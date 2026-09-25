import argparse
import os
import time
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from kan import KAN

# --- CONFIGURAZIONE ---
parser = argparse.ArgumentParser(description="Runner Unificato CICIoT2023")
parser.add_argument("--smoke", action="store_true", help="Esegue un test rapido (1% dati, iterazioni ridotte)")
args = parser.parse_args()

SEEDS = [101, 202, 303, 404, 505]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\archive\CICIOT23\cleaned"
OUT_DIR = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\results"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INIT] Modalità Smoke: {args.smoke} | Backend: {DEVICE}")

# --- CARICAMENTO DATI ---
df_train = pl.read_csv(f"{BASE_PATH}\\train_clean.csv")
df_val = pl.read_csv(f"{BASE_PATH}\\validation_clean.csv")
df_test = pl.read_csv(f"{BASE_PATH}\\test_clean.csv")

if args.smoke:
    df_train = df_train.sample(fraction=0.01, seed=42)
    df_val = df_val.sample(fraction=0.05, seed=42)
    df_test = df_test.sample(fraction=0.05, seed=42)

X_train_raw, y_train_raw = df_train.drop("label").to_numpy(), df_train.select("label").to_numpy().squeeze()
X_val_raw, y_val_raw = df_val.drop("label").to_numpy(), df_val.select("label").to_numpy().squeeze()
X_test_raw, y_test_raw = df_test.drop("label").to_numpy(), df_test.select("label").to_numpy().squeeze()

classes = np.unique(y_train_raw)
label_map = {label: idx for idx, label in enumerate(classes)}
target_names = [str(c) for c in classes]

y_train = np.array([label_map[l] for l in y_train_raw])
y_val = np.array([label_map.get(l, -1) for l in y_val_raw])
y_test = np.array([label_map.get(l, -1) for l in y_test_raw])

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

# --- PREPROCESSING KAN ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

kan_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=8192, shuffle=True)
kan_val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=8192, shuffle=False)
kan_test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), batch_size=8192, shuffle=False)

def profile_cuda_inference(model, dummy_input):
    if DEVICE.type != 'cuda': return 0, 0
    # Warmup
    for _ in range(5): _ = model(dummy_input)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.reset_peak_memory_stats()
    start_event.record()
    with torch.no_grad():
        _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event), torch.cuda.max_memory_allocated() / (1024**2)

results_f1 = {'RF': [], 'XGB': [], 'LGBM': [], 'KAN': []}

for seed in SEEDS:
    print(f"\n{'='*40}\nAVVIO SEED: {seed}\n{'='*40}")
    
    # --- LIGHTGBM ---
    print("[1/2] Addestramento LightGBM...")
    lgb_train = lgb.Dataset(X_train_raw, y_train, weight=class_weights[y_train])
    lgb_val = lgb.Dataset(X_val_raw, y_val, reference=lgb_train)
    lgb_model = lgb.train(
        {'objective': 'multiclass', 'num_class': len(classes), 'seed': seed, 'verbose': -1, 'n_jobs': -1},
        lgb_train, num_boost_round=5 if args.smoke else 100, valid_sets=[lgb_val], 
        callbacks=[lgb.early_stopping(stopping_rounds=10)] if not args.smoke else []
    )
    preds = np.argmax(lgb_model.predict(X_test_raw), axis=1)
    results_f1['LGBM'].append(f1_score(y_test, preds, average='macro'))
    
    # --- KAN ---
    print("[2/2] Addestramento KAN...")
    torch.manual_seed(seed)
    kan = KAN(width=[X_train_scaled.shape[1], 64, len(classes)], grid=5, k=3, seed=seed).to(DEVICE)
    optimizer = optim.Adam(kan.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    best_loss, patience, max_epochs = float('inf'), 0, (2 if args.smoke else 100)
    best_path = os.path.join(OUT_DIR, f"kan_best_{seed}.pt")
    
    for epoch in range(max_epochs):
        kan.train()
        for bx, by in kan_train_loader:
            optimizer.zero_grad()
            loss = criterion(kan(bx.to(DEVICE)), by.to(DEVICE))
            loss.backward()
            optimizer.step()
            
        kan.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in kan_val_loader:
                val_loss += criterion(kan(bx.to(DEVICE)), by.to(DEVICE)).item()
        val_loss /= len(kan_val_loader)
        
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(kan.state_dict(), best_path)
        else:
            patience += 1
            if patience >= 10 and not args.smoke:
                print(f"  [!] Early stopping KAN a epoca {epoch+1}")
                break

    kan.load_state_dict(torch.load(best_path))
    kan.eval()
    
    # Test Evaluation
    all_preds = []
    with torch.no_grad():
        for bx, _ in kan_test_loader:
            all_preds.extend(torch.max(kan(bx.to(DEVICE)), 1)[1].cpu().numpy())
    results_f1['KAN'].append(f1_score(y_test, all_preds, average='macro'))
    
    # Profilazione Hardware KAN
    dummy_input = torch.FloatTensor(X_test_scaled[:8192]).to(DEVICE)
    inf_time_ms, vram_mb = profile_cuda_inference(kan, dummy_input)
    param_count = sum(p.numel() for p in kan.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(best_path) / (1024**2)
    print(f"  [*] KAN Profiling - Parametri: {param_count} | Dimensione su disco: {model_size_mb:.2f} MB")
    print(f"  [*] Inferenza (Batch 8192) - Tempo: {inf_time_ms:.2f} ms | Picco VRAM: {vram_mb:.2f} MB")

print("\n=== RIEPILOGO FINALE SUI 5 SEED (TEST SET) ===")
for model_name, f1_scores in results_f1.items():
    if f1_scores: 
        print(f"{model_name} Macro-F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")