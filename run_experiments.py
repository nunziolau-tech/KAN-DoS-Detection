import argparse
import os
import time
import json
import joblib
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
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from kan import KAN

# --- CONFIGURAZIONE ---
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="Smoke test: 1% stratificato, no Test set")
args = parser.parse_args()

SEEDS = [101, 202, 303, 404, 505]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\archive\CICIOT23\cleaned"
OUT_DIR = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\results\smoke" if args.smoke else r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\results\final"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INIT] Modalità Smoke: {args.smoke} | Backend: {DEVICE} | Output: {OUT_DIR}")

# --- CARICAMENTO E PREPROCESSING ---
df_train = pl.read_csv(f"{BASE_PATH}\\train_clean.csv")
df_val = pl.read_csv(f"{BASE_PATH}\\validation_clean.csv")

feature_cols = [c for c in df_train.columns if c != "label"]
X_train_raw, y_train_raw = df_train.select(feature_cols).to_numpy(), df_train.select("label").to_numpy().squeeze()
X_val_raw, y_val_raw = df_val.select(feature_cols).to_numpy(), df_val.select("label").to_numpy().squeeze()

# Gestione Smoke Test: Campionamento stratificato su Train e Val (Test escluso)
if args.smoke:
    X_train_raw, _, y_train_raw, _ = train_test_split(X_train_raw, y_train_raw, train_size=0.01, stratify=y_train_raw, random_state=42)
    X_val_raw, _, y_val_raw, _ = train_test_split(X_val_raw, y_val_raw, train_size=0.05, stratify=y_val_raw, random_state=42)

classes = np.unique(y_train_raw)
label_map = {label: int(idx) for idx, label in enumerate(classes)}
target_names = [str(c) for c in classes]

y_train = np.array([label_map[l] for l in y_train_raw])
y_val = np.array([label_map.get(l, -1) for l in y_val_raw])

# Salvataggio Metadati Globali
joblib.dump(label_map, os.path.join(OUT_DIR, "label_map.joblib"))
with open(os.path.join(OUT_DIR, "feature_order.json"), "w") as f:
    json.dump(feature_cols, f)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

kan_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=8192, shuffle=True)
kan_val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=8192, shuffle=False)

# --- FUNZIONI DI SUPPORTO ---
def save_artifacts(model_name, seed, y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    with open(os.path.join(OUT_DIR, f"{model_name}_seed{seed}_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    np.save(os.path.join(OUT_DIR, f"{model_name}_seed{seed}_cm.npy"), cm)
    return report['macro avg']['f1-score'], balanced_accuracy_score(y_true, y_pred)

def profile_inference(model, dummy_x, is_torch=False):
    start_time = time.time()
    if is_torch:
        torch.cuda.empty_cache() # Pulisce lo stato residuo dell'addestramento
        with torch.no_grad():
            for _ in range(3): _ = model(dummy_x.to(DEVICE)) # Warmup includendo trasferimento
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_x.to(DEVICE)) # Misura inferenza reale + trasferimento RAM->VRAM
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        vram = torch.cuda.max_memory_allocated() / (1024**2)
        return (t1 - t0) * 1000, vram
    else:
        t0 = time.perf_counter()
        _ = model.predict(dummy_x)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000, 0.0 # CPU models non usano VRAM CUDA

# --- LOOP SUI SEED ---
for seed in SEEDS:
    print(f"\n{'='*40}\nAVVIO SEED: {seed}\n{'='*40}")
    
    # 1. Random Forest
    print("[1/4] Addestramento Random Forest...")
    rf = RandomForestClassifier(n_estimators=10 if args.smoke else 100, class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_train_raw, y_train)
    rf_preds = rf.predict(X_val_raw)
    save_artifacts("RF", seed, y_val, rf_preds)
    rf_time, _ = profile_inference(rf, X_val_raw[:8192])
    print(f"  [*] Inferenza (Batch 8192) - Tempo: {rf_time:.2f} ms")

    # 2. XGBoost
    print("[2/4] Addestramento XGBoost...")
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(classes), tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu', seed=seed, n_jobs=-1)
    xgb_model.fit(X_train_raw, y_train, sample_weight=class_weights[y_train])
    xgb_preds = xgb_model.predict(X_val_raw)
    save_artifacts("XGB", seed, y_val, xgb_preds)
    xgb_time, xgb_vram = profile_inference(xgb_model, X_val_raw[:8192])
    print(f"  [*] Inferenza (Batch 8192) - Tempo: {xgb_time:.2f} ms")

    # 3. LightGBM
    print("[3/4] Addestramento LightGBM...")
    lgb_train = lgb.Dataset(X_train_raw, y_train, weight=class_weights[y_train])
    lgb_val = lgb.Dataset(X_val_raw, y_val, reference=lgb_train)
    lgb_model = lgb.train({'objective': 'multiclass', 'num_class': len(classes), 'seed': seed, 'verbose': -1, 'n_jobs': -1}, lgb_train, num_boost_round=10 if args.smoke else 100, valid_sets=[lgb_val], callbacks=[lgb.early_stopping(stopping_rounds=10)] if not args.smoke else [])
    lgb_preds = np.argmax(lgb_model.predict(X_val_raw), axis=1)
    save_artifacts("LGBM", seed, y_val, lgb_preds)
    lgb_time, _ = profile_inference(lgb_model, X_val_raw[:8192])
    print(f"  [*] Inferenza (Batch 8192) - Tempo: {lgb_time:.2f} ms")

    # 4. KAN
    print("[4/4] Addestramento KAN...")
    torch.manual_seed(seed)
    kan = KAN(width=[X_train_scaled.shape[1], 64, len(classes)], grid=5, k=3, seed=seed).to(DEVICE)
    optimizer = optim.Adam(kan.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    best_loss, patience, max_epochs = float('inf'), 0, (2 if args.smoke else 100)
    best_path = os.path.join(OUT_DIR, f"kan_best_seed_{seed}.pt")
    
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
    
    all_preds = []
    with torch.no_grad():
        for bx, _ in kan_val_loader: # Valutazione sempre su Val in pre-flight/smoke
            all_preds.extend(torch.max(kan(bx.to(DEVICE)), 1)[1].cpu().numpy())
    save_artifacts("KAN", seed, y_val, all_preds)
    
    # Profilazione KAN esatta
    kan_time, kan_vram = profile_inference(kan, torch.FloatTensor(X_val_scaled[:8192]), is_torch=True)
    param_count = sum(p.numel() for p in kan.parameters() if p.requires_grad)
    print(f"  [*] KAN Parametri: {param_count}")
    print(f"  [*] Inferenza (Batch 8192) - Tempo (incluso trsf. GPU): {kan_time:.2f} ms | VRAM netta: {kan_vram:.2f} MB")
    
    if args.smoke:
        break # In smoke test facciamo solo il primo seed per validare l'infrastruttura

print("\n[!] Pre-Flight / Smoke completato con successo.")