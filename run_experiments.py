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

try:
    from kan import KAN
except ImportError:
    print("[ERRORE] Libreria 'kan' non trovata.")
    exit()

# 1. SETUP E SEED (DIVERSI DAL PILOT 42)
SEEDS = [101, 202, 303, 404, 505]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SETUP] Backend: {DEVICE} - Avvio pipeline pre-flight sui seed: {SEEDS}")

# 2. CARICAMENTO DATI
base_path = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\archive\CICIOT23\cleaned"
df_train = pl.read_csv(f"{base_path}\\train_clean.csv")
df_val = pl.read_csv(f"{base_path}\\validation_clean.csv")

X_train_raw = df_train.drop("label").to_numpy()
y_train_raw = df_train.select("label").to_numpy().squeeze()
X_val_raw = df_val.drop("label").to_numpy()
y_val_raw = df_val.select("label").to_numpy().squeeze()

classes = np.unique(y_train_raw)
label_map = {label: idx for idx, label in enumerate(classes)}
target_names = [str(c) for c in classes]

y_train = np.array([label_map[l] for l in y_train_raw])
y_val = np.array([label_map[l] for l in y_val_raw])

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# 3. PREPROCESSING DIFFERENZIATO
# Modelli basati su alberi (RF, XGB, LGBM) usano i dati raw per preservare le distribuzioni
# La rete KAN richiede i dati standardizzati
print("[PREPROCESSING] Standardizzazione feature esclusiva per architettura KAN...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)

class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
kan_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), batch_size=8192, shuffle=True)
kan_val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)), batch_size=8192, shuffle=False)

def log_metrics(model_name, seed, y_true, y_pred):
    print(f"\n[{model_name} - SEED {seed}] Report Generato")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")

# 4. LOOP DI ADDESTRAMENTO
# NOTA PRE-FLIGHT: Il loop è predisposto, ma in attesa di autorizzazione per l'esecuzione massiva.
for current_seed in SEEDS:
    print(f"\n{'='*50}\nINIZIALIZZAZIONE RUN - SEED: {current_seed}\n{'='*50}")
    
    # --- RANDOM FOREST ---
    print("\n[1/4] Addestramento Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=current_seed, n_jobs=-1)
    rf.fit(X_train_raw, y_train)
    log_metrics("Random Forest", current_seed, y_val, rf.predict(X_val_raw))

    # --- XGBOOST ---
    print("\n[2/4] Addestramento XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax', num_class=len(classes),
        tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=current_seed, n_jobs=-1
    )
    xgb_model.fit(X_train_raw, y_train, sample_weight=class_weights[y_train])
    log_metrics("XGBoost", current_seed, y_val, xgb_model.predict(X_val_raw))

    # --- LIGHTGBM ---
    print("\n[3/4] Addestramento LightGBM...")
    lgb_train = lgb.Dataset(X_train_raw, y_train, weight=class_weights[y_train])
    lgb_model = lgb.train({'objective': 'multiclass', 'num_class': len(classes), 'seed': current_seed, 'verbose': -1, 'n_jobs': -1}, lgb_train, num_boost_round=100)
    lgb_preds = np.argmax(lgb_model.predict(X_val_raw), axis=1)
    log_metrics("LightGBM", current_seed, y_val, lgb_preds)

    # --- KAN ---
    print("\n[4/4] Addestramento KAN...")
    torch.manual_seed(current_seed)
    model = KAN(width=[X_train_scaled.shape[1], 64, len(classes)], grid=5, k=3, seed=current_seed).to(DEVICE)
    
    # Telemetria Architetturale KAN
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Parametri addestrabili KAN: {total_params}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    # Logica semplificata per pre-flight (il training reale impiegherà ore)
    # L'early stopping completo e il salvataggio sono omessi qui per brevità strutturale,
    # ma l'infrastruttura di telemetria è iniettata per il report.
    model.train()
    for batch_x, batch_y in kan_train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
        loss.backward()
        optimizer.step()
        break # PRE-FLIGHT BREAK: Interrompe dopo 1 batch per validazione infrastruttura
        
    execution_time = time.time() - start_time
    vram_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print(f"[*] KAN Telemetria Hardware -> Tempo batch test: {execution_time:.2f}s | Picco VRAM: {vram_used:.2f} MB")
    print(f"[!] Run per il seed {current_seed} completata (modalità pre-flight).")
    break # Interrompe il loop dei seed per il pre-flight