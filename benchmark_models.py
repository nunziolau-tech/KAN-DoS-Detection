import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import time
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
try:
    from kan import KAN
except ImportError:
    print("[ERRORE LIBRERIA] Libreria 'kan' non trovata. Assicurati che sia installata nel tuo venv.")
    exit()

# ==========================================
# 1. SETUP E SEED (PILOT = 42)
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SETUP] Hardware backend: {DEVICE}")

# ==========================================
# 2. CARICAMENTO DATI (PATH LOCALI ASSOLUTI)
# ==========================================
print("[DATA] Caricamento dataset bonificati...")
base_path = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\archive\CICIOT23\cleaned"
df_train = pl.read_csv(f"{base_path}\\train_clean.csv")
df_val = pl.read_csv(f"{base_path}\\validation_clean.csv")
df_test = pl.read_csv(f"{base_path}\\test_clean.csv") # Caricato ma ESCLUSO dalle decisioni

X_train_raw = df_train.drop("label").to_numpy()
y_train_raw = df_train.select("label").to_numpy().squeeze()
X_val_raw = df_val.drop("label").to_numpy()
y_val_raw = df_val.select("label").to_numpy().squeeze()

classes = np.unique(y_train_raw)
label_map = {label: idx for idx, label in enumerate(classes)}
target_names = [str(c) for c in classes]

y_train = np.array([label_map[l] for l in y_train_raw])
y_val = np.array([label_map[l] for l in y_val_raw])

print("[DATA] Fit dello StandardScaler SOLO sul Train...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

# ==========================================
# 3. CALCOLO PESI DI CLASSE (Punto 2 del Relatore)
# ==========================================
print("[DATA] Calcolo class weights bilanciati sul Train...")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

# ==========================================
# 4. LIGHTGBM - PILOT SEED 42
# ==========================================
print("\n--- [LightGBM] Avvio addestramento Pilot ---")
lgb_train = lgb.Dataset(X_train, y_train, weight=class_weights[y_train])
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

lgb_params = {
    'objective': 'multiclass',
    'num_class': len(classes),
    'metric': 'multi_logloss',
    'seed': SEED,
    'n_jobs': 16, # Ryzen 7 9850X3D
    'verbose': -1
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

lgb_preds_prob = lgb_model.predict(X_val)
lgb_preds = np.argmax(lgb_preds_prob, axis=1)

print("\n[LightGBM] Metriche su Validation Set:")
print(f"Macro-F1: {f1_score(y_val, lgb_preds, average='macro'):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_val, lgb_preds):.4f}")

# ==========================================
# 5. KAN REALE - PILOT SEED 42
# ==========================================
print("\n--- [KAN] Avvio addestramento Pilot ---")
input_dim = X_train.shape[1]
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=1024, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=1024, shuffle=False)

# Inizializzazione della tua rete
model = KAN(width=[input_dim, 32, 16, len(classes)], grid=5, k=3, seed=SEED).to(DEVICE)
print(f"[KAN] Architettura definita: width={[input_dim, 32, 16, len(classes)]}, grid=5, k=3")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience_counter = 0
best_model_path = f'kan_best_seed_{SEED}.pt'

for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x.to(DEVICE))
        loss = criterion(outputs, batch_y.to(DEVICE))
        loss.backward()
        optimizer.step()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.to(DEVICE))
            val_loss += criterion(outputs, batch_y.to(DEVICE)).item()
    val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f"[KAN] Early stopping innescato all'epoca {epoch+1} (Miglior Val Loss: {best_val_loss:.4f})")
            break

# Ripristino e Metriche Finali KAN
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        outputs = model(batch_x.to(DEVICE))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

print("\n[KAN] Metriche su Validation Set:")
print(f"Macro-F1: {f1_score(y_val, all_preds, average='macro'):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_val, all_preds):.4f}")
print("\n[KAN] Classification Report (Validation Set):")
print(classification_report(y_val, all_preds, target_names=target_names, zero_division=0))

print("\n[COMPLETATO] Pilot Seed 42 terminato. Invia questi log completi al relatore.")