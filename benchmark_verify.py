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
from sklearn.metrics import classification_report

# ==========================================
# 1. CONFIGURAZIONE E SEED
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SETUP] Dispositivo in uso: {DEVICE}")

# ==========================================
# 2. CARICAMENTO DATI BONIFICATI
# ==========================================
print("[DATA] Caricamento dataset bonificati in corso...")
df_train = pl.read_csv("archive/CICIOT23/cleaned/train_clean.csv")
df_val = pl.read_csv("archive/CICIOT23/cleaned/validation_clean.csv")
df_test = pl.read_csv("archive/CICIOT23/cleaned/test_clean.csv")

# Separazione feature e target (Assumiamo che la label sia testuale e vada mappata, o già numerica)
# Per semplicità in questo verify, estraiamo le matrici numpy. 
# NOTA: Adatta 'label' col nome esatto della tua colonna target.
X_train_raw = df_train.drop("label").to_numpy()
y_train_raw = df_train.select("label").to_numpy().squeeze()
X_val_raw = df_val.drop("label").to_numpy()
y_val_raw = df_val.select("label").to_numpy().squeeze()
X_test_raw = df_test.drop("label").to_numpy()
y_test_raw = df_test.select("label").to_numpy().squeeze()

# Mapping delle label (se testuali) in interi 0-33
classes = np.unique(y_train_raw)
label_map = {label: idx for idx, label in enumerate(classes)}
y_train = np.array([label_map[l] for l in y_train_raw])
y_val = np.array([label_map[l] for l in y_val_raw])
y_test = np.array([label_map[l] for l in y_test_raw])

# Scaler SOLO sul train
print("[DATA] Fit dello StandardScaler solo sul Train...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# ==========================================
# 3. VERIFICA LIGHTGBM
# ==========================================
print("\n[LightGBM] Avvio addestramento...")
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

lgb_params = {
    'objective': 'multiclass', # Forzato come da protocollo
    'num_class': len(classes),
    'metric': 'multi_logloss',
    'seed': SEED,
    'n_jobs': 16, # Ryzen 9850X3D
    'verbose': -1
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Dimensione su disco LightGBM
lgb_model.save_model('temp_lgb.txt')
lgb_size = os.path.getsize('temp_lgb.txt') / (1024 * 1024)
os.remove('temp_lgb.txt')
print(f"[LightGBM] Dimensione modello su disco: {lgb_size:.2f} MB")

# ==========================================
# 4. VERIFICA RETE NEURALE (PyTorch)
# ==========================================
print("\n[PyTorch] Preparazione tensori...")
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=1024, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=1024, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=1024, shuffle=False)

# Rete Placeholder (Sostituisci con la tua classe KAN effettiva)
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))
    def forward(self, x):
        return self.fc(x)

model = SimpleNet(X_train.shape[1], len(classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss() # Aggiungi i pesi qui se li hai calcolati
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("[PyTorch] Avvio addestramento con Early Stopping (Patience 10)...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
        loss.backward()
        optimizer.step()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.to(DEVICE))
            val_loss += criterion(outputs, batch_y.to(DEVICE)).item()
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f"[PyTorch] Early stopping innescato all'epoca {epoch+1}")
            break

# Ripristino checkpoint migliore
model.load_state_dict(torch.load('best_model.pt'))

# Calcolo dimensione su disco
nn_size = os.path.getsize('best_model.pt') / (1024 * 1024)
print(f"[PyTorch] Dimensione modello su disco: {nn_size:.2f} MB")

# ==========================================
# 5. PROFILAZIONE CUDA E TEMPI
# ==========================================
print("\n[PyTorch] Avvio profilazione hardware...")
model.eval()

# Warm-up (un batch)
with torch.no_grad():
    for batch_x, _ in test_loader:
        _ = model(batch_x.to(DEVICE))
        break

torch.cuda.synchronize()
start_time = time.perf_counter()

total_samples = 0
with torch.no_grad():
    for batch_x, _ in test_loader:
        # Il trasferimento CPU->GPU è incluso nel timer come richiesto
        outputs = model(batch_x.to(DEVICE))
        total_samples += batch_x.size(0)

torch.cuda.synchronize()
end_time = time.perf_counter()

time_ms_per_sample = ((end_time - start_time) / total_samples) * 1000
print(f"[PyTorch] Tempo medio di inferenza per campione: {time_ms_per_sample:.4f} ms")

print("\n[COMPLETATO] Verifica terminata. Raccogli questi log per il Prof.")