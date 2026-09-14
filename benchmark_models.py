import polars as pl
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from kan import KAN

# ==========================================
# CONFIGURAZIONE GENERALE E SEEDS
# ==========================================
SEEDS = [42, 123, 456, 789, 1024]
DATASET_PATH = "archive/CICIOT23/train/train.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device per reti neurali/XGB: {device}")

# Dizionario per accumulare i risultati
results = {
    'KAN': {'acc': [], 'f1': [], 'inf_time': []},
    'RF': {'acc': [], 'f1': [], 'inf_time': []},
    'XGB': {'acc': [], 'f1': [], 'inf_time': []},
    'LGBM': {'acc': [], 'f1': [], 'inf_time': []}
}

# ==========================================
# LETTURA E PREPARAZIONE DATI (Eseguita una sola volta)
# ==========================================
print("\nLettura dataset e campionamento asimmetrico in corso...")
df = pl.read_csv(DATASET_PATH)

MIN_SAMPLES = 300
MAX_SAMPLES = 10000

dfs_sampled = []
for label, group in df.group_by('label'):
    n_samples = group.height
    if n_samples < MIN_SAMPLES:
        dfs_sampled.append(group)
    elif n_samples > MAX_SAMPLES:
        dfs_sampled.append(group.sample(n=MAX_SAMPLES, seed=42))
    else:
        dfs_sampled.append(group)

df_balanced = pl.concat(dfs_sampled)
print(f"Dataset bilanciato: {df_balanced.height} campioni.")

X_raw = df_balanced.drop("label").to_numpy()
y_text = df_balanced["label"].to_numpy()

le = LabelEncoder()
y_raw = le.fit_transform(y_text)
num_classes = len(np.unique(y_raw))

# ==========================================
# CICLO SUI SEED
# ==========================================
for seed in SEEDS:
    print(f"\n{'='*40}")
    print(f"AVVIO RUN CON SEED: {seed}")
    print(f"{'='*40}")
    
    # 1. Split e Scaling specifico per questo seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=seed
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tensori per KAN
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
   # ------------- 1. KAN -------------
    print("Addestramento KAN (Mini-batch 5000)...")
    kan_model = KAN(width=[X_train_scaled.shape[1], 32, 16, num_classes], grid=5, k=3, seed=seed).to(device)
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Creazione del DataLoader per non far esplodere la GPU
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
    
    kan_model.train()
    for epoch in range(20):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = kan_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    
    # Inferenza KAN (qui possiamo fare full-batch perché non c'è calcolo dei gradienti)
    kan_model.eval()
    start_time = time.time()
    with torch.no_grad():
        out = kan_model(X_test_t)
        kan_preds = torch.argmax(out, dim=1).cpu().numpy()
    end_time = time.time()
    
    results['KAN']['acc'].append(accuracy_score(y_test, kan_preds))
    results['KAN']['f1'].append(f1_score(y_test, kan_preds, average='macro'))
    results['KAN']['inf_time'].append((end_time - start_time) / len(y_test))

    # ------------- 2. Random Forest -------------
    print("Addestramento Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=seed)
    rf.fit(X_train_scaled, y_train)
    
    start_time = time.time()
    rf_preds = rf.predict(X_test_scaled)
    end_time = time.time()
    
    results['RF']['acc'].append(accuracy_score(y_test, rf_preds))
    results['RF']['f1'].append(f1_score(y_test, rf_preds, average='macro'))
    results['RF']['inf_time'].append((end_time - start_time) / len(y_test))

    # ------------- 3. XGBoost -------------
    print("Addestramento XGBoost...")
    # tree_method='hist' e device='cuda' spostano l'addestramento sulla tua 5080
    xgb = XGBClassifier(n_estimators=100, tree_method='hist', device='cuda', random_state=seed)
    xgb.fit(X_train_scaled, y_train)
    
    start_time = time.time()
    xgb_preds = xgb.predict(X_test_scaled)
    end_time = time.time()
    
    results['XGB']['acc'].append(accuracy_score(y_test, xgb_preds))
    results['XGB']['f1'].append(f1_score(y_test, xgb_preds, average='macro'))
    results['XGB']['inf_time'].append((end_time - start_time) / len(y_test))

    # ------------- 4. LightGBM -------------
    print("Addestramento LightGBM...")
    # n_jobs=-1 usa la CPU per LGBM (su Windows la GPU per LGBM richiede build custom, meglio CPU)
    lgbm = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=seed, verbose=-1)
    lgbm.fit(X_train_scaled, y_train)
    
    start_time = time.time()
    lgbm_preds = lgbm.predict(X_test_scaled)
    end_time = time.time()
    
    results['LGBM']['acc'].append(accuracy_score(y_test, lgbm_preds))
    results['LGBM']['f1'].append(f1_score(y_test, lgbm_preds, average='macro'))
    results['LGBM']['inf_time'].append((end_time - start_time) / len(y_test))


# ==========================================
# REPORT FINALE AGGREGATO
# ==========================================
print("\n" + "="*50)
print("RISULTATI FINALI SUI 5 SEED (MEDIA ± DEV. STD)")
print("="*50)
for model_name, metrics in results.items():
    acc_mean = np.mean(metrics['acc'])
    acc_std = np.std(metrics['acc'])
    f1_mean = np.mean(metrics['f1'])
    f1_std = np.std(metrics['f1'])
    time_mean = np.mean(metrics['inf_time']) * 1e6 # Convertito in microsecondi
    
    print(f"\nModello: {model_name}")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1-Macro: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Inf. Time per sample: {time_mean:.2f} µs")