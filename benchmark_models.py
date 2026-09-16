import polars as pl
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from kan import KAN

# ==========================================
# CONFIGURAZIONE GENERALE
# ==========================================
SEEDS = [42, 123, 456, 789, 1024]
TRAIN_PATH = "archive/CICIOT23/train/train.csv"
TEST_PATH = "archive/CICIOT23/test/test.csv"
RARE_CLASSES = [28, 31, 33] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device attivo: {device}")

results = {
    'KAN': {'f1': [], 'bal_acc': [], 'recall_rare': [], 'inf_time': []},
    'RF': {'f1': [], 'bal_acc': [], 'recall_rare': [], 'inf_time': []},
    'XGB': {'f1': [], 'bal_acc': [], 'recall_rare': [], 'inf_time': []},
    'LGBM': {'f1': [], 'bal_acc': [], 'recall_rare': [], 'inf_time': []}
}

# ==========================================
# LETTURA DATI ASSOLUTI (NO LEAKAGE)
# ==========================================
print("\nLettura Train e Test set in corso...")
df_train_raw = pl.read_csv(TRAIN_PATH)
df_test_raw = pl.read_csv(TEST_PATH)

# Preparazione Test Set (Fisso per tutti i seed)
X_test_raw = df_test_raw.drop("label").to_numpy()
y_test_text = df_test_raw["label"].to_numpy()

le = LabelEncoder()
# Fit sull'intero spettro delle label per sicurezza
le.fit(np.concatenate((df_train_raw["label"].to_numpy(), y_test_text)))
y_test = le.transform(y_test_text)
num_classes = len(le.classes_)

MIN_SAMPLES = 300
MAX_SAMPLES = 10000

# ==========================================
# CICLO SUI SEED
# ==========================================
for seed in SEEDS:
    print(f"\n{'='*40}")
    print(f"AVVIO RUN CON SEED: {seed}")
    print(f"{'='*40}")
    
    # Campionamento del Train dipendente dal seed (garantisce varianza)
    dfs_sampled = []
    for label, group in df_train_raw.group_by('label'):
        n_samples = group.height
        if n_samples < MIN_SAMPLES:
            dfs_sampled.append(group)
        elif n_samples > MAX_SAMPLES:
            dfs_sampled.append(group.sample(n=MAX_SAMPLES, seed=seed))
        else:
            dfs_sampled.append(group)
            
    df_train_balanced = pl.concat(dfs_sampled)
    X_train_raw = df_train_balanced.drop("label").to_numpy()
    y_train = le.transform(df_train_balanced["label"].to_numpy())
    
    # Preprocessing appreso SOLO sul train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Tensori
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    # ------------- 1. KAN -------------
    print("Addestramento KAN (Mini-batch 5000)...")
    kan_model = KAN(width=[X_train_scaled.shape[1], 32, 16, num_classes], grid=5, k=3, seed=seed).to(device)
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
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
    
    kan_model.eval()
    start_time = time.time()
    with torch.no_grad():
        out = kan_model(X_test_t)
        preds = torch.argmax(out, dim=1).cpu().numpy()
    end_time = time.time()
    
    results['KAN']['f1'].append(f1_score(y_test, preds, average='macro'))
    results['KAN']['bal_acc'].append(balanced_accuracy_score(y_test, preds))
    recalls = recall_score(y_test, preds, average=None, zero_division=0)
    results['KAN']['recall_rare'].append([recalls[i] for i in RARE_CLASSES])
    results['KAN']['inf_time'].append((end_time - start_time) / len(y_test))

    # ------------- 2. Random Forest -------------
    print("Addestramento Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=seed)
    rf.fit(X_train_scaled, y_train)
    
    start_time = time.time()
    preds = rf.predict(X_test_scaled)
    end_time = time.time()
    
    results['RF']['f1'].append(f1_score(y_test, preds, average='macro'))
    results['RF']['bal_acc'].append(balanced_accuracy_score(y_test, preds))
    recalls = recall_score(y_test, preds, average=None, zero_division=0)
    results['RF']['recall_rare'].append([recalls[i] for i in RARE_CLASSES])
    results['RF']['inf_time'].append((end_time - start_time) / len(y_test))

    # ------------- 3. XGBoost -------------
    print("Addestramento XGBoost...")
    xgb = XGBClassifier(n_estimators=100, tree_method='hist', device='cuda', random_state=seed)
    xgb.fit(X_train_scaled, y_train)
    
    start_time = time.time()
    preds = xgb.predict(X_test_scaled)
    end_time = time.time()
    
    results['XGB']['f1'].append(f1_score(y_test, preds, average='macro'))
    results['XGB']['bal_acc'].append(balanced_accuracy_score(y_test, preds))
    recalls = recall_score(y_test, preds, average=None, zero_division=0)
    results['XGB']['recall_rare'].append([recalls[i] for i in RARE_CLASSES])
    results['XGB']['inf_time'].append((end_time - start_time) / len(y_test))

    # ------------- 4. LightGBM -------------
    print("Addestramento LightGBM...")
    lgbm = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=seed, verbose=-1)
    lgbm.fit(X_train_scaled, y_train)
    
    start_time = time.time()
    preds = lgbm.predict(X_test_scaled)
    end_time = time.time()
    
    results['LGBM']['f1'].append(f1_score(y_test, preds, average='macro'))
    results['LGBM']['bal_acc'].append(balanced_accuracy_score(y_test, preds))
    recalls = recall_score(y_test, preds, average=None, zero_division=0)
    results['LGBM']['recall_rare'].append([recalls[i] for i in RARE_CLASSES])
    results['LGBM']['inf_time'].append((end_time - start_time) / len(y_test))

# ==========================================
# REPORT FINALE AGGREGATO
# ==========================================
print("\n" + "="*60)
print("RISULTATI FINALI SUI 5 SEED (MEDIA ± DEV. STD)")
print("="*60)
for model_name, metrics in results.items():
    f1_mean = np.mean(metrics['f1'])
    f1_std = np.std(metrics['f1'])
    bal_acc_mean = np.mean(metrics['bal_acc'])
    bal_acc_std = np.std(metrics['bal_acc'])
    time_mean = np.mean(metrics['inf_time']) * 1e6
    
    # Calcolo media delle recall per le singole classi rare sui 5 seed
    rare_recalls_matrix = np.array(metrics['recall_rare'])
    rare_recalls_mean = np.mean(rare_recalls_matrix, axis=0)
    
    print(f"\nModello: {model_name}")
    print(f"  F1-Macro:          {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Balanced Accuracy: {bal_acc_mean:.4f} ± {bal_acc_std:.4f}")
    print(f"  Recall C-28:       {rare_recalls_mean[0]:.4f}")
    print(f"  Recall C-31:       {rare_recalls_mean[1]:.4f}")
    print(f"  Recall C-33:       {rare_recalls_mean[2]:.4f}")
    print(f"  Inf. Time/sample:  {time_mean:.2f} µs")