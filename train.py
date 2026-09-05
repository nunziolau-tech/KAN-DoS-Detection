import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kan import KAN
from sklearn.metrics import precision_recall_fscore_support
import gc

# 1. Configurazione Dispositivo Locale (RTX 5080)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo di addestramento attivo: {device}")

# 2. Caricamento del dataset locale
file_path = 'archive/CICIOT23/train/train.csv'  # <-- Inseriscilo qui
print("Caricamento del dataset in corso...")
df = pd.read_csv(file_path)

# 3. Creazione etichetta binaria
label_col = 'label' if 'label' in df.columns else 'Label'
is_benign = df[label_col].str.lower().str.contains('benign')
df['binary_label'] = (~is_benign).astype(int)

print(f"Totale righe caricate: {len(df)}")

print("Avvio campionamento stratificato per preservare gli attacchi rari...")
df_benign = df[df['binary_label'] == 0].sample(n=50000, random_state=42)
df_attack_full = df[df['binary_label'] == 1]
n_attack_samples = 50000

df_attack = df_attack_full.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=max(1, int(len(x) * (n_attack_samples / len(df_attack_full)))), random_state=42)
)

df_balanced = pd.concat([df_benign, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Righe finali bilanciate: {len(df_balanced)}")

del df
gc.collect()
print("RAM liberata dal dataset originale.")

# 4. Isolamento delle feature numeriche
feature_cols = df_balanced.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
feature_cols = [c for c in feature_cols if c not in ['label', 'binary_label', label_col]]

print(f"Feature numeriche selezionate: {len(feature_cols)}")

# 5. Normalizzazione (StandardScaler)
scaler = StandardScaler()
X = scaler.fit_transform(df_balanced[feature_cols])
y = df_balanced['binary_label'].values

# 6. Split Stratificato (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Costruzione dei Tensori PyTorch per la GPU (RTX 5080)
dataset = {
    'train_input': torch.FloatTensor(X_train).to(device),
    'train_label': torch.FloatTensor(y_train).reshape(-1, 1).to(device),
    'test_input': torch.FloatTensor(X_test).to(device),
    'test_label': torch.FloatTensor(y_test).reshape(-1, 1).to(device)
}

print(f"\n--- DATASET PRONTO PER KAN ---")
print(f"Train Input Shape: {dataset['train_input'].shape}")
print(f"Test Input Shape: {dataset['test_input'].shape}")

print("Inizializzazione del modello KAN su GPU...")
input_dim = dataset['train_input'].shape[1]

# Architettura KAN originale
model = KAN(width=[input_dim, 32, 16, 1], grid=5, k=3, seed=42).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Calcolo dinamico del pos_weight per la BCE Loss
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
pos_weight_val = num_neg / num_pos
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

print(f"Bilanciamento dinamico loss applicato (pos_weight): {pos_weight_val:.4f}")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
epochs = 50
print(f"Avvio training KAN per {epochs} epoche...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    train_out = model(dataset['train_input'])
    loss = criterion(train_out, dataset['train_label'])

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}')

print("\n--- VALUTAZIONE FINALE SUL CICIoT2023 ---")
model.eval()
with torch.no_grad():
    test_outputs = model(dataset['test_input'])
    preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy()
    labels = dataset['test_label'].cpu().numpy()

    accuracy = (preds == labels).mean()
    precision, recall, f1_binary, _ = precision_recall_fscore_support(labels, preds, average='binary')
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-Score (Binary): {f1_binary:.4f}")
    print(f"F1-Score (Macro):  {f1_macro:.4f}")