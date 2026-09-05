import polars as pl
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from kan import KAN
import gc

# 1. Configurazione Dispositivo (RTX 5080)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo di addestramento attivo: {device}")

# 2. Caricamento efficiente con Polars ed estrazione delle feature
file_path = 'archive/CICIOT23/train/train.csv'
print("Lettura del dataset con Polars in corso...")

df_pl = pl.read_csv(file_path)
print(f"Totale righe originarie caricate: {len(df_pl)}")

label_col = 'label' if 'label' in df_pl.columns else 'Label'
labels_raw = df_pl[label_col].to_numpy()

feature_cols = [c for c in df_pl.columns if c != label_col]
X_np = df_pl.select(feature_cols).to_numpy().astype("float32")
del df_pl
gc.collect()
print("RAM liberata dalla struttura Polars.")

# 3. Label Encoding delle 34 classi
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels_raw)
num_classes = len(encoder.classes_)
print(f"Numero totale di classi multi-classe rilevate: {num_classes}")

# 4. Sottocampionamento Stratificato INIZIALE (150k campioni totali per tempi di calcolo realistici)
max_total_samples = 150000
if len(X_np) > max_total_samples:
    fraction_to_keep = max_total_samples / len(X_np)
    print(f"Riduzione del dataset a {max_total_samples} campioni mantenendo la stratificazione (Prof. Req. #1)...")
    X_np, _, y_encoded, _ = train_test_split(
        X_np, y_encoded, train_size=fraction_to_keep, random_state=42, stratify=y_encoded
    )

# Split in Train e Test (80% / 20%)
print("Esecuzione dello split Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Normalizzazione (StandardScaler) - Prevenzione rigorosa del Data Leakage (Prof. Req. #2)
print("Normalizzazione delle feature con StandardScaler (fit SOLO sul train)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Solo transform sul test!

# 6. Calcolo dei pesi di classe per la CrossEntropyLoss (Prof. Req. #3)
print("Calcolo dei pesi per compensare lo sbilanciamento delle 34 classi...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# 7. Conversione in Tensori PyTorch su GPU
dataset = {
    'train_input': torch.FloatTensor(X_train).to(device),
    'train_label': torch.LongTensor(y_train).to(device),
    'test_input': torch.FloatTensor(X_test).to(device),
    'test_label': torch.LongTensor(y_test).to(device)
}

print(f"\n--- DATASET MULTI-CLASSE PRONTO ---")
print(f"Train Input Shape: {dataset['train_input'].shape}")
print(f"Test Input Shape: {dataset['test_input'].shape}")

# 8. Inizializzazione della KAN per Multi-Classe
input_dim = dataset['train_input'].shape[1]
print(f"Inizializzazione del modello KAN Multi-Classe (Output: {num_classes} classi)...")

model = KAN(width=[input_dim, 64, 32, num_classes], grid=5, k=3, seed=42).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Passiamo i pesi bilanciati alla CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

epochs = 70
batch_size = 5000  # Mini-batch training per preservare la VRAM
print(f"Avvio training KAN multi-classe per {epochs} epoche (mini-batch da {batch_size})...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Shuffle dei dati ad ogni epoca
    perm = torch.randperm(len(dataset['train_input']))
    train_x_shuf = dataset['train_input'][perm]
    train_y_shuf = dataset['train_label'][perm]
    
    for i in range(0, len(train_x_shuf), batch_size):
        bx = train_x_shuf[i:i + batch_size]
        by = train_y_shuf[i:i + batch_size]
        
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / (len(train_x_shuf) / batch_size)
        print(f'Epoch [{epoch+1}/{epochs}] | Loss Media: {avg_loss:.4f}')

print("\n--- VALUTAZIONE FINALE MULTI-CLASSE (A BATCH) ---")
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    test_inputs = dataset['test_input']
    test_labels = dataset['test_label']
    
    for i in range(0, len(test_inputs), batch_size):
        batch_X = test_inputs[i:i + batch_size]
        batch_outputs = model(batch_X)
        preds = torch.argmax(batch_outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)

    labels = test_labels.cpu().numpy()

all_preds = np.array(all_preds)

accuracy = (all_preds == labels).mean()
precision, recall, f1_macro, _ = precision_recall_fscore_support(labels, all_preds, average='macro', zero_division=0)
_, _, f1_weighted, _ = precision_recall_fscore_support(labels, all_preds, average='weighted', zero_division=0)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro):    {recall:.4f}")
print(f"F1-Score (Macro):  {f1_macro:.4f}")
print(f"F1-Score (Wgt):    {f1_weighted:.4f}")

print("\n--- GENERAZIONE HEATMAP MATRICE DI CONFUSIONE ---")
cm = confusion_matrix(labels, all_preds)
plt.figure(figsize=(20, 16))
sns.heatmap(cm, xticklabels=encoder.classes_, yticklabels=encoder.classes_, 
            annot=False, cmap='Blues', fmt='g')
plt.xlabel('Classe Predetta', fontsize=14)
plt.ylabel('Classe Reale', fontsize=14)
plt.title('Matrice di Confusione KAN - CICIoT2023 (34 Classi)', fontsize=18)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix_kan.png', dpi=300)
print("Matrice salvata come 'confusion_matrix_kan.png' nella cartella del progetto.")