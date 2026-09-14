
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from kan import KAN
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configurazione Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo di addestramento attivo: {device}")

# 1. LETTURA E CAMPIONAMENTO ASIMMETRICO (Prof. Req. #2)
print("Lettura del dataset con Polars in corso...")
# Sostituisci col tuo percorso file corretto
df = pl.read_csv("archive/CICIOT23/train/train.csv")

print(f"Totale righe originarie caricate: {df.height}")

MIN_SAMPLES = 300    # Soglia di sopravvivenza per le classi rare
MAX_SAMPLES = 10000  # Tappo per le classi volumetriche (es. DDoS) per non saturare la RAM

dfs_sampled = []
# Raggruppa per classe ed esegui il campionamento logico
for label, group in df.group_by('label'):
    n_samples = group.height
    if n_samples < MIN_SAMPLES:
        # Classe rara: prendi il 100% dei dati, zero tagli.
        dfs_sampled.append(group)
    elif n_samples > MAX_SAMPLES:
        # Classe volumetrica: sottocampiona pesantemente
        dfs_sampled.append(group.sample(n=MAX_SAMPLES, seed=42))
    else:
        # Classe intermedia: prendi tutto
        dfs_sampled.append(group)

df_balanced = pl.concat(dfs_sampled)
print(f"Dataset bilanciato per il training: {df_balanced.height} campioni totali.")

# Estrazione feature e target
X = df_balanced.drop("label").to_numpy()
y_text = df_balanced["label"].to_numpy() 

# Traduce le etichette di testo in numeri (0-33)
le = LabelEncoder()
y = le.fit_transform(y_text)

del df, df_balanced, dfs_sampled # Pulizia RAM

# 2. SPLIT E SCALING (Nessun Data Leakage)
print("Esecuzione dello split Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

print("Normalizzazione delle feature con StandardScaler (fit SOLO sul train)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversione in Tensori PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)

# 3. CLASS WEIGHTING DINAMICO
print("Calcolo dei pesi per compensare lo sbilanciamento...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

# 4. INIZIALIZZAZIONE MODELLO 
print("Inizializzazione del modello KAN Multi-Classe...")
input_dim = X_train_t.shape[1]
model = KAN(width=[input_dim, 32, 16, 34], grid=5, k=3, seed=42).to(device)

# Loss e Ottimizzatore
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. CONVERGENZA AUTOMATIZZATA (Prof. Req. #3)
# Scheduler: riduce il learning rate se la loss non scende per 5 epoche
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Early Stopping
max_epochs = 250
patience = 15
best_loss = float('inf')
patience_counter = 0

print(f"Avvio training KAN multi-classe (Max {max_epochs} epoche, Patience: {patience})...")

for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    
    # Training iterando sui mini-batch
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
        
    epoch_train_loss = running_loss / len(train_dataset)
    
    # Validation step (fatto in full-batch se il test set è piccolo, altrimenti va batchato anch'esso)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_t)
        val_loss = criterion(val_outputs, y_test_t)
    
    # Lo scheduler osserva la loss di validazione
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")
        
    # Logica Early Stopping
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        patience_counter = 0
        torch.save(model.state_dict(), 'model/best_kan.pth') # Salva sempre il peso migliore
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping innescato all'epoca {epoch+1}. La loss non migliora da {patience} epoche.")
        # Ricarica i pesi migliori prima della valutazione finale
        model.load_state_dict(torch.load('model/best_kan.pth'))
        break

# 6. VALUTAZIONE FINALE E REPORTISTICA (Prof. Req. #1)
print("\n--- VALUTAZIONE FINALE MULTI-CLASSE ---")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, y_pred_t = torch.max(test_outputs, 1)
    y_pred = y_pred_t.cpu().numpy()

# Classification Report Dettagliato
print("\n--- CLASSIFICATION REPORT DETTAGLIATO ---")
print(classification_report(y_test, y_pred, digits=4))

# Metriche globali
print(f"Accuracy:        {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

# Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - KAN Multi-Class (Balanced Sampling)')
plt.savefig('confusion_matrix_kan.png', dpi=300, bbox_inches='tight')
print("Matrice salvata come 'confusion_matrix_kan.png'.")

print("\n--- ANALISI EFFICIENZA (KAN) ---")
# 1. Calcolo parametri
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametri addestrabili KAN: {total_params:,}")

# 2. Calcolo tempo di inferenza (su batch completo)
model.eval()
start_time = time.time()
with torch.no_grad():
    _ = model(X_test_t)
end_time = time.time()

total_time = end_time - start_time
time_per_sample = total_time / len(X_test_t)

print(f"Tempo totale inferenza (su {len(X_test_t)} campioni): {total_time:.4f} secondi")
print(f"Tempo di inferenza per singolo campione: {time_per_sample:.8f} secondi")