import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import gc
import time

# 1. Caricamento Dataset (Stessa pipeline della KAN)
file_path = 'archive/CICIOT23/train/train.csv'
print("Lettura del dataset con Polars in corso...")

df_pl = pl.read_csv(file_path)
label_col = 'label' if 'label' in df_pl.columns else 'Label'
labels_raw = df_pl[label_col].to_numpy()

feature_cols = [c for c in df_pl.columns if c != label_col]
X_np = df_pl.select(feature_cols).to_numpy().astype("float32")
del df_pl
gc.collect()

# 2. Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels_raw)

# 3. Campionamento Stratificato INIZIALE (Esattamente 150k campioni come la KAN)
max_total_samples = 150000
fraction_to_keep = max_total_samples / len(X_np)
X_np, _, y_encoded, _ = train_test_split(
    X_np, y_encoded, train_size=fraction_to_keep, random_state=42, stratify=y_encoded
)

# Split Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Normalizzazione (Fit solo sul train per evitare Data Leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Addestramento Random Forest Benchmark
print("\nAvvio addestramento Random Forest (100 alberi) sui thread della CPU...")
start_time = time.time()

# class_weight='balanced' gestisce lo sbilanciamento internamente
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

end_time = time.time()
print(f"Addestramento completato in {end_time - start_time:.2f} secondi.")

# 6. Valutazione
preds = rf.predict(X_test)

accuracy = (preds == y_test).mean()
precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
_, _, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)

print(f"\n--- CONFRONTO RANDOM FOREST (BASELINE) ---")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro):    {recall:.4f}")
print(f"F1-Score (Macro):  {f1_macro:.4f}")
print(f"F1-Score (Wgt):    {f1_weighted:.4f}")