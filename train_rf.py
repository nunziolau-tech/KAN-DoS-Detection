import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LETTURA E CAMPIONAMENTO (Identico alla KAN)
print("Lettura del dataset con Polars in corso...")
df = pl.read_csv("archive/CICIOT23/train/train.csv") 

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
print(f"Dataset bilanciato per il training RF: {df_balanced.height} campioni totali.")

# Estrazione e codifica etichette
X = df_balanced.drop("label").to_numpy()
y_text = df_balanced["label"].to_numpy()

le = LabelEncoder()
y = le.fit_transform(y_text)

# 2. SPLIT E SCALING
print("Esecuzione dello split e normalizzazione...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. TRAINING RANDOM FOREST
print("Addestramento Random Forest in corso...")
# n_jobs=-1 usa tutti i core della CPU
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# 4. VALUTAZIONE FINALE
y_pred = rf.predict(X_test)

print("\n--- CLASSIFICATION REPORT DETTAGLIATO (RANDOM FOREST) ---")
print(classification_report(y_test, y_pred, digits=4))

print(f"F1-Score (Macro) Random Forest: {f1_score(y_test, y_pred, average='macro'):.4f}")

# Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=False, cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest (Balanced Sampling)')
plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("Matrice salvata come 'confusion_matrix_rf.png'.")