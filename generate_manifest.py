import polars as pl
import hashlib
import os
import json

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
    return h.hexdigest()

base_path = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\archive\CICIOT23\cleaned"
files = {"Train": "train_clean.csv", "Validation": "validation_clean.csv", "Test": "test_clean.csv"}

print("=== MANIFEST DATASET BONIFICATI ===")
dfs = {}
for name, filename in files.items():
    path = os.path.join(base_path, filename)
    df = pl.read_csv(path)
    dfs[name] = df
    print(f"\n[{name}] {filename}")
    print(f"- SHA-256: {get_sha256(path)}")
    print(f"- Righe: {df.height} | Colonne: {df.width}")
    print(f"- Ordine Feature (prime 5): {df.columns[:5]} ...")
    
    class_dist = df.select("label").to_series().value_counts().to_dicts()
    print(f"- Distribuzione Classi: {json.dumps(class_dist)[:100]}... (troncato)")

print("\n=== AUDIT ZERO EXACT-FEATURE OVERLAP ===")
feat_cols = [c for c in dfs["Train"].columns if c != 'label']
train_f = dfs["Train"].select(feat_cols).unique()
val_f = dfs["Validation"].select(feat_cols).unique()
test_f = dfs["Test"].select(feat_cols).unique()

print(f"Train Intersezione Validation: {train_f.join(val_f, on=feat_cols, how='inner').height}")
print(f"Train Intersezione Test:       {train_f.join(test_f, on=feat_cols, how='inner').height}")
print(f"Validation Intersezione Test:  {val_f.join(test_f, on=feat_cols, how='inner').height}")