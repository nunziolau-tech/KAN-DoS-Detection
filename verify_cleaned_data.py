import polars as pl

print("Caricamento dataset bonificati per l'audit...")
base_path = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\archive\CICIOT23\cleaned"
df_train = pl.read_csv(f"{base_path}\\train_clean.csv")
df_val = pl.read_csv(f"{base_path}\\validation_clean.csv")
df_test = pl.read_csv(f"{base_path}\\test_clean.csv")

feature_cols = [c for c in df_train.columns if c != 'label']

print("Estrazione feature uniche...")
test_features = df_test.select(feature_cols).unique()
val_features = df_val.select(feature_cols).unique()
train_features = df_train.select(feature_cols).unique()

print("\n--- AUDIT ZERO LEAKAGE (FILE BONIFICATI) ---")
val_test_leak = val_features.join(test_features, on=feature_cols, how="inner").height
train_test_leak = train_features.join(test_features, on=feature_cols, how="inner").height
train_val_leak = train_features.join(val_features, on=feature_cols, how="inner").height

print(f"Intersezione Validation-Test: {val_test_leak}")
print(f"Intersezione Train-Test: {train_test_leak}")
print(f"Intersezione Train-Validation: {train_val_leak}")