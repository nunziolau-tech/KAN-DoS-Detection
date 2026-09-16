import polars as pl
import os

TRAIN_PATH = "archive/CICIOT23/train/train.csv"
VAL_PATH = "archive/CICIOT23/validation/validation.csv"
TEST_PATH = "archive/CICIOT23/test/test.csv"
OUT_DIR = "archive/CICIOT23/cleaned"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Lettura dataset originali...")
    df_train = pl.read_csv(TRAIN_PATH)
    df_val = pl.read_csv(VAL_PATH)
    df_test = pl.read_csv(TEST_PATH)
    
    feature_cols = [c for c in df_train.columns if c != 'label']
    
    # 1. Analisi duplicati interni al Test (Richiesta esplicita del Prof)
    print("\n--- ANALISI DUPLICATI INTERNI AL TEST ---")
    test_dupes = df_test.group_by(feature_cols).len().filter(pl.col("len") > 1)
    print(f"Righe con feature duplicate all'interno del Test Set stesso: {test_dupes.height}")
    
    # 2. Bonifica Validation (Elimino dal Val le righe presenti nel Test)
    print("\n--- BONIFICA VALIDATION ---")
    test_features_unique = df_test.select(feature_cols).unique()
    df_val_clean = df_val.join(test_features_unique, on=feature_cols, how="anti")
    print(f"Validation originale: {df_val.height}")
    print(f"Validation bonificato: {df_val_clean.height}")
    print(f"Intersezione Test-Val (rimossa): {df_val.height - df_val_clean.height}")
    
    # 3. Bonifica Train (Elimino dal Train le righe presenti nel Test e nel Val bonificato)
    print("\n--- BONIFICA TRAIN ---")
    val_clean_features_unique = df_val_clean.select(feature_cols).unique()
    
    # Prima rimuovo quelle nel Test
    df_train_step1 = df_train.join(test_features_unique, on=feature_cols, how="anti")
    # Poi rimuovo quelle nel Validation pulito
    df_train_clean = df_train_step1.join(val_clean_features_unique, on=feature_cols, how="anti")
    
    print(f"Train originale: {df_train.height}")
    print(f"Train bonificato: {df_train_clean.height}")
    print(f"Intersezione Test-Train (rimossa): {df_train.height - df_train_step1.height}")
    print(f"Intersezione Val-Train (rimossa): {df_train_step1.height - df_train_clean.height}")
    
    # 4. Verifica di Sopravvivenza delle Classi nel Train
    print("\n--- VERIFICA CLASSI TARGET NEL TRAIN ---")
    train_classes = df_train_clean.select("label").unique().height
    print(f"Classi presenti nel Train bonificato: {train_classes}/34")
    if train_classes < 34:
        print("CRITICO: Alcune classi sono state completamente cancellate dalla bonifica!")
        
    # 5. Salvataggio
    print("\n--- SALVATAGGIO FILE BONIFICATI ---")
    df_test.write_csv(f"{OUT_DIR}/test_clean.csv")
    df_val_clean.write_csv(f"{OUT_DIR}/validation_clean.csv")
    df_train_clean.write_csv(f"{OUT_DIR}/train_clean.csv")
    print("Bonifica completata. Usa la cartella 'cleaned' per addestrare e calcolare lo scaler.")

if __name__ == "__main__":
    main()