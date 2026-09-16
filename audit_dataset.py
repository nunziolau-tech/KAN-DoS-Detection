import polars as pl

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
TRAIN_PATH = "archive/CICIOT23/train/train.csv"
VAL_PATH = "archive/CICIOT23/validation/validation.csv"
TEST_PATH = "archive/CICIOT23/test/test.csv"

def main():
    print("==================================================")
    print("AVVIO AUDIT FORENSE DATASET CICIOT2023")
    print("==================================================\n")
    
    print("1. Lettura dei dataset fisici e iniezione metadata di partizione...")
    df_train = pl.read_csv(TRAIN_PATH).with_columns(pl.lit("train").alias("partition"))
    df_val = pl.read_csv(VAL_PATH).with_columns(pl.lit("val").alias("partition"))
    df_test = pl.read_csv(TEST_PATH).with_columns(pl.lit("test").alias("partition"))

    # Estrazione dinamica delle feature (escludendo 'label' e 'partition')
    feature_cols = [c for c in df_train.columns if c not in ['label', 'partition']]

    # Calcolo distribuzione per rispondere al Punto 1 del Relatore
    dist_train = df_train.group_by("label").len().rename({"len": "train_count"})
    dist_val = df_val.group_by("label").len().rename({"len": "val_count"})
    dist_test = df_test.group_by("label").len().rename({"len": "test_count"})
    
   # Estrazione di tutte le label uniche dai tre set
    unique_labels = pl.concat([
        dist_train.select("label"), 
        dist_val.select("label"), 
        dist_test.select("label")
    ]).unique()
    
    # Left join progressivo, senza conflitti di colonne
    dist_all = (
        unique_labels
        .join(dist_train, on="label", how="left")
        .join(dist_val, on="label", how="left")
        .join(dist_test, on="label", how="left")
        .fill_null(0)
        .sort("label")
    )
    
    print("\n--- RISPOSTA PUNTO 1: DISTRIBUZIONE CLASSI ---")
    with pl.Config(tbl_rows=40):
        print(dist_all)

    print("\n2. Unione dei dataframe in RAM per analisi incrociata...")
    df_all = pl.concat([df_train, df_val, df_test])
    
    print("3. Scansione raggruppata per feature esatte (Attenzione: operazione intensiva)...")
    # Raggruppiamo le righe che hanno esattamente gli stessi valori matematici
    grouped = df_all.group_by(feature_cols).agg([
        pl.col("label").unique().alias("unique_labels"),
        pl.col("partition").unique().alias("unique_partitions"),
        pl.len().alias("count")
    ])

    # Isoliamo i gruppi che compaiono più di una volta
    anomalies = grouped.filter(pl.col("count") > 1)

    # LEAKAGE: Stesse feature, stessa label, ma sparpagliate in partizioni diverse
    leakage = anomalies.filter(
        (pl.col("unique_labels").list.len() == 1) & 
        (pl.col("unique_partitions").list.len() > 1)
    )
    
    # COLLISIONI: Stesse feature, ma etichette DIVERSE (Errore grave del dataset originario)
    collisions = anomalies.filter(
        pl.col("unique_labels").list.len() > 1
    )

    print("\n==================================================")
    print("RISULTATI AUDIT - PUNTO 2 DEL RELATORE")
    print("==================================================")
    
    print(f"\n[LEAKAGE INTER-PARTIZIONE]: Trovati {leakage.height} record unici duplicati tra Train/Val/Test.")
    if leakage.height > 0:
        print("Esempio di partizioni inquinate (Le stesse righe appaiono in partizioni multiple):")
        print(leakage.select(["unique_labels", "unique_partitions", "count"]).head(5))

    print(f"\n[COLLISIONI SEMANTICHE]: Trovati {collisions.height} record con feature identiche ma ETICHETTE OPPOSTE.")
    if collisions.height > 0:
        print("Esempio di collisioni (Es. la rete fa la stessa cosa, ma il dataset dice due attacchi diversi):")
        print(collisions.select(["unique_labels", "unique_partitions", "count"]).head(5))

if __name__ == "__main__":
    main()