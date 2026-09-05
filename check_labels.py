import polars as pl

print("Scansione delle etichette in corso...")
lazy_df = pl.scan_csv('archive/CICIOT23/train/train.csv')
labels = lazy_df.select(pl.col('label').unique()).collect()
print("\Classi trovate nel dataset:")
print(labels)