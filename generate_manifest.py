import hashlib
import os

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

base_path = r"C:\Users\Admin\Desktop\Scuola\Tesi\CICIoT2023\archive\CICIOT23\cleaned"
files = ["train_clean.csv", "validation_clean.csv", "test_clean.csv"]

print("--- MANIFEST DATASET BONIFICATI ---")
for file in files:
    path = os.path.join(base_path, file)
    if os.path.exists(path):
        print(f"{file}: {calculate_sha256(path)}")
    else:
        print(f"[ERRORE] File non trovato: {path}")