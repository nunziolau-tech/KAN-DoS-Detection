# CICIoT2023 - Benchmark Framework (Multiclass)

## Obiettivo
Confronto multiclasse (34 categorie) tra RF, XGBoost, LightGBM e KAN.

## Esecuzione
I percorsi sono configurabili in testata a `run_experiments.py`.
Le metriche (JSON), le matrici di confusione (NPY) e i metadati (Scaler, Label Map) vengono salvati dinamicamente.

### Modalità Smoke (Diagnostica)
Usa Train e Validation (campionati all'1% con stratificazione `train_test_split`). **Non carica il Test set.** Salva gli artefatti in `results/smoke`.
`python run_experiments.py --smoke`

### Modalità Definitiva
Addestra i 4 modelli sui 5 seed (intero dataset). Salva in `results/final`.
`python run_experiments.py`