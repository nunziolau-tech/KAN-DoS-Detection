# CICIoT2023 - Benchmark Framework (Multiclass)

## Obiettivo
Infrastruttura di addestramento riproducibile per il confronto multiclasse (34 categorie) tra algoritmi basati su alberi (Random Forest, XGBoost, LightGBM) e reti neurali (Kolmogorov-Arnold Network) sul dataset CICIoT2023 bonificato.

## Esecuzione
1. Attivare l'ambiente virtuale locale.
2. Installare le dipendenze: `pip install -r requirements.txt`.
3. I percorsi assoluti dei dataset (Train, Validation, Test) e delle directory di output per i checkpoint sono configurabili nelle costanti `BASE_PATH` e `OUT_DIR` all'inizio del file `run_experiments.py`.

### Modalità Smoke (Verifica Infrastrutturale Rapida)
Esegue un campionamento stratificato all'1% dei dati e tronca le epoche di addestramento. Da utilizzare esclusivamente per validare l'integrità del codice, il salvataggio dei pesi e la pipeline di inferenza (profilazione CUDA).
`python run_experiments.py --smoke`

### Modalità Definitiva
Avvia il calcolo completo sul 100% del dataset per i 5 seed definitivi. Include calcolo dei pesi bilanciati sul Train set, early stopping (patience 10) e aggregazione finale delle metriche (Macro-F1, Balanced Accuracy) sul Test set.
`python run_experiments.py`