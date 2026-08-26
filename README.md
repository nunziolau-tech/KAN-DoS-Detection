# KAN per Intrusion Detection (CICIoT2023) - Binary Classification

## Obiettivo
Implementazione di una Kolmogorov-Arnold Network (KAN) in PyTorch per la classificazione binaria (Benign vs Attack) sul dataset CICIoT2023. Il progetto è stato strutturato per affrontare le criticità tipiche del network traffic analysis, in particolare l'estremo sbilanciamento delle classi.

## Soluzioni Architetturali Implementate
Per garantire un addestramento robusto e metriche affidabili, sono stati applicati i seguenti accorgimenti:

*   **Campionamento Stratificato:** Il subsampling del dataset è stato eseguito preservando la distribuzione originale degli attacchi rari. È stato garantito l'inserimento di almeno un campione per le classi minoritarie microscopiche.
*   **Gestione dello Sbilanciamento in Training:** Per prevenire il collasso della rete sulla classe maggioritaria, la funzione di costo (`BCEWithLogitsLoss`) integra un peso dinamico (`pos_weight`). Il peso viene calcolato a runtime in base al rapporto esatto tra campioni negativi e positivi nel set di addestramento.
*   **Metriche di Valutazione Non Polarizzate:** Oltre alle standard Accuracy e F1-Binary, le performance vengono misurate utilizzando la metrica **Macro-F1**, per valutare la capacità di generalizzazione del modello al netto della frequenza delle singole classi.

## Struttura del Notebook
Il codice è ottimizzato per l'esecuzione su Google Colab ed è diviso in tre blocchi logici per evitare ricaricamenti inutili in RAM:
1.  **Setup & Ingestion:** Caricamento del file CSV.
2.  **Preprocessing & Tensorization:** Stratificazione, scaling (StandardScaler), pulizia garbage collector e push dei tensori su GPU.
3.  **Model Training & Evaluation:** Istanziazione della KAN, calcolo pesi della Loss, addestramento ed estrazione delle metriche.
