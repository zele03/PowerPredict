# Uputstvo Za Treniranje GRU Modela

Ovaj dokument objasnjava kako se pokrece GRU model nakon kloniranja ili
pull-ovanja projekta.

Podrazumevani workflow projekta ne trenira GRU model, zato sto GRU trening
zahteva PyTorch okruzenje i moze da traje znatno duze od preprocessinga,
baseline evaluacije i baseline vizualizacije.

## 1. Pokrenuti Osnovni Pipeline

Prvo pokrenuti:

```bash
uv run main.py
```

Ekvivalentne komande:

```bash
uv run python main.py
python main.py
```

Ovo pokrece:

```text
step01 preprocessing
step02 feature engineering
step03 vremenski train/validation/test split
step04 baseline model
step05 baseline vizualizacije
```

Pre GRU treninga treba da postoje sledeci fajlovi:

```text
data/processed/split/train.csv
data/processed/split/validation.csv
data/processed/split/test.csv
data/logs/baseline_report.csv
data/graphs/baseline_graphs/
```

## 2. Pripremiti PyTorch

GRU trening zahteva PyTorch. GPU/CUDA podrska je preporucena, ali model moze da
radi i na CPU-u.

Provera PyTorch instalacije:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Ako `torch.cuda.is_available()` ispise `True`, model ce koristiti GPU.

Ako PyTorch nije instaliran ili instalirana verzija ne odgovara CUDA setup-u na
toj masini, instalirati PyTorch zvanicnom komandom za konkretno okruzenje:

```text
https://pytorch.org/get-started/locally/
```

## 3. Trenirati GRU Model

Kada se osnovni pipeline zavrsi, pokrenuti:

```bash
uv run main.py gru-train
```

Ekvivalentna module komanda:

```bash
uv run python -m src.step06_gru_model
```

Ovaj korak kreira:

```text
models/gru_model.pt
data/logs/gru_training_history.csv
data/logs/gru_report.csv
data/logs/baseline_gru_compare_report.csv
data/predictions/gru_validation_predictions.csv
data/predictions/gru_test_predictions.csv
```

Model se trenira samo na train skupu. Validation skup se koristi za early
stopping. Test skup se koristi samo za finalnu evaluaciju.

## 4. Kreirati GRU Grafike

Nakon treninga, pokrenuti:

```bash
uv run main.py gru-visualizations
```

Ova komanda kreira i GRU-only grafike i baseline-vs-GRU uporedne grafike:

```text
data/graphs/gru_graphs/
data/graphs/baseline_gru_compare_graphs/
```

GRU-only grafici:

```text
gru_test_hourly_wape.png
gru_test_hourly_mae.png
gru_test_actual_vs_predicted_scatter.png
gru_test_error_distribution.png
```

Uporedni grafici:

```text
baseline_gru_compare_test_hourly_wape.png
baseline_gru_compare_test_hourly_mae.png
baseline_gru_compare_test_actual_vs_predicted_scatter.png
baseline_gru_compare_test_error_distribution.png
```

## 5. Opcione Komande

Pokrenuti samo GRU grafike:

```bash
uv run main.py gru-graphs
```

Pokrenuti samo baseline-vs-GRU uporedne grafike:

```bash
uv run main.py gru-compare-graphs
```

Pokrenuti GRU trening i sve GRU vizualizacije jednom komandom:

```bash
uv run main.py gru-pipeline
```

Komandu `gru-pipeline` koristiti samo kada masina vec ima ispravno podeseno
PyTorch okruzenje.

## 6. Pregled Glavnih Komandi

```bash
uv run main.py                    # osnovni baseline workflow
uv run main.py baseline           # isto kao osnovno pokretanje
uv run main.py gru-train          # trenira GRU model
uv run main.py gru-graphs         # pravi samo GRU grafike
uv run main.py gru-compare-graphs # pravi uporedne grafike
uv run main.py gru-visualizations # pravi sve GRU grafike
uv run main.py gru-pipeline       # trenira GRU i pravi sve GRU grafike
```
