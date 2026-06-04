# Uputstvo Za Treniranje GRU Modela

Ovaj dokument objasnjava kako se pokreće GRU deo projekta nakon kloniranja ili
pull-ovanja projekta.

Osnovni workflow ne trenira GRU model, zato što GRU trening zahteva PyTorch
okruženje i može da traje znatno duze od preprocessinga, baseline evaluacije i
baseline vizualizacije, pogotovo zato što trenira više modela sa različitim
hiperparametrima jedan za drugim.

## 1. Pokrenuti Osnovni Pipeline

Prvo pokrenuti:

```bash
uv run main.py
```

Ovo pokreće:

```text
step01 preprocessing
step02 feature engineering
step03 vremenski train/validation/test split
step04 baseline model
step05 baseline vizualizacije
```

Pre GRU treninga treba da postoje:

```text
data/processed/split/train.csv
data/processed/split/validation.csv
data/processed/split/test.csv
data/logs/baseline_report.csv
data/graphs/baseline_graphs/
```

## 2. Pripremiti PyTorch

GRU trening zahteva PyTorch. GPU/CUDA podrška je preporučena, ali model može da
radi i na CPU-u.

Provera PyTorch instalacije:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Ako `torch.cuda.is_available()` ispise `True`, model ce koristiti GPU.

Ako PyTorch nije instaliran ili instalirana verzija ne odgovara CUDA setup-u na
toj mašini, instalirati PyTorch zvaničnom komandom za konkretno okruzenje:

```text
https://pytorch.org/get-started/locally/
```

## 3. Pokrenuti GRU Pipeline

Kada se osnovni pipeline završi, pokrenuti:

```bash
uv run main.py gru-pipeline
```

Ova komanda radi:

```text
step06 treniranje svih GRU konfiguracija iz HYPERPARAMETER_CONFIGS
step07 crtanje grafova za sve GRU modele
step07 crtanje baseline vs najbolji GRU grafova
```

GRU modeli se treniraju jedan po jedan, redom iz liste
`HYPERPARAMETER_CONFIGS` u `src/step06_gru_model.py`.

## 4. Kako Se Bira Najbolji GRU

Svaki GRU model se trenira na train skupu.

Validation skup se koristi za:

```text
early stopping
izbor najboljih hiperparametara
```

Najbolji GRU se bira po:

```text
validation_MAE
```

Test skup se koristi za:

```text
finalnu evaluaciju
GRU grafove
baseline vs najbolji GRU compare report
baseline vs najbolji GRU uporedne grafove
```

Ovo znači da se hiperparametri ne biraju po test skupu, već se test koristi kao
finalna provera modela.

## 5. Output Fajlovi

Trenirani modeli:

```text
models/{model_name}.pt
```

GRU predikcije:

```text
data/predictions/gru_predictions/{model_name}_validation_predictions.csv
data/predictions/gru_predictions/{model_name}_test_predictions.csv
```

GRU logovi:

```text
data/logs/gru_logs/{model_name}_training_history.csv
data/logs/gru_logs/{model_name}_report.csv
data/logs/gru_logs/gru_hyperparameter_report.csv
data/logs/gru_logs/gru_model_summaries.csv
data/logs/gru_logs/best_gru_model_summary.csv
```

Compare logovi:

```text
data/logs/compare_logs/baseline_gru_compare_report.csv
```

GRU grafovi za svaki model:

```text
data/graphs/gru_graphs/{model_name}_test_hourly_wape.png
data/graphs/gru_graphs/{model_name}_test_hourly_mae.png
data/graphs/gru_graphs/{model_name}_test_actual_vs_predicted_scatter.png
data/graphs/gru_graphs/{model_name}_test_error_distribution.png
```

Baseline vs najbolji GRU grafovi:

```text
data/graphs/baseline_gru_compare_graphs/
```

Nazivi grafova sadrže naziv najboljeg GRU modela, a na samim grafovima su
ispisani hiperparametri i glavne metrike.

## 6. Dodavanje Novih GRU Modela

Za dodavanje novog modela dodati novu konfiguraciju u
`HYPERPARAMETER_CONFIGS`.

Primer:

```python
{
    "name": "gru_seq48_h128_l2_d0.3_lr0.0005_bs64",
    "sequence_length": 48,
    "batch_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 0.0005,
    "max_epochs": 80,
    "patience": 10,
}
```

Polje `name` mora biti jedinstveno, jer se koristi u nazivima modela,
predikcija, reportova i grafova.

## 7. Glavne Komande

```bash
uv run main.py              # osnovni baseline workflow
uv run main.py gru-pipeline # trenira sve GRU modele, poredi ih i pravi GRU grafove
```
