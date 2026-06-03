# GRU Training Instructions

Ovaj step se pokrece na masini koja ima PyTorch okruzenje, po mogucnosti sa CUDA/GPU podrskom.

## 1. Provera podataka

Pre treninga treba da postoje:

```text
data/processed/split/train.csv
data/processed/split/validation.csv
data/processed/split/test.csv
data/logs/baseline_report.csv
```

Ako ne postoje, prvo pokrenuti:

```bash
python main.py
```

## 2. Trening GRU modela

Pokrenuti:

```bash
python -m src.step06_gru_model
```

Ovaj korak snima:

```text
models/gru_model.pt
data/logs/gru_training_history.csv
data/logs/gru_report.csv
data/predictions/gru_validation_predictions.csv
data/predictions/gru_test_predictions.csv
```

`gru_report.csv` ima isti princip kao baseline report:

```text
summary redovi za validation i test
hourly redovi za svaki sat 0-23
WAPE, MAE, RMSE, MAPE, sMAPE
```

## 3. GRU grafici

Kada se trening zavrsi, pokrenuti:

```bash
python -m src.step07_gru_visualization
```

Grafici se snimaju u:

```text
data/graphs/gru_graphs/
```

Generisu se samo test grafici:

```text
gru_test_hourly_wape.png
gru_test_hourly_mae.png
gru_test_actual_vs_predicted_scatter.png
gru_test_error_distribution.png
```

## 4. Napomena za PyTorch/CUDA

Ako instalacija `torch` dependency-ja ne povuce GPU verziju, instalirati PyTorch komandom sa zvanicnog PyTorch sajta za konkretnu CUDA verziju na toj masini.
