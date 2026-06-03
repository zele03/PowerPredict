# GRU Training Instructions

This document explains how to run the GRU model after cloning or pulling the
project.

The default project workflow does not train the GRU model, because GRU training
requires a PyTorch environment and can be much slower than preprocessing,
baseline evaluation, and baseline visualization.

## 1. Run the default pipeline

Run this first:

```bash
uv run main.py
```

Equivalent commands:

```bash
uv run python main.py
python main.py
```

This runs:

```text
step01 preprocessing
step02 feature engineering
step03 time-based train/validation/test split
step04 baseline model
step05 baseline visualizations
```

It creates the files needed before GRU training, including:

```text
data/processed/split/train.csv
data/processed/split/validation.csv
data/processed/split/test.csv
data/logs/baseline_report.csv
data/graphs/baseline_graphs/
```

## 2. Prepare PyTorch

GRU training requires PyTorch. GPU/CUDA support is recommended, but the model can
also run on CPU.

Check the PyTorch installation:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `True`, the model will use GPU.

If PyTorch is missing or the installed version does not match your CUDA setup,
install PyTorch using the official command for your machine:

```text
https://pytorch.org/get-started/locally/
```

## 3. Train the GRU model

After the default pipeline is finished, run:

```bash
uv run main.py gru-train
```

Equivalent module command:

```bash
uv run python -m src.step06_gru_model
```

This creates:

```text
models/gru_model.pt
data/logs/gru_training_history.csv
data/logs/gru_report.csv
data/logs/baseline_gru_compare_report.csv
data/predictions/gru_validation_predictions.csv
data/predictions/gru_test_predictions.csv
```

The model is trained only on the train split. Validation is used for early
stopping. Test is used only for final evaluation.

## 4. Create GRU graphs

After training, run:

```bash
uv run main.py gru-visualizations
```

This creates both GRU-only graphs and baseline-vs-GRU comparison graphs:

```text
data/graphs/gru_graphs/
data/graphs/baseline_gru_compare_graphs/
```

GRU-only graphs:

```text
gru_test_hourly_wape.png
gru_test_hourly_mae.png
gru_test_actual_vs_predicted_scatter.png
gru_test_error_distribution.png
```

Comparison graphs:

```text
baseline_gru_compare_test_hourly_wape.png
baseline_gru_compare_test_hourly_mae.png
baseline_gru_compare_test_actual_vs_predicted_scatter.png
baseline_gru_compare_test_error_distribution.png
```

## 5. Optional commands

Run only the GRU graphs:

```bash
uv run main.py gru-graphs
```

Run only the baseline-vs-GRU comparison graphs:

```bash
uv run main.py gru-compare-graphs
```

Run GRU training and all GRU visualizations in one command:

```bash
uv run main.py gru-pipeline
```

Use `gru-pipeline` only when the machine already has a working PyTorch setup.

## 6. Main command summary

```bash
uv run main.py                    # default baseline workflow
uv run main.py baseline           # same as default
uv run main.py gru-train          # train GRU model
uv run main.py gru-graphs         # create GRU-only graphs
uv run main.py gru-compare-graphs # create comparison graphs
uv run main.py gru-visualizations # create all GRU graphs
uv run main.py gru-pipeline       # train GRU and create all GRU graphs
```
