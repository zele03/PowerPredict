# PowerPredict

PowerPredict is a student project for forecasting household electricity
consumption from the Household Power Consumption dataset.

The system predicts hourly energy consumption and evaluates models on validation
and test splits using WAPE, MAE, RMSE, MAPE, and sMAPE.

## Project Structure

```text
main.py
src/
data/
docs/
models/
```

Main workflow steps:

```text
src/step01_preprocessing.py
src/step02_features.py
src/step03_split.py
src/step04_baseline_model.py
src/step05_baseline_visualization.py
src/step06_gru_model.py
src/step07_gru_visualization.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/zele03/PowerPredict.git
cd PowerPredict
```

Create the virtual environment and install dependencies with `uv`:

```bash
uv sync
```

## Default Run

Run the standard pipeline:

```bash
uv run main.py
```

This runs preprocessing, feature engineering, train/validation/test split,
baseline evaluation, and baseline visualizations.

It does not train the GRU model.

## GRU Model

The GRU model is not executed by default because it requires a PyTorch setup and
can benefit from CUDA/GPU support.

After the default pipeline is finished, train GRU explicitly:

```bash
uv run main.py gru-train
```

Then generate GRU and baseline-vs-GRU graphs:

```bash
uv run main.py gru-visualizations
```

For detailed GRU setup and command explanations, see:

```text
docs/gru_training_instructions.md
```

## Main Commands

```bash
uv run main.py                    # default baseline workflow
uv run main.py baseline           # same as default
uv run main.py gru-train          # train GRU model
uv run main.py gru-graphs         # create GRU-only graphs
uv run main.py gru-compare-graphs # create comparison graphs
uv run main.py gru-visualizations # create all GRU graphs
uv run main.py gru-pipeline       # train GRU and create all GRU graphs
```

## Documentation

Detailed project documentation:

```text
docs/documentation.md
```
