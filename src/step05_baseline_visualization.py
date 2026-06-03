from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMN = "Energy_kWh"
BASELINE_COLUMN = "lag_24h"
ERROR_BIN_WIDTH = 0.1
ERROR_X_TICK_STEP = 0.5
ERROR_Y_TICK_STEP = 200


def load_split_datasets():
    """
    Ucitava test skup iz split foldera.
    """
    split_dir = Path("data/processed/split")

    test_df = pd.read_csv(split_dir / "test.csv", parse_dates=["datetime"])

    return {
        "test": test_df,
    }


def load_baseline_report():
    """
    Ucitava baseline report sa ukupnim i hourly metrikama.
    """
    report_path = Path("data/logs/baseline_report.csv")
    report_df = pd.read_csv(report_path)

    return report_df


def add_error_columns(df):
    """
    Dodaje kolone sa baseline predikcijom i apsolutnom greskom.
    """
    df = df.copy()
    df["predicted_Energy_kWh"] = df[BASELINE_COLUMN]
    df["absolute_error"] = (df[TARGET_COLUMN] - df["predicted_Energy_kWh"]).abs()

    return df


def load_gru_error_values_if_available():
    """
    Ucitava GRU apsolutne greske ako su predikcije vec generisane.
    """
    predictions_path = Path("data/predictions/gru_test_predictions.csv")

    if not predictions_path.exists():
        return None

    predictions_df = pd.read_csv(predictions_path)
    return predictions_df["absolute_error"].to_numpy()


def create_error_axis_config(error_values_list):
    """
    Racuna zajednicke binove i podeoke za error distribution grafike.
    """
    max_error = max(error_values.max() for error_values in error_values_list)
    x_max = np.ceil(max_error / ERROR_X_TICK_STEP) * ERROR_X_TICK_STEP
    x_max = max(x_max, ERROR_X_TICK_STEP)
    bins = np.arange(0, x_max + ERROR_BIN_WIDTH, ERROR_BIN_WIDTH)

    max_count = max(
        np.histogram(error_values, bins=bins)[0].max()
        for error_values in error_values_list
    )
    y_max = np.ceil(max_count / ERROR_Y_TICK_STEP) * ERROR_Y_TICK_STEP
    y_max = max(y_max, ERROR_Y_TICK_STEP)

    return {
        "bins": bins,
        "x_lim": (0, x_max),
        "y_lim": (0, y_max),
        "x_ticks": np.arange(0, x_max + ERROR_X_TICK_STEP, ERROR_X_TICK_STEP),
        "y_ticks": np.arange(0, y_max + ERROR_Y_TICK_STEP, ERROR_Y_TICK_STEP),
    }


def apply_error_axis_config(ax, axis_config):
    """
    Primenjuje zajednicke ose i podeoke za error distribution grafike.
    """
    ax.set_xlim(axis_config["x_lim"])
    ax.set_ylim(axis_config["y_lim"])
    ax.set_xticks(axis_config["x_ticks"])
    ax.set_yticks(axis_config["y_ticks"])


def plot_hourly_metric(report_df, dataset_name, metric, graphs_dir):
    """
    Prikazuje hourly metriku iz baseline reporta.
    """
    hourly_df = report_df[
        (report_df["Report_Type"] == "hourly")
        & (report_df["Dataset"] == dataset_name)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(hourly_df["Hour"], hourly_df[metric], color="tab:blue")

    ax.set_title(f"Baseline hourly {metric} - {dataset_name}")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / f"baseline_{dataset_name}_hourly_{metric.lower()}.png")
    plt.close(fig)


def plot_actual_vs_predicted_scatter(df, dataset_name, graphs_dir):
    """
    Prikazuje odnos stvarnih i predvidjenih vrednosti.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        df[TARGET_COLUMN],
        df["predicted_Energy_kWh"],
        alpha=0.35,
        s=12,
    )

    min_value = min(df[TARGET_COLUMN].min(), df["predicted_Energy_kWh"].min())
    max_value = max(df[TARGET_COLUMN].max(), df["predicted_Energy_kWh"].max())
    ax.plot([min_value, max_value], [min_value, max_value], color="tab:red")

    ax.set_title(f"Baseline actual vs predicted scatter - {dataset_name}")
    ax.set_xlabel("Actual Energy_kWh")
    ax.set_ylabel("Predicted Energy_kWh")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / f"baseline_{dataset_name}_actual_vs_predicted_scatter.png")
    plt.close(fig)


def plot_error_distribution(df, dataset_name, graphs_dir, axis_config):
    """
    Prikazuje distribuciju apsolutnih gresaka baseline modela.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["absolute_error"],
        bins=axis_config["bins"],
        color="tab:purple",
        edgecolor="black",
    )

    ax.set_title(f"Baseline error distribution - {dataset_name}")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    apply_error_axis_config(ax, axis_config)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / f"baseline_{dataset_name}_error_distribution.png")
    plt.close(fig)


def create_baseline_graphs():
    """
    Kreira graficke prikaze baseline modela za test skup.
    """
    graphs_dir = Path("data/graphs/baseline_graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    split_datasets = load_split_datasets()
    report_df = load_baseline_report()

    for dataset_name, df in split_datasets.items():
        df = add_error_columns(df)
        error_values_list = [df["absolute_error"].to_numpy()]
        gru_error_values = load_gru_error_values_if_available()

        if gru_error_values is not None:
            error_values_list.append(gru_error_values)

        error_axis_config = create_error_axis_config(error_values_list)

        plot_hourly_metric(report_df, dataset_name, "WAPE", graphs_dir)
        plot_hourly_metric(report_df, dataset_name, "MAE", graphs_dir)
        plot_actual_vs_predicted_scatter(df, dataset_name, graphs_dir)
        plot_error_distribution(df, dataset_name, graphs_dir, error_axis_config)


if __name__ == "__main__":
    create_baseline_graphs()
