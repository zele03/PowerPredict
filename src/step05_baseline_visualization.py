from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGET_COLUMN = "Energy_kWh"
BASELINE_COLUMN = "lag_24h"


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


def plot_error_distribution(df, dataset_name, graphs_dir):
    """
    Prikazuje distribuciju apsolutnih gresaka baseline modela.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["absolute_error"], bins=40, color="tab:purple", edgecolor="black")

    ax.set_title(f"Baseline error distribution - {dataset_name}")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
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

        plot_hourly_metric(report_df, dataset_name, "WAPE", graphs_dir)
        plot_hourly_metric(report_df, dataset_name, "MAE", graphs_dir)
        plot_actual_vs_predicted_scatter(df, dataset_name, graphs_dir)
        plot_error_distribution(df, dataset_name, graphs_dir)


if __name__ == "__main__":
    create_baseline_graphs()
