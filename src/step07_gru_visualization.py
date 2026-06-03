from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_gru_outputs():
    """
    Ucitava GRU predikcije i report.
    """
    predictions_path = Path("data/predictions/gru_test_predictions.csv")
    report_path = Path("data/logs/gru_report.csv")

    predictions_df = pd.read_csv(predictions_path, parse_dates=["datetime"])
    report_df = pd.read_csv(report_path)

    return predictions_df, report_df


def plot_hourly_metric(report_df, metric, graphs_dir):
    """
    Prikazuje hourly metriku za test skup.
    """
    hourly_df = report_df[
        (report_df["Report_Type"] == "hourly")
        & (report_df["Dataset"] == "test")
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(hourly_df["Hour"], hourly_df[metric], color="tab:green")

    ax.set_title(f"GRU hourly {metric} - test")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / f"gru_test_hourly_{metric.lower()}.png")
    plt.close(fig)


def plot_actual_vs_predicted_scatter(predictions_df, graphs_dir):
    """
    Prikazuje odnos stvarnih i predvidjenih vrednosti.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        predictions_df["actual_Energy_kWh"],
        predictions_df["predicted_Energy_kWh"],
        alpha=0.35,
        s=12,
        color="tab:green",
    )

    min_value = min(
        predictions_df["actual_Energy_kWh"].min(),
        predictions_df["predicted_Energy_kWh"].min(),
    )
    max_value = max(
        predictions_df["actual_Energy_kWh"].max(),
        predictions_df["predicted_Energy_kWh"].max(),
    )
    ax.plot([min_value, max_value], [min_value, max_value], color="tab:red")

    ax.set_title("GRU actual vs predicted scatter - test")
    ax.set_xlabel("Actual Energy_kWh")
    ax.set_ylabel("Predicted Energy_kWh")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / "gru_test_actual_vs_predicted_scatter.png")
    plt.close(fig)


def plot_error_distribution(predictions_df, graphs_dir):
    """
    Prikazuje distribuciju apsolutnih gresaka GRU modela.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        predictions_df["absolute_error"],
        bins=40,
        color="tab:green",
        edgecolor="black",
    )

    ax.set_title("GRU error distribution - test")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / "gru_test_error_distribution.png")
    plt.close(fig)


def create_gru_graphs():
    """
    Kreira graficke prikaze GRU modela za test skup.
    """
    graphs_dir = Path("data/graphs/gru_graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    predictions_df, report_df = load_gru_outputs()

    plot_hourly_metric(report_df, "WAPE", graphs_dir)
    plot_hourly_metric(report_df, "MAE", graphs_dir)
    plot_actual_vs_predicted_scatter(predictions_df, graphs_dir)
    plot_error_distribution(predictions_df, graphs_dir)


if __name__ == "__main__":
    create_gru_graphs()
