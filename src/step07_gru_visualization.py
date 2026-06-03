from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMN = "Energy_kWh"
BASELINE_COLUMN = "lag_24h"
ERROR_BIN_WIDTH = 0.1
ERROR_X_TICK_STEP = 0.5
ERROR_Y_TICK_STEP = 200


def load_gru_outputs():
    """
    Ucitava GRU predikcije i report.
    """
    predictions_path = Path("data/predictions/gru_test_predictions.csv")
    report_path = Path("data/logs/gru_report.csv")

    predictions_df = pd.read_csv(predictions_path, parse_dates=["datetime"])
    report_df = pd.read_csv(report_path)

    return predictions_df, report_df


def load_baseline_outputs():
    """
    Ucitava baseline test skup i baseline report.
    """
    test_path = Path("data/processed/split/test.csv")
    report_path = Path("data/logs/baseline_report.csv")

    test_df = pd.read_csv(test_path, parse_dates=["datetime"])
    report_df = pd.read_csv(report_path)

    return test_df, report_df


def create_compare_predictions_df(baseline_test_df, gru_predictions_df):
    """
    Spaja baseline i GRU predikcije za iste test sate.
    """
    baseline_df = baseline_test_df[
        ["datetime", TARGET_COLUMN, BASELINE_COLUMN]
    ].copy()
    baseline_df = baseline_df.rename(
        columns={
            TARGET_COLUMN: "actual_Energy_kWh",
            BASELINE_COLUMN: "baseline_predicted_Energy_kWh",
        }
    )

    gru_df = gru_predictions_df[
        ["datetime", "actual_Energy_kWh", "predicted_Energy_kWh"]
    ].copy()
    gru_df = gru_df.rename(
        columns={"predicted_Energy_kWh": "gru_predicted_Energy_kWh"}
    )

    compare_df = baseline_df.merge(
        gru_df,
        on=["datetime", "actual_Energy_kWh"],
        how="inner",
    )
    compare_df["baseline_absolute_error"] = (
        compare_df["actual_Energy_kWh"]
        - compare_df["baseline_predicted_Energy_kWh"]
    ).abs()
    compare_df["gru_absolute_error"] = (
        compare_df["actual_Energy_kWh"] - compare_df["gru_predicted_Energy_kWh"]
    ).abs()

    return compare_df


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


def plot_compare_hourly_metric(baseline_report_df, gru_report_df, metric, graphs_dir):
    """
    Prikazuje baseline i GRU hourly metriku za test skup na istom grafiku.
    """
    baseline_hourly_df = baseline_report_df[
        (baseline_report_df["Report_Type"] == "hourly")
        & (baseline_report_df["Dataset"] == "test")
    ].copy()
    gru_hourly_df = gru_report_df[
        (gru_report_df["Report_Type"] == "hourly")
        & (gru_report_df["Dataset"] == "test")
    ].copy()

    compare_df = baseline_hourly_df[["Hour", metric]].merge(
        gru_hourly_df[["Hour", metric]],
        on="Hour",
        suffixes=("_baseline", "_gru"),
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    hours = compare_df["Hour"].astype(int)
    bar_width = 0.4

    ax.bar(
        hours - bar_width / 2,
        compare_df[f"{metric}_baseline"],
        width=bar_width,
        label="Baseline lag_24h",
        color="tab:blue",
    )
    ax.bar(
        hours + bar_width / 2,
        compare_df[f"{metric}_gru"],
        width=bar_width,
        label="GRU",
        color="tab:green",
    )

    ax.set_title(f"Baseline vs GRU hourly {metric} - test")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / f"baseline_gru_compare_test_hourly_{metric.lower()}.png")
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


def plot_compare_actual_vs_predicted_scatter(compare_df, graphs_dir):
    """
    Prikazuje odnos stvarnih i predvidjenih vrednosti za baseline i GRU.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        compare_df["actual_Energy_kWh"],
        compare_df["baseline_predicted_Energy_kWh"],
        alpha=0.28,
        s=12,
        label="Baseline lag_24h",
        color="tab:blue",
    )
    ax.scatter(
        compare_df["actual_Energy_kWh"],
        compare_df["gru_predicted_Energy_kWh"],
        alpha=0.28,
        s=12,
        label="GRU",
        color="tab:green",
    )

    min_value = min(
        compare_df["actual_Energy_kWh"].min(),
        compare_df["baseline_predicted_Energy_kWh"].min(),
        compare_df["gru_predicted_Energy_kWh"].min(),
    )
    max_value = max(
        compare_df["actual_Energy_kWh"].max(),
        compare_df["baseline_predicted_Energy_kWh"].max(),
        compare_df["gru_predicted_Energy_kWh"].max(),
    )
    ax.plot([min_value, max_value], [min_value, max_value], color="tab:red")

    ax.set_title("Baseline vs GRU actual vs predicted scatter - test")
    ax.set_xlabel("Actual Energy_kWh")
    ax.set_ylabel("Predicted Energy_kWh")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        graphs_dir / "baseline_gru_compare_test_actual_vs_predicted_scatter.png"
    )
    plt.close(fig)


def plot_error_distribution(predictions_df, graphs_dir, axis_config):
    """
    Prikazuje distribuciju apsolutnih gresaka GRU modela.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        predictions_df["absolute_error"],
        bins=axis_config["bins"],
        color="tab:green",
        edgecolor="black",
    )

    ax.set_title("GRU error distribution - test")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    apply_error_axis_config(ax, axis_config)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / "gru_test_error_distribution.png")
    plt.close(fig)


def plot_compare_error_distribution(compare_df, graphs_dir, axis_config):
    """
    Prikazuje distribuciju apsolutnih gresaka za baseline i GRU.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = axis_config["bins"]
    baseline_counts, _ = np.histogram(compare_df["baseline_absolute_error"], bins=bins)
    gru_counts, _ = np.histogram(compare_df["gru_absolute_error"], bins=bins)
    bin_width = bins[1] - bins[0]
    baseline_positions = bins[:-1]
    gru_positions = bins[:-1] + bin_width / 2

    ax.bar(
        baseline_positions,
        baseline_counts,
        width=bin_width / 2,
        align="edge",
        label="Baseline lag_24h",
        color="tab:blue",
        edgecolor="black",
    )
    ax.bar(
        gru_positions,
        gru_counts,
        width=bin_width / 2,
        align="edge",
        label="GRU",
        color="tab:green",
        edgecolor="black",
    )

    ax.set_title("Baseline vs GRU error distribution - test")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    ax.legend()
    apply_error_axis_config(ax, axis_config)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / "baseline_gru_compare_test_error_distribution.png")
    plt.close(fig)


def create_gru_graphs():
    """
    Kreira graficke prikaze GRU modela za test skup.
    """
    graphs_dir = Path("data/graphs/gru_graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    predictions_df, report_df = load_gru_outputs()
    baseline_test_df, _ = load_baseline_outputs()
    baseline_test_df = baseline_test_df.copy()
    baseline_test_df["absolute_error"] = (
        baseline_test_df[TARGET_COLUMN] - baseline_test_df[BASELINE_COLUMN]
    ).abs()
    error_axis_config = create_error_axis_config(
        [
            baseline_test_df["absolute_error"].to_numpy(),
            predictions_df["absolute_error"].to_numpy(),
        ]
    )

    plot_hourly_metric(report_df, "WAPE", graphs_dir)
    plot_hourly_metric(report_df, "MAE", graphs_dir)
    plot_actual_vs_predicted_scatter(predictions_df, graphs_dir)
    plot_error_distribution(predictions_df, graphs_dir, error_axis_config)


def create_baseline_gru_compare_graphs():
    """
    Kreira uporedne graficke prikaze baseline i GRU modela za test skup.
    """
    graphs_dir = Path("data/graphs/baseline_gru_compare_graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    gru_predictions_df, gru_report_df = load_gru_outputs()
    baseline_test_df, baseline_report_df = load_baseline_outputs()
    compare_df = create_compare_predictions_df(baseline_test_df, gru_predictions_df)
    error_axis_config = create_error_axis_config(
        [
            compare_df["baseline_absolute_error"].to_numpy(),
            compare_df["gru_absolute_error"].to_numpy(),
        ]
    )

    plot_compare_hourly_metric(baseline_report_df, gru_report_df, "WAPE", graphs_dir)
    plot_compare_hourly_metric(baseline_report_df, gru_report_df, "MAE", graphs_dir)
    plot_compare_actual_vs_predicted_scatter(compare_df, graphs_dir)
    plot_compare_error_distribution(compare_df, graphs_dir, error_axis_config)


if __name__ == "__main__":
    create_gru_graphs()
    create_baseline_gru_compare_graphs()
