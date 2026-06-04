from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMN = "Energy_kWh"
BASELINE_COLUMN = "lag_24h"
BASELINE_MODEL_NAME = "Baseline lag_24h"
ERROR_BIN_WIDTH = 0.1
ERROR_X_TICK_STEP = 0.5
ERROR_Y_TICK_STEP = 200


def load_model_summaries():
    """
    Ucitava indeks svih GRU modela ako ga je step06 napravio.
    """
    summaries_path = Path("data/logs/gru_logs/gru_model_summaries.csv")

    if summaries_path.exists():
        summaries_df = pd.read_csv(summaries_path)
        if "validation_MAE" in summaries_df.columns:
            return summaries_df

    return create_model_summaries_from_reports()


def create_model_summaries_from_reports():
    """
    Rekonstruise summary tabelu iz pojedinacnih GRU reportova.
    """
    summary_rows = []
    existing_summaries_path = Path("data/logs/gru_logs/gru_model_summaries.csv")
    existing_summaries_df = (
        pd.read_csv(existing_summaries_path)
        if existing_summaries_path.exists()
        else pd.DataFrame()
    )

    for report_path in Path("data/logs/gru_logs").glob("*_report.csv"):
        model_name = report_path.stem.removesuffix("_report")
        if model_name == "gru_hyperparameter":
            continue

        report_df = pd.read_csv(report_path)

        validation_summary = report_df[
            (report_df["Report_Type"] == "summary")
            & (report_df["Dataset"] == "validation")
        ]
        test_summary = report_df[
            (report_df["Report_Type"] == "summary")
            & (report_df["Dataset"] == "test")
        ]

        if validation_summary.empty or test_summary.empty:
            continue

        validation_summary = validation_summary.iloc[0]
        test_summary = test_summary.iloc[0]
        existing_row = existing_summaries_df[
            existing_summaries_df.get("Model", pd.Series(dtype=str)) == model_name
        ]
        existing_row = existing_row.iloc[0] if not existing_row.empty else {}

        summary_rows.append(
            {
                "Model": model_name,
                "sequence_length": existing_row.get("sequence_length", ""),
                "batch_size": existing_row.get("batch_size", ""),
                "hidden_size": existing_row.get("hidden_size", ""),
                "num_layers": existing_row.get("num_layers", ""),
                "dropout": existing_row.get("dropout", ""),
                "learning_rate": existing_row.get("learning_rate", ""),
                "validation_MAE": validation_summary["MAE"],
                "validation_RMSE": validation_summary["RMSE"],
                "validation_MAPE": validation_summary["MAPE"],
                "validation_sMAPE": validation_summary["sMAPE"],
                "validation_WAPE": validation_summary["WAPE"],
                "test_MAE": test_summary["MAE"],
                "test_RMSE": test_summary["RMSE"],
                "test_MAPE": test_summary["MAPE"],
                "test_sMAPE": test_summary["sMAPE"],
                "test_WAPE": test_summary["WAPE"],
            }
        )

    return pd.DataFrame(summary_rows)


def get_gru_model_names():
    """
    Vraca nazive svih GRU modela za koje postoje reportovi.
    """
    summaries_df = load_model_summaries()
    if not summaries_df.empty:
        return summaries_df["Model"].tolist()

    report_paths = Path("data/logs/gru_logs").glob("*_report.csv")
    excluded_names = {
        "baseline",
        "baseline_gru_compare",
        "gru_hyperparameter",
    }
    model_names = []

    for report_path in report_paths:
        model_name = report_path.stem.removesuffix("_report")
        if model_name not in excluded_names:
            model_names.append(model_name)

    return sorted(model_names)


def get_model_summary_row(model_name):
    """
    Vraca summary red sa hiperparametrima za trazeni GRU model.
    """
    summaries_df = load_model_summaries()

    if summaries_df.empty:
        return None

    matching_rows = summaries_df[summaries_df["Model"] == model_name]
    if matching_rows.empty:
        return None

    return matching_rows.iloc[0]


def get_graph_details(model_name):
    """
    Pravi tekst koji se ispisuje na grafovima.
    """
    summary_row = get_model_summary_row(model_name)

    if summary_row is None:
        return f"Model: {model_name}"

    return (
        f"Model: {model_name}\n"
        f"seq={summary_row['sequence_length']} | "
        f"hidden={summary_row['hidden_size']} | "
        f"layers={summary_row['num_layers']} | "
        f"dropout={summary_row['dropout']} | "
        f"lr={summary_row['learning_rate']} | "
        f"batch={summary_row['batch_size']}\n"
        f"validation MAE={summary_row['validation_MAE']} | "
        f"test MAE={summary_row['test_MAE']} | "
        f"test RMSE={summary_row['test_RMSE']} | "
        f"test MAPE={summary_row['test_MAPE']} | "
        f"test sMAPE={summary_row['test_sMAPE']} | "
        f"test WAPE={summary_row['test_WAPE']}"
    )


def add_graph_details(fig, details, extra_text=None):
    """
    Dodaje opis modela na sam graf.
    """
    if extra_text:
        details = f"{details}\n{extra_text}"

    fig.text(0.01, 0.01, details, ha="left", va="bottom", fontsize=8)


def load_gru_outputs(model_name):
    """
    Ucitava GRU predikcije i report za jedan model.
    """
    predictions_path = Path(
        f"data/predictions/gru_predictions/{model_name}_test_predictions.csv"
    )
    report_path = Path(f"data/logs/gru_logs/{model_name}_report.csv")

    predictions_df = pd.read_csv(predictions_path, parse_dates=["datetime"])
    report_df = pd.read_csv(report_path)

    return predictions_df, report_df


def load_best_gru_model_name():
    """
    Vraca naziv najboljeg GRU modela iz step06 outputa.
    """
    best_summary_path = Path("data/logs/gru_logs/best_gru_model_summary.csv")

    if best_summary_path.exists():
        best_summary_df = pd.read_csv(best_summary_path)
        selection_metric = best_summary_df.iloc[0].get("selection_metric")
        if selection_metric == "validation_MAE":
            return best_summary_df.iloc[0]["Model"]

    summaries_df = load_model_summaries()
    if not summaries_df.empty:
        best_index = summaries_df["validation_MAE"].astype(float).idxmin()
        return summaries_df.loc[best_index, "Model"]

    model_names = get_gru_model_names()
    if not model_names:
        raise FileNotFoundError("Nije pronadjen nijedan GRU report za crtanje.")

    return model_names[0]


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


def plot_hourly_metric(report_df, metric, graphs_dir, model_name, details):
    """
    Prikazuje hourly metriku za test skup.
    """
    hourly_df = report_df[
        (report_df["Report_Type"] == "hourly")
        & (report_df["Dataset"] == "test")
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly_df["Hour"], hourly_df[metric], color="tab:green")

    ax.set_title(f"GRU hourly {metric} - test - {model_name}")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(fig, details)

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(graphs_dir / f"{model_name}_test_hourly_{metric.lower()}.png")
    plt.close(fig)


def plot_compare_hourly_metric(
    baseline_report_df,
    gru_report_df,
    metric,
    graphs_dir,
    model_name,
    details,
):
    """
    Prikazuje baseline i najbolji GRU hourly metriku za test skup.
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
        label=BASELINE_MODEL_NAME,
        color="tab:blue",
    )
    ax.bar(
        hours + bar_width / 2,
        compare_df[f"{metric}_gru"],
        width=bar_width,
        label=f"Best GRU: {model_name}",
        color="tab:green",
    )

    ax.set_title(f"Baseline vs best GRU hourly {metric} - test")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(fig, details, "Comparison: baseline lag_24h vs best GRU")

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(
        graphs_dir / f"best_{model_name}_baseline_gru_compare_test_hourly_{metric.lower()}.png"
    )
    plt.close(fig)


def plot_actual_vs_predicted_scatter(predictions_df, graphs_dir, model_name, details):
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

    ax.set_title(f"GRU actual vs predicted scatter - test - {model_name}")
    ax.set_xlabel("Actual Energy_kWh")
    ax.set_ylabel("Predicted Energy_kWh")
    ax.grid(True, alpha=0.3)
    add_graph_details(fig, details)

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(graphs_dir / f"{model_name}_test_actual_vs_predicted_scatter.png")
    plt.close(fig)


def plot_compare_actual_vs_predicted_scatter(compare_df, graphs_dir, model_name, details):
    """
    Prikazuje odnos stvarnih i predvidjenih vrednosti za baseline i najbolji GRU.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        compare_df["actual_Energy_kWh"],
        compare_df["baseline_predicted_Energy_kWh"],
        alpha=0.28,
        s=12,
        label=BASELINE_MODEL_NAME,
        color="tab:blue",
    )
    ax.scatter(
        compare_df["actual_Energy_kWh"],
        compare_df["gru_predicted_Energy_kWh"],
        alpha=0.28,
        s=12,
        label=f"Best GRU: {model_name}",
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

    ax.set_title("Baseline vs best GRU actual vs predicted scatter - test")
    ax.set_xlabel("Actual Energy_kWh")
    ax.set_ylabel("Predicted Energy_kWh")
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_graph_details(fig, details, "Comparison: baseline lag_24h vs best GRU")

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(
        graphs_dir
        / f"best_{model_name}_baseline_gru_compare_test_actual_vs_predicted_scatter.png"
    )
    plt.close(fig)


def plot_error_distribution(predictions_df, graphs_dir, axis_config, model_name, details):
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

    ax.set_title(f"GRU error distribution - test - {model_name}")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    apply_error_axis_config(ax, axis_config)
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(fig, details)

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(graphs_dir / f"{model_name}_test_error_distribution.png")
    plt.close(fig)


def plot_compare_error_distribution(compare_df, graphs_dir, axis_config, model_name, details):
    """
    Prikazuje distribuciju apsolutnih gresaka za baseline i najbolji GRU.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = axis_config["bins"]
    baseline_counts, _ = np.histogram(compare_df["baseline_absolute_error"], bins=bins)
    gru_counts, _ = np.histogram(compare_df["gru_absolute_error"], bins=bins)
    bin_width = bins[1] - bins[0]

    ax.bar(
        bins[:-1],
        baseline_counts,
        width=bin_width / 2,
        align="edge",
        label=BASELINE_MODEL_NAME,
        color="tab:blue",
        edgecolor="black",
    )
    ax.bar(
        bins[:-1] + bin_width / 2,
        gru_counts,
        width=bin_width / 2,
        align="edge",
        label=f"Best GRU: {model_name}",
        color="tab:green",
        edgecolor="black",
    )

    ax.set_title("Baseline vs best GRU error distribution - test")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    ax.legend()
    apply_error_axis_config(ax, axis_config)
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(fig, details, "Comparison: baseline lag_24h vs best GRU")

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(
        graphs_dir / f"best_{model_name}_baseline_gru_compare_test_error_distribution.png"
    )
    plt.close(fig)


def create_gru_graphs():
    """
    Kreira 4 graficka prikaza za svaki GRU model iz step06.
    """
    graphs_dir = Path("data/graphs/gru_graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    baseline_test_df, _ = load_baseline_outputs()
    baseline_test_df = baseline_test_df.copy()
    baseline_test_df["absolute_error"] = (
        baseline_test_df[TARGET_COLUMN] - baseline_test_df[BASELINE_COLUMN]
    ).abs()

    for model_name in get_gru_model_names():
        predictions_df, report_df = load_gru_outputs(model_name)
        details = get_graph_details(model_name)
        error_axis_config = create_error_axis_config(
            [
                baseline_test_df["absolute_error"].to_numpy(),
                predictions_df["absolute_error"].to_numpy(),
            ]
        )

        plot_hourly_metric(report_df, "WAPE", graphs_dir, model_name, details)
        plot_hourly_metric(report_df, "MAE", graphs_dir, model_name, details)
        plot_actual_vs_predicted_scatter(
            predictions_df,
            graphs_dir,
            model_name,
            details,
        )
        plot_error_distribution(
            predictions_df,
            graphs_dir,
            error_axis_config,
            model_name,
            details,
        )


def create_baseline_gru_compare_graphs():
    """
    Kreira 4 uporedna grafa za baseline i najbolji GRU model.
    """
    graphs_dir = Path("data/graphs/baseline_gru_compare_graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    best_model_name = load_best_gru_model_name()
    gru_predictions_df, gru_report_df = load_gru_outputs(best_model_name)
    baseline_test_df, baseline_report_df = load_baseline_outputs()
    compare_df = create_compare_predictions_df(baseline_test_df, gru_predictions_df)
    details = get_graph_details(best_model_name)
    error_axis_config = create_error_axis_config(
        [
            compare_df["baseline_absolute_error"].to_numpy(),
            compare_df["gru_absolute_error"].to_numpy(),
        ]
    )

    plot_compare_hourly_metric(
        baseline_report_df,
        gru_report_df,
        "WAPE",
        graphs_dir,
        best_model_name,
        details,
    )
    plot_compare_hourly_metric(
        baseline_report_df,
        gru_report_df,
        "MAE",
        graphs_dir,
        best_model_name,
        details,
    )
    plot_compare_actual_vs_predicted_scatter(
        compare_df,
        graphs_dir,
        best_model_name,
        details,
    )
    plot_compare_error_distribution(
        compare_df,
        graphs_dir,
        error_axis_config,
        best_model_name,
        details,
    )


if __name__ == "__main__":
    create_gru_graphs()
    create_baseline_gru_compare_graphs()
