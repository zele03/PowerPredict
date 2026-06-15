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
HOURS_IN_DAY = 24


TREE_MODEL_SPECS = {
    "random_forest": {
        "display_name": "Random Forest",
        "short_label": "Random Forest",
        "color": "tab:orange",
        "logs_dir": Path("data/logs/random_forest_logs"),
        "predictions_dir": Path("data/predictions/random_forest_predictions"),
        "graphs_dir": Path("data/graphs/random_forest_graphs"),
        "baseline_compare_graphs_dir": Path(
            "data/graphs/baseline_random_forest_compare_graphs"
        ),
        "best_summary_path": Path(
            "data/logs/random_forest_logs/best_random_forest_model_summary.csv"
        ),
        "model_summaries_path": Path(
            "data/logs/random_forest_logs/random_forest_model_summaries.csv"
        ),
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "short_label": "Gradient Boosting",
        "color": "tab:purple",
        "logs_dir": Path("data/logs/gradient_boosting_logs"),
        "predictions_dir": Path("data/predictions/gradient_boosting_predictions"),
        "graphs_dir": Path("data/graphs/gradient_boosting_graphs"),
        "baseline_compare_graphs_dir": Path(
            "data/graphs/baseline_gradient_boosting_compare_graphs"
        ),
        "best_summary_path": Path(
            "data/logs/gradient_boosting_logs/best_gradient_boosting_model_summary.csv"
        ),
        "model_summaries_path": Path(
            "data/logs/gradient_boosting_logs/gradient_boosting_model_summaries.csv"
        ),
    },
}


SEQUENCE_MODEL_SPECS = {
    "gru": {
        "display_name": "GRU",
        "short_label": "GRU",
        "color": "tab:green",
        "logs_dir": Path("data/logs/gru_logs"),
        "predictions_dir": Path("data/predictions/gru_predictions"),
        "best_summary_path": Path("data/logs/gru_logs/best_gru_model_summary.csv"),
    },
    "lstm": {
        "display_name": "LSTM",
        "short_label": "LSTM",
        "color": "tab:red",
        "logs_dir": Path("data/logs/lstm_logs"),
        "predictions_dir": Path("data/predictions/lstm_predictions"),
        "best_summary_path": Path("data/logs/lstm_logs/best_lstm_model_summary.csv"),
    },
}


ALL_MODELS_COMPARE_GRAPHS_DIR = Path("data/graphs/all_models_compare_graphs")


def load_baseline_outputs():
    """
    Ucitava baseline test podatke i baseline report.
    Baseline predikcija je kolona lag_24h iz test skupa.
    """
    test_df = pd.read_csv(
        Path("data/processed/split/test.csv"),
        parse_dates=["datetime"],
    )
    report_df = pd.read_csv(Path("data/logs/baseline_report.csv"))

    predictions_df = test_df[["datetime", TARGET_COLUMN, BASELINE_COLUMN]].copy()
    predictions_df = predictions_df.rename(
        columns={
            TARGET_COLUMN: "actual_Energy_kWh",
            BASELINE_COLUMN: "predicted_Energy_kWh",
        }
    )
    predictions_df["absolute_error"] = (
        predictions_df["actual_Energy_kWh"]
        - predictions_df["predicted_Energy_kWh"]
    ).abs()

    return predictions_df, report_df


def load_model_summaries(model_key):
    """
    Ucitava summary tabelu za sve konfiguracije jednog tree modela.
    """
    summaries_path = TREE_MODEL_SPECS[model_key]["model_summaries_path"]

    if not summaries_path.exists():
        return pd.DataFrame()

    return pd.read_csv(summaries_path)


def get_tree_model_names(model_key):
    """
    Vraca nazive svih tree modela za koje postoje reportovi.
    Prvo koristi model_summaries.csv, a ako ga nema, gleda pojedinacne reportove.
    """
    summaries_df = load_model_summaries(model_key)
    if not summaries_df.empty:
        return summaries_df["Model"].tolist()

    logs_dir = TREE_MODEL_SPECS[model_key]["logs_dir"]
    model_names = []

    for report_path in logs_dir.glob("*_report.csv"):
        model_name = report_path.stem.removesuffix("_report")
        if "hyperparameter" not in model_name:
            model_names.append(model_name)

    return sorted(model_names)


def get_model_summary_row(model_key, model_name):
    """
    Vraca red iz summary tabele za konkretnu konfiguraciju.
    Koristi se za tekst ispod grafika.
    """
    summaries_df = load_model_summaries(model_key)

    if summaries_df.empty:
        return None

    matching_rows = summaries_df[summaries_df["Model"] == model_name]
    if matching_rows.empty:
        return None

    return matching_rows.iloc[0]


def get_tree_graph_details(model_key, model_name):
    """
    Pravi kratak opis modela koji se upisuje ispod grafa.
    """
    summary_row = get_model_summary_row(model_key, model_name)
    display_name = TREE_MODEL_SPECS[model_key]["display_name"]

    if summary_row is None:
        return f"Model: {display_name} - {model_name}"

    hyperparameter_parts = []
    for column in summary_row.index:
        if column not in {
            "Model",
            "validation_MAE",
            "validation_RMSE",
            "validation_MAPE",
            "validation_sMAPE",
            "validation_WAPE",
            "test_MAE",
            "test_RMSE",
            "test_MAPE",
            "test_sMAPE",
            "test_WAPE",
        }:
            hyperparameter_parts.append(f"{column}={summary_row[column]}")

    return (
        f"Model: {display_name} - {model_name}\n"
        f"{' | '.join(hyperparameter_parts)}\n"
        f"validation MAE={summary_row['validation_MAE']} | "
        f"test MAE={summary_row['test_MAE']} | "
        f"test RMSE={summary_row['test_RMSE']} | "
        f"test MAPE={summary_row['test_MAPE']} | "
        f"test sMAPE={summary_row['test_sMAPE']} | "
        f"test WAPE={summary_row['test_WAPE']}"
    )


def add_graph_details(fig, details, extra_text=None):
    """
    Dodaje tekst ispod grafa.
    Ovo je korisno jer se uz svaki PNG odmah vidi koji model i parametri su korisceni.
    """
    if extra_text:
        details = f"{details}\n{extra_text}"

    fig.text(0.01, 0.01, details, ha="left", va="bottom", fontsize=8)


def load_tree_outputs(model_key, model_name):
    """
    Ucitava test predikcije i report za jedan Random Forest ili Gradient Boosting model.
    """
    spec = TREE_MODEL_SPECS[model_key]
    predictions_path = spec["predictions_dir"] / f"{model_name}_test_predictions.csv"
    report_path = spec["logs_dir"] / f"{model_name}_report.csv"

    predictions_df = pd.read_csv(predictions_path, parse_dates=["datetime"])
    report_df = pd.read_csv(report_path)

    return predictions_df, report_df


def load_feature_importance(model_key, model_name):
    """
    Ucitava feature importance CSV koji je napravio step10.
    """
    logs_dir = TREE_MODEL_SPECS[model_key]["logs_dir"]
    importance_path = logs_dir / f"{model_name}_feature_importance.csv"

    if not importance_path.exists():
        return pd.DataFrame()

    return pd.read_csv(importance_path)


def load_best_model_name(best_summary_path):
    """
    Iz best summary fajla cita naziv najbolje konfiguracije.
    """
    if not best_summary_path.exists():
        return None

    best_summary_df = pd.read_csv(best_summary_path)
    if best_summary_df.empty:
        return None

    return best_summary_df.iloc[0]["Model"]


def create_error_axis_config(error_values_list):
    """
    Racuna zajednicke binove i ose za error distribution grafike.
    Kada poredimo modele, zajednicka osa sprecava varljiv vizuelni utisak.
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
    Primenjuje zajednicku konfiguraciju osa za histogram gresaka.
    """
    ax.set_xlim(axis_config["x_lim"])
    ax.set_ylim(axis_config["y_lim"])
    ax.set_xticks(axis_config["x_ticks"])
    ax.set_yticks(axis_config["y_ticks"])


def get_hourly_metric(report_df, metric):
    """
    Iz reporta uzima hourly vrednosti jedne metrike za test skup.
    """
    hourly_df = report_df[
        (report_df["Report_Type"] == "hourly")
        & (report_df["Dataset"] == "test")
    ].copy()
    hourly_df["Hour"] = hourly_df["Hour"].astype(int)

    return hourly_df[["Hour", metric]].sort_values("Hour")


def get_summary_metrics(report_df):
    """
    Iz reporta uzima summary metrike za test skup.
    """
    summary_row = report_df[
        (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == "test")
    ].iloc[0]

    return {
        "MAE": float(summary_row["MAE"]),
        "RMSE": float(summary_row["RMSE"]),
        "MAPE": float(summary_row["MAPE"]),
        "sMAPE": float(summary_row["sMAPE"]),
        "WAPE": float(summary_row["WAPE"]),
    }


def plot_hourly_metric(report_df, metric, graphs_dir, model_key, model_name, details):
    """
    Crta hourly MAE ili WAPE za jedan tree model.
    """
    spec = TREE_MODEL_SPECS[model_key]
    hourly_df = get_hourly_metric(report_df, metric)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly_df["Hour"], hourly_df[metric], color=spec["color"])

    ax.set_title(f"{spec['display_name']} hourly {metric} - test - {model_name}")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(fig, details)

    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(graphs_dir / f"{model_name}_test_hourly_{metric.lower()}.png")
    plt.close(fig)


def plot_actual_vs_predicted_scatter(
    predictions_df,
    graphs_dir,
    model_key,
    model_name,
    details,
):
    """
    Crta scatter odnos stvarne i predvidjene potrosnje.
    Idealna predikcija bi lezala na crvenoj dijagonali.
    """
    spec = TREE_MODEL_SPECS[model_key]
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        predictions_df["actual_Energy_kWh"],
        predictions_df["predicted_Energy_kWh"],
        alpha=0.35,
        s=12,
        color=spec["color"],
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

    ax.set_title(
        f"{spec['display_name']} actual vs predicted scatter - test - {model_name}"
    )
    ax.set_xlabel("Actual Energy_kWh")
    ax.set_ylabel("Predicted Energy_kWh")
    ax.grid(True, alpha=0.3)
    add_graph_details(fig, details)

    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(graphs_dir / f"{model_name}_test_actual_vs_predicted_scatter.png")
    plt.close(fig)


def plot_error_distribution(
    predictions_df,
    graphs_dir,
    axis_config,
    model_key,
    model_name,
    details,
):
    """
    Crta distribuciju apsolutnih gresaka za jedan tree model.
    """
    spec = TREE_MODEL_SPECS[model_key]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        predictions_df["absolute_error"],
        bins=axis_config["bins"],
        color=spec["color"],
        edgecolor="black",
    )

    ax.set_title(f"{spec['display_name']} error distribution - test - {model_name}")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    apply_error_axis_config(ax, axis_config)
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(fig, details)

    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(graphs_dir / f"{model_name}_test_error_distribution.png")
    plt.close(fig)


def plot_feature_importance(importance_df, graphs_dir, model_key, model_name, details):
    """
    Crta feature importance za tree modele.
    Ovo pokazuje koje kolone su najvise doprinosile podelama u stablima.
    """
    if importance_df.empty:
        return

    spec = TREE_MODEL_SPECS[model_key]
    plot_df = importance_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["Feature"], plot_df["Importance"], color=spec["color"])

    ax.set_title(f"{spec['display_name']} feature importance - {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)
    add_graph_details(fig, details)

    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(graphs_dir / f"{model_name}_feature_importance.png")
    plt.close(fig)


def create_compare_predictions_df(baseline_predictions_df, model_predictions_df):
    """
    Spaja baseline i tree predikcije za iste test sate.
    """
    baseline_df = baseline_predictions_df[
        ["datetime", "actual_Energy_kWh", "predicted_Energy_kWh"]
    ].copy()
    baseline_df = baseline_df.rename(
        columns={"predicted_Energy_kWh": "baseline_predicted_Energy_kWh"}
    )

    model_df = model_predictions_df[
        ["datetime", "actual_Energy_kWh", "predicted_Energy_kWh"]
    ].copy()
    model_df = model_df.rename(
        columns={"predicted_Energy_kWh": "model_predicted_Energy_kWh"}
    )

    compare_df = baseline_df.merge(
        model_df,
        on=["datetime", "actual_Energy_kWh"],
        how="inner",
    )
    compare_df["baseline_absolute_error"] = (
        compare_df["actual_Energy_kWh"]
        - compare_df["baseline_predicted_Energy_kWh"]
    ).abs()
    compare_df["model_absolute_error"] = (
        compare_df["actual_Energy_kWh"] - compare_df["model_predicted_Energy_kWh"]
    ).abs()

    return compare_df


def plot_compare_hourly_metric(
    baseline_report_df,
    model_report_df,
    metric,
    graphs_dir,
    model_key,
    model_name,
    details,
):
    """
    Crta baseline i najbolji tree model po satima.
    """
    spec = TREE_MODEL_SPECS[model_key]
    baseline_hourly_df = get_hourly_metric(baseline_report_df, metric)
    model_hourly_df = get_hourly_metric(model_report_df, metric)
    compare_df = baseline_hourly_df.merge(
        model_hourly_df,
        on="Hour",
        suffixes=("_baseline", "_model"),
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
        compare_df[f"{metric}_model"],
        width=bar_width,
        label=f"Best {spec['display_name']}: {model_name}",
        color=spec["color"],
    )

    ax.set_title(f"Baseline vs best {spec['display_name']} hourly {metric} - test")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(
        fig,
        details,
        f"Comparison: baseline lag_24h vs best {spec['display_name']}",
    )

    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(
        graphs_dir
        / f"best_{model_name}_baseline_{model_key}_compare_test_hourly_{metric.lower()}.png"
    )
    plt.close(fig)


def plot_compare_actual_vs_predicted_scatter(
    compare_df,
    graphs_dir,
    model_key,
    model_name,
    details,
):
    """
    Crta scatter baseline i najboljeg tree modela na istim test satima.
    """
    spec = TREE_MODEL_SPECS[model_key]
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
        compare_df["model_predicted_Energy_kWh"],
        alpha=0.28,
        s=12,
        label=f"Best {spec['display_name']}: {model_name}",
        color=spec["color"],
    )

    min_value = min(
        compare_df["actual_Energy_kWh"].min(),
        compare_df["baseline_predicted_Energy_kWh"].min(),
        compare_df["model_predicted_Energy_kWh"].min(),
    )
    max_value = max(
        compare_df["actual_Energy_kWh"].max(),
        compare_df["baseline_predicted_Energy_kWh"].max(),
        compare_df["model_predicted_Energy_kWh"].max(),
    )
    ax.plot([min_value, max_value], [min_value, max_value], color="tab:red")

    ax.set_title(f"Baseline vs best {spec['display_name']} actual vs predicted - test")
    ax.set_xlabel("Actual Energy_kWh")
    ax.set_ylabel("Predicted Energy_kWh")
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_graph_details(
        fig,
        details,
        f"Comparison: baseline lag_24h vs best {spec['display_name']}",
    )

    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(
        graphs_dir
        / f"best_{model_name}_baseline_{model_key}_compare_test_actual_vs_predicted_scatter.png"
    )
    plt.close(fig)


def plot_compare_error_distribution(
    compare_df,
    graphs_dir,
    axis_config,
    model_key,
    model_name,
    details,
):
    """
    Crta distribuciju gresaka baseline-a i najboljeg tree modela.
    """
    spec = TREE_MODEL_SPECS[model_key]
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = axis_config["bins"]
    baseline_counts, _ = np.histogram(compare_df["baseline_absolute_error"], bins=bins)
    model_counts, _ = np.histogram(compare_df["model_absolute_error"], bins=bins)
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
        model_counts,
        width=bin_width / 2,
        align="edge",
        label=f"Best {spec['display_name']}: {model_name}",
        color=spec["color"],
        edgecolor="black",
    )

    ax.set_title(f"Baseline vs best {spec['display_name']} error distribution - test")
    ax.set_xlabel("Absolute error (kWh)")
    ax.set_ylabel("Number of hours")
    ax.legend()
    apply_error_axis_config(ax, axis_config)
    ax.grid(True, axis="y", alpha=0.3)
    add_graph_details(
        fig,
        details,
        f"Comparison: baseline lag_24h vs best {spec['display_name']}",
    )

    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(
        graphs_dir
        / f"best_{model_name}_baseline_{model_key}_compare_test_error_distribution.png"
    )
    plt.close(fig)


def create_tree_graphs():
    """
    Kreira grafove za svaku Random Forest i Gradient Boosting konfiguraciju.
    """
    baseline_predictions_df, _ = load_baseline_outputs()

    for model_key, spec in TREE_MODEL_SPECS.items():
        graphs_dir = spec["graphs_dir"]
        graphs_dir.mkdir(parents=True, exist_ok=True)

        for model_name in get_tree_model_names(model_key):
            predictions_df, report_df = load_tree_outputs(model_key, model_name)
            importance_df = load_feature_importance(model_key, model_name)
            details = get_tree_graph_details(model_key, model_name)
            error_axis_config = create_error_axis_config(
                [
                    baseline_predictions_df["absolute_error"].to_numpy(),
                    predictions_df["absolute_error"].to_numpy(),
                ]
            )

            plot_hourly_metric(report_df, "WAPE", graphs_dir, model_key, model_name, details)
            plot_hourly_metric(report_df, "MAE", graphs_dir, model_key, model_name, details)
            plot_actual_vs_predicted_scatter(
                predictions_df,
                graphs_dir,
                model_key,
                model_name,
                details,
            )
            plot_error_distribution(
                predictions_df,
                graphs_dir,
                error_axis_config,
                model_key,
                model_name,
                details,
            )
            plot_feature_importance(
                importance_df,
                graphs_dir,
                model_key,
                model_name,
                details,
            )


def create_baseline_tree_compare_graphs():
    """
    Kreira baseline poredenje za najbolji Random Forest i najbolji Gradient Boosting.
    """
    baseline_predictions_df, baseline_report_df = load_baseline_outputs()

    for model_key, spec in TREE_MODEL_SPECS.items():
        best_model_name = load_best_model_name(spec["best_summary_path"])
        if best_model_name is None:
            continue

        graphs_dir = spec["baseline_compare_graphs_dir"]
        graphs_dir.mkdir(parents=True, exist_ok=True)

        predictions_df, report_df = load_tree_outputs(model_key, best_model_name)
        compare_df = create_compare_predictions_df(
            baseline_predictions_df,
            predictions_df,
        )
        details = get_tree_graph_details(model_key, best_model_name)
        error_axis_config = create_error_axis_config(
            [
                compare_df["baseline_absolute_error"].to_numpy(),
                compare_df["model_absolute_error"].to_numpy(),
            ]
        )

        plot_compare_hourly_metric(
            baseline_report_df,
            report_df,
            "WAPE",
            graphs_dir,
            model_key,
            best_model_name,
            details,
        )
        plot_compare_hourly_metric(
            baseline_report_df,
            report_df,
            "MAE",
            graphs_dir,
            model_key,
            best_model_name,
            details,
        )
        plot_compare_actual_vs_predicted_scatter(
            compare_df,
            graphs_dir,
            model_key,
            best_model_name,
            details,
        )
        plot_compare_error_distribution(
            compare_df,
            graphs_dir,
            error_axis_config,
            model_key,
            best_model_name,
            details,
        )


def load_sequence_best_outputs(model_key):
    """
    Ucitava najbolji GRU ili LSTM ako su njegovi reportovi i predikcije dostupni.
    """
    spec = SEQUENCE_MODEL_SPECS[model_key]
    best_model_name = load_best_model_name(spec["best_summary_path"])

    if best_model_name is None:
        return None

    predictions_path = spec["predictions_dir"] / f"{best_model_name}_test_predictions.csv"
    report_path = spec["logs_dir"] / f"{best_model_name}_report.csv"

    if not predictions_path.exists() or not report_path.exists():
        return None

    return {
        "label": spec["short_label"],
        "model_name": best_model_name,
        "color": spec["color"],
        "predictions_df": pd.read_csv(predictions_path, parse_dates=["datetime"]),
        "report_df": pd.read_csv(report_path),
    }


def load_tree_best_outputs(model_key):
    """
    Ucitava najbolji Random Forest ili Gradient Boosting model.
    """
    spec = TREE_MODEL_SPECS[model_key]
    best_model_name = load_best_model_name(spec["best_summary_path"])

    if best_model_name is None:
        return None

    predictions_df, report_df = load_tree_outputs(model_key, best_model_name)

    return {
        "label": spec["short_label"],
        "model_name": best_model_name,
        "color": spec["color"],
        "predictions_df": predictions_df,
        "report_df": report_df,
    }


def load_all_best_outputs():
    """
    Ucitava baseline i najbolje dostupne modele za globalno poredjenje.
    Ako neki model nije istreniran, samo se preskace.
    """
    baseline_predictions_df, baseline_report_df = load_baseline_outputs()
    outputs = [
        {
            "label": "Baseline",
            "model_name": BASELINE_MODEL_NAME,
            "color": "tab:blue",
            "predictions_df": baseline_predictions_df,
            "report_df": baseline_report_df,
        }
    ]

    for model_key in ["gru", "lstm"]:
        model_output = load_sequence_best_outputs(model_key)
        if model_output is not None:
            outputs.append(model_output)

    for model_key in ["random_forest", "gradient_boosting"]:
        model_output = load_tree_best_outputs(model_key)
        if model_output is not None:
            outputs.append(model_output)

    return outputs


def plot_all_models_summary_metrics(model_outputs, graphs_dir):
    """
    Crta summary metrike svih najboljih modela.
    """
    metrics = ["MAE", "RMSE"]
    labels = [output["label"] for output in model_outputs]
    x_positions = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    for metric_index, metric in enumerate(metrics):
        values = [
            get_summary_metrics(output["report_df"])[metric]
            for output in model_outputs
        ]
        offset = (metric_index - 0.5) * bar_width
        bars = ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=metric,
        )
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)

    ax.set_title("Best models MAE and RMSE - test")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / "best_models_test_summary_metrics.png")
    plt.close(fig)


def plot_all_models_hourly_metric(model_outputs, graphs_dir, metric):
    """
    Crta jednu hourly metriku za sve najbolje modele na istom grafu.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for output in model_outputs:
        hourly_df = get_hourly_metric(output["report_df"], metric)
        ax.plot(
            hourly_df["Hour"],
            hourly_df[metric],
            marker="o",
            linewidth=2,
            label=output["label"],
            color=output["color"],
        )

    ax.set_title(f"Best models hourly {metric} - test")
    ax.set_xlabel("Hour")
    ax.set_ylabel(metric)
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(graphs_dir / f"best_models_test_hourly_{metric.lower()}.png")
    plt.close(fig)


def create_daily_error_summary(model_outputs):
    """
    Racuna prosecnu apsolutnu gresku po danu.
    Koristimo prosek svih modela osim baseline-a da nadjemo jedan dobar i jedan los dan.
    """
    baseline_output = model_outputs[0]
    daily_df = baseline_output["predictions_df"][["datetime"]].copy()
    daily_df["date"] = daily_df["datetime"].dt.date

    model_error_columns = []
    outputs_for_selection = model_outputs[1:] if len(model_outputs) > 1 else model_outputs

    for output in outputs_for_selection:
        column_name = f"{output['label']}_absolute_error"
        model_error_columns.append(column_name)
        errors_df = output["predictions_df"][["datetime", "absolute_error"]].copy()
        errors_df = errors_df.rename(columns={"absolute_error": column_name})
        daily_df = daily_df.merge(errors_df, on="datetime", how="inner")

    daily_summary_df = daily_df.groupby("date")[model_error_columns].mean()
    daily_summary_df["mean_model_absolute_error"] = daily_summary_df.mean(axis=1)

    return daily_summary_df.reset_index()


def plot_all_models_one_day_predictions(model_outputs, graphs_dir, selected_date, label):
    """
    Crta stvarnu potrosnju i predikcije svih modela za jedan dan iz test skupa.
    """
    baseline_output = model_outputs[0]
    base_df = baseline_output["predictions_df"][
        ["datetime", "actual_Energy_kWh"]
    ].copy()
    base_df = base_df[base_df["datetime"].dt.date == selected_date].head(HOURS_IN_DAY)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        base_df["datetime"],
        base_df["actual_Energy_kWh"],
        linewidth=2.5,
        label="Actual",
        color="black",
    )

    for output in model_outputs:
        predictions_df = output["predictions_df"][
            ["datetime", "predicted_Energy_kWh"]
        ].copy()
        plot_df = base_df[["datetime"]].merge(
            predictions_df,
            on="datetime",
            how="left",
        )
        ax.plot(
            plot_df["datetime"],
            plot_df["predicted_Energy_kWh"],
            linewidth=1.7,
            alpha=0.9,
            label=output["label"],
            color=output["color"],
        )

    ax.set_title(f"Best models predictions - {label} day - {selected_date}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Energy_kWh")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(graphs_dir / f"best_models_{label}_day_predictions.png")
    plt.close(fig)


def plot_all_models_best_day(model_outputs, graphs_dir):
    """
    Pravi dnevni graf za dan gde su modeli najbolje pogadjali.
    """
    daily_summary_df = create_daily_error_summary(model_outputs)
    best_day = daily_summary_df.sort_values("mean_model_absolute_error").iloc[0]["date"]

    plot_all_models_one_day_predictions(model_outputs, graphs_dir, best_day, "best")


def create_all_models_compare_graphs():
    """
    Kreira zajednicke grafove za Baseline, GRU, LSTM, Random Forest i Gradient Boosting.
    """
    graphs_dir = ALL_MODELS_COMPARE_GRAPHS_DIR
    graphs_dir.mkdir(parents=True, exist_ok=True)

    model_outputs = load_all_best_outputs()

    plot_all_models_summary_metrics(model_outputs, graphs_dir)
    plot_all_models_hourly_metric(model_outputs, graphs_dir, "MAE")
    plot_all_models_hourly_metric(model_outputs, graphs_dir, "WAPE")
    plot_all_models_best_day(model_outputs, graphs_dir)


def create_tree_visualizations():
    """
    Glavna funkcija za step11.
    Pravi tree grafove, baseline-tree poredenja i zajednicko poredjenje svih modela.
    """
    create_tree_graphs()
    create_baseline_tree_compare_graphs()
    create_all_models_compare_graphs()


if __name__ == "__main__":
    create_tree_visualizations()
