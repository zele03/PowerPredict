from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Energy_kWh"
BASELINE_COLUMN = "lag_24h"
PERCENTAGE_MULTIPLIER = 100


def load_split_datasets():
    """
    Ucitava validation i test skupove iz split foldera.
    """
    split_dir = Path("data/processed/split")

    validation_df = pd.read_csv(split_dir / "validation.csv", parse_dates=["datetime"])
    test_df = pd.read_csv(split_dir / "test.csv", parse_dates=["datetime"])

    return validation_df, test_df


def evaluate_baseline(df, dataset_name):
    """
    Racuna glavne i dijagnosticke metrike za baseline predikciju.
    Baseline koristi potrosnju istog sata prethodnog dana.
    """
    y_true = df[TARGET_COLUMN]
    y_pred = df[BASELINE_COLUMN]

    errors = y_true - y_pred
    absolute_errors = errors.abs()

    mae = absolute_errors.mean()
    rmse = (errors.pow(2).mean()) ** 0.5
    wape = absolute_errors.sum() / y_true.sum() * PERCENTAGE_MULTIPLIER

    valid_mape_mask = y_true > 0
    mape = (
        (absolute_errors[valid_mape_mask] / y_true[valid_mape_mask]).mean()
        * PERCENTAGE_MULTIPLIER
    )

    smape_denominator = y_true.abs() + y_pred.abs()
    valid_smape_mask = smape_denominator > 0
    smape = (
        (2 * absolute_errors[valid_smape_mask] / smape_denominator[valid_smape_mask]).mean()
        * PERCENTAGE_MULTIPLIER
    )

    return {
        "Report_Type": "summary",
        "Dataset": dataset_name,
        "Baseline": BASELINE_COLUMN,
        "Hour": "",
        "WAPE": round(wape, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "sMAPE": round(smape, 4),
        "Broj_redova": len(df),
    }


def evaluate_baseline_by_hour(df, dataset_name):
    """
    Racuna metrike posebno za svaki sat dana.
    """
    hourly_report = []

    for hour, hour_df in df.groupby("hour"):
        metrics = evaluate_baseline(hour_df, dataset_name)
        metrics["Report_Type"] = "hourly"
        metrics["Hour"] = int(hour)
        hourly_report.append(metrics)

    return hourly_report


def create_baseline_report(validation_df, test_df):
    """
    Kreira baseline izvestaj za validation i test skup.
    """
    summary_report = [
        evaluate_baseline(validation_df, "validation"),
        evaluate_baseline(test_df, "test"),
    ]

    hourly_report = []
    hourly_report.extend(evaluate_baseline_by_hour(validation_df, "validation"))
    hourly_report.extend(evaluate_baseline_by_hour(test_df, "test"))

    report = summary_report + hourly_report

    return pd.DataFrame(report)


def save_baseline_report(report_df):
    """
    Cuva baseline izvestaj u logs folder.
    """
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(logs_dir / "baseline_report.csv", index=False)


def create_baseline_model():
    """
    Glavna funkcija za baseline evaluaciju.
    """
    validation_df, test_df = load_split_datasets()

    report_df = create_baseline_report(validation_df, test_df)

    save_baseline_report(report_df)

    return report_df
