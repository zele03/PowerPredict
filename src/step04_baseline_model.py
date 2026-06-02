from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Energy_kWh"
BASELINE_COLUMN = "lag_24h"


def load_split_datasets():
    """
    Učitava validation i test skupove iz split foldera.
    """
    split_dir = Path("data/processed/split")

    validation_df = pd.read_csv(split_dir / "validation.csv", parse_dates=["datetime"])
    test_df = pd.read_csv(split_dir / "test.csv", parse_dates=["datetime"])

    return validation_df, test_df


def evaluate_baseline(df, dataset_name):
    """
    Računa MAE i RMSE za baseline predikciju.
    Baseline koristi potrošnju istog sata prethodnog dana.
    """
    y_true = df[TARGET_COLUMN]
    y_pred = df[BASELINE_COLUMN]

    errors = y_true - y_pred

    mae = errors.abs().mean()
    rmse = (errors.pow(2).mean()) ** 0.5

    return {
        "Dataset": dataset_name,
        "Baseline": BASELINE_COLUMN,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Broj_redova": len(df),
    }


def create_baseline_report(validation_df, test_df):
    """
    Kreira baseline izveštaj za validation i test skup.
    """
    report = [
        evaluate_baseline(validation_df, "validation"),
        evaluate_baseline(test_df, "test"),
    ]

    return pd.DataFrame(report)


def save_baseline_report(report_df):
    """
    Čuva baseline izveštaj u logs folder.
    """
    logs_dir = Path("data/processed/logs")
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
