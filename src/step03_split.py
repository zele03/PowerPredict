from pathlib import Path

import pandas as pd


def load_features_dataset(features_path):
    """
    Učitava features dataset.
    """
    df = pd.read_csv(features_path, parse_dates=["datetime"])
    return df


def sort_by_datetime(df):
    """
    Sortira podatke hronološki jer radimo vremensku seriju.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def split_dataset(df, train_ratio=0.7, validation_ratio=0.15):
    """
    Deli dataset na train, validation i test bez random mešanja.
    """
    total_rows = len(df)

    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def create_split_report(train_df, validation_df, test_df):
    """
    Kreira izveštaj o podeli dataseta.
    """
    report_entries = [
        {"Opis": "Broj redova u train skupu", "Vrednost": len(train_df)},
        {"Opis": "Broj redova u validation skupu", "Vrednost": len(validation_df)},
        {"Opis": "Broj redova u test skupu", "Vrednost": len(test_df)},
        {
            "Opis": "Train period",
            "Vrednost": f"{train_df['datetime'].min()} - {train_df['datetime'].max()}",
        },
        {
            "Opis": "Validation period",
            "Vrednost": f"{validation_df['datetime'].min()} - {validation_df['datetime'].max()}",
        },
        {
            "Opis": "Test period",
            "Vrednost": f"{test_df['datetime'].min()} - {test_df['datetime'].max()}",
        },
    ]

    report_df = pd.DataFrame(report_entries)
    return report_df


def save_split_outputs(train_df, validation_df, test_df, report_df):
    """
    Čuva train, validation, test i split report fajlove.
    """
    split_dir = Path("data/processed/split")
    logs_dir = Path("data/logs")

    split_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(split_dir / "train.csv", index=False)
    validation_df.to_csv(split_dir / "validation.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    report_df.to_csv(logs_dir / "split_report.csv", index=False)


def create_train_validation_test_split(
    features_path="data/processed/features.csv",
):
    """
    Glavna funkcija za podelu features dataseta.

    Ako split fajlovi već postoje i isti su kao novi split, ne prepisuje ih.
    Ako se split razlikuje, prepisuje stare fajlove.
    """
    features_path = Path(features_path)

    split_dir = Path("data/processed/split")
    report_path = Path("data/logs/split_report.csv")

    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"

    df = load_features_dataset(features_path)
    df = sort_by_datetime(df)

    train_df, validation_df, test_df = split_dataset(df)
    report_df = create_split_report(train_df, validation_df, test_df)

    if (
        train_path.exists()
        and validation_path.exists()
        and test_path.exists()
        and report_path.exists()
    ):
        old_train = pd.read_csv(train_path, parse_dates=["datetime"])
        old_validation = pd.read_csv(validation_path, parse_dates=["datetime"])
        old_test = pd.read_csv(test_path, parse_dates=["datetime"])
        old_report = pd.read_csv(report_path)

        if (
            old_train.equals(train_df)
            and old_validation.equals(validation_df)
            and old_test.equals(test_df)
            and old_report.equals(report_df)
        ):
            return old_train, old_validation, old_test, old_report

    save_split_outputs(train_df, validation_df, test_df, report_df)

    return train_df, validation_df, test_df, report_df
