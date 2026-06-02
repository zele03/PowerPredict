from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

SELECTED_COLUMNS = [
    "datetime",
    "Global_active_power",
    "Global_reactive_power",
    "Global_intensity",
]


def load_raw_dataset(raw_path):
    """
    Učitava raw dataset i znak ? tretira kao missing vrednost.
    """
    df = pd.read_csv(raw_path, sep=";", na_values=["?"])
    return df


def add_datetime_columns(df):
    """
    Spaja Date i Time u datetime kolonu i dodaje date_only za grupisanje po danima.
    """
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )

    df["date_only"] = df["datetime"].dt.date
    return df


def mark_missing_rows(df):
    """
    Dodaje kolonu has_missing koja označava redove sa missing vrednostima.
    """
    df["has_missing"] = df[NUMERIC_COLUMNS].isna().any(axis=1)
    return df


def get_daily_statistics(df):
    """
    Računa broj merenja i broj missing redova za svaki dan.
    """
    measurements_per_day = df.groupby("date_only").size()
    missing_rows_per_day = df.groupby("date_only")["has_missing"].sum()

    return measurements_per_day, missing_rows_per_day


def find_days_to_remove(
    measurements_per_day,
    missing_rows_per_day,
    expected_rows_per_day=1440,
    missing_limit=60,
):
    """
    Pronalazi dane koje treba potpuno obrisati.

    Dan se briše ako mu fali više od missing_limit redova
    ili ako ima više od missing_limit redova sa missing vrednostima.
    """
    incomplete_rows_per_day = expected_rows_per_day - measurements_per_day

    days_with_too_few_rows = incomplete_rows_per_day[
        incomplete_rows_per_day > missing_limit
    ]

    days_with_too_many_missing = missing_rows_per_day[
        missing_rows_per_day > missing_limit
    ]

    days_to_remove = set(days_with_too_few_rows.index) | set(
        days_with_too_many_missing.index
    )

    return days_to_remove


def clean_dataset(df, days_to_remove):
    """
    Briše loše dane i zatim briše preostale redove sa missing vrednostima.
    """
    df = df[~df["date_only"].isin(days_to_remove)].copy()
    df = df[~df["has_missing"]].copy()

    return df


def create_hourly_dataset(df):
    """
    Bira relevantne kolone i pretvara minutne podatke u satne proseke.
    """
    processed_df = df[SELECTED_COLUMNS].copy()

    # datetime postaje indeks da bismo mogli da koristimo resample
    processed_df = processed_df.set_index("datetime")

    # Od minutnih podataka pravimo satne proseke
    hourly_df = processed_df.resample("h").mean()

    # Brišemo prazne sate
    hourly_df = hourly_df.dropna()

    # datetime vraćamo kao običnu kolonu
    hourly_df = hourly_df.reset_index()

    # Zaokružujemo numeričke kolone radi čitljivijeg CSV fajla
    numeric_cols = hourly_df.select_dtypes(include="number").columns
    hourly_df[numeric_cols] = hourly_df[numeric_cols].round(4)

    return hourly_df


def create_report(
    measurements_per_day,
    missing_rows_per_day,
    days_to_remove,
    hourly_df,
):
    """
    Kreira izveštaj o preprocessing koraku.
    """
    report_entries = [
        {
            "Opis": "Ukupan broj dana u raw datasetu",
            "Vrednost": len(measurements_per_day),
        },
        {
            "Opis": "Broj dana obrisanih",
            "Vrednost": len(days_to_remove),
        },
        {
            "Opis": "Lista obrisanih dana",
            "Vrednost": sorted(days_to_remove),
        },
        {
            "Opis": "Broj redova u processed datasetu",
            "Vrednost": len(hourly_df),
        },
        {
            "Opis": "Kolone u processed datasetu",
            "Vrednost": hourly_df.columns.tolist(),
        },
        {
            "Opis": "Prosečan broj redova po danu",
            "Vrednost": measurements_per_day.mean(),
        },
        {
            "Opis": "Prosečan broj missing vrednosti po danu",
            "Vrednost": missing_rows_per_day.mean(),
        },
    ]

    report_df = pd.DataFrame(report_entries)
    return report_df


def save_outputs(hourly_df, report_df, processed_path, report_path):
    """
    Čuva processed dataset i preprocessing report.
    """
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    hourly_df.to_csv(processed_path, index=False)
    report_df.to_csv(report_path, index=False)


def create_processed_dataset():
    """
    Glavna funkcija za preprocessing.

    Od raw dataseta pravi processed.csv i full_report.csv.
    """
    raw_path = Path("data/raw/raw.txt")
    processed_path = Path("data/processed/processed.csv")
    report_path = Path("data/processed/logs/full_report.csv")

    # Ako fajlovi već postoje, samo ih učitaj
    if processed_path.exists() and report_path.exists():
        df = pd.read_csv(processed_path)
        report = pd.read_csv(report_path)
        return df, report

    df = load_raw_dataset(raw_path)
    df = add_datetime_columns(df)
    df = mark_missing_rows(df)

    measurements_per_day, missing_rows_per_day = get_daily_statistics(df)

    days_to_remove = find_days_to_remove(
        measurements_per_day,
        missing_rows_per_day,
    )

    df = clean_dataset(df, days_to_remove)
    hourly_df = create_hourly_dataset(df)

    report_df = create_report(
        measurements_per_day,
        missing_rows_per_day,
        days_to_remove,
        hourly_df,
    )

    save_outputs(hourly_df, report_df, processed_path, report_path)

    return hourly_df, report_df