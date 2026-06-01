from pathlib import Path

import pandas as pd


def create_processed_dataset():
    raw_path = Path("data/raw/raw.txt")
    processed_path = Path("data/processed/processed.csv")

    df = pd.read_csv(raw_path, sep=";", na_values=["?"])

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )

    df["date_only"] = df["datetime"].dt.date

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    df["has_missing"] = df[numeric_columns].isna().any(axis=1)

    measurements_per_day = df.groupby("date_only").size()
    missing_rows_per_day = df.groupby("date_only")["has_missing"].sum()

    expected_rows_per_day = 1440
    missing_limit = 60

    # Koliko redova fali do punog dana od 1440 minuta
    incomplete_rows_per_day = expected_rows_per_day - measurements_per_day

    # Brišemo ceo dan samo ako mu fali više od 60 redova
    days_with_too_few_rows = incomplete_rows_per_day[
        incomplete_rows_per_day > missing_limit
    ]

    # Brišemo ceo dan ako ima više od 60 redova sa missing vrednostima
    days_with_too_many_missing = missing_rows_per_day[
        missing_rows_per_day > missing_limit
    ]

    days_to_remove = set(days_with_too_few_rows.index) | set(
        days_with_too_many_missing.index
    )

    print("Broj dana za potpuno brisanje:")
    print(len(days_to_remove))

    print("\nDani za potpuno brisanje:")
    print(sorted(days_to_remove))

    df = df[~df["date_only"].isin(days_to_remove)].copy()

    # Kod dana koji ostaju brišemo samo redove sa missing vrednostima
    df = df[~df["has_missing"]].copy()

    selected_columns = [
        "datetime",
        "Global_active_power",
        "Global_reactive_power",
        "Global_intensity",
    ]

    processed_df = df[selected_columns].copy()
    processed_df = df[selected_columns].copy()

    # datetime postaje indeks za grupisanje po satima
    processed_df = processed_df.set_index("datetime")

    # Od minutnih podataka pravimo satne proseke
    hourly_df = processed_df.resample("h").mean()

    # Brišemo sate koji su ostali prazni
    hourly_df = hourly_df.dropna()

    # datetime vraćamo kao običnu kolonu
    hourly_df = hourly_df.reset_index()

    # Zaokružujemo numeričke vrednosti da CSV bude čitljiviji
    hourly_df = hourly_df.round(4)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(processed_path, index=False)

    print("\nProcessed dataset je sačuvan u:")
    print(processed_path)

    print("\nBroj redova u processed datasetu:")
    print(len(hourly_df))

    print("\nKolone u processed datasetu:")
    print(hourly_df.columns.tolist())
