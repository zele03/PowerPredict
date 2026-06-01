from pathlib import Path
import pandas as pd

def create_processed_dataset():
    raw_path = Path("data/raw/raw.txt")
    processed_path = Path("data/processed/processed.csv")
    report_path = Path("data/processed/logs/full_report.csv")

    # Ako fajlovi već postoje, samo ih učitaj
    if processed_path.exists() and report_path.exists():
        df = pd.read_csv(processed_path)
        report = pd.read_csv(report_path)
        return df, report

    # 1) Učitavanje raw podataka
    df = pd.read_csv(raw_path, sep=";", na_values=["?"])

    # 2) Spajanje Date i Time u jednu datetime kolonu
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    df["date_only"] = df["datetime"].dt.date

    # 3) Definisanje numeričkih kolona koje proveravamo za missing vrednosti
    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    # 4) Obeležavanje redova koji imaju missing vrednosti
    df["has_missing"] = df[numeric_columns].isna().any(axis=1)

    # 5) Grupisanje po danima – broj merenja i broj missing redova
    measurements_per_day = df.groupby("date_only").size()
    missing_rows_per_day = df.groupby("date_only")["has_missing"].sum()

    expected_rows_per_day = 1440  # broj minuta u danu
    missing_limit = 60            # prag tolerancije

    # 6) Računamo koliko redova fali do punog dana
    incomplete_rows_per_day = expected_rows_per_day - measurements_per_day

    # 7) Dani sa previše malo redova
    days_with_too_few_rows = incomplete_rows_per_day[
        incomplete_rows_per_day > missing_limit
    ]

    # 8) Dani sa previše missing vrednosti
    days_with_too_many_missing = missing_rows_per_day[
        missing_rows_per_day > missing_limit
    ]

    # 9) Dani koje potpuno brišemo (spajamo oba kriterijuma)
    days_to_remove = set(days_with_too_few_rows.index) | set(
        days_with_too_many_missing.index
    )

    # 10) Brišemo cele dane koji ne zadovoljavaju kriterijume
    df = df[~df["date_only"].isin(days_to_remove)].copy()

    # 11) Kod preostalih dana brišemo samo redove sa missing vrednostima
    df = df[~df["has_missing"]].copy()

    # 12) Biramo samo relevantne kolone
    selected_columns = [
        "datetime",
        "Global_active_power",
        "Global_reactive_power",
        "Global_intensity",
    ]
    processed_df = df[selected_columns].copy()

    # 13) Postavljamo datetime kao indeks radi resamplovanja
    processed_df = processed_df.set_index("datetime")

    # 14) Resamplujemo minutne podatke u satne proseke
    hourly_df = processed_df.resample("h").mean()

    # 15) Brišemo prazne sate
    hourly_df = hourly_df.dropna()

    # 16) Vraćamo datetime kao običnu kolonu
    hourly_df = hourly_df.reset_index()

    # 17) Zaokružujemo vrednosti radi čitljivosti
    numeric_cols = hourly_df.select_dtypes(include="number").columns
    hourly_df[numeric_cols] = hourly_df[numeric_cols].round(4)


    # 18) Snimamo rezultat u processed.csv
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(processed_path, index=False)

    # 19) Snimamo detaljan report u logs/full_report.csv
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_entries = [
      {"Opis": "Ukupan broj dana u raw datasetu", "Vrednost": len(measurements_per_day)},
        {"Opis": "Broj dana obrisanih", "Vrednost": len(days_to_remove)},
        {"Opis": "Lista obrisanih dana", "Vrednost": sorted(days_to_remove)},
        {"Opis": "Broj redova u processed datasetu", "Vrednost": len(hourly_df)},
        {"Opis": "Kolone u processed datasetu", "Vrednost": hourly_df.columns.tolist()},
        {"Opis": "Prosečan broj redova po danu", "Vrednost": measurements_per_day.mean()},
        {"Opis": "Prosečan broj missing vrednosti po danu", "Vrednost": missing_rows_per_day.mean()},
    ]

    report_df = pd.DataFrame(report_entries)
    report_df.to_csv(report_path, index=False)


    return hourly_df, report_df
