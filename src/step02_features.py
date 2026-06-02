import pandas as pd  # noqa: F401
from datetime import date  # noqa: F401
from pathlib import Path


def add_energy_feature(df, power_col="Global_active_power"):
    """
    Dodaje kolonu Energy_kWh iz Global_active_power.
    Pošto je dataset agregiran na satni nivo, vrednosti ostaju iste.
    """
    df["Energy_kWh"] = df[power_col] * 1
    return df


def add_datetime_features(df, datetime_col="datetime"):
    """
    Dodaje osnovne vremenske feature‑e.
    """
    df["hour"] = df[datetime_col].dt.hour
    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(0, 6).astype(int)
    return df


def add_lag_features(df, target_col="Energy_kWh"):
    """
    Dodaje lag feature‑e (istoriju).
    """
    df["lag_1h"] = df[target_col].shift(1)  # prosli sat
    df["lag_24h"] = df[target_col].shift(24)  # isti sat dan pre
    df["lag_168h"] = df[target_col].shift(168)  # isti sat 7 dana pre
    return df


def add_rolling_features(df, target_col="Energy_kWh"):
    """
    Dodaje rolling statistike.
    """
    df["rolling_mean_24h"] = (
        df[target_col].rolling(window=24).mean()
    )  # Za svaki sat računa se prosek vrednosti u poslednjih 24 sata zaključno sa tim satom
    df["rolling_std_7d"] = (
        df[target_col].rolling(window=24 * 7).std()
    )  # Za svaki sat računa se standardna devijacija vrednosti u poslednjih 7 dana (168 sati) zaključno sa tim satom
    return df


def add_holiday_feature(df, datetime_col="datetime", holidays=None):
    """
    Dodaje binarni indikator praznika.
    Ako se lista holidays ne prosledi, koristi se default lista francuskih praznika za 2026.
    """
    if holidays is None:
        holidays = [
            (1, 1),  # Nova godina
            (5, 1),  # Praznik rada
            (5, 8),  # Dan pobede 1945
            (7, 14),  # Dan Bastilje
            (8, 15),  # Velika Gospojina
            (11, 1),  # Svi sveti
            (11, 11),  # Dan primirja 1918
            (12, 25),  # Božić
        ]

    df["is_holiday"] = df[datetime_col].apply(
        lambda d: int((d.month, d.day) in holidays)
    )
    return df


def create_feature_dataset(
    processed_path="data/processed/processed.csv",
    features_path="data/processed/features.csv",
):
    """
    Kreira feature dataset iz processed.csv i snima ga u features.csv.
    Ako features.csv već postoji i identičan je novom, preskače snimanje.
    Ako postoji ali se razlikuje, prepisuje ga.
    """
    processed_path = Path(processed_path)
    features_path = Path(features_path)

    if not processed_path.exists():
        raise FileNotFoundError(
            f"{processed_path} ne postoji. Prvo pokreni preprocessing."
        )

    # učitaj processed.csv
    df = pd.read_csv(processed_path, parse_dates=["datetime"])

    # pozovi sve feature funkcije redom
    df_new = add_energy_feature(df.copy())
    df_new = add_datetime_features(df_new)
    df_new = add_lag_features(df_new)
    df_new = add_rolling_features(df_new)
    df_new = add_holiday_feature(df_new)

    # zaštita: ako fajl već postoji, uporedi
    if features_path.exists():
        df_old = pd.read_csv(features_path, parse_dates=["datetime"])
        if df_old.equals(df_new):
            return df_old

    # snimi novi fajl
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(features_path, index=False)
    return df_new
