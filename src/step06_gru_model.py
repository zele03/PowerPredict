from itertools import product
from pathlib import Path
from tqdm import trange

import numpy as np
import pandas as pd


TARGET_COLUMN = "Energy_kWh"  # Kolona koju model pokusava da predvidi
MODEL_NAME = "GRU"
BASELINE_MODEL_NAME = "Baseline lag_24h"
RANDOM_SEED = 42  # Seed za ponovljivost rezultata
PERCENTAGE_MULTIPLIER = 100  # Koristi se da procentualne metrike budu u procentima

DEFAULT_SEQUENCE_LENGTH = 24  # Podrazumevani broj prethodnih sati za jednu predikciju
DEFAULT_BATCH_SIZE = 64  # Podrazumevani broj sekvenci u jednom trening batch-u
DEFAULT_HIDDEN_SIZE = 64  # Podrazumevana velicina skrivene memorije GRU sloja
DEFAULT_NUM_LAYERS = 2  # Podrazumevani broj naslaganih GRU slojeva
DEFAULT_DROPOUT = 0.2  # Podrazumevana dropout regularizacija
DEFAULT_LEARNING_RATE = 0.001  # Podrazumevana brzina ucenja optimizatora
DEFAULT_MAX_EPOCHS = 80  # Podrazumevani maksimalan broj epoha treninga
DEFAULT_PATIENCE = 10  # Podrazumevani early stopping patience


GRU_GRID_SEARCH_SPACE = {
    "sequence_length": [12, 24, 48],
    "batch_size": [64],
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.2, 0.3],
    "learning_rate": [0.001, 0.0005],
    "max_epochs": [80],
    "patience": [10],
}

EXCLUDED_FEATURE_COLUMNS = [
    "datetime",
    "Global_active_power",
]


def format_hyperparameter_value(value):
    """
    Formatira vrednosti hiperparametara za naziv modela.
    """
    return str(value).replace(".", "p")


def build_gru_config_name(config):
    """
    Kreira jedinstven naziv modela na osnovu hiperparametara.
    """
    return (
        f"gru_seq{config['sequence_length']}"
        f"_h{config['hidden_size']}"
        f"_l{config['num_layers']}"
        f"_d{format_hyperparameter_value(config['dropout'])}"
        f"_lr{format_hyperparameter_value(config['learning_rate'])}"
        f"_bs{config['batch_size']}"
    )


def generate_gru_grid_configs(search_space=GRU_GRID_SEARCH_SPACE):
    """
    Generise sve validne kombinacije hiperparametara za GRU grid search.
    """
    search_keys = [
        "sequence_length",
        "batch_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "max_epochs",
        "patience",
    ]
    configs = []

    for values in product(*(search_space[key] for key in search_keys)):
        config = dict(zip(search_keys, values))

        if config["num_layers"] == 1 and config["dropout"] > 0:
            continue

        config["name"] = build_gru_config_name(config)
        configs.append(config)

    return configs


HYPERPARAMETER_CONFIGS = generate_gru_grid_configs()


def import_torch():
    """
    Ucitava torch tek kada se pokrece GRU trening.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch nije instaliran. Instaliraj torch u okruzenju sa CUDA podrskom "
            "pre pokretanja GRU treninga."
        ) from exc

    return torch, nn, DataLoader, TensorDataset


class StandardScaler:
    """
    Jednostavan standard scaler za numpy nizove.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, values):
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        self.std_[self.std_ == 0] = 1

    def transform(self, values):
        return (values - self.mean_) / self.std_

    def inverse_transform(self, values):
        return values * self.std_ + self.mean_

    def to_dict(self):
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }


def load_split_datasets():
    """
    Ucitava train, validation i test skupove iz split foldera.
    """
    split_dir = Path("data/processed/split")

    train_df = pd.read_csv(split_dir / "train.csv", parse_dates=["datetime"])
    validation_df = pd.read_csv(split_dir / "validation.csv", parse_dates=["datetime"])
    test_df = pd.read_csv(split_dir / "test.csv", parse_dates=["datetime"])

    return train_df, validation_df, test_df


def get_feature_columns(df):
    """
    Bira feature kolone koje ne sadrze target za isti sat.
    """
    feature_columns = [
        column for column in df.columns if column not in EXCLUDED_FEATURE_COLUMNS
    ]

    return feature_columns


def fit_scalers(train_df, feature_columns):
    """
    Fit-uje scalere samo na train skupu.
    """
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(train_df[feature_columns].to_numpy(dtype=np.float32))
    target_scaler.fit(train_df[[TARGET_COLUMN]].to_numpy(dtype=np.float32))

    return feature_scaler, target_scaler


def has_continuous_time_window(datetimes):
    """
    Proverava da li su svi sati u sekvenci uzastopni.
    """
    diffs = datetimes.diff().dropna()
    return (diffs == pd.Timedelta(hours=1)).all()


def create_sequences(
    context_df,
    eval_df,
    feature_columns,
    feature_scaler,
    target_scaler,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
):
    """
    Kreira GRU sekvence koristeci istoriju iz context_df i targete iz eval_df.
    """
    combined_df = pd.concat(
        [context_df.tail(sequence_length), eval_df],
        ignore_index=True,
    )

    features = feature_scaler.transform(
        combined_df[feature_columns].to_numpy(dtype=np.float32)
    )
    targets = target_scaler.transform(
        combined_df[[TARGET_COLUMN]].to_numpy(dtype=np.float32)
    ).reshape(-1)

    x_values = []
    y_values = []
    datetimes = []
    actual_values = []

    first_eval_index = len(context_df.tail(sequence_length))

    for target_index in range(first_eval_index, len(combined_df)):
        sequence_start = target_index - sequence_length

        if sequence_start < 0:
            continue

        time_window = combined_df.loc[sequence_start:target_index, "datetime"]
        if not has_continuous_time_window(time_window):
            continue

        x_values.append(features[sequence_start:target_index])
        y_values.append(targets[target_index])
        datetimes.append(combined_df.loc[target_index, "datetime"])
        actual_values.append(combined_df.loc[target_index, TARGET_COLUMN])

    return (
        np.array(x_values, dtype=np.float32),
        np.array(y_values, dtype=np.float32),
        pd.DataFrame(
            {
                "datetime": datetimes,
                "actual_Energy_kWh": actual_values,
            }
        ),
    )


def build_gru_model_class(nn, config):
    """
    Kreira GRU model klasu nakon sto je torch dostupan.
    """

    class GRURegressor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                batch_first=True,
                dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            )
            self.output_layer = nn.Linear(config["hidden_size"], 1)

        def forward(self, x):
            output, _ = self.gru(x)
            last_output = output[:, -1, :]
            prediction = self.output_layer(last_output)
            return prediction.squeeze(-1)

    return GRURegressor


def create_data_loader(
    torch,
    DataLoader,
    TensorDataset,
    x_values,
    y_values,
    batch_size,
    shuffle,
):
    """
    Kreira PyTorch DataLoader.
    """
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(torch, model, data_loader, loss_fn, optimizer, device):
    """
    Trenira model tokom jedne epohe.
    """
    model.train()
    losses = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def evaluate_loss(torch, model, data_loader, loss_fn, device):
    """
    Racuna loss bez treniranja.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            losses.append(loss.item())

    return float(np.mean(losses))


def train_gru_model(model, train_loader, validation_loader, device, torch, nn, config):
    """
    Trenira GRU model sa early stopping logikom.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_validation_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    history_rows = []

    progress_bar = trange(
        1,
        config["max_epochs"] + 1,
        desc=f"Training {config['name']}",
    )

    for epoch in progress_bar:
        train_loss = train_one_epoch(
            torch,
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )
        validation_loss = evaluate_loss(
            torch,
            model,
            validation_loader,
            loss_fn,
            device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "validation_loss": round(validation_loss, 6),
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress_bar.set_description(
            f"Epoch {epoch} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {validation_loss:.4f} | "
            f"Best: {best_validation_loss:.4f}"
        )

        if epochs_without_improvement >= config["patience"]:
            break

    model.load_state_dict(best_state_dict)

    return model, pd.DataFrame(history_rows)


def predict(torch, model, x_values, target_scaler, device):
    """
    Vraca predikcije u originalnoj kWh skali.
    """
    model.eval()
    x_tensor = torch.tensor(x_values, dtype=torch.float32).to(device)

    with torch.no_grad():
        scaled_predictions = model(x_tensor).cpu().numpy().reshape(-1, 1)

    predictions = target_scaler.inverse_transform(scaled_predictions).reshape(-1)
    predictions = np.maximum(predictions, 0)

    return predictions


def calculate_metrics(y_true, y_pred):
    """
    Racuna MAE, RMSE, MAPE, sMAPE i WAPE.
    """
    errors = y_true - y_pred
    absolute_errors = np.abs(errors)

    mae = absolute_errors.mean()
    rmse = np.sqrt(np.mean(errors**2))
    wape = absolute_errors.sum() / y_true.sum() * PERCENTAGE_MULTIPLIER

    valid_mape_mask = y_true > 0
    mape = (
        np.mean(absolute_errors[valid_mape_mask] / y_true[valid_mape_mask])
        * PERCENTAGE_MULTIPLIER
    )

    smape_denominator = np.abs(y_true) + np.abs(y_pred)
    valid_smape_mask = smape_denominator > 0
    smape = (
        np.mean(
            2 * absolute_errors[valid_smape_mask] / smape_denominator[valid_smape_mask]
        )
        * PERCENTAGE_MULTIPLIER
    )

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "sMAPE": round(smape, 4),
        "WAPE": round(wape, 4),
    }


def create_prediction_report(predictions_df, dataset_name, model_name=MODEL_NAME):
    """
    Kreira summary i hourly report za GRU predikcije.
    """
    y_true = predictions_df["actual_Energy_kWh"].to_numpy()
    y_pred = predictions_df["predicted_Energy_kWh"].to_numpy()

    summary_metrics = calculate_metrics(y_true, y_pred)
    report_rows = [
        {
            "Report_Type": "summary",
            "Dataset": dataset_name,
            "Model": model_name,
            "Hour": "",
            **summary_metrics,
            "Broj_redova": len(predictions_df),
        }
    ]

    predictions_df = predictions_df.copy()
    predictions_df["hour"] = predictions_df["datetime"].dt.hour

    for hour, hour_df in predictions_df.groupby("hour"):
        hourly_metrics = calculate_metrics(
            hour_df["actual_Energy_kWh"].to_numpy(),
            hour_df["predicted_Energy_kWh"].to_numpy(),
        )
        report_rows.append(
            {
                "Report_Type": "hourly",
                "Dataset": dataset_name,
                "Model": model_name,
                "Hour": int(hour),
                **hourly_metrics,
                "Broj_redova": len(hour_df),
            }
        )

    return pd.DataFrame(report_rows)


def load_baseline_report():
    """
    Ucitava baseline report ako postoji.
    """
    report_path = Path("data/logs/baseline_report.csv")

    if not report_path.exists():
        return None

    return pd.read_csv(report_path)


def create_baseline_gru_compare_report(
    baseline_report_df, gru_report_df, model_name=MODEL_NAME
):
    """
    Kreira kompaktan uporedni report sa baseline i GRU metrikama.
    """
    baseline_df = baseline_report_df.copy()
    gru_df = gru_report_df.copy()

    baseline_df["Model"] = BASELINE_MODEL_NAME
    baseline_df = baseline_df.drop(columns=["Baseline"], errors="ignore")

    for df in [baseline_df, gru_df]:
        df["Hour"] = (
            df["Hour"]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

    metric_columns = ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE"]
    key_columns = ["Report_Type", "Dataset", "Hour"]

    baseline_metrics = baseline_df[
        key_columns + metric_columns + ["Broj_redova"]
    ].rename(
        columns={
            **{metric: f"Baseline_{metric}" for metric in metric_columns},
            "Broj_redova": "Baseline_Broj_redova",
        }
    )
    gru_metrics = gru_df[key_columns + metric_columns + ["Broj_redova"]].rename(
        columns={
            **{metric: f"GRU_{metric}" for metric in metric_columns},
            "Broj_redova": "GRU_Broj_redova",
        }
    )

    compare_df = baseline_metrics.merge(gru_metrics, on=key_columns, how="inner")
    compare_df["Broj_redova"] = compare_df[
        ["Baseline_Broj_redova", "GRU_Broj_redova"]
    ].min(axis=1)

    for metric in metric_columns:
        difference_column = f"{metric}_Difference_Baseline_minus_GRU"
        improvement_column = f"{metric}_Improvement_percent"

        compare_df[difference_column] = (
            compare_df[f"Baseline_{metric}"] - compare_df[f"GRU_{metric}"]
        ).round(4)
        compare_df[improvement_column] = (
            compare_df[difference_column] / compare_df[f"Baseline_{metric}"] * 100
        ).round(4)

    compare_df["Better_Model_by_MAE"] = np.where(
        compare_df["MAE_Difference_Baseline_minus_GRU"] > 0,
        model_name,
        BASELINE_MODEL_NAME,
    )

    output_columns = [
        "Report_Type",
        "Dataset",
        "Hour",
        "Broj_redova",
        "Better_Model_by_MAE",
        "Baseline_MAE",
        "GRU_MAE",
        "MAE_Improvement_percent",
        "Baseline_RMSE",
        "GRU_RMSE",
        "RMSE_Improvement_percent",
        "Baseline_MAPE",
        "GRU_MAPE",
        "MAPE_Improvement_percent",
        "Baseline_sMAPE",
        "GRU_sMAPE",
        "sMAPE_Improvement_percent",
        "Baseline_WAPE",
        "GRU_WAPE",
        "WAPE_Improvement_percent",
    ]

    return compare_df[output_columns]


def get_summary_metric(report_df, dataset_name, metric):
    """
    Vraca summary metriku za trazeni skup.
    """
    summary_row = report_df[
        (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == dataset_name)
    ].iloc[0]

    return float(summary_row[metric])


def create_best_gru_summary(best_result):
    """
    Kreira kratak zapis o najboljoj GRU konfiguraciji.
    """
    config = best_result["config"]
    report_df = best_result["report_df"]
    validation_summary = report_df[
        (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == "validation")
    ].iloc[0]
    test_summary = report_df[
        (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == "test")
    ].iloc[0]

    return pd.DataFrame(
        [
            {
                "Model": config["name"],
                "selection_metric": "validation_MAE",
                "sequence_length": config["sequence_length"],
                "batch_size": config["batch_size"],
                "hidden_size": config["hidden_size"],
                "num_layers": config["num_layers"],
                "dropout": config["dropout"],
                "learning_rate": config["learning_rate"],
                "max_epochs": config["max_epochs"],
                "patience": config["patience"],
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
        ]
    )


def create_gru_model_summaries(results):
    """
    Kreira tabelu sa hiperparametrima i test metrikama svih GRU modela.
    """
    summary_rows = []

    for result in results:
        config = result["config"]
        report_df = result["report_df"]
        validation_summary = report_df[
            (report_df["Report_Type"] == "summary")
            & (report_df["Dataset"] == "validation")
        ].iloc[0]
        test_summary = report_df[
            (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == "test")
        ].iloc[0]

        summary_rows.append(
            {
                "Model": config["name"],
                "sequence_length": config["sequence_length"],
                "batch_size": config["batch_size"],
                "hidden_size": config["hidden_size"],
                "num_layers": config["num_layers"],
                "dropout": config["dropout"],
                "learning_rate": config["learning_rate"],
                "max_epochs": config["max_epochs"],
                "patience": config["patience"],
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


def save_best_gru_outputs(best_result):
    """
    Cuva summary najboljeg GRU modela i compare report sa baseline modelom.
    """
    gru_logs_dir = Path("data/logs/gru_logs")
    compare_logs_dir = Path("data/logs/compare_logs")
    gru_logs_dir.mkdir(parents=True, exist_ok=True)
    compare_logs_dir.mkdir(parents=True, exist_ok=True)

    best_summary_df = create_best_gru_summary(best_result)
    best_summary_df.to_csv(gru_logs_dir / "best_gru_model_summary.csv", index=False)

    baseline_report_df = load_baseline_report()
    if baseline_report_df is None:
        return None

    baseline_test_report_df = baseline_report_df[
        baseline_report_df["Dataset"] == "test"
    ].copy()
    best_gru_test_report_df = best_result["report_df"][
        best_result["report_df"]["Dataset"] == "test"
    ].copy()

    compare_report_df = create_baseline_gru_compare_report(
        baseline_test_report_df,
        best_gru_test_report_df,
        best_result["config"]["name"],
    )
    compare_report_df.to_csv(
        compare_logs_dir / "baseline_gru_compare_report.csv",
        index=False,
    )

    return compare_report_df


def save_outputs(
    torch,
    model,
    feature_columns,
    feature_scaler,
    target_scaler,
    history_df,
    validation_predictions_df,
    test_predictions_df,
    report_df,
    compare_report_df=None,
    config=None,
):
    """
    Cuva model, istoriju treninga, predikcije i GRU report.
    """
    models_dir = Path("models/gru")
    logs_dir = Path("data/logs/gru_logs")
    predictions_dir = Path("data/predictions/gru_predictions")

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    config = config or {
        "name": "gru_model",
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "batch_size": DEFAULT_BATCH_SIZE,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "num_layers": DEFAULT_NUM_LAYERS,
        "dropout": DEFAULT_DROPOUT,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "patience": DEFAULT_PATIENCE,
    }
    output_name = config["name"]

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "feature_scaler": feature_scaler.to_dict(),
        "target_scaler": target_scaler.to_dict(),
        "sequence_length": config["sequence_length"],
        "batch_size": config["batch_size"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "max_epochs": config["max_epochs"],
        "patience": config["patience"],
    }

    torch.save(checkpoint, models_dir / f"{output_name}.pt")
    history_df.to_csv(logs_dir / f"{output_name}_training_history.csv", index=False)
    validation_predictions_df.to_csv(
        predictions_dir / f"{output_name}_validation_predictions.csv",
        index=False,
    )
    test_predictions_df.to_csv(
        predictions_dir / f"{output_name}_test_predictions.csv",
        index=False,
    )
    report_df.to_csv(logs_dir / f"{output_name}_report.csv", index=False)

    if compare_report_df is not None:
        compare_report_df.to_csv(
            logs_dir / f"{output_name}_baseline_compare_report.csv",
            index=False,
        )


def create_gru_model(config):
    """
    Glavna funkcija za trening i evaluaciju GRU modela.
    """
    torch, nn, DataLoader, TensorDataset = import_torch()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, validation_df, test_df = load_split_datasets()
    feature_columns = get_feature_columns(train_df)
    feature_scaler, target_scaler = fit_scalers(train_df, feature_columns)

    x_train, y_train, _ = create_sequences(
        train_df.head(config["sequence_length"]),
        train_df.iloc[config["sequence_length"] :].copy(),
        feature_columns,
        feature_scaler,
        target_scaler,
        sequence_length=config["sequence_length"],
    )
    x_validation, y_validation, validation_predictions_df = create_sequences(
        train_df,
        validation_df,
        feature_columns,
        feature_scaler,
        target_scaler,
        sequence_length=config["sequence_length"],
    )
    x_test, y_test, test_predictions_df = create_sequences(
        validation_df,
        test_df,
        feature_columns,
        feature_scaler,
        target_scaler,
        sequence_length=config["sequence_length"],
    )

    train_loader = create_data_loader(
        torch,
        DataLoader,
        TensorDataset,
        x_train,
        y_train,
        config["batch_size"],
        shuffle=True,
    )
    validation_loader = create_data_loader(
        torch,
        DataLoader,
        TensorDataset,
        x_validation,
        y_validation,
        config["batch_size"],
        shuffle=False,
    )

    GRURegressor = build_gru_model_class(nn, config)
    model = GRURegressor(input_size=len(feature_columns)).to(device)

    model, history_df = train_gru_model(
        model,
        train_loader,
        validation_loader,
        device,
        torch,
        nn,
        config,
    )

    validation_predictions = predict(
        torch,
        model,
        x_validation,
        target_scaler,
        device,
    )
    test_predictions = predict(torch, model, x_test, target_scaler, device)

    validation_predictions_df["predicted_Energy_kWh"] = np.round(
        validation_predictions,
        4,
    )
    test_predictions_df["predicted_Energy_kWh"] = np.round(test_predictions, 4)

    validation_predictions_df["absolute_error"] = (
        validation_predictions_df["actual_Energy_kWh"]
        - validation_predictions_df["predicted_Energy_kWh"]
    ).abs()
    test_predictions_df["absolute_error"] = (
        test_predictions_df["actual_Energy_kWh"]
        - test_predictions_df["predicted_Energy_kWh"]
    ).abs()

    report_df = pd.concat(
        [
            create_prediction_report(
                validation_predictions_df,
                "validation",
                config["name"],
            ),
            create_prediction_report(test_predictions_df, "test", config["name"]),
        ],
        ignore_index=True,
    )

    save_outputs(
        torch,
        model,
        feature_columns,
        feature_scaler,
        target_scaler,
        history_df,
        validation_predictions_df,
        test_predictions_df,
        report_df,
        compare_report_df=None,
        config=config,
    )

    return {
        "config": config,
        "report_df": report_df,
        "history_df": history_df,
        "validation_predictions_df": validation_predictions_df,
        "test_predictions_df": test_predictions_df,
        "validation_mae": get_summary_metric(report_df, "validation", "MAE"),
        "test_mae": get_summary_metric(report_df, "test", "MAE"),
    }


def create_gru_models():
    """
    Trenira po jedan GRU model za svaku konfiguraciju hiperparametara.
    """
    reports = []
    results = []

    for config in HYPERPARAMETER_CONFIGS:
        result = create_gru_model(config)
        results.append(result)
        reports.append(result["report_df"])

    combined_report_df = pd.concat(reports, ignore_index=True)
    logs_dir = Path("data/logs/gru_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    combined_report_df.to_csv(logs_dir / "gru_hyperparameter_report.csv", index=False)
    create_gru_model_summaries(results).to_csv(
        logs_dir / "gru_model_summaries.csv",
        index=False,
    )

    best_result = min(results, key=lambda result: result["validation_mae"])
    save_best_gru_outputs(best_result)

    return combined_report_df


if __name__ == "__main__":
    create_gru_models()
