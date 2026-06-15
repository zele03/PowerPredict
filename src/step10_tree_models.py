from itertools import product
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


TARGET_COLUMN = "Energy_kWh"
BASELINE_MODEL_NAME = "Baseline lag_24h"
RANDOM_SEED = 42
PERCENTAGE_MULTIPLIER = 100

# Ove kolone se ne koriste kao ulaz u modele.
# datetime nije numericka vrednost za sklearn model.
# Energy_kWh i Global_active_power predstavljaju potrosnju u istom satu koji
# zelimo da predvidimo, pa bi njihovo koriscenje bilo data leakage.
# Global_reactive_power i Global_intensity su takodje merenja iz trenutnog sata,
# pa ih izbacujemo da bi model radio samo sa informacijama dostupnim unapred.
EXCLUDED_FEATURE_COLUMNS = [
    "datetime",
    "Energy_kWh",
    "Global_active_power",
    "Global_reactive_power",
    "Global_intensity",
]


RANDOM_FOREST_GRID_SEARCH_SPACE = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 5],
}


GRADIENT_BOOSTING_GRID_SEARCH_SPACE = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 5],
}


MODEL_DEFINITIONS = {
    "random_forest": {
        "display_name": "Random Forest",
        "short_name": "RF",
        "estimator_class": RandomForestRegressor,
        "search_space": RANDOM_FOREST_GRID_SEARCH_SPACE,
        "logs_dir": Path("data/logs/random_forest_logs"),
        "predictions_dir": Path("data/predictions/random_forest_predictions"),
        "compare_report_path": Path(
            "data/logs/compare_logs/baseline_random_forest_compare_report.csv"
        ),
        "best_summary_filename": "best_random_forest_model_summary.csv",
        "model_summaries_filename": "random_forest_model_summaries.csv",
        "hyperparameter_report_filename": "random_forest_hyperparameter_report.csv",
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "short_name": "GB",
        "estimator_class": GradientBoostingRegressor,
        "search_space": GRADIENT_BOOSTING_GRID_SEARCH_SPACE,
        "logs_dir": Path("data/logs/gradient_boosting_logs"),
        "predictions_dir": Path("data/predictions/gradient_boosting_predictions"),
        "compare_report_path": Path(
            "data/logs/compare_logs/baseline_gradient_boosting_compare_report.csv"
        ),
        "best_summary_filename": "best_gradient_boosting_model_summary.csv",
        "model_summaries_filename": "gradient_boosting_model_summaries.csv",
        "hyperparameter_report_filename": "gradient_boosting_hyperparameter_report.csv",
    },
}


def format_hyperparameter_value(value):
    """
    Formatira hiperparametar za naziv fajla/modela.
    None pretvaramo u "none", a decimalnu tacku u "p" zbog citljivih imena.
    """
    if value is None:
        return "none"

    return str(value).replace(".", "p")


def build_model_config_name(model_key, config):
    """
    Kreira jedinstveno ime konfiguracije.
    Ime se kasnije koristi za model, predikcije, report i summary fajlove.
    """
    if model_key == "random_forest":
        return (
            f"random_forest"
            f"_n{config['n_estimators']}"
            f"_depth{format_hyperparameter_value(config['max_depth'])}"
            f"_leaf{config['min_samples_leaf']}"
        )

    return (
        f"gradient_boosting"
        f"_n{config['n_estimators']}"
        f"_lr{format_hyperparameter_value(config['learning_rate'])}"
        f"_depth{config['max_depth']}"
    )


def generate_grid_configs(model_key, search_space):
    """
    Od prostora hiperparametara pravi listu svih kombinacija.
    Ovo je isti princip kao kod GRU/LSTM grid search-a, samo za sklearn modele.
    """
    search_keys = list(search_space.keys())
    configs = []

    for values in product(*(search_space[key] for key in search_keys)):
        config = dict(zip(search_keys, values))
        config["name"] = build_model_config_name(model_key, config)
        configs.append(config)

    return configs


def load_split_datasets():
    """
    Ucitava vec napravljene train, validation i test skupove.
    Ovi modeli ne rade preprocessing sami, nego koriste postojece CSV fajlove.
    """
    split_dir = Path("data/processed/split")

    train_df = pd.read_csv(split_dir / "train.csv", parse_dates=["datetime"])
    validation_df = pd.read_csv(split_dir / "validation.csv", parse_dates=["datetime"])
    test_df = pd.read_csv(split_dir / "test.csv", parse_dates=["datetime"])

    return train_df, validation_df, test_df


def get_feature_columns(df):
    """
    Bira ulazne kolone za tabularne modele.
    Ostaju vremenski indikatori, lag vrednosti i rolling statistike.
    """
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
    ]


def split_features_and_target(df, feature_columns):
    """
    Od dataframe-a pravi X i y.
    X su ulazni feature-i, a y je target koji model uci da predvidja.
    """
    x_values = df[feature_columns]
    y_values = df[TARGET_COLUMN]

    return x_values, y_values


def build_estimator(model_key, config):
    """
    Pravi konkretan sklearn estimator za zadatu konfiguraciju.
    Random seed drzimo fiksnim da rezultati budu ponovljivi.
    """
    definition = MODEL_DEFINITIONS[model_key]
    estimator_class = definition["estimator_class"]

    return estimator_class(
        **{key: value for key, value in config.items() if key != "name"},
        random_state=RANDOM_SEED,
    )


def calculate_metrics(y_true, y_pred):
    """
    Racuna iste metrike kao baseline, GRU i LSTM: MAE, RMSE, MAPE, sMAPE i WAPE.
    Tako svi modeli mogu direktno da se porede.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

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


def create_predictions_df(df, predictions):
    """
    Pravi tabelu predikcija u istom formatu kao GRU/LSTM predikcije.
    Negativne predikcije secemo na 0 jer potrosnja energije ne moze biti negativna.
    """
    predictions_df = df[["datetime", TARGET_COLUMN]].copy()
    predictions_df = predictions_df.rename(
        columns={TARGET_COLUMN: "actual_Energy_kWh"}
    )

    predictions = np.maximum(predictions, 0)
    predictions_df["predicted_Energy_kWh"] = np.round(predictions, 4)
    predictions_df["absolute_error"] = (
        predictions_df["actual_Energy_kWh"]
        - predictions_df["predicted_Energy_kWh"]
    ).abs()

    return predictions_df


def create_prediction_report(predictions_df, dataset_name, model_name):
    """
    Kreira summary red i hourly redove za jedan skup podataka.
    Summary govori ukupan kvalitet, a hourly redovi pokazuju po kojim satima
    model najvise gresi.
    """
    y_true = predictions_df["actual_Energy_kWh"].to_numpy()
    y_pred = predictions_df["predicted_Energy_kWh"].to_numpy()

    report_rows = [
        {
            "Report_Type": "summary",
            "Dataset": dataset_name,
            "Model": model_name,
            "Hour": "",
            **calculate_metrics(y_true, y_pred),
            "Broj_redova": len(predictions_df),
        }
    ]

    predictions_df = predictions_df.copy()
    predictions_df["hour"] = predictions_df["datetime"].dt.hour

    for hour, hour_df in predictions_df.groupby("hour"):
        report_rows.append(
            {
                "Report_Type": "hourly",
                "Dataset": dataset_name,
                "Model": model_name,
                "Hour": int(hour),
                **calculate_metrics(
                    hour_df["actual_Energy_kWh"].to_numpy(),
                    hour_df["predicted_Energy_kWh"].to_numpy(),
                ),
                "Broj_redova": len(hour_df),
            }
        )

    return pd.DataFrame(report_rows)


def get_summary_metric(report_df, dataset_name, metric):
    """
    Iz reporta vadi jednu summary metriku, npr. validation MAE.
    To koristimo za izbor najbolje konfiguracije.
    """
    summary_row = report_df[
        (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == dataset_name)
    ].iloc[0]

    return float(summary_row[metric])


def load_baseline_report():
    """
    Ucitava baseline report ako postoji.
    Ako ne postoji, model se i dalje trenira, samo se preskace poredjenje.
    """
    report_path = Path("data/logs/baseline_report.csv")

    if not report_path.exists():
        return None

    return pd.read_csv(report_path)


def create_baseline_compare_report(baseline_report_df, model_report_df, model_name):
    """
    Poredi baseline i najbolji tree model nad istim test satima.
    Pozitivno poboljsanje znaci da je novi model smanjio gresku u odnosu na baseline.
    """
    baseline_df = baseline_report_df.copy()
    tree_df = model_report_df.copy()

    baseline_df["Model"] = BASELINE_MODEL_NAME
    baseline_df = baseline_df.drop(columns=["Baseline"], errors="ignore")

    for df in [baseline_df, tree_df]:
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
    tree_metrics = tree_df[key_columns + metric_columns + ["Broj_redova"]].rename(
        columns={
            **{metric: f"Tree_Model_{metric}" for metric in metric_columns},
            "Broj_redova": "Tree_Model_Broj_redova",
        }
    )

    compare_df = baseline_metrics.merge(tree_metrics, on=key_columns, how="inner")
    compare_df["Broj_redova"] = compare_df[
        ["Baseline_Broj_redova", "Tree_Model_Broj_redova"]
    ].min(axis=1)

    for metric in metric_columns:
        difference_column = f"{metric}_Difference_Baseline_minus_Tree_Model"
        improvement_column = f"{metric}_Improvement_percent"

        compare_df[difference_column] = (
            compare_df[f"Baseline_{metric}"] - compare_df[f"Tree_Model_{metric}"]
        ).round(4)
        compare_df[improvement_column] = (
            compare_df[difference_column] / compare_df[f"Baseline_{metric}"] * 100
        ).round(4)

    compare_df["Better_Model_by_MAE"] = np.where(
        compare_df["MAE_Difference_Baseline_minus_Tree_Model"] > 0,
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
        "Tree_Model_MAE",
        "MAE_Improvement_percent",
        "Baseline_RMSE",
        "Tree_Model_RMSE",
        "RMSE_Improvement_percent",
        "Baseline_MAPE",
        "Tree_Model_MAPE",
        "MAPE_Improvement_percent",
        "Baseline_sMAPE",
        "Tree_Model_sMAPE",
        "sMAPE_Improvement_percent",
        "Baseline_WAPE",
        "Tree_Model_WAPE",
        "WAPE_Improvement_percent",
    ]

    return compare_df[output_columns]


def create_model_summary_row(result):
    """
    Pravi jedan red summary tabele za jednu konfiguraciju.
    U taj red ulaze hiperparametri i glavne validation/test metrike.
    """
    config = result["config"]
    report_df = result["report_df"]
    validation_summary = report_df[
        (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == "validation")
    ].iloc[0]
    test_summary = report_df[
        (report_df["Report_Type"] == "summary") & (report_df["Dataset"] == "test")
    ].iloc[0]

    row = {
        "Model": config["name"],
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

    for key, value in config.items():
        if key != "name":
            row[key] = value

    return row


def save_model_pickle(estimator, feature_columns, config, model_key):
    """
    Cuva istrenirani sklearn model zajedno sa listom feature kolona.
    Feature kolone cuvamo jer su redosled i nazivi bitni pri kasnijem koriscenju.
    """
    models_dir = Path("models") / model_key
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_key": model_key,
        "config": config,
        "feature_columns": feature_columns,
        "estimator": estimator,
    }

    with open(models_dir / f"{config['name']}.pkl", "wb") as model_file:
        pickle.dump(checkpoint, model_file)


def create_feature_importance_df(estimator, feature_columns):
    """
    Tree modeli mogu da izracunaju koliko je svaki feature bio vazan.
    Veca vrednost znaci da je model taj feature vise koristio za dobre podele.
    """
    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": estimator.feature_importances_,
        }
    )
    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)
    importance_df["Importance"] = importance_df["Importance"].round(6)

    return importance_df


def save_model_outputs(model_key, result):
    """
    Cuva sve fajlove za jednu konfiguraciju: model, predikcije i report.
    """
    definition = MODEL_DEFINITIONS[model_key]
    config = result["config"]
    logs_dir = definition["logs_dir"]
    predictions_dir = definition["predictions_dir"]

    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    save_model_pickle(
        result["estimator"],
        result["feature_columns"],
        config,
        model_key,
    )
    result["validation_predictions_df"].to_csv(
        predictions_dir / f"{config['name']}_validation_predictions.csv",
        index=False,
    )
    result["test_predictions_df"].to_csv(
        predictions_dir / f"{config['name']}_test_predictions.csv",
        index=False,
    )
    result["report_df"].to_csv(logs_dir / f"{config['name']}_report.csv", index=False)
    create_feature_importance_df(
        result["estimator"],
        result["feature_columns"],
    ).to_csv(logs_dir / f"{config['name']}_feature_importance.csv", index=False)


def save_best_model_outputs(model_key, best_result):
    """
    Cuva summary najboljeg modela i baseline compare report.
    Najbolji model biramo po validation MAE, a test koristimo za finalni izvestaj.
    """
    definition = MODEL_DEFINITIONS[model_key]
    logs_dir = definition["logs_dir"]
    compare_report_path = definition["compare_report_path"]

    logs_dir.mkdir(parents=True, exist_ok=True)
    compare_report_path.parent.mkdir(parents=True, exist_ok=True)

    best_summary_df = pd.DataFrame([create_model_summary_row(best_result)])
    best_summary_df.insert(1, "selection_metric", "validation_MAE")
    best_summary_df.to_csv(
        logs_dir / definition["best_summary_filename"],
        index=False,
    )

    baseline_report_df = load_baseline_report()
    if baseline_report_df is None:
        return None

    baseline_test_report_df = baseline_report_df[
        baseline_report_df["Dataset"] == "test"
    ].copy()
    best_model_test_report_df = best_result["report_df"][
        best_result["report_df"]["Dataset"] == "test"
    ].copy()

    compare_report_df = create_baseline_compare_report(
        baseline_test_report_df,
        best_model_test_report_df,
        best_result["config"]["name"],
    )
    compare_report_df.to_csv(compare_report_path, index=False)

    return compare_report_df


def train_and_evaluate_model(model_key, config, datasets, feature_columns):
    """
    Trenira jednu konfiguraciju i odmah pravi validation/test predikcije.
    Validation metrika sluzi za izbor najboljeg modela, test metrika za kraj.
    """
    train_df, validation_df, test_df = datasets
    x_train, y_train = split_features_and_target(train_df, feature_columns)
    x_validation, _ = split_features_and_target(validation_df, feature_columns)
    x_test, _ = split_features_and_target(test_df, feature_columns)

    estimator = build_estimator(model_key, config)
    estimator.fit(x_train, y_train)

    validation_predictions_df = create_predictions_df(
        validation_df,
        estimator.predict(x_validation),
    )
    test_predictions_df = create_predictions_df(
        test_df,
        estimator.predict(x_test),
    )

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

    return {
        "config": config,
        "estimator": estimator,
        "feature_columns": feature_columns,
        "report_df": report_df,
        "validation_predictions_df": validation_predictions_df,
        "test_predictions_df": test_predictions_df,
        "validation_mae": get_summary_metric(report_df, "validation", "MAE"),
        "test_mae": get_summary_metric(report_df, "test", "MAE"),
    }


def create_tree_model_family(model_key):
    """
    Pokrece grid search za jednu porodicu modela: Random Forest ili Gradient Boosting.
    Svaka konfiguracija se trenira, evaluira i cuva u posebne fajlove.
    """
    definition = MODEL_DEFINITIONS[model_key]
    configs = generate_grid_configs(model_key, definition["search_space"])
    datasets = load_split_datasets()
    train_df, _, _ = datasets
    feature_columns = get_feature_columns(train_df)

    reports = []
    results = []

    for config in configs:
        result = train_and_evaluate_model(
            model_key,
            config,
            datasets,
            feature_columns,
        )
        save_model_outputs(model_key, result)
        results.append(result)
        reports.append(result["report_df"])

    combined_report_df = pd.concat(reports, ignore_index=True)
    logs_dir = definition["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    combined_report_df.to_csv(
        logs_dir / definition["hyperparameter_report_filename"],
        index=False,
    )

    model_summaries_df = pd.DataFrame(
        create_model_summary_row(result) for result in results
    )
    model_summaries_df.to_csv(
        logs_dir / definition["model_summaries_filename"],
        index=False,
    )

    best_result = min(results, key=lambda result: result["validation_mae"])
    save_best_model_outputs(model_key, best_result)

    return combined_report_df


def create_tree_models():
    """
    Glavna funkcija za step10.
    Treniraju se Random Forest i Gradient Boosting, jedan za drugim.
    """
    outputs = {}

    for model_key in MODEL_DEFINITIONS:
        outputs[model_key] = create_tree_model_family(model_key)

    return outputs


if __name__ == "__main__":
    create_tree_models()
