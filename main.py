import argparse

from src.step01_preprocessing import create_processed_dataset
from src.step02_features import create_feature_dataset
from src.step03_split import create_train_validation_test_split
from src.step04_baseline_model import create_baseline_model
from src.step05_baseline_visualization import create_baseline_graphs


def run_baseline_pipeline():
    """
    Pokrece standardni workflow bez GRU/LSTM treninga.
    """
    create_processed_dataset()
    create_feature_dataset()
    create_train_validation_test_split()
    create_baseline_model()
    create_baseline_graphs()


def run_gru_pipeline():
    """
    Pokrece GRU trening i zatim sve GRU vizualizacije.
    """
    from src.step06_gru_model import create_gru_models
    from src.step07_gru_visualization import (
        create_baseline_gru_compare_graphs,
        create_gru_graphs,
    )

    create_gru_models()
    create_gru_graphs()
    create_baseline_gru_compare_graphs()


def run_lstm_pipeline():
    """
    Pokrece LSTM trening i zatim sve LSTM vizualizacije.
    """
    from src.step08_lstm_model import create_lstm_models
    from src.step09_lstm_visualization import (
        create_baseline_lstm_compare_graphs,
        create_lstm_graphs,
    )

    create_lstm_models()
    create_lstm_graphs()
    create_baseline_lstm_compare_graphs()


def parse_args():
    parser = argparse.ArgumentParser(
        description="PowerPredict workflow runner.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=[
            "gru-pipeline",
            "lstm-pipeline",
        ],
        help=(
            "Opcioni workflow. Bez argumenta pokrece sve do neural network treninga."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "gru-pipeline":
        run_gru_pipeline()
    elif args.command == "lstm-pipeline":
        run_lstm_pipeline()
    else:
        run_baseline_pipeline()


if __name__ == "__main__":
    main()
