import argparse

from src.step01_preprocessing import create_processed_dataset
from src.step02_features import create_feature_dataset
from src.step03_split import create_train_validation_test_split
from src.step04_baseline_model import create_baseline_model
from src.step05_baseline_visualization import create_baseline_graphs


def run_baseline_pipeline():
    """
    Pokrece standardni workflow bez GRU treninga.
    """
    create_processed_dataset()
    create_feature_dataset()
    create_train_validation_test_split()
    create_baseline_model()
    create_baseline_graphs()


def run_gru_training():
    """
    Pokrece GRU trening samo kada korisnik eksplicitno izabere tu komandu.
    """
    from src.step06_gru_model import create_gru_model

    create_gru_model()


def run_gru_graphs():
    """
    Kreira osnovne GRU grafike iz vec istreniranog modela.
    """
    from src.step07_gru_visualization import create_gru_graphs

    create_gru_graphs()


def run_gru_compare_graphs():
    """
    Kreira uporedne baseline vs GRU grafike.
    """
    from src.step07_gru_visualization import create_baseline_gru_compare_graphs

    create_baseline_gru_compare_graphs()


def run_gru_visualizations():
    """
    Kreira sve GRU vizualizacije nakon treninga.
    """
    run_gru_graphs()
    run_gru_compare_graphs()


def run_gru_pipeline():
    """
    Pokrece GRU trening i zatim sve GRU vizualizacije.
    """
    run_gru_training()
    run_gru_visualizations()


def parse_args():
    parser = argparse.ArgumentParser(
        description="PowerPredict workflow runner.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="baseline",
        choices=[
            "baseline",
            "gru-train",
            "gru-graphs",
            "gru-compare-graphs",
            "gru-visualizations",
            "gru-pipeline",
        ],
        help=(
            "Workflow koji treba pokrenuti. "
            "Default je 'baseline', bez PyTorch/GRU treninga."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "baseline":
        run_baseline_pipeline()
    elif args.command == "gru-train":
        run_gru_training()
    elif args.command == "gru-graphs":
        run_gru_graphs()
    elif args.command == "gru-compare-graphs":
        run_gru_compare_graphs()
    elif args.command == "gru-visualizations":
        run_gru_visualizations()
    elif args.command == "gru-pipeline":
        run_gru_pipeline()


if __name__ == "__main__":
    main()
