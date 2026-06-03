from src.step01_preprocessing import create_processed_dataset
from src.step02_features import create_feature_dataset
from src.step03_split import create_train_validation_test_split
from src.step04_baseline_model import create_baseline_model
from src.step05_baseline_visualization import create_baseline_graphs


def main():
    create_processed_dataset()
    create_feature_dataset()
    create_train_validation_test_split()
    create_baseline_model()
    create_baseline_graphs()


if __name__ == "__main__":
    main()
