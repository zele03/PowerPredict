from src.step01_preprocessing import create_processed_dataset
from src.step02_features import create_feature_dataset
from src.step03_split import create_train_validation_test_split


def main():
    create_processed_dataset()
    create_feature_dataset()
    create_train_validation_test_split()


if __name__ == "__main__":
    main()
