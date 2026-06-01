from src.step01_preprocessing import create_processed_dataset
from src.step02_features import create_feature_dataset


def main():
    create_processed_dataset()
    create_feature_dataset()


if __name__ == "__main__":
    main()
