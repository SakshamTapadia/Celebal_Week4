"""
main.py: Entry point for loading and verifying Spambase dataset (numerical format)
"""

from utils.config import Config
from utils.logger import setup_logger
from data.data_loader import DataLoader
from data.data_splitter import train_test_split_stratified

logger = setup_logger(__name__)

def main():
    # Load config
    config = Config("config/config.yaml")
    data_path = config.get("path.dataset")
    test_size = config.get("training.test_size", 0.2)
    seed = config.get("training.random_state", 42)

    # Load data
    logger.info("Loading dataset from %s", data_path)
    loader = DataLoader(data_path)
    df = loader.load_data()

    # Use the last column as label for Spambase dataset
    feature_columns = df.columns[:-1]
    label_column = df.columns[-1]

    df.rename(columns={label_column: "label"}, inplace=True)
    df["label"] = df["label"].astype(int)

    X = df[feature_columns]
    y = df["label"]

    # Split
    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=test_size, seed=seed)

    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    logger.info("Label distribution in training: %s", y_train.value_counts().to_dict())

    logger.info("Dataset loaded and split successfully. You can now run `train.py`.")

if __name__ == "__main__":
    main()
