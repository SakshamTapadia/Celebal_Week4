"""
train.py: Full training pipeline for numerical Spambase dataset.
"""

from utils.config import Config
from utils.logger import setup_logger
from utils.helpers import set_seed
from data.data_loader import DataLoader
from data.data_splitter import train_test_split_stratified
from models.model_selector import ModelSelector
from evaluation.evaluator import ModelEvaluator
from evaluation.visualizer import plot_confusion_matrix, plot_roc_curve

import joblib
import os

logger = setup_logger(__name__)

def main():
    # Load config
    cfg = Config("config/config.yaml")
    set_seed(cfg.get("training.random_state", 42))

    # Load data
    data_path = cfg.get("path.dataset")
    loader = DataLoader(data_path)
    df = loader.load_data()

    # Identify label and features
    feature_columns = df.columns[:-1]
    label_column = df.columns[-1]

    df.rename(columns={label_column: "label"}, inplace=True)
    df["label"] = df["label"].astype(int)

    # Sanitize feature column names (remove/replace invalid characters)
    sanitized_columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in feature_columns]
    df.columns = sanitized_columns + ["label"]

    X = df[sanitized_columns]
    y = df["label"]

    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=cfg.get("training.test_size", 0.2), seed=cfg.get("training.random_state", 42)
    )

    # Model selection & training
    model_name = cfg.get("model.type", "XGBoost")
    model_params = cfg.get("model.parameters", {})
    selector = ModelSelector(model_name=model_name, param_grid=model_params)
    model = selector.get_model_class()(**model_params)
    model.train(X_train, y_train)

    # Evaluation
    evaluator = ModelEvaluator()
    y_pred = model.predict(X_test)
    y_proba = getattr(model.model, "predict_proba", lambda x: None)(X_test)
    y_proba = y_proba[:, 1] if y_proba is not None else None

    evaluator.evaluate_model(model_name, y_test, y_pred, y_proba)
    print(evaluator.summarize_results())

    # Plots
    plot_confusion_matrix(y_test, y_pred)
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba)

    # Save model
    model_dir = f"artifacts/{model_name.lower()}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.joblib"))

    logger.info("Model training and saving complete.")

if __name__ == "__main__":
    main()
