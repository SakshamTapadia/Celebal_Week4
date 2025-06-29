"""
predict.py: Predicts spam/not spam from a CSV file of numerical features (Spambase dataset format).
Each row must contain exactly 57 features (no header).
Outputs prediction and spam probability.
"""

import joblib
import pandas as pd
import sys
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_model(model_path: str):
    return joblib.load(model_path)

def predict_csv(df: pd.DataFrame, model):
    predictions = model.predict(df)
    
    # Get probabilities if available
    try:
        proba = model.predict_proba(df)[:, 1]  # Probability of class 1 (spam)
    except AttributeError:
        proba = [None] * len(predictions)

    labels = ["spam" if pred == 1 else "not spam" for pred in predictions]
    return labels, proba

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_csv_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    try:
        logger.info("Reading input CSV...")
        df = pd.read_csv(input_path)

        if df.shape[1] != 57:
            raise ValueError(f"Each row must have 57 features, but got {df.shape[1]}.")

        model_path = "artifacts/xgboost/model.joblib"
        logger.info("Loading model...")
        model = load_model(model_path)

        logger.info("Running predictions...")
        predictions, probabilities = predict_csv(df, model)

        # Print results
        print("\nResults:")
        for i, (label, prob) in enumerate(zip(predictions, probabilities), 1):
            if prob is not None:
                print(f"Row {i}: {label.upper()} (Spam Probability: {prob:.4f})")
            else:
                print(f"Row {i}: {label.upper()}")

        # Optional: Write to CSV
        output_df = df.copy()
        output_df["Prediction"] = predictions
        output_df["Spam_Probability"] = probabilities
        output_file = os.path.splitext(input_path)[0] + "_predicted.csv"
        output_df.to_csv(output_file, index=False, header=False)
        print(f"\nPredictions saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
