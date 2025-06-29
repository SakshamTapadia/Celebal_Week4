"""
evaluation/evaluator.py: Compare multiple models and perform statistical significance tests.
"""

from typing import List, Dict, Tuple
from evaluation.metrics import compute_metrics
from scipy.stats import ttest_rel, wilcoxon
import numpy as np
import pandas as pd


class ModelEvaluator:
    def __init__(self):
        self.results: List[Dict] = []

    def evaluate_model(self, name: str, y_true, y_pred, y_proba=None) -> None:
        metrics = compute_metrics(y_true, y_pred, y_proba)
        metrics["model"] = name
        self.results.append(metrics)

    def summarize_results(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results)
        return df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].sort_values(by="f1_score", ascending=False)

    def compare_models(self, preds_a: np.ndarray, preds_b: np.ndarray, method: str = "ttest") -> Tuple[str, float]:
        """Compares two prediction arrays using statistical tests."""
        if method == "ttest":
            stat, p_value = ttest_rel(preds_a, preds_b)
        elif method == "wilcoxon":
            stat, p_value = wilcoxon(preds_a, preds_b)
        else:
            raise ValueError("Invalid comparison method. Use 'ttest' or 'wilcoxon'.")

        return method, p_value

