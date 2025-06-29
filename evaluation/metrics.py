"""
evaluation/metrics.py: Centralizes evaluation metric functions for all models.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute evaluation metrics for binary classification."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except:
            metrics["roc_auc"] = None

    return metrics
