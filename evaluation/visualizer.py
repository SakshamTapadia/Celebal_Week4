"""
evaluation/visualizer.py: Visualizes model performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import numpy as np
from typing import Optional


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(y_true, y_proba, title: str = "ROC Curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall(y_true, y_proba, title: str = "Precision-Recall Curve") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision, marker=".", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_feature_importance(importances: np.ndarray, feature_names: list[str], top_n: int = 20, title: str = "Feature Importance") -> None:
    sorted_idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
