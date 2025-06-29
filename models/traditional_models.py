"""
models/traditional_models.py: Implements traditional ML models using scikit-learn and XGBoost.
"""

import joblib
import numpy as np
from typing import Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def train(self, X: Any, y: Any) -> None:
        self.model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: Any, y: Any) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X: Any, y: Any) -> None:
        self.model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: Any, y: Any) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def train(self, X: Any, y: Any) -> None:
        self.model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: Any, y: Any) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)

    def train(self, X: Any, y: Any) -> None:
        self.model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: Any, y: Any) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
