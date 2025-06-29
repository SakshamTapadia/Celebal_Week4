"""
models/ensemble_models.py: Implements ensemble techniques such as Voting and Stacking classifiers.
"""

import joblib
import numpy as np
from typing import Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.base_model import BaseModel


class VotingEnsembleModel(BaseModel):
    def __init__(self, estimators: list[tuple], voting: str = "hard"):
        self.model = VotingClassifier(estimators=estimators, voting=voting)

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
            "f1_score": f1_score(y, y_pred)
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


class StackingEnsembleModel(BaseModel):
    def __init__(self, estimators: list[tuple], final_estimator=None, cv: int = 5):
        if final_estimator is None:
            final_estimator = LogisticRegression()
        self.model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=cv)

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
            "f1_score": f1_score(y, y_pred)
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
