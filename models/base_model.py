"""
models/base_model.py: Abstract base class defining the interface for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd

class BaseModel(ABC):
    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Predict the target for given input features."""
        pass

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> dict:
        """Evaluate model and return performance metrics."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass
