"""
data/data_splitter.py: Stratified data splitting for training, validation, and testing.
"""

from sklearn.model_selection import train_test_split
from typing import Tuple, Any


def train_test_split_stratified(
    X: Any, y: Any, test_size: float = 0.2, seed: int = 42
) -> Tuple:
    """Performs stratified train-test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def train_val_test_split_stratified(
    X: Any, y: Any, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
) -> Tuple:
    """Performs stratified train-val-test split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split_stratified(
        X, y, test_size=test_size, seed=seed
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, stratify=y_train_val, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
