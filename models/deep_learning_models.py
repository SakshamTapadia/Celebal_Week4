"""
models/deep_learning_models.py: Implements deep learning models like LSTM, CNN, and DistilBERT for text classification.
"""

import numpy as np
from typing import Any
from models.base_model import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# LSTM / CNN
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# DistilBERT
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf


class LSTMModel(BaseModel):
    def __init__(self, vocab_size=10000, embedding_dim=64, input_length=100):
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=input_length),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X: Any, y: Any) -> None:
        self.model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=2)])

    def predict(self, X: Any) -> np.ndarray:
        return (self.model.predict(X) > 0.5).astype("int32").flatten()

    def evaluate(self, X: Any, y: Any) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = load_model(path)


class CNNModel(BaseModel):
    def __init__(self, vocab_size=10000, embedding_dim=64, input_length=100):
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=input_length),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X: Any, y: Any) -> None:
        self.model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=2)])

    def predict(self, X: Any) -> np.ndarray:
        return (self.model.predict(X) > 0.5).astype("int32").flatten()

    def evaluate(self, X: Any, y: Any) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = load_model(path)


class DistilBERTModel(BaseModel):
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    def tokenize(self, texts: list[str]):
        return self.tokenizer(texts, truncation=True, padding=True, return_tensors='tf')

    def train(self, X: list[str], y: Any) -> None:
        X_tokenized = self.tokenize(X)
        self.model.fit(X_tokenized.data, y, epochs=3, batch_size=16)

    def predict(self, X: list[str]) -> np.ndarray:
        X_tokenized = self.tokenize(X)
        predictions = self.model(X_tokenized.data).logits
        return tf.argmax(predictions, axis=1).numpy()

    def evaluate(self, X: list[str], y: Any) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(path)
