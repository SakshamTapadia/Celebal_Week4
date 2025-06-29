"""
feature_extractor.py: Feature transformation using TF-IDF and n-grams.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
import pandas as pd

class FeatureExtractor:
    def __init__(self, ngram_range=(1, 2), max_features=10000):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

    def fit_transform(self, X_train: pd.Series) -> Tuple:
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        return X_train_tfidf

    def transform(self, X: pd.Series):
        return self.vectorizer.transform(X)
