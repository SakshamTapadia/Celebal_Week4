"""
data/preprocessor.py: Handles text preprocessing such as lowercasing, stopword removal, punctuation removal, and stemming.
"""

import re
import string
from typing import List
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class TextPreprocessor:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        tokens = text.split()
        cleaned = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return " ".join(cleaned)

    def preprocess(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        df[text_column] = df[text_column].astype(str).apply(self.clean_text)
        return df

    # Backward compatibility alias
    def preprocess_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        return self.preprocess(df, column)