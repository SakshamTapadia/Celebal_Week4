"""
utils/helpers.py: Common reusable helper functions for tokenization, reproducibility, etc.
"""

import random
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, List


def set_seed(seed: int = 42) -> None:
    """Ensure reproducible results."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def tokenize_and_pad(
    texts: List[str], 
    num_words: int = 10000, 
    max_length: int = 100
) -> Tuple[np.ndarray, Tokenizer]:
    """Tokenizes and pads text sequences for CNN/LSTM models."""
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded, tokenizer


def print_separator(title: str = "") -> None:
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60 + "\n")
