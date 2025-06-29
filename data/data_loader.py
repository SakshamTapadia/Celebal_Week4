import pandas as pd
from typing import Tuple
import os
import logging

class DataLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.path, encoding='latin-1')
            self.logger.info("Data loaded successfully from %s", self.path)
            return data
        except FileNotFoundError as e:
            self.logger.error("File not found: %s", e)
            raise
        except Exception as e:
            self.logger.error("Unexpected error: %s", e)
            raise
