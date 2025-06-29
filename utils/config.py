"""
utils/config.py: Parses configuration YAML files for modular pipeline control.
"""

import yaml
from typing import Any


class Config:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
        except KeyError:
            return default
        return value

    def get_all(self) -> dict:
        return self.config
