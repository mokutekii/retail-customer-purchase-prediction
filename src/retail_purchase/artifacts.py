"""Portable model artifacts used by training and inference processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionBundle:
    """A calibrated estimator plus the feature contract and chosen operating point."""

    model: Any
    threshold: float
    feature_columns: tuple[str, ...]

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Prediction input must be a pandas DataFrame.")
        expected = set(self.feature_columns)
        received = set(features.columns)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError(f"Feature schema mismatch ({'; '.join(details)}).")
        return features.loc[:, list(self.feature_columns)]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare(features))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= self.threshold).astype("int8")
