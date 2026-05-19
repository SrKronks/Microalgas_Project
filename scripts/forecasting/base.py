from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ModelSkipped(RuntimeError):
    """Raised when a model cannot be run for the current data or dependencies."""


@dataclass
class ForecastRecord:
    BIM: str
    target: str
    category: str
    model: str
    status: str
    fit_seconds: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecastModel:
    name = "base"
    category = "base"
    min_points = 5
    n_params = 1

    def __init__(self) -> None:
        self.fitted_model: Any = None
        self.metadata: dict[str, Any] = {}
        self.hyperparameters: dict[str, Any] = {}

    def fit_predict(
        self,
        train: pd.Series,
        horizon: int,
        frame_train: pd.DataFrame | None = None,
        frame_test: pd.DataFrame | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.fitted_model if self.fitted_model is not None else self, handle)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame()

    def model_card(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "model": self.name,
            "min_points": self.min_points,
            "n_params": self.n_params,
            "hyperparameters": self.hyperparameters,
        }


def require_points(series: pd.Series, min_points: int, model_name: str) -> None:
    if pd.to_numeric(series, errors="coerce").dropna().shape[0] < min_points:
        raise ModelSkipped(f"{model_name} requires at least {min_points} observations")


def clean_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean = pd.Series(clean.to_numpy(dtype=float), index=range(len(clean)))
    return clean.astype(float)


class SkippedModel(ForecastModel):
    def __init__(self, name: str, category: str, reason: str) -> None:
        super().__init__()
        self.name = name
        self.category = category
        self.reason = reason

    def fit_predict(
        self,
        train: pd.Series,
        horizon: int,
        frame_train: pd.DataFrame | None = None,
        frame_test: pd.DataFrame | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        raise ModelSkipped(self.reason)
