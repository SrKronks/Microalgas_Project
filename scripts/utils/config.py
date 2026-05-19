from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "Microalgas_Project",
        "description": "Advanced time-series analytics and forecasting for microalgae cultures.",
    },
    "data": {
        "raw_file": "data/raw/historial-monitoreos-2026-05-14.xlsx",
        "sheet_name": None,
        "date_column": None,
        "group_column": None,
        "target_columns": ["OD"],
        "preferred_analysis_columns": ["OD", "pH", "EC", "Temperatura", "Temp", "mu", "growth"],
        "drop_normalized_duplicates": False,
    },
    "execution": {
        "random_state": 42,
        "n_jobs": -1,
        "forecast_horizon": 5,
        "test_fraction": 0.25,
        "min_train_points": 5,
        "max_bims": None,
        "continue_on_error": True,
        "save_models": True,
        "make_png": True,
        "make_svg": True,
        "make_pdf": True,
        "heavy_deep_learning": False,
    },
    "validation": {
        "strategy": "synthetic_full_cycle",
        "imputation_strategy": "time_interpolate",
        "outlier_method": "iqr",
        "outlier_iqr_multiplier": 1.5,
        "rolling_windows": [3, 5],
        "seasonal_periods": 3,
    },
    "synthetic_training": {
        "enabled": True,
        "n_cycles": 2000,
        "min_cycle_points": 8,
        "max_cycle_points": 32,
        "noise_fraction": 0.035,
        "decline_probability": 0.70,
        "seasonality_probability": 0.45,
        "save_dataset": True,
    },
    "models": {
        "enabled_groups": {
            "classical": True,
            "statistical": True,
            "biological": True,
            "differential_equations": True,
            "probabilistic": True,
            "machine_learning": True,
            "deep_learning": True,
            "hybrid": True,
        },
        "optional_models": {
            "xgboost": True,
            "lightgbm": True,
            "catboost": True,
            "prophet": True,
            "darts": False,
            "pymc": False,
        },
    },
    "model_hyperparameters": {
        "classical": {
            "Moving_Average": {"window": 3},
            "Weighted_Moving_Average": {"window": 3},
            "Croston": {"alpha": 0.1},
        },
        "statistical": {
            "VAR": {"maxlags": 2},
            "VARMAX": {"varmax_order": [1, 0], "maxiter": 100},
            "ARCH": {"p": 1, "q": 0},
            "GARCH": {"p": 1, "q": 1},
            "State_Space_Model": {"level": "local linear trend"},
        },
        "biological": {"Curve_Fit": {"maxfev": 20_000}},
        "differential_equations": {"Curve_Fit": {"maxfev": 20_000}},
        "probabilistic": {
            "Hidden_Markov_Models": {"n_components": 3, "n_iter": 200},
            "Markov_Chains": {"n_states": 3},
            "Gaussian_Processes": {"normalize_y": True},
            "Particle_Filters": {"particles": 500},
            "Monte_Carlo": {"simulations": 500},
            "Sequential_Monte_Carlo": {"particles": 500},
        },
        "machine_learning": {
            "Linear_Regression": {"lags": 3},
            "Polynomial_Regression": {"lags": 3, "degree": 2},
            "Ridge": {"lags": 3, "alpha": 1.0},
            "Lasso": {"lags": 3, "alpha": 0.001, "max_iter": 10_000},
            "Elastic_Net": {"lags": 3, "alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10_000},
            "Decision_Trees": {"lags": 3, "max_depth": 4},
            "Random_Forest": {"lags": 3, "n_estimators": 200},
            "Extra_Trees": {"lags": 3, "n_estimators": 200},
            "Gradient_Boosting": {"lags": 3},
            "AdaBoost": {"lags": 3},
            "SVR": {"lags": 3, "C": 10.0, "epsilon": 0.01},
            "KNN": {"lags": 3, "n_neighbors": 3},
            "Gaussian_Process_Regression": {"lags": 3, "normalize_y": True},
            "PLS": {"lags": 3, "n_components": 1},
            "PCR": {"lags": 3, "n_components": 1},
            "XGBoost": {"lags": 3, "n_estimators": 200, "objective": "reg:squarederror"},
            "LightGBM": {"lags": 3, "n_estimators": 200, "verbose": -1},
            "CatBoost": {"lags": 3, "iterations": 200, "verbose": False},
        },
        "deep_learning": {
            "MLP": {"lags": 4, "hidden_layer_sizes": [32, 16], "max_iter": 1000},
            "RNN": {"lags": 4, "epochs": 40, "units": 16},
            "LSTM": {"lags": 4, "epochs": 40, "units": 16},
            "Bidirectional_LSTM": {"lags": 4, "epochs": 40, "units": 16},
            "GRU": {"lags": 4, "epochs": 40, "units": 16},
            "CNN_temporal": {"lags": 4, "epochs": 40, "units": 16},
        },
    },
    "reporting": {
        "language": "es",
        "title": "Plataforma de analisis y pronostico de microalgas",
        "author": "Sebastian Pizarro",
        "export_excel": True,
        "export_html": True,
        "export_pdf": True,
    },
}


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    root: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted_key: str) -> Path:
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Missing path config: {dotted_key}")
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_if_available(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def load_config(root: Path, config_path: Path | None = None) -> ProjectConfig:
    path = config_path or root / "configs" / "config.yaml"
    loaded = load_yaml_if_available(path)
    return ProjectConfig(raw=deep_merge(DEFAULT_CONFIG, loaded), root=root)
