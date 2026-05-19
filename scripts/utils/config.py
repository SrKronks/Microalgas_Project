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
    "reporting": {
        "language": "es",
        "title": "Plataforma de analisis y pronostico de microalgas",
        "author": "Codex",
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
