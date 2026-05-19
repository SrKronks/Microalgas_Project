from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DependencyStatus:
    python_version: str
    platform: str
    pandas: bool
    numpy: bool
    scipy: bool
    statsmodels: bool
    sklearn: bool
    matplotlib: bool
    seaborn: bool
    plotly: bool
    shap: bool
    xgboost: bool
    lightgbm: bool
    catboost: bool
    torch: bool
    tensorflow: bool
    cuda_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_cuda() -> bool:
    if has_module("torch"):
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False
    if has_module("tensorflow"):
        try:
            import tensorflow as tf  # type: ignore

            return bool(tf.config.list_physical_devices("GPU"))
        except Exception:
            return False
    return False


def dependency_status() -> DependencyStatus:
    return DependencyStatus(
        python_version=platform.python_version(),
        platform=platform.platform(),
        pandas=has_module("pandas"),
        numpy=has_module("numpy"),
        scipy=has_module("scipy"),
        statsmodels=has_module("statsmodels"),
        sklearn=has_module("sklearn"),
        matplotlib=has_module("matplotlib"),
        seaborn=has_module("seaborn"),
        plotly=has_module("plotly"),
        shap=has_module("shap"),
        xgboost=has_module("xgboost"),
        lightgbm=has_module("lightgbm"),
        catboost=has_module("catboost"),
        torch=has_module("torch"),
        tensorflow=has_module("tensorflow"),
        cuda_available=detect_cuda(),
    )
