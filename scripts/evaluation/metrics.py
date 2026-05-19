from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y_true: Any, y_pred: Any, n_params: int = 1) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    n = len(true)
    if n == 0:
        return {metric: math.nan for metric in ["RMSE", "MAE", "MAPE", "SMAPE", "R2", "Adjusted_R2", "AIC", "BIC", "LogLikelihood"]}

    residuals = true - pred
    sse = float(np.sum(residuals**2))
    mse = sse / n
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(residuals)))
    non_zero = np.abs(true) > 1e-12
    mape = float(np.mean(np.abs((true[non_zero] - pred[non_zero]) / true[non_zero])) * 100) if non_zero.any() else math.nan
    smape = float(np.mean(2 * np.abs(pred - true) / np.maximum(np.abs(true) + np.abs(pred), 1e-12)) * 100)

    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1 - sse / ss_tot if ss_tot > 0 else math.nan
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_params - 1, 1) if not math.isnan(r2) and n > 2 else math.nan
    sigma2 = max(mse, 1e-12)
    loglik = float(-0.5 * n * (math.log(2 * math.pi * sigma2) + 1))
    aic = float(2 * n_params - 2 * loglik)
    bic = float(math.log(max(n, 1)) * n_params - 2 * loglik)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "SMAPE": smape,
        "R2": float(r2),
        "Adjusted_R2": float(adjusted_r2),
        "AIC": aic,
        "BIC": bic,
        "LogLikelihood": loglik,
    }


def rank_models(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    successful = metrics[metrics["status"].eq("ok")].copy()
    if successful.empty:
        return successful
    ascending_cols = ["RMSE", "MAE", "SMAPE", "AIC", "BIC"]
    for col in ascending_cols:
        if col in successful:
            successful[f"{col}_rank"] = successful.groupby(["BIM", "target"])[col].rank(method="min", ascending=True)
    if "R2" in successful:
        successful["R2_rank"] = successful.groupby(["BIM", "target"])["R2"].rank(method="min", ascending=False)
    rank_cols = [col for col in successful.columns if col.endswith("_rank")]
    successful["mean_rank"] = successful[rank_cols].mean(axis=1)
    return successful.sort_values(["BIM", "target", "mean_rank", "RMSE"], na_position="last")
