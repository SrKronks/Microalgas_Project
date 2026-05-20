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


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    labels: list[str] | None = None,
) -> dict[str, float]:
    try:
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            balanced_accuracy_score,
            cohen_kappa_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
        )
    except Exception:
        true = np.asarray(y_true, dtype=object)
        pred = np.asarray(y_pred, dtype=object)
        mask = pd.notna(true) & pd.notna(pred)
        true = true[mask]
        pred = pred[mask]
        metric_labels = labels or sorted({str(value) for value in true})
        accuracy = float(np.mean(true == pred)) if len(true) else math.nan
        precisions = []
        recalls = []
        f1_scores = []
        weighted_f1 = 0.0
        total = max(len(true), 1)
        expected_accuracy = 0.0
        for label in metric_labels:
            true_mask = true == label
            pred_mask = pred == label
            tp = float(np.sum(true_mask & pred_mask))
            fp = float(np.sum(~true_mask & pred_mask))
            fn = float(np.sum(true_mask & ~pred_mask))
            support = float(np.sum(true_mask))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            weighted_f1 += f1 * support / total
            expected_accuracy += (support / total) * (float(np.sum(pred_mask)) / total)
        observed_accuracy = accuracy if not math.isnan(accuracy) else 0.0
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy) if expected_accuracy < 1 else math.nan
        return {
            "Accuracy": accuracy,
            "Balanced_Accuracy": float(np.mean(recalls)) if recalls else math.nan,
            "Macro_Precision": float(np.mean(precisions)) if precisions else math.nan,
            "Macro_Recall": float(np.mean(recalls)) if recalls else math.nan,
            "Macro_F1": float(np.mean(f1_scores)) if f1_scores else math.nan,
            "Weighted_F1": float(weighted_f1) if len(true) else math.nan,
            "Cohen_Kappa": float(kappa),
            "LogLoss": math.nan,
        }

    true = pd.Series(y_true, dtype="object").astype(str)
    pred = pd.Series(y_pred, dtype="object").astype(str)
    metric_labels = labels or sorted(true.dropna().unique().tolist())
    result = {
        "Accuracy": float(accuracy_score(true, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(true, pred)),
        "Macro_Precision": float(precision_score(true, pred, labels=metric_labels, average="macro", zero_division=0)),
        "Macro_Recall": float(recall_score(true, pred, labels=metric_labels, average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(true, pred, labels=metric_labels, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(true, pred, labels=metric_labels, average="weighted", zero_division=0)),
        "Cohen_Kappa": float(cohen_kappa_score(true, pred, labels=metric_labels)),
        "LogLoss": math.nan,
    }
    if y_proba is not None:
        try:
            result["LogLoss"] = float(log_loss(true, y_proba, labels=metric_labels))
        except Exception:
            result["LogLoss"] = math.nan
    return result


def rank_classifiers(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    successful = metrics[metrics["status"].eq("ok")].copy()
    if successful.empty:
        return successful
    for col in ["Macro_F1", "Balanced_Accuracy", "Weighted_F1", "Accuracy"]:
        if col in successful:
            successful[f"{col}_rank"] = successful.groupby("label_target")[col].rank(method="min", ascending=False)
    if "LogLoss" in successful:
        successful["LogLoss_rank"] = successful.groupby("label_target")["LogLoss"].rank(method="min", ascending=True)
    rank_cols = [col for col in successful.columns if col.endswith("_rank")]
    successful["mean_rank"] = successful[rank_cols].mean(axis=1)
    return successful.sort_values(["label_target", "mean_rank", "Macro_F1"], ascending=[True, True, False], na_position="last")
