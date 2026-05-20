from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticQualityResult:
    summary: pd.DataFrame
    by_phase: pd.DataFrame


def evaluate_synthetic_cycles(
    real_values: pd.Series,
    synthetic_cycles: pd.DataFrame,
    target: str,
    group: str | None = None,
) -> SyntheticQualityResult:
    """Compare synthetic cycle values against the real training segment.

    The diagnostics are intentionally dependency-light so they always run in
    the main pipeline. They are not a proof that synthetic data is perfect; they
    are guardrails against obvious distribution, slope, and range failures.
    """

    real = pd.to_numeric(real_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    syn = (
        pd.to_numeric(synthetic_cycles.get("value", pd.Series(dtype=float)), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    rows: list[dict[str, Any]] = []
    group_value = group if group is not None else "ALL"
    if real.empty or syn.empty:
        rows.append(
            {
                "BIM": group_value,
                "target": target,
                "status": "skipped",
                "reason": "empty real or synthetic values",
            }
        )
        return SyntheticQualityResult(pd.DataFrame(rows), pd.DataFrame())

    real_arr = real.to_numpy(dtype=float)
    syn_arr = syn.to_numpy(dtype=float)
    real_p = _percentiles(real_arr)
    syn_p = _percentiles(syn_arr)
    lower, upper = real_p["p01"], real_p["p99"]
    tolerance = max(1e-9, 0.10 * (upper - lower))
    out_of_range = np.mean((syn_arr < lower - tolerance) | (syn_arr > upper + tolerance))

    real_slopes = np.diff(real_arr)
    syn_slopes = _synthetic_slopes(synthetic_cycles)
    slope_ks = _ks_statistic(real_slopes, syn_slopes) if len(real_slopes) and len(syn_slopes) else np.nan

    rows.append(
        {
            "BIM": group_value,
            "target": target,
            "status": "ok",
            "real_points": int(len(real_arr)),
            "synthetic_points": int(len(syn_arr)),
            "synthetic_cycles": int(synthetic_cycles["cycle_id"].nunique()) if "cycle_id" in synthetic_cycles else np.nan,
            "ks_value": float(_ks_statistic(real_arr, syn_arr)),
            "ks_slope": float(slope_ks) if np.isfinite(slope_ks) else np.nan,
            "mean_real": float(np.mean(real_arr)),
            "mean_synthetic": float(np.mean(syn_arr)),
            "std_real": float(np.std(real_arr, ddof=1)) if len(real_arr) > 1 else 0.0,
            "std_synthetic": float(np.std(syn_arr, ddof=1)) if len(syn_arr) > 1 else 0.0,
            "p05_real": real_p["p05"],
            "p05_synthetic": syn_p["p05"],
            "p50_real": real_p["p50"],
            "p50_synthetic": syn_p["p50"],
            "p95_real": real_p["p95"],
            "p95_synthetic": syn_p["p95"],
            "synthetic_out_of_real_range_pct": float(out_of_range * 100.0),
            "nearest_real_distance_mean": float(_nearest_distance_mean(real_arr, syn_arr)),
        }
    )

    by_phase = pd.DataFrame()
    if {"phase", "value"}.issubset(synthetic_cycles.columns):
        by_phase = (
            synthetic_cycles.assign(value=pd.to_numeric(synthetic_cycles["value"], errors="coerce"))
            .dropna(subset=["value"])
            .groupby("phase", dropna=False)
            .agg(
                synthetic_points=("value", "size"),
                synthetic_mean=("value", "mean"),
                synthetic_std=("value", "std"),
                synthetic_min=("value", "min"),
                synthetic_max=("value", "max"),
            )
            .reset_index()
        )
        by_phase.insert(0, "target", target)
        by_phase.insert(0, "BIM", group_value)

    return SyntheticQualityResult(pd.DataFrame(rows), by_phase)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    p01, p05, p50, p95, p99 = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {"p01": float(p01), "p05": float(p05), "p50": float(p50), "p95": float(p95), "p99": float(p99)}


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _synthetic_slopes(cycles: pd.DataFrame) -> np.ndarray:
    if not {"cycle_id", "step", "value"}.issubset(cycles.columns):
        return np.asarray([], dtype=float)
    slopes: list[float] = []
    for _, frame in cycles.groupby("cycle_id", sort=False):
        values = pd.to_numeric(frame.sort_values("step")["value"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) > 1:
            slopes.extend(np.diff(values).tolist())
    return np.asarray(slopes, dtype=float)


def _nearest_distance_mean(real: np.ndarray, syn: np.ndarray) -> float:
    if len(real) == 0 or len(syn) == 0:
        return float("nan")
    sample = syn
    if len(sample) > 5000:
        rng = np.random.default_rng(42)
        sample = rng.choice(sample, size=5000, replace=False)
    distances = np.min(np.abs(sample[:, None] - real[None, :]), axis=1)
    return float(np.mean(distances))
