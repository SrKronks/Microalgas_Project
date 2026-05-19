from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from scripts.evaluation.metrics import regression_metrics
from scripts.preprocessing.data_loader import detect_schema
from scripts.sensitivity.decline_sensitivity import _parse_probabilities, _summarize, _winners
from scripts.synthetic.growth_cycles import SyntheticGrowthCycleGenerator
from scripts.utils.config import ProjectConfig, DEFAULT_CONFIG


def test_schema_detection() -> None:
    df = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-01-01", periods=4),
            "BIM": ["BIM-1"] * 4,
            "OD": [0.1, 0.2, 0.3, 0.4],
            "pH": [7.0, 7.1, 7.2, 7.3],
        }
    )
    config = ProjectConfig(raw=DEFAULT_CONFIG, root=Path("."))
    schema = detect_schema(df, config, __import__("logging").getLogger("test"))
    assert schema.date_col == "Fecha"
    assert schema.group_col == "BIM"
    assert "OD" in schema.target_columns


def test_metrics_are_finite_for_good_forecast() -> None:
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["R2"] == 1


def test_synthetic_growth_cycles_are_complete() -> None:
    raw = deepcopy(DEFAULT_CONFIG)
    raw["synthetic_training"]["n_cycles"] = 6
    raw["synthetic_training"]["min_cycle_points"] = 9
    raw["synthetic_training"]["max_cycle_points"] = 12
    config = ProjectConfig(raw=raw, root=Path("."))

    cycles = SyntheticGrowthCycleGenerator(config).generate(
        observed_values=pd.Series([0.2, 0.35, 0.8, 0.95, 0.7]),
        observed_lengths=[9, 11, 12],
        target="OD",
    )

    assert cycles["cycle_id"].nunique() == 6
    assert cycles.groupby("cycle_id").size().between(9, 12).all()
    assert {"lag", "exponential"}.issubset(set(cycles["phase"]))
    assert (cycles["value"] > 0).all()


def test_decline_sensitivity_summary_selects_best_probability() -> None:
    assert _parse_probabilities("0.4,0.7") == [0.4, 0.7]
    metrics = pd.DataFrame(
        [
            {"decline_probability": 0.4, "target": "OD", "category": "machine_learning", "model": "Ridge", "status": "ok", "RMSE": 0.3, "MAE": 0.2, "SMAPE": 10, "R2": 0.4, "fit_seconds": 0.1},
            {"decline_probability": 0.7, "target": "OD", "category": "machine_learning", "model": "Ridge", "status": "ok", "RMSE": 0.2, "MAE": 0.1, "SMAPE": 8, "R2": 0.6, "fit_seconds": 0.1},
        ]
    )
    summary = _summarize(metrics)
    winners = _winners(summary)
    assert float(winners.iloc[0]["decline_probability"]) == 0.7
