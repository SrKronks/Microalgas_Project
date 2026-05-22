from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from scripts.classification.runner import ClassificationRunner
from scripts.evaluation.metrics import classification_metrics
from scripts.evaluation.metrics import regression_metrics
from scripts.forecasting.runner import ForecastRunner
from scripts.preprocessing.data_loader import detect_schema
from scripts.sensitivity.decline_sensitivity import _parse_probabilities, _summarize, _winners
from scripts.synthetic.growth_cycles import SyntheticGrowthCycleGenerator
from scripts.synthetic.quality import evaluate_synthetic_cycles
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


def test_schema_detection_finds_classification_labels() -> None:
    df = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-01-01", periods=6),
            "BIM": ["BIM-1"] * 6,
            "OD": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Ritmo": ["RAPIDO", "RAPIDO", "MODERADO", "MODERADO", "DECLIVE", "DECLIVE"],
            "Estado_Cultivo": ["CRECIMIENTO", "CRECIMIENTO", "PRODUCTO", "PRODUCTO", "DESCARTE", "DESCARTE"],
        }
    )
    config = ProjectConfig(raw=DEFAULT_CONFIG, root=Path("."))
    schema = detect_schema(df, config, __import__("logging").getLogger("test"))
    assert schema.label_columns == ["Ritmo", "Estado_Cultivo"]


def test_metrics_are_finite_for_good_forecast() -> None:
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["R2"] == 1


def test_classification_metrics_are_finite_for_good_classifier() -> None:
    metrics = classification_metrics(["A", "B", "A"], ["A", "B", "A"], labels=["A", "B"])
    assert metrics["Accuracy"] == 1
    assert metrics["Macro_F1"] == 1


def test_classification_runner_outputs_rankings(tmp_path: Path) -> None:
    n = 36
    df = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-01-01", periods=n),
            "BIM": [f"BIM-{idx % 3}" for idx in range(n)],
            "OD": [0.2 + idx * 0.02 for idx in range(n)],
            "pH": [7.0 + (idx % 6) * 0.05 for idx in range(n)],
            "Ritmo": ["RAPIDO" if idx % 3 == 0 else "MODERADO" if idx % 3 == 1 else "DECLIVE" for idx in range(n)],
            "Estado_Cultivo": ["CRECIMIENTO" if idx % 2 == 0 else "PRODUCTO" for idx in range(n)],
        }
    )
    raw = deepcopy(DEFAULT_CONFIG)
    raw["classification"]["min_samples"] = 12
    raw["classification"]["save_models"] = False
    config = ProjectConfig(raw=raw, root=tmp_path)
    schema = detect_schema(df, config, __import__("logging").getLogger("test"))
    dirs = {
        "metrics": tmp_path / "outputs" / "metrics",
        "rankings": tmp_path / "outputs" / "rankings",
        "diagnostics": tmp_path / "outputs" / "diagnostics",
        "models": tmp_path / "outputs" / "models",
    }
    metrics, rankings, predictions, confusion = ClassificationRunner(
        config,
        schema,
        dirs,
        __import__("logging").getLogger("test"),
    ).run(df)

    assert not metrics.empty
    assert metrics["status"].eq("ok").any()
    assert not rankings.empty
    assert not predictions.empty
    assert not confusion.empty


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
    assert {"BIM", "cycle_age_days", "specific_growth_rate", "synthetic_ritmo", "synthetic_estado_cultivo", "optimality_score", "pH", "EC", "Temperatura (°C)"}.issubset(cycles.columns)
    assert (cycles["value"] > 0).all()


def test_synthetic_growth_cycles_repair_tight_carrying_bounds() -> None:
    raw = deepcopy(DEFAULT_CONFIG)
    raw["synthetic_training"]["n_cycles"] = 8
    raw["synthetic_training"]["min_cycle_points"] = 6
    raw["synthetic_training"]["max_cycle_points"] = 8
    raw["synthetic_training"]["baseline_low"] = 0.7
    raw["synthetic_training"]["baseline_high"] = 1.0
    raw["synthetic_training"]["carrying_low"] = 0.8
    raw["synthetic_training"]["carrying_high"] = 0.9
    config = ProjectConfig(raw=raw, root=Path("."))

    cycles = SyntheticGrowthCycleGenerator(config).generate(
        observed_values=pd.Series([0.62, 0.64, 0.66, 0.65, 0.63]),
        observed_lengths=[6],
        target="OD",
    )

    params = cycles.groupby("cycle_id")[["baseline", "carrying_capacity"]].first()
    assert cycles["cycle_id"].nunique() == 8
    assert (params["carrying_capacity"] >= params["baseline"] * 1.2).all()


def test_synthetic_quality_reports_distribution_guardrails() -> None:
    raw = deepcopy(DEFAULT_CONFIG)
    raw["synthetic_training"]["n_cycles"] = 4
    raw["synthetic_training"]["min_cycle_points"] = 6
    raw["synthetic_training"]["max_cycle_points"] = 8
    config = ProjectConfig(raw=raw, root=Path("."))
    real = pd.Series([0.25, 0.31, 0.44, 0.56, 0.62, 0.58])
    cycles = SyntheticGrowthCycleGenerator(config).generate(real, [len(real)], target="OD")

    result = evaluate_synthetic_cycles(real, cycles, target="OD", group="BIM-TEST")

    assert not result.summary.empty
    assert result.summary.iloc[0]["status"] == "ok"
    assert "ks_value" in result.summary.columns
    assert not result.by_phase.empty


def test_synthetic_forecast_uses_real_train_holdout_protocol(tmp_path: Path) -> None:
    n = 14
    df = pd.DataFrame(
        {
            "Fecha": pd.date_range("2026-01-01", periods=n),
            "BIM": ["BIM-1"] * n,
            "OD": [0.2, 0.24, 0.31, 0.39, 0.51, 0.62, 0.70, 0.74, 0.73, 0.69, 0.64, 0.58, 0.53, 0.49],
        }
    )
    raw = deepcopy(DEFAULT_CONFIG)
    raw["execution"]["forecast_horizon"] = 3
    raw["execution"]["min_train_points"] = 6
    raw["execution"]["save_models"] = False
    raw["execution"]["make_png"] = False
    raw["execution"]["make_svg"] = False
    raw["synthetic_training"]["enabled"] = True
    raw["synthetic_training"]["n_cycles"] = 8
    raw["models"]["enabled_groups"] = {key: False for key in raw["models"]["enabled_groups"]}
    raw["models"]["enabled_groups"]["machine_learning"] = True
    raw["model_hyperparameters"]["machine_learning"]["Linear_Regression"]["lags"] = 2
    config = ProjectConfig(raw=raw, root=tmp_path)
    schema = detect_schema(df, config, __import__("logging").getLogger("test"))
    dirs = {
        "metrics": tmp_path / "outputs" / "metrics",
        "forecasts": tmp_path / "outputs" / "forecasts",
        "processed": tmp_path / "outputs" / "processed",
        "diagnostics": tmp_path / "outputs" / "diagnostics",
        "models": tmp_path / "outputs" / "models",
        "figures": tmp_path / "outputs" / "figures",
        "shap": tmp_path / "outputs" / "shap",
    }

    metrics, forecasts = ForecastRunner(config, schema, dirs, __import__("logging").getLogger("test")).run(df)

    assert not metrics.empty
    assert set(metrics["validation_mode"].dropna()) == {"TSTR_temporal_holdout"}
    assert int(metrics.iloc[0]["real_train_points"]) == n - 3
    assert int(metrics.iloc[0]["real_test_points"]) == 3
    if metrics["status"].eq("ok").any():
        assert not forecasts.empty
        metadata = metrics.loc[metrics["status"].eq("ok"), "metadata"].dropna().iloc[0]
        assert "phase_conditioned_one_step_ahead" in metadata
    assert (dirs["diagnostics"] / "synthetic_quality").exists()


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
