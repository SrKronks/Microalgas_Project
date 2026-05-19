from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.evaluation.metrics import rank_models
from scripts.forecasting.runner import ForecastRunner
from scripts.sensitivity.decline_sensitivity import _apply_cli_overrides, _parse_model_groups, _prepare_features
from scripts.utils.config import ProjectConfig, load_config
from scripts.utils.logging_utils import setup_logging
from scripts.utils.paths import ensure_directories, output_dirs, project_root, safe_name


DEFAULT_GRID = (
    "machine_learning.Ridge.alpha=0.1,1.0,10.0;"
    "machine_learning.Lasso.alpha=0.0001,0.001,0.01;"
    "machine_learning.Elastic_Net.l1_ratio=0.2,0.5,0.8;"
    "machine_learning.Random_Forest.n_estimators=100,200,400;"
    "machine_learning.Random_Forest.lags=2,3,5;"
    "machine_learning.SVR.C=1.0,10.0,50.0;"
    "probabilistic.Monte_Carlo.simulations=200,500,1000;"
    "synthetic_training.decline_probability=0.4,0.7,0.9"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-at-a-time hyperparameter sensitivity scenarios.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input", type=str, default=None, help="Optional Excel input override")
    parser.add_argument("--sheet", type=str, default=None, help="Optional Excel sheet override")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated target columns override")
    parser.add_argument(
        "--grid",
        type=str,
        default=DEFAULT_GRID,
        help=(
            "Semicolon-separated specs like "
            "machine_learning.Ridge.alpha=0.1,1,10;machine_learning.Random_Forest.lags=2,3,5"
        ),
    )
    parser.add_argument("--model-groups", type=str, default="machine_learning,probabilistic", help="Model groups to enable")
    parser.add_argument("--n-cycles", type=int, default=None, help="Synthetic cycles per scenario")
    parser.add_argument("--max-bims", type=int, default=None, help="Limit BIM groups for quick experiments")
    parser.add_argument("--output", type=str, default=None, help="Default: outputs/sensitivity/hyperparameters")
    parser.add_argument("--include-baseline", action="store_true", help="Run the unmodified baseline configuration first")
    return parser.parse_args()


def main() -> None:
    root = project_root()
    args = parse_args()
    base_config = load_config(root, Path(args.config) if args.config else None)
    base_config = _apply_cli_overrides(base_config, args)
    out_root = Path(args.output) if args.output else root / "outputs" / "sensitivity" / "hyperparameters"
    out_root = out_root if out_root.is_absolute() else root / out_root
    logger = setup_logging(out_root / "logs")
    logger.info("Starting hyperparameter sensitivity analysis in %s", out_root)

    features, schema = _prepare_features(base_config, logger)
    base_dirs = output_dirs(root)
    model_groups = _parse_model_groups(args.model_groups)
    scenarios = _scenario_specs(args.grid)
    if args.include_baseline:
        scenarios.insert(0, {"key": "baseline", "group": "", "model": "", "param": "", "value": None})

    all_metrics: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    for index, spec in enumerate(scenarios, start=1):
        scenario_name = _scenario_name(index, spec)
        scenario_root = out_root / scenario_name
        scenario_dirs = _scenario_output_dirs(scenario_root, base_dirs)
        ensure_directories(scenario_dirs.values())
        scenario_config = _scenario_config(base_config, spec, model_groups, args.n_cycles, args.max_bims)

        logger.info("Running scenario=%s spec=%s", scenario_name, spec)
        metrics, _forecasts = ForecastRunner(scenario_config, schema, scenario_dirs, logger).run(features)
        metrics = _annotate(metrics, scenario_name, spec)
        metrics.to_csv(scenario_root / "metrics" / "model_metrics.csv", index=False, encoding="utf-8-sig")
        rankings = rank_models(metrics)
        rankings = _annotate(rankings, scenario_name, spec) if not rankings.empty else rankings
        rankings_path = scenario_root / "rankings" / "model_rankings.csv"
        rankings_path.parent.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(rankings_path, index=False, encoding="utf-8-sig")
        all_metrics.append(metrics)
        all_rankings.append(rankings)

    combined_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    combined_rankings = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    per_model = _summarize_by_model(combined_metrics)
    project = _summarize_project(combined_metrics)
    winners = _best_scenarios(per_model)

    combined_metrics.to_csv(out_root / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    combined_rankings.to_csv(out_root / "all_model_rankings.csv", index=False, encoding="utf-8-sig")
    per_model.to_csv(out_root / "per_model_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    project.to_csv(out_root / "project_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(out_root / "recommended_hyperparameter_scenarios.csv", index=False, encoding="utf-8-sig")
    _write_manifest(out_root, scenarios, model_groups, args)
    logger.info("Hyperparameter sensitivity complete. Summary=%s", out_root / "per_model_sensitivity_summary.csv")


def _scenario_specs(raw: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        left, raw_values = item.split("=", 1)
        parts = [part.strip() for part in left.split(".") if part.strip()]
        if len(parts) < 2:
            raise ValueError(f"Invalid sensitivity key: {left}")
        if parts[0] == "synthetic_training":
            group, model, param = "synthetic_training", "", ".".join(parts[1:])
        else:
            if len(parts) != 3:
                raise ValueError(f"Expected group.model.param for model hyperparameter: {left}")
            group, model, param = parts
        for value in [_parse_value(value) for value in raw_values.split(",") if value.strip()]:
            specs.append({"key": left, "group": group, "model": model, "param": param, "value": value})
    if not specs:
        raise ValueError("At least one sensitivity scenario is required")
    return specs


def _parse_value(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _scenario_config(
    base_config: ProjectConfig,
    spec: dict[str, Any],
    model_groups: set[str],
    n_cycles: int | None,
    max_bims: int | None,
) -> ProjectConfig:
    raw = deepcopy(base_config.raw)
    enabled = raw.setdefault("models", {}).setdefault("enabled_groups", {})
    for group in list(enabled):
        enabled[group] = group in model_groups
    for group in model_groups:
        enabled[group] = True
    if n_cycles is not None:
        raw.setdefault("synthetic_training", {})["n_cycles"] = n_cycles
    if max_bims is not None:
        raw.setdefault("execution", {})["max_bims"] = max_bims
    if spec["key"] != "baseline":
        if spec["group"] == "synthetic_training":
            raw.setdefault("synthetic_training", {})[spec["param"]] = spec["value"]
        else:
            raw.setdefault("model_hyperparameters", {}).setdefault(spec["group"], {}).setdefault(spec["model"], {})[spec["param"]] = spec["value"]
    return ProjectConfig(raw=raw, root=base_config.root)


def _scenario_output_dirs(scenario_root: Path, base_dirs: dict[str, Path]) -> dict[str, Path]:
    return {key: scenario_root / key for key in base_dirs if key not in {"logs"}} | {"logs": scenario_root / "logs"}


def _scenario_name(index: int, spec: dict[str, Any]) -> str:
    if spec["key"] == "baseline":
        return f"{index:03d}_baseline"
    return f"{index:03d}_{safe_name(spec['key'])}_{safe_name(spec['value'])}"


def _annotate(frame: pd.DataFrame, scenario_name: str, spec: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["scenario"] = scenario_name
    result["sensitivity_key"] = spec["key"]
    result["sensitivity_value"] = "" if spec["value"] is None else spec["value"]
    return result


def _summarize_by_model(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    ok = metrics[metrics["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    summary = (
        ok.groupby(["sensitivity_key", "sensitivity_value", "target", "category", "model"], dropna=False)
        .agg(
            ok_runs=("status", "size"),
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            mean_MAE=("MAE", "mean"),
            mean_SMAPE=("SMAPE", "mean"),
            mean_R2=("R2", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    summary["RMSE_rank"] = summary.groupby(["target", "category", "model", "sensitivity_key"])["mean_RMSE"].rank(method="min", ascending=True)
    summary["SMAPE_rank"] = summary.groupby(["target", "category", "model", "sensitivity_key"])["mean_SMAPE"].rank(method="min", ascending=True)
    summary["R2_rank"] = summary.groupby(["target", "category", "model", "sensitivity_key"])["mean_R2"].rank(method="min", ascending=False)
    summary["mean_rank"] = summary[["RMSE_rank", "SMAPE_rank", "R2_rank"]].mean(axis=1)
    return summary.sort_values(["target", "category", "model", "sensitivity_key", "mean_rank"], na_position="last")


def _summarize_project(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return (
        metrics.groupby(["sensitivity_key", "sensitivity_value"], dropna=False)
        .agg(
            rows=("status", "size"),
            ok_runs=("status", lambda x: int((x == "ok").sum())),
            skipped_runs=("status", lambda x: int((x == "skipped").sum())),
            failed_runs=("status", lambda x: int((x == "failed").sum())),
            mean_RMSE=("RMSE", "mean"),
            mean_MAE=("MAE", "mean"),
            mean_SMAPE=("SMAPE", "mean"),
            mean_R2=("R2", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["sensitivity_key", "mean_RMSE"], na_position="last")
    )


def _best_scenarios(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return (
        summary.sort_values(["target", "category", "model", "sensitivity_key", "mean_rank", "mean_RMSE"], na_position="last")
        .groupby(["target", "category", "model", "sensitivity_key"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def _write_manifest(out_root: Path, scenarios: list[dict[str, Any]], model_groups: set[str], args: argparse.Namespace) -> None:
    manifest = {
        "analysis": "hyperparameter_sensitivity",
        "model_groups": sorted(model_groups),
        "n_cycles": args.n_cycles,
        "max_bims": args.max_bims,
        "scenarios": scenarios,
        "outputs": {
            "all_metrics": "all_model_metrics.csv",
            "all_rankings": "all_model_rankings.csv",
            "per_model": "per_model_sensitivity_summary.csv",
            "project": "project_sensitivity_summary.csv",
            "recommended": "recommended_hyperparameter_scenarios.csv",
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
