from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.evaluation.metrics import rank_models
from scripts.feature_engineering.features import FeatureEngineer
from scripts.forecasting.runner import ForecastRunner
from scripts.preprocessing.data_loader import coerce_numeric_columns, detect_schema, load_excel
from scripts.preprocessing.quality import DataQualityProcessor
from scripts.utils.config import ProjectConfig, load_config
from scripts.utils.logging_utils import setup_logging
from scripts.utils.paths import ensure_directories, output_dirs, project_root, safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for synthetic growth-cycle decline probability."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input", type=str, default=None, help="Optional Excel input override")
    parser.add_argument("--sheet", type=str, default=None, help="Optional Excel sheet override")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated target columns override")
    parser.add_argument(
        "--probabilities",
        type=str,
        default="0.4,0.7,0.9",
        help="Comma-separated decline probabilities to evaluate",
    )
    parser.add_argument("--n-cycles", type=int, default=None, help="Synthetic cycles per scenario")
    parser.add_argument("--max-bims", type=int, default=None, help="Limit BIM groups for quick experiments")
    parser.add_argument(
        "--model-groups",
        type=str,
        default="machine_learning",
        help="Comma-separated model groups to enable for the sensitivity run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: outputs/sensitivity/decline_probability",
    )
    return parser.parse_args()


def main() -> None:
    root = project_root()
    args = parse_args()
    base_config = load_config(root, Path(args.config) if args.config else None)
    base_config = _apply_cli_overrides(base_config, args)

    base_dirs = output_dirs(root)
    out_root = Path(args.output) if args.output else root / "outputs" / "sensitivity" / "decline_probability"
    out_root = out_root if out_root.is_absolute() else root / out_root
    logger = setup_logging(out_root / "logs")
    logger.info("Starting decline sensitivity analysis in %s", out_root)

    probabilities = _parse_probabilities(args.probabilities)
    model_groups = _parse_model_groups(args.model_groups)
    features, schema = _prepare_features(base_config, logger)
    all_metrics: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []

    for probability in probabilities:
        scenario_name = f"decline_{probability:.2f}".replace(".", "_")
        scenario_root = out_root / scenario_name
        scenario_dirs = _scenario_output_dirs(scenario_root, base_dirs)
        ensure_directories(scenario_dirs.values())
        scenario_config = _scenario_config(
            base_config,
            probability=probability,
            model_groups=model_groups,
            n_cycles=args.n_cycles,
            max_bims=args.max_bims,
        )

        logger.info("Running scenario=%s decline_probability=%.3f", scenario_name, probability)
        metrics, _forecasts = ForecastRunner(scenario_config, schema, scenario_dirs, logger).run(features)
        metrics["decline_probability"] = probability
        metrics["scenario"] = scenario_name
        metrics_path = scenario_root / "metrics" / "model_metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        rankings = rank_models(metrics)
        if not rankings.empty:
            rankings["decline_probability"] = probability
            rankings["scenario"] = scenario_name
        rankings_path = scenario_root / "rankings" / "model_rankings.csv"
        rankings_path.parent.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(rankings_path, index=False, encoding="utf-8-sig")

        all_metrics.append(metrics)
        all_rankings.append(rankings)

    combined_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    combined_rankings = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    summary = _summarize(combined_metrics)
    winners = _winners(summary)

    combined_metrics.to_csv(out_root / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
    combined_rankings.to_csv(out_root / "all_model_rankings.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_root / "decline_probability_summary.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(out_root / "recommended_decline_probability.csv", index=False, encoding="utf-8-sig")

    _write_manifest(
        out_root,
        probabilities=probabilities,
        model_groups=model_groups,
        n_cycles=args.n_cycles or int(base_config.get("synthetic_training.n_cycles", 2000)),
        max_bims=args.max_bims if args.max_bims is not None else base_config.get("execution.max_bims"),
    )
    logger.info("Sensitivity analysis complete. Summary=%s", out_root / "decline_probability_summary.csv")


def _prepare_features(config: ProjectConfig, logger: logging.Logger) -> tuple[pd.DataFrame, Any]:
    raw_path = _resolve_raw_file(config, logger)
    raw_df = load_excel(raw_path, config.get("data.sheet_name"), logger)
    schema = detect_schema(raw_df, config, logger)
    raw_df = coerce_numeric_columns(raw_df, schema.numeric_columns)
    quality = DataQualityProcessor(
        schema=schema,
        imputation_strategy=str(config.get("validation.imputation_strategy", "time_interpolate")),
        iqr_multiplier=float(config.get("validation.outlier_iqr_multiplier", 1.5)),
        logger=logger,
    ).run(raw_df)
    features = FeatureEngineer(
        schema=schema,
        rolling_windows=list(config.get("validation.rolling_windows", [3, 5])),
        logger=logger,
    ).transform(quality.clean_data)
    return features, schema


def _scenario_config(
    base_config: ProjectConfig,
    probability: float,
    model_groups: set[str],
    n_cycles: int | None,
    max_bims: int | None,
) -> ProjectConfig:
    raw = deepcopy(base_config.raw)
    raw.setdefault("validation", {})["strategy"] = "synthetic_full_cycle"
    raw.setdefault("synthetic_training", {})["enabled"] = True
    raw["synthetic_training"]["decline_probability"] = probability
    raw["synthetic_training"]["save_dataset"] = True
    if n_cycles is not None:
        raw["synthetic_training"]["n_cycles"] = n_cycles
    if max_bims is not None:
        raw.setdefault("execution", {})["max_bims"] = max_bims

    enabled = raw.setdefault("models", {}).setdefault("enabled_groups", {})
    for group in list(enabled):
        enabled[group] = group in model_groups
    for group in model_groups:
        enabled[group] = True

    raw.setdefault("execution", {})["save_models"] = False
    return ProjectConfig(raw=raw, root=base_config.root)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    ok = metrics[metrics["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    group_cols = ["decline_probability", "target", "category", "model"]
    agg = (
        ok.groupby(group_cols, dropna=False)
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
    agg["RMSE_rank"] = agg.groupby(["target", "category", "model"])["mean_RMSE"].rank(method="min", ascending=True)
    agg["SMAPE_rank"] = agg.groupby(["target", "category", "model"])["mean_SMAPE"].rank(method="min", ascending=True)
    agg["R2_rank"] = agg.groupby(["target", "category", "model"])["mean_R2"].rank(method="min", ascending=False)
    rank_cols = ["RMSE_rank", "SMAPE_rank", "R2_rank"]
    agg["mean_rank"] = agg[rank_cols].mean(axis=1)
    return agg.sort_values(["target", "model", "mean_rank", "mean_RMSE"], na_position="last")


def _winners(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    winners = (
        summary.sort_values(["target", "category", "model", "mean_rank", "mean_RMSE"], na_position="last")
        .groupby(["target", "category", "model"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    return winners


def _scenario_output_dirs(scenario_root: Path, base_dirs: dict[str, Path]) -> dict[str, Path]:
    return {
        key: scenario_root / key
        for key in base_dirs
        if key not in {"logs"}
    } | {"logs": scenario_root / "logs"}


def _apply_cli_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    raw = json.loads(json.dumps(config.raw))
    if args.input:
        raw["data"]["raw_file"] = args.input
    if args.sheet:
        raw["data"]["sheet_name"] = args.sheet
    if args.targets:
        raw["data"]["target_columns"] = [item.strip() for item in args.targets.split(",") if item.strip()]
    return ProjectConfig(raw=raw, root=config.root)


def _resolve_raw_file(config: ProjectConfig, logger: logging.Logger) -> Path:
    raw_path = config.path("data.raw_file")
    if raw_path.exists():
        return raw_path
    raise FileNotFoundError(
        f"Raw Excel not found: {raw_path}. "
        "Place the file at the configured data.raw_file path or run with --input /path/to/file.xlsx."
    )


def _parse_probabilities(raw: str) -> list[float]:
    probabilities = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not probabilities:
        raise ValueError("At least one decline probability is required")
    invalid = [value for value in probabilities if value < 0 or value > 1]
    if invalid:
        raise ValueError(f"Decline probabilities must be between 0 and 1: {invalid}")
    return probabilities


def _parse_model_groups(raw: str) -> set[str]:
    groups = {safe_name(item.strip()) for item in raw.split(",") if item.strip()}
    return groups or {"machine_learning"}


def _write_manifest(
    out_root: Path,
    probabilities: list[float],
    model_groups: set[str],
    n_cycles: int,
    max_bims: Any,
) -> None:
    manifest = {
        "analysis": "decline_probability_sensitivity",
        "probabilities": probabilities,
        "model_groups": sorted(model_groups),
        "n_cycles": n_cycles,
        "max_bims": max_bims,
        "outputs": {
            "all_metrics": "all_model_metrics.csv",
            "all_rankings": "all_model_rankings.csv",
            "summary": "decline_probability_summary.csv",
            "recommended": "recommended_decline_probability.csv",
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
