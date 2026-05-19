from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.eda.exploratory import generate_eda_figures, generate_lag_and_rolling_plots, save_descriptive_by_group
from scripts.evaluation.metrics import rank_models
from scripts.feature_engineering.features import FeatureEngineer
from scripts.forecasting.runner import ForecastRunner
from scripts.preprocessing.data_loader import coerce_numeric_columns, detect_schema, load_excel, summarize_dataset
from scripts.preprocessing.quality import DataQualityProcessor
from scripts.reporting.reports import ReportBuilder
from scripts.statistical_analysis.multivariate import multivariate_analysis
from scripts.statistical_analysis.temporal import cross_correlation_table, temporal_diagnostics
from scripts.utils.config import ProjectConfig, load_config
from scripts.utils.dependencies import dependency_status
from scripts.utils.logging_utils import setup_logging
from scripts.utils.paths import ensure_directories, output_dirs, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microalgae analytics and forecasting pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input", type=str, default=None, help="Optional Excel input override")
    parser.add_argument("--sheet", type=str, default=None, help="Optional Excel sheet override")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated target columns override")
    return parser.parse_args()


def main() -> None:
    root = project_root()
    args = parse_args()
    config = load_config(root, Path(args.config) if args.config else None)
    config = _apply_cli_overrides(config, args)
    dirs = output_dirs(root)
    ensure_directories(dirs.values())
    logger = setup_logging(dirs["logs"])

    logger.info("Starting Microalgas pipeline in %s", root)
    deps = dependency_status()
    (dirs["diagnostics"] / "dependency_status.json").write_text(
        json.dumps(deps.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("CUDA available: %s", deps.cuda_available)

    raw_path = _resolve_raw_file(config, logger)
    raw_df = load_excel(raw_path, config.get("data.sheet_name"), logger)
    schema = detect_schema(raw_df, config, logger)
    raw_df = coerce_numeric_columns(raw_df, schema.numeric_columns)
    summary = summarize_dataset(raw_df, schema)
    _write_json(dirs["diagnostics"] / "dataset_summary.json", summary)

    quality_processor = DataQualityProcessor(
        schema=schema,
        imputation_strategy=str(config.get("validation.imputation_strategy", "time_interpolate")),
        iqr_multiplier=float(config.get("validation.outlier_iqr_multiplier", 1.5)),
        logger=logger,
    )
    quality_result = quality_processor.run(raw_df)
    _save_quality_outputs(quality_result, dirs["diagnostics"])

    features = FeatureEngineer(
        schema=schema,
        rolling_windows=list(config.get("validation.rolling_windows", [3, 5])),
        logger=logger,
    ).transform(quality_result.clean_data)
    processed_path = dirs["processed"] / "microalgas_processed.csv"
    features.to_csv(processed_path, index=False, encoding="utf-8-sig")
    logger.info("Processed data saved: %s", processed_path)

    descriptive = save_descriptive_by_group(features, schema, dirs["metrics"], logger)
    generate_eda_figures(
        features,
        schema,
        dirs["figures"],
        logger,
        make_png=bool(config.get("execution.make_png", True)),
        make_svg=bool(config.get("execution.make_svg", True)),
    )
    generate_lag_and_rolling_plots(
        features,
        schema,
        dirs["figures"],
        logger,
        windows=list(config.get("validation.rolling_windows", [3, 5])),
        make_png=bool(config.get("execution.make_png", True)),
        make_svg=bool(config.get("execution.make_svg", True)),
    )

    temporal_diagnostics(
        features,
        schema,
        dirs["diagnostics"],
        logger,
        seasonal_periods=int(config.get("validation.seasonal_periods", 3)),
        make_png=bool(config.get("execution.make_png", True)),
        make_svg=bool(config.get("execution.make_svg", True)),
    )
    cross_correlation_table(features, schema, dirs["diagnostics"])
    multivariate_analysis(features, schema, dirs["diagnostics"], logger)

    runner = ForecastRunner(config, schema, dirs, logger)
    metrics, forecasts = runner.run(features)
    rankings = rank_models(metrics)
    rankings_path = dirs["rankings"] / "model_rankings.csv"
    rankings.to_csv(rankings_path, index=False, encoding="utf-8-sig")
    logger.info("Rankings saved: %s", rankings_path)

    reports = ReportBuilder(config, dirs, logger).build(
        dataset_summary=summary,
        dependency_summary=deps.to_dict(),
        quality=quality_result.quality_table,
        descriptive=descriptive,
        metrics=metrics,
        rankings=rankings,
        forecasts=forecasts,
    )
    logger.info("Pipeline completed. Reports: %s", reports)


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
    candidate = Path("C:/Users/Asus/Downloads/historial-monitoreos-2026-05-14.xlsx")
    if candidate.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, raw_path)
        logger.info("Copied raw Excel from %s to %s", candidate, raw_path)
        return raw_path
    raise FileNotFoundError(f"Raw Excel not found: {raw_path}")


def _save_quality_outputs(quality_result: Any, diagnostics_dir: Path) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    quality_result.quality_table.to_csv(diagnostics_dir / "data_quality.csv", index=False, encoding="utf-8-sig")
    quality_result.outlier_table.to_csv(diagnostics_dir / "outliers_iqr.csv", index=False, encoding="utf-8-sig")
    quality_result.inconsistencies.to_csv(diagnostics_dir / "inconsistencies.csv", index=False, encoding="utf-8-sig")
    quality_result.clean_data.to_csv(diagnostics_dir / "clean_data_snapshot.csv", index=False, encoding="utf-8-sig")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
