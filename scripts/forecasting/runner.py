from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.biological_models.growth_models import get_biological_models
from scripts.deep_learning.deep_models import get_deep_learning_models
from scripts.differential_equations.ode_models import get_differential_equation_models
from scripts.evaluation.interpretability import save_feature_importance, save_shap_status
from scripts.evaluation.metrics import regression_metrics
from scripts.forecasting.base import ForecastModel, ModelSkipped
from scripts.forecasting.classical import get_classical_models
from scripts.forecasting.statistical import get_statistical_models
from scripts.hybrid_models.hybrid import get_hybrid_models
from scripts.machine_learning.ml_models import get_machine_learning_models
from scripts.preprocessing.data_loader import DataSchema
from scripts.probabilistic_models.probabilistic import get_probabilistic_models
from scripts.synthetic.growth_cycles import SyntheticGrowthCycleGenerator
from scripts.synthetic.quality import evaluate_synthetic_cycles
from scripts.utils.config import ProjectConfig
from scripts.utils.dependencies import has_module
from scripts.utils.paths import safe_name
from scripts.utils.plot_style import PALETTE, apply_plot_style, polish_axis, save_figure_no_return


class ForecastRunner:
    def __init__(
        self,
        config: ProjectConfig,
        schema: DataSchema,
        output_dirs: dict[str, Path],
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.schema = schema
        self.output_dirs = output_dirs
        self.logger = logger
        self.models = self._build_registry()
        self._save_model_catalog()
        self.synthetic_generator = SyntheticGrowthCycleGenerator(config)
        self._synthetic_cache: dict[str, pd.DataFrame] = {}

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        metric_rows: list[dict[str, Any]] = []
        forecast_rows: list[dict[str, Any]] = []
        groups = list(df.groupby(self.schema.group_col, dropna=False))
        max_bims = self.config.get("execution.max_bims")
        if max_bims:
            groups = groups[: int(max_bims)]

        synthetic_mode = self._use_synthetic_training()
        for group, frame in groups:
            for target in self.schema.target_columns:
                if synthetic_mode:
                    metrics, forecasts = self._run_synthetic_group_target(str(group), frame, target)
                else:
                    metrics, forecasts = self._run_group_target(str(group), frame, target)
                metric_rows.extend(metrics)
                forecast_rows.extend(forecasts)

        metrics_df = pd.DataFrame(metric_rows)
        forecasts_df = pd.DataFrame(forecast_rows)
        metrics_path = self.output_dirs["metrics"] / "model_metrics.csv"
        forecast_path = self.output_dirs["forecasts"] / "all_forecasts.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        forecasts_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
        self.logger.info("Saved metrics=%s forecasts=%s", metrics_path, forecast_path)
        return metrics_df, forecasts_df

    def _run_synthetic_group_target(
        self,
        group: str,
        frame: pd.DataFrame,
        target: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        metrics: list[dict[str, Any]] = []
        forecasts: list[dict[str, Any]] = []
        if target not in frame.columns:
            return metrics, forecasts

        frame = frame.sort_values(self.schema.date_col).copy()
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        valid = frame.dropna(subset=[target, self.schema.date_col]).copy()
        n = len(valid)
        min_train = int(self.config.get("execution.min_train_points", 5))
        if n <= min_train:
            metrics.append(self._status_row(group, target, "pipeline", "TSTR_temporal_holdout", "skipped", 0, "too_few_real_points"))
            return metrics, forecasts

        configured_horizon = int(self.config.get("execution.forecast_horizon", 5))
        fraction_horizon = max(1, int(math.ceil(n * float(self.config.get("execution.test_fraction", 0.25)))))
        test_size = min(configured_horizon, fraction_horizon, n - min_train)
        train_frame = valid.iloc[:-test_size].copy()
        test_frame = valid.iloc[-test_size:].copy()
        train = pd.Series(train_frame[target].to_numpy(dtype=float), index=pd.to_datetime(train_frame[self.schema.date_col]), name=target)
        y_true_holdout = test_frame[target].to_numpy(dtype=float)
        test_dates = pd.to_datetime(test_frame[self.schema.date_col])
        synthetic_cycles = self._synthetic_cycles_for_training_segment(train_frame, target, group)
        self._save_synthetic_quality(group, target, train, synthetic_cycles)
        self.logger.info(
            "Synthetic training holdout for BIM=%s target=%s train_real=%d test_real=%d synthetic_cycles=%d",
            group,
            target,
            len(train),
            len(test_frame),
            int(synthetic_cycles["cycle_id"].nunique()) if "cycle_id" in synthetic_cycles else 0,
        )

        for model in self.models:
            started = time.perf_counter()
            try:
                synthetic_method = getattr(model, "fit_predict_from_synthetic", None)
                if not callable(synthetic_method):
                    raise ModelSkipped("model does not implement synthetic-cycle training")
                lags = max(1, int(getattr(model, "lags", self.config.get("model_hyperparameters.machine_learning.Linear_Regression.lags", 3))))
                if len(train) < lags:
                    raise ModelSkipped(f"requires at least {lags} real warmup points")
                validation = pd.concat([train.tail(lags), pd.Series(y_true_holdout, index=test_dates, name=target)])
                pred, y_true, test_dates = synthetic_method(
                    synthetic_cycles,
                    validation,
                    frame_validation=valid,
                    target=target,
                )
                pred = np.asarray(pred, dtype=float).reshape(-1)
                y_true = np.asarray(y_true, dtype=float).reshape(-1)
                if len(pred) != len(y_true):
                    raise ValueError(f"expected {len(y_true)} predictions, got {len(pred)}")
                elapsed = time.perf_counter() - started
                metric = regression_metrics(y_true, pred, n_params=getattr(model, "n_params", 1))
                row = self._status_row(group, target, model.category, model.name, "ok", elapsed, None)
                row.update(metric)
                row["validation_mode"] = "TSTR_temporal_holdout"
                row["train_source"] = "synthetic_growth_cycles_from_real_train"
                row["train_points"] = int(len(synthetic_cycles))
                row["test_points"] = int(len(y_true))
                row["real_train_points"] = int(len(train))
                row["real_test_points"] = int(len(y_true_holdout))
                row["metadata"] = json.dumps(_jsonable(model.metadata), ensure_ascii=False)
                metrics.append(row)
                forecasts.extend(
                    self._forecast_rows(
                        group,
                        target,
                        model,
                        test_dates,
                        y_true,
                        pred,
                        validation_mode="TSTR_temporal_holdout",
                        train_source="synthetic_growth_cycles_from_real_train",
                    )
                )
                self._save_model_artifact(model, group, target)
                self._save_interpretability(model, group, target)
                self._plot_synthetic_validation(group, target, model.name, valid, test_dates, pred)
            except ModelSkipped as exc:
                elapsed = time.perf_counter() - started
                row = self._status_row(group, target, model.category, model.name, "skipped", elapsed, str(exc))
                row["validation_mode"] = "TSTR_temporal_holdout"
                row["train_source"] = "synthetic_growth_cycles_from_real_train"
                row["real_train_points"] = int(len(train))
                row["real_test_points"] = int(len(y_true_holdout))
                metrics.append(row)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.logger.exception("Synthetic validation failed BIM=%s target=%s model=%s", group, target, model.name)
                row = self._status_row(group, target, model.category, model.name, "failed", elapsed, str(exc))
                row["validation_mode"] = "TSTR_temporal_holdout"
                row["train_source"] = "synthetic_growth_cycles_from_real_train"
                row["real_train_points"] = int(len(train))
                row["real_test_points"] = int(len(y_true_holdout))
                metrics.append(row)
                if not self.config.get("execution.continue_on_error", True):
                    raise
        return metrics, forecasts

    def _run_group_target(
        self,
        group: str,
        frame: pd.DataFrame,
        target: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        metrics: list[dict[str, Any]] = []
        forecasts: list[dict[str, Any]] = []
        if target not in frame.columns:
            return metrics, forecasts

        frame = frame.sort_values(self.schema.date_col).copy()
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        valid = frame.dropna(subset=[target, self.schema.date_col]).copy()
        n = len(valid)
        min_train = int(self.config.get("execution.min_train_points", 5))
        if n <= min_train:
            metrics.append(self._status_row(group, target, "pipeline", "split", "skipped", 0, "too_few_points"))
            return metrics, forecasts

        configured_horizon = int(self.config.get("execution.forecast_horizon", 5))
        fraction_horizon = max(1, int(math.ceil(n * float(self.config.get("execution.test_fraction", 0.25)))))
        test_size = min(configured_horizon, fraction_horizon, n - min_train)
        train_frame = valid.iloc[:-test_size].copy()
        test_frame = valid.iloc[-test_size:].copy()
        train = pd.Series(train_frame[target].to_numpy(dtype=float), index=pd.to_datetime(train_frame[self.schema.date_col]), name=target)
        y_true = test_frame[target].to_numpy(dtype=float)
        test_dates = pd.to_datetime(test_frame[self.schema.date_col])

        self.logger.info("Training %d models for BIM=%s target=%s train=%d test=%d", len(self.models), group, target, len(train), len(y_true))
        for model in self.models:
            started = time.perf_counter()
            try:
                if len(train.dropna()) < model.min_points:
                    raise ModelSkipped(f"requires at least {model.min_points} training points")
                pred = model.fit_predict(train, len(y_true), frame_train=train_frame, frame_test=test_frame, target=target)
                pred = np.asarray(pred, dtype=float).reshape(-1)[: len(y_true)]
                if len(pred) != len(y_true):
                    raise ValueError(f"expected {len(y_true)} predictions, got {len(pred)}")
                elapsed = time.perf_counter() - started
                metric = regression_metrics(y_true, pred, n_params=getattr(model, "n_params", 1))
                row = self._status_row(group, target, model.category, model.name, "ok", elapsed, None)
                row.update(metric)
                row["validation_mode"] = "temporal_holdout"
                row["train_source"] = "real_prefix"
                row["train_points"] = int(len(train))
                row["test_points"] = int(len(y_true))
                row["metadata"] = json.dumps(_jsonable(model.metadata), ensure_ascii=False)
                metrics.append(row)
                forecasts.extend(self._forecast_rows(group, target, model, test_dates, y_true, pred))
                self._save_model_artifact(model, group, target)
                self._save_interpretability(model, group, target)
                self._plot_forecast(group, target, model.name, train_frame, test_frame, pred)
            except ModelSkipped as exc:
                elapsed = time.perf_counter() - started
                metrics.append(self._status_row(group, target, model.category, model.name, "skipped", elapsed, str(exc)))
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.logger.exception("Model failed BIM=%s target=%s model=%s", group, target, model.name)
                metrics.append(self._status_row(group, target, model.category, model.name, "failed", elapsed, str(exc)))
                if not self.config.get("execution.continue_on_error", True):
                    raise
        return metrics, forecasts

    def _use_synthetic_training(self) -> bool:
        strategy = str(self.config.get("validation.strategy", "") or "").strip().lower()
        return bool(self.config.get("synthetic_training.enabled", False)) or strategy == "synthetic_full_cycle"

    def _synthetic_cycles_for_training_segment(self, train_frame: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
        valid = train_frame.dropna(subset=[target, self.schema.date_col]).copy()
        synthetic = self.synthetic_generator.generate(
            valid[target],
            [len(valid)],
            target,
            group=group,
            observed_frame=valid,
        )
        if self.config.get("synthetic_training.save_dataset", True):
            processed_dir = self.output_dirs["processed"]
            processed_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"{safe_name(group)}_{safe_name(target)}"
            path = processed_dir / f"synthetic_growth_cycles_train_{suffix}.csv"
            synthetic.to_csv(path, index=False, encoding="utf-8-sig")
            self.logger.info("Synthetic growth cycles saved for BIM=%s target=%s path=%s rows=%d", group, target, path, len(synthetic))
        return synthetic

    def _save_synthetic_quality(self, group: str, target: str, train: pd.Series, synthetic_cycles: pd.DataFrame) -> None:
        result = evaluate_synthetic_cycles(train, synthetic_cycles, target=target, group=group)
        diagnostics_dir = self.output_dirs["diagnostics"] / "synthetic_quality"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{safe_name(group)}_{safe_name(target)}"
        result.summary.to_csv(diagnostics_dir / f"quality_{suffix}.csv", index=False, encoding="utf-8-sig")
        if not result.by_phase.empty:
            result.by_phase.to_csv(diagnostics_dir / f"phase_profile_{suffix}.csv", index=False, encoding="utf-8-sig")

    def _build_registry(self) -> list[ForecastModel]:
        enabled = self.config.get("models.enabled_groups", {})
        seasonal_periods = int(self.config.get("validation.seasonal_periods", 3))
        random_state = int(self.config.get("execution.random_state", 42))
        heavy = bool(self.config.get("execution.heavy_deep_learning", False))
        hyper = self.config.get("model_hyperparameters", {}) or {}
        registry: list[ForecastModel] = []
        if enabled.get("classical", True):
            registry.extend(get_classical_models(seasonal_periods, dict(hyper.get("classical", {}) or {})))
        if enabled.get("statistical", True):
            registry.extend(get_statistical_models(seasonal_periods, dict(hyper.get("statistical", {}) or {})))
        if enabled.get("biological", True):
            registry.extend(get_biological_models(dict(hyper.get("biological", {}) or {})))
        if enabled.get("differential_equations", True):
            registry.extend(get_differential_equation_models(dict(hyper.get("differential_equations", {}) or {})))
        if enabled.get("probabilistic", True):
            registry.extend(get_probabilistic_models(dict(hyper.get("probabilistic", {}) or {}), random_state=random_state))
        if enabled.get("machine_learning", True):
            registry.extend(get_machine_learning_models(random_state, dict(hyper.get("machine_learning", {}) or {})))
        if enabled.get("deep_learning", True):
            registry.extend(get_deep_learning_models(random_state, heavy, dict(hyper.get("deep_learning", {}) or {})))
        if enabled.get("hybrid", True):
            registry.extend(get_hybrid_models())
        return registry

    def _save_model_catalog(self) -> None:
        diagnostics_dir = self.output_dirs["diagnostics"]
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for model in self.models:
            card = model.model_card()
            rows.append(
                {
                    "category": card["category"],
                    "model": card["model"],
                    "min_points": card["min_points"],
                    "n_params": card["n_params"],
                    "hyperparameters": json.dumps(_jsonable(card["hyperparameters"]), ensure_ascii=False),
                }
            )
        pd.DataFrame(rows).to_csv(diagnostics_dir / "model_hyperparameters.csv", index=False, encoding="utf-8-sig")
        (diagnostics_dir / "model_hyperparameters.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def _status_row(
        self,
        group: str,
        target: str,
        category: str,
        model: str,
        status: str,
        elapsed: float,
        error: str | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "BIM": group,
            "target": target,
            "category": category,
            "model": model,
            "status": status,
            "fit_seconds": elapsed,
            "error": error,
        }
        for metric in ["RMSE", "MAE", "MAPE", "SMAPE", "R2", "Adjusted_R2", "AIC", "BIC", "LogLikelihood"]:
            row.setdefault(metric, np.nan)
        return row

    def _forecast_rows(
        self,
        group: str,
        target: str,
        model: ForecastModel,
        dates: pd.Series | pd.Index,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        validation_mode: str = "temporal_holdout",
        train_source: str = "real_prefix",
    ) -> list[dict[str, Any]]:
        rows = []
        for date, true, pred in zip(dates, y_true, y_pred):
            rows.append(
                {
                    "BIM": group,
                    "target": target,
                    "category": model.category,
                    "model": model.name,
                    "date": str(pd.to_datetime(date)),
                    "y_true": float(true),
                    "y_pred": float(pred),
                    "residual": float(true - pred),
                    "validation_mode": validation_mode,
                    "train_source": train_source,
                }
            )
        model_dir = self.output_dirs["forecasts"] / safe_name(group) / safe_name(target)
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(model_dir / f"{safe_name(model.name)}.csv", index=False, encoding="utf-8-sig")
        return rows

    def _save_model_artifact(self, model: ForecastModel, group: str, target: str) -> None:
        if not self.config.get("execution.save_models", True):
            return
        model_dir = self.output_dirs["models"] / safe_name(group) / safe_name(target)
        model_dir.mkdir(parents=True, exist_ok=True)
        card = {
            "BIM": group,
            "target": target,
            "category": model.category,
            "model": model.name,
            "metadata": _jsonable(model.metadata),
        }
        (model_dir / f"{safe_name(model.name)}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            model.save(model_dir / f"{safe_name(model.name)}.pkl")
        except Exception as exc:
            self.logger.debug("Could not pickle model %s: %s", model.name, exc)

    def _save_interpretability(self, model: ForecastModel, group: str, target: str) -> None:
        out_dir = self.output_dirs["shap"] / safe_name(group) / safe_name(target)
        importance = model.feature_importance()
        if not importance.empty:
            save_feature_importance(importance, out_dir / f"{safe_name(model.name)}_feature_importance.csv")
        if not has_module("shap"):
            save_shap_status(out_dir / f"{safe_name(model.name)}_shap_status.csv", model.name, "shap package is not installed")

    def _plot_forecast(
        self,
        group: str,
        target: str,
        model_name: str,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        pred: np.ndarray,
    ) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        apply_plot_style()
        make_png = bool(self.config.get("execution.make_png", True))
        make_svg = bool(self.config.get("execution.make_svg", True))
        out_dir = self.output_dirs["figures"] / safe_name(group) / "forecasts" / safe_name(target)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        ax.plot(train_frame[self.schema.date_col], train_frame[target], label="entrenamiento", marker="o", linewidth=2.4, color=PALETTE["primary"])
        ax.plot(test_frame[self.schema.date_col], test_frame[target], label="observado", marker="o", linewidth=2.4, color=PALETTE["secondary"])
        ax.plot(test_frame[self.schema.date_col], pred, label=f"prediccion {model_name}", marker="X", linewidth=2.5, linestyle="--", color=PALETTE["accent"])
        polish_axis(ax, f"{group} - {target} - {model_name}", "Fecha", target, legend=True)
        fig.autofmt_xdate()
        save_figure_no_return(fig, out_dir / safe_name(model_name), make_png, make_svg)
        plt.close(fig)

    def _plot_synthetic_validation(
        self,
        group: str,
        target: str,
        model_name: str,
        real_frame: pd.DataFrame,
        pred_dates: pd.Series | pd.Index,
        pred: np.ndarray,
    ) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        apply_plot_style()
        make_png = bool(self.config.get("execution.make_png", True))
        make_svg = bool(self.config.get("execution.make_svg", True))
        out_dir = self.output_dirs["figures"] / safe_name(group) / "forecasts" / safe_name(target)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        real_dates = pd.to_datetime(real_frame[self.schema.date_col])
        ax.plot(real_dates, real_frame[target], label="ciclo real", marker="o", linewidth=2.5, color=PALETTE["primary"])
        ax.plot(pd.to_datetime(pred_dates), pred, label=f"prediccion {model_name}", marker="X", linewidth=2.5, linestyle="--", color=PALETTE["accent"])
        polish_axis(ax, f"{group} - {target} - {model_name} validacion ciclo completo", "Fecha", target, legend=True)
        fig.autofmt_xdate()
        save_figure_no_return(fig, out_dir / safe_name(model_name), make_png, make_svg)
        plt.close(fig)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
