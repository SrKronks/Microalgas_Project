from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.evaluation.metrics import classification_metrics, rank_classifiers
from scripts.preprocessing.data_loader import DataSchema
from scripts.utils.config import ProjectConfig
from scripts.utils.paths import safe_name


class ClassificationRunner:
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
        self.random_state = int(config.get("execution.random_state", 42))

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.config.get("classification.enabled", True) or not self.schema.label_columns:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        confusion_rows: list[dict[str, Any]] = []
        feature_rows: list[dict[str, Any]] = []

        for label_col in self.schema.label_columns:
            try:
                metrics, predictions, confusion, features = self._run_label(df, label_col)
                metric_rows.extend(metrics)
                prediction_rows.extend(predictions)
                confusion_rows.extend(confusion)
                feature_rows.extend(features)
            except Exception as exc:
                self.logger.exception("Classification failed for label=%s", label_col)
                metric_rows.append(self._status_row(label_col, "pipeline", "failed", 0.0, str(exc)))
                if not self.config.get("execution.continue_on_error", True):
                    raise

        metrics_df = pd.DataFrame(metric_rows)
        predictions_df = pd.DataFrame(prediction_rows)
        confusion_df = pd.DataFrame(confusion_rows)
        feature_importance_df = pd.DataFrame(feature_rows)
        rankings_df = rank_classifiers(metrics_df)

        metrics_dir = self.output_dirs["metrics"]
        rankings_dir = self.output_dirs["rankings"]
        diagnostics_dir = self.output_dirs["diagnostics"]
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rankings_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_dir / "classification_metrics.csv", index=False, encoding="utf-8-sig")
        rankings_df.to_csv(rankings_dir / "classification_rankings.csv", index=False, encoding="utf-8-sig")
        predictions_df.to_csv(metrics_dir / "classification_predictions.csv", index=False, encoding="utf-8-sig")
        confusion_df.to_csv(diagnostics_dir / "classification_confusion_matrices.csv", index=False, encoding="utf-8-sig")
        feature_importance_df.to_csv(diagnostics_dir / "classification_feature_importance.csv", index=False, encoding="utf-8-sig")
        self.logger.info("Classification metrics saved for labels=%s", self.schema.label_columns)
        return metrics_df, rankings_df, predictions_df, confusion_df

    def _run_label(
        self,
        df: pd.DataFrame,
        label_col: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        prepared = self._prepare_dataset(df, label_col)
        if prepared.empty:
            return [self._status_row(label_col, "all", "skipped", 0.0, "no valid labelled rows")], [], [], []

        y = prepared[label_col].astype(str)
        class_counts = y.value_counts()
        min_samples = int(self.config.get("classification.min_samples", 20))
        min_class_count = int(self.config.get("classification.min_class_count", 2))
        if len(prepared) < min_samples:
            return [self._status_row(label_col, "all", "skipped", 0.0, f"requires at least {min_samples} labelled samples")], [], [], []
        if class_counts.min() < min_class_count:
            return [self._status_row(label_col, "all", "skipped", 0.0, f"each class requires at least {min_class_count} samples")], [], [], []

        feature_cols = [col for col in prepared.columns if col not in {label_col, self.schema.date_col, self.schema.group_col, "_row_id"}]
        x = prepared[feature_cols].copy()
        groups = prepared[self.schema.group_col].astype(str)
        indices = np.arange(len(prepared))
        train_idx, test_idx, validation_mode = self._split(indices, y, groups)
        labels = sorted(y.unique().tolist())

        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        confusion_rows: list[dict[str, Any]] = []
        feature_rows: list[dict[str, Any]] = []

        for model_name, estimator in self._models():
            started = time.perf_counter()
            try:
                estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
                y_pred = pd.Series(estimator.predict(x.iloc[test_idx]), dtype="object").astype(str)
                y_proba = estimator.predict_proba(x.iloc[test_idx]) if hasattr(estimator, "predict_proba") else None
                elapsed = time.perf_counter() - started
                metric = classification_metrics(y.iloc[test_idx], y_pred, y_proba, labels)
                row = self._status_row(label_col, model_name, "ok", elapsed, None)
                row.update(metric)
                row.update(
                    {
                        "validation_mode": validation_mode,
                        "train_points": int(len(train_idx)),
                        "test_points": int(len(test_idx)),
                        "n_classes": int(len(labels)),
                        "classes": json.dumps(labels, ensure_ascii=False),
                        "baseline_majority_accuracy": float(class_counts.max() / class_counts.sum()),
                        "feature_count": int(len(feature_cols)),
                    }
                )
                metric_rows.append(row)
                prediction_rows.extend(
                    self._prediction_rows(
                        prepared.iloc[test_idx],
                        label_col,
                        model_name,
                        y.iloc[test_idx].reset_index(drop=True),
                        y_pred.reset_index(drop=True),
                    )
                )
                matrix = _confusion_matrix(y.iloc[test_idx], y_pred, labels)
                for true_idx, true_label in enumerate(labels):
                    for pred_idx, pred_label in enumerate(labels):
                        confusion_rows.append(
                            {
                                "label_target": label_col,
                                "model": model_name,
                                "true_label": true_label,
                                "predicted_label": pred_label,
                                "count": int(matrix[true_idx, pred_idx]),
                            }
                        )
                feature_rows.extend(self._feature_importance_rows(estimator, feature_cols, label_col, model_name))
                self._save_model(estimator, label_col, model_name)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.logger.exception("Classifier failed label=%s model=%s", label_col, model_name)
                metric_rows.append(self._status_row(label_col, model_name, "failed", elapsed, str(exc)))
                if not self.config.get("execution.continue_on_error", True):
                    raise

        return metric_rows, prediction_rows, confusion_rows, feature_rows

    def _prepare_dataset(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        working = df.copy()
        working["_row_id"] = np.arange(len(working))
        working[label_col] = working[label_col].astype("string").str.strip()
        working = working[working[label_col].notna() & working[label_col].ne("")]

        excluded = {self.schema.date_col, self.schema.group_col, "_row_id", *self.schema.label_columns}
        numeric_cols = [
            col
            for col in working.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(working[col])
        ]
        max_missing = float(self.config.get("classification.max_missing_feature_pct", 0.60))
        feature_cols = [col for col in numeric_cols if float(working[col].isna().mean()) <= max_missing]
        keep_cols = ["_row_id", self.schema.date_col, self.schema.group_col, label_col, *feature_cols]
        prepared = working[keep_cols].replace([np.inf, -np.inf], np.nan).copy()
        prepared = prepared.dropna(subset=[label_col])
        return prepared.reset_index(drop=True)

    def _split(
        self,
        indices: np.ndarray,
        y: pd.Series,
        groups: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        fraction = float(self.config.get("classification.test_fraction", 0.25))
        fraction = min(max(fraction, (y.nunique() + 1) / max(len(y), 1)), 0.40)
        strategy = str(self.config.get("classification.validation_strategy", "stratified_holdout")).lower()
        rng = np.random.default_rng(self.random_state)
        if strategy == "group_holdout" and groups.nunique() >= 2:
            unique_groups = groups.drop_duplicates().to_numpy()
            rng.shuffle(unique_groups)
            n_test_groups = max(1, int(round(len(unique_groups) * fraction)))
            test_groups = set(unique_groups[:n_test_groups])
            test_mask = groups.isin(test_groups).to_numpy()
            train_idx = indices[~test_mask]
            test_idx = indices[test_mask]
            if len(train_idx) and len(test_idx):
                return train_idx, test_idx, "group_holdout"

        try:
            from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=self.random_state)
            train_idx, test_idx = next(splitter.split(indices, y))
            return train_idx, test_idx, "stratified_holdout"
        except Exception:
            test_indices: list[int] = []
            for label in sorted(y.unique()):
                label_indices = indices[y.to_numpy() == label].copy()
                rng.shuffle(label_indices)
                n_test = max(1, int(round(len(label_indices) * fraction)))
                test_indices.extend(label_indices[:n_test].tolist())
            test_idx = np.asarray(sorted(test_indices), dtype=int)
            train_idx = np.asarray([idx for idx in indices if idx not in set(test_idx)], dtype=int)
            return train_idx, test_idx, "stratified_holdout_numpy"

    def _models(self) -> list[tuple[str, Any]]:
        try:
            from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier  # type: ignore
            from sklearn.impute import SimpleImputer  # type: ignore
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.naive_bayes import GaussianNB  # type: ignore
            from sklearn.pipeline import make_pipeline  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
            from sklearn.svm import SVC  # type: ignore
        except Exception as exc:
            self.logger.warning("Using NumPy fallback classifiers because scikit-learn is unavailable: %s", exc)
            return [
                ("Fallback_Centroid", _CentroidClassifier()),
                ("Fallback_Gaussian_NB", _GaussianNBClassifier()),
                ("Fallback_KNN", _KNNClassifier(k=5)),
            ]

        return [
            (
                "Logistic_Regression",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.random_state),
                ),
            ),
            (
                "Random_Forest",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=self.random_state),
                ),
            ),
            (
                "Extra_Trees",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=self.random_state),
                ),
            ),
            (
                "Gradient_Boosting",
                make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=self.random_state)),
            ),
            (
                "SVC_RBF",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=self.random_state),
                ),
            ),
            ("Gaussian_NB", make_pipeline(SimpleImputer(strategy="median"), GaussianNB())),
        ]

    def _prediction_rows(
        self,
        frame: pd.DataFrame,
        label_col: str,
        model_name: str,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
            rows.append(
                {
                    "label_target": label_col,
                    "model": model_name,
                    "row_id": int(row["_row_id"]),
                    "BIM": row.get(self.schema.group_col),
                    "date": str(pd.to_datetime(row.get(self.schema.date_col), errors="coerce")),
                    "y_true": y_true.iloc[idx],
                    "y_pred": y_pred.iloc[idx],
                    "correct": bool(y_true.iloc[idx] == y_pred.iloc[idx]),
                }
            )
        return rows

    def _feature_importance_rows(
        self,
        estimator: Any,
        feature_cols: list[str],
        label_col: str,
        model_name: str,
    ) -> list[dict[str, Any]]:
        final = estimator
        if hasattr(estimator, "named_steps"):
            final = list(estimator.named_steps.values())[-1]
        values = None
        if hasattr(final, "feature_importances_"):
            values = np.asarray(final.feature_importances_, dtype=float)
        elif hasattr(final, "coef_"):
            values = np.mean(np.abs(np.asarray(final.coef_, dtype=float)), axis=0)
        if values is None:
            return []
        rows = []
        for feature, importance in sorted(zip(feature_cols, values), key=lambda item: abs(float(item[1])), reverse=True):
            rows.append(
                {
                    "label_target": label_col,
                    "model": model_name,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
        return rows

    def _save_model(self, estimator: Any, label_col: str, model_name: str) -> None:
        if not self.config.get("classification.save_models", True):
            return
        try:
            import joblib  # type: ignore
        except Exception:
            return
        model_dir = self.output_dirs["models"] / "classification" / safe_name(label_col)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, model_dir / f"{safe_name(model_name)}.joblib")

    def _status_row(self, label_col: str, model_name: str, status: str, elapsed: float, error: str | None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "label_target": label_col,
            "model": model_name,
            "status": status,
            "fit_seconds": elapsed,
            "error": error,
        }
        for metric in [
            "Accuracy",
            "Balanced_Accuracy",
            "Macro_Precision",
            "Macro_Recall",
            "Macro_F1",
            "Weighted_F1",
            "Cohen_Kappa",
            "LogLoss",
        ]:
            row.setdefault(metric, np.nan)
        return row


def _confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true.astype(str), y_pred.astype(str)):
        if true in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[true], label_to_idx[pred]] += 1
    return matrix


def _impute_matrix(frame: pd.DataFrame, medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame.astype(float).to_numpy()
    if medians is None:
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    missing_rows, missing_cols = np.where(~np.isfinite(matrix))
    if len(missing_rows):
        matrix[missing_rows, missing_cols] = medians[missing_cols]
    return matrix, medians


def _scale_matrix(
    matrix: np.ndarray,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if means is None:
        means = matrix.mean(axis=0)
    if stds is None:
        stds = matrix.std(axis=0)
        stds = np.where(stds > 1e-12, stds, 1.0)
    return (matrix - means) / stds, means, stds


def _softmax(scores: np.ndarray) -> np.ndarray:
    stable = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(stable)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


class _CentroidClassifier:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_CentroidClassifier":
        matrix, self.medians = _impute_matrix(x)
        scaled, self.means, self.stds = _scale_matrix(matrix)
        self.classes_ = np.asarray(sorted(pd.Series(y).astype(str).unique()))
        self.centroids_ = np.vstack([scaled[pd.Series(y).astype(str).to_numpy() == label].mean(axis=0) for label in self.classes_])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        matrix, _ = _impute_matrix(x, self.medians)
        scaled, _, _ = _scale_matrix(matrix, self.means, self.stds)
        distances = np.linalg.norm(scaled[:, None, :] - self.centroids_[None, :, :], axis=2)
        return _softmax(-distances)


class _GaussianNBClassifier:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_GaussianNBClassifier":
        matrix, self.medians = _impute_matrix(x)
        labels = pd.Series(y).astype(str).to_numpy()
        self.classes_ = np.asarray(sorted(pd.Series(labels).unique()))
        self.priors_ = np.asarray([np.mean(labels == label) for label in self.classes_])
        self.means_ = np.vstack([matrix[labels == label].mean(axis=0) for label in self.classes_])
        self.vars_ = np.vstack([matrix[labels == label].var(axis=0) + 1e-6 for label in self.classes_])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        matrix, _ = _impute_matrix(x, self.medians)
        log_probs = []
        for prior, mean, var in zip(self.priors_, self.means_, self.vars_):
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((matrix - mean) ** 2 / var), axis=1)
            log_probs.append(np.log(max(float(prior), 1e-12)) + log_likelihood)
        return _softmax(np.vstack(log_probs).T)


class _KNNClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_KNNClassifier":
        matrix, self.medians = _impute_matrix(x)
        self.x_train_, self.means, self.stds = _scale_matrix(matrix)
        self.y_train_ = pd.Series(y).astype(str).to_numpy()
        self.classes_ = np.asarray(sorted(pd.Series(self.y_train_).unique()))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        matrix, _ = _impute_matrix(x, self.medians)
        scaled, _, _ = _scale_matrix(matrix, self.means, self.stds)
        distances = np.linalg.norm(scaled[:, None, :] - self.x_train_[None, :, :], axis=2)
        k = min(self.k, len(self.y_train_))
        nearest = np.argsort(distances, axis=1)[:, :k]
        proba = np.zeros((len(scaled), len(self.classes_)), dtype=float)
        for row_idx, neighbor_idx in enumerate(nearest):
            labels = self.y_train_[neighbor_idx]
            for class_idx, label in enumerate(self.classes_):
                proba[row_idx, class_idx] = np.mean(labels == label)
        return proba
