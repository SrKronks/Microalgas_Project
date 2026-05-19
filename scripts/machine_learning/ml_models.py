from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from scripts.forecasting.base import ForecastModel, ModelSkipped, SkippedModel, clean_series, require_points


def supervised_lags(series: pd.Series, lags: int) -> tuple[np.ndarray, np.ndarray]:
    values = clean_series(series).to_numpy(dtype=float)
    x_rows, y_rows = [], []
    for idx in range(lags, len(values)):
        x_rows.append(values[idx - lags : idx][::-1])
        y_rows.append(values[idx])
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def supervised_lags_from_synthetic_cycles(
    cycles: pd.DataFrame,
    lags: int,
    value_col: str = "value",
    cycle_col: str = "cycle_id",
) -> tuple[np.ndarray, np.ndarray]:
    if cycles.empty or value_col not in cycles.columns or cycle_col not in cycles.columns:
        raise ModelSkipped("Synthetic growth cycles are empty or malformed")

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for _, frame in cycles.groupby(cycle_col, sort=False):
        values = pd.to_numeric(frame.sort_values("step")[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        for idx in range(lags, len(values)):
            x_rows.append(values[idx - lags : idx][::-1])
            y_rows.append(float(values[idx]))

    if not x_rows:
        raise ModelSkipped("Synthetic growth cycles do not contain enough lagged samples")
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


class RecursiveSklearnModel(ForecastModel):
    category = "machine_learning"
    min_points = 8

    def __init__(self, name: str, estimator_factory: Callable[[], object], lags: int = 3, estimator_params: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.name = name
        self.estimator_factory = estimator_factory
        self.lags = lags
        self.estimator_params = estimator_params or {}
        self.hyperparameters = {"lags": lags, **self.estimator_params}
        self._synthetic_fit_key: tuple[Any, ...] | None = None
        self._synthetic_fit_metadata: dict[str, Any] = {}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, max(self.min_points, self.lags + 3), self.name)
        x_train, y_train = supervised_lags(series, self.lags)
        if len(x_train) < 3:
            raise ModelSkipped("Not enough lagged samples")
        self.fitted_model = self.estimator_factory()
        self.fitted_model.fit(x_train, y_train)
        history = list(series.to_numpy(dtype=float))
        preds: list[float] = []
        for _step in range(horizon):
            features = np.asarray(history[-self.lags:][::-1], dtype=float).reshape(1, -1)
            pred = float(np.asarray(self.fitted_model.predict(features)).ravel()[0])
            preds.append(pred)
            history.append(pred)
        self.metadata = {"lags": self.lags}
        return np.asarray(preds, dtype=float)

    def fit_predict_from_synthetic(
        self,
        synthetic_cycles: pd.DataFrame,
        validation: pd.Series,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        self._ensure_synthetic_fit(synthetic_cycles)
        real = clean_series(validation)
        require_points(real, self.lags + 1, self.name)
        values = real.to_numpy(dtype=float)
        if len(values) <= self.lags:
            raise ModelSkipped(f"{self.name} requires more than {self.lags} real observations for validation")

        x_validation = []
        for idx in range(self.lags, len(values)):
            x_validation.append(values[idx - self.lags : idx][::-1])
        x_array = np.asarray(x_validation, dtype=float)
        pred = np.asarray(self.fitted_model.predict(x_array), dtype=float).reshape(-1)
        y_true = values[self.lags :]
        dates = pd.Index(real.index[self.lags :])
        self.metadata = {
            **self._synthetic_fit_metadata,
            "validation_mode": "full_real_cycle_walk_forward",
            "validation_points": int(len(y_true)),
            "warmup_points": int(self.lags),
        }
        return pred, y_true, dates

    def _ensure_synthetic_fit(self, synthetic_cycles: pd.DataFrame) -> None:
        synthetic_id = synthetic_cycles.attrs.get("synthetic_id", id(synthetic_cycles))
        fit_key = (synthetic_id, self.lags, self.name)
        if self._synthetic_fit_key == fit_key and self.fitted_model is not None:
            return

        x_train, y_train = supervised_lags_from_synthetic_cycles(synthetic_cycles, self.lags)
        if len(x_train) < 3:
            raise ModelSkipped("Not enough synthetic lagged samples")
        self.fitted_model = self.estimator_factory()
        self.fitted_model.fit(x_train, y_train)
        self._synthetic_fit_key = fit_key
        self._synthetic_fit_metadata = {
            "lags": self.lags,
            "train_source": "synthetic_growth_cycles",
            "synthetic_cycles": int(synthetic_cycles["cycle_id"].nunique()),
            "synthetic_samples": int(len(x_train)),
            "prediction_protocol": "one_step_ahead_with_observed_real_lags",
        }

    def feature_importance(self) -> pd.DataFrame:
        if self.fitted_model is None:
            return pd.DataFrame()
        names = [f"lag_{lag}" for lag in range(1, self.lags + 1)]
        values = None
        if hasattr(self.fitted_model, "feature_importances_"):
            values = getattr(self.fitted_model, "feature_importances_")
        elif hasattr(self.fitted_model, "coef_"):
            values = np.ravel(getattr(self.fitted_model, "coef_"))
        elif hasattr(self.fitted_model, "named_steps"):
            final = list(self.fitted_model.named_steps.values())[-1]
            if hasattr(final, "coef_"):
                values = np.ravel(final.coef_)[: len(names)]
        if values is None:
            return pd.DataFrame()
        values = np.asarray(values, dtype=float)[: len(names)]
        return pd.DataFrame({"feature": names[: len(values)], "importance": values, "model": self.name})


def get_machine_learning_models(random_state: int = 42, params: dict[str, object] | None = None) -> list[ForecastModel]:
    params = params or {}

    def cfg(model_name: str, defaults: dict[str, Any]) -> dict[str, Any]:
        merged = defaults.copy()
        merged.update(dict(params.get(model_name, {}) or {}))
        return merged

    def lags(model_name: str, default: int = 3) -> int:
        return int(cfg(model_name, {"lags": default}).get("lags", default))

    try:
        from sklearn.cross_decomposition import PLSRegression  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor  # type: ignore
        from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
        from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # type: ignore
        from sklearn.neighbors import KNeighborsRegressor  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # type: ignore
        from sklearn.svm import SVR  # type: ignore
        from sklearn.tree import DecisionTreeRegressor  # type: ignore
    except Exception as exc:
        reason = f"scikit-learn is required for ML models: {exc}"
        return [
            SkippedModel(name, "machine_learning", reason)
            for name in [
                "Linear_Regression",
                "Polynomial_Regression",
                "Ridge",
                "Lasso",
                "Elastic_Net",
                "Decision_Trees",
                "Random_Forest",
                "Extra_Trees",
                "Gradient_Boosting",
                "AdaBoost",
                "XGBoost",
                "LightGBM",
                "CatBoost",
                "SVR",
                "KNN",
                "Gaussian_Process_Regression",
                "PLS",
                "PCR",
            ]
        ]

    poly = cfg("Polynomial_Regression", {"lags": 3, "degree": 2})
    ridge = cfg("Ridge", {"lags": 3, "alpha": 1.0})
    lasso = cfg("Lasso", {"lags": 3, "alpha": 0.001, "max_iter": 10_000})
    elastic = cfg("Elastic_Net", {"lags": 3, "alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10_000})
    tree = cfg("Decision_Trees", {"lags": 3, "max_depth": 4})
    rf = cfg("Random_Forest", {"lags": 3, "n_estimators": 200})
    extra = cfg("Extra_Trees", {"lags": 3, "n_estimators": 200})
    gb = cfg("Gradient_Boosting", {"lags": 3})
    ada = cfg("AdaBoost", {"lags": 3})
    svr = cfg("SVR", {"lags": 3, "C": 10.0, "epsilon": 0.01})
    knn = cfg("KNN", {"lags": 3, "n_neighbors": 3})
    gp = cfg("Gaussian_Process_Regression", {"lags": 3, "normalize_y": True})
    pls = cfg("PLS", {"lags": 3, "n_components": 1})
    pcr = cfg("PCR", {"lags": 3, "n_components": 1})

    models: list[ForecastModel] = [
        RecursiveSklearnModel("Linear_Regression", lambda: LinearRegression(), lags=lags("Linear_Regression"), estimator_params=cfg("Linear_Regression", {"lags": 3})),
        RecursiveSklearnModel("Polynomial_Regression", lambda: make_pipeline(PolynomialFeatures(int(poly["degree"])), LinearRegression()), lags=int(poly["lags"]), estimator_params=poly),
        RecursiveSklearnModel("Ridge", lambda: Ridge(alpha=float(ridge["alpha"]), random_state=random_state), lags=int(ridge["lags"]), estimator_params=ridge),
        RecursiveSklearnModel("Lasso", lambda: Lasso(alpha=float(lasso["alpha"]), random_state=random_state, max_iter=int(lasso["max_iter"])), lags=int(lasso["lags"]), estimator_params=lasso),
        RecursiveSklearnModel("Elastic_Net", lambda: ElasticNet(alpha=float(elastic["alpha"]), l1_ratio=float(elastic["l1_ratio"]), random_state=random_state, max_iter=int(elastic["max_iter"])), lags=int(elastic["lags"]), estimator_params=elastic),
        RecursiveSklearnModel("Decision_Trees", lambda: DecisionTreeRegressor(max_depth=int(tree["max_depth"]), random_state=random_state), lags=int(tree["lags"]), estimator_params=tree),
        RecursiveSklearnModel("Random_Forest", lambda: RandomForestRegressor(n_estimators=int(rf["n_estimators"]), random_state=random_state), lags=int(rf["lags"]), estimator_params=rf),
        RecursiveSklearnModel("Extra_Trees", lambda: ExtraTreesRegressor(n_estimators=int(extra["n_estimators"]), random_state=random_state), lags=int(extra["lags"]), estimator_params=extra),
        RecursiveSklearnModel("Gradient_Boosting", lambda: GradientBoostingRegressor(random_state=random_state), lags=int(gb["lags"]), estimator_params=gb),
        RecursiveSklearnModel("AdaBoost", lambda: AdaBoostRegressor(random_state=random_state), lags=int(ada["lags"]), estimator_params=ada),
        RecursiveSklearnModel("SVR", lambda: make_pipeline(StandardScaler(), SVR(C=float(svr["C"]), epsilon=float(svr["epsilon"]))), lags=int(svr["lags"]), estimator_params=svr),
        RecursiveSklearnModel("KNN", lambda: KNeighborsRegressor(n_neighbors=int(knn["n_neighbors"])), lags=int(knn["lags"]), estimator_params=knn),
        RecursiveSklearnModel("Gaussian_Process_Regression", lambda: GaussianProcessRegressor(random_state=random_state, normalize_y=bool(gp["normalize_y"])), lags=int(gp["lags"]), estimator_params=gp),
        RecursiveSklearnModel("PLS", lambda: PLSRegression(n_components=int(pls["n_components"])), lags=int(pls["lags"]), estimator_params=pls),
        RecursiveSklearnModel("PCR", lambda: make_pipeline(StandardScaler(), PCA(n_components=int(pcr["n_components"])), LinearRegression()), lags=int(pcr["lags"]), estimator_params=pcr),
    ]

    optional_specs = [
        ("XGBoost", "xgboost", "XGBRegressor", cfg("XGBoost", {"lags": 3, "n_estimators": 200, "objective": "reg:squarederror"})),
        ("LightGBM", "lightgbm", "LGBMRegressor", cfg("LightGBM", {"lags": 3, "n_estimators": 200, "verbose": -1})),
        ("CatBoost", "catboost", "CatBoostRegressor", cfg("CatBoost", {"lags": 3, "iterations": 200, "verbose": False})),
    ]
    for model_name, module_name, class_name, kwargs in optional_specs:
        try:
            module = __import__(module_name, fromlist=[class_name])
            estimator_cls = getattr(module, class_name)
            estimator_kwargs = dict(kwargs)
            model_lags = int(estimator_kwargs.pop("lags", 3))
            if model_name == "CatBoost":
                estimator_kwargs["random_seed"] = random_state
            else:
                estimator_kwargs["random_state"] = random_state
            models.append(RecursiveSklearnModel(model_name, lambda cls=estimator_cls, kw=estimator_kwargs: cls(**kw), lags=model_lags, estimator_params={**kwargs, **estimator_kwargs}))
        except Exception as exc:
            models.append(SkippedModel(model_name, "machine_learning", f"{module_name} not available: {exc}"))
    return models
