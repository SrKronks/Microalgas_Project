from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.biological_models.growth_models import gompertz, _p0_gompertz
from scripts.forecasting.base import ForecastModel, ModelSkipped, SkippedModel, clean_series, require_points
from scripts.machine_learning.ml_models import supervised_lags


class ARIMAResidualForest(ForecastModel):
    name = "ARIMA_XGBoost"
    category = "hybrid"
    min_points = 12

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"statsmodels and scikit-learn are required for {self.name}: {exc}") from exc
        arima = ARIMA(series, order=(1, 1, 1)).fit()
        fitted = pd.Series(arima.fittedvalues, index=series.index).reindex(series.index).fillna(method="bfill")
        residuals = series - fitted
        x, y = supervised_lags(residuals, lags=3)
        if len(x) < 4:
            raise ModelSkipped("Not enough residual lag samples")
        rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(x, y)
        base = np.asarray(arima.forecast(horizon), dtype=float)
        history = list(residuals.to_numpy(dtype=float))
        residual_pred = []
        for _step in range(horizon):
            pred = float(rf.predict(np.asarray(history[-3:][::-1]).reshape(1, -1))[0])
            residual_pred.append(pred)
            history.append(pred)
        self.fitted_model = {"arima": arima, "residual_model": rf}
        return base + np.asarray(residual_pred, dtype=float)


class GompertzResidualForest(ForecastModel):
    name = "Gompertz_Random_Forest"
    category = "hybrid"
    min_points = 8

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from scipy.optimize import curve_fit  # type: ignore
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"scipy and scikit-learn are required for {self.name}: {exc}") from exc
        y = series.to_numpy(dtype=float)
        x_time = np.arange(len(y), dtype=float)
        params, _ = curve_fit(gompertz, x_time, y, p0=_p0_gompertz(x_time, y), maxfev=20_000)
        fitted = gompertz(x_time, *params)
        residuals = y - fitted
        x = x_time.reshape(-1, 1)
        rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(x, residuals)
        future = np.arange(len(y), len(y) + horizon, dtype=float)
        self.fitted_model = {"params": params.tolist(), "residual_model": rf}
        return gompertz(future, *params) + rf.predict(future.reshape(-1, 1))


class ODELinearResidual(ForecastModel):
    name = "ODE_ML"
    category = "hybrid"
    min_points = 8

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from scipy.optimize import curve_fit  # type: ignore
            from sklearn.linear_model import Ridge  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"scipy and scikit-learn are required for {self.name}: {exc}") from exc
        y = series.to_numpy(dtype=float)
        t = np.arange(len(y), dtype=float)

        def logistic(t_: np.ndarray, k: float, a: float, r: float) -> np.ndarray:
            return k / (1 + a * np.exp(-r * t_))

        params, _ = curve_fit(logistic, t, y, p0=[max(y) * 1.2, 1.0, 0.1], maxfev=20_000)
        residuals = y - logistic(t, *params)
        model = Ridge(alpha=1.0).fit(t.reshape(-1, 1), residuals)
        future = np.arange(len(y), len(y) + horizon, dtype=float)
        self.fitted_model = {"params": params.tolist(), "residual_model": model}
        return logistic(future, *params) + model.predict(future.reshape(-1, 1))


def get_hybrid_models() -> list[ForecastModel]:
    return [
        ARIMAResidualForest(),
        SkippedModel("SARIMA_LSTM", "hybrid", "Requires statsmodels plus tensorflow sequence training."),
        ODELinearResidual(),
        GompertzResidualForest(),
        SkippedModel("Monod_Neural_Networks", "hybrid", "Requires substrate covariates and neural-network backend."),
        SkippedModel("Kalman_LSTM", "hybrid", "Requires statsmodels state-space model plus tensorflow."),
        SkippedModel("PINNs", "hybrid", "Requires physics-informed loss specification and torch/tensorflow backend."),
        SkippedModel("Neural_ODEs", "hybrid", "Requires torchdiffeq or equivalent neural ODE backend."),
        SkippedModel("Digital_Twin_Hybrid", "hybrid", "Requires process-control inputs and calibrated mechanistic model."),
    ]
