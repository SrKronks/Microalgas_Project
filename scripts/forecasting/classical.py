from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.forecasting.base import ForecastModel, ModelSkipped, clean_series, require_points


class NaiveLast(ForecastModel):
    name = "Naive_Last"
    category = "classical"
    min_points = 2

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        require_points(train, self.min_points, self.name)
        value = float(clean_series(train).iloc[-1])
        return np.repeat(value, horizon)


class MovingAverage(ForecastModel):
    name = "Moving_Average"
    category = "classical"
    min_points = 3

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = window

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        require_points(train, self.min_points, self.name)
        value = float(clean_series(train).tail(self.window).mean())
        self.metadata = {"window": self.window}
        return np.repeat(value, horizon)


class WeightedMovingAverage(ForecastModel):
    name = "Weighted_Moving_Average"
    category = "classical"
    min_points = 3

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = window

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        values = series.tail(self.window).to_numpy(dtype=float)
        weights = np.arange(1, len(values) + 1, dtype=float)
        value = float(np.average(values, weights=weights))
        self.metadata = {"window": self.window}
        return np.repeat(value, horizon)


class DriftMethod(ForecastModel):
    name = "Drift"
    category = "classical"
    min_points = 3

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        slope = (series.iloc[-1] - series.iloc[0]) / max(len(series) - 1, 1)
        return np.asarray([series.iloc[-1] + slope * step for step in range(1, horizon + 1)], dtype=float)


class Croston(ForecastModel):
    name = "Croston"
    category = "classical"
    min_points = 5

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        values = np.maximum(series.to_numpy(dtype=float), 0)
        demand = values[values > 0]
        if len(demand) == 0:
            return np.zeros(horizon)
        z = demand[0]
        p = 1.0
        interval = 1.0
        for value in values[1:]:
            if value > 0:
                z = self.alpha * value + (1 - self.alpha) * z
                p = self.alpha * interval + (1 - self.alpha) * p
                interval = 1.0
            else:
                interval += 1
        forecast = z / max(p, 1e-12)
        self.metadata = {"alpha": self.alpha}
        return np.repeat(float(forecast), horizon)


class StatsmodelsETS(ForecastModel):
    category = "classical"
    min_points = 5

    def __init__(self, name: str, mode: str, seasonal_periods: int = 3) -> None:
        super().__init__()
        self.name = name
        self.mode = mode
        self.seasonal_periods = seasonal_periods

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing  # type: ignore
            from statsmodels.tsa.forecasting.theta import ThetaModel  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"statsmodels is required for {self.name}: {exc}") from exc

        if self.mode == "simple":
            model = SimpleExpSmoothing(series, initialization_method="estimated")
        elif self.mode == "holt":
            model = Holt(series, initialization_method="estimated")
        elif self.mode == "brown":
            model = Holt(series, exponential=False, damped_trend=False, initialization_method="estimated")
        elif self.mode == "hw_add":
            if len(series) < self.seasonal_periods * 2:
                raise ModelSkipped("Holt-Winters additive requires at least two seasonal cycles")
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
        elif self.mode == "hw_mul":
            if (series <= 0).any():
                raise ModelSkipped("Holt-Winters multiplicative requires positive values")
            if len(series) < self.seasonal_periods * 2:
                raise ModelSkipped("Holt-Winters multiplicative requires at least two seasonal cycles")
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="mul",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
        elif self.mode == "theta":
            self.fitted_model = ThetaModel(series, period=max(2, self.seasonal_periods)).fit()
            return np.asarray(self.fitted_model.forecast(horizon), dtype=float)
        else:
            raise ModelSkipped(f"Unknown ETS mode {self.mode}")
        self.fitted_model = model.fit(optimized=True)
        return np.asarray(self.fitted_model.forecast(horizon), dtype=float)


def get_classical_models(seasonal_periods: int = 3) -> list[ForecastModel]:
    return [
        NaiveLast(),
        MovingAverage(window=3),
        WeightedMovingAverage(window=3),
        DriftMethod(),
        StatsmodelsETS("Exponential_Smoothing", "simple", seasonal_periods),
        StatsmodelsETS("Holt", "holt", seasonal_periods),
        StatsmodelsETS("Holt_Winters_Additive", "hw_add", seasonal_periods),
        StatsmodelsETS("Holt_Winters_Multiplicative", "hw_mul", seasonal_periods),
        StatsmodelsETS("Brown_Double_Exponential", "brown", seasonal_periods),
        StatsmodelsETS("Winters", "hw_add", seasonal_periods),
        Croston(),
        StatsmodelsETS("Theta_Method", "theta", seasonal_periods),
    ]
