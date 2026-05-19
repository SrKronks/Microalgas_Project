from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.forecasting.base import ForecastModel, ModelSkipped, clean_series, require_points


class ARIMAFamily(ForecastModel):
    category = "statistical"
    min_points = 8

    def __init__(
        self,
        name: str,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] | None = None,
        use_exog: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.n_params = sum(order) + 1
        self.hyperparameters = {"order": order, "seasonal_order": seasonal_order, "use_exog": use_exog}

    def fit_predict(
        self,
        train: pd.Series,
        horizon: int,
        frame_train: pd.DataFrame | None = None,
        frame_test: pd.DataFrame | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"statsmodels is required for {self.name}: {exc}") from exc

        exog_train = None
        exog_test = None
        if self.use_exog and frame_train is not None and frame_test is not None and target is not None:
            numeric = frame_train.select_dtypes(include="number").drop(columns=[target], errors="ignore")
            numeric_test = frame_test[numeric.columns] if set(numeric.columns).issubset(frame_test.columns) else None
            if numeric.shape[1] > 0 and numeric_test is not None:
                exog_train = numeric.fillna(method="ffill").fillna(method="bfill").fillna(0)
                exog_test = numeric_test.fillna(method="ffill").fillna(method="bfill").fillna(0)
        if self.use_exog and exog_train is None:
            raise ModelSkipped("No exogenous numeric variables available")

        kwargs = {"order": self.order, "enforce_stationarity": False, "enforce_invertibility": False}
        if self.seasonal_order is not None:
            kwargs["seasonal_order"] = self.seasonal_order
        self.fitted_model = SARIMAX(series, exog=exog_train, **kwargs).fit(disp=False)
        pred = self.fitted_model.forecast(steps=horizon, exog=exog_test)
        self.metadata = {"order": self.order, "seasonal_order": self.seasonal_order, "use_exog": self.use_exog}
        return np.asarray(pred, dtype=float)


class VARFamily(ForecastModel):
    category = "statistical"
    min_points = 8

    def __init__(self, name: str, mode: str = "var", maxlags: int = 2, varmax_order: tuple[int, int] = (1, 0), maxiter: int = 100) -> None:
        super().__init__()
        self.name = name
        self.mode = mode
        self.maxlags = maxlags
        self.varmax_order = varmax_order
        self.maxiter = maxiter
        self.hyperparameters = {"mode": mode, "maxlags": maxlags, "varmax_order": varmax_order, "maxiter": maxiter}

    def fit_predict(
        self,
        train: pd.Series,
        horizon: int,
        frame_train: pd.DataFrame | None = None,
        frame_test: pd.DataFrame | None = None,
        target: str | None = None,
    ) -> np.ndarray:
        if frame_train is None or target is None:
            raise ModelSkipped("VAR family requires a multivariate frame")
        require_points(train, self.min_points, self.name)
        numeric = frame_train.select_dtypes(include="number").dropna(axis=1, how="all")
        numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1].dropna()
        if target not in numeric or numeric.shape[1] < 2 or len(numeric) < self.min_points:
            raise ModelSkipped("VAR family requires at least two complete numeric variables")
        try:
            from statsmodels.tsa.api import VAR  # type: ignore
            from statsmodels.tsa.statespace.varmax import VARMAX  # type: ignore
            from statsmodels.tsa.vector_ar.vecm import VECM  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"statsmodels is required for {self.name}: {exc}") from exc

        if self.mode == "var":
            self.fitted_model = VAR(numeric).fit(maxlags=min(self.maxlags, len(numeric) // 3), ic=None)
            forecast = self.fitted_model.forecast(numeric.values[-self.fitted_model.k_ar :], steps=horizon)
            return np.asarray(pd.DataFrame(forecast, columns=numeric.columns)[target], dtype=float)
        if self.mode == "varmax":
            self.fitted_model = VARMAX(numeric, order=self.varmax_order, enforce_stationarity=False).fit(disp=False, maxiter=self.maxiter)
            return np.asarray(self.fitted_model.forecast(horizon)[target], dtype=float)
        if self.mode == "vecm":
            self.fitted_model = VECM(numeric, k_ar_diff=1, coint_rank=1).fit()
            return np.asarray(pd.DataFrame(self.fitted_model.predict(steps=horizon), columns=numeric.columns)[target], dtype=float)
        raise ModelSkipped(f"Unknown VAR mode {self.mode}")


class VolatilityModel(ForecastModel):
    category = "statistical"
    min_points = 10

    def __init__(self, name: str, vol: str, p: int = 1, q: int | None = None) -> None:
        super().__init__()
        self.name = name
        self.vol = vol
        self.p = p
        self.q = 1 if q is None and vol.upper() == "GARCH" else int(q or 0)
        self.hyperparameters = {"vol": vol, "p": self.p, "q": self.q}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from arch import arch_model  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"arch package is required for {self.name}: {exc}") from exc
        returns = series.diff().dropna()
        if len(returns) < self.min_points:
            raise ModelSkipped("ARCH/GARCH requires enough differenced observations")
        self.fitted_model = arch_model(returns, vol=self.vol, p=self.p, q=self.q).fit(disp="off")
        mean_step = float(returns.mean())
        start = float(series.iloc[-1])
        return np.asarray([start + mean_step * step for step in range(1, horizon + 1)], dtype=float)


class StateSpaceModel(ForecastModel):
    category = "statistical"
    min_points = 8

    def __init__(self, name: str, level: str = "local linear trend") -> None:
        super().__init__()
        self.name = name
        self.level = level
        self.hyperparameters = {"level": level}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"statsmodels is required for {self.name}: {exc}") from exc
        self.fitted_model = UnobservedComponents(series, level=self.level).fit(disp=False)
        return np.asarray(self.fitted_model.forecast(horizon), dtype=float)


def get_statistical_models(seasonal_periods: int = 3, params: dict[str, object] | None = None) -> list[ForecastModel]:
    params = params or {}

    def cfg(model_name: str, defaults: dict[str, object]) -> dict[str, object]:
        merged = defaults.copy()
        merged.update(dict(params.get(model_name, {}) or {}))
        return merged

    var = cfg("VAR", {"maxlags": 2})
    varmax = cfg("VARMAX", {"varmax_order": (1, 0), "maxiter": 100})
    arch = cfg("ARCH", {"p": 1, "q": 0})
    garch = cfg("GARCH", {"p": 1, "q": 1})
    state = cfg("State_Space_Model", {"level": "local linear trend"})
    return [
        ARIMAFamily("AR", (1, 0, 0)),
        ARIMAFamily("MA", (0, 0, 1)),
        ARIMAFamily("ARMA", (1, 0, 1)),
        ARIMAFamily("ARIMA", (1, 1, 1)),
        ARIMAFamily("SARIMA", (1, 1, 1), (1, 0, 0, max(2, seasonal_periods))),
        ARIMAFamily("ARIMAX", (1, 1, 1), None, use_exog=True),
        ARIMAFamily("SARIMAX", (1, 1, 1), (1, 0, 0, max(2, seasonal_periods)), use_exog=True),
        VARFamily("VAR", "var", maxlags=int(var["maxlags"])),
        VARFamily("VARMAX", "varmax", varmax_order=tuple(varmax["varmax_order"]), maxiter=int(varmax["maxiter"])),
        VARFamily("VECM", "vecm"),
        VolatilityModel("ARCH", "ARCH", p=int(arch["p"]), q=int(arch["q"])),
        VolatilityModel("GARCH", "GARCH", p=int(garch["p"]), q=int(garch["q"])),
        StateSpaceModel("State_Space_Model", level=str(state["level"])),
        StateSpaceModel("Dynamic_Linear_Model", level=str(cfg("Dynamic_Linear_Model", {"level": "local linear trend"})["level"])),
        StateSpaceModel("Kalman_Filter", level=str(cfg("Kalman_Filter", {"level": "local linear trend"})["level"])),
        ARIMAFamily("BSTS_proxy", (1, 1, 1)),
    ]
