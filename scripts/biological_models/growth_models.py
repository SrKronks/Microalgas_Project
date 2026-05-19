from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from scripts.forecasting.base import ForecastModel, ModelSkipped, clean_series, require_points


ArrayFunc = Callable[..., np.ndarray]


class CurveFitGrowthModel(ForecastModel):
    category = "biological"
    min_points = 6

    def __init__(
        self,
        name: str,
        func: ArrayFunc,
        p0: Callable[[np.ndarray, np.ndarray], list[float]],
        bounds: tuple[list[float], list[float]] | None = None,
        note: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.func = func
        self.p0_factory = p0
        self.bounds = bounds
        self.note = note

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from scipy.optimize import curve_fit  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"scipy is required for biological model {self.name}: {exc}") from exc
        y = np.asarray(series, dtype=float)
        x = np.arange(len(y), dtype=float)
        p0 = self.p0_factory(x, y)
        kwargs = {"maxfev": 20_000}
        if self.bounds is not None:
            kwargs["bounds"] = self.bounds
        try:
            params, covariance = curve_fit(self.func, x, y, p0=p0, **kwargs)
        except Exception as exc:
            raise ModelSkipped(f"curve_fit failed: {exc}") from exc
        future_x = np.arange(len(y), len(y) + horizon, dtype=float)
        pred = np.asarray(self.func(future_x, *params), dtype=float)
        self.fitted_model = {"params": params.tolist(), "covariance": np.asarray(covariance).tolist()}
        self.metadata = {"params": params.tolist(), "note": self.note}
        self.n_params = len(params)
        return pred


def _scale(y: np.ndarray) -> tuple[float, float, float]:
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    span = max(ymax - ymin, 1e-6)
    return ymin, ymax, span


def exponential(t: np.ndarray, y0: float, mu: float) -> np.ndarray:
    return y0 * np.exp(mu * t)


def logistic(t: np.ndarray, k: float, a: float, r: float) -> np.ndarray:
    return k / (1 + a * np.exp(-r * t))


def gompertz(t: np.ndarray, k: float, a: float, r: float) -> np.ndarray:
    return k * np.exp(-a * np.exp(-r * t))


def richards(t: np.ndarray, k: float, a: float, r: float, nu: float) -> np.ndarray:
    return k / np.power(1 + a * np.exp(-r * t), 1 / np.maximum(nu, 1e-6))


def modified_gompertz(t: np.ndarray, a: float, mu: float, lag: float, baseline: float) -> np.ndarray:
    return baseline + a * np.exp(-np.exp((mu * np.e / np.maximum(a, 1e-6)) * (lag - t) + 1))


def bertalanffy(t: np.ndarray, k: float, b: float, r: float) -> np.ndarray:
    return k * np.power(np.maximum(1 - b * np.exp(-r * t), 1e-6), 3)


def monod_proxy(t: np.ndarray, y0: float, mu_max: float, ks: float) -> np.ndarray:
    substrate = t + 1
    return y0 + mu_max * substrate / (ks + substrate)


def haldane_proxy(t: np.ndarray, y0: float, mu_max: float, ks: float, ki: float) -> np.ndarray:
    substrate = t + 1
    return y0 + mu_max * substrate / (ks + substrate + substrate**2 / np.maximum(ki, 1e-6))


def tessier_proxy(t: np.ndarray, y0: float, mu_max: float, k: float) -> np.ndarray:
    substrate = t + 1
    return y0 + mu_max * (1 - np.exp(-substrate / np.maximum(k, 1e-6)))


def moser_proxy(t: np.ndarray, y0: float, mu_max: float, ks: float, n: float) -> np.ndarray:
    substrate = t + 1
    return y0 + mu_max * substrate**n / (ks + substrate**n)


def light_inhibition_proxy(t: np.ndarray, y0: float, alpha: float, beta: float) -> np.ndarray:
    light = t + 1
    return y0 + alpha * light * np.exp(-beta * light)


def mortality_proxy(t: np.ndarray, k: float, a: float, r: float, d: float) -> np.ndarray:
    return logistic(t, k, a, r) * np.exp(-d * t)


def lotka_volterra_proxy(t: np.ndarray, y0: float, r: float, interaction: float) -> np.ndarray:
    return y0 * np.exp((r - interaction * t / (t.max() + 1 if len(t) else 1)) * t)


def _p0_exp(_: np.ndarray, y: np.ndarray) -> list[float]:
    return [max(float(y[0]), 1e-6), 0.05]


def _p0_sigmoid(_: np.ndarray, y: np.ndarray) -> list[float]:
    _, ymax, span = _scale(y)
    return [max(ymax * 1.3, 1e-6), max(span, 1e-3), 0.1]


def _p0_richards(_: np.ndarray, y: np.ndarray) -> list[float]:
    return [max(float(np.nanmax(y)) * 1.3, 1e-6), 1.0, 0.1, 1.0]


def _p0_gompertz(_: np.ndarray, y: np.ndarray) -> list[float]:
    _, ymax, span = _scale(y)
    return [max(ymax * 1.3, 1e-6), max(span, 1e-3), 0.1]


def _p0_modified(_: np.ndarray, y: np.ndarray) -> list[float]:
    ymin, _, span = _scale(y)
    return [max(span, 1e-6), 0.1, 1.0, ymin]


def _p0_proxy(_: np.ndarray, y: np.ndarray) -> list[float]:
    return [max(float(y[0]), 1e-6), max(float(np.nanmax(y) - np.nanmin(y)), 1e-3), 1.0]


def _p0_haldane(_: np.ndarray, y: np.ndarray) -> list[float]:
    return [max(float(y[0]), 1e-6), max(float(np.nanmax(y) - np.nanmin(y)), 1e-3), 1.0, 10.0]


def _p0_moser(_: np.ndarray, y: np.ndarray) -> list[float]:
    return [max(float(y[0]), 1e-6), max(float(np.nanmax(y) - np.nanmin(y)), 1e-3), 1.0, 1.0]


def get_biological_models() -> list[ForecastModel]:
    proxy_note = "Proxy form: substrate/light/metabolic variables were not present, so elapsed time is used as proxy."
    return [
        CurveFitGrowthModel("Exponential", exponential, _p0_exp),
        CurveFitGrowthModel("Logistic", logistic, _p0_sigmoid),
        CurveFitGrowthModel("Gompertz", gompertz, _p0_gompertz),
        CurveFitGrowthModel("Richards", richards, _p0_richards),
        CurveFitGrowthModel("Verhulst", logistic, _p0_sigmoid),
        CurveFitGrowthModel("Baranyi", modified_gompertz, _p0_modified),
        CurveFitGrowthModel("Bertalanffy", bertalanffy, _p0_sigmoid),
        CurveFitGrowthModel("Monod", monod_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Droop", monod_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Contois", monod_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Haldane", haldane_proxy, _p0_haldane, note=proxy_note),
        CurveFitGrowthModel("Andrews", haldane_proxy, _p0_haldane, note=proxy_note),
        CurveFitGrowthModel("Tessier", tessier_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Moser", moser_proxy, _p0_moser, note=proxy_note),
        CurveFitGrowthModel("Eilers_Peeters", light_inhibition_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Steele", light_inhibition_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Photo_Limited", monod_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Photo_Inhibited", light_inhibition_proxy, _p0_proxy, note=proxy_note),
        CurveFitGrowthModel("Cell_Mortality", mortality_proxy, lambda x, y: [*_p0_sigmoid(x, y), 0.01]),
        CurveFitGrowthModel("Lotka_Volterra", lotka_volterra_proxy, lambda _x, y: [max(float(y[0]), 1e-6), 0.05, 0.01]),
        CurveFitGrowthModel("Bioenergetic_Dynamic", gompertz, _p0_gompertz, note=proxy_note),
        CurveFitGrowthModel("Dynamic_Metabolic", richards, _p0_richards, note=proxy_note),
    ]
