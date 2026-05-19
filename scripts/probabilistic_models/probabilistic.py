from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.forecasting.base import ForecastModel, ModelSkipped, SkippedModel, clean_series, require_points


class MarkovChainForecaster(ForecastModel):
    name = "Markov_Chains"
    category = "probabilistic"
    min_points = 8

    def __init__(self, n_states: int = 3) -> None:
        super().__init__()
        self.n_states = n_states
        self.hyperparameters = {"n_states": n_states}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        quantiles = np.linspace(0, 1, self.n_states + 1)
        bins = np.unique(series.quantile(quantiles).to_numpy(dtype=float))
        if len(bins) < 3:
            raise ModelSkipped("Not enough variation to build states")
        states = np.digitize(series, bins[1:-1], right=True)
        transition = np.ones((len(bins) - 1, len(bins) - 1), dtype=float)
        for a, b in zip(states[:-1], states[1:]):
            transition[int(a), int(b)] += 1
        transition = transition / transition.sum(axis=1, keepdims=True)
        centers = np.asarray([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
        current = int(states[-1])
        preds = []
        for _step in range(horizon):
            probs = transition[current]
            preds.append(float(np.dot(probs, centers)))
            current = int(np.argmax(probs))
        self.fitted_model = {"bins": bins.tolist(), "transition": transition.tolist()}
        return np.asarray(preds, dtype=float)


class GaussianProcessForecaster(ForecastModel):
    name = "Gaussian_Processes"
    category = "probabilistic"
    min_points = 6

    def __init__(self, random_state: int = 42, normalize_y: bool = True) -> None:
        super().__init__()
        self.random_state = random_state
        self.normalize_y = normalize_y
        self.hyperparameters = {"random_state": random_state, "normalize_y": normalize_y, "kernel": "RBF + WhiteKernel"}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"scikit-learn is required for Gaussian Processes: {exc}") from exc
        x = np.arange(len(series), dtype=float).reshape(-1, 1)
        y = series.to_numpy(dtype=float)
        self.fitted_model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=self.random_state, normalize_y=self.normalize_y)
        self.fitted_model.fit(x, y)
        future = np.arange(len(series), len(series) + horizon, dtype=float).reshape(-1, 1)
        return np.asarray(self.fitted_model.predict(future), dtype=float)


class MonteCarloDrift(ForecastModel):
    name = "Monte_Carlo"
    category = "probabilistic"
    min_points = 5

    def __init__(self, simulations: int = 500, random_state: int = 42) -> None:
        super().__init__()
        self.simulations = simulations
        self.random_state = random_state
        self.hyperparameters = {"simulations": simulations, "random_state": random_state}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        rng = np.random.default_rng(self.random_state)
        increments = series.diff().dropna().to_numpy(dtype=float)
        if len(increments) == 0:
            return np.repeat(float(series.iloc[-1]), horizon)
        paths = np.zeros((self.simulations, horizon), dtype=float)
        for sim in range(self.simulations):
            shocks = rng.choice(increments, size=horizon, replace=True)
            paths[sim] = float(series.iloc[-1]) + np.cumsum(shocks)
        self.fitted_model = {"simulations": self.simulations}
        return np.median(paths, axis=0)


class ParticleFilterForecaster(ForecastModel):
    name = "Particle_Filters"
    category = "probabilistic"
    min_points = 5

    def __init__(self, particles: int = 500, random_state: int = 42) -> None:
        super().__init__()
        self.particles = particles
        self.random_state = random_state
        self.hyperparameters = {"particles": particles, "random_state": random_state}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        rng = np.random.default_rng(self.random_state)
        diff = series.diff().dropna()
        drift = float(diff.mean()) if not diff.empty else 0.0
        noise = float(diff.std(ddof=1)) if len(diff) > 1 else max(abs(drift), 1e-3)
        particles = rng.normal(float(series.iloc[-1]), noise, size=self.particles)
        preds = []
        for _ in range(horizon):
            particles = particles + rng.normal(drift, max(noise, 1e-6), size=self.particles)
            preds.append(float(np.mean(particles)))
        self.fitted_model = {"particles": self.particles, "drift": drift, "noise": noise}
        return np.asarray(preds, dtype=float)


class HMMForecaster(ForecastModel):
    name = "Hidden_Markov_Models"
    category = "probabilistic"
    min_points = 10

    def __init__(self, n_components: int = 3, n_iter: int = 200, random_state: int = 42) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.hyperparameters = {"n_components": n_components, "n_iter": n_iter, "covariance_type": "diag", "random_state": random_state}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, self.min_points, self.name)
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"hmmlearn is required for HMM: {exc}") from exc
        x = series.to_numpy(dtype=float).reshape(-1, 1)
        self.fitted_model = GaussianHMM(n_components=min(self.n_components, len(series) // 3), covariance_type="diag", n_iter=self.n_iter, random_state=self.random_state)
        self.fitted_model.fit(x)
        states = self.fitted_model.predict(x)
        current_state = int(states[-1])
        means = self.fitted_model.means_.ravel()
        trans = self.fitted_model.transmat_
        preds = []
        for _ in range(horizon):
            probs = trans[current_state]
            preds.append(float(np.dot(probs, means)))
            current_state = int(np.argmax(probs))
        return np.asarray(preds, dtype=float)


def get_probabilistic_models(params: dict[str, object] | None = None, random_state: int = 42) -> list[ForecastModel]:
    params = params or {}

    def cfg(model_name: str, defaults: dict[str, object]) -> dict[str, object]:
        merged = defaults.copy()
        merged.update(dict(params.get(model_name, {}) or {}))
        return merged

    hmm = cfg("Hidden_Markov_Models", {"n_components": 3, "n_iter": 200, "random_state": random_state})
    markov = cfg("Markov_Chains", {"n_states": 3})
    gp = cfg("Gaussian_Processes", {"random_state": random_state, "normalize_y": True})
    pf = cfg("Particle_Filters", {"particles": 500, "random_state": random_state})
    mc = cfg("Monte_Carlo", {"simulations": 500, "random_state": random_state})
    smc_params = cfg("Sequential_Monte_Carlo", {"particles": 500, "random_state": random_state})
    smc = ParticleFilterForecaster(particles=int(smc_params["particles"]), random_state=int(smc_params["random_state"]))
    smc.name = "Sequential_Monte_Carlo"
    return [
        HMMForecaster(n_components=int(hmm["n_components"]), n_iter=int(hmm["n_iter"]), random_state=int(hmm["random_state"])),
        MarkovChainForecaster(n_states=int(markov["n_states"])),
        SkippedModel("Semi_Markov", "probabilistic", "Semi-Markov requires explicit state dwell-time labels."),
        GaussianProcessForecaster(random_state=int(gp["random_state"]), normalize_y=bool(gp["normalize_y"])),
        SkippedModel("Bayesian_Networks", "probabilistic", "Bayesian network structure learning is available through pgmpy/pymc when installed."),
        ParticleFilterForecaster(particles=int(pf["particles"]), random_state=int(pf["random_state"])),
        MonteCarloDrift(simulations=int(mc["simulations"]), random_state=int(mc["random_state"])),
        smc,
        SkippedModel("Probabilistic_Graphical_Models", "probabilistic", "Probabilistic graphical models require structural assumptions or graph data."),
    ]
