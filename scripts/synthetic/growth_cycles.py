from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticGrowthProfile:
    n_cycles: int
    min_points: int
    max_points: int
    baseline_low: float
    baseline_high: float
    carrying_low: float
    carrying_high: float
    noise_fraction: float
    decline_probability: float
    seasonality_probability: float
    random_state: int


class SyntheticGrowthCycleGenerator:
    """Generate complete lag-growth-stationary-decline microalgae cycles."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def generate(
        self,
        observed_values: pd.Series,
        observed_lengths: Iterable[int],
        target: str,
    ) -> pd.DataFrame:
        profile = self._profile(observed_values, observed_lengths)
        rng = np.random.default_rng(profile.random_state)
        rows: list[dict[str, Any]] = []

        for cycle_idx in range(profile.n_cycles):
            length = int(rng.integers(profile.min_points, profile.max_points + 1))
            values, params, phases = _simulate_cycle(length, profile, rng)
            for step, (value, phase) in enumerate(zip(values, phases)):
                rows.append(
                    {
                        "cycle_id": f"SYN-{cycle_idx + 1:05d}",
                        "target": target,
                        "step": step,
                        "relative_time": step / max(length - 1, 1),
                        "value": float(value),
                        "phase": phase,
                        "baseline": params["baseline"],
                        "carrying_capacity": params["carrying_capacity"],
                        "lag_fraction": params["lag_fraction"],
                        "decline_start": params["decline_start"],
                        "growth_rate": params["growth_rate"],
                    }
                )

        synthetic = pd.DataFrame(rows)
        synthetic.attrs["synthetic_id"] = f"{target}:{_series_signature(observed_values)}:{profile.random_state}:{profile.n_cycles}:{profile.min_points}-{profile.max_points}"
        return synthetic

    def _profile(self, observed_values: pd.Series, observed_lengths: Iterable[int]) -> SyntheticGrowthProfile:
        values = pd.to_numeric(observed_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values > 0]
        if values.empty:
            q05, q25, q50, q75, q95 = 0.05, 0.1, 0.4, 0.8, 1.0
        else:
            q05, q25, q50, q75, q95 = np.quantile(values.to_numpy(dtype=float), [0.05, 0.25, 0.5, 0.75, 0.95])

        spread = max(float(q95 - q05), float(values.std(ddof=1) if len(values) > 1 else 0.0), abs(float(q50)) * 0.15, 1e-3)
        lengths = [int(length) for length in observed_lengths if int(length) > 0]
        if lengths:
            default_min = max(6, int(np.quantile(lengths, 0.10)))
            default_max = max(default_min + 2, int(np.quantile(lengths, 0.95)))
        else:
            default_min, default_max = 8, 32

        min_points = int(_cfg(self.config, "synthetic_training.min_cycle_points", default_min))
        max_points = int(_cfg(self.config, "synthetic_training.max_cycle_points", default_max))
        min_points = max(5, min_points)
        max_points = max(min_points + 1, max_points)

        baseline_low = max(1e-6, float(_cfg(self.config, "synthetic_training.baseline_low", max(q05 - 0.35 * spread, q05 * 0.45))))
        baseline_high = max(baseline_low * 1.05, float(_cfg(self.config, "synthetic_training.baseline_high", max(q25, q50 - 0.15 * spread))))
        carrying_low = max(baseline_high * 1.05, float(_cfg(self.config, "synthetic_training.carrying_low", max(q75, q50 + 0.30 * spread))))
        carrying_high = max(carrying_low * 1.05, float(_cfg(self.config, "synthetic_training.carrying_high", q95 + 0.80 * spread)))

        return SyntheticGrowthProfile(
            n_cycles=max(1, int(_cfg(self.config, "synthetic_training.n_cycles", 2000))),
            min_points=min_points,
            max_points=max_points,
            baseline_low=baseline_low,
            baseline_high=baseline_high,
            carrying_low=carrying_low,
            carrying_high=carrying_high,
            noise_fraction=max(0.0, float(_cfg(self.config, "synthetic_training.noise_fraction", 0.035))),
            decline_probability=float(_cfg(self.config, "synthetic_training.decline_probability", 0.70)),
            seasonality_probability=float(_cfg(self.config, "synthetic_training.seasonality_probability", 0.45)),
            random_state=int(_cfg(self.config, "execution.random_state", 42)),
        )


def _simulate_cycle(
    length: int,
    profile: SyntheticGrowthProfile,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    t = np.linspace(0.0, 1.0, length)
    baseline = float(rng.uniform(profile.baseline_low, profile.baseline_high))
    carrying_capacity = float(rng.uniform(max(profile.carrying_low, baseline * 1.2), profile.carrying_high))
    amplitude = max(carrying_capacity - baseline, 1e-6)
    lag_fraction = float(rng.uniform(0.04, 0.28))
    midpoint = float(rng.uniform(lag_fraction + 0.10, min(0.72, lag_fraction + 0.42)))
    growth_rate = float(rng.uniform(7.0, 17.0))
    asymmetry = float(rng.uniform(0.75, 1.55))

    sigmoid = baseline + amplitude / np.power(1.0 + np.exp(-growth_rate * (t - midpoint)), asymmetry)
    lag_blend = np.clip((t - lag_fraction) / max(midpoint - lag_fraction, 1e-6), 0.0, 1.0)
    values = baseline * (1.0 - lag_blend) + sigmoid * lag_blend

    has_decline = rng.random() < profile.decline_probability
    decline_start = float(rng.uniform(0.66, 0.92))
    if has_decline:
        decline_strength = float(rng.uniform(0.04, 0.42)) * amplitude
        decline_power = float(rng.uniform(1.0, 2.4))
        decline_progress = np.clip((t - decline_start) / max(1.0 - decline_start, 1e-6), 0.0, 1.0)
        values = values - decline_strength * np.power(decline_progress, decline_power)

    if rng.random() < profile.seasonality_probability:
        season_amp = float(rng.uniform(0.01, 0.08)) * amplitude
        period = float(rng.uniform(0.22, 0.62))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        envelope = np.clip((t - lag_fraction) / max(1.0 - lag_fraction, 1e-6), 0.0, 1.0)
        values = values + season_amp * envelope * np.sin(2.0 * np.pi * t / period + phase)

    noise_sd = max(amplitude * profile.noise_fraction * float(rng.uniform(0.45, 1.55)), 1e-8)
    noise = np.zeros(length, dtype=float)
    for idx in range(length):
        innovation = float(rng.normal(0.0, noise_sd))
        noise[idx] = innovation if idx == 0 else 0.45 * noise[idx - 1] + innovation
    values = np.clip(values + noise, max(1e-8, baseline * 0.05), None)

    growth_low = max(midpoint + 0.08, 0.48)
    growth_high = 0.86
    if growth_low >= growth_high:
        growth_end_candidate = min(0.92, midpoint + 0.08)
    else:
        growth_end_candidate = float(rng.uniform(growth_low, growth_high))
    growth_end = min(decline_start, growth_end_candidate)
    phases: list[str] = []
    for item in t:
        if item < lag_fraction:
            phases.append("lag")
        elif item < growth_end:
            phases.append("exponential")
        elif has_decline and item >= decline_start:
            phases.append("decline")
        else:
            phases.append("stationary")

    params = {
        "baseline": baseline,
        "carrying_capacity": carrying_capacity,
        "lag_fraction": lag_fraction,
        "decline_start": decline_start if has_decline else 1.0,
        "growth_rate": growth_rate,
    }
    return values, params, phases


def _cfg(config: Any, dotted_key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(dotted_key, default)
    return default


def _series_signature(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return "empty"
    q05, q50, q95 = np.quantile(clean, [0.05, 0.50, 0.95])
    return f"n{len(clean)}_m{np.mean(clean):.6g}_s{np.std(clean):.6g}_q{q05:.6g}_{q50:.6g}_{q95:.6g}"
