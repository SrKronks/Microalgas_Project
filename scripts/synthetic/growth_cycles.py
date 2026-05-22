from __future__ import annotations

import hashlib
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
    sampling_interval_days: float
    ph_low: float
    ph_high: float
    ec_low: float
    ec_high: float
    temperature_center: float
    temperature_sd: float
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
        group: str | None = None,
        observed_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        profile = self._profile(observed_values, observed_lengths, observed_frame)
        simulation_seed = _stable_seed(profile.random_state, group, target)
        rng = np.random.default_rng(simulation_seed)
        rows: list[dict[str, Any]] = []

        for cycle_idx in range(profile.n_cycles):
            length = int(rng.integers(profile.min_points, profile.max_points + 1))
            values, params, phases = _simulate_cycle(length, profile, rng)
            covariates = _simulate_covariates(values, phases, profile, rng)
            deltas = np.r_[np.nan, np.diff(values)]
            specific_growth = _specific_growth(values, profile.sampling_interval_days)
            normalized = (values - np.min(values)) / max(float(np.max(values) - np.min(values)), 1e-9)
            for step, (value, phase) in enumerate(zip(values, phases)):
                state_label = _state_label(phase, normalized[step])
                rows.append(
                    {
                        "BIM": group if group is not None else "SYNTHETIC",
                        "cycle_id": f"SYN-{cycle_idx + 1:05d}",
                        "target": target,
                        "step": step,
                        "cycle_age_days": float(step * profile.sampling_interval_days),
                        "relative_time": step / max(length - 1, 1),
                        "value": float(value),
                        f"{target}_synthetic": float(value),
                        "normalized_value": float(normalized[step]),
                        "delta_value": float(deltas[step]) if np.isfinite(deltas[step]) else np.nan,
                        "specific_growth_rate": float(specific_growth[step]) if np.isfinite(specific_growth[step]) else np.nan,
                        "phase": phase,
                        "phase_code": _phase_code(phase),
                        "synthetic_ritmo": _ritmo_label(phase, specific_growth[step]),
                        "synthetic_estado_cultivo": state_label,
                        "is_optimal_phase": bool(state_label in {"CRECIMIENTO", "PRODUCTO"}),
                        "optimality_score": float(_optimality_score(phase, normalized[step], specific_growth[step])),
                        "pH": float(covariates["pH"][step]),
                        "EC": float(covariates["EC"][step]),
                        "Temperatura (°C)": float(covariates["temperature"][step]),
                        "baseline": params["baseline"],
                        "carrying_capacity": params["carrying_capacity"],
                        "lag_fraction": params["lag_fraction"],
                        "decline_start": params["decline_start"],
                        "growth_rate": params["growth_rate"],
                    }
                )

        synthetic = pd.DataFrame(rows)
        synthetic.attrs["synthetic_id"] = (
            f"{group or 'ALL'}:{target}:{_series_signature(observed_values)}:"
            f"{simulation_seed}:{profile.n_cycles}:{profile.min_points}-{profile.max_points}"
        )
        return synthetic

    def _profile(
        self,
        observed_values: pd.Series,
        observed_lengths: Iterable[int],
        observed_frame: pd.DataFrame | None = None,
    ) -> SyntheticGrowthProfile:
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

        min_points = int(_cfg(self.config, "synthetic_training.min_cycle_points", max(default_min, 48)))
        max_points = int(_cfg(self.config, "synthetic_training.max_cycle_points", max(default_max, 96)))
        min_points = max(5, min_points)
        max_points = max(min_points + 1, max_points)
        frame = observed_frame if observed_frame is not None else pd.DataFrame()
        ph_low, ph_high = _observed_bounds(frame, ["pH", "ph"], 6.8, 8.8)
        ec_low, ec_high = _observed_bounds(frame, ["EC", "conductividad", "conductivity"], 0.6, 3.2)
        temp_low, temp_high = _observed_bounds(frame, ["Temperatura", "Temp", "temperature"], 18.0, 28.0)

        baseline_low = max(1e-6, float(_cfg(self.config, "synthetic_training.baseline_low", max(q05 - 0.35 * spread, q05 * 0.45))))
        baseline_high = max(baseline_low * 1.05, float(_cfg(self.config, "synthetic_training.baseline_high", max(q25, q50 - 0.15 * spread))))
        carrying_low = max(baseline_high * 1.05, float(_cfg(self.config, "synthetic_training.carrying_low", max(q75, q50 + 0.30 * spread))))
        carrying_high = max(carrying_low * 1.05, float(_cfg(self.config, "synthetic_training.carrying_high", q95 + 0.80 * spread)))
        carrying_high = max(carrying_high, baseline_high * 1.2 * 1.05)

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
            sampling_interval_days=max(0.05, float(_cfg(self.config, "synthetic_training.sampling_interval_days", 0.25))),
            ph_low=ph_low,
            ph_high=ph_high,
            ec_low=ec_low,
            ec_high=ec_high,
            temperature_center=float((temp_low + temp_high) / 2.0),
            temperature_sd=max(0.15, float((temp_high - temp_low) / 5.0)),
            random_state=int(_cfg(self.config, "execution.random_state", 42)),
        )


def _simulate_cycle(
    length: int,
    profile: SyntheticGrowthProfile,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    t = np.linspace(0.0, 1.0, length)
    baseline = float(rng.uniform(profile.baseline_low, profile.baseline_high))
    carrying_floor = max(profile.carrying_low, baseline * 1.2)
    carrying_ceiling = max(profile.carrying_high, carrying_floor * 1.05)
    carrying_capacity = float(rng.uniform(carrying_floor, carrying_ceiling))
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


def _simulate_covariates(
    values: np.ndarray,
    phases: list[str],
    profile: SyntheticGrowthProfile,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    normalized = (values - np.min(values)) / max(float(np.max(values) - np.min(values)), 1e-9)
    phase_arr = np.asarray([_phase_code(phase) for phase in phases], dtype=float)
    phase_center = phase_arr / max(float(phase_arr.max()), 1.0)

    ph_span = max(profile.ph_high - profile.ph_low, 0.1)
    ph = profile.ph_low + ph_span * (0.25 + 0.55 * normalized - 0.18 * (phase_arr == _phase_code("decline")))
    ph += rng.normal(0.0, max(0.015, ph_span * 0.025), size=len(values))
    ph = np.clip(ph, profile.ph_low - 0.15 * ph_span, profile.ph_high + 0.15 * ph_span)

    ec_span = max(profile.ec_high - profile.ec_low, 0.1)
    ec = profile.ec_low + ec_span * (0.35 + 0.35 * normalized + 0.12 * phase_center)
    ec += rng.normal(0.0, max(0.01, ec_span * 0.025), size=len(values))
    ec = np.clip(ec, profile.ec_low - 0.10 * ec_span, profile.ec_high + 0.10 * ec_span)

    t = np.linspace(0.0, 2.0 * np.pi, len(values))
    temperature = profile.temperature_center + profile.temperature_sd * 0.45 * np.sin(t + rng.uniform(0, 2.0 * np.pi))
    temperature += rng.normal(0.0, profile.temperature_sd * 0.35, size=len(values))
    return {"pH": ph, "EC": ec, "temperature": temperature}


def _specific_growth(values: np.ndarray, interval_days: float) -> np.ndarray:
    safe = np.clip(values.astype(float), 1e-9, None)
    growth = np.r_[np.nan, np.diff(np.log(safe)) / max(interval_days, 1e-9)]
    return growth


def _phase_code(phase: str) -> int:
    return {"lag": 0, "exponential": 1, "stationary": 2, "decline": 3}.get(str(phase).lower(), -1)


def _ritmo_label(phase: str, growth_rate: float) -> str:
    if phase == "decline" or (np.isfinite(growth_rate) and growth_rate < -0.02):
        return "DECLIVE"
    if not np.isfinite(growth_rate):
        return "SIN_DATO"
    if growth_rate >= 0.10:
        return "RAPIDO"
    if growth_rate >= 0.025:
        return "MODERADO"
    return "ESTACIONARIO"


def _state_label(phase: str, normalized_value: float) -> str:
    if phase == "lag":
        return "INOCULO_PLANTA"
    if phase == "exponential":
        return "CRECIMIENTO"
    if phase == "decline":
        return "DESCARTE"
    if normalized_value >= 0.72:
        return "PRODUCTO"
    return "OBSERVACION"


def _optimality_score(phase: str, normalized_value: float, growth_rate: float) -> float:
    growth_bonus = 0.0 if not np.isfinite(growth_rate) else np.clip(growth_rate / 0.20, -1.0, 1.0)
    phase_bonus = {"lag": -0.25, "exponential": 0.30, "stationary": 0.18, "decline": -0.45}.get(phase, 0.0)
    return float(np.clip(0.55 * normalized_value + 0.25 * growth_bonus + phase_bonus, 0.0, 1.0))


def _observed_bounds(frame: pd.DataFrame, candidates: list[str], default_low: float, default_high: float) -> tuple[float, float]:
    if frame.empty:
        return default_low, default_high
    col = _first_matching_column(frame, candidates)
    if col is None:
        return default_low, default_high
    values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3:
        return default_low, default_high
    low, high = np.quantile(values.to_numpy(dtype=float), [0.05, 0.95])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return default_low, default_high
    span = max(float(high - low), 1e-6)
    return float(low - 0.15 * span), float(high + 0.15 * span)


def _first_matching_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {col: str(col).lower() for col in frame.columns}
    for candidate in candidates:
        needle = candidate.lower()
        for col, col_norm in normalized.items():
            if needle in col_norm:
                return col
    return None


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


def _stable_seed(base_seed: int, group: str | None, target: str) -> int:
    token = f"{base_seed}:{group or 'ALL'}:{target}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()[:8]
    return (int(digest, 16) + int(base_seed)) % (2**32 - 1)
