from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing.data_loader import DataSchema
from scripts.utils.paths import safe_name
from scripts.utils.plot_style import apply_plot_style, polish_axis, save_figure_no_return


def stationarity_tests(series: pd.Series) -> list[dict[str, object]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    rows: list[dict[str, object]] = []
    if len(clean) < 6:
        return [{"test": "stationarity", "status": "skipped", "reason": "too_few_points"}]

    try:
        from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore

        adf = adfuller(clean, autolag="AIC")
        rows.append({"test": "ADF", "statistic": float(adf[0]), "p_value": float(adf[1]), "status": "ok"})
        try:
            kpss_result = kpss(clean, regression="c", nlags="auto")
            rows.append(
                {
                    "test": "KPSS",
                    "statistic": float(kpss_result[0]),
                    "p_value": float(kpss_result[1]),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"test": "KPSS", "status": "failed", "reason": str(exc)})
    except Exception as exc:
        rows.append({"test": "ADF/KPSS", "status": "skipped", "reason": str(exc)})

    try:
        from arch.unitroot import PhillipsPerron  # type: ignore

        pp = PhillipsPerron(clean)
        rows.append({"test": "Phillips-Perron", "statistic": float(pp.stat), "p_value": float(pp.pvalue), "status": "ok"})
    except Exception as exc:
        rows.append({"test": "Phillips-Perron", "status": "skipped", "reason": str(exc)})
    return rows


def temporal_diagnostics(
    df: pd.DataFrame,
    schema: DataSchema,
    output_dir: Path,
    logger: logging.Logger,
    seasonal_periods: int = 3,
    make_png: bool = True,
    make_svg: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for group, frame in df.groupby(schema.group_col, dropna=False):
        bim_dir = output_dir / safe_name(group)
        bim_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.sort_values(schema.date_col)
        for col in schema.target_columns:
            if col not in frame:
                continue
            series = pd.to_numeric(frame[col], errors="coerce")
            clean = series.dropna()
            for row in stationarity_tests(clean):
                rows.append({"BIM": group, "variable": col, **row})
            rolling = pd.DataFrame(
                {
                    schema.date_col: frame[schema.date_col],
                    "value": series,
                    "rolling_mean": series.rolling(seasonal_periods, min_periods=1).mean(),
                    "rolling_variance": series.rolling(seasonal_periods, min_periods=2).var(),
                }
            )
            rolling.to_csv(bim_dir / f"{safe_name(col)}_rolling_diagnostics.csv", index=False, encoding="utf-8-sig")
            _plot_acf_pacf_decomposition(clean, group, col, bim_dir, logger, seasonal_periods, make_png, make_svg)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "stationarity_tests.csv", index=False, encoding="utf-8-sig")
    return result


def cross_correlation_table(df: pd.DataFrame, schema: DataSchema, output_dir: Path, max_lag: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, frame in df.groupby(schema.group_col, dropna=False):
        numeric = frame[schema.analysis_columns].apply(pd.to_numeric, errors="coerce")
        for target in schema.target_columns:
            if target not in numeric:
                continue
            for col in numeric.columns:
                if col == target:
                    continue
                for lag in range(-max_lag, max_lag + 1):
                    corr = numeric[target].corr(numeric[col].shift(lag))
                    rows.append({"BIM": group, "target": target, "variable": col, "lag": lag, "correlation": corr})
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "cross_correlation.csv", index=False, encoding="utf-8-sig")
    return result


def _plot_acf_pacf_decomposition(
    series: pd.Series,
    group: object,
    col: str,
    output_dir: Path,
    logger: logging.Logger,
    seasonal_periods: int,
    make_png: bool,
    make_svg: bool,
) -> None:
    if len(series) < 6:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore
        from statsmodels.tsa.seasonal import STL, seasonal_decompose  # type: ignore
    except Exception as exc:
        logger.warning("Skipping temporal plots for %s/%s: %s", group, col, exc)
        return

    apply_plot_style()
    lags = max(1, min(12, len(series) // 2 - 1))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1], method="ywm")
    polish_axis(axes[0], f"{group} - ACF {col}", "Rezago", "Correlacion")
    polish_axis(axes[1], f"{group} - PACF {col}", "Rezago", "Correlacion parcial")
    _save(fig, output_dir / f"{safe_name(col)}_acf_pacf", make_png, make_svg)
    plt.close(fig)

    if len(series) >= seasonal_periods * 2:
        try:
            stl = STL(series.reset_index(drop=True), period=max(2, seasonal_periods), robust=True).fit()
            fig = stl.plot()
            fig.set_size_inches(10, 7)
            _save(fig, output_dir / f"{safe_name(col)}_stl_decomposition", make_png, make_svg)
            plt.close(fig)
        except Exception as exc:
            logger.warning("STL failed for %s/%s: %s", group, col, exc)
        try:
            decomposition = seasonal_decompose(
                series.reset_index(drop=True),
                period=max(2, seasonal_periods),
                model="additive",
                extrapolate_trend="freq",
            )
            fig = decomposition.plot()
            fig.set_size_inches(10, 7)
            _save(fig, output_dir / f"{safe_name(col)}_seasonal_decomposition", make_png, make_svg)
            plt.close(fig)
        except Exception as exc:
            logger.warning("seasonal_decompose failed for %s/%s: %s", group, col, exc)


def _save(fig: object, base: Path, make_png: bool, make_svg: bool) -> None:
    save_figure_no_return(fig, base, make_png, make_svg)
