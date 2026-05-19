from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing.data_loader import DataSchema
from scripts.utils.paths import safe_name


def descriptive_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        mode = series.mode(dropna=True)
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        rows.append(
            {
                "variable": col,
                "count": int(series.count()),
                "mean": mean,
                "median": float(series.median()),
                "mode": float(mode.iloc[0]) if not mode.empty else np.nan,
                "std": std,
                "variance": float(series.var(ddof=1)) if len(series) > 1 else np.nan,
                "min": float(series.min()),
                "p05": float(series.quantile(0.05)),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
                "skewness": float(series.skew()) if len(series) > 2 else np.nan,
                "kurtosis": float(series.kurt()) if len(series) > 3 else np.nan,
                "coefficient_variation": float(std / mean) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_descriptive_by_group(
    df: pd.DataFrame,
    schema: DataSchema,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_stats = descriptive_statistics(df, schema.analysis_columns)
    global_stats.insert(0, "BIM", "ALL")
    pieces = [global_stats]
    for group, frame in df.groupby(schema.group_col, dropna=False):
        stats = descriptive_statistics(frame, schema.analysis_columns)
        if not stats.empty:
            stats.insert(0, "BIM", group)
            pieces.append(stats)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    result.to_csv(output_dir / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved descriptive statistics: %s", output_dir / "descriptive_statistics.csv")
    return result


def generate_eda_figures(
    df: pd.DataFrame,
    schema: DataSchema,
    output_dir: Path,
    logger: logging.Logger,
    make_png: bool = True,
    make_svg: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except Exception as exc:
        note = output_dir / "figures_skipped.txt"
        note.write_text(f"Matplotlib/seaborn not available: {exc}\n", encoding="utf-8")
        logger.warning("Skipping EDA figures because plotting libraries are unavailable: %s", exc)
        return [note]

    saved: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk")

    for group, frame in df.groupby(schema.group_col, dropna=False):
        bim_dir = output_dir / safe_name(group)
        bim_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.sort_values(schema.date_col)
        for col in schema.analysis_columns:
            if col not in frame:
                continue
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().sum() < 2:
                continue

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(frame[schema.date_col], series, marker="o", linewidth=2)
            ax.set_title(f"{group} - serie temporal de {col}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel(col)
            fig.autofmt_xdate()
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_timeseries", make_png, make_svg))
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            sns.histplot(series.dropna(), kde=True, ax=axes[0], color="#2A9D8F")
            axes[0].set_title("Histograma + KDE")
            sns.boxplot(x=series, ax=axes[1], color="#E9C46A")
            axes[1].set_title("Boxplot")
            sns.violinplot(x=series, ax=axes[2], color="#F4A261")
            axes[2].set_title("Violin")
            fig.suptitle(f"{group} - distribucion de {col}", y=1.03)
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_distribution", make_png, make_svg))
            plt.close(fig)

        numeric = frame[schema.analysis_columns].select_dtypes(include="number")
        if numeric.shape[1] >= 2 and numeric.dropna().shape[0] >= 3:
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
            ax.set_title(f"{group} - matriz de correlacion")
            saved.extend(_save_figure(fig, bim_dir / "correlation_heatmap", make_png, make_svg))
            plt.close(fig)

    logger.info("Generated %d EDA figure files", len(saved))
    return saved


def generate_lag_and_rolling_plots(
    df: pd.DataFrame,
    schema: DataSchema,
    output_dir: Path,
    logger: logging.Logger,
    windows: list[int],
    make_png: bool = True,
    make_svg: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from pandas.plotting import lag_plot  # type: ignore
    except Exception as exc:
        logger.warning("Skipping lag/rolling plots: %s", exc)
        return []

    saved: list[Path] = []
    for group, frame in df.groupby(schema.group_col, dropna=False):
        bim_dir = output_dir / safe_name(group)
        bim_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.sort_values(schema.date_col)
        for col in schema.target_columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().sum() < 4:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            lag_plot(series.dropna(), ax=ax)
            ax.set_title(f"{group} - lag plot {col}")
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_lag_plot", make_png, make_svg))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(frame[schema.date_col], series, label=col, marker="o")
            for window in windows:
                ax.plot(frame[schema.date_col], series.rolling(window, min_periods=1).mean(), label=f"rolling mean {window}")
                ax.plot(frame[schema.date_col], series.rolling(window, min_periods=2).std(), label=f"rolling std {window}")
            ax.legend()
            ax.set_title(f"{group} - estadisticas moviles {col}")
            fig.autofmt_xdate()
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_rolling_statistics", make_png, make_svg))
            plt.close(fig)
    return saved


def _save_figure(fig: object, base_path: Path, make_png: bool, make_svg: bool) -> list[Path]:
    saved: list[Path] = []
    if make_png:
        png = base_path.with_suffix(".png")
        fig.savefig(png, dpi=160, bbox_inches="tight")  # type: ignore[attr-defined]
        saved.append(png)
    if make_svg:
        svg = base_path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight")  # type: ignore[attr-defined]
        saved.append(svg)
    return saved
