from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing.data_loader import DataSchema
from scripts.utils.paths import safe_name
from scripts.utils.plot_style import PALETTE, apply_plot_style, polish_axis, save_figure, soften_spines


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
    apply_plot_style()

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

            fig, ax = plt.subplots(figsize=(11.5, 5.4))
            ax.plot(
                frame[schema.date_col],
                series,
                marker="o",
                markersize=5,
                linewidth=2.6,
                color=PALETTE["primary"],
            )
            ax.fill_between(frame[schema.date_col], series, alpha=0.08, color=PALETTE["primary"])
            polish_axis(ax, f"{group} - serie temporal de {col}", "Fecha", col)
            fig.autofmt_xdate()
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_timeseries", make_png, make_svg))
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
            sns.histplot(series.dropna(), kde=True, ax=axes[0], color=PALETTE["primary"], edgecolor="white", linewidth=0.7)
            axes[0].set_title("Histograma + KDE")
            sns.boxplot(x=series, ax=axes[1], color=PALETTE["accent"], linewidth=1.2)
            axes[1].set_title("Boxplot")
            sns.violinplot(x=series, ax=axes[2], color=PALETTE["secondary"], linewidth=1.1)
            axes[2].set_title("Violin")
            soften_spines(axes)
            for axis in axes:
                polish_axis(axis)
            fig.suptitle(f"{group} - distribucion de {col}", x=0.05, y=1.03, ha="left", fontweight="bold", color=PALETTE["primary_dark"])
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_distribution", make_png, make_svg))
            plt.close(fig)

        numeric = frame[schema.analysis_columns].select_dtypes(include="number")
        if numeric.shape[1] >= 2 and numeric.dropna().shape[0] >= 3:
            fig, ax = plt.subplots(figsize=(9.5, 7.4))
            sns.heatmap(
                numeric.corr(),
                annot=True,
                fmt=".2f",
                cmap="BrBG",
                center=0,
                linewidths=0.6,
                linecolor="white",
                square=True,
                cbar_kws={"shrink": 0.82},
                ax=ax,
            )
            ax.set_title(f"{group} - matriz de correlacion", loc="left", fontweight="bold", pad=12)
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
    apply_plot_style()
    for group, frame in df.groupby(schema.group_col, dropna=False):
        bim_dir = output_dir / safe_name(group)
        bim_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.sort_values(schema.date_col)
        for col in schema.target_columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().sum() < 4:
                continue
            fig, ax = plt.subplots(figsize=(5.6, 5.4))
            lag_plot(series.dropna(), ax=ax)
            polish_axis(ax, f"{group} - lag plot {col}", f"{col}(t)", f"{col}(t+1)")
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_lag_plot", make_png, make_svg))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11.5, 5.4))
            ax.plot(frame[schema.date_col], series, label=col, marker="o", linewidth=2.4, color=PALETTE["primary"])
            for idx, window in enumerate(windows):
                color = PALETTE["secondary"] if idx % 2 == 0 else PALETTE["accent"]
                ax.plot(frame[schema.date_col], series.rolling(window, min_periods=1).mean(), label=f"media movil {window}", linewidth=2, color=color)
                ax.plot(frame[schema.date_col], series.rolling(window, min_periods=2).std(), label=f"desv. movil {window}", linewidth=1.7, linestyle="--", color=PALETTE["muted"])
            polish_axis(ax, f"{group} - estadisticas moviles {col}", "Fecha", col, legend=True)
            fig.autofmt_xdate()
            saved.extend(_save_figure(fig, bim_dir / f"{safe_name(col)}_rolling_statistics", make_png, make_svg))
            plt.close(fig)
    return saved


def _save_figure(fig: object, base_path: Path, make_png: bool, make_svg: bool) -> list[Path]:
    return save_figure(fig, base_path, make_png, make_svg)
