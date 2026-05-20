from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.utils.config import ProjectConfig


@dataclass(frozen=True)
class DataSchema:
    date_col: str
    group_col: str
    numeric_columns: list[str]
    analysis_columns: list[str]
    target_columns: list[str]
    label_columns: list[str]
    categorical_columns: list[str]


def _norm(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _first_matching_column(columns: Iterable[str], patterns: Iterable[str]) -> str | None:
    normalized = {col: _norm(col) for col in columns}
    for pattern in patterns:
        pattern_norm = _norm(pattern)
        for col, normed in normalized.items():
            if pattern_norm and pattern_norm in normed:
                return col
    return None


def load_excel(path: Path, sheet_name: str | None, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    excel = pd.ExcelFile(path)
    selected_sheet = sheet_name or excel.sheet_names[0]
    logger.info("Reading Excel file=%s sheet=%s", path, selected_sheet)
    df = pd.read_excel(path, sheet_name=selected_sheet)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def detect_schema(df: pd.DataFrame, config: ProjectConfig, logger: logging.Logger) -> DataSchema:
    configured_date = config.get("data.date_column")
    configured_group = config.get("data.group_column")

    date_col = configured_date if configured_date in df.columns else None
    if date_col is None:
        date_col = _first_matching_column(df.columns, ["fecha", "date", "datetime", "timestamp", "time"])
    if date_col is None:
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        date_col = datetime_cols[0] if datetime_cols else None
    if date_col is None:
        raise ValueError("No temporal column detected. Configure data.date_column.")

    group_col = configured_group if configured_group in df.columns else None
    if group_col is None:
        group_col = _first_matching_column(df.columns, ["bim-id", "bim", "reactor", "cultivo", "unidad", "tank"])
    if group_col is None:
        raise ValueError("No BIM/group column detected. Configure data.group_column.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_columns = [col for col in df.columns if col not in numeric_columns and col != date_col]

    preferred = list(config.get("data.preferred_analysis_columns", []))
    analysis_columns: list[str] = []
    for col in numeric_columns:
        normed = _norm(col)
        if any(_norm(pattern) in normed for pattern in preferred):
            analysis_columns.append(col)
    if not analysis_columns:
        analysis_columns = numeric_columns.copy()

    configured_targets = list(config.get("data.target_columns", []) or [])
    target_columns = [col for col in configured_targets if col in df.columns and col in numeric_columns]
    if not target_columns:
        od_col = _first_matching_column(numeric_columns, ["od", "absorbancia", "absorbance"])
        target_columns = [od_col] if od_col else analysis_columns[:1]

    configured_labels = list(config.get("data.classification_targets", []) or [])
    label_columns = [col for col in configured_labels if col in df.columns]
    if not label_columns:
        detected_labels = []
        for pattern in ["estado", "ritmo"]:
            match = _first_matching_column(categorical_columns, [pattern])
            if match and match not in detected_labels:
                detected_labels.append(match)
        label_columns = detected_labels

    logger.info(
        "Detected schema date=%s group=%s numeric=%d targets=%s labels=%s",
        date_col,
        group_col,
        len(numeric_columns),
        target_columns,
        label_columns,
    )
    return DataSchema(
        date_col=date_col,
        group_col=group_col,
        numeric_columns=numeric_columns,
        analysis_columns=analysis_columns,
        target_columns=target_columns,
        label_columns=label_columns,
        categorical_columns=categorical_columns,
    )


def summarize_dataset(df: pd.DataFrame, schema: DataSchema) -> dict[str, object]:
    dates = pd.to_datetime(df[schema.date_col], errors="coerce")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "date_column": schema.date_col,
        "group_column": schema.group_col,
        "numeric_columns": schema.numeric_columns,
        "analysis_columns": schema.analysis_columns,
        "target_columns": schema.target_columns,
        "label_columns": schema.label_columns,
        "n_groups": int(df[schema.group_col].nunique(dropna=True)),
        "groups": sorted(map(str, df[schema.group_col].dropna().unique())),
        "date_min": str(dates.min()) if not dates.isna().all() else None,
        "date_max": str(dates.max()) if not dates.isna().all() else None,
        "missing_pct": {str(k): float(v) for k, v in (df.isna().mean() * 100).round(4).items()},
    }


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
    return result
