from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.preprocessing.data_loader import DataSchema


@dataclass(frozen=True)
class QualityResult:
    clean_data: pd.DataFrame
    quality_table: pd.DataFrame
    outlier_table: pd.DataFrame
    inconsistencies: pd.DataFrame


class DataQualityProcessor:
    def __init__(
        self,
        schema: DataSchema,
        imputation_strategy: str = "time_interpolate",
        iqr_multiplier: float = 1.5,
        logger: logging.Logger | None = None,
    ) -> None:
        self.schema = schema
        self.imputation_strategy = imputation_strategy
        self.iqr_multiplier = iqr_multiplier
        self.logger = logger or logging.getLogger("microalgas")

    def run(self, df: pd.DataFrame) -> QualityResult:
        working = df.copy()
        working[self.schema.date_col] = pd.to_datetime(working[self.schema.date_col], errors="coerce")
        working = working.sort_values([self.schema.group_col, self.schema.date_col]).reset_index(drop=True)

        quality = self._quality_table(working)
        outliers = self._outlier_table(working)
        inconsistencies = self._inconsistencies(working)
        clean = self._impute(working)
        self.logger.info("Quality validation complete: missing cells=%d", int(working.isna().sum().sum()))
        return QualityResult(clean, quality, outliers, inconsistencies)

    def _quality_table(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        duplicated_rows = int(df.duplicated().sum())
        for col in df.columns:
            rows.append(
                {
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_pct": float(df[col].isna().mean() * 100),
                    "unique_count": int(df[col].nunique(dropna=True)),
                    "duplicated_rows_total": duplicated_rows,
                }
            )
        return pd.DataFrame(rows)

    def _outlier_table(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for group, frame in df.groupby(self.schema.group_col, dropna=False):
            for col in self.schema.numeric_columns:
                series = pd.to_numeric(frame[col], errors="coerce").dropna()
                if len(series) < 4:
                    continue
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
                count = int(((series < lower) | (series > upper)).sum())
                rows.append(
                    {
                        "BIM": group,
                        "column": col,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "outlier_count": count,
                    }
                )
        return pd.DataFrame(rows)

    def _inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        non_negative_markers = ("od", "abs", "ec", "temperatura", "temp")
        for col in self.schema.numeric_columns:
            norm = col.lower()
            if any(marker in norm for marker in non_negative_markers):
                bad = df[pd.to_numeric(df[col], errors="coerce") < 0]
                for idx in bad.index:
                    rows.append(
                        {
                            "row": int(idx),
                            "column": col,
                            "value": bad.loc[idx, col],
                            "issue": "negative_value_in_non_negative_variable",
                        }
                    )
        invalid_dates = df[df[self.schema.date_col].isna()]
        for idx in invalid_dates.index:
            rows.append({"row": int(idx), "column": self.schema.date_col, "value": None, "issue": "invalid_date"})
        return pd.DataFrame(rows)

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()
        if self.imputation_strategy == "none":
            return clean

        def impute_group(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.sort_values(self.schema.date_col).copy()
            for col in self.schema.numeric_columns:
                series = pd.to_numeric(frame[col], errors="coerce")
                if self.imputation_strategy == "median":
                    frame[col] = series.fillna(series.median())
                else:
                    frame[col] = series.interpolate(method="linear", limit_direction="both")
                    frame[col] = frame[col].fillna(series.median())
            return frame

        frames = [impute_group(group) for _, group in clean.groupby(self.schema.group_col, dropna=False)]
        clean = pd.concat(frames, ignore_index=True) if frames else clean
        for col in self.schema.categorical_columns:
            if col in clean.columns:
                clean[col] = clean[col].ffill().bfill()
        return clean.reset_index(drop=True)
