from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.preprocessing.data_loader import DataSchema


class FeatureEngineer:
    def __init__(
        self,
        schema: DataSchema,
        rolling_windows: list[int],
        logger: logging.Logger | None = None,
    ) -> None:
        self.schema = schema
        self.rolling_windows = rolling_windows
        self.logger = logger or logging.getLogger("microalgas")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        working[self.schema.date_col] = pd.to_datetime(working[self.schema.date_col], errors="coerce")
        working = working.sort_values([self.schema.group_col, self.schema.date_col]).reset_index(drop=True)

        frames = []
        for _, group in working.groupby(self.schema.group_col, dropna=False):
            frames.append(self._transform_group(group.copy()))
        engineered = pd.concat(frames, ignore_index=True) if frames else working
        self.logger.info("Feature engineering complete: %d columns", engineered.shape[1])
        return engineered

    def _transform_group(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(self.schema.date_col).copy()
        dates = pd.to_datetime(frame[self.schema.date_col], errors="coerce")
        frame["time_idx"] = np.arange(len(frame), dtype=float)
        frame["elapsed_days"] = (dates - dates.min()).dt.total_seconds() / 86_400
        frame["elapsed_days"] = frame["elapsed_days"].fillna(frame["time_idx"])

        for col in self.schema.analysis_columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            frame[f"{col}__diff"] = series.diff()
            frame[f"{col}__pct_change"] = series.pct_change().replace([np.inf, -np.inf], np.nan)
            for lag in (1, 2, 3):
                frame[f"{col}__lag_{lag}"] = series.shift(lag)
            for window in self.rolling_windows:
                frame[f"{col}__roll_mean_{window}"] = series.rolling(window, min_periods=1).mean()
                frame[f"{col}__roll_std_{window}"] = series.rolling(window, min_periods=2).std()

        for target in self.schema.target_columns:
            if target in frame.columns:
                target_series = pd.to_numeric(frame[target], errors="coerce")
                delta_days = frame["elapsed_days"].diff().replace(0, np.nan)
                frame[f"{target}__specific_growth_rate"] = np.log(target_series / target_series.shift(1)) / delta_days
        return frame
