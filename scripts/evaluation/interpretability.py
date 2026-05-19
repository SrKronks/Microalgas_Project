from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_feature_importance(table: pd.DataFrame, output_path: Path) -> None:
    if table.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_shap_status(output_path: Path, model_name: str, reason: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": model_name, "status": "skipped", "reason": reason}]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
