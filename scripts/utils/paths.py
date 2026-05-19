from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(value: object) -> str:
    text = str(value).strip() or "unknown"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "unknown"


def output_dirs(root: Path) -> dict[str, Path]:
    outputs = root / "outputs"
    return {
        "figures": outputs / "figures",
        "models": outputs / "models",
        "metrics": outputs / "metrics",
        "forecasts": outputs / "forecasts",
        "reports": outputs / "reports",
        "shap": outputs / "shap",
        "diagnostics": outputs / "diagnostics",
        "rankings": outputs / "rankings",
        "processed": root / "data" / "processed",
        "logs": root / "logs",
    }
