from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.utils.config import ProjectConfig


class ReportBuilder:
    def __init__(self, config: ProjectConfig, output_dirs: dict[str, Path], logger: logging.Logger) -> None:
        self.config = config
        self.output_dirs = output_dirs
        self.logger = logger

    def build(
        self,
        dataset_summary: dict[str, Any],
        dependency_summary: dict[str, Any],
        quality: pd.DataFrame,
        descriptive: pd.DataFrame,
        metrics: pd.DataFrame,
        rankings: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> dict[str, Path]:
        reports_dir = self.output_dirs["reports"]
        reports_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}
        if self.config.get("reporting.export_html", True):
            html_path = reports_dir / "microalgas_report.html"
            html_path.write_text(
                self._html(dataset_summary, dependency_summary, quality, descriptive, metrics, rankings),
                encoding="utf-8",
            )
            dashboard_path = reports_dir / "dashboard_summary.html"
            dashboard_path.write_text(self._dashboard(metrics, rankings, dataset_summary), encoding="utf-8")
            artifacts["html_report"] = html_path
            artifacts["dashboard"] = dashboard_path
        if self.config.get("reporting.export_excel", True):
            excel_path = reports_dir / "microalgas_consolidated_results.xlsx"
            self._excel(excel_path, dataset_summary, dependency_summary, quality, descriptive, metrics, rankings, forecasts)
            artifacts["excel"] = excel_path
        if self.config.get("reporting.export_pdf", True) and self.config.get("execution.make_pdf", True):
            pdf_path = reports_dir / "microalgas_report.pdf"
            if self._pdf(pdf_path, dataset_summary, metrics, rankings):
                artifacts["pdf"] = pdf_path
        self.logger.info("Reports generated: %s", artifacts)
        return artifacts

    def _html(
        self,
        dataset_summary: dict[str, Any],
        dependency_summary: dict[str, Any],
        quality: pd.DataFrame,
        descriptive: pd.DataFrame,
        metrics: pd.DataFrame,
        rankings: pd.DataFrame,
    ) -> str:
        title = self.config.get("reporting.title", "Microalgae report")
        best = rankings.head(20) if not rankings.empty else pd.DataFrame()
        status = metrics["status"].value_counts().rename_axis("status").reset_index(name="count") if "status" in metrics else pd.DataFrame()
        validation_modes = (
            metrics["validation_mode"].value_counts().rename_axis("validation_mode").reset_index(name="count")
            if "validation_mode" in metrics
            else pd.DataFrame()
        )
        return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2933; }}
    h1, h2 {{ color: #183A37; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; background: #fbfcfd; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px; text-align: left; }}
    th {{ background: #edf2f7; }}
    code {{ background: #edf2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <h2>Resumen del dataset</h2>
  <div class="grid">
    <div class="card"><b>Filas</b><br>{dataset_summary.get('rows')}</div>
    <div class="card"><b>Columnas</b><br>{dataset_summary.get('columns')}</div>
    <div class="card"><b>BIM detectados</b><br>{dataset_summary.get('n_groups')}</div>
    <div class="card"><b>Periodo</b><br>{dataset_summary.get('date_min')} a {dataset_summary.get('date_max')}</div>
  </div>
  <h2>Variables objetivo</h2>
  <p><code>{', '.join(map(str, dataset_summary.get('target_columns', [])))}</code></p>
  <h2>Estado de dependencias</h2>
  {_dict_table(dependency_summary)}
  <h2>Calidad de datos</h2>
  {_df_table(quality.head(40))}
  <h2>Estadistica descriptiva</h2>
  {_df_table(descriptive.head(80))}
  <h2>Estado de modelos</h2>
  {_df_table(status)}
  <h2>Protocolo de validacion</h2>
  {_df_table(validation_modes)}
  <h2>Top modelos por BIM y objetivo</h2>
  {_df_table(best)}
</body>
</html>"""

    def _dashboard(self, metrics: pd.DataFrame, rankings: pd.DataFrame, dataset_summary: dict[str, Any]) -> str:
        ok = int(metrics["status"].eq("ok").sum()) if "status" in metrics else 0
        failed = int(metrics["status"].eq("failed").sum()) if "status" in metrics else 0
        skipped = int(metrics["status"].eq("skipped").sum()) if "status" in metrics else 0
        best_model = rankings.iloc[0].to_dict() if not rankings.empty else {}
        validation_mode = metrics["validation_mode"].dropna().iloc[0] if "validation_mode" in metrics and not metrics["validation_mode"].dropna().empty else "n/a"
        return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Dashboard Microalgas</title>
<style>body{{font-family:Arial;margin:28px}}.kpi{{display:inline-block;border:1px solid #ddd;padding:16px;margin:8px;border-radius:8px;min-width:150px}}</style>
</head><body>
<h1>Dashboard resumen</h1>
<div class="kpi"><b>BIM</b><br>{dataset_summary.get('n_groups')}</div>
<div class="kpi"><b>Modelos OK</b><br>{ok}</div>
<div class="kpi"><b>Omitidos</b><br>{skipped}</div>
<div class="kpi"><b>Fallidos</b><br>{failed}</div>
<div class="kpi"><b>Validacion</b><br>{validation_mode}</div>
<h2>Mejor modelo global</h2>
{_dict_table(best_model)}
</body></html>"""

    def _excel(
        self,
        path: Path,
        dataset_summary: dict[str, Any],
        dependency_summary: dict[str, Any],
        quality: pd.DataFrame,
        descriptive: pd.DataFrame,
        metrics: pd.DataFrame,
        rankings: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            dataset_rows = [{"key": key, "value": _cell_value(value)} for key, value in dataset_summary.items()]
            dependency_rows = [{"dependency": key, "available": _cell_value(value)} for key, value in dependency_summary.items()]
            pd.DataFrame(dataset_rows).to_excel(writer, sheet_name="Dataset", index=False)
            pd.DataFrame(dependency_rows).to_excel(writer, sheet_name="Dependencies", index=False)
            quality.to_excel(writer, sheet_name="Quality", index=False)
            descriptive.to_excel(writer, sheet_name="Descriptive", index=False)
            metrics.to_excel(writer, sheet_name="Metrics", index=False)
            rankings.to_excel(writer, sheet_name="Rankings", index=False)
            forecasts.head(100_000).to_excel(writer, sheet_name="Forecasts", index=False)

    def _pdf(self, path: Path, dataset_summary: dict[str, Any], metrics: pd.DataFrame, rankings: pd.DataFrame) -> bool:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
        except Exception as exc:
            self.logger.warning("Matplotlib unavailable for PDF; using reportlab fallback: %s", exc)
            return self._pdf_reportlab(path, dataset_summary, metrics, rankings)
        with PdfPages(path) as pdf:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            text = [
                self.config.get("reporting.title", "Microalgae report"),
                "",
                f"Rows: {dataset_summary.get('rows')}",
                f"BIM groups: {dataset_summary.get('n_groups')}",
                f"Period: {dataset_summary.get('date_min')} to {dataset_summary.get('date_max')}",
                f"Successful models: {int(metrics['status'].eq('ok').sum()) if 'status' in metrics else 0}",
            ]
            if not rankings.empty:
                best = rankings.iloc[0]
                text.append(f"Best model: {best.get('model')} ({best.get('BIM')} / {best.get('target')})")
                text.append(f"RMSE: {best.get('RMSE'):.4f}")
            ax.text(0.05, 0.95, "\n".join(text), va="top", fontsize=14)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        return True

    def _pdf_reportlab(self, path: Path, dataset_summary: dict[str, Any], metrics: pd.DataFrame, rankings: pd.DataFrame) -> bool:
        try:
            from reportlab.lib.pagesizes import letter  # type: ignore
            from reportlab.pdfgen import canvas  # type: ignore
        except Exception as exc:
            self.logger.warning("Skipping PDF report because reportlab is unavailable: %s", exc)
            return False

        c = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        y = height - 54
        c.setFont("Helvetica-Bold", 16)
        c.drawString(54, y, str(self.config.get("reporting.title", "Microalgae report"))[:90])
        y -= 32
        c.setFont("Helvetica", 10)
        lines = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Rows: {dataset_summary.get('rows')}",
            f"Columns: {dataset_summary.get('columns')}",
            f"BIM groups: {dataset_summary.get('n_groups')}",
            f"Period: {dataset_summary.get('date_min')} to {dataset_summary.get('date_max')}",
            f"Successful models: {int(metrics['status'].eq('ok').sum()) if 'status' in metrics else 0}",
            f"Skipped models: {int(metrics['status'].eq('skipped').sum()) if 'status' in metrics else 0}",
        ]
        for line in lines:
            c.drawString(54, y, line[:110])
            y -= 16

        y -= 14
        c.setFont("Helvetica-Bold", 12)
        c.drawString(54, y, "Top model rankings")
        y -= 20
        c.setFont("Helvetica", 8)
        if rankings.empty:
            c.drawString(54, y, "No successful model rankings available.")
        else:
            cols = ["BIM", "target", "category", "model", "RMSE", "MAE", "mean_rank"]
            for _, row in rankings.head(24).iterrows():
                text = " | ".join(f"{col}: {row.get(col)}" for col in cols)
                c.drawString(54, y, text[:130])
                y -= 12
                if y < 54:
                    c.showPage()
                    y = height - 54
                    c.setFont("Helvetica", 8)
        c.save()
        return True


def _df_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No hay datos disponibles.</p>"
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].astype(str)
    return safe.to_html(index=False, escape=True)


def _dict_table(data: dict[str, Any]) -> str:
    if not data:
        return "<p>No hay datos disponibles.</p>"
    rows = [{"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in data.items()]
    return _df_table(pd.DataFrame(rows))


def _cell_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value
