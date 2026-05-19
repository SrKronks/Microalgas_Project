# Microalgas Project

Proyecto profesional de ciencia de datos para analisis, modelado y pronostico
del crecimiento de microalgas por BIM-ID a partir de archivos Excel.

El pipeline detecta columnas temporales, variables numericas, BIM/grupo,
calidad de datos, estadistica descriptiva, diagnosticos temporales,
analisis multivariado, entrenamiento de modelos, ranking y reportes
automaticos.

## Estructura

```text
data/raw/                  Excel original
data/processed/            Dataset limpio y con features
scripts/                   Codigo modular por etapa
outputs/figures/           Graficos PNG/SVG por BIM
outputs/models/            Artefactos y tarjetas de modelos
outputs/metrics/           Metricas y descriptivos
outputs/forecasts/         Predicciones por modelo
outputs/reports/           HTML, PDF y Excel consolidado
outputs/diagnostics/       Calidad, estacionariedad y multivariado
outputs/rankings/          Ranking de modelos
configs/config.yaml        Configuracion reproducible
logs/pipeline.log          Logging rotativo
tests/                     Tests de humo
```

## Instalacion

Con `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Con Conda:

```bash
conda env create -f environment.yml
conda activate microalgas-project
```

## Ejecucion

```bash
python main.py
```

El archivo esperado por defecto es:

```text
data/raw/historial-monitoreos-2026-05-14.xlsx
```

Tambien puedes usar otro Excel:

```bash
python main.py --input ruta/al/archivo.xlsx
```

O cambiar objetivos de pronostico:

```bash
python main.py --targets OD,pH,EC
```

## Modelos incluidos

El proyecto registra modelos clasicos, ARIMA/estado espacio, biologicos,
ecuaciones diferenciales, probabilisticos, machine learning, deep learning e
hibridos. Las dependencias pesadas son opcionales en ejecucion: si una libreria
o dato biologico externo no existe, el modelo queda marcado como `skipped` y el
pipeline continua.

## Validacion biologica con ciclos sinteticos

Por defecto el pipeline evita el split temporal 80/20 sobre cada BIM, porque en
un cultivo biologico ese corte suele entrenar con lag/exponencial y validar
solo contra estacionaria/declive. En su lugar, `validation.strategy:
synthetic_full_cycle` genera ciclos completos sinteticos de crecimiento de
microalgas, entrena los modelos supervisados por rezagos con esos ciclos y
valida sobre todo el ciclo real disponible mediante prediccion one-step-ahead
con rezagos observados.

La configuracion esta en `synthetic_training`:

- `n_cycles`: cantidad de ciclos sinteticos, por defecto 2000
- `min_cycle_points` / `max_cycle_points`: largo de cada ciclo
- `noise_fraction`: ruido de medicion autocorrelacionado
- `decline_probability`: probabilidad de fase de declive
- `seasonality_probability`: oscilacion suave en fase madura
- `save_dataset`: guarda `data/processed/synthetic_growth_cycles_<target>.csv`

Los modelos que aun no implementan entrenamiento desde ciclos sinteticos quedan
marcados como `skipped` en ese modo, para evitar metricas optimistas o
comparaciones biologicamente inconsistentes. Para recuperar el holdout temporal
anterior, usa `validation.strategy: temporal_holdout` y
`synthetic_training.enabled: false`.

Para comparar si conviene usar 40%, 70%, 90% u otro porcentaje de ciclos con
declive, ejecuta el analisis de sensibilidad:

```bash
python -m scripts.sensitivity.decline_sensitivity --probabilities 0.4,0.7,0.9 --n-cycles 2000
```

Salidas principales:

- `outputs/sensitivity/decline_probability/all_model_metrics.csv`
- `outputs/sensitivity/decline_probability/decline_probability_summary.csv`
- `outputs/sensitivity/decline_probability/recommended_decline_probability.csv`

Para una prueba rapida:

```bash
python -m scripts.sensitivity.decline_sensitivity --probabilities 0.4,0.7 --n-cycles 200 --max-bims 3
```

Esto permite correr una version minima con `pandas/numpy/openpyxl`, y una
version completa al instalar `requirements.txt`.

## Salidas principales

- `outputs/reports/microalgas_report.html`
- `outputs/reports/dashboard_summary.html`
- `outputs/reports/microalgas_consolidated_results.xlsx`
- `outputs/reports/microalgas_report.pdf` si `matplotlib` esta instalado
- `outputs/rankings/model_rankings.csv`
- `outputs/metrics/model_metrics.csv`
- `outputs/forecasts/all_forecasts.csv`

## Configuracion

Edita `configs/config.yaml` para:

- cambiar targets
- ajustar horizonte de pronostico
- activar/desactivar familias de modelos
- cambiar entre validacion sintetica de ciclo completo y holdout temporal
- controlar la cantidad y forma de ciclos sinteticos
- modificar imputacion y ventanas moviles
- controlar PNG/SVG/PDF
- limitar BIMs durante pruebas

## Reproducibilidad

El pipeline usa rutas relativas, logging, configuracion central, manejo robusto
de errores y deteccion automatica de GPU/CUDA cuando `torch` o `tensorflow`
estan disponibles.
