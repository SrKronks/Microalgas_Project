# Microalgas Project

Proyecto profesional de ciencia de datos para analisis, modelado y pronostico
del crecimiento de microalgas por BIM-ID a partir de archivos Excel.

El pipeline detecta columnas temporales, variables numericas, BIM/grupo,
calidad de datos, estadistica descriptiva, diagnosticos temporales,
analisis multivariado, entrenamiento de modelos, ranking y reportes
automaticos.

Si el dataset incluye etiquetas categoricas como `Ritmo` y
`Estado_Cultivo`, el pipeline tambien entrena clasificadores para esas
etiquetas y genera metricas especificas de clasificacion.

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

```powershell
.\setup_venv.bat
```

Activar en PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\venv\Scripts\activate
```

Activar en `cmd.exe`:

```bat
venv\Scripts\activate.bat
```

Tambien puedes abrir una consola `cmd` ya activada con:

```bat
activar_venv.bat
```

Para instalar todas las dependencias opcionales:

```bash
python -m pip install -r requirements.txt
```

En Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
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

En Windows, si quieres probar el entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\venv\Scripts\activate
python -m pytest -q
```

El archivo esperado por defecto es:

```text
data/raw/historial-monitoreos-2026-05-14.xlsx
```

Tambien puedes usar otro Excel:

```bash
python main.py --input ruta/al/archivo.xlsx
```

Si el archivo no esta en la ruta configurada, el pipeline se detiene con un
mensaje explicito. Esto evita depender de rutas locales de un computador
especifico y facilita ejecutar el proyecto en Linux, servidor o CI.

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

## Entrenamiento e hiperparametros

Los hiperparametros editables estan centralizados en:

```text
configs/config.yaml
```

Apartado principal:

```yaml
model_hyperparameters:
  machine_learning:
    Ridge:
      lags: 3
      alpha: 1.0
    Random_Forest:
      lags: 3
      n_estimators: 200
  probabilistic:
    Monte_Carlo:
      simulations: 500
```

Durante el pipeline, los modelos supervisados por rezagos se entrenan asi:

1. Se genera una matriz supervisada usando los ultimos `lags` valores del ciclo.
2. Si `validation.strategy: synthetic_full_cycle`, el entrenamiento se hace con
   ciclos sinteticos completos.
3. La validacion se hace contra el ciclo real con prediccion one-step-ahead,
   usando rezagos reales observados.
4. Las metricas y rankings se guardan en `outputs/metrics` y
   `outputs/rankings`.

Cada ejecucion guarda el catalogo efectivo de hiperparametros en:

- `outputs/diagnostics/model_hyperparameters.csv`
- `outputs/diagnostics/model_hyperparameters.json`

Esto permite verificar exactamente con que configuracion se entreno cada
modelo.

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

## Analisis de sensibilidad de hiperparametros

Para evaluar sensibilidad por modelo y a nivel general del proyecto:

```bash
python -m scripts.sensitivity.hyperparameter_sensitivity --include-baseline
```

Por defecto evalua parametros relevantes como:

- `Ridge.alpha`
- `Lasso.alpha`
- `Elastic_Net.l1_ratio`
- `Random_Forest.n_estimators`
- `Random_Forest.lags`
- `SVR.C`
- `Monte_Carlo.simulations`
- `synthetic_training.decline_probability`

Salidas principales:

- `outputs/sensitivity/hyperparameters/all_model_metrics.csv`
- `outputs/sensitivity/hyperparameters/per_model_sensitivity_summary.csv`
- `outputs/sensitivity/hyperparameters/project_sensitivity_summary.csv`
- `outputs/sensitivity/hyperparameters/recommended_hyperparameter_scenarios.csv`

Para una prueba rapida:

```bash
python -m scripts.sensitivity.hyperparameter_sensitivity --include-baseline --n-cycles 200 --max-bims 3
```

Para definir una grilla especifica:

```bash
python -m scripts.sensitivity.hyperparameter_sensitivity \
  --grid "machine_learning.Ridge.alpha=0.1,1,10;machine_learning.Random_Forest.lags=2,3,5" \
  --model-groups machine_learning \
  --n-cycles 500
```

## Graficos de ciclos sinteticos

Despues de ejecutar el pipeline con `synthetic_training.save_dataset: true`,
puedes graficar ejemplos de los ciclos simulados usados para entrenar:

```bash
python -m scripts.synthetic.plot_examples
```

Salidas por defecto:

- `outputs/figures/synthetic_examples/OD_synthetic_cycle_overlay.png`
- `outputs/figures/synthetic_examples/OD_SYN-00001_phases.png`

Opciones utiles:

```bash
python -m scripts.synthetic.plot_examples --n-cycles 20
python -m scripts.synthetic.plot_examples --cycle-id SYN-00025
python -m scripts.synthetic.plot_examples --input data/processed/synthetic_growth_cycles_OD.csv
```

## Salidas principales

- `outputs/reports/microalgas_report.html`
- `outputs/reports/dashboard_summary.html`
- `outputs/reports/microalgas_consolidated_results.xlsx`
- `outputs/reports/microalgas_report.pdf` si `matplotlib` esta instalado
- `outputs/rankings/model_rankings.csv`
- `outputs/metrics/model_metrics.csv`
- `outputs/forecasts/all_forecasts.csv`

## Clasificacion de estado y ritmo

Para datasets clasificados, configura las etiquetas en:

```yaml
data:
  classification_targets:
    - Ritmo
    - Estado_Cultivo
```

El pipeline entrena una rama supervisada para cada etiqueta usando las
variables numericas y features temporales disponibles. Por defecto usa
validacion `stratified_holdout` para preservar la proporcion de clases. Si
`scikit-learn` esta disponible, usa modelos como regresion logistica, Random
Forest, Extra Trees, Gradient Boosting, SVC y Naive Bayes. Si no esta
disponible, ejecuta clasificadores NumPy de respaldo para no perder la corrida.

Salidas principales:

- `outputs/metrics/classification_metrics.csv`
- `outputs/metrics/classification_predictions.csv`
- `outputs/rankings/classification_rankings.csv`
- `outputs/diagnostics/classification_confusion_matrices.csv`
- `outputs/diagnostics/classification_feature_importance.csv`
- hojas `Class_Metrics`, `Class_Rankings`, `Class_Predictions` y
  `Class_Confusion` en el Excel consolidado.

Metricas incluidas:

- Accuracy
- Balanced Accuracy
- Macro Precision / Recall / F1
- Weighted F1
- Cohen Kappa
- LogLoss cuando el modelo entrega probabilidades

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

Los datos crudos y salidas generadas (`data/raw`, `data/processed`, `outputs`,
`logs`, `catboost_info`) estan pensados como artefactos locales y quedan
ignorados por Git para evitar subir archivos pesados o sensibles por accidente.
