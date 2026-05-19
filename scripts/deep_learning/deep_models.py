from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.forecasting.base import ForecastModel, ModelSkipped, SkippedModel, clean_series, require_points
from scripts.machine_learning.ml_models import RecursiveSklearnModel


class TensorFlowSequenceModel(ForecastModel):
    category = "deep_learning"
    min_points = 14

    def __init__(self, name: str, cell: str, epochs: int = 80, lags: int = 4, units: int = 16) -> None:
        super().__init__()
        self.name = name
        self.cell = cell
        self.epochs = epochs
        self.lags = lags
        self.units = units
        self.hyperparameters = {"cell": cell, "epochs": epochs, "lags": lags, "units": units, "optimizer": "adam", "loss": "mse"}

    def fit_predict(self, train: pd.Series, horizon: int, **_: object) -> np.ndarray:
        series = clean_series(train)
        require_points(series, max(self.min_points, self.lags + 5), self.name)
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:
            raise ModelSkipped(f"tensorflow is required for {self.name}: {exc}") from exc
        values = series.to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1) or 1.0)
        scaled = (values - mean) / std
        x_rows, y_rows = [], []
        for idx in range(self.lags, len(scaled)):
            x_rows.append(scaled[idx - self.lags : idx])
            y_rows.append(scaled[idx])
        x_train = np.asarray(x_rows, dtype=float).reshape(-1, self.lags, 1)
        y_train = np.asarray(y_rows, dtype=float)
        if len(x_train) < 4:
            raise ModelSkipped("Not enough sequence samples")

        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential()
        if self.cell == "lstm":
            model.add(tf.keras.layers.LSTM(self.units, input_shape=(self.lags, 1)))
        elif self.cell == "bilstm":
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units), input_shape=(self.lags, 1)))
        elif self.cell == "gru":
            model.add(tf.keras.layers.GRU(self.units, input_shape=(self.lags, 1)))
        elif self.cell == "cnn":
            model.add(tf.keras.layers.Conv1D(self.units, kernel_size=2, activation="relu", input_shape=(self.lags, 1)))
            model.add(tf.keras.layers.Flatten())
        else:
            model.add(tf.keras.layers.SimpleRNN(self.units, input_shape=(self.lags, 1)))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=self.epochs, verbose=0)
        self.fitted_model = model

        history = list(scaled)
        preds = []
        for _step in range(horizon):
            x = np.asarray(history[-self.lags:], dtype=float).reshape(1, self.lags, 1)
            pred_scaled = float(model.predict(x, verbose=0).ravel()[0])
            preds.append(pred_scaled * std + mean)
            history.append(pred_scaled)
        self.metadata = {"lags": self.lags, "epochs": self.epochs, "cell": self.cell, "units": self.units}
        return np.asarray(preds, dtype=float)


def get_deep_learning_models(random_state: int = 42, heavy: bool = False, params: dict[str, object] | None = None) -> list[ForecastModel]:
    params = params or {}

    def cfg(model_name: str, defaults: dict[str, object]) -> dict[str, object]:
        merged = defaults.copy()
        merged.update(dict(params.get(model_name, {}) or {}))
        return merged

    models: list[ForecastModel] = []
    try:
        from sklearn.neural_network import MLPRegressor  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        mlp = cfg("MLP", {"lags": 4, "hidden_layer_sizes": [32, 16], "max_iter": 1000})
        hidden = tuple(int(item) for item in mlp["hidden_layer_sizes"])
        models.append(
            RecursiveSklearnModel(
                "MLP",
                lambda: make_pipeline(
                    StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=hidden, max_iter=int(mlp["max_iter"]), random_state=random_state),
                ),
                lags=int(mlp["lags"]),
                estimator_params=mlp,
            )
        )
        models[-1].category = "deep_learning"
    except Exception as exc:
        models.append(SkippedModel("MLP", "deep_learning", f"scikit-learn is required for MLP: {exc}"))

    sequence_names = [
        ("RNN", "rnn"),
        ("LSTM", "lstm"),
        ("Bidirectional_LSTM", "bilstm"),
        ("GRU", "gru"),
        ("CNN_temporal", "cnn"),
    ]
    for name, cell in sequence_names:
        sequence = cfg(name, {"epochs": 120 if heavy else 40, "lags": 4, "units": 16})
        models.append(TensorFlowSequenceModel(name, cell, epochs=int(sequence["epochs"]), lags=int(sequence["lags"]), units=int(sequence["units"])))

    advanced = [
        "Seq2Seq",
        "TCN",
        "DeepAR",
        "N_BEATS",
        "N_HiTS",
        "TFT",
        "Temporal_Transformers",
        "Informer",
        "Autoformer",
        "PatchTST",
    ]
    for name in advanced:
        models.append(
            SkippedModel(
                name,
                "deep_learning",
                "Install darts/neuralforecast or implement a project-specific architecture to enable this advanced model.",
            )
        )
    return models
