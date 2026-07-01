import numpy as np
import warnings
from models.base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    raise ImportError(f"TensorFlow is required for LSTMForecaster but could not be imported: {e}")


class LSTMForecaster(BaseForecaster):

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.look_back = self.params.get("look_back", 8)   # REDUCED from 12
        self.lstm_units = self.params.get("lstm_units", 64)
        self.epochs = self.params.get("epochs", 50)
        self.batch_size = self.params.get("batch_size", 16)
        self.patience = self.params.get("patience", 5)
        self.seed = self.params.get("seed", 42)
        self.model = None
        self._last_window = None
        self._trained_horizon = 4

    def _build_sequences(self, series: np.ndarray, horizon: int):
        X, y = [], []
        for i in range(len(series) - self.look_back - horizon + 1):
            X.append(series[i: i + self.look_back])
            y.append(series[i + self.look_back: i + self.look_back + horizon])
        return np.array(X), np.array(y)

    def _build_model(self, horizon: int):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.look_back, 1),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units // 2),
            Dropout(0.2),
            Dense(horizon),
        ])
        model.compile(optimizer="adam", loss="mae")
        return model

    def fit(self, y_train: np.ndarray) -> None:
        tf.random.set_seed(self.seed)
        horizon = self.params.get("horizon", 4)

        # ✅ KEY FIX 1: Check minimum length BEFORE doing anything
        min_required = self.look_back + horizon + 5
        if len(y_train) < min_required:
            raise ValueError(
                f"Training series too short: {len(y_train)} steps. "
                f"Need at least {min_required} for look_back={self.look_back}, horizon={horizon}."
            )

        # ✅ KEY FIX 2: Only use internal val split if enough data remains
        val_size = max(self.look_back + horizon, int(len(y_train) * 0.1))
        if len(y_train) - val_size < min_required:
            # Not enough for a val split — train on full series, use train loss
            train_series = y_train
            use_val = False
        else:
            train_series = y_train[:-val_size]
            val_series = y_train[-(val_size + self.look_back):]
            use_val = True

        X_train, y_tr = self._build_sequences(train_series, horizon)

        if len(X_train) == 0:
            raise ValueError("Could not build any training sequences.")

        X_train = X_train.reshape(-1, self.look_back, 1)

        self.model = self._build_model(horizon)

        callbacks = []
        val_data = None

        if use_val:
            X_val, y_val = self._build_sequences(val_series, horizon)
            if len(X_val) > 0:
                X_val = X_val.reshape(-1, self.look_back, 1)
                val_data = (X_val, y_val)
                callbacks.append(
                    EarlyStopping(
                        monitor="val_loss",
                        patience=self.patience,
                        restore_best_weights=True
                    )
                )

        self.model.fit(
            X_train, y_tr,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        self._last_window = y_train[-self.look_back:]
        self._trained_horizon = horizon
        self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        window = self._last_window.reshape(1, self.look_back, 1)
        pred = self.model.predict(window, verbose=0)[0]

        if horizon <= self._trained_horizon:
            return pred[:horizon]

        predictions = list(pred)
        current_window = list(self._last_window) + list(pred)
        while len(predictions) < horizon:
            new_window = np.array(
                current_window[-self.look_back:]
            ).reshape(1, self.look_back, 1)
            next_pred = self.model.predict(new_window, verbose=0)[0]
            predictions.extend(next_pred)
            current_window.extend(next_pred)

        return np.array(predictions[:horizon])

    def get_params(self) -> dict:
        return {
            "look_back": self.look_back,
            "lstm_units": self.lstm_units,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }