import numpy as np
import warnings
from models.base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")


class LSTMForecaster(BaseForecaster):
    """
    LSTM Seq2Seq encoder-decoder for multi-step supply chain forecasting.
    Input: sliding window of look_back steps.
    Output: next `horizon` steps via dense decoder.
    Trained with early stopping on validation MAE — runs on CPU in <30 min on DataCo.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.look_back = self.params.get("look_back", 12)
        self.lstm_units = self.params.get("lstm_units", 64)
        self.epochs = self.params.get("epochs", 50)
        self.batch_size = self.params.get("batch_size", 16)
        self.patience = self.params.get("patience", 5)
        self.seed = self.params.get("seed", 42)
        self.model = None
        self._last_window = None

    def _build_sequences(self, series: np.ndarray, horizon: int):
        """Create sliding window (X, y) sequences from a 1D series."""
        X, y = [], []
        for i in range(len(series) - self.look_back - horizon + 1):
            X.append(series[i: i + self.look_back])
            y.append(series[i + self.look_back: i + self.look_back + horizon])
        return np.array(X), np.array(y)

    def _build_model(self, horizon: int):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

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
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(self.seed)

        # Use last 10% of training data as internal validation for early stopping
        val_split = max(1, int(len(y_train) * 0.1))
        train_series = y_train[:-val_split]
        val_series = y_train[-val_split - self.look_back:]

        horizon = self.params.get("horizon", 4)

        X_train, y_tr = self._build_sequences(train_series, horizon)
        X_val, y_val = self._build_sequences(val_series, horizon)

        if len(X_train) == 0:
            raise ValueError("Training series too short for the given look_back and horizon.")

        # Reshape for LSTM: (samples, timesteps, features)
        X_train = X_train.reshape(-1, self.look_back, 1)
        X_val = X_val.reshape(-1, self.look_back, 1)

        self.model = self._build_model(horizon)
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )

        self.model.fit(
            X_train, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
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

        # If requested horizon differs from trained horizon, truncate or recurse
        if horizon <= self._trained_horizon:
            return pred[:horizon]

        # For longer horizons, do recursive prediction
        predictions = list(pred)
        current_window = list(self._last_window) + list(pred)
        while len(predictions) < horizon:
            new_window = np.array(current_window[-self.look_back:]).reshape(1, self.look_back, 1)
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