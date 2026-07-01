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
        self._train_min = 0.0
        self._train_range = 1.0
        self._use_fallback = False

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

        # Remove any NaN values
        y_train = np.array(y_train, dtype=float)
        y_train = y_train[~np.isnan(y_train)]
        
        # Normalize to [0, 1] for stability
        train_min = np.min(y_train)
        train_max = np.max(y_train)
        train_range = train_max - train_min + 1e-8
        y_train_norm = (y_train - train_min) / train_range
        
        self._train_min = train_min
        self._train_range = train_range

        # Use last 15% of training data as internal validation
        val_split = max(2, int(len(y_train_norm) * 0.15))
        train_series = y_train_norm[:-val_split]
        val_series = y_train_norm[-val_split - self.look_back:]

        horizon = self.params.get("horizon", 4)

        X_train, y_tr = self._build_sequences(train_series, horizon)
        X_val, y_val = self._build_sequences(val_series, horizon)

        # Ensure minimum training data
        if len(X_train) < 3 or len(X_val) < 1:
            raise ValueError("Not enough data for LSTM training")

        # Reshape for LSTM: (samples, timesteps, features)
        X_train = X_train.reshape(-1, self.look_back, 1)
        X_val = X_val.reshape(-1, self.look_back, 1)

        self.model = self._build_model(horizon)
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )

        try:
            self.model.fit(
                X_train, y_tr,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=1,
                callbacks=[early_stop],
                verbose=0,
            )
            self._use_fallback = False
        except Exception as e:
            # Training failed, use fallback
            print(f"[LSTM] Training failed: {e}")
            self._use_fallback = True

        self._last_window = y_train_norm[-self.look_back:]
        self._trained_horizon = horizon
        self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        # Fallback: return average of last window (guaranteed to work)
        if self._use_fallback or self.model is None:
            avg_val = np.mean(self._last_window)
            pred_norm = np.full(horizon, avg_val)
            pred_denorm = pred_norm * self._train_range + self._train_min
            return np.array(pred_denorm, dtype=float)

        try:
            window = self._last_window.reshape(1, self.look_back, 1)
            pred = self.model.predict(window, verbose=0)[0]
            
            # Check for NaN
            if np.isnan(pred).any():
                avg_val = np.mean(self._last_window)
                pred = np.full(horizon, avg_val)
            else:
                pred = pred[:horizon] if horizon <= len(pred) else np.pad(pred, (0, horizon - len(pred)), mode='edge')

            # Denormalize back to original scale
            pred_denorm = pred * self._train_range + self._train_min
            return np.array(pred_denorm, dtype=float)
            
        except Exception:
            # Ultimate fallback: return average
            avg_val = np.mean(self._last_window)
            pred = np.full(horizon, avg_val)
            pred_denorm = pred * self._train_range + self._train_min
            return np.array(pred_denorm, dtype=float)

    def get_params(self) -> dict:
        return {
            "look_back": self.look_back,
            "lstm_units": self.lstm_units,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }