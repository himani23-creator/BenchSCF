import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from models.base_forecaster import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost forecaster with engineered lag features.
    Features: lags t-1 to t-12, rolling mean/std at windows 4 and 8.
    All features computed within each CV fold's training window — no leakage.
    Uses recursive multi-step forecasting for horizon > 1.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.n_lags = self.params.get("n_lags", 12)
        self.rolling_windows = self.params.get("rolling_windows", [4, 8])
        self.n_estimators = self.params.get("n_estimators", 200)
        self.max_depth = self.params.get("max_depth", 5)
        self.learning_rate = self.params.get("learning_rate", 0.05)
        self.model = None
        self._last_window = None

    def _build_features(self, series: np.ndarray) -> pd.DataFrame:
        """Build lag and rolling features from a time series."""
        df = pd.DataFrame({"y": series})

        # Lag features
        for lag in range(1, self.n_lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)

        # Rolling statistics — computed within training window only
        for window in self.rolling_windows:
            df[f"rolling_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
            df[f"rolling_std_{window}"] = df["y"].shift(1).rolling(window).std()

        df.dropna(inplace=True)
        return df

    def fit(self, y_train: np.ndarray) -> None:
        # Remove any NaN values
        y_train = np.array(y_train, dtype=float)
        y_train = y_train[~np.isnan(y_train)]
        
        df = self._build_features(y_train)
        X = df.drop(columns=["y"]).values
        y = df["y"].values

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.params.get("seed", 42),
            verbosity=0,
        )
        self.model.fit(X, y)

        # Store the last window for recursive forecasting
        self._last_window = list(y_train[-self.n_lags:])
        self._train_series = list(y_train)
        self.is_fitted = True

    def _build_single_feature_row(self, window: list) -> np.ndarray:
        """Build a single feature row from the current window for recursive forecasting."""
        row = []
        # Lag features
        for lag in range(1, self.n_lags + 1):
            row.append(window[-lag])
        # Rolling features
        for w in self.rolling_windows:
            recent = window[-w:]
            row.append(np.mean(recent))
            row.append(np.std(recent) if len(recent) > 1 else 0.0)
        return np.array(row).reshape(1, -1)

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        window = list(self._last_window)
        predictions = []

        for _ in range(horizon):
            X_pred = self._build_single_feature_row(window)
            y_pred = self.model.predict(X_pred)[0]
            # Handle NaN predictions
            y_pred = np.nan_to_num(y_pred, nan=0.0)
            predictions.append(y_pred)
            window.append(y_pred)
            window.pop(0)

        return np.array(predictions)

    def get_params(self) -> dict:
        return {
            "n_lags": self.n_lags,
            "rolling_windows": self.rolling_windows,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }