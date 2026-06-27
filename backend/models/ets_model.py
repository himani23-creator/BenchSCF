import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from models.base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")


class ETSForecaster(BaseForecaster):
    """
    Exponential Smoothing (Holt-Winters) forecaster via statsmodels.
    Handles trend and seasonality. Strong performer on short supply chain series.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.trend = self.params.get("trend", "add")
        self.seasonal = self.params.get("seasonal", "add")
        self.seasonal_periods = self.params.get("seasonal_periods", 52)
        self.model_fit = None

    def fit(self, y_train: np.ndarray) -> None:
        # Clip to avoid zeros/negatives which break multiplicative models
        y_train = np.clip(y_train, 1e-3, None)
        model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self.model_fit = model.fit(optimized=True)
        self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        forecast = self.model_fit.forecast(horizon)
        return np.array(forecast)

    def get_params(self) -> dict:
        return {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
        }