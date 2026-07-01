import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")


class SARIMAForecaster(BaseForecaster):
    """
    Seasonal ARIMA forecaster using statsmodels SARIMAX.
    Default order (1,1,1) x (1,1,0,52) for weekly supply chain data.
    Parameters can be overridden via config.yaml.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.order = tuple(self.params.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(self.params.get("seasonal_order", [1, 1, 0, 52]))
        self.model_fit = None

    def fit(self, y_train: np.ndarray) -> None:
        # Remove any NaN values
        y_train = np.array(y_train, dtype=float)
        y_train = y_train[~np.isnan(y_train)]
        
        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False)
        self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        forecast = self.model_fit.forecast(steps=horizon)
        forecast = np.array(forecast, dtype=float)
        # Handle any NaN values in forecast
        forecast = np.nan_to_num(forecast, nan=0.0)
        return forecast

    def get_params(self) -> dict:
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
        }