import numpy as np
import pandas as pd
import warnings
from models.base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except ImportError as e:
    raise ImportError(f"Prophet is required for ProphetForecaster but could not be imported: {e}")


class ProphetForecaster(BaseForecaster):
    """
    Meta Prophet forecaster. Handles missing data, holidays, and multiple seasonalities.
    Widely used in industry supply chain contexts.
    Requires prophet package: pip install prophet
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.yearly_seasonality = self.params.get("yearly_seasonality", True)
        self.weekly_seasonality = self.params.get("weekly_seasonality", True)
        self.daily_seasonality = self.params.get("daily_seasonality", False)
        self.changepoint_prior_scale = self.params.get("changepoint_prior_scale", 0.05)
        self.model = None
        self._last_date = None

    def fit(self, y_train: np.ndarray) -> None:

        # Remove any NaN values
        y_train = np.array(y_train, dtype=float)
        y_train = y_train[~np.isnan(y_train)]

        # Prophet requires a DataFrame with 'ds' and 'y' columns
        # Generate weekly date range for the training series
        dates = pd.date_range(end="2018-12-31", periods=len(y_train), freq="W")
        self._last_date = dates[-1]

        df = pd.DataFrame({"ds": dates, "y": y_train})
        df["y"] = df["y"].clip(lower=0)

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        self.model.fit(df)
        self.is_fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        future = self.model.make_future_dataframe(periods=horizon, freq="W")
        forecast = self.model.predict(future)
        # Return only the future horizon predictions
        preds = forecast["yhat"].values[-horizon:]
        preds = np.array(preds, dtype=float)
        # Handle NaN values
        preds = np.nan_to_num(preds, nan=0.0)
        return np.clip(preds, 0, None)

    def get_params(self) -> dict:
        return {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
        }