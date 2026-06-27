from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models in BenchSCF.
    Every model must implement fit(), predict(), and get_params().
    Adding a new model = subclass this + update config.yaml. Zero eval logic changes.
    """

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, y_train: np.ndarray) -> None:
        """
        Train the model on the provided training series.
        Normalisation has already been applied upstream — do not re-normalise here.
        Args:
            y_train: 1D numpy array of training values (normalised)
        """
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts for the given horizon.
        Args:
            horizon: Number of steps ahead to forecast
        Returns:
            1D numpy array of length `horizon`
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return model hyperparameters for logging to experiment record.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.get_params()})"