"""
BenchSCF Model Registry
========================
Maps config.yaml model keys → BaseForecaster subclasses.
Adding a new model: subclass BaseForecaster, add one entry here. Zero eval changes.

Imports are lazy so the framework runs even if optional dependencies (xgboost,
tensorflow, prophet) are not installed — those models will simply be unavailable.
"""

MODEL_REGISTRY = {}

def _try_import(name, module_path, class_name):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        MODEL_REGISTRY[name] = getattr(mod, class_name)
    except ImportError as e:
        import warnings
        warnings.warn(f"[BenchSCF] Model '{name}' unavailable: {e}")

_try_import("sarima",   "models.sarima_model",   "SARIMAForecaster")
_try_import("ets",      "models.ets_model",      "ETSForecaster")
_try_import("prophet",  "models.prophet_model",  "ProphetForecaster")
_try_import("xgboost",  "models.xgboost_model",  "XGBoostForecaster")
_try_import("lstm",     "models.lstm_model",     "LSTMForecaster")

__all__ = ["MODEL_REGISTRY"]
