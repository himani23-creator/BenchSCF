import numpy as np
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, R² between true and predicted values.
    All four metrics required by the BenchSCF evaluation protocol.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE: avoid division by zero
    mask_nonzero = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100 if mask_nonzero.any() else np.nan

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


class RollingOriginEvaluator:
    """
    Rolling-origin (expanding window) cross-validator for time series.
    Each fold advances the origin by one forecast horizon.
    Returns mean ± std across folds for each metric — no single-run point values.

    Leakage is architecturally prevented:
    - Normalisation stats are computed per fold on training slice only
    - Models receive only y_train with no access to future values
    """

    def __init__(self, n_folds: int = 5, horizon: int = 4):
        self.n_folds = n_folds
        self.horizon = horizon

    def run(
        self,
        model_class,
        model_params: dict,
        full_series: np.ndarray,
        train_ratio: float = 0.70,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run rolling-origin CV for a single model on a single series.

        Args:
            model_class: Uninstantiated BaseForecaster subclass
            model_params: Hyperparameters dict (from config.yaml)
            full_series: Full normalised time series (1D array)
            train_ratio: Initial training proportion

        Returns:
            Dict with metrics as keys, each containing 'mean' and 'std' across folds.
        """
        # Remove any NaN values from the series
        full_series = np.array(full_series, dtype=float)
        full_series = full_series[~np.isnan(full_series)]
        
        if len(full_series) < 52:
            return {m: {"mean": np.nan, "std": np.nan} for m in ["MAE", "RMSE", "MAPE", "R2"]}
        
        n = len(full_series)
        initial_train_end = int(n * train_ratio)

        fold_results = []

        for fold in range(self.n_folds):
            # Each fold advances origin by one horizon
            train_end = initial_train_end + fold * self.horizon
            test_start = train_end
            test_end = test_start + self.horizon

            if test_end > n:
                break  # Not enough data for this fold

            y_train = full_series[:train_end]
            y_test = full_series[test_start:test_end]

            # Compute normalisation on THIS fold's training slice only
            fold_mean = np.mean(y_train)
            fold_std = np.std(y_train) + 1e-8

            y_train_norm = (y_train - fold_mean) / fold_std
            y_test_norm = (y_test - fold_mean) / fold_std  # same stats

            try:
                # Instantiate fresh model per fold — no state leakage between folds
                params = {**model_params, "horizon": self.horizon}
                model = model_class(params=params)
                model.fit(y_train_norm)
                y_pred_norm = model.predict(self.horizon)

                # Denormalise both for metric computation in original scale
                y_pred = y_pred_norm * fold_std + fold_mean
                y_true = y_test  # original scale

                metrics = compute_metrics(y_true, y_pred)
                fold_results.append(metrics)

            except Exception as e:
                print(f"[Evaluator] Fold {fold} failed: {e}")
                continue

        if not fold_results:
            return {m: {"mean": np.nan, "std": np.nan} for m in ["MAE", "RMSE", "MAPE", "R2"]}

        # Aggregate across folds: mean ± std
        aggregated = {}
        for metric in ["MAE", "RMSE", "MAPE", "R2"]:
            values = [f[metric] for f in fold_results if not np.isnan(f[metric])]
            aggregated[metric] = {
                "mean": float(np.mean(values)) if values else np.nan,
                "std": float(np.std(values)) if len(values) > 1 else 0.0,
                "n_folds": len(values),
            }

        return aggregated


class BenchmarkRunner:
    """
    Orchestrates the full benchmark: iterates over (dataset, model, series) triples.
    Each combination is independent — results are aggregated across series then across folds.
    """

    def __init__(self, config: dict):
        self.config = config
        self.n_folds = config.get("cv_folds", 5)
        self.horizon = config.get("horizon", 4)
        self.train_ratio = config.get("train_ratio", 0.70)
        self.evaluator = RollingOriginEvaluator(
            n_folds=self.n_folds, horizon=self.horizon
        )

    def run_model_on_dataset(
        self,
        model_name: str,
        model_class,
        model_params: dict,
        adapter,
        series_keys: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Run one model across all series in a dataset.
        Returns metrics averaged across series, with std reflecting series-level variance.
        """
        all_series_metrics = []

        for key in series_keys:
            try:
                y_train_norm, _, y_test_norm, norm_mean, norm_std = adapter.get_train_val_test(
                    key, train_ratio=self.train_ratio
                )
                # Use raw (unnormalised) series — evaluator handles per-fold normalisation
                raw_series = adapter._series_map[key].values.astype(float)
                # Remove any NaN values from the series
                raw_series = raw_series[~np.isnan(raw_series)]
                
                if len(raw_series) < 52:
                    continue  # Skip series that are too short

                fold_metrics = self.evaluator.run(
                    model_class, model_params, raw_series, self.train_ratio
                )
                all_series_metrics.append(fold_metrics)

            except Exception as e:
                print(f"[BenchmarkRunner] Skipping series {key} for {model_name}: {e}")
                continue

        if not all_series_metrics:
            return {m: {"mean": np.nan, "std": np.nan} for m in ["MAE", "RMSE", "MAPE", "R2"]}

        # Average metrics across all series
        final = {}
        for metric in ["MAE", "RMSE", "MAPE", "R2"]:
            means = [s[metric]["mean"] for s in all_series_metrics if not np.isnan(s[metric]["mean"])]
            final[metric] = {
                "mean": float(np.mean(means)) if means else np.nan,
                "std": float(np.std(means)) if len(means) > 1 else 0.0,
            }

        return final