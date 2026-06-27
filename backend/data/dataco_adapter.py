import pandas as pd
import numpy as np
import os


class DataCoAdapter:
    """
    Adapter for the DataCo Smart Supply Chain dataset.
    Aggregates transaction-level data to weekly demand series
    per product_category × market pair.
    Applies deterministic preprocessing — all stats computed on training slice only.
    """

    REQUIRED_COLUMNS = ["Order Date", "Order Item Quantity", "Category Name", "Market"]

    def __init__(self, file_path: str, seed: int = 42):
        self.file_path = file_path
        self.seed = seed
        self._raw = None
        self._series_map = {}      # key: (category, market) → pd.Series
        self._norm_stats = {}      # key: series_key → (mean, std) computed on train only

    def load(self) -> None:
        """Load and parse the raw DataCo CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"DataCo file not found at: {self.file_path}")

        df = pd.read_csv(self.file_path, encoding="latin-1", low_memory=False)

        # Validate required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataCo dataset: {missing}")

        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        df.dropna(subset=["Order Date", "Order Item Quantity"], inplace=True)
        df["week"] = df["Order Date"].dt.to_period("W").apply(lambda r: r.start_time)

        # Aggregate to weekly demand per category × market
        grouped = (
            df.groupby(["Category Name", "Market", "week"])["Order Item Quantity"]
            .sum()
            .reset_index()
        )

        for (category, market), grp in grouped.groupby(["Category Name", "Market"]):
            series = grp.set_index("week")["Order Item Quantity"].sort_index()
            # Fill missing weeks with 0
            full_index = pd.date_range(series.index.min(), series.index.max(), freq="W")
            series = series.reindex(full_index, fill_value=0)
            # Only keep series with enough observations
            if len(series) >= 52:
                key = f"{category}__{market}"
                self._series_map[key] = series

        print(f"[DataCoAdapter] Loaded {len(self._series_map)} series from DataCo.")

    def get_series_keys(self) -> list:
        return list(self._series_map.keys())

    def get_train_val_test(
        self, series_key: str, train_ratio=0.70, val_ratio=0.10
    ) -> tuple:
        """
        Split a series chronologically into train, val, test.
        Normalisation stats (mean, std) computed on train slice ONLY.
        Returns: (y_train, y_val, y_test, norm_mean, norm_std)
        """
        series = self._series_map[series_key].values.astype(float)
        n = len(series)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        y_train = series[:train_end]
        y_val = series[train_end:val_end]
        y_test = series[val_end:]

        # Compute normalisation stats on TRAIN only — enforced here at API level
        norm_mean = np.mean(y_train)
        norm_std = np.std(y_train) + 1e-8  # avoid division by zero

        y_train_norm = (y_train - norm_mean) / norm_std
        y_val_norm = (y_val - norm_mean) / norm_std
        y_test_norm = (y_test - norm_mean) / norm_std

        self._norm_stats[series_key] = (norm_mean, norm_std)

        return y_train_norm, y_val_norm, y_test_norm, norm_mean, norm_std

    def denormalise(self, series_key: str, values: np.ndarray) -> np.ndarray:
        """Reverse normalisation using stored train stats."""
        mean, std = self._norm_stats[series_key]
        return values * std + mean

    def get_sample_keys(self, n: int = 20) -> list:
        """Return a stratified sample of series keys for faster benchmark runs."""
        np.random.seed(self.seed)
        keys = self.get_series_keys()
        n = min(n, len(keys))
        return list(np.random.choice(keys, size=n, replace=False))
