import pandas as pd
import numpy as np
import os


class RossmannAdapter:
    """
    Adapter for the Rossmann Store Sales dataset (Kaggle).
    Uses daily store-level sales aggregated to weekly.
    Stratified sample of 100 stores preserving store-type distribution.
    Merges store.csv for promo and store-type features (informational, not fed to models).
    """

    def __init__(self, train_path: str, store_path: str, seed: int = 42, n_stores: int = 100):
        self.train_path = train_path
        self.store_path = store_path
        self.seed = seed
        self.n_stores = n_stores
        self._series_map = {}
        self._norm_stats = {}

    def load(self) -> None:
        """Load and preprocess Rossmann data."""
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Rossmann train file not found: {self.train_path}")
        if not os.path.exists(self.store_path):
            raise FileNotFoundError(f"Rossmann store file not found: {self.store_path}")

        train_df = pd.read_csv(self.train_path, low_memory=False)
        store_df = pd.read_csv(self.store_path)

        train_df["Date"] = pd.to_datetime(train_df["Date"])
        train_df = train_df[train_df["Open"] == 1]  # Only open days

        # Merge store metadata
        df = train_df.merge(store_df, on="Store", how="left")

        # Stratified sample: preserve StoreType distribution
        store_types = store_df.set_index("Store")["StoreType"]
        np.random.seed(self.seed)

        sampled_stores = []
        for stype, group in store_df.groupby("StoreType"):
            store_ids = group["Store"].values
            n_sample = max(1, int(self.n_stores * len(store_ids) / len(store_df)))
            n_sample = min(n_sample, len(store_ids))
            sampled = np.random.choice(store_ids, size=n_sample, replace=False)
            sampled_stores.extend(sampled)

        sampled_stores = sampled_stores[:self.n_stores]
        df = df[df["Store"].isin(sampled_stores)]

        # Aggregate to weekly
        df["week"] = df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly = (
            df.groupby(["Store", "week"])["Sales"]
            .sum()
            .reset_index()
        )

        for store_id, grp in weekly.groupby("Store"):
            series = grp.set_index("week")["Sales"].sort_index()
            full_index = pd.date_range(series.index.min(), series.index.max(), freq="W")
            series = series.reindex(full_index, fill_value=0)
            if len(series) >= 52:
                key = f"store_{store_id}"
                self._series_map[key] = series

        print(f"[RossmannAdapter] Loaded {len(self._series_map)} store series.")

    def get_series_keys(self) -> list:
        return list(self._series_map.keys())

    def get_train_val_test(
        self, series_key: str, train_ratio=0.70, val_ratio=0.10
    ) -> tuple:
        """
        Chronological split with normalisation computed on train slice only.
        Returns: (y_train_norm, y_val_norm, y_test_norm, norm_mean, norm_std)
        """
        series = self._series_map[series_key].values.astype(float)
        n = len(series)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        y_train = series[:train_end]
        y_val = series[train_end:val_end]
        y_test = series[val_end:]

        norm_mean = np.mean(y_train)
        norm_std = np.std(y_train) + 1e-8

        y_train_norm = (y_train - norm_mean) / norm_std
        y_val_norm = (y_val - norm_mean) / norm_std
        y_test_norm = (y_test - norm_mean) / norm_std

        self._norm_stats[series_key] = (norm_mean, norm_std)

        return y_train_norm, y_val_norm, y_test_norm, norm_mean, norm_std

    def denormalise(self, series_key: str, values: np.ndarray) -> np.ndarray:
        mean, std = self._norm_stats[series_key]
        return values * std + mean

    def get_sample_keys(self, n: int = 20) -> list:
        np.random.seed(self.seed)
        keys = self.get_series_keys()
        n = min(n, len(keys))
        return list(np.random.choice(keys, size=n, replace=False))
