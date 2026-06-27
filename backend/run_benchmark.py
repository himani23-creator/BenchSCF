"""
BenchSCF — Main Benchmark Runner
=================================
Usage:
    python run_benchmark.py --config config.yaml

This script is the single entry point for all experiments.
It reads config.yaml, loads datasets, runs all enabled models
via rolling-origin CV, and writes results to results/.

Every output is deterministic given the same config.yaml and seed.
"""

import argparse
import json
import os
import sys
import random
import time
import csv
from datetime import datetime

import numpy as np
import yaml

# Fix global seeds immediately — before any imports that use randomness
def fix_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(name: str, dataset_cfg: dict, seed: int):
    """Instantiate and load the correct dataset adapter."""
    if name == "dataco":
        from data.dataco_adapter import DataCoAdapter
        adapter = DataCoAdapter(
            file_path=dataset_cfg["file_path"],
            seed=seed,
        )
        adapter.load()
        keys = adapter.get_sample_keys(n=dataset_cfg.get("n_sample_series", 20))
        return adapter, keys

    elif name == "rossmann":
        from data.rossmann_adapter import RossmannAdapter
        adapter = RossmannAdapter(
            train_path=dataset_cfg["train_path"],
            store_path=dataset_cfg["store_path"],
            seed=seed,
            n_stores=dataset_cfg.get("n_stores", 100),
        )
        adapter.load()
        keys = adapter.get_sample_keys(n=dataset_cfg.get("n_sample_series", 20))
        return adapter, keys

    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_model_class(name: str):
    """Return the model class for a given model name."""
    from models import MODEL_REGISTRY
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def save_results(results: dict, output_cfg: dict, run_id: str):
    """Save results to JSON and CSV."""
    os.makedirs(output_cfg["results_dir"], exist_ok=True)

    if output_cfg.get("save_json", True):
        json_path = os.path.join(output_cfg["results_dir"], f"run_{run_id}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Results] JSON saved to {json_path}")

    if output_cfg.get("save_csv", True):
        csv_path = os.path.join(output_cfg["results_dir"], f"run_{run_id}.csv")
        rows = []
        for dataset_name, dataset_results in results["results"].items():
            for model_name, metrics in dataset_results.items():
                row = {"dataset": dataset_name, "model": model_name}
                for metric, vals in metrics.items():
                    row[f"{metric}_mean"] = round(vals["mean"], 4) if vals["mean"] is not None else "N/A"
                    row[f"{metric}_std"] = round(vals["std"], 4) if vals["std"] is not None else "N/A"
                rows.append(row)

        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"[Results] CSV saved to {csv_path}")


def print_results_table(results: dict):
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 80)
    print("BENCHSCF RESULTS — Mean ± Std across Rolling-Origin CV Folds")
    print("=" * 80)

    for dataset_name, dataset_results in results["results"].items():
        print(f"\nDataset: {dataset_name.upper()}")
        print(f"{'Model':<12} {'MAE':>14} {'RMSE':>14} {'MAPE':>14} {'R²':>14}")
        print("-" * 70)

        for model_name, metrics in dataset_results.items():
            def fmt(m):
                if metrics[m]["mean"] is None or np.isnan(metrics[m]["mean"]):
                    return "     N/A"
                return f"{metrics[m]['mean']:>6.3f}±{metrics[m]['std']:>5.3f}"

            print(f"{model_name:<12} {fmt('MAE'):>14} {fmt('RMSE'):>14} {fmt('MAPE'):>14} {fmt('R2'):>14}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="BenchSCF Benchmark Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    print(f"\n[BenchSCF] Loading config from: {args.config}")
    config = load_config(args.config)

    seed = config.get("seed", 42)
    fix_seeds(seed)
    print(f"[BenchSCF] Global seed fixed to {seed}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    results = {
        "run_id": run_id,
        "config": config,
        "results": {},
    }

    from evaluation.evaluator import BenchmarkRunner
    runner = BenchmarkRunner(config)

    # ── Iterate over datasets ────────────────────────────────────
    for dataset_name, dataset_cfg in config["datasets"].items():
        if not dataset_cfg.get("enabled", True):
            print(f"[BenchSCF] Skipping dataset: {dataset_name} (disabled in config)")
            continue

        print(f"\n[BenchSCF] Loading dataset: {dataset_name}")
        try:
            adapter, series_keys = load_dataset(dataset_name, dataset_cfg, seed)
        except FileNotFoundError as e:
            print(f"[BenchSCF] WARNING: {e} — skipping {dataset_name}")
            continue

        print(f"[BenchSCF] Running benchmark on {len(series_keys)} series from {dataset_name}")
        results["results"][dataset_name] = {}

        # ── Iterate over models ──────────────────────────────────
        for model_name, model_cfg in config["models"].items():
            if not model_cfg.get("enabled", True):
                print(f"  [Skip] {model_name} disabled in config")
                continue

            print(f"  [Running] {model_name} on {dataset_name}...")
            model_class = get_model_class(model_name)
            model_params = {k: v for k, v in model_cfg.items() if k != "enabled"}
            model_params["seed"] = seed

            t0 = time.time()
            metrics = runner.run_model_on_dataset(
                model_name=model_name,
                model_class=model_class,
                model_params=model_params,
                adapter=adapter,
                series_keys=series_keys,
            )
            elapsed = time.time() - t0

            results["results"][dataset_name][model_name] = metrics
            print(f"  [Done] {model_name} in {elapsed:.1f}s — MAE: {metrics['MAE']['mean']:.3f} ± {metrics['MAE']['std']:.3f}")

    # ── Save and display ─────────────────────────────────────────
    print_results_table(results)
    results["total_time_seconds"] = round(time.time() - start_time, 2)
    save_results(results, config.get("output", {"results_dir": "results/"}), run_id)

    print(f"\n[BenchSCF] Benchmark complete in {results['total_time_seconds']}s. Run ID: {run_id}")


if __name__ == "__main__":
    main()