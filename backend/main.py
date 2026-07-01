"""
BenchSCF FastAPI Backend
========================
Serves the React frontend with benchmark results via REST API.
Endpoints:
  GET  /health                  — health check
  GET  /datasets                — list available datasets
  GET  /models                  — list available models
  POST /run-benchmark           — run benchmark, returns results
  GET  /results/{run_id}        — fetch a past run by ID
"""

import json
import os
import sys
import random
import time
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(
    title="BenchSCF API",
    description="Multi-Baseline Benchmarking Framework for Supply Chain Forecasting",
    version="1.0.0",
)

# Allow React frontend (localhost:5173 for Vite, and Vercel URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for run results (use Redis/DB in production)
_run_store = {}


# ── Pydantic Models ──────────────────────────────────────────────

class BenchmarkConfig(BaseModel):
    datasets: List[str]           # e.g. ["dataco", "rossmann"]
    models: List[str]             # e.g. ["sarima", "ets", "xgboost"]
    horizon: int = 4              # forecast horizon in weeks
    cv_folds: int = 5
    train_ratio: float = 0.70
    seed: int = 42
    n_sample_series: int = 10     # how many series per dataset (speed control)


class RunStatusResponse(BaseModel):
    run_id: str
    status: str                   # "running" | "complete" | "error"
    results: Optional[dict] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None

# ── Helpers ──────────────────────────────────────────────────────

def fix_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_base_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def resolve_backend_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def run_benchmark_task(run_id: str, config: BenchmarkConfig):
    """
    Background task: runs the full benchmark and stores results.
    Called asynchronously so the API returns immediately.
    """
    _run_store[run_id]["status"] = "running"
    _run_store[run_id]["logs"] = []
    start = time.time()

    def log(msg: str, level: str = "info"):
        entry = {"ts": datetime.now().isoformat(), "level": level, "msg": msg}
        _run_store[run_id]["logs"].append(entry)
        print(f"[BenchSCF/{run_id}] {msg}")

    try:
        fix_seeds(config.seed)
        base_cfg = load_base_config()

        log(f"Global seed fixed to {config.seed}")
        log(f"Datasets: {', '.join(config.datasets)}")
        log(f"Models: {', '.join(config.models)}")
        log(f"Horizon: {config.horizon} weeks | CV Folds: {config.cv_folds} | Series/dataset: {config.n_sample_series}")

        from evaluation.evaluator import BenchmarkRunner
        from models import MODEL_REGISTRY

        # Build a config dict compatible with BenchmarkRunner
        runner_config = {
            "cv_folds": config.cv_folds,
            "horizon": config.horizon,
            "train_ratio": config.train_ratio,
            "seed": config.seed,
        }
        runner = BenchmarkRunner(runner_config)

        all_results = {}

        for dataset_name in config.datasets:
            dataset_cfg = base_cfg.get("datasets", {}).get(dataset_name, {})
            dataset_cfg["n_sample_series"] = config.n_sample_series

            log(f"Loading dataset: {dataset_name}")
            try:
                adapter, series_keys = _load_dataset(dataset_name, dataset_cfg, config.seed)
                log(f"  {len(series_keys)} series loaded from {dataset_name}")
            except Exception as e:
                log(f"  ERROR loading {dataset_name}: {e}", "error")
                all_results[dataset_name] = {"error": str(e)}
                continue

            all_results[dataset_name] = {}

            for model_name in config.models:
                if model_name not in MODEL_REGISTRY:
                    log(f"  Skipping unknown model: {model_name}", "warn")
                    continue

                log(f"  Running {model_name} on {dataset_name}…")
                model_class = MODEL_REGISTRY[model_name]
                model_params = base_cfg.get("models", {}).get(model_name, {})
                model_params = {k: v for k, v in model_params.items() if k != "enabled"}
                model_params["seed"] = config.seed
                model_params["horizon"] = config.horizon

                t0 = time.time()
                try:
                    metrics = runner.run_model_on_dataset(
                        model_name=model_name,
                        model_class=model_class,
                        model_params=model_params,
                        adapter=adapter,
                        series_keys=series_keys,
                    )
                    elapsed_model = round(time.time() - t0, 1)
                    mae = metrics.get("MAE", {}).get("mean")
                    mae_str = f"{mae:.3f}" if mae is not None and not np.isnan(mae) else "N/A"
                    log(f"  ✓ {model_name} done in {elapsed_model}s — MAE: {mae_str}")
                    all_results[dataset_name][model_name] = metrics
                except Exception as e:
                    log(f"  ✗ {model_name} failed: {e}", "error")
                    all_results[dataset_name][model_name] = {"error": str(e)}

        _run_store[run_id]["status"] = "complete"
        _run_store[run_id]["results"] = all_results
        _run_store[run_id]["elapsed_seconds"] = round(time.time() - start, 2)
        log(f"Benchmark complete in {_run_store[run_id]['elapsed_seconds']}s", "success")

    except Exception as e:
        _run_store[run_id]["status"] = "error"
        _run_store[run_id]["error"] = str(e)
        log(f"Fatal error: {e}", "error")

def _load_dataset(name: str, cfg: dict, seed: int):
    if name == "dataco":
        from data.dataco_adapter import DataCoAdapter
        adapter = DataCoAdapter(
            file_path=resolve_backend_path(cfg.get("file_path", "data/raw/DataCoSupplyChainDataset.csv")),
            seed=seed,
        )
        adapter.load()
        keys = adapter.get_sample_keys(n=cfg.get("n_sample_series", 10))
        return adapter, keys
    elif name == "rossmann":
        from data.rossmann_adapter import RossmannAdapter
        adapter = RossmannAdapter(
            train_path=resolve_backend_path(cfg.get("train_path", "data/raw/rossmann_train.csv")),
            store_path=resolve_backend_path(cfg.get("store_path", "data/raw/rossmann_store.csv")),
            seed=seed,
            n_stores=cfg.get("n_stores", 100),
        )
        adapter.load()
        keys = adapter.get_sample_keys(n=cfg.get("n_sample_series", 10))
        return adapter, keys
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ── Routes ───────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "BenchSCF API", "version": "1.0.0"}


@app.get("/datasets")
def list_datasets():
    return {
        "datasets": [
            {
                "id": "dataco",
                "name": "DataCo Smart Supply Chain",
                "description": "180K+ transaction records, 2015–2018. Aggregated to weekly demand per product-category × market pair.",
                "series_count": "~120",
                "frequency": "Weekly",
            },
            {
                "id": "rossmann",
                "name": "Rossmann Store Sales",
                "description": "Daily store-level sales for 1,115 stores. Used as domain stress-test for cross-domain generalisability.",
                "series_count": "100 stores (stratified sample)",
                "frequency": "Weekly (aggregated)",
            },
        ]
    }


@app.get("/models")
def list_models():
    return {
        "models": [
            {"id": "sarima", "name": "SARIMA", "family": "Classical Statistical", "library": "statsmodels"},
            {"id": "ets", "name": "ETS (Holt-Winters)", "family": "Exponential Smoothing", "library": "statsmodels"},
            {"id": "prophet", "name": "Prophet", "family": "Decomposition", "library": "prophet (Meta)"},
            {"id": "xgboost", "name": "XGBoost", "family": "Gradient Boosting", "library": "xgboost"},
            {"id": "lstm", "name": "LSTM Seq2Seq", "family": "Deep Learning", "library": "keras/tensorflow"},
        ]
    }


@app.post("/run-benchmark", response_model=RunStatusResponse)
def start_benchmark(config: BenchmarkConfig, background_tasks: BackgroundTasks):
    """
    Start a benchmark run asynchronously.
    Returns a run_id immediately. Poll /results/{run_id} for status and results.
    """
    run_id = str(uuid.uuid4())[:8]
    _run_store[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "results": None,
        "error": None,
        "elapsed_seconds": None,
        "config": config.dict(),
        "created_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(run_benchmark_task, run_id, config)

    return RunStatusResponse(run_id=run_id, status="queued")


@app.get("/results/{run_id}", response_model=RunStatusResponse)
def get_results(run_id: str):
    """Poll this endpoint after POST /run-benchmark."""
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")

    run = _run_store[run_id]
    return RunStatusResponse(
        run_id=run["run_id"],
        status=run["status"],
        results=run.get("results"),
        error=run.get("error"),
        elapsed_seconds=run.get("elapsed_seconds"),
    )


@app.get("/results")
def list_results():
    """List all completed runs."""
    return {
        "runs": [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "created_at": r.get("created_at"),
                "elapsed_seconds": r.get("elapsed_seconds"),
            }
            for r in _run_store.values()
        ]
    }


@app.get("/logs/{run_id}")
def get_logs(run_id: str):
    """Fetch server-side log entries for a run."""
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")
    return {"logs": _run_store[run_id].get("logs", [])}
