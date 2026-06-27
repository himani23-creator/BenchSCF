# BenchSCF — Multi-Baseline Benchmarking Framework for Supply Chain Forecasting

> **Problem Statement 1** · Config-driven, pluggable, fully reproducible benchmark for supply chain demand forecasting.

## Architecture

```
BenchSCF/
├── backend/
│   ├── config.yaml              ← Single config: datasets, models, eval params
│   ├── run_benchmark.py         ← CLI entry: python run_benchmark.py --config config.yaml
│   ├── main.py                  ← FastAPI REST server (for UI)
│   ├── requirements.txt
│   ├── data/
│   │   ├── dataco_adapter.py    ← DataCo Smart Supply Chain (weekly demand)
│   │   └── rossmann_adapter.py  ← Rossmann Store Sales (domain stress-test)
│   ├── models/
│   │   ├── base_forecaster.py   ← Abstract BaseForecaster (fit/predict/get_params)
│   │   ├── sarima_model.py      ← SARIMA via statsmodels
│   │   ├── ets_model.py         ← ETS Holt-Winters via statsmodels
│   │   ├── prophet_model.py     ← Prophet by Meta
│   │   ├── xgboost_model.py     ← XGBoost with lag/rolling features
│   │   └── lstm_model.py        ← LSTM Seq2Seq encoder-decoder (Keras)
│   └── evaluation/
│       └── evaluator.py         ← RollingOriginEvaluator + BenchmarkRunner
└── frontend/                    ← React + Vite UI (AI Agent style)
    ├── src/
    │   ├── pages/Home.jsx       ← Config panel + terminal log + results
    │   └── pages/Results.jsx    ← Past run history
    └── ...
```

## Quickstart

### 1. Environment Setup (conda — Python 3.10)

```bash
conda env create -f environment.yml
conda activate benchscf
```

### 2. Download Datasets

Place the following raw files in `backend/data/raw/`:

| File | Source |
|------|--------|
| `DataCoSupplyChainDataset.csv` | [Kaggle DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) |
| `rossmann_train.csv` | [Kaggle Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) |
| `rossmann_store.csv` | Same as above |

### 3. Run CLI Benchmark

```bash
cd backend
python run_benchmark.py --config config.yaml
```

Results are saved to `backend/results/run_<timestamp>.json` and `.csv`.

### 4. Run UI (React)

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Evaluation Protocol

- **Split**: 70% train / 10% val / 20% test — chronological, no shuffle
- **CV**: 5-fold rolling-origin (expanding window), advancing by one horizon per fold
- **Metrics**: MAE, RMSE, MAPE, R² — reported as mean ± std across folds
- **Normalisation**: μ and σ computed on each fold's training slice only — leakage architecturally prevented
- **Reproducibility**: All seeds fixed (numpy, random, tensorflow) via config `seed` field

## Models

| Model | Family | Library |
|-------|--------|---------|
| SARIMA | Classical Statistical | statsmodels |
| ETS Holt-Winters | Exponential Smoothing | statsmodels |
| Prophet | Decomposition | Meta prophet |
| XGBoost | Gradient Boosting | xgboost (lag t-1..t-12, rolling window 4,8) |
| LSTM Seq2Seq | Deep Learning | Keras/TensorFlow (early stopping on val MAE) |

## Adding a New Model

1. Subclass `BaseForecaster` in `backend/models/`
2. Implement `fit(y_train)`, `predict(horizon)`, `get_params()`
3. Register in `models/__init__.py`
4. Add config entry in `config.yaml`

Zero changes to evaluation logic required.

## Feasibility

Runs on a 16GB RAM machine without GPU. LSTM training on DataCo weekly series completes in under 30 minutes on CPU. The Rossmann 100-store sample keeps runtime tractable.