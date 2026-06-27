import { useState, useEffect, useRef, useCallback } from 'react'
import { Play, Square, RefreshCw, Database, Cpu, Settings2, Terminal, ChevronRight, Check, AlertTriangle, Info, Download } from 'lucide-react'
import ResultsTable from '../components/ResultsTable.jsx'
import ComparisonChart from '../components/ComparisonChart.jsx'

const API_BASE = '/api'

const DATASETS = [
  {
    id: 'dataco',
    name: 'DataCo Smart Supply Chain',
    desc: 'Weekly demand · ~120 series · 2015–2018',
    tag: '180K+ records',
    color: 'var(--neon-cyan)',
  },
  {
    id: 'rossmann',
    name: 'Rossmann Store Sales',
    desc: 'Weekly store sales · 100 stores (stratified)',
    tag: 'Domain stress-test',
    color: 'var(--neon-purple)',
  },
]

const MODELS = [
  { id: 'sarima',   name: 'SARIMA',           family: 'Classical Statistical',  lib: 'statsmodels',    color: '#00e5ff' },
  { id: 'ets',      name: 'ETS Holt-Winters', family: 'Exponential Smoothing',  lib: 'statsmodels',    color: '#7c4dff' },
  { id: 'prophet',  name: 'Prophet',          family: 'Decomposition',          lib: 'Meta Prophet',   color: '#00e676' },
  { id: 'xgboost',  name: 'XGBoost',          family: 'Gradient Boosting',      lib: 'xgboost',        color: '#ffab40' },
  { id: 'lstm',     name: 'LSTM Seq2Seq',     family: 'Deep Learning',          lib: 'Keras/TF',       color: '#ff4081' },
]

function useLogStream() {
  const [logs, setLogs] = useState([])
  const append = useCallback((type, text) => {
    setLogs(prev => [...prev, { type, text, ts: Date.now() }])
  }, [])
  const clear = useCallback(() => setLogs([]), [])
  return { logs, append, clear }
}

export default function HomePage({ apiStatus, setApiStatus }) {
  // Config state
  const [selectedDatasets, setSelectedDatasets] = useState(['dataco'])
  const [selectedModels, setSelectedModels]     = useState(['sarima', 'ets', 'prophet', 'xgboost'])
  const [horizon, setHorizon]                   = useState(4)
  const [cvFolds, setCvFolds]                   = useState(5)
  const [nSeries, setNSeries]                   = useState(10)
  const [seed, setSeed]                         = useState(42)

  // Run state
  const [runId, setRunId]         = useState(null)
  const [runStatus, setRunStatus] = useState('idle')  // idle | queued | running | complete | error
  const [results, setResults]     = useState(null)
  const [elapsed, setElapsed]     = useState(null)
  const [error, setError]         = useState(null)
  const [activeTab, setActiveTab] = useState('table')

  const { logs, append, clear } = useLogStream()
  const logRef = useRef(null)
  const pollRef = useRef(null)

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  // Cleanup on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const toggleDataset = (id) => {
    setSelectedDatasets(prev =>
      prev.includes(id) ? prev.filter(d => d !== id) : [...prev, id]
    )
  }

  const toggleModel = (id) => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    )
  }

  const pollResults = useCallback((id) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/results/${id}`)
        const data = await res.json()

        if (data.status === 'complete') {
          clearInterval(pollRef.current)
          setRunStatus('complete')
          setResults(data.results)
          setElapsed(data.elapsed_seconds)
          append('success', `✓ Benchmark complete in ${data.elapsed_seconds}s`)
          append('success', `✓ Run ID: ${id}`)
        } else if (data.status === 'error') {
          clearInterval(pollRef.current)
          setRunStatus('error')
          setError(data.error)
          append('error', `✗ Error: ${data.error}`)
        } else {
          append('dim', `  … ${data.status} (polling run ${id})`)
        }
      } catch (e) {
        append('warn', `  ⚠ Poll failed: ${e.message}`)
      }
    }, 3000)
  }, [append])

  const runBenchmark = async () => {
    if (selectedDatasets.length === 0) return append('error', '✗ Select at least one dataset.')
    if (selectedModels.length === 0)   return append('error', '✗ Select at least one model.')

    clear()
    setRunStatus('queued')
    setResults(null)
    setError(null)
    setElapsed(null)

    append('cmd', `$ python run_benchmark.py --config config.yaml`)
    append('info', `  Datasets : ${selectedDatasets.join(', ')}`)
    append('info', `  Models   : ${selectedModels.join(', ')}`)
    append('info', `  Horizon  : ${horizon} weeks  |  CV Folds: ${cvFolds}  |  Series: ${nSeries}  |  Seed: ${seed}`)
    append('info', `  Fixing global seed → ${seed}`)
    append('dim',  `  Submitting benchmark run…`)

    try {
      const res = await fetch(`${API_BASE}/run-benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasets: selectedDatasets,
          models: selectedModels,
          horizon,
          cv_folds: cvFolds,
          n_sample_series: nSeries,
          seed,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'API error')
      }

      const data = await res.json()
      setRunId(data.run_id)
      setRunStatus('running')
      append('success', `✓ Run queued — ID: ${data.run_id}`)
      append('info', `  Rolling-origin CV (${cvFolds} folds × ${horizon}-week horizon)`)
      append('dim',  `  Running evaluation… this may take several minutes on first run.`)
      append('dim',  `  (LSTM ~30 min CPU, others much faster)`)
      pollResults(data.run_id)
    } catch (e) {
      setRunStatus('error')
      setError(e.message)
      append('error', `✗ ${e.message}`)
    }
  }

  const stopRun = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    setRunStatus('idle')
    append('warn', '  ⚠ Run polling stopped by user.')
  }

  const downloadResults = () => {
    if (!results) return
    const blob = new Blob([JSON.stringify({ run_id: runId, results, elapsed_seconds: elapsed }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `benchscf_run_${runId}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const isRunning = runStatus === 'running' || runStatus === 'queued'
  const canRun = apiStatus === 'online' && !isRunning

  return (
    <div className="fade-in">
      {/* ── Page Header ─────────────────────────────────────── */}
      <div className="page-header">
        <div className="page-eyebrow">
          <Terminal size={12} /> BenchSCF Agent
        </div>
        <h1 className="page-title">
          Multi-Baseline Benchmarking<br />
          <span style={{ background: 'var(--grad-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
            Framework
          </span>
        </h1>
        <p className="page-subtitle">
          Config-driven, pluggable evaluation framework for supply chain forecasting.
          Five-fold rolling-origin CV · Leakage-free by architecture · MAE, RMSE, MAPE, R².
        </p>
      </div>

      {/* ── API Offline Banner ───────────────────────────────── */}
      {apiStatus === 'offline' && (
        <div className="card mb-6 fade-in" style={{ borderColor: 'rgba(255,171,64,0.3)', background: 'rgba(255,171,64,0.05)' }}>
          <div className="flex items-center gap-3">
            <AlertTriangle size={16} style={{ color: 'var(--neon-amber)', flexShrink: 0 }} />
            <div>
              <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--neon-amber)', marginBottom: 2 }}>
                Backend API Offline
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                Start the FastAPI server: <code className="text-mono" style={{ color: 'var(--neon-cyan)', fontSize: 11 }}>cd backend && uvicorn main:app --reload --port 8000</code>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="layout-sidebar">
        {/* ── LEFT: Config Panel ─────────────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Datasets */}
          <div className="card slide-up stagger-1">
            <div className="card-header">
              <div className="card-title">
                <Database size={14} style={{ color: 'var(--neon-cyan)' }} />
                Datasets
              </div>
              <span className="badge badge-cyan">{selectedDatasets.length} selected</span>
            </div>
            <div className="toggle-group">
              {DATASETS.map(ds => (
                <div
                  key={ds.id}
                  className={`toggle-item ${selectedDatasets.includes(ds.id) ? 'selected' : ''}`}
                  onClick={() => toggleDataset(ds.id)}
                  id={`dataset-${ds.id}`}
                >
                  <div className="toggle-label">
                    <div className="flex items-center gap-2">
                      <div style={{ width: 6, height: 6, borderRadius: '50%', background: ds.color, flexShrink: 0 }} />
                      <span className="toggle-name">{ds.name}</span>
                    </div>
                    <span className="toggle-desc">{ds.desc}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="badge badge-cyan" style={{ fontSize: 10 }}>{ds.tag}</span>
                    <div className={`toggle-check ${selectedDatasets.includes(ds.id) ? 'checked' : ''}`}>
                      {selectedDatasets.includes(ds.id) && <Check size={11} color="#000" />}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Models */}
          <div className="card slide-up stagger-2">
            <div className="card-header">
              <div className="card-title">
                <Cpu size={14} style={{ color: 'var(--neon-cyan)' }} />
                Models
              </div>
              <span className="badge badge-purple">{selectedModels.length} selected</span>
            </div>
            <div className="toggle-group">
              {MODELS.map(m => (
                <div
                  key={m.id}
                  className={`toggle-item ${selectedModels.includes(m.id) ? 'selected' : ''}`}
                  onClick={() => toggleModel(m.id)}
                  id={`model-${m.id}`}
                >
                  <div className="toggle-label">
                    <div className="flex items-center gap-2">
                      <div style={{ width: 6, height: 6, borderRadius: '50%', background: m.color, flexShrink: 0 }} />
                      <span className="toggle-name">{m.name}</span>
                    </div>
                    <span className="toggle-desc">{m.family} · {m.lib}</span>
                  </div>
                  <div className={`toggle-check ${selectedModels.includes(m.id) ? 'checked' : ''}`}>
                    {selectedModels.includes(m.id) && <Check size={11} color="#000" />}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Config params */}
          <div className="card slide-up stagger-3">
            <div className="card-header">
              <div className="card-title">
                <Settings2 size={14} style={{ color: 'var(--neon-cyan)' }} />
                Parameters
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

              <div className="form-group">
                <label className="form-label">Forecast Horizon (weeks) — {horizon}</label>
                <input
                  id="param-horizon"
                  type="range" min={1} max={12} value={horizon}
                  className="form-range"
                  onChange={e => setHorizon(+e.target.value)}
                />
                <div className="flex justify-between text-sm text-muted">
                  <span>1w</span><span>12w</span>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">CV Folds (rolling-origin) — {cvFolds}</label>
                <input
                  id="param-cv-folds"
                  type="range" min={2} max={10} value={cvFolds}
                  className="form-range"
                  onChange={e => setCvFolds(+e.target.value)}
                />
                <div className="flex justify-between text-sm text-muted">
                  <span>2</span><span>10</span>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Series per Dataset — {nSeries}</label>
                <input
                  id="param-n-series"
                  type="range" min={2} max={30} value={nSeries}
                  className="form-range"
                  onChange={e => setNSeries(+e.target.value)}
                />
                <div className="flex justify-between text-sm text-muted">
                  <span>2 (fast)</span><span>30 (full)</span>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Random Seed</label>
                <input
                  id="param-seed"
                  type="number" value={seed}
                  className="form-input"
                  style={{ fontFamily: 'var(--font-mono)' }}
                  onChange={e => setSeed(+e.target.value)}
                />
              </div>

              <div
                style={{
                  padding: '10px 12px',
                  background: 'rgba(0,229,255,0.04)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: 11,
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  lineHeight: 1.6,
                }}
              >
                <span style={{ color: 'var(--neon-cyan)' }}>70/10/20</span> train/val/test split · chronological<br />
                Normalisation computed on training slice only.<br />
                All seeds fixed → numpy, random, tensorflow.
              </div>
            </div>
          </div>

          {/* Run button */}
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              id="btn-run-benchmark"
              className="btn btn-primary btn-lg full-width"
              onClick={runBenchmark}
              disabled={!canRun}
            >
              {isRunning ? (
                <><div className="spinner" /><span>Running…</span></>
              ) : (
                <><Play size={16} /><span>Run Benchmark</span></>
              )}
            </button>
            {isRunning && (
              <button className="btn btn-danger btn-icon" onClick={stopRun} title="Stop polling">
                <Square size={16} />
              </button>
            )}
          </div>
        </div>

        {/* ── RIGHT: Output Panel ────────────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Terminal log */}
          <div className="terminal slide-up stagger-1">
            <div className="terminal-header">
              <div className="terminal-dot terminal-dot-red" />
              <div className="terminal-dot terminal-dot-yellow" />
              <div className="terminal-dot terminal-dot-green" />
              <span className="terminal-title">benchscf · agent log</span>
              {runId && <span className="badge badge-cyan" style={{ marginLeft: 8 }}>RUN: {runId}</span>}
            </div>
            <div className="terminal-body" ref={logRef}>
              {logs.length === 0 && (
                <div className="terminal-line">
                  <span className="terminal-prompt">$</span>
                  <span className="terminal-dim">Configure datasets and models, then click Run Benchmark</span>
                  <span className="terminal-cursor" />
                </div>
              )}
              {logs.map((l, i) => (
                <div key={i} className="terminal-line">
                  <span className={`terminal-${l.type}`}>{l.text}</span>
                </div>
              ))}
              {isRunning && (
                <div className="terminal-line">
                  <span className="terminal-prompt">$</span>
                  <span className="terminal-dim">Evaluating</span>
                  <span className="terminal-cursor" />
                </div>
              )}
            </div>
          </div>

          {/* Status bar */}
          {runStatus !== 'idle' && (
            <div className="card fade-in" style={{ padding: '16px 20px' }}>
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-10">
                  <div>
                    <div className="text-sm text-muted">Status</div>
                    <div style={{ fontSize: 13, fontWeight: 700, marginTop: 2 }}>
                      <span
                        style={{
                          color: runStatus === 'complete' ? 'var(--neon-green)'
                               : runStatus === 'error'    ? 'var(--neon-red)'
                               : 'var(--neon-amber)',
                        }}
                      >
                        {runStatus.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  {elapsed && (
                    <div>
                      <div className="text-sm text-muted">Wall time</div>
                      <div style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--neon-cyan)', marginTop: 2 }}>
                        {elapsed}s
                      </div>
                    </div>
                  )}
                  {runId && (
                    <div>
                      <div className="text-sm text-muted">Run ID</div>
                      <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginTop: 2 }}>
                        {runId}
                      </div>
                    </div>
                  )}
                </div>
                {results && (
                  <button className="btn btn-secondary btn-sm" onClick={downloadResults}>
                    <Download size={13} />
                    Export JSON
                  </button>
                )}
              </div>

              {isRunning && (
                <div className="mt-3">
                  <div className="progress-track">
                    <div className="progress-fill progress-bar-animated" style={{ width: '100%' }} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error display */}
          {runStatus === 'error' && error && (
            <div className="card fade-in" style={{ borderColor: 'rgba(255,82,82,0.3)', background: 'rgba(255,82,82,0.05)' }}>
              <div className="flex items-center gap-3">
                <AlertTriangle size={16} style={{ color: 'var(--neon-red)', flexShrink: 0 }} />
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--neon-red)' }}>Run Failed</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 4, lineHeight: 1.5 }}>
                    {error}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {runStatus === 'complete' && results && (
            <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Results</div>
                  <div className="tab-bar">
                    <button className={`tab-btn ${activeTab === 'table' ? 'active' : ''}`} onClick={() => setActiveTab('table')}>
                      Table
                    </button>
                    <button className={`tab-btn ${activeTab === 'chart' ? 'active' : ''}`} onClick={() => setActiveTab('chart')}>
                      Charts
                    </button>
                  </div>
                </div>

                {activeTab === 'table' && (
                  <ResultsTable results={results} models={MODELS} />
                )}
                {activeTab === 'chart' && (
                  <ComparisonChart results={results} models={MODELS} />
                )}
              </div>

              {/* Interpretation note */}
              <div className="card" style={{ padding: '14px 18px', background: 'rgba(0,229,255,0.03)' }}>
                <div className="flex items-center gap-2 mb-2">
                  <Info size={13} style={{ color: 'var(--neon-cyan)' }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--neon-cyan)', textTransform: 'uppercase', letterSpacing: '0.6px' }}>
                    Interpretation
                  </span>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                  All metrics are <strong style={{ color: 'var(--text-secondary)' }}>mean ± std across rolling-origin CV folds</strong>.
                  Reported std reflects uncertainty over time, not single-point estimates. Lower MAE, RMSE, MAPE is better;
                  higher R² is better. <span style={{ color: 'var(--neon-green)' }}>Green</span> highlights best per metric per dataset.
                  Normalisation is computed per fold on training slice only — leakage is architecturally prevented.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
