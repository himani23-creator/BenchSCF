import { useState, useEffect } from 'react'
import { BarChart3, RefreshCw, Download, Clock, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import ResultsTable from '../components/ResultsTable.jsx'
import ComparisonChart from '../components/ComparisonChart.jsx'

const API_BASE = '/api'

const MODELS = [
  { id: 'sarima',   name: 'SARIMA',           family: 'Classical Statistical',  lib: 'statsmodels',    color: '#00e5ff' },
  { id: 'ets',      name: 'ETS Holt-Winters', family: 'Exponential Smoothing',  lib: 'statsmodels',    color: '#7c4dff' },
  { id: 'prophet',  name: 'Prophet',          family: 'Decomposition',          lib: 'Meta Prophet',   color: '#00e676' },
  { id: 'xgboost',  name: 'XGBoost',          family: 'Gradient Boosting',      lib: 'xgboost',        color: '#ffab40' },
  { id: 'lstm',     name: 'LSTM Seq2Seq',     family: 'Deep Learning',          lib: 'Keras/TF',       color: '#ff4081' },
]

function StatusIcon({ status }) {
  if (status === 'complete') return <CheckCircle2 size={14} style={{ color: 'var(--neon-green)' }} />
  if (status === 'error')    return <AlertCircle  size={14} style={{ color: 'var(--neon-red)' }} />
  if (status === 'running' || status === 'queued') return <Loader2 size={14} style={{ color: 'var(--neon-amber)', animation: 'spin 1s linear infinite' }} />
  return null
}

export default function ResultsPage() {
  const [runs, setRuns]         = useState([])
  const [loading, setLoading]   = useState(true)
  const [selected, setSelected] = useState(null)
  const [detail, setDetail]     = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('table')

  const fetchRuns = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/results`)
      if (!res.ok) throw new Error('API unavailable')
      const data = await res.json()
      const sorted = (data.runs || []).sort((a, b) =>
        new Date(b.created_at || 0) - new Date(a.created_at || 0)
      )
      setRuns(sorted)
    } catch (e) {
      setRuns([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchRuns() }, [])

  const loadDetail = async (runId) => {
    setSelected(runId)
    setDetailLoading(true)
    setDetail(null)
    try {
      const res = await fetch(`${API_BASE}/results/${runId}`)
      const data = await res.json()
      setDetail(data)
    } catch {
      setDetail(null)
    } finally {
      setDetailLoading(false)
    }
  }

  const downloadRun = (run) => {
    if (!detail?.results) return
    const blob = new Blob(
      [JSON.stringify({ run_id: run.run_id, ...detail }, null, 2)],
      { type: 'application/json' }
    )
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `benchscf_run_${run.run_id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div className="page-eyebrow">
          <BarChart3 size={12} /> Experiment History
        </div>
        <h1 className="page-title">Past Runs</h1>
        <p className="page-subtitle">
          Every run is logged with its config, metrics, and timestamp. Deterministic reproduction
          via <code className="text-mono" style={{ color: 'var(--neon-cyan)', fontSize: 14 }}>python run_benchmark.py --config config.yaml</code>.
        </p>
      </div>

      <div className="layout-sidebar">
        {/* Run list */}
        <div>
          <div className="card">
            <div className="card-header">
              <div className="card-title">Runs</div>
              <button className="btn btn-ghost btn-sm" onClick={fetchRuns}>
                <RefreshCw size={12} /> Refresh
              </button>
            </div>

            {loading && (
              <div className="empty-state">
                <div className="spinner spinner-lg" style={{ margin: '0 auto 16px' }} />
                <div className="empty-state-text">Loading runs…</div>
              </div>
            )}

            {!loading && runs.length === 0 && (
              <div className="empty-state">
                <BarChart3 size={36} className="empty-state-icon" />
                <div className="empty-state-text">
                  No runs yet.<br />Run a benchmark from the home page.
                </div>
              </div>
            )}

            {!loading && runs.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {runs.map(run => (
                  <div
                    key={run.run_id}
                    className={`toggle-item ${selected === run.run_id ? 'selected' : ''}`}
                    onClick={() => loadDetail(run.run_id)}
                    id={`run-${run.run_id}`}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="toggle-label">
                      <div className="flex items-center gap-2">
                        <StatusIcon status={run.status} />
                        <span className="toggle-name" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                          {run.run_id}
                        </span>
                      </div>
                      <span className="toggle-desc">
                        {run.created_at ? new Date(run.created_at).toLocaleString() : 'Unknown time'}
                        {run.elapsed_seconds ? ` · ${run.elapsed_seconds}s` : ''}
                      </span>
                    </div>
                    <span
                      className={`badge badge-${
                        run.status === 'complete' ? 'green'
                        : run.status === 'error' ? 'red'
                        : 'amber'
                      }`}
                    >
                      {run.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Run detail */}
        <div>
          {!selected && (
            <div className="card" style={{ padding: '60px 32px', textAlign: 'center' }}>
              <BarChart3 size={40} style={{ margin: '0 auto 16px', color: 'var(--text-dim)' }} />
              <div style={{ color: 'var(--text-muted)', fontSize: 14 }}>
                Select a run to view detailed results
              </div>
            </div>
          )}

          {selected && detailLoading && (
            <div className="card" style={{ padding: '60px 32px', textAlign: 'center' }}>
              <div className="spinner spinner-lg" style={{ margin: '0 auto 16px' }} />
              <div style={{ color: 'var(--text-muted)', fontSize: 14 }}>Loading results…</div>
            </div>
          )}

          {selected && !detailLoading && detail && (
            <div className="card fade-in">
              <div className="card-header">
                <div className="card-title">
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--neon-cyan)' }}>
                    {selected}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="tab-bar">
                    <button className={`tab-btn ${activeTab === 'table' ? 'active' : ''}`} onClick={() => setActiveTab('table')}>Table</button>
                    <button className={`tab-btn ${activeTab === 'chart' ? 'active' : ''}`} onClick={() => setActiveTab('chart')}>Charts</button>
                  </div>
                  {detail.results && (
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => downloadRun(runs.find(r => r.run_id === selected))}
                    >
                      <Download size={12} />
                    </button>
                  )}
                </div>
              </div>

              {detail.status === 'error' && (
                <div style={{ padding: '16px', background: 'rgba(255,82,82,0.07)', borderRadius: 'var(--radius-md)', color: 'var(--neon-red)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>
                  Error: {detail.error}
                </div>
              )}

              {(detail.status === 'running' || detail.status === 'queued') && (
                <div className="flex items-center gap-3" style={{ padding: 16 }}>
                  <div className="spinner" />
                  <span style={{ color: 'var(--neon-amber)', fontSize: 13 }}>Run in progress…</span>
                </div>
              )}

              {detail.status === 'complete' && detail.results && activeTab === 'table' && (
                <ResultsTable results={detail.results} models={MODELS} />
              )}

              {detail.status === 'complete' && detail.results && activeTab === 'chart' && (
                <ComparisonChart results={detail.results} models={MODELS} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
