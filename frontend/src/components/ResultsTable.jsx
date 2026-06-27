import { useMemo } from 'react'
import { TrendingDown, TrendingUp } from 'lucide-react'

const METRICS = [
  { key: 'MAE',  label: 'MAE',  lowerBetter: true,  desc: 'Mean Absolute Error' },
  { key: 'RMSE', label: 'RMSE', lowerBetter: true,  desc: 'Root Mean Squared Error' },
  { key: 'MAPE', label: 'MAPE %', lowerBetter: true,  desc: 'Mean Absolute Percentage Error' },
  { key: 'R2',   label: 'R²',   lowerBetter: false, desc: 'Coefficient of Determination' },
]

function findBest(datasetResults, metric) {
  const lowerBetter = METRICS.find(m => m.key === metric)?.lowerBetter
  let bestModel = null
  let bestVal = lowerBetter ? Infinity : -Infinity

  for (const [model, metrics] of Object.entries(datasetResults)) {
    if (metrics?.error) continue
    const val = metrics?.[metric]?.mean
    if (val == null || isNaN(val)) continue
    if (lowerBetter ? val < bestVal : val > bestVal) {
      bestVal = val
      bestModel = model
    }
  }
  return bestModel
}

function formatVal(mean, std) {
  if (mean == null || isNaN(mean)) return 'N/A'
  const m = mean.toFixed(3)
  const s = (std ?? 0).toFixed(3)
  return `${m} ±${s}`
}

export default function ResultsTable({ results, models }) {
  const datasets = Object.keys(results || {})

  if (!datasets.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-text">No results to display.</div>
      </div>
    )
  }

  const modelColors = Object.fromEntries((models || []).map(m => [m.id, m.color]))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {datasets.map(dataset => {
        const dsResults = results[dataset]
        if (dsResults?.error) {
          return (
            <div key={dataset}>
              <div style={{ fontWeight: 700, marginBottom: 10, color: 'var(--text-primary)', fontSize: 14 }}>
                {dataset.toUpperCase()}
              </div>
              <div style={{ color: 'var(--neon-red)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>
                Error: {dsResults.error}
              </div>
            </div>
          )
        }

        const modelNames = Object.keys(dsResults || {})
        const bestPerMetric = Object.fromEntries(
          METRICS.map(m => [m.key, findBest(dsResults, m.key)])
        )

        return (
          <div key={dataset}>
            <div className="flex items-center gap-10 mb-3">
              <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
                {dataset === 'dataco' ? 'DataCo Smart Supply Chain' : 'Rossmann Store Sales'}
              </div>
              <span className="badge badge-cyan">{dataset.toUpperCase()}</span>
            </div>

            <div className="results-table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    {METRICS.map(m => (
                      <th key={m.key} title={m.desc}>
                        {m.label}
                        <span style={{ fontSize: 10, marginLeft: 4, color: 'var(--text-dim)' }}>
                          {m.lowerBetter ? '↓' : '↑'}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {modelNames.map(modelId => {
                    const mData = dsResults[modelId]
                    if (mData?.error) {
                      return (
                        <tr key={modelId}>
                          <td>
                            <div className="model-cell">
                              <div className="model-dot" style={{ background: modelColors[modelId] || '#888' }} />
                              <span>{modelId}</span>
                            </div>
                          </td>
                          <td colSpan={4} style={{ color: 'var(--neon-red)', fontSize: 11 }}>
                            Error: {mData.error}
                          </td>
                        </tr>
                      )
                    }

                    return (
                      <tr key={modelId}>
                        <td>
                          <div className="model-cell" style={{ fontFamily: 'var(--font-sans)' }}>
                            <div className="model-dot" style={{ background: modelColors[modelId] || '#888' }} />
                            {(models || []).find(m => m.id === modelId)?.name || modelId}
                          </div>
                        </td>
                        {METRICS.map(m => {
                          const isBest = bestPerMetric[m.key] === modelId
                          const mean = mData?.[m.key]?.mean
                          const std  = mData?.[m.key]?.std
                          return (
                            <td
                              key={m.key}
                              className={isBest ? 'best-cell' : ''}
                              title={isBest ? `Best ${m.key}` : ''}
                            >
                              {isBest && (
                                <span style={{ marginRight: 4, fontSize: 10 }}>★</span>
                              )}
                              {formatVal(mean, std)}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}
    </div>
  )
}
