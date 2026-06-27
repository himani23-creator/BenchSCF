import { useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'

const METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']
const METRIC_LABELS = { MAE: 'MAE', RMSE: 'RMSE', MAPE: 'MAPE %', R2: 'R²' }

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <div className="tooltip-label">{label}</div>
      {payload.map(p => (
        <div key={p.name} className="tooltip-row">
          <span style={{ color: p.color }}>{p.name}</span>
          <span>{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function ComparisonChart({ results, models }) {
  const [activeMetric, setActiveMetric] = useState('MAE')
  const [activeDataset, setActiveDataset] = useState(Object.keys(results || {})[0])

  const datasets = Object.keys(results || {})

  if (!datasets.length) return null

  const dsResults = results[activeDataset] || {}
  const modelIds  = Object.keys(dsResults).filter(m => !dsResults[m]?.error)

  // Bar chart data for selected metric
  const barData = modelIds.map(id => ({
    model: (models || []).find(m => m.id === id)?.name || id,
    value: dsResults[id]?.[activeMetric]?.mean ?? 0,
    std:   dsResults[id]?.[activeMetric]?.std ?? 0,
  }))

  const modelColors = Object.fromEntries((models || []).map(m => [m.id, m.color]))

  // Radar chart — normalise each metric to [0,1] for comparison
  const radarData = METRICS.map(metric => {
    const vals = modelIds.map(id => dsResults[id]?.[metric]?.mean).filter(v => v != null && !isNaN(v))
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const range = max - min || 1
    const entry = { metric: METRIC_LABELS[metric] }
    modelIds.forEach(id => {
      const raw = dsResults[id]?.[metric]?.mean ?? 0
      // Normalise (for R2, higher is better so flip)
      const norm = metric === 'R2'
        ? (raw - min) / range         // higher norm = better
        : 1 - (raw - min) / range     // lower raw = higher norm = better
      entry[id] = Math.round(norm * 100) / 100
    })
    return entry
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Controls */}
      <div className="flex items-center gap-3" style={{ flexWrap: 'wrap' }}>
        {datasets.length > 1 && (
          <div className="tab-bar">
            {datasets.map(ds => (
              <button
                key={ds}
                className={`tab-btn ${activeDataset === ds ? 'active' : ''}`}
                onClick={() => setActiveDataset(ds)}
              >
                {ds === 'dataco' ? 'DataCo' : 'Rossmann'}
              </button>
            ))}
          </div>
        )}
        <div className="tab-bar">
          {METRICS.map(m => (
            <button
              key={m}
              className={`tab-btn ${activeMetric === m ? 'active' : ''}`}
              onClick={() => setActiveMetric(m)}
            >
              {METRIC_LABELS[m]}
            </button>
          ))}
        </div>
      </div>

      {/* Bar chart */}
      <div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.6px' }}>
          {METRIC_LABELS[activeMetric]} — Mean across CV folds
          {activeMetric !== 'R2' ? ' (lower is better)' : ' (higher is better)'}
        </div>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,255,0.06)" />
              <XAxis
                dataKey="model"
                tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-sans)' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                axisLine={false}
                tickLine={false}
                width={60}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" name={METRIC_LABELS[activeMetric]} radius={[4, 4, 0, 0]}>
                {barData.map((entry, i) => (
                  <Cell key={i} fill={modelColors[modelIds[i]] || '#00e5ff'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Radar chart */}
      {modelIds.length >= 2 && (
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.6px' }}>
            Performance Radar — Normalised (higher = better on each axis)
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                <PolarGrid stroke="rgba(0,229,255,0.1)" />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{ fill: 'var(--text-secondary)', fontSize: 12, fontFamily: 'var(--font-sans)', fontWeight: 600 }}
                />
                <PolarRadiusAxis
                  domain={[0, 1]}
                  tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                  tickCount={3}
                />
                {modelIds.map(id => (
                  <Radar
                    key={id}
                    name={(models || []).find(m => m.id === id)?.name || id}
                    dataKey={id}
                    stroke={modelColors[id] || '#00e5ff'}
                    fill={modelColors[id] || '#00e5ff'}
                    fillOpacity={0.08}
                    strokeWidth={2}
                  />
                ))}
                <Legend
                  wrapperStyle={{ fontSize: 12, fontFamily: 'var(--font-sans)', color: 'var(--text-secondary)' }}
                />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Per-model metric grid */}
      <div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.6px' }}>
          All Metrics Summary
        </div>
        <div className="grid-4" style={{ gap: 12 }}>
          {modelIds.map(id => {
            const mData = dsResults[id]
            const model = (models || []).find(m => m.id === id)
            return (
              <div key={id} className="metric-card">
                <div className="flex items-center gap-2 mb-2">
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: modelColors[id] || '#00e5ff' }} />
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>{model?.name || id}</div>
                </div>
                {METRICS.map(metric => (
                  <div key={metric} className="flex justify-between" style={{ fontSize: 11 }}>
                    <span style={{ color: 'var(--text-muted)' }}>{METRIC_LABELS[metric]}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                      {mData?.[metric]?.mean != null && !isNaN(mData[metric].mean)
                        ? mData[metric].mean.toFixed(3)
                        : 'N/A'}
                    </span>
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
