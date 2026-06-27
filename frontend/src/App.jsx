import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Activity, BarChart3, Home, Cpu, Zap } from 'lucide-react'
import HomePage from './pages/Home.jsx'
import ResultsPage from './pages/Results.jsx'

const API_BASE = '/api'

function Navbar({ apiStatus }) {
  return (
    <nav className="navbar">
      <a href="/" className="navbar-brand">
        <div className="navbar-logo">B</div>
        <div>
          <div className="navbar-title">BenchSCF</div>
          <div className="navbar-subtitle">Supply Chain Forecasting</div>
        </div>
      </a>

      <div className="navbar-nav">
        <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          <Home size={14} />
          Benchmark
        </NavLink>
        <NavLink to="/results" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          <BarChart3 size={14} />
          Results
        </NavLink>
      </div>

      <div className="navbar-status">
        <div className={`status-dot ${apiStatus}`} />
        <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
          {apiStatus === 'online' ? 'API ONLINE' : apiStatus === 'running' ? 'RUNNING' : 'API OFFLINE'}
        </span>
        <span style={{ color: 'var(--text-dim)', marginLeft: 4 }}>v1.0.0</span>
      </div>
    </nav>
  )
}

export default function App() {
  const [apiStatus, setApiStatus] = useState('offline')

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`)
        if (res.ok) setApiStatus('online')
        else setApiStatus('offline')
      } catch {
        setApiStatus('offline')
      }
    }
    check()
    const interval = setInterval(check, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar apiStatus={apiStatus} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage apiStatus={apiStatus} setApiStatus={setApiStatus} />} />
            <Route path="/results" element={<ResultsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export { API_BASE }
