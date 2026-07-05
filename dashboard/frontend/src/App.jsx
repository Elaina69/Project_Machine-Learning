import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const metricLabels = {
  MAE: 'MAE',
  RMSE: 'RMSE',
  MAPE: 'MAPE',
  R2: 'R2',
  MASE: 'MASE',
};

const rawMetricLabels = {
  flow: 'Flow',
  speed: 'Speed',
  occupancy: 'Occupancy',
};

const baselineLabels = {
  Baseline_A: 'Baseline A',
  Baseline_B: 'Baseline B',
};

const modelLabels = {
  '5_RandomForest': 'Random Forest',
  '6_XGBoost': 'XGBoost',
};

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-';
  }
  return new Intl.NumberFormat('vi-VN', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number(value));
}

function formatHorizon(minutes) {
  const value = Number(minutes);
  if (!Number.isFinite(value)) return '-';
  if (value < 60) return `${Math.round(value)} phút`;
  const hours = value / 60;
  if (hours < 24) return `${formatNumber(hours, hours % 1 === 0 ? 0 : 1)} giờ`;
  return `${formatNumber(hours / 24, 1)} ngày`;
}

function toDateTimeLocal(value) {
  if (!value) return '';
  return String(value).replace(' ', 'T').slice(0, 16);
}

function toApiDateTime(value) {
  if (!value) return '';
  const normalized = String(value).replace('T', ' ');
  return normalized.length === 16 ? `${normalized}:00` : normalized;
}

async function fetchJson(path, params = {}) {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, value);
    }
  });
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function useAsyncData(loader, deps) {
  const [state, setState] = useState({ data: null, loading: true, error: '' });

  useEffect(() => {
    let cancelled = false;
    setState((prev) => ({ ...prev, loading: !prev.data, error: '' }));
    loader()
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: '' });
      })
      .catch((error) => {
        if (!cancelled) setState({ data: null, loading: false, error: error.message });
      });
    return () => {
      cancelled = true;
    };
  }, deps);

  return state;
}

function Chart({ data, lines, valueFormatter = formatNumber }) {
  const width = 900;
  const height = 260;
  const pad = { top: 22, right: 22, bottom: 30, left: 54 };
  const values = [];
  data.forEach((row) => {
    lines.forEach((line) => {
      const value = Number(row[line.key]);
      if (Number.isFinite(value)) values.push(value);
    });
  });
  if (!data.length || !values.length) {
    return <div className="empty-state">Không có dữ liệu để hiển thị.</div>;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max === min ? Math.max(Math.abs(max), 1) : max - min;
  const domainMin = min - span * 0.12;
  const domainMax = max + span * 0.14;
  const domainSpan = domainMax - domainMin || 1;
  const xFor = (index) => {
    if (data.length === 1) return width / 2;
    return pad.left + (index / (data.length - 1)) * (width - pad.left - pad.right);
  };
  const yFor = (value) => pad.top + (1 - (value - domainMin) / domainSpan) * (height - pad.top - pad.bottom);
  const yTicks = [min, min + span / 2, max];

  return (
    <div className="chart-shell">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={yFor(tick)}
              y2={yFor(tick)}
              className="grid-line"
            />
            <text x={12} y={yFor(tick) + 4} className="axis-label">
              {valueFormatter(tick)}
            </text>
          </g>
        ))}
        {lines.map((line) => {
          const points = data
            .map((row, index) => {
              const value = Number(row[line.key]);
              if (!Number.isFinite(value)) return null;
              return `${xFor(index)},${yFor(value)}`;
            })
            .filter(Boolean)
            .join(' ');
          return <polyline key={line.key} points={points} fill="none" stroke={line.color} strokeWidth="2.5" />;
        })}
        <line x1={pad.left} x2={width - pad.right} y1={height - pad.bottom} y2={height - pad.bottom} className="axis-line" />
      </svg>
      <div className="legend-row">
        {lines.map((line) => (
          <span key={line.key} className="legend-item">
            <span className="legend-swatch" style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{formatNumber(value, label === 'R2' ? 4 : 2)}</strong>
    </div>
  );
}

function HorizontalBars({ rows, valueKey, labelKey, color = '#0f766e' }) {
  if (!rows?.length) return <div className="empty-state">Không có dữ liệu.</div>;
  const max = Math.max(...rows.map((row) => Number(row[valueKey]) || 0), 1);
  return (
    <div className="bar-list">
      {rows.map((row) => {
        const value = Number(row[valueKey]) || 0;
        return (
          <div className="bar-row" key={`${row[labelKey]}-${row.rank || ''}`}>
            <div className="bar-label">{row[labelKey]}</div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(value / max) * 100}%`, background: color }} />
            </div>
            <div className="bar-value">{formatNumber(value, 2)}</div>
          </div>
        );
      })}
    </div>
  );
}

function AlertPanel({ alertsData }) {
  const alerts = alertsData?.alerts || [];
  if (!alerts.length) return <div className="empty-state">Chưa có cảnh báo.</div>;
  return (
    <div className="alert-list">
      {alerts.map((item, index) => (
        <div className={`alert-card ${item.level}`} key={`${item.title}-${index}`}>
          <strong>{item.title}</strong>
          <p>{item.message}</p>
          <span>{item.recommendation}</span>
        </div>
      ))}
    </div>
  );
}

function FlowForecastCard({ forecast, loading }) {
  if (loading) return <LoadingBlock label="Đang tải dự báo lưu lượng..." />;
  if (!forecast) return <div className="empty-state">Chưa có dự báo lưu lượng.</div>;

  return (
    <section className={`flow-forecast ${forecast.level || 'ok'}`}>
      <div>
        <p className="flow-label">Dự báo lưu lượng dòng xe</p>
        <strong>{formatNumber(forecast.predicted_flow)} xe/5 phút</strong>
        <span>Dự báo sau: {formatHorizon(forecast.requested_minutes)}</span>
        <span>Mốc người dùng chọn: {forecast.requested_base_time}</span>
        <span>Mốc dữ liệu gần nhất: {forecast.demo_now}</span>
        <span>Thời điểm dự báo: {forecast.timestamp}</span>
      </div>
      <div className="flow-status-block">
        <span className={`flow-status ${forecast.level || 'ok'}`}>{forecast.status}</span>
        <p>{forecast.recommendation}</p>
        <small>
          Cập nhật khi thay đổi mốc thời gian, slider hoặc bộ lọc
          {forecast.effective_minutes !== forecast.requested_minutes
            ? ` | mốc gần nhất theo dữ liệu: ${formatHorizon(forecast.effective_minutes)}`
            : ''}
        </small>
      </div>
      <div className="flow-mini-metrics">
        <span>Actual: {formatNumber(forecast.actual_flow)}</span>
        <span>|Error|: {formatNumber(forecast.abs_error)}</span>
        <span>Ngưỡng cao Q75: {formatNumber(forecast.q75)}</span>
      </div>
    </section>
  );
}

function LoadingBlock({ label = 'Đang tải dữ liệu...' }) {
  return <div className="empty-state">{label}</div>;
}

export default function App() {
  const [filters, setFilters] = useState({
    baseline: 'Baseline_A',
    model: '6_XGBoost',
    sensor: '',
    rawMetric: 'flow',
    nPoints: 300,
    forecastMinutes: 15,
    forecastBaseTime: '',
  });

  const healthState = useAsyncData(() => fetchJson('/api/health'), []);
  const optionsState = useAsyncData(() => fetchJson('/api/options'), []);
  const options = optionsState.data;

  useEffect(() => {
    if (!options?.defaults) return;
    setFilters((prev) => ({
      ...prev,
      baseline: options.defaults.baseline,
      model: options.defaults.model,
      sensor: options.defaults.sensor,
      rawMetric: options.defaults.raw_metric,
      nPoints: options.defaults.n_points,
      forecastBaseTime: prev.forecastBaseTime || toDateTimeLocal(options.defaults.forecast_base_time),
    }));
  }, [options]);

  useEffect(() => {
    if (!options?.sensors) return;
    const available = options.sensors[filters.baseline] || [];
    const range = options.time_ranges?.[filters.baseline];
    const minTime = toDateTimeLocal(range?.start);
    const maxTime = toDateTimeLocal(range?.end);
    const defaultTime = toDateTimeLocal(range?.default_forecast_time);
    const shouldResetSensor = available.length && !available.includes(filters.sensor);
    const shouldResetTime = (
      !filters.forecastBaseTime
      || (minTime && filters.forecastBaseTime < minTime)
      || (maxTime && filters.forecastBaseTime > maxTime)
    );

    if (shouldResetSensor || shouldResetTime) {
      setFilters((prev) => ({
        ...prev,
        sensor: shouldResetSensor ? available[0] : prev.sensor,
        forecastBaseTime: shouldResetTime ? defaultTime : prev.forecastBaseTime,
      }));
    }
  }, [filters.baseline, filters.sensor, filters.forecastBaseTime, options]);

  const rawState = useAsyncData(
    () => fetchJson('/api/raw-series', {
      sensor: filters.sensor,
      metric: filters.rawMetric,
      n_points: filters.nPoints,
    }),
    [filters.sensor, filters.rawMetric, filters.nPoints],
  );

  const predictionState = useAsyncData(
    () => fetchJson('/api/predictions', {
      baseline: filters.baseline,
      model: filters.model,
      sensor: filters.sensor,
      n_points: filters.nPoints,
    }),
    [filters.baseline, filters.model, filters.sensor, filters.nPoints],
  );

  const metricsState = useAsyncData(
    () => fetchJson('/api/metrics', {
      baseline: filters.baseline,
      model: filters.model,
      sensor: filters.sensor,
    }),
    [filters.baseline, filters.model, filters.sensor],
  );

  const comparisonState = useAsyncData(
    () => fetchJson('/api/comparison', {
      model: filters.model,
      n_points: 200,
    }),
    [filters.model],
  );

  const xaiState = useAsyncData(
    () => fetchJson('/api/xai', {
      baseline: filters.baseline,
      model: filters.model,
    }),
    [filters.baseline, filters.model],
  );

  const alertsState = useAsyncData(
    () => fetchJson('/api/alerts', {
      baseline: filters.baseline,
      model: filters.model,
      sensor: filters.sensor,
      forecast_minutes: filters.forecastMinutes,
      base_time: toApiDateTime(filters.forecastBaseTime),
    }),
    [filters.baseline, filters.model, filters.sensor, filters.forecastMinutes, filters.forecastBaseTime],
  );

  const metricValues = metricsState.data?.computed_metrics || metricsState.data?.registry_metrics || {};
  const sensors = options?.sensors?.[filters.baseline] || [];
  const predictionSeries = predictionState.data?.series || [];
  const rawSeries = rawState.data?.series || [];
  const comparisonRows = comparisonState.data?.rows || [];
  const xaiRows = xaiState.data?.features || [];
  const timeRange = options?.time_ranges?.[filters.baseline] || {};
  const forecastMinTime = toDateTimeLocal(timeRange.start);
  const forecastMaxTime = toDateTimeLocal(timeRange.end);

  const comparisonBars = useMemo(
    () => comparisonRows.map((row) => ({
      label: baselineLabels[row.baseline] || row.baseline,
      RMSE: row.RMSE,
    })),
    [comparisonRows],
  );

  const errors = [
    healthState.error,
    optionsState.error,
    rawState.error,
    predictionState.error,
    metricsState.error,
    comparisonState.error,
    xaiState.error,
    alertsState.error,
  ].filter(Boolean);

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">SV16 PeMSD3</p>
          <h1>Dashboard dự báo giao thông</h1>
          <p className="subtitle">Một trang web cho dữ liệu gốc, dự báo, metrics, SHAP và cảnh báo từ 4 mô hình baseline tối ưu.</p>
        </div>
        <div className={`health-pill ${healthState.data?.ok ? 'ok' : 'warn'}`}>
          {healthState.loading ? 'Đang kiểm tra' : `${healthState.data?.loaded_model_count || 0}/4 model sẵn sàng`}
        </div>
      </header>

      {errors.length > 0 && (
        <section className="notice">
          <strong>Có lỗi khi tải dữ liệu</strong>
          <p>{errors[0]}</p>
        </section>
      )}

      <section className="filters-panel">
        <label>
          Baseline
          <select value={filters.baseline} onChange={(event) => setFilters((prev) => ({ ...prev, baseline: event.target.value }))}>
            {(options?.baselines || ['Baseline_A', 'Baseline_B']).map((baseline) => (
              <option value={baseline} key={baseline}>{baselineLabels[baseline] || baseline}</option>
            ))}
          </select>
        </label>
        <label>
          Model
          <select value={filters.model} onChange={(event) => setFilters((prev) => ({ ...prev, model: event.target.value }))}>
            {(options?.models || ['5_RandomForest', '6_XGBoost']).map((model) => (
              <option value={model} key={model}>{modelLabels[model] || model}</option>
            ))}
          </select>
        </label>
        <label>
          Sensor
          <select value={filters.sensor} onChange={(event) => setFilters((prev) => ({ ...prev, sensor: event.target.value }))}>
            {sensors.map((sensor) => (
              <option value={sensor} key={sensor}>{sensor}</option>
            ))}
          </select>
        </label>
        <label>
          Dữ liệu gốc
          <select value={filters.rawMetric} onChange={(event) => setFilters((prev) => ({ ...prev, rawMetric: event.target.value }))}>
            {(options?.raw_metrics || ['flow', 'speed', 'occupancy']).map((metric) => (
              <option value={metric} key={metric}>{rawMetricLabels[metric]}</option>
            ))}
          </select>
        </label>
        <label>
          Mốc bắt đầu dự báo
          <input
            type="datetime-local"
            min={forecastMinTime}
            max={forecastMaxTime}
            step="300"
            value={filters.forecastBaseTime}
            onChange={(event) => setFilters((prev) => ({ ...prev, forecastBaseTime: event.target.value }))}
          />
        </label>
        <label>
          Dự báo sau
          <input
            type="range"
            min="5"
            max="1440"
            step="5"
            value={filters.forecastMinutes}
            onChange={(event) => setFilters((prev) => ({ ...prev, forecastMinutes: Number(event.target.value) }))}
          />
          <span className="range-value">{formatHorizon(filters.forecastMinutes)}</span>
        </label>
        <label>
          Số điểm trên biểu đồ
          <input
            type="range"
            min="50"
            max="1000"
            step="50"
            value={filters.nPoints}
            onChange={(event) => setFilters((prev) => ({ ...prev, nPoints: Number(event.target.value) }))}
          />
          <span className="range-value">{filters.nPoints} điểm</span>
        </label>
      </section>

      <section className="metric-grid" aria-label="Chỉ số đánh giá">
        {Object.keys(metricLabels).map((key) => (
          <MetricCard key={key} label={metricLabels[key]} value={metricValues[key]} />
        ))}
      </section>

      <FlowForecastCard forecast={alertsState.data?.flow_forecast} loading={alertsState.loading} />

      <section className="dashboard-grid">
        <article className="panel wide">
          <div className="panel-heading">
            <div>
              <h2>Dữ liệu gốc</h2>
              <p>{filters.sensor} - {rawMetricLabels[filters.rawMetric]}</p>
            </div>
            {rawState.data?.stats && (
              <span className="panel-stat">
                mean {formatNumber(rawState.data.stats.mean)} | max {formatNumber(rawState.data.stats.max)}
              </span>
            )}
          </div>
          {rawState.loading ? <LoadingBlock /> : (
            <Chart
              data={rawSeries}
              lines={[{ key: filters.rawMetric, label: rawMetricLabels[filters.rawMetric], color: '#0f766e' }]}
            />
          )}
        </article>

        <article className="panel wide">
          <div className="panel-heading">
            <div>
              <h2>Actual vs Predicted</h2>
              <p>{baselineLabels[filters.baseline]} - {modelLabels[filters.model]} - {filters.sensor}</p>
            </div>
            <span className="panel-stat">RMSE {formatNumber(predictionState.data?.metrics?.RMSE)}</span>
          </div>
          {predictionState.loading ? <LoadingBlock /> : (
            <Chart
              data={predictionSeries}
              lines={[
                { key: 'actual', label: 'Actual', color: '#2563eb' },
                { key: 'predicted', label: 'Predicted', color: '#dc2626' },
              ]}
            />
          )}
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>So sánh A/B</h2>
              <p>Cùng model {modelLabels[filters.model]}</p>
            </div>
            <span className="panel-stat">Best: {baselineLabels[comparisonState.data?.best_baseline] || '-'}</span>
          </div>
          {comparisonState.loading ? <LoadingBlock /> : (
            <HorizontalBars rows={comparisonBars} valueKey="RMSE" labelKey="label" color="#2563eb" />
          )}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Baseline</th>
                  <th>RMSE</th>
                  <th>MAE</th>
                  <th>R2</th>
                </tr>
              </thead>
              <tbody>
                {comparisonRows.map((row) => (
                  <tr key={row.baseline}>
                    <td>{baselineLabels[row.baseline]}</td>
                    <td>{formatNumber(row.RMSE)}</td>
                    <td>{formatNumber(row.MAE)}</td>
                    <td>{formatNumber(row.R2, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>SHAP top 14 + other</h2>
              <p>Độ nhạy feature của model đang chọn</p>
            </div>
          </div>
          {xaiState.loading ? <LoadingBlock /> : (
            <HorizontalBars rows={xaiRows} valueKey="mean_abs_shap" labelKey="feature" color="#0f766e" />
          )}
        </article>

        <article className="panel wide">
          <div className="panel-heading">
            <div>
              <h2>Cảnh báo và khuyến nghị</h2>
              <p>Rule-based theo flow, occupancy, sai số sensor/hour và RMSE registry.</p>
            </div>
          </div>
          {alertsState.loading ? <LoadingBlock /> : <AlertPanel alertsData={alertsState.data} />}
        </article>
      </section>
    </main>
  );
}
