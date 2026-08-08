import React, { useEffect, useMemo, useState } from 'react';
import { Activity, Box, Cpu, Database, Gauge, Radio, RefreshCw, Zap } from 'lucide-react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const fmt = (n, digits = 2) => n == null ? '—' : Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
const metricFmt = n => {
  if (n == null) return '—';
  const value = Number(n);
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0.000000';
  if (Math.abs(value) < 0.01) return value.toPrecision(9);
  return value.toLocaleString(undefined, { maximumFractionDigits: 9 });
};
const SMOOTH_WINDOW = 60;
const LOG_EPSILON = 1e-6;

const makeAxisTicks = values => {
  if (!values.length) return [];
  const count = Math.min(6, values.length);
  return Array.from({ length: count }, (_, index) => {
    const position = count === 1 ? 0 : index / (count - 1);
    const point = values[Math.round(position * (values.length - 1))];
    return { position: position * 100, step: point.step };
  }).filter((tick, index, ticks) => index === 0 || index === ticks.length - 1 || tick.position - ticks[index - 1].position >= 12);
};

function logAxis(logValues) {
  const finiteValues = logValues.filter(Number.isFinite);
  if (!finiteValues.length) return { minimum: -1, maximum: 1, range: 2, ticks: [{ value: -1, exponent: -1 }, { value: 0, exponent: 0 }, { value: 1, exponent: 1 }] };
  let minimum = Math.floor(Math.min(...finiteValues));
  let maximum = Math.ceil(Math.max(...finiteValues));
  if (minimum === maximum) { minimum -= 1; maximum += 1; }
  const ticks = [];
  const addTick = (value, exponent = value) => {
    if (!ticks.some(tick => Math.abs(tick.value - value) < 1e-9)) ticks.push({ value, exponent });
  };
  for (let exponent = minimum; exponent <= maximum; exponent += 1) addTick(exponent);
  ticks.sort((left, right) => left.value - right.value);
  return { minimum, maximum, range: maximum - minimum, ticks };
}

function PowerLabel({ exponent }) {
  return <>10<sup>{String(exponent)}</sup></>;
}

function App() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState('yolov3');
  const [data, setData] = useState(null);
  const [scale, setScale] = useState('linear');
  const [lossWindow, setLossWindow] = useState('all');

  const loadRuns = () => fetch('/api/runs', { cache: 'no-store' })
    .then(response => { if (!response.ok) throw new Error(`Run list failed (${response.status})`); return response.json(); })
    .then(names => {
      setRuns(names);
      setRun(current => names.length && !names.includes(current) ? names[0] : current);
    })
    .catch(error => console.warn(error));

  const fetchSnapshot = signal => fetch(`/api/run?name=${encodeURIComponent(run)}${lossWindow === 'all' ? '' : `&window=${lossWindow}`}`, {
    cache: 'no-store',
    signal
  }).then(response => {
    if (!response.ok) throw new Error(`Monitor request failed (${response.status})`);
    return response.json();
  });

  const refresh = () => fetchSnapshot().then(setData).catch(error => console.warn(error));

  useEffect(() => {
    loadRuns();
    const id = setInterval(loadRuns, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let stopped = false;
    let inFlight = false;
    let controller;

    const poll = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      controller = new AbortController();
      try {
        const next = await fetchSnapshot(controller.signal);
        if (!stopped) setData(next);
      } catch (error) {
        if (!stopped && error.name !== 'AbortError') console.warn(error);
      } finally {
        inFlight = false;
      }
    };

    poll();
    const id = setInterval(poll, 2500);
    return () => { stopped = true; controller?.abort(); clearInterval(id); };
  }, [run, lossWindow]);

  const history = data?.points || [];
  const bucketCount = Math.min(48, history.length);
  const bucketSize = Math.max(1, Math.ceil(history.length / Math.max(bucketCount, 1)));
  const points = useMemo(() => Array.from({ length: Math.ceil(history.length / bucketSize) }, (_, i) => {
    const bucket = history.slice(i * bucketSize, (i + 1) * bucketSize);
    return bucket.length ? {
      step: bucket.at(-1).step,
      avg: bucket.reduce((sum, point) => sum + point.avg, 0) / bucket.length
    } : null;
  }).filter(Boolean), [history, bucketSize]);
  const axisTicks = useMemo(() => {
    if (!history.length) return [];
    const count = Math.min(8, history.length);
    const candidates = Array.from({ length: count }, (_, index) => {
      const position = count === 1 ? 0 : index / (count - 1);
      const historyIndex = Math.round(position * (history.length - 1));
      const step = history[historyIndex].step;
      return { position: position * 100, step, width: String(step).length * 7 + 12 };
    });
    const accepted = [];
    for (const tick of candidates) {
      const previous = accepted.at(-1);
      const gap = previous ? tick.position - previous.position : Infinity;
      const required = previous ? ((tick.width + previous.width) / 2) / 7 : 0;
      if (!previous || gap >= required) accepted.push(tick);
    }
    if (accepted.length > 1) {
      const last = candidates.at(-1);
      while (accepted.length > 1 && last.position - accepted.at(-2).position < ((last.width + accepted.at(-2).width) / 2) / 7) {
        accepted.pop();
      }
      if (accepted.at(-1).step !== last.step) accepted.push(last);
    }
    return accepted.map(({ position, step }) => ({ position, step }));
  }, [history]);

  const values = points.map(point => point.avg).filter(Number.isFinite);
  const max = values.length ? Math.max(...values) : 1;
  const min = values.length ? Math.min(...values) : 0;
  const linearRange = max > min ? max - min : LOG_EPSILON;
  const logAxisData = logAxis(points.map(point => Math.log10(Math.max(point.avg, LOG_EPSILON))));
  const height = value => {
    if (values.length === 1 || max === min) return 100;
    const normalized = scale === 'log'
      ? (Math.log10(Math.max(value, LOG_EPSILON)) - logAxisData.minimum) / logAxisData.range
      : (value - min) / linearRange;
    return Math.max(0, Math.min(100, normalized * 100));
  };
  const lossHue = value => {
    const ratio = scale === 'log'
      ? (Math.log10(Math.max(value, LOG_EPSILON)) - logAxisData.minimum) / logAxisData.range
      : (value - min) / linearRange;
    const palette = [105, 125, 52, 35, 12, 350];
    return palette[Math.round(Math.max(0, Math.min(1, ratio)) * (palette.length - 1))];
  };

  const meta = data?.metadata || {};
  const latest = data?.latest;

  return <main>
    <header>
      <div className="brand"><div className="mark"><Zap size={19} /></div><div><b>PIXIENN</b><span>TRAINING CONTROL</span></div></div>
      <div className="run-picker"><span>RUN</span><select value={run} onChange={event => setRun(event.target.value)}>{runs.map(name => <option key={name}>{name}</option>)}</select><button onClick={() => { loadRuns(); refresh(); }}><RefreshCw size={16} /></button></div>
    </header>

    <section className="hero">
      <div><p className="eyebrow"><span className="live-dot" /> TRAINING TELEMETRY</p><h1><span>PIXIENN</span><strong>{run}</strong><em>/ RUN MONITOR</em></h1><p className="sub">TensorFlow event files, checkpoints, and live training signals.</p></div>
      <div className={`status ${data?.active ? '' : 'idle'}`}><Radio size={16} /><span>{data?.active ? 'RUNNING' : 'IDLE'}</span></div>
    </section>

    <nav><small>updated {data ? new Date(data.updatedAt).toLocaleTimeString() : '—'}</small></nav>

    <section className="cards">
      <Card icon={<Activity />} label="AVG LOSS" value={fmt(latest?.avg)} accent="pink" note="event-file avg-loss" />
      <Card icon={<Gauge />} label="STEP" value={latest?.step?.toLocaleString() || '—'} accent="blue" note="optimizer step" />
      <Card icon={<Cpu />} label="LEARNING RATE" value={latest ? latest.lr?.toExponential(2) || '—' : '—'} accent="lime" note="event-file learning-rate" />
      <Card icon={<Box />} label="CHECKPOINT" value={data?.checkpoints?.[0]?.name || '—'} title={data?.checkpoints?.[0]?.name} accent="gold" note="latest saved weights" />
    </section>

    <section className="grid">
      <div className="panel chart">
        <div className="panel-head"><div><span className="kicker">LOSS TRAJECTORY</span><h2>Average loss</h2></div><div style={{ display: 'flex', gap: 8 }}><select value={lossWindow} onChange={event => setLossWindow(event.target.value)}><option value="all">Full run · auto-scaled</option><option value="10000">Last 10,000 steps</option><option value="2000">Last 2,000 steps</option><option value="500">Last 500 steps</option></select><select value={scale} onChange={event => setScale(event.target.value)}><option value="linear">Linear</option><option value="log">Log scale</option></select></div></div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#68707d', margin: '18px 0 8px' }}><span>{scale === 'log' ? 'Avg. loss · log10(loss)' : 'Avg. loss'}</span><span>{lossWindow === 'all' ? 'Full training history' : `Last ${Number(lossWindow).toLocaleString()} optimizer steps`}</span></div>
        <div style={{ display: 'flex', gap: 10, width: '100%' }}><div className="loss-y-labels">{scale === 'log' ? logAxisData.ticks.map(tick => <span key={tick.value} style={{ top: `${(1 - (tick.value - logAxisData.minimum) / logAxisData.range) * 100}%` }}><PowerLabel exponent={tick.exponent} /></span>) : <><span style={{ top: '0%' }}>{max.toPrecision(3)}</span><span style={{ top: '50%' }}>{((max + min) / 2).toPrecision(3)}</span><span style={{ top: '100%' }}>{min.toPrecision(3)}</span></>}</div><div className="spark" style={{ flex: 1, width: '100%', minWidth: 0, height: 260 }}>{scale === 'log' && logAxisData.ticks.map(tick => <span className="loss-grid-line" key={`loss-grid-${tick.value}`} style={{ bottom: `${((tick.value - logAxisData.minimum) / logAxisData.range) * 100}%` }} />)}{points.length ? points.map((point, i) => <div className="loss-bar" key={i} style={{ height: `${height(point.avg)}%`, '--bar-hue': `${lossHue(point.avg)}` }} title={`step ${point.step} · avg loss ${point.avg.toFixed(3)}`} />) : <span className="empty">Waiting for event-file data…</span>}</div></div>
        <div className="axis loss-x-axis" aria-label="optimizer steps">{axisTicks.map((tick, index) => <span key={`${tick.step}-${index}`} style={{ left: `${tick.position}%` }}>{tick.step.toLocaleString()}</span>)}</div>
      </div>
      <div className="panel meta"><div className="panel-head"><div><span className="kicker">RUN DNA</span><h2>Metadata</h2></div><Database size={18} /></div><Meta label="mode" value={meta.mode} /><Meta label="started" value={meta.started_utc} /><Meta label="configuration" value={meta.configuration?.split('/').pop()} /><Meta label="weights target" value={meta.weights?.split('/').slice(-2).join('/')} /><Meta label="engine" value={meta.executable?.split('/').pop()} /></div>
    </section>

    <EventScalarsPanel series={data?.eventFile?.series || {}} />
    <footer><span><span className="live-dot" /> event stream connected</span><span>PIXIENN / LOCAL RUN OBSERVATORY</span></footer>
  </main>;
}

function EventScalarsPanel({ series }) {
  const [hidden, setHidden] = useState(() => new Set());
  const [logScale, setLogScale] = useState(() => new Set());
  const [rawSeries, setRawSeries] = useState(() => new Set());

  const toggle = tag => setHidden(previous => {
    const next = new Set(previous);
    if (next.has(tag)) next.delete(tag); else next.add(tag);
    return next;
  });

  const toggleLogScale = tag => setLogScale(previous => {
    const next = new Set(previous);
    if (next.has(tag)) next.delete(tag); else next.add(tag);
    return next;
  });

  const setSmoothing = (tag, mode) => setRawSeries(previous => {
    const next = new Set(previous);
    if (mode === 'raw') next.add(tag); else next.delete(tag);
    return next;
  });

  return <section className="panel chart event-file-panel">
    <div className="tb-head"><span className="kicker">TFEVENT SCALARS · X / Y</span><span>{Object.keys(series).length} scalar cards · full run · auto-scaled</span></div>
    <div className="tb-cards">{Object.entries(series).map(([tag, values]) => {
      const logarithmic = logScale.has(tag);
      const raw = rawSeries.has(tag);
      const displayValues = values.map((point, index) => {
        if (raw) return point.value;
        const slice = values.slice(Math.max(0, index - SMOOTH_WINDOW + 1), index + 1);
        return slice.reduce((sum, item) => sum + item.value, 0) / slice.length;
      });
      const logarithms = displayValues.map(value => Math.log10(Math.max(value, LOG_EPSILON)));
      const logAxisData = logAxis(logarithms);
      const transformed = logarithmic ? logarithms : displayValues;
      const finiteValues = transformed.filter(Number.isFinite);
      const maximum = finiteValues.length
        ? (logarithmic ? logAxisData.maximum : Math.max(...finiteValues))
        : 1;
      const minimum = finiteValues.length
        ? (logarithmic ? logAxisData.minimum : Math.min(...finiteValues))
        : 0;
      const range = maximum > minimum ? maximum - minimum : LOG_EPSILON;
      const pointY = value => maximum === minimum ? 43 : 86 - ((value - minimum) / range) * 86;
      const svgPoints = transformed.map((value, index) => `${(index / Math.max(transformed.length - 1, 1)) * 220},${pointY(value)}`).join(' ');
      const singlePoint = transformed.length === 1 && Number.isFinite(transformed[0]) ? { x: 110, y: pointY(transformed[0]) } : null;
      const axisText = value => {
        if (logarithmic) return null;
        const number = value;
        if (!Number.isFinite(number)) return '—';
        if (number !== 0 && Math.abs(number) < 0.01) return number.toExponential(1);
        return number.toPrecision(3);
      };
      const yTicks = logarithmic
        ? logAxisData.ticks.map(tick => ({ exponent: tick.exponent, top: (1 - (tick.value - logAxisData.minimum) / logAxisData.range) * 100 }))
        : [{ top: 0, label: axisText(maximum) }, { top: 50, label: axisText((maximum + minimum) / 2) }, { top: 100, label: axisText(minimum) }];
      const latest = values.at(-1);
      const scalarAxisTicks = makeAxisTicks(values);
      const unit = tag.includes('learning-rate') ? 'learning rate' : tag.includes('objectness') || tag.includes('avg-iou') || tag.includes('recall') || tag.includes('avg-class') ? 'ratio' : 'loss';
      return <article className={`tb-card ${hidden.has(tag) ? 'tb-off' : ''}`} key={tag}>
        <div className="tb-toggle"><label><input type="checkbox" checked={!hidden.has(tag)} onChange={() => toggle(tag)} /><span>{tag}</span></label><select className="tb-smoothing" value={raw ? 'raw' : 'smooth'} onChange={event => setSmoothing(tag, event.target.value)}><option value="smooth">Smooth</option><option value="raw">Raw</option></select><button type="button" className="tb-scale" onClick={() => toggleLogScale(tag)}>{logarithmic ? 'Log' : 'Linear'}</button></div>
        <div className="tb-axis-units"><span>Y · {logarithmic ? `log10(${unit})` : unit}</span><span>X · optimizer step</span></div>
        <div className="tb-plot"><div className="tb-y-labels" aria-hidden="true">{yTicks.map((tick, index) => <span key={`${tag}-${tick.top}-${index}`} style={{ top: `${tick.top}%` }}>{logarithmic ? <PowerLabel exponent={tick.exponent} /> : tick.label}</span>)}</div><div className="tb-plot-area"><svg viewBox="0 0 220 86" preserveAspectRatio="none" role="img" aria-label={`${tag} ${logarithmic ? 'logarithmic' : 'linear'} scale`}>{logarithmic && yTicks.map(tick => <line key={`grid-${tag}-${tick.exponent}`} className="tb-grid-line" x1="0" y1={(tick.top / 100) * 86} x2="220" y2={(tick.top / 100) * 86} />)}<line x1="0" y1="86" x2="220" y2="86" /><line x1="0" y1="0" x2="0" y2="86" /><polyline points={svgPoints} />{singlePoint && <circle className="tb-single-point" cx={singlePoint.x} cy={singlePoint.y} r="2" />}</svg><div className="tb-x-axis">{scalarAxisTicks.map((tick, index) => <span key={`${tag}-x-${index}`} style={{ left: `${tick.position}%` }}>{tick.step.toLocaleString()}</span>)}</div></div></div>
        <div className="tb-values"><span>step {latest?.step ?? '—'}</span><b>{metricFmt(latest?.value)}</b></div>
      </article>;
    }) || <span className="empty">No scalar data found in the event file.</span>}</div>
  </section>;
}

function Card({ icon, label, value, title, accent, note }) { return <div className={`card ${accent}`}><div className="card-top"><span>{icon}</span><label>{label}</label></div><strong title={title || undefined}>{value}</strong><small>{note}</small></div>; }
function Meta({ label, value }) { return <div className="meta-row"><span>{label}</span><b>{value || '—'}</b></div>; }

createRoot(document.getElementById('root')).render(<App />);
