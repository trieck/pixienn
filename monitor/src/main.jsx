import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Box, Cpu, Gauge, RefreshCw } from 'lucide-react';
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

function elapsedLabel(timestamp, now = Date.now()) {
  if (!Number.isFinite(timestamp)) return '—';
  const seconds = Math.max(0, Math.round((now - timestamp) / 1000));
  if (seconds < 10) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m ago`;
}

function countdownLabel(timestamp, now = Date.now()) {
  if (!Number.isFinite(timestamp)) return 'estimating…';
  const seconds = Math.max(0, Math.round((timestamp - now) / 1000));
  if (seconds < 60) return `in ${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return minutes < 60 ? `in ${minutes}m ${seconds % 60}s` : `in ${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

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

function logTicksForRange(minimum, maximum) {
  const first = Math.ceil(minimum);
  const last = Math.floor(maximum);
  return Array.from({ length: Math.max(0, last - first + 1) }, (_, index) => {
    const exponent = first + index;
    return { value: exponent, exponent };
  });
}

function paddedAxisRange(minimum, maximum) {
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return { minimum: 0, maximum: 1 };
  if (maximum <= minimum) {
    const padding = Math.max(Math.abs(maximum), 1) * 0.05;
    return { minimum: minimum - padding, maximum: maximum + padding };
  }
  const padding = (maximum - minimum) * 0.05;
  return { minimum: minimum - padding, maximum: maximum + padding };
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
  const [loading, setLoading] = useState(false);
  const [clockNow, setClockNow] = useState(Date.now());

  useEffect(() => {
    const id = setInterval(() => setClockNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

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

  const refresh = () => {
    setLoading(true);
    return fetchSnapshot().then(setData).catch(error => console.warn(error)).finally(() => setLoading(false));
  };

  useEffect(() => {
    loadRuns();
    const id = setInterval(loadRuns, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let stopped = false;
    let inFlight = false;
    let controller;
    setLoading(true);

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
        if (!stopped) setLoading(false);
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
    // Continuous traffic-light hue: green at low loss, yellow in the middle,
    // red at high loss, without abrupt palette jumps between bars.
    return 120 * (1 - Math.max(0, Math.min(1, ratio)));
  };

  const meta = data?.metadata || {};
  const latest = data?.latest;
  const scalarSeries = data?.eventFile?.series || {};
  const validationMetric = tag => {
    const values = scalarSeries[tag] || [];
    return [...values].sort((a, b) => Number(a.step) - Number(b.step)).at(-1);
  };
  const validationDelta = tag => {
    const values = scalarSeries[tag] || [];
    return values.length > 1 ? values.at(-1).value - values.at(-2).value : null;
  };
  const eventAge = data?.eventFile?.eventUpdatedAt ? Math.max(0, Date.now() - data.eventFile.eventUpdatedAt) : null;
  const validationStep = validationMetric('mAP50')?.step || validationMetric('micro-avg-f1')?.step;
  const freshness = eventAge == null ? 'waiting' : eventAge < 120000 ? 'fresh' : 'stale';
  const checkpoint = data?.checkpoints?.[0];
  const checkpointAge = checkpoint?.mtime ? Math.max(0, Date.now() - checkpoint.mtime) : null;
  const movingAverageWindow = 12;

  return <main>
    <header><div className="run-picker"><span>RUN</span><select value={run} onChange={event => setRun(event.target.value)}>{runs.map(name => <option key={name}>{name}</option>)}</select><button onClick={() => { loadRuns(); refresh(); }}><RefreshCw size={16} /></button></div></header>

    <nav><small>updated {data ? new Date(data.updatedAt).toLocaleTimeString() : '—'}</small></nav>
    {loading && <div className="monitor-loading-overlay" role="status" aria-live="polite"><div className="monitor-loading-dialog"><span className="monitor-loading-spinner" /><strong>Loading training history</strong><small>Reading event data…</small></div></div>}

    <section className="pixienn-banner" aria-label="PixieNN neural network illustration">
      <div className="banner-meta"><Meta label="mode" value={meta.mode} /><Meta label="started" value={meta.started_utc} /><Meta label="configuration" value={meta.configuration?.split('/').pop()} /><Meta label="engine" value={meta.executable?.split('/').pop()} /></div>
    </section>

    <section className="cards">
      <Card icon={<Activity />} label="AVG LOSS" value={fmt(latest?.avg)} accent="pink" note="event-file avg-loss" />
      <Card icon={<Gauge />} label="STEP" value={latest?.step?.toLocaleString() || '—'} accent="blue" note="optimizer step" />
      <Card icon={<Cpu />} label="LEARNING RATE" value={latest ? latest.lr?.toFixed(9) || '—' : '—'} accent="lime" note="learning rate" />
      <Card icon={<Box />} label="CHECKPOINT" value={checkpoint?.name || '—'} title={checkpoint?.name} accent="gold" note="latest saved weights" />
    </section>

    <section className="grid">
      <div className="panel chart">
        <div className="panel-head"><div><span className="kicker">LOSS TRAJECTORY</span><h2>Average loss</h2></div><div style={{ display: 'flex', gap: 8 }}><select value={lossWindow} onChange={event => setLossWindow(event.target.value)}><option value="all">Full run · auto-scaled</option><option value="10000">Last 10,000 steps</option><option value="2000">Last 2,000 steps</option><option value="500">Last 500 steps</option></select><select value={scale} onChange={event => setScale(event.target.value)}><option value="linear">Linear</option><option value="log">Log scale</option></select></div></div>
        <div className="loss-caption"><span>{scale === 'log' ? 'Avg. loss · log10(loss)' : 'Avg. loss'}</span><span className="loss-legend"><i className="low" /> lower loss <i className="high" /> higher loss</span><span>{lossWindow === 'all' ? 'Full training history' : `Last ${Number(lossWindow).toLocaleString()} optimizer steps`}</span></div>
        <div style={{ display: 'flex', gap: 10, width: '100%' }}><div className="loss-y-labels">{scale === 'log' ? logAxisData.ticks.map(tick => <span key={tick.value} style={{ top: `${(1 - (tick.value - logAxisData.minimum) / logAxisData.range) * 100}%` }}><PowerLabel exponent={tick.exponent} /></span>) : <><span style={{ top: '0%' }}>{max.toPrecision(3)}</span><span style={{ top: '50%' }}>{((max + min) / 2).toPrecision(3)}</span><span style={{ top: '100%' }}>{min.toPrecision(3)}</span></>}</div><div className="spark" style={{ flex: 1, width: '100%', minWidth: 0, height: 260 }}>{scale === 'log' && logAxisData.ticks.map(tick => <span className="loss-grid-line" key={`loss-grid-${tick.value}`} style={{ bottom: `${((tick.value - logAxisData.minimum) / logAxisData.range) * 100}%` }} />)}{points.length ? points.map((point, i) => <div className="loss-bar" key={i} style={{ height: `${height(point.avg)}%`, '--bar-hue': `${lossHue(point.avg)}` }} title={`step ${point.step} · avg loss ${point.avg.toFixed(3)}`} />) : <span className="empty">Waiting for event-file data…</span>}</div></div>
        <div className="axis loss-x-axis" aria-label="optimizer steps">{axisTicks.map((tick, index) => <span key={`${tick.step}-${index}`} style={{ left: `${tick.position}%` }}>{tick.step.toLocaleString()}</span>)}</div>
      </div>
    </section>

    <section className="health-panel panel">
      <div className="health-heading"><div><span className="kicker">VALIDATION HEALTH</span><h2>Is the detector improving?</h2><p className="health-legend">Numbers compare the latest validation with the moving average of the last {movingAverageWindow} validations.</p></div><span className={`freshness ${freshness}`}><span /> {freshness}</span></div>
      <div className="health-grid">
        <HealthMetric label="mAP50" point={validationMetric('mAP50')} values={scalarSeries['mAP50']} />
        <HealthMetric label="Micro-Avg-F1" point={validationMetric('micro-avg-f1')} values={scalarSeries['micro-avg-f1']} />
        <HealthMetric label="Avg. recall" point={validationMetric('avg-recall')} values={scalarSeries['avg-recall']} />
        <HealthMetric label="Val. loss" point={validationMetric('avg-val-loss')} values={scalarSeries['avg-val-loss']} invert />
      </div>
      <div className="health-footer"><ValidationClock schedule={data?.validationSchedule} now={clockNow} /><span>confidence threshold <b>{data?.validationThreshold == null ? '—' : data.validationThreshold}</b></span><span>events <b>{eventAge == null ? '—' : eventAge < 1000 ? 'just now' : `${Math.round(eventAge / 1000)}s ago`}</b></span><span>checkpoint <b>{checkpointAge == null ? '—' : checkpointAge < 1000 ? 'just now' : `${Math.round(checkpointAge / 60000)}m ago`}</b></span></div>
    </section>

    <PRCurvePanel curve={data?.eventFile?.prCurves?.['validation/micro-pr/curve'] || data?.eventFile?.prCurves?.['validation/micro-pr/pr_curves']} />
    <EventScalarsPanel run={run} series={data?.eventFile?.series || {}} tails={data?.eventFile?.tails || {}} />
    <footer><span><span className="live-dot" /> event stream connected</span><span>PIXIENN / LOCAL RUN OBSERVATORY</span></footer>
  </main>;
}

function ValidationClock({ schedule, now }) {
  return <div className="validation-clock"><span>last validation <b>{elapsedLabel(schedule?.lastAt, now)}</b>{schedule?.lastStep != null && <small>step {Number(schedule.lastStep).toLocaleString()}</small>}</span><span>next likely <b>{countdownLabel(schedule?.nextAt, now)}</b>{schedule?.nextStep != null && <small>step {Number(schedule.nextStep).toLocaleString()}</small>}</span></div>;
}

function PRCurvePanel({ curve }) {
  const points = [...(curve?.points || [])]
    .filter(point => Number.isFinite(Number(point.precision)) && Number.isFinite(Number(point.recall)))
    .sort((lhs, rhs) => Number(lhs.recall) - Number(rhs.recall));
  const xMaximum = 1;
  const yMaximum = 1;
  const coordinate = point => ({
    x: Math.max(0, Math.min(xMaximum, Number(point.recall))) / xMaximum * 220,
    y: 86 - Math.max(0, Math.min(yMaximum, Number(point.precision))) / yMaximum * 86
  });
  const svgPoints = points.map(point => {
    const current = coordinate(point);
    return `${current.x},${current.y}`;
  }).join(' ');
  const axisLabel = value => value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  return <section className="tb-card pr-panel">
    <div className="tb-toggle"><span>Precision–recall curve</span><span className="pr-meta">{curve ? `validation step ${Number(curve.step).toLocaleString()}` : 'Waiting for validation data'}</span></div>
    <div className="tb-axis-units"><span>Y · precision</span><span>X · recall</span></div>
    <div className="tb-plot"><div className="tb-y-labels"><span style={{ top: '0%' }}>{axisLabel(yMaximum)}</span><span style={{ top: '50%' }}>{axisLabel(yMaximum / 2)}</span><span style={{ top: '100%' }}>0</span></div><div className="tb-plot-area"><svg viewBox="0 0 220 86" preserveAspectRatio="none" role="img" aria-label="Precision versus recall curve"><defs><linearGradient id="pr-line-gradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#d5fbff" /><stop offset="16%" stopColor="#27dfff" /><stop offset="52%" stopColor="#087ff5" /><stop offset="82%" stopColor="#5140df" /><stop offset="100%" stopColor="#a818ff" /></linearGradient></defs><line x1="0" y1="0" x2="220" y2="0" /><line x1="0" y1="43" x2="220" y2="43" /><line x1="0" y1="86" x2="220" y2="86" /><polyline className="tb-tube-body" stroke="url(#pr-line-gradient)" points={svgPoints} /></svg><div className="tb-x-axis"><span style={{ left: '50%' }}>{axisLabel(xMaximum / 2)}</span><span style={{ left: '100%' }}>{axisLabel(xMaximum)}</span></div></div></div>
  </section>;
}

function EventScalarsPanel({ run, series, tails }) {
  const [hidden, setHidden] = useState(() => new Set());
  const [logScale, setLogScale] = useState(() => new Set());
  const [rawSeries, setRawSeries] = useState(() => new Set());
  const [cardWindows, setCardWindows] = useState({});
  const axisRanges = useRef(new Map());

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

  const setCardWindow = (tag, window) => setCardWindows(previous => ({ ...previous, [tag]: window }));

  return <section className="panel chart event-file-panel">
    <div className="tb-head"><span className="kicker">EVENT STREAM SCALARS · X / Y</span><span>{Object.keys(series).length} scalar cards · per-card windows · stable y-scale</span></div>
    <div className="tb-cards">{Object.entries(series).map(([tag, values]) => {
      const logarithmic = logScale.has(tag);
      const raw = rawSeries.has(tag);
      const window = cardWindows[tag] || 'all';
      const recentValues = tails[tag]?.length ? tails[tag] : values;
      const endStep = Number(recentValues.at(-1)?.step);
      const windowCutoff = window === 'all' || !Number.isFinite(endStep) ? null : endStep - Number(window);
      const visibleValues = windowCutoff == null
        ? values
        : recentValues.filter(point => Number(point.step) >= windowCutoff);
      const smoothingInput = windowCutoff == null
        ? visibleValues
        : [...recentValues.filter(point => Number(point.step) < windowCutoff).slice(-(SMOOTH_WINDOW - 1)), ...visibleValues];
      const smoothedValues = smoothingInput.map((point, index) => {
        if (raw) return point.value;
        const slice = smoothingInput.slice(Math.max(0, index - SMOOTH_WINDOW + 1), index + 1);
        return slice.reduce((sum, item) => sum + item.value, 0) / slice.length;
      });
      const displayValues = windowCutoff == null
        ? smoothedValues
        : visibleValues.length ? smoothedValues.slice(-visibleValues.length) : [];
      const logarithms = displayValues.map(value => Math.log10(Math.max(value, LOG_EPSILON)));
      const logAxisData = logAxis(logarithms);
      const transformed = logarithmic ? logarithms : displayValues;
      const finiteValues = transformed.filter(Number.isFinite);
      const candidateMaximum = finiteValues.length
        ? (logarithmic ? logAxisData.maximum : Math.max(...finiteValues))
        : 1;
      const candidateMinimum = finiteValues.length
        ? (logarithmic ? logAxisData.minimum : Math.min(...finiteValues))
        : 0;
      const axisKey = `${run}:${tag}:${window}:${raw ? 'raw' : 'smooth'}:${logarithmic ? 'log' : 'linear'}`;
      const previousAxis = axisRanges.current.get(axisKey);
      const candidateAxis = paddedAxisRange(candidateMinimum, candidateMaximum);
      const axis = previousAxis
        ? {
          minimum: Math.min(previousAxis.minimum, candidateAxis.minimum),
          maximum: Math.max(previousAxis.maximum, candidateAxis.maximum)
        }
        : candidateAxis;
      axisRanges.current.set(axisKey, axis);
      const maximum = axis.maximum;
      const minimum = axis.minimum;
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
        ? logTicksForRange(minimum, maximum).map(tick => ({ exponent: tick.exponent, top: (1 - (tick.value - minimum) / range) * 100 }))
        : [{ top: 0, label: axisText(maximum) }, { top: 50, label: axisText((maximum + minimum) / 2) }, { top: 100, label: axisText(minimum) }];
      const latest = visibleValues.at(-1);
      const scalarAxisTicks = makeAxisTicks(visibleValues);
      const unit = tag.includes('learning-rate') ? 'learning rate' : tag.includes('objectness') || tag.includes('avg-iou') || tag.includes('recall') || tag.includes('avg-class') ? 'ratio' : 'loss';
      return <article className={`tb-card ${hidden.has(tag) ? 'tb-off' : ''}`} key={tag}>
        <div className="tb-toggle"><label><input type="checkbox" checked={!hidden.has(tag)} onChange={() => toggle(tag)} /><span>{tag}</span></label><select className="tb-window" aria-label={`${tag} time window`} value={window} onChange={event => setCardWindow(tag, event.target.value)}><option value="all">Full</option><option value="1000">Last 1,000</option><option value="5000">Last 5,000</option></select><select className="tb-smoothing" value={raw ? 'raw' : 'smooth'} onChange={event => setSmoothing(tag, event.target.value)}><option value="smooth">Smooth</option><option value="raw">Raw</option></select><button type="button" className="tb-scale" onClick={() => toggleLogScale(tag)}>{logarithmic ? 'Log' : 'Linear'}</button></div>
        <div className="tb-axis-units"><span>Y · {logarithmic ? `log10(${unit})` : unit}</span><span>X · optimizer step</span></div>
        <div className="tb-plot"><div className="tb-y-labels" aria-hidden="true">{yTicks.map((tick, index) => <span key={`${tag}-${tick.top}-${index}`} style={{ top: `${tick.top}%` }}>{logarithmic ? <PowerLabel exponent={tick.exponent} /> : tick.label}</span>)}</div><div className="tb-plot-area"><svg viewBox="0 0 220 86" preserveAspectRatio="none" role="img" aria-label={`${tag} ${logarithmic ? 'logarithmic' : 'linear'} scale`}><defs><linearGradient id={`line-tube-${tag.replace(/[^a-zA-Z0-9_-]/g, '-')}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#d5fbff" /><stop offset="16%" stopColor="#27dfff" /><stop offset="52%" stopColor="#087ff5" /><stop offset="82%" stopColor="#5140df" /><stop offset="100%" stopColor="#a818ff" /></linearGradient></defs>{logarithmic && yTicks.map(tick => <line key={`grid-${tag}-${tick.exponent}`} className="tb-grid-line" x1="0" y1={(tick.top / 100) * 86} x2="220" y2={(tick.top / 100) * 86} />)}<line x1="0" y1="86" x2="220" y2="86" /><line x1="0" y1="0" x2="0" y2="86" /><polyline className="tb-tube-body" stroke={`url(#line-tube-${tag.replace(/[^a-zA-Z0-9_-]/g, '-')})`} points={svgPoints} />{singlePoint && <circle className="tb-single-point" cx={singlePoint.x} cy={singlePoint.y} r="3" />}</svg><div className="tb-x-axis">{scalarAxisTicks.map((tick, index) => <span key={`${tag}-x-${index}`} style={{ left: `${tick.position}%` }}>{tick.step.toLocaleString()}</span>)}</div></div></div>
        <div className="tb-values"><span>step {latest?.step ?? '—'}</span><b>{metricFmt(latest?.value)}</b></div>
      </article>;
    }) || <span className="empty">No scalar data found in the event file.</span>}</div>
  </section>;
}

function Card({ icon, label, value, title, accent, note }) { return <div className={`card ${accent}`}><div className="card-top"><span>{icon}</span><label>{label}</label></div><strong title={title || undefined}>{value}</strong><small>{note}</small></div>; }
function HealthMetric({ label, point, values = [], invert = false }) {
  const ordered = [...values].sort((a, b) => Number(a.step) - Number(b.step));
  const current = ordered.at(-1) || point;
  const recent = ordered.slice(-12).map(item => Number(item.value)).filter(Number.isFinite);
  const average = recent.length ? recent.reduce((sum, value) => sum + value, 0) / recent.length : null;
  const currentValue = current == null ? null : Number(current.value);
  const improving = average == null || !Number.isFinite(currentValue) ? null : invert ? currentValue < average : currentValue > average;
  const previous = ordered.length > 1 ? Number(ordered.at(-2)?.value) : null;
  const changedFromPrevious = Number.isFinite(currentValue) && Number.isFinite(previous) && currentValue !== previous;
  const improvedFromPrevious = changedFromPrevious && (invert ? currentValue < previous : currentValue > previous);
  const previousClass = !changedFromPrevious ? '' : improvedFromPrevious ? 'up' : 'down';
  const previousArrow = !changedFromPrevious ? '→' : improvedFromPrevious ? '↑' : '↓';
  return <div className="health-metric"><span>{label}</span><b title={average == null ? undefined : `Recent moving average: ${metricFmt(average)}`} className={improving == null ? '' : improving ? 'metric-improving' : 'metric-declining'}>{current ? metricFmt(currentValue) : '—'}</b><small className="moving-average">moving average {average == null ? '—' : metricFmt(average)}</small><small className={previousClass}>{previous == null ? 'previous —' : `vs previous ${previousArrow} ${metricFmt(previous)} → ${metricFmt(currentValue)}`}</small></div>;
}
function Meta({ label, value }) { return <div className="meta-row"><span>{label}</span><b>{value || '—'}</b></div>; }

createRoot(document.getElementById('root')).render(<App />);
