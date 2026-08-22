import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Cpu, Gauge, RefreshCw } from 'lucide-react';
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
// Six validations balances responsiveness with noisy COCO metrics.
const VALIDATION_MOVING_AVERAGE_WINDOW = 6;
const LOG_EPSILON = 1e-6;

function elapsedLabel(timestamp, now = Date.now()) {
  if (!Number.isFinite(timestamp)) return '—';
  const seconds = Math.max(0, Math.round((now - timestamp) / 1000));
  if (seconds < 10) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  return days ? `${days}d ${hours % 24}h ${minutes % 60}m ago` : `${hours}h ${minutes % 60}m ago`;
}

function localDateLabel(timestamp) {
  return timestamp == null ? null : new Date(timestamp).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
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
  const [run, setRun] = useState(null);
  const [data, setData] = useState(null);
  const [scale, setScale] = useState('linear');
  const [lossWindow, setLossWindow] = useState('all');
  const [lossBucketCount, setLossBucketCount] = useState(48);
  const [theme, setTheme] = useState(() => localStorage.getItem('pixienn-monitor-theme') || 'parchment');
  const [loading, setLoading] = useState(false);
  const [clockNow, setClockNow] = useState(Date.now());

  useEffect(() => {
    localStorage.setItem('pixienn-monitor-theme', theme);
    const root = document.getElementById('root');
    if (root) {
      root.classList.remove('theme-parchment', 'theme-night', 'theme-terminal', 'theme-brown', 'theme-davinci', 'theme-anime', 'theme-sico');
      root.classList.add(`theme-${theme}`);
    }
  }, [theme]);

  useEffect(() => {
    const id = setInterval(() => setClockNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const loadRuns = () => fetch('/api/runs', { cache: 'no-store' })
    .then(response => { if (!response.ok) throw new Error(`Run list failed (${response.status})`); return response.json(); })
    .then(names => {
      setRuns(names);
      setRun(current => names.length && (!current || !names.includes(current)) ? names[0] : current);
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
    if (!run) return undefined;
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
  const bucketCount = Math.min(lossBucketCount, history.length);
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
  const movingAverageWindow = VALIDATION_MOVING_AVERAGE_WINDOW;

  return <main className={`theme-${theme}`}>
    <header><div className="run-picker"><span>RUN</span><select value={run} onChange={event => setRun(event.target.value)}>{runs.map(name => <option key={name}>{name}</option>)}</select><button onClick={() => { loadRuns(); refresh(); }}><RefreshCw size={16} /></button><label className="theme-picker"><span>THEME</span><select aria-label="Theme" value={theme} onChange={event => setTheme(event.target.value)}><option value="parchment">Parchment</option><option value="night">Night control room</option><option value="terminal">Terminal green</option><option value="brown">Cocoa</option><option value="davinci">Da Vinci 95</option><option value="anime">Japanese ink anime</option><option value="sico">Sico robot</option></select></label></div></header>

    <nav><small>updated {data ? new Date(data.updatedAt).toLocaleTimeString() : '—'}</small></nav>
    {loading && <div className="monitor-loading-overlay" role="status" aria-live="polite"><div className="monitor-loading-dialog"><span className="monitor-loading-spinner" /><strong>Loading training history</strong><small>Reading event data…</small></div></div>}

    <section className="pixienn-banner" aria-label="PixieNN neural network illustration">
      <div className="banner-meta"><Meta label="mode" value={meta.mode} /><Meta label="resumed" value={meta.started_utc ? localDateLabel(Date.parse(meta.started_utc)) : null} /><Meta label="training since" value={localDateLabel(data?.earliestTrainingEvent?.at)} /><Meta label="training duration" value={data?.earliestTrainingEvent?.at ? elapsedLabel(data.earliestTrainingEvent.at, clockNow) : null} /><Meta label="configuration" value={meta.configuration?.split('/').pop()} /></div>
      <div className="sico-face" aria-hidden="true"><i /><i /></div>
      <div className="sico-banner-copy"><span>SICO // TRAINING COMPANION</span><strong>KEEP WATCH. KEEP LEARNING.</strong><em>status: observing the green lights</em></div>
    </section>

    <section className="cards">
      <LossCard latest={latest} trend={data?.trend} losses={scalarSeries['avg-loss']} />
      <StepProgressCard progress={data?.progress} />
      <LearningRateCard latest={latest} rates={data?.eventFile?.series?.['learning-rate']} policy={data?.learningRatePolicy} />
    </section>

    <section className="health-panel panel">
      <div className="health-heading"><div><span className="kicker">VALIDATION HEALTH</span><h2>Is the detector improving?</h2><p className="health-legend">Numbers compare the latest validation with the moving average of the last {movingAverageWindow} validations.</p></div><span className={`freshness ${freshness}`}><span /> {freshness}</span></div>
      <div className="health-grid">
        <HealthMetric label="mAP50" point={validationMetric('mAP50')} values={scalarSeries['mAP50']} window={movingAverageWindow} />
        <HealthMetric label="Micro-Avg-F1" point={validationMetric('micro-avg-f1')} values={scalarSeries['micro-avg-f1']} window={movingAverageWindow} />
        <HealthMetric label="Avg. recall" point={validationMetric('avg-recall')} values={scalarSeries['avg-recall']} window={movingAverageWindow} />
        <HealthMetric label="Val. loss" point={validationMetric('avg-val-loss')} values={scalarSeries['avg-val-loss']} window={movingAverageWindow} invert />
      </div>
      <div className="health-footer"><ValidationClock schedule={data?.validationSchedule} now={clockNow} /><span>confidence threshold <b>{data?.validationThreshold == null ? '—' : data.validationThreshold}</b></span><span>events <b>{eventAge == null ? '—' : eventAge < 1000 ? 'just now' : `${Math.round(eventAge / 1000)}s ago`}</b></span><span>checkpoint <b>{checkpointAge == null ? '—' : checkpointAge < 1000 ? 'just now' : `${Math.round(checkpointAge / 60000)}m ago`}</b></span></div>
    </section>

    <section className="grid">
      <div className="panel chart">
        <div className="panel-head"><div><span className="kicker">LOSS TRAJECTORY</span><h2>Average loss</h2></div><div style={{ display: 'flex', gap: 8 }}><select value={lossWindow} onChange={event => setLossWindow(event.target.value)}><option value="all">Full run · auto-scaled</option><option value="10000">Last 10,000 steps</option><option value="2000">Last 2,000 steps</option><option value="500">Last 500 steps</option></select><select value={scale} onChange={event => setScale(event.target.value)}><option value="linear">Linear</option><option value="log">Log scale</option></select><select aria-label="Average loss bucket count" value={lossBucketCount} onChange={event => setLossBucketCount(Number(event.target.value))}><option value="24">24 buckets</option><option value="48">48 buckets</option><option value="72">72 buckets</option><option value="120">120 buckets</option></select></div></div>
        <div className="loss-caption"><span>{scale === 'log' ? 'Avg. loss · log10(loss)' : 'Avg. loss'}</span><span className="loss-legend"><i className="low" /> lower loss <i className="high" /> higher loss</span><span>{lossWindow === 'all' ? 'Full training history' : `Last ${Number(lossWindow).toLocaleString()} optimizer steps`}</span></div>
        <div style={{ display: 'flex', gap: 10, width: '100%' }}><div className="loss-y-labels">{scale === 'log' ? logAxisData.ticks.map(tick => <span key={tick.value} style={{ top: `${(1 - (tick.value - logAxisData.minimum) / logAxisData.range) * 100}%` }}><PowerLabel exponent={tick.exponent} /></span>) : <><span style={{ top: '0%' }}>{max.toPrecision(3)}</span><span style={{ top: '50%' }}>{((max + min) / 2).toPrecision(3)}</span><span style={{ top: '100%' }}>{min.toPrecision(3)}</span></>}</div><div className="spark" style={{ flex: 1, width: '100%', minWidth: 0, height: 260 }}>{scale === 'log' && logAxisData.ticks.map(tick => <span className="loss-grid-line" key={`loss-grid-${tick.value}`} style={{ bottom: `${((tick.value - logAxisData.minimum) / logAxisData.range) * 100}%` }} />)}{points.length ? points.map((point, i) => <div className="loss-bar" key={i} style={{ height: `${height(point.avg)}%`, '--bar-hue': `${lossHue(point.avg)}` }} title={`step ${point.step} · avg loss ${point.avg.toFixed(3)}`} />) : <span className="empty">Waiting for event-file data…</span>}</div></div>
        <div className="axis loss-x-axis" aria-label="optimizer steps">{axisTicks.map((tick, index) => <span key={`${tick.step}-${index}`} style={{ left: `${tick.position}%` }}>{tick.step.toLocaleString()}</span>)}</div>
        <p className="chart-explanation">Average loss is the model's training error: a summary of how far its predictions are from the expected outputs. It is calculated using the loss functions defined by the selected model. Lower loss generally means the model is fitting the training examples better, but loss is not itself a measure of accuracy.</p>
      </div>
    </section>
    <section className="validation-card-grid">
      <ValidationBarsPanel series={scalarSeries} tag="mAP50" label="mAP50" />
      <ValidationBarsPanel series={scalarSeries} tag="micro-avg-f1" label="Micro-Avg-F1" />
    </section>
    <PRCurvePanel curve={data?.eventFile?.prCurves?.['validation/micro-pr/curve'] || data?.eventFile?.prCurves?.['validation/micro-pr/pr_curves']} />
    <ConfusionMatrixPanel matrix={data?.eventFile?.confusionMatrix} />
    <ValidationGallery image={data?.eventFile?.images?.['validation/error-gallery']} />
    <TrainingActivityPanel activity={data?.eventFile?.activity} />
    <EventScalarsPanel run={run} series={data?.eventFile?.series || {}} tails={data?.eventFile?.tails || {}} />
    <footer><span><span className="live-dot" /> event stream connected</span><span>PIXIENN / LOCAL RUN OBSERVATORY</span></footer>
  </main>;
}

function ConfusionMatrixPanel({ matrix }) {
  if (!matrix?.values || !matrix?.size) return <section className="panel detection-profile"><div className="panel-head"><div><span className="kicker">VALIDATION COUNTS</span><h2>Detection profile</h2></div></div><div className="empty">Waiting for validation counts…</div></section>;
  const labels = matrix.labels?.length === matrix.size ? matrix.labels : Array.from({ length: matrix.size }, (_, i) => i === matrix.size - 1 ? 'background' : `class ${i}`);
  const values = matrix.values.map(Number);
  const background = matrix.size - 1;
  const classLabels = labels.slice(0, background);
  const metricValues = metric => classLabels.map((_, cls) => {
    if (metric === 'TP') return values[cls * matrix.size + cls] || 0;
    if (metric === 'FP') return values.reduce((sum, value, i) => sum + (i % matrix.size === cls && Math.floor(i / matrix.size) !== cls ? value : 0), 0);
    return values.slice(cls * matrix.size, (cls + 1) * matrix.size).reduce((sum, value, col) => sum + (col !== cls ? value : 0), 0);
  });
  return <section className="panel detection-profile">
    <div className="panel-head"><div><span className="kicker">VALIDATION COUNTS · PER CLASS</span><h2>Class detection profile</h2></div></div>
    <CompactErrorHeatmap labels={classLabels} tp={metricValues('TP')} fp={metricValues('FP')} fn={metricValues('FN')} />
    <p className="pr-insight">Each class is shown as a composition of true-positive, false-positive, and false-negative results. Wedge size uses <strong>2×TP</strong>, FP, and FN so the true-positive share corresponds directly to the F1 calculation; the key below each class gives the exact counts. Greener means a stronger positive TP contribution, while redder means a larger FP or FN penalty. The centered score is that class&apos;s F1, calculated as <strong>2 × TP ÷ (2 × TP + FP + FN)</strong>. Use the view control to switch between pie wedges and independently scaled bars.</p>
  </section>;
}

function CompactErrorHeatmap({ labels, tp, fp, fn }) {
  const [view, setView] = useState('pies');
  const columns = [{ key: 'tp', label: 'TP', values: tp }, { key: 'fp', label: 'FP', values: fp }, { key: 'fn', label: 'FN', values: fn }];
  const classQuality = row => { const denominator = 2 * tp[row] + fp[row] + fn[row]; return denominator ? (2 * tp[row]) / denominator : 0; };
  const impactStyle = (key, value, row) => {
    const denominator = 2 * tp[row] + fp[row] + fn[row];
    const impact = denominator ? (key === 'tp' ? 2 * value : value) / denominator : 0;
    const positive = key === 'tp';
    const hue = positive ? 68 + impact * 58 : 38 - impact * 38;
    const saturation = positive ? 88 : 96;
    const topLightness = positive ? 30 + impact * 58 : 88 - impact * 48;
    const bottomLightness = positive ? 18 + impact * 48 : 68 - impact * 48;
    return {
      '--bar-color': `hsl(${hue} ${saturation}% ${topLightness}%)`,
      '--bar-top-color': `hsl(${hue} ${saturation}% ${topLightness}%)`,
      '--bar-bottom-color': `hsl(${hue} ${saturation}% ${bottomLightness}%)`,
      '--bar-border-color': `hsl(${hue} ${saturation}% ${Math.max(16, bottomLightness - 8)}%)`
    };
  };
  const classMaximum = row => Math.max(1, ...columns.map(column => column.values[row]));
  const impactColor = (key, value, row) => impactStyle(key, value, row)['--bar-color'];
  return <div className="compact-heatmap">
    <div className="class-profile-controls"><label>VIEW <select aria-label="Class detection profile view" value={view} onChange={event => setView(event.target.value)}><option value="bars">Bars</option><option value="pies">Pie wedges</option></select></label></div>
    {view === 'bars' ? <div className="class-bar-chart">
      <div className="class-bar-groups">
        {labels.map((label, row) => {
          const score = classQuality(row);
          return <div className="class-bar-group" key={label}>
            <div className="class-bar-scale"><span>{classMaximum(row).toLocaleString()}</span><span>0</span></div>
            <div className="class-bar-columns">{columns.map(column => {
              const value = column.values[row];
              return <div className={`class-bar ${column.key}`} key={`${label}-${column.key}`} style={{ ...impactStyle(column.key, value, row), height: `${value / classMaximum(row) * 100}%` }} title={`${label} · ${column.label}: ${value.toLocaleString()}`}><span>{value.toLocaleString()}</span><small>{column.label}</small></div>;
            })}</div>
            <div className="class-bar-f1" style={{ '--quality-hue': `${120 * score}` }} title={`${label} · F1 ${score.toFixed(3)}`}>F1 {score.toFixed(2)}</div>
            <div className="class-bar-label" title={label}>{label}</div>
          </div>;
        })}
      </div>
    </div> : <div className="class-pie-chart">
      {labels.map((label, row) => {
        const denominator = 2 * tp[row] + fp[row] + fn[row];
        const tpEnd = denominator ? 2 * tp[row] / denominator * 100 : 0;
        const fpEnd = denominator ? tpEnd + fp[row] / denominator * 100 : 0;
        const score = classQuality(row);
        const piePath = (start, end) => {
          const point = percentage => {
            const angle = percentage / 100 * Math.PI * 2;
            return [50 + 48 * Math.sin(angle), 50 - 48 * Math.cos(angle)];
          };
          const [startX, startY] = point(start);
          const [endX, endY] = point(end);
          return `M 50 50 L ${startX} ${startY} A 48 48 0 ${end - start > 50 ? 1 : 0} 1 ${endX} ${endY} Z`;
        };
        const pieSegments = [
          { key: 'tp', label: 'TP', value: tp[row], start: 0, end: tpEnd },
          { key: 'fp', label: 'FP', value: fp[row], start: tpEnd, end: fpEnd },
          { key: 'fn', label: 'FN', value: fn[row], start: fpEnd, end: 100 }
        ].filter(segment => segment.value > 0);
        const f1Id = `class-pie-f1-${row}`;
        return <div className="class-pie-group" key={label}>
          <div className="class-pie"><svg viewBox="0 0 100 100" role="img" aria-label={`${label}: TP ${tp[row]}, FP ${fp[row]}, FN ${fn[row]}, F1 ${score.toFixed(2)}`}><defs>{pieSegments.map(segment => { const style = impactStyle(segment.key, segment.value, row); return <linearGradient key={segment.key} id={`class-pie-${row}-${segment.key}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={style['--bar-top-color']} /><stop offset="100%" stopColor={style['--bar-bottom-color']} /></linearGradient>; })}<linearGradient id={f1Id} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={`hsl(${120 * score} 100% 86%)`} /><stop offset="48%" stopColor={`hsl(${120 * score} 100% 70%)`} /><stop offset="100%" stopColor={`hsl(${120 * score} 100% 53%)`} /></linearGradient></defs>{pieSegments.map(segment => <path key={segment.key} d={piePath(segment.start, segment.end)} fill={`url(#class-pie-${row}-${segment.key})`} stroke="#f6ead3" strokeWidth="1.5"><title>{`${label} · ${segment.label}: ${segment.value.toLocaleString()}`}</title></path>)}<circle cx="50" cy="50" r="22" fill={`url(#${f1Id})`} stroke="#ffffff88" strokeWidth="1" /><text x="50" y="51" textAnchor="middle">{score.toFixed(2)}</text></svg></div>
          <div className="class-pie-label">{label}</div>
          <div className="class-pie-keys">{pieSegments.map(segment => { const style = impactStyle(segment.key, segment.value, row); return <span key={segment.key} style={{ '--pie-top-color': style['--bar-top-color'], '--pie-bottom-color': style['--bar-bottom-color'] }}><i />{segment.label} {segment.value.toLocaleString()}</span>; })}</div>
        </div>;
      })}
    </div>}
  </div>;
}

function ValidationGallery({ image }) {
  return <section className="panel validation-gallery">
    <div className="panel-head"><h2>Validation gallery</h2></div>
    {image ? <img src={image.data} alt={`Validation predictions and errors at step ${image.step}`} /> : <div className="empty gallery-empty">A gallery is written after the configured interval when at least one validation prediction clears the confidence threshold.</div>}
    <p className="chart-explanation">Green boxes are matched predictions, red boxes are false positives, and yellow boxes are missed ground-truth objects.</p>
  </section>;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const totalMinutes = Math.round(Math.max(0, seconds) / 60);
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  return days ? `${days}d ${hours}h ${minutes}m` : hours ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function TrainingActivityPanel({ activity }) {
  if (!activity) {
    return <section className="training-activity panel"><div className="panel-head"><div><span className="kicker">RUN ACTIVITY</span><h2>Training time vs. gaps</h2></div></div><div className="empty">Waiting for enough timestamped training events…</div></section>;
  }
  const start = Number(activity.start);
  const end = Number(activity.end);
  const crossesDayBoundary = new Date(start).toDateString() !== new Date(end).toDateString();
  // Activity durations are reported in seconds, while timestamps are in
  // milliseconds. Keep the timeline widths in the same unit as segments.
  const span = Math.max(1, (end - start) / 1000);
  const visibleSegments = (activity.segments || [])
    .filter(segment => segment.seconds > 0)
    .reduce((segments, segment) => {
      const previous = segments[segments.length - 1];
      if (previous && previous.kind === segment.kind) previous.seconds += segment.seconds;
      else segments.push({ ...segment });
      return segments;
    }, []);
  const tickCount = Math.min(7, Math.max(3, Math.ceil(span / (4 * 60 * 60)) + 1));
  const ticks = Array.from({ length: tickCount }, (_, index) => {
    const position = index / (tickCount - 1);
    const timestamp = start + (end - start) * position;
    const date = new Date(timestamp);
    const label = crossesDayBoundary || span >= 24 * 60 * 60
      ? date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric' })
      : date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
    return { label, position };
  });
  return <section className="training-activity panel">
    <div className="panel-head"><div><span className="kicker">RUN ACTIVITY · INFERRED FROM EVENT TIMESTAMPS</span><h2>Training time vs. gaps</h2></div><span className="training-activity-range">{new Date(start).toLocaleDateString()} → {new Date(end).toLocaleDateString()}</span></div>
    <div className="activity-summary"><span><i className="active" /> active training <b>{formatDuration(activity.activeSeconds)}</b></span><span><i className="validation" /> validation <b>{formatDuration(activity.validationSeconds)}</b></span><span><i className="offline" /> offline gaps <b>{formatDuration(activity.offlineSeconds)}</b></span></div>
    <div className="activity-track" aria-label="Training activity timeline">{visibleSegments.map((segment, index) => <span key={`${segment.kind}-${index}`} className={segment.kind} style={{ flex: `0 0 ${segment.seconds / span * 100}%`, width: `${segment.seconds / span * 100}%`, background: segment.kind === 'active' ? '#355e3b' : segment.kind === 'validation' ? '#f4d35e' : '#8b1e1e' }} title={`${segment.kind} · ${formatDuration(segment.seconds)}`} />)}</div>
    <div className="activity-axis">{ticks.map((tick, index) => <span key={`${tick.position}-${index}`} className={index === 0 ? 'first' : index === ticks.length - 1 ? 'last' : ''} style={{ left: `${tick.position * 100}%` }}>{tick.label}</span>)}</div>
  </section>;
}

function ValidationBarsPanel({ series, tag, label }) {
  const [scale, setScale] = useState('linear');
  const [bucketCount, setBucketCount] = useState(48);
  const [range, setRange] = useState('all');
  const byStep = new Map();
  (series[tag] || []).forEach(point => {
    const step = Number(point.step);
    const value = Number(point.value);
    if (!Number.isFinite(step) || !Number.isFinite(value)) return;
    const bucket = byStep.get(step) || { step };
    bucket.value = Math.max(0, Math.min(1, value));
    byStep.set(step, bucket);
  });
  const history = [...byStep.values()].sort((left, right) => left.step - right.step);
  const visibleHistory = range === 'all' ? history : history.slice(-Number(range));
  const visibleBucketCount = Math.min(bucketCount, visibleHistory.length);
  const bucketSize = Math.max(1, Math.ceil(visibleHistory.length / Math.max(visibleBucketCount, 1)));
  const buckets = Array.from({ length: Math.ceil(visibleHistory.length / bucketSize) }, (_, index) => {
    const points = visibleHistory.slice(index * bucketSize, (index + 1) * bucketSize);
    const values = points.map(point => point.value).filter(Number.isFinite);
    return { step: points.at(-1)?.step, value: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null };
  });
  const firstStep = buckets[0]?.step;
  const lastStep = buckets.at(-1)?.step;
  const values = buckets.map(bucket => bucket.value).filter(Number.isFinite);
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 1;
  const linearRange = Math.max(maximum - minimum, LOG_EPSILON);
  const logValues = values.map(value => Math.log10(Math.max(value, LOG_EPSILON)));
  const logAxisData = logAxis(logValues);
  const logMinimum = logAxisData.minimum;
  const logMaximum = logAxisData.maximum;
  const logRange = logAxisData.range;
  const logTicks = logAxisData.ticks;
  const height = value => {
    if (!Number.isFinite(value)) return 0;
    if (scale === 'log') return Math.max(0, Math.min(100, (Math.log10(Math.max(value, LOG_EPSILON)) - logMinimum) / logRange * 100));
    return values.length === 1 || maximum === minimum ? 100 : Math.max(0, Math.min(100, (value - minimum) / linearRange * 100));
  };
  const hue = value => Math.round(height(value) * 1.2);
  return <div className="validation-bars panel chart">
    <div className="panel-head"><div><span className="kicker">VALIDATION HISTORY · BUCKETED</span><h2>{label}</h2></div><div style={{ display: 'flex', gap: 8 }}><select aria-label={`${label} validation history range`} value={range} onChange={event => setRange(event.target.value)}><option value="all">Full run</option><option value="100">Last 100</option><option value="20">Last 20</option></select><select aria-label={`${label} scale`} value={scale} onChange={event => setScale(event.target.value)}><option value="linear">Linear</option><option value="log">Log</option></select><select aria-label={`${label} bucket count`} value={bucketCount} onChange={event => setBucketCount(Number(event.target.value))}><option value="24">24 buckets</option><option value="48">48 buckets</option><option value="72">72 buckets</option><option value="120">120 buckets</option></select></div></div>
    <div className="validation-bars-caption"><span>{scale === 'log' ? 'Validation score · log10(score)' : 'Validation score'} · higher is better</span><span>{visibleHistory.length} validations · {range === 'all' ? 'full run' : `last ${Number(range)}`} · bucketed averages</span></div>
    {buckets.length ? <><div className="validation-bars-chart"><div className="loss-y-labels">{scale === 'log' ? [...logTicks].reverse().map(tick => <span key={tick.value}><PowerLabel exponent={tick.exponent} /></span>) : <><span>{maximum.toPrecision(3)}</span><span>{((maximum + minimum) / 2).toPrecision(3)}</span><span>{minimum.toPrecision(3)}</span></>}</div><div className="spark validation-spark" style={{ flex: 1, width: '100%', minWidth: 0, height: 230 }}>{scale === 'log' && logTicks.map(tick => <span className="loss-grid-line" key={`validation-grid-${tick.value}`} style={{ bottom: `${((tick.value - logMinimum) / logRange) * 100}%` }} />)}{buckets.map((bucket, index) => <div className="loss-bar" key={bucket.step ?? index} style={{ height: `${height(bucket.value)}%`, '--bar-hue': `${hue(bucket.value)}` }} title={`step ${Number(bucket.step).toLocaleString()} · ${label} ${Number(bucket.value).toFixed(3)}`} />)}</div></div><div className="axis loss-x-axis validation-bars-axis" aria-label={`${label} validation steps`}><span>step {Number(firstStep).toLocaleString()}</span><span>step {Number(lastStep).toLocaleString()}</span></div>{tag === 'micro-avg-f1' && <p className="validation-bars-note validation-bars-explanation">Micro-Avg-F1 is one score for the whole validation set. It pools all images and classes, counting correct detections, incorrect detections, and missed ground-truth objects. Precision is correct detections divided by all detections; recall is correct detections divided by all ground-truth objects. F1 combines precision and recall, so it is high only when both are high.</p>}{tag === 'mAP50' && <p className="validation-bars-note validation-bars-explanation">mAP50 summarizes detection quality across the validation set. A prediction counts as correct when it has the right class and its box overlaps the matching ground-truth box by at least 50%. For each class, detections are ranked by confidence to form a precision–recall curve; the area under that curve is the class's average precision. mAP50 is the average of those class scores, so higher values mean better performance across the classes represented in the validation set.</p>}</> : <div className="empty validation-bars-empty">Waiting for validation data…</div>}
  </div>;
}

function MapProgressPanel({ data }) {
  const raw = (data?.eventFile?.series?.mAP50 || []).filter(point => Number.isFinite(Number(point.value)));
  const points = raw.length > 120 ? raw.filter((_, index) => index % Math.ceil(raw.length / 120) === 0 || index === raw.length - 1) : raw;
  const width = 1000;
  const height = 230;
  const values = points.map(point => Number(point.value));
  const minimum = values.length ? Math.max(0, Math.floor((Math.min(...values) - 0.02) * 100) / 100) : 0;
  const maximum = values.length ? Math.min(1, Math.ceil((Math.max(...values) + 0.02) * 100) / 100) : 1;
  const range = Math.max(maximum - minimum, 0.01);
  const coordinate = (point, index) => ({
    x: points.length < 2 ? width / 2 : index / (points.length - 1) * width,
    y: height - 20 - (Number(point.value) - minimum) / range * (height - 40)
  });
  const coords = points.map(coordinate);
  const linePoints = coords.map(point => `${point.x},${point.y}`).join(' ');
  const best = raw.reduce((winner, point) => Number(point.value) > Number(winner?.value ?? -Infinity) ? point : winner, null);
  const latest = raw.at(-1);
  const fmtPercent = value => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
  return <section className="map-progress-panel panel">
    <div className="map-progress-head"><div><span className="kicker">VALIDATION TRAJECTORY</span><h2>mAP50 movement across training</h2><p>Higher on the chart means better validation performance.</p></div><span className={`timeline-status ${data?.active ? 'live' : 'idle'}`}><i />{data?.active ? 'training live' : 'idle'}</span></div>
    {coords.length ? <><div className="map-chart"><div className="map-y-axis"><span>{maximum.toFixed(2)}</span><span>{((maximum + minimum) / 2).toFixed(2)}</span><span>{minimum.toFixed(2)}</span></div><svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="mAP50 validation movement"><line className="map-grid" x1="0" y1="20" x2={width} y2="20" /><line className="map-grid" x1="0" y1={height / 2} x2={width} y2={height / 2} /><line className="map-grid" x1="0" y1={height - 20} x2={width} y2={height - 20} /><polyline className="map-line" points={linePoints} />{coords.map((point, index) => { const source = points[index]; const isLatest = index === coords.length - 1; const isBest = Number(source.value) === Number(best?.value); if (!isLatest && !isBest) return null; return <circle key={index} className={`map-point ${isLatest ? 'latest' : ''} ${isBest ? 'best' : ''}`} cx={point.x} cy={point.y} r="5"><title>{`${isLatest ? 'Latest · ' : ''}${isBest ? 'Best · ' : ''}step ${Number(source.raw_step ?? source.step).toLocaleString()} · mAP50 ${fmtPercent(source.value)}`}</title></circle>; })}</svg></div><div className="map-x-axis"><span>step {Number(points[0].raw_step ?? points[0].step).toLocaleString()}</span><span>step {Number(points.at(-1).raw_step ?? points.at(-1).step).toLocaleString()}</span></div><div className="map-summary"><span><small>VALIDATIONS</small><b>{raw.length}</b></span><span><small>LATEST mAP50</small><b>{fmtPercent(latest?.value)}</b></span><span><small>BEST mAP50</small><b>{fmtPercent(best?.value)}</b><em>step {best ? Number(best.raw_step ?? best.step).toLocaleString() : '—'}</em></span></div></> : <div className="empty map-empty">Waiting for validation data…</div>}
  </section>;
}

function ValidationClock({ schedule, now }) {
  return <div className="validation-clock"><span>last validation <b>{elapsedLabel(schedule?.lastAt, now)}</b>{schedule?.lastStep != null && <small>step {Number(schedule.lastStep).toLocaleString()}</small>}</span><span>next likely <b>{countdownLabel(schedule?.nextAt, now)}</b>{schedule?.nextStep != null && <small>step {Number(schedule.nextStep).toLocaleString()}</small>}</span></div>;
}

function PRCurvePanel({ curve }) {
  const sampledPoints = [...(curve?.points || [])]
    .filter(point => Number.isFinite(Number(point.precision)) && Number.isFinite(Number(point.recall)));
  const points = [...sampledPoints]
    .sort((lhs, rhs) => Number(lhs.recall) - Number(rhs.recall));
  const xMaximum = 1;
  const yMaximum = 1;
  const coordinate = point => ({
    x: 2 + Math.max(0, Math.min(xMaximum, Number(point.recall))) / xMaximum * 216,
    y: 85 - Math.max(0, Math.min(yMaximum, Number(point.precision))) / yMaximum * 84
  });
  const axisLabel = value => value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  const activePoints = sampledPoints.filter(point => Number(point.precision) > 0 || Number(point.recall) > 0);
  const lowerThreshold = activePoints.reduce((lowest, point) => (
    !lowest || Number(point.confidence) < Number(lowest.confidence) ? point : lowest
  ), null);
  const upperThreshold = activePoints.reduce((highest, point) => (
    !highest || Number(point.confidence) > Number(highest.confidence) ? point : highest
  ), null);
  const middleThreshold = lowerThreshold && upperThreshold
    ? activePoints.reduce((middle, point) => {
      const midpoint = (Number(lowerThreshold.confidence) + Number(upperThreshold.confidence)) / 2;
      return !middle || Math.abs(Number(point.confidence) - midpoint) < Math.abs(Number(middle.confidence) - midpoint)
        ? point
        : middle;
    }, null)
    : null;
  const percent = value => `${Math.round(Number(value) * 100)}%`;
  const confidenceLabel = point => Number(point.confidence).toFixed(3);
  const thresholdInsight = (point, description) => {
    const precision = Number(point.precision);
    const recall = Number(point.recall);
    const recallText = recall >= 0.9995
      ? 'It finds essentially all of the ground-truth objects.'
      : `It misses about ${percent(1 - recall)} of the ground-truth objects (false negatives).`;
    const precisionText = precision >= 0.9995
      ? 'Nearly all of its detections are correct.'
      : `About ${percent(1 - precision)} of its detections are false positives.`;
    return <p className="pr-insight">At confidence {confidenceLabel(point)}—{description}—the model&apos;s recall is about {percent(recall)}. {recallText} Its precision is about {percent(precision)}. {precisionText}</p>;
  };
  const areaPoints = [];
  for (const point of points) {
    const recall = Number(point.recall);
    const precision = Number(point.precision);
    const previous = areaPoints[areaPoints.length - 1];
    if (previous && Math.abs(previous.recall - recall) < 1e-12) {
      previous.precision = Math.max(previous.precision, precision);
    } else {
      areaPoints.push({ recall, precision });
    }
  }
  let runningPrecision = 0;
  for (let index = areaPoints.length - 1; index >= 0; --index) {
    runningPrecision = Math.max(runningPrecision, areaPoints[index].precision);
    areaPoints[index].precision = runningPrecision;
  }
  let prArea = 0;
  let previousRecall = 0;
  for (const point of areaPoints) {
    if (point.recall > previousRecall) {
      prArea += (point.recall - previousRecall) * point.precision;
      previousRecall = point.recall;
    }
  }
  const prScore = points.length === 0
    ? null
    : Math.max(0, Math.min(1, prArea));
  const envelopePoints = areaPoints.length && areaPoints[0].recall > 0
    ? [{ recall: 0, precision: areaPoints[0].precision }, ...areaPoints]
    : areaPoints;
  const smoothEnvelopePoints = envelopePoints.length < 2
    ? envelopePoints
    : Array.from({ length: 201 }, (_, index) => {
      const recall = index / 200;
      let right = 1;
      while (right < envelopePoints.length && envelopePoints[right].recall < recall) right++;
      if (right >= envelopePoints.length) {
        return { recall, precision: envelopePoints[envelopePoints.length - 1].precision };
      }
      const left = Math.max(0, right - 1);
      const before = envelopePoints[left];
      const after = envelopePoints[right];
      const span = after.recall - before.recall;
      const fraction = span > 0 ? (recall - before.recall) / span : 0;
      return {
        recall,
        precision: before.precision + (after.precision - before.precision) * fraction
      };
    });
  const svgPoints = smoothEnvelopePoints.map(point => {
    const current = coordinate(point);
    return `${current.x},${current.y}`;
  }).join(' ');
  const chartPoints = svgPoints || points.map(point => {
    const current = coordinate(point);
    return `${current.x},${current.y}`;
  }).join(' ');
  return <section className="tb-card pr-panel">
    <div className="tb-toggle"><span>Precision–recall curve</span></div>
    <div className="tb-axis-units"><span>Y · precision</span><span>X · recall</span></div>
    <div className="tb-plot"><div className="tb-y-labels"><span style={{ top: '0%' }}>{axisLabel(yMaximum)}</span><span style={{ top: '50%' }}>{axisLabel(yMaximum / 2)}</span><span style={{ top: '100%' }}>0</span></div><div className="tb-plot-area"><svg viewBox="0 0 220 86" preserveAspectRatio="none" role="img" aria-label="Precision versus recall curve"><line x1="0" y1="0" x2="220" y2="0" /><line x1="0" y1="43" x2="220" y2="43" /><line x1="0" y1="86" x2="220" y2="86" /><polyline className="tb-tube-body" points={chartPoints} fill="none" stroke="#087ff5" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" /></svg><div className="tb-x-axis"><span style={{ left: '50%' }}>{axisLabel(xMaximum / 2)}</span><span style={{ left: '100%' }}>{axisLabel(xMaximum)}</span></div></div></div>
    <div className="tb-values"><span>{points.length ? `${points.length} threshold points` : 'No PR data'}</span><span>micro PR score <b>{prScore == null ? '—' : prScore.toFixed(2)}</b></span><span>lowest-threshold precision <b>{lowerThreshold ? metricFmt(lowerThreshold.precision) : '—'}</b></span><span>lowest-threshold recall <b>{lowerThreshold ? metricFmt(lowerThreshold.recall) : '—'}</b></span></div>
    {prScore != null && <p className="pr-insight">A micro PR score of {prScore.toFixed(2)} summarizes the entire confidence sweep, not one cutoff. A score of 1.00 means the curve reaches the ideal precision–recall envelope: at one confidence setting, the model finds every ground-truth object and its detections are all correct.</p>}
    {lowerThreshold && thresholdInsight(lowerThreshold, 'the lowest threshold and most permissive setting')}
    {middleThreshold && thresholdInsight(middleThreshold, 'a middle confidence setting between the two extremes')}
    {upperThreshold && thresholdInsight(upperThreshold, 'the highest threshold that produces any detections')}
    <p className="pr-insight">The curve uses confidence cutoffs from 0.000 through 1.000 in 0.005 increments. Precision is the percentage of detections that are correct. Recall is the percentage of ground-truth objects the model finds.</p>
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
function LossCard({ latest, trend, losses = [] }) {
  const direction = trend?.direction || 'waiting';
  const marker = direction === 'improving' ? '↓' : direction === 'worsening' ? '↑' : direction === 'flat' ? '→' : '•';
  const label = direction === 'improving' ? 'improving' : direction === 'worsening' ? 'needs attention' : direction === 'flat' ? 'holding steady' : 'awaiting trend';
  const history = losses.filter(point => Number.isFinite(Number(point.step)) && Number.isFinite(Number(point.value))).slice(-60);
  const values = history.map(point => Number(point.value));
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 1;
  const range = Math.max(maximum - minimum, 1e-6);
  const points = history.map((point, index) => `${(history.length > 1 ? index / (history.length - 1) * 220 : 110).toFixed(1)},${(44 - (Number(point.value) - minimum) / range * 38).toFixed(1)}`).join(' ');
  return <div className={`card pink loss-card ${direction}`}>
    <div className="card-top"><span><Activity /></span><label>AVG LOSS</label></div>
    <strong>{fmt(latest?.avg)}</strong>
    <div className="loss-trend-chart"><svg viewBox="0 0 220 52" preserveAspectRatio="none" role="img" aria-label="Average loss trend"><line x1="0" y1="44" x2="220" y2="44" /><polyline points={points} /></svg></div>
    <div className="loss-status"><b>{marker}</b> {label}</div>
    <small>{trend?.average == null ? 'building trend' : `recent average ${fmt(trend.average)}`}</small>
  </div>;
}
function LearningRateCard({ latest, rates = [], policy }) {
  const history = rates.filter(point => Number.isFinite(Number(point.step)) && Number.isFinite(Number(point.value))).slice(-80);
  const values = history.map(point => Number(point.value));
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 1;
  const padding = Math.max((maximum - minimum) * 0.12, Math.abs(maximum) * 0.02, 1e-9);
  const low = minimum - padding;
  const high = maximum + padding;
  const points = history.map((point, index) => {
    const x = history.length > 1 ? index / (history.length - 1) * 220 : 110;
    const y = 44 - ((Number(point.value) - low) / (high - low)) * 38;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const first = history[0];
  const last = history.at(-1);
  return <div className="card lime learning-rate-card">
    <div className="card-top"><span><Cpu /></span><label>LEARNING RATE</label></div>
    <strong>{latest?.lr?.toFixed(9) || '—'}</strong>
    <div className="lr-trend-chart"><svg viewBox="0 0 220 52" preserveAspectRatio="none" role="img" aria-label="Learning-rate trend"><line x1="0" y1="44" x2="220" y2="44" /><polyline points={points} /></svg></div>
    <small>{policy ? `policy: ${policy}` : 'policy unavailable'}{first && last ? ` · steps ${Number(first.step).toLocaleString()} → ${Number(last.step).toLocaleString()}` : ''}</small>
  </div>;
}
function StepProgressCard({ progress }) {
  const percentage = progress?.percentage;
  const current = Number.isFinite(Number(progress?.currentBatches))
    ? Number(progress.currentBatches).toLocaleString()
    : '—';
  const target = Number.isFinite(Number(progress?.targetBatches))
    ? Number(progress.targetBatches).toLocaleString()
    : '—';
  const value = progress ? `${current} / ${target}` : '—';
  return <div className="card blue step-progress-card">
    <div className="card-top"><span><Gauge /></span><label>STEP PROGRESS</label></div>
    <strong>{value}</strong>
    <div className="step-progress-track" role="progressbar" aria-label="Training step progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow={percentage ?? 0}>
      <span style={{ width: `${Math.max(0, Math.min(100, percentage ?? 0))}%` }} />
    </div>
    <small>{percentage == null ? 'optimizer steps' : `${percentage.toFixed(1)}% complete`}</small>
  </div>;
}
function HealthMetric({ label, point, values = [], window = VALIDATION_MOVING_AVERAGE_WINDOW, invert = false }) {
  const ordered = [...values].sort((a, b) => Number(a.step) - Number(b.step));
  const current = ordered.at(-1) || point;
  const recent = ordered.slice(-window).map(item => Number(item.value)).filter(Number.isFinite);
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
