import fs from 'node:fs';
import path from 'node:path';
import { webcrypto } from 'node:crypto';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

// Vite expects the Web Crypto API. Some Node versions expose a partial
// `globalThis.crypto` object, so nullish assignment alone is not sufficient.
if (typeof globalThis.crypto?.getRandomValues !== 'function') {
  globalThis.crypto = webcrypto;
}
const { createServer: createViteServer } = await import('vite');

const monitorDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(monitorDir, '..');
const runs = path.join(root, 'runs');
const port = Number(process.env.PORT || 4173);

const checkpointCaches = new Map();
const eventReaders = new Map();
const snapshotRequests = new Map();
const LOSS_WINDOWS = new Set([500, 2000, 10000]);
const eventReaderTimeoutMs = Number(process.env.PIXIENN_EVENT_READER_TIMEOUT_MS || 300000);

function lossWindow(value) {
  const parsed = Number(value);
  return LOSS_WINDOWS.has(parsed) ? parsed : null;
}

function safeRun(name = 'yolov3') {
  const clean = name.replace(/[^a-zA-Z0-9_-]/g, '');
  const dir = path.join(runs, clean);
  if (!dir.startsWith(`${runs}${path.sep}`)) throw new Error('Invalid run');
  return { name: clean, dir };
}

function availableRuns() {
  if (!fs.existsSync(runs)) return [];
  return fs.readdirSync(runs, { withFileTypes: true })
    .filter(entry => entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== '.locks' && entry.name !== 'archive')
    .map(entry => entry.name)
    .sort();
}

function read(name) {
  try { return fs.readFileSync(name, 'utf8'); } catch { return ''; }
}

function validationThreshold(metadata) {
  if (metadata.validation_threshold != null) return Number(metadata.validation_threshold);
  const configuration = metadata.configuration;
  if (!configuration) return null;
  const configText = read(configuration);
  const modelMatch = configText.match(/^\s*model:\s*(\S+)\s*$/m);
  const modelPath = modelMatch
    ? path.resolve(path.dirname(configuration), modelMatch[1])
    : configuration;
  const match = read(modelPath).match(/^\s+threshold:\s*([0-9.+-eE]+)\s*$/m);
  return match ? Number(match[1]) : null;
}

function validationInterval(metadata) {
  if (metadata.validation_interval != null) return Number(metadata.validation_interval);
  const configuration = metadata.configuration;
  if (!configuration) return null;
  const configText = read(configuration);
  const modelMatch = configText.match(/^\s*model:\s*(\S+)\s*$/m);
  const modelPath = modelMatch
    ? path.resolve(path.dirname(configuration), modelMatch[1])
    : configuration;
  const modelText = read(modelPath);
  const match = modelText.match(/validation:[\s\S]{0,500}?^\s+interval:\s*(\d+)\s*$/m);
  return match ? Number(match[1]) : null;
}

function validationSchedule(metadata, eventFile) {
  const interval = validationInterval(metadata);
  const validationSeries = [...(eventFile.series['mAP50'] || [])]
    .filter(point => Number.isFinite(Number(point.step)) && Number.isFinite(Number(point.wall_time)))
    .sort((a, b) => Number(a.step) - Number(b.step));
  const trainingSeries = [...(eventFile.series['avg-loss'] || [])]
    .filter(point => Number.isFinite(Number(point.step)) && Number.isFinite(Number(point.wall_time)))
    .sort((a, b) => Number(a.step) - Number(b.step));
  const stepOf = point => Number(point.raw_step ?? point.step);
  const current = trainingSeries.at(-1);
  // Validation is scheduled on optimizer-step boundaries.  Do not derive the
  // next boundary from the last event: resumed runs can retain older events
  // whose step is offset (or ahead of the restored checkpoint).
  const last = current
    ? validationSeries.filter(point => stepOf(point) <= stepOf(current)).at(-1)
    : validationSeries.at(-1);
  if (!interval || !last) return { interval, lastStep: last?.step ?? null, lastAt: last?.wall_time ? last.wall_time * 1000 : null, nextStep: null, nextAt: null };

  const nextStep = current
    ? Math.floor(stepOf(current) / interval + 1) * interval
    : stepOf(last) + interval;
  const rates = [];
  for (let i = Math.max(1, trainingSeries.length - 20); i < trainingSeries.length; ++i) {
    const previous = trainingSeries[i - 1];
    const point = trainingSeries[i];
    const stepDelta = stepOf(point) - stepOf(previous);
    const timeDelta = Number(point.wall_time) - Number(previous.wall_time);
    if (stepDelta > 0 && timeDelta > 0) rates.push(timeDelta / stepDelta);
  }
  // Validation pauses and resume boundaries create large/out-of-order gaps in
  // the event stream. A median over the recent positive intervals gives a
  // useful moving training rate without letting one pause dominate the clock.
  rates.sort((a, b) => a - b);
  const secondsPerStep = rates.length
    ? rates[Math.floor(rates.length / 2)]
    : null;
  const nextAt = secondsPerStep == null ? null
    : (current
      ? Number(current.wall_time) * 1000 + (nextStep - stepOf(current)) * secondsPerStep * 1000
      : Number(last.wall_time) * 1000 + (nextStep - stepOf(last)) * secondsPerStep * 1000);
  return { interval, lastStep: stepOf(last), lastAt: Number(last.wall_time) * 1000, nextStep, nextAt, secondsPerStep };
}

function trainingProgress(metadata, latestStep) {
  if (metadata.max_batches != null) {
    const maxBatches = Number(metadata.max_batches);
    const currentBatches = Number.isFinite(Number(latestStep)) ? Number(latestStep) : null;
    return {
      currentBatches,
      targetBatches: maxBatches,
      percentage: currentBatches == null ? null : Math.min(100, currentBatches / maxBatches * 100)
    };
  }
  const configuration = metadata.configuration;
  if (!configuration) return null;
  const configText = read(configuration);
  const modelMatch = configText.match(/^\s*model:\s*(\S+)\s*$/m);
  const modelPath = modelMatch ? path.resolve(path.dirname(configuration), modelMatch[1]) : configuration;
  const modelText = read(modelPath);
  const maxBatches = Number(modelText.match(/^\s*max_batches:\s*(\d+)\s*$/m)?.[1]);
  if (!Number.isFinite(maxBatches)) return null;
  const currentBatches = Number.isFinite(Number(latestStep)) ? Number(latestStep) : null;
  return {
    currentBatches,
    targetBatches: maxBatches,
    percentage: currentBatches == null ? null : Math.min(100, currentBatches / maxBatches * 100)
  };
}

function learningRatePolicy(metadata) {
  if (metadata.learning_rate_policy) return metadata.learning_rate_policy;
  const configuration = metadata.configuration;
  if (!configuration) return null;
  const configText = read(configuration);
  const modelMatch = configText.match(/^\s*model:\s*(\S+)\s*$/m);
  const modelPath = modelMatch ? path.resolve(path.dirname(configuration), modelMatch[1]) : configuration;
  const modelText = read(modelPath);
  return modelText.match(/learning_rate:\s*[\s\S]{0,500}?^\s*policy:\s*([^\s#]+)\s*$/m)?.[1] || null;
}

function localDate(value) {
  if (!value) return value;
  const match = value.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  const date = match
    ? new Date(Date.UTC(+match[1], +match[2] - 1, +match[3], +match[4], +match[5], +match[6]))
    : new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

function metadataEpochSeconds(value) {
  const match = String(value || '').match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  if (!match) return null;
  return Date.UTC(+match[1], +match[2] - 1, +match[3], +match[4], +match[5], +match[6]) / 1000;
}

function latestEventFile(dir) {
  try {
    return fs.readdirSync(dir, { withFileTypes: true })
      .filter(entry => entry.isFile() && entry.name === 'events.tfevents')
      .map(entry => {
        const file = path.join(dir, entry.name);
        return { file, mtime: fs.statSync(file).mtimeMs };
      })
      .sort((a, b) => b.mtime - a.mtime)[0] || null;
  } catch {
    return null;
  }
}

class EventFileReader {
  constructor(event, startTime = null) {
    this.event = event;
    this.startTime = startTime;
    const python = process.env.PIXIENN_PYTHON || 'python3';
    const args = [path.join(monitorDir, 'event_file_reader.py'), event];
    if (startTime != null) args.push(String(startTime));
    this.child = spawn(python, args, {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    this.child.stdout.setEncoding('utf8');
    this.child.stderr.setEncoding('utf8');
    this.buffer = '';
    this.bufferParts = [];
    this.waiters = [];
    this.active = null;
    this.failure = null;
    this.last = null;
    this.lastMtime = null;
    this.stderr = '';

    this.child.stdout.on('data', data => this.receive(data));
    this.child.stderr.on('data', data => {
      this.stderr = `${this.stderr}${data}`.slice(-2000);
    });
    this.child.on('error', error => this.fail(error));
    this.child.on('close', (code, signal) => {
      if (!this.failure && !this.closed) {
        const detail = this.stderr.trim();
        this.fail(new Error(`Event-file reader exited (${code ?? 'null'}${signal ? `, ${signal}` : ''})${detail ? `: ${detail}` : ''}`));
      }
    });
  }

  request(mtime) {
    if (this.failure) return Promise.reject(this.failure);
    if (this.last && this.lastMtime === mtime) return Promise.resolve(this.last);
    return new Promise((resolve, reject) => {
      this.waiters.push({ resolve, reject, mtime });
      this.pump();
    });
  }

  pump() {
    if (this.failure || this.active || !this.waiters.length) return;
    const waiters = this.waiters;
    this.waiters = [];
    // Full-run scalar reconstruction is intentionally bounded, but a long
    // event file can still take several seconds on the first read. Do not
    // turn a slow refresh into an empty monitor snapshot.
    const timeout = setTimeout(() => this.fail(new Error('Event-file reader timed out')), eventReaderTimeoutMs);
    this.active = { waiters, timeout };
    try {
      this.child.stdin.write('\n');
    } catch (error) {
      this.fail(error);
    }
  }

  receive(data) {
    this.bufferParts.push(data);
    if (!data.includes('\n')) return;
    this.buffer += this.bufferParts.join('');
    this.bufferParts = [];
    let newline;
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (!this.active) continue;
      let result;
      try {
        result = JSON.parse(line);
      } catch (error) {
        this.fail(new Error(`Invalid event-file reader response: ${error.message}`));
        return;
      }
      if (result.error) {
        this.fail(new Error(result.error));
        return;
      }
      const active = this.active;
      clearTimeout(active.timeout);
      this.active = null;
      this.last = result;
      this.lastMtime = active.mtime;
      // Requests that arrived while the reader was working can share this
      // response; sending another full reload for each poll is unnecessary.
      const waiters = active.waiters.concat(this.waiters);
      this.waiters = [];
      waiters.forEach(waiter => waiter.resolve(result));
      return;
    }
  }

  fail(error) {
    if (this.failure) return;
    this.failure = error;
    if (this.active) {
      clearTimeout(this.active.timeout);
      this.active.waiters.forEach(waiter => waiter.reject(error));
      this.active = null;
    }
    this.waiters.splice(0).forEach(waiter => waiter.reject(error));
    this.child.kill();
  }

  close() {
    this.closed = true;
    this.child.kill();
  }
}

function readerFor(dir, event, startTime = null) {
  const previous = eventReaders.get(dir);
  if (previous && (previous.event !== event.file || previous.startTime !== startTime)) {
    previous.reader.close();
    eventReaders.delete(dir);
  }
  let current = eventReaders.get(dir);
  if (!current || current.reader.failure) {
    current?.reader.close();
    current = { event: event.file, startTime, reader: new EventFileReader(event.file, startTime) };
    eventReaders.set(dir, current);
  }
  return { reader: current.reader, mtime: event.mtime };
}

async function eventFileSnapshot(dir, startTime = null) {
  const event = latestEventFile(dir);
  if (!event) return { tags: [], series: {}, latest: {}, windows: {}, tails: {}, prCurves: {} };
  const { reader, mtime } = readerFor(dir, event, startTime);
  try {
    const result = await reader.request(mtime);
    const series = result.series || {};
    const latest = Object.fromEntries(Object.entries(series).map(([tag, values]) => [tag, values.at(-1) || null]));
    return { tags: Object.keys(series), series, latest, windows: result.windows || {}, tails: result.tails || {}, prCurves: result.prCurves || {}, activity: result.activity || null, eventUpdatedAt: event.mtime };
  } catch (error) {
    const result = reader.last?.series || {};
    const latest = Object.fromEntries(Object.entries(result).map(([tag, values]) => [tag, values.at(-1) || null]));
    return { tags: Object.keys(result), series: result, latest, windows: reader.last?.windows || {}, tails: reader.last?.tails || {}, prCurves: reader.last?.prCurves || {}, activity: reader.last?.activity || null, eventUpdatedAt: event.mtime, error: error.message };
  }
}

function metricPoints(series) {
  const losses = series['avg-loss'] || [];
  const rates = series['learning-rate'] || [];
  let rateIndex = -1;
  let currentRate = null;

  return losses.map(loss => {
    while (rateIndex + 1 < rates.length && rates[rateIndex + 1].step <= loss.step) {
      rateIndex++;
      currentRate = rates[rateIndex].value;
    }
    return { step: loss.step, raw_step: loss.raw_step, loss: loss.value, avg: loss.value, lr: currentRate };
  });
}

function latestStep(series) {
  return Math.max(...Object.values(series).flatMap(values => values.map(point => point.step).filter(Number.isFinite)), -Infinity);
}

function stepWindow(values, cutoff, carryPrevious = false) {
  const selected = values.filter(point => point.step >= cutoff);
  if (!carryPrevious || !selected.length) return selected;
  let previous = null;
  for (const point of values) {
    if (point.step >= cutoff) break;
    previous = point;
  }
  return previous ? [previous, ...selected] : selected;
}

function checkpoints(dir) {
  const backup = path.join(dir, 'backup');
  let stat;
  try { stat = fs.statSync(backup); } catch { return []; }
  let latestStat;
  try { latestStat = fs.statSync(path.join(backup, `${path.basename(dir)}_latest.weights`)); } catch { latestStat = null; }
  const cached = checkpointCaches.get(backup);
  if (cached && cached.mtimeMs === stat.mtimeMs && cached.latestMtimeMs === (latestStat?.mtimeMs ?? null)) return cached.files;

  let files = [];
  try {
    files = fs.readdirSync(backup)
      .filter(file => file.endsWith('.weights'))
      .map(file => {
        const full = path.join(backup, file);
        const match = file.match(/_(\d+)\.weights$/);
        return { name: file, step: match ? Number(match[1]) : null, mtime: fs.statSync(full).mtimeMs };
      })
      .sort((a, b) => b.mtime - a.mtime)
      .slice(0, 4);
  } catch {
    files = [];
  }
  checkpointCaches.set(backup, { mtimeMs: stat.mtimeMs, latestMtimeMs: latestStat?.mtimeMs ?? null, files });
  return files;
}

async function snapshot(runName, selectedLossWindow = null) {
  const { name, dir } = safeRun(runName);
  const metadata = Object.fromEntries(read(path.join(dir, 'run-metadata.txt')).split('\n').filter(Boolean).map(line => {
    const i = line.indexOf('='); return [line.slice(0, i), line.slice(i + 1)];
  }));
  // Older runs predate run-metadata.txt and keep their configuration locally.
  if (!metadata.configuration) {
    const legacyConfiguration = path.join(dir, 'config.yml');
    if (fs.existsSync(legacyConfiguration)) metadata.configuration = legacyConfiguration;
  }
  const eventStart = metadata.mode === 'fresh' ? metadataEpochSeconds(metadata.started_utc) : null;
  if (metadata.started_utc) metadata.started_utc = localDate(metadata.started_utc);
  const eventFile = await eventFileSnapshot(dir, eventStart);
  const earliestTrainingEvent = [...(eventFile.series['avg-loss'] || [])]
    .filter(point => Number.isFinite(Number(point.wall_time)))
    .sort((a, b) => Number(a.wall_time) - Number(b.wall_time))[0] || null;
  const metricSeries = selectedLossWindow
    ? (() => {
      const lossSource = eventFile.windows['avg-loss'] || eventFile.series['avg-loss'] || [];
      const rateSource = eventFile.windows['learning-rate'] || eventFile.series['learning-rate'] || [];
      const endStep = latestStep({ 'avg-loss': lossSource, 'learning-rate': rateSource });
      if (!Number.isFinite(endStep)) return eventFile.series;
      const cutoff = endStep - selectedLossWindow;
      return {
        ...eventFile.series,
        'avg-loss': stepWindow(lossSource, cutoff),
        'learning-rate': stepWindow(rateSource, cutoff, true)
      };
    })()
    : eventFile.series;
  const points = metricPoints(metricSeries);
  const latest = points.at(-1) || null;
  const recent = points.slice(-60);
  const prior = points.slice(-120, -60);
  const mean = values => values.length ? values.reduce((sum, p) => sum + p.avg, 0) / values.length : null;
  const trendAverage = mean(recent);
  const priorAverage = mean(prior);
  const change = trendAverage != null && priorAverage != null ? trendAverage - priorAverage : null;
  const direction = change == null ? 'waiting' : change < -0.5 ? 'improving' : change > 0.5 ? 'worsening' : 'flat';
  // Event-file freshness is the source of truth for run liveness. This avoids
  // coupling the monitor to the unbounded console log.
  const active = Boolean(eventFile.eventUpdatedAt && Date.now() - eventFile.eventUpdatedAt < 120000);
  const { windows, ...publicEventFile } = eventFile;
  const progressStep = latest?.raw_step ?? latest?.step;
  const policy = learningRatePolicy(metadata);
  const schedule = validationSchedule(metadata, eventFile);
  const runCheckpoints = checkpoints(dir);
  const threshold = validationThreshold(metadata);
  const result = { name, metadata, earliestTrainingEvent: earliestTrainingEvent ? { at: Number(earliestTrainingEvent.wall_time) * 1000, step: Number(earliestTrainingEvent.raw_step ?? earliestTrainingEvent.step) } : null, latest, progress: trainingProgress(metadata, progressStep), learningRatePolicy: policy, points, trend: { average: trendAverage, priorAverage, change, direction }, eventFile: publicEventFile, validationSchedule: schedule, checkpoints: runCheckpoints, validationThreshold: threshold, active, updatedAt: Date.now() };
  return result;
}

function sharedSnapshot(runName, selectedLossWindow = null) {
  const key = `${runName}:${selectedLossWindow || 'all'}`;
  const existing = snapshotRequests.get(key);
  if (existing) return existing;
  const request = snapshot(runName, selectedLossWindow).finally(() => {
    if (snapshotRequests.get(key) === request) snapshotRequests.delete(key);
  });
  snapshotRequests.set(key, request);
  return request;
}

const vite = await createViteServer({ root: path.dirname(fileURLToPath(import.meta.url)), server: { middlewareMode: true, hmr: false, watch: { ignored: [monitorDir, '**/*'] } }, appType: 'spa' });
createServer(async (req, res) => {
  try {
    if (req.url === '/api/runs') {
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'no-store');
      res.end(JSON.stringify(availableRuns()));
      return;
    }
    if (req.url.startsWith('/api/run')) {
      const query = new URL(req.url, 'http://localhost').searchParams;
      const runName = query.get('name') || 'yolov3';
      const selectedLossWindow = lossWindow(query.get('window'));
      const data = await sharedSnapshot(runName, selectedLossWindow);
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'no-store');
      res.end(JSON.stringify(data));
      return;
    }
    vite.middlewares(req, res, () => { res.statusCode = 404; res.end('Not found'); });
  } catch (error) {
    if (!res.headersSent) {
      res.statusCode = 500;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: error.message }));
    } else {
      res.destroy(error);
    }
  }
}).listen(port, () => console.log(`PixieNN monitor: http://localhost:${port}`));

process.once('exit', () => {
  for (const { reader } of eventReaders.values()) reader.close();
});
