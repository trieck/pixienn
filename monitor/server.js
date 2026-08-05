import fs from 'node:fs';
import path from 'node:path';
import { webcrypto } from 'node:crypto';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';

// Vite expects the Web Crypto API. Some Node versions expose a partial
// `globalThis.crypto` object, so nullish assignment alone is not sufficient.
if (typeof globalThis.crypto?.getRandomValues !== 'function') {
  globalThis.crypto = webcrypto;
}
const { createServer: createViteServer } = await import('vite');

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runs = path.join(root, 'runs');
const port = Number(process.env.PORT || 4173);

function safeRun(name = 'yolov3') {
  const clean = name.replace(/[^a-zA-Z0-9_-]/g, '');
  const dir = path.join(runs, clean);
  if (!dir.startsWith(`${runs}${path.sep}`)) throw new Error('Invalid run');
  return { name: clean, dir };
}

function availableRuns() {
  if (!fs.existsSync(runs)) return [];
  return fs.readdirSync(runs, { withFileTypes: true })
    .filter(entry => entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== '.locks')
    .map(entry => entry.name)
    .sort();
}

function read(name) {
  try { return fs.readFileSync(name, 'utf8'); } catch { return ''; }
}

function localDate(value) {
  if (!value) return value;
  const match = value.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  const date = match
    ? new Date(Date.UTC(+match[1], +match[2] - 1, +match[3], +match[4], +match[5], +match[6]))
    : new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

function snapshot(runName) {
  const { name, dir } = safeRun(runName);
  const log = read(path.join(dir, 'training.log'));
  const metadata = Object.fromEntries(read(path.join(dir, 'run-metadata.txt')).split('\n').filter(Boolean).map(line => {
    const i = line.indexOf('='); return [line.slice(0, i), line.slice(i + 1)];
  }));
  if (metadata.started_utc) metadata.started_utc = localDate(metadata.started_utc);
  const lines = log.trim().split('\n').filter(Boolean);
  const re = /Epoch:\s*(\d+), Seen:\s*(\d+), Loss:\s*([\d.e+-]+), Avg\. Loss:\s*([\d.e+-]+), LR:\s*([\d.e+-]+)/;
  // Keep the complete scalar history. The client compresses it into a fixed
  // number of buckets so the chart shows the trajectory across the whole run.
  const points = lines.flatMap(line => { const m = line.match(re); return m ? [{ step: +m[2], loss: +m[3], avg: +m[4], lr: +m[5] }] : []; });
  const latest = points.at(-1) || null;
  const recent = points.slice(-60);
  const prior = points.slice(-120, -60);
  const mean = values => values.length ? values.reduce((sum, p) => sum + p.avg, 0) / values.length : null;
  const trendAverage = mean(recent);
  const priorAverage = mean(prior);
  const change = trendAverage != null && priorAverage != null ? trendAverage - priorAverage : null;
  const direction = change == null ? 'waiting' : change < -0.5 ? 'improving' : change > 0.5 ? 'worsening' : 'flat';
  const files = fs.existsSync(path.join(dir, 'backup')) ? fs.readdirSync(path.join(dir, 'backup')).filter(f => f.endsWith('.weights')).map(f => ({ name: f, mtime: fs.statSync(path.join(dir, 'backup', f)).mtimeMs })).sort((a,b) => b.mtime-a.mtime).slice(0, 4) : [];
  // A metric in an old log is not evidence that training is still running.
  // Treat the run as live only while its log has been updated recently.
  const logFresh = fs.existsSync(path.join(dir, 'training.log'))
    && Date.now() - fs.statSync(path.join(dir, 'training.log')).mtimeMs < 120000;
  const active = Boolean(latest && logFresh && !/trained in|early stopping/i.test(lines.at(-1) || ''));
  return { name, metadata, latest, points, trend: { average: trendAverage, priorAverage, change, direction }, log: lines.slice(-120), checkpoints: files, active, updatedAt: Date.now() };
}

const vite = await createViteServer({ root: path.dirname(fileURLToPath(import.meta.url)), server: { middlewareMode: true }, appType: 'spa' });
createServer((req, res) => {
  if (req.url === '/api/runs') {
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(availableRuns()));
    return;
  }
  if (req.url.startsWith('/api/run')) {
    try { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(snapshot(new URL(req.url, 'http://localhost').searchParams.get('name') || 'yolov3'))); }
    catch (e) { res.statusCode = 400; res.end(JSON.stringify({ error: e.message })); }
    return;
  }
  vite.middlewares(req, res, () => { res.statusCode = 404; res.end('Not found'); });
}).listen(port, () => console.log(`PixieNN monitor: http://localhost:${port}`));
