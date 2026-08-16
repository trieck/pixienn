# PixieNN Training Monitor

The monitor is a small local web application for inspecting PixieNN training runs. It reads the files produced under the repository's `runs/` directory and presents training loss, learning-rate history, validation metrics, PR curves, checkpoints, and inferred run activity in a browser.

It is intentionally a local observability tool. It does not control, start, stop, or modify training.

## Architecture

The monitor has three layers:

```text
Browser
  React UI (src/main.jsx)
        │ fetches JSON every 2.5 seconds
        ▼
Node server (server.js)
  HTTP API + Vite middleware
        │ starts one persistent reader process per event file
        ▼
Python event reader (event_file_reader.py)
  TensorBoard event-file parsing and bounded series reduction
        │
        ▼
runs/<run>/events.tfevents
```

### `server.js`: Node.js host and API

`server.js` is an ES module executed by Node. It:

- discovers available run directories;
- reads `run-metadata.txt`, model configuration, and checkpoint metadata;
- starts Vite in middleware mode to serve the React application;
- exposes the JSON API;
- maintains a cached reader process for each active event file;
- derives display values such as training progress, LR policy, validation schedule, and run freshness.

The server listens on port `4173` by default. Set `PORT` to use another port.

Endpoints:

- `GET /api/runs` — returns available run names.
- `GET /api/run?name=yolov3` — returns the current snapshot for a run.
- `GET /api/run?name=yolov3&window=5000` — limits the loss/LR chart window. Supported windows are `500`, `2000`, and `10000` optimizer steps.

The browser polls the snapshot endpoint; there is no WebSocket or training-process connection.

### `src/main.jsx`: React presentation layer

`main.jsx` is the React entry point. It fetches the API snapshot, holds display state such as the selected run, theme, loss window, and chart scale, and renders the cards and charts.

The UI is client-side only. It does not parse TensorBoard files and does not need direct filesystem access. `src/styles.css` contains the visual themes, including the Da Vinci 95 theme.

### `event_file_reader.py`: Python TensorBoard adapter

The training executable writes TensorBoard-compatible event data. Python is used here because TensorBoard provides the mature, maintained event-file and tensor-summary readers:

- `EventFileLoader` reads TensorFlow event records;
- `tensor_util` extracts scalar values stored as tensor summaries;
- the reader handles scalar metrics, PR-curve data, timestamps, and normalized steps.

The Node server launches the reader with the event-file path. The reader stays alive and receives a newline on stdin for each refresh request, returning one JSON response per request on stdout. Its stderr is retained by Node for diagnostics.

Long runs can contain millions of events, so the reader does not send every point to the browser. It keeps bounded tails for interactive windows, uses bounded sampling for full-run charts, preserves extrema when reducing a series, and stores an incremental cache in the system temporary directory.

## Run directory contract

For a run named `yolov3`, the monitor looks under:

```text
runs/yolov3/
├── events.tfevents       # scalar, PR, and timestamp data
├── run-metadata.txt      # launcher metadata and configuration path
├── training.log          # human-readable trainer output
└── backup/               # saved weight checkpoints
```

The event file is the source of truth for the live/stale indicator. A run is considered active when its event file was updated within the server's freshness window.

## Development

From this directory:

```bash
npm install
npm run dev
```

Then open <http://localhost:4173/>.

The `dev` script runs `node server.js`; Vite serves and hot-reloads the React source through Node's middleware. There is currently no separate production build script—the monitor is designed for local use.

The Python reader requires the Python environment used by the project to provide TensorBoard. Override the interpreter with:

```bash
PIXIENN_PYTHON=/path/to/python npm run dev
```

For a different runs directory:

```bash
PIXIENN_RUNS_DIR=/path/to/runs npm run dev
```

Other useful settings include `PIXIENN_EVENT_READER_TIMEOUT_MS` for the event-reader request timeout and `PORT` for the HTTP port.

## Design boundaries

- The monitor is read-only with respect to training artifacts.
- It does not invoke `pixienn-train`.
- It does not use TensorBoard's web server; it reads the event files directly.
- Node owns HTTP, process management, configuration interpretation, and snapshot assembly.
- Python is limited to TensorBoard event-file decoding and bounded data reduction.
- React owns presentation and user interaction.
