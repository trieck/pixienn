#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SERVICE_DIR="$ROOT_DIR/model-graph"
PORT="${MODEL_GRAPH_PORT:-5179}"
PID_FILE="$SERVICE_DIR/.model-graph.pid"
LOG_FILE="$SERVICE_DIR/model-graph.log"
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="$PYTHON"
elif python3 -c 'import yaml' >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif [[ -x /usr/bin/python3 ]] && /usr/bin/python3 -c 'import yaml' >/dev/null 2>&1; then
    PYTHON=/usr/bin/python3
else
    echo "model-graph requires PyYAML; install it in the active Python environment or set PYTHON=/path/to/python" >&2
    exit 1
fi

find_graph_pids() {
    while read -r pid args; do
        [[ "$args" == *"model-graph/server.py"* || "$args" == *"server.py"* ]] || continue
        [[ -r "/proc/$pid/cwd" ]] || continue
        [[ "$(readlink -f "/proc/$pid/cwd")" == "$ROOT_DIR" || "$(readlink -f "/proc/$pid/cwd")" == "$SERVICE_DIR" ]] || continue
        printf '%s\n' "$pid"
    done < <(ps -eo pid=,args=)
}

stop_graph() {
    local pids pid descendants all_pids deadline
    pids="$(find_graph_pids || true)"
    if [[ -f "$PID_FILE" ]]; then
        pid="$(<"$PID_FILE")"
        if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cwd" ]]; then
            pids="$(printf '%s\n%s\n' "$pids" "$pid")"
        fi
    fi
    pids="$(printf '%s\n' "$pids" | awk 'NF && !seen[$1]++ { print $1 }')"
    [[ -z "$pids" ]] && { rm -f -- "$PID_FILE"; return; }
    descendants=""
    for pid in $pids; do
        descendants+="$(pgrep -P "$pid" || true)\n"
    done
    all_pids="$(printf '%s\n%s\n' "$pids" "$descendants" | awk 'NF && !seen[$1]++ { print $1 }')"
    kill $all_pids 2>/dev/null || true
    deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )) && kill -0 $all_pids 2>/dev/null; do sleep 0.1; done
    kill -KILL $all_pids 2>/dev/null || true
    rm -f -- "$PID_FILE"
}

wait_for_port_free() {
    local deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )); do
        if ! (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then return 0; fi
        sleep 0.1
    done
    echo "model-graph port $PORT is still occupied" >&2
    return 1
}

wait_for_http() {
    local url="$1" deadline=$((SECONDS + 15))
    while (( SECONDS < deadline )); do
        if curl --silent --show-error --fail --max-time 1 "$url" >/dev/null 2>&1; then return 0; fi
        sleep 0.2
    done
    return 1
}

cd -- "$ROOT_DIR"
echo "Stopping model-graph..."
stop_graph
wait_for_port_free

echo "Building model-graph frontend..."
(cd -- "$SERVICE_DIR" && timeout 30s npm run build)

echo "Starting model-graph server..."
nohup setsid "$PYTHON" "$SERVICE_DIR/server.py" \
    --model "$ROOT_DIR/resources/models/centernet-prosopo.yml" \
    --host 127.0.0.1 --port "$PORT" >"$LOG_FILE" 2>&1 < /dev/null &
graph_pid=$!
printf '%s\n' "$graph_pid" >"$PID_FILE"

if ! wait_for_http "http://127.0.0.1:$PORT/"; then
    echo "model graph failed to start; see $LOG_FILE" >&2
    exit 1
fi

echo "PixieNN model graph restarted on http://localhost:$PORT (pid $graph_pid)"
