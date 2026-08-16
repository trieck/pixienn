#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER="$SCRIPT_DIR/server.js"
PORT="${PORT:-4173}"
PID_FILE="$SCRIPT_DIR/.monitor.pid"
LOG_FILE="$SCRIPT_DIR/monitor.log"

select_event_python() {
    local candidate
    for candidate in \
        "$SCRIPT_DIR/../.venv-pixienn-cuda/bin/python3" \
        "$HOME/miniforge3/bin/python3" \
        "$(command -v python3)"; do
        [[ -x "$candidate" ]] || continue
        if "$candidate" -c 'import tensorboard' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "no Python interpreter with TensorBoard is available" >&2
    return 1
}

find_monitor_pids() {
    while read -r pid args; do
        [[ "$args" == *"server.js"* ]] || continue
        [[ -r "/proc/$pid/cwd" ]] || continue
        [[ "$(readlink -f "/proc/$pid/cwd")" == "$SCRIPT_DIR" ]] || continue
        printf '%s\n' "$pid"
    done < <(ps -eo pid=,args=)
}

stop_monitor() {
    local pids pid deadline child descendants all_pids
    pids="$(find_monitor_pids || true)"
    if [[ -f "$PID_FILE" ]]; then
        pid="$(<"$PID_FILE")"
        if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cwd" && "$(readlink -f "/proc/$pid/cwd")" == "$SCRIPT_DIR" ]]; then
            pids="$(printf '%s\n%s\n' "$pids" "$pid")"
        fi
    fi

    pids="$(printf '%s\n' "$pids" | awk 'NF && !seen[$1]++ { print $1 }')"
    if [[ -n "$pids" ]]; then
        # Killing only node leaves Vite, esbuild, and event readers behind.
        # Collect descendants first so the whole monitor tree is stopped.
        descendants=""
        for pid in $pids; do
            while read -r child; do
                [[ -n "$child" ]] || continue
                descendants+="$child\n"
            done < <(ps -eo pid=,ppid= | awk -v parent="$pid" '$2 == parent { print $1 }')
        done
        all_pids="$(printf '%s\n%s\n' "$pids" "$descendants" | awk 'NF && !seen[$1]++ { print $1 }')"
        kill $all_pids 2>/dev/null || true
        deadline=$((SECONDS + 5))
        while (( SECONDS < deadline )); do
            if ! kill -0 $all_pids 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        kill -KILL $all_pids 2>/dev/null || true
    fi
    rm -f -- "$PID_FILE"
}

wait_for_port_free() {
    local deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )); do
        if ! (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    echo "monitor port $PORT is still occupied" >&2
    return 1
}

stop_monitor
wait_for_port_free
export PIXIENN_PYTHON="$(select_event_python)"

cd -- "$SCRIPT_DIR"
nohup setsid node "$SERVER" >"$LOG_FILE" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$PID_FILE"

deadline=$((SECONDS + 10))
while (( SECONDS < deadline )); do
    if curl --silent --show-error --fail --max-time 1 \
        "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
        echo "PixieNN monitor restarted on http://localhost:$PORT (pid $monitor_pid)"
        exit 0
    fi
    if ! kill -0 "$monitor_pid" 2>/dev/null; then
        echo "monitor failed to start; see $LOG_FILE" >&2
        exit 1
    fi
    sleep 0.2
done

echo "monitor did not become ready; see $LOG_FILE" >&2
exit 1
