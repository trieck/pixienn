#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/../.." && pwd -P)
temp_root=$(mktemp -d -t pixienn-training-scripts.XXXXXX)

cleanup()
{
    [[ -n "${started_tensorboard_pid:-}" ]] && kill -TERM "$started_tensorboard_pid" 2>/dev/null || true
    [[ -n "${fake_tensorboard_pid:-}" ]] && kill -TERM "$fake_tensorboard_pid" 2>/dev/null || true
    rm -rf -- "$temp_root"
}
trap cleanup EXIT

fail()
{
    echo "FAIL: $*" >&2
    exit 1
}

fake_bin="$temp_root/bin"
runs="$temp_root/runs"
mkdir -p -- "$fake_bin"

cat > "$fake_bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == -L ]]; then
    echo "GPU 0: Fake CUDA GPU"
else
    echo "Fake CUDA GPU"
fi
EOF

cat > "$fake_bin/pixienn-train" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
weights=${!#}
run_dir=$(dirname -- "$weights")
model=$(basename -- "$weights" .weights)
printf '%s\n' "$*" > "$run_dir/fake-invocation.txt"
mkdir -p -- "$run_dir/backup"
touch -- "$weights" "$weights.optimizer" "$weights.training"
touch -- "$run_dir/backup/${model}_latest.weights"
touch -- "$run_dir/backup/${model}_latest.weights.optimizer"
touch -- "$run_dir/backup/${model}_latest.weights.training"
echo "fake training completed"
EOF

cat > "$fake_bin/lsof" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${FAKE_TENSORBOARD_PID:-}" && -r "/proc/$FAKE_TENSORBOARD_PID/stat" &&
      "$(awk '{print $3}' "/proc/$FAKE_TENSORBOARD_PID/stat")" != Z ]]; then
    printf '%s\n' "$FAKE_TENSORBOARD_PID"
fi
if [[ -n "${FAKE_STARTED_TENSORBOARD_PID_FILE:-}" && -s "$FAKE_STARTED_TENSORBOARD_PID_FILE" ]]; then
    cat "$FAKE_STARTED_TENSORBOARD_PID_FILE"
fi
EOF

cat > "$fake_bin/tensorboard" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$$" > "$FAKE_STARTED_TENSORBOARD_PID_FILE"
trap 'exit 0' TERM
while true; do
    sleep 1
done
EOF
chmod +x "$fake_bin/nvidia-smi" "$fake_bin/pixienn-train" "$fake_bin/lsof" "$fake_bin/tensorboard"

export PATH="$fake_bin:$PATH"
export PIXIENN_TRAIN_BIN="$fake_bin/pixienn-train"
export PIXIENN_RUNS_DIR="$runs"
export FAKE_STARTED_TENSORBOARD_PID_FILE="$temp_root/started-tensorboard.pid"

"$repo_root/shell/train-model.sh" yolo-nano --fresh --dry-run >/dev/null
[[ ! -e "$runs/yolo-nano" ]] || fail "dry-run created a run directory"

mkdir -p -- "$runs/yolo-nano"
echo old > "$runs/yolo-nano/old-run-marker"
touch -- "$runs/yolo-nano/events.out.tfevents.active"
mkdir -p -- "$runs/archive/yolo-nano-older"
touch -- "$runs/archive/yolo-nano-older/events.out.tfevents.archived"

FAKE_STARTED_TENSORBOARD_PID_FILE="$temp_root/old-tensorboard.pid" "$fake_bin/tensorboard" &
fake_tensorboard_pid=$!
export FAKE_TENSORBOARD_PID=$fake_tensorboard_pid
"$repo_root/shell/train-model.sh" yolo-nano --fresh >"$temp_root/fresh.log"
wait "$fake_tensorboard_pid" 2>/dev/null || true

if kill -0 "$fake_tensorboard_pid" 2>/dev/null; then
    fail "fresh run did not stop TensorBoard on the configured port"
fi
if find "$runs" -type f -name 'events.out.tfevents.*' -print -quit | grep -q .; then
    fail "fresh run retained TensorBoard event data"
fi
unset FAKE_TENSORBOARD_PID

started_tensorboard_pid=$(cat "$FAKE_STARTED_TENSORBOARD_PID_FILE")
if ! kill -0 "$started_tensorboard_pid" 2>/dev/null; then
    fail "training wrapper did not start TensorBoard"
fi
grep -q 'TensorBoard: http://localhost:6006/' "$temp_root/fresh.log" || \
    fail "training wrapper did not print the TensorBoard URL"

[[ -f "$runs/yolo-nano/run-metadata.txt" ]] || fail "fresh run did not write metadata"
[[ -f "$runs/yolo-nano/training.log" ]] || fail "fresh run did not write a log"
[[ -f "$runs/yolo-nano/yolo-nano.weights" ]] || fail "fresh run did not produce primary weights"
grep -q -- '--clear-weights' "$runs/yolo-nano/fake-invocation.txt" || fail "fresh run omitted --clear-weights"

archive_marker=$(find "$runs/archive" -name old-run-marker -print -quit)
[[ -n "$archive_marker" ]] || fail "fresh run did not archive the previous run"

"$repo_root/shell/train-model.sh" yolo-nano --resume >/dev/null

resume_archive=$(find "$runs/yolo-nano/archive" -name 'yolo-nano-before-resume-*.weights' -print -quit)
[[ -n "$resume_archive" ]] || fail "resume did not preserve the older primary checkpoint"
[[ -f "$runs/yolo-nano/yolo-nano.weights" ]] || fail "resume did not recreate primary weights"
[[ ! -e "$runs/.locks/yolo-nano" ]] || fail "training lock was not cleaned up"

mkdir -p -- "$runs/.locks/yolo-nano"
printf '%s\n' "$$" > "$runs/.locks/yolo-nano/pid"
if "$repo_root/shell/train-model.sh" yolo-nano --resume >"$temp_root/locked.log" 2>&1; then
    fail "active training lock did not block a second run"
fi
grep -q 'already registered' "$temp_root/locked.log" || fail "active-lock failure was not explained"
rm -f -- "$runs/.locks/yolo-nano/pid"
rmdir -- "$runs/.locks/yolo-nano"

echo "Training script integration tests passed."

kill -TERM "$started_tensorboard_pid" 2>/dev/null || true
wait "$started_tensorboard_pid" 2>/dev/null || true
