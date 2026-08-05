#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)

models=(
    centernet-smoke-voc
    centernet-tiny-voc
    resnet18
    tiny-yolo-voc
    yolo-nano
    yolov1-tiny
    yolov3-tiny-voc
    yolov3-tiny
    yolov3
    yolov7
)

usage()
{
    cat <<'EOF'
Usage: train-model.sh MODEL (--fresh | --resume) [options]
       train-model.sh --list

Modes:
  --fresh                  Stop TensorBoard, purge this model's event data, archive
                           the remaining run files, then start with clear weights.
  --resume                 Resume the latest checkpoint in the model's run directory.

Options:
  --verify-data            Check every image and label before training.
  --allow-reset-optimizer  Permit resume without an Adam optimizer sidecar.
  --view-image             Pass PixieNN's training-image viewer option.
  --dry-run                Perform preflight and print the command without changing a run.
  -h, --help               Show this help.

Environment:
  PIXIENN_TRAIN_BIN         Override the CUDA pixienn-train executable.
  PIXIENN_RUNS_DIR          Override the run root (default: REPOSITORY/runs).
  PIXIENN_TENSORBOARD_PORT  TensorBoard port checked by --fresh (default: 6006).
EOF
}

list_models()
{
    cat <<'EOF'
MODEL               DATASET/PURPOSE         CONFIG
centernet-smoke-voc VOC pipeline smoke test resources/cfg/centernet-smoke-voc-cfg.yml
centernet-tiny-voc  VOC anchor-free detector resources/cfg/centernet-tiny-voc-cfg.yml
resnet18            ImageNet paths required resources/cfg/resnet18-cfg.yml
tiny-yolo-voc       VOC smoke preset        resources/cfg/tiny-yolo-voc-cfg.yml
yolo-nano           VOC full manifests      resources/cfg/yolo-nano-cfg.yml
yolov1-tiny         VOC full manifests      resources/cfg/yolov1-tiny-cfg.yml
yolov3-tiny-voc     VOC full manifests      resources/cfg/yolov3-tiny-voc-cfg.yml
yolov3-tiny         COCO smoke preset       resources/cfg/yolov3-tiny-cfg.yml
yolov3              COCO smoke preset       resources/cfg/yolov3-cfg.yml
yolov7              COCO 82k/1k manifests  resources/cfg/yolov7-cfg.yml
EOF
}

is_model()
{
    local candidate=$1
    local known
    for known in "${models[@]}"; do
        [[ "$candidate" == "$known" ]] && return 0
    done
    return 1
}

config_for_model()
{
    printf '%s/resources/cfg/%s-cfg.yml\n' "$repo_root" "$1"
}

find_train_binary()
{
    local candidate
    if [[ -n "${PIXIENN_TRAIN_BIN:-}" ]]; then
        candidate=$(realpath -m -- "$PIXIENN_TRAIN_BIN")
        [[ -x "$candidate" ]] || {
            echo "PIXIENN_TRAIN_BIN is not executable: $candidate" >&2
            return 1
        }
        printf '%s\n' "$candidate"
        return
    fi

    for candidate in \
        "$repo_root/build/bin/pixienn-train" \
        "$repo_root/cmake-build-release-cuda/bin/pixienn-train" \
        "$repo_root/cmake-build-debug-cuda/bin/pixienn-train"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    echo "No CUDA pixienn-train binary found. Build a *-cuda configuration first." >&2
    return 1
}

verify_cuda_binary()
{
    local binary=$1
    if file -- "$binary" | grep -q 'ELF'; then
        if ! ldd -- "$binary" 2>/dev/null | grep -Eq 'libcudart|libcuda|libcudnn'; then
            echo "Training executable does not appear to be CUDA-enabled: $binary" >&2
            echo "Refusing to risk CPU training." >&2
            return 1
        fi
    fi
}

model_uses_adam()
{
    local model_file=$1
    awk '
        /^[[:space:]]+adam:[[:space:]]*$/ { in_adam=1; next }
        in_adam && /^[[:space:]]+enabled:/ {
            value=tolower($2)
            exit(value == "true" ? 0 : 1)
        }
        in_adam && /^[[:space:]]+[a-zA-Z0-9_-]+:/ { exit 1 }
        END { if (!in_adam) exit 1 }
    ' "$model_file"
}

yaml_model_path()
{
    local config=$1
    local value
    value=$(awk '$1 == "model:" { print $2; exit }' "$config")
    if [[ "$value" = /* ]]; then
        realpath -m -- "$value"
    else
        realpath -m -- "$(dirname -- "$config")/$value"
    fi
}

stop_tensorboard_on_port()
{
    local port=$1
    local pid command
    local -a tensorboard_pids=()

    command -v lsof >/dev/null 2>&1 || {
        echo "Warning: lsof is unavailable; cannot inspect TensorBoard port $port." >&2
        return 0
    }

    while IFS= read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
        command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
        if [[ "${command,,}" == *tensorboard* ]]; then
            tensorboard_pids+=("$pid")
        else
            echo "Port $port is occupied by a non-TensorBoard process (PID $pid); leaving it running." >&2
        fi
    done < <(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u)

    ((${#tensorboard_pids[@]})) || return 0

    echo "Stopping TensorBoard on port $port (PID(s): ${tensorboard_pids[*]})"
    kill -TERM "${tensorboard_pids[@]}" 2>/dev/null || true

    for _ in {1..50}; do
        local alive=false
        for pid in "${tensorboard_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null && [[ "$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)" != Z ]]; then
                alive=true
            fi
        done
        if ! $alive; then
            return 0
        fi
        sleep 0.1
    done

    echo "TensorBoard did not stop cleanly; forcing termination." >&2
    kill -KILL "${tensorboard_pids[@]}" 2>/dev/null || true
}

purge_tensorboard_data()
{
    local active_run=$1
    local archive_root=$2
    local model_name=$3
    local directory event_file
    local removed=0
    local -a directories=()

    [[ -d "$active_run" ]] && directories+=("$active_run")
    if [[ -d "$archive_root" ]]; then
        while IFS= read -r -d '' directory; do
            directories+=("$directory")
        done < <(find "$archive_root" -mindepth 1 -maxdepth 1 -type d -name "${model_name}-*" -print0)
    fi

    for directory in "${directories[@]}"; do
        while IFS= read -r -d '' event_file; do
            rm -f -- "$event_file"
            ((removed += 1))
        done < <(find "$directory" -type f -name 'events.out.tfevents.*' -print0)
    done

    echo "Removed $removed TensorBoard event file(s) for $model_name."
}

start_tensorboard()
{
    local log_dir=$1
    local port=$2
    local pid command listener candidate shebang shebang_body interpreter shebang_arg python3_bin
    local -a tensorboard_command=()

    # A user-local tensorboard launcher can point at a different Python
    # installation than the active environment.  Resolve that mismatch before
    # starting the server; otherwise the wrapper reports a false startup
    # failure with "No module named 'tensorboard'".
    candidate=$(command -v tensorboard 2>/dev/null || true)
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        shebang=$(head -n 1 -- "$candidate" 2>/dev/null || true)
        if [[ "$shebang" == '#!'* ]]; then
            shebang_body=${shebang#\#!}
            read -r interpreter shebang_arg <<<"$shebang_body"
            if [[ "$interpreter" == "/usr/bin/env" ]]; then
                interpreter=$(command -v "$shebang_arg" 2>/dev/null || true)
            fi
            if [[ "$interpreter" == *python* ]]; then
                if "$interpreter" -c 'import tensorboard' >/dev/null 2>&1; then
                    tensorboard_command=("$candidate")
                fi
            else
                # Keep test doubles and non-Python launchers intact.
                tensorboard_command=("$candidate")
            fi
        else
            tensorboard_command=("$candidate")
        fi
    fi

    if ((${#tensorboard_command[@]} == 0)); then
        python3_bin=$(command -v python3 2>/dev/null || true)
        if [[ -n "$python3_bin" ]] && "$python3_bin" -c 'import tensorboard' >/dev/null 2>&1; then
            tensorboard_command=("$python3_bin" -m tensorboard.main)
        else
            echo "TensorBoard is not installed or not available to the active Python environment." >&2
            return 1
        fi
    fi

    while IFS= read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
        command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
        if [[ "${command,,}" == *tensorboard* ]]; then
            echo "TensorBoard is already running: http://localhost:$port/"
            return 0
        fi
        echo "Cannot start TensorBoard: port $port is occupied by PID $pid ($command)." >&2
        return 1
    done < <(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u)

    "${tensorboard_command[@]}" --logdir="$log_dir" --port="$port" --bind_all \
        >"$log_dir/tensorboard.log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$log_dir/tensorboard.pid"

    for _ in {1..50}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "TensorBoard failed to start. See $log_dir/tensorboard.log" >&2
            return 1
        fi
        listener=$(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        if grep -qx "$pid" <<<"$listener"; then
            echo "TensorBoard: http://localhost:$port/"
            echo "TensorBoard log: $log_dir/tensorboard.log"
            return 0
        fi
        sleep 0.1
    done

    echo "TensorBoard did not begin listening on port $port. See $log_dir/tensorboard.log" >&2
    kill -TERM "$pid" 2>/dev/null || true
    return 1
}

model=""
mode=""
verify_data=false
allow_reset_optimizer=false
view_image=false
dry_run=false

while (($#)); do
    case "$1" in
        --fresh|--resume)
            if [[ -n "$mode" ]]; then
                echo "Choose exactly one of --fresh or --resume." >&2
                exit 2
            fi
            mode=${1#--}
            ;;
        --verify-data)
            verify_data=true
            ;;
        --allow-reset-optimizer)
            allow_reset_optimizer=true
            ;;
        --view-image)
            view_image=true
            ;;
        --dry-run)
            dry_run=true
            ;;
        --list)
            list_models
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -* )
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$model" ]]; then
                echo "Only one model may be trained at a time." >&2
                exit 2
            fi
            model=$1
            ;;
    esac
    shift
done

if [[ -z "$model" || -z "$mode" ]]; then
    usage >&2
    exit 2
fi

if ! is_model "$model"; then
    echo "Unknown model: $model" >&2
    list_models >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "No usable NVIDIA GPU was found; refusing to fall back to CPU training." >&2
    exit 1
fi

train_bin=$(find_train_binary)
verify_cuda_binary "$train_bin"
config=$(config_for_model "$model")
model_file=$(yaml_model_path "$config")
runs_root=$(realpath -m -- "${PIXIENN_RUNS_DIR:-$repo_root/runs}")
tensorboard_port=${PIXIENN_TENSORBOARD_PORT:-6006}
run_dir="$runs_root/$model"
weights="$run_dir/$model.weights"
latest="$run_dir/backup/${model}_latest.weights"

[[ -f "$config" ]] || { echo "Configuration not found: $config" >&2; exit 1; }
[[ -f "$model_file" ]] || { echo "Model file not found: $model_file" >&2; exit 1; }
[[ "$run_dir" == "$runs_root/"* ]] || { echo "Unsafe run path: $run_dir" >&2; exit 1; }
[[ ! -L "$run_dir" ]] || { echo "Run directory may not be a symbolic link: $run_dir" >&2; exit 1; }
[[ "$tensorboard_port" =~ ^[0-9]+$ && tensorboard_port -ge 1 && tensorboard_port -le 65535 ]] || {
    echo "Invalid PIXIENN_TENSORBOARD_PORT: $tensorboard_port" >&2
    exit 1
}

check_args=(--quick "$config")
$verify_data && check_args=("$config")
"$script_dir/check-training-data.sh" "${check_args[@]}"

git_revision=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf 'unknown')
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "Warning: tracked source changes are present; the run metadata will record a dirty tree." >&2
    git_revision="${git_revision}-dirty"
fi

trainer_options=()
$view_image && trainer_options+=(--view-image)
[[ "$mode" == fresh ]] && trainer_options+=(--clear-weights)

echo
echo "Training plan"
printf '  model:       %s\n' "$model"
printf '  mode:        %s\n' "$mode"
printf '  executable:  %s\n' "$train_bin"
printf '  config:      %s\n' "$config"
printf '  run dir:     %s\n' "$run_dir"
printf '  weights:     %s\n' "$weights"
printf '  git commit:  %s\n' "$git_revision"
printf '  GPU:         %s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ', ' -)"

if $dry_run; then
    if [[ "$mode" == resume ]]; then
        [[ -d "$run_dir" ]] || { echo "No run directory to resume: $run_dir" >&2; exit 1; }
        dry_resume_source=""
        [[ -f "$latest" ]] && dry_resume_source=$latest
        [[ -z "$dry_resume_source" && -f "$weights" ]] && dry_resume_source=$weights
        [[ -n "$dry_resume_source" ]] || { echo "No checkpoint found to resume in $run_dir" >&2; exit 1; }
        if model_uses_adam "$model_file" && [[ ! -f "$dry_resume_source.optimizer" ]] && ! $allow_reset_optimizer; then
            echo "Adam state is missing: $dry_resume_source.optimizer" >&2
            exit 1
        fi
        printf '  resume from: %s\n' "$dry_resume_source"
    elif [[ -d "$run_dir" ]]; then
        printf '  cleanup:     stop TensorBoard on port %s, purge model events, archive remaining files\n' \
            "$tensorboard_port"
    fi
    printf '  command:     '
    printf '%q ' "$train_bin" "${trainer_options[@]}" "$config" "$weights"
    echo
    echo "Dry run complete; no run files were changed."
    exit 0
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
locks_root="$runs_root/.locks"
lock_dir="$locks_root/$model"
mkdir -p -- "$locks_root"
if ! mkdir -- "$lock_dir" 2>/dev/null; then
    lock_pid=""
    [[ -f "$lock_dir/pid" ]] && read -r lock_pid < "$lock_dir/pid"
    if [[ "$lock_pid" =~ ^[0-9]+$ ]] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "A training process is already registered for $model (PID $lock_pid)." >&2
        exit 1
    fi
    echo "Removing stale training lock for $model." >&2
    rm -f -- "$lock_dir/pid"
    rmdir -- "$lock_dir"
    mkdir -- "$lock_dir"
fi

printf '%s\n' "$$" > "$lock_dir/pid"
cleanup_lock()
{
    rm -f -- "$lock_dir/pid"
    rmdir -- "$lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT

if [[ "$mode" == fresh ]]; then
    stop_tensorboard_on_port "$tensorboard_port"
    purge_tensorboard_data "$run_dir" "$runs_root/archive" "$model"
    if [[ -d "$run_dir" ]]; then
        archive_root="$runs_root/archive"
        archive_dir="$archive_root/${model}-${timestamp}"
        [[ ! -e "$archive_dir" ]] || archive_dir="${archive_dir}-$$"
        mkdir -p -- "$archive_root"
        mv -- "$run_dir" "$archive_dir"
        echo "Archived previous run to $archive_dir"
    fi
    mkdir -p -- "$run_dir"
else
    [[ -d "$run_dir" ]] || { echo "No run directory to resume: $run_dir" >&2; exit 1; }
    resume_source=""
    [[ -f "$latest" ]] && resume_source=$latest
    [[ -z "$resume_source" && -f "$weights" ]] && resume_source=$weights
    [[ -n "$resume_source" ]] || { echo "No checkpoint found to resume in $run_dir" >&2; exit 1; }

    if model_uses_adam "$model_file" && [[ ! -f "$resume_source.optimizer" ]]; then
        if ! $allow_reset_optimizer; then
            echo "Adam state is missing: $resume_source.optimizer" >&2
            echo "Use --allow-reset-optimizer only if restarting Adam moments is intentional." >&2
            exit 1
        fi
        echo "Warning: resuming without Adam moments." >&2
    fi
    if [[ ! -f "$resume_source.training" ]]; then
        echo "Warning: training-control state is missing; best metrics and early stopping will restart." >&2
    fi

    # Model::loadWeights prefers the primary target when it exists. Preserve an
    # older final target so the loader can fall back to the newer latest file.
    if [[ "$resume_source" == "$latest" && -f "$weights" ]]; then
        resume_archive="$run_dir/archive"
        mkdir -p -- "$resume_archive"
        for suffix in "" .optimizer .training; do
            if [[ -f "$weights$suffix" ]]; then
                mv -- "$weights$suffix" \
                    "$resume_archive/${model}-before-resume-${timestamp}.weights${suffix}"
            fi
        done
        echo "Archived the older primary checkpoint before resuming latest."
    fi
fi

cat > "$run_dir/run-metadata.txt" <<EOF
model=$model
mode=$mode
started_utc=$timestamp
git_revision=$git_revision
executable=$train_bin
configuration=$config
weights=$weights
EOF

start_tensorboard "$run_dir" "$tensorboard_port"

echo "Starting training. Output is also written to $run_dir/training.log"
echo

cd -- "$run_dir"
set +e
"$train_bin" "${trainer_options[@]}" "$config" "$weights" 2>&1 | tee -a training.log
train_status=${PIPESTATUS[0]}
set -e

if ((train_status != 0)); then
    echo "Training exited with status $train_status. Check $run_dir/training.log" >&2
    exit "$train_status"
fi

echo "Training completed successfully."
