#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

usage()
{
    cat <<'EOF'
Usage: train-all-models.sh (--fresh | --resume) [options]

Run every registered PixieNN training configuration sequentially on one GPU.
The script stops immediately if any model fails.

Options are forwarded to train-model.sh:
  --verify-data
  --allow-reset-optimizer
  --dry-run

Selection:
  --yolo-only    Skip the ResNet18 configuration.
EOF
}

mode=""
yolo_only=false
forwarded=()

while (($#)); do
    case "$1" in
        --fresh|--resume)
            [[ -z "$mode" ]] || { echo "Choose one training mode." >&2; exit 2; }
            mode=$1
            forwarded+=("$1")
            ;;
        --verify-data|--allow-reset-optimizer|--dry-run)
            forwarded+=("$1")
            ;;
        --yolo-only)
            yolo_only=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

[[ -n "$mode" ]] || { usage >&2; exit 2; }

models=(
    resnet18
    tiny-yolo-voc
    yolo-nano
    yolov1-tiny
    yolov3-tiny-voc
    yolov3-tiny
    yolov3
    yolov7
)

for model in "${models[@]}"; do
    $yolo_only && [[ "$model" == resnet18 ]] && continue
    echo
    echo "======================================================================"
    echo "Training $model"
    echo "======================================================================"
    "$script_dir/train-model.sh" "$model" "${forwarded[@]}"
done

echo "All requested model runs completed successfully."
