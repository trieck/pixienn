# PixieNN training scripts

The scripts in this directory run CUDA training from isolated directories under
`runs/`. Each run keeps its TensorBoard events, log, weights, optimizer state,
training-control state, and checkpoint backup directory together.

## Train one model

List the available presets:

```bash
./shell/train-model.sh --list
```

Start a clean YOLOv7 run:

```bash
./shell/train-model.sh yolov7 --fresh --verify-data
```

`--fresh` does not delete an earlier run. It moves it under `runs/archive/`
before creating the new run directory.

Resume the latest checkpoint:

```bash
./shell/train-model.sh yolov7 --resume
```

Resume prefers `backup/yolov7_latest.weights`. If an older primary
`yolov7.weights` exists, the script archives it first so PixieNN cannot silently
load it instead of the latest checkpoint. Adam-based models require the matching
`.optimizer` sidecar unless `--allow-reset-optimizer` is explicitly supplied.

Preview a run without changing files:

```bash
./shell/train-model.sh yolov3-tiny-voc --fresh --dry-run
```

## Validate data

Training performs a quick configuration preflight automatically. To verify every
manifest image and corresponding label before committing GPU time:

```bash
./shell/check-training-data.sh resources/cfg/yolov3-tiny-voc-cfg.yml
```

## Train every preset

The following runs every configuration sequentially on one GPU and can take a
very long time:

```bash
./shell/train-all-models.sh --fresh --verify-data
```

To run only detector configurations:

```bash
./shell/train-all-models.sh --fresh --verify-data --yolo-only
```

This intentionally runs models sequentially. Concurrent jobs would compete for
GPU memory and make timing and out-of-memory failures difficult to interpret.

## Monitoring

For a single model:

```bash
tensorboard --logdir=runs/yolov7 --port=6006
```

For all isolated runs:

```bash
tensorboard --logdir=runs --port=6006
```

## Dataset scope

The scripts use the checked-in configurations exactly as written:

| Model | Current manifest scope |
|---|---|
| `resnet18` | ImageNet smoke manifests; local `/opt/imagenet` label paths must be configured |
| `tiny-yolo-voc` | VOC smoke preset (`train-200`, `val-1`) |
| `yolo-nano` | Full configured VOC manifests |
| `yolov1-tiny` | Full configured VOC manifests |
| `yolov3-tiny-voc` | Full configured VOC manifests (`train-10000`, `val-2000`) |
| `yolov3-tiny` | COCO smoke preset (`train-2`, `val-1`) |
| `yolov3` | COCO smoke preset (`train-200`, `val-1`) |
| `yolov7` | COCO 82,081-image training and 1,000-image validation manifests |

Smoke presets are useful for correctness and overfitting tests, but their metrics
are not publication benchmarks.

The checked-in ResNet18 configuration currently references machine-local
`/opt/imagenet` label directories. The preflight intentionally blocks that run
until those paths exist or the configuration is updated. Use `--yolo-only` when
running all currently available detector presets.

## Overrides

The scripts first use `build/bin/pixienn-train` from the README's standard
CMake build, then try the release and debug CUDA IDE build directories.
Override the executable or run root when needed:

```bash
PIXIENN_TRAIN_BIN=/path/to/pixienn-train \
PIXIENN_RUNS_DIR=/data/pixienn-runs \
./shell/train-model.sh yolov7 --fresh
```

The GPU preflight deliberately refuses to run when `nvidia-smi` cannot access an
NVIDIA GPU. These scripts never add PixieNN's `--no-gpu` option.

## Script integration test

The integration test uses fake GPU and trainer executables, so it exercises
fresh-run archival, resume selection, sidecars, logging, and lock cleanup without
starting a real training job:

```bash
./tests/shell/training_scripts.sh
```
