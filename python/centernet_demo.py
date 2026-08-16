"""Run a CenterNet experiment through the native PixieNN CUDA engine.

Python owns only experiment orchestration and the HTML report. Model creation,
data loading, augmentation, CenterNet targets/loss, CUDA execution, native
checkpoints, and decoding all stay in PixieNN's C++ implementation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def import_pixienn(root: Path):
    source_package = str(root / "python")
    for build in ("cmake-build-python-cuda", "cmake-build-debug-cuda", "cmake-build-release-cuda", "cmake-build-python"):
        package = root / build / "python"
        if list((package / "pixienn").glob("_native*.so")):
            sys.path.insert(0, str(package))
            break
    else:
        # When installed with pip, do not let the source checkout's package
        # directory shadow the installed package (the checkout has no .so).
        while source_package in sys.path:
            sys.path.remove(source_package)
    from pixienn import Model
    return Model


def local_voc_images(root: Path, count: int, excluded: set[str] | None = None) -> list[Path]:
    excluded = excluded or set()
    paths = []
    for list_name in ("train.txt", "val.txt"):
        for line in (root / "resources/data/voc" / list_name).read_text().splitlines():
            listed = Path(line.strip())
            image = listed if listed.exists() else root / "resources/data/voc/images" / f"{listed.stem}.jpg"
            if listed.stem in excluded:
                continue
            if image.exists() and image not in paths:
                paths.append(image)
            if len(paths) == count:
                break
        if len(paths) == count:
            break
    if len(paths) != count:
        raise RuntimeError(f"expected {count} local VOC images, found {len(paths)}")
    return paths


def write_image_list(path: Path, images: list[Path]) -> None:
    path.write_text("".join(f"{image}\n" for image in images))


def prepare_resume_checkpoint(output: Path) -> bool:
    """Promote the latest native checkpoint when the canonical file is absent."""
    weights = output / "centernet.weights"
    latest = output / "backup/centernet_latest.weights"
    if weights.exists() or not latest.exists():
        return weights.exists()

    for suffix in ("", ".optimizer", ".training"):
        source = Path(f"{latest}{suffix}")
        if source.exists():
            shutil.copyfile(source, Path(f"{weights}{suffix}"))
    return True


def native_model(Model, output: Path, train_list: Path, val_list: Path, *, batch: int,
                 height: int, width: int, max_batches: int, validation_threshold: float):
    options = {
        "subdivisions": 1,
        "max_batches": max_batches,
        "momentum": 0.9,
        "decay": 0.0005,
        "augmentation": {
            "enabled": True,
            "flip": True,
            "jitter": 0.2,
            "saturation": 1.5,
            "exposure": 1.5,
            "hue": 0.1,
        },
        "adam": {
            "enabled": True,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        },
        "validation": {
            "enabled": True,
            "interval": max(1, max_batches // 10),
            "confidence_threshold": validation_threshold,
            "ap_confidence_threshold": 0.001,
            "iou_threshold": 0.5,
            "nms_threshold": 0.4,
        },
        "gradient_rescale": {"enabled": True, "threshold": 100.0},
        "gradient_clipping": {"enabled": False},
        "save_weights_interval": max(1, max_batches // 10),
        "write_metrics_interval": 50,
        "backup-dir": str(output / "backup"),
        "weights-file": str(output / "centernet.weights"),
        "event_file": str(output / "events.tfevents"),
        "learning_rate": {
            "initial_learning_rate": 0.0002,
            "policy": "sigmoid",
            "sigmoid": {
                "target_learning_rate": 0.000005,
                "factor": 8.0,
            },
        },
    }
    model = Model(3, height, width, batch=batch, device="cuda", **options)
    model.set_labels(CLASSES)
    model.configure_training(**{
        "train-images": str(train_list),
        "train-labels": str(rooted_label_dir(train_list)),
        "val-images": str(val_list),
        "val-labels": str(rooted_label_dir(val_list)),
    })
    model = build_centernet_graph(model).build()
    # Python-built native graphs do not implicitly load weights during build.
    # Explicitly load the canonical checkpoint so native optimizer/training
    # state is restored before Model.train() starts.
    weights = output / "centernet.weights"
    if weights.exists():
        model.load_weights(weights)
    return model


def rooted_label_dir(image_list: Path) -> Path:
    return image_list.parents[2] / "resources/data/voc/labels"


def build_centernet_graph(model):
    """Build the larger native stride-4 CenterNet encoder/decoder graph."""
    return (model
            .conv(32, 3, stride=2, pad=True, activation="mish", batch_normalize=True)
            .conv(64, 3, stride=2, pad=True, activation="mish", batch_normalize=True)
            .conv(128, 3, stride=2, pad=True, activation="mish", batch_normalize=True)
            .conv(128, 3, pad=True, activation="mish", batch_normalize=True)
            .conv(256, 3, stride=2, pad=True, activation="mish", batch_normalize=True)
            .conv(256, 3, pad=True, activation="mish", batch_normalize=True)
            .conv(384, 3, stride=2, pad=True, activation="mish", batch_normalize=True)
            .conv(384, 3, pad=True, activation="mish", batch_normalize=True)
            .conv(256, 1, pad=False, activation="mish", batch_normalize=True)
            .upsample(2)
            .route([-1, 5])
            .conv(256, 3, pad=True, activation="mish", batch_normalize=True)
            .conv(128, 1, pad=False, activation="mish", batch_normalize=True)
            .upsample(2)
            .route([-1, 3])
            .conv(128, 3, pad=True, activation="mish", batch_normalize=True)
            .conv(64, 1, pad=False, activation="mish", batch_normalize=True)
            .upsample(2)
            .route([-1, 1])
            .conv(128, 3, pad=True, activation="mish", batch_normalize=True)
            .conv(64, 3, pad=True, activation="mish", batch_normalize=True)
            .conv(24, 1, pad=False, activation="linear")
            .centernet(focal_alpha=2.0, focal_beta=4.0, heatmap_bias=-2.19,
                       size_weight=0.1, max_detections=100))


def write_run_metadata(root: Path, output: Path, *, batch: int, image_size: int,
                       max_batches: int, mode: str, validation_images: int,
                       validation_threshold: float) -> None:
    """Write the dynamic run contract consumed by the training monitor."""
    validation_interval = max(1, max_batches // 10) if max_batches else 1
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    metadata = {
        "model": "centernet-python",
        "mode": mode,
        "started_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_revision": revision,
        "executable": sys.executable,
        "batch": batch,
        "image_size": image_size,
        "max_batches": max_batches,
        "validation_interval": validation_interval,
        "validation_images": validation_images,
        "validation_threshold": validation_threshold,
        "learning_rate_policy": "sigmoid",
        "weights": str(output / "centernet.weights"),
    }
    (output / "run-metadata.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items()))


def read_run_metadata(output: Path) -> dict[str, str]:
    metadata = output / "run-metadata.txt"
    if not metadata.exists():
        return {}
    values = {}
    for line in metadata.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def native_inference_model(Model, weights: Path, *, height: int, width: int, batch: int):
    model = Model(3, height, width, batch=batch, device="cuda")
    model.set_labels(CLASSES)
    model = build_centernet_graph(model).build()
    return model.load_weights(weights)


def write_report(output: Path, rows: list[dict], epochs: int | str, batch: int,
                 max_batches: int, confidence: float) -> None:
    cards = []
    for row in rows:
        detections = row["detections"]
        if detections:
            summary = "".join(
                f'<li>{d["label"]} · {d["confidence"]:.1%}</li>'
                for d in detections)
            detection_text = f'<p><b>{len(detections)} detections</b> at confidence ≥ {confidence:.2f}</p><ul>{summary}</ul>'
        else:
            detection_text = f'<p><b>No detections</b> at confidence ≥ {confidence:.2f}</p>'
        cards.append(f'''<article><h2>{row["name"]}</h2>
<img src="images/{row["rendered"].name}" alt="native PixieNN detections">
{detection_text}
 </article>''')
    run_label = (f"{epochs} epochs, {max_batches} optimizer steps"
                 if isinstance(epochs, int)
                 else f"{epochs}; trained for {max_batches} optimizer steps")
    html = f'''<!doctype html><meta charset="utf-8"><title>PixieNN native CenterNet</title>
<style>body{{font:16px system-ui;background:#17202a;color:#edf2f7;margin:2rem}}h1{{color:#72e0a5}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1.5rem}}
article{{background:#243447;padding:1rem;border-radius:12px}}img{{width:100%;border-radius:8px}}
</style>
<h1>PixieNN native CenterNet</h1>
<p>Native C++/CUDA training and inference; 100 VOC images, {run_label},
batch {batch}, inference confidence {confidence:.2f}.</p><div class="grid">{"".join(cards)}</div>'''
    (output / "index.html").write_text(html)


def publish_report(root: Path, output: Path) -> None:
    """Publish the image-only report to the monitor's Vite public tree."""
    public = root / "monitor/public/centernet"
    public_images = public / "images"
    public_images.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "index.html", public / "index.html")
    for image in (output / "images").glob("*.jpg"):
        shutil.copyfile(image, public_images / image.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50,
                        help="training epochs when --steps is not provided")
    parser.add_argument("--steps", type=int, default=None,
                        help="exact number of optimizer steps; overrides --epochs")
    parser.add_argument("--image-count", type=int, default=100,
                        help="number of VOC training images")
    parser.add_argument("--exclude-list", type=Path, default=None,
                        help="image-list file whose image stems must be excluded from training")
    parser.add_argument("--validation-count", type=int, default=10,
                        help="number of validation images kept disjoint from training")
    parser.add_argument("--validation-threshold", type=float, default=0.05,
                        help="confidence threshold used for validation metrics")
    parser.add_argument("--inference-list", type=Path, default=None,
                        help="image-list used for the final inference mosaic")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="training batch size; inference defaults to the checkpoint run batch")
    parser.add_argument("--image-size", type=int, default=None,
                        help="network input size; inference defaults to the checkpoint run size")
    parser.add_argument("--confidence", type=float, default=0.05,
                        help="native detection confidence threshold")
    parser.add_argument("--inference-only", action="store_true",
                        help="skip training and infer using the existing checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="discard this output's CenterNet checkpoints/events and start over")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "runs/centernet").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "backup").mkdir(exist_ok=True)
    run_metadata = read_run_metadata(output)
    if args.image_size is None:
        args.image_size = int(run_metadata.get("image_size", 128))
    if args.batch_size is None:
        args.batch_size = int(run_metadata.get("batch", 8))
    if args.fresh:
        for path in output.glob("centernet.weights*"):
            path.unlink()
        for path in (output / "backup").glob("centernet*.weights*"):
            path.unlink()
        event_file = output / "events.tfevents"
        if event_file.exists():
            event_file.unlink()
    excluded = set()
    if args.exclude_list:
        excluded = {Path(line.strip()).stem for line in args.exclude_list.read_text().splitlines()
                    if line.strip()}
    images = local_voc_images(root, args.image_count, excluded)
    validation_excluded = excluded | {image.stem for image in images}
    validation_images = local_voc_images(root, args.validation_count, validation_excluded)
    max_batches = (args.steps if args.steps is not None else
                   args.epochs * ((len(images) + args.batch_size - 1) // args.batch_size))
    if max_batches <= 0:
        raise ValueError("training steps must be positive")
    checkpoint_exists = prepare_resume_checkpoint(output)
    fresh_training = not args.inference_only and not checkpoint_exists
    if fresh_training:
        # Event files are append-only. A fresh experiment in a reused output
        # directory must not inherit the previous run's timeline or metrics.
        event_file = output / "events.tfevents"
        if event_file.exists():
            event_file.unlink()
    metadata_path = output / "run-metadata.txt"
    if not args.inference_only or not metadata_path.exists():
        write_run_metadata(root, output, batch=args.batch_size, image_size=args.image_size,
                           max_batches=max_batches,
                           validation_images=len(validation_images),
                           validation_threshold=args.validation_threshold,
                           mode=("inference-only" if args.inference_only else
                                 ("resume" if checkpoint_exists else "fresh")))
    Model = import_pixienn(root)
    if not args.inference_only:
        train_list = output / f"train-{len(images)}.txt"
        val_list = output / f"val-{len(validation_images)}.txt"
        write_image_list(train_list, images)
        write_image_list(val_list, validation_images)
        model = native_model(Model, output, train_list, val_list, batch=args.batch_size,
                             height=args.image_size, width=args.image_size,
                             max_batches=max_batches,
                             validation_threshold=args.validation_threshold)
        print(f"training native CenterNet on CUDA for {max_batches} optimizer steps", flush=True)
        model.train()
    weights = output / "centernet.weights"
    if not weights.exists():
        latest = output / "backup/centernet_latest.weights"
        if latest.exists():
            weights = latest
    if args.inference_only and not weights.exists():
        raise FileNotFoundError(
            f"no checkpoint found in {output}; run without --inference-only first")

    inference_model = native_inference_model(Model, weights, height=args.image_size,
                                             width=args.image_size, batch=args.batch_size)

    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    rows = []
    inference_list = (args.inference_list.resolve() if args.inference_list
                      else output / "inference-10.txt")
    if args.inference_list is None:
        write_image_list(inference_list, images[:10])
    previous = Path.cwd()
    try:
        # One native batched forward pass writes the complete predictions mosaic.
        import os
        os.chdir(image_dir)
        result = inference_model.predict_batch_image_list(
            inference_list, confidence=args.confidence, nms_threshold=0.4)
        document = json.loads(result)
        detections = [
            {
                "label": f"image {int(feature['properties']['batch_id']) + 1}: "
                         f"{feature['properties']['class']}",
                "confidence": float(feature["properties"]["confidence"]),
            }
            for feature in document.get("features", [])
        ]
        rendered = image_dir / "predictions-mosaic.jpg"
        shutil.copyfile(image_dir / "predictions.jpg", rendered)
        rows.append({"name": "validation images (native batch mosaic)",
                     "rendered": rendered, "detections": detections})
    finally:
        os.chdir(previous)
    report_epochs = ("inference-only checkpoint" if args.inference_only else
                     (f"{args.steps} steps" if args.steps is not None else args.epochs))
    report_steps = (int(run_metadata.get("max_batches", 0))
                    if args.inference_only else max_batches)
    write_report(output, rows, report_epochs,
                 args.batch_size, report_steps,
                 args.confidence)
    publish_report(root, output)
    print(f"report: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
