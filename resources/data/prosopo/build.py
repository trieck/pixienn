#!/usr/bin/env python3
"""Build the one-class Prosopo person-detection dataset.

The source datasets remain untouched.  Images are hard-linked into Prosopo
when possible, so the generated dataset is self-contained under this
directory without needlessly duplicating large image files.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
# The current CenterNet Prosopo model uses a 512x512 input. Mosaic places each
# source image in a 256x256 quadrant, so curate boxes against that effective
# per-source resolution.
TRAINING_SIZE = 256
MIN_BOX_PIXELS = 10


@dataclass(frozen=True)
class Record:
    source: str
    source_split: str
    image: Path
    label: Path
    person_classes: frozenset[int]


def read_source_list(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def source_records() -> list[Record]:
    records: list[Record] = []

    # COCO labels use class 0 for person.
    for split in ("train2014", "val2014"):
        image_dir = DATA / "coco" / split
        label_dir = DATA / "coco" / "labels" / split
        for image in sorted(image_dir.glob("*.jpg")):
            records.append(Record("coco", split, image, label_dir / f"{image.stem}.txt", frozenset({0})))

    # KITTI uses pedestrian=3 and person_sitting=4.  Cyclist=5 is omitted:
    # its annotation is the cyclist+bicycle object, not a person-only box.
    for split in ("train", "val"):
        for image in read_source_list(DATA / "kitti" / f"{split}.txt"):
            records.append(Record("kitti", split, image, DATA / "kitti" / "labels" / f"{image.stem}.txt", frozenset({3, 4})))

    # VOC uses person=14 in the local YOLO label conversion.
    for split in ("train", "val"):
        for image in read_source_list(DATA / "voc" / f"{split}.txt"):
            records.append(Record("voc", split, image, DATA / "voc" / "labels" / f"{image.stem}.txt", frozenset({14})))

    return records


def person_boxes(record: Record) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not record.image.is_file() or not record.label.is_file():
        return boxes
    for line_number, line in enumerate(record.label.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 5 or int(fields[0]) not in record.person_classes:
            continue
        values = tuple(float(value) for value in fields[1:])
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"out-of-range box in {record.label}:{line_number}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            continue
        boxes.append(values)
    return boxes


def usable_person_boxes(record: Record, boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Keep boxes large enough to provide a useful target at training size."""
    with Image.open(record.image) as image:
        width, height = image.size
    scale = min(TRAINING_SIZE / width, TRAINING_SIZE / height)
    return [box for box in boxes
            if box[2] * width * scale >= MIN_BOX_PIXELS
            and box[3] * height * scale >= MIN_BOX_PIXELS]


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def diagnostic_validation_set(paths: list[str], manifest: list[dict], limit: int = 1000) -> list[str]:
    """Choose a deterministic, source-balanced validation subset."""
    if len(paths) <= limit:
        return paths
    path_set = set(paths)
    by_source: dict[str, list[str]] = defaultdict(list)
    for item in manifest:
        if item["image"] in path_set:
            by_source[item["source"]].append(item["image"])

    total = len(paths)
    quotas = {source: max(1, round(limit * len(items) / total))
              for source, items in by_source.items()}
    while sum(quotas.values()) > limit:
        source = max(quotas, key=lambda value: quotas[value])
        if quotas[source] > 1:
            quotas[source] -= 1
    while sum(quotas.values()) < limit:
        source = max(by_source, key=lambda value: len(by_source[value]) - quotas[value])
        quotas[source] += 1

    selected = []
    for source, items in by_source.items():
        ranked = sorted(items, key=lambda value: hashlib.sha256(value.encode()).digest())
        selected.extend(ranked[:quotas[source]])
    return sorted(selected)


def main() -> None:
    images_dir = ROOT / "images"
    labels_dir = ROOT / "labels"
    for directory in (images_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # A clean rebuild prevents stale records if source lists change.
    for path in images_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
    for path in labels_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()

    splits: dict[str, list[str]] = {"train": [], "val": []}
    manifest = []
    seen: set[Path] = set()
    source_stats: dict[str, dict[str, int]] = {}
    filtered_images = 0
    filtered_boxes = 0

    for record in source_records():
        resolved_image = record.image.resolve()
        if resolved_image in seen:
            continue
        seen.add(resolved_image)
        boxes = person_boxes(record)
        if not boxes:
            continue
        usable_boxes = usable_person_boxes(record, boxes)
        filtered_boxes += len(boxes) - len(usable_boxes)
        if not usable_boxes:
            filtered_images += 1
            continue
        boxes = usable_boxes

        output_split = {
            "train2014": "train",
            "val2014": "val",
        }.get(record.source_split, record.source_split)
        # COCO's source stem already contains both the dataset and split;
        # retain only its stable numeric image ID in the Prosopo filename.
        image_id = record.image.stem.rsplit("_", 1)[-1] if record.source == "coco" else record.image.stem
        output_stem = f"{record.source}_{output_split}_{image_id}"
        output_image = images_dir / f"{output_stem}{record.image.suffix.lower()}"
        output_label = labels_dir / f"{output_stem}.txt"
        link_or_copy(record.image, output_image)
        output_label.write_text("".join("0 %.9f %.9f %.9f %.9f\n" % box for box in boxes))

        output_path = str(output_image.resolve())
        splits["train" if record.source_split == "train" or record.source == "coco" and record.source_split == "train2014" else "val"].append(output_path)
        manifest.append({
            "image": output_path,
            "label": str(output_label.resolve()),
            "source": record.source,
            "source_split": record.source_split,
            "source_image": str(record.image.resolve()),
            "source_label": str(record.label.resolve()),
            "person_boxes": len(boxes),
        })

        stats = source_stats.setdefault(record.source, {"images": 0, "boxes": 0})
        stats["images"] += 1
        stats["boxes"] += len(boxes)

    (ROOT / "train.txt").write_text("\n".join(splits["train"]) + "\n")
    diagnostic_val = diagnostic_validation_set(splits["val"], manifest)
    (ROOT / "val.txt").write_text("\n".join(diagnostic_val) + "\n")

    (ROOT / "prosopo.names").write_text("person\n")
    (ROOT / "manifest.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest))
    train_images = set(splits["train"])
    diagnostic_images = set(diagnostic_val)
    summary = {
        "dataset": "prosopo",
        "classes": ["person"],
        "train_images": len(splits["train"]),
        "val_images": len(diagnostic_val),
        "train_boxes": sum(item["person_boxes"] for item in manifest if item["image"] in train_images),
        "val_boxes": sum(item["person_boxes"] for item in manifest if item["image"] in diagnostic_images),
        "sources": source_stats,
        "curation": {
            "training_size": TRAINING_SIZE,
            "minimum_box_pixels": MIN_BOX_PIXELS,
            "filtered_boxes": filtered_boxes,
            "removed_images": filtered_images,
        },
        "kitti_person_classes": {"3": "pedestrian", "4": "person_sitting"},
        "kitti_excluded_classes": {"5": "cyclist"},
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
