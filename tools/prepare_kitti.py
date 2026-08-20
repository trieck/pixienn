#!/usr/bin/env python3
"""Convert the KITTI object-detection training set to Darknet layout."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


CLASSES = ("car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc")
CLASS_IDS = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
}


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def convert_label(source: Path, destination: Path, width: int, height: int) -> None:
    rows: list[str] = []
    for line in source.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        class_name = fields[0]
        if class_name == "DontCare":
            continue
        if class_name not in CLASS_IDS:
            raise ValueError(f"unknown KITTI class {class_name!r} in {source}")
        left, top, right, bottom = map(float, fields[4:8])
        center_x = ((left + right) * 0.5) / width
        center_y = ((top + bottom) * 0.5) / height
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        rows.append(f"{CLASS_IDS[class_name]} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
    destination.write_text("\n".join(rows) + ("\n" if rows else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Extracted KITTI object directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image_source = args.source / "training" / "image_2"
    label_source = args.source / "training" / "label_2"
    images = sorted(image_source.glob("*.png"))
    if not images:
        raise SystemExit(f"no KITTI images found under {image_source}")

    image_output = args.output / "images"
    label_output = args.output / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    train: list[str] = []
    validation: list[str] = []
    split_point = int(len(images) * 0.8)
    for index, source_image in enumerate(images):
        image = image_output / source_image.name
        label = label_output / f"{source_image.stem}.txt"
        if not image.exists():
            image.symlink_to(source_image.resolve())
        width, height = png_size(source_image)
        convert_label(label_source / f"{source_image.stem}.txt", label, width, height)
        (train if index < split_point else validation).append(str(image.resolve()))

    (args.output / "train.txt").write_text("\n".join(train) + "\n")
    (args.output / "val.txt").write_text("\n".join(validation) + "\n")
    (args.output / "kitti.names").write_text("\n".join(CLASSES) + "\n")
    print(f"converted {len(images)} images: {len(train)} train, {len(validation)} validation")


if __name__ == "__main__":
    main()
