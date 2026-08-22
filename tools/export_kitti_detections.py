#!/usr/bin/env python3
"""Export PixieNN detections in the official KITTI object format."""

import argparse
import json
from pathlib import Path

import yaml

from pixienn import Model


KITTI_CLASSES = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist",
}


def build_model(model_path: Path, weights_path: Path) -> Model:
    spec = yaml.safe_load(model_path.read_text())["model"]
    model = Model(
        spec["channels"], spec["height"], spec["width"], spec["batch"],
        device="cuda",
        **{key: value for key, value in spec.items()
           if key not in {"channels", "height", "width", "batch", "layers"}},
    )
    for layer in spec["layers"]:
        model.add_layer(layer)
    model.set_labels(["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"])
    model.build()
    model.load_weights(weights_path)
    return model


def feature_box(feature):
    points = feature["geometry"]["coordinates"][0]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    # PixieNN GeoJSON uses image x and negated image y coordinates.
    return min(xs), -max(ys), max(xs), -min(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms", type=float, default=0.4)
    args = parser.parse_args()

    image_paths = [Path(line.strip()) for line in args.images.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.model, args.weights)
    exported = 0
    detections = 0

    for index, image_path in enumerate(image_paths, 1):
        document = json.loads(model.predict_json(image_path, confidence=args.confidence,
                                                  nms_threshold=args.nms))
        lines = []
        for feature in document["features"]:
            label = KITTI_CLASSES.get(feature["properties"]["class"])
            if label is None:
                continue
            x1, y1, x2, y2 = feature_box(feature)
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(feature["properties"]["confidence"])
            lines.append(
                f"{label} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.8f}"
            )
        (args.output / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        detections += len(lines)
        exported += 1
        if index % 100 == 0 or index == len(image_paths):
            print(f"exported {index}/{len(image_paths)} images, {detections} detections", flush=True)


if __name__ == "__main__":
    main()
