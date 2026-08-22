#!/usr/bin/env python3
"""Evaluate local KITTI detections with the KITTI 2D difficulty rules.

This is a local, KITTI-style evaluator.  It reports AP at 40 recall samples
for the supplied split; it is not the official server score.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


DIFFICULTIES = {
    "easy": (40, 0, 0.15),
    "moderate": (25, 1, 0.30),
    "hard": (25, 2, 0.50),
}
IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
IGNORE_FOR = {"Car": {"Van"}, "Pedestrian": {"Sitting Person"}}


@dataclass
class Box:
    kind: str
    box: tuple[float, float, float, float]
    score: float = 0.0
    truncation: float = 0.0
    occlusion: int = 0


def parse_gt(path: Path):
    result = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        result.append(Box(fields[0], tuple(map(float, fields[4:8])),
                          truncation=float(fields[1]), occlusion=int(fields[2])))
    return result


def parse_detections(path: Path):
    result = []
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 16:
            raise ValueError(f"{path}: expected 16 fields, got {len(fields)}")
        result.append(Box(fields[0], tuple(map(float, fields[4:8])),
                          score=float(fields[15])))
    return result


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1 + 1.0), max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0] + 1.0) * max(0.0, a[3] - a[1] + 1.0)
    area_b = max(0.0, b[2] - b[0] + 1.0) * max(0.0, b[3] - b[1] + 1.0)
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def ap_for_class(kind, difficulty, stems, gt_dir, det_dir):
    min_height, max_occ, max_trunc = DIFFICULTIES[difficulty]
    threshold = IOU_THRESHOLDS[kind]
    records = []
    total_valid = 0
    for stem in stems:
        gt = parse_gt(gt_dir / f"{stem}.txt")
        det = [d for d in parse_detections(det_dir / f"{stem}.txt") if d.kind == kind]
        valid, ignored = [], []
        for obj in gt:
            height = obj.box[3] - obj.box[1] + 1.0
            eligible = obj.kind == kind and height >= min_height and obj.occlusion <= max_occ and obj.truncation <= max_trunc
            if eligible:
                valid.append(obj)
            elif obj.kind == kind or obj.kind in IGNORE_FOR.get(kind, set()):
                ignored.append(obj)
        total_valid += len(valid)
        matched = [False] * len(valid)
        for d in det:
            best_i, best_overlap = -1, threshold
            for i, obj in enumerate(valid):
                overlap = iou(d.box, obj.box)
                if not matched[i] and overlap >= best_overlap:
                    best_i, best_overlap = i, overlap
            if best_i >= 0:
                matched[best_i] = True
                records.append((d.score, 1, 0))
                continue
            if any(iou(d.box, obj.box) >= threshold for obj in ignored):
                records.append((d.score, 0, 1))
                continue
            if kind == "Car":
                dont_care = [obj for obj in gt if obj.kind == "DontCare"]
                if any(iou(d.box, obj.box) > 0.5 for obj in dont_care):
                    records.append((d.score, 0, 1))
                    continue
            records.append((d.score, 0, 0))
    records.sort(key=lambda x: x[0], reverse=True)
    tp = fp = 0
    points = []
    for score, is_tp, is_ignored in records:
        if is_ignored:
            continue
        tp += is_tp
        fp += 1 - is_tp
        points.append((tp / total_valid if total_valid else 0.0,
                       tp / (tp + fp) if tp + fp else 0.0))
    precisions = []
    for i in range(40):
        target = i / 40.0
        precisions.append(max((p for r, p in points if r >= target), default=0.0))
    ap = sum(precisions) / 40.0
    return (ap, total_valid, sum(x[1] for x in records),
            sum(1 - x[1] - x[2] for x in records), sum(x[2] for x in records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    args = parser.parse_args()
    stems = [Path(line.strip()).stem for line in args.images.read_text().splitlines() if line.strip()]
    print(f"split: {len(stems)} images")
    print("class        easy AP   moderate AP   hard AP   valid GT (moderate)   TP/FP/ignored")
    for kind in IOU_THRESHOLDS:
        values = [ap_for_class(kind, diff, stems, args.gt_dir, args.detections) for diff in DIFFICULTIES]
        print(f"{kind:<12} {values[0][0]:8.4f}     {values[1][0]:8.4f}     {values[2][0]:8.4f}"
              f"        {values[1][1]:5d}          {values[1][2]}/{values[1][3]}/{values[1][4]}")
        print("             " + "  ".join(f"{d}: {v[0]:.4f}" for d, v in zip(DIFFICULTIES, values)))


if __name__ == "__main__":
    main()
