#!/usr/bin/env python3
"""Incremental, full-run TensorFlow event-file scalar reader for the local monitor.

The accumulator uses bounded reservoir sampling so memory stays
bounded as a run grows. Each response is then reduced to an extrema-preserving
series that spans the complete run, keeping the browser render bounded too.
"""

import json
import base64
import hashlib
import math
import os
import sys
import tempfile
from collections import deque

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util


event_file = sys.argv[1]
start_time = float(sys.argv[2]) if len(sys.argv) > 2 else None
MAX_DISPLAY_POINTS = 1_200
MAX_WINDOW_POINTS = 10_000
MAX_CARD_WINDOW_POINTS = 5_000
MAX_CONFUSION_CLASSES = 24
WINDOW_TAGS = {"avg-loss", "learning-rate"}
CACHE_VERSION = 14
CACHE_UPDATE_BYTES = 1_000_000
# RawEventFileLoader preserves the original Summary.Value oneof. TensorBoard's
# higher-level EventFileLoader migrates Summary.Image into a TensorProto.
event_loader = RawEventFileLoader(event_file)
tail_series = {tag: deque(maxlen=MAX_WINDOW_POINTS) for tag in WINDOW_TAGS}
card_tail_series = {}
pr_curves = {}
images = {}
confusion_matrix = None
full_series = {}
previous_raw_step = -1
previous_normalized_step = -1
step_offset = 0
cache_key = hashlib.sha256(f"{os.path.abspath(event_file)}:{start_time}".encode()).hexdigest()[:24]
cache_file = os.path.join(tempfile.gettempdir(), f"pixienn-event-cache-{cache_key}.json")
cache_file_size = 0
loaded_file_size = 0
loaded_file_mtime_ns = None
loaded_event_count = 0


def display_series(values, limit=MAX_DISPLAY_POINTS):
    """Return a bounded series covering the complete input range.

    Keeping the first, last, minimum, and maximum value from each time bucket
    preserves long-run shape and spikes without creating one DOM/SVG point per
    training event.
    """
    if len(values) <= limit:
        return values

    bucket_count = max(1, limit // 4)
    sampled = []
    for bucket in range(bucket_count):
        start = bucket * len(values) // bucket_count
        end = (bucket + 1) * len(values) // bucket_count
        if end <= start:
            continue
        indexes = {start, end - 1}
        indexes.add(min(range(start, end), key=lambda index: values[index]["value"]))
        indexes.add(max(range(start, end), key=lambda index: values[index]["value"]))
        sampled.extend(values[index] for index in sorted(indexes))
    return sampled


def scalar_value(summary):
    if summary.HasField("simple_value"):
        return float(summary.simple_value)
    if summary.HasField("tensor"):
        array = tensor_util.make_ndarray(summary.tensor)
        if array.size != 1:
            return None
        return float(array.reshape(-1)[0])
    return None


def update_series():
    global previous_raw_step, previous_normalized_step, step_offset
    global loaded_file_size, loaded_file_mtime_ns, loaded_event_count, confusion_matrix
    current_stat = os.stat(event_file)
    if (loaded_file_size == current_stat.st_size
            and loaded_file_mtime_ns == current_stat.st_mtime_ns):
        return
    if loaded_file_size > current_stat.st_size:
        full_series.clear()
        card_tail_series.clear()
        pr_curves.clear()
        images.clear()
        confusion_matrix = None
        for values in tail_series.values():
            values.clear()
        previous_raw_step = -1
        previous_normalized_step = -1
        step_offset = 0
        loaded_event_count = 0
    for record in event_loader.Load():
        event = event_pb2.Event.FromString(record)
        if start_time is not None and event.wall_time < start_time:
            continue
        if event.step < previous_raw_step:
            step_offset += previous_normalized_step + 1 - (event.step + step_offset)
        normalized_step = event.step + step_offset
        previous_raw_step = event.step
        previous_normalized_step = normalized_step
        for summary in event.summary.value:
            if summary.HasField("image"):
                encoded = base64.b64encode(summary.image.encoded_image_string).decode("ascii")
                images[summary.tag] = {
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "width": summary.image.width,
                    "height": summary.image.height,
                    "data": f"data:image/jpeg;base64,{encoded}",
                }
                continue
            if summary.tag == "validation/confusion-matrix" and summary.HasField("tensor"):
                array = tensor_util.make_ndarray(summary.tensor)
                confusion_matrix = {"step": event.step, "values": array.astype(int).reshape(-1).tolist(),
                                    "size": int(array.shape[0])}
                continue
            if summary.tag == "validation/confusion-matrix/labels" and summary.HasField("tensor"):
                if confusion_matrix is None: confusion_matrix = {"step": event.step}
                labels = list(summary.tensor.string_val)
                confusion_matrix["labels"] = [item.decode("utf-8") if isinstance(item, bytes) else str(item)
                                                for item in labels]
                confusion_matrix["step"] = event.step
                continue
            if summary.tag.startswith("validation/micro-pr/") and summary.HasField("tensor"):
                array = tensor_util.make_ndarray(summary.tensor)
                if array.ndim == 2 and 3 in array.shape:
                    rows = array if array.shape[1] == 3 else array.T
                    points = [{"confidence": float(row[0]), "precision": float(row[1]), "recall": float(row[2])} for row in rows]
                    # PR curves are validation snapshots. Keep the raw
                    # optimizer step so resumed-run display offsets cannot
                    # turn a valid interval boundary into a fake step.
                    pr_curves[summary.tag] = {"step": event.step, "points": points}
                continue
            value = scalar_value(summary)
            if value is None or not math.isfinite(value):
                continue
            # Keep the on-disk optimizer step as well as the display step.
            # Display steps are made monotonic across resumed runs, but
            # validation scheduling must use the raw optimizer step because
            # boundaries are defined by the model's actual step counter.
            point = {"step": normalized_step, "raw_step": event.step, "value": value, "wall_time": event.wall_time}
            full_series.setdefault(summary.tag, []).append(point)
            card_tail_series.setdefault(summary.tag, deque(maxlen=MAX_CARD_WINDOW_POINTS)).append(point)
            if summary.tag in tail_series:
                tail_series[summary.tag].append(point)
    loaded_file_size = current_stat.st_size
    loaded_file_mtime_ns = current_stat.st_mtime_ns
    loaded_event_count = 0


def load_cache():
    global cache_file_size
    try:
        with open(cache_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        current_size = os.path.getsize(event_file)
        cached_size = int(payload.get("file_size", 0))
        response = payload.get("response")
        if payload.get("version") != CACHE_VERSION or cached_size > current_size:
            return None
        if not isinstance(response, dict) or not isinstance(response.get("series"), dict):
            return None
        response["series"] = {
            tag: display_series(values, MAX_DISPLAY_POINTS if tag in WINDOW_TAGS else 240)
            for tag, values in response["series"].items()
        }
        if isinstance(response.get("tails"), dict):
            response["tails"] = {tag: values for tag, values in response["tails"].items() if tag in WINDOW_TAGS}
        cache_file_size = cached_size
        return response
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def save_cache(response):
    global cache_file_size
    try:
        current_size = os.path.getsize(event_file)
        if cache_file_size and current_size - cache_file_size < CACHE_UPDATE_BYTES:
            return
        temporary = f"{cache_file}.{os.getpid()}.tmp"
        payload = {"version": CACHE_VERSION, "file_size": current_size, "response": response}
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary, cache_file)
        cache_file_size = current_size
    except OSError:
        pass


def activity_summary():
    events = sorted((point for point in full_series.get("avg-loss", []) if math.isfinite(point.get("wall_time", math.nan))), key=lambda point: point["wall_time"])
    if len(events) < 2:
        return None
    validation_ends = sorted((point for tag in ("mAP50", "micro-avg-f1") for point in full_series.get(tag, []) if math.isfinite(point.get("wall_time", math.nan))), key=lambda point: point["wall_time"])
    exact = sorted((point for point in full_series.get("validation/duration-seconds", []) if math.isfinite(point.get("wall_time", math.nan)) and math.isfinite(point.get("value", math.nan))), key=lambda point: point["wall_time"])
    gaps = sorted(b["wall_time"] - a["wall_time"] for a, b in zip(events, events[1:]) if 0 < b["wall_time"] - a["wall_time"] <= 300)
    cadence = gaps[len(gaps) // 2] if gaps else 60.0
    candidates = []
    for left, right in zip(events, events[1:]):
        gap = right["wall_time"] - left["wall_time"]
        if gap <= cadence * 2 or gap > 2 * 60 * 60:
            continue
        if any(left["wall_time"] < point["wall_time"] <= right["wall_time"] for point in validation_ends):
            candidates.append(gap - cadence)
    candidates.sort()
    inferred = candidates[len(candidates) // 2] if candidates else 0.0
    active = validation = offline = 0.0
    segments = []
    for left, right in zip(events, events[1:]):
        gap = max(0.0, right["wall_time"] - left["wall_time"])
        exact_seconds = sum(max(0.0, point["value"]) for point in exact if left["wall_time"] < point["wall_time"] <= right["wall_time"])
        has_validation = any(left["wall_time"] < point["wall_time"] <= right["wall_time"] for point in validation_ends)
        validation_gap = min(max(0.0, gap - cadence), inferred) if has_validation and gap > cadence * 2 else 0.0
        validation_gap = exact_seconds or validation_gap
        remaining = max(0.0, gap - validation_gap)
        if remaining > 15 * 60:
            active_part = min(cadence, remaining)
            offline_part = remaining - active_part
            active += active_part
            offline += offline_part
            if active_part: segments.append({"kind": "active", "seconds": active_part})
            if offline_part: segments.append({"kind": "offline", "seconds": offline_part})
        else:
            active += remaining
            if remaining: segments.append({"kind": "active", "seconds": remaining})
        validation += validation_gap
        if validation_gap: segments.append({"kind": "validation", "seconds": validation_gap})
    return {"start": events[0]["wall_time"] * 1000, "end": events[-1]["wall_time"] * 1000, "activeSeconds": active, "validationSeconds": validation, "offlineSeconds": offline, "segments": segments}


def current_response():
    series = {tag: display_series(values, MAX_DISPLAY_POINTS if tag in WINDOW_TAGS else 240)
              for tag, values in full_series.items()}
    windows = {tag: list(values) for tag, values in tail_series.items() if values}
    # Long tails are needed for the loss-window controls. Other scalar cards
    # already have bounded display series and do not need a second copy of
    # thousands of recent points in every response.
    tails = {tag: list(card_tail_series[tag]) for tag in WINDOW_TAGS if tag in card_tail_series}
    return {"series": series, "windows": windows, "tails": tails, "prCurves": pr_curves,
            "images": images, "confusionMatrix": condensed_confusion_matrix(), "activity": activity_summary()}


def condensed_confusion_matrix():
    """Keep the most informative classes and aggregate the long tail.

    The event file retains the full matrix, but the browser receives at most
    24 active classes plus Other and Background. Ranking uses both actual and
    predicted traffic so rare-but-noisy classes are not silently discarded.
    """
    if not confusion_matrix or not confusion_matrix.get("values"):
        return confusion_matrix
    size = int(confusion_matrix["size"])
    labels = confusion_matrix.get("labels") or [f"class {i}" for i in range(size)]
    values = confusion_matrix["values"]
    if size <= MAX_CONFUSION_CLASSES + 2:
        return confusion_matrix
    background = size - 1
    totals = []
    for index in range(background):
        row = sum(values[index * size:(index + 1) * size])
        column = sum(values[index::size])
        totals.append((row + column, index))
    keep = [index for _, index in sorted(totals, reverse=True)[:MAX_CONFUSION_CLASSES]]
    keep_set = set(keep)
    other = [index for index in range(background) if index not in keep_set]
    groups = keep + [other, [background]]
    condensed = []
    for row_group in groups:
        for col_group in groups:
            condensed.append(sum(values[row * size + col] for row in row_group for col in col_group))
    condensed_labels = [labels[index] for index in keep] + ["other", labels[background] if background < len(labels) else "background"]
    return {"step": confusion_matrix.get("step"), "values": condensed, "size": len(groups),
            "labels": condensed_labels, "condensed": True, "hiddenClasses": len(other)}


cached_response = load_cache()
for _ in sys.stdin:
    try:
        if cached_response is not None:
            response = cached_response
            cached_response = None
            print(json.dumps(response, separators=(",", ":")), flush=True)
            # Let the caller render the cached snapshot immediately, then
            # catch the reader up before the next poll arrives.
            update_series()
            response = current_response()
            save_cache(response)
            continue
        else:
            update_series()
            response = current_response()
            save_cache(response)
        print(json.dumps(response, separators=(",", ":")), flush=True)
    except Exception as error:  # Keep the helper alive for the next poll.
        print(json.dumps({"error": str(error)}), flush=True)
