#!/usr/bin/env python3
"""Incremental, full-run TensorFlow event-file scalar reader for the local monitor.

The accumulator uses bounded reservoir sampling so memory stays
bounded as a run grows. Each response is then reduced to an extrema-preserving
series that spans the complete run, keeping the browser render bounded too.
"""

import json
import math
import sys
from collections import deque

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


event_file = sys.argv[1]
MAX_STORED_SCALARS = 10_000
MAX_DISPLAY_POINTS = 1_200
MAX_WINDOW_POINTS = 10_000
WINDOW_TAGS = {"avg-loss", "learning-rate"}
tail_loader = EventFileLoader(event_file)
tail_series = {tag: deque(maxlen=MAX_WINDOW_POINTS) for tag in WINDOW_TAGS}


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


def update_tail_series():
    for event in tail_loader.Load():
        for summary in event.summary.value:
            if summary.tag not in tail_series:
                continue
            value = scalar_value(summary)
            if value is not None and math.isfinite(value):
                tail_series[summary.tag].append({"step": event.step, "value": value})


def read_full_series():
    series = {}
    previous_raw_step = -1
    previous_normalized_step = -1
    step_offset = 0
    for event in EventFileLoader(event_file).Load():
        if event.step < previous_raw_step:
            step_offset += previous_normalized_step + 1 - (event.step + step_offset)
        normalized_step = event.step + step_offset
        previous_raw_step = event.step
        previous_normalized_step = normalized_step
        for summary in event.summary.value:
            value = scalar_value(summary)
            if value is None or not math.isfinite(value):
                continue
            series.setdefault(summary.tag, []).append({"step": normalized_step, "value": value})
    for values in series.values():
        values.sort(key=lambda item: item["step"])
    return series


for _ in sys.stdin:
    try:
        update_tail_series()
        series = {tag: display_series(values) for tag, values in read_full_series().items()}
        windows = {tag: list(values) for tag, values in tail_series.items() if values}
        print(json.dumps({"series": series, "windows": windows}, separators=(",", ":")), flush=True)
    except Exception as error:  # Keep the helper alive for the next poll.
        print(json.dumps({"error": str(error)}), flush=True)
