#!/usr/bin/env python3
"""Incremental, full-run TensorFlow event-file scalar reader for the local monitor.

The accumulator uses bounded reservoir sampling so memory stays
bounded as a run grows. Each response is then reduced to an extrema-preserving
series that spans the complete run, keeping the browser render bounded too.
"""

import json
import hashlib
import math
import os
import sys
import tempfile
from collections import deque

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


event_file = sys.argv[1]
MAX_DISPLAY_POINTS = 1_200
MAX_WINDOW_POINTS = 10_000
MAX_CARD_WINDOW_POINTS = 5_000
WINDOW_TAGS = {"avg-loss", "learning-rate"}
CACHE_VERSION = 1
CACHE_UPDATE_BYTES = 1_000_000
event_loader = EventFileLoader(event_file)
tail_series = {tag: deque(maxlen=MAX_WINDOW_POINTS) for tag in WINDOW_TAGS}
card_tail_series = {}
full_series = {}
previous_raw_step = -1
previous_normalized_step = -1
step_offset = 0
cache_key = hashlib.sha256(os.path.abspath(event_file).encode()).hexdigest()[:24]
cache_file = os.path.join(tempfile.gettempdir(), f"pixienn-event-cache-{cache_key}.json")
cache_file_size = 0


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
    for event in event_loader.Load():
        if event.step < previous_raw_step:
            step_offset += previous_normalized_step + 1 - (event.step + step_offset)
        normalized_step = event.step + step_offset
        previous_raw_step = event.step
        previous_normalized_step = normalized_step
        for summary in event.summary.value:
            value = scalar_value(summary)
            if value is None or not math.isfinite(value):
                continue
            point = {"step": normalized_step, "value": value}
            full_series.setdefault(summary.tag, []).append(point)
            card_tail_series.setdefault(summary.tag, deque(maxlen=MAX_CARD_WINDOW_POINTS)).append(point)
            if summary.tag in tail_series:
                tail_series[summary.tag].append(point)


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


def current_response():
    series = {tag: display_series(values) for tag, values in full_series.items()}
    windows = {tag: list(values) for tag, values in tail_series.items() if values}
    tails = {tag: list(values) for tag, values in card_tail_series.items() if values}
    return {"series": series, "windows": windows, "tails": tails}


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
