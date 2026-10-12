"""Frozen full-video probability smoothing shared by evaluation and the local page."""

from __future__ import annotations

import numpy as np


def average_probs(probs: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return probs
    radius = width // 2
    return np.stack([probs[max(0, i - radius):min(len(probs), i + radius + 1)].mean(0)
                     for i in range(len(probs))])


def suppress_short_runs(labels: np.ndarray, averaged: np.ndarray, min_run: int) -> np.ndarray:
    labels = labels.copy()
    if len(labels) == 0 or min_run <= 1:
        return labels
    for _ in range(3):
        runs = []
        begin = 0
        for end in range(1, len(labels) + 1):
            if end == len(labels) or labels[end] != labels[begin]:
                runs.append((begin, end, int(labels[begin])))
                begin = end
        changed = False
        for i, (begin, end, label) in enumerate(runs):
            if end - begin >= min_run or len(runs) == 1:
                continue
            candidates = []
            if i:
                candidates.append(runs[i - 1][2])
            if i + 1 < len(runs):
                candidates.append(runs[i + 1][2])
            if candidates:
                replacement = max(candidates, key=lambda c: float(averaged[begin:end, c].mean()))
                if replacement != label:
                    labels[begin:end] = replacement
                    changed = True
        if not changed:
            break
    return labels


def predict_labels(probs: np.ndarray, setting: dict[str, object]) -> np.ndarray:
    width = int(setting["width"])
    min_run = int(setting["min_run"])
    averaged = average_probs(probs, width)
    if setting["method"] != "hysteresis":
        return suppress_short_runs(averaged.argmax(1).astype(np.int64), averaged, min_run)
    confirm_frames = int(setting["confirm_frames"])
    margin = float(setting["margin"])
    raw = averaged.argmax(1).astype(np.int64)
    labels = np.empty_like(raw)
    state = int(raw[0])
    labels[0] = state
    pending_label: int | None = None
    pending_start = 0
    for index in range(1, len(raw)):
        candidate = int(raw[index])
        if candidate == state:
            labels[index] = state
            pending_label = None
            continue
        if candidate != pending_label:
            pending_label = candidate
            pending_start = index
        labels[index] = state
        advantage = float(np.mean(averaged[pending_start:index + 1, candidate]
                                  - averaged[pending_start:index + 1, state]))
        if index - pending_start + 1 >= confirm_frames and advantage >= margin:
            labels[pending_start:index + 1] = candidate
            state = candidate
            pending_label = None
    return suppress_short_runs(labels, averaged, min_run)
