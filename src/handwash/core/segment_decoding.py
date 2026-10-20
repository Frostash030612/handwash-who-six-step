"""Offline, order-free action segmentation from frozen network probabilities.

The exact semi-Markov dynamic program scores whole segments. A transition costs
``change_penalty + boundary_weight * min(-log(edge), boundary_cost_cap)``.
Thus an edge can lower a transition's cost but cannot reward making a spurious
cut. Internal short WHO segments have a soft duration cost; background and
video-censored first/last segments are exempt. Strong evidence can retain a
real short action. No reference labels or canonical class order are inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SegmentDecoderConfig:
    sample_fps: float = 5.0
    change_penalty: float = 3.0
    boundary_weight: float = 0.5
    boundary_cost_cap: float = 4.0
    short_action_penalty_per_second: float = 2.0
    preferred_action_seconds: float = 3.0
    background_class: int = 0

    def validate(self, classes: int) -> None:
        for name in (
            "sample_fps",
            "change_penalty",
            "boundary_weight",
            "boundary_cost_cap",
            "short_action_penalty_per_second",
            "preferred_action_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.sample_fps <= 0 or self.preferred_action_seconds <= 0:
            raise ValueError("sample_fps and preferred_action_seconds must be positive")
        if not 0 <= self.background_class < classes:
            raise ValueError("background_class must name an existing class")


@dataclass
class SegmentDecoding:
    labels: np.ndarray
    objective: float
    segments: list[dict]
    config: dict


def decode_action_segments(
    class_probs: ArrayLike,
    boundary_probs: ArrayLike | None,
    config: SegmentDecoderConfig,
) -> SegmentDecoding:
    """Return discrete segment decisions plus *raw* model evidence, not posteriors.

    All segment lengths and every class transition remain possible. Complexity
    is O(T² C + T C²); intended for complete, sampled offline videos.
    """
    probs = np.asarray(class_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("class_probs must have shape [T,C] with at least two classes")
    length, classes = probs.shape
    config.validate(classes)
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs.sum(1) <= 0):
        raise ValueError("class probabilities must be finite, nonnegative and nonempty per row")
    probs = probs / probs.sum(1, keepdims=True)
    edges = None if boundary_probs is None else np.asarray(boundary_probs, dtype=np.float64)
    if edges is not None and (
        edges.shape != (length,) or not np.isfinite(edges).all() or np.any(edges < 0) or np.any(edges > 1)
    ):
        raise ValueError("boundary probabilities must be a finite [T] array in [0,1]")
    if config.boundary_weight > 0 and edges is None:
        raise ValueError("positive boundary_weight requires boundary probabilities")
    if length == 0:
        return SegmentDecoding(np.empty(0, dtype=np.int64), 0.0, [], asdict(config))

    prefix = np.vstack((np.zeros(classes), np.cumsum(np.log(np.clip(probs, 1e-6, 1.0)), axis=0)))
    switch_cost = np.full(length, config.change_penalty, dtype=np.float64)
    if config.boundary_weight:
        switch_cost += config.boundary_weight * np.minimum(
            -np.log(np.clip(edges, 1e-6, 1.0)), config.boundary_cost_cap
        )
    switch_cost[0] = 0.0
    # previous[b,c] is the best prefix ending at b with a class other than c.
    previous = np.full((length + 1, classes), -np.inf)
    previous[0] = 0.0
    previous_class = np.full((length + 1, classes), -1, dtype=np.int64)
    starts = np.zeros((length + 1, classes), dtype=np.int64)
    best = np.zeros(classes)
    for end in range(1, length + 1):
        candidate = previous[:end] + prefix[end] - prefix[:end] - switch_cost[:end, None]
        duration = (end - np.arange(end)) / config.sample_fps
        short_cost = config.short_action_penalty_per_second * np.maximum(
            0.0, config.preferred_action_seconds - duration
        )
        # The first and last observed segments may be truncated by recording.
        short_cost[0] = 0.0
        if end == length:
            short_cost[:] = 0.0
        candidate -= short_cost[:, None]
        candidate[:, config.background_class] += short_cost
        starts[end] = candidate.argmax(axis=0)
        best = candidate[starts[end], np.arange(classes)]
        alternatives = np.broadcast_to(best[:, None], (classes, classes)).copy()
        np.fill_diagonal(alternatives, -np.inf)
        previous_class[end] = alternatives.argmax(axis=0)
        previous[end] = alternatives.max(axis=0)

    label = int(best.argmax())
    objective = float(best[label])
    labels = np.empty(length, dtype=np.int64)
    segments = []
    end = length
    raw_labels = probs.argmax(axis=1)
    while end:
        begin = int(starts[end, label])
        labels[begin:end] = label
        means = probs[begin:end].mean(axis=0)
        runner_up = float(np.delete(means, label).max())
        segments.append(
            {
                "class": label,
                "start_index": begin,
                "end_index_exclusive": end,
                "duration_s": (end - begin) / config.sample_fps,
                "mean_raw_class_probability": float(means[label]),
                "mean_raw_probability_margin": float(means[label] - runner_up),
                "raw_agreement_fraction": float(np.mean(raw_labels[begin:end] == label)),
                "short_observed_action": label != config.background_class
                and (end - begin) / config.sample_fps < config.preferred_action_seconds,
                "video_edge_censored": begin == 0 or end == length,
            }
        )
        label = int(previous_class[begin, label])
        end = begin
    return SegmentDecoding(labels, objective, segments[::-1], asdict(config))
