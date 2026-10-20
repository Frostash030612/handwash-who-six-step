"""Boundary-guided decoding of per-sample class probabilities (NumPy only).

Two decoders share one boundary-peak rule:

* :func:`decode_boundaries` is the original segment-mean decoder. Its behaviour
  is unchanged so existing checkpoints and scripts keep their results.
* :func:`decode_boundary_blend` additionally keeps sustained, confident raw
  evidence as its own segment and blends raw with segment-mean probabilities,
  so a segment mean cannot overwrite e.g. a clearly recognised opening "other".

Neither decoder uses an expected WHO step order. Decoder settings travel with
checkpoints as plain dictionaries understood by :func:`apply_decoder`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

DECODER_METHODS = ("raw", "boundary_segment", "boundary_blend")


def _validated(
    class_probs: ArrayLike, boundary_probs: ArrayLike, *, threshold: float, min_gap: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    probs = np.asarray(class_probs, dtype=np.float32)
    edges = np.asarray(boundary_probs, dtype=np.float32).reshape(-1)
    if probs.ndim != 2 or len(probs) != len(edges):
        raise ValueError("Expected class probabilities [T, C] and boundary scores [T]")
    if not np.isfinite(probs).all() or not np.isfinite(edges).all():
        raise ValueError("Probabilities must be finite")
    if not 0.0 <= threshold <= 1.0 or min_gap < 1:
        raise ValueError("threshold must be in [0, 1] and min_gap must be positive")
    return probs, edges


def boundary_peaks(boundary_probs: ArrayLike, *, threshold: float, min_gap: int) -> list[int]:
    """Local maxima at or above ``threshold``, strongest first, ``min_gap`` samples apart.

    A peak at index ``i`` means the action changes between samples ``i-1`` and ``i``.
    """
    edges = np.asarray(boundary_probs, dtype=np.float32).reshape(-1)
    left = np.r_[-np.inf, edges[:-1]]
    right = np.r_[edges[1:], -np.inf]
    peaks = np.flatnonzero((edges >= threshold) & (edges >= left) & (edges > right))
    chosen: list[int] = []
    for position in sorted(peaks.tolist(), key=lambda index: float(edges[index]), reverse=True):
        if position == 0 or position >= len(edges):
            continue
        if all(abs(position - other) >= min_gap for other in chosen):
            chosen.append(position)
    return sorted(chosen)


def decode_boundaries(
    class_probs: ArrayLike,
    boundary_probs: ArrayLike,
    *,
    threshold: float,
    min_gap: int,
) -> NDArray[np.float32]:
    """Use confident boundary peaks to average class probabilities per segment.

    ``min_gap`` is measured in sampled frames (at the project default of 5 fps,
    two frames represent about 0.4 s). No class order is imposed.
    """
    probs, edges = _validated(class_probs, boundary_probs, threshold=threshold, min_gap=min_gap)
    if len(probs) < 2:
        return probs.copy()
    boundaries = boundary_peaks(edges, threshold=threshold, min_gap=min_gap)
    result = probs.copy()
    for begin, end in zip([0, *boundaries], [*boundaries, len(probs)], strict=True):
        if end > begin:
            result[begin:end] = probs[begin:end].mean(axis=0, keepdims=True)
    return result


def confident_runs(
    class_probs: ArrayLike, *, min_frames: int, min_confidence: float
) -> list[tuple[int, int]]:
    """Half-open runs whose raw top class stays the same at ``>= min_confidence``.

    Every sample of a run must be confident on its own; a run shorter than
    ``min_frames`` is not returned, so one or two confident outliers are not
    protected from segment averaging.
    """
    probs = np.asarray(class_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("Expected class probabilities [T, C]")
    if min_frames < 1 or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_frames must be positive and min_confidence must be in [0, 1]")
    labels = probs.argmax(axis=1)
    confident = probs.max(axis=1) >= min_confidence
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(len(probs) + 1):
        inside = index < len(probs) and bool(confident[index])
        if start is not None and inside and labels[index] == labels[start]:
            continue
        if start is not None and index - start >= min_frames:
            runs.append((start, index))
        start = index if inside else None
    return runs


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Smallest whole number of sampled frames covering ``seconds`` (at least one)."""
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("seconds and fps must be finite and positive")
    return max(1, math.ceil(seconds * fps - 1e-9))


def decode_boundary_blend(
    class_probs: ArrayLike,
    boundary_probs: ArrayLike,
    *,
    threshold: float,
    min_gap: int,
    raw_weight: float,
    protect_min_frames: int,
    protect_confidence: float,
) -> NDArray[np.float32]:
    """Boundary segment averaging that cannot overwrite sustained confident evidence.

    1. Boundary peaks (same rule as :func:`decode_boundaries`) cut the sequence.
    2. Each confident raw run (see :func:`confident_runs`) is cut out as its own
       segment, so its evidence is neither diluted by nor spread onto neighbours.
    3. Output = ``raw_weight * raw + (1 - raw_weight) * segment mean``;
       ``raw_weight=1`` returns the raw probabilities (no segment averaging).
    """
    probs, edges = _validated(class_probs, boundary_probs, threshold=threshold, min_gap=min_gap)
    if not 0.0 <= raw_weight <= 1.0:
        raise ValueError("raw_weight must be in [0, 1]")
    runs = confident_runs(probs, min_frames=protect_min_frames, min_confidence=protect_confidence)
    if raw_weight == 1.0 or len(probs) < 2:
        return probs.copy()
    cuts = set(boundary_peaks(edges, threshold=threshold, min_gap=min_gap))
    for begin, end in runs:
        cuts.update((begin, end))
    ordered = sorted(cut for cut in cuts if 0 < cut < len(probs))
    means = np.empty_like(probs)
    for begin, end in zip([0, *ordered], [*ordered, len(probs)], strict=True):
        means[begin:end] = probs[begin:end].mean(axis=0, keepdims=True)
    return (raw_weight * probs + (1.0 - raw_weight) * means).astype(np.float32)


def apply_decoder(
    class_probs: ArrayLike, boundary_probs: ArrayLike | None, decoder: Mapping[str, Any]
) -> NDArray[np.float32]:
    """Decode with a checkpoint's decoder settings.

    ``{"method": "raw"}`` returns the probabilities unchanged. Older
    checkpoints that store only ``threshold``/``min_gap`` are read as the
    original ``boundary_segment`` decoder.
    """
    method = decoder.get("method") or ("boundary_segment" if "threshold" in decoder else "raw")
    if method == "raw":
        return np.asarray(class_probs, dtype=np.float32).copy()
    if boundary_probs is None:
        raise ValueError(f"Decoder {method!r} needs boundary probabilities")
    if method == "boundary_segment":
        return decode_boundaries(
            class_probs, boundary_probs,
            threshold=float(decoder["threshold"]), min_gap=int(decoder["min_gap"]),
        )
    if method == "boundary_blend":
        return decode_boundary_blend(
            class_probs, boundary_probs,
            threshold=float(decoder["threshold"]), min_gap=int(decoder["min_gap"]),
            raw_weight=float(decoder["raw_weight"]),
            protect_min_frames=seconds_to_frames(
                float(decoder["protect_min_seconds"]), float(decoder["sample_fps"])
            ),
            protect_confidence=float(decoder["protect_confidence"]),
        )
    raise ValueError(f"Unknown decoder method {method!r}; expected one of {DECODER_METHODS}")
