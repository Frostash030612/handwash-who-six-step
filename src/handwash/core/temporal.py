"""Shared temporal-window indexing used by training and inference."""

from __future__ import annotations


def window_starts(num_frames: int, window: int, stride: int) -> list[int]:
    """Return full-coverage sliding-window starts, including the final tail.

    A clip shorter than ``window`` is represented by one shorter window. Longer
    clips advance by the configured stride until the previous window covers the
    tail. The last window may be shorter; this avoids a nearly duplicate window
    shifted by one frame at clip boundaries.
    """
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if window < 1 or stride < 1 or stride > window:
        raise ValueError("window/stride must satisfy 1 <= stride <= window")
    if num_frames == 0:
        return []
    starts = [0]
    while starts[-1] + window < num_frames:
        starts.append(starts[-1] + stride)
    return starts


__all__ = ["window_starts"]
