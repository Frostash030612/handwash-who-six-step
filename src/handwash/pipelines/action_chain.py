"""Confirm sustained action proposals without changing frame predictions."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from handwash.errors import ProtocolError


def sustained_action_chain(
    labels: Sequence[int], *, fps: float, min_action_seconds: float,
    smoothing_window_seconds: float, background_class: int,
) -> dict:
    """Build an offline chain with explicit rejected short proposals.

    Majority smoothing is centred (not a streaming decoder). Only a single
    continuous smoothed run meeting the duration threshold is confirmed.
    Short runs are not relabelled as a neighbouring action, and their time
    never increases that neighbour's duration. No expected step order is used.
    Durations use sampled-frame coverage N/fps, not last-first timestamps.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ProtocolError('action_chain.fps must be finite and positive')
    if not math.isfinite(min_action_seconds) or min_action_seconds <= 0:
        raise ProtocolError('action_chain.min_action_seconds must be finite and positive')
    if not math.isfinite(smoothing_window_seconds) or smoothing_window_seconds < 0:
        raise ProtocolError('action_chain.smoothing_window_seconds must be finite and nonnegative')
    values = [int(label) for label in labels]
    window = max(1, int(round(smoothing_window_seconds * fps)))
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = []
    for i, label in enumerate(values):
        neighbours = values[max(0, i-half):min(len(values), i+half+1)]
        counts = Counter(neighbours)
        most = max(counts.values())
        tied = {key for key, count in counts.items() if count == most}
        # Preserve the current observation on ties; never prefer canonical order.
        winner = label if label in tied else next(x for x in neighbours if x in tied)
        smoothed.append(winner)
    minimum_frames = max(1, math.ceil(min_action_seconds * fps - 1e-9))
    segments = []
    start = 0
    for end in range(1, len(smoothed)+1):
        if end == len(smoothed) or smoothed[end] != smoothed[start]:
            label = smoothed[start]
            status = ('background' if label == background_class else
                      'confirmed' if end-start >= minimum_frames else 'unconfirmed_short')
            segments.append({'class':label, 'start_index':start, 'end_index_exclusive':end,
                             'sampled_frames':end-start, 'duration_s':(end-start)/fps,
                             'status':status})
            start = end
    order = []
    for segment in segments:
        if segment['status'] == 'confirmed':
            label = segment['class']
            if not order or label != order[-1]:
                order.append(label)
    return {'order':order, 'segments':segments, 'smoothed_labels':smoothed,
            'minimum_frames':minimum_frames, 'smoothing_window_frames':window}
