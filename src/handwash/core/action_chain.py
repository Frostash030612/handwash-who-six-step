"""ATEC: Adaptive Temporal Evidence Confirmation for action chains."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from handwash.errors import ProtocolError


def atec_action_chain(
    labels: Sequence[int], *, fps: float, smoothing_window_seconds: float,
    background_class: int, class_probabilities: Sequence[Sequence[float]] | None = None,
    winner_confidences: Sequence[float] | None = None,
    outlier_z: float = -2.0, minimum_comparable_segments: int = 4,
) -> dict:
    """Confirm action runs using within-video duration and confidence evidence.

    Durations are compared on a log scale using a robust median and MAD. A run
    is unconfirmed only when it is a strong short-duration outlier *and*, when
    probabilities are available, its class confidence is below the video's
    median action-run confidence. No absolute action duration or expected class
    order is used. With too few comparable runs, the algorithm abstains from
    duration-outlier rejection; this limitation is recorded in the result.

    ``winner_confidences`` is for archived predictions that retained only the
    probability of each original winning label. Full ``class_probabilities``
    is preferred and never reconstructed from winning confidences.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ProtocolError('action_chain.fps must be finite and positive')
    if not math.isfinite(smoothing_window_seconds) or smoothing_window_seconds < 0:
        raise ProtocolError('action_chain.smoothing_window_seconds must be finite and nonnegative')
    if not math.isfinite(outlier_z) or outlier_z >= 0:
        raise ProtocolError('action_chain.outlier_z must be finite and negative')
    if minimum_comparable_segments < 3:
        raise ProtocolError('action_chain.minimum_comparable_segments must be at least 3')
    values = np.asarray(labels, dtype=np.int64)
    n = len(values)
    if class_probabilities is not None and winner_confidences is not None:
        raise ProtocolError('provide full class probabilities or winner confidences, not both')
    probabilities = None
    confidence = None
    if class_probabilities is not None:
        probabilities = np.asarray(class_probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] != n or probabilities.shape[1] < 2:
            raise ProtocolError('class_probabilities must have shape [T,C]')
        if np.any(values < 0) or np.any(values >= probabilities.shape[1]):
            raise ProtocolError('labels must name classes in class_probabilities')
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ProtocolError('class_probabilities must be finite and in [0,1]')
    elif winner_confidences is not None:
        confidence = np.asarray(winner_confidences, dtype=np.float64)
        if confidence.shape != (n,) or not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1):
            raise ProtocolError('winner_confidences must be a finite [T] vector in [0,1]')

    window = max(1, int(round(smoothing_window_seconds * fps)))
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = []
    for i, label in enumerate(values):
        neighbours = values[max(0, i-half):min(n, i+half+1)]
        counts = Counter(neighbours)
        most = max(counts.values())
        tied = {key for key, count in counts.items() if count == most}
        winner = int(label) if label in tied else next(int(x) for x in neighbours if x in tied)
        smoothed.append(winner)

    segments = []
    start = 0
    for end in range(1, n+1):
        if end == n or smoothed[end] != smoothed[start]:
            label = smoothed[start]
            segment = {'class': label, 'start_index': start, 'end_index_exclusive': end,
                       'sampled_frames': end-start, 'duration_s': (end-start)/fps,
                       'status': 'background' if label == background_class else 'confirmed'}
            if label != background_class:
                if probabilities is not None:
                    segment['mean_class_probability'] = float(probabilities[start:end, label].mean())
                elif confidence is not None:
                    # A smoothed label may differ from the original winner.
                    matching = values[start:end] == label
                    segment['mean_class_probability'] = (
                        float(confidence[start:end][matching].mean()) if np.any(matching) else None
                    )
            segments.append(segment)
            start = end

    action_segments = [s for s in segments if s['class'] != background_class]
    comparable = len(action_segments) >= minimum_comparable_segments
    if comparable:
        log_durations = np.log([s['duration_s'] for s in action_segments])
        median = float(np.median(log_durations))
        mad = float(np.median(np.abs(log_durations - median)))
        # A scale floor avoids flagging harmless small differences when all
        # actions have nearly identical durations. It is dimensionless.
        robust_scale = 1.4826 * max(mad, 0.2)
        confidence_values = [s.get('mean_class_probability') for s in action_segments]
        available_confidence = [x for x in confidence_values if x is not None]
        median_confidence = float(np.median(available_confidence)) if available_confidence else None
        for segment in action_segments:
            z = (math.log(segment['duration_s']) - median) / robust_scale
            segment['duration_outlier_z'] = z
            segment['relative_duration'] = segment['duration_s'] / math.exp(median)
            below_typical_confidence = (median_confidence is None or
                                        segment.get('mean_class_probability') is None or
                                        segment['mean_class_probability'] < median_confidence)
            if z < outlier_z and below_typical_confidence:
                segment['status'] = 'unconfirmed_outlier'
    else:
        median_confidence = None

    order = []
    for segment in segments:
        if segment['status'] == 'confirmed':
            label = segment['class']
            if not order or label != order[-1]:
                order.append(label)
    return {'order': order, 'segments': segments, 'smoothed_labels': smoothed,
            'smoothing_window_frames': window, 'comparable_segments': len(action_segments),
            'duration_comparison_available': comparable, 'median_action_confidence': median_confidence,
            'method': 'ATEC/1'}
