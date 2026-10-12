"""One-to-one temporal overlap scores for aligned WHO action segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionSegment:
    label: int
    start: int
    end: int


def action_segments(labels: np.ndarray, usable: np.ndarray) -> list[ActionSegment]:
    """Make WHO segments on the original centre axis; unknown/other break runs."""
    if len(labels) != len(usable):
        raise ValueError("labels and usable must be aligned")
    segments: list[ActionSegment] = []
    start: int | None = None
    current = 0
    for index in range(len(labels) + 1):
        value = int(labels[index]) if index < len(labels) and usable[index] else 0
        if value == current:
            continue
        if start is not None:
            segments.append(ActionSegment(current, start, index))
        start = index if value > 0 else None
        current = value
    return segments


def segment_counts(
    reference: list[ActionSegment], predicted: list[ActionSegment], *, overlap: float = 0.5
) -> tuple[int, int, int]:
    """One-to-one, same-class segment matching with centre-level temporal IoU."""
    if not 0 < overlap <= 1:
        raise ValueError("overlap must be in (0, 1]")
    used: set[int] = set()
    true_positive = 0
    for proposal in predicted:
        candidate = None
        best_iou = 0.0
        for index, actual in enumerate(reference):
            if index in used or proposal.label != actual.label:
                continue
            intersection = max(0, min(proposal.end, actual.end) - max(proposal.start, actual.start))
            union = max(proposal.end, actual.end) - min(proposal.start, actual.start)
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                candidate, best_iou = index, iou
        if candidate is not None and best_iou >= overlap:
            used.add(candidate)
            true_positive += 1
    return true_positive, len(predicted) - true_positive, len(reference) - true_positive
