"""Boundary F1 and timing error against annotated transition intervals.

A labelled change is known only up to its transition interval: from the last
labelled sample of the previous action to the first labelled sample of the
next one (unlabelled transition samples in between widen it). A predicted
change may match it once, if it lies inside that interval widened by a
tolerance on both sides. Timing error is the distance outside the interval
(zero inside it), because the exact moment within the interval is unknown.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ReferenceBoundary:
    start_s: float
    end_s: float
    before: int
    after: int


@dataclass(frozen=True)
class PredictedBoundary:
    time_s: float
    before: int
    after: int
    index: int  # first sample carrying the new prediction


@dataclass(frozen=True)
class BoundaryMatch:
    reference: int
    predicted: int
    signed_error_s: float  # negative = early, positive = late, 0 = inside the interval


@dataclass
class BoundaryScore:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    signed_errors_s: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 1.0

    @property
    def recall(self) -> float:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 1.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 1.0

    @property
    def mean_abs_error_s(self) -> float | None:
        if not self.signed_errors_s:
            return None
        return float(np.mean(np.abs(self.signed_errors_s)))

    @property
    def mean_signed_error_s(self) -> float | None:
        return float(np.mean(self.signed_errors_s)) if self.signed_errors_s else None

    def add(self, other: BoundaryScore) -> None:
        self.true_positive += other.true_positive
        self.false_positive += other.false_positive
        self.false_negative += other.false_negative
        self.signed_errors_s.extend(other.signed_errors_s)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_abs_error_s": self.mean_abs_error_s,
            "mean_signed_error_s": self.mean_signed_error_s,
        }


def _aligned(labels: ArrayLike, times: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    stamps = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(values) != len(stamps):
        raise ValueError("labels and times must be aligned")
    if not np.isfinite(stamps).all() or (len(stamps) > 1 and np.any(np.diff(stamps) <= 0)):
        raise ValueError("times must be finite and strictly increasing")
    return values, stamps


def reference_boundaries(labels: ArrayLike, times: ArrayLike) -> list[ReferenceBoundary]:
    """Changes between consecutive labelled samples; negative labels are unlabelled."""
    values, stamps = _aligned(labels, times)
    valid = np.flatnonzero(values >= 0)
    return [
        ReferenceBoundary(float(stamps[previous]), float(stamps[current]),
                          int(values[previous]), int(values[current]))
        for previous, current in zip(valid[:-1], valid[1:], strict=True)
        if values[previous] != values[current]
    ]


def predicted_boundaries(labels: ArrayLike, times: ArrayLike) -> list[PredictedBoundary]:
    """Every change in a per-sample prediction, placed midway between the two samples."""
    values, stamps = _aligned(labels, times)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    return [
        PredictedBoundary(float((stamps[index - 1] + stamps[index]) / 2),
                          int(values[index - 1]), int(values[index]), int(index))
        for index in changes
    ]


def interval_offset_s(time_s: float, boundary: ReferenceBoundary) -> float:
    """Signed distance outside the annotated interval (0 inside, negative = early)."""
    if time_s < boundary.start_s:
        return time_s - boundary.start_s
    if time_s > boundary.end_s:
        return time_s - boundary.end_s
    return 0.0


def match_boundaries(
    reference: Sequence[ReferenceBoundary],
    predicted: Sequence[PredictedBoundary],
    *,
    tolerance_s: float,
) -> list[BoundaryMatch]:
    """Maximum one-to-one matching, then minimum total timing error (exact).

    Both lists are ordered in time and the cost is the distance to ordered,
    disjoint intervals, so crossing pairs can always be uncrossed without
    losing a match or adding error; an alignment DP is therefore exact.
    """
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    if any(b.time_s < a.time_s for a, b in zip(predicted[:-1], predicted[1:], strict=True)):
        raise ValueError("predicted boundaries must be ordered in time")
    if any(b.start_s < a.end_s for a, b in zip(reference[:-1], reference[1:], strict=True)):
        raise ValueError("reference boundaries must be ordered and non-overlapping")
    rows, columns = len(predicted), len(reference)
    # best[i][j] = (matches, -total error) for the first i predictions and j references.
    best = [[(0, 0.0)] * (columns + 1) for _ in range(rows + 1)]
    move = [[0] * (columns + 1) for _ in range(rows + 1)]  # 1 skip prediction, 2 skip reference, 3 match
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            options = [(best[i - 1][j], 1), (best[i][j - 1], 2)]
            offset = interval_offset_s(predicted[i - 1].time_s, reference[j - 1])
            if abs(offset) <= tolerance_s + 1e-9:
                matched, error = best[i - 1][j - 1]
                options.append(((matched + 1, error - abs(offset)), 3))
            best[i][j], move[i][j] = max(options, key=lambda option: option[0])
    matches: list[BoundaryMatch] = []
    i, j = rows, columns
    while i > 0 and j > 0:
        if move[i][j] == 3:
            offset = interval_offset_s(predicted[i - 1].time_s, reference[j - 1])
            matches.append(BoundaryMatch(j - 1, i - 1, offset))
            i, j = i - 1, j - 1
        elif move[i][j] == 1:
            i -= 1
        else:
            j -= 1
    return matches[::-1]


def score_boundaries(
    reference: Sequence[ReferenceBoundary],
    predicted: Sequence[PredictedBoundary],
    *,
    tolerance_s: float,
) -> tuple[BoundaryScore, list[BoundaryMatch]]:
    """One video's boundary counts; pool videos with :meth:`BoundaryScore.add`."""
    matches = match_boundaries(reference, predicted, tolerance_s=tolerance_s)
    score = BoundaryScore(
        true_positive=len(matches),
        false_positive=len(predicted) - len(matches),
        false_negative=len(reference) - len(matches),
        signed_errors_s=[match.signed_error_s for match in matches],
    )
    return score, matches
