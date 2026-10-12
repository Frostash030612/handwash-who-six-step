"""Human-readable sequence discrepancy without changing model predictions."""

from __future__ import annotations

from collections.abc import Sequence


def sequence_edit_distance(observed: Sequence[int], reference: Sequence[int]) -> int:
    """Minimum inserted, deleted, or substituted WHO actions in order.

    This is an explanation of the predicted action *sequence*, not a measure
    of action timing, correctness of individual frames, or clinical hygiene.
    """
    if any(not 1 <= step <= 6 for step in (*observed, *reference)):
        raise ValueError("Action sequences must contain only WHO steps 1–6")
    previous = list(range(len(reference) + 1))
    for row, step in enumerate(observed, start=1):
        current = [row]
        for column, expected in enumerate(reference, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (step != expected),
            ))
        previous = current
    return previous[-1]
