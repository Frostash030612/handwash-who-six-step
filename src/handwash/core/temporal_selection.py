"""Validation-only selection score and the rule for replacing the main temporal model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: 40% Macro-F1 + 30% continuous-video action-chain match rate + 30% boundary F1.
SELECTION_WEIGHTS: Mapping[str, float] = {
    "macro_f1": 0.4,
    "continuous_chain_match_rate": 0.3,
    "boundary_f1": 0.3,
}
#: A replacement may not be worse than the retrained GRU on any of these (seed means)...
NOT_LOWER: tuple[str, ...] = ("accuracy", "macro_f1", "continuous_chain_match_rate", "boundary_f1")
#: ...must improve at least one of them, and may not increase boundary timing error.
NOT_HIGHER: tuple[str, ...] = ("boundary_mean_abs_error_s",)


def composite_score(metrics: Mapping[str, float], weights: Mapping[str, float] = SELECTION_WEIGHTS) -> float:
    """Weighted selection score; weights must sum to one."""
    if not weights or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("selection weights must sum to 1")
    return float(sum(weight * float(metrics[key]) for key, weight in weights.items()))


def _worst_if_missing(value: float | None) -> float:
    # No matched boundary means no timing evidence; never let that look like zero error.
    return math.inf if value is None else float(value)


def replacement_decision(
    candidate: Mapping[str, float | None],
    baseline: Mapping[str, float | None],
    *,
    not_lower: Sequence[str] = NOT_LOWER,
    not_higher: Sequence[str] = NOT_HIGHER,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Recommend replacement only if nothing gets worse and something gets better."""
    failures: list[dict[str, Any]] = []
    improved: list[str] = []
    for key in not_lower:
        new, old = float(candidate[key]), float(baseline[key])  # type: ignore[arg-type]
        if new < old - tolerance:
            failures.append({"metric": key, "candidate": new, "baseline": old, "rule": "not_lower"})
        elif new > old + tolerance:
            improved.append(key)
    for key in not_higher:
        new, old = _worst_if_missing(candidate.get(key)), _worst_if_missing(baseline.get(key))
        if new > old + tolerance:
            failures.append({"metric": key, "candidate": candidate.get(key),
                             "baseline": baseline.get(key), "rule": "not_higher"})
    if not improved:
        failures.append({"metric": "|".join(not_lower), "rule": "at_least_one_improves"})
    return {"recommend_replacement": not failures, "improved": improved, "failures": failures}
