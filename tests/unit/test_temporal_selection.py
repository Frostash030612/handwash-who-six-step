"""Selection score and the conservative rule for replacing the main GRU."""

from __future__ import annotations

import pytest

from handwash.core.temporal_selection import composite_score, replacement_decision

BASELINE = {"accuracy": 0.95, "macro_f1": 0.94, "continuous_chain_match_rate": 1.0, "boundary_f1": 0.6,
            "boundary_mean_abs_error_s": 0.05}


def test_composite_uses_forty_thirty_thirty_weights() -> None:
    assert composite_score({"macro_f1": 0.9, "continuous_chain_match_rate": 0.5, "boundary_f1": 0.8}) == (
        pytest.approx(0.4 * 0.9 + 0.3 * 0.5 + 0.3 * 0.8))
    with pytest.raises(ValueError):
        composite_score({"macro_f1": 1.0}, {"macro_f1": 0.5})


def test_identical_results_are_not_a_reason_to_replace() -> None:
    decision = replacement_decision(BASELINE, BASELINE)

    assert not decision["recommend_replacement"]
    assert decision["failures"][-1]["rule"] == "at_least_one_improves"


def test_one_gain_and_no_loss_recommends_replacement() -> None:
    decision = replacement_decision({**BASELINE, "boundary_f1": 0.7}, BASELINE)

    assert decision == {"recommend_replacement": True, "improved": ["boundary_f1"], "failures": []}


def test_any_loss_or_later_boundaries_block_replacement() -> None:
    worse = replacement_decision({**BASELINE, "boundary_f1": 0.9, "macro_f1": 0.93}, BASELINE)
    later = replacement_decision({**BASELINE, "boundary_f1": 0.9, "boundary_mean_abs_error_s": 0.08}, BASELINE)
    unmatched = replacement_decision({**BASELINE, "boundary_f1": 0.9, "boundary_mean_abs_error_s": None}, BASELINE)

    assert [f["metric"] for f in worse["failures"]] == ["macro_f1"]
    assert [f["metric"] for f in later["failures"]] == ["boundary_mean_abs_error_s"]
    assert [f["metric"] for f in unmatched["failures"]] == ["boundary_mean_abs_error_s"]
