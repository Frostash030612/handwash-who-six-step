"""A single extra or missing step should remain one explainable discrepancy."""

import pytest

from handwash.core.sequence_review import sequence_edit_distance


def test_one_extra_step_is_one_discrepancy() -> None:
    assert sequence_edit_distance([1, 2, 3, 5, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 2
    assert sequence_edit_distance([1, 2, 3, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 1


def test_missing_and_reordered_actions_are_not_marked_correct() -> None:
    assert sequence_edit_distance([1, 2, 3, 4, 6], [1, 2, 3, 4, 5, 6]) == 1
    assert sequence_edit_distance([1, 3, 2, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 2


def test_only_who_step_numbers_are_accepted() -> None:
    with pytest.raises(ValueError):
        sequence_edit_distance([0, 1], [1, 2])
