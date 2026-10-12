"""The verifier and fold generator must agree on held-out video membership."""

from scripts.build_oof_boundary_predictions import _fold as training_fold, _inner
from scripts.train_short_segment_verifier import _fold as verifier_fold


def test_every_video_has_one_matching_temporal_head_holdout() -> None:
    ids = [f"DataSet{i % 4 + 1}/video-{i:03d}" for i in range(100)]
    for video_id in ids:
        assert training_fold(video_id) == verifier_fold(video_id)
        assert sum(training_fold(video_id) == fold for fold in range(5)) == 1
        assert isinstance(_inner(video_id), bool)
