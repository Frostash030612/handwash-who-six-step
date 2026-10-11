"""本地演示核验的时间因果性与报告入口。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from handwash.core.config import load_config
from handwash.core.labels import STEP_ORDER, LabelSpace, Step
from handwash.pipelines.demo_audit import _causal_mean, _report_from_probabilities
from handwash.pipelines.live import LiveSession


def test_causal_mean_never_uses_a_future_frame() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    actual = _causal_mean(probabilities, window=2)
    np.testing.assert_allclose(actual, [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])


def test_report_from_probabilities_preserves_class_order() -> None:
    rc = load_config(
        ["configs/config.yaml"],
        overrides={"assess": {"min_segment_frames": 1, "min_segment_s": 0.2}},
    )
    classifier = SimpleNamespace(rc=rc, space=LabelSpace("test_demo", (Step.OTHER, *STEP_ORDER)))
    probabilities = np.full((10, 7), 0.01, dtype=np.float32)
    probabilities[:5, 1] = 0.94
    probabilities[5:, 0] = 0.94

    report = _report_from_probabilities(
        classifier, Path("example.mp4"), probabilities, model_name="unit-test"
    )

    assert report.clip_id == "example"
    assert report.statistics[0].detected
    assert all(not stat.detected for stat in report.statistics[1:])
    assert report.is_in_order


def test_offline_frame_report_matches_regular_live_session() -> None:
    rc = load_config(
        ["configs/config.yaml"],
        overrides={"assess": {"smooth_window": 3, "min_segment_frames": 1, "min_segment_s": 0.2}},
    )
    space = LabelSpace("test_demo", (Step.OTHER, *STEP_ORDER))
    probabilities = np.full((10, 7), 0.01, dtype=np.float32)
    probabilities[:5, 1] = 0.94
    probabilities[5:, 2] = 0.94

    class SequentialClassifier:
        model_name = "unit-test"

        def __init__(self) -> None:
            self.rc = rc
            self.space = space
            self.index = 0

        def predict(self, _frame: np.ndarray) -> np.ndarray:
            answer = probabilities[self.index]
            self.index += 1
            return answer

    classifier = SequentialClassifier()
    session = LiveSession(classifier, session_id="example")
    for index in range(len(probabilities)):
        session.add_frame(np.zeros((2, 2, 3), dtype=np.uint8), capture_time_s=index / rc.dataset.prep.fps)
    offline = _report_from_probabilities(
        classifier,
        Path("example.mp4"),
        _causal_mean(probabilities, window=rc.assess.smooth_window),
        model_name=classifier.model_name,
    )
    assert offline.to_dict() == session.report().to_dict()
