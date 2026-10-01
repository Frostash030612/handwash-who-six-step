"""实时 YOLO 分类时间轴和模型输入的契约检查。"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from handwash.core.config import load_config
from handwash.core.labels import Step, get_label_space
from handwash.errors import DataError
from handwash.pipelines.live import LiveClassifier, LiveSession


def _config():
    return load_config(
        ["configs/config.yaml", "configs/experiments/live_yolo_frame.yaml"],
        overrides={"runtime": {"device": "cpu"}, "assess": {"smooth_window": 1}},
    )


class _FixedModel(torch.nn.Module):
    arch_name = "yolo26n-cls"

    def __init__(self, classes: int, winner: int) -> None:
        super().__init__()
        self.classes = classes
        self.winner = winner

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((images.shape[0], images.shape[1], self.classes))
        output[..., self.winner] = 4.0
        return output


class LivePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rc = _config()
        self.space = get_label_space(self.rc.label_space)
        model = _FixedModel(len(self.space), self.space.to_index(Step.STEP_1))
        self.classifier = LiveClassifier(self.rc, model)
        self.frame = np.zeros((32, 48, 3), dtype=np.uint8)

    def test_frame_classifier_returns_matching_probabilities(self) -> None:
        probs = self.classifier.predict(self.frame)
        self.assertEqual(probs.shape, (len(self.space),))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(self.space.to_label(int(np.argmax(probs))), Step.STEP_1)

    def test_camera_gap_is_unknown_in_report_timeline(self) -> None:
        session = LiveSession(self.classifier, session_id="case")
        session.add_frame(self.frame, capture_time_s=10.0)
        session.add_frame(self.frame, capture_time_s=11.0)
        labels, confidences = session._uniform_observations()
        self.assertGreaterEqual(labels.count(Step.UNKNOWN), 3)
        self.assertIn(0.0, confidences)
        self.assertLess(session.report().total_wash_duration_s, 0.5)
        self.assertAlmostEqual(session.report().total_duration_s, len(labels) / 5.0)

    def test_rejects_backward_capture_time_without_adding_frame(self) -> None:
        session = LiveSession(self.classifier, session_id="case")
        session.add_frame(self.frame, capture_time_s=10.0)
        with self.assertRaises(DataError):
            session.add_frame(self.frame, capture_time_s=9.9)
        self.assertEqual(session.num_frames, 1)

    def test_rejects_non_rgb_frame(self) -> None:
        with self.assertRaises(DataError):
            self.classifier.predict(np.zeros((32, 48), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
