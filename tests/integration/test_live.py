"""实时 YOLO 分类时间轴和模型输入的契约检查。"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from handwash.core.config import load_config  # noqa: E402
from handwash.core.labels import Step, get_label_space  # noqa: E402
from handwash.errors import DataError  # noqa: E402
from handwash.pipelines.live import LiveClassifier, LiveSession  # noqa: E402

pytestmark = pytest.mark.integration


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


class _CountingStream:
    """假时序状态：第几帧就输出第几步，用来观察状态是否被重置。"""

    def __init__(self, size: int) -> None:
        self.size = size
        self.seen = 0

    def predict(self, frame: np.ndarray) -> np.ndarray:
        self.seen += 1
        probs = np.zeros(self.size, dtype=np.float32)
        probs[min(self.seen, self.size - 1)] = 1.0
        return probs


class _StreamingClassifier:
    model_name = "fake-temporal"

    def __init__(self, rc) -> None:
        self.rc = rc
        self.space = get_label_space(rc.label_space)
        self.streams: list[_CountingStream] = []

    def predict(self, frame: np.ndarray) -> np.ndarray:
        raise AssertionError("带时序状态的分类器不应走逐帧 predict")

    def new_stream(self) -> _CountingStream:
        self.streams.append(_CountingStream(len(self.space)))
        return self.streams[-1]


class TemporalStreamSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        # 故意保留较大的平滑窗口：时序分类器应绕开概率平均，否则会增加切换延迟。
        self.rc = load_config(
            ["configs/config.yaml", "configs/experiments/live_yolo_frame.yaml"],
            overrides={"runtime": {"device": "cpu"}, "assess": {"smooth_window": 9, "min_confidence": 0.0}},
        )
        self.classifier = _StreamingClassifier(self.rc)
        self.space = self.classifier.space
        self.frame = np.zeros((32, 48, 3), dtype=np.uint8)

    def test_stream_state_is_per_session_and_not_averaged(self) -> None:
        first = LiveSession(self.classifier, session_id="a")
        first.add_frame(self.frame, capture_time_s=0.0)
        update = first.add_frame(self.frame, capture_time_s=0.2)
        self.assertEqual(update["label"], self.space.to_label(2).value)
        second = LiveSession(self.classifier, session_id="b")
        update = second.add_frame(self.frame, capture_time_s=0.0)
        self.assertEqual(update["label"], self.space.to_label(1).value)
        self.assertEqual([stream.seen for stream in self.classifier.streams], [2, 1])

    def test_capture_gap_restarts_stream_state(self) -> None:
        session = LiveSession(self.classifier, session_id="gap")
        session.add_frame(self.frame, capture_time_s=0.0)
        session.add_frame(self.frame, capture_time_s=0.2)
        update = session.add_frame(self.frame, capture_time_s=5.0)
        self.assertEqual(update["label"], self.space.to_label(1).value)
        self.assertEqual(len(self.classifier.streams), 2)


def test_temporal_head_roundtrip_checks_exp_fingerprint_and_fps(tmp_path) -> None:
    from handwash.errors import ModelError
    from handwash.pipelines.exp_demo import (
        ExpTemporalHead,
        file_sha256,
        load_temporal_head,
        save_temporal_head,
    )

    exp = tmp_path / "exp.pt"
    exp.write_bytes(b"weights-v1")
    head = ExpTemporalHead(feature_dim=16, hidden=8)
    head.feature_mean.fill_(0.5)
    path = save_temporal_head(
        tmp_path / "head.pt", head, exp_sha256=file_sha256(exp), fps=5.0, metrics={"best_epoch": 3}
    )

    loaded = load_temporal_head(path, exp_checkpoint=exp, fps=5.0)
    features = torch.randn(1, 4, 16)
    with torch.no_grad():
        expected, _ = head.eval()(features)
        actual, _ = loaded(features)
    assert torch.allclose(expected, actual)
    # 逐帧流式推理与整段推理结果一致（单向 GRU）
    with torch.no_grad():
        step_state = None
        outputs = []
        for t in range(4):
            logits, step_state = loaded(features[:, t : t + 1], step_state)
            outputs.append(logits)
    assert torch.allclose(torch.cat(outputs, dim=1), actual, atol=1e-6)

    with pytest.raises(ModelError):
        load_temporal_head(path, exp_checkpoint=exp, fps=10.0)
    exp.write_bytes(b"weights-v2")
    with pytest.raises(ModelError):
        load_temporal_head(path, exp_checkpoint=exp, fps=5.0)


if __name__ == "__main__":
    unittest.main()
