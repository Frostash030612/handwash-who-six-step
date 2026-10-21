"""Apply the archived DataSet1–4 models to one user supplied complete video."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from handwash.core.labels import STEP_ORDER, Step
from handwash.core.protocol import build_report
from handwash.core.sequence_review import sequence_edit_distance
from handwash.errors import DataError, ModelError
from handwash.paths import MODELS_DIRNAME, ensure_dir, project_path
from handwash.pipelines.video_smoothing import predict_labels

MODEL_ROOT = project_path(MODELS_DIRNAME, "pskuss_ds1234")
EXPECTED_YOLO_NAMES = (
    "0_other_or_nonstandard", "1_palm_to_palm", "2_palm_over_dorsum",
    "3_palm_to_palm_interlaced", "4_backs_of_fingers", "5_thumb_rubbing",
    "6_fingertips_to_palm",
)
EXPECTED_VIDEOMAE_NAMES = (
    "0_other", "1_palm_to_palm", "2_palm_over_dorsum", "3_fingers_interlaced",
    "4_backs_of_fingers", "5_rotational_thumbs", "6_rotational_fingertips",
)
LABELS = (Step.OTHER, *STEP_ORDER)


def _decode(path: Path, fps: float) -> tuple[list[np.ndarray], list[int], float, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise DataError("无法打开视频文件")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, round(source_fps / fps))
    frames: list[np.ndarray] = []
    indices: list[int] = []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % stride == 0:
                height, width = image.shape[:2]
                if max(height, width) > 640:
                    scale = 640 / max(height, width)
                    image = cv2.resize(image, (round(width * scale), round(height * scale)))
                frames.append(image)
                indices.append(index)
                if len(frames) > 1200:
                    raise DataError("单段视频最长约 4 分钟；请先裁短再上传")
            index += 1
    finally:
        capture.release()
    if len(frames) < 16:
        raise DataError("视频太短：至少需要约 3.2 秒的可解码画面")
    return frames, indices, source_fps, index / source_fps


def _window_centres(frame_count: int, window: int, stride: int) -> tuple[list[int], np.ndarray]:
    starts = list(range(0, frame_count - window + 1, stride))
    tail = frame_count - window
    if starts[-1] != tail:
        starts.append(tail)
    return starts, np.asarray([start + window // 2 for start in starts], dtype=np.int64)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_rows(path: Path) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    times: list[float] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                code = int(row["movement_code"])
                washing = row.get("is_washing", "1").strip() == "1"
                labels.append(0 if not washing or code in (0, 7) else code if 1 <= code <= 6 else -1)
                times.append(float(row["frame_time"]) / 1000)
            except (KeyError, TypeError, ValueError):
                labels.append(-1)
                times.append(float("nan"))
    return labels, times


def _known_reference(upload: Path, indices: list[int], positions: np.ndarray) -> dict | None:
    """Use original two-annotator labels only for an exact bundled video match."""
    known_root = project_path("data", "external", "pskus_demo_validation")
    if not known_root.is_dir():
        return None
    fingerprint = _file_sha256(upload)
    source: Path | None = None
    for candidate in sorted(known_root.rglob("*.mp4")):
        if candidate.stat().st_size == upload.stat().st_size and _file_sha256(candidate) == fingerprint:
            source = candidate
            break
    if source is None:
        return None
    original_root = project_path("data", "raw", "pskuss", "extracted")
    for dataset, pair in (("DataSet1", ("Annotator2", "Annotator3")),
                          ("DataSet4", ("Annotator1", "Annotator2"))):
        root = original_root / dataset
        if not (root / "Videos" / source.name).is_file():
            continue
        left_path = root / "Annotations" / pair[0] / f"{source.stem}.csv"
        right_path = root / "Annotations" / pair[1] / f"{source.stem}.csv"
        if not (left_path.is_file() and right_path.is_file()):
            continue
        left, left_times = _annotation_rows(left_path)
        right, right_times = _annotation_rows(right_path)
        targets: list[int] = []
        for position in positions:
            frame_id = indices[int(position)]
            if (frame_id < min(len(left), len(right)) and left[frame_id] >= 0
                    and left[frame_id] == right[frame_id]
                    and abs(left_times[frame_id] - right_times[frame_id]) <= 0.001):
                targets.append(left[frame_id])
            else:
                targets.append(-1)
        sequence: list[int] = []
        for value in targets:
            if value > 0 and (not sequence or sequence[-1] != value):
                sequence.append(value)
        return {
            "source": f"{dataset}/{source.stem}",
            "annotators": list(pair), "targets": targets,
            "valid_centres": sum(value >= 0 for value in targets),
            "total_centres": len(targets), "sequence": sequence,
        }
    return None


def _result(name: str, probs: np.ndarray, setting: dict, positions: np.ndarray,
            indices: list[int], source_fps: float, duration: float, cfg,
            reference: dict | None = None) -> dict:
    # These settings were selected on the validation split, never on this upload.
    chosen = predict_labels(probs, setting)
    times = np.asarray([indices[int(pos)] / source_fps for pos in positions], dtype=float)
    boundaries = np.concatenate(([0.0], (times[:-1] + times[1:]) / 2, [duration]))
    # Restore a 5 fps axis to pass true durations to the project's existing rules.
    ticks = np.arange(max(1, round(duration * 5.0))) / 5.0
    centre_for_tick = np.searchsorted(boundaries[1:], ticks, side="right").clip(0, len(chosen) - 1)
    report = build_report(
        clip_id="local-upload", labels=[LABELS[int(chosen[i])] for i in centre_for_tick],
        confidences=[float(probs[i, int(chosen[i])]) for i in centre_for_tick],
        fps=5.0, cfg=cfg, model_name=name, apply_smoothing=False,
    )
    payload = report.to_dict()
    timeline = [
        {"label": part["step"], "step_no": part["step_no"],
         "start_s": part["start_s"], "end_s": part["end_s"],
         "confidence": part["mean_confidence"]}
        for part in payload["timeline"]
    ]
    result = {
        "model": name, "timeline": timeline,
        "steps_seen": [stat["step_no"] for stat in payload["statistics"] if stat["detected"]],
        "report": payload, "window_centres": len(chosen),
        "inference_note": (
            "Actions and completeness are model estimates; accuracy uses only two-annotator consensus centres."
            if reference is not None else
            "Actions and completeness are model estimates. Accuracy requires human labels for this video."
        ),
    }
    predicted_sequence: list[int] = []
    for part in timeline:
        step_no = int(part["step_no"])
        if step_no > 0 and (not predicted_sequence or predicted_sequence[-1] != step_no):
            predicted_sequence.append(step_no)
    target_sequence = list(reference["sequence"]) if reference is not None else list(range(1, 7))
    result["sequence_review"] = {
        "predicted": predicted_sequence,
        "reference_kind": "two_annotator" if reference is not None else "who_order",
        "reference": target_sequence,
        "edit_distance": sequence_edit_distance(predicted_sequence, target_sequence),
        "note": "Sequence differences are descriptive; no predicted action or rule warning is removed.",
    }
    if reference is not None and reference["valid_centres"]:
        valid = [(int(predicted), target) for predicted, target in zip(chosen, reference["targets"], strict=True)
                 if target >= 0]
        result["consensus_accuracy"] = round(sum(predicted == target for predicted, target in valid)
                                               / len(valid), 4)
        result["consensus_correct"] = sum(predicted == target for predicted, target in valid)
    return result


def compare_video(path: Path, cfg, progress: Callable[[str], None]) -> dict:
    import cv2
    import torch

    from handwash.pipelines.exp_demo import ExpFeatureExtractor, load_temporal_head

    settings = json.loads((MODEL_ROOT / "web_inference.json").read_text(encoding="utf-8"))
    yolo_path = MODEL_ROOT / "yolo/best.pt"
    head_path = MODEL_ROOT / "yolo_gru/head.pt"
    videomae_path = MODEL_ROOT / "videomae_center_stable"
    for required in (yolo_path, head_path, videomae_path / "model.safetensors"):
        if not required.is_file() or required.stat().st_size < 100_000:
            raise DataError(f"缺少模型权重或只拿到了 Git LFS 指针：{required}；请运行 git lfs pull")
    progress("读取视频并按 5 fps 采样")
    frames, indices, source_fps, duration = _decode(path, settings["sample_fps"])
    starts, positions = _window_centres(len(frames), settings["num_frames"], settings["window_stride"])
    reference = _known_reference(path, indices, positions)
    results: dict[str, dict] = {}

    progress("运行 YOLO 和 YOLO＋GRU")
    extractor = ExpFeatureExtractor(yolo_path, device="cpu",
                                    cache_dir=ensure_dir(cfg.resolve_out_dir() / "compare_runtime"))
    if tuple(extractor.model.names[i] for i in range(7)) != EXPECTED_YOLO_NAMES:
        raise ModelError("YOLO 类别顺序与这组三模型的训练标签不一致")
    feature_chunks, prob_chunks = [], []
    for offset in range(0, len(frames), 16):
        features, probabilities = extractor.forward_bgr(frames[offset:offset + 16])
        feature_chunks.append(features)
        prob_chunks.append(probabilities)
        if offset % 80 == 0 or offset + 16 >= len(frames):
            progress(f"YOLO 帧 {min(offset + 16, len(frames))}/{len(frames)}")
    features = np.concatenate(feature_chunks)
    yolo_probs = np.concatenate(prob_chunks)[positions]
    results["yolo"] = _result("YOLO", yolo_probs, settings["yolo"], positions,
                              indices, source_fps, duration, cfg.assess, reference)
    del extractor, feature_chunks, prob_chunks
    gc.collect()

    head = load_temporal_head(head_path, exp_checkpoint=yolo_path, fps=settings["sample_fps"])
    if head.hparams["feature_dim"] != features.shape[1]:
        raise ModelError("GRU 输入维度与 YOLO 特征不一致")
    with torch.inference_mode():
        gru_probs = head(torch.from_numpy(features.astype(np.float32))[None])[0][0]
        gru_probs = gru_probs.softmax(dim=-1).cpu().numpy()[positions]
    results["yolo_gru"] = _result("YOLO + GRU", gru_probs, settings["yolo_gru"], positions,
                                  indices, source_fps, duration, cfg.assess, reference)
    del head, features
    gc.collect()

    progress("运行 VideoMAE（本机首次加载可能较慢）")
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained(videomae_path, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(videomae_path, local_files_only=True).eval()
    names = tuple(model.config.id2label[i] for i in range(7))
    if names != EXPECTED_VIDEOMAE_NAMES:
        raise ModelError(f"VideoMAE 类别顺序与训练标签不一致：{names}")
    mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
    small = [cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
             for frame in frames]
    window_probs = []
    # Batch one limits memory on the 8 GB local machine.
    with torch.inference_mode():
        for number, start in enumerate(starts, 1):
            array = np.stack(small[start:start + 16])
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2)[None].float().div_(255)
            tensor = (tensor - mean) / std
            probability = model(pixel_values=tensor).logits.softmax(dim=-1)[0].cpu().numpy()
            window_probs.append(probability)
            if number % 20 == 0 or number == len(starts):
                progress(f"VideoMAE 窗口 {number}/{len(starts)}")
    results["videomae"] = _result("VideoMAE", np.stack(window_probs), settings["videomae"],
                                  positions, indices, source_fps, duration, cfg.assess, reference)
    return {"duration_s": round(duration, 2), "sampled_frames": len(frames),
            "aligned_centres": len(positions), "reference": reference, "models": results}
