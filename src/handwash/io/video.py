"""视频解码与抽帧。

后端策略（CONTRIBUTING.md R16）
--------------------------------
优先 OpenCV（快，且能直接按目标 fps 抽样）；不可用时回退 imageio-ffmpeg。
两个后端都不可用时抛 ``BackendUnavailableError``，并给出安装命令。
**解码细节只存在这个文件里**，其余模块只调用 ``probe_video`` / ``iter_frames`` / ``extract_frames``。

时基约定（重要）
-----------------
时间戳使用原视频帧号/原生FPS；OpenCV 提供有效解码时间时采用其时间戳。
不能用原始帧号除以目标采样FPS，否则会把视频时长缩短约采样倍率。
当帧数元数据缺失但设置了 ``max_frames`` 时，先统计帧数再第二遍均匀抽样，
避免只保留开头或给时序模型输入不均匀的帧间隔。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from handwash.errors import BackendUnavailableError, VideoDecodeError
from handwash.io.utils import VIDEO_EXTENSIONS
from handwash.logging import get_logger
from handwash.paths import ensure_dir

__all__ = [
    "VideoMeta",
    "probe_video",
    "iter_frames",
    "extract_frames",
    "save_frame",
    "write_video",
    "available_backend",
]

log = get_logger(__name__)

Backend = Literal["opencv", "imageio"]


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """视频元信息（探测失败时字段可能为 None，但 path 一定有效）。"""

    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_s: float
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "frame_count": self.frame_count,
            "fps": round(self.fps, 4),
            "width": self.width,
            "height": self.height,
            "duration_s": round(self.duration_s, 4),
            "backend": self.backend,
        }


def available_backend() -> Backend:
    """返回当前可用的解码后端名。"""
    try:
        import cv2  # noqa: F401

        return "opencv"
    except ImportError:
        pass
    try:
        import imageio.v3 as iio  # noqa: F401

        return "imageio"
    except ImportError as exc:  # pragma: no cover - 环境缺依赖
        raise BackendUnavailableError(
            "视频解码",
            "conda env update -f environment.yml（需要 opencv 或 imageio-ffmpeg）",
        ) from exc


def _require_video_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise VideoDecodeError(target, "文件不存在")
    if target.suffix.lower() not in VIDEO_EXTENSIONS:
        log.warning("文件扩展名不在已知视频列表中：%s（仍尝试解码）", target.suffix)
    return target


def probe_video(path: str | Path, *, backend: Backend | None = None) -> VideoMeta:
    """读取视频元信息（不加载全部帧）。"""
    target = _require_video_path(path)
    chosen = backend or available_backend()

    if chosen == "opencv":
        import cv2

        cap = cv2.VideoCapture(str(target))
        if not cap.isOpened():
            raise VideoDecodeError(target, "OpenCV 无法打开（编码不支持或文件损坏）")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        finally:
            cap.release()
        duration = (count / fps) if fps > 0 and count > 0 else 0.0
        return VideoMeta(str(target), count, fps, width, height, duration, "opencv")

    import imageio.v3 as iio

    try:
        meta = iio.immeta(str(target), plugin="pyav")
    except Exception:  # noqa: BLE001 - imageio 各插件报错类型不统一
        meta = iio.immeta(str(target))
    fps = float(meta.get("fps", 0.0) or 0.0)
    duration = float(meta.get("duration", 0.0) or 0.0)
    shape = meta.get("shape") or (0, 0, 3)
    height = int(shape[0]) if len(shape) >= 2 else 0
    width = int(shape[1]) if len(shape) >= 2 else 0
    count = int(round(duration * fps)) if fps > 0 else 0
    return VideoMeta(str(target), count, fps, width, height, duration, "imageio")


def iter_frames(
    path: str | Path,
    *,
    sample_fps: float | None = None,
    frame_step: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """按顺序产出 ``(原始帧号, 时间戳秒, RGB uint8 HWC 数组)``。

    Parameters
    ----------
    sample_fps:
        目标抽帧率。None 表示"每帧都取"（仍受 ``frame_step`` 限制）。
    frame_step:
        固定步长降采样（与 ``sample_fps`` 二选一，同时给则以 ``sample_fps`` 为准）。
    """
    target = _require_video_path(path)
    if sample_fps is not None and sample_fps <= 0:
        raise VideoDecodeError(target, f"sample_fps 必须为正，实际 {sample_fps}")
    if frame_step < 1:
        raise VideoDecodeError(target, f"frame_step 必须 >= 1，实际 {frame_step}")
    if max_frames is not None and max_frames < 1:
        raise VideoDecodeError(target, f"max_frames 必须 >= 1，实际 {max_frames}")

    backend = available_backend()
    if backend == "opencv":
        yield from _iter_opencv(target, sample_fps, frame_step, max_frames)
    else:
        yield from _iter_imageio(target, sample_fps, frame_step, max_frames)


def _iter_opencv(
    target: Path,
    sample_fps: float | None,
    frame_step: int,
    max_frames: int | None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        raise VideoDecodeError(target, "OpenCV 无法打开")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None and raw_frame_count <= 0:
        raw_frame_count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            raw_frame_count += 1
        cap.release()
        cap = cv2.VideoCapture(str(target))
        if not cap.isOpened():
            raise VideoDecodeError(target, "无法在统计帧数后重新打开视频")

    # 目标采样间隔（原始帧为单位）：sample_fps=5 且原生 30fps -> 每 6 帧取一帧
    if sample_fps is not None and native_fps > 0:
        stride = max(1, int(round(native_fps / sample_fps)))
    else:
        stride = frame_step
        sample_fps = native_fps / stride if native_fps > 0 else (sample_fps or 1.0)

    selected = _evenly_selected_raw_indices(raw_frame_count, stride, max_frames)
    emitted = 0
    index = 0
    last_timestamp = -1.0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if index % stride == 0 and (selected is None or index in selected):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp = (
                    index / native_fps
                    if native_fps > 0
                    else (index // stride) / max(sample_fps, 1.0)
                )
                position_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if index > 0 and np.isfinite(position_ms) and position_ms > last_timestamp * 1000.0:
                    timestamp = position_ms / 1000.0
                last_timestamp = timestamp
                yield index, timestamp, rgb
                emitted += 1
            index += 1
    finally:
        cap.release()

    if emitted == 0:
        raise VideoDecodeError(target, "未解码出任何帧")


def _iter_imageio(
    target: Path,
    sample_fps: float | None,
    frame_step: int,
    max_frames: int | None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    import imageio.v3 as iio

    meta = probe_video(target, backend="imageio")
    native_fps = meta.fps
    if max_frames is not None and meta.frame_count <= 0:
        try:
            frame_count = sum(1 for _ in iio.imiter(str(target)))
        except Exception as exc:  # noqa: BLE001 - decoder errors vary by plugin
            raise VideoDecodeError(target, f"无法统计视频帧数：{exc}") from exc
    else:
        frame_count = meta.frame_count

    if sample_fps is not None and native_fps > 0:
        stride = max(1, int(round(native_fps / sample_fps)))
        eff_fps = native_fps / stride
    else:
        stride = frame_step
        eff_fps = native_fps / stride if native_fps > 0 else (sample_fps or 1.0)

    selected = _evenly_selected_raw_indices(frame_count, stride, max_frames)
    emitted = 0
    for index, frame in enumerate(iio.imiter(str(target))):
        if index % stride != 0 or (selected is not None and index not in selected):
            continue
        arr = np.asarray(frame)
        if arr.ndim == 2:  # 灰度视频：复制成三通道
            arr = np.stack([arr] * 3, axis=-1)
        timestamp = (
            index / native_fps
            if native_fps > 0
            else (index // stride) / eff_fps
            if eff_fps > 0
            else float(index)
        )
        rgb = arr.astype(np.uint8, copy=False)
        yield index, timestamp, rgb
        emitted += 1

    if emitted == 0:
        raise VideoDecodeError(target, "未解码出任何帧")


def _evenly_selected_raw_indices(frame_count: int, stride: int, max_frames: int | None) -> set[int] | None:
    """对完整视频等间隔限帧，确保上限不会只保留视频开头。"""
    if max_frames is None or frame_count <= 0:
        return None
    candidates = np.arange(0, frame_count, stride, dtype=np.int64)
    if len(candidates) <= max_frames:
        return set(int(index) for index in candidates)
    if max_frames == 1:
        return {int(candidates[len(candidates) // 2])}
    positions = np.linspace(0, len(candidates) - 1, num=max_frames)
    selected_positions = np.rint(positions).astype(np.int64)
    return set(int(index) for index in candidates[selected_positions])


def extract_frames(
    path: str | Path,
    *,
    sample_fps: float,
    frame_step: int = 1,
    max_frames: int | None = None,
) -> tuple[list[int], list[float], list[np.ndarray]]:
    """一次性抽出全部帧（小视频用；长视频请用 ``iter_frames`` 流式处理）。"""
    indices: list[int] = []
    stamps: list[float] = []
    frames: list[np.ndarray] = []
    for index, stamp, frame in iter_frames(
        path, sample_fps=sample_fps, frame_step=frame_step, max_frames=max_frames
    ):
        indices.append(index)
        stamps.append(stamp)
        frames.append(frame)
    return indices, stamps, frames


def save_frame(frame: np.ndarray, path: str | Path, *, quality: int = 92) -> Path:
    """把一帧写盘（jpg/png）。用 Pillow，避免为写图引入 cv2 依赖。"""
    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(frame))
    if target.suffix.lower() in (".jpg", ".jpeg"):
        image.save(target, format="JPEG", quality=int(quality), subsampling=0)
    else:
        image.save(target)
    return target


def write_video(
    frames: Sequence[np.ndarray],
    path: str | Path,
    *,
    fps: float = 5.0,
) -> Path:
    """把一组 RGB 帧写成视频。

    用途：**在没有真实拍摄素材时生成可解码的测试视频**（例如把一段动作序列
    渲染成 GIF 用于端到端验证、或把抽帧结果回灌成视频做演示）。

    后端策略：有 imageio-ffmpeg 就写 mp4（体积小）；否则回退写 GIF
    （Pillow 直接支持，且 ``io.video`` 能解码回来，链路仍然完整）。
    """
    if not frames:
        raise VideoDecodeError(path, "frames 为空，无法写出视频")
    if fps <= 0:
        raise VideoDecodeError(path, f"fps 必须为正，实际 {fps}")

    target = Path(path)
    ensure_dir(target.parent)
    arrays = [np.asarray(frame) for frame in frames]
    arrays = [np.stack([a] * 3, axis=-1) if a.ndim == 2 else a for a in arrays]

    if target.suffix.lower() == ".gif":
        return _write_gif(arrays, target, fps)

    try:
        import imageio.v3 as iio
    except ImportError:
        log.warning("未安装 imageio-ffmpeg，改为写出 GIF：%s", target.with_suffix(".gif"))
        return _write_gif(arrays, target.with_suffix(".gif"), fps)

    try:
        iio.imwrite(str(target), np.stack(arrays), fps=fps, codec="libx264")
        return target
    except Exception as exc:  # noqa: BLE001 - 具体异常类型由 ffmpeg 后端决定
        log.warning("mp4 编码失败（%s），改为写出 GIF", exc)
        return _write_gif(arrays, target.with_suffix(".gif"), fps)


def _write_gif(frames: Sequence[np.ndarray], target: Path, fps: float) -> Path:
    from PIL import Image

    images = [Image.fromarray(np.asarray(f)) for f in frames]
    duration_ms = max(20, int(round(1000.0 / fps)))
    images[0].save(
        target,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    log.info("已写出 GIF：%s（%d 帧 @ %.1ffps）", target, len(images), fps)
    return target
