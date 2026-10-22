"""Video decoding, sampling and export utilities."""

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
    "available_backend",
    "extract_frames",
    "iter_frames",
    "probe_video",
    "save_frame",
    "write_video",
]

log = get_logger(__name__)

Backend = Literal["opencv", "imageio"]


@dataclass(frozen=True, slots=True)
class VideoMeta:
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

    try:
        import cv2  # noqa: F401

        return "opencv"
    except ImportError:
        pass
    try:
        import imageio.v3 as iio  # noqa: F401

        return "imageio"
    except ImportError as exc:
        raise BackendUnavailableError(
            "Video decoding",
            "Install opencv-python or imageio-ffmpeg in the current environment.",
        ) from exc


def _require_video_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise VideoDecodeError(target, "File not found")
    if target.suffix.lower() not in VIDEO_EXTENSIONS:
        log.warning("Unknown video extension: %s (attempting decoding)", target.suffix)
    return target


def probe_video(path: str | Path, *, backend: Backend | None = None) -> VideoMeta:

    target = _require_video_path(path)
    chosen = backend or available_backend()

    if chosen == "opencv":
        import cv2

        cap = cv2.VideoCapture(str(target))
        if not cap.isOpened():
            raise VideoDecodeError(
                target, "OpenCV could not open the video (unsupported codec or damaged file)"
            )
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
    except Exception:
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

    target = _require_video_path(path)
    if sample_fps is not None and sample_fps <= 0:
        raise VideoDecodeError(target, f"sample_fps must be positive; received {sample_fps}")
    if frame_step < 1:
        raise VideoDecodeError(target, f"frame_step must be >= 1; received {frame_step}")
    if max_frames is not None and max_frames < 1:
        raise VideoDecodeError(target, f"max_frames must be >= 1; received {max_frames}")

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
        raise VideoDecodeError(target, "OpenCV could not open the video")

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
            raise VideoDecodeError(target, "Unable to reopen the video after counting frames")

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
                timestamp = index / native_fps if native_fps > 0 else (index // stride) / max(sample_fps, 1.0)
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
        raise VideoDecodeError(target, "No frames were decoded")


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
        except Exception as exc:
            raise VideoDecodeError(target, f"Cannot count video frames: {exc}") from exc
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
        if arr.ndim == 2:
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
        raise VideoDecodeError(target, "No frames were decoded")


def _evenly_selected_raw_indices(frame_count: int, stride: int, max_frames: int | None) -> set[int] | None:

    if max_frames is None or frame_count <= 0:
        return None
    candidates = np.arange(0, frame_count, stride, dtype=np.int64)
    if len(candidates) <= max_frames:
        return {int(index) for index in candidates}
    if max_frames == 1:
        return {int(candidates[len(candidates) // 2])}
    positions = np.linspace(0, len(candidates) - 1, num=max_frames)
    selected_positions = np.rint(positions).astype(np.int64)
    return {int(index) for index in candidates[selected_positions]}


def extract_frames(
    path: str | Path,
    *,
    sample_fps: float,
    frame_step: int = 1,
    max_frames: int | None = None,
) -> tuple[list[int], list[float], list[np.ndarray]]:

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

    if not frames:
        raise VideoDecodeError(path, "Cannot write a video with no frames")
    if fps <= 0:
        raise VideoDecodeError(path, f"fps must be positive; received {fps}")

    target = Path(path)
    ensure_dir(target.parent)
    arrays = [np.asarray(frame) for frame in frames]
    arrays = [np.stack([a] * 3, axis=-1) if a.ndim == 2 else a for a in arrays]

    if target.suffix.lower() == ".gif":
        return _write_gif(arrays, target, fps)

    try:
        import imageio.v3 as iio
    except ImportError:
        log.warning("imageio-ffmpeg is unavailable; writing GIF: %s", target.with_suffix(".gif"))
        return _write_gif(arrays, target.with_suffix(".gif"), fps)

    try:
        iio.imwrite(str(target), np.stack(arrays), fps=fps, codec="libx264")
        return target
    except Exception as exc:
        log.warning("MP4 encoding failed (%s); writing GIF instead", exc)
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
    log.info("GIF written: %s (%d frames @ %.1ffps)", target, len(images), fps)
    return target
