"""合成数据：让"没有下载任何数据集"的组员也能跑通全流程。

两个用途
--------
1. **冒烟训练**（``make train-smoke`` / ``configs/experiments/smoke.yaml``）：
   30 秒内验证"数据 → 模型 → 指标 → 报告"链路没有断。
2. **单元测试**：给 ``pipelines`` 提供完全确定、无需网络的输入。

设计要点
--------
* 每类合成图有**确定的视觉签名**（基色 + 形状 + 噪声），所以一个正常的小模型
  应该能很快把它学到接近满分 —— 这正好用来验证训练代码本身没写错。
* 同样的 ``seed`` 一定产出同样的字节序列（像素级确定），因此可以断言指标。
* 生成的是**真实目录结构 + manifest**，不是内存里的假对象，
  这样连 io 层、抽帧后的训练入口都被真实覆盖。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from handwash.core.labels import CANONICAL_STEPS, Step
from handwash.core.schema import ClipRecord, FrameRecord, Split
from handwash.io.manifest import write_manifest
from handwash.io.video import save_frame
from handwash.logging import get_logger
from handwash.paths import ensure_dir

__all__ = ["render_step_frame", "SyntheticFrameDataset", "write_synthetic_dataset"]

log = get_logger(__name__)

#: 每类动作的基色（RGB）。刻意选差异大的颜色，让分类任务"可学但需要真的看图像"。
_STEP_COLORS: dict[Step, tuple[int, int, int]] = {
    Step.STEP_1: (220, 60, 60),
    Step.STEP_2: (60, 200, 90),
    Step.STEP_3: (60, 110, 230),
    Step.STEP_4: (230, 200, 50),
    Step.STEP_5: (170, 70, 210),
    Step.STEP_6: (50, 210, 210),
}

#: 每类动作的几何签名：圆/方/十字等，避免模型只靠颜色就分开（留一点真实难度）
_STEP_SHAPES: dict[Step, str] = {
    Step.STEP_1: "disc",
    Step.STEP_2: "square",
    Step.STEP_3: "bars",
    Step.STEP_4: "cross",
    Step.STEP_5: "ring",
    Step.STEP_6: "dots",
}


def render_step_frame(
    step: Step,
    *,
    size: int = 96,
    rng: np.random.Generator,
    noise: float = 18.0,
    brightness: float = 1.0,
) -> np.ndarray:
    """渲染一帧合成图像（HWC uint8）。同一 ``rng`` 状态 -> 同一结果。"""
    if size < 16:
        raise ValueError(f"size 太小：{size}")

    base = np.asarray(_STEP_COLORS.get(step, (128, 128, 128)), dtype=np.float32)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    # 背景：加一点渐变，模拟真实场景的光照不均
    ramp = np.linspace(0.75, 1.05, size, dtype=np.float32).reshape(size, 1, 1)
    canvas[:] = base.reshape(1, 1, 3) * ramp * 0.45

    shape = _STEP_SHAPES.get(step, "disc")
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = size / 2 + float(rng.uniform(-2, 2))
    cx = size / 2 + float(rng.uniform(-2, 2))
    radius = size * float(rng.uniform(0.28, 0.36))
    mask = _shape_mask(shape, xx - cx, yy - cy, radius, size)
    canvas[mask] = base * brightness

    noise_layer = rng.normal(0.0, noise, size=(size, size, 3)).astype(np.float32)
    canvas = np.clip(canvas + noise_layer, 0, 255)
    return canvas.astype(np.uint8)


def _shape_mask(shape: str, dx: np.ndarray, dy: np.ndarray, radius: float, size: int) -> np.ndarray:
    dist = np.sqrt(dx**2 + dy**2)
    if shape == "disc":
        return dist <= radius
    if shape == "ring":
        return (dist <= radius) & (dist >= radius * 0.55)
    if shape == "square":
        return (np.abs(dx) <= radius * 0.85) & (np.abs(dy) <= radius * 0.85)
    if shape == "cross":
        return ((np.abs(dx) <= radius * 0.25) & (np.abs(dy) <= radius)) | (
            (np.abs(dy) <= radius * 0.25) & (np.abs(dx) <= radius)
        )
    if shape == "bars":
        return (np.abs(dy) <= radius) & ((np.abs(dx) <= radius * 0.18) | (np.abs(dx - radius * 0.5) <= radius * 0.18))
    if shape == "dots":
        grid = size / 5.0
        return ((dx % grid - grid / 2) ** 2 + (dy % grid - grid / 2) ** 2) <= (radius * 0.28) ** 2
    return dist <= radius  # pragma: no cover - 兜底


# ============================================================================
# 内存版：单元测试与最快速冒烟
# ============================================================================
class SyntheticFrameDataset:
    """不落盘的合成帧数据集，接口与 ``FrameManifestDataset`` 一致（返回 dict）。

    这是**唯一**允许不依赖 torch 的 Dataset —— 它返回 numpy 数组，
    ``collate`` 由调用方决定。这样单元测试不需要 GPU/大数据。
    """

    def __init__(
        self,
        *,
        num_samples: int = 128,
        num_classes: int = 6,
        image_size: int = 64,
        seed: int = 42,
        transform=None,
        scramble: bool = False,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples 必须 >= 1")
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.transform = transform
        self.scramble = bool(scramble)
        self._rng = np.random.default_rng(seed)
        steps = list(CANONICAL_STEPS)[: self.num_classes]
        if self.scramble:
            self._labels = [steps[int(self._rng.integers(0, len(steps)))] for _ in range(self.num_samples)]
        else:
            self._labels = [steps[i % len(steps)] for i in range(self.num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    @property
    def labels(self) -> list[Step]:
        return list(self._labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        step = self._labels[index]
        # 每帧独立可复现：由 (seed, index) 派生局部 rng
        local = np.random.default_rng([self.seed, index])
        image = render_step_frame(step, size=self.image_size, rng=local)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label_index": index % self.num_classes if not self.scramble else self._labels_index(step),
            "clip_id": f"synthetic_{index // 8:04d}",
            "frame_index": index % 8,
            "label": step.value,
        }

    def _labels_index(self, step: Step) -> int:
        return list(CANONICAL_STEPS).index(step)

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self._labels:
            counts[step.value] = counts.get(step.value, 0) + 1
        return counts


# ============================================================================
# 落盘版：端到端演练
# ============================================================================
def write_synthetic_dataset(
    root: str | Path,
    *,
    num_clips: int = 24,
    frames_per_clip: int = 12,
    image_size: int = 96,
    seed: int = 42,
    fps: float = 5.0,
    mode: str = "who_order",
    splits: Sequence[Split] = ("train", "val", "test"),
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[list[FrameRecord], list[ClipRecord]]:
    """生成一个结构完全标准的合成数据集（帧图像 + manifest）。

    Parameters
    ----------
    mode:
        * ``"who_order"``：每段视频严格按 WHO 六步顺序执行（用于验证"完整流程"判定）。
        * ``"single_step"``：每段视频只做一个步骤（用于帧级分类平衡）。
        * ``"missing_step"``：随机漏掉一步（用于验证漏步检出）。
        * ``"shuffled"``：随机打乱顺序（用于验证乱序检出）。

    Returns
    -------
    (frames, clips)
        ``frames`` 可直接写入 manifest；``clips`` 可直接送进 ``io.split``。
    """
    base = ensure_dir(Path(root))
    image_root = ensure_dir(base / "frames")
    rng = np.random.default_rng(seed)
    steps = list(CANONICAL_STEPS)

    # 1) 先决定每段视频的步骤序列（**先划分再抽帧**，此处直接生成已划好的 split）
    clip_plans: list[tuple[Split, list[Step]]] = []
    split_cycle: list[Split] = []
    n_train = int(round(ratios[0] * num_clips))
    n_val = int(round(ratios[1] * num_clips))
    split_cycle = ["train"] * n_train + ["val"] * n_val
    split_cycle += ["test"] * max(0, num_clips - len(split_cycle))
    for split_name in split_cycle:
        if mode == "single_step":
            plan = [steps[int(rng.integers(0, len(steps)))]]
        elif mode == "missing_step":
            skip = int(rng.integers(0, len(steps)))
            plan = [s for i, s in enumerate(steps) if i != skip]
            rng.shuffle(plan)
        elif mode == "shuffled":
            plan = list(steps)
            rng.shuffle(plan)
        else:  # who_order
            plan = list(steps)
        clip_plans.append((split_name, plan))

    frames: list[FrameRecord] = []
    clips: list[ClipRecord] = []
    frames_per_step = max(1, frames_per_clip // 6)

    for clip_index, (split_name, plan) in enumerate(clip_plans):
        clip_id = f"syn_{clip_index:04d}"
        video_path = str(image_root / clip_id / "source.mp4")  # 占位路径：合成数据无真实视频
        label_sequence: list[Step] = []
        frame_index = 0
        for step in plan:
            for _ in range(frames_per_step):
                local = np.random.default_rng([seed, clip_index, frame_index])
                image = render_step_frame(step, size=image_size, rng=local)
                rel = Path("frames") / clip_id / f"{frame_index:05d}.jpg"
                save_frame(image, base / rel, quality=90)
                frames.append(
                    FrameRecord(
                        clip_id=clip_id,
                        frame_index=frame_index,
                        image_path=str(rel).replace("\\", "/"),
                        label=step,
                        dataset="synthetic",
                        split=split_name,
                        timestamp_s=frame_index / fps,
                    )
                )
                label_sequence.append(step)
                frame_index += 1

        clips.append(
            ClipRecord(
                clip_id=clip_id,
                dataset="synthetic",
                split=split_name,
                video_path=video_path,
                frame_count=frame_index,
                fps=fps,
                duration_s=frame_index / fps,
                label_sequence=tuple(label_sequence),
                metadata={"synthetic": True, "mode": mode},
            )
        )

    write_manifest(base / "manifest.csv", frames)
    log.info(
        "合成数据集已生成：%s（%d 段视频 / %d 帧 / 模式 %s）", base, len(clips), len(frames), mode
    )
    return frames, clips
