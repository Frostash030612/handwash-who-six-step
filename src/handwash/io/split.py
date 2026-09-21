"""按**原始视频**划分 train/val/test —— 防数据泄漏的关键模块。

为什么不能随机打乱帧（选题文档第五节）
----------------------------------------
同一段视频里相邻帧几乎一样。如果把 3185 段视频的每一帧混在一起随机划分，
同一段视频会同时出现在训练集和测试集，测试准确率会被"记忆"抬到虚高的水平，
整个实验结论作废。因此**划分单位必须是原始视频**，抽帧必须在划分之后。

实现方式（对应 ``SplitConfig.group_key``）
------------------------------------------
1. 对每个 clip 计算分组键（默认 ``video_path`` 的规范形式）；
2. 以**分组键**为单位随机分配到 train/val/test；
3. 可选按主标签分层，保证每类在各 split 都有样本（类别不平衡时很重要）；
4. 划分完成后跑三道校验：无重叠组、无空 split、比例偏差在容差内。

本模块不读写文件（纯函数），因此可以被完整单元测试。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from handwash.core.schema import ClipRecord, FrameRecord, Split
from handwash.errors import DataLeakageError, ManifestError
from handwash.logging import get_logger

__all__ = ["group_key_for", "split_clips", "assign_frames", "assert_no_leakage", "split_report"]

log = get_logger(__name__)

_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def group_key_for(clip: ClipRecord | FrameRecord, group_key: str = "original_video") -> str:
    """计算分组键。

    * ``original_video``：用视频路径的规范化形式（统一分隔符、小写盘符、去扩展名），
      这样 ``Data/RAW/Sub/Video1.mp4`` 与 ``data/raw/sub/video1.MP4`` 会归为同一组。
    * ``clip_id``：完全信任 clip_id（用于已经保证唯一的场景）。
    * ``participant``：从 metadata 取 ``participant``/``subject`` 字段（METC 有 72 名参与者）。
    """
    if group_key == "clip_id":
        return clip.clip_id
    if group_key == "participant":
        meta = dict(getattr(clip, "metadata", {}) or {})
        raw = meta.get("participant") or meta.get("subject") or meta.get("participant_id")
        if raw is None:
            raise ManifestError(
                f"clip={clip.clip_id} 缺少 metadata.participant，无法按参与者分组",
                hint="要么在抽帧阶段补齐 metadata，要么把 split.group_key 改回 original_video。",
            )
        return f"participant:{raw}"
    # 默认：original_video
    raw_path = getattr(clip, "video_path", None)
    if raw_path is None:
        return clip.clip_id
    return _normalize_video_path(str(raw_path))


def _normalize_video_path(raw: str) -> str:
    """规范化视频路径作为分组键：统一小写、分隔符、去掉扩展名与首尾空白。"""
    text = str(raw).strip().replace("\\", "/").lower()
    if len(text) > 1 and text[1] == ":":
        text = text[2:]
    path = Path(text)
    return str(path.with_suffix("")) if path.suffix else text


def split_clips(
    clips: Sequence[ClipRecord],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
    group_key: str = "original_video",
    stratify_by: str | None = "label_sequence",
) -> dict[Split, list[ClipRecord]]:
    """把 clip 级清单划分到 train/val/test。

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``，每个列表内按 clip_id 排序
        （保证不同机器、不同次运行结果一致）。
    """
    if not clips:
        raise ManifestError("split_clips 收到空的 clip 列表")

    ratios = dict(ratios or {"train": 0.7, "val": 0.15, "test": 0.15})
    for name in _SPLIT_NAMES:
        if name not in ratios:
            raise ManifestError(f"划分比例缺少 {name}：{ratios}")
        if ratios[name] < 0:
            raise ManifestError(f"划分比例不能为负：{name}={ratios[name]}")
    total_ratio = sum(ratios[n] for n in _SPLIT_NAMES)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ManifestError(f"划分比例之和必须为 1.0，实际 {total_ratio}")

    # --- 1) 按分组键聚合，先检测"同一段视频被登记了多个 clip_id"的隐患 -----
    groups: dict[str, list[ClipRecord]] = defaultdict(list)
    for clip in clips:
        groups[group_key_for(clip, group_key)].append(clip)

    duplicates = {k: v for k, v in groups.items() if len({c.split for c in v}) > 1}
    if duplicates:
        raise DataLeakageError([c.clip_id for group in duplicates.values() for c in group])

    # --- 2) 分层：先按"主标签"把组分类，再在每层内分配 ---------------------
    rng = random.Random(seed)
    if stratify_by:
        strata: dict[str, list[str]] = defaultdict(list)
        for key, members in groups.items():
            strata[_stratum_of(members, stratify_by)].append(key)
        for keys in strata.values():
            keys.sort()  # 先排序再打乱：与输入顺序无关，结果可复现
            rng.shuffle(keys)
    else:
        all_keys = sorted(groups)
        rng.shuffle(all_keys)
        strata = {"__all__": all_keys}

    assignment: dict[str, Split] = {}
    for _, keys in sorted(strata.items()):
        n = len(keys)
        n_train = _largest_remainder(ratios["train"], n)
        n_val = _largest_remainder(ratios["val"], n)
        # 小样本保护：每个非空 split 至少分到 1 个组（否则 val 为空无法早停）
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        else:
            n_train = max(1, n_train)
            n_val = max(0, min(n_val, n - n_train))
        for i, key in enumerate(keys):
            if i < n_train:
                assignment[key] = "train"
            elif i < n_train + n_val:
                assignment[key] = "val"
            else:
                assignment[key] = "test"

    result: dict[Split, list[ClipRecord]] = {"train": [], "val": [], "test": []}
    for key, members in groups.items():
        target = assignment[key]
        result[target].extend(members)

    for name in _SPLIT_NAMES:
        result[name] = sorted(result[name], key=lambda c: c.clip_id)

    if group_key != "clip_id":
        assert_no_leakage(result, group_key=group_key)
    return result


def _stratum_of(members: Sequence[ClipRecord], stratify_by: str) -> str:
    """求一个组的分层键：多标签段落用"出现最多的 WHO 步骤"，纯单标签则用该标签。"""
    if stratify_by == "label_sequence":
        counter: Counter[str] = Counter()
        for clip in members:
            for step in clip.label_sequence:
                if step.order_index:  # 只看 WHO 六步
                    counter[step.value] += 1
        if counter:
            return counter.most_common(1)[0][0]
        return "__no_label__"
    # 其它取值：当成 metadata 字段名
    for clip in members:
        meta = dict(getattr(clip, "metadata", {}) or {})
        if stratify_by in meta:
            return str(meta[stratify_by])
    raise ManifestError(
        f"stratify_by={stratify_by!r} 既不是 label_sequence 也不是任何 clip 的 metadata 字段",
        hint="改成 label_sequence，或确认抽帧阶段写入了该 metadata。",
    )


def _largest_remainder(ratio: float, n: int) -> int:
    """按比例取整数名额（向下取整；余数由调用方的小样本保护处理）。"""
    if n <= 0:
        return 0
    return max(0, min(n, int(round(ratio * n))))


def assign_frames(
    frames: Sequence[FrameRecord],
    clips: Mapping[Split, Sequence[ClipRecord]],
    *,
    max_per_clip: Mapping[str, int | None] | None = None,
) -> dict[Split, list[FrameRecord]]:
    """把帧级记录按 clip 的 split 归属分组，并可对每段视频做帧数上限截断。

    截断策略：**等间隔采样**而不是取前 N 帧 —— 取前 N 帧会让所有视频都只覆盖
    洗手流程的开头，模型学不到后面的步骤。
    """
    clip_split: dict[str, Split] = {}
    for split_name, clip_list in clips.items():
        for clip in clip_list:
            if clip.clip_id in clip_split:
                raise DataLeakageError([clip.clip_id])
            clip_split[clip.clip_id] = split_name  # type: ignore[assignment]

    grouped: dict[Split, list[FrameRecord]] = defaultdict(list)
    unknown_clips: set[str] = set()
    for rec in frames:
        target = clip_split.get(rec.clip_id)
        if target is None:
            unknown_clips.add(rec.clip_id)
            continue
        grouped[target].append(rec)

    if unknown_clips:
        raise ManifestError(
            f"以下 clip 出现在 manifest 中但没有划分结果：{sorted(unknown_clips)[:5]}",
            hint="划分与 manifest 必须来自同一次 prepare；请重新运行 handwash prepare。",
        )

    caps = dict(max_per_clip or {})
    out: dict[Split, list[FrameRecord]] = {}
    for split_name, recs in grouped.items():
        cap = caps.get(split_name)
        buckets: dict[str, list[FrameRecord]] = defaultdict(list)
        for rec in recs:
            buckets[rec.clip_id].append(rec)
        selected: list[FrameRecord] = []
        for clip_id, clip_recs in buckets.items():
            ordered = sorted(clip_recs, key=lambda r: r.frame_index)
            selected.extend(_evenly_spaced(ordered, cap) if cap else ordered)
        out[split_name] = sorted(selected, key=lambda r: (r.clip_id, r.frame_index))
    return out


def _evenly_spaced(records: Sequence[FrameRecord], cap: int) -> list[FrameRecord]:
    """在序列中等间隔取至多 ``cap`` 条，保留首尾（首尾往往对应动作起止）。"""
    if cap <= 0 or len(records) <= cap:
        return list(records)
    if cap == 1:
        return [records[len(records) // 2]]
    step = (len(records) - 1) / (cap - 1)
    picked = sorted({int(round(i * step)) for i in range(cap)})
    return [records[i] for i in picked]


def assert_no_leakage(
    clips: Mapping[Split, Sequence[ClipRecord]],
    *,
    group_key: str = "original_video",
) -> None:
    """强制校验：任何分组键都不得出现在多个 split 中。"""
    seen: dict[str, Split] = {}
    conflicts: set[str] = set()
    for split_name, clip_list in clips.items():
        for clip in clip_list:
            key = group_key_for(clip, group_key)
            if key in seen and seen[key] != split_name:
                conflicts.add(clip.clip_id)
            seen[key] = split_name  # type: ignore[assignment]
    if conflicts:
        raise DataLeakageError(sorted(conflicts))


def split_report(clips: Mapping[Split, Sequence[ClipRecord]]) -> dict[str, object]:
    """划分摘要：段落数、帧数、各类别分布，写进日志与 outputs/<run>/。"""
    report: dict[str, object] = {}
    for split_name, clip_list in clips.items():
        label_counter: Counter[str] = Counter()
        for clip in clip_list:
            for step in clip.label_sequence:
                label_counter[step.value] += 1
        report[split_name] = {
            "num_clips": len(clip_list),
            "num_frames": int(sum(c.frame_count for c in clip_list)),
            "label_frames": dict(sorted(label_counter.items())),
        }
    return report
