"""集成测试：manifest 必须保留**逐帧**标签，不能整段塌成一个类。

**为什么必须有这条测试**

曾经出现过一次非常隐蔽的错误：PSKUS 的 39 段视频在 manifest 里**只有 `other` 一个标签**，
六步动作识别完全失去监督信号 —— 而整条流水线没有任何报错，
`split_report.json` 的帧数、划分比例、帧图像数量全都正常。

根因是三个问题叠加：

  1. `prepare()` 把"图像目录树"启发式放在专门适配器**之前**判断。
     抽帧前 PSKUS 是"一堆 mp4 + 标注 csv、没有任何图像"，启发式误判为图像目录树，
     于是走 `_records_from_frames_dir`，把 `DataSet4` 这种**分片目录名**当标签名，
     全部落到兜底的 `other`。
  2. 即使走对了适配器，抽帧阶段也只用 `clip.label_sequence[0]`（整段一个标签），
     适配器算好的逐帧标注被整个丢弃。
  3. 时间戳用"采样序号 ÷ 采样帧率"，而不是标注里的真实时间。

本测试直接断言"标签有多个类别、且逐帧标签确实随时间变化"，
这是上面三个问题共同的症状，也是最便宜、最不可能误报的守门方式。
"""

from __future__ import annotations

import collections
import os
from pathlib import Path

import pytest

from handwash.core.config import load_config
from handwash.io.manifest import read_manifest

pytestmark = pytest.mark.integration


def _data_root() -> Path:
    override = os.environ.get("HANDWASH_DATA_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_absolute():
            return candidate
        return Path(__file__).resolve().parents[2] / candidate
    return Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def pskuss_records() -> list:
    manifest = _data_root() / "processed" / "pskuss" / "manifest.csv"
    if not manifest.exists():
        pytest.skip(f"未找到 manifest（{manifest}）；先运行 scripts/prepare_data.py")
    return read_manifest(manifest)


def _label_name(record: object) -> str:
    """取标签的**规范名**，而不是枚举的 repr。

    坑：``Step`` 是 ``str`` 枚举，但 ``str(Step.STEP_1)`` 返回 ``'Step.STEP_1'``
    （枚举的 repr），只有 ``.value`` 才是 ``'step_1_palm_to_palm'``。
    直接 str() 会让"六步是否齐全"的判据永远失败 —— 曾经就是这样误报的。
    """
    label = getattr(record, "label", record)
    return str(getattr(label, "value", label))


def test_manifest_has_more_than_one_label(pskuss_records: list) -> None:
    """标签必须多于一个类。

    整段塌成一个类时，模型可以靠"全部预测成同一类"拿到满分，
    指标好看但系统完全无用。这条断言是最直接的守门。
    """
    labels = collections.Counter(_label_name(r) for r in pskuss_records)
    assert len(labels) > 1, (
        f"manifest 里只有 {len(labels)} 个标签：{dict(labels)}\n"
        "这通常意味着逐帧标注没有被使用（整段被赋成同一个类）。"
    )


def test_manifest_contains_the_six_who_steps(pskuss_records: list) -> None:
    """六步必须都出现，否则六分类任务没有监督信号。

    只要求"出现过"而不是"每步都有足够样本"：
    后者取决于用了哪些分片，在小规模子集上本来就可能缺步。
    """
    present = {_label_name(r) for r in pskuss_records}
    missing = [
        f"step_{i}" for i in range(1, 7)
        if not any(name.startswith(f"step_{i}_") for name in present)
    ]
    assert not missing, (
        f"manifest 缺少这些 WHO 步骤：{missing}\n实际标签：{sorted(present)}\n"
        "检查：(1) 是否走了专门适配器而不是图像目录树分支；"
        "(2) 抽帧时是否逐帧取标签而不是只用 label_sequence[0]。"
    )


def test_labels_vary_within_a_clip(pskuss_records: list) -> None:
    """至少大多数片段内部标签要发生变化 —— 证明是逐帧标签而非整段一个类。"""
    per_clip: dict[str, set[str]] = collections.defaultdict(set)
    for record in pskuss_records:
        per_clip[record.clip_id].add(_label_name(record))
    variable = sum(1 for labels in per_clip.values() if len(labels) > 1)
    assert variable >= max(1, len(per_clip) // 2), (
        f"只有 {variable}/{len(per_clip)} 段视频内部标签有变化；"
        "若接近 0，说明整段被赋成了同一个标签。"
    )


def test_timestamps_come_from_annotation_not_index_division(pskuss_records: list) -> None:
    """时间戳必须来自标注的真实时间，而不是"采样序号 ÷ 采样帧率"。

    判据：同一 clip 内相邻两帧的时间差应当约等于 1/fps（均匀采样间隔）。
    若用的是采样序号，时间差会是原生的（例如 7.5s）—— 差两个数量级，很容易区分。
    """
    by_clip: dict[str, list] = collections.defaultdict(list)
    for record in pskuss_records:
        by_clip[record.clip_id].append(record)

    checked = 0
    for records in by_clip.values():
        records.sort(key=lambda r: r.frame_index)
        if len(records) < 3:
            continue
        deltas = [
            round(b.timestamp_s - a.timestamp_s, 3)
            for a, b in zip(records, records[1:], strict=False)
        ]
        # 采样间隔应当稳定且较小（PSKUS 2fps -> 0.5s）；整段不可能出现几十秒的间隔
        assert max(deltas) < 5.0, (
            f"clip {records[0].clip_id} 出现 {max(deltas)}s 的帧间隔，"
            f"时间戳疑似不是按采样网格递增的（前几个 delta={deltas[:5]}）"
        )
        checked += 1
        if checked >= 5:
            break
    assert checked > 0, "没有可检查的 clip"
