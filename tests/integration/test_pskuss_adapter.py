"""集成测试：PSKUS 真实目录结构的适配器。

标记为 ``integration``：**需要已下载的数据**，因此在没有数据时会自动跳过。
默认测试集合（``-m "not integration"``）不含本文件，所以别人 clone 下来跑测试不会失败。

跑法::

    python scripts/download_data.py --dataset pskuss --files DataSet4.zip --extract
    python -m pytest tests/integration -m integration -v

为什么值得写这个测试
    适配器曾经是"照着猜测的结构"写的（假设目录名即标签、或每段视频一个顶层 csv），
    而 PSKUS 的真实结构完全不同：``<root>/DataSetN/{Videos,Annotations/AnnotatorM}/*``，
    标注 CSV 的列是 ``frame_time, is_washing, movement_code``。
    这类"结构与假设不符"的问题如果不测，会在抽帧跑了几小时之后才暴露。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from handwash.core.config import load_config
from handwash.core.labels import Step, get_label_space

pytestmark = pytest.mark.integration


def _data_root() -> Path:
    """与 scripts/download_data.py 保持一致：HANDWASH_DATA_ROOT 优先。"""
    override = os.environ.get("HANDWASH_DATA_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_absolute():
            return candidate
        return Path(__file__).resolve().parents[2] / candidate
    return Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def pskuss_root() -> Path:
    root = _data_root() / "raw" / "pskuss" / "extracted"
    if not root.is_dir() or not any(root.rglob("DataSet*")):
        pytest.skip(f"未找到 PSKUS 数据（{root}）；先运行 scripts/download_data.py --dataset pskuss")
    return root


def test_pskuss_directory_structure_matches_adapter(pskuss_root: Path) -> None:
    """真实结构必须包含 DataSetN/Videos 与 DataSetN/Annotations/AnnotatorN。"""
    datasets = sorted(p for p in pskuss_root.rglob("DataSet*") if p.is_dir())
    assert datasets, "解压目录下应当有 DataSet1..DataSet11"

    sample = datasets[0]
    assert (sample / "Videos").is_dir(), f"{sample.name} 缺 Videos 目录"
    annotations = sample / "Annotations"
    assert annotations.is_dir(), f"{sample.name} 缺 Annotations 目录"

    annotators = sorted(p for p in annotations.iterdir() if p.is_dir())
    assert annotators, "Annotations 下应当有 Annotator1 / Annotator2"

    videos = sorted((sample / "Videos").glob("*.mp4"))
    csvs = sorted(annotators[0].glob("*.csv"))
    assert videos, "Videos 下应当有 mp4"
    assert csvs, "Annotator 目录下应当有标注 csv"


def test_pskuss_annotation_columns_and_codes(pskuss_root: Path) -> None:
    """标注 CSV 的列名与 movement_code 取值必须与 labels.py 的映射一致。

    这条测试是"标签空间"和"真实数据"之间的契约检查：
    如果上游改了列名，或者我们的映射写错了 code，这里会立刻失败，
    而不是等到训练出来一个准确率异常低的模型才发现。
    """
    from handwash.io.utils import read_csv

    sample_csv = sorted(pskuss_root.rglob("Annotations/Annotator1/*.csv"))[0]
    rows = read_csv(sample_csv)
    assert rows, f"标注文件为空：{sample_csv}"

    assert set(rows[0]) >= {"frame_time", "is_washing", "movement_code"}, (
        f"标注列名变了：{list(rows[0])}"
    )
    for row in rows[:200]:
        int(row["movement_code"])  # 必须能转成整数

    space = get_label_space("pskuss")
    for code in range(8):
        step = space.canonicalize(str(code))
        assert isinstance(step, Step)

    # 用真实数据核对映射方向：Other（code 0）应当占多数，与 summary.csv 的
    # "Other movement" 是最后一项相吻合。方向反了就说明映射错了。
    counts: dict[str, int] = {}
    for path in sorted(pskuss_root.rglob("Annotations/Annotator1/*.csv"))[:10]:
        for row in read_csv(path):
            counts[row["movement_code"]] = counts.get(row["movement_code"], 0) + 1
    assert counts, "没有读到任何标注行"
    dominant = max(counts, key=lambda code: counts[code])
    assert dominant == "0", (
        f"占比最大的 movement_code 是 {dominant}，期望 0（Other movement）。"
        "映射方向可能写反了，见 core/labels.py 中 pskuss 的映射注释。"
    )


def test_pskuss_adapter_produces_clips_and_frames(pskuss_root: Path) -> None:
    """走一遍真实适配器：应当产出 clip 级记录，且每段视频的标签序列非空。"""
    from handwash.pipelines.prepare import _records_from_pskuss

    rc = load_config(["configs/config.yaml", "configs/data/pskuss.yaml"])
    frames, clips = _records_from_pskuss(rc, pskuss_root)

    assert clips, "适配器没有解析出任何片段"
    assert len(frames) > 0

    for clip in clips[:5]:
        assert clip.label_sequence, f"{clip.clip_id} 的标签序列为空"
        assert clip.frame_count == len(clip.label_sequence)
        assert clip.video_path.endswith(".mp4")
        assert clip.metadata.get("annotation", "").endswith(".csv")

    # 六个 WHO 步骤中至少应有一部分在数据里真实出现过
    seen = {step for clip in clips for step in clip.label_sequence}
    who_steps = {step for step in seen if step.order_index}
    assert who_steps, "一个 WHO 步骤都没解析出来，映射或列名很可能不对"


def test_pskuss_split_keeps_videos_disjoint(pskuss_root: Path) -> None:
    """划分必须按视频互斥：同一原始视频不得跨 split（防数据泄漏的第一道验证）。"""
    from handwash.io.split import assert_no_leakage, split_clips
    from handwash.pipelines.prepare import _records_from_pskuss

    rc = load_config(["configs/config.yaml", "configs/data/pskuss.yaml"])
    _, clips = _records_from_pskuss(rc, pskuss_root)
    if len(clips) < 3:
        pytest.skip("片段太少，无法验证划分")

    split_map = split_clips(
        clips,
        ratios={"train": rc.split.train, "val": rc.split.val, "test": rc.split.test},
        seed=rc.split.seed,
        group_key=rc.split.group_key,
        stratify_by=rc.split.stratify_by,
    )
    assert_no_leakage(split_map, group_key=rc.split.group_key)

    total = sum(len(bucket) for bucket in split_map.values())
    assert total == len(clips), "划分后片段总数不一致"
