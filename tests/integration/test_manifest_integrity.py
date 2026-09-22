"""集成测试：manifest 里的每一条路径都必须在磁盘上真实存在。

**为什么必须有这条测试**

曾经出现过两个真实 bug，都只有在"真跑训练"时才暴露，而单跑 `prepare_data.py`
看起来一切正常：

  1. **image_path 与 image_root 不同源**
     `prepare.py` 把 image_path 写成 `frames/<clip_id>/00000.jpg`（相对数据集 root），
     而 `data/dataset.py` 把 image_root 解析成 `dataset.root`（原始视频目录）。
     结果训练时报 `帧图像不存在：<root>/frames/...`，而帧其实好好地躺在
     `data/processed/<dataset>/frames/` 下。

  2. **配置叠加顺序导致跑到别的数据集上**
     `configs/experiments/*.yaml` 里也写了 `dataset.name`；若把它叠在数据配置**之后**，
     数据配置就被覆盖回默认数据集，训练会静默地跑在另一个数据集上
     （日志里出现 "252 帧 / 6 类" 这种合成数据的特征）。

本测试直接验证"manifest 说的路径，文件真的在"，这是上面两个 bug 的共同症状，
也是最便宜、最不可能误报的守门方式。
"""

from __future__ import annotations

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
def pskuss_manifest() -> tuple[Path, list]:
    manifest = _data_root() / "processed" / "pskuss" / "manifest.csv"
    if not manifest.exists():
        pytest.skip(f"未找到 manifest（{manifest}）；先运行 scripts/prepare_data.py")
    return manifest, read_manifest(manifest)


def test_manifest_paths_resolve_and_exist(pskuss_manifest: tuple[Path, list]) -> None:
    """每一条 image_path 用配置推导出的 image_root 拼起来，必须指向真实文件。

    这条断言同时锁住两件事：
        * `prepare.py` 写路径的口径；
        * `dataset.py` 解析 image_root 的口径。
    两者只要有一个改动而另一个没跟上，这里就会红。
    """
    from handwash.data.dataset import _image_root_from

    _, records = pskuss_manifest
    rc = load_config(["configs/config.yaml", "configs/data/pskuss.yaml"])
    image_root = _image_root_from(rc)

    assert image_root.is_dir(), (
        f"配置推导出的 image_root 不存在：{image_root}\n"
        "（检查 configs 里的 frames_dir / processed_dir 是否与实际抽帧落盘位置一致）"
    )

    missing: list[str] = []
    for record in records[:200]:  # 抽查前 200 条，够暴露系统性问题且很快
        candidate = image_root / record.image_path
        if not candidate.exists():
            missing.append(f"{record.image_path} -> {candidate}")

    assert not missing, (
        f"{len(missing)} 条 manifest 路径在磁盘上不存在，前 3 条：\n  "
        + "\n  ".join(missing[:3])
        + "\n\n这通常意味着 image_path 的写法与 image_root 的解析口径不一致。"
    )


def test_manifest_paths_are_relative_not_absolute(pskuss_manifest: tuple[Path, list]) -> None:
    """image_path 必须是相对路径（不含盘符/前导斜杠）。

    绝对路径会让整包数据无法搬移，也会让"队友解包到自己的 data 目录"失效 ——
    而我们的协作方案正是靠"相对路径 + manifest 随包分发"成立的。
    """
    _, records = pskuss_manifest
    for record in records[:50]:
        assert not Path(record.image_path).is_absolute(), (
            f"image_path 是绝对路径：{record.image_path}"
        )
        assert not record.image_path.startswith(("/", "\\")), (
            f"image_path 不应以分隔符开头：{record.image_path}"
        )


def test_manifest_has_all_three_splits(pskuss_manifest: tuple[Path, list]) -> None:
    """三个 split 都要有数据：没有 val 就无法早停选模，没有 test 就没有结论。"""
    _, records = pskuss_manifest
    splits = {record.split for record in records}
    for expected in ("train", "val", "test"):
        assert expected in splits, f"manifest 缺少 split={expected}（实际有 {sorted(splits)}）"
