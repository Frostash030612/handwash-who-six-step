"""契约层：全项目共享的数据结构、标签空间、配置、指标与业务规则。

**这一层是全项目最稳定的部分**，也是"防破坏"的核心：
    * 只允许依赖 numpy / PyYAML / 标准库。
    * 禁止 import torch、ultralytics、cv2、pandas、matplotlib。
    * 禁止读写文件（除 config.py 读 YAML 配置这一处明确例外）。
    * 修改本层任何公开结构 = 破坏性变更，必须走 CONTRIBUTING.md 的 RFC 流程。

子模块：
    labels.py    标签空间（WHO 六步 + 各数据集命名空间的唯一映射）
    schema.py    跨层数据契约（FrameRecord / ClipRecord / Prediction / Report ...）
    config.py    YAML 配置的严格加载与校验
    metrics.py   分类指标（Accuracy / Macro-F1 / 混淆矩阵 ...）
    protocol.py  WHO 完整性判定规则（漏步 / 乱序 / 时长不足）
    registry.py  按字符串名注册模型与数据集，避免 core 依赖具体框架
    seeding.py   随机种子统一控制
"""

from __future__ import annotations

__all__ = [
    "config",
    "labels",
    "metrics",
    "protocol",
    "registry",
    "schema",
    "seeding",
]
