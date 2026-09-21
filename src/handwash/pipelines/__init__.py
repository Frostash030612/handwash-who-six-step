"""Pipeline 层：把 config / data / models 串成可执行流程。

四个入口（与 CLI 子命令一一对应）
-----------------------------------
``train.py``     训练（frame / clip 两种模式）
``evaluate.py``  评估（内部测试 + 跨场景）
``infer.py``     推理（单段视频 -> 逐帧预测）
``assess.py``    把预测结果变成 WHO 完整性报告

``common.py`` 提供共享设施：设备选择、DataLoader 构造、数据集解析。

分层位置：L4。允许 import 以上所有层。这里写"流程"，不写"规则"与"契约"。
"""

from __future__ import annotations

__all__ = ["assess", "common", "evaluate", "infer", "train"]
