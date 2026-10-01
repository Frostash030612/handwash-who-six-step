> [English](/docs/CONFIG.md) | **中文**

# 配置参考（CONFIG）

摄像头产品使用 `configs/experiments/live_yolo_frame.yaml`：YOLO 分类模型、
`train.mode=frame`、`model.temporal.kind=none`、`infer.mode=frame`。
云端用此配置训练后，本机载入同一配置与对应的项目格式 checkpoint：

```bash
python -m handwash.cli camera config=configs/experiments/live_yolo_frame.yaml --checkpoint path/to/best.pt
```

网页按 `dataset.prep.fps` 采样；实时展示最多平均
`assess.smooth_window` 个已到达帧，并使用 `assess.min_confidence`；
最终报告继续使用配置中的六步判定阈值。

配置文件的键名与 `src/handwash/core/config.py` 的 dataclass 字段**逐字对应**，
拼错的键会**直接报错**（不会静默忽略）。本文件是每个键的说明。

用法回顾：

```bash
# 默认加载 configs/config.yaml
python -m handwash.cli train

# 叠加一个覆盖文件（推荐：只写与默认不同的部分）
python -m handwash.cli train config=configs/experiments/smoke.yaml

# 临时改单个参数（key.sub=value，值按 YAML 标量解析）
python -m handwash.cli train train.epochs=5 runtime.device=cpu

# 替换基础配置（文件必须自洽）
python -m handwash.cli train --config configs/config.yaml
```

> `config=`（位置参数）是**叠加**，`--config` 是**替换**。这是刻意的区分：
> 叠加用于实验覆盖，替换用于完全独立的一套配置。

---

## schema_version

配置结构版本，当前为 `2`。破坏性改动时递增，并同步 `CHANGELOG.md` 与迁移说明。
**不要为了"让它跑起来"而手工改这个数字。**

## project

| 键 | 类型 | 说明 |
| --- | --- | --- |
| `name` | str | 项目名，写入产物元数据 |
| `task` | str | 任务标识，默认 `who_six_step_recognition` |
| `language` | str | 报告语言，默认 `zh` |

## runtime

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `seed` | int | 42 | 全局随机种子。**固定种子是结果可复现的前提** |
| `deterministic` | bool | true | 让 cuDNN 走确定性算法（略慢但可复现） |
| `device` | str | auto | `auto` / `cpu` / `cuda` / `cuda:0` |
| `num_workers` | int | 4 | DataLoader 进程数。**Windows 报错时改成 0** |
| `pin_memory` | bool | true | 有 GPU 时通常更快 |
| `log_level` | str | INFO | DEBUG / INFO / WARNING / ERROR |
| `tracking` | str | csv | `none` / `csv`；CSV 写入 `run_log.csv`，JSONL 仍是标准 epoch 日志 |
| `tracking_project` | str | handwash | 写入 `env_info.json` 的项目标识 |
| `run_name` | str\|null | null | 输出目录名。null 时用时间戳；**对比实验建议写死** |

## paths

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `out_dir` | `outputs` | 实验产物根目录 |
| `models_dir` | `models` | 外部权重存放处（不进 Git） |
| `cache_dir` | `.cache/handwash` | 预留的缓存路径；当前运行时代码不会读取或创建此目录 |

相对路径一律相对**仓库根目录**解析（不是当前工作目录）。

## dataset

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `name` | str | pskuss | 当前使用的数据集，必须出现在 `datasets` 段中 |
| `root` | str | `data/raw/pskuss/extracted` | 数据集原始目录 |
| `variants` | list[str] | `[]` | 数据集变体标记（预留） |
| `label_space` | str\|null | pskuss | 标签空间名；null 表示与 `name` 相同 |
| `include_non_wash` | bool | true | 是否把开关水龙头等辅助动作当类别 |
| `prep.fps` | float | 5.0 | **抽帧率**。动作级任务 5 fps 足够 |
| `prep.frame_step` | int | 1 | 固定步长降采样（与 fps 二选一，fps 优先） |
| `prep.resize_hw` | [int,int] | [224,224] | 落盘尺寸，与当前 YOLO 模型输入一致 |
| `prep.image_ext` | str | jpg | jpg / jpeg / png |
| `prep.jpeg_quality` | int | 85 | 1—100 |
| `prep.min_frames_per_clip` | int | 10 | 短于此值的视频会被跳过 |

## datasets

数据集档案字典：`datasets.<name>.root / processed_dir / frames_dir / manifest`。
训练脚本只读 `dataset.name` 对应的那一份；其余档案不影响运行。

**manifest 里的 `image_path` 是相对 `<root>` 的路径** —— 因此整个数据目录
可以搬走或挂载到别的盘，manifest 不需要改。

## split

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `group_key` | str | original_video | 划分分组键：`original_video` / `clip_id` / `participant` |
| `train` / `val` / `test` | float | 0.70/0.15/0.15 | 三者之和必须为 1.0 |
| `stratify_by` | str\|null | label_sequence | 分层依据；null 表示不分层 |
| `seed` | int | 42 | 划分随机种子 |
| `max_frames_per_clip_train` | int\|null | 200 | 长视频**等间隔**截断（不是取前 N 帧） |
| `max_frames_per_clip_eval` | int\|null | 100 | 同上，评估集 |
| `guard_leakage` | bool | true | 发现跨 split 的原始视频直接报错 |

> `group_key` 是防数据泄漏的关键。**改成 `clip_id` 就失去了防泄漏能力**，
> 除非你能保证 clip_id 与原始视频一一对应。

## model

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `name` | str | yolo26n-cls | 便于阅读的模型名（不影响行为） |
| `arch` | str | yolo26n-cls | **注册名**，见 `src/handwash/models/` |
| `pretrained` | str\|bool | auto | `auto` / `true` / `false` / `imagenet`；本地 `.pt` 路径仅由 Ultralytics 适配器支持 |
| `num_classes` | int\|null | null | **保持 null**：由标签空间自动推导 |
| `image_size` | int | 224 | 必须是 32 的倍数；当前支持 64 到 320（以配置校验为准） |
| `normalize` | str | zero_one | `imagenet` / `zero_one` / `minus_one_one` |
| `dropout` | float | 0.2 | [0, 1) |
| `temporal.kind` | str | none | `gru` / `tcn` / `mean_pool` / `none` |
| `temporal.hidden_size` | int | 128 | 时序头隐藏维度 |
| `temporal.num_layers` | int | 1 | GRU 层数 |
| `temporal.bidirectional` | bool | false | true 会用到未来帧，**不能用于实时演示** |
| `temporal.window` | int | 16 | 时序窗口长度（帧） |
| `temporal.stride` | int | 8 | clip 训练和推理共用的窗口步长（`1..window`）；最后一个窗口可以较短 |
| `temporal.kernel_size` | int | 3 | TCN 卷积核 |
| `temporal.dilations` | list[int] | [1,2,4,8] | TCN 膨胀率 |
| `temporal.dropout` | float | 0.1 | 时序头 dropout |

`pretrained: auto` 会加载对应架构的默认预训练权重。本地权重先从 `paths.models_dir`
查找，再从仓库根目录查找。Ultralytics 适配器可以用兼容的分类模型 `.pt`（例如
`exp.pt`）初始化骨干网络；这不会把它变成本项目 checkpoint。项目评估仍需要本项目
生成的 `outputs/<run>/models/best.pt`。外部 `.pt` 是否兼容取决于 Ultralytics 版本和模型结构。

**`normalize` 与 arch 的搭配（最常见的坑）**

| arch | 必须的 normalize | 原因 |
| --- | --- | --- |
| `yolo26n-cls` / `yolon-cls` / `yolov8n-cls` | `zero_one` | Ultralytics 分类输入应缩放到 [0, 1]；再做 ImageNet 均值/方差标准化会改变输入范围 |
| `mobilenet_v2` / `resnet18` / `efficientnet_b0` | `imagenet` | 用 ImageNet 预训练权重，必须匹配其输入分布 |

配置加载时会拒绝不匹配的组合；`handwash doctor` 也会显示预期的输入归一化方式。
Ultralytics 分类预处理说明见[官方文档](https://docs.ultralytics.com/tasks/classify/)。

## train

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `mode` | str | frame | `frame` / `clip`；时序头必须用 `clip`。`hybrid` 仅用于推理融合 |
| `epochs` | int | 20 | 训练轮数 |
| `batch_size` | int | 32 | **clip 模式下含义不同**：窗口数 × 窗口长度 |
| `eval_batch_size` | int | 64 | 评估 batch |
| `lr` | float | 3e-4 | 学习率 |
| `weight_decay` | float | 5e-4 | 权重衰减 |
| `optimizer` | str | adamw | `adamw` / `sgd` |
| `momentum` | float | 0.9 | 仅 SGD |
| `scheduler` | str | cosine | `cosine` / `step` / `none` |
| `warmup_epochs` | float | 1.0 | 预热轮数 |
| `label_smoothing` | float | 0.05 | [0, 1) |
| `class_weights` | str | none | `none` / `balanced`（只平衡训练集中出现的类别；缺失类别权重为 0） |
| `early_stopping_patience` | int | 5 | 验证 Macro-F1 连续 N 轮不提升就停 |
| `grad_clip_norm` | float | 1.0 | 梯度裁剪；0 表示关闭 |
| `precision` | str | fp32 | `fp32` / `fp16` / `bf16`（fp16、bf16 需要受支持的 CUDA；CPU/MPS 请用 fp32） |
| `accumulate_grad_batches` | int | 1 | 梯度累积，显存不足时放大等效 batch |
| `focal_gamma` | float | 0.0 | >0 时改用焦点损失 |
| `augment.*` | | | 见下 |

`train.augment`：`random_resized_crop`(true) / `crop_scale`([0.7,1.0]) /
`horizontal_flip`(true) / `color_jitter`(0.2) / `rotation_deg`(8.0) /
`gaussian_blur`(0.1) / `randaugment`(false；启用会报错，因为尚未实现)
clip 训练时，同一视频同一轮内的随机增强参数会在所有帧间共享，避免人为制造逐帧闪烁。

> **选模标准是验证集 Macro-F1**，不是 Accuracy。类别不平衡时 Accuracy 会掩盖
> 小类别（第 5、6 步）的退化。
> 验证预测会使用配置的推理模式、TTA、时序窗口和概率平滑，使选 checkpoint 的输出路径
> 与正式评估一致。

## eval

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `splits` | list[str] | [val, test] | 要评估的 split |
| `metrics` | list[str] | 见文件 | 记录用（实际指标固定在 core/metrics） |
| `save_confusion_matrix` | bool | true | 输出 PNG |
| `save_predictions` | bool | true | 输出逐帧 JSONL（复核失败案例用） |
| `bootstrap_ci` | bool | false | 自助法置信区间（证明提升不是波动） |
| `bootstrap_samples` | int | 1000 | 自助采样次数 |
| `extra_datasets` | list[str] | [] | 跨场景数据集名，如 `[metc]` |

评估会复用 `handwash infer` 的逐视频推理流程：`infer.mode`、TTA 和概率平滑都会影响
指标与预测文件，保证离线结果与演示时采用的推理设置一致。

## assess（完整性判定阈值）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `smooth_window` | 9 | 多数投票滑窗（帧） |
| `min_confidence` | 0.4 | 低于此置信度的帧不参与投票 |
| `min_segment_frames` | 5 | 有效片段的最少帧数 |
| `min_segment_s` | 1.0 | 有效片段的最短秒数 |
| `min_total_duration_s` | 40.0 | 总搓洗时长下限（WHO 建议 40—60 s） |
| `reference_total_duration_s` | 50.0 | 打分用的参考总时长 |
| `min_step_duration_s` | 3.0 | 单步时长下限（`duration_check=seconds` 时生效） |
| `step_duration_ratio` | 0.4 | 单步时长占"平均份额"的最低比例（`ratio` 时生效） |
| `allow_repeats` | false | 是否允许重复步骤 |
| `missing_tolerance` | 0 | 允许漏几步仍判"基本完整" |
| `duration_check` | ratio | `seconds` / `ratio` / `none` |
| `order_check` | true | 是否检查顺序 |
| `require_faucet_events` | false | 是否要求检出开关水龙头 |
| `report_language` | zh | 报告语言 |

判定口径详见 [`PROTOCOL.md`](PROTOCOL.md)。

## infer

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `mode` | frame | `frame` / `clip` / `hybrid`；默认与逐帧训练一致 |
| `temporal_apply` | false | 是否做概率滑动平均 |
| `smooth_window` | 9 | 滑动平均窗口 |
| `save_frame_predictions` | true | 输出逐帧 JSONL |
| `save_overlay_video` | false | 输出叠加逐帧标签的 GIF 预览 |
| `batch_size` | 64 | 推理 batch |
| `tta` | false | 对原图和水平翻转图的概率取平均（更慢） |

---

## 常见改法速查

```bash
# 显存不足
train.batch_size=8 train.accumulate_grad_batches=4
# 换成 CPU 快速验证链路
runtime.device=cpu runtime.num_workers=0
# Windows DataLoader 报错
runtime.num_workers=0
# 试 TCN 替代 GRU
model.temporal.kind=tcn train.mode=clip infer.mode=clip
# 关闭增强做消融（验证增强到底有没有用）
train.augment.random_resized_crop=false train.augment.horizontal_flip=false \
train.augment.color_jitter=0.0 train.augment.rotation_deg=0.0
# 只做漏步/顺序判定，不管时长
assess.duration_check=none
# 允许回头补做
assess.order_check=false
```
