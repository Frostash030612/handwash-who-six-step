# 架构说明（ARCHITECTURE）

本文件解释**为什么这样分层**，以及**为什么`core/` 的接口不能随便改**。
读代码前先读这份文件，能省掉大量"这个函数为什么在这儿"的困惑。

---

## 1. 分层图

```
                        ┌──────────────────────────────────────────┐
   L4  入口与编排        │  cli.py  cli_doctor.py  scripts/*.py     │
                        │  pipelines/: prepare train evaluate      │
                        │              infer  assess  common       │
                        └────────────────┬─────────────────────────┘
                                         │ 可以 import 下面所有层
                        ┌────────────────▼─────────────────────────┐
   L3  模型              │  models/: base frame_cnn yolo26_cls      │
                        │           temporal voting factory        │
                        │           checkpoint                     │
                        └────────────────┬─────────────────────────┘
                                         │
                        ┌────────────────▼─────────────────────────┐
   L2  数据与落盘        │  io/: utils video manifest split         │
                        │  data/: dataset transforms synthetic     │
                        └────────────────┬─────────────────────────┘
                                         │
                        ┌────────────────▼─────────────────────────┐
   L1  契约（最稳定）    │  core/: labels schema config metrics     │
                        │         protocol registry seeding        │
                        └────────────────┬─────────────────────────┘
                                         │
                        ┌────────────────▼─────────────────────────┐
   L0  基础设施          │  paths.py   logging.py   errors.py       │
                        │  （只依赖标准库）                          │
                        └──────────────────────────────────────────┘

        依赖方向：只能从上层指向下层。反向依赖 = 结构违规。
```

### 每层的职责与禁止事项

| 层 | 职责 | **绝对禁止** |
| --- | --- | --- |
| L0 基础设施 | 路径常量、日志、异常体系 | import 任何第三方库（除标准库） |
| L1 契约 | 数据契约、标签、配置、指标、业务规则 | import torch/ultralytics/cv2/pandas/matplotlib；读写文件（`config.py` 读 YAML 是唯一例外） |
| L2 IO / 数据 | 加解码、manifest、划分、预处理、Dataset | import models / pipelines |
| L3 模型 | 网络结构与权重存取 | import pipelines / cli |
| L4 编排 | 把上面串成流程、命令行 | 实现业务规则（规则只能在 `core/protocol.py`） |

**为什么坚持这条线？** 因为小组项目最容易崩的地方是"规则散落在各处"：
如果漏步判定同时存在于评估脚本、demo 界面和一个 notebook 里，
三者迟早会给出不一致的答案，而且没人能说清哪个对。
分层把"规则"钉死在一个文件里，其他人只能调用它。

---

## 2. 五个关键契约（改了就要走 RFC）

### 2.1 标签空间 —— `core/labels.py`

```python
class Step(str, Enum):            # 规范名，全项目唯一
    STEP_1 = "step_1_palm_to_palm"
    ...
STEP_ORDER = (STEP_1, ..., STEP_6)   # WHO 六步的**唯一权威顺序**
LabelSpace.get/to_index/to_label/to_space
```

* 任何数据集的原始标签必须先经 `LabelSpace.canonicalize()` 转成 `Step`，
  再经 `to_index()` 转成模型通道号。**代码里不许出现步骤字面量。**
* `LABEL_SPACES` 里每个命名空间的**索引顺序即模型输出通道顺序**，
  只能追加、不能插入或重排 —— 否则所有旧 checkpoint 的输出会静默错位。

### 2.2 模型输出形状 —— `(B, T, C)`

所有模型（单帧、时序、YOLO 适配器）都必须实现：

```python
model.logits(x) -> Tensor (B, T, C)   # 逐时间步的分类 logits
model.embed(x)  -> Tensor (B, T, D)   # 时序头输入特征
model.num_classes: int
model.feature_dim: int
```

单帧模型也要返回 `(B, 1, C)`，并且**要能接受 `(B, T, 3, H, W)`**
（把 T 折叠进 batch）。这样训练 / 评估 / 推理三处只有一套形状逻辑。

### 2.3 DataLoader 输出 —— 也统一成 `(B, T, …)`

`collate_samples`（帧模式）与 `clip_collate`（时序模式）都产出：

```python
{
  "image":       (B, T, 3, H, W) float32
  "label_index": (B, T)  int64      # 标签空间的通道下标
  "frame_index": (B, T)  int64
  "clip_id":     list[str] 长度 B
}
```

**不要**让某个路径产出一维 `label_index`：那会逼着下游写分支，
分支就是 bug 的温床。

### 2.4 请求 / 响应对象 —— `core/schema.py`

`FrameRecord` / `ClipRecord`（数据）、`Clip` / `Sample`（输入）、
`FramePrediction` / `ClipPrediction`（预测）、`StepStatistic` /
`ProtocolViolation` / `ProtocolReport`（结论）、`EvalResult`（指标）。

* 时间单位统一**秒（float）**；帧号统一**从 0 开始**；
* 新增字段必须给默认值（不破坏旧调用点），否则属于 BREAKING；
* 序列化只走 `to_dict()` / `from_dict()`。

### 2.5 配置结构 —— `core/config.py`

YAML 键名与 dataclass 字段**逐字对应**，未知键直接报错。
配置段：`project / runtime / paths / dataset / datasets / split / model /
train / eval / assess / infer`。

---

## 3. 一次训练的数据流（把契约串起来看）

```
原始视频 (PSKUS/METC/自采)
   │  ① io.video.iter_frames(sample_fps=5)  —— 抽帧（**在划分之后**）
   ▼
帧图像 + manifest.csv                    ← io.manifest（含 6 条不变量校验）
   │  ② io.split.split_clips(group_key=original_video)
   ▼
train / val / test 三个 split            ← 按**原始视频**划分，防泄漏
   │  ③ data.dataset.FrameManifestDataset + data.transforms.build_transforms
   ▼
batch: image (B,T,3,H,W), label_index (B,T)   ← pipelines.common.collate_*
   │  ④ models.factory.build_model(rc).logits(x) -> (B, T, C)
   ▼
logits -> CrossEntropy / FocalLoss -> 反向传播 -> best.pt   ← pipelines.train
   │  ⑤ pipelines.evaluate：Accuracy / Macro-F1 / 混淆矩阵   ← core.metrics
   ▼
EvalResult（落盘 JSON + PNG + CSV）
   │  ⑥ pipelines.infer：视频 -> 逐帧概率 (T, C)
   ▼
core.protocol.build_report：平滑 -> 分段 -> 覆盖/顺序/时长判定
   ▼
ProtocolReport（漏步、乱序、时长不足、综合得分）-> JSON + Markdown
```

第 ⑥ 步是**唯一**允许做业务判定的地方。评估脚本只看指标，不看漏步；
demo 界面只调 `build_report`，不自己判断顺序。

---

## 4. 为什么用注册表（`core/registry.py`）

契约层不能 import torch，但流程层需要"按配置字符串拿实现"。
注册表就是这个解耦点：

```python
# models/my_model.py（L3，可以 import torch）
@register_model("my-model")
class MyModel(BaseClassifier): ...

# models/factory.py 的 register_all() 里 import 触发注册
# pipelines/train.py（L4）只跟 registry 打交道
```

好处：新增模型 = 新增一个文件 + 一行 import + 一份配置，
**不改任何既有代码**。这直接减少了 L1/L2 冲突的概率。

---

## 5. RFC 流程（L1 改动的唯一合法入口）

在下面追加一条记录（新条目写在最上面），然后在 PR 里链接它。

```markdown
### RFC-0003：把 min_segment_s 的默认值从 1.0 调整为 0.5
- 日期：2026-04-11    提出人：@某组员    状态：已通过
- 现状：自采视频里第 6 步常被判为"太短"，导致时长不足告警偏多。
- 改动：core/config.py 的 AssessConfig.min_segment_s 默认 1.0 -> 0.5；
        configs/config.yaml 同步；docs/PROTOCOL.md 更新口径说明。
- 影响：所有已跑的 assess 报告中的 insufficient_duration 数量会变化；
        **实验 E1—E3 的完整性判定结论需要重跑**（分类指标不受影响）。
- 迁移：无需改代码；历史报告保留在原目录，不删除，对比时注明阈值版本。
- 同意：@leader-a ✅  @leader-b ✅
```

### 已通过的 RFC

#### RFC-0001：统一模型输出为 `(B, T, C)`
- 日期：项目初始化    状态：已通过（本框架的初始设计）
- 原因：避免为"帧模型/时序模型"各写一套训练与评估分支。
- 影响：所有模型必须遵守；`BaseClassifier.check_logits_shape` 强制校验。

#### RFC-0002：先划分、后抽帧
- 日期：项目初始化    状态：已通过
- 原因：先抽帧再随机划分会导致同一段视频的相邻帧跨集合，指标虚高。
- 影响：`pipelines/prepare.py` 的流程顺序固定为 scan -> split -> frames；
  `io.split` 提供三道泄漏检查（manifest 校验 / 划分校验 / config 开关）。

---

## 6. 扩展指南

| 想做的事 | 改哪里 | 级别 |
| --- | --- | --- |
| 换模型架构 | 新建 `models/xxx.py` + `configs/models/xxx.yaml` | L3 |
| 换骨干 / 时序头组合 | 只改配置（`model.arch` / `model.temporal.kind`） | L3 |
| 加数据集 | `core/labels.py` 别名（追加）+ `configs/data/xxx.yaml` + `docs/DATA.md` | L1（别名表）+ L3 |
| 改判定阈值 | 只改配置（`assess` 段） | L3 |
| 改判定**规则**（例如允许乱序） | `core/protocol.py` | **L1 + RFC** |
| 加指标 | 追加 `core/metrics.py` 函数 + 接入 `evaluate.py` | L1（追加不破坏） |
| 加命令行子命令 | `cli.py` + `Makefile`（两处必须同时改） | L2 |
| 加单元测试 | `tests/unit/<层>/test_<模块>.py` | L3 |

---

## 7. 机器如何强制这些规则

| 检查 | 位置 | 拦什么 |
| --- | --- | --- |
| 格式与换行 | `.editorconfig` + pre-commit hooks | R2 |
| 分层依赖 / 硬编码路径 / print | `scripts/check_structure.py` | R5、R8、R16 |
| 配置合法性 | `scripts/check_config.py` | R3、R14 |
| 契约行为 | `tests/unit/**` | L1 语义不被悄悄改掉 |
| 大文件 / 密钥 | pre-commit hooks | R6 |
| PR 审阅 | `CODEOWNERS` | L1 / L2 流程 |
| 全量门禁 | `.github/workflows/ci.yml` | 以上全部 |

**改了 L1 却不改测试，CI 会红；改了测试来迁就代码，review 会拦。**
这是"框架不被破坏"的实际保障机制。
