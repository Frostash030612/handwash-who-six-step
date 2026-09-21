> [English](/CONTRIBUTING.md) | **中文**

# 修改规则（CONTRIBUTING）

> **一句话规则：改代码前先看这张表 —— 你要动的是 L1 / L2 / L3 哪一级？**

本文件是这个仓库的"法律"。它解决小组项目最常见的失败模式：**五个人各自改一点，
两周后没人能跑通全流程，也没人说得清哪个结果是哪份代码跑出来的。**

- 规则的目标不是限制你，而是保证：**任何人 clone 下来都能复现任何人的结果。**
- 违反规则的代价不是"被批评"，而是**全组的实验结论作废**。
- 这份规则由 `scripts/check_structure.py`、pre-commit、CI 三道机器检查强制执行，
  不依赖任何人自觉。

---

## 0. 五分钟上手（新组员第一天的全部动作）

```bash
git clone <repo>                       # 1) 拿代码
conda env create -f environment.yml    # 2) 建环境（Python 3.12 + torch + 依赖）
conda activate handwash
pip install -e ".[all]"                # 3) 把本仓库装进环境
python -m pre_commit install           # 4) 装提交前钩子（= make hooks）
python -m handwash.cli doctor          # 5) 自检：缺什么、配好了没
make train-smoke                       # 6) 30 秒跑通"数据→模型→指标"链路
```

六步全部成功，才算"环境就绪"。任何一步失败，
**先看 `handwash doctor` 的输出**，不要自己猜着改代码。

---

## 1. 十条铁律（记不住全部，至少记住这十条）

| # | 规则 | 为什么 | 谁在检查 |
| --- | --- | --- | --- |
| **R1** | 只改自己负责的模块；跨模块改动先开 issue 说一声 | 避免两人同时重构同一处 | CODEOWNERS review |
| **R2** | 文件编码 UTF-8、换行 LF、缩进 4 空格（YAML/JSON 2 空格） | 否则每次 diff 都是整文件重写，无法 review | `.editorconfig` + pre-commit |
| **R3** | 一切可调参数写进 `configs/`，代码里不允许出现魔数 | 结果要能复现、要能对比 | CI 的 archcheck + review |
| **R4** | 新增依赖必须同时改 `pyproject.toml` 和 `environment.yml` | 否则别人装不上、跑出不同结果 | review |
| **R5** | 遵守分层依赖：`core` 不许 import `torch`，`io` 不许 import `models` | 契约层可单测、可复用；改了模型不影响数据 | `scripts/check_structure.py` |
| **R6** | 数据、权重、输出一律不进 Git（`data/` `models/` `outputs/` 已忽略） | 仓库会被 20GB 数据撑爆 | `.gitignore` + pre-commit 大文件检查 |
| **R7** | 新工作流必须同时加到 `Makefile` 和 CLI，命名一致 | 组员不用记两套命令 | review |
| **R8** | 不用 `print` 输出运行信息，不用 `sys.exit` 抛错 | 日志要能统一收集、错误要能统一处理 | archcheck |
| **R9** | 抛 `HandwashError` 子类，消息里说清"哪个文件/字段/期望/实际" | 报错信息就是最好的文档 | review |
| **R10** | 洗手步骤的字符串只在 `core/labels.py` 出现一次 | 否则标签错位，指标全废且极难查 | review + 单测 |

---

## 2. 改动分级：先判断你动的是哪一级

### L1 冻结区 —— 改动需要 RFC（先写文档，再改代码）

```
src/handwash/core/labels.py      标签空间（类别顺序一旦冻结，旧权重才有意义）
src/handwash/core/schema.py      跨层数据契约（字段/含义/单位）
src/handwash/core/config.py      配置结构（键名就是全组的"接口"）
src/handwash/core/protocol.py    完整性判定规则（结论口径）
src/handwash/core/registry.py    注册机制
src/handwash/errors.py  paths.py  logging.py
configs/config.yaml              公共默认配置
src/handwash/io/split.py         数据划分规则（划分一变，历史指标全部不可比）
```

**L1 改动的标准流程：**

1. 在 `docs/ARCHITECTURE.md` 第 5 节追加一条 RFC 记录（模板已给），写清：
   现状 → 为什么要改 → 影响谁 → 迁移办法 → 谁同意。
2. 至少 1 名负责人 review 通过，并在 PR 描述里粘贴 RFC 链接。
3. 同步更新 `CHANGELOG.md`（破坏性改动要标注 **BREAKING**）。
4. 如果改动影响已产出的结果，**在 PR 里明确写出"哪些历史实验需要重跑"。**

### L2 受限区 —— 需要 1 名负责人 review

```
src/handwash/io/**             src/handwash/data/**
src/handwash/models/**         src/handwash/pipelines/**
scripts/**                     tests/**
pyproject.toml  environment.yml  Makefile
.pre-commit-config.yaml  .github/**  docs/**（除 ARCHITECTURE/PROTOCOL）
```

可以自由实现，但**不得改变对外契约**：
函数签名、返回结构、配置键名、日志语义都不能变。
需要变 → 按 L1 走。

### L3 自由区 —— 随便改，只要 CI 通过

```
configs/experiments/**.yaml    你自己的实验配置
configs/models/**.yaml         新模型的配置
configs/data/**.yaml           新数据集的配置
docs/EXPERIMENTS.md            实验结果记录
outputs/  data/  models/       本地产物（不进 Git）
notebooks/**.ipynb             个人探索（提交前请清空输出）
```

**这是你 90% 的工作所在。** 想试新想法 → 新建一个
`configs/experiments/expNN_描述.yaml`，跑，把结果写进 `docs/EXPERIMENTS.md`，
不需要任何人批准。

> 判断口诀：**加文件是 L3，改接口是 L1，改实现是 L2。**

---

## 3. 加新东西的正确姿势（照抄即可）

### 3.1 加一个新模型

```python
# src/handwash/models/my_model.py
from handwash.core.registry import register_model
from handwash.models.base import BaseClassifier

@register_model("my-model")            # ① 必须注册，名字就是配置里的 model.arch
class MyModel(BaseClassifier):
    arch_name = "my-model"

    def __init__(self, num_classes: int, *, pretrained=True, dropout: float = 0.2, **kwargs):
        super().__init__(num_classes, dropout=dropout)
        ...                            # ② 自己搭网络

    @property
    def feature_dim(self) -> int: ...  # ③ 时序头需要它

    def _forward_logits(self, x):      # ④ 必须返回 (B, T, C)
        ...                            #    单帧模型也要补一个时间维

    def embed(self, x):                # ⑤ 想用于 GRU/TCN 两级训练才需要
        ...
```

然后在 `src/handwash/models/factory.py` 的 `register_all()` 里 import 你的模块，
在 `configs/models/` 加一份配置，在 `tests/unit/models/` 加一个"能前向、形状对"的测试。
**不要改 `BaseClassifier` 的接口**（那是 L1）。

### 3.2 加一个新数据集

1. **只改 `core/labels.py` 的 `_DATASET_ALIASES`**：把该数据集的原始标签名映射到规范 `Step`。
   这一步最容易出错也最重要 —— 映射错了，模型学到的东西就是错位的。
   若该数据集有新类别（不是六步之一），再在 `LABEL_SPACES` 里加一个命名空间，
   **并且只能追加在末尾**（顺序即通道号，插入会让旧权重失效 → 属于 L1 改动）。
2. 在 `configs/data/<name>.yaml` 写数据路径与 `assess` 阈值。
3. 在 `configs/config.yaml` 的 `datasets:` 段登记路径。
4. 若目录结构与已有适配器不同，改 `pipelines/prepare.py` 的适配器函数（L2）。
5. 在 `docs/DATA.md` 补：来源链接、规模、目录结构、下载命令。

### 3.3 加一个新实验

```bash
cp configs/config.yaml configs/experiments/exp05_my_idea.yaml
# 只写与默认不同的键（叠加语义，不要复制整份配置）
python -m handwash.cli train config=configs/experiments/exp05_my_idea.yaml
```

跑完把结果记进 `docs/EXPERIMENTS.md` 的那张表（**必须填 config_hash**）。

### 3.4 加一个新评估指标

1. 纯计算放 `core/metrics.py`（L1，但只**追加**函数、不改已有函数的语义）。
2. 加入 `EvalConfig.metrics` 的合法取值，并在 `pipelines/evaluate.py` 里接上。
3. 加单测：给一个能手算的例子，断言具体数值。

---

## 4. 目录与命名约定

```
src/handwash/
  paths.py logging.py errors.py     L0 基础设施（只依赖标准库）
  core/                             L1 契约：schema / labels / config / metrics / protocol
  io/                               L2 落盘与解码：manifest / split / video / utils
  data/                             L2 数据集与预处理
  models/                           L3 模型（唯一允许 import torch/ultralytics 的地方）
  pipelines/                        L4 流程编排：prepare / train / evaluate / infer / assess
  cli.py cli_doctor.py              L4 命令行入口
scripts/                            薄封装脚本（多步组合、演示用）
configs/                            config.yaml + data/ + models/ + experiments/
docs/                               文档
tests/                              与 src/ 分层对应的测试
data/ models/ outputs/              产物（不进 Git）
```

命名约定（机器检查 + review 同时把关）：

| 对象 | 约定 | 例子 |
| --- | --- | --- |
| 模块/函数/变量 | `snake_case` | `build_dataset`、`macro_f1` |
| 类 | `PascalCase` | `TemporalClassifier` |
| 常量 | `UPPER_SNAKE` | `CANONICAL_STEPS` |
| 私有 | 前缀 `_` | `_forward_logits` |
| 配置键 | `snake_case`，与 dataclass 字段**逐字一致** | `train.epochs` |
| 类别名 | 只出现在 `core/labels.py` | `step_1_palm_to_palm` |
| 测试文件 | `test_<被测模块>.py` | `test_protocol.py` |
| 实验配置 | `expNN_描述.yaml` | `exp02_yolo26n_gru.yaml` |

---

## 5. 提交与分支（每天都会用）

### 分支

```
main            永远可运行、永远通过 CI。禁止直接 push（pre-commit 会拦）
feat/<模块>/<简述>    新功能，如 feat/models/tcn-head
fix/<模块>/<简述>     修 bug，如 fix/io/manifest-dup-frame
exp/<你的名字>/<编号>  跑实验（通常只改 configs/experiments/）
docs/<简述>          只改文档
```

### 提交信息（Conventional Commits，中文描述）

```
<类型>(<范围>): <一句话说明>

范围取值：core / io / data / models / pipelines / configs / docs / tests / scripts
类型取值：feat / fix / refactor / docs / test / chore / perf / BREAKING

例：
feat(models): 加入 TCN 时序头并注册 tcn 架构
fix(core): 修正时长判定在 total_wash_s=0 时的除零
docs(configs): 说明 YOLO 适配器必须用 zero_one 归一化
BREAKING(core): 标签空间 pskuss 新增 other 类别（旧权重需重训）
```

### 提交前必跑（三选一，推荐第一个）

```bash
python -m pre_commit run --all-files    # 快：格式 + 结构检查 + 快速单测
make check                              # 全：lint + archcheck + typecheck + test
python -m pytest -m "not integration and not slow and not gpu"
```

### PR 检查清单（复制到 PR 描述里逐条打勾）

```markdown
- [ ] 我改动的文件属于我负责的模块，或已获得负责人同意
- [ ] 我没有修改任何 L1 冻结文件（若有，已附 RFC 链接：____）
- [ ] 所有新增参数都进了 configs/，代码里没有新增魔数
- [ ] 新增/修改的模块都有对应测试（tests/ 下同名文件）
- [ ] `python -m pre_commit run --all-files` 通过
- [ ] 我在本机跑过受影响的最小流程（贴命令与结果：____）
- [ ] 若改了模型/数据，已记录实验到 docs/EXPERIMENTS.md（含 config_hash）
- [ ] 若影响历史结果，已在 PR 里列出"需要重跑的实验"
- [ ] 没有提交数据/权重/输出（`git status` 干净）
```

---

## 6. 常见错误与正确处理（看到报错先查这里）

| 现象 | 真正原因 | 正确处理 |
| --- | --- | --- |
| `ConfigError: 出现未知配置键` | 配置键拼错，或把新参数写进了 YAML 但没加到 dataclass | 核对 `core/config.py` 的字段名；新参数走 RFC |
| `DataLeakageError` | 同一段原始视频落进了两个 split | 检查 `split.group_key=original_video`；**必须先划分再抽帧** |
| `ManifestError: 缺少必需列` | 用了旧版 manifest | 删掉旧的 `data/processed/*/manifest.csv` 重跑 `prepare` |
| 准确率异常高（>0.98） | 极可能数据泄漏，或测试集进了训练 | 检查划分；参考 `docs/DATA.md` 的"自检清单" |
| 准确率异常低 | 归一化用错（YOLO 用 imagenet）、标签映射错位 | 跑 `handwash doctor`；核对 `core/labels.py` |
| 找不到 `handwash` 模块 | 没 `pip install -e .`，或脚本缺 `from _bootstrap import ...` | 二选一修好 |
| 别人跑不出我的结果 | 没固定 seed / 没记录 config_hash | 用 `runtime.seed`，实验记录里填 `config_hash` |
| 显存不足 | clip 模式 batch 含义不同（窗口数 × 窗口长度） | 调小 `train.batch_size` 或 `model.temporal.window` |
| Windows DataLoader 报错 | 多进程 + 中文路径 | `runtime.num_workers: 0` |

---

## 7. 文档与记录的义务

**改了东西不写文档 = 没改。**

| 你改了什么 | 必须更新 |
| --- | --- |
| 新增/修改配置键 | `docs/CONFIG.md`（逐键说明） |
| 新增数据集或标注结构 | `docs/DATA.md` |
| 修改判定阈值或口径 | `docs/PROTOCOL.md` |
| 跑了实验（无论成功失败） | `docs/EXPERIMENTS.md`（含 `config_hash`） |
| 破坏性改动 | `CHANGELOG.md`（标 `BREAKING`）+ RFC |
| 新的分层/架构决定 | `docs/ARCHITECTURE.md` 第 5 节 RFC |

---

## 8. 遇到规则妨碍你的时候

规则是为人服务的。如果你觉得某条规则挡住了正确的事：

1. **不要偷偷绕过**（例如把 magic number 藏进函数默认值、把测试改成 skip）。
2. 在 PR 里写明"我认为 R× 在这里不适用，因为……"。
3. 由负责人决定是**豁免一次**（在 `scripts/check_structure.py` 的
   `ALLOWED_EXCEPTIONS` 里逐文件登记理由）还是**改规则**（走 L1 RFC）。

**唯一不可协商的两条：数据泄漏（R6 相关的划分规则）与标签空间唯一性（R10）。**
这两条一旦破，全组的时间都白花。
