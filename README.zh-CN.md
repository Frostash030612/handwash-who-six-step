> [English](/README.md) | **中文**

# 六步洗手动作识别与完整性评估

> 基于视频的 WHO 六步洗手法动作识别系统：逐帧判断正在做哪一步，
> 并检查**漏步、顺序异常、动作时长不足**。项目不做疾病诊断，只做动作规范性分析。

- **主模型**：YOLO26n-cls（Ultralytics 分类模型）+ GRU/TCN 时序模块
- **基线**：MobileNetV2（对照开源基线）、YOLOv8n-cls
- **数据**：PSKUS（主）、METC（跨场景）、Kaggle（快速原型）、组员自采视频（最终验证）

---

## 目录

1. [五分钟上手](#1-五分钟上手)
2. [现在就能跑的三条命令](#2-现在就能跑的三条命令)
3. [项目中有什么](#3-项目中有什么)
4. [改代码之前必须读](#4-改代码之前必须读)
5. [常见问题](#5-常见问题)
6. [文档索引](#6-文档索引)

---

## 1. 五分钟上手

```bash
git clone <repo-url> && cd "Group Project PRS"

conda env create -f environment.yml     # Python 3.12 + PyTorch + 依赖
conda activate handwash
pip install -e ".[all]"                 # 把本仓库装进环境（可编辑模式）
python -m pre_commit install            # 装提交前钩子

python -m handwash.cli doctor           # 自检：依赖、配置、数据、GPU、模型权重
```

`doctor` 全绿（或只剩不阻塞的 WARN）就算环境就绪。
**Windows 用户**也可以直接跑 `pwsh scripts/setup_windows.ps1`，它把上面几步做完。

> 没有 conda？装 Miniconda 即可（<https://docs.conda.io/en/latest/miniconda.html>）。
> 不想用 conda？可以 `python -m venv .venv` + `pip install -e ".[all]"`，
> 但请把结果记进 `docs/EXPERIMENTS.md`，并注意 `environment.yml` 仍是团队基准。

## 2. 现在就能跑的三条命令

```bash
# ① 30 秒冒烟：不需要任何数据集，验证"数据→模型→指标"链路
make train-smoke

# ② 对一段视频做完整性评估，输出中文报告（漏步 / 乱序 / 时长）
python -m handwash.cli assess --video data/external/self_recorded/full/demo.mp4

# ③ 在 Kaggle 小数据上跑通真实训练（需先下载数据，见 docs/DATA.md）
python -m handwash.cli prepare config=configs/data/kaggle.yaml
python -m handwash.cli train  config=configs/experiments/exp01_baseline_frame.yaml
python -m handwash.cli evaluate
```

产物全部落在 `outputs/<run_name>/`，其中：

```
resolved_config.yaml     本次实验的完整配置 + config_hash（写报告必须引用）
history.json / run_log.jsonl   逐 epoch 指标
models/best.pt           按验证集 Macro-F1 选出的最佳权重
eval/eval_*.json         各 split 的指标
eval/confusion_matrix_*.png    混淆矩阵
assess/<clip>.md         中文完整性报告（可直接贴进作业）
```

## 3. 项目中有什么

```
src/handwash/
  core/       契约层：标签空间、数据结构、配置、指标、WHO 完整性规则
  io/         视频解码、manifest、按原始视频-safe 划分
  data/       数据集、预处理与增强、合成数据（冒烟用）
  models/     YOLO26n-cls 适配器、MobileNetV2/ResNet 基线、GRU/TCN 时序头
  pipelines/  prepare / train / evaluate / infer / assess 五个流程
  cli.py      命令行入口（handwash <子命令>）
scripts/      薄封装脚本 + 结构体检 + Windows 一键初始化
configs/      config.yaml + data/ + models/ + experiments/
docs/         数据卡、判定口径、配置参考、实验记录、架构、RFC
tests/        与源码分层对应的单元测试
```

三条命令的区别（别再混淆）：

| 命令 | 干什么 | 何时用 |
| --- | --- | --- |
| `handwash prepare` | 扫描 → **按视频划分** → 抽帧 → 生成 manifest | 拿到新数据后 |
| `handwash train` | 训练并保存 checkpoint | 改模型/配置后 |
| `handwash evaluate` | 在 val/test/external 上算指标 | 训练后 |
| `handwash assess` | 视频 → **漏步/乱序/时长**报告 | 演示与最终验证 |

## 4. 改代码之前必须读

**这个仓库有明确的修改规则，不是建议。**

- [`docs/RULES_CARD.md`](docs/RULES_CARD.md) —— **一页速查卡**（建议打印贴桌上）：
  三条命令起步、我该改哪一级、十条铁律、四个"绝对不能"、PR 检查清单。
- [`CONTRIBUTING.md`](CONTRIBUTING.md) —— **完整规则**。十条铁律、改动分级
  （L1 冻结 / L2 受限 / L3 自由）、提交规范、PR 检查清单、常见错误对照表。
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) —— 分层图、五个关键契约、
  RFC 流程、扩展指南。

一句话版本：**加文件是 L3（随便改），改实现是 L2（要 review），
改接口是 L1（要写 RFC）**。`handwash` 的 `core/` 目录属于 L1。

规则由三道机器检查强制执行，不靠自觉：

```bash
python scripts/check_structure.py       # 分层依赖 / 硬编码路径 / print
python -m pytest -m "not integration and not slow and not gpu"
python -m pre_commit run --all-files
```

## 5. 常见问题

<details>
<summary><b>训练时提示"本次使用合成数据"？</b></summary>

说明真实 manifest 不存在，代码自动回退到合成数据以验证链路。
**合成数据的指标不能写进报告。** 先跑 `handwash prepare`，见 `docs/DATA.md`。
</details>

<details>
<summary><b>准确率异常高（>0.98）？</b></summary>

几乎一定是数据泄漏：同一段原始视频的帧同时进了训练集和测试集。
检查 `split.group_key` 是否为 `original_video`，并确认流程是
**先划分、后抽帧**。`io/manifest.py` 与 `io/split.py` 各有校验会直接报错。
</details>

<details>
<summary><b>提示 ultralytics 缺失 / yolo26 权重下载失败？</b></summary>

主模型需要 `ultralytics`：`conda env update -f environment.yml`。
若 `yolo26n-cls.pt` 尚不可用，用现成权重先跑：
`python -m handwash.cli train model.arch=yolon-cls model.pretrained=yolov8n-cls.pt`
**换权重只改配置，不要改代码**，并在报告里写明实际使用的权重。
</details>

<details>
<summary><b>`ConfigError: 出现未知配置键`？</b></summary>

配置键拼错，或你写了新参数但没加到 `core/config.py`。
这是刻意设计：拼错的键**不会被静默忽略**。对照 `docs/CONFIG.md`。
</details>

<details>
<summary><b>Windows 上 DataLoader 报错？</b></summary>

设 `runtime.num_workers: 0`（在 `configs/local.yaml` 里覆盖，不要改公共配置）。
</details>

<details>
<summary><b>我需要提交数据或权重吗？</b></summary>

**不需要也不允许。** `data/`、`models/`、`outputs/` 已在 `.gitignore` 中。
数据获取方式见 `docs/DATA.md`；权重通过 `outputs/<run>/models/` 或共享盘传递。
</details>

## 6. 文档索引

| 文档 | 内容 |
| --- | --- |
| [`docs/PROJECT_PLAN_4_WEEK.md`](docs/zh/PROJECT_PLAN_4_WEEK.md) | **四周团队计划（4 人）**：角色分工、逐周任务、计分追溯、风险清单 |
| [`docs/PROJECT_PROPOSAL.md`](docs/zh/PROJECT_PROPOSAL.md) | **提案内容包**：模板每一栏的可粘贴文案（截止 9/30） |
| [`docs/RULES_CARD.md`](docs/RULES_CARD.md) | **一页速查卡**：改代码前先看这一页 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **修改规则（必读）**：铁律、分级、提交与 PR 流程 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层图、五个契约、RFC 流程、扩展指南 |
| [`docs/DATA.md`](docs/DATA.md) | 五个公开数据集的来源、结构、下载与准备 |
| [`docs/CONFIG.md`](docs/CONFIG.md) | 配置逐键说明与常见改法 |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | 完整性判定口径：漏步/乱序/时长的精确定义 |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | 实验记录表（每跑一次都要填） |
| [`docs/SELF_RECORDING.md`](docs/SELF_RECORDING.md) | 自采视频拍摄规范与标注表 |
| [`CHANGELOG.md`](CHANGELOG.md) | 破坏性改动与迁移说明 |
| [`tests/README.md`](tests/README.md) | 测试怎么跑、标记含义、加测试的规则 |

---

## 参考文献

1. Hand-Washing Video Dataset Annotated According to WHO Guidelines (2021) —
   <https://doi.org/10.3390/data6040038>（PSKUS 数据集论文）
2. Towards Automated Hand Hygiene Assessment in Hospitals (2022) —
   <https://www.edi.lv/wp-content/uploads/2022/12/main.pdf>
3. Learning to Recognize Hand-Washing Activities in Hospital Settings (2020) —
   <https://arxiv.org/abs/2011.11383>
4. Hand Washing Gesture Recognition Using a Synthetic Dataset (2025) —
   <https://doi.org/10.3390/jimaging11070208>
5. 开源基线代码 — <https://github.com/edi-riga/handwash>
6. WHO How to Handwash — <https://www.who.int/publications/m/item/how-to-handwash>
7. Ultralytics YOLO26 — <https://docs.ultralytics.com/models/yolo26>
