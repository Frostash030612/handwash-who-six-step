> [English](/README.md) | **中文**

# 六步洗手动作识别与完整性评估

> **当前实施主计划**：[数据清洗、云端 YOLO 分类训练与本地摄像头网页](docs/IMPLEMENTATION_PLAN.zh-CN.md)。
> 早期 GRU/TCN 与 PE 时序模型方案保留为可选研究实验。

> 外接摄像头持续采集画面，YOLO 图像分类器逐帧识别正在做的动作；
> 连续预测再用于检查**漏步、顺序异常、动作时长不足**。
> 项目不做疾病诊断，只做动作规范性分析。

- **主模型**：在云端用清洗后的数据训练 YOLO 分类模型，部署电脑逐帧实时推理
- **实时展示**：仅用已到达帧的概率平均稳定显示；结束后按预测时间线判断六步
- **可选实验**：MobileNetV2、GRU/TCN 与 PE 时序分段
- **数据**：PSKUS（主）、METC（跨场景）、Kaggle（快速原型）、组员自采视频（最终验证）

---

## 目录

1. [五分钟上手](#1-五分钟上手)
2. [从数据到实时摄像头](#2-从数据到实时摄像头)
3. [项目中有什么](#3-项目中有什么)
4. [改代码之前必须读](#4-改代码之前必须读)
5. [常见问题](#5-常见问题)
6. [文档索引](#6-文档索引)

---

## 1. 五分钟上手

在 **macOS（本机推理）** 上，从项目目录运行：

```bash
conda create -n handwash python=3.11 -y
conda activate handwash
python -m pip install -e ".[torch,yolo,video]"
python -m handwash.cli doctor
```

`environment.yml` 含 NVIDIA CUDA 依赖，适用于相应的 Linux 训练机，不要直接用于 macOS。
如果终端里的 `python --version` 仍显示 Python 2，说明系统 Python 排在 conda 环境之前；
把下文命令的 `python` 改成 `"$CONDA_PREFIX/bin/python"`。
在配有相应 CUDA 环境的云端 Linux 训练机，可以使用：

```bash
conda env create -f environment.yml
conda activate handwash
python -m pip install -e ".[all]"
python -m handwash.cli doctor
```

`doctor` 全绿（或只剩不阻塞的 WARN）就算环境就绪。
**Windows 用户**可参考 `pwsh scripts/setup_windows.ps1`。

> 没有 conda？装 Miniconda 即可（<https://docs.conda.io/en/latest/miniconda.html>）。
> 不想用 conda？可以 `python -m venv .venv` + `pip install -e ".[all]"`，
> 但请把结果记进 `docs/EXPERIMENTS.md`，并注意 `environment.yml` 仍是团队基准。

## 2. 从数据到实时摄像头

### 直接体验 exp.pt 演示

根目录的 `exp.pt` 是当前七类 Ultralytics 帧分类模型；若同目录存在
`exp_temporal_head.pt`，演示会自动叠加 GRU 时序头，用于直接体验摄像头网页。
用于训练的 NDJSON 导出清单不随仓库发布；正式数据请按下文从公开来源获取。
无需下载正式数据即可体验摄像头网页：

```bash
python scripts/run_camera.py --demo-exp
```

打开 `http://127.0.0.1:8765/`，连接并选择外接摄像头，然后点击“开始识别”。
页面会标明“演示”，结果保存在 `outputs/exp_demo/camera/`。可用
`--no-temporal-head` 查看纯帧模型对照。模型类别顺序已单独核对；演示报告不能当作
正式准确率或最终实验结果。
公开上传前用 `git add .` 和 `git status --short` 核对待提交文件；
只允许根目录的演示权重 `exp.pt` 与 `exp_temporal_head.pt` 入库，NDJSON 导出清单、训练权重、
`deliverables/` 和 `outputs/` 不应上传。`.gitignore` 不会清除已经提交的历史文件。

### 正式数据与云端训练

```bash
# ① 仓库不附带原始数据；下载并解压 PSKUS，再清洗、人工核对原视频/标注
python scripts/download_data.py --dataset pskuss --all --extract
# 只想先验证流程，可改用 --files DataSet4.zip --extract；该子集不能作为正式结果
# ② 按原始视频分组生成 manifest
python scripts/prepare_data.py config=configs/data/pskuss.yaml --inspect
python scripts/prepare_data.py config=configs/data/pskuss.yaml

# ③ 在云端训练逐帧 YOLO 分类模型并评估
python scripts/train_model.py config=configs/experiments/live_yolo_frame.yaml --evaluate

# ④ 把项目格式的 best.pt 与对应配置复制到部署电脑，连接外接摄像头
# ⑤ 启动网页，浏览器打开 http://127.0.0.1:8765/
python scripts/run_camera.py config=configs/experiments/live_yolo_frame.yaml \
  --checkpoint path/to/best.pt
```

摄像头页面需要已训练的项目格式 checkpoint。代码入口已经具备；
模型效果和目标电脑上的延迟仍须用正式数据与设备验收。

产物全部落在 `outputs/<run_name>/`，其中：

```
resolved_config.yaml     本次实验的完整配置 + config_hash（写报告必须引用）
history.json / run_log.jsonl   逐 epoch 指标
models/best.pt           按验证集 Macro-F1 选出的最佳权重
eval/eval_*.json         各 split 的指标
eval/confusion_matrix_*.png    混淆矩阵
assess/<clip>.md         中文完整性报告（可直接贴进作业）
camera/<session>/        实时会话的逐帧预测与最终报告
```

## 3. 项目中有什么

```
src/handwash/
  core/       契约层：标签空间、数据结构、配置、指标、WHO 完整性规则
  io/         视频解码、manifest、按原始视频-safe 划分
  data/       数据集、预处理与增强、合成数据（冒烟用）
  models/     YOLO 分类适配器；实验用的其他骨干与时序头
  pipelines/  prepare / train / evaluate / infer / assess / live
  camera_app.py + static/   本机网页服务与摄像头页面
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
| `handwash infer` | 单段视频 → 逐帧动作预测 | 检查模型逐帧输出 |
| `handwash assess` | 视频 → **漏步/乱序/时长**报告 | 演示与最终验证 |
| `handwash camera` | 外接摄像头 → 实时 YOLO 分类 → 最终报告 | 云端训练后在本机演示 |

示例：`python -m handwash.cli infer --video demo.mp4 --checkpoint outputs/run/models/best.pt`。

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

说明配置显式选择了 `dataset.name: synthetic` 做冒烟训练。真实数据的 manifest 缺失时
会报错，不会自动改用合成数据。**合成数据的指标不能写进报告。**
真实训练前先跑 `handwash prepare`，见 `docs/DATA.md`。
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

**数据和训练权重不提交；根目录的 `exp.pt` 是唯一演示例外。**
`data/`、`models/`、`outputs/` 和其他 `*.pt` 已在 `.gitignore` 中。
数据获取方式见 `docs/DATA.md`；正式模型仍需训练后通过 `--checkpoint` 指定。
</details>

## 6. 文档索引

| 文档 | 内容 |
| --- | --- |
| [`docs/IMPLEMENTATION_PLAN.zh-CN.md`](docs/IMPLEMENTATION_PLAN.zh-CN.md) | **逐步实施主计划**：数据清洗 → 云端训练 → 本地摄像头网页 |
| [`docs/PROJECT_PLAN_4_WEEK.md`](docs/zh/PROJECT_PLAN_4_WEEK.md) | 旧课程排期与提交要求存档 |
| [`docs/PROJECT_PROPOSAL.md`](docs/zh/PROJECT_PROPOSAL.md) | **提案内容包**：模板每一栏的可粘贴文案（截止 9/30） |
| [`docs/RULES_CARD.md`](docs/RULES_CARD.md) | **一页速查卡**：改代码前先看这一页 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **修改规则（必读）**：铁律、分级、提交与 PR 流程 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层图、五个契约、RFC 流程、扩展指南 |
| [`docs/DATA.md`](docs/DATA.md) | 五个公开数据集的来源、结构、下载与准备 |
| [`docs/DATA_COLLABORATION.md`](docs/zh/DATA_COLLABORATION.md) | **不上传任何东西，四个人怎么共用 2–17 GB 数据** |
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
