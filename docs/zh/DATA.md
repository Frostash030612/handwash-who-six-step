> [English](/docs/DATA.md) | **中文**

# 数据说明（DATA）

本文件是数据部分的**唯一权威说明**：数据集从哪来、目录要怎么摆、标签怎么映射、
怎么划分、以及最容易踩的三个坑。

> 报告中"数据集"一节可以直接引用本文件的结构说明，但**必须自己跑一遍
> `handwash prepare` 并把实际数字（段数 / 帧数 / 划分）填进去**，
> 不要照抄本文件的估计值。

---

## 0. 三条硬规则（违反则结论作废）

1. **先划分、后抽帧。** 划分单位必须是**原始视频**。把同一段视频的相邻帧
   同时放进训练集和测试集，准确率会虚高到 0.95+，而且没有任何报错。
   框架里的三道防线：
   - `io/manifest.py`：同一 `clip_id` 跨 split → 抛 `DataLeakageError`；
   - `io/split.py`：按 `split.group_key` 分组划分 + 划分后复核；
   - `configs/config.yaml`：`split.guard_leakage: true`。

2. **数据不进 Git。** `data/`、`models/`、`outputs/` 都已在 `.gitignore`。
   数据放在本地或共享盘，用 `HANDWASH_DATA_ROOT` 环境变量指向外部位置：
   ```bash
   set HANDWASH_DATA_ROOT=E:\datasets\handwash     # Windows
   export HANDWASH_DATA_ROOT=/mnt/data/handwash    # Linux/macOS
   ```

3. **标签映射只改一处。** 数据集原始标签名 → 规范 `Step` 的映射**只允许**
   写在 `src/handwash/core/labels.py` 的 `_DATASET_ALIASES`。
   在别处写 `if label == "Step1_water"` 会导致训练标签与评估标签悄悄错位。

---

## 1. 数据集总览

| 数据集 | 规模 | 环境 | 用途 | 优先级 |
| --- | --- | --- | --- | --- |
| **PSKUS** | 3185 段 / ~18.4 GB | 真实医院 | 主训练数据 | 必须（可先取子集） |
| **METC** | 212 段 / ~2.1 GB / 72 人 | 实验室 | 跨场景测试 | 必须 |
| **Kaggle** | 小（数百 MB） | 混合 | 快速原型 | 建议先跑 |
| **Jurmala** | 2427 段 / ~17 GB | 真实环境 | 扩展训练 | 可选 |
| **合成数据** | 自带生成 | 3D 渲染 | 冒烟测试/预训练 | 可选 |
| **自采视频** | 每人 4—6 段 | 普通洗手池 | 最终验证 | 必须 |

---

## 2. PSKUS（主数据集）

- 链接：<https://zenodo.org/records/4537209>
- 论文：<https://doi.org/10.3390/data6040038>
- 内容：真实医院环境，WHO 六步动作 + 开关水龙头 + 其他动作，**逐帧标注**。

**目录结构（框架期望）**

```
data/raw/pskuss/
  ├── step_1_palm_to_palm/        # 目录名 = 标签（可用别名，见 core/labels.py）
  │   ├── clip_0001.mp4
  │   └── clip_0002.mp4
  ├── faucet_on/
  └── ...
```

或者"每段视频一个逐帧标注 CSV"的形式：

```
data/raw/pskuss/<clip_id>.csv      # 列：frame, label[, timestamp]
```

**建议**：不要一次下全量。先取 300—500 段跑通，确认抽帧参数与标签映射正确，
再补全 —— 18.4 GB 抽帧后会占用数十 GB 磁盘。

**配置**：`configs/data/pskuss.yaml`（标签空间 10 类，`include_non_wash: true`）

---

## 3. METC（跨场景测试集）

- 链接：<https://zenodo.org/records/5808789>
- 内容：212 段、72 名参与者的实验室环境数据，帧级标签。

**正确用法（很重要）**

```
① 用 PSKUS 训练 → outputs/e4_pskuss_train/models/best.pt
② 准备 METC 的帧（configs/data/metc.yaml）
③ 用**同一个** checkpoint 在 METC 上评估：
     python -m handwash.cli evaluate --checkpoint <上面的 best.pt> --splits external
```

**不要**用 METC 训练再在 METC 上测试 —— 那就不是跨场景实验了。
报告里应给出三组数字：PSKUS test（同场景）、METC（跨场景）、自采视频（真实使用）。

**配置**：`configs/data/metc.yaml`

---

## 4. Kaggle 洗手数据集（快速原型）

- 链接：<https://www.kaggle.com/datasets/realtimear/hand-wash-dataset>
- 七分类整理版：<https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar>

**目录结构**

```
data/raw/kaggle/
  ├── Step1_water/<clip>/*.jpg      # 或直接 Step1_water/*.jpg
  ├── Step2_water/
  └── ...
```

**说明**：`Step7_water` / `not_washing` 在框架里归入 `other`，而 `kaggle`
标签空间只有六步，因此这些样本不会被计入六分类训练（见 `core/labels.py`）。

**配置**：`configs/data/kaggle.yaml`（默认 `dataset.name`，用于一键冒烟）

---

## 5. Jurmala（可选扩展）

- 链接：<https://zenodo.org/records/5808764>
- 规模：2427 段 / ~17 GB。采集规范与 PSKUS 同源。
- 用法：与 PSKUS 同为六步 6 类标签空间，可在时间与算力充足时加入训练，
  **不是完成作业的必要条件**。

---

## 6. 合成洗手数据集（可选）

- 数据与代码：<https://github.com/r-ozakar/synthetic-hand-washing>
- 论文：<https://doi.org/10.3390/jimaging11070208>
- 特点：3D 场景生成，含 RGB、深度与手部掩膜，可研究合成数据预训练。
- 建议：**只作为扩展实验**。若做，报告中必须说明"合成预训练 + 真实微调"
  与"纯真实训练"的对比，否则无法体现价值。

---

## 7. 组员自采视频（最终验证）

拍摄与标注规范见 [`SELF_RECORDING.md`](/docs/SELF_RECORDING.md)。三条要点：

1. 每位组员 4—6 段，必须覆盖：**完整正确 / 漏一步 / 换序 / 某步过短**；
2. 机位参考公开数据：双手全程清晰可见（俯拍或侧前方 45°）；
3. **必须记录人工答案**（真实步骤序列），否则无法与模型结果对比。

自采视频的评估用 `handwash assess`（完整性），而不是 `handwash evaluate`（帧级分类）——
因为自采视频的价值在于**完整流程**，人工也标不出逐帧标签。

---

## 8. 常见问题（按报错查）

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `DatasetNotFoundError` | `dataset.root` 路径不对 | 核对 `configs/config.yaml` 的 `datasets` 段；或用 `HANDWASH_DATA_ROOT` |
| `标签空间 X 无法识别标签：'yyy'` | 数据集的标签名没登记 | 在 `core/labels.py` 的 `_DATASET_ALIASES` 补映射（L1，改动要 review） |
| `DataLeakageError` | 同一原始视频跨 split | 确认 `split.group_key=original_video`；确认先划分后抽帧 |
| `manifest 缺少必需列` | 用了旧版 manifest | 删除 `data/processed/*/manifest.csv` 后重跑 `handwash prepare` |
| 抽帧阶段"视频过短，已跳过" | 视频短于 `min_frames_per_clip` | 调小该值，或检查视频是否损坏 |
| 准确率 > 0.98 | 极可能数据泄漏 | 见上；另外检查是否把 test 放进了 train |

**抽帧参数建议**：`fps: 5.0` 对动作级任务足够（30fps 的视频每 6 帧取 1 帧），
存储可降到约 1/6；`resize_hw: [256, 256]` 落盘、模型再裁到 224。

---

## 9. 划分与统计（自检清单）

跑完 `handwash prepare` 后，检查 `data/processed/<dataset>/split_report.json`：

- [ ] 三个 split 的段数比例接近 0.7 / 0.15 / 0.15（小数据集会有偏差，属正常）；
- [ ] 任一原始视频**没有**同时出现在两个 split（框架已强制）；
- [ ] 每个 split 里六步都有样本（`label_frames` 不应有 0）；
- [ ] `val` 不为空（否则无法早停选模）；
- [ ] 每段视频的帧数与时长合理（异常值通常意味着解码失败或标注错位）。

把这份检查结果写进报告的数据部分，比任何形容词都有说服力。
