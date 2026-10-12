# YOLO、YOLO+GRU 与 VideoMAE：整段视频公平对照

这里的 YOLO 是逐帧七类分类基线；YOLO+GRU 用同一 YOLO 权重提取逐帧特征，再训练独立的时序头。VideoMAE 直接看连续 16 帧。旧演示权重 `exp.pt` 和 `exp_temporal_head.pt` 保留，但不能把旧 GRU 头接到新 YOLO 权重上。

## 共同口径

- 数据仅为 PSKUS DataSet1/2/3/4，原始视频按 `data/processed/pskuss_ds1234_yolo/video_splits.csv` 锁定：430/92/93 段 train/val/test。不能重新计算视频划分。
- 两名指定标注员的中心帧标签一致且时间戳对齐时，才纳入评分。`other` 包括非洗手和其他/非标准动作，并不只代表“没洗手”。
- 两种模型都读取完整视频，按 5 fps 采样，并在同一组 16 帧窗口中心（4 个采样帧的步长）评分。YOLO 只看该中心单帧；GRU 还看此前的 YOLO 特征；VideoMAE 看整个 16 帧窗口。输入信息量不同，所以这是**方案对照**，不是只换网络结构的消融。
- 报未平滑结果，也报使用相同参数候选集合、各自在验证集选定参数后的结果。测试集从不用于调参。由于相邻窗口重叠，5599 个中心点不是 5599 个独立视频；同时报 93 段视频的序列指标。
- 单独报动作切换附近的 Macro-F1。综合 Macro-F1 上升不保证边界预测改善。序列 exact-match 只是窗口中心标签折叠后的匹配率，不等于人工判定的“六步合规率”。

## 当前实验

主脚本：`scripts/compare_full_video_models.py`。它强制校验三个模型的测试视频 ID、中心帧编号、标签和有效样本掩码一致；每段的原始概率与摘要分别保存。新 GRU 的训练脚本是 `scripts/train_current_yolo_gru.py`，权重保存于 `outputs/pskuss_ds1234_yolo_gru/head.pt`，并记录当前 YOLO checkpoint 的 SHA-256。

VideoMAE 的改进实验用现有最佳权重继续微调，目标为 `center_stable` 窗口：**中心标签**仍须双人一致且前后 7 原始帧稳定，但输入窗口允许包含附近的不同动作。这让训练输入更接近完整视频的切换场景，又不把有争议的边界帧当训练标签。旧的 `full_stable` 构造仍为默认，不覆盖旧权重。新旧 VideoMAE 只根据验证集上的完整视频 Macro-F1 选最终模型，再看锁定测试集。

所有产物分开保存；不要把此前重新计算划分所得的整段视频指标与本实验放在同一张“公平对照”表中。

在已经备齐相应权重和原视频的训练机上，可按以下顺序复现。运行前先确认 615 段划分表与权重的指纹；输出目录不能指向旧演示权重。

```bash
python scripts/train_videomae_pskus.py --full-pskus \
  --datasets DataSet1,DataSet2,DataSet3,DataSet4 \
  --split-reference data/processed/pskuss_ds1234_yolo/video_splits.csv \
  --checkpoint outputs/pskuss_ds1234_videomae/best_model \
  --output outputs/pskuss_ds1234_videomae_center_stable \
  --window-label-policy center_stable --other-to-wash-ratio 1.0 \
  --epochs 8 --freeze-backbone-epochs 0 --skip-test-eval

python scripts/train_current_yolo_gru.py \
  --split-reference data/processed/pskuss_ds1234_yolo/video_splits.csv \
  --model outputs/pskuss_ds1234_yolo/weights/best.pt \
  --output outputs/pskuss_ds1234_yolo_gru

python scripts/compare_full_video_models.py \
  --split-reference data/processed/pskuss_ds1234_yolo/video_splits.csv \
  --models yolo,videomae,videomae_center,yolo_gru
```

## 2026-10-11 首轮锁定测试集结果

评估配置：原视频划分表 SHA-256 为 `b9e0b066ca8cf8694408027fd32db14609ff4ec12a6ca580c73cfa348cd551f2`；YOLO 权重 SHA-256 为 `7ec0e5ec7a7c3475c0fb9b0a1c745c847057ad2be1ccbbd6fda7764ddef20613`，原 VideoMAE 权重 SHA-256 为 `975564856d7befd70691603d9e351bc3fa4f0dac8667b7c09d48b6717de35660`。同 93 段测试视频、5599 个双标注一致窗口中心、其中 1068 个接近动作切换。

| 模型 | 未平滑 Macro-F1 | 验证集选定平滑后 Macro-F1 | 切换附近 Macro-F1 | 序列完全匹配 |
| --- | ---: | ---: | ---: | ---: |
| YOLO 基线 | 0.8871 | 0.9025 | 0.8327 | 38/93 |
| 原 VideoMAE | 0.7634 | 0.7721 | 0.5842 | 20/93 |

YOLO 的平滑后总分提高，但切换附近 Macro-F1 从未平滑的 0.8481 降为 0.8327；不能把平滑描述成所有场景都改善。原 VideoMAE 在训练时只见整段稳定窗口，在完整视频的混合动作输入中明显掉分，因此另做了上述 `center_stable` 训练实验。以上数值不是“六步合规准确率”，也不是自采视频的跨场景性能。

按原视频成对重采样 1000 次，YOLO 相对原 VideoMAE 的测试 Macro-F1 差值为 0.1304，95% 重采样区间为 0.1051～0.1620。区间只以视频为重采样单位；同一医院环境、人员或水池可能相关，不能解释为跨医院泛化置信区间。

## 最终公平对照（2026-10-12）

所有四行仍是同一批 93 段锁定测试视频、5599 个可评估中心，其中 1068 个靠近动作切换。后处理参数只在 92 段验证视频上选择；改进版 VideoMAE 是否替代原版也只按完整视频验证集 Macro-F1 选择（原版 0.7878、改进版 0.8823）。新 GRU 头与本轮 YOLO `best.pt` 的 SHA-256 绑定，不使用旧演示头。

| 方案 | 中心准确率 | 中心 Macro-F1 | 切换附近 Macro-F1 | 序列完全匹配 |
| --- | ---: | ---: | ---: |
| YOLO 逐帧基线 | 0.9484 | 0.9025 | 0.8327 | 38/93 |
| 原 VideoMAE | 0.8903 | 0.7721 | 0.5842 | 20/93 |
| 改进 VideoMAE（`center_stable`） | 0.9339 | 0.8745 | 0.7944 | 39/93 |
| YOLO + 新 GRU | **0.9598** | **0.9271** | **0.9006** | **48/93** |

表格列顺序为中心准确率、中心 Macro-F1、切换附近 Macro-F1、序列完全匹配。准确率按所有可评分中心计数，Macro-F1 对七类等权；`other` 占 3904/5599 个中心（69.7%），因此单看准确率会被多数类主导。序列完全匹配是把一致标注中心上的标签折叠后，预测序列与参考序列完全相同的视频数；它不是人工审核的完整性分数。

### 各类别 F1（验证集选定后处理）

| 类别 | 一致中心数 | YOLO | YOLO + GRU | 改进 VideoMAE |
| --- | ---: | ---: | ---: | ---: |
| Other／非步骤动作 | 3904 | 0.9734 | **0.9772** | 0.9664 |
| 第 1 步：掌心相对 | 311 | 0.8285 | **0.8736** | 0.7713 |
| 第 2 步：手背对掌心 | 366 | 0.9193 | **0.9550** | 0.9339 |
| 第 3 步：十指交叉 | 294 | 0.8657 | **0.8927** | 0.8408 |
| 第 4 步：指背互扣 | 283 | 0.9260 | **0.9431** | 0.9113 |
| 第 5 步：拇指旋转 | 229 | 0.8798 | **0.9091** | 0.8162 |
| 第 6 步：指尖旋转 | 212 | 0.9245 | **0.9393** | 0.8816 |

GRU 在七类上都有最高的测试 F1。VideoMAE 相比 YOLO 的主要短板是第 1、3、5 步；这些类别也比 `other` 少得多。每类分数是 93 段视频合并后的中心帧指标，不代表每段视频都有同样表现。

### 一个共同测试视频的错误示例

测试视频 `DataSet1/2020-06-30_13-08-45_camera101` 的双人一致参考序列为 `1→3→2→5→4`。三个模型在这段视频上的结果如下：

| 模型 | 中心准确率 | Macro-F1 | 预测序列 | 主要问题 |
| --- | ---: | ---: | --- | --- |
| YOLO | 84.8% | 0.5204 | `1→3→2→4` | 漏掉第 5 步 |
| YOLO + GRU | **95.7%** | **0.7905** | `1→3→2→5→4→5→4` | 额外重复第 5、4 步 |
| 改进 VideoMAE | 87.0% | 0.4811 | `1→2→4` | 漏掉第 3、5 步，并把第 3 步识别成第 2 步 |

这个例子显示 GRU 的逐中心分类明显更准，但仍可能把动作边界附近的短暂预测写进整段动作序列。单个案例用于说明错误类型，不用于代表总体模型排名。

改进 VideoMAE 相比原版 Macro-F1 增加 0.1024；按原视频配对重采样的 95% 区间为 0.0767～0.1330。YOLO+GRU 相比 YOLO 单帧基线增加 0.0247，区间为 0.0143～0.0384。YOLO+GRU 相比改进 VideoMAE 高 0.0526，区间为 0.0353～0.0727。它们描述的是当前 PSKUS 测试划分，不代表换水池或自录视频的性能。

改进 VideoMAE 权重 `model.safetensors` 的 SHA-256 为 `068726bf4d69ae6b9903c4b64c3b769c757ae00424b6536c374197fcbc7365ce`；新 GRU 头 `head.pt` 的 SHA-256 为 `eb0dfa16c286f13594458afd48d9df75fc86822398d99d1f5cde9784ee5e2189`。每个模型的逐视频、混淆矩阵、验证集参数搜索和原始预测概率都在 `outputs/pskuss_ds1234_fair_video_comparison/`。GRU 最佳训练 epoch 为 17；改进 VideoMAE 为 6。训练窗口级验证分数与上表完整视频测试分数不是同一个指标。

用于复现的 YOLO、GRU 和改进 VideoMAE 权重已归档至 [`models/`](../models/README.md)，并附带 SHA-256 校验值；原始视频、标注和预测缓存不随仓库分发。

序列完全匹配来自双标注一致中心点的动作标签折叠，不是经人工复核的六步完整性、顺序和时长合规率。真正的交付验收还需对未见过的自录视频逐段人工标注后测试。
