# VideoMAE 洗手动作识别实验

> 本页记录早期 DataSet1/4 的试验方法与结果。DataSet1/2/3/4 的最终
> YOLO、YOLO+GRU、原版及改进 VideoMAE 同口径对照，请看
> [公平整段视频对照](FAIR_VIDEO_COMPARISON.zh-CN.md)。两处测试划分不同，不可直接混排指标。

本实验使用 VideoMAE 对连续视频片段做七类动作分类，并与项目逐帧 YOLO26-cls + GRU 方案比较。它使用公开 VideoMAE 预训练权重进行微调，不从零预训练。

## 数据与标注

- 输入为 PSKUS 原始视频及逐帧 CSV 标注，不使用已经抽平的分类图片。
- 只读取 DataSet1 的 Annotator2/3 与 DataSet4 的 Annotator1/2；两位标注必须时间轴对齐且动作一致。
- 任一标签切换前后各 7 个原始标注帧作为边界排除区。
- 训练标签为 `other` + WHO 六步；movement code 0、7 以及非洗手帧合并到 `other`。
- 每个样本为 16 帧、5 fps 的连续窗口；同一原始视频只属于 train、val、test 中的一个划分。

按当前参数检查，153 段双标注视频可组成 4,937 个窗口。`other` 占多数（训练集 3,034 / 3,401 个窗口），训练使用类别加权损失；报告需要同时说明类别不均衡，不能只看总体准确率。

## AutoDL 环境

推荐使用单张 CUDA GPU；3080 Ti 12GB 可运行，首轮 batch size 设为 1、梯度累积 4、FP16。PyTorch 镜像中保留与镜像匹配的 CUDA 版本。

```bash
python -m pip install -e .[video]
python -m pip install 'transformers>=4.45,<5' scikit-learn

# 如果 AutoDL 无法直接访问 Hugging Face，可用镜像下载预训练权重
export HF_ENDPOINT=https://hf-mirror.com

# 先核对双标注窗口数量和类别分布，不解码、不训练
python scripts/train_videomae_pskus.py \
  --root data/raw/pskuss/extracted --inspect

# 运行微调
python scripts/train_videomae_pskus.py \
  --root data/raw/pskuss/extracted \
  --output outputs/videomae_pskus \
  --batch-size 1 --grad-accum 4 --epochs 20
```

如果显存不足，可先将 `--num-frames` 改成 8；此时仍须在报告中注明输入窗口长度发生变化。

## 整段视频测试

整段视频按 5 fps 采样，用 16 帧窗口、4 帧步长滑动推理。每个窗口的预测落在窗口中心时间点，随后合并为连续动作时间轴。评估时保留动作切换附近的窗口，只要求中心帧的两位标注员标签一致，因此它比只测试稳定动作片段更接近实际使用。

```bash
python scripts/evaluate_videomae_full_videos.py \
  --model outputs/videomae_pskus/best_model \
  --root data/raw/pskuss/extracted \
  --output outputs/videomae_full_video_test \
  --split test --smooth-windows 3 --batch-size 4
```

概率平滑宽度先在验证集选择：1、3、5 个窗口的 Macro-F1 分别为 0.7150、0.7242、0.7169，因此固定使用 3，不根据测试集调参。最终 23 段测试视频产生 1,357 个双人一致窗口中心，Accuracy 为 0.8732，Macro-F1 为 0.7500。先前稳定窗口测试的 Accuracy 0.9656、Macro-F1 0.8776 不能替代这个整段视频结果，因为它排除了动作切换边界。

## 产物

- `outputs/videomae_pskus/best_model/`：可移至本地加载的 VideoMAE 权重、配置、预处理器和类别映射。
- `history.json`：每轮训练损失和验证指标。
- `test_metrics.json`：按原始视频隔离的测试集 Accuracy、Macro-F1 和分类报告。
- `frame_cache/`：按原始视频缓存抽取帧，避免每轮反复解码；可在确认模型和指标已备份后删除并重建。
- `outputs/videomae_full_video_test/summary.json`：整段视频汇总指标、逐类报告和混淆矩阵。
- `outputs/videomae_full_video_test/per_video.csv`：每段测试视频的指标与预测动作顺序。
- `outputs/videomae_full_video_test/timelines/`：23 段视频各自的动作时间轴 JSON。

当前已经完成完整视频滑窗推理和动作时间轴输出。若要给出最终的漏步、错序与时长合规结论，还需把时间轴接到现有 assess 规则层；窗口分类分数本身不等于整套洗手流程合规率。

## 同口径 YOLO `exp.pt` 对照（2026-10-11）

把本地 `exp.pt` 上传至 AutoDL，使用 `scripts/evaluate_yolo_full_videos.py` 在同一原始视频划分、同一 5 fps 采样和同一窗口中心时间点评估。YOLO 输入单张原始视频帧；VideoMAE 输入连续 16 帧。两者都只在中心帧双人标注一致时计分，动作切换附近的样本没有被整体剔除。`exp.pt` 的训练脚本记录的按视频划分规则与 VideoMAE 一致，但两个模型的训练样本构造和输入信息量不同，结果是模型方案对照，不是只改变网络结构的消融实验。

```bash
python scripts/evaluate_yolo_full_videos.py \
  --model exp.pt --root data/raw/pskuss/extracted \
  --output outputs/yolo_full_video_test
```

离线平滑先对相邻帧的七类概率取中心滑动平均，再把不足 `min_run` 帧的孤立类别段并入概率更支持的相邻段；确认是真正的新动作后，边界回填到该动作首次出现的帧。只在 23 段验证视频中比较平均宽度 1/3/5 帧、最短段 1/2/3 帧，选出 `5 + 3`；之后固定参数跑 23 段测试视频。按 5 fps 计算，这分别约是 1 秒证据窗口和 0.6 秒最短段。因为是中心窗口和回填，**这是离线视频方案，不能直接声称实时零延迟**。

| 测试方式 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| YOLO 单帧，未平滑 | 0.8276 | 0.7128 |
| YOLO 单帧，验证集选定平滑 | 0.8696 | 0.7817 |
| VideoMAE 16 帧窗口，验证集选定平滑 | 0.8732 | 0.7500 |

三行使用同样的 23 段测试视频、1,357 个可评估中心点。YOLO 在接近双人一致标签切换的位置（本脚本定义的 248 个中心点），平滑前/后的 Accuracy 为 0.7056/0.7460，Macro-F1 为 0.6907/0.7490。这说明短时抖动有所改善，但不是洗手流程正确率；例如 `other` 占 964/1,357 个中心点，且第 5 步的 F1 仍只有 0.5345。自录视频的跨场景效果也须另测。

详细结果见 `outputs/yolo_full_video_test/summary.json` 与 `per_video.csv`。若需要实时摄像头平滑，应改用只看过去帧的概率平均与“连续确认新类别”机制，并接受约 0.4–0.6 秒的切换提示延迟；不能把上述离线数字直接作为实时性能。
