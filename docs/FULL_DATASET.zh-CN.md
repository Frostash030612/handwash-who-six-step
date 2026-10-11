# 全量数据训练方案（2026-10-10）

## 哪些数据进入训练

| 来源 | 原始规模 | 角色 | 原因 |
| --- | ---: | --- | --- |
| [PSKUS](https://zenodo.org/records/4537209) | 3,185 段 / 11 个 zip | 训练、验证、内部测试 | 有多位逐帧标注者；使用 Annotator1 与 Annotator2 一致的帧 |
| [Jurmala](https://zenodo.org/records/5808764) | 2,427 段 / 4 个 zip | 训练、验证、内部测试 | 几乎所有视频有双人标注；使用 Annotator_1 与 Annotator_2 一致的帧 |
| [METC](https://zenodo.org/records/5808789) | 212 段 / 3 个 zip | **独立跨场景测试** | 公开版只有一位实时标注者，不能冒充双标注训练样本 |
| [Kaggle 整理版](https://github.com/edi-riga/handwash) | 本地校验为 300 段 / 1 个 tar | 独立补充测试、原型 | 七类视频已整理，但没有同等双人逐帧标注 |

组员自录视频仍是**最终自验证集**，不能用公开数据代替。合成样本仅供代码冒烟测试，不进入结果表。

## 统一规则

1. 原始视频为划分单位：70% train / 15% val / 15% test，同一视频的帧不可跨集。
2. 两位标注者对同一时间帧的动作编号必须相同；缺少其中一位或时间轴不齐的整段视频不进入训练。
3. 一致帧的前后各 7 个**原始标注帧**内必须保持同一动作，去掉动作切换边界。该掩码必须一直传到最终 `manifest.csv`；2026-10-10 修复了此前扫描通过、抽帧时又取回原始标签的问题。
4. 最终分类用七类：`0_other` 和 WHO 第 1–6 步。原始 `movement_code=7`（纸巾关水龙头）投影到 `0_other`。`other` 是“非六步动作”的合并类，**不等于单纯没有洗手**。
5. 项目里现成的 8 段 PSKUS 演示视频若通过上述筛选，则从训练/验证中移到 test，防止现场演示用到训练样本。METC、Kaggle、自录都不混入这份七类训练集。

## 本地文件与命令

```bash
python scripts/download_data.py --dataset pskuss --all
python scripts/download_data.py --dataset jurmala --all
python scripts/download_data.py --dataset metc --all

python scripts/download_data.py --dataset pskuss --extract
python scripts/prepare_data.py config=configs/data/pskuss.yaml
python scripts/download_data.py --dataset jurmala --extract
python scripts/prepare_data.py config=configs/data/jurmala.yaml
python scripts/build_full_hospital_dataset.py

# 项目内训练七类 YOLO 逐帧模型：
python scripts/train_model.py config=configs/experiments/full_hospital_yolo_frame.yaml
```

七类 Ultralytics 目录在 `data/processed/hospitals_full_7class/ultralytics/{train,val,test}/{class}/`；项目内训练使用同目录的 `manifest.csv` 与 `frames/`。两套目录用硬链接指向同一图像，不会重复占用图像字节。不要在两家医院数据**下全且 MD5 校验通过前**运行组合脚本；它会主动拒绝不完整来源。

磁盘注意：三个 Zenodo 原始压缩包合计约 35 GiB，再完整解压并抽帧会接近本机空余空间。若空间紧张，应按顺序处理一家医院、验证 `manifest.csv` 与帧数量后，只清理这家已可从 zip 重建的 `extracted/` 目录和对应 `.extracted` 标记，再处理另一家；**原始 zip 和已生成的帧不要删**。

Kaggle 整理版已从上游 GitHub 下载为 `data/raw/kaggle/kaggle-dataset-6classes.tar`，大小 1,297,551,360 字节；SHA-256：`3f9e1eef50fca0b5359650f4aff5c3906561525ee0b230c763276ee6ed753436`。它不包含双标注，故不纳入上述全量训练。
