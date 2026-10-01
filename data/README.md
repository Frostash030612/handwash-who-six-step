# 数据与产物目录（占位说明）

本目录**不进 Git**（见 `.gitignore`），只保留结构说明，方便新组员知道东西该放哪。

```
data/
  raw/          公开数据集的原始文件（PSKUS / METC / Kaggle / Jurmala）
  interim/      中间产物：合成数据、临时抽帧结果
  processed/    正式产物：<dataset>/frames/ + <dataset>/manifest.csv + split_report.json
  external/     组员自采视频（self_recorded/full/ 放整段视频）
```

## 命名与来源

| 子目录 | 内容 | 获取方式 |
| --- | --- | --- |
| `raw/pskuss/` | PSKUS 医院洗手数据集 | <https://zenodo.org/records/4537209> |
| `raw/metc/` | METC 跨场景数据集 | <https://zenodo.org/records/5808789> |
| `raw/kaggle/` | Kaggle 七分类数据集 | <https://www.kaggle.com/datasets/realtimear/hand-wash-dataset> |
| `raw/jurmala/` | Jurmala 扩展数据集 | <https://zenodo.org/records/5808764> |
| `external/self_recorded/full/` | 组员自采视频 | 见 `docs/SELF_RECORDING.md` |

详细结构、标签映射与常见问题见 [`docs/DATA.md`](/docs/DATA.md)（中文版见 [`docs/zh/DATA.md`](/docs/zh/DATA.md)）。

## 两条硬规则

1. **先划分、后抽帧**：划分单位是原始视频，不是帧。反了会造成数据泄漏。
2. **数据不进 Git**：默认使用项目下的 `data/`。数据在别处时，设置
   `HANDWASH_DATA_ROOT` 环境变量指向自己选择的数据目录；配置里的
   `data/raw/...` 会自动映射到该目录。
