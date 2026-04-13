# 数据集说明

## 1. 数据集总览

| 数据集 | 场景 | 原始主要格式 | 加载器 |
|---|---|---|---|
| 4TU.ResearchD | 极端生理场景 | 雷达 `.bin` + 参考 `.csv/.ods` | `FTUDataLoader` |
| BGT60TR13C | 室内长时监测 | 雷达 `.npy` + 参考 `.csv` + 配置 `.json` | `BGT60TR13CDataLoader` |
| PhysDrive | 车载驾驶员监测 | 雷达/参考 `.mat` | `PhysDriveDataLoader` |

## 2. 本地目录约定

```text
Dataset/
├── 4TU.ResearchD/
├── BGT60TR13C/
└── PhysDrive/
```

## 3. 当前统一导出目标

统一导出脚本：`src/data/loaders/export_all_datasets.py`  
统一 `.npz` 字段：

- `radar`
- `time`
- `heart_rate`
- `respiration_rate`

统一 `meta` 关键字段：

- `radar_shape`, `radar_dtype`, `radar_axes`
- `radar_source_shape`, `radar_source_dtype`
- `radar_preprocess`
- `reference_len`, `reference_fields`

## 4. 三数据集当前处理步骤

### 4.1 FTU

1. 加载原始雷达：`(frames, rx, chirps, adc) = (1200, 4, 128, 250)`，复数。  
2. 加载参考：`timestamps + heart_rate`。  
3. 参考字段统一：`timestamps -> time`，`respiration_rate` 置为 `NaN`。  
4. 雷达统一到 RDA：`(frames, doppler, angle, range) = (1200, 8, 16, 8)`。  
5. `range=8` 采用“全样本能量中心 + 连续窗口”策略，并写入 `range_center_bin/range_selected_bins`。

### 4.2 BGT60TR13C

1. 加载原始雷达：`(frames, rx, chirps, samples)`，短时/长时帧长不同。  
2. 加载参考：`time + heart_rate + respiration_rate`。  
3. 雷达统一到 RDA：`(frames, 8, 16, 8)`。  
4. `range=8` 与 FTU 一致，使用全样本固定窗口策略。  
5. `long` 模式按固定采集距离 `0.6m` 处理与记录。

### 4.3 PhysDrive

1. 加载雷达：公开包为 `processed mmWave`，默认复数输出形状 `(~600, 8, 16, 8)`。  
2. 加载参考：`ecg + respiration` 波形。  
3. 参考转换：估计并输出统一 `time + heart_rate + respiration_rate`。  
4. 雷达对齐：保持 RDA 语义，必要时做尺寸对齐。  
5. 路径兼容：支持样本目录前导零不一致（如 `_10` / `_010`）。  
6. 帧数兼容：帧数非 600 时，默认截断或补零到 600 后导出。

## 5. 已知差异与注意事项

- 三个数据集原始层级不同：FTU/BGT60 更接近原始 ADC，PhysDrive 已是较高层特征。  
- BGT60 为实采样，频域处理链路与复采样雷达不完全相同。  
- 当前统一导出重点是“跨数据集输入结构一致”，不是严格物理标定一致。  
- 若做物理解释分析（精确距离/速度/角度），建议保留更高维原始 bins 并单独建链路。

## 6. PhysDrive 关键事实（已核对）

- 数据论文：`PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring`（arXiv:2507.19172，2025-07-25）。  
- 论文描述：生理信号原始采样率 `1000 Hz`，RGB/IR 为 `30 fps`，mmWave 分支对齐到 `20 Hz`。  
- 公开包说明：仅提供 `processed mmWave data`，并附 `ecg.mat` 与 `resp.mat`。

## 7. 维护约定

- 涉及加载器输出字段、导出格式、时间轴语义变更时，需同步更新本文件与 `README.md`。  
- 代码变更后建议先跑小范围导出，再用 `src/valid/validate_ftu_export.py` 与可视化工具做验收。
