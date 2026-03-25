# TODO

## 2026-03-16

### 统一 `reference_data` 和 `radar_data`

- 目标：
  - 统一三个加载器的输出结构，减少上层训练和评估代码中的分支判断。

- `reference_data` 统一方案：
  - 统一返回字段：`time`、`heart_rate`、`respiration_rate`、`meta`。
  - FTU：`timestamps -> time`，`heart_rate` 直接映射，`respiration_rate` 可暂空。
  - BGT60：保留现有 `time/heart_rate/respiration_rate` 字段名。
  - PhysDrive：由 `ecg/respiration` 估计 `heart_rate/respiration_rate`；若暂未估计，字段置空并在 `meta` 标注来源。

- `radar_data` 统一方案：
  - 统一返回容器：`{"data": np.ndarray, "meta": {...}}`。
  - `meta` 至少包含：`dataset`、`shape`、`dtype`、`is_complex`、`axes`、`sample_rate/frame_rate`。
  - 短期先统一语义描述（`axes`），中期增加可选标准化输出（如 `NCHW` 或 `T,C,...`）。

## 2026-03-18

- 将 `radar_vital_signs` 继续完善后上传至 Docker Hub 或 GitHub Container Registry。

## 2026-03-19

### 统一特征中间格式（FTU/BGT60 -> PhysDrive 风格）

- 新增离线预处理脚本：将 `FTU/BGT60` 的原始雷达数据统一转换为 `RDA` 特征。
- 目标输出轴顺序：`(frames, doppler, angle, range)`，并可选对齐到固定尺寸（如 `8x16x8`）。
- 输出容器统一为 `npz`，标准字段：
  - `radar`：统一后的雷达特征（建议 `complex64` 或 `real/imag` 双通道）。
  - `ref`：统一后的真值信号（`time/heart_rate/respiration_rate`）。
  - `meta`：元信息（`dataset/axes/shape/dtype/is_complex/frame_rate/sample_rate/preprocess`）。
- 明确保存预处理配置（窗函数、FFT 点数、去静态方法、归一化方法），确保可复现。
- [ ] PhysDrive 真值统一：新增 `ECG/resp -> time/heart_rate/respiration_rate` 转换流程，输出与 FTU/BGT60 一致的参考格式（含时间戳和 bpm 序列）。


## 3.23
docker需新增pandas


## 3.25
为这次组会介绍DANN/MMD/CORAL + 增强这些方法，但是先要对这些方法进行分析，组会就汇报这些内容。