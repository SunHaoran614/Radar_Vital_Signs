# TODO

## 2026-03-16

### 统一 `reference_data` 和 `radar_data`

- 目标：
  - 统一三个加载器的输出结构，减少上层训练/评估代码中的分支判断。

- `reference_data` 统一方法：
  - 统一返回字段：`time`、`heart_rate`、`respiration_rate`、`meta`。
  - FTU：`timestamps -> time`，`heart_rate` 直接映射，`respiration_rate` 置空。
  - BGT60：保留现有 `time/heart_rate/respiration_rate` 字段名。
  - PhysDrive：由 `ecg/respiration` 通过滑窗峰值法估计 `heart_rate/respiration_rate`；若暂未估计，字段置空并在 `meta` 标注来源。

- `radar_data` 统一方法：
  - 统一返回容器：`{"data": np.ndarray, "meta": {...}}`。
  - `meta` 至少包含：`dataset`、`shape`、`dtype`、`is_complex`、`axes`、`sample_rate/frame_rate`。
  - 短期不强行统一张量形状，仅统一语义描述（`axes`）；中期增加可选标准化输出（如 `NCHW` 或 `T,C,...`）。
