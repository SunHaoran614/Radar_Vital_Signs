# 固定模型推理

`radar-infer`使用`src/models/dar_waveform_reader.py`中的三视图波形模型，以明确给定的固定权重执行复数ROI→BPM推理。不会导入实验脚本、读取参考标签或在评估记录上更新模型。

## 输入

使用`numpy.savez`保存NPZ，且只包含以下三个键：

| 键 | 要求 |
| --- | --- |
| `complex_stream` | 一条物理记录的有限复数数组`[native_frames, cells]` |
| `window_starts` | 从0开始、严格递增的整数原始帧起点；必须保留完整12.8秒窗 |
| `sample_rate_hz` | 标量，支持10、20或30Hz；必须是该输入的实际采样率 |

设备ADC到ROI由相应设备前端处理；不能把ADC整数、真实功率矩阵或不同物理记录拼接后作为复数ROI。起点必须能精确映射到内部20Hz时间网格。

## 调用

```bash
radar-infer --input /path/to/radar.npz \
  --checkpoint /path/to/frozen.pt \
  --checkpoint-sha256 EXPECTED_SHA256 \
  --record-id physical_record_001 \
  --output /path/to/new_predictions.npz
```

权重必须包含匹配模型的`model_state`。错误SHA或不兼容参数将拒绝加载；输出已存在时拒绝覆盖。权重不随代码分发，也不会从本地归档自动选择。请从自己的已冻结权重配置中取路径和SHA。

```python
from src.inference.dar_pipeline import FrozenDarPipeline

pipeline = FrozenDarPipeline(checkpoint, expected_sha256)
result = pipeline.predict(complex_stream, sample_rate_hz, window_starts,
                          record_id="physical_record_001")
```

输出NPZ包括原始起点、起点秒数、raw BPM和EWMA BPM；JSON记录输入、权重、输出哈希及参考未读取身份。窗口编号按调用输入的完整起点序列生成；不要用它重新编号经过标签筛选的窗口。

波形采用整记录零相位滤波，属于离线处理；EWMA按一条物理记录重置，不能宣称整体是严格在线算法。保留代码和权重兼容不意味着已证明源域、跨域或新设备精度。不得用目标标签重新挑选权重或真值校准。
