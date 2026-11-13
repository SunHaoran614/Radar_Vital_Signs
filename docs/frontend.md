# RDA前端与表征

`src/radar/`提供可替换的RDA算法池：clutter、localization、range selection、beamforming和phase。`RDAConfig`选择各阶段的方法；`list_stage_methods`枚举可用实现。

```python
from src.radar import RDAConfig, list_stage_methods

config = RDAConfig()
print(config)
print(list_stage_methods())
```

`apply_rda_pipeline`接收保留复数信息的统一cube；各阶段的轴和返回值遵循函数签名。该算法池不能替代设备专属的ADC解析、chirp选择、虚拟阵列几何和FFT配置。FTU与BGT的前端配置分别位于`src/radar/ftu_adc_config.py`和`bgt_adc_config.py`。

`src/features/`保留EDACM、HR-AdaVMD、训练表征装配及固定推理使用的相位/波形处理。`radar-build-windows --help`查看可配置的窗口与表征选项。固定DAR推理采用独立的整记录相位/波形视图，不等同于通用EDACM/VMD训练路径。

相位、谱能量或浅层探针改善本身不证明深度模型或跨域方法有效；精度结论需要固定协议下的独立评估。
