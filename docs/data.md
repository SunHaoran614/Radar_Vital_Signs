# 数据接口

原始数据不随代码分发。使用 `radar-export --help` 查看 FTU、BGT60TR13C 和 PhysDrive 的导出参数，用 `--dataset-root` 指定本地数据路径，用 `--output-dir` 指定新导出目录。

导出按数据集生成 `manifest.csv` 与 `samples/*.npz`。`radar-build-windows --help` 提供窗口构建参数；训练入口读取所指定窗口目录中的 `manifest.csv`、`build_config.json` 和窗口文件。不要将旧导出作为不同前端版本的可互换输入。

## 数据层级

| 数据 | 输入及限制 |
| --- | --- |
| FTU | 原始ADC；训练须按受试者分组，并排除确认的物理记录别名 |
| BGT60TR13C | ADC前端受设备配置和采样率约束；不得仅按张量形状推断语义 |
| Nature | 发布的复数range FFT；`src/data/nature_alignment.py`提供雷达/ECG时间处理工具，不能代替原始ADC验证 |
| PhysDrive | 发布的已处理RDA；参考时间轴和HR质量未认证，不作为可信HR训练或新定量结论的依据 |

FTU去别名输入通过 `radar-export --ftu-alias-audit PATH` 显式提供，不从历史目录自动读取。它是用户提供的数据身份清单，不是自动生成的保证。启用FTU ADC前端配置时必须明确提供该清单，避免遗漏去别名约束。

## 角色和时间

FTU源协议只使用FTU标签训练；独立Nature源协议只使用Nature源标签。目标标签不能进入源训练或无标签适配。候选在源train选择，固定后才复核val，test只作冻结后的评估；已经评分的test不能重调同一方法。

固定推理接收一条物理记录的复数ROI流，而非上述导出包的全部字段，详见[推理输入](inference.md)。不同设备的原始采样率、窗口起点和物理时长必须明确，不能用筛选后相邻行代替原始窗口ID。
