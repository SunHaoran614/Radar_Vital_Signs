# 数据集说明（精简版）

## 1. 数据集总览

| 数据集 | 场景 | 主要格式 | 加载器 |
|---|---|---|---|
| 4TU.ResearchD | 极端生理场景 | 雷达 `.bin` + 参考 `.csv/.ods` | `FTUDataLoader` |
| PhysDrive | 车载驾驶员监测 | 雷达/参考 `.mat` | `PhysDriveDataLoader` |
| BGT60TR13C_Indoor | 室内长时监测 | 雷达 `.npy` + 参考 `.csv` + 配置 `.json` | `BGT60TR13CDataLoader` |

## 2. 本地目录约定

建议在仓库根目录放置：

```text
Dataset/
├── 4TU.ResearchD/
├── PhysDrive/
└── BGT60TR13C_Indoor/
```

## 3. 加载器统一接口

所有加载器继承 `BaseDataLoader`，核心方法：

- `load_radar_data(...) -> np.ndarray`
- `load_reference_data(...) -> Dict[str, Any]`
- `get_dataset_info() -> Dict[str, Any]`

## 4. 最小使用示例

```python
from src.data.loaders.ftu_loader import FTUDataLoader

loader = FTUDataLoader('Dataset/4TU.ResearchD')
radar = loader.load_radar_data(1, 'Distance', '80 cm', 1)
ref = loader.load_reference_data(1, 'Distance', '80 cm', 1)
info = loader.get_dataset_info()
```

## 5. 数据处理注意事项

- 三个数据集采样维度与预处理链路不同，不能直接复用同一处理参数。
- PhysDrive 数据通常已是较高层特征表示，和原始 ADC 流程不同。
- BGT60TR13C 为实采样数据，频域处理需注意与复采样数据的 FFT 差异。

## 6. 新增数据集接入规范

新增加载器时请满足：

1. 继承 `BaseDataLoader` 并实现两个抽象方法。
2. 在 `__init__` 中校验目录结构和关键文件。
3. 明确返回张量形状、dtype、单位和时间轴语义。
4. 在本文件补一条“数据集总览”记录与最小示例。
