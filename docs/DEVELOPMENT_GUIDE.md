# 代码开发守则

**所属项目**: 面向跨域场景的毫米波雷达生命体征检测关键技术研究

## 文档概述

本文档定义了"面向跨域场景的毫米波雷达生命体征检测关键技术研究"项目的代码开发规范、项目架构和最佳实践，确保代码质量、可维护性和研究可重复性。

---

## 1. 项目架构设计

### 1.1 整体架构原则

**核心原则**：
- ✅ **模块化**: 功能独立，职责单一
- ✅ **可复用**: 避免代码重复，提高复用性
- ✅ **可测试**: 便于单元测试和集成测试
- ✅ **可扩展**: 易于添加新功能和新数据集
- ✅ **可重现**: 确保实验结果可重复

### 1.2 项目目录结构

```
Radar_Vital_Signs/
├── Dataset/                          # 数据集（不纳入版本控制）
│   ├── 4TU.ResearchD/
│   └── PhysDrive/
│
├── src/                              # 源代码
│   ├── __init__.py
│   │
│   ├── data/                         # 数据处理模块
│   │   ├── __init__.py
│   │   ├── loaders/                  # 数据加载器
│   │   │   ├── __init__.py
│   │   │   ├── base_loader.py        # 基类
│   │   │   ├── ftu_loader.py         # 4TU数据加载
│   │   │   └── physdrive_loader.py   # PhysDrive数据加载
│   │   ├── preprocessors/            # 数据预处理
│   │   │   ├── __init__.py
│   │   │   ├── radar_preprocessor.py # 雷达信号预处理
│   │   │   └── reference_preprocessor.py # 参考信号预处理
│   │   └── augmentation.py           # 数据增强
│   │
│   ├── signal_processing/            # 信号处理模块
│   │   ├── __init__.py
│   │   ├── range_processing.py       # Range FFT处理
│   │   ├── phase_extraction.py       # 相位提取
│   │   ├── vital_signs.py            # 生命体征提取
│   │   └── filters.py                # 滤波器
│   │
│   ├── models/                       # 深度学习模型
│   │   ├── __init__.py
│   │   ├── base_model.py             # 模型基类
│   │   ├── cnn_models.py             # CNN模型
│   │   ├── rnn_models.py             # RNN/LSTM模型
│   │   └── transformer_models.py     # Transformer模型
│   │
│   ├── training/                     # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py                # 训练器
│   │   ├── losses.py                 # 损失函数
│   │   └── optimizers.py             # 优化器配置
│   │
│   ├── evaluation/                   # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py                # 评估指标
│   │   ├── evaluator.py              # 评估器
│   │   └── visualization.py          # 结果可视化
│   │
│   ├── domain_adaptation/            # 域适应模块
│   │   ├── __init__.py
│   │   ├── transfer_learning.py      # 迁移学习
│   │   └── domain_adversarial.py     # 域对抗训练
│   │
│   └── utils/                        # 工具函数
│       ├── __init__.py
│       ├── config.py                 # 配置管理
│       ├── logger.py                 # 日志工具
│       └── io_utils.py               # 文件IO工具
│
├── configs/                          # 配置文件
│   ├── data_config.yaml              # 数据配置
│   ├── model_config.yaml             # 模型配置
│   └── train_config.yaml             # 训练配置
│
├── experiments/                      # 实验脚本
│   ├── baseline/                     # 基线实验
│   ├── cross_dataset/                # 跨数据集实验
│   └── extreme_scenarios/            # 极端场景实验
│
├── notebooks/                        # Jupyter笔记本
│   ├── 01_data_exploration.ipynb     # 数据探索
│   ├── 02_signal_processing.ipynb    # 信号处理分析
│   └── 03_results_analysis.ipynb     # 结果分析
│
├── tests/                            # 单元测试
│   ├── test_data/
│   ├── test_signal_processing/
│   └── test_models/
│
├── results/                          # 实验结果（不纳入版本控制）
│   ├── checkpoints/                  # 模型检查点
│   ├── logs/                         # 训练日志
│   ├── metrics/                      # 评估指标
│   └── figures/                      # 图表
│
├── docs/                             # 文档
│   ├── CROSS_DOMAIN_RESEARCH_PLAN.md # 跨域研究完整方案
│   ├── DATASETS.md                   # 数据集文档
│   ├── DEVELOPMENT_GUIDE.md          # 开发守则（本文件）
│   └── QUICK_REFERENCE.md            # 快速参考
│
├── .gitignore                        # Git忽略文件
├── requirements.txt                  # Python依赖
├── setup.py                          # 安装配置
└── README.md                         # 项目根README
```

### 1.3 模块职责划分

| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **data.loaders** | 加载原始数据 | 文件路径 | 原始数据数组 |
| **data.preprocessors** | 数据预处理 | 原始数据 | 预处理后数据 |
| **signal_processing** | 信号处理算法 | 雷达数据 | 生命体征估计 |
| **models** | 深度学习模型 | 特征数据 | 预测结果 |
| **training** | 模型训练 | 数据+模型 | 训练好的模型 |
| **evaluation** | 性能评估 | 预测+真值 | 评估指标 |
| **domain_adaptation** | 域适应 | 源域+目标域 | 适应后模型 |

---

## 2. 代码编写规范

### 2.1 Python代码风格

**遵循 PEP 8 规范**

#### 命名规范

```python
# 模块名：小写+下划线
# 文件名: radar_preprocessor.py

# 类名：大驼峰命名法
class RadarDataLoader:
    pass

class VitalSignExtractor:
    pass

# 函数名：小写+下划线
def load_radar_data(file_path):
    pass

def extract_heart_rate(signal):
    pass

# 常量：全大写+下划线
MAX_HEART_RATE = 200
MIN_BREATHING_RATE = 6
SAMPLING_RATE = 6250

# 变量名：小写+下划线
participant_id = 1
radar_data = None
heart_rate_estimate = 0

# 私有变量/方法：前缀单下划线
class DataProcessor:
    def __init__(self):
        self._internal_buffer = []
    
    def _internal_method(self):
        pass
```

#### 代码格式

```python
# 导入顺序
# 1. 标准库
import os
import sys
from pathlib import Path

# 2. 第三方库
import numpy as np
import pandas as pd
import torch

# 3. 本地模块
from src.data.loaders import FTULoader
from src.utils.config import load_config

# 每行最大长度：88字符（Black格式化工具标准）
# 使用4个空格缩进，不使用Tab

# 函数定义之间空2行
def function_one():
    pass


def function_two():
    pass


# 类定义之间空2行
class ClassOne:
    pass


class ClassTwo:
    pass
```

### 2.2 文档字符串规范

**使用 Google Style Docstrings**

```python
def extract_vital_signs(radar_data, config):
    """从雷达数据中提取生命体征。
    
    该函数实现了基于FFT的生命体征提取算法，包括Range FFT、
    相位提取和带通滤波等步骤。
    
    Args:
        radar_data (np.ndarray): 雷达原始数据，shape为(num_rx, num_adc, num_chirps)
        config (dict): 配置参数字典，包含以下键：
            - 'hr_band': tuple, 心率频带范围 (Hz)
            - 'rr_band': tuple, 呼吸率频带范围 (Hz)
            - 'window_size': int, 窗口大小（帧数）
    
    Returns:
        dict: 包含以下键的字典：
            - 'heart_rate': float, 估计的心率 (bpm)
            - 'respiratory_rate': float, 估计的呼吸率 (bpm)
            - 'confidence': float, 置信度 (0-1)
    
    Raises:
        ValueError: 如果radar_data的shape不正确
        KeyError: 如果config缺少必需的键
    
    Example:
        >>> config = {'hr_band': (0.8, 2.5), 'rr_band': (0.1, 0.6), 'window_size': 200}
        >>> result = extract_vital_signs(radar_data, config)
        >>> print(f"Heart Rate: {result['heart_rate']:.1f} bpm")
        Heart Rate: 72.5 bpm
    
    Note:
        - 输入数据应已完成预处理（去直流、去噪等）
        - 建议window_size至少为200帧（10秒）以获得稳定估计
    
    References:
        [1] Sadeghi et al., "Comprehensive mm-Wave FMCW Radar Dataset...", 2024
    """
    # 实现代码
    pass


class RadarDataLoader:
    """4TU数据集的雷达数据加载器。
    
    该类负责加载和解析4TU.ResearchD数据集的雷达二进制文件，
    将原始ADC数据重组为标准的数据立方体格式。
    
    Attributes:
        dataset_path (Path): 数据集根目录路径
        num_adc (int): ADC采样点数，默认250
        num_rx (int): 接收天线数，默认4
        num_frames (int): 帧数，默认1200
        num_chirps (int): 每帧chirp数，默认128
    
    Example:
        >>> loader = RadarDataLoader('Dataset/4TU.ResearchD')
        >>> data = loader.load_participant(1, 'Distance', '80 cm', 1)
        >>> print(data.shape)
        (4, 250, 153600)
    """
    
    def __init__(self, dataset_path):
        """初始化数据加载器。
        
        Args:
            dataset_path (str or Path): 数据集根目录路径
        """
        pass
    
    def load_participant(self, participant_id, scenario, distance, repeat):
        """加载指定参与者的雷达数据。
        
        Args:
            participant_id (int): 参与者ID (1-10)
            scenario (str): 场景类型 ('Distance', 'Orientation', 'Angle', 'Elevated')
            distance (str): 距离 ('40 cm', '80 cm', '120 cm', '160 cm')
            repeat (int): 重复次数 (1-4)
        
        Returns:
            np.ndarray: 雷达数据立方体，shape为(num_rx, num_adc, num_chirps)
        """
        pass
```

### 2.3 类型注解

**使用Python类型提示**

```python
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
from pathlib import Path

def process_radar_signal(
    data: np.ndarray,
    sampling_rate: float,
    filter_band: Tuple[float, float],
    window_size: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """处理雷达信号。
    
    Args:
        data: 输入信号数组
        sampling_rate: 采样率 (Hz)
        filter_band: 滤波频带 (低频, 高频)
        window_size: 可选的窗口大小
    
    Returns:
        包含处理结果的字典
    """
    pass


class DataConfig:
    """数据配置类。"""
    
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool = True
    ) -> None:
        self.dataset_name: str = dataset_name
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle: bool = shuffle
```

---

## 3. 函数设计规范

### 3.1 函数设计原则

**SOLID原则应用于函数**

1. **单一职责原则 (SRP)**
   - 一个函数只做一件事
   - 函数名应清晰表达其功能

```python
# ❌ 不好：函数做了太多事情
def process_data(file_path):
    data = load_file(file_path)
    data = remove_dc(data)
    data = apply_filter(data)
    hr = extract_hr(data)
    rr = extract_rr(data)
    save_results(hr, rr)
    return hr, rr

# ✅ 好：职责分离
def load_radar_data(file_path: Path) -> np.ndarray:
    """只负责加载数据"""
    pass

def preprocess_signal(data: np.ndarray) -> np.ndarray:
    """只负责预处理"""
    pass

def extract_vital_signs(data: np.ndarray) -> Dict[str, float]:
    """只负责提取生命体征"""
    pass
```

2. **函数长度控制**
   - 单个函数不超过50行
   - 如果超过，考虑拆分

3. **参数数量控制**
   - 参数不超过5个
   - 超过5个使用配置对象或字典

```python
# ❌ 不好：参数太多
def train_model(data, labels, lr, epochs, batch_size, optimizer, loss_fn, device):
    pass

# ✅ 好：使用配置对象
from dataclasses import dataclass

@dataclass
class TrainConfig:
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    optimizer: str = 'adam'
    loss_fn: str = 'mse'
    device: str = 'cuda'

def train_model(data, labels, config: TrainConfig):
    pass
```

### 3.2 错误处理

```python
def load_participant_data(participant_id: int, scenario: str) -> np.ndarray:
    """加载参与者数据，包含完整的错误处理。"""
    
    # 参数验证
    if not 1 <= participant_id <= 10:
        raise ValueError(f"participant_id必须在1-10之间，得到: {participant_id}")
    
    valid_scenarios = ['Distance', 'Orientation', 'Angle', 'Elevated']
    if scenario not in valid_scenarios:
        raise ValueError(f"scenario必须是{valid_scenarios}之一，得到: {scenario}")
    
    # 文件存在性检查
    file_path = get_data_path(participant_id, scenario)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 数据加载with异常处理
    try:
        data = np.fromfile(file_path, dtype=np.int16)
    except Exception as e:
        raise IOError(f"读取文件失败: {file_path}, 错误: {e}")
    
    # 数据验证
    expected_size = 250 * 4 * 1200 * 128 * 2  # ADC * RX * Frames * Chirps * IQ
    if len(data) != expected_size:
        raise ValueError(
            f"数据大小不匹配。期望: {expected_size}, 实际: {len(data)}"
        )
    
    return data
```

### 3.3 函数返回值

```python
# ✅ 好：明确的返回类型
def extract_heart_rate(signal: np.ndarray) -> float:
    """返回单一值"""
    return hr_value

def extract_vital_signs(signal: np.ndarray) -> Tuple[float, float]:
    """返回多个值，使用元组"""
    return heart_rate, respiratory_rate

def analyze_signal(signal: np.ndarray) -> Dict[str, Any]:
    """返回复杂结果，使用字典"""
    return {
        'heart_rate': 72.5,
        'respiratory_rate': 15.2,
        'confidence': 0.95,
        'quality_score': 0.88
    }

# 使用dataclass返回结构化数据
from dataclasses import dataclass

@dataclass
class VitalSignsResult:
    heart_rate: float
    respiratory_rate: float
    confidence: float
    timestamp: float

def extract_vital_signs_v2(signal: np.ndarray) -> VitalSignsResult:
    """返回结构化对象"""
    return VitalSignsResult(
        heart_rate=72.5,
        respiratory_rate=15.2,
        confidence=0.95,
        timestamp=time.time()
    )
```

---

## 4. 类设计规范

### 4.1 基类设计

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class BaseDataLoader(ABC):
    """数据加载器基类。
    
    所有数据加载器都应继承此类并实现抽象方法。
    """
    
    def __init__(self, dataset_path: Path):
        """初始化数据加载器。
        
        Args:
            dataset_path: 数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)
        self._validate_dataset_path()
    
    def _validate_dataset_path(self) -> None:
        """验证数据集路径是否有效。"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
    
    @abstractmethod
    def load_radar_data(self, **kwargs) -> np.ndarray:
        """加载雷达数据（子类必须实现）。"""
        pass
    
    @abstractmethod
    def load_reference_data(self, **kwargs) -> Dict[str, Any]:
        """加载参考数据（子类必须实现）。"""
        pass
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息（可选重写）。"""
        return {
            'dataset_path': str(self.dataset_path),
            'dataset_type': self.__class__.__name__
        }
```

### 4.2 具体实现类

```python
class FTUDataLoader(BaseDataLoader):
    """4TU数据集加载器。"""
    
    # 类常量
    NUM_ADC = 250
    NUM_RX = 4
    NUM_FRAMES = 1200
    NUM_CHIRPS = 128
    
    def __init__(self, dataset_path: Path):
        super().__init__(dataset_path)
        self.radar_data_path = self.dataset_path / 'Radar data'
        self.hr_ref_path = self.dataset_path / 'HR_Ref_Values'
    
    def load_radar_data(
        self,
        participant_id: int,
        scenario: str,
        distance: str,
        repeat: int
    ) -> np.ndarray:
        """加载4TU雷达数据。
        
        Args:
            participant_id: 参与者ID (1-10)
            scenario: 场景类型
            distance: 距离
            repeat: 重复次数
        
        Returns:
            雷达数据立方体，shape为(num_rx, num_adc, num_chirps)
        """
        file_path = self._get_radar_file_path(
            participant_id, scenario, distance, repeat
        )
        raw_data = np.fromfile(file_path, dtype=np.int16)
        return self._reshape_radar_data(raw_data)
    
    def load_reference_data(
        self,
        participant_id: int,
        scenario: str,
        distance: str,
        repeat: int
    ) -> Dict[str, Any]:
        """加载参考心率数据。"""
        hr_file = self._get_hr_file_path(
            participant_id, scenario, distance, repeat
        )
        hr_data = pd.read_csv(hr_file, skiprows=3)
        return {
            'heart_rate': hr_data.iloc[:, 2].values,
            'timestamps': hr_data.iloc[:, 1].values
        }
    
    def _get_radar_file_path(self, *args) -> Path:
        """构建雷达数据文件路径（私有方法）。"""
        # 实现细节
        pass
    
    def _reshape_radar_data(self, raw_data: np.ndarray) -> np.ndarray:
        """重塑雷达数据为标准格式（私有方法）。"""
        # 分离IQ
        I = raw_data[0::2]
        Q = raw_data[1::2]
        complex_data = I + 1j * Q
        
        # 重塑
        data_cube = complex_data.reshape(
            self.NUM_RX,
            self.NUM_ADC,
            self.NUM_FRAMES,
            self.NUM_CHIRPS
        )
        
        return data_cube.reshape(
            self.NUM_RX,
            self.NUM_ADC,
            self.NUM_FRAMES * self.NUM_CHIRPS
        )
```

---

## 5. 配置管理

### 5.1 使用YAML配置文件

```yaml
# configs/data_config.yaml
dataset:
  name: "4TU"
  path: "Dataset/4TU.ResearchD"
  
  # 数据划分
  split:
    train_participants: [1, 2, 3, 4, 5, 6, 7]
    val_participants: [8, 9]
    test_participants: [10]
  
  # 场景选择
  scenarios:
    - "Distance"
    - "Orientation"
    - "Angle"
    - "Elevated"
  
  # 数据加载参数
  loader:
    num_workers: 4
    batch_size: 16
    shuffle: true

# 雷达参数
radar:
  num_adc: 250
  num_rx: 4
  num_frames: 1200
  num_chirps: 128
  sampling_rate: 6250  # ksps
  start_freq: 77  # GHz
  end_freq: 81  # GHz
```

### 5.2 配置加载工具

```python
# src/utils/config.py
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field

@dataclass
class RadarConfig:
    """雷达配置"""
    num_adc: int = 250
    num_rx: int = 4
    num_frames: int = 1200
    num_chirps: int = 128
    sampling_rate: int = 6250
    start_freq: float = 77.0
    end_freq: float = 81.0

@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    path: str
    train_participants: List[int] = field(default_factory=list)
    val_participants: List[int] = field(default_factory=list)
    test_participants: List[int] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=list)

class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_yaml(config_path: Path) -> Dict[str, Any]:
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_data_config(config_path: Path) -> DatasetConfig:
        """加载数据集配置"""
        config_dict = ConfigManager.load_yaml(config_path)
        dataset_dict = config_dict['dataset']
        
        return DatasetConfig(
            name=dataset_dict['name'],
            path=dataset_dict['path'],
            train_participants=dataset_dict['split']['train_participants'],
            val_participants=dataset_dict['split']['val_participants'],
            test_participants=dataset_dict['split']['test_participants'],
            scenarios=dataset_dict['scenarios']
        )
```

---

## 6. 日志和调试

### 6.1 日志规范

```python
# src/utils/logger.py
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str,
    log_dir: Path,
    level: int = logging.INFO
) -> logging.Logger:
    """设置日志记录器。
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        level: 日志级别
    
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 文件处理器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'{name}_{timestamp}.log',
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 使用示例
logger = setup_logger('data_loader', Path('results/logs'))

logger.info("开始加载数据...")
logger.debug(f"参数: participant_id={participant_id}")
logger.warning("数据质量较低，SNR < 10 dB")
logger.error("文件读取失败")
```

---

## 7. 测试规范

### 7.1 单元测试

```python
# tests/test_data/test_ftu_loader.py
import pytest
import numpy as np
from pathlib import Path
from src.data.loaders.ftu_loader import FTUDataLoader

class TestFTUDataLoader:
    """测试4TU数据加载器"""
    
    @pytest.fixture
    def loader(self):
        """创建测试用的加载器实例"""
        dataset_path = Path('Dataset/4TU.ResearchD')
        return FTUDataLoader(dataset_path)
    
    def test_load_radar_data_shape(self, loader):
        """测试加载的雷达数据shape是否正确"""
        data = loader.load_radar_data(
            participant_id=1,
            scenario='Distance',
            distance='80 cm',
            repeat=1
        )
        
        expected_shape = (4, 250, 153600)  # (RX, ADC, Chirps)
        assert data.shape == expected_shape, \
            f"数据shape不正确。期望: {expected_shape}, 实际: {data.shape}"
    
    def test_load_radar_data_dtype(self, loader):
        """测试数据类型是否为复数"""
        data = loader.load_radar_data(1, 'Distance', '80 cm', 1)
        assert np.iscomplexobj(data), "数据应该是复数类型"
    
    def test_invalid_participant_id(self, loader):
        """测试无效的participant_id是否抛出异常"""
        with pytest.raises(ValueError):
            loader.load_radar_data(11, 'Distance', '80 cm', 1)
    
    def test_load_reference_data(self, loader):
        """测试加载参考数据"""
        ref_data = loader.load_reference_data(1, 'Distance', '80 cm', 1)
        
        assert 'heart_rate' in ref_data
        assert 'timestamps' in ref_data
        assert len(ref_data['heart_rate']) > 0

# 运行测试: pytest tests/test_data/test_ftu_loader.py -v
```

---

## 8. 文件管理规范

### 8.1 临时文件管理

**原则**：保持工作目录整洁，及时清理临时文件。

**临时文件定义**：
- 测试/验证用的临时脚本
- 数据处理过程中的中间文件
- 调试用的输出文件
- 实验性代码文件

**管理规范**：

✅ **应该做的**：
1. **使用临时目录**：将临时文件放在专门的临时目录
   ```
   temp/              # 临时文件目录（加入.gitignore）
   ├── test_*.py      # 临时测试脚本
   ├── debug_*.png    # 调试图片
   └── temp_*.npy     # 临时数据文件
   ```

2. **命名规范**：临时文件使用明确的前缀
   - `temp_` - 临时文件
   - `test_` - 测试文件
   - `debug_` - 调试文件
   - `tmp_` - 临时数据

3. **及时删除**：使用完毕后立即删除
   ```python
   import tempfile
   import os

   # 使用临时文件
   with tempfile.NamedTemporaryFile(delete=True) as tmp:
       # 使用tmp文件
       pass
   # 自动删除

   # 或手动删除
   temp_file = 'temp_data.npy'
   try:
       # 使用文件
       pass
   finally:
       if os.path.exists(temp_file):
           os.remove(temp_file)
   ```

4. **添加到.gitignore**：确保临时文件不被提交
   ```gitignore
   # 临时文件
   temp/
   tmp/
   temp_*
   test_*
   debug_*
   *.tmp
   ```

❌ **不应该做的**：
- 在项目根目录创建临时文件
- 临时文件使用不明确的名称
- 使用完不删除临时文件
- 将临时文件提交到版本控制

**检查清单**：
- [ ] 临时文件放在专门目录
- [ ] 使用明确的命名前缀
- [ ] 使用完毕后已删除
- [ ] 已添加到.gitignore

---

## 9. 版本控制规范

### 9.1 Git工作流

**分支策略**：

```
main (主分支)
  ├── develop (开发分支)
  │   ├── feature/data-loader (功能分支)
  │   ├── feature/signal-processing (功能分支)
  │   └── fix/bug-name (修复分支)
  └── release/v1.0.0 (发布分支)
```

**分支命名规范**：

| 分支类型 | 命名格式 | 说明 | 示例 |
|---------|---------|------|------|
| 主分支 | `main` | 稳定的生产版本 | `main` |
| 开发分支 | `develop` | 日常开发集成分支 | `develop` |
| 功能分支 | `feature/功能名` | 新功能开发 | `feature/ftu-loader` |
| 修复分支 | `fix/问题描述` | Bug修复 | `fix/phase-unwrap` |
| 实验分支 | `experiment/实验名` | 实验性功能 | `experiment/transformer` |
| 发布分支 | `release/版本号` | 版本发布准备 | `release/v1.0.0` |

**提交信息规范** (Conventional Commits)：

```bash
# 格式: <type>(<scope>): <subject>

# 类型 (type)
feat:     # 新功能
fix:      # Bug修复
docs:     # 文档更新
style:    # 代码格式（不影响功能）
refactor: # 重构（既不是新功能也不是修复）
test:     # 测试相关
chore:    # 构建/工具链相关
perf:     # 性能优化

# 示例
git commit -m "feat(data): 添加4TU数据加载器"
git commit -m "fix(signal): 修复相位解包错误"
git commit -m "docs: 更新数据集文档"
git commit -m "test(loader): 添加数据加载器单元测试"
git commit -m "refactor(preprocessing): 重构数据预处理模块"
git commit -m "perf(fft): 优化Range FFT性能"
```

### 9.2 GitHub工作流程

#### 常用指令
```bash
# 本地存储的远程分支缓存
git branch -r

# 远程仓库中实时的分支状态
git ls-remote --heads origin

# 清理过时的远程分支缓存
git fetch origin --prune
```

#### 初次上传到GitHub

**步骤1: 创建GitHub仓库**
1. 登录GitHub
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `Radar_Vital_Signs`
   - Description: "面向跨域场景的毫米波雷达生命体征检测关键技术研究"
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize with README"（本地已有）
4. 点击 "Create repository"

**步骤2: 本地Git初始化**
```bash
# 进入项目目录
cd d:\Code\Radar_Vital_Signs

# 初始化Git仓库（如果还没有）
git init

# 添加所有文件到暂存区
git add .

# 首次提交
git commit -m "feat: 初始化项目，添加数据加载器和开发守则"

# 设置主分支名称为main
git branch -M main

# 添加远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/Radar_Vital_Signs.git

# 推送到GitHub
git push -u origin main
```

**步骤3: 创建develop分支**
```bash
# 创建并切换到develop分支
git checkout -b develop

# 推送develop分支到GitHub
git push -u origin develop
```

#### 日常开发工作流

**开发新功能**：
```bash
# 1. 确保在develop分支
git checkout develop

# 2. 拉取最新代码
git pull origin develop

# 3. 创建功能分支
git checkout -b feature/new-feature

# 4. 开发并提交
git add .
git commit -m "feat: 添加新功能"

# 5. 推送到GitHub
git push -u origin feature/new-feature

# 6. 在GitHub上创建Pull Request
#    从 feature/new-feature 到 develop

# 7. 代码审查通过后，合并到develop
#    在GitHub上点击 "Merge pull request"

# 8. 删除本地功能分支
git checkout develop
git pull origin develop
git branch -d feature/new-feature
```

**修复Bug**：
```bash
# 1. 从develop创建修复分支
git checkout develop
git checkout -b fix/bug-description

# 2. 修复并提交
git add .
git commit -m "fix: 修复XXX问题"

# 3. 推送并创建PR
git push -u origin fix/bug-description
```

**发布版本**：
```bash
# 1. 从develop创建发布分支
git checkout develop
git checkout -b release/v1.0.0

# 2. 更新版本号、文档等
# 编辑 src/__init__.py 中的 __version__
git add .
git commit -m "chore: 准备v1.0.0发布"

# 3. 合并到main
git checkout main
git merge release/v1.0.0

# 4. 打标签
git tag -a v1.0.0 -m "Release version 1.0.0"

# 5. 推送
git push origin main
git push origin v1.0.0

# 6. 合并回develop
git checkout develop
git merge release/v1.0.0
git push origin develop

# 7. 删除发布分支
git branch -d release/v1.0.0
```

### 9.3 持续维护规范

**每日工作流程**：
```bash
# 1. 早上开始工作前
git checkout develop
git pull origin develop

# 2. 创建今天的工作分支
git checkout -b feature/today-work

# 3. 工作中定期提交
git add .
git commit -m "feat: 完成XXX功能"

# 4. 下班前推送
git push origin feature/today-work

# 5. 创建PR进行代码审查
```

**定期维护任务**：
- [ ] 每周检查并关闭已完成的Issue
- [ ] 每月更新依赖包版本
- [ ] 每季度审查和更新文档
- [ ] 定期清理已合并的分支

### 9.4 .gitignore配置

```gitignore
# 数据集（太大，不纳入版本控制）
Dataset/
*.bin
*.mat

# 实验结果
results/
checkpoints/
logs/
*.pth
*.h5

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db
```

---

## 10. 文档编写规范

### 10.1 文档类型划分

**项目文档** (docs/目录)：
- README.md - 项目总体说明
- DATASETS.md - 数据集说明文档
- DEVELOPMENT_GUIDE.md - 开发守则
- QUICK_REFERENCE.md - 快速参考

**代码文档** (源代码中)：
- 文档字符串 (Docstrings)
- 行内注释 (Comments)

### 10.2 文档编写原则

**项目文档规范**：

✅ **应该包含**：
- 概念说明和原理
- 数据格式和结构描述
- 配置参数说明
- 使用流程和步骤
- 参考文献和链接

❌ **不应该包含**：
- **完整的代码实现**（代码应该在源文件中）
- 具体的函数调用示例（可以指向源文件）
- 详细的代码片段（超过5行）

**例外情况**：
- QUICK_REFERENCE.md 可以包含简短代码片段（<10行）作为速查
- 配置文件示例（YAML/JSON格式）可以完整展示

**正确做法**：
```markdown
## 数据读取方法

**读取步骤**：
1. 读取二进制文件
2. 重塑数据为标准格式
3. 分离IQ分量
4. 组合成复数数据

**参考实现**：详见 `src/data/loaders/ftu_loader.py`
```

**错误做法**：
```markdown
## 数据读取方法

```python
import numpy as np
raw_data = np.fromfile('data.bin', dtype=np.int16)
# ... 30行代码 ...
```
```

### 10.3 文档维护原则

1. **单一数据源**：代码实现只在源文件中，文档只描述概念
2. **避免重复**：不在文档中复制代码，避免不同步
3. **引用源文件**：通过文件路径引用实际实现
4. **保持简洁**：文档重在说明"是什么"和"为什么"，而非"怎么做"

### 10.4 代码文档规范

**文档字符串**（必须包含）：
- 函数/类的功能说明
- 参数说明
- 返回值说明
- 使用示例（简短）

**行内注释**（适度使用）：
- 解释复杂逻辑
- 说明关键算法步骤
- 标注重要的数值含义

---

## 11. 代码审查清单

在提交代码前，请检查以下项目：

**代码质量**：
- [ ] 代码符合PEP 8规范
- [ ] 所有函数都有完整的文档字符串
- [ ] 添加了类型注解
- [ ] 包含适当的错误处理
- [ ] 代码可读性良好，有适当的注释
- [ ] 没有重复代码
- [ ] 函数和类的职责单一明确

**测试和配置**：
- [ ] 编写了单元测试
- [ ] 测试全部通过
- [ ] 没有硬编码的路径或参数
- [ ] 使用配置文件管理参数
- [ ] 添加了必要的日志记录

**文档**：
- [ ] 更新了相关文档（如有必要）
- [ ] 文档中没有包含完整代码实现
- [ ] 文档引用了正确的源文件路径

**文件管理**：
- [ ] 删除了所有临时文件
- [ ] 临时文件已添加到.gitignore
- [ ] 工作目录整洁，无调试文件残留

---

## 12. 最佳实践总结

### 12.1 DO - 应该做的

✅ **使用有意义的变量名**
✅ **保持函数简短（<50行）**
✅ **编写文档字符串**
✅ **使用类型注解**
✅ **编写单元测试**
✅ **使用配置文件**
✅ **添加日志记录**
✅ **处理异常情况**
✅ **代码复用**
✅ **遵循DRY原则（Don't Repeat Yourself）**

### 12.2 DON'T - 不应该做的

❌ **硬编码路径和参数**
❌ **使用全局变量**
❌ **忽略异常**
❌ **写超长函数**
❌ **过度嵌套**
❌ **缺少文档**
❌ **不写测试**
❌ **提交调试代码**
❌ **复制粘贴代码**
❌ **使用魔法数字**

---

## 附录：快速参考

### 常用命令

```bash
# 代码格式化
black src/
isort src/

# 代码检查
flake8 src/
pylint src/

# 类型检查
mypy src/

# 运行测试
pytest tests/ -v
pytest tests/ --cov=src

# 生成文档
sphinx-build -b html docs/ docs/_build/
```

### 推荐工具

- **代码格式化**: Black, isort
- **代码检查**: flake8, pylint
- **类型检查**: mypy
- **测试**: pytest, pytest-cov
- **文档**: Sphinx
- **版本控制**: Git
- **依赖管理**: pip, conda

---

**项目名称**: 面向跨域场景的毫米波雷达生命体征检测关键技术研究
**文档版本**: 1.1
**最后更新**: 2025-11-11
**维护者**: 项目团队

