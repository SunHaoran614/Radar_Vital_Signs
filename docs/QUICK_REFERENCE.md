# 快速参考卡片

## 📋 项目结构速查

```
src/
├── data/              # 数据处理
│   ├── loaders/       # 数据加载器
│   ├── preprocessors/ # 预处理
│   └── augmentation.py
├── signal_processing/ # 信号处理
├── models/            # 深度学习模型
├── training/          # 训练模块
├── evaluation/        # 评估模块
├── domain_adaptation/ # 域适应
└── utils/             # 工具函数
```

## 🎯 命名规范速查

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块/文件 | 小写+下划线 | `radar_preprocessor.py` |
| 类 | 大驼峰 | `RadarDataLoader` |
| 函数 | 小写+下划线 | `load_radar_data()` |
| 常量 | 全大写+下划线 | `MAX_HEART_RATE` |
| 变量 | 小写+下划线 | `participant_id` |
| 私有 | 前缀_ | `_internal_method()` |

## 📝 文档字符串模板

```python
def function_name(arg1, arg2):
    """简短描述（一行）。
    
    详细描述（可选，多行）。
    
    Args:
        arg1 (type): 参数1说明
        arg2 (type): 参数2说明
    
    Returns:
        type: 返回值说明
    
    Raises:
        ExceptionType: 异常说明
    
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

## 🔧 常用代码片段

### 数据加载

```python
from src.data.loaders.ftu_loader import FTUDataLoader

loader = FTUDataLoader('Dataset/4TU.ResearchD')
data = loader.load_radar_data(1, 'Distance', '80 cm', 1)
ref = loader.load_reference_data(1, 'Distance', '80 cm', 1)
```

### 配置管理

```python
from src.utils.config import ConfigManager

config = ConfigManager.load_yaml('configs/data_config.yaml')
dataset_config = ConfigManager.load_data_config('configs/data_config.yaml')
```

### 日志记录

```python
from src.utils.logger import setup_logger

logger = setup_logger('my_module', Path('results/logs'))
logger.info("信息")
logger.warning("警告")
logger.error("错误")
```

### 错误处理

```python
def my_function(param):
    # 参数验证
    if not valid(param):
        raise ValueError(f"参数无效: {param}")
    
    # 文件检查
    if not file.exists():
        raise FileNotFoundError(f"文件不存在: {file}")
    
    # 异常捕获
    try:
        result = risky_operation()
    except Exception as e:
        raise IOError(f"操作失败: {e}")
    
    return result
```

## 🧪 测试模板

```python
import pytest
from src.module import MyClass

class TestMyClass:
    @pytest.fixture
    def instance(self):
        return MyClass()
    
    def test_method(self, instance):
        result = instance.method()
        assert result == expected
    
    def test_exception(self, instance):
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

## 📊 评估指标

```python
from src.evaluation.metrics import calculate_mae, calculate_rmse

mae = calculate_mae(predictions, ground_truth)
rmse = calculate_rmse(predictions, ground_truth)
correlation = np.corrcoef(predictions, ground_truth)[0, 1]
```

## 🚀 Git工作流

```bash
# 创建功能分支
git checkout -b feature/my-feature

# 提交代码
git add .
git commit -m "feat: 添加新功能"

# 推送分支
git push origin feature/my-feature

# 合并到develop
git checkout develop
git merge feature/my-feature
```

## 📦 依赖管理

```bash
# 安装依赖
pip install -r requirements.txt

# 添加新依赖
pip install package_name
pip freeze > requirements.txt

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## 🔍 代码检查

```bash
# 格式化
black src/
isort src/

# 检查
flake8 src/
pylint src/
mypy src/

# 测试
pytest tests/ -v
pytest tests/ --cov=src
```

## 📈 实验流程

1. **准备数据**
   ```python
   loader = FTUDataLoader(dataset_path)
   data = loader.load_radar_data(...)
   ```

2. **信号处理**
   ```python
   range_fft = range_processor.process(data)
   phase = extract_phase(range_fft)
   ```

3. **提取生命体征**
   ```python
   vital_signs = extractor.extract(phase)
   hr = vital_signs['heart_rate']
   ```

4. **评估**
   ```python
   mae = calculate_mae(predictions, ground_truth)
   ```

5. **保存结果**
   ```python
   np.savez('results/experiment.npz', **results)
   ```

## 🎨 数据可视化

```python
import matplotlib.pyplot as plt

# 绘制信号
plt.figure(figsize=(12, 4))
plt.plot(signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Radar Signal')
plt.savefig('results/figures/signal.png', dpi=300)
plt.close()

# 绘制频谱
plt.figure(figsize=(10, 6))
plt.plot(freq, power)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.savefig('results/figures/spectrum.png', dpi=300)
plt.close()
```

## ⚠️ 常见错误

### 1. 路径问题
```python
# ❌ 错误
path = 'Dataset/4TU.ResearchD'

# ✅ 正确
from pathlib import Path
path = Path('Dataset/4TU.ResearchD')
```

### 2. 数据类型
```python
# ❌ 错误
data = np.fromfile(file, dtype=np.float32)  # 应该是int16

# ✅ 正确
data = np.fromfile(file, dtype=np.int16)
```

### 3. 维度错误
```python
# ❌ 错误
data.reshape(4, 250, -1)  # 顺序错误

# ✅ 正确
data.reshape(num_rx, num_adc, -1)
```

## 💡 最佳实践提示

- ✅ 使用类型注解
- ✅ 编写文档字符串
- ✅ 添加单元测试
- ✅ 使用配置文件
- ✅ 记录日志
- ✅ 处理异常
- ✅ 代码复用
- ✅ 遵循PEP 8

## 📞 获取帮助

- 查看 [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) 了解详细规范
- 查看 [CODE_EXAMPLES.md](CODE_EXAMPLES.md) 了解代码示例
- 查看 [DATASETS.md](DATASETS.md) 了解数据集详情

---

**保持代码整洁，保持逻辑清晰！** 🚀

