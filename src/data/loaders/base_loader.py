"""数据加载器基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
import numpy as np


class BaseDataLoader(ABC):
    """数据加载器基类。
    
    所有数据加载器都应继承此类并实现抽象方法。
    
    Attributes:
        dataset_path (Path): 数据集根目录路径
    """
    
    def __init__(self, dataset_path: str):
        """初始化数据加载器。
        
        Args:
            dataset_path: 数据集根目录路径
        
        Raises:
            FileNotFoundError: 如果数据集路径不存在
        """
        self.dataset_path = Path(dataset_path)
        self._validate_dataset_path()
    
    def _validate_dataset_path(self) -> None:
        """验证数据集路径是否有效。
        
        Raises:
            FileNotFoundError: 如果数据集路径不存在
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
    
    @abstractmethod
    def load_radar_data(self, **kwargs) -> np.ndarray:
        """加载雷达数据（子类必须实现）。
        
        Returns:
            雷达数据数组
        """
        pass
    
    @abstractmethod
    def load_reference_data(self, **kwargs) -> Dict[str, Any]:
        """加载参考数据（子类必须实现）。
        
        Returns:
            包含参考数据的字典
        """
        pass
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息（可选重写）。
        
        Returns:
            包含数据集信息的字典
        """
        return {
            'dataset_path': str(self.dataset_path),
            'dataset_type': self.__class__.__name__
        }

