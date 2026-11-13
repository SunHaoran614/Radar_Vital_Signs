"""数据加载器模块"""

from .ftu_loader import FTUDataLoader

__all__ = ['FTUDataLoader', 'PhysDriveDataLoader', 'BGT60TR13CDataLoader']


def __getattr__(name: str):
    if name == 'PhysDriveDataLoader':
        from .physdrive_loader import PhysDriveDataLoader
        return PhysDriveDataLoader
    if name == 'BGT60TR13CDataLoader':
        from .bgt60_loader import BGT60TR13CDataLoader
        return BGT60TR13CDataLoader
    raise AttributeError(name)

