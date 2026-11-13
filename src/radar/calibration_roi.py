"""Motion localization from a specified observed interval; no HR inputs."""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class CalibrationRangeConfig:
    calibration_frames: int = 4608
    search_bins: tuple[int,int] = (8,60)
    roi_bins: int = 8

def select_calibration_roi(z, config=CalibrationRangeConfig()):
    a=np.asarray(z)
    if a.ndim!=3 or not np.iscomplexobj(a) or not 2<=config.calibration_frames<=len(a):raise ValueError('Complex frames x RX x range and observed interval required')
    lo,hi=config.search_bins
    if not 0<=lo<hi<=a.shape[-1] or not 1<=config.roi_bins<=a.shape[-1]:raise ValueError('Invalid search/ROI width')
    observed=a[:config.calibration_frames]
    if not np.isfinite(observed).all():raise ValueError('Nonfinite calibration radar')
    profile=np.mean(np.var(observed,axis=0),axis=0).real
    center=lo+int(np.argmax(profile[lo:hi]));start=max(0,min(center-config.roi_bins//2,a.shape[-1]-config.roi_bins));bins=np.arange(start,start+config.roi_bins,dtype=np.int32)
    return center,bins,profile
