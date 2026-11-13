"""Slice Nature radar/ECG to shared absolute observation timestamps.

No interpolation, sample repair, reference filtering or HR computation occurs.
Indices refer to original arrays, including gaps or repeated radar content.
"""
from __future__ import annotations

import numpy as np


def _timestamps_ns(values, count, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != count:
        raise ValueError(f'{name} timestamps must match the signal length')
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError(f'{name} timestamps must be datetime64 values')
    values = values.astype('datetime64[ns]')
    if count < 2 or np.isnat(values).any():
        raise ValueError(f'{name} needs at least two finite timestamps')
    ns = values.astype(np.int64)
    if np.any(ns[1:] <= ns[:-1]):
        raise ValueError(f'{name} timestamps must be strictly increasing')
    return ns


def align_radar_ecg(radar, radar_timestamps, ecg, ecg_timestamps):
    """Return slices on the closed intersection of observed timestamp ranges.

    The common interval is [max(first timestamps), min(last timestamps)].
    A stream keeps only its original samples inside that interval; its first
    retained sample can be later than the interval start by one sample period.
    No claim of sub-frame synchronization or hardware freshness is made.
    """
    radar, ecg = np.asarray(radar), np.asarray(ecg)
    if radar.ndim != 3 or ecg.ndim != 1:
        raise ValueError('Expected frame x antenna x range radar and 1D ECG')
    rt = _timestamps_ns(radar_timestamps, len(radar), 'radar')
    et = _timestamps_ns(ecg_timestamps, len(ecg), 'ECG')
    start, end = max(int(rt[0]), int(et[0])), min(int(rt[-1]), int(et[-1]))
    if start >= end:
        raise ValueError('Radar and ECG have no positive shared time interval')
    ra, rb = int(np.searchsorted(rt, start)), int(np.searchsorted(rt, end, side='right'))
    ea, eb = int(np.searchsorted(et, start)), int(np.searchsorted(et, end, side='right'))
    if rb - ra < 2 or eb - ea < 2:
        raise ValueError('Shared interval contains fewer than two samples per stream')
    info = dict(mode='timestamp_intersection', interval='closed_observation_bounds',
                start_ns=start, end_ns=end, duration_s=(end-start)/1e9,
                radar_start_index=ra, radar_stop_index=rb,
                ecg_start_index=ea, ecg_stop_index=eb,
                original_radar_frames=len(radar), original_ecg_samples=len(ecg),
                radar_frames=rb-ra, ecg_samples=eb-ea,
                radar_first_ns=int(rt[ra]), radar_last_ns=int(rt[rb-1]),
                ecg_first_ns=int(et[ea]), ecg_last_ns=int(et[eb-1]))
    return radar[ra:rb], ecg[ea:eb], info
