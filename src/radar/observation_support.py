"""Causal record-history feature from exact raw-frame repetition.

Different frames are not certified fresh hardware observations. This module
does not repair samples, compress time or turn support into a probability.
"""
import numpy as np
from src.radar.frame_support import frame_support


def observation_support(cube):
    support = frame_support(cube)
    index = np.arange(len(cube))
    last_anchor = np.maximum.accumulate(np.where(support['anchor'], index, -1))
    age = index - last_anchor
    pair = np.zeros(len(cube), dtype=np.float64)
    pair[1:] = 1.0 / (1 + np.maximum(age[1:], age[:-1]))
    return dict(**support, age=age, pair_support=pair)
