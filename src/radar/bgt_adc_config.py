"""Versioned BGT60TR13C raw-ADC to RDA front-end candidate.

The published real ADC has 16 chirps per frame. Retaining their coherent
zero-Doppler component and excluding the near-zero range bins is required for
the subject-distance target to remain in the eight-bin RDA ROI. This physical
repair is not itself evidence of heart-rate prediction performance.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ftu_adc_config import FTUADCConfig


@dataclass(frozen=True)
class BGTADCConfig(FTUADCConfig):
    chirp_selection_method: str = "all"
    clutter_method: str = "none"
    doppler_method: str = "full_center_bins"
    range_selection_method: str = "motion_zero_doppler"
    range_search_bins: tuple[int, int] | None = (8, 60)
