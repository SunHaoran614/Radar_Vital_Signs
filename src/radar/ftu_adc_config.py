"""Versioned FTU raw-ADC front-end specification used by conversion/export.

This config covers stages *before* a complex RDA cube exists. The separate
RDAConfig pool operates on an already formed cube and must not be presented as
configuring chirp selection, intraframe clutter, or the Doppler FFT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FTUADCConfig:
    schema_version: int = 1
    target_doppler: int = 8
    target_angle: int = 16
    target_range: int = 8
    chirp_selection_method: str = "legacy_64"
    clutter_method: str = "chirp_mean"
    doppler_method: str = "legacy_truncate"
    range_selection_method: str = "global_energy"
    range_search_bins: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported FTU ADC config schema version")
        dimensions = (self.target_doppler, self.target_angle, self.target_range)
        if any(type(size) is not int or size <= 0 for size in dimensions):
            raise ValueError("Target RDA dimensions must be positive")
        choices = {
            "chirp_selection_method": {"legacy_64", "all"},
            "clutter_method": {"chirp_mean", "none"},
            "doppler_method": {"legacy_truncate", "full_center_bins"},
            "range_selection_method": {"global_energy", "motion_zero_doppler"},
        }
        for field, allowed in choices.items():
            if getattr(self, field) not in allowed:
                raise ValueError(f"Unknown {field}: {getattr(self, field)}")
        if self.range_selection_method == "motion_zero_doppler" and (
            self.doppler_method != "full_center_bins" or self.clutter_method != "none"
        ):
            raise ValueError("motion_zero_doppler requires full_center_bins and no chirp-mean subtraction")
        if self.range_search_bins is not None:
            if len(self.range_search_bins) != 2 or any(
                type(value) is not int for value in self.range_search_bins
            ):
                raise ValueError("range_search_bins must contain exactly two integers")
            lo, hi = self.range_search_bins
            if not (0 <= lo < hi):
                raise ValueError("range_search_bins must be increasing nonnegative integers")

    @classmethod
    def coherent_motion_candidate(cls) -> "FTUADCConfig":
        """Unpromoted train candidate that preserves complex zero Doppler."""
        return cls(chirp_selection_method="all", clutter_method="none",
                   doppler_method="full_center_bins",
                   range_selection_method="motion_zero_doppler")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FTUADCConfig":
        payload = dict(value)
        if payload.get("range_search_bins") is not None:
            payload["range_search_bins"] = tuple(payload["range_search_bins"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def converter_kwargs(self) -> dict[str, Any]:
        return {key: value for key, value in self.to_dict().items()
                if key != "schema_version"}

    def convert(self, radar: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        from src.data.loaders.export_all_datasets import convert_adc_cube_to_rda

        return convert_adc_cube_to_rda(radar, **self.converter_kwargs())
