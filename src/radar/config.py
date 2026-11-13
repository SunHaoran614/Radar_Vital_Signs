"""RDA configuration: one selectable method per processing stage.

A single :class:`RDAConfig` describes the post-conversion cube stages. The
default is an identity transform: an experiment must explicitly select every
spatial mask or weighting stage it intends to test.

``fs`` is the cube frame rate in Hz (a dataset property). It is injected into
every cube stage that accepts an ``fs`` keyword, so stages never hardcode it.
Defaults to 20.0 (FTU / codebase convention); set 10.0 for mmwave-897.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from .pipeline import RDA_STAGE_REGISTRY, DEFAULT_FS


# The historical defaults applied energy localization and a second range mask
# to an already cropped RDA cube. That was not the ADC exporter baseline.
# Identity defaults prevent an unrequested spatial change to model inputs.
@dataclass
class RDAConfig:
    clutter: str = "none"
    localization: str = "none"
    range_selection: str = "none"
    beamforming: str = "none"
    phase: str = "none"

    # Frame rate of the cube in Hz (dataset property, injected into cube stages).
    fs: float = DEFAULT_FS

    # Optional stage-specific kwargs forwarded to the selected method.
    clutter_kwargs: Dict[str, Any] = field(default_factory=dict)
    localization_kwargs: Dict[str, Any] = field(default_factory=dict)
    range_selection_kwargs: Dict[str, Any] = field(default_factory=dict)
    beamforming_kwargs: Dict[str, Any] = field(default_factory=dict)
    phase_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RDAConfig":
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = {k: v for k, v in data.items() if k not in known}
        if extra:
            raise ValueError(f"Unknown RDAConfig field(s): {sorted(extra)}")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def cube_stages(self) -> tuple[str, str, str, str]:
        """The four stages that return a cube of the same shape contract."""
        return ("clutter", "localization", "range_selection", "beamforming")

    def validate(self) -> None:
        """Raise if any selected method is not registered."""
        for stage, method in (
            ("clutter", self.clutter),
            ("localization", self.localization),
            ("range_selection", self.range_selection),
            ("beamforming", self.beamforming),
            ("phase", self.phase),
        ):
            # "none" / "" / None means "stage disabled"; always valid.
            if method in (None, "none", ""):
                continue
            if method not in RDA_STAGE_REGISTRY.get(stage, {}):
                raise StageNotFoundError(stage, method, list(RDA_STAGE_REGISTRY.get(stage, {})))


class StageNotFoundError(KeyError):
    """Raised when a config references a stage method that is not registered."""

    def __init__(self, stage: str, method: str, available: list):
        self.stage = stage
        self.method = method
        self.available = available
        super().__init__(
            f"RDA stage '{stage}' has no method '{method}'. "
            f"Available: {available}"
        )
