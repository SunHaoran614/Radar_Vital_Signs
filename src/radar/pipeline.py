"""Stage registry + ordered pipeline runner for the RDA front-end pool.

Each stage is a named function over a complex RDA cube ``(F, D, A, R)`` that
returns a new complex cube of the same shape *contract* (see notes per stage).
Stages are registered into ``RDA_STAGE_REGISTRY`` so :class:`RDAConfig` can
select them by string and so new paper-derived methods can be plugged in without
touching the runner.

Cube shape contract (used everywhere in this package):
    axis 0 = frames (time)
    axis 1 = doppler bins            (PhysDrive: 8; mmwave-897 has none -> insert D=1)
    axis 2 = angle / RX-spatial bins (PhysDrive: 16; mmwave-897: 8 virtual antennas)
    axis 3 = range bins             (PhysDrive: 8;  mmwave-897: 64)

mmwave-897 note: its loader yields a 3-D cube ``(F, A, R)`` (chirps were averaged
at storage time, so there is no Doppler axis). Use :func:`ensure_rda_4d` before
calling :func:`apply_rda_pipeline`; it inserts a singleton Doppler axis so the 4-D
pool can run unchanged (D=1).

``fs`` (frame rate, Hz) is a dataset property, not a per-stage constant. It is
pulled from :class:`RDAConfig.fs` and injected into every cube stage that accepts
an ``fs`` keyword; individual stage kwargs may override it per stage.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

# Nominal frame rate assumed when a config does not specify one. Matches the FTU
# loader (20 fps) and the codebase-wide DEFAULT_SAMPLING_RATE_HZ; mmwave-897 (10 Hz)
# must set RDAConfig.fs = 10.0.
DEFAULT_FS = 20.0

# cube shape convention used everywhere in this package:
#   axis 0 = frames (time, ~20 Hz)
#   axis 1 = doppler bins
#   axis 2 = angle / RX-spatial bins
#   axis 3 = range bins
RDARCube = np.ndarray  # complex64, shape (F, D, A, R)

# stage name -> {method_name: callable}
# Cube stages return a complex cube of the same (F,D,A,R) shape. The "phase"
# stage is cross-layer: it consumes the localized cube and returns a 1-D phase
# trace (see src/radar/phase.py and docs/frontend.md §2 layer boundary).
RDA_STAGE_REGISTRY: Dict[str, Dict[str, Callable[..., RDARCube]]] = {
    "clutter": {},
    "localization": {},
    "range_selection": {},
    "beamforming": {},
    "phase": {},
}

# Cube stages run in this fixed order; output is still a cube.
STAGE_ORDER = ("clutter", "localization", "range_selection", "beamforming")

# Cube stages that understand an ``fs`` (frame-rate) keyword. ``fs`` is injected
# from RDAConfig.fs so individual stages never hardcode the frame rate.
_FS_AWARE_STAGES = ("clutter", "localization", "range_selection", "beamforming")


def register_stage(stage: str, name: str, func: Callable[..., RDARCube]) -> None:
    if stage not in RDA_STAGE_REGISTRY:
        RDA_STAGE_REGISTRY[stage] = {}
    RDA_STAGE_REGISTRY[stage][name] = func


def list_stage_methods() -> Dict[str, List[str]]:
    return {stage: sorted(methods) for stage, methods in RDA_STAGE_REGISTRY.items()}


def ensure_rda_4d(cube: np.ndarray) -> np.ndarray:
    """Coerce a cube into the 4-D ``(F, D, A, R)`` contract.

    - 4-D input is returned as-is (complex64).
    - 3-D input ``(F, A, R)`` (e.g. mmwave-897, no Doppler axis) gets a singleton
      Doppler axis inserted at position 1 -> ``(F, 1, A, R)``.
    - 2-D input ``(F, R)`` is treated as single-antenna, no Doppler.

    Raises ValueError on any other rank.
    """
    cube = np.asarray(cube)
    if not np.iscomplexobj(cube):
        cube = cube.astype(np.complex64)
    else:
        cube = cube.astype(np.complex64)
    ndim = cube.ndim
    if ndim == 4:
        return cube
    if ndim == 3:
        return cube[:, np.newaxis, ...]  # (F, A, R) -> (F, 1, A, R)
    if ndim == 2:
        return cube[:, np.newaxis, np.newaxis, ...]  # (F, R) -> (F, 1, 1, R)
    raise ValueError(
        f"RDA cube must be 2-D/3-D/4-D, got {ndim}-D shape {cube.shape}"
    )


def _resolve(stage: str, method: str) -> Callable[..., RDARCube]:
    methods = RDA_STAGE_REGISTRY.get(stage, {})
    if method not in methods:
        from .config import StageNotFoundError

        raise StageNotFoundError(stage, method, list(methods))
    return methods[method]


def apply_rda_pipeline(
    cube: np.ndarray,
    config,
    *,
    record_meta: bool = False,
    return_phase: bool = False,
) -> Any:
    """Run the selected stages in order.

    Args:
        cube: complex RDA cube ``(F, D, A, R)`` (or 3-D/2-D; see :func:`ensure_rda_4d`).
        config: :class:`src.radar.config.RDAConfig` (or dataclass-compatible dict).
        record_meta: if True, also return a per-stage diagnostic dict.
        return_phase: if True and ``config.phase != "none"``, return the phase
            trace produced by the phase stage instead of the cube. Useful for the
            Phase 0 diagnostic / representation-probe gate without touching EDACM.

    Returns:
        The transformed cube, the phase trace, or ``(obj, meta)`` when
        ``record_meta`` is True.
    """
    cube = ensure_rda_4d(cube)
    if not np.iscomplexobj(cube):
        cube = cube.astype(np.complex64)
    else:
        cube = cube.astype(np.complex64)

    if hasattr(config, "to_dict"):
        cfg = config.to_dict()
    else:
        cfg = dict(config)

    fs = cfg.get("fs", None)
    meta: Dict[str, Any] = {}
    current = cube
    for stage in STAGE_ORDER:
        method = cfg.get(stage, "none")
        if method in (None, "none", ""):
            continue
        kwargs = dict(cfg.get(f"{stage}_kwargs", {}) or {})
        if fs is not None and stage in _FS_AWARE_STAGES:
            # per-stage kwarg wins over the config-wide fs
            kwargs.setdefault("fs", fs)
        func = _resolve(stage, method)
        before = current
        current = func(current, **kwargs)
        # enforce shape contract: stages may not change (F, D, A, R)
        if current.shape != before.shape:
            raise ValueError(
                f"RDA stage '{stage}/{method}' changed cube shape "
                f"{before.shape} -> {current.shape}; stages must preserve (F,D,A,R)."
            )
        if record_meta:
            meta[stage] = {"method": method, "kwargs": kwargs}

    # Cross-layer phase stage: operates on the localized cube, returns a trace.
    phase_method = cfg.get("phase", "none")
    phase_trace: Any = None
    if phase_method not in (None, "none", ""):
        phase_kwargs = cfg.get("phase_kwargs", {}) or {}
        phase_func = _resolve("phase", phase_method)
        phase_trace = phase_func(current, **phase_kwargs)
        if record_meta:
            meta["phase"] = {"method": phase_method, "kwargs": phase_kwargs}

    # Metadata must describe the same object returned by the phase-only call.
    # With no selected phase stage, retain the documented cube fallback.
    result = phase_trace if return_phase and phase_trace is not None else current
    if record_meta:
        return result, meta
    return result
