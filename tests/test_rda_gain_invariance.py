"""Known-target and gain-unit regressions for the experimental RDA pool."""
import numpy as np
import pytest

from src.radar.localization import (
    localization_energy, localization_spatial_consistency,
    localization_temporal_consistency, localization_hr_band,
)
from src.radar.range_selection import (
    range_global_energy, range_peak, range_weighted_center, range_hr_band,
)
from src.radar.phase import (
    _select_target_bin, phase_single_bin, phase_interframe_diff,
    phase_multibin_fusion,
)
from src.radar.numerics import magnitude_squared


def example():
    t = np.arange(256) / 20
    phase = .3 * np.sin(2 * np.pi * 1.25 * t)
    cube = np.zeros((256, 1, 1, 12), np.complex64)
    cube[:, 0, 0, 9] = np.exp(1j * phase)
    cube[:, 0, 0, 2] = .2
    return cube, phase


@pytest.mark.parametrize('gain', [1e-25, 1e-8, 1e8, 1e25])
@pytest.mark.parametrize('operator', [
    localization_energy, localization_spatial_consistency,
    localization_temporal_consistency, localization_hr_band,
    range_global_energy, range_peak, range_weighted_center, range_hr_band,
])
def test_spatial_selection_is_independent_of_iq_gain_units(operator, gain):
    cube, _ = example()
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        reference = operator(cube)
        scaled = operator(cube * gain) / gain
    np.testing.assert_allclose(scaled, reference, rtol=2e-6, atol=1e-7)
    assert abs(scaled[:, 0, 0, 9]).mean() > .99


@pytest.mark.parametrize('gain', [1e-25, 1e-8, 1., 1e8, 1e25])
def test_energy_picker_and_conjugate_phase_retain_known_target(gain):
    cube, phase = example()
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        assert _select_target_bin(cube * gain) == (0, 0, 9)
        delta = phase_interframe_diff(cube * gain, normalize=False)
    expected = np.r_[0., np.diff(phase)]
    np.testing.assert_allclose(delta, expected, rtol=0, atol=2e-7)


@pytest.mark.parametrize('gain', [1e-25, 1e-8, 1e8, 1e25])
@pytest.mark.parametrize('operator', [phase_single_bin, phase_multibin_fusion])
def test_phase_selection_and_fusion_are_gain_invariant(operator, gain):
    cube, _ = example()
    kwargs = dict(normalize=False)
    if operator is phase_multibin_fusion:
        kwargs['top_bins'] = 2
    np.testing.assert_allclose(operator(cube * gain, **kwargs),
                               operator(cube, **kwargs), rtol=0, atol=3e-7)


def test_zero_energy_fallback_and_float32_power_extremes():
    cube, _ = example()
    for operator in (range_weighted_center, range_hr_band):
        np.testing.assert_array_equal(operator(np.zeros_like(cube)),
                                       np.zeros_like(cube))
    z = np.array([1e-25+1e-25j, 1e25+1e25j], dtype=np.complex64)
    power = magnitude_squared(z)
    assert power.dtype == np.float64
    assert (power > 0).all() and np.isfinite(power).all()
    np.testing.assert_allclose(np.sqrt(power / 2), z.real.astype(float), rtol=1e-14)
