"""Already-formed Doppler/angle spectra keep their bin meaning on resize."""

import numpy as np

from src.data.loaders.export_all_datasets import align_rda_shape


def test_center_crop_keeps_physical_spectral_bin_value() -> None:
    radar = np.zeros((2, 12, 20, 8), dtype=np.complex64)
    radar[:, 6, 10, 3] = 3 + 4j
    aligned = align_rda_shape(radar, 8, 16, 8)
    assert aligned.shape == (2, 8, 16, 8)
    np.testing.assert_array_equal(aligned[:, 4, 8, 3], [3 + 4j, 3 + 4j])
    assert np.count_nonzero(aligned) == 2


def test_center_pad_keeps_physical_spectral_bin_value() -> None:
    radar = np.zeros((1, 4, 8, 8), dtype=np.complex64)
    radar[0, 2, 4, 3] = 3 + 4j
    aligned = align_rda_shape(radar, 8, 16, 8)
    assert aligned.shape == (1, 8, 16, 8)
    assert aligned[0, 4, 8, 3] == 3 + 4j
    assert np.count_nonzero(aligned) == 1


def test_already_target_shape_is_unchanged() -> None:
    radar = np.arange(8 * 16 * 8).reshape(1, 8, 16, 8).astype(np.complex64)
    np.testing.assert_array_equal(align_rda_shape(radar), radar)
