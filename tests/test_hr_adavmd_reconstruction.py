"""Regression checks for the decomposition failure found in legacy exports."""

import unittest

import numpy as np

from src.features.edacm import zscore_1d
from src.features.hr_adavmd import (
    DECOMPOSITION_ALGORITHM_VERSION,
    hr_adavmd_decompose,
    vmd_decompose,
)


class VmdReconstructionTests(unittest.TestCase):
    def test_single_heart_tone_reconstructs(self) -> None:
        t = np.arange(256) / 20.0
        signal = np.sin(2 * np.pi * 1.25 * t)
        modes = vmd_decompose(signal)
        self.assertEqual(modes.shape, (7, 256))
        self.assertTrue(np.isfinite(modes).all())
        self.assertGreater(float(np.corrcoef(zscore_1d(signal), modes.sum(axis=0))[0, 1]), 0.99)

    def test_respiration_and_heart_tones_reconstruct(self) -> None:
        t = np.arange(256) / 20.0
        signal = 3 * np.sin(2 * np.pi * 0.25 * t) + 0.3 * np.sin(2 * np.pi * 1.25 * t)
        modes = vmd_decompose(signal)
        self.assertGreater(float(np.corrcoef(zscore_1d(signal), modes.sum(axis=0))[0, 1]), 0.99)
        weighted, meta = hr_adavmd_decompose(signal)
        self.assertTrue(np.isfinite(weighted).all())
        self.assertEqual(meta["decomposition_algorithm_version"], DECOMPOSITION_ALGORITHM_VERSION)

    def test_constant_signal_stays_zero(self) -> None:
        modes = vmd_decompose(np.ones(256))
        self.assertTrue(np.array_equal(modes, np.zeros((7, 256), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
