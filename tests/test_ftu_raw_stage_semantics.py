"""Raw FTU IQ and frequency-axis physical invariants."""

import unittest

import numpy as np

from src.data.loaders.ftu_loader import FTUDataLoader
from src.data.loaders.export_all_datasets import _build_rda_cube_from_frame


class FtuRawStageSemanticsTest(unittest.TestCase):
    def test_i1_i2_q1_q2_unpack_and_axes(self) -> None:
        loader = FTUDataLoader.__new__(FTUDataLoader)
        loader.NUM_FRAMES = 1
        loader.NUM_CHIRPS = 2
        loader.NUM_RX = 2
        loader.NUM_ADC = 4
        raw = []
        for chirp in range(2):
            for rx in range(2):
                b = 100 * chirp + 10 * rx
                raw.extend((b + 1, b + 2, b + 11, b + 12,
                            b + 3, b + 4, b + 13, b + 14))
        cube = loader._reshape_radar_data(np.asarray(raw, dtype=np.int16), num_frames=1)
        self.assertEqual(cube.shape, (1, 2, 2, 4))
        np.testing.assert_array_equal(cube[0, 1, 1],
                                      [111 + 121j, 112 + 122j, 113 + 123j, 114 + 124j])

    def test_chirp_mean_erases_coherent_zero_doppler(self) -> None:
        samples = np.arange(32)
        tone = np.exp(2j * np.pi * 5 * samples / 32)
        frame = np.broadcast_to(tone, (4, 16, 32)).astype(np.complex64).copy()
        no_clutter = _build_rda_cube_from_frame(
            frame, np.arange(16), samples, 8, 16,
            doppler_method="full_center_bins", clutter_method="none")
        mean_removed = _build_rda_cube_from_frame(
            frame, np.arange(16), samples, 8, 16,
            doppler_method="full_center_bins", clutter_method="chirp_mean")
        self.assertGreater(float(np.max(abs(no_clutter[4]))), 100)
        self.assertLess(float(np.max(abs(mean_removed))), 1e-4)


if __name__ == "__main__":
    unittest.main()
