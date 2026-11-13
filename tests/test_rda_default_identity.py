"""Default post-conversion RDA configuration must preserve the input cube."""

import unittest

import numpy as np

from src.radar import RDAConfig, apply_rda_pipeline


class RDADefaultIdentityTests(unittest.TestCase):
    def test_default_does_not_silently_mask_a_converted_cube(self) -> None:
        rng = np.random.default_rng(20261016)
        cube = (rng.normal(size=(16, 8, 16, 8)) +
                1j * rng.normal(size=(16, 8, 16, 8))).astype(np.complex64)
        actual = apply_rda_pipeline(cube, RDAConfig())
        np.testing.assert_array_equal(actual, cube)

    def test_explicit_localization_still_changes_spatial_weights(self) -> None:
        cube = np.ones((16, 8, 16, 8), dtype=np.complex64)
        actual = apply_rda_pipeline(cube, RDAConfig(localization="energy"))
        self.assertEqual(actual.shape, cube.shape)
        self.assertGreater(float(np.max(np.abs(actual - cube))), 0.5)


if __name__ == "__main__":
    unittest.main()
