"""Regression checks for explicit Doppler chirp policies."""

import unittest

import numpy as np

from src.data.loaders.export_all_datasets import _build_rda_cube_from_frame, convert_adc_cube_to_rda


class DopplerPolicyTests(unittest.TestCase):
    def test_motion_range_policy_keeps_slow_time_phase(self) -> None:
        frames, chirps, samples = 32, 16, 32
        adc = np.zeros((frames, 4, chirps, samples), dtype=np.complex64)
        n = np.arange(samples)
        for frame in range(frames):
            adc[frame] += 100 * np.exp(2j * np.pi * 5 * n / samples)
            adc[frame] += (20 * np.exp(0.8j * np.sin(2 * np.pi * 1.4 * frame / 20))
                           * np.exp(2j * np.pi * 8 * n / samples))
        kwargs = dict(target_doppler=8, target_angle=8, target_range=8,
                      doppler_method="full_center_bins", clutter_method="none",
                      chirp_selection_method="all", range_search_bins=(2, 13))
        _, _, energy_center = convert_adc_cube_to_rda(adc, **kwargs)
        cube, bins, motion_center = convert_adc_cube_to_rda(
            adc, range_selection_method="motion_zero_doppler", **kwargs)
        self.assertEqual(energy_center, 5)
        self.assertEqual(motion_center, 8)
        trace = cube[:, 4, 4, int(np.flatnonzero(bins == 8)[0])]
        self.assertGreater(float(np.std(np.unwrap(np.angle(trace)))), 0.1)

    def test_motion_policy_rejects_zero_doppler_removal(self) -> None:
        adc = np.zeros((2, 4, 16, 32), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "requires full_center_bins"):
            convert_adc_cube_to_rda(adc, target_doppler=8, target_angle=8,
                doppler_method="full_center_bins", clutter_method="chirp_mean",
                chirp_selection_method="all", range_selection_method="motion_zero_doppler")

    def test_cached_range_matches_full_chirp_zero_doppler_angle_slice(self) -> None:
        rng = np.random.default_rng(17)
        adc = (rng.normal(size=(6, 4, 16, 32))
               + 1j * rng.normal(size=(6, 4, 16, 32))).astype(np.complex64)
        cached_range = np.fft.fft(adc.mean(axis=2), axis=-1)[..., :16]
        expected = np.fft.fftshift(np.fft.fft(cached_range, n=8, axis=1), axes=1) * 16
        actual = np.stack([_build_rda_cube_from_frame(
            frame, np.arange(16), np.arange(32), 8, 8,
            doppler_method="full_center_bins", clutter_method="none")[4]
            for frame in adc])
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-3)

    def test_all_chirps_policy_uses_late_chirps(self) -> None:
        adc = np.zeros((3, 4, 128, 32), dtype=np.complex64)
        n = np.arange(32)
        adc[:, :, 64:] = np.exp(2j * np.pi * 6 * n / 32)
        kwargs = dict(target_doppler=8, target_angle=8, target_range=8,
                      doppler_method="full_center_bins", clutter_method="none",
                      range_selection_method="global_energy", range_search_bins=(3, 10))
        full, _, _ = convert_adc_cube_to_rda(adc, chirp_selection_method="all", **kwargs)
        legacy, _, _ = convert_adc_cube_to_rda(adc, chirp_selection_method="legacy_64", **kwargs)
        self.assertGreater(float(np.max(np.abs(full))), float(np.max(np.abs(legacy))) + 100)

    def test_full_method_uses_chirps_discarded_by_legacy(self) -> None:
        frame = np.zeros((2, 16, 32), dtype=np.complex64)
        frame[:, 8:, :] = np.arange(8, dtype=np.float32)[None, :, None] + 1
        chirps = np.arange(16)
        samples = np.arange(32)
        legacy = _build_rda_cube_from_frame(frame, chirps, samples, 8, 4,
                                            doppler_method="legacy_truncate")
        full = _build_rda_cube_from_frame(frame, chirps, samples, 8, 4,
                                          doppler_method="full_center_bins")
        self.assertEqual(full.shape, (8, 4, 16))
        self.assertGreater(float(np.max(np.abs(full - legacy))), 1.0)

    def test_chirp_mean_removes_zero_doppler_only_when_using_all_chirps(self) -> None:
        rng = np.random.default_rng(4)
        frame = (rng.normal(size=(2, 16, 32)) + 1j * rng.normal(size=(2, 16, 32))).astype(np.complex64)
        full = _build_rda_cube_from_frame(frame, np.arange(16), np.arange(32), 8, 4,
                                          doppler_method="full_center_bins",
                                          clutter_method="chirp_mean")
        self.assertLess(float(np.max(np.abs(full[4]))), 1e-4)


if __name__ == "__main__":
    unittest.main()
