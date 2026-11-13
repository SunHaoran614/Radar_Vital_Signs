import unittest
from src.features.waveform_contract import WaveformContract


class WaveformContractTests(unittest.TestCase):
    def test_original_and_native_duration(self):
        self.assertEqual(WaveformContract(3600, 120., 4).validate([45., 180.], 10.),
                         dict(output_samples=900, output_hz=30., duration_seconds=30., crop_samples=300))
        for n, fs in [(256, 20.), (128, 10.), (384, 30.)]:
            c = WaveformContract(n, fs)
            self.assertEqual(c.validate([46.875, 206.25], 10.)['crop_samples'], 10*fs)
            self.assertEqual(c.duration_seconds, 12.8)

    def test_unsupported_grid_and_crop_fail(self):
        for n, fs, boundary in [(256, 20., 150.), (128, 10., 75.)]:
            c = WaveformContract(n, fs, 4)
            with self.assertRaisesRegex(ValueError, 'Nyquist'):
                c.validate([boundary], 10.)
            with self.assertRaisesRegex(ValueError, 'Nyquist'):
                c.validate([206.25], 10.)
        for n, fs in [(256, 20.), (128, 10.), (384, 30.)]:
            c = WaveformContract(n, fs, 4)
            with self.assertRaisesRegex(ValueError, 'outside'):
                c.validate([50.], 300/c.output_hz)
        with self.assertRaisesRegex(ValueError, 'integer samples'):
            WaveformContract(128, 10., 4).validate([50.], 1.)

    def test_invalid_physical_parameters(self):
        for n, fs, factor in [(255, 20., 4), (256, float('nan'), 4), (256, 0., 4),
                              (256, 20., 0), (256., 20., 4), (256, 20., True)]:
            with self.assertRaises(ValueError):
                WaveformContract(n, fs, factor)
        for grid, crop in [([], 10.), ([float('nan')], 10.), ([0.], 10.),
                            ([60.], -1.), ([60.], float('inf'))]:
            with self.assertRaises(ValueError):
                WaveformContract(256, 20.).validate(grid, crop)


if __name__ == '__main__':
    unittest.main()
