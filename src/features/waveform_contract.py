"""Physical time/frequency contract for waveform-output models (no resampling)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WaveformContract:
    input_samples: int
    input_hz: float
    downsample: int = 1

    def __post_init__(self):
        if type(self.input_samples) is not int or self.input_samples < 2:
            raise ValueError('input_samples must be an integer >= 2')
        if type(self.downsample) is not int or self.downsample < 1:
            raise ValueError('downsample must be a positive integer')
        if not math.isfinite(self.input_hz) or self.input_hz <= 0:
            raise ValueError('input_hz must be positive and finite')
        if self.input_samples % self.downsample:
            raise ValueError('Exact duration requires divisible input length')

    @property
    def output_samples(self):
        return self.input_samples // self.downsample

    @property
    def output_hz(self):
        return self.input_hz / self.downsample

    @property
    def duration_seconds(self):
        # N/fs is DFT/block duration, not timestamp-last minus timestamp-first.
        return self.input_samples / self.input_hz

    def validate(self, bpm_grid, crop_seconds):
        grid = list(bpm_grid)
        if not grid or any(not math.isfinite(b) or b <= 0 for b in grid):
            raise ValueError('Positive finite BPM grid required')
        if any(b >= 30 * self.output_hz for b in grid):
            raise ValueError('BPM grid reaches/exceeds output Nyquist')
        if not math.isfinite(crop_seconds) or crop_seconds <= 0:
            raise ValueError('Positive finite crop duration required')
        length = crop_seconds * self.output_hz
        if not math.isclose(length, round(length), abs_tol=1e-9, rel_tol=0):
            raise ValueError('Crop duration does not give integer samples')
        if not 2 <= round(length) <= self.output_samples:
            raise ValueError('Crop samples outside output length')
        return dict(output_samples=self.output_samples, output_hz=self.output_hz,
                    duration_seconds=self.duration_seconds, crop_samples=round(length))
