"""Analizator widma FFT z uśrednianiem."""
import numpy as np


class FFTAnalyzer:
    def __init__(self, fs, fft_size=4096, avg=2):
        self.fs = fs
        self.fft_size = fft_size
        self.avg = max(1, avg)
        self.window = np.hanning(fft_size).astype(np.float32)
        self.buffer = np.zeros(fft_size, dtype=np.float32)
        self.spec_db = None
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / fs).astype(np.float32)

    def process(self, samples):
        n = len(samples)
        if n >= self.fft_size:
            self.buffer[:] = samples[-self.fft_size:]
        else:
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = samples
        x = self.buffer * self.window
        X = np.fft.rfft(x)
        mag = np.abs(X) / (self.fft_size / 2.0)
        spec_db = 20.0 * np.log10(mag + 1e-9)
        if self.spec_db is None:
            self.spec_db = spec_db.astype(np.float32)
        else:
            a = 1.0 / self.avg
            self.spec_db = (1.0 - a) * self.spec_db + a * spec_db
        return self.spec_db
