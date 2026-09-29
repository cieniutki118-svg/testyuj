"""Łańcuch DSP: bandpass + spectral subtraction."""
import numpy as np
from scipy import signal

from core.spectral import SpectralSubtractor


class BiquadFilter:
    def __init__(self, b, a):
        self.b = b
        self.a = a
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def process(self, x):
        y, self.zi = signal.lfilter(self.b, self.a, x, zi=self.zi)
        return y

    def reset(self):
        self.zi[:] = 0


def design_bandpass(low_hz, high_hz, fs, order=4):
    nyq = fs / 2.0
    low = max(1e-3, min(low_hz / nyq, 0.999))
    high = max(low + 1e-3, min(high_hz / nyq, 0.999))
    b, a = signal.butter(order, [low, high], btype="band")
    return BiquadFilter(b, a)


class DSPChain:
    def __init__(self, fs, low=300, high=3000, use_spectral=True,
                 fft_size=1024):
        self.fs = fs
        self.bandpass = design_bandpass(low, high, fs)
        self.spectral = (SpectralSubtractor(fs, fft_size=fft_size)
                         if use_spectral else None)

    def process(self, x):
        y = self.bandpass.process(x)
        if self.spectral is not None:
            y = self.spectral.process(y)
        return y

    def reset(self):
        self.bandpass.reset()
        if self.spectral is not None:
            self.spectral.reset()
