"""
Streaming spectral subtraction.

- Jawna, stała latencja = fft_size.
- Bufor wyjściowy prefillowany latencją zer -> wyjście ma zawsze tyle
  próbek co wejście i jest identyczne dla dowolnego podziału na bloki.
- Minimum statistics: histogram PSD + wygładzenie min.
"""
import numpy as np


class SpectralSubtractor:
    def __init__(self, fs, fft_size=1024, alpha=2.0, beta=0.05,
                 min_stat_win=40):
        if fft_size % 2 != 0:
            raise ValueError("fft_size musi byc parzyste")
        self.fs = int(fs)
        self.fft_size = int(fft_size)
        self.hop = self.fft_size // 2
        self.latency = self.fft_size
        self.alpha = float(alpha)
        self.beta = float(beta)

        n = np.arange(self.fft_size, dtype=np.float32)
        self.window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / self.fft_size)
                       ).astype(np.float32)

        self.bins = self.fft_size // 2 + 1
        self.noise_psd = np.full(self.bins, 1e-6, dtype=np.float32)
        self.min_win = max(4, int(min_stat_win))
        self._psd_hist = np.full((self.min_win, self.bins), 1e-6,
                                 dtype=np.float32)
        self._hist_idx = 0

        self._in_buf = np.zeros(0, dtype=np.float32)
        self._ola_buf = np.zeros(self.fft_size, dtype=np.float32)
        self._out_queue = np.zeros(self.latency, dtype=np.float32)

    def reset(self):
        self.noise_psd[:] = 1e-6
        self._psd_hist[:] = 1e-6
        self._hist_idx = 0
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._ola_buf[:] = 0.0
        self._out_queue = np.zeros(self.latency, dtype=np.float32)

    def process(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        n_out = x.size
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)

        if self._in_buf.size:
            self._in_buf = np.concatenate([self._in_buf, x])
        else:
            self._in_buf = x.copy()

        produced = []
        while self._in_buf.size >= self.fft_size:
            frame = self._in_buf[:self.fft_size] * self.window
            X = np.fft.rfft(frame)
            P = (X.real ** 2 + X.imag ** 2).astype(np.float32)

            self._psd_hist[self._hist_idx] = P
            self._hist_idx = (self._hist_idx + 1) % self.min_win
            P_min = self._psd_hist.min(axis=0)
            self.noise_psd = 0.95 * self.noise_psd + 0.05 * P_min

            P_clean = np.maximum(P - self.alpha * self.noise_psd,
                                 self.beta * P)
            gain = np.sqrt(P_clean / (P + 1e-12)).astype(np.float32)
            Y = X * gain
            y = np.fft.irfft(Y, n=self.fft_size).astype(np.float32)

            self._ola_buf += y
            produced.append(self._ola_buf[:self.hop].copy())
            self._ola_buf[:-self.hop] = self._ola_buf[self.hop:].copy()
            self._ola_buf[-self.hop:] = 0.0
            self._in_buf = self._in_buf[self.hop:].copy()

        if produced:
            self._out_queue = np.concatenate([self._out_queue] + produced)

        if self._out_queue.size < n_out:
            pad = np.zeros(n_out - self._out_queue.size, dtype=np.float32)
            out = np.concatenate([self._out_queue, pad])
            self._out_queue = np.zeros(0, dtype=np.float32)
        else:
            out = self._out_queue[:n_out].copy()
            self._out_queue = self._out_queue[n_out:].copy()
        return out
