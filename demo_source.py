"""
Tryb DEMO: generuje prawdziwy, syntetyczny sygnał CW i emituje go
w blokach tak, jak zrobiłby to capture audio. Ten sam tor DSP.
"""
import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.cw_synth import generate_cw


class DemoSource(QThread):
    block_ready = Signal(object)

    def __init__(self, fs=48000, blocksize=1024,
                 text="CQ CQ DE SP5MIG TEST", wpm=20.0,
                 tone=700.0, snr_db=25.0):
        super().__init__()
        self.fs = fs
        self.blocksize = blocksize
        self.text = text
        self.wpm = wpm
        self.tone = tone
        self.snr_db = snr_db
        self._running = False
        self._signal = None
        self._pos = 0

    def _build_signal(self):
        sig, _ = generate_cw(
            self.text, wpm=self.wpm, tone=self.tone, fs=self.fs,
            amp=0.5, snr_db=self.snr_db, lead=0.5, trail=1.5, seed=1234)
        return sig

    def run(self):
        self._signal = self._build_signal()
        self._pos = 0
        self._running = True
        period = self.blocksize / self.fs
        next_t = time.perf_counter()
        while self._running:
            n = self.blocksize
            end = self._pos + n
            if end > self._signal.size:
                # pętla
                self._pos = 0
                end = n
            block = self._signal[self._pos:end].copy()
            if block.size < n:
                pad = np.zeros(n - block.size, dtype=np.float32)
                block = np.concatenate([block, pad])
            self._pos = end
            self.block_ready.emit(block)
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def stop(self):
        self._running = False
        self.wait(2000)

    def set_text(self, text):
        self.text = text
        self._signal = None
        self._pos = 0
