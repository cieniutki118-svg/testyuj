"""Nagrywanie WAV."""
import numpy as np
import soundfile as sf
from datetime import datetime
from pathlib import Path


class WAVRecorder:
    def __init__(self, samplerate=48000, out_dir="recordings"):
        self.samplerate = samplerate
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._path = None
        self._recording = False

    def start(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self.out_dir / f"sp5mig_{ts}.wav"
        self._file = sf.SoundFile(
            str(self._path), mode="w", samplerate=self.samplerate,
            channels=1, subtype="PCM_16")
        self._recording = True
        return self._path

    def write(self, samples):
        if not self._recording or self._file is None:
            return
        data = np.clip(samples, -1.0, 1.0).astype(np.float32)
        self._file.write(data)

    def stop(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self._recording = False
        return self._path

    @property
    def recording(self):
        return self._recording

    @property
    def path(self):
        return self._path
