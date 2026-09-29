"""Worker przetwarzania: DSP, FFT, Morse, recorder, output."""
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from core.dsp import DSPChain
from core.fft_analyzer import FFTAnalyzer
from core.morse import MorseDecoder
from core.audio_out import AudioOutput
from core.recorder import WAVRecorder


class ProcessingWorker(QObject):
    spectrum_ready = Signal(object, object)
    level_ready = Signal(float, float, float)
    text_ready = Signal(str)
    wpm_ready = Signal(float)
    recording_started = Signal(str)
    recording_stopped = Signal(str)
    shutdown_done = Signal()

    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.dsp = DSPChain(fs, low=300, high=3000, use_spectral=True)
        self.fft = FFTAnalyzer(fs, fft_size=4096, avg=2)
        self.morse = MorseDecoder(fs, tone_freq=700.0)
        self.recorder = WAVRecorder(fs)
        self.output = AudioOutput(fs)

        self.dsp_enabled = True
        self.morse_enabled = True
        self._shutdown = False

    @Slot(object)
    def process_block(self, samples):
        if self._shutdown or samples is None or len(samples) == 0:
            return
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        level_db = 20.0 * np.log10(rms + 1e-9)

        processed = self.dsp.process(samples) if self.dsp_enabled else samples

        spec_db = self.fft.process(processed)
        self.spectrum_ready.emit(self.fft.freqs, spec_db)

        snr_db, peak_hz = self._compute_snr(self.fft.freqs, spec_db)
        self.level_ready.emit(level_db, snr_db, peak_hz)

        if self.morse_enabled:
            txt = self.morse.process(samples)
            if txt:
                self.text_ready.emit(txt)
            self.wpm_ready.emit(self.morse.wpm)

        if self.recorder.recording:
            self.recorder.write(processed)

        self.output.feed(processed)

    @staticmethod
    def _compute_snr(freqs, spec_db, f_min=200.0):
        mask = freqs >= f_min
        s = spec_db[mask]
        if s.size < 20:
            return 0.0, 0.0
        sorted_s = np.sort(s)
        noise = float(np.median(sorted_s[: s.size // 2]))
        peak_idx = int(np.argmax(s))
        peak = float(s[peak_idx])
        return peak - noise, float(freqs[mask][peak_idx])

    @Slot(float)
    def set_tone(self, freq):
        self.morse.set_tone_freq(freq)

    @Slot(bool)
    def set_dsp_enabled(self, v):
        self.dsp_enabled = bool(v)
        if not v:
            self.dsp.reset()

    @Slot(bool)
    def set_morse_enabled(self, v):
        self.morse_enabled = bool(v)

    @Slot(bool)
    def set_output_enabled(self, v):
        if v and not self.output.enabled:
            try:
                self.output.start()
            except Exception:
                pass
        elif not v and self.output.enabled:
            self.output.stop()

    @Slot(int)
    def set_output_device(self, device_index):
        try:
            self.output.set_device(device_index)
        except Exception:
            pass

    @Slot()
    def reset_morse(self):
        self.morse.reset()

    @Slot()
    def start_recording(self):
        if self.recorder.recording:
            return
        path = self.recorder.start()
        self.recording_started.emit(str(path) if path else "")

    @Slot()
    def stop_recording(self):
        if not self.recorder.recording:
            return
        path = self.recorder.stop()
        self.recording_stopped.emit(str(path) if path else "")

    @Slot()
    def flush_morse(self):
        tail = self.morse.flush()
        if tail:
            self.text_ready.emit(tail)

    @Slot()
    def shutdown(self):
        if self._shutdown:
            self.shutdown_done.emit()
            return
        self._shutdown = True
        try:
            self.flush_morse()
        except Exception:
            pass
        try:
            if self.recorder.recording:
                self.recorder.stop()
        except Exception:
            pass
        try:
            self.output.stop()
        except Exception:
            pass
        self.shutdown_done.emit()
