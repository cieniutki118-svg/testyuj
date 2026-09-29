"""Wyjście audio z ring bufferem SPSC."""
import threading
import numpy as np
import sounddevice as sd


class AudioOutput:
    def __init__(self, samplerate=48000, blocksize=1024,
                 capacity_sec=0.5):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.capacity = max(blocksize * 8,
                            int(samplerate * capacity_sec))
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._read = 0
        self._write = 0
        self._size = 0
        self._lock = threading.Lock()
        self._stream = None
        self.enabled = False

    def feed(self, samples):
        if not self.enabled or samples is None or len(samples) == 0:
            return
        x = np.asarray(samples, dtype=np.float32)
        n = len(x)
        if n > self.capacity:
            x = x[-self.capacity:]
            n = self.capacity
        with self._lock:
            free = self.capacity - self._size
            if n > free:
                drop = n - free
                self._read = (self._read + drop) % self.capacity
                self._size -= drop
            end = self._write + n
            if end <= self.capacity:
                self._buf[self._write:end] = x
            else:
                first = self.capacity - self._write
                self._buf[self._write:] = x[:first]
                self._buf[:end - self.capacity] = x[first:]
            self._write = end % self.capacity
            self._size += n

    def _pull(self, frames, out):
        with self._lock:
            avail = self._size
            take = min(avail, frames)
            if take > 0:
                end = self._read + take
                if end <= self.capacity:
                    out[:take, 0] = self._buf[self._read:end]
                else:
                    first = self.capacity - self._read
                    out[:first, 0] = self._buf[self._read:]
                    out[first:take, 0] = self._buf[:end - self.capacity]
                self._read = end % self.capacity
                self._size -= take
            if take < frames:
                out[take:, 0] = 0.0

    def start(self, device=None):
        def _cb(outdata, frames, time_info, status):
            self._pull(frames, outdata)

        self._stream = sd.OutputStream(
            device=device, samplerate=self.samplerate,
            blocksize=self.blocksize, channels=1,
            dtype="float32", callback=_cb)
        self._stream.start()
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._read = self._write = self._size = 0

    def set_device(self, device_index):
        was_enabled = self.enabled
        if was_enabled:
            self.stop()
        self.device = device_index
        if was_enabled:
            self.start(device=device_index)
