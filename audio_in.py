"""Wejście audio: lista urządzeń + capture (callback -> sygnał Qt)."""
import sounddevice as sd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from PySide6.QtCore import QObject, Signal


USB_KEYWORDS = ("usb", "codec", "uac", "audio device", "soundblaster",
                "focusrite", "behringer", "scarlett", "yamaha", "steinberg")


@dataclass
class AudioDeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float
    is_usb: bool


def list_input_devices():
    out = []
    try:
        devices = sd.query_devices()
    except Exception:
        return out
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            name = d.get("name", f"Device {i}")
            is_usb = any(k in name.lower() for k in USB_KEYWORDS)
            out.append(AudioDeviceInfo(
                i, name, int(d["max_input_channels"]),
                float(d["default_samplerate"]), is_usb))
    return out


def list_output_devices():
    out = []
    try:
        devices = sd.query_devices()
    except Exception:
        return out
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            out.append((i, d.get("name", f"Output {i}")))
    return out


def find_usb_input():
    devices = list_input_devices()
    for d in devices:
        if d.is_usb:
            return d
    return devices[0] if devices else None


class AudioCapture(QObject):
    block_ready = Signal(object)
    error = Signal(str)

    def __init__(self, device_index, samplerate=48000, blocksize=1024,
                 channels=1):
        super().__init__()
        self.device_index = device_index
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self._stream = None

    def start(self):
        def _cb(indata, frames, time_info, status):
            try:
                data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                self.block_ready.emit(data)
            except Exception as e:
                self.error.emit(str(e))

        try:
            self._stream = sd.InputStream(
                device=self.device_index, samplerate=self.samplerate,
                blocksize=self.blocksize, channels=self.channels,
                dtype="float32", callback=_cb)
            self._stream.start()
        except Exception as e:
            self.error.emit(f"Nie mozna otworzyc wejscia: {e}")

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
