"""Syntetyczny generator CW (używany przez tryb DEMO i testy)."""
import numpy as np

from core.morse import MORSE_TABLE

MORSE_REV = {v: k for k, v in MORSE_TABLE.items()}


def text_to_elements(text):
    out = []
    words = text.upper().strip().split()
    for wi, word in enumerate(words):
        if wi > 0:
            out.append((" ", 7))
        for ci, ch in enumerate(word):
            code = MORSE_REV.get(ch)
            if code is None:
                continue
            if ci > 0:
                out.append((" ", 3))
            for si, sym in enumerate(code):
                if si > 0:
                    out.append((" ", 1))
                out.append(("*", 1 if sym == "." else 3))
    return out


def generate_cw(text, wpm=20.0, tone=700.0, fs=48000, amp=0.5,
                snr_db=None, lead=0.3, trail=1.5, seed=42):
    rng = np.random.default_rng(seed)
    dot = 1.2 / wpm
    chunks = [np.zeros(int(lead * fs), dtype=np.float32)]
    for sym, units in text_to_elements(text):
        n = int(round(units * dot * fs))
        if sym == "*":
            t = np.arange(n, dtype=np.float64) / fs
            chunks.append((amp * np.sin(2 * np.pi * tone * t)).astype(np.float32))
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    chunks.append(np.zeros(int(trail * fs), dtype=np.float32))
    sig = np.concatenate(chunks).astype(np.float32)
    if snr_db is not None:
        mark_power = (amp ** 2) / 2.0
        noise_power = mark_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=len(sig))
        sig = (sig + noise).astype(np.float32)
    return sig, tone
