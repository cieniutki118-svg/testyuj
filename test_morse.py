"""Testy dekodera CW - w tym niezależność od rozmiaru bloku."""
import numpy as np
import pytest

from core.morse import MorseDecoder
from core.cw_synth import generate_cw

FS = 48000
TONE = 700.0


def _decode(signal, wpm_hint=20.0):
    dec = MorseDecoder(FS, tone_freq=TONE, wpm_hint=wpm_hint)
    out = dec.process(signal)
    out += dec.flush()
    return out


def _decode_blocks(signal, block_size, wpm_hint=20.0):
    dec = MorseDecoder(FS, tone_freq=TONE, wpm_hint=wpm_hint)
    out = ""
    for i in range(0, signal.size, block_size):
        out += dec.process(signal[i:i + block_size])
    out += dec.flush()
    return out


@pytest.mark.parametrize("text", ["PARIS", "HELLO", "TEST", "CQ", "DE"])
def test_decode_clean(text):
    sig, _ = generate_cw(text, wpm=20, tone=TONE, fs=FS)
    assert text in _decode(sig, 20)


def test_last_char_flushed_explicit():
    sig, _ = generate_cw("E", wpm=20, tone=TONE, fs=FS, trail=2.0)
    assert "E" in _decode(sig, 20)


def test_last_char_flushed_idle():
    sig, _ = generate_cw("E", wpm=20, tone=TONE, fs=FS, trail=2.0)
    dec = MorseDecoder(FS, tone_freq=TONE, wpm_hint=20)
    out = dec.process(sig)
    assert "E" in out


def test_word_gap_present():
    sig, _ = generate_cw("HI HI", wpm=20, tone=TONE, fs=FS)
    out = _decode(sig, 20)
    assert " " in out.strip()
    assert "HI" in out


def test_no_word_gap_inside_word():
    sig, _ = generate_cw("PARIS", wpm=20, tone=TONE, fs=FS)
    out = _decode(sig, 20)
    assert " " not in out.strip()


@pytest.mark.parametrize("snr_db", [30, 20, 15])
def test_decode_with_noise(snr_db):
    sig, _ = generate_cw("CQ DE SP5MIG", wpm=20, tone=TONE, fs=FS,
                         snr_db=snr_db)
    out = _decode(sig, 20).replace(" ", "")
    assert "CQ" in out
    assert "DE" in out
    assert "SP5MIG" in out


@pytest.mark.parametrize("wpm", [10, 15, 20, 25, 30])
def test_decode_various_wpm(wpm):
    sig, _ = generate_cw("PARIS", wpm=wpm, tone=TONE, fs=FS)
    assert "PARIS" in _decode(sig, wpm)


@pytest.mark.parametrize("text", ["PARIS", "CQ DE SP5MIG", "HELLO WORLD"])
@pytest.mark.parametrize("block_size", [128, 256, 512, 1024, 4096])
def test_block_size_invariance(text, block_size):
    sig, _ = generate_cw(text, wpm=20, tone=TONE, fs=FS, trail=2.0)
    ref = _decode(sig, 20)
    got = _decode_blocks(sig, block_size, 20)
    assert got == ref, (
        f"block={block_size} text={text!r}\n  ref={ref!r}\n  got={got!r}")


def test_block_size_invariance_cross():
    sig, _ = generate_cw("CQ DE SP5MIG TEST", wpm=18, tone=TONE,
                         fs=FS, trail=2.0)
    results = {bs: _decode_blocks(sig, bs, 18)
               for bs in (128, 256, 512, 1024, 4096)}
    ref = results[256]
    for bs, v in results.items():
        assert v == ref, f"block={bs}: {v!r} != 256: {ref!r}"


def test_reset_clears_state():
    sig, _ = generate_cw("TEST", wpm=20, tone=TONE, fs=FS)
    dec = MorseDecoder(FS, tone_freq=TONE, wpm_hint=20)
    dec.process(sig)
    dec.flush()
    assert dec.text
    dec.reset()
    assert dec.text == "" and dec.current_symbol == ""
