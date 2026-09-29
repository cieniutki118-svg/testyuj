"""Testy łańcucha DSP."""
import numpy as np
import pytest

from core.dsp import DSPChain

FS = 48000


def test_dsp_reset_exists_and_works():
    d = DSPChain(FS, low=300, high=3000, use_spectral=True)
    x = np.random.default_rng(0).normal(0, 0.1, 4096).astype(np.float32)
    y1 = d.process(x)
    d.reset()
    y2 = d.process(x)
    assert y1.shape == y2.shape == x.shape


def test_dsp_output_length_matches_input():
    d = DSPChain(FS, low=300, high=3000, use_spectral=True)
    for n in (128, 512, 1024, 4096):
        x = np.zeros(n, dtype=np.float32)
        y = d.process(x)
        assert y.size == n


def test_dsp_bandpass_attenuates_dc():
    d = DSPChain(FS, low=300, high=3000, use_spectral=False)
    x = np.ones(48000, dtype=np.float32) * 0.5
    y = d.process(x)
    # DC powinno być mocno stłumione
    assert float(np.abs(y[10000:]).mean()) < 0.05
