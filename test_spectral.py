"""Testy odszumiania spektralnego - ciągłość, latencja, blok-niezależność."""
import numpy as np
import pytest

from core.spectral import SpectralSubtractor

FS = 48000
TONE = 700.0


def test_continuity_length_and_prefill():
    ss = SpectralSubtractor(FS, fft_size=1024)
    assert ss.latency == 1024
    n = 8000
    x = np.random.default_rng(0).normal(0, 0.1, n).astype(np.float32)
    out = ss.process(x)
    assert out.size == n
    assert np.allclose(out[:ss.latency], 0.0)


@pytest.mark.parametrize("block_size", [128, 256, 512, 1024, 4096])
def test_block_size_invariance(block_size):
    rng = np.random.default_rng(1)
    sig = rng.normal(0, 0.1, 24000).astype(np.float32)

    ss = SpectralSubtractor(FS, fft_size=1024)
    out = np.concatenate(
        [ss.process(sig[i:i + block_size])
         for i in range(0, len(sig), block_size)])
    latency = ss.latency
    body = out[latency:]

    ss_ref = SpectralSubtractor(FS, fft_size=1024)
    ref = ss_ref.process(sig)[latency:]

    assert body.shape == ref.shape
    assert np.allclose(body, ref, atol=1e-5), (
        f"block={block_size}: max diff "
        f"{np.max(np.abs(body - ref)):.3e}")


def _band_energies(sig, fs, tone, bw=50.0):
    S = np.abs(np.fft.rfft(sig))
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    in_b = (f > tone - bw) & (f < tone + bw)
    return float(np.sum(S[in_b] ** 2)), float(np.sum(S[~in_b] ** 2))


def test_spectral_reduces_out_of_band_noise():
    rng = np.random.default_rng(2)
    n = int(FS * 2.0)
    t = np.arange(n, dtype=np.float64) / FS
    clean = (0.3 * np.sin(2 * np.pi * TONE * t)).astype(np.float32)
    noisy = clean + rng.normal(0, 0.05, n).astype(np.float32)

    ss = SpectralSubtractor(FS, fft_size=1024)
    block = 512
    half = n // 2
    # rozgrzewka: pierwsza połowa
    for i in range(0, half - block, block):
        ss.process(noisy[i:i + block])
    # pomiar: druga połowa BEZ reset
    out_chunks = []
    for i in range(half, n - block, block):
        out_chunks.append(ss.process(noisy[i:i + block]))
    out = np.concatenate(out_chunks)

    trim = 2 * 1024
    out_t = out[trim:-trim]
    noisy_t = noisy[half + trim: half + trim + out_t.size]

    in_n, out_n = _band_energies(noisy_t, FS, TONE)
    in_d, out_d = _band_energies(out_t, FS, TONE)

    # ton zachowany
    assert in_d > 0.5 * in_n, f"ton uszkodzony: {in_d:.3e} vs {in_n:.3e}"
    # szum poza pasmem zredukowany
    assert out_d < 0.8 * out_n, f"szum nie zredukowany: {out_d:.3e} vs {out_n:.3e}"
