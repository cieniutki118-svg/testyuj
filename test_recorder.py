"""Testy rejestratora WAV - wielokrotne start/stop."""
import os
import tempfile
import numpy as np

from core.recorder import WAVRecorder


def test_recorder_start_stop_write(tmp_path):
    rec = WAVRecorder(samplerate=48000, out_dir=str(tmp_path))
    assert not rec.recording
    p = rec.start()
    assert rec.recording
    assert os.path.exists(p)
    x = np.zeros(4800, dtype=np.float32)
    rec.write(x)
    out = rec.stop()
    assert not rec.recording
    assert out == p
    assert os.path.getsize(p) > 44  # nagłówek WAV


def test_recorder_multiple_cycles(tmp_path):
    rec = WAVRecorder(samplerate=48000, out_dir=str(tmp_path))
    for i in range(3):
        p = rec.start()
        rec.write(np.zeros(1024, dtype=np.float32))
        out = rec.stop()
        assert out == p
        assert os.path.exists(out)
    files = sorted(os.listdir(tmp_path))
    assert len(files) == 3
