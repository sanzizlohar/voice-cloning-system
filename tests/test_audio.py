"""Unit tests for audio helpers."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.services import audio as audio_svc
from app.services.audio import AudioError


def test_prepare_reference_rejects_short_audio(tmp_path):
    sr = 24000
    path = tmp_path / "short.wav"
    sf.write(str(path), np.zeros(sr, dtype=np.float32), sr)  # 1 s
    with pytest.raises(AudioError, match="too short"):
        audio_svc.prepare_reference(path, sr, min_seconds=3.0, max_seconds=30.0)


def test_prepare_reference_normalizes_and_caps(tmp_path):
    sr = 24000
    t = np.arange(10 * sr) / sr
    loud = (0.9 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "loud.wav"
    sf.write(str(path), loud, sr)
    wav, duration = audio_svc.prepare_reference(path, sr, min_seconds=3.0, max_seconds=5.0)
    assert duration == pytest.approx(5.0, abs=0.05)  # capped at max_seconds
    assert float(np.max(np.abs(wav))) <= 0.951
    assert wav.dtype == np.float32


def test_prepare_reference_rejects_silence(tmp_path):
    sr = 24000
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros(5 * sr, dtype=np.float32), sr)
    with pytest.raises(AudioError, match="too short|silent"):
        audio_svc.prepare_reference(path, sr, min_seconds=3.0, max_seconds=30.0)


def test_wav_bytes_roundtrip(tmp_path):
    sr = 24000
    wav = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    data = audio_svc.wav_bytes(wav, sr)
    path = tmp_path / "roundtrip.wav"
    path.write_bytes(data)
    decoded, out_sr = sf.read(str(path), dtype="float32")
    assert out_sr == sr
    assert len(decoded) == len(wav)
    assert np.max(np.abs(decoded)) == pytest.approx(np.max(np.abs(wav)), abs=1e-3)


def test_add_fades_removes_clicks():
    sr = 24000
    wav = np.ones(sr, dtype=np.float32)
    out = audio_svc.add_fades(wav, sr, ms=10)
    assert len(out) == len(wav)
    assert out[0] == 0.0
    assert abs(out[sr // 2] - 1.0) < 1e-6


def test_load_mono_rejects_garbage(tmp_path):
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"this is not audio")
    with pytest.raises(AudioError):
        audio_svc.load_mono(path, 24000)
