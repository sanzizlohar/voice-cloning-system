"""Shared fixtures — the whole test suite runs against the offline fake engine."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# must be set before app.config is imported
os.environ.setdefault("VC_ENGINE", "fake")
os.environ.setdefault("VC_PRELOAD", "0")
os.environ.setdefault("VC_DATA_DIR", tempfile.mkdtemp(prefix="vc_test_data_"))
os.environ.setdefault("VC_ENABLE_SIMILARITY", "0")   # never download ECAPA in tests
os.environ.setdefault("VC_ENABLE_TRANSCRIPTION", "0")  # never download whisper in tests
os.environ.setdefault("VC_CONSENT_REQUIRED", "1")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def wav_file(tmp_path):
    """A 4-second 220 Hz tone — valid audio for prepare_reference()."""
    import numpy as np
    import soundfile as sf

    sr = 24000
    t = np.arange(4 * sr) / sr
    wav = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), wav, sr, subtype="PCM_16")
    return path


@pytest.fixture()
def enrolled_speaker(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/api/v1/speakers",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"name": "TestVoice", "consent": "true", "ref_text": "test transcript"},
        )
    assert r.status_code == 201, r.text
    return r.json()
