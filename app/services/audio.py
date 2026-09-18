"""Audio decoding, reference preparation, and WAV encoding."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioError(ValueError):
    """Raised for undecodable or invalid audio."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def load_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode any common audio format to mono float32 at `sample_rate`."""
    try:
        wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001 — fall back to ffmpeg for exotic containers
        if not ffmpeg_available():
            raise AudioError(f"could not decode audio: {exc}") from exc
        logger.info("librosa could not decode file (%s); retrying via ffmpeg", exc)
        try:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "converted.wav"
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(path), "-ac", "1", "-ar", str(sample_rate), str(out),
                    ],
                    check=True,
                    capture_output=True,
                )
                wav, _ = sf.read(str(out), dtype="float32")
        except Exception as exc2:  # noqa: BLE001
            raise AudioError(f"could not decode audio with ffmpeg: {exc2}") from exc2

    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim != 1 or wav.size == 0:
        raise AudioError("decoded audio is empty")
    return wav


def normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize to a safe level."""
    maximum = float(np.max(np.abs(wav)))
    if maximum < 1e-9:
        raise AudioError("audio is silent")
    return (wav / maximum * peak).astype(np.float32)


def prepare_reference(
    path: str | Path,
    sample_rate: int,
    min_seconds: float,
    max_seconds: float,
) -> tuple[np.ndarray, float]:
    """Decode, trim silence, cap length, and normalize a voice reference.

    Returns (wav float32 mono, duration seconds).
    """
    wav = load_mono(path, sample_rate)
    duration = len(wav) / sample_rate
    if duration < min_seconds:
        raise AudioError(
            f"reference audio too short ({duration:.1f}s); need at least {min_seconds:.0f}s"
        )
    trimmed, _ = librosa.effects.trim(wav, top_db=40)
    if len(trimmed) > 0:
        wav = trimmed
    max_samples = int(max_seconds * sample_rate)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    duration = round(len(wav) / sample_rate, 3)
    wav = normalize(wav)
    return wav, duration


def add_fades(wav: np.ndarray, sample_rate: int, ms: int = 15) -> np.ndarray:
    """Short fade in/out to remove clicks at chunk boundaries."""
    n = min(len(wav) // 2, int(sample_rate * ms / 1000))
    if n <= 0:
        return wav
    out = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def write_wav(path: str | Path, wav: np.ndarray, sample_rate: int) -> None:
    """Write float32 mono to disk as 16-bit PCM WAV."""
    sf.write(str(path), wav, sample_rate, subtype="PCM_16")


def wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono as 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
