"""Deterministic offline engine — runs the full API without any model download.

Used by CI, tests (VC_ENGINE=fake), and quick UI development.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from app.engines.base import SynthesisResult, TTSEngine


class FakeEngine(TTSEngine):
    """Synthesizes deterministic pseudo-speech (pitch contour + pauses) instead of real speech."""

    name = "fake"
    sample_rate = 24000
    device = "cpu"
    supports_dynamic_quantization = True

    def __init__(self, settings=None):
        self.settings = settings
        self._loaded = False
        self._quantized = False

    def load(self) -> None:
        self._loaded = True

    def is_model_loaded(self, key: str) -> bool:
        return self._loaded  # fake engine serves every language instantly

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def quantize(self) -> bool:
        if not self._loaded:
            return False
        self._quantized = True
        return True

    def synthesize(
        self,
        *,
        ref_audio_path: str,
        ref_text: str,
        text: str,
        language: str | None = None,
        nfe_steps: int | None = None,
        cfg_strength: float | None = None,
        speed: float | None = None,
        seed: int | None = None,
    ) -> SynthesisResult:
        if not self._loaded:
            raise RuntimeError("engine not loaded")
        if not (text or "").strip():
            raise ValueError("text is empty")
        t0 = time.perf_counter()
        digest = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(digest % (2**32))

        duration = min(6.0, 1.2 + len(text) * 0.055)
        sr = self.sample_rate
        t = np.arange(int(duration * sr)) / sr
        pitch = 110 + 30 * np.sin(2 * np.pi * 0.7 * t)
        phase = 2 * np.pi * np.cumsum(pitch) / sr
        wav = 0.4 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)

        n_frames = int(np.ceil(len(t) / 480))  # 20 ms envelopes, ceil so we cover the tail
        envelope = rng.uniform(0.3, 1.0, size=n_frames).repeat(480)[: len(t)]
        pauses = (rng.random(n_frames) < 0.12).repeat(480)[: len(t)]
        wav = wav * envelope
        wav[pauses] = 0.0

        wav = wav.astype(np.float32)
        wav /= max(float(np.max(np.abs(wav))), 1e-9)
        if seed is not None:  # keep determinism but acknowledge the knob
            wav *= 1.0

        elapsed = time.perf_counter() - t0
        return SynthesisResult(
            audio=wav,
            sample_rate=sr,
            inference_seconds=round(elapsed, 4),
            rtf=round(elapsed / max(duration, 1e-6), 4),
            model="fake",
            quantized=self._quantized,
            nfe_steps=nfe_steps or 0,
            seed=seed,
        )
