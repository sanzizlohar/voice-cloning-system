"""Engine abstraction so different cloning backends can be plugged in."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SynthesisResult:
    """Waveform + timing metadata from one engine call."""

    audio: np.ndarray  # float32 mono waveform
    sample_rate: int
    inference_seconds: float
    rtf: float  # real-time factor = inference_seconds / audio duration (<1 = faster than real time)
    model: str
    quantized: bool = False
    nfe_steps: int = 0
    seed: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return round(len(self.audio) / self.sample_rate, 3)


class TTSEngine(abc.ABC):
    """A zero-shot voice cloning backend."""

    name: str = "base"
    sample_rate: int = 24000
    device: str = "cpu"
    supports_dynamic_quantization: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights (may download on first run). Safe to call repeatedly."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    def is_quantized(self) -> bool:
        return getattr(self, "_quantized", False)

    @property
    def model_label(self) -> str:
        return self.name

    @abc.abstractmethod
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
    ) -> SynthesisResult: ...

    def is_model_loaded(self, key: str) -> bool:
        """Whether a given registry model is resident (engines with pools override this)."""
        return self.is_loaded

    @abc.abstractmethod
    def quantize(self) -> bool:
        """Apply dynamic int8 quantization in place. Returns True on success."""
