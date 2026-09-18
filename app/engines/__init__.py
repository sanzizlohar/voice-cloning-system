"""Engine registry: VC_ENGINE=f5 (default) or VC_ENGINE=fake for offline dev/CI."""

from __future__ import annotations

from app.config import Settings
from app.engines.base import SynthesisResult, TTSEngine

__all__ = ["TTSEngine", "SynthesisResult", "build_engine"]


def build_engine(settings: Settings) -> TTSEngine:
    if settings.engine == "f5":
        from app.engines.f5 import F5Engine

        return F5Engine(settings)
    if settings.engine == "fake":
        from app.engines.fake import FakeEngine

        return FakeEngine(settings)
    raise ValueError(f"unknown engine '{settings.engine}' (expected 'f5' or 'fake')")
