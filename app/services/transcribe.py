"""Optional reference-audio transcription via faster-whisper (lazy-loaded, int8 on CPU)."""

from __future__ import annotations

import logging
import threading

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def transcribe(path: str, language: str | None = None) -> str:
    """Transcribe an audio file; downloads the 'base' model on first use (~145 MB)."""
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed") from exc

    with _lock:
        if _model is None:
            logger.info(
                "Loading faster-whisper '%s' model (first use downloads it) ...",
                settings.whisper_model,
            )
            _model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
        segments, _info = _model.transcribe(
            str(path), language=language or None, beam_size=1, vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments).strip()
