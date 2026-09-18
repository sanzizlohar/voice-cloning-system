"""Environment-driven settings (every variable is prefixed with VC_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str) -> str:
    return os.getenv("VC_" + name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv("VC_" + name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv("VC_" + name, str(default)))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv("VC_" + name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # server
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    # engine
    engine: str = "f5"
    model_name: str = "F5TTS_v1_Base"
    ckpt_file: str = ""
    vocab_file: str = ""
    device: str = ""  # empty -> autodetect (cuda if available, else cpu)
    quantized: bool = False  # int8 dynamic quantization at load time (CPU only)
    nfe_steps: int = 32  # ODE solver steps: quality vs speed knob
    cfg_strength: float = 2.0
    speed: float = 1.0
    hf_cache_dir: str = ""
    default_language: str = "en"
    max_loaded_models: int = 2  # LRU pool size for multilingual checkpoints
    whisper_model: str = "base"  # faster-whisper size for reference transcription

    # audio
    sample_rate: int = 24000
    min_ref_seconds: float = 3.0
    max_ref_seconds: float = 30.0
    max_upload_mb: int = 25

    # storage
    data_dir: Path = PROJECT_ROOT / "data"
    cache_max_files: int = 200
    enable_cache: bool = True

    # features
    preload_model: bool = True
    enable_similarity: bool = True
    enable_transcription: bool = True
    consent_required: bool = True
    max_text_chars: int = 5000
    stream_chunk_chars: int = 180

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            host=_str("HOST", "127.0.0.1"),
            port=_int("PORT", 8000),
            engine=_str("ENGINE", "f5"),
            model_name=_str("MODEL", "F5TTS_v1_Base"),
            ckpt_file=_str("CKPT_FILE", ""),
            vocab_file=_str("VOCAB_FILE", ""),
            device=_str("DEVICE", ""),
            quantized=_bool("QUANTIZED", False),
            nfe_steps=_int("NFE_STEPS", 32),
            cfg_strength=_float("CFG_STRENGTH", 2.0),
            speed=_float("SPEED", 1.0),
            hf_cache_dir=_str("HF_CACHE_DIR", ""),
            default_language=_str("DEFAULT_LANGUAGE", "en"),
            max_loaded_models=_int("MAX_LOADED_MODELS", 2),
            whisper_model=_str("WHISPER_MODEL", "base"),
            sample_rate=_int("SAMPLE_RATE", 24000),
            min_ref_seconds=_float("MIN_REF_SECONDS", 3.0),
            max_ref_seconds=_float("MAX_REF_SECONDS", 30.0),
            max_upload_mb=_int("MAX_UPLOAD_MB", 25),
            data_dir=Path(_str("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve(),
            cache_max_files=_int("CACHE_MAX_FILES", 200),
            enable_cache=_bool("ENABLE_CACHE", True),
            preload_model=_bool("PRELOAD", True),
            enable_similarity=_bool("ENABLE_SIMILARITY", True),
            enable_transcription=_bool("ENABLE_TRANSCRIPTION", True),
            consent_required=_bool("CONSENT_REQUIRED", True),
            max_text_chars=_int("MAX_TEXT_CHARS", 5000),
            stream_chunk_chars=_int("STREAM_CHUNK_CHARS", 180),
        )
        origins = os.getenv("VC_CORS_ORIGINS")
        if origins:
            s.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]
        return s

    @property
    def speakers_dir(self) -> Path:
        return self.data_dir / "speakers"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


settings = Settings.from_env()
