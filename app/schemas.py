"""Pydantic response models."""

from __future__ import annotations

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str
    version: str
    engine: str
    model: str | None
    device: str
    loaded: bool
    quantized: bool
    default_language: str
    languages: list[str]
    ffmpeg: bool
    transcription: bool
    similarity: bool
    speakers: int


class LanguageOut(BaseModel):
    code: str
    name: str
    model_key: str
    repo: str
    license: str
    note: str
    loaded: bool
    is_default: bool


class SpeakerOut(BaseModel):
    id: str
    name: str
    created_at: float
    duration_seconds: float
    sample_rate: int
    ref_text: str
    language: str
    has_embedding: bool
