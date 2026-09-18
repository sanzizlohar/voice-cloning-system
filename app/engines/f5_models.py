"""Registry of F5-TTS checkpoints served by this system (language -> model).

All repos are public on HuggingFace. The official AI4Bharat/IndicF5 repo is
auto-gated (requires a login), so the registry points at a byte-identical
ungated mirror by default; set VC_LANGUAGE_MODELS to point at the official
repo with an HF_TOKEN if you prefer.

Extend or override the registry without code changes:
    VC_LANGUAGE_MODELS='{"de": {"repo": "user/F5-German", "ckpt_file": "model.safetensors",
                                "vocab_file": "vocab.txt", "languages": ["de"]}}'
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """One F5-TTS checkpoint and the language codes it can synthesize."""

    key: str
    repo: str
    ckpt_file: str
    vocab_file: str
    languages: tuple[str, ...]
    license: str = ""
    note: str = ""
    arch: str = "F5TTS_v1_Base"


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "or": "Odia",
    "pa": "Punjabi",
    "as": "Assamese",
    "fr": "French",
}

_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        key="en",
        repo="SWivid/F5-TTS",
        ckpt_file="F5TTS_v1_Base/model_1250000.safetensors",
        vocab_file="F5TTS_v1_Base/vocab.txt",
        languages=("en", "zh"),
        license="CC-BY-NC-4.0 (checkpoints; code MIT)",
        note="F5-TTS v1 base — English + Chinese",
    ),
    ModelSpec(
        key="indic",
        repo="6Morpheus6/IndicF5",
        ckpt_file="model.safetensors",
        vocab_file="checkpoints/vocab.txt",
        languages=("hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "or", "pa", "as"),
        license="MIT (AI4Bharat/IndicF5)",
        note="IndicF5 — 11 Indian languages incl. Hindi & Bengali (ungated mirror)",
    ),
    ModelSpec(
        key="fr",
        repo="RASPIAUDIO/F5-French-MixedSpeakers-reduced",
        ckpt_file="model_last_reduced.pt",
        vocab_file="vocab.txt",
        languages=("fr",),
        license="CC-BY-NC-4.0",
        note="French fine-tune (mixed speakers)",
    ),
]


def _load_overrides() -> None:
    raw = os.getenv("VC_LANGUAGE_MODELS")
    if not raw:
        return
    try:
        extra = json.loads(raw)
        for key, cfg in extra.items():
            spec = ModelSpec(
                key=key,
                repo=cfg["repo"],
                ckpt_file=cfg["ckpt_file"],
                vocab_file=cfg.get("vocab_file", ""),
                languages=tuple(cfg.get("languages", [key])),
                license=cfg.get("license", ""),
                note=cfg.get("note", "custom (VC_LANGUAGE_MODELS)"),
            )
            _REGISTRY[:] = [s for s in _REGISTRY if s.key != key] + [spec]
        logger.info("loaded %d custom model spec(s) from VC_LANGUAGE_MODELS", len(extra))
    except Exception as exc:  # noqa: BLE001 — bad config should not crash the app
        logger.warning("could not parse VC_LANGUAGE_MODELS (%s); using built-in registry", exc)


_load_overrides()


def all_specs() -> list[ModelSpec]:
    return list(_REGISTRY)


def spec_for_language(language: str | None) -> ModelSpec:
    """Map a language code to its checkpoint; falls back to the first (default) spec."""
    language = (language or "").strip().lower()
    for spec in _REGISTRY:
        if language in spec.languages:
            return spec
    return _REGISTRY[0]


def language_entries() -> list[dict]:
    """Flat list of {code, name, model_key, ...} for the API/UI."""
    out = []
    for spec in _REGISTRY:
        for code in spec.languages:
            out.append(
                {
                    "code": code,
                    "name": LANGUAGE_NAMES.get(code, code),
                    "model_key": spec.key,
                    "repo": spec.repo,
                    "license": spec.license,
                    "note": spec.note,
                }
            )
    return out
