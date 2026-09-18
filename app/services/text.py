"""Lightweight sentence segmentation used for chunked/streaming synthesis."""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")


def split_sentences(text: str, max_chars: int = 180, min_chars: int = 40) -> list[str]:
    """Split text into synthesis chunks.

    Splits on sentence punctuation first, merges fragments that are too short,
    and hard-wraps anything longer than `max_chars` on word boundaries.
    """
    text = " ".join((text or "").split())
    if not text:
        return []

    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p.strip()]
    chunks: list[str] = []
    for part in parts:
        if len(part) > max_chars:
            chunks.extend(_wrap_words(part, max_chars))
        elif chunks and len(chunks[-1]) < min_chars and len(chunks[-1]) + len(part) + 1 <= max_chars:
            chunks[-1] = f"{chunks[-1]} {part}"
        else:
            chunks.append(part)
    return chunks


def _wrap_words(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
