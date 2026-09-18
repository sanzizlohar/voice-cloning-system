"""Speaker profile store — enroll a reference voice once, reuse it for many syntheses."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.services import audio as audio_svc
from app.services import similarity as similarity_svc

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    id: str
    name: str
    created_at: float
    duration_seconds: float
    sample_rate: int
    ref_text: str = ""
    language: str = ""
    consent_recorded_at: float | None = None
    has_embedding: bool = False


class SpeakerStore:
    """Filesystem-backed store: data/speakers/<id>/{profile.json, reference.wav, embedding.npy}."""

    def __init__(
        self,
        root: Path,
        *,
        sample_rate: int,
        min_ref_seconds: float,
        max_ref_seconds: float,
        enable_embeddings: bool = True,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.min_ref_seconds = min_ref_seconds
        self.max_ref_seconds = max_ref_seconds
        self.enable_embeddings = enable_embeddings and similarity_svc.available()
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------
    def _dir(self, speaker_id: str) -> Path:
        return self.root / speaker_id

    def _meta_path(self, speaker_id: str) -> Path:
        return self._dir(speaker_id) / "profile.json"

    def _wav_path(self, speaker_id: str) -> Path:
        return self._dir(speaker_id) / "reference.wav"

    def _emb_path(self, speaker_id: str) -> Path:
        return self._dir(speaker_id) / "embedding.npy"

    # -- operations ------------------------------------------------------------
    def enroll(
        self,
        *,
        name: str,
        source_path: str | Path,
        ref_text: str = "",
        language: str = "",
        consent: bool = False,
    ) -> SpeakerProfile:
        wav, duration = audio_svc.prepare_reference(
            source_path, self.sample_rate, self.min_ref_seconds, self.max_ref_seconds
        )
        speaker_id = uuid.uuid4().hex[:12]
        directory = self._dir(speaker_id)
        directory.mkdir(parents=True, exist_ok=False)
        audio_svc.write_wav(self._wav_path(speaker_id), wav, self.sample_rate)

        profile = SpeakerProfile(
            id=speaker_id,
            name=(name or "").strip() or f"speaker-{speaker_id}",
            created_at=time.time(),
            duration_seconds=duration,
            sample_rate=self.sample_rate,
            ref_text=(ref_text or "").strip(),
            language=(language or "").strip(),
            consent_recorded_at=time.time() if consent else None,
        )
        if self.enable_embeddings:
            try:
                emb = similarity_svc.embedding_from_wav(wav, self.sample_rate)
                np.save(self._emb_path(speaker_id), emb)
                profile.has_embedding = True
            except Exception as exc:  # noqa: BLE001 — embeddings are optional metadata
                logger.warning("speaker embedding skipped: %s", exc)
        self._meta_path(speaker_id).write_text(
            json.dumps(asdict(profile), indent=2), encoding="utf-8"
        )
        logger.info("enrolled speaker '%s' as %s (%.1fs)", profile.name, speaker_id, duration)
        return profile

    def get(self, speaker_id: str) -> SpeakerProfile | None:
        path = self._meta_path(speaker_id)
        if not path.is_file():
            return None
        try:
            return SpeakerProfile(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — corrupt metadata should not 500
            return None

    def list(self) -> list[SpeakerProfile]:
        profiles = []
        for directory in self.root.iterdir():
            if directory.is_dir():
                profile = self.get(directory.name)
                if profile is not None:
                    profiles.append(profile)
        return sorted(profiles, key=lambda p: p.created_at, reverse=True)

    def delete(self, speaker_id: str) -> bool:
        directory = self._dir(speaker_id)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True

    def reference_path(self, speaker_id: str) -> Path:
        return self._wav_path(speaker_id)

    def load_reference(self, speaker_id: str) -> tuple[np.ndarray, int]:
        profile = self.get(speaker_id)
        if profile is None:
            raise KeyError(speaker_id)
        wav = audio_svc.load_mono(self._wav_path(speaker_id), profile.sample_rate)
        return wav, profile.sample_rate

    def embedding(self, speaker_id: str) -> np.ndarray | None:
        path = self._emb_path(speaker_id)
        return np.load(path) if path.is_file() else None

    def update_ref_text(self, speaker_id: str, ref_text: str) -> None:
        profile = self.get(speaker_id)
        if profile is None:
            return
        profile.ref_text = " ".join(ref_text.split())
        with self._lock:
            self._meta_path(speaker_id).write_text(
                json.dumps(asdict(profile), indent=2), encoding="utf-8"
            )
