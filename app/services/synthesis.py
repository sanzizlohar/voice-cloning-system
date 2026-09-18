"""End-to-end cloning orchestration: reference resolution, transcripts, caching, metrics."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.engines.base import TTSEngine
from app.services import audio as audio_svc
from app.services import similarity as similarity_svc
from app.services import transcribe as transcribe_svc
from app.services.speakers import SpeakerProfile, SpeakerStore
from app.services.text import split_sentences

logger = logging.getLogger(__name__)


class SynthesisError(RuntimeError):
    """User-facing synthesis failure (bad input, unknown speaker, ...)."""


@dataclass
class CloneOutput:
    wav_bytes: bytes
    sample_rate: int
    metrics: dict
    audio: np.ndarray | None = None


class CloningService:
    def __init__(self, *, engine: TTSEngine, speakers: SpeakerStore, settings: Settings):
        self.engine = engine
        self.speakers = speakers
        self.settings = settings
        self._engine_lock = threading.Lock()  # one synthesis at a time
        self._transcript_cache: dict[tuple[str, str], str] = {}  # (ref_hash, language) -> text
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ clone
    def clone(
        self,
        *,
        text: str,
        speaker_id: str = "",
        upload_path: str | Path | None = None,
        ref_text: str = "",
        language: str = "",
        nfe_steps: int | None = None,
        cfg_strength: float | None = None,
        speed: float | None = None,
        seed: int | None = None,
        with_similarity: bool = False,
        use_cache: bool = True,
    ) -> CloneOutput:
        text = " ".join((text or "").split())
        if not text:
            raise SynthesisError("text is empty")
        if len(text) > self.settings.max_text_chars:
            raise SynthesisError(
                f"text too long ({len(text)} chars; max {self.settings.max_text_chars})"
            )

        profile: SpeakerProfile | None = None
        ref_wav = ref_sr = None
        tmp_ref: Path | None = None
        if speaker_id:
            profile = self.speakers.get(speaker_id)
            if profile is None:
                raise SynthesisError(f"unknown speaker '{speaker_id}'")
            ref_wav, ref_sr = self.speakers.load_reference(speaker_id)
            ref_path = self.speakers.reference_path(speaker_id)
        elif upload_path:
            ref_wav, _duration = audio_svc.prepare_reference(
                upload_path,
                self.settings.sample_rate,
                self.settings.min_ref_seconds,
                self.settings.max_ref_seconds,
            )
            ref_sr = self.settings.sample_rate  # prepare_reference decodes at this rate
            ref_path = None  # written below once prepared
        else:
            raise SynthesisError("no voice reference: pass 'speaker_id' or 'reference_audio'")

        ref_hash = hashlib.sha1(audio_svc.wav_bytes(ref_wav, ref_sr)).hexdigest()[:16]

        if ref_path is None:
            # persist the prepared upload so the engine gets identical input every time
            fd = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix="vc_ref_", dir=self.settings.cache_dir, delete=False
            )
            fd.close()
            tmp_ref = Path(fd.name)
            audio_svc.write_wav(tmp_ref, ref_wav, ref_sr)
            ref_path = tmp_ref

        effective_ref_text = (ref_text or "").strip() or (profile.ref_text if profile else "")
        if not effective_ref_text:
            effective_ref_text = self._reference_transcript(ref_hash, ref_path, language)

        cache_key = self._cache_key(
            ref_hash, effective_ref_text, text, language, nfe_steps, cfg_strength, speed, seed
        )
        if use_cache and self.settings.enable_cache:
            hit = self._cache_read(cache_key)
            if hit is not None:
                wav_bytes, metrics = hit
                metrics["cached"] = True
                return CloneOutput(wav_bytes=wav_bytes, sample_rate=ref_sr, metrics=metrics)

        try:
            with self._engine_lock:
                if not self.engine.is_loaded:
                    self.engine.load()
                result = self.engine.synthesize(
                    ref_audio_path=str(ref_path),
                    ref_text=effective_ref_text,
                    text=text,
                    language=(language or "").strip().lower() or None,
                    nfe_steps=nfe_steps,
                    cfg_strength=cfg_strength,
                    speed=speed,
                    seed=seed,
                )
        finally:
            if tmp_ref is not None:
                tmp_ref.unlink(missing_ok=True)

        out_wav = audio_svc.add_fades(result.audio, result.sample_rate)
        wav_bytes = audio_svc.wav_bytes(out_wav, result.sample_rate)

        metrics = {
            "model": result.model,
            "device": self.engine.device,
            "quantized": result.quantized,
            "nfe_steps": result.nfe_steps,
            "latency_s": result.inference_seconds,
            "audio_seconds": result.duration_seconds,
            "rtf": result.rtf,
            "cached": False,
            "speaker_id": speaker_id or None,
            "speaker_similarity": None,
        }
        if with_similarity and self.settings.enable_similarity:
            try:
                sim = self._speaker_similarity(profile, ref_wav, ref_sr, out_wav, result.sample_rate)
                metrics["speaker_similarity"] = round(sim, 4) if sim is not None else None
            except Exception as exc:  # noqa: BLE001 — scoring is best-effort
                logger.warning("similarity scoring failed: %s", exc)

        if use_cache and self.settings.enable_cache:
            self._cache_write(cache_key, wav_bytes, metrics)
        return CloneOutput(
            wav_bytes=wav_bytes, sample_rate=result.sample_rate, metrics=metrics, audio=out_wav
        )

    # ---------------------------------------------------------------- streaming
    def stream(
        self,
        *,
        text: str,
        **clone_kwargs,
    ) -> Iterator[dict]:
        """Yield one event dict per sentence chunk, then a final summary event."""
        started = time.perf_counter()
        chunks = split_sentences(text, max_chars=self.settings.stream_chunk_chars)
        if not chunks:
            raise SynthesisError("text is empty")
        clone_kwargs.setdefault("use_cache", True)
        for i, chunk in enumerate(chunks):
            out = self.clone(text=chunk, **clone_kwargs)
            yield {
                "type": "chunk",
                "index": i,
                "n_chunks": len(chunks),
                "sentence": chunk,
                "audio_base64": base64.b64encode(out.wav_bytes).decode("ascii"),
                "latency_s": out.metrics.get("latency_s"),
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
        yield {
            "type": "done",
            "n_chunks": len(chunks),
            "total_latency_s": round(time.perf_counter() - started, 3),
        }

    # ------------------------------------------------------------------ helpers
    def _reference_transcript(self, ref_hash: str, ref_path: Path, language: str) -> str:
        """Explicit ref_text missing — use the cache or transcribe with faster-whisper."""
        key = (ref_hash, language)
        cached = self._transcript_cache.get(key)
        if cached:
            return cached
        if not (self.settings.enable_transcription and transcribe_svc.available()):
            raise SynthesisError(
                "no reference transcript: pass 'ref_text', enroll with one, or enable transcription"
            )
        logger.info("transcribing reference audio %s ...", ref_path)
        text = transcribe_svc.transcribe(str(ref_path), language=language or None)
        if not text:
            raise SynthesisError(
                "reference transcript came back empty — pass 'ref_text' explicitly"
            )
        self._transcript_cache[key] = text
        return text

    def _speaker_similarity(
        self,
        profile: SpeakerProfile | None,
        ref_wav: np.ndarray,
        ref_sr: int,
        out_wav: np.ndarray,
        out_sr: int,
    ) -> float | None:
        if not similarity_svc.available():
            return None
        out_emb = similarity_svc.embedding_from_wav(out_wav, out_sr)
        ref_emb = None
        if profile is not None and profile.has_embedding:
            ref_emb = self.speakers.embedding(profile.id)
        if ref_emb is None:
            ref_emb = similarity_svc.embedding_from_wav(ref_wav, ref_sr)
        return similarity_svc.cosine_similarity(ref_emb, out_emb)

    # --------------------------------------------------------------------- cache
    def _cache_key(
        self,
        ref_hash: str,
        ref_text: str,
        text: str,
        language: str | None,
        nfe: int | None,
        cfg: float | None,
        speed: float | None,
        seed: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "engine": self.engine.name,
                "model": self.engine.model_label,
                "quantized": self.engine.is_quantized,
                "ref": ref_hash,
                "ref_text": ref_text,
                "text": text,
                "language": (language or "").strip().lower(),
                "nfe": nfe,
                "cfg": cfg,
                "speed": speed,
                "seed": seed,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> tuple[bytes, dict] | None:
        wav_path = self._cache_path(key)
        meta_path = wav_path.with_suffix(".json")
        if wav_path.is_file() and meta_path.is_file():
            try:
                metrics = json.loads(meta_path.read_text(encoding="utf-8"))
                wav_path.touch()  # LRU touch
                return wav_path.read_bytes(), metrics
            except Exception:  # noqa: BLE001
                return None
        return None

    def _cache_write(self, key: str, wav_bytes: bytes, metrics: dict) -> None:
        try:
            wav_path = self._cache_path(key)
            wav_path.write_bytes(wav_bytes)
            wav_path.with_suffix(".json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            self._cache_prune()
        except Exception as exc:  # noqa: BLE001 — cache is best-effort
            logger.warning("cache write failed: %s", exc)

    def _cache_path(self, key: str) -> Path:
        return self.settings.cache_dir / f"{key}.wav"

    def _cache_prune(self) -> None:
        files = sorted(self.settings.cache_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        excess = len(files) - self.settings.cache_max_files
        for path in files[:max(0, excess)]:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
