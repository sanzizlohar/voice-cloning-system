"""REST API v1 — health, speaker enrollment, cloning (sync + streaming)."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import __version__
from app.engines import f5_models
from app.schemas import HealthOut, LanguageOut, SpeakerOut
from app.services import audio as audio_svc
from app.services import similarity as similarity_svc
from app.services import transcribe as transcribe_svc
from app.services.audio import AudioError
from app.services.synthesis import SynthesisError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_SPEAKER_ID_RE = re.compile(r"^[0-9a-f]{12}$")
CONSENT_MESSAGE = (
    "By passing consent=true you affirm you have the legal right to use this voice "
    "recording (your own voice, or the speaker's written permission) and will not use "
    "cloned audio to deceive anyone."
)

METRIC_HEADERS = [
    "X-Voice-Model",
    "X-Voice-Device",
    "X-Voice-Quantized",
    "X-Voice-Nfe-Steps",
    "X-Voice-Latency-S",
    "X-Voice-Audio-S",
    "X-Voice-RTF",
    "X-Voice-Cached",
    "X-Voice-Similarity",
]


# --------------------------------------------------------------------- helpers
def _state(request: Request):
    return request.app.state


def _require_service(request: Request):
    state = _state(request)
    if getattr(state, "service", None) is None:
        raise HTTPException(status_code=503, detail="service is starting; retry shortly")
    return state


def _validate_speaker_id(speaker_id: str) -> str:
    speaker_id = (speaker_id or "").strip()
    if not _SPEAKER_ID_RE.fullmatch(speaker_id):
        raise HTTPException(status_code=400, detail="invalid speaker id format")
    return speaker_id


async def _save_upload(upload: UploadFile, max_mb: int) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded audio file is empty")
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"upload larger than {max_mb} MB")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="vc_upload_") as tmp:
        tmp.write(data)
    return Path(tmp.name)


def _clone_params(
    text: str,
    speaker_id: str,
    ref_text: str,
    language: str,
    nfe_steps: int | None,
    cfg_strength: float | None,
    speed: float | None,
    seed: int | None,
) -> None:
    """Shared validation for /clone and /clone/stream."""
    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="text is required")
    if nfe_steps is not None and not 1 <= nfe_steps <= 128:
        raise HTTPException(status_code=400, detail="nfe_steps must be between 1 and 128")
    if cfg_strength is not None and not 0.1 <= cfg_strength <= 5.0:
        raise HTTPException(status_code=400, detail="cfg_strength must be between 0.1 and 5.0")
    if speed is not None and not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=400, detail="speed must be between 0.5 and 2.0")
    if speaker_id:
        _validate_speaker_id(speaker_id)


def _synthesis_error(exc: Exception) -> HTTPException:
    message = str(exc)
    code = 404 if "unknown speaker" in message else 400
    return HTTPException(status_code=code, detail=message)


def _metric_headers(metrics: dict) -> dict[str, str]:
    headers = {
        "X-Voice-Model": str(metrics.get("model", "")),
        "X-Voice-Device": str(metrics.get("device", "")),
        "X-Voice-Quantized": str(bool(metrics.get("quantized"))).lower(),
        "X-Voice-Nfe-Steps": str(metrics.get("nfe_steps", "")),
        "X-Voice-Latency-S": str(metrics.get("latency_s", "")),
        "X-Voice-Audio-S": str(metrics.get("audio_seconds", "")),
        "X-Voice-RTF": str(metrics.get("rtf", "")),
        "X-Voice-Cached": str(bool(metrics.get("cached"))).lower(),
    }
    if metrics.get("speaker_similarity") is not None:
        headers["X-Voice-Similarity"] = str(metrics["speaker_similarity"])
    return headers


# ----------------------------------------------------------------------- health
@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    state = _state(request)
    engine = getattr(state, "engine", None)
    speakers = getattr(state, "speakers", None)
    return HealthOut(
        status="ok" if engine is not None else "starting",
        version=__version__,
        engine=state.settings.engine,
        model=engine.model_label if engine else None,
        device=engine.device if engine else "unknown",
        loaded=bool(engine.is_loaded) if engine else False,
        quantized=engine.is_quantized if engine else state.settings.quantized,
        default_language=state.settings.default_language,
        languages=[e["code"] for e in f5_models.language_entries()],
        ffmpeg=audio_svc.ffmpeg_available(),
        transcription=transcribe_svc.available(),
        similarity=similarity_svc.available() and state.settings.enable_similarity,
        speakers=len(speakers.list()) if speakers else 0,
    )


# -------------------------------------------------------------------- languages
@router.get("/languages", response_model=list[LanguageOut])
def languages(request: Request) -> list[LanguageOut]:
    """Language registry: which languages are served, by which checkpoint."""
    state = _state(request)
    engine = getattr(state, "engine", None)
    default = state.settings.default_language.lower()
    out = []
    for entry in f5_models.language_entries():
        loaded = engine.is_model_loaded(entry["model_key"]) if engine else False
        out.append(
            LanguageOut(
                **entry,
                loaded=loaded,
                is_default=entry["code"] == default,
            )
        )
    return out


# --------------------------------------------------------------------- speakers
@router.post("/speakers", response_model=SpeakerOut, status_code=201)
async def enroll_speaker(
    request: Request,
    reference_audio: UploadFile = File(...),
    name: str = Form(...),
    consent: bool = Form(False),
    ref_text: str = Form(""),
    language: str = Form(""),
) -> SpeakerOut:
    """Enroll a voice: upload 6-30 s of clean speech (consent required)."""
    state = _require_service(request)
    s = state.settings
    if s.consent_required and not consent:
        raise HTTPException(status_code=403, detail=f"consent required. {CONSENT_MESSAGE}")

    name = (name or "").strip()
    if not name or len(name) > 80:
        raise HTTPException(status_code=400, detail="name must be 1-80 characters")

    upload_path = await _save_upload(reference_audio, s.max_upload_mb)

    def _enroll() -> SpeakerOut:
        profile = state.speakers.enroll(
            name=name,
            source_path=upload_path,
            ref_text=ref_text,
            language=language,
            consent=True,
        )
        if not profile.ref_text and s.enable_transcription and transcribe_svc.available():
            try:
                transcript = transcribe_svc.transcribe(
                    str(state.speakers.reference_path(profile.id)), language=language or None
                )
                if transcript:
                    state.speakers.update_ref_text(profile.id, transcript)
                    profile.ref_text = transcript
            except Exception as exc:  # noqa: BLE001 — transcription is best-effort
                logger.warning("auto-transcription failed: %s", exc)
        return SpeakerOut(**asdict(profile))

    try:
        return await run_in_threadpool(_enroll)
    except AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("enrollment failed")
        raise HTTPException(status_code=500, detail=f"enrollment failed: {exc}") from exc
    finally:
        upload_path.unlink(missing_ok=True)


@router.get("/speakers", response_model=list[SpeakerOut])
def list_speakers(request: Request) -> list[SpeakerOut]:
    state = _require_service(request)
    return [SpeakerOut(**asdict(p)) for p in state.speakers.list()]


@router.get("/speakers/{speaker_id}", response_model=SpeakerOut)
def get_speaker(request: Request, speaker_id: str) -> SpeakerOut:
    state = _require_service(request)
    _validate_speaker_id(speaker_id)
    profile = state.speakers.get(speaker_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="speaker not found")
    return SpeakerOut(**asdict(profile))


@router.delete("/speakers/{speaker_id}", status_code=204)
def delete_speaker(request: Request, speaker_id: str) -> Response:
    state = _require_service(request)
    _validate_speaker_id(speaker_id)
    if not state.speakers.delete(speaker_id):
        raise HTTPException(status_code=404, detail="speaker not found")
    return Response(status_code=204)


@router.get("/speakers/{speaker_id}/reference")
def speaker_reference(request: Request, speaker_id: str) -> FileResponse:
    state = _require_service(request)
    _validate_speaker_id(speaker_id)
    profile = state.speakers.get(speaker_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="speaker not found")
    return FileResponse(
        state.speakers.reference_path(speaker_id),
        media_type="audio/wav",
        filename=f"{profile.name}-reference.wav",
    )


# ----------------------------------------------------------------------- clone
async def _clone_common(request: Request, **params) -> tuple[object, Path | None]:
    """Shared logic for /clone and /clone/stream: consent + upload handling."""
    state = _require_service(request)
    s = state.settings
    upload = params.pop("reference_audio")
    consent = params.pop("consent")

    upload_path: Path | None = None
    if upload is not None and (upload.filename or "").strip():
        if s.consent_required and not consent:
            raise HTTPException(
                status_code=403, detail=f"consent required when uploading a reference. {CONSENT_MESSAGE}"
            )
        upload_path = await _save_upload(upload, s.max_upload_mb)
    return state, upload_path


@router.post("/clone")
async def clone(
    request: Request,
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    speaker_id: str = Form(""),
    ref_text: str = Form(""),
    language: str = Form(""),
    nfe_steps: int | None = Form(None),
    cfg_strength: float | None = Form(None),
    speed: float | None = Form(None),
    seed: int | None = Form(None),
    with_metrics: bool = Form(False),
    consent: bool = Form(False),
) -> Response:
    """Synthesize `text` in a cloned voice; returns audio/wav (metrics in X-Voice-* headers)."""
    _clone_params(text, speaker_id, ref_text, language, nfe_steps, cfg_strength, speed, seed)
    state, upload_path = await _clone_common(
        request,
        reference_audio=reference_audio,
        consent=consent,
    )
    try:
        out = await run_in_threadpool(
            state.service.clone,
            text=text,
            speaker_id=speaker_id.strip(),
            upload_path=upload_path,
            ref_text=ref_text,
            language=language,
            nfe_steps=nfe_steps,
            cfg_strength=cfg_strength,
            speed=speed,
            seed=seed,
            with_similarity=with_metrics,
        )
    except SynthesisError as exc:
        raise _synthesis_error(exc) from exc
    except AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:  # noqa: BLE001 — engine failures
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)

    return Response(content=out.wav_bytes, media_type="audio/wav", headers=_metric_headers(out.metrics))


@router.post("/clone/stream")
async def clone_stream(
    request: Request,
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    speaker_id: str = Form(""),
    ref_text: str = Form(""),
    language: str = Form(""),
    nfe_steps: int | None = Form(None),
    cfg_strength: float | None = Form(None),
    speed: float | None = Form(None),
    seed: int | None = Form(None),
    consent: bool = Form(False),
) -> StreamingResponse:
    """Stream NDJSON events: one per sentence chunk (base64 WAV), then a done summary."""
    _clone_params(text, speaker_id, ref_text, language, nfe_steps, cfg_strength, speed, seed)
    state, upload_path = await _clone_common(
        request,
        reference_audio=reference_audio,
        consent=consent,
    )

    def _generate():
        try:
            for event in state.service.stream(
                text=text,
                speaker_id=speaker_id.strip(),
                upload_path=upload_path,
                ref_text=ref_text,
                language=language,
                nfe_steps=nfe_steps,
                cfg_strength=cfg_strength,
                speed=speed,
                seed=seed,
            ):
                yield json.dumps(event) + "\n"
        except SynthesisError as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        except Exception as exc:  # noqa: BLE001 — mid-stream failures surface as error events
            logger.exception("stream failed")
            yield json.dumps({"type": "error", "message": f"synthesis failed: {exc}"}) + "\n"
        finally:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)

    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
