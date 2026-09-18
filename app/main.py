"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import METRIC_HEADERS
from app.api.routes import router
from app.config import settings
from app.engines import build_engine
from app.services.speakers import SpeakerStore
from app.services.synthesis import CloningService

logger = logging.getLogger(__name__)


def _configure_torch_threads() -> None:
    """Use all logical cores for inference (torch defaults to physical cores,
    which leaves ~30% of small CPUs on the table). VC_TORCH_THREADS overrides."""
    try:
        import torch

        n = int(os.getenv("VC_TORCH_THREADS", "0")) or (os.cpu_count() or 2)
        torch.set_num_threads(max(1, n))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # parallel work already started — safe to ignore
        logger.info("torch threads: %d intra-op", n)
    except Exception:  # noqa: BLE001 — threading config must never break startup
        logger.warning("could not configure torch threads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = build_engine(settings)
    speakers = SpeakerStore(
        settings.speakers_dir,
        sample_rate=settings.sample_rate,
        min_ref_seconds=settings.min_ref_seconds,
        max_ref_seconds=settings.max_ref_seconds,
        enable_embeddings=settings.enable_similarity,
    )
    service = CloningService(engine=engine, speakers=speakers, settings=settings)

    app.state.settings = settings
    app.state.engine = engine
    app.state.speakers = speakers
    app.state.service = service

    if settings.preload_model:

        def _load() -> None:
            try:
                engine.load()
            except Exception:  # noqa: BLE001 — server stays up; first request retries
                logger.exception("background model load failed")

        threading.Thread(target=_load, name="model-loader", daemon=True).start()
    yield


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    _configure_torch_threads()
    app = FastAPI(
        title="Voice Cloning System",
        description=(
            "Zero-shot voice cloning service: enroll a short reference recording, "
            "then synthesize arbitrary text in that voice (F5-TTS + FastAPI)."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=METRIC_HEADERS,
    )
    app.include_router(router)

    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
