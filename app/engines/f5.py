"""F5-TTS engine — zero-shot voice cloning via flow matching.

Runs a small LRU pool of F5-TTS checkpoints from the language registry
(app/engines/f5_models.py), so one service can clone across languages:
en/zh (F5 v1 base), 11 Indic languages incl. Hindi & Bengali (IndicF5),
and French. At most `VC_MAX_LOADED_MODELS` checkpoints stay in memory.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from app.config import Settings
from app.engines.base import SynthesisResult, TTSEngine
from app.engines.f5_models import ModelSpec, spec_for_language

logger = logging.getLogger(__name__)


class _QuietProgress:
    """f5's infer() expects an object exposing .tqdm; this shim keeps server logs clean."""

    def tqdm(self, iterable, *args, **kwargs):
        return iterable


class F5Engine(TTSEngine):
    name = "f5"
    sample_rate = 24000
    supports_dynamic_quantization = True

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = settings.device or self._autodetect_device()
        self._models: OrderedDict[str, object] = OrderedDict()  # spec.key -> F5TTS instance
        self._quantize_flag = bool(settings.quantized)
        self._quantized = False
        self._lock = threading.RLock()
        if settings.quantized and self.device != "cpu":
            logger.warning("VC_QUANTIZED=1 ignored: dynamic int8 quantization is CPU-only")

    # ------------------------------------------------------------------ basics
    @staticmethod
    def _autodetect_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 — torch import issues shouldn't kill the API
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        return bool(self._models)

    @property
    def model_label(self) -> str:
        loaded = list(self._models)
        return f"{self.name}:{loaded[-1]}" if loaded else f"{self.name}:not-loaded"

    def is_model_loaded(self, key: str) -> bool:
        return key in self._models

    @property
    def loaded_keys(self) -> list[str]:
        return list(self._models)

    # ------------------------------------------------------------------- load
    def load(self) -> None:
        """Preload the default language's checkpoint (safe to call repeatedly)."""
        self._get_model(spec_for_language(self.settings.default_language))

    def _download(self, spec: ModelSpec) -> tuple[Path, Path | None]:
        from huggingface_hub import hf_hub_download

        cache_dir = self.settings.hf_cache_dir or None
        ckpt = Path(
            hf_hub_download(repo_id=spec.repo, filename=spec.ckpt_file, cache_dir=cache_dir)
        )
        vocab = None
        if spec.vocab_file:
            vocab = Path(
                hf_hub_download(repo_id=spec.repo, filename=spec.vocab_file, cache_dir=cache_dir)
            )
        return ckpt, vocab

    def _build_model(self, spec: ModelSpec):
        from f5_tts.api import F5TTS  # deferred: heavy import

        ckpt, vocab = self._download(spec)
        logger.info("building %s from %s (%s)", spec.key, spec.repo, spec.note)
        f5 = F5TTS(
            model=spec.arch,
            ckpt_file=str(ckpt),
            vocab_file=str(vocab) if vocab else "",
            device=self.device,
            hf_cache_dir=self.settings.hf_cache_dir or None,
        )
        if self._quantize_flag and self.device == "cpu":
            self._quantize_instance(f5)
        return f5

    def _get_model(self, spec: ModelSpec):
        with self._lock:
            if spec.key in self._models:
                self._models.move_to_end(spec.key)
                return self._models[spec.key]
            logger.info("Loading TTS model '%s' (%s) on %s ...", spec.key, spec.repo, self.device)
            t0 = time.perf_counter()
            self._models[spec.key] = self._build_model(spec)
            self._evict_locked()
            logger.info("model '%s' ready in %.1fs", spec.key, time.perf_counter() - t0)
            return self._models[spec.key]

    def _evict_locked(self) -> None:
        while len(self._models) > self.settings.max_loaded_models:
            key, model = self._models.popitem(last=False)
            logger.info("evicting model '%s' from memory (LRU)", key)
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------- quantization
    def quantize(self) -> bool:
        """Dynamic int8 quantization of Linear layers — CPU only, in place.

        Applies to every loaded model and to any model loaded later.
        """
        if self.device != "cpu":
            return False
        self._quantize_flag = True
        with self._lock:
            for f5 in self._models.values():
                self._quantize_instance(f5)
        return True

    def _quantize_instance(self, f5) -> bool:
        import torch
        from torch.ao.quantization import quantize_dynamic

        try:
            target = f5.ema_model  # the CFM backbone lives here in f5_tts >= 1.1
            n_linear = sum(1 for m in target.modules() if isinstance(m, torch.nn.Linear))
            if n_linear == 0:
                return True  # already quantized
            logger.info("quantizing %d Linear layers to int8 ...", n_linear)
            f5.ema_model = quantize_dynamic(target, {torch.nn.Linear}, dtype=torch.qint8)
            self._quantized = True
            return True
        except Exception as exc:  # noqa: BLE001 — keep fp32 weights if quantization fails
            logger.warning("dynamic quantization failed (%s); keeping fp32 weights", exc)
            return False

    # -------------------------------------------------------------- synthesis
    def synthesize(
        self,
        *,
        ref_audio_path: str,
        ref_text: str,
        text: str,
        language: str | None = None,
        nfe_steps: int | None = None,
        cfg_strength: float | None = None,
        speed: float | None = None,
        seed: int | None = None,
    ) -> SynthesisResult:
        spec = spec_for_language(language or self.settings.default_language)
        f5 = self._get_model(spec)

        nfe = int(nfe_steps or self.settings.nfe_steps)
        cfg = float(cfg_strength if cfg_strength is not None else self.settings.cfg_strength)
        spd = float(speed if speed is not None else self.settings.speed)
        gen_text = " ".join((text or "").split())
        if not gen_text:
            raise ValueError("text is empty")

        t0 = time.perf_counter()
        wav, sr, _spec_out = f5.infer(
            ref_file=str(ref_audio_path),
            ref_text=ref_text,
            gen_text=gen_text,
            nfe_step=nfe,
            cfg_strength=cfg,
            speed=spd,
            target_rms=0.1,
            cross_fade_duration=0.15,
            seed=seed,
            show_info=lambda msg: logger.info("%s", msg),
            progress=_QuietProgress(),
        )
        elapsed = time.perf_counter() - t0

        audio = np.asarray(wav, dtype=np.float32).squeeze()
        duration = max(len(audio) / float(sr), 1e-6)
        return SynthesisResult(
            audio=audio,
            sample_rate=int(sr),
            inference_seconds=round(elapsed, 3),
            rtf=round(elapsed / duration, 3),
            model=f"{self.name}:{spec.key}",
            quantized=self._quantized,
            nfe_steps=nfe,
            seed=seed,
            extra={"language": spec.languages, "repo": spec.repo},
        )
