"""Speaker-verification embeddings (ECAPA-TDNN) used to score cloning fidelity."""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_encoder = None


def available() -> bool:
    try:
        import speechbrain  # noqa: F401

        return True
    except ImportError:
        return False


def _get_encoder():
    global _encoder
    with _lock:
        if _encoder is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # speechbrain < 1.0
                from speechbrain.pretrained import EncoderClassifier  # type: ignore

            savedir = Path(settings.models_dir) / "ecapa-voxceleb"
            if not (savedir / "hyperparams.yaml").is_file():
                # fetch the whole repo with plain file copies — speechbrain's own
                # fetch tries symlinks, which need admin/dev-mode on Windows
                from huggingface_hub import snapshot_download

                logger.info("Downloading ECAPA-TDNN speaker encoder (~100 MB) ...")
                snapshot_download(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    local_dir=str(savedir),
                    cache_dir=settings.hf_cache_dir or None,
                )
            # speechbrain's hyperparams reference alias names the HF repo stores
            # under different filenames; copy (don't symlink) them into place —
            # its own fetch() would symlink, which needs admin rights on Windows
            for src_name, dst_name in {"label_encoder.txt": "label_encoder.ckpt"}.items():
                src, dst = savedir / src_name, savedir / dst_name
                if src.is_file() and not dst.is_file():
                    shutil.copy2(src, dst)
            logger.info("Loading ECAPA-TDNN speaker encoder ...")
            _encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(savedir),
                run_opts={"device": "cpu"},
            )
        return _encoder


def embedding_from_wav(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Speaker embedding for a mono float32 waveform (resampled to 16 kHz internally)."""
    import librosa
    import torch

    if sample_rate != 16000:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
    wav = wav[: 16000 * 30]  # ECAPA is trained on short utterances; cap at 30 s

    encoder = _get_encoder()
    tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(tensor, wav_lens=torch.tensor([1.0]))
    return emb.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def similarity_from_wavs(
    ref_wav: np.ndarray, ref_sr: int, out_wav: np.ndarray, out_sr: int
) -> float:
    """Cosine similarity between the reference and generated speaker embeddings.

    Typical range for a good clone: ~0.6-0.9 (ECAPA on 24 kHz vocoded speech).
    """
    ref_emb = embedding_from_wav(ref_wav, ref_sr)
    out_emb = embedding_from_wav(out_wav, out_sr)
    return cosine_similarity(ref_emb, out_emb)


def similarity_from_paths(ref_path: str | Path, out_path: str | Path) -> float:
    import librosa

    ref, sr = librosa.load(str(ref_path), sr=None, mono=True)
    out, sr2 = librosa.load(str(out_path), sr=None, mono=True)
    return similarity_from_wavs(np.asarray(ref), int(sr), np.asarray(out), int(sr2))
