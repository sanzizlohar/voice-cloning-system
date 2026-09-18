"""Evaluate cloning quality: speaker similarity of generated audio vs the reference.

For every reference clip (default examples/*.wav) synthesizes a few texts and
computes the ECAPA speaker-embedding cosine similarity between reference and
clone. Values of ~0.6-0.9 indicate the same voice identity.

Usage:
    python scripts/evaluate.py [--refs examples/basic_ref_en.wav] [--nfe 16] [--runs 2]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.services.synthesis import CloningService  # noqa: E402
from app.engines import build_engine  # noqa: E402
from app.services import similarity as similarity_svc  # noqa: E402
from app.services.speakers import SpeakerStore  # noqa: E402
from app.services.text import split_sentences  # noqa: E402

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog while the rain gently falls outside.",
    "Speech synthesis has improved dramatically; cloned voices are now nearly indistinguishable from real recordings.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", nargs="*", default=None, help="reference audio files")
    parser.add_argument(
        "--ref-text",
        default="Some call me nature. Others call me Mother Nature.",
        help="transcript of the reference clips (skips transcription)",
    )
    parser.add_argument("--texts", nargs="*", default=DEFAULT_TEXTS)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    if not similarity_svc.available():
        sys.exit("speechbrain is required for evaluation (pip install speechbrain)")

    refs = [Path(p) for p in (args.refs or [str(PROJECT_ROOT / "examples" / "basic_ref_en.wav")])]
    missing = [p for p in refs if not p.is_file()]
    if missing:
        sys.exit(f"reference audio not found: {missing}")

    engine = build_engine(settings)
    engine.load()
    speakers = SpeakerStore(
        settings.speakers_dir,
        sample_rate=settings.sample_rate,
        min_ref_seconds=settings.min_ref_seconds,
        max_ref_seconds=settings.max_ref_seconds,
        enable_embeddings=False,
    )
    service = CloningService(engine=engine, speakers=speakers, settings=settings)

    rows = []
    for ref in refs:
        print(f"\n=== {ref.name} ===", flush=True)
        # a temp speaker profile gives us stored transcripts + embeddings for free
        profile = speakers.enroll(
            name=f"eval-{ref.stem}", source_path=ref, ref_text=args.ref_text, consent=False
        )
        sims = []
        try:
            for i, text in enumerate(args.texts):
                out = service.clone(
                    text=text,
                    speaker_id=profile.id,
                    nfe_steps=args.nfe,
                    with_similarity=True,
                    use_cache=False,
                )
                sim = out.metrics.get("speaker_similarity")
                if sim is None:
                    print(f"  text {i + 1}: similarity unavailable")
                    continue
                sims.append(sim)
                print(
                    f"  text {i + 1}: similarity={sim:.4f}  "
                    f"latency={out.metrics['latency_s']}s  audio={out.metrics['audio_seconds']}s",
                    flush=True,
                )
                # keep one sample per reference for manual listening
                sample = settings.data_dir / "eval_samples"
                sample.mkdir(parents=True, exist_ok=True)
                (sample / f"{ref.stem}_{i + 1}.wav").write_bytes(out.wav_bytes)
        finally:
            speakers.delete(profile.id)
        if sims:
            rows.append({"reference": ref.name, "nfe": args.nfe,
                         "mean_similarity": round(statistics.mean(sims), 4),
                         "min_similarity": round(min(sims), 4), "n_texts": len(sims)})

    if rows:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md = out_dir / "evaluation.md"
        lines = [
            "# Cloning quality — speaker similarity (ECAPA-TDNN cosine)",
            "",
            "| reference | nfe | mean similarity | min | texts |",
            "|---|---|---|---|---|",
            *[f"| {r['reference']} | {r['nfe']} | {r['mean_similarity']} | {r['min_similarity']} | {r['n_texts']} |"
              for r in rows],
            "",
            "Samples saved to `data/eval_samples/` for manual listening.",
        ]
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / "evaluation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
