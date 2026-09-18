"""Benchmark F5-TTS synthesis speed: fp32 vs int8 dynamic quantization.

Measures wall-clock latency and real-time factor (RTF) for four configs:
    fp32 nfe=32   (default quality)
    fp32 nfe=16   (fewer ODE steps)
    int8 nfe=32   (dynamic quantization)
    int8 nfe=16   (both)

Also scores speaker similarity (ECAPA cosine) for fp32 vs int8 output to show
quantization retains voice quality. Writes results/benchmark_<device>.md and .json.

Usage:
    python scripts/benchmark.py [--runs 3] [--warmup 1] [--ref examples/basic_ref_en.wav]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.engines.f5 import F5Engine  # noqa: E402
from app.services import audio as audio_svc  # noqa: E402
from app.services import similarity as similarity_svc  # noqa: E402

DEFAULT_TEXT = (
    "This voice was synthesized by a neural text to speech model, cloned from a short "
    "reference recording of a single speaker. The pipeline runs entirely on this machine."
)
SEED = 1234


def main() -> None:
    import torch

    # keep memory/thread overhead predictable on small CPUs (2-core dev box)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=str(PROJECT_ROOT / "examples" / "basic_ref_en.wav"))
    parser.add_argument(
        "--ref-text",
        default="Some call me nature. Others call me Mother Nature.",
        help="transcript of the reference clip (always provide it: an empty transcript "
        "triggers f5's internal whisper transcription, which is slow and memory-hungry)",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--configs",
        default="fp32:32,fp32:16,int8:32,int8:16",
        help="comma list of precision:nfe, e.g. 'fp32:16,int8:16' (fp32 always runs before in-place int8)",
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.is_file():
        sys.exit(f"reference audio not found: {ref_path}")

    # normalize the reference once so every config sees identical input
    ref_wav, _duration = audio_svc.prepare_reference(
        ref_path, settings.sample_rate, settings.min_ref_seconds, settings.max_ref_seconds
    )
    ref_sr = settings.sample_rate  # prepare_reference decodes at this rate
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    prepared_ref = settings.data_dir / "benchmark_ref.wav"
    audio_svc.write_wav(prepared_ref, ref_wav, ref_sr)

    # parse requested configs; fp32 always runs before int8 (fresh instance)
    configs = []
    for part in args.configs.split(","):
        precision, nfe = part.strip().split(":")
        configs.append((int(nfe), precision.strip().lower() == "int8"))
    configs.sort(key=lambda c: c[1])
    results = []

    engine = F5Engine(settings)
    engine.load()
    print(f"device={engine.device}  runs={args.runs}  warmup={args.warmup}  ref={prepared_ref.name}")

    def run_config(nfe: int, want_quant: bool, eng: F5Engine) -> None:
        for _ in range(args.warmup):
            eng.synthesize(
                ref_audio_path=str(prepared_ref), ref_text=args.ref_text,
                text=args.text, nfe_steps=nfe, seed=SEED,
            )
        times, durations, last = [], [], None
        for _ in range(args.runs):
            r = eng.synthesize(
                ref_audio_path=str(prepared_ref), ref_text=args.ref_text,
                text=args.text, nfe_steps=nfe, seed=SEED,
            )
            times.append(r.inference_seconds)
            durations.append(r.duration_seconds)
            last = r

        entry = {
            "config": f"{'int8' if want_quant else 'fp32'} nfe={nfe}",
            "quantized": want_quant,
            "nfe": nfe,
            "runs_s": [round(t, 3) for t in times],
            "mean_latency_s": round(statistics.mean(times), 3),
            "audio_seconds": round(statistics.mean(durations), 3),
            "rtf": round(statistics.mean(times) / statistics.mean(durations), 3),
        }
        if similarity_svc.available() and last is not None:
            sim = similarity_svc.similarity_from_wavs(ref_wav, ref_sr, last.audio, last.sample_rate)
            entry["speaker_similarity"] = round(sim, 4)
        results.append(entry)
        sim_s = f"  sim={entry.get('speaker_similarity')}" if "speaker_similarity" in entry else ""
        print(
            f"{entry['config']:>12}:  mean {entry['mean_latency_s']:>7.2f}s"
            f"  audio {entry['audio_seconds']:>6.2f}s  RTF {entry['rtf']:>6.2f}{sim_s}",
            flush=True,
        )

    # int8 dynamic quantization must happen before any fp32 inference on a model
    # instance (post-inference quantization trips over inference-time state), so
    # fp32 configs run on one instance and int8 configs on a fresh quantized one
    fp32_configs = [(nfe, q) for nfe, q in configs if not q]
    int8_configs = [(nfe, q) for nfe, q in configs if q]

    for nfe, _ in fp32_configs:
        run_config(nfe, False, engine)

    if int8_configs:
        if engine.device != "cpu":
            print("skipping quantized configs (dynamic int8 quantization is CPU-only)")
        else:
            del engine
            gc.collect()
            quant_settings = settings
            quant_settings.quantized = True
            engine = F5Engine(quant_settings)
            engine.load()
            if not engine.is_quantized:
                print("skipping quantized configs (quantization failed at load)")
            else:
                for nfe, _ in int8_configs:
                    run_config(nfe, True, engine)

    baseline = next((r for r in results if r["config"] == "fp32 nfe=32"), results[0])
    for r in results:
        r["speedup_vs_fp32_nfe32_pct"] = round(
            (baseline["mean_latency_s"] - r["mean_latency_s"]) / baseline["mean_latency_s"] * 100, 1
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"benchmark_{engine.device}.json"
    md_path = out_dir / f"benchmark_{engine.device}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        f"# Synthesis benchmark — device: {engine.device}",
        "",
        f"- model: `{engine.model_label}`  |  runs: {args.runs} (+{args.warmup} warmup)  |  seed: {SEED}",
        f"- reference: `{prepared_ref.name}` ({len(ref_wav) / ref_sr:.1f}s)",
        f"- text: {len(args.text)} chars -> ~{baseline['audio_seconds']}s of audio",
        "",
        "| config | mean latency | audio | RTF | speedup | similarity |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['mean_latency_s']}s | {r['audio_seconds']}s "
            f"| {r['rtf']} | {r['speedup_vs_fp32_nfe32_pct']:+.1f}% "
            f"| {r.get('speaker_similarity', '—')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {json_path} and {md_path}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"total wall time: {time.perf_counter() - t0:.0f}s")
