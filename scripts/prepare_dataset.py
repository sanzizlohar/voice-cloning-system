"""Prepare a fine-tuning dataset from raw recordings (the "100+ hours" pipeline).

Takes a folder of audio files (any format) and produces a training-ready corpus:
    <out>/wavs/*.wav        24 kHz mono, silence-trimmed, peak-normalized
    <out>/metadata.csv      rows of: wavs/<name>.wav|<transcript>

Transcripts come from (in priority order):
    1. same-stem .txt files next to the audio
    2. --transcribe (faster-whisper, downloads a model on first use)
    3. empty (fine if you will transcribe later; training requires text)

Long recordings are optionally chopped at silence gaps (--split-long), which is
how you turn hours of raw session audio into short training utterances.

Usage:
    python scripts/prepare_dataset.py --input ~/recordings --output data/dataset --split-long
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm", ".wma", ".aac"}


def split_on_silence(
    wav: np.ndarray,
    sr: int,
    top_db: float,
    min_silence_ms: int,
    min_chunk_s: float,
    max_chunk_s: float,
) -> list[np.ndarray]:
    """Chop `wav` at the middle of silence gaps longer than min_silence_ms."""
    hop = 512
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-9, ref=max(float(rms.max()), 1e-9))
    quiet = db < -abs(top_db)
    min_frames = int(min_silence_ms * sr / 1000 / hop)

    cuts, run_start = [0], None
    for idx, is_quiet in enumerate(quiet):
        if is_quiet and run_start is None:
            run_start = idx
        elif not is_quiet and run_start is not None:
            if idx - run_start >= min_frames:
                cuts.append(int((run_start + idx) / 2 * hop))
            run_start = None
    cuts.append(len(wav))

    chunks = []
    for a, b in zip(cuts, cuts[1:]):
        duration = (b - a) / sr
        if duration < min_chunk_s:
            continue
        if duration > max_chunk_s:  # hard-split anything still too long
            n = int(np.ceil(duration / max_chunk_s))
            edges = np.linspace(a, b, n + 1).astype(int)
            chunks.extend(wav[s:e] for s, e in zip(edges, edges[1:]))
        else:
            chunks.append(wav[a:b])
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="folder of raw audio files")
    parser.add_argument("--output", required=True, help="output dataset folder")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-sec", type=float, default=1.0)
    parser.add_argument("--max-sec", type=float, default=30.0)
    parser.add_argument("--top-db", type=float, default=40.0)
    parser.add_argument("--split-long", action="store_true",
                        help="chop long recordings at silence gaps (min 350ms)")
    parser.add_argument("--transcribe", action="store_true",
                        help="transcribe with faster-whisper when no .txt exists")
    parser.add_argument("--language", default=None, help="whisper language hint, e.g. en")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        if args.transcribe:
            sys.exit("--transcribe requires faster-whisper (pip install faster-whisper)")

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        sys.exit(f"input folder not found: {in_dir}")
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"no audio files found in {in_dir}")
    print(f"{len(files)} source files -> {wavs_dir}")

    whisper = None
    if args.transcribe:
        print("loading faster-whisper 'base' ...")
        whisper = WhisperModel("base", device="cpu", compute_type="int8")

    rows, skipped = [], 0
    for src in files:
        try:
            wav, _ = librosa.load(str(src), sr=args.sample_rate, mono=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {src.name}: decode failed ({exc})")
            skipped += 1
            continue
        wav, _ = librosa.effects.trim(wav, top_db=args.top_db)
        if len(wav) < args.min_sec * args.sample_rate:
            skipped += 1
            continue

        if args.split_long and len(wav) > args.max_sec * args.sample_rate:
            pieces = split_on_silence(
                wav, args.sample_rate, args.top_db,
                min_silence_ms=350, min_chunk_s=args.min_sec, max_chunk_s=args.max_sec,
            )
        else:
            pieces = [wav[: int(args.max_sec * args.sample_rate)]]

        # transcript source 1: sibling .txt; source 2: whisper
        txt_path = src.with_suffix(".txt")
        transcript = txt_path.read_text(encoding="utf-8").strip() if txt_path.is_file() else ""
        if not transcript and whisper is not None:
            segments, _ = whisper.transcribe(str(src), language=args.language, vad_filter=True)
            transcript = " ".join(s.text.strip() for s in segments).strip()

        stem = src.stem
        for i, piece in enumerate(pieces):
            peak = float(np.max(np.abs(piece)))
            if peak < 1e-4:
                continue
            piece = (piece / peak * 0.95).astype(np.float32)
            name = f"{stem}_{i:03d}" if len(pieces) > 1 else stem
            out_path = wavs_dir / f"{name}.wav"
            sf.write(str(out_path), piece, args.sample_rate, subtype="PCM_16")
            rows.append([f"wavs/{name}.wav", " ".join(transcript.split())])
        print(f"  {src.name}: {len(pieces)} clip(s)")

    meta = out_dir / "metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(rows)

    total_sec = sum(
        sf.info(str(wavs_dir / Path(r[0]).name)).duration for r in rows if (wavs_dir / Path(r[0]).name).is_file()
    )
    print(f"\nwrote {meta} with {len(rows)} clips ({total_sec / 3600:.2f} hours, {skipped} sources skipped)")
    if any(not r[1] for r in rows):
        print("NOTE: some clips have empty transcripts — training requires text; "
              "re-run with --transcribe or provide .txt files.")


if __name__ == "__main__":
    main()
