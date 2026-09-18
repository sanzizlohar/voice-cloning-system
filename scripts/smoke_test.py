"""End-to-end smoke test against a running server (default http://127.0.0.1:8000).

Exercises health, enrollment, sync clone (with metrics), streaming, and cleanup.

Usage:
    python scripts/smoke_test.py [--base http://127.0.0.1:8000] [--ref examples/basic_ref_en.wav]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--ref", default=str(PROJECT_ROOT / "examples" / "basic_ref_en.wav"))
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--language", default="en", help="synthesis language (en, hi, bn, fr, ...)")
    parser.add_argument("--nfe", type=int, default=16, help="NFE steps; lower is faster")
    parser.add_argument("--skip-stream", action="store_true", help="skip the streaming check (slow CPUs)")
    parser.add_argument("--keep", action="store_true", help="keep the enrolled speaker")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.is_file():
        sys.exit(f"reference audio not found: {ref_path} (see examples/)")

    client = httpx.Client(base_url=args.base, timeout=600)

    health = client.get("/api/v1/health").json()
    print(f"[1/5] health: engine={health['engine']} model={health['model']} "
          f"device={health['device']} loaded={health['loaded']}")

    print("[2/5] enrolling voice ...")
    with open(ref_path, "rb") as f:
        r = client.post(
            "/api/v1/speakers",
            files={"reference_audio": (ref_path.name, f, "audio/wav")},
            data={"name": "smoke-test", "consent": "true", "ref_text": args.ref_text},
        )
    r.raise_for_status()
    speaker = r.json()
    speaker_id = speaker["id"]
    print(f"      speaker {speaker_id} ({speaker['duration_seconds']}s, "
          f"transcript: {speaker['ref_text'][:60]!r})")

    try:
        print("[3/5] cloning (sync, with metrics) ...")
        t0 = time.perf_counter()
        r = client.post(
            "/api/v1/clone",
            data={
                "text": "This is a smoke test.",
                "speaker_id": speaker_id,
                "language": args.language,
                "nfe_steps": str(args.nfe),
                "with_metrics": "true",
            },
        )
        r.raise_for_status()
        out = Path("outputs") / "smoke_test.wav"
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(r.content)
        m = r.headers
        print(f"      {out} ({len(r.content) / 1024:.0f} KB) in {time.perf_counter() - t0:.1f}s | "
              f"RTF={m.get('x-voice-rtf')} similarity={m.get('x-voice-similarity')}")

        if args.skip_stream:
            print("[4/5] streaming skipped (--skip-stream)")
            print("[5/5] PASS (partial)")
            return 0

        print(f"[4/5] streaming ...")
        chunks, first_chunk = 0, None
        with client.stream(
            "POST",
            "/api/v1/clone/stream",
            data={
                "text": "Streaming audio begins as soon as the very first sentence is fully ready. "
                        "Every following sentence arrives as its own chunk. "
                        "This is the final one.",
                "speaker_id": speaker_id,
                "language": args.language,
                "nfe_steps": str(args.nfe),
            },
        ) as stream:
            for line in stream.iter_lines():
                if not line.strip():
                    continue
                evt = json.loads(line)
                if evt["type"] == "chunk":
                    chunks += 1
                    if first_chunk is None:
                        first_chunk = evt.get("first_chunk_latency_s") or evt.get("latency_s")
                    base64.b64decode(evt["audio_base64"])
                elif evt["type"] == "error":
                    raise RuntimeError(evt["message"])
        print(f"      {chunks} chunks, first audio after {first_chunk}s")
        ok = chunks == 3 and first_chunk is not None
        print(f"[5/5] {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        if not args.keep:
            client.delete(f"/api/v1/speakers/{speaker_id}")


if __name__ == "__main__":
    sys.exit(main())
