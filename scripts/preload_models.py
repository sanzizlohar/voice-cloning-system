"""Pre-download every checkpoint in the language registry (no model loading).

Useful before going offline or to warm a fresh deployment:
    python scripts/preload_models.py           # all languages
    python scripts/preload_models.py --keys en indic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.engines import f5_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="*", default=None, help="model keys (en, indic, fr)")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    for spec in f5_models.all_specs():
        if args.keys and spec.key not in args.keys:
            continue
        for filename in (spec.ckpt_file, spec.vocab_file):
            if not filename:
                continue
            t0 = time.perf_counter()
            print(f"[{spec.key}] {spec.repo}/{filename} ...", flush=True)
            path = hf_hub_download(
                repo_id=spec.repo, filename=filename, cache_dir=settings.hf_cache_dir or None
            )
            size_mb = Path(path).stat().st_size / 1e6
            print(f"  done: {size_mb:.0f} MB in {time.perf_counter() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
