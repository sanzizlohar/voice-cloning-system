"""One-command deploy of VOXCLONE to a Hugging Face Space (Docker SDK).

Stages the space layout (HF README metadata + Dockerfile + sources) and
uploads it. Creates the Space on first run, updates it on every later run.

Usage:
    set HF_TOKEN=hf_xxx                     (Windows)   or  export HF_TOKEN=...
    python scripts/deploy_hf.py             # uses env token or cached login
    python scripts/deploy_hf.py --token hf_xxx --name voxclone
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# what lands in the Space repo (flat): HF metadata README + Dockerfile + sources
COPY_DIRS = ["app", "scripts", "examples"]
COPY_FILES = ["requirements.txt"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""), help="HF token (or set HF_TOKEN)")
    parser.add_argument("--name", default="voxclone", help="Space name")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub missing — pip install huggingface_hub")

    api = HfApi(token=args.token or None)
    who = api.whoami()
    username = who["name"]
    repo_id = f"{username}/{args.name}"
    print(f"deploying to Space: {repo_id}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    # stage a flat copy the way HF expects: README.md + Dockerfile at root
    staging = Path(tempfile.mkdtemp(prefix="voxclone_space_"))
    src = PROJECT_ROOT / "deploy" / "hf-space"
    shutil.copy2(src / "README.md", staging / "README.md")
    shutil.copy2(src / "Dockerfile", staging / "Dockerfile")
    for d in COPY_DIRS:
        shutil.copytree(PROJECT_ROOT / d, staging / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in COPY_FILES:
        shutil.copy2(PROJECT_ROOT / f, staging / f)

    url = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(staging),
        commit_message="VOXCLONE deploy from voice-cloning-system",
    )
    print(f"uploaded -> {url}")
    print(f"live at  -> https://{username}-{args.name}.hf.space  (first boot: ~5 min build + ~2 min model download)")
    shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
