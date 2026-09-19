---
title: Voxclone
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
license: other
short_description: Zero-shot voice cloning — 14 languages, runs on free CPU
---

# VOXCLONE on Hugging Face Spaces

This folder is a drop-in Hugging Face Space (Docker SDK). It serves the full
VOXCLONE console + REST API on a free public URL. Visitors need no account.

Zero-shot cloning from a 6–30 s reference. int8-quantized F5-TTS inference.
Consent is enforced by the API for every enrollment/render from a fresh upload.

## Deploy (2 minutes)

```bash
# one-time
pip install -U huggingface_hub
huggingface-cli login

# create a Space and push this folder as its repo
huggingface-cli repo create voxclone --repo_type space --space_sdk docker
git clone https://huggingface.co/spaces/<your-username>/voxclone
cp -r deploy/hf-space/* voxclone/
cp -r app scripts examples voxclone/          # sources live inside the space too
cd voxclone && git add -A && git commit -m "VOXCLONE" && git push
```

The Space builds (~5 min), downloads the active checkpoint on first boot (~1.3 GB),
then serves `https://<your-username>-voxclone.hf.space` — console at `/`, API at `/api/v1`.

## Notes & limits

- Free CPU basic: 2 vCPU / 16 GB RAM — plenty here (the dev machine ran on 3.8 GB).
  Expect RTF ≈ 10–30 per request; renders are serialized (one at a time by design).
- Space storage is ephemeral on the free tier: enrolled voices and the model cache
  reset on restart. Add persistent storage (paid) or accept the reset.
- Checkpoint licenses: F5-TTS v1 Base & the French fine-tune are CC-BY-NC-4.0
  (non-commercial); IndicF5 is MIT. Keep the demo non-commercial accordingly.
- The consent gate is an attestation enforced by the API (`consent=true` required);
  for a truly public deployment consider adding rate limiting (e.g. a reverse
  proxy or `slowapi`) and an abuse-report contact.
