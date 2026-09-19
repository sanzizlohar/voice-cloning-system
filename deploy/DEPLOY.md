# Deployment guide — "anyone can use it, no account"

Two real paths. GitHub Pages / Cloudflare Pages **cannot** host this project's
inference (they serve static files only — no Python/PyTorch backend), but the
options below both give you a public URL where anyone can use the console
without creating an account.

## Option 1 — Instant: Cloudflare quick tunnel (no account, runs on your PC)

```bat
cloudflared.exe tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

It prints a `https://<random>.trycloudflare.com` URL — share that. The console,
API, and audio all work through it (CORS and ports are handled by the tunnel).

- ✅ live in 2 minutes, zero accounts, TLS included
- ⚠️ temporary: URL changes each run, and your machine must stay on
- ⚠️ this dev box is a 2-core CPU with 3.8 GB RAM — fine for 1–2 concurrent
  visitors (renders are serialized), not a crowd

## Option 2 — Permanent: Hugging Face Space (free, always on)

See [`hf-space/`](./hf-space/) — a ready Docker Space (2 vCPU / 16 GB free tier).
Deploy steps are in [`hf-space/README.md`](./hf-space/README.md). Result:
`https://<you>-voxclone.hf.space`, always on, no visitor accounts.

- ✅ survives reboots, 16 GB RAM (no OOM), public and shareable
- ⚠️ free-tier storage is ephemeral (enrolled voices reset on restart)
- ⚠️ first boot downloads the model (~1.3 GB), so allow a slow warm-up

## Split-frontend variant (advanced)

The console (`app/static/index.html`) is backend-agnostic. You *can* host it on
GitHub Pages / Cloudflare Pages and point it at a deployed API (set
`VC_CORS_ORIGINS` on the backend to allow the Pages origin). The console alone
is not usable without a backend — that split only makes sense once an API is
public somewhere.

## Responsible exposure

- The consent gate is an API-enforced attestation, not identity verification.
- Renders are serialized (`VC_MAX_LOADED_MODELS` + engine lock) — a natural
  throttle, but add rate limiting (`slowapi`, or a Cloudflare proxy in front)
  before inviting traffic.
- Checkpoint licenses are CC-BY-NC-4.0 (en/fr) and MIT (IndicF5) — keep the
  public demo non-commercial.
