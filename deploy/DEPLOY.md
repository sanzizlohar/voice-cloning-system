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

## Option 2 — Permanent & free: Oracle Cloud Always Free ARM VM

HF changed policy: Docker/Gradio Spaces now require PRO ($9/mo); free tiers of
Render/Koyeb are 512 MB (too small for a 1.3 GB model). The genuinely-free
permanent path is an Oracle Cloud **Always Free** ARM VM (4 cores / 24 GB):

1. Sign up at oracle.com/cloud/free (card needed for verification, no charge)
2. Create an ARM (Ampere A1) instance — Ubuntu 22.04, 4 OCPU / 24 GB
3. On the VM: install docker, clone this repo, `docker compose up -d`
   (or run uvicorn + the watchdog), open port 8000 in the security list
4. Put Cloudflare (free) in front for TLS + caching: `cloudflared tunnel`

Also viable:
- **HF PRO ($9/mo)** — the prepared Space in `hf-space/` deploys with
  `python scripts/deploy_hf.py` (token + auto-deploy already wired via GitHub
  Actions; gated by the `HF_DEPLOY_ENABLED` repo variable)
- **Modal.com** — $30/mo free credits, serverless CPU/GPU functions

The quick tunnel (Option 1) remains free and unlimited-time on a personal machine.

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
