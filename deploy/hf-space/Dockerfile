# VOXCLONE — Hugging Face Space (Docker SDK)
# Free tier: 2 vCPU / 16 GB RAM. Serves console + API on port 7860.

FROM python:3.12-slim

# ffmpeg provides the shared libraries torchcodec needs for audio decoding
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 vox

WORKDIR /srv/app

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/srv/app/hf \
    VC_PRELOAD=1 \
    VC_DEVICE=cpu \
    VC_QUANTIZED=1 \
    VC_NFE_STEPS=16 \
    VC_MAX_LOADED_MODELS=1 \
    VC_DATA_DIR=/srv/app/data \
    HF_HUB_ETAG_TIMEOUT=5

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY examples ./examples

RUN mkdir -p /srv/app/data/speakers /srv/app/data/cache /srv/app/hf \
    && chown -R vox:vox /srv/app
USER vox

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -sf http://localhost:7860/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
