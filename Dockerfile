FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

ENV PIP_NO_CACHE_DIR=1 \
    VC_PRELOAD=1 \
    VC_DEVICE=cpu \
    VC_DATA_DIR=/srv/app/data

# torch first (CPU wheels from PyPI), then the rest
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY examples ./examples

VOLUME /srv/app/data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -sf http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
