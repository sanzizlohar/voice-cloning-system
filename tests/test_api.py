"""API tests — full request/response cycles against the fake engine."""

from __future__ import annotations


def test_languages_endpoint(client):
    r = client.get("/api/v1/languages")
    assert r.status_code == 200
    langs = r.json()
    codes = {l["code"] for l in langs}
    assert {"en", "hi", "bn", "fr"} <= codes  # the headline multilingual set
    assert all(l["model_key"] for l in langs)
    assert any(l["is_default"] for l in langs)


def test_clone_with_language_param(client, enrolled_speaker):
    r = client.post(
        "/api/v1/clone",
        data={"text": "Nomoshkar! Eta ekta bhasha porikkha.", "speaker_id": enrolled_speaker["id"],
              "language": "bn"},
    )
    assert r.status_code == 200
    assert r.headers["x-voice-model"].startswith("fake")
    # a different language must not collide with the default-language cache
    r2 = client.post(
        "/api/v1/clone",
        data={"text": "Nomoshkar! Eta ekta bhasha porikkha.", "speaker_id": enrolled_speaker["id"]},
    )
    assert r2.headers["x-voice-cached"] == "false"


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "fake"
    assert body["version"]


def test_enroll_requires_consent(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/api/v1/speakers",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"name": "NoConsent"},
        )
    assert r.status_code == 403
    assert "consent" in r.json()["detail"].lower()


def test_enroll_rejects_invalid_name(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/api/v1/speakers",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"name": "   ", "consent": "true"},
        )
    assert r.status_code == 400


def test_enroll_rejects_empty_upload(client):
    r = client.post(
        "/api/v1/speakers",
        files={"reference_audio": ("empty.wav", b"", "audio/wav")},
        data={"name": "X", "consent": "true"},
    )
    assert r.status_code == 400


def test_enroll_and_list(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/api/v1/speakers",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"name": "Listed", "consent": "true", "ref_text": "a transcript"},
        )
    assert r.status_code == 201
    speaker = r.json()
    assert speaker["name"] == "Listed"
    assert speaker["ref_text"] == "a transcript"
    assert speaker["duration_seconds"] > 0

    speakers = client.get("/api/v1/speakers").json()
    assert any(s["id"] == speaker["id"] for s in speakers)

    assert client.delete(f"/api/v1/speakers/{speaker['id']}").status_code == 204
    assert client.get(f"/api/v1/speakers/{speaker['id']}").status_code == 404


def test_clone_with_enrolled_speaker(client, enrolled_speaker):
    r = client.post(
        "/api/v1/clone",
        data={"text": "Hello from the fake engine.", "speaker_id": enrolled_speaker["id"]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 1000
    assert r.headers["x-voice-cached"] == "false"
    assert float(r.headers["x-voice-rtf"]) >= 0
    assert r.headers["x-voice-model"] == "fake"


def test_clone_caches_repeat_requests(client, enrolled_speaker):
    data = {"text": "Cache me once.", "speaker_id": enrolled_speaker["id"]}
    first = client.post("/api/v1/clone", data=data)
    second = client.post("/api/v1/clone", data=data)
    assert first.status_code == second.status_code == 200
    assert first.headers["x-voice-cached"] == "false"
    assert second.headers["x-voice-cached"] == "true"
    assert first.content == second.content


def test_clone_with_upload_requires_consent(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/api/v1/clone",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "no consent given"},
        )
    assert r.status_code == 403


def test_clone_with_upload_and_consent(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/api/v1/clone",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "Cloned from a direct upload.", "consent": "true",
                  "ref_text": "test transcript"},
        )
    assert r.status_code == 200
    assert len(r.content) > 1000


def test_clone_without_reference_fails(client):
    r = client.post("/api/v1/clone", data={"text": "no reference"})
    assert r.status_code == 400
    assert "reference" in r.json()["detail"]


def test_clone_unknown_speaker_is_404(client):
    r = client.post("/api/v1/clone", data={"text": "hi", "speaker_id": "0" * 12})
    assert r.status_code == 404


def test_clone_invalid_speaker_id_format(client):
    r = client.post("/api/v1/clone", data={"text": "hi", "speaker_id": "../../etc"})
    assert r.status_code == 400


def test_clone_validates_params(client, enrolled_speaker):
    base = {"speaker_id": enrolled_speaker["id"]}
    assert client.post("/api/v1/clone", data={**base, "text": "x", "nfe_steps": "500"}).status_code == 400
    assert client.post("/api/v1/clone", data={**base, "text": "x", "speed": "9"}).status_code == 400
    assert client.post("/api/v1/clone", data={**base, "text": ""}).status_code in (400, 422)


def test_clone_stream_returns_ndjson(client, enrolled_speaker):
    r = client.post(
        "/api/v1/clone/stream",
        data={
            "text": "The first sentence is long enough to stand entirely on its own here. "
                    "The second sentence also carries plenty of words for the chunker. "
                    "And the third sentence finishes the stream with a complete thought.",
            "speaker_id": enrolled_speaker["id"],
        },
    )
    assert r.status_code == 200
    events = [__import__("json").loads(line) for line in r.text.splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types.count("chunk") == 3
    assert types[-1] == "done"
    for e in events:
        if e["type"] == "chunk":
            assert len(e["audio_base64"]) > 100
            assert e["sentence"]
        if e["type"] == "done":
            assert e["total_latency_s"] >= 0


def test_speaker_reference_download(client, enrolled_speaker):
    r = client.get(f"/api/v1/speakers/{enrolled_speaker['id']}/reference")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 1000


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Voice" in r.content and b"Cloning" in r.content
