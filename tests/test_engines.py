"""Engine-level tests using the offline fake engine."""

from __future__ import annotations

import numpy as np

from app.engines.fake import FakeEngine


def make_engine() -> FakeEngine:
    engine = FakeEngine()
    engine.load()
    return engine


def test_synthesize_deterministic():
    engine = make_engine()
    kwargs = dict(ref_audio_path="ref.wav", ref_text="ref text", text="Hello world.")
    a = engine.synthesize(**kwargs)
    b = engine.synthesize(**kwargs)
    assert np.array_equal(a.audio, b.audio)
    assert a.sample_rate == 24000
    assert 0 < a.duration_seconds <= 6.0
    assert a.rtf >= 0


def test_synthesize_rejects_empty_text():
    engine = make_engine()
    try:
        engine.synthesize(ref_audio_path="r.wav", ref_text="t", text="   ")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_quantize_requires_load():
    engine = FakeEngine()
    assert engine.quantize() is False
    engine.load()
    assert engine.quantize() is True
    assert engine.is_quantized is True
    result = engine.synthesize(ref_audio_path="r.wav", ref_text="t", text="quantized")
    assert result.quantized is True


def test_synth_before_load_raises():
    engine = FakeEngine()
    try:
        engine.synthesize(ref_audio_path="r.wav", ref_text="t", text="x")
        raised = False
    except RuntimeError:
        raised = True
    assert raised
