"""Unit tests for sentence segmentation."""

from __future__ import annotations

from app.services.text import split_sentences


def test_splits_on_sentence_boundaries():
    # min_chars=0 disables merging so we can assert pure boundary behavior
    text = "This is the first sentence. Here comes another one! And a final question?"
    chunks = split_sentences(text, min_chars=0)
    assert chunks == [
        "This is the first sentence.",
        "Here comes another one!",
        "And a final question?",
    ]


def test_merges_short_prefixes_by_default():
    # a chunk shorter than min_chars pulls the next sentence in
    text = "This is the first sentence. Here comes another one! And a final question?"
    chunks = split_sentences(text)  # default min_chars=40
    assert len(chunks) == 2
    assert chunks[0] == "This is the first sentence. Here comes another one!"


def test_runs_of_fragments_merge_into_one_chunk():
    assert split_sentences("Hi. Ok. Sure.", min_chars=40) == ["Hi. Ok. Sure."]


def test_hard_wraps_long_sentences():
    text = " ".join(["word"] * 100) + "."  # one ~500 char "sentence"
    chunks = split_sentences(text, max_chars=180)
    assert all(len(c) <= 180 for c in chunks)
    assert " ".join(chunks) == text


def test_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_normalizes_whitespace():
    assert split_sentences("Hello\n   world.  Next  one!", min_chars=0) == [
        "Hello world.",
        "Next one!",
    ]
