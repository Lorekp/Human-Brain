import re
from dataclasses import FrozenInstanceError

import pytest

from humanbrain.byte_events import MAX_EVENT_BYTES, ByteEvent, ByteEventSegmenter, segment_bytes
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig


def test_tokenizer_is_lossless_for_unicode_and_unseen_text() -> None:
    tokenizer = BytePhraseTokenizer.fit(
        ["cell membrane cell membrane"],
        TokenizerConfig(max_phrases=32, min_frequency=2),
    )
    text = "cell membrane — Δ unseen"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size > 256


def test_phrase_tokens_reduce_repeated_text() -> None:
    text = "pyramidal pyramidal pyramidal"
    tokenizer = BytePhraseTokenizer.fit([text])
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


_EVENT_FIXTURES = [
    b"", b"a", b"?", b"Record k00 has value ab.", b"Az09_\t\n\v\f\r !?",
    b"\xff\xc0\x80\xed\xa0\x80\x00a", bytes(range(256)),
    *[b"x" * length for length in (15, 16, 17, 31, 32, 33)],
    *[(b" \t\n\v\f\r" * 6)[:length] for length in (15, 16, 17, 31, 32, 33)],
]


def _regex_byte_events(raw):
    """Independent standard-library span oracle, with no streaming state machine."""
    events = []
    for match in re.finditer(rb"[A-Za-z0-9_]+|[\x09-\x0d\x20]+|[\s\S]", raw):
        for start in range(match.start(), match.end(), 16):
            end = min(start + 16, match.end())
            events.append(ByteEvent(start, end, raw[start:end]))
    return tuple(events)


@pytest.mark.parametrize("raw", _EVENT_FIXTURES)
def test_byte_events_exact_spans_reconstruction_and_chunk_invariance(raw):
    expected = _regex_byte_events(raw)
    assert segment_bytes(raw) == expected
    assert b"".join(event.payload for event in expected) == raw
    assert all(0 < len(event.payload) <= MAX_EVENT_BYTES for event in expected)
    assert all(event.payload == raw[event.start:event.end] for event in expected)
    # Every split of the short fixtures; longer all-byte fixture also gets byte chunks.
    splits = range(len(raw) + 1) if len(raw) <= 48 else (0, len(raw) // 2, len(raw))
    chunkings = [(raw[:split], raw[split:]) for split in splits]
    chunkings.append(tuple(raw[index:index + 1] for index in range(len(raw))))
    for chunks in chunkings:
        segmenter = ByteEventSegmenter()
        actual = ()
        for chunk in chunks:
            actual += segmenter.feed(chunk)
            assert segmenter.buffered_bytes < MAX_EVENT_BYTES
        actual += segmenter.finish()
        assert actual == expected
        assert segmenter.bytes_seen == len(raw)
        assert segmenter.buffered_bytes == 0
        assert segmenter.max_buffered_bytes <= MAX_EVENT_BYTES
        assert segmenter.finish() == ()


def test_byte_events_emit_caps_immediately_and_retain_only_unfinished_run():
    segmenter = ByteEventSegmenter()
    assert segmenter.feed(b"a" * 15) == ()
    assert segmenter.buffered_bytes == 15
    assert segmenter.feed(b"a") == (ByteEvent(0, 16, b"a" * 16),)
    assert segmenter.buffered_bytes == 0
    assert segmenter.max_buffered_bytes == 16
    assert segmenter.feed(b"b? ") == (ByteEvent(16, 17, b"b"), ByteEvent(17, 18, b"?"))
    assert segmenter.finish() == (ByteEvent(18, 19, b" "),)
    # Object owns counters plus the empty unfinished buffer, no emitted history.
    assert not any(isinstance(value, (bytes, list, tuple, dict))
                   for value in vars(segmenter).values())
    assert segmenter._buffer == bytearray()


@pytest.mark.parametrize("invalid", ["text", bytearray(b"bytes"), memoryview(b"bytes"), None, [97]])
def test_byte_events_reject_invalid_input_without_consuming_pending_bytes(invalid):
    segmenter = ByteEventSegmenter()
    assert segmenter.feed(b"abc") == ()
    with pytest.raises(ValueError, match="raw bytes"):
        segmenter.feed(invalid)
    assert segmenter.bytes_seen == 3 and segmenter.buffered_bytes == 3
    assert segmenter.finish() == (ByteEvent(0, 3, b"abc"),)
    with pytest.raises(ValueError, match="raw bytes"):
        segment_bytes(invalid)


def test_byte_events_finish_has_no_model_eos_and_streams_are_independent():
    left, right = ByteEventSegmenter(), ByteEventSegmenter()
    assert segment_bytes(b"") == ()
    assert left.feed(b"a") == right.feed(b"b") == ()
    assert left.finish() == (ByteEvent(0, 1, b"a"),)
    assert right.feed(b"c") == ()
    assert right.finish() == (ByteEvent(0, 2, b"bc"),)
    with pytest.raises(ValueError, match="finished"):
        left.feed(b"")
    event = segment_bytes(b"x")[0]
    with pytest.raises(FrozenInstanceError):
        event.payload = b"y"


def test_byte_event_preflight_counts_every_chunking_without_neural_eos():
    from experiments.r4_integrated.local_byte_event_preflight_v1 import (
        count_invocations,
        qualify_input,
    )

    result = qualify_input({"id": "tiny", "kind": "fixture", "input_hex": b"ab \xff".hex()})
    assert result["events"] == [
        {"start": 0, "end": 2, "payload_hex": "6162"},
        {"start": 2, "end": 3, "payload_hex": "20"},
        {"start": 3, "end": 4, "payload_hex": "ff"},
    ]
    assert [row["name"] for row in result["variants"]] == [
        "single", "bytes", "split-0", "split-1", "split-2", "split-3", "split-4"
    ]
    assert count_invocations(result["variants"]) == {
        "segmentation_invocations": 7, "raw_bytes_offered": 28,
        "raw_events_emitted": 21, "feed_calls": 15, "finish_calls": 7,
        "conceptual_end_events": 7, "conceptual_byte_baseline_steps": 35,
        "conceptual_controller_events": 28, "max_buffered_bytes": 2,
    }
    empty = qualify_input({"id": "empty", "kind": "fixture", "input_hex": ""})
    assert empty["events"] == []
    counts = count_invocations(empty["variants"])
    assert counts["segmentation_invocations"] == counts["conceptual_end_events"] == 3
    assert counts["raw_bytes_offered"] == counts["raw_events_emitted"] == 0
