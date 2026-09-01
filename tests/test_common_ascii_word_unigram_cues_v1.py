from __future__ import annotations

from dataclasses import replace

import pytest
from experiments.r3_pilot.common_ascii_word_unigram_cues_v1 import (
    MAX_LINE_BYTES,
    MAX_SOURCE_BYTES,
    MAX_TOKEN_CHARACTERS,
    MAX_UNIQUE_TOKENS,
    AsciiWordUnigramCueError,
    ascii_word_unigram_cue_sha256_v1,
    ascii_word_unigrams_v1,
    project_ascii_word_unigram_cues_v1,
    seal_ascii_byte_span_v1,
)


def _project(source: bytes, token: bytes):
    byte_start = source.index(token)
    span = seal_ascii_byte_span_v1(
        source,
        byte_start=byte_start,
        byte_end=byte_start + len(token),
    )
    return project_ascii_word_unigram_cues_v1(source, span=span)


def test_complete_crlf_line_is_derived_and_tokenized_canonically() -> None:
    source = b"header\r\n  Alpha_beta ALPHA 42; beta!\r\ntrailer"
    envelope = _project(source, b"beta!")

    envelope.authenticate()
    envelope.span.authenticate(source=source)
    assert envelope.span.span_bytes == b"beta!"
    assert envelope.line == b"  Alpha_beta ALPHA 42; beta!"
    assert envelope.line_byte_start == len(b"header\r\n")
    assert envelope.line_byte_end == source.index(b"\r\ntrailer")
    assert envelope.tokens == ("42", "alpha", "beta")
    assert envelope.cue_sha256s == tuple(
        ascii_word_unigram_cue_sha256_v1(token) for token in envelope.tokens
    )


def test_normalized_token_roots_are_source_and_role_neutral() -> None:
    uppercase = _project(b"SIGNAL", b"SIGNAL")
    lowercase = _project(b"signal", b"signal")
    expected = ascii_word_unigram_cue_sha256_v1("signal")

    assert ascii_word_unigrams_v1("SIGNAL signal") == ("signal",)
    assert uppercase.tokens == lowercase.tokens == ("signal",)
    assert uppercase.cue_sha256s == lowercase.cue_sha256s == (expected,)
    assert uppercase.span.sha256 != lowercase.span.sha256
    assert uppercase.sha256 != lowercase.sha256


def test_invalid_or_cross_source_spans_fail_closed() -> None:
    with pytest.raises(TypeError, match="exact bytes"):
        seal_ascii_byte_span_v1(bytearray(b"abc"), byte_start=0, byte_end=1)  # type: ignore[arg-type]

    for byte_start, byte_end in ((0, 0), (2, 1), (-1, 1), (0, 4)):
        with pytest.raises(AsciiWordUnigramCueError, match="outside its source"):
            seal_ascii_byte_span_v1(
                b"abc",
                byte_start=byte_start,
                byte_end=byte_end,
            )

    for byte_start, byte_end in ((2, 5), (3, 4)):
        with pytest.raises(AsciiWordUnigramCueError, match="line boundary"):
            seal_ascii_byte_span_v1(
                b"abc\ndef",
                byte_start=byte_start,
                byte_end=byte_end,
            )

    with pytest.raises(AsciiWordUnigramCueError, match="non-ASCII"):
        seal_ascii_byte_span_v1(b"\xff", byte_start=0, byte_end=1)

    alpha_span = seal_ascii_byte_span_v1(b"alpha", byte_start=0, byte_end=5)
    with pytest.raises(AsciiWordUnigramCueError, match="authentication failed"):
        project_ascii_word_unigram_cues_v1(b"omega", span=alpha_span)


def test_source_line_token_and_inventory_capacities_fail_closed() -> None:
    at_capacity = " ".join(f"w{index}" for index in range(MAX_UNIQUE_TOKENS))
    over_capacity = f"{at_capacity} w{MAX_UNIQUE_TOKENS}"

    assert len(ascii_word_unigrams_v1(at_capacity)) == MAX_UNIQUE_TOKENS
    with pytest.raises(AsciiWordUnigramCueError, match="inventory"):
        ascii_word_unigrams_v1(over_capacity)
    with pytest.raises(AsciiWordUnigramCueError, match="capacity exceeded"):
        ascii_word_unigrams_v1("a" * (MAX_TOKEN_CHARACTERS + 1))
    with pytest.raises(AsciiWordUnigramCueError, match="source is empty"):
        seal_ascii_byte_span_v1(
            b"a" * (MAX_SOURCE_BYTES + 1),
            byte_start=0,
            byte_end=1,
        )

    overlong_line = b"a" + b"!" * MAX_LINE_BYTES
    span = seal_ascii_byte_span_v1(overlong_line, byte_start=0, byte_end=1)
    with pytest.raises(AsciiWordUnigramCueError, match="line is empty or over capacity"):
        project_ascii_word_unigram_cues_v1(overlong_line, span=span)


def test_envelope_mutations_are_rejected_by_reauthentication() -> None:
    envelope = _project(b"alpha beta", b"alpha")
    mutations = (
        replace(envelope, source=b"omega beta"),
        replace(envelope, source_size_bytes=True),
        replace(envelope, line=b"alpha zeta"),
        replace(envelope, tokens=("alpha", "zeta")),
        replace(envelope, cue_sha256s=("0" * 64, "1" * 64)),
        replace(envelope, span=replace(envelope.span, span_bytes=b"omega")),
        replace(envelope, sha256="0" * 64),
    )

    for mutation in mutations:
        with pytest.raises(AsciiWordUnigramCueError):
            mutation.authenticate()
