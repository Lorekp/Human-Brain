"""Authenticate ASCII source-line word-unigram cue projection.

This module is a deterministic text-region component.  It retains exact ASCII
source bytes and an authenticated non-empty byte span, derives the complete
containing line, and projects canonical lowercase ASCII-alphanumeric word
unigrams.  Cue roots depend only on the normalized token and one fixed generic
namespace; source, line, and span provenance is confined to the envelope root.

There is no parser, provider abstraction, fallback, truncation, or learning
claim here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256

SCHEMA = "COMMON_ASCII_WORD_UNIGRAM_CUES_V1"
SPAN_SCHEMA = "COMMON_ASCII_BYTE_SPAN_V1"

MAX_SOURCE_BYTES = 65_536
MAX_LINE_BYTES = 8_192
MAX_TEXT_UTF8_BYTES = 8_192
MAX_TOKEN_CHARACTERS = 64
MAX_UNIQUE_TOKENS = 32

_ASCII_WORD_UNIGRAM_CUE_NAMESPACE = b"common-ascii-word-unigram-cue-v1"
_SPAN_NAMESPACE = b"common-ascii-byte-span-v1"
_ENVELOPE_NAMESPACE = b"common-ascii-word-unigram-envelope-v1"


class AsciiWordUnigramCueError(ValueError):
    """ASCII source, span, line, token, or cue custody failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_ascii_source(source: object) -> bytes:
    if type(source) is not bytes:
        raise TypeError("ASCII cue source must be exact bytes")
    if not source or len(source) > MAX_SOURCE_BYTES or any(value > 0x7F for value in source):
        raise AsciiWordUnigramCueError("ASCII cue source is empty, non-ASCII, or over capacity")
    return source


def _source_span_bytes(source: bytes, *, byte_start: object, byte_end: object) -> bytes:
    _validate_ascii_source(source)
    if (
        type(byte_start) is not int
        or type(byte_end) is not int
        or not 0 <= byte_start < byte_end <= len(source)
    ):
        raise AsciiWordUnigramCueError("ASCII byte span is empty or outside its source")
    span_bytes = source[byte_start:byte_end]
    if b"\r" in span_bytes or b"\n" in span_bytes:
        raise AsciiWordUnigramCueError("ASCII byte span crosses a line boundary")
    return span_bytes


def _flush_ascii_word(current: list[str], tokens: set[str]) -> None:
    if not current:
        return
    token = "".join(current).lower()
    current.clear()
    if len(token) > MAX_TOKEN_CHARACTERS:
        raise AsciiWordUnigramCueError("ASCII word unigram capacity exceeded")
    tokens.add(token)


def ascii_word_unigrams_v1(text: str) -> tuple[str, ...]:
    """Return all unique lowercase ASCII-alphanumeric word unigrams."""

    if type(text) is not str or not text:
        raise TypeError("ASCII word-unigram text must be one non-empty exact string")
    if len(text.encode("utf-8")) > MAX_TEXT_UTF8_BYTES:
        raise AsciiWordUnigramCueError("ASCII word-unigram text capacity exceeded")

    tokens: set[str] = set()
    current: list[str] = []
    for character in text:
        if "A" <= character <= "Z" or "a" <= character <= "z" or "0" <= character <= "9":
            current.append(character)
        else:
            _flush_ascii_word(current, tokens)
    _flush_ascii_word(current, tokens)
    if not tokens or len(tokens) > MAX_UNIQUE_TOKENS:
        raise AsciiWordUnigramCueError("ASCII word-unigram inventory is empty or over capacity")
    return tuple(sorted(tokens))


def ascii_word_unigram_cue_sha256_v1(token: str) -> str:
    """Root one canonical token without source, span, provider, or role context."""

    if (
        type(token) is not str
        or not 1 <= len(token) <= MAX_TOKEN_CHARACTERS
        or any(not ("a" <= character <= "z" or "0" <= character <= "9") for character in token)
    ):
        raise AsciiWordUnigramCueError("ASCII word-unigram cue token is not canonical")
    return _digest(_ASCII_WORD_UNIGRAM_CUE_NAMESPACE, {"token": token})


@dataclass(frozen=True, slots=True)
class AsciiByteSpanV1:
    """Exact non-empty single-line byte span bound to one ASCII source."""

    source_sha256: str
    source_size_bytes: int
    byte_start: int
    byte_end: int
    span_bytes: bytes
    span_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SPAN_SCHEMA,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "span_hex": self.span_bytes.hex(),
            "span_sha256": self.span_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self, *, source: bytes) -> None:
        expected_source = _validate_ascii_source(source)
        expected_span = _source_span_bytes(
            expected_source,
            byte_start=self.byte_start,
            byte_end=self.byte_end,
        )
        if (
            type(self.span_bytes) is not bytes
            or not _is_sha256(self.source_sha256)
            or type(self.source_size_bytes) is not int
            or type(self.byte_start) is not int
            or type(self.byte_end) is not int
            or not _is_sha256(self.span_sha256)
            or not _is_sha256(self.sha256)
            or self.source_sha256 != sha256(expected_source).hexdigest()
            or self.source_size_bytes != len(expected_source)
            or self.span_bytes != expected_span
            or self.span_sha256 != sha256(expected_span).hexdigest()
            or self.sha256 != _digest(_SPAN_NAMESPACE, self.record(include_sha256=False))
        ):
            raise AsciiWordUnigramCueError("ASCII byte-span authentication failed")


def seal_ascii_byte_span_v1(
    source: bytes,
    *,
    byte_start: int,
    byte_end: int,
) -> AsciiByteSpanV1:
    """Seal one exact non-empty single-line span of an ASCII source."""

    source = _validate_ascii_source(source)
    span_bytes = _source_span_bytes(source, byte_start=byte_start, byte_end=byte_end)
    partial = AsciiByteSpanV1(
        source_sha256=sha256(source).hexdigest(),
        source_size_bytes=len(source),
        byte_start=byte_start,
        byte_end=byte_end,
        span_bytes=span_bytes,
        span_sha256=sha256(span_bytes).hexdigest(),
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_SPAN_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate(source=source)
    return result


def _containing_line(source: bytes, span: AsciiByteSpanV1) -> tuple[int, int, bytes]:
    span.authenticate(source=source)
    line_start = span.byte_start
    while line_start > 0 and source[line_start - 1] not in (0x0A, 0x0D):
        line_start -= 1
    line_end = span.byte_end
    while line_end < len(source) and source[line_end] not in (0x0A, 0x0D):
        line_end += 1
    line = source[line_start:line_end]
    if not line or len(line) > MAX_LINE_BYTES:
        raise AsciiWordUnigramCueError("containing ASCII line is empty or over capacity")
    return line_start, line_end, line


@dataclass(frozen=True, slots=True)
class AsciiWordUnigramCueEnvelopeV1:
    """Frozen provenance that rederives source + span -> line -> tokens -> cues."""

    source: bytes
    source_sha256: str
    source_size_bytes: int
    span: AsciiByteSpanV1
    line_byte_start: int
    line_byte_end: int
    line: bytes
    line_sha256: str
    tokens: tuple[str, ...]
    cue_sha256s: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "source_hex": self.source.hex(),
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "span": self.span.record(),
            "line_byte_start": self.line_byte_start,
            "line_byte_end": self.line_byte_end,
            "line_hex": self.line.hex(),
            "line_sha256": self.line_sha256,
            "tokens": self.tokens,
            "cue_sha256s": self.cue_sha256s,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        source = _validate_ascii_source(self.source)
        if type(self.span) is not AsciiByteSpanV1:
            raise TypeError("ASCII word-unigram envelope span must be exact")
        if type(self.line) is not bytes:
            raise TypeError("ASCII word-unigram envelope line must be exact bytes")
        if type(self.tokens) is not tuple or type(self.cue_sha256s) is not tuple:
            raise TypeError("ASCII word-unigram tokens and cue roots must be exact tuples")
        if (
            not _is_sha256(self.source_sha256)
            or type(self.source_size_bytes) is not int
            or type(self.line_byte_start) is not int
            or type(self.line_byte_end) is not int
            or not _is_sha256(self.line_sha256)
            or not _is_sha256(self.sha256)
            or any(type(token) is not str for token in self.tokens)
            or any(not _is_sha256(cue) for cue in self.cue_sha256s)
        ):
            raise AsciiWordUnigramCueError("ASCII word-unigram envelope fields are not exact")
        line_start, line_end, line = _containing_line(source, self.span)
        tokens = ascii_word_unigrams_v1(line.decode("ascii"))
        cues = tuple(ascii_word_unigram_cue_sha256_v1(token) for token in tokens)
        if (
            self.source_sha256 != sha256(source).hexdigest()
            or self.source_size_bytes != len(source)
            or self.line_byte_start != line_start
            or self.line_byte_end != line_end
            or self.line != line
            or self.line_sha256 != sha256(line).hexdigest()
            or self.tokens != tokens
            or self.cue_sha256s != cues
            or len(set(cues)) != len(cues)
            or self.sha256 != _digest(_ENVELOPE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise AsciiWordUnigramCueError("ASCII word-unigram envelope authentication failed")


def project_ascii_word_unigram_cues_v1(
    source: bytes,
    *,
    span: AsciiByteSpanV1,
) -> AsciiWordUnigramCueEnvelopeV1:
    """Project one authenticated ASCII source span to its complete line cues."""

    source = _validate_ascii_source(source)
    if type(span) is not AsciiByteSpanV1:
        raise TypeError("ASCII word-unigram projection span must be exact")
    line_start, line_end, line = _containing_line(source, span)
    tokens = ascii_word_unigrams_v1(line.decode("ascii"))
    cues = tuple(ascii_word_unigram_cue_sha256_v1(token) for token in tokens)
    partial = AsciiWordUnigramCueEnvelopeV1(
        source=source,
        source_sha256=sha256(source).hexdigest(),
        source_size_bytes=len(source),
        span=span,
        line_byte_start=line_start,
        line_byte_end=line_end,
        line=line,
        line_sha256=sha256(line).hexdigest(),
        tokens=tokens,
        cue_sha256s=cues,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_ENVELOPE_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result
