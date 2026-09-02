"""Lossless hierarchical events for opaque query and document byte streams.

Lexical atoms remain byte-identical wherever they occur.  Position-framed
occurrence events keep repeated atoms distinct, while generic boundaries retain
query, document, sentence, provenance, and byte-offset structure.  This module
only observes bytes; it does not attach semantic roles or make decisions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Self

__all__ = [
    "OpaqueDocument",
    "OpaqueDocumentEventTape",
    "OpaqueDocumentEventTapeError",
    "OpaqueSentenceOccurrence",
    "OpaqueTokenOccurrence",
]


_LEXICAL = re.compile(rb"[A-Za-z0-9_]+")
_SENTENCE_TERMINATOR = re.compile(rb"[.!?]+")
_WHITESPACE = frozenset(b" \t\r\n")
_LENGTH_BYTES = 8

_OBSERVATION_START = b"\x00opaque-document-event-tape/observation-start/v1"
_OBSERVATION_END = b"\x00opaque-document-event-tape/observation-end/v1"
_QUERY_START = b"\x00opaque-document-event-tape/query-start/v1"
_QUERY_END = b"\x00opaque-document-event-tape/query-end/v1"
_DOCUMENTS_START = b"\x00opaque-document-event-tape/documents-start/v1"
_DOCUMENTS_END = b"\x00opaque-document-event-tape/documents-end/v1"
_DOCUMENT_START = b"\x00opaque-document-event-tape/document-start/v1"
_DOCUMENT_END = b"\x00opaque-document-event-tape/document-end/v1"
_CONTENT_START = b"\x00opaque-document-event-tape/content-start/v1"
_CONTENT_END = b"\x00opaque-document-event-tape/content-end/v1"

_DOCUMENT_ID = b"\x00opaque-document-event-tape/document-id/v1\x00"
_PROVENANCE_ID = b"\x00opaque-document-event-tape/provenance-id/v1\x00"
_LITERAL = b"\x00opaque-document-event-tape/literal/v1\x00"
_SENTENCE_START = b"\x00opaque-document-event-tape/sentence-start/v1\x00"
_SENTENCE_END = b"\x00opaque-document-event-tape/sentence-end/v1\x00"
_TOKEN_OCCURRENCE = b"\x00opaque-document-event-tape/token-occurrence/v1\x00"
_CONSTRUCTION_TOKEN = object()


class OpaqueDocumentEventTapeError(ValueError):
    """The supplied bytes cannot be represented by this exact observation."""


@dataclass(frozen=True, slots=True)
class OpaqueDocument:
    """One opaque document byte stream with opaque identity and provenance."""

    document_id: bytes
    provenance_id: bytes
    content: bytes

    def __post_init__(self) -> None:
        for name, value in (
            ("document_id", self.document_id),
            ("provenance_id", self.provenance_id),
        ):
            if not isinstance(value, bytes) or not value:
                raise OpaqueDocumentEventTapeError(f"{name} must be non-empty bytes")
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")


@dataclass(frozen=True, slots=True)
class OpaqueTokenOccurrence:
    """One positioned lexical identity in the current observation."""

    identity_event: bytes
    occurrence_event: bytes
    document_index: int | None
    document_id: bytes | None
    provenance_id: bytes | None
    sentence_index: int
    token_index: int
    sentence_token_index: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class OpaqueSentenceOccurrence:
    """One exact sentence-scale span and its immutable event hierarchy."""

    document_index: int | None
    document_id: bytes | None
    provenance_id: bytes | None
    sentence_index: int
    byte_start: int
    byte_end: int
    events: tuple[bytes, ...]
    token_occurrences: tuple[OpaqueTokenOccurrence, ...]

    def reconstruct(self) -> bytes:
        """Reconstruct this exact span from its events."""

        return _decode_sentence_events(self.events)


def _frame(prefix: bytes, payload: bytes) -> bytes:
    return prefix + len(payload).to_bytes(_LENGTH_BYTES, "big") + payload


def _unframe(event: bytes, prefix: bytes, name: str) -> bytes:
    if not isinstance(event, bytes) or not event.startswith(prefix):
        raise OpaqueDocumentEventTapeError(f"{name} frame changed")
    length_start = len(prefix)
    payload_start = length_start + _LENGTH_BYTES
    if len(event) < payload_start:
        raise OpaqueDocumentEventTapeError(f"{name} frame length is missing")
    length = int.from_bytes(event[length_start:payload_start], "big")
    payload = event[payload_start:]
    if len(payload) != length:
        raise OpaqueDocumentEventTapeError(f"{name} frame length changed")
    return payload


def _position_event(prefix: bytes, *positions: int) -> bytes:
    return prefix + b"".join(position.to_bytes(_LENGTH_BYTES, "big") for position in positions)


def _token_occurrence_event(
    *,
    stream_index: int,
    sentence_index: int,
    token_index: int,
    sentence_token_index: int,
    byte_start: int,
    byte_end: int,
    identity: bytes,
) -> bytes:
    return _frame(
        _TOKEN_OCCURRENCE,
        b"".join(
            position.to_bytes(_LENGTH_BYTES, "big")
            for position in (
                stream_index,
                sentence_index,
                token_index,
                sentence_token_index,
                byte_start,
                byte_end,
            )
        )
        + identity,
    )


def _sentence_spans(data: bytes) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_TERMINATOR.finditer(data):
        end = match.end()
        while end < len(data) and data[end] in _WHITESPACE:
            end += 1
        if end > start:
            spans.append((start, end))
            start = end
    if start < len(data):
        spans.append((start, len(data)))
    return tuple(spans)


def _stream_occurrences(
    data: bytes,
    *,
    stream_index: int,
    document_index: int | None,
    document_id: bytes | None,
    provenance_id: bytes | None,
) -> tuple[tuple[OpaqueSentenceOccurrence, ...], tuple[OpaqueTokenOccurrence, ...]]:
    sentences: list[OpaqueSentenceOccurrence] = []
    stream_tokens: list[OpaqueTokenOccurrence] = []

    for sentence_index, (byte_start, byte_end) in enumerate(_sentence_spans(data)):
        sentence = data[byte_start:byte_end]
        sentence_start = _position_event(
            _SENTENCE_START,
            stream_index,
            sentence_index,
            byte_start,
            byte_end,
        )
        sentence_end = _position_event(
            _SENTENCE_END,
            stream_index,
            sentence_index,
            byte_start,
            byte_end,
        )
        events: list[bytes] = [sentence_start]
        sentence_tokens: list[OpaqueTokenOccurrence] = []
        cursor = 0

        for sentence_token_index, match in enumerate(_LEXICAL.finditer(sentence)):
            if cursor < match.start():
                events.append(_frame(_LITERAL, sentence[cursor : match.start()]))
            identity = match.group(0)
            absolute_start = byte_start + match.start()
            absolute_end = byte_start + match.end()
            occurrence_event = _token_occurrence_event(
                stream_index=stream_index,
                sentence_index=sentence_index,
                token_index=len(stream_tokens),
                sentence_token_index=sentence_token_index,
                byte_start=absolute_start,
                byte_end=absolute_end,
                identity=identity,
            )
            occurrence = OpaqueTokenOccurrence(
                identity_event=identity,
                occurrence_event=occurrence_event,
                document_index=document_index,
                document_id=document_id,
                provenance_id=provenance_id,
                sentence_index=sentence_index,
                token_index=len(stream_tokens),
                sentence_token_index=sentence_token_index,
                byte_start=absolute_start,
                byte_end=absolute_end,
            )
            events.extend((occurrence_event, identity))
            sentence_tokens.append(occurrence)
            stream_tokens.append(occurrence)
            cursor = match.end()

        if cursor < len(sentence):
            events.append(_frame(_LITERAL, sentence[cursor:]))
        events.append(sentence_end)
        sentence_occurrence = OpaqueSentenceOccurrence(
            document_index=document_index,
            document_id=document_id,
            provenance_id=provenance_id,
            sentence_index=sentence_index,
            byte_start=byte_start,
            byte_end=byte_end,
            events=tuple(events),
            token_occurrences=tuple(sentence_tokens),
        )
        if sentence_occurrence.reconstruct() != sentence:
            raise RuntimeError("sentence event hierarchy ceased to be lossless")
        sentences.append(sentence_occurrence)

    return tuple(sentences), tuple(stream_tokens)


def _decode_sentence_events(events: tuple[bytes, ...]) -> bytes:
    if (
        len(events) < 2
        or not events[0].startswith(_SENTENCE_START)
        or not events[-1].startswith(_SENTENCE_END)
    ):
        raise OpaqueDocumentEventTapeError("sentence boundaries changed")

    parts: list[bytes] = []
    cursor = 1
    while cursor < len(events) - 1:
        event = events[cursor]
        if event.startswith(_LITERAL):
            parts.append(_unframe(event, _LITERAL, "literal"))
            cursor += 1
            continue
        if not event.startswith(_TOKEN_OCCURRENCE) or cursor + 1 >= len(events) - 1:
            raise OpaqueDocumentEventTapeError("token occurrence hierarchy changed")
        payload = _unframe(event, _TOKEN_OCCURRENCE, "token occurrence")
        fixed_position_bytes = 6 * _LENGTH_BYTES
        if len(payload) <= fixed_position_bytes:
            raise OpaqueDocumentEventTapeError("token occurrence identity is missing")
        identity = payload[fixed_position_bytes:]
        if _LEXICAL.fullmatch(identity) is None or events[cursor + 1] != identity:
            raise OpaqueDocumentEventTapeError("token occurrence identity changed")
        parts.append(identity)
        cursor += 2
    return b"".join(parts)


def _query_events(sentences: tuple[OpaqueSentenceOccurrence, ...]) -> tuple[bytes, ...]:
    return (
        _QUERY_START,
        *(event for sentence in sentences for event in sentence.events),
        _QUERY_END,
    )


def _document_events(
    document: OpaqueDocument,
    sentences: tuple[OpaqueSentenceOccurrence, ...],
) -> tuple[bytes, ...]:
    return (
        _DOCUMENT_START,
        _frame(_DOCUMENT_ID, document.document_id),
        _frame(_PROVENANCE_ID, document.provenance_id),
        _CONTENT_START,
        *(event for sentence in sentences for event in sentence.events),
        _CONTENT_END,
        _DOCUMENT_END,
    )


def _decode_stream_events(
    events: tuple[bytes, ...],
    *,
    start_event: bytes,
    end_event: bytes,
) -> bytes:
    if len(events) < 2 or events[0] != start_event or events[-1] != end_event:
        raise OpaqueDocumentEventTapeError("stream boundaries changed")
    parts: list[bytes] = []
    cursor = 1
    while cursor < len(events) - 1:
        if not events[cursor].startswith(_SENTENCE_START):
            raise OpaqueDocumentEventTapeError("sentence start changed")
        sentence_start = cursor
        cursor += 1
        while cursor < len(events) - 1 and not events[cursor].startswith(_SENTENCE_END):
            cursor += 1
        if cursor >= len(events) - 1:
            raise OpaqueDocumentEventTapeError("sentence end is missing")
        parts.append(_decode_sentence_events(events[sentence_start : cursor + 1]))
        cursor += 1
    return b"".join(parts)


def _decode_document_events(events: tuple[bytes, ...]) -> OpaqueDocument:
    if (
        len(events) < 6
        or events[0] != _DOCUMENT_START
        or events[3] != _CONTENT_START
        or events[-2] != _CONTENT_END
        or events[-1] != _DOCUMENT_END
    ):
        raise OpaqueDocumentEventTapeError("document hierarchy changed")
    document_id = _unframe(events[1], _DOCUMENT_ID, "document identity")
    provenance_id = _unframe(events[2], _PROVENANCE_ID, "provenance identity")
    content_events = (_CONTENT_START, *events[4:-2], _CONTENT_END)
    content = _decode_stream_events(
        content_events,
        start_event=_CONTENT_START,
        end_event=_CONTENT_END,
    )
    return OpaqueDocument(document_id, provenance_id, content)


def _stable_event_hash(events: tuple[bytes, ...]) -> bytes:
    digest = hashlib.sha256()
    for event in events:
        digest.update(len(event).to_bytes(_LENGTH_BYTES, "big"))
        digest.update(event)
    return digest.digest()


@dataclass(frozen=True, slots=True, init=False)
class OpaqueDocumentEventTape:
    """One immutable query and ordered multi-document observation."""

    _events: tuple[bytes, ...]
    _query_events: tuple[bytes, ...]
    _document_events: tuple[tuple[bytes, ...], ...]
    _query_sentences: tuple[OpaqueSentenceOccurrence, ...]
    _document_sentences: tuple[tuple[OpaqueSentenceOccurrence, ...], ...]
    _query_token_occurrences: tuple[OpaqueTokenOccurrence, ...]
    _document_token_occurrences: tuple[tuple[OpaqueTokenOccurrence, ...], ...]
    _stable_hash: bytes

    def __init__(
        self,
        token: object,
        *,
        events: tuple[bytes, ...],
        query_events: tuple[bytes, ...],
        document_events: tuple[tuple[bytes, ...], ...],
        query_sentences: tuple[OpaqueSentenceOccurrence, ...],
        document_sentences: tuple[tuple[OpaqueSentenceOccurrence, ...], ...],
        query_token_occurrences: tuple[OpaqueTokenOccurrence, ...],
        document_token_occurrences: tuple[tuple[OpaqueTokenOccurrence, ...], ...],
        stable_hash: bytes,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use OpaqueDocumentEventTape.observe")
        object.__setattr__(self, "_events", events)
        object.__setattr__(self, "_query_events", query_events)
        object.__setattr__(self, "_document_events", document_events)
        object.__setattr__(self, "_query_sentences", query_sentences)
        object.__setattr__(self, "_document_sentences", document_sentences)
        object.__setattr__(self, "_query_token_occurrences", query_token_occurrences)
        object.__setattr__(self, "_document_token_occurrences", document_token_occurrences)
        object.__setattr__(self, "_stable_hash", stable_hash)

    @classmethod
    def observe(
        cls,
        query: bytes,
        documents: tuple[OpaqueDocument, ...],
    ) -> Self:
        """Derive one deterministic observation from complete raw byte streams."""

        if not isinstance(query, bytes) or not query:
            raise OpaqueDocumentEventTapeError("query must be non-empty bytes")
        if type(documents) is not tuple or not documents:
            raise OpaqueDocumentEventTapeError("documents must be a non-empty tuple")
        if any(not isinstance(document, OpaqueDocument) for document in documents):
            raise TypeError("documents must contain OpaqueDocument values")
        if len({document.document_id for document in documents}) != len(documents):
            raise OpaqueDocumentEventTapeError("document identities must be unique")

        query_sentences, query_tokens = _stream_occurrences(
            query,
            stream_index=0,
            document_index=None,
            document_id=None,
            provenance_id=None,
        )
        query_events = _query_events(query_sentences)

        document_sentences: list[tuple[OpaqueSentenceOccurrence, ...]] = []
        document_tokens: list[tuple[OpaqueTokenOccurrence, ...]] = []
        document_event_streams: list[tuple[bytes, ...]] = []
        for document_index, document in enumerate(documents):
            sentences, tokens = _stream_occurrences(
                document.content,
                stream_index=document_index + 1,
                document_index=document_index,
                document_id=document.document_id,
                provenance_id=document.provenance_id,
            )
            document_sentences.append(sentences)
            document_tokens.append(tokens)
            document_event_streams.append(_document_events(document, sentences))

        frozen_document_events = tuple(document_event_streams)
        events = (
            _OBSERVATION_START,
            *query_events,
            _DOCUMENTS_START,
            *(event for stream in frozen_document_events for event in stream),
            _DOCUMENTS_END,
            _OBSERVATION_END,
        )
        observation = cls(
            _CONSTRUCTION_TOKEN,
            events=events,
            query_events=query_events,
            document_events=frozen_document_events,
            query_sentences=query_sentences,
            document_sentences=tuple(document_sentences),
            query_token_occurrences=query_tokens,
            document_token_occurrences=tuple(document_tokens),
            stable_hash=_stable_event_hash(events),
        )
        if observation.reconstruct_query() != query:
            raise RuntimeError("query event hierarchy ceased to be lossless")
        if observation.reconstruct_documents() != documents:
            raise RuntimeError("document event hierarchy ceased to be lossless")
        return observation

    @property
    def events(self) -> tuple[bytes, ...]:
        """Complete observation events in causal presentation order."""

        return self._events

    @property
    def query_events(self) -> tuple[bytes, ...]:
        """Complete query hierarchy."""

        return self._query_events

    @property
    def document_events(self) -> tuple[tuple[bytes, ...], ...]:
        """Complete document hierarchies, aligned to presentation order."""

        return self._document_events

    @property
    def query_sentences(self) -> tuple[OpaqueSentenceOccurrence, ...]:
        return self._query_sentences

    @property
    def document_sentences(self) -> tuple[tuple[OpaqueSentenceOccurrence, ...], ...]:
        return self._document_sentences

    @property
    def query_token_occurrences(self) -> tuple[OpaqueTokenOccurrence, ...]:
        return self._query_token_occurrences

    @property
    def document_token_occurrences(self) -> tuple[tuple[OpaqueTokenOccurrence, ...], ...]:
        return self._document_token_occurrences

    @property
    def token_occurrences(self) -> tuple[OpaqueTokenOccurrence, ...]:
        """Every current lexical occurrence, retaining identity and position."""

        return (
            *self._query_token_occurrences,
            *(token for document in self._document_token_occurrences for token in document),
        )

    @property
    def stable_hash(self) -> bytes:
        """A process-independent SHA-256 digest of the complete event sequence."""

        return self._stable_hash

    def reconstruct_query(self) -> bytes:
        """Decode the exact query only from its event hierarchy."""

        return _decode_stream_events(
            self._query_events,
            start_event=_QUERY_START,
            end_event=_QUERY_END,
        )

    def reconstruct_documents(self) -> tuple[OpaqueDocument, ...]:
        """Decode every exact document and identity in presentation order."""

        return tuple(_decode_document_events(events) for events in self._document_events)
