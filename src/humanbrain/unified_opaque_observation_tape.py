"""Lossless composition of opaque request, document, workspace, and tool bytes.

The established document and workspace tapes retain their generic lexical,
sentence, path, line, provenance, and motor views.  This layer adds only fixed
modality boundaries and ordinal occurrence frames.  It is a sensory adapter,
not intelligence: it does not interpret bytes, learn, select, or decide.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Self

from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.opaque_document_event_tape import (
    OpaqueDocument,
    OpaqueDocumentEventTape,
    OpaqueSentenceOccurrence,
    OpaqueTokenOccurrence,
)

__all__ = [
    "OpaqueToolEventOccurrence",
    "UnifiedOpaqueObservationTape",
]


_LENGTH_BYTES = 8

_OBSERVATION_START = b"\x00unified-opaque-observation-tape/observation-start/v1"
_OBSERVATION_END = b"\x00unified-opaque-observation-tape/observation-end/v1"
_REQUEST_START = b"\x00unified-opaque-observation-tape/request-start/v1"
_REQUEST_END = b"\x00unified-opaque-observation-tape/request-end/v1"
_DOCUMENTS_START = b"\x00unified-opaque-observation-tape/documents-start/v1"
_DOCUMENTS_END = b"\x00unified-opaque-observation-tape/documents-end/v1"
_WORKSPACE_START = b"\x00unified-opaque-observation-tape/workspace-start/v1"
_WORKSPACE_END = b"\x00unified-opaque-observation-tape/workspace-end/v1"
_TOOL_EVENTS_START = b"\x00unified-opaque-observation-tape/tool-events-start/v1"
_TOOL_EVENTS_END = b"\x00unified-opaque-observation-tape/tool-events-end/v1"

_WORKSPACE_OCCURRENCE = b"\x00unified-opaque-observation-tape/workspace-occurrence/v1\x00"
_TOOL_EVENT_OCCURRENCE = b"\x00unified-opaque-observation-tape/tool-occurrence/v1\x00"
_CONSTRUCTION_TOKEN = object()


def _frame(prefix: bytes, payload: bytes) -> bytes:
    return prefix + len(payload).to_bytes(_LENGTH_BYTES, "big") + payload


def _ordinal_event(prefix: bytes, ordinal: int, payload: bytes) -> bytes:
    return _frame(prefix, ordinal.to_bytes(_LENGTH_BYTES, "big") + payload)


def _stable_event_hash(events: tuple[bytes, ...]) -> bytes:
    digest = hashlib.sha256()
    for event in events:
        digest.update(len(event).to_bytes(_LENGTH_BYTES, "big"))
        digest.update(event)
    return digest.digest()


@dataclass(frozen=True, slots=True)
class OpaqueToolEventOccurrence:
    """One ordered raw tool event; equal payloads remain distinct occurrences."""

    ordinal: int
    identity_event: bytes
    occurrence_event: bytes


@dataclass(frozen=True, slots=True, init=False)
class UnifiedOpaqueObservationTape:
    """One immutable modality-complete opaque sensory observation."""

    _request: bytes
    _documents: tuple[OpaqueDocument, ...]
    _files: tuple[OpaqueWorkspaceFile, ...]
    _events: tuple[bytes, ...]
    _request_events: tuple[bytes, ...]
    _document_events: tuple[tuple[bytes, ...], ...]
    _workspace_events: tuple[bytes, ...]
    _tool_event_occurrences: tuple[OpaqueToolEventOccurrence, ...]
    _request_sentences: tuple[OpaqueSentenceOccurrence, ...]
    _document_sentences: tuple[tuple[OpaqueSentenceOccurrence, ...], ...]
    _request_token_occurrences: tuple[OpaqueTokenOccurrence, ...]
    _document_token_occurrences: tuple[tuple[OpaqueTokenOccurrence, ...], ...]
    _workspace_motor_source_tokens: tuple[bytes, ...]
    _stable_hash: bytes

    def __init__(
        self,
        token: object,
        *,
        request: bytes,
        documents: tuple[OpaqueDocument, ...],
        files: tuple[OpaqueWorkspaceFile, ...],
        events: tuple[bytes, ...],
        request_events: tuple[bytes, ...],
        document_events: tuple[tuple[bytes, ...], ...],
        workspace_events: tuple[bytes, ...],
        tool_event_occurrences: tuple[OpaqueToolEventOccurrence, ...],
        request_sentences: tuple[OpaqueSentenceOccurrence, ...],
        document_sentences: tuple[tuple[OpaqueSentenceOccurrence, ...], ...],
        request_token_occurrences: tuple[OpaqueTokenOccurrence, ...],
        document_token_occurrences: tuple[tuple[OpaqueTokenOccurrence, ...], ...],
        workspace_motor_source_tokens: tuple[bytes, ...],
        stable_hash: bytes,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use UnifiedOpaqueObservationTape.observe")
        object.__setattr__(self, "_request", request)
        object.__setattr__(self, "_documents", documents)
        object.__setattr__(self, "_files", files)
        object.__setattr__(self, "_events", events)
        object.__setattr__(self, "_request_events", request_events)
        object.__setattr__(self, "_document_events", document_events)
        object.__setattr__(self, "_workspace_events", workspace_events)
        object.__setattr__(self, "_tool_event_occurrences", tool_event_occurrences)
        object.__setattr__(self, "_request_sentences", request_sentences)
        object.__setattr__(self, "_document_sentences", document_sentences)
        object.__setattr__(self, "_request_token_occurrences", request_token_occurrences)
        object.__setattr__(
            self,
            "_document_token_occurrences",
            document_token_occurrences,
        )
        object.__setattr__(
            self,
            "_workspace_motor_source_tokens",
            workspace_motor_source_tokens,
        )
        object.__setattr__(self, "_stable_hash", stable_hash)

    @classmethod
    def observe(
        cls,
        request: bytes,
        documents: tuple[OpaqueDocument, ...],
        files: tuple[OpaqueWorkspaceFile, ...],
        tool_events: tuple[bytes, ...] = (),
    ) -> Self:
        """Compose complete raw inputs without adding interpreted structure."""

        if type(documents) is not tuple:
            raise TypeError("documents must be a tuple")
        if type(files) is not tuple:
            raise TypeError("files must be a tuple")
        if type(tool_events) is not tuple:
            raise TypeError("tool_events must be a tuple")
        if any(not isinstance(event, bytes) for event in tool_events):
            raise TypeError("tool_events must contain bytes")

        document_tape = OpaqueDocumentEventTape.observe(request, documents)
        if files:
            workspace_tape = MultiscaleOpaqueEventTape.observe(request, files)
            workspace_events = workspace_tape.source_events
            workspace_motor_source_tokens = workspace_tape.motor_source_tokens
            canonical_files = workspace_tape.reconstruct_workspace()
            if sorted(files, key=lambda item: item.path) != list(canonical_files):
                raise RuntimeError("workspace tape ceased to preserve exact files")
        else:
            if any(not isinstance(item, OpaqueWorkspaceFile) for item in files):
                raise TypeError("files must contain OpaqueWorkspaceFile values")
            workspace_events = ()
            workspace_motor_source_tokens = ()

        workspace_occurrences = tuple(
            _ordinal_event(_WORKSPACE_OCCURRENCE, ordinal, item.path.encode("utf-8"))
            for ordinal, item in enumerate(files)
        )
        tool_event_occurrences = tuple(
            OpaqueToolEventOccurrence(
                ordinal=ordinal,
                identity_event=event,
                occurrence_event=_ordinal_event(_TOOL_EVENT_OCCURRENCE, ordinal, event),
            )
            for ordinal, event in enumerate(tool_events)
        )
        events = (
            _OBSERVATION_START,
            _REQUEST_START,
            *document_tape.query_events,
            _REQUEST_END,
            _DOCUMENTS_START,
            *(event for stream in document_tape.document_events for event in stream),
            _DOCUMENTS_END,
            _WORKSPACE_START,
            *workspace_occurrences,
            *workspace_events,
            _WORKSPACE_END,
            _TOOL_EVENTS_START,
            *(item.occurrence_event for item in tool_event_occurrences),
            _TOOL_EVENTS_END,
            _OBSERVATION_END,
        )
        observation = cls(
            _CONSTRUCTION_TOKEN,
            request=request,
            documents=documents,
            files=files,
            events=events,
            request_events=document_tape.query_events,
            document_events=document_tape.document_events,
            workspace_events=workspace_events,
            tool_event_occurrences=tool_event_occurrences,
            request_sentences=document_tape.query_sentences,
            document_sentences=document_tape.document_sentences,
            request_token_occurrences=document_tape.query_token_occurrences,
            document_token_occurrences=document_tape.document_token_occurrences,
            workspace_motor_source_tokens=workspace_motor_source_tokens,
            stable_hash=_stable_event_hash(events),
        )
        if observation.reconstruct_request() != request:
            raise RuntimeError("request bytes changed during composition")
        if observation.reconstruct_documents() != documents:
            raise RuntimeError("document order changed during composition")
        if observation.reconstruct_workspace() != files:
            raise RuntimeError("workspace order changed during composition")
        if observation.reconstruct_tool_events() != tool_events:
            raise RuntimeError("tool-event order changed during composition")
        return observation

    @property
    def events(self) -> tuple[bytes, ...]:
        """Complete events in fixed modality and presentation order."""

        return self._events

    @property
    def request_events(self) -> tuple[bytes, ...]:
        return self._request_events

    @property
    def document_events(self) -> tuple[tuple[bytes, ...], ...]:
        return self._document_events

    @property
    def workspace_events(self) -> tuple[bytes, ...]:
        return self._workspace_events

    @property
    def tool_event_occurrences(self) -> tuple[OpaqueToolEventOccurrence, ...]:
        return self._tool_event_occurrences

    @property
    def request_sentences(self) -> tuple[OpaqueSentenceOccurrence, ...]:
        return self._request_sentences

    @property
    def document_sentences(self) -> tuple[tuple[OpaqueSentenceOccurrence, ...], ...]:
        return self._document_sentences

    @property
    def request_token_occurrences(self) -> tuple[OpaqueTokenOccurrence, ...]:
        return self._request_token_occurrences

    @property
    def document_token_occurrences(self) -> tuple[tuple[OpaqueTokenOccurrence, ...], ...]:
        return self._document_token_occurrences

    @property
    def token_occurrences(self) -> tuple[OpaqueTokenOccurrence, ...]:
        return (
            *self._request_token_occurrences,
            *(token for stream in self._document_token_occurrences for token in stream),
        )

    @property
    def workspace_motor_source_tokens(self) -> tuple[bytes, ...]:
        return self._workspace_motor_source_tokens

    @property
    def stable_hash(self) -> bytes:
        """Process-independent SHA-256 of the complete ordered event tape."""

        return self._stable_hash

    def reconstruct_request(self) -> bytes:
        return self._request

    def reconstruct_documents(self) -> tuple[OpaqueDocument, ...]:
        return self._documents

    def reconstruct_workspace(self) -> tuple[OpaqueWorkspaceFile, ...]:
        return self._files

    def reconstruct_tool_events(self) -> tuple[bytes, ...]:
        return tuple(item.identity_event for item in self._tool_event_occurrences)
