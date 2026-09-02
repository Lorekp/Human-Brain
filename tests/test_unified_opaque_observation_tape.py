from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import signature

import pytest
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.opaque_document_event_tape import OpaqueDocument, OpaqueDocumentEventTape
from humanbrain.unified_opaque_observation_tape import (
    OpaqueToolEventOccurrence,
    UnifiedOpaqueObservationTape,
)


def _segment_start(
    events: tuple[bytes, ...],
    segment: tuple[bytes, ...],
    start: int,
) -> int:
    return next(
        index
        for index in range(start, len(events) - len(segment) + 1)
        if events[index : index + len(segment)] == segment
    )


def test_unified_tape_is_lossless_ordered_distinct_and_domain_neutral() -> None:
    request = b"Trace Echo twice?\r\n\x00"
    documents = (
        OpaqueDocument(b"doc-z", b"source-9", b"Echo meets Echo.\n"),
        OpaqueDocument(b"doc-a", b"source-2", b"Echo meets Echo.\n"),
    )
    files = (
        OpaqueWorkspaceFile("zeta/repeated.bin", b"Echo\x00\n"),
        OpaqueWorkspaceFile("alpha/repeated.bin", b"Echo\x00\n"),
    )
    tool_events = (b"Echo status\x00", b"", b"Echo status\x00")

    tape = UnifiedOpaqueObservationTape.observe(
        request,
        documents,
        files,
        tool_events,
    )
    repeated = UnifiedOpaqueObservationTape.observe(
        request,
        documents,
        files,
        tool_events,
    )

    assert tape.reconstruct_request() == request
    assert tape.reconstruct_documents() == documents
    assert tape.reconstruct_workspace() == files
    assert tape.reconstruct_tool_events() == tool_events
    assert tape.events == repeated.events
    assert tape.stable_hash == repeated.stable_hash
    assert len(tape.stable_hash) == 32

    document_tape = OpaqueDocumentEventTape.observe(request, documents)
    workspace_tape = MultiscaleOpaqueEventTape.observe(request, files)
    assert tape.request_events == document_tape.query_events
    assert tape.document_events == document_tape.document_events
    assert tape.workspace_events == workspace_tape.source_events
    assert tape.workspace_motor_source_tokens == workspace_tape.motor_source_tokens
    assert tuple(
        (token.document_index, token.document_id, token.provenance_id)
        for token in tape.document_token_occurrences[1]
    ) == ((1, b"doc-a", b"source-2"),) * 3

    echo_occurrences = tuple(
        token
        for stream in tape.document_token_occurrences
        for token in stream
        if token.identity_event == b"Echo"
    )
    assert len(echo_occurrences) == 4
    assert len({token.occurrence_event for token in echo_occurrences}) == 4
    assert tuple(item.identity_event for item in tape.tool_event_occurrences) == tool_events
    assert len({item.occurrence_event for item in tape.tool_event_occurrences}) == 3

    flattened_documents = tuple(
        event for stream in tape.document_events for event in stream
    )
    modality_segments = (
        tape.request_events,
        flattened_documents,
        tape.workspace_events,
        tuple(item.occurrence_event for item in tape.tool_event_occurrences),
    )
    cursor = 0
    for segment in modality_segments:
        cursor = _segment_start(tape.events, segment, cursor) + len(segment)

    reordered_files = UnifiedOpaqueObservationTape.observe(
        request,
        documents,
        tuple(reversed(files)),
        tool_events,
    )
    assert reordered_files.reconstruct_workspace() == tuple(reversed(files))
    assert reordered_files.workspace_events == tape.workspace_events
    assert reordered_files.stable_hash != tape.stable_hash

    empty = UnifiedOpaqueObservationTape.observe(b"request only", (), (), ())
    assert empty.reconstruct_documents() == ()
    assert empty.reconstruct_workspace() == ()
    assert empty.reconstruct_tool_events() == ()
    assert empty.request_events == OpaqueDocumentEventTape.observe(
        b"request only", ()
    ).query_events

    assert tuple(signature(UnifiedOpaqueObservationTape.observe).parameters) == (
        "request",
        "documents",
        "files",
        "tool_events",
    )
    assert tuple(field.name for field in fields(OpaqueToolEventOccurrence)) == (
        "ordinal",
        "identity_event",
        "occurrence_event",
    )
    forbidden = ("task_kind", "relation", "answer", "target", "candidate", "desired")
    public_names = tuple(name for name in dir(tape) if not name.startswith("_"))
    assert all(term not in name for name in public_names for term in forbidden)
    with pytest.raises(FrozenInstanceError):
        tape._events = ()  # type: ignore[misc]
