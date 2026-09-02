from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import signature

import pytest
from humanbrain.opaque_document_event_tape import (
    OpaqueDocument,
    OpaqueDocumentEventTape,
    OpaqueSentenceOccurrence,
    OpaqueTokenOccurrence,
)


def _documents() -> tuple[OpaqueDocument, ...]:
    return (
        OpaqueDocument(
            document_id=b"brief-b",
            provenance_id=b"observatory-7",
            content=b"Nara mentors Elio.\r\nElio studies quartz!\x00tail",
        ),
        OpaqueDocument(
            document_id=b"brief-a",
            provenance_id=b"field-notes-2",
            content=b"Quartz is stable?  Yes.\n",
        ),
    )


def test_round_trip_preserves_exact_bytes_identity_provenance_and_order() -> None:
    query = b"What does Nara's mentee study?\r\n"
    documents = _documents()

    tape = OpaqueDocumentEventTape.observe(query, documents)

    assert tape.reconstruct_query() == query
    assert tape.reconstruct_documents() == documents
    assert tuple(sentence.reconstruct() for sentence in tape.query_sentences) == (query,)
    assert tuple(
        sentence.reconstruct() for sentence in tape.document_sentences[0]
    ) == (b"Nara mentors Elio.\r\n", b"Elio studies quartz!", b"\x00tail")
    assert tuple(
        (token.document_index, token.document_id, token.provenance_id)
        for token in tape.document_token_occurrences[1]
    ) == ((1, b"brief-a", b"field-notes-2"),) * 4


def test_sentence_and_token_positions_are_hierarchical_and_byte_exact() -> None:
    document = OpaqueDocument(b"d-1", b"p-1", b"Aster sees Comet. Comet moves.")
    tape = OpaqueDocumentEventTape.observe(b"Who sees Comet?", (document,))

    first, second = tape.document_sentences[0]
    assert (first.sentence_index, first.byte_start, first.byte_end) == (0, 0, 18)
    assert (second.sentence_index, second.byte_start, second.byte_end) == (1, 18, 30)
    assert first.reconstruct() == document.content[first.byte_start : first.byte_end]
    assert second.reconstruct() == document.content[second.byte_start : second.byte_end]
    assert tuple(token.identity_event for token in first.token_occurrences) == (
        b"Aster",
        b"sees",
        b"Comet",
    )
    assert tuple(
        document.content[token.byte_start : token.byte_end]
        for token in tape.document_token_occurrences[0]
    ) == tuple(
        token.identity_event for token in tape.document_token_occurrences[0]
    )


def test_repeated_lexical_values_remain_distinct_live_occurrences() -> None:
    tape = OpaqueDocumentEventTape.observe(
        b"Where is Echo?",
        (OpaqueDocument(b"d", b"p", b"Echo sees Echo. Echo rests."),),
    )

    repeated = tuple(
        token
        for token in tape.document_token_occurrences[0]
        if token.identity_event == b"Echo"
    )
    assert len(repeated) == 3
    assert len({token.occurrence_event for token in repeated}) == 3
    assert tuple(token.token_index for token in repeated) == (0, 2, 3)
    assert all(token in tape.token_occurrences for token in repeated)


def test_complete_event_sequence_has_a_deterministic_stable_hash() -> None:
    query = b"What does Nara study?"
    documents = _documents()

    first = OpaqueDocumentEventTape.observe(query, documents)
    repeated = OpaqueDocumentEventTape.observe(query, documents)
    reordered = OpaqueDocumentEventTape.observe(query, tuple(reversed(documents)))

    assert first.events == repeated.events
    assert first.stable_hash == repeated.stable_hash
    assert len(first.stable_hash) == 32
    assert first.stable_hash != reordered.stable_hash


def test_public_observation_surface_contains_only_raw_inputs_and_generic_views() -> None:
    assert tuple(field.name for field in fields(OpaqueDocument)) == (
        "document_id",
        "provenance_id",
        "content",
    )
    assert tuple(signature(OpaqueDocumentEventTape.observe).parameters) == (
        "query",
        "documents",
    )
    public_fields = {
        *(field.name for field in fields(OpaqueDocument)),
        *(field.name for field in fields(OpaqueSentenceOccurrence)),
        *(field.name for field in fields(OpaqueTokenOccurrence)),
    }
    forbidden = {"answer", "target", "relation", "predicate", "file", "path", "edit"}
    assert all(term not in name for name in public_fields for term in forbidden)

    document = OpaqueDocument(b"d", b"p", b"immutable")
    with pytest.raises(FrozenInstanceError):
        document.content = b"changed"  # type: ignore[misc]
