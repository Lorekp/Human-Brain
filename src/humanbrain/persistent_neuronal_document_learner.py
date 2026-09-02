"""Replay-free continual growth for one-hop opaque relational reading.

The learner accepts one :class:`RelationalReadingSuccess` at a time.  An
answer-supervised host compiler locates one exact one-hop training path and
retains only its literal query/document surfaces plus opaque pointer and
provenance digests.  Raw queries, documents, subjects, answers, and training
rows are not retained.

An exact surface is promoted only after two independently named filler pairs
from two independent success and document-provenance contexts.  Frozen
inference then enumerates all live token pairs, but fitted simulated-cell
layers perform every query-frame match, document-frame match, subject join,
candidate competition, and motor verification.  Missing or ambiguous evidence
loses to a tonic STOP cell.

This is a narrow answer-supervised exact-surface mechanism.  It is not
paraphrase induction, document-only learning, or broad language understanding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256

from humanbrain import neuronal_relational_reader as _reader
from humanbrain.neuronal_relational_reader import (
    RelationalRaceTrace,
    RelationalReadingSuccess,
)
from humanbrain.opaque_document_event_tape import (
    OpaqueDocument,
    OpaqueDocumentEventTape,
    OpaqueTokenOccurrence,
)
from humanbrain.sparse_neuronal_role_filler import (
    SparseBlockPointer,
    SparseNeuronalRoleFiller,
)

__all__ = [
    "CLAIM_BOUNDARY",
    "FrozenPersistentNeuronalDocumentLearner",
    "PersistentDocumentAblation",
    "PersistentDocumentDecision",
    "PersistentEvidenceChunk",
    "PersistentLearningSummary",
    "PersistentLearningTransition",
    "PersistentNeuronalDocumentLearnerError",
    "advance",
    "start",
]


CLAIM_BOUNDARY = (
    "answer-supervised exact-surface one-hop continual learning; host training "
    "path compilation and sensory projection are scaffolding"
)

_MIN_PROMOTION_SUPPORT = 2
_MAX_SCHEMAS = 64
_MAX_EVIDENCE_CHUNKS = 256
_CONSTRUCTION_TOKEN = object()
_SUBSTRATE = SparseNeuronalRoleFiller()

_CHECKPOINT_NAMESPACE = b"humanbrain.persistent-document.checkpoint.v1\0"
_STATE_NAMESPACE = b"humanbrain.persistent-document.state.v1\0"
_SCHEMA_KEY_NAMESPACE = b"humanbrain.persistent-document.schema-key.v1\0"
_SCHEMA_NAMESPACE = b"humanbrain.persistent-document.schema.v1\0"
_CHUNK_NAMESPACE = b"humanbrain.persistent-document.chunk.v1\0"
_QUERY_NAMESPACE = b"humanbrain.persistent-document.query.v1\0"
_DECISION_NAMESPACE = b"humanbrain.persistent-document.decision.v1\0"
_FILLER_CONTEXT_NAMESPACE = b"humanbrain.persistent-document.filler-context.v1\0"
_SOURCE_CONTEXT_NAMESPACE = b"humanbrain.persistent-document.source-context.v1\0"


class PersistentNeuronalDocumentLearnerError(ValueError):
    """The raw success cannot provide one exact one-hop training observation."""


class PersistentDocumentAblation(StrEnum):
    NONE = "none"
    QUERY_FRAME_OFF = "query_frame_off"
    DOCUMENT_FRAME_OFF = "document_frame_off"
    SUBJECT_JOIN_OFF = "subject_join_off"
    ANSWER_SELECTION_OFF = "answer_selection_off"
    MOTOR_OFF = "motor_off"


@dataclass(frozen=True, slots=True)
class PersistentEvidenceChunk:
    """One immutable distilled episode delta with no raw training row."""

    parent_checkpoint_sha256: str
    schema_key_sha256: str
    success_provenance_sha256: str
    document_provenance_sha256: str
    document_id_sha256: str
    subject_sha256: str
    answer_sha256: str
    subject_occurrence_sha256: str
    answer_occurrence_sha256: str
    filler_context_pointer: SparseBlockPointer
    source_context_pointer: SparseBlockPointer
    binding_trace_sha256s: tuple[str, str]
    learning_cell_steps: int
    learning_active_synapses: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PersistentLearningSummary:
    accepted_episode_count: int
    delta_chunk_count: int
    schema_count: int
    pending_schema_count: int
    promoted_schema_count: int
    promotion_support: int
    retained_raw_training_row_count: int
    retained_raw_document_count: int
    parent_episode_replay_count: int
    retained_surface_literal_byte_count: int
    retained_canonical_state_byte_count: int
    retained_pointer_count: int
    retained_active_cell_count: int
    cumulative_binding_update_count: int
    cumulative_learning_cell_steps: int
    cumulative_learning_active_synapses: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class PersistentDocumentDecision:
    decision: str
    answer: bytes | None
    answer_occurrence_event: bytes | None
    answer_document_id: bytes | None
    answer_provenance_id: bytes | None
    schema_sha256: str | None
    completed_hops: int
    stop_stage: str | None
    stop_reason: str | None
    races: tuple[RelationalRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PersistentLearningTransition:
    status: str
    updated: bool
    promoted_schema: bool
    parent_checkpoint_sha256: str
    successor_checkpoint_sha256: str
    schema_key_sha256: str | None
    schema_support_before: int
    schema_support_after: int
    delta_chunks_added: int
    current_episode_count: int
    parent_episode_replay_count: int
    parent_mutation_count: int
    retained_raw_training_row_count: int
    learning_cell_steps: int
    learning_active_synapses: int
    successor: FrozenPersistentNeuronalDocumentLearner


@dataclass(frozen=True, slots=True)
class _QueryFrame:
    sentence_lengths: tuple[int, ...]
    subject_flat_index: int
    literal_events: tuple[bytes | None, ...]


@dataclass(frozen=True, slots=True)
class _DocumentFrame:
    token_count: int
    source_token_index: int
    destination_token_index: int
    literal_events: tuple[bytes | None, ...]


@dataclass(frozen=True, slots=True)
class _SurfaceSchema:
    key_sha256: str
    query_frame: _QueryFrame
    document_frame: _DocumentFrame
    relation_pointer: SparseBlockPointer
    subject_sha256s: tuple[str, ...]
    answer_sha256s: tuple[str, ...]
    success_provenance_sha256s: tuple[str, ...]
    document_id_sha256s: tuple[str, ...]
    document_provenance_sha256s: tuple[str, ...]
    filler_context_pointer_sha256s: tuple[str, ...]
    source_context_pointer_sha256s: tuple[str, ...]
    sha256: str

    @property
    def support_ticks(self) -> int:
        return len(self.success_provenance_sha256s)

    @property
    def promoted(self) -> bool:
        return self.support_ticks >= _MIN_PROMOTION_SUPPORT


@dataclass(frozen=True, slots=True)
class _CompiledEvidence:
    query_frame: _QueryFrame
    document_frame: _DocumentFrame
    schema_key_sha256: str
    subject: bytes
    answer: bytes
    success_provenance: bytes
    document_provenance: bytes
    document_id: bytes
    subject_occurrence_event: bytes
    answer_occurrence_event: bytes


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{name} is not a canonical SHA-256 digest")


def _frame_events_record(events: tuple[bytes | None, ...]) -> tuple[str | None, ...]:
    return tuple(None if event is None else event.hex() for event in events)


def _query_frame_record(frame: _QueryFrame) -> dict[str, object]:
    return {
        "literal_events_hex": _frame_events_record(frame.literal_events),
        "sentence_lengths": frame.sentence_lengths,
        "subject_flat_index": frame.subject_flat_index,
    }


def _document_frame_record(frame: _DocumentFrame) -> dict[str, object]:
    return {
        "destination_token_index": frame.destination_token_index,
        "literal_events_hex": _frame_events_record(frame.literal_events),
        "source_token_index": frame.source_token_index,
        "token_count": frame.token_count,
    }


def _pointer_record(pointer: SparseBlockPointer) -> dict[str, object]:
    return {
        "block_cells": pointer.block_cells,
        "sha256": pointer.sha256,
    }


def _schema_key(query_frame: _QueryFrame, document_frame: _DocumentFrame) -> str:
    return _digest(
        _SCHEMA_KEY_NAMESPACE,
        {
            "document_frame": _document_frame_record(document_frame),
            "query_frame": _query_frame_record(query_frame),
        },
    )


def _schema_record(schema: _SurfaceSchema, *, include_sha256: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "answer_sha256s": schema.answer_sha256s,
        "document_frame": _document_frame_record(schema.document_frame),
        "document_id_sha256s": schema.document_id_sha256s,
        "document_provenance_sha256s": schema.document_provenance_sha256s,
        "filler_context_pointer_sha256s": schema.filler_context_pointer_sha256s,
        "key_sha256": schema.key_sha256,
        "query_frame": _query_frame_record(schema.query_frame),
        "relation_pointer": _pointer_record(schema.relation_pointer),
        "source_context_pointer_sha256s": schema.source_context_pointer_sha256s,
        "subject_sha256s": schema.subject_sha256s,
        "success_provenance_sha256s": schema.success_provenance_sha256s,
    }
    if include_sha256:
        record["sha256"] = schema.sha256
    return record


def _chunk_record(chunk: PersistentEvidenceChunk, *, include_sha256: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "answer_occurrence_sha256": chunk.answer_occurrence_sha256,
        "answer_sha256": chunk.answer_sha256,
        "binding_trace_sha256s": chunk.binding_trace_sha256s,
        "document_id_sha256": chunk.document_id_sha256,
        "document_provenance_sha256": chunk.document_provenance_sha256,
        "filler_context_pointer": _pointer_record(chunk.filler_context_pointer),
        "learning_cell_steps": chunk.learning_cell_steps,
        "learning_active_synapses": chunk.learning_active_synapses,
        "parent_checkpoint_sha256": chunk.parent_checkpoint_sha256,
        "schema_key_sha256": chunk.schema_key_sha256,
        "source_context_pointer": _pointer_record(chunk.source_context_pointer),
        "subject_occurrence_sha256": chunk.subject_occurrence_sha256,
        "subject_sha256": chunk.subject_sha256,
        "success_provenance_sha256": chunk.success_provenance_sha256,
    }
    if include_sha256:
        record["sha256"] = chunk.sha256
    return record


def _make_schema(
    evidence: _CompiledEvidence,
    *,
    old: _SurfaceSchema | None,
    chunk: PersistentEvidenceChunk,
) -> _SurfaceSchema:
    if old is None:
        relation_pointer = _SUBSTRATE.project(bytes.fromhex(evidence.schema_key_sha256))
        subject_sha256s: tuple[str, ...] = ()
        answer_sha256s: tuple[str, ...] = ()
        success_provenance_sha256s: tuple[str, ...] = ()
        document_id_sha256s: tuple[str, ...] = ()
        document_provenance_sha256s: tuple[str, ...] = ()
        filler_context_pointer_sha256s: tuple[str, ...] = ()
        source_context_pointer_sha256s: tuple[str, ...] = ()
    else:
        relation_pointer = old.relation_pointer
        subject_sha256s = old.subject_sha256s
        answer_sha256s = old.answer_sha256s
        success_provenance_sha256s = old.success_provenance_sha256s
        document_id_sha256s = old.document_id_sha256s
        document_provenance_sha256s = old.document_provenance_sha256s
        filler_context_pointer_sha256s = old.filler_context_pointer_sha256s
        source_context_pointer_sha256s = old.source_context_pointer_sha256s

    provisional = _SurfaceSchema(
        key_sha256=evidence.schema_key_sha256,
        query_frame=evidence.query_frame,
        document_frame=evidence.document_frame,
        relation_pointer=relation_pointer,
        subject_sha256s=(*subject_sha256s, chunk.subject_sha256),
        answer_sha256s=(*answer_sha256s, chunk.answer_sha256),
        success_provenance_sha256s=(
            *success_provenance_sha256s,
            chunk.success_provenance_sha256,
        ),
        document_id_sha256s=(*document_id_sha256s, chunk.document_id_sha256),
        document_provenance_sha256s=(
            *document_provenance_sha256s,
            chunk.document_provenance_sha256,
        ),
        filler_context_pointer_sha256s=(
            *filler_context_pointer_sha256s,
            chunk.filler_context_pointer.sha256,
        ),
        source_context_pointer_sha256s=(
            *source_context_pointer_sha256s,
            chunk.source_context_pointer.sha256,
        ),
        sha256="",
    )
    return _SurfaceSchema(
        key_sha256=provisional.key_sha256,
        query_frame=provisional.query_frame,
        document_frame=provisional.document_frame,
        relation_pointer=provisional.relation_pointer,
        subject_sha256s=provisional.subject_sha256s,
        answer_sha256s=provisional.answer_sha256s,
        success_provenance_sha256s=provisional.success_provenance_sha256s,
        document_id_sha256s=provisional.document_id_sha256s,
        document_provenance_sha256s=provisional.document_provenance_sha256s,
        filler_context_pointer_sha256s=(provisional.filler_context_pointer_sha256s),
        source_context_pointer_sha256s=provisional.source_context_pointer_sha256s,
        sha256=_digest(
            _SCHEMA_NAMESPACE,
            _schema_record(provisional, include_sha256=False),
        ),
    )


def _compile_success(success: RelationalReadingSuccess) -> _CompiledEvidence:
    tape = OpaqueDocumentEventTape.observe(success.query, success.documents)
    query_tokens = _reader._query_tokens(tape)
    if not query_tokens:
        raise PersistentNeuronalDocumentLearnerError(
            "the query must contain at least one lexical occurrence"
        )
    edges = _reader._all_edges(tape)
    candidates = tuple(
        (query_index, query_token, edge)
        for query_index, query_token in enumerate(query_tokens)
        for edge in edges
        if query_token.identity_event == edge.source.identity_event
        and edge.destination.identity_event == success.observed_answer
    )
    if len(candidates) != 1:
        raise PersistentNeuronalDocumentLearnerError(
            "each success must expose one unique answer-supervised one-hop path"
        )
    subject_index, subject, edge = candidates[0]
    if edge.source.provenance_id is None or edge.source.document_id is None:
        raise RuntimeError("training edge lost its document provenance")
    query_frame = _QueryFrame(
        sentence_lengths=_reader._query_sentence_lengths(tape),
        subject_flat_index=subject_index,
        literal_events=tuple(
            None if index == subject_index else token.identity_event
            for index, token in enumerate(query_tokens)
        ),
    )
    source_index, destination_index, literal_events = _reader._surface_key(edge)
    document_frame = _DocumentFrame(
        token_count=len(edge.sentence_tokens),
        source_token_index=source_index,
        destination_token_index=destination_index,
        literal_events=literal_events,
    )
    endpoints = {subject.identity_event, edge.destination.identity_event}
    retained_literals = {
        event
        for event in (*query_frame.literal_events, *document_frame.literal_events)
        if event is not None
    }
    if endpoints & retained_literals:
        raise PersistentNeuronalDocumentLearnerError(
            "a learned literal aliases a concrete training endpoint"
        )
    return _CompiledEvidence(
        query_frame=query_frame,
        document_frame=document_frame,
        schema_key_sha256=_schema_key(query_frame, document_frame),
        subject=subject.identity_event,
        answer=edge.destination.identity_event,
        success_provenance=success.provenance_id,
        document_provenance=edge.source.provenance_id,
        document_id=edge.source.document_id,
        subject_occurrence_event=edge.source.occurrence_event,
        answer_occurrence_event=edge.destination.occurrence_event,
    )


def _project_digest(namespace: bytes, value: bytes) -> SparseBlockPointer:
    digest = sha256(namespace + value).digest()
    return _SUBSTRATE.project(digest)


def _make_chunk(
    *,
    parent_checkpoint_sha256: str,
    evidence: _CompiledEvidence,
) -> PersistentEvidenceChunk:
    subject_pointer = _project_digest(_FILLER_CONTEXT_NAMESPACE, evidence.subject)
    answer_pointer = _project_digest(_FILLER_CONTEXT_NAMESPACE, evidence.answer)
    success_pointer = _project_digest(
        _SOURCE_CONTEXT_NAMESPACE,
        evidence.success_provenance,
    )
    document_pointer = _project_digest(
        _SOURCE_CONTEXT_NAMESPACE,
        evidence.document_provenance,
    )
    filler_binding = _SUBSTRATE.bind(subject_pointer, answer_pointer)
    source_binding = _SUBSTRATE.bind(success_pointer, document_pointer)
    if filler_binding.pointer is None or source_binding.pointer is None:
        raise RuntimeError("unablated context binding selected tonic STOP")
    provisional = PersistentEvidenceChunk(
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        schema_key_sha256=evidence.schema_key_sha256,
        success_provenance_sha256=_bytes_sha256(evidence.success_provenance),
        document_provenance_sha256=_bytes_sha256(evidence.document_provenance),
        document_id_sha256=_bytes_sha256(evidence.document_id),
        subject_sha256=_bytes_sha256(evidence.subject),
        answer_sha256=_bytes_sha256(evidence.answer),
        subject_occurrence_sha256=_bytes_sha256(evidence.subject_occurrence_event),
        answer_occurrence_sha256=_bytes_sha256(evidence.answer_occurrence_event),
        filler_context_pointer=filler_binding.pointer,
        source_context_pointer=source_binding.pointer,
        binding_trace_sha256s=(
            filler_binding.trace.sha256,
            source_binding.trace.sha256,
        ),
        learning_cell_steps=(
            filler_binding.trace.physical_cell_steps + source_binding.trace.physical_cell_steps
        ),
        learning_active_synapses=(
            filler_binding.trace.active_synapses + source_binding.trace.active_synapses
        ),
        sha256="",
    )
    return PersistentEvidenceChunk(
        parent_checkpoint_sha256=provisional.parent_checkpoint_sha256,
        schema_key_sha256=provisional.schema_key_sha256,
        success_provenance_sha256=provisional.success_provenance_sha256,
        document_provenance_sha256=provisional.document_provenance_sha256,
        document_id_sha256=provisional.document_id_sha256,
        subject_sha256=provisional.subject_sha256,
        answer_sha256=provisional.answer_sha256,
        subject_occurrence_sha256=provisional.subject_occurrence_sha256,
        answer_occurrence_sha256=provisional.answer_occurrence_sha256,
        filler_context_pointer=provisional.filler_context_pointer,
        source_context_pointer=provisional.source_context_pointer,
        binding_trace_sha256s=provisional.binding_trace_sha256s,
        learning_cell_steps=provisional.learning_cell_steps,
        learning_active_synapses=provisional.learning_active_synapses,
        sha256=_digest(
            _CHUNK_NAMESPACE,
            _chunk_record(provisional, include_sha256=False),
        ),
    )


def _persistent_state_record(
    schemas: tuple[_SurfaceSchema, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
) -> dict[str, object]:
    return {
        "chunks": tuple(_chunk_record(chunk, include_sha256=True) for chunk in chunks),
        "schemas": tuple(_schema_record(schema, include_sha256=True) for schema in schemas),
    }


def _retained_raw_artifact_counts(
    schemas: tuple[_SurfaceSchema, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
) -> tuple[int, int]:
    def visit(value: object) -> tuple[int, int]:
        rows = int(type(value) is RelationalReadingSuccess)
        documents = int(type(value) in (OpaqueDocument, OpaqueDocumentEventTape))
        if type(value) is tuple:
            for item in value:
                child_rows, child_documents = visit(item)
                rows += child_rows
                documents += child_documents
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                child_rows, child_documents = visit(getattr(value, field.name))
                rows += child_rows
                documents += child_documents
        return rows, documents

    return visit((schemas, chunks))


def _retained_pointers(
    schemas: tuple[_SurfaceSchema, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
) -> tuple[SparseBlockPointer, ...]:
    return (
        *(schema.relation_pointer for schema in schemas),
        *(chunk.filler_context_pointer for chunk in chunks),
        *(chunk.source_context_pointer for chunk in chunks),
    )


def _retained_canonical_state_bytes(
    schemas: tuple[_SurfaceSchema, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
) -> int:
    return len(_canonical_json(_persistent_state_record(schemas, chunks)))


def _checkpoint_record(
    schemas: tuple[_SurfaceSchema, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
) -> dict[str, object]:
    state_record = _persistent_state_record(schemas, chunks)
    raw_rows, raw_documents = _retained_raw_artifact_counts(schemas, chunks)
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "chunk_sha256s": tuple(chunk.sha256 for chunk in chunks),
        "promotion_support": _MIN_PROMOTION_SUPPORT,
        "retained_canonical_state_byte_count": len(_canonical_json(state_record)),
        "retained_raw_document_count": raw_documents,
        "retained_raw_training_row_count": raw_rows,
        "schema_sha256s": tuple(schema.sha256 for schema in schemas),
        "state_sha256": _digest(_STATE_NAMESPACE, state_record),
    }


def _retained_surface_literal_bytes(schemas: tuple[_SurfaceSchema, ...]) -> int:
    return sum(
        len(event)
        for schema in schemas
        for event in (
            *schema.query_frame.literal_events,
            *schema.document_frame.literal_events,
        )
        if event is not None
    )


def _freeze(
    *,
    schemas: tuple[_SurfaceSchema, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
) -> FrozenPersistentNeuronalDocumentLearner:
    raw_rows, raw_documents = _retained_raw_artifact_counts(schemas, chunks)
    pointers = _retained_pointers(schemas, chunks)
    checkpoint_sha256 = _digest(
        _CHECKPOINT_NAMESPACE,
        _checkpoint_record(schemas, chunks),
    )
    promoted = sum(schema.promoted for schema in schemas)
    summary = PersistentLearningSummary(
        accepted_episode_count=len(chunks),
        delta_chunk_count=len(chunks),
        schema_count=len(schemas),
        pending_schema_count=len(schemas) - promoted,
        promoted_schema_count=promoted,
        promotion_support=_MIN_PROMOTION_SUPPORT,
        retained_raw_training_row_count=raw_rows,
        retained_raw_document_count=raw_documents,
        parent_episode_replay_count=0,
        retained_surface_literal_byte_count=_retained_surface_literal_bytes(schemas),
        retained_canonical_state_byte_count=_retained_canonical_state_bytes(
            schemas,
            chunks,
        ),
        retained_pointer_count=len(pointers),
        retained_active_cell_count=sum(pointer.active_cell_count for pointer in pointers),
        cumulative_binding_update_count=sum(len(chunk.binding_trace_sha256s) for chunk in chunks),
        cumulative_learning_cell_steps=sum(chunk.learning_cell_steps for chunk in chunks),
        cumulative_learning_active_synapses=sum(chunk.learning_active_synapses for chunk in chunks),
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenPersistentNeuronalDocumentLearner(
        _CONSTRUCTION_TOKEN,
        schemas=schemas,
        chunks=chunks,
        summary=summary,
        checkpoint_sha256=checkpoint_sha256,
    )


class FrozenPersistentNeuronalDocumentLearner:
    """Immutable exact-surface schemas plus structurally shared episode deltas."""

    __slots__ = ("_checkpoint_sha256", "_chunks", "_schemas", "_summary")

    def __init__(
        self,
        token: object,
        *,
        schemas: tuple[_SurfaceSchema, ...],
        chunks: tuple[PersistentEvidenceChunk, ...],
        summary: PersistentLearningSummary,
        checkpoint_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use start() and advance()")
        self._schemas = schemas
        self._chunks = chunks
        self._summary = summary
        self._checkpoint_sha256 = checkpoint_sha256
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> PersistentLearningSummary:
        self.authenticate()
        return self._summary

    @property
    def evidence_chunks(self) -> tuple[PersistentEvidenceChunk, ...]:
        self.authenticate()
        return self._chunks

    @property
    def promoted_schema_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return tuple(schema.sha256 for schema in self._schemas if schema.promoted)

    @property
    def schema_support_ticks(self) -> tuple[int, ...]:
        self.authenticate()
        return tuple(schema.support_ticks for schema in self._schemas)

    def authenticate(self) -> None:
        for chunk in self._chunks:
            _require_sha256(chunk.sha256, "chunk sha256")
            SparseBlockPointer(
                chunk.filler_context_pointer.block_cells,
                chunk.filler_context_pointer.sha256,
            )
            SparseBlockPointer(
                chunk.source_context_pointer.block_cells,
                chunk.source_context_pointer.sha256,
            )
            if (
                len(chunk.binding_trace_sha256s) != 2
                or chunk.learning_cell_steps <= 0
                or chunk.learning_active_synapses <= 0
            ):
                raise RuntimeError("persistent chunk lost physical binding accounting")
            if chunk.sha256 != _digest(
                _CHUNK_NAMESPACE,
                _chunk_record(chunk, include_sha256=False),
            ):
                raise RuntimeError("persistent evidence chunk identity changed")
        for schema in self._schemas:
            _require_sha256(schema.key_sha256, "schema key sha256")
            _require_sha256(schema.sha256, "schema sha256")
            SparseBlockPointer(
                schema.relation_pointer.block_cells,
                schema.relation_pointer.sha256,
            )
            if schema.key_sha256 != _schema_key(
                schema.query_frame,
                schema.document_frame,
            ):
                raise RuntimeError("persistent schema key changed")
            if schema.sha256 != _digest(
                _SCHEMA_NAMESPACE,
                _schema_record(schema, include_sha256=False),
            ):
                raise RuntimeError("persistent schema identity changed")
            support = schema.support_ticks
            context_lengths = {
                len(schema.subject_sha256s),
                len(schema.answer_sha256s),
                len(schema.success_provenance_sha256s),
                len(schema.document_id_sha256s),
                len(schema.document_provenance_sha256s),
                len(schema.filler_context_pointer_sha256s),
                len(schema.source_context_pointer_sha256s),
            }
            if context_lengths != {support}:
                raise RuntimeError("persistent schema support geometry changed")
            if any(
                len(set(values)) != support
                for values in (
                    schema.subject_sha256s,
                    schema.answer_sha256s,
                    schema.success_provenance_sha256s,
                    schema.document_id_sha256s,
                    schema.document_provenance_sha256s,
                    schema.filler_context_pointer_sha256s,
                    schema.source_context_pointer_sha256s,
                )
            ):
                raise RuntimeError("persistent schema lost independent contexts")
        expected_root = _digest(
            _CHECKPOINT_NAMESPACE,
            _checkpoint_record(self._schemas, self._chunks),
        )
        if expected_root != self._checkpoint_sha256:
            raise RuntimeError("persistent learner checkpoint authentication failed")
        raw_rows, raw_documents = _retained_raw_artifact_counts(
            self._schemas,
            self._chunks,
        )
        if raw_rows or raw_documents:
            raise RuntimeError("persistent learner retained a raw training artifact")
        pointers = _retained_pointers(self._schemas, self._chunks)
        promoted = sum(schema.promoted for schema in self._schemas)
        expected_summary = PersistentLearningSummary(
            accepted_episode_count=len(self._chunks),
            delta_chunk_count=len(self._chunks),
            schema_count=len(self._schemas),
            pending_schema_count=len(self._schemas) - promoted,
            promoted_schema_count=promoted,
            promotion_support=_MIN_PROMOTION_SUPPORT,
            retained_raw_training_row_count=raw_rows,
            retained_raw_document_count=raw_documents,
            parent_episode_replay_count=0,
            retained_surface_literal_byte_count=_retained_surface_literal_bytes(self._schemas),
            retained_canonical_state_byte_count=_retained_canonical_state_bytes(
                self._schemas,
                self._chunks,
            ),
            retained_pointer_count=len(pointers),
            retained_active_cell_count=sum(pointer.active_cell_count for pointer in pointers),
            cumulative_binding_update_count=sum(
                len(chunk.binding_trace_sha256s) for chunk in self._chunks
            ),
            cumulative_learning_cell_steps=sum(chunk.learning_cell_steps for chunk in self._chunks),
            cumulative_learning_active_synapses=sum(
                chunk.learning_active_synapses for chunk in self._chunks
            ),
            checkpoint_sha256=expected_root,
        )
        if self._summary != expected_summary:
            raise RuntimeError("persistent learner training summary changed")

    def respond(
        self,
        query: bytes,
        documents: tuple[OpaqueDocument, ...],
        ablation: PersistentDocumentAblation = PersistentDocumentAblation.NONE,
    ) -> PersistentDocumentDecision:
        """Execute one physical exact-surface read over the complete live tape."""

        self.authenticate()
        if not isinstance(query, bytes) or not query:
            raise ValueError("query must be non-empty bytes")
        if type(documents) is not tuple or not 1 <= len(documents) <= 4:
            raise ValueError("documents must contain one to four values")
        if any(not isinstance(document, OpaqueDocument) for document in documents):
            raise TypeError("documents must contain OpaqueDocument values")
        if not isinstance(ablation, PersistentDocumentAblation):
            raise TypeError("ablation must be PersistentDocumentAblation")

        tape = OpaqueDocumentEventTape.observe(query, documents)
        query_tokens = _reader._query_tokens(tape)
        races: list[RelationalRaceTrace] = []
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": self._checkpoint_sha256,
                "observation_sha256": tape.stable_hash.hex(),
            },
        )
        query_control = _reader._control_signal(
            population="persistent/query_frame_control",
            enabled=ablation is not PersistentDocumentAblation.QUERY_FRAME_OFF,
            races=races,
        )
        document_control = _reader._control_signal(
            population="persistent/document_frame_control",
            enabled=ablation is not PersistentDocumentAblation.DOCUMENT_FRAME_OFF,
            races=races,
        )
        subject_control = _reader._control_signal(
            population="persistent/subject_join_control",
            enabled=ablation is not PersistentDocumentAblation.SUBJECT_JOIN_OFF,
            races=races,
        )
        answer_control = _reader._control_signal(
            population="persistent/answer_selection_control",
            enabled=ablation is not PersistentDocumentAblation.ANSWER_SELECTION_OFF,
            races=races,
        )
        motor_control = _reader._control_signal(
            population="persistent/motor_control",
            enabled=ablation is not PersistentDocumentAblation.MOTOR_OFF,
            races=races,
        )
        schemas = tuple(schema for schema in self._schemas if schema.promoted)
        if not schemas:
            return self._empty_stop(
                query_sha256=query_sha256,
                races=races,
                stage="schema",
            )

        query_signals = tuple(
            self._query_frame_signal(
                schema,
                tape,
                query_tokens,
                ordinal=ordinal,
                control=query_control,
                races=races,
            )
            for ordinal, schema in enumerate(schemas)
        )
        any_query = _reader._disjunction_signal(
            query_signals,
            population="persistent/query_frames",
            races=races,
        )
        query_choice = _reader._physical_competition(
            population="persistent/query_frame_match",
            signals=(any_query,),
            races=races,
        )
        if query_choice.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                stage="query_frame",
                reason=query_choice.trace.stop_reason,
            )

        edges = _reader._all_edges(tape)
        if not edges:
            return self._empty_stop(
                query_sha256=query_sha256,
                races=races,
                stage="answer_selection",
            )
        candidates: list[tuple[_SurfaceSchema, _reader._EdgeCandidate]] = []
        signals: list[_reader._CellSignal] = []
        for schema_index, (schema, query_signal) in enumerate(
            zip(schemas, query_signals, strict=True)
        ):
            subject = (
                query_tokens[schema.query_frame.subject_flat_index].identity_event
                if schema.query_frame.subject_flat_index < len(query_tokens)
                else b""
            )
            for edge_index, edge in enumerate(edges):
                surface_signal = self._document_frame_signal(
                    schema,
                    edge,
                    schema_index=schema_index,
                    edge_index=edge_index,
                    control=document_control,
                    races=races,
                )
                equality_signal = _reader._event_equality_signal(
                    subject,
                    edge.source.identity_event,
                    population=(
                        f"persistent/schema_{schema_index}/edge_{edge_index}/subject_equality"
                    ),
                    races=races,
                )
                signals.append(
                    _reader._conjunction_signal(
                        (
                            query_signal,
                            surface_signal,
                            equality_signal,
                            subject_control,
                            answer_control,
                        ),
                        population=(
                            f"persistent/schema_{schema_index}/edge_{edge_index}/one_hop_join"
                        ),
                        races=races,
                    )
                )
                candidates.append((schema, edge))

        answer_choice = _reader._physical_competition(
            population="persistent/unique_answer_selection",
            signals=signals,
            races=races,
        )
        if answer_choice.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                stage="answer_selection",
                reason=answer_choice.trace.stop_reason,
            )
        schema, selected_edge = candidates[answer_choice.column]

        live_tokens = tuple(
            occurrence for document in tape.document_token_occurrences for occurrence in document
        )
        motor_signals: list[_reader._CellSignal] = []
        for ordinal, occurrence in enumerate(live_tokens):
            byte_equality = _reader._event_equality_signal(
                selected_edge.destination.identity_event,
                occurrence.identity_event,
                population=f"persistent/motor_{ordinal}/byte_equality",
                races=races,
            )
            document_equality = _reader._scalar_equality_signal(
                selected_edge.destination.document_index or 0,
                occurrence.document_index or 0,
                population=f"persistent/motor_{ordinal}/document_index_equality",
                races=races,
            )
            sentence_equality = _reader._scalar_equality_signal(
                selected_edge.destination.sentence_index,
                occurrence.sentence_index,
                population=f"persistent/motor_{ordinal}/sentence_index_equality",
                races=races,
            )
            token_equality = _reader._scalar_equality_signal(
                selected_edge.destination.token_index,
                occurrence.token_index,
                population=f"persistent/motor_{ordinal}/token_index_equality",
                races=races,
            )
            motor_signals.append(
                _reader._conjunction_signal(
                    (
                        byte_equality,
                        document_equality,
                        sentence_equality,
                        token_equality,
                        motor_control,
                    ),
                    population=f"persistent/motor_{ordinal}/verification",
                    races=races,
                )
            )
        motor = _reader._physical_competition(
            population="persistent/live_occurrence_motor",
            signals=motor_signals,
            races=races,
        )
        if motor.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                stage="motor",
                reason=motor.trace.stop_reason,
            )
        answer_occurrence = live_tokens[motor.column]
        return self._decision(
            decision="EMIT",
            answer_occurrence=answer_occurrence,
            schema_sha256=schema.sha256,
            completed_hops=1,
            stop_stage=None,
            stop_reason=None,
            races=races,
            query_sha256=query_sha256,
        )

    @staticmethod
    def _query_frame_signal(
        schema: _SurfaceSchema,
        tape: OpaqueDocumentEventTape,
        tokens: tuple[OpaqueTokenOccurrence, ...],
        *,
        ordinal: int,
        control: _reader._CellSignal,
        races: list[RelationalRaceTrace],
    ) -> _reader._CellSignal:
        frame = schema.query_frame
        prefix = f"persistent/schema_{ordinal}/query_frame"
        observed_sentence_lengths = _reader._query_sentence_lengths(tape)
        features: list[_reader._CellSignal] = [control]
        features.append(
            _reader._scalar_equality_signal(
                len(observed_sentence_lengths),
                len(frame.sentence_lengths),
                population=f"{prefix}/sentence_count_equality",
                races=races,
            )
        )
        for index, expected_length in enumerate(frame.sentence_lengths):
            observed_length = (
                observed_sentence_lengths[index] if index < len(observed_sentence_lengths) else 0
            )
            features.append(
                _reader._scalar_equality_signal(
                    observed_length,
                    expected_length,
                    population=f"{prefix}/sentence_{index}_length_equality",
                    races=races,
                )
            )
        features.append(
            _reader._scalar_equality_signal(
                len(tokens),
                len(frame.literal_events),
                population=f"{prefix}/token_count_equality",
                races=races,
            )
        )
        for index, expected in enumerate(frame.literal_events):
            if expected is None:
                continue
            observed = tokens[index].identity_event if index < len(tokens) else b""
            features.append(
                _reader._event_equality_signal(
                    observed,
                    expected,
                    population=f"{prefix}/literal_{index}",
                    races=races,
                )
            )
        return _reader._conjunction_signal(
            features,
            population=prefix,
            races=races,
        )

    @staticmethod
    def _document_frame_signal(
        schema: _SurfaceSchema,
        edge: _reader._EdgeCandidate,
        *,
        schema_index: int,
        edge_index: int,
        control: _reader._CellSignal,
        races: list[RelationalRaceTrace],
    ) -> _reader._CellSignal:
        frame = schema.document_frame
        prefix = f"persistent/schema_{schema_index}/edge_{edge_index}/document_frame"
        features: list[_reader._CellSignal] = [control]
        features.extend(
            (
                _reader._scalar_equality_signal(
                    len(edge.sentence_tokens),
                    frame.token_count,
                    population=f"{prefix}/token_count_equality",
                    races=races,
                ),
                _reader._scalar_equality_signal(
                    edge.source_token_index,
                    frame.source_token_index,
                    population=f"{prefix}/source_orientation_equality",
                    races=races,
                ),
                _reader._scalar_equality_signal(
                    edge.destination_token_index,
                    frame.destination_token_index,
                    population=f"{prefix}/destination_orientation_equality",
                    races=races,
                ),
            )
        )
        for index, expected in enumerate(frame.literal_events):
            if expected is None:
                continue
            observed = (
                edge.sentence_tokens[index].identity_event
                if index < len(edge.sentence_tokens)
                else b""
            )
            features.append(
                _reader._event_equality_signal(
                    observed,
                    expected,
                    population=f"{prefix}/literal_{index}",
                    races=races,
                )
            )
        return _reader._conjunction_signal(
            features,
            population=prefix,
            races=races,
        )

    def _empty_stop(
        self,
        *,
        query_sha256: str,
        races: list[RelationalRaceTrace],
        stage: str,
    ) -> PersistentDocumentDecision:
        inactive = _reader._control_signal(
            population=f"persistent/{stage}/inactive_input",
            enabled=False,
            races=races,
        )
        stop = _reader._physical_competition(
            population=f"persistent/{stage}/tonic_stop",
            signals=(inactive,),
            races=races,
        )
        return self._stop(
            query_sha256=query_sha256,
            races=races,
            stage=stage,
            reason=stop.trace.stop_reason,
        )

    def _stop(
        self,
        *,
        query_sha256: str,
        races: list[RelationalRaceTrace],
        stage: str,
        reason: str | None,
    ) -> PersistentDocumentDecision:
        if reason is None:
            raise RuntimeError("physical STOP lost its race reason")
        return self._decision(
            decision="STOP",
            answer_occurrence=None,
            schema_sha256=None,
            completed_hops=0,
            stop_stage=stage,
            stop_reason=f"{stage}:{reason}",
            races=races,
            query_sha256=query_sha256,
        )

    def _decision(
        self,
        *,
        decision: str,
        answer_occurrence: OpaqueTokenOccurrence | None,
        schema_sha256: str | None,
        completed_hops: int,
        stop_stage: str | None,
        stop_reason: str | None,
        races: list[RelationalRaceTrace],
        query_sha256: str,
    ) -> PersistentDocumentDecision:
        race_tuple = tuple(races)
        physical_cell_steps = sum(race.physical_cell_steps for race in race_tuple)
        answer = None if answer_occurrence is None else answer_occurrence.identity_event
        occurrence_event = None if answer_occurrence is None else answer_occurrence.occurrence_event
        document_id = None if answer_occurrence is None else answer_occurrence.document_id
        provenance_id = None if answer_occurrence is None else answer_occurrence.provenance_id
        record = {
            "answer_document_id_hex": None if document_id is None else document_id.hex(),
            "answer_hex": None if answer is None else answer.hex(),
            "answer_occurrence_event_hex": (
                None if occurrence_event is None else occurrence_event.hex()
            ),
            "answer_provenance_id_hex": (None if provenance_id is None else provenance_id.hex()),
            "checkpoint_sha256": self._checkpoint_sha256,
            "completed_hops": completed_hops,
            "decision": decision,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in race_tuple),
            "schema_sha256": schema_sha256,
            "stop_reason": stop_reason,
            "stop_stage": stop_stage,
        }
        return PersistentDocumentDecision(
            decision=decision,
            answer=answer,
            answer_occurrence_event=occurrence_event,
            answer_document_id=document_id,
            answer_provenance_id=provenance_id,
            schema_sha256=schema_sha256,
            completed_hops=completed_hops,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            races=race_tuple,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )


def _measure_parent_mutation(
    parent: FrozenPersistentNeuronalDocumentLearner,
    *,
    checkpoint_before: str,
    state_before: bytes,
) -> int:
    parent.authenticate()
    return int(
        parent.checkpoint_sha256 != checkpoint_before
        or _canonical_json(_persistent_state_record(parent._schemas, parent._chunks))
        != state_before
    )


def start() -> FrozenPersistentNeuronalDocumentLearner:
    """Return the authenticated empty learner."""

    return _freeze(schemas=(), chunks=())


def advance(
    parent: FrozenPersistentNeuronalDocumentLearner,
    success: RelationalReadingSuccess,
) -> PersistentLearningTransition:
    """Consume one raw success without replaying or retaining any parent row."""

    if type(parent) is not FrozenPersistentNeuronalDocumentLearner:
        raise TypeError("parent must be one exact frozen persistent learner")
    if type(success) is not RelationalReadingSuccess:
        raise TypeError("success must be one exact RelationalReadingSuccess")
    parent.authenticate()
    parent_root = parent.checkpoint_sha256
    parent_state_before = _canonical_json(_persistent_state_record(parent._schemas, parent._chunks))
    provenance_sha256 = _bytes_sha256(success.provenance_id)
    if any(chunk.success_provenance_sha256 == provenance_sha256 for chunk in parent._chunks):
        parent_mutation_count = _measure_parent_mutation(
            parent,
            checkpoint_before=parent_root,
            state_before=parent_state_before,
        )
        return PersistentLearningTransition(
            status="NO_UPDATE",
            updated=False,
            promoted_schema=False,
            parent_checkpoint_sha256=parent_root,
            successor_checkpoint_sha256=parent_root,
            schema_key_sha256=None,
            schema_support_before=0,
            schema_support_after=0,
            delta_chunks_added=0,
            current_episode_count=1,
            parent_episode_replay_count=0,
            parent_mutation_count=parent_mutation_count,
            retained_raw_training_row_count=(
                parent.training_summary.retained_raw_training_row_count
            ),
            learning_cell_steps=0,
            learning_active_synapses=0,
            successor=parent,
        )

    evidence = _compile_success(success)
    schema_index = next(
        (
            index
            for index, schema in enumerate(parent._schemas)
            if schema.key_sha256 == evidence.schema_key_sha256
        ),
        None,
    )
    old = None if schema_index is None else parent._schemas[schema_index]
    if len(parent._chunks) == _MAX_EVIDENCE_CHUNKS:
        raise PersistentNeuronalDocumentLearnerError("evidence chunk capacity exhausted")
    if old is None and len(parent._schemas) == _MAX_SCHEMAS:
        raise PersistentNeuronalDocumentLearnerError("surface schema capacity exhausted")

    chunk = _make_chunk(
        parent_checkpoint_sha256=parent_root,
        evidence=evidence,
    )
    if old is not None and (
        chunk.subject_sha256 in old.subject_sha256s
        or chunk.answer_sha256 in old.answer_sha256s
        or chunk.success_provenance_sha256 in old.success_provenance_sha256s
        or chunk.document_id_sha256 in old.document_id_sha256s
        or chunk.document_provenance_sha256 in old.document_provenance_sha256s
        or chunk.filler_context_pointer.sha256 in old.filler_context_pointer_sha256s
        or chunk.source_context_pointer.sha256 in old.source_context_pointer_sha256s
    ):
        parent_mutation_count = _measure_parent_mutation(
            parent,
            checkpoint_before=parent_root,
            state_before=parent_state_before,
        )
        return PersistentLearningTransition(
            status="NO_UPDATE",
            updated=False,
            promoted_schema=False,
            parent_checkpoint_sha256=parent_root,
            successor_checkpoint_sha256=parent_root,
            schema_key_sha256=evidence.schema_key_sha256,
            schema_support_before=old.support_ticks,
            schema_support_after=old.support_ticks,
            delta_chunks_added=0,
            current_episode_count=1,
            parent_episode_replay_count=0,
            parent_mutation_count=parent_mutation_count,
            retained_raw_training_row_count=(
                parent.training_summary.retained_raw_training_row_count
            ),
            learning_cell_steps=chunk.learning_cell_steps,
            learning_active_synapses=chunk.learning_active_synapses,
            successor=parent,
        )
    schema = _make_schema(evidence, old=old, chunk=chunk)
    if old is None:
        schemas = (*parent._schemas, schema)
        support_before = 0
    else:
        mutable = list(parent._schemas)
        mutable[schema_index] = schema
        schemas = tuple(mutable)
        support_before = old.support_ticks
    chunks = (*parent._chunks, chunk)
    successor = _freeze(schemas=schemas, chunks=chunks)
    parent_mutation_count = _measure_parent_mutation(
        parent,
        checkpoint_before=parent_root,
        state_before=parent_state_before,
    )
    if parent_mutation_count:
        raise RuntimeError("persistent advance mutated its parent")
    promoted_now = support_before < _MIN_PROMOTION_SUPPORT <= schema.support_ticks
    if promoted_now:
        status = "PROMOTED"
    elif schema.promoted:
        status = "SUPPORTED"
    else:
        status = "PENDING"
    return PersistentLearningTransition(
        status=status,
        updated=True,
        promoted_schema=promoted_now,
        parent_checkpoint_sha256=parent_root,
        successor_checkpoint_sha256=successor.checkpoint_sha256,
        schema_key_sha256=schema.key_sha256,
        schema_support_before=support_before,
        schema_support_after=schema.support_ticks,
        delta_chunks_added=1,
        current_episode_count=1,
        parent_episode_replay_count=0,
        parent_mutation_count=parent_mutation_count,
        retained_raw_training_row_count=(
            successor.training_summary.retained_raw_training_row_count
        ),
        learning_cell_steps=chunk.learning_cell_steps,
        learning_active_synapses=chunk.learning_active_synapses,
        successor=successor,
    )
