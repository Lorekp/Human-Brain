"""Replay-free continual growth for one-hop opaque relational reading.

The learner accepts one :class:`RelationalReadingSuccess` at a time.  An
answer-supervised host compiler locates one exact one-hop training path and
retains only its literal query/document surfaces plus opaque pointer and
provenance digests.  Raw queries, documents, subjects, answers, and training
rows are not retained.

An exact surface is promoted after two independently named filler/source
contexts.  Two distinct surfaces can instead become an anonymous equivalence
when physical cells find that their successful episodes share one complete
opaque subject/answer binding.  Frozen inference then uses fitted sparse-cell
cleanup to decide whether independently matching query and document surfaces
belong to the same learned association.  Every frame match, subject join,
candidate competition, and motor verification remains a simulated-cell
decision; missing or ambiguous evidence loses to a tonic STOP cell.

Once a surface is learned, a separate neutral-document step may consolidate
facts expressed on that surface without an answer label.  It retains only
counted endpoint lexemes, provenance, occurrence digests, and a sparse neuronal
fact assembly.  Later inference can retrieve that assembly when the source
document is absent and emits the retained answer lexeme only after neuronal
relation, subject, unique-fact, and motor decisions.

This is narrow answer-supervised exact-surface acquisition followed by neutral
fact consolidation and one-hop delayed recall.  It is not free-form paraphrase
understanding, neutral-only surface induction, revision, or multi-hop reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass, replace
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
    SparseCleanupRaceTrace,
    SparseNeuronalRoleFiller,
    SparsePointerSuperposition,
    SparseRoleFillerAblation,
    SparseTransformRaceTrace,
)

__all__ = [
    "CLAIM_BOUNDARY",
    "FrozenPersistentNeuronalDocumentLearner",
    "PersistentConsolidationTransition",
    "PersistentDocumentAblation",
    "PersistentDocumentDecision",
    "PersistentEvidenceChunk",
    "PersistentLearningSummary",
    "PersistentLearningTransition",
    "PersistentNeuronalDocumentLearnerError",
    "advance",
    "consolidate",
    "start",
]


CLAIM_BOUNDARY = (
    "answer-supervised exact-surface one-hop continual learning plus one-shot "
    "co-binding-induced exact-surface association, neutral facts on learned surfaces, "
    "and exact one-hop delayed recall; host training path compilation, candidate "
    "enumeration, and sensory projection are scaffolding"
)

_MIN_PROMOTION_SUPPORT = 2
_MAX_SCHEMAS = 64
_MAX_EVIDENCE_CHUNKS = 256
_MAX_CONSOLIDATED_FACTS = 1024
_CONSTRUCTION_TOKEN = object()
_SUBSTRATE = SparseNeuronalRoleFiller()

_CHECKPOINT_NAMESPACE = b"humanbrain.persistent-document.checkpoint.v1\0"
_STATE_NAMESPACE = b"humanbrain.persistent-document.state.v1\0"
_SCHEMA_KEY_NAMESPACE = b"humanbrain.persistent-document.schema-key.v1\0"
_SCHEMA_NAMESPACE = b"humanbrain.persistent-document.schema.v1\0"
_ASSOCIATION_NAMESPACE = b"humanbrain.persistent-document.association.v1\0"
_CHUNK_NAMESPACE = b"humanbrain.persistent-document.chunk.v1\0"
_QUERY_NAMESPACE = b"humanbrain.persistent-document.query.v1\0"
_DECISION_NAMESPACE = b"humanbrain.persistent-document.decision.v1\0"
_FILLER_CONTEXT_NAMESPACE = b"humanbrain.persistent-document.filler-context.v1\0"
_SOURCE_CONTEXT_NAMESPACE = b"humanbrain.persistent-document.source-context.v1\0"
_FACT_NAMESPACE = b"humanbrain.persistent-document.fact.v1\0"
_FACT_SOURCE_NAMESPACE = b"humanbrain.persistent-document.fact-source.v1\0"
_LEXEME_NAMESPACE = b"humanbrain.persistent-document.lexeme.v1\0"
_FACT_FILLER_NAMESPACE = b"humanbrain.persistent-document.fact-filler.v1\0"
_FACT_PROVENANCE_NAMESPACE = b"humanbrain.persistent-document.fact-provenance.v1\0"
_NEUTRAL_EPISODE_NAMESPACE = b"humanbrain.persistent-document.neutral-episode.v1\0"


class PersistentNeuronalDocumentLearnerError(ValueError):
    """The raw success cannot provide one exact one-hop training observation."""


class PersistentDocumentAblation(StrEnum):
    NONE = "none"
    QUERY_FRAME_OFF = "query_frame_off"
    DOCUMENT_FRAME_OFF = "document_frame_off"
    SUBJECT_JOIN_OFF = "subject_join_off"
    RELATION_BINDING_OFF = "relation_binding_off"
    RELATION_CLEANUP_OFF = "relation_cleanup_off"
    ANSWER_SELECTION_OFF = "answer_selection_off"
    MOTOR_OFF = "motor_off"
    CONSOLIDATION_OFF = "consolidation_off"
    RETRIEVAL_OFF = "retrieval_off"


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
    association_probe_trace_sha256s: tuple[str, ...]
    learning_cell_steps: int
    association_probe_cell_steps: int
    learning_active_synapses: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PersistentLearningSummary:
    accepted_episode_count: int
    delta_chunk_count: int
    schema_count: int
    pending_schema_count: int
    promoted_schema_count: int
    association_count: int
    learned_equivalence_count: int
    associated_surface_count: int
    promotion_support: int
    retained_raw_training_row_count: int
    retained_raw_document_count: int
    parent_episode_replay_count: int
    retained_surface_literal_byte_count: int
    retained_canonical_state_byte_count: int
    retained_pointer_count: int
    retained_active_cell_count: int
    retained_association_pointer_count: int
    consolidated_fact_count: int
    neutral_episode_count: int
    retained_fact_lexical_byte_count: int
    retained_fact_provenance_digest_byte_count: int
    cumulative_binding_update_count: int
    cumulative_learning_cell_steps: int
    cumulative_association_cell_steps: int
    cumulative_consolidation_cell_steps: int
    cumulative_learning_active_synapses: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class PersistentDocumentDecision:
    decision: str
    answer: bytes | None
    answer_occurrence_event: bytes | None
    answer_document_id: bytes | None
    answer_provenance_id: bytes | None
    answer_source: str | None
    memory_fact_sha256: str | None
    answer_document_sha256s: tuple[str, ...]
    answer_provenance_sha256s: tuple[str, ...]
    schema_sha256: str | None
    completed_hops: int
    stop_stage: str | None
    stop_reason: str | None
    races: tuple[RelationalRaceTrace, ...]
    relation_binding_races: tuple[SparseTransformRaceTrace, ...]
    relation_cleanup_races: tuple[SparseCleanupRaceTrace, ...]
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
    association_updated: bool
    association_link_count: int
    delta_chunks_added: int
    current_episode_count: int
    parent_episode_replay_count: int
    parent_mutation_count: int
    retained_raw_training_row_count: int
    learning_cell_steps: int
    learning_active_synapses: int
    association_binding_races: tuple[SparseTransformRaceTrace, ...]
    association_cleanup_races: tuple[SparseCleanupRaceTrace, ...]
    successor: FrozenPersistentNeuronalDocumentLearner


@dataclass(frozen=True, slots=True)
class PersistentConsolidationTransition:
    status: str
    updated: bool
    parent_checkpoint_sha256: str
    successor_checkpoint_sha256: str
    fact_sha256s: tuple[str, ...]
    facts_added: int
    current_episode_count: int
    parent_episode_replay_count: int
    parent_mutation_count: int
    retained_raw_document_count: int
    retained_fact_lexical_byte_count: int
    consolidation_cell_steps: int
    consolidation_active_synapses: int
    stop_stage: str | None
    stop_reason: str | None
    races: tuple[RelationalRaceTrace, ...]
    binding_races: tuple[SparseTransformRaceTrace, ...]
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
    surface_pointer: SparseBlockPointer
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
class _SurfaceAssociation:
    query_schema_key_sha256: str
    document_schema_key_sha256: str
    link_pointer: SparseBlockPointer
    evidence_chunk_sha256s: tuple[str, ...]
    binding_trace_sha256: str
    binding_cell_steps: int
    binding_active_synapses: int
    sha256: str

    @property
    def learned_equivalence(self) -> bool:
        return self.query_schema_key_sha256 != self.document_schema_key_sha256


@dataclass(frozen=True, slots=True)
class _FactSource:
    neutral_episode_sha256: str
    document_id_sha256: str
    provenance_id_sha256: str
    subject_occurrence_sha256: str
    answer_occurrence_sha256: str
    custody_pointer: SparseBlockPointer
    binding_trace_sha256s: tuple[str, str]
    binding_cell_steps: int
    binding_active_synapses: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _RetainedLexeme:
    identity_event: bytes
    pointer: SparseBlockPointer
    sha256: str


@dataclass(frozen=True, slots=True)
class _ConsolidatedFact:
    created_parent_checkpoint_sha256: str
    document_schema_key_sha256: str
    subject_sha256: str
    answer_sha256: str
    answer_lexeme_sha256: str
    key_pointer: SparseBlockPointer
    fact_pointer: SparseBlockPointer
    sources: tuple[_FactSource, ...]
    binding_trace_sha256s: tuple[str, ...]
    consolidation_trace_sha256s: tuple[str, ...]
    binding_cell_steps: int
    consolidation_cell_steps: int
    binding_active_synapses: int
    sha256: str


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
        "surface_pointer": _pointer_record(schema.surface_pointer),
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
        "association_probe_cell_steps": chunk.association_probe_cell_steps,
        "association_probe_trace_sha256s": chunk.association_probe_trace_sha256s,
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


def _association_record(
    association: _SurfaceAssociation,
    *,
    include_sha256: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "binding_active_synapses": association.binding_active_synapses,
        "binding_cell_steps": association.binding_cell_steps,
        "binding_trace_sha256": association.binding_trace_sha256,
        "document_schema_key_sha256": association.document_schema_key_sha256,
        "evidence_chunk_sha256s": association.evidence_chunk_sha256s,
        "link_pointer": _pointer_record(association.link_pointer),
        "query_schema_key_sha256": association.query_schema_key_sha256,
    }
    if include_sha256:
        record["sha256"] = association.sha256
    return record


def _fact_source_record(source: _FactSource, *, include_sha256: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "answer_occurrence_sha256": source.answer_occurrence_sha256,
        "binding_active_synapses": source.binding_active_synapses,
        "binding_cell_steps": source.binding_cell_steps,
        "binding_trace_sha256s": source.binding_trace_sha256s,
        "custody_pointer": _pointer_record(source.custody_pointer),
        "document_id_sha256": source.document_id_sha256,
        "neutral_episode_sha256": source.neutral_episode_sha256,
        "provenance_id_sha256": source.provenance_id_sha256,
        "subject_occurrence_sha256": source.subject_occurrence_sha256,
    }
    if include_sha256:
        record["sha256"] = source.sha256
    return record


def _lexeme_record(lexeme: _RetainedLexeme, *, include_sha256: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "identity_event_hex": lexeme.identity_event.hex(),
        "pointer": _pointer_record(lexeme.pointer),
    }
    if include_sha256:
        record["sha256"] = lexeme.sha256
    return record


def _fact_record(fact: _ConsolidatedFact, *, include_sha256: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "answer_lexeme_sha256": fact.answer_lexeme_sha256,
        "answer_sha256": fact.answer_sha256,
        "binding_active_synapses": fact.binding_active_synapses,
        "binding_cell_steps": fact.binding_cell_steps,
        "binding_trace_sha256s": fact.binding_trace_sha256s,
        "consolidation_cell_steps": fact.consolidation_cell_steps,
        "consolidation_trace_sha256s": fact.consolidation_trace_sha256s,
        "created_parent_checkpoint_sha256": fact.created_parent_checkpoint_sha256,
        "document_schema_key_sha256": fact.document_schema_key_sha256,
        "fact_pointer": _pointer_record(fact.fact_pointer),
        "key_pointer": _pointer_record(fact.key_pointer),
        "sources": tuple(
            _fact_source_record(source, include_sha256=True) for source in fact.sources
        ),
        "subject_sha256": fact.subject_sha256,
    }
    if include_sha256:
        record["sha256"] = fact.sha256
    return record


def _make_lexeme(identity_event: bytes, pointer: SparseBlockPointer) -> _RetainedLexeme:
    provisional = _RetainedLexeme(identity_event=identity_event, pointer=pointer, sha256="")
    return replace(
        provisional,
        sha256=_digest(_LEXEME_NAMESPACE, _lexeme_record(provisional, include_sha256=False)),
    )


def _make_fact_source(
    *,
    neutral_episode_sha256: str,
    edge: _reader._EdgeCandidate,
    fact_pointer: SparseBlockPointer,
) -> tuple[_FactSource, tuple[SparseTransformRaceTrace, SparseTransformRaceTrace]]:
    if edge.source.document_id is None or edge.source.provenance_id is None:
        raise RuntimeError("neutral fact edge lost its source provenance")
    document_pointer = _project_digest(
        _FACT_PROVENANCE_NAMESPACE + b"document\0",
        edge.source.document_id,
    )
    provenance_pointer = _project_digest(
        _FACT_PROVENANCE_NAMESPACE + b"provenance\0",
        edge.source.provenance_id,
    )
    source_binding = _SUBSTRATE.bind(document_pointer, provenance_pointer)
    if source_binding.pointer is None:
        raise RuntimeError("unablated neutral source binding selected tonic STOP")
    custody_binding = _SUBSTRATE.bind(fact_pointer, source_binding.pointer)
    if custody_binding.pointer is None:
        raise RuntimeError("unablated neutral custody binding selected tonic STOP")
    races = (source_binding.trace, custody_binding.trace)
    provisional = _FactSource(
        neutral_episode_sha256=neutral_episode_sha256,
        document_id_sha256=_bytes_sha256(edge.source.document_id),
        provenance_id_sha256=_bytes_sha256(edge.source.provenance_id),
        subject_occurrence_sha256=_bytes_sha256(edge.source.occurrence_event),
        answer_occurrence_sha256=_bytes_sha256(edge.destination.occurrence_event),
        custody_pointer=custody_binding.pointer,
        binding_trace_sha256s=(source_binding.trace.sha256, custody_binding.trace.sha256),
        binding_cell_steps=sum(race.physical_cell_steps for race in races),
        binding_active_synapses=sum(race.active_synapses for race in races),
        sha256="",
    )
    return (
        replace(
            provisional,
            sha256=_digest(
                _FACT_SOURCE_NAMESPACE,
                _fact_source_record(provisional, include_sha256=False),
            ),
        ),
        races,
    )


def _make_consolidated_fact(
    *,
    parent_checkpoint_sha256: str,
    neutral_episode_sha256: str,
    document_schema: _SurfaceSchema,
    edge: _reader._EdgeCandidate,
    consolidation_races: tuple[RelationalRaceTrace, ...],
) -> tuple[
    _ConsolidatedFact,
    _RetainedLexeme,
    tuple[SparseTransformRaceTrace, ...],
]:
    subject_pointer = _project_digest(
        _FACT_FILLER_NAMESPACE + b"subject\0",
        edge.source.identity_event,
    )
    answer_pointer = _project_digest(
        _FACT_FILLER_NAMESPACE + b"answer\0",
        edge.destination.identity_event,
    )
    key_binding = _SUBSTRATE.bind(document_schema.surface_pointer, subject_pointer)
    if key_binding.pointer is None:
        raise RuntimeError("unablated neutral fact key binding selected tonic STOP")
    fact_binding = _SUBSTRATE.bind(key_binding.pointer, answer_pointer)
    if fact_binding.pointer is None:
        raise RuntimeError("unablated neutral fact binding selected tonic STOP")
    source, source_races = _make_fact_source(
        neutral_episode_sha256=neutral_episode_sha256,
        edge=edge,
        fact_pointer=fact_binding.pointer,
    )
    binding_races = (key_binding.trace, fact_binding.trace, *source_races)
    lexeme = _make_lexeme(edge.destination.identity_event, answer_pointer)
    provisional = _ConsolidatedFact(
        created_parent_checkpoint_sha256=parent_checkpoint_sha256,
        document_schema_key_sha256=document_schema.key_sha256,
        subject_sha256=_bytes_sha256(edge.source.identity_event),
        answer_sha256=_bytes_sha256(edge.destination.identity_event),
        answer_lexeme_sha256=lexeme.sha256,
        key_pointer=key_binding.pointer,
        fact_pointer=fact_binding.pointer,
        sources=(source,),
        binding_trace_sha256s=(key_binding.trace.sha256, fact_binding.trace.sha256),
        consolidation_trace_sha256s=tuple(race.sha256 for race in consolidation_races),
        binding_cell_steps=sum(race.physical_cell_steps for race in binding_races),
        consolidation_cell_steps=(
            sum(race.physical_cell_steps for race in consolidation_races)
            + sum(race.physical_cell_steps for race in binding_races)
        ),
        binding_active_synapses=sum(race.active_synapses for race in binding_races),
        sha256="",
    )
    return (
        replace(
            provisional,
            sha256=_digest(_FACT_NAMESPACE, _fact_record(provisional, include_sha256=False)),
        ),
        lexeme,
        binding_races,
    )


def _merge_fact_source(
    old: _ConsolidatedFact,
    candidate: _ConsolidatedFact,
) -> tuple[_ConsolidatedFact, bool]:
    if (
        old.key_pointer.sha256 != candidate.key_pointer.sha256
        or old.fact_pointer.sha256 != candidate.fact_pointer.sha256
        or len(candidate.sources) != 1
    ):
        raise ValueError("only an identical fact assembly can merge provenance")
    source = candidate.sources[0]
    if any(existing.sha256 == source.sha256 for existing in old.sources):
        return old, False
    provisional = replace(
        old,
        sources=(*old.sources, source),
        binding_trace_sha256s=(
            *old.binding_trace_sha256s,
            *candidate.binding_trace_sha256s,
        ),
        consolidation_trace_sha256s=(
            *old.consolidation_trace_sha256s,
            *candidate.consolidation_trace_sha256s,
        ),
        binding_cell_steps=old.binding_cell_steps + candidate.binding_cell_steps,
        consolidation_cell_steps=(
            old.consolidation_cell_steps + candidate.consolidation_cell_steps
        ),
        binding_active_synapses=(
            old.binding_active_synapses + candidate.binding_active_synapses
        ),
        sha256="",
    )
    return (
        replace(
            provisional,
            sha256=_digest(_FACT_NAMESPACE, _fact_record(provisional, include_sha256=False)),
        ),
        True,
    )


def _make_schema(
    evidence: _CompiledEvidence,
    *,
    old: _SurfaceSchema | None,
    chunk: PersistentEvidenceChunk,
) -> _SurfaceSchema:
    if old is None:
        surface_pointer = _SUBSTRATE.project(bytes.fromhex(evidence.schema_key_sha256))
        subject_sha256s: tuple[str, ...] = ()
        answer_sha256s: tuple[str, ...] = ()
        success_provenance_sha256s: tuple[str, ...] = ()
        document_id_sha256s: tuple[str, ...] = ()
        document_provenance_sha256s: tuple[str, ...] = ()
        filler_context_pointer_sha256s: tuple[str, ...] = ()
        source_context_pointer_sha256s: tuple[str, ...] = ()
    else:
        surface_pointer = old.surface_pointer
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
        surface_pointer=surface_pointer,
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
        surface_pointer=provisional.surface_pointer,
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


def _make_association(
    *,
    query_schema: _SurfaceSchema,
    document_schema: _SurfaceSchema,
    evidence_chunk_sha256s: tuple[str, ...],
) -> tuple[_SurfaceAssociation, SparseTransformRaceTrace]:
    binding = _SUBSTRATE.bind(
        query_schema.surface_pointer,
        document_schema.surface_pointer,
    )
    if binding.pointer is None:
        raise RuntimeError("unablated surface association binding selected tonic STOP")
    provisional = _SurfaceAssociation(
        query_schema_key_sha256=query_schema.key_sha256,
        document_schema_key_sha256=document_schema.key_sha256,
        link_pointer=binding.pointer,
        evidence_chunk_sha256s=evidence_chunk_sha256s,
        binding_trace_sha256=binding.trace.sha256,
        binding_cell_steps=binding.trace.physical_cell_steps,
        binding_active_synapses=binding.trace.active_synapses,
        sha256="",
    )
    association = _SurfaceAssociation(
        query_schema_key_sha256=provisional.query_schema_key_sha256,
        document_schema_key_sha256=provisional.document_schema_key_sha256,
        link_pointer=provisional.link_pointer,
        evidence_chunk_sha256s=provisional.evidence_chunk_sha256s,
        binding_trace_sha256=provisional.binding_trace_sha256,
        binding_cell_steps=provisional.binding_cell_steps,
        binding_active_synapses=provisional.binding_active_synapses,
        sha256=_digest(
            _ASSOCIATION_NAMESPACE,
            _association_record(provisional, include_sha256=False),
        ),
    )
    return association, binding.trace


def _append_ordered_association(
    associations: tuple[_SurfaceAssociation, ...],
    *,
    query_schema: _SurfaceSchema,
    document_schema: _SurfaceSchema,
    evidence_chunk_sha256s: tuple[str, ...],
) -> tuple[
    tuple[_SurfaceAssociation, ...],
    _SurfaceAssociation | None,
    SparseTransformRaceTrace | None,
]:
    if any(
        association.query_schema_key_sha256 == query_schema.key_sha256
        and association.document_schema_key_sha256 == document_schema.key_sha256
        for association in associations
    ):
        return associations, None, None
    association, binding_race = _make_association(
        query_schema=query_schema,
        document_schema=document_schema,
        evidence_chunk_sha256s=evidence_chunk_sha256s,
    )
    return (*associations, association), association, binding_race


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
        association_probe_trace_sha256s=(),
        learning_cell_steps=(
            filler_binding.trace.physical_cell_steps + source_binding.trace.physical_cell_steps
        ),
        association_probe_cell_steps=0,
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
        association_probe_trace_sha256s=provisional.association_probe_trace_sha256s,
        learning_cell_steps=provisional.learning_cell_steps,
        association_probe_cell_steps=provisional.association_probe_cell_steps,
        learning_active_synapses=provisional.learning_active_synapses,
        sha256=_digest(
            _CHUNK_NAMESPACE,
            _chunk_record(provisional, include_sha256=False),
        ),
    )


def _with_association_probe(
    chunk: PersistentEvidenceChunk,
    cleanup_races: tuple[SparseCleanupRaceTrace, ...],
) -> PersistentEvidenceChunk:
    provisional = replace(
        chunk,
        association_probe_trace_sha256s=tuple(race.sha256 for race in cleanup_races),
        association_probe_cell_steps=sum(race.physical_cell_steps for race in cleanup_races),
        sha256="",
    )
    return replace(
        provisional,
        sha256=_digest(
            _CHUNK_NAMESPACE,
            _chunk_record(provisional, include_sha256=False),
        ),
    )


def _shared_binding_chunk(
    chunks: tuple[PersistentEvidenceChunk, ...],
    current: PersistentEvidenceChunk,
) -> tuple[
    PersistentEvidenceChunk | None,
    tuple[SparseCleanupRaceTrace, ...],
]:
    if not chunks:
        return None, ()
    cue = SparsePointerSuperposition.empty().accumulate(current.filler_context_pointer)
    live_pointers = tuple(chunk.filler_context_pointer for chunk in chunks)
    cleanup = _SUBSTRATE.cleanup(cue, live_pointers)
    if cleanup.pointer is None:
        return None, (cleanup.trace,)
    selected_index = next(
        index for index, pointer in enumerate(live_pointers) if pointer is cleanup.pointer
    )
    return chunks[selected_index], (cleanup.trace,)


def _persistent_state_record(
    schemas: tuple[_SurfaceSchema, ...],
    associations: tuple[_SurfaceAssociation, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
    facts: tuple[_ConsolidatedFact, ...],
    lexemes: tuple[_RetainedLexeme, ...],
) -> dict[str, object]:
    return {
        "associations": tuple(
            _association_record(association, include_sha256=True) for association in associations
        ),
        "chunks": tuple(_chunk_record(chunk, include_sha256=True) for chunk in chunks),
        "facts": tuple(_fact_record(fact, include_sha256=True) for fact in facts),
        "lexemes": tuple(_lexeme_record(lexeme, include_sha256=True) for lexeme in lexemes),
        "schemas": tuple(_schema_record(schema, include_sha256=True) for schema in schemas),
    }


def _retained_raw_artifact_counts(
    schemas: tuple[_SurfaceSchema, ...],
    associations: tuple[_SurfaceAssociation, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
    facts: tuple[_ConsolidatedFact, ...],
    lexemes: tuple[_RetainedLexeme, ...],
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

    return visit((schemas, associations, chunks, facts, lexemes))


def _retained_pointers(
    schemas: tuple[_SurfaceSchema, ...],
    associations: tuple[_SurfaceAssociation, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
    facts: tuple[_ConsolidatedFact, ...],
    lexemes: tuple[_RetainedLexeme, ...],
) -> tuple[SparseBlockPointer, ...]:
    return (
        *(schema.surface_pointer for schema in schemas),
        *(association.link_pointer for association in associations),
        *(chunk.filler_context_pointer for chunk in chunks),
        *(chunk.source_context_pointer for chunk in chunks),
        *(fact.key_pointer for fact in facts),
        *(fact.fact_pointer for fact in facts),
        *(source.custody_pointer for fact in facts for source in fact.sources),
        *(lexeme.pointer for lexeme in lexemes),
    )


def _retained_canonical_state_bytes(
    schemas: tuple[_SurfaceSchema, ...],
    associations: tuple[_SurfaceAssociation, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
    facts: tuple[_ConsolidatedFact, ...],
    lexemes: tuple[_RetainedLexeme, ...],
) -> int:
    return len(
        _canonical_json(_persistent_state_record(schemas, associations, chunks, facts, lexemes))
    )


def _checkpoint_record(
    schemas: tuple[_SurfaceSchema, ...],
    associations: tuple[_SurfaceAssociation, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
    facts: tuple[_ConsolidatedFact, ...],
    lexemes: tuple[_RetainedLexeme, ...],
) -> dict[str, object]:
    state_record = _persistent_state_record(schemas, associations, chunks, facts, lexemes)
    raw_rows, raw_documents = _retained_raw_artifact_counts(
        schemas,
        associations,
        chunks,
        facts,
        lexemes,
    )
    return {
        "association_sha256s": tuple(association.sha256 for association in associations),
        "claim_boundary": CLAIM_BOUNDARY,
        "chunk_sha256s": tuple(chunk.sha256 for chunk in chunks),
        "fact_sha256s": tuple(fact.sha256 for fact in facts),
        "lexeme_sha256s": tuple(lexeme.sha256 for lexeme in lexemes),
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


def _retained_fact_lexical_bytes(lexemes: tuple[_RetainedLexeme, ...]) -> int:
    return sum(len(lexeme.identity_event) for lexeme in lexemes)


def _retained_fact_provenance_digest_bytes(facts: tuple[_ConsolidatedFact, ...]) -> int:
    return sum(
        len(bytes.fromhex(source.document_id_sha256))
        + len(bytes.fromhex(source.provenance_id_sha256))
        for fact in facts
        for source in fact.sources
    )


def _freeze(
    *,
    schemas: tuple[_SurfaceSchema, ...],
    associations: tuple[_SurfaceAssociation, ...],
    chunks: tuple[PersistentEvidenceChunk, ...],
    facts: tuple[_ConsolidatedFact, ...],
    lexemes: tuple[_RetainedLexeme, ...],
) -> FrozenPersistentNeuronalDocumentLearner:
    raw_rows, raw_documents = _retained_raw_artifact_counts(
        schemas,
        associations,
        chunks,
        facts,
        lexemes,
    )
    pointers = _retained_pointers(schemas, associations, chunks, facts, lexemes)
    checkpoint_sha256 = _digest(
        _CHECKPOINT_NAMESPACE,
        _checkpoint_record(schemas, associations, chunks, facts, lexemes),
    )
    promoted = sum(schema.promoted for schema in schemas)
    summary = PersistentLearningSummary(
        accepted_episode_count=len(chunks),
        delta_chunk_count=len(chunks),
        schema_count=len(schemas),
        pending_schema_count=len(schemas) - promoted,
        promoted_schema_count=promoted,
        association_count=len(associations),
        learned_equivalence_count=len(
            {
                tuple(
                    sorted(
                        (
                            association.query_schema_key_sha256,
                            association.document_schema_key_sha256,
                        )
                    )
                )
                for association in associations
                if association.learned_equivalence
            }
        ),
        associated_surface_count=len(
            {
                key
                for association in associations
                for key in (
                    association.query_schema_key_sha256,
                    association.document_schema_key_sha256,
                )
            }
        ),
        promotion_support=_MIN_PROMOTION_SUPPORT,
        retained_raw_training_row_count=raw_rows,
        retained_raw_document_count=raw_documents,
        parent_episode_replay_count=0,
        retained_surface_literal_byte_count=_retained_surface_literal_bytes(schemas),
        retained_canonical_state_byte_count=_retained_canonical_state_bytes(
            schemas,
            associations,
            chunks,
            facts,
            lexemes,
        ),
        retained_pointer_count=len(pointers),
        retained_active_cell_count=sum(pointer.active_cell_count for pointer in pointers),
        retained_association_pointer_count=len(associations),
        consolidated_fact_count=len(facts),
        neutral_episode_count=len(
            {source.neutral_episode_sha256 for fact in facts for source in fact.sources}
        ),
        retained_fact_lexical_byte_count=_retained_fact_lexical_bytes(lexemes),
        retained_fact_provenance_digest_byte_count=(
            _retained_fact_provenance_digest_bytes(facts)
        ),
        cumulative_binding_update_count=sum(len(chunk.binding_trace_sha256s) for chunk in chunks)
        + sum(
            len(fact.binding_trace_sha256s)
            + sum(len(source.binding_trace_sha256s) for source in fact.sources)
            for fact in facts
        ),
        cumulative_learning_cell_steps=sum(
            chunk.learning_cell_steps + chunk.association_probe_cell_steps for chunk in chunks
        )
        + sum(association.binding_cell_steps for association in associations)
        + sum(fact.consolidation_cell_steps for fact in facts),
        cumulative_association_cell_steps=sum(
            chunk.association_probe_cell_steps for chunk in chunks
        )
        + sum(association.binding_cell_steps for association in associations),
        cumulative_consolidation_cell_steps=sum(
            fact.consolidation_cell_steps for fact in facts
        ),
        cumulative_learning_active_synapses=sum(chunk.learning_active_synapses for chunk in chunks)
        + sum(association.binding_active_synapses for association in associations)
        + sum(fact.binding_active_synapses for fact in facts),
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenPersistentNeuronalDocumentLearner(
        _CONSTRUCTION_TOKEN,
        schemas=schemas,
        associations=associations,
        chunks=chunks,
        facts=facts,
        lexemes=lexemes,
        summary=summary,
        checkpoint_sha256=checkpoint_sha256,
    )


class FrozenPersistentNeuronalDocumentLearner:
    """Immutable exact surfaces, ordered associations, and shared deltas."""

    __slots__ = (
        "_associations",
        "_checkpoint_sha256",
        "_chunks",
        "_facts",
        "_lexemes",
        "_schemas",
        "_summary",
    )

    def __init__(
        self,
        token: object,
        *,
        schemas: tuple[_SurfaceSchema, ...],
        associations: tuple[_SurfaceAssociation, ...],
        chunks: tuple[PersistentEvidenceChunk, ...],
        facts: tuple[_ConsolidatedFact, ...],
        lexemes: tuple[_RetainedLexeme, ...],
        summary: PersistentLearningSummary,
        checkpoint_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use start() and advance()")
        self._schemas = schemas
        self._associations = associations
        self._chunks = chunks
        self._facts = facts
        self._lexemes = lexemes
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

    @property
    def consolidated_fact_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return tuple(fact.sha256 for fact in self._facts)

    @property
    def consolidated_provenance_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return tuple(
            source.provenance_id_sha256 for fact in self._facts for source in fact.sources
        )

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
                schema.surface_pointer.block_cells,
                schema.surface_pointer.sha256,
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
        schema_keys = {schema.key_sha256 for schema in self._schemas}
        chunk_sha256s = {chunk.sha256 for chunk in self._chunks}
        association_pairs: set[tuple[str, str]] = set()
        association_pointer_sha256s: set[str] = set()
        for association in self._associations:
            _require_sha256(association.sha256, "association sha256")
            _require_sha256(
                association.binding_trace_sha256,
                "association binding trace sha256",
            )
            pair = (
                association.query_schema_key_sha256,
                association.document_schema_key_sha256,
            )
            if pair in association_pairs or not set(pair) <= schema_keys:
                raise RuntimeError("persistent association endpoints changed")
            association_pairs.add(pair)
            SparseBlockPointer(
                association.link_pointer.block_cells,
                association.link_pointer.sha256,
            )
            if association.link_pointer.sha256 in association_pointer_sha256s:
                raise RuntimeError("persistent association pointers must be unique")
            association_pointer_sha256s.add(association.link_pointer.sha256)
            if (
                not association.evidence_chunk_sha256s
                or not set(association.evidence_chunk_sha256s) <= chunk_sha256s
                or association.binding_cell_steps <= 0
                or association.binding_active_synapses <= 0
            ):
                raise RuntimeError("persistent association evidence changed")
            if association.sha256 != _digest(
                _ASSOCIATION_NAMESPACE,
                _association_record(association, include_sha256=False),
            ):
                raise RuntimeError("persistent association identity changed")
        lexemes_by_sha256: dict[str, _RetainedLexeme] = {}
        lexeme_pointer_sha256s: set[str] = set()
        for lexeme in self._lexemes:
            _require_sha256(lexeme.sha256, "lexeme sha256")
            if not lexeme.identity_event:
                raise RuntimeError("persistent lexeme lost its exact bytes")
            SparseBlockPointer(lexeme.pointer.block_cells, lexeme.pointer.sha256)
            if (
                lexeme.sha256 in lexemes_by_sha256
                or lexeme.pointer.sha256 in lexeme_pointer_sha256s
            ):
                raise RuntimeError("persistent lexemes must be deduplicated")
            if lexeme.sha256 != _digest(
                _LEXEME_NAMESPACE,
                _lexeme_record(lexeme, include_sha256=False),
            ):
                raise RuntimeError("persistent lexeme identity changed")
            lexemes_by_sha256[lexeme.sha256] = lexeme
            lexeme_pointer_sha256s.add(lexeme.pointer.sha256)
        document_schema_keys = {
            association.document_schema_key_sha256 for association in self._associations
        }
        fact_assemblies: set[tuple[str, str]] = set()
        for fact in self._facts:
            for digest, name in (
                (fact.sha256, "fact sha256"),
                (fact.created_parent_checkpoint_sha256, "fact parent checkpoint sha256"),
                (fact.document_schema_key_sha256, "fact document schema sha256"),
                (fact.subject_sha256, "fact subject sha256"),
                (fact.answer_sha256, "fact answer sha256"),
                (fact.answer_lexeme_sha256, "fact answer lexeme sha256"),
            ):
                _require_sha256(digest, name)
            if fact.document_schema_key_sha256 not in document_schema_keys:
                raise RuntimeError("persistent fact lost its learned document surface")
            lexeme = lexemes_by_sha256.get(fact.answer_lexeme_sha256)
            if (
                lexeme is None
                or fact.answer_sha256 != _bytes_sha256(lexeme.identity_event)
                or fact.fact_pointer.sha256 == fact.key_pointer.sha256
            ):
                raise RuntimeError("persistent fact lost its answer lexeme assembly")
            SparseBlockPointer(fact.key_pointer.block_cells, fact.key_pointer.sha256)
            SparseBlockPointer(fact.fact_pointer.block_cells, fact.fact_pointer.sha256)
            assembly = (fact.key_pointer.sha256, fact.fact_pointer.sha256)
            if assembly in fact_assemblies:
                raise RuntimeError("identical persistent facts must merge provenance")
            fact_assemblies.add(assembly)
            if (
                not fact.sources
                or len(fact.binding_trace_sha256s) != 2 * len(fact.sources)
                or not fact.consolidation_trace_sha256s
                or fact.binding_cell_steps <= 0
                or fact.consolidation_cell_steps <= 0
                or fact.binding_active_synapses <= 0
            ):
                raise RuntimeError("persistent fact lost physical consolidation accounting")
            source_sha256s: set[str] = set()
            for source in fact.sources:
                for digest, name in (
                    (source.sha256, "fact source sha256"),
                    (source.neutral_episode_sha256, "neutral episode sha256"),
                    (source.document_id_sha256, "fact document id sha256"),
                    (source.provenance_id_sha256, "fact provenance sha256"),
                    (source.subject_occurrence_sha256, "fact subject occurrence sha256"),
                    (source.answer_occurrence_sha256, "fact answer occurrence sha256"),
                ):
                    _require_sha256(digest, name)
                SparseBlockPointer(
                    source.custody_pointer.block_cells,
                    source.custody_pointer.sha256,
                )
                if (
                    source.sha256 in source_sha256s
                    or len(source.binding_trace_sha256s) != 2
                    or source.binding_cell_steps <= 0
                    or source.binding_active_synapses <= 0
                    or source.sha256
                    != _digest(
                        _FACT_SOURCE_NAMESPACE,
                        _fact_source_record(source, include_sha256=False),
                    )
                ):
                    raise RuntimeError("persistent fact source identity changed")
                source_sha256s.add(source.sha256)
            if fact.sha256 != _digest(
                _FACT_NAMESPACE,
                _fact_record(fact, include_sha256=False),
            ):
                raise RuntimeError("persistent fact identity changed")
        expected_root = _digest(
            _CHECKPOINT_NAMESPACE,
            _checkpoint_record(
                self._schemas,
                self._associations,
                self._chunks,
                self._facts,
                self._lexemes,
            ),
        )
        if expected_root != self._checkpoint_sha256:
            raise RuntimeError("persistent learner checkpoint authentication failed")
        raw_rows, raw_documents = _retained_raw_artifact_counts(
            self._schemas,
            self._associations,
            self._chunks,
            self._facts,
            self._lexemes,
        )
        if raw_rows or raw_documents:
            raise RuntimeError("persistent learner retained a raw training artifact")
        pointers = _retained_pointers(
            self._schemas,
            self._associations,
            self._chunks,
            self._facts,
            self._lexemes,
        )
        promoted = sum(schema.promoted for schema in self._schemas)
        expected_summary = PersistentLearningSummary(
            accepted_episode_count=len(self._chunks),
            delta_chunk_count=len(self._chunks),
            schema_count=len(self._schemas),
            pending_schema_count=len(self._schemas) - promoted,
            promoted_schema_count=promoted,
            association_count=len(self._associations),
            learned_equivalence_count=len(
                {
                    tuple(
                        sorted(
                            (
                                association.query_schema_key_sha256,
                                association.document_schema_key_sha256,
                            )
                        )
                    )
                    for association in self._associations
                    if association.learned_equivalence
                }
            ),
            associated_surface_count=len(
                {
                    key
                    for association in self._associations
                    for key in (
                        association.query_schema_key_sha256,
                        association.document_schema_key_sha256,
                    )
                }
            ),
            promotion_support=_MIN_PROMOTION_SUPPORT,
            retained_raw_training_row_count=raw_rows,
            retained_raw_document_count=raw_documents,
            parent_episode_replay_count=0,
            retained_surface_literal_byte_count=_retained_surface_literal_bytes(self._schemas),
            retained_canonical_state_byte_count=_retained_canonical_state_bytes(
                self._schemas,
                self._associations,
                self._chunks,
                self._facts,
                self._lexemes,
            ),
            retained_pointer_count=len(pointers),
            retained_active_cell_count=sum(pointer.active_cell_count for pointer in pointers),
            retained_association_pointer_count=len(self._associations),
            consolidated_fact_count=len(self._facts),
            neutral_episode_count=len(
                {
                    source.neutral_episode_sha256
                    for fact in self._facts
                    for source in fact.sources
                }
            ),
            retained_fact_lexical_byte_count=_retained_fact_lexical_bytes(self._lexemes),
            retained_fact_provenance_digest_byte_count=(
                _retained_fact_provenance_digest_bytes(self._facts)
            ),
            cumulative_binding_update_count=sum(
                len(chunk.binding_trace_sha256s) for chunk in self._chunks
            )
            + sum(
                len(fact.binding_trace_sha256s)
                + sum(len(source.binding_trace_sha256s) for source in fact.sources)
                for fact in self._facts
            ),
            cumulative_learning_cell_steps=sum(
                chunk.learning_cell_steps + chunk.association_probe_cell_steps
                for chunk in self._chunks
            )
            + sum(association.binding_cell_steps for association in self._associations)
            + sum(fact.consolidation_cell_steps for fact in self._facts),
            cumulative_association_cell_steps=sum(
                chunk.association_probe_cell_steps for chunk in self._chunks
            )
            + sum(association.binding_cell_steps for association in self._associations),
            cumulative_consolidation_cell_steps=sum(
                fact.consolidation_cell_steps for fact in self._facts
            ),
            cumulative_learning_active_synapses=sum(
                chunk.learning_active_synapses for chunk in self._chunks
            )
            + sum(association.binding_active_synapses for association in self._associations)
            + sum(fact.binding_active_synapses for fact in self._facts),
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
        relation_binding_races: list[SparseTransformRaceTrace] = []
        relation_cleanup_races: list[SparseCleanupRaceTrace] = []
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
        associated_schema_keys = {
            key
            for association in self._associations
            for key in (
                association.query_schema_key_sha256,
                association.document_schema_key_sha256,
            )
        }
        schemas = tuple(
            schema for schema in self._schemas if schema.key_sha256 in associated_schema_keys
        )
        if not self._associations or not schemas:
            return self._empty_stop(
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
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
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="query_frame",
                reason=query_choice.trace.stop_reason,
            )

        edges = _reader._all_edges(tape)
        if not edges:
            if not self._facts:
                return self._empty_stop(
                    query_sha256=query_sha256,
                    races=races,
                    relation_binding_races=relation_binding_races,
                    relation_cleanup_races=relation_cleanup_races,
                    stage="answer_selection",
                )
            return self._retrieve(
                query_tokens=query_tokens,
                schemas=schemas,
                query_signals=query_signals,
                subject_control=subject_control,
                answer_control=answer_control,
                motor_control=motor_control,
                ablation=ablation,
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
            )
        edge_paths: list[list[_reader._CellSignal]] = [[] for _ in edges]
        learned_link_pointers = tuple(
            association.link_pointer for association in self._associations
        )
        for query_schema_index, (query_schema, query_signal) in enumerate(
            zip(schemas, query_signals, strict=True)
        ):
            subject = (
                query_tokens[query_schema.query_frame.subject_flat_index].identity_event
                if query_schema.query_frame.subject_flat_index < len(query_tokens)
                else b""
            )
            for document_schema_index, document_schema in enumerate(schemas):
                relation_binding = _SUBSTRATE.bind(
                    query_schema.surface_pointer,
                    document_schema.surface_pointer,
                    ablation=(
                        SparseRoleFillerAblation.BINDING_OFF
                        if ablation is PersistentDocumentAblation.RELATION_BINDING_OFF
                        else SparseRoleFillerAblation.NONE
                    ),
                )
                relation_binding_races.append(relation_binding.trace)
                if relation_binding.pointer is None:
                    relation_signal = _reader._CellSignal((0, 1))
                else:
                    relation_cue = SparsePointerSuperposition.empty().accumulate(
                        relation_binding.pointer
                    )
                    relation_cleanup = _SUBSTRATE.cleanup(
                        relation_cue,
                        learned_link_pointers,
                        ablation=(
                            SparseRoleFillerAblation.CLEANUP_OFF
                            if ablation is PersistentDocumentAblation.RELATION_CLEANUP_OFF
                            else SparseRoleFillerAblation.NONE
                        ),
                    )
                    relation_cleanup_races.append(relation_cleanup.trace)
                    relation_signal = _reader._CellSignal(
                        (1, 0) if relation_cleanup.pointer is not None else (0, 1)
                    )
                for edge_index, edge in enumerate(edges):
                    surface_signal = self._document_frame_signal(
                        document_schema,
                        edge,
                        schema_index=document_schema_index,
                        edge_index=edge_index,
                        control=document_control,
                        races=races,
                    )
                    equality_signal = _reader._event_equality_signal(
                        subject,
                        edge.source.identity_event,
                        population=(
                            f"persistent/query_schema_{query_schema_index}/"
                            f"document_schema_{document_schema_index}/edge_{edge_index}/"
                            "subject_equality"
                        ),
                        races=races,
                    )
                    edge_paths[edge_index].append(
                        _reader._conjunction_signal(
                            (
                                query_signal,
                                surface_signal,
                                equality_signal,
                                relation_signal,
                                subject_control,
                                answer_control,
                            ),
                            population=(
                                f"persistent/query_schema_{query_schema_index}/"
                                f"document_schema_{document_schema_index}/edge_{edge_index}/"
                                "associated_one_hop_join"
                            ),
                            races=races,
                        )
                    )

        edge_signals = tuple(
            _reader._disjunction_signal(
                paths,
                population=f"persistent/edge_{edge_index}/associated_path_or",
                races=races,
            )
            for edge_index, paths in enumerate(edge_paths)
        )
        answer_choice = _reader._physical_competition(
            population="persistent/unique_answer_selection",
            signals=edge_signals,
            races=races,
        )
        if answer_choice.column is None:
            if answer_choice.trace.stop_reason == "tonic_stop" and self._facts:
                return self._retrieve(
                    query_tokens=query_tokens,
                    schemas=schemas,
                    query_signals=query_signals,
                    subject_control=subject_control,
                    answer_control=answer_control,
                    motor_control=motor_control,
                    ablation=ablation,
                    query_sha256=query_sha256,
                    races=races,
                    relation_binding_races=relation_binding_races,
                    relation_cleanup_races=relation_cleanup_races,
                )
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="answer_selection",
                reason=answer_choice.trace.stop_reason,
            )
        selected_edge = edges[answer_choice.column]
        active_query_schemas = tuple(
            schema
            for schema, signal in zip(schemas, query_signals, strict=True)
            if signal.output_spikes[0]
        )
        selected_schema_sha256 = (
            active_query_schemas[0].sha256 if len(active_query_schemas) == 1 else None
        )

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
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="motor",
                reason=motor.trace.stop_reason,
            )
        answer_occurrence = live_tokens[motor.column]
        return self._decision(
            decision="EMIT",
            answer_occurrence=answer_occurrence,
            memory_fact=None,
            memory_lexeme=None,
            answer_source="live",
            schema_sha256=selected_schema_sha256,
            completed_hops=1,
            stop_stage=None,
            stop_reason=None,
            races=races,
            relation_binding_races=relation_binding_races,
            relation_cleanup_races=relation_cleanup_races,
            query_sha256=query_sha256,
        )

    def _retrieve(
        self,
        *,
        query_tokens: tuple[OpaqueTokenOccurrence, ...],
        schemas: tuple[_SurfaceSchema, ...],
        query_signals: tuple[_reader._CellSignal, ...],
        subject_control: _reader._CellSignal,
        answer_control: _reader._CellSignal,
        motor_control: _reader._CellSignal,
        ablation: PersistentDocumentAblation,
        query_sha256: str,
        races: list[RelationalRaceTrace],
        relation_binding_races: list[SparseTransformRaceTrace],
        relation_cleanup_races: list[SparseCleanupRaceTrace],
    ) -> PersistentDocumentDecision:
        retrieval_control = _reader._control_signal(
            population="persistent/retrieval_control",
            enabled=ablation is not PersistentDocumentAblation.RETRIEVAL_OFF,
            races=races,
        )
        if not self._facts or not self._lexemes:
            return self._empty_stop(
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="retrieval",
            )

        learned_link_pointers = tuple(
            association.link_pointer for association in self._associations
        )
        unique_key_pointers: list[SparseBlockPointer] = []
        seen_key_sha256s: set[str] = set()
        for fact in self._facts:
            if fact.key_pointer.sha256 not in seen_key_sha256s:
                seen_key_sha256s.add(fact.key_pointer.sha256)
                unique_key_pointers.append(fact.key_pointer)

        fact_paths: list[list[_reader._CellSignal]] = [[] for _ in self._facts]
        for query_schema_index, (query_schema, query_signal) in enumerate(
            zip(schemas, query_signals, strict=True)
        ):
            subject = (
                query_tokens[query_schema.query_frame.subject_flat_index].identity_event
                if query_schema.query_frame.subject_flat_index < len(query_tokens)
                else b""
            )
            subject_pointer = _project_digest(
                _FACT_FILLER_NAMESPACE + b"subject\0",
                subject,
            )
            for document_schema_index, document_schema in enumerate(schemas):
                relation_binding = _SUBSTRATE.bind(
                    query_schema.surface_pointer,
                    document_schema.surface_pointer,
                    ablation=(
                        SparseRoleFillerAblation.BINDING_OFF
                        if ablation is PersistentDocumentAblation.RELATION_BINDING_OFF
                        else SparseRoleFillerAblation.NONE
                    ),
                )
                relation_binding_races.append(relation_binding.trace)
                if relation_binding.pointer is None:
                    continue
                relation_cleanup = _SUBSTRATE.cleanup(
                    SparsePointerSuperposition.empty().accumulate(relation_binding.pointer),
                    learned_link_pointers,
                    ablation=(
                        SparseRoleFillerAblation.CLEANUP_OFF
                        if ablation is PersistentDocumentAblation.RELATION_CLEANUP_OFF
                        else SparseRoleFillerAblation.NONE
                    ),
                )
                relation_cleanup_races.append(relation_cleanup.trace)
                if relation_cleanup.pointer is None:
                    continue
                relation_signal = _reader._control_signal(
                    population=(
                        f"persistent/retrieval/query_schema_{query_schema_index}/"
                        f"document_schema_{document_schema_index}/relation_cleanup"
                    ),
                    enabled=True,
                    races=races,
                )
                key_binding = _SUBSTRATE.bind(
                    document_schema.surface_pointer,
                    subject_pointer,
                )
                relation_binding_races.append(key_binding.trace)
                if key_binding.pointer is None:
                    continue
                key_cleanup = _SUBSTRATE.cleanup(
                    SparsePointerSuperposition.empty().accumulate(key_binding.pointer),
                    tuple(unique_key_pointers),
                    ablation=(
                        SparseRoleFillerAblation.CLEANUP_OFF
                        if ablation is PersistentDocumentAblation.RETRIEVAL_OFF
                        else SparseRoleFillerAblation.NONE
                    ),
                )
                relation_cleanup_races.append(key_cleanup.trace)
                recovered_key = key_cleanup.pointer
                cleanup_signal = _reader._control_signal(
                    population=(
                        f"persistent/retrieval/query_schema_{query_schema_index}/"
                        f"document_schema_{document_schema_index}/key_cleanup"
                    ),
                    enabled=recovered_key is not None,
                    races=races,
                )
                recovered_key_event = (
                    b"" if recovered_key is None else bytes(recovered_key.block_cells)
                )
                for fact_index, fact in enumerate(self._facts):
                    key_equality = _reader._event_equality_signal(
                        recovered_key_event,
                        bytes(fact.key_pointer.block_cells),
                        population=(
                            f"persistent/retrieval/query_schema_{query_schema_index}/"
                            f"document_schema_{document_schema_index}/fact_{fact_index}/"
                            "key_equality"
                        ),
                        races=races,
                    )
                    fact_paths[fact_index].append(
                        _reader._conjunction_signal(
                            (
                                query_signal,
                                relation_signal,
                                cleanup_signal,
                                key_equality,
                                subject_control,
                                retrieval_control,
                                answer_control,
                            ),
                            population=(
                                f"persistent/retrieval/query_schema_{query_schema_index}/"
                                f"document_schema_{document_schema_index}/fact_{fact_index}/"
                                "associated_fact_join"
                            ),
                            races=races,
                        )
                    )

        fact_signals = tuple(
            _reader._disjunction_signal(
                paths,
                population=f"persistent/retrieval/fact_{fact_index}/path_or",
                races=races,
            )
            for fact_index, paths in enumerate(fact_paths)
        )
        fact_choice = _reader._physical_competition(
            population="persistent/retrieval/unique_fact_selection",
            signals=fact_signals,
            races=races,
        )
        if fact_choice.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="retrieval",
                reason=fact_choice.trace.stop_reason,
            )
        selected_fact = self._facts[fact_choice.column]
        unbinding = _SUBSTRATE.unbind(
            selected_fact.key_pointer,
            selected_fact.fact_pointer,
            ablation=(
                SparseRoleFillerAblation.BINDING_OFF
                if ablation is PersistentDocumentAblation.RETRIEVAL_OFF
                else SparseRoleFillerAblation.NONE
            ),
        )
        relation_binding_races.append(unbinding.trace)
        if unbinding.pointer is None:
            return self._empty_stop(
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="retrieval",
            )
        lexeme_cleanup = _SUBSTRATE.cleanup(
            SparsePointerSuperposition.empty().accumulate(unbinding.pointer),
            tuple(lexeme.pointer for lexeme in self._lexemes),
            ablation=(
                SparseRoleFillerAblation.CLEANUP_OFF
                if ablation is PersistentDocumentAblation.RETRIEVAL_OFF
                else SparseRoleFillerAblation.NONE
            ),
        )
        relation_cleanup_races.append(lexeme_cleanup.trace)
        if lexeme_cleanup.pointer is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="retrieval",
                reason=lexeme_cleanup.trace.stop_reason,
            )

        selected_pointer_event = bytes(lexeme_cleanup.pointer.block_cells)
        motor_signals: list[_reader._CellSignal] = []
        for lexeme_index, lexeme in enumerate(self._lexemes):
            pointer_equality = _reader._event_equality_signal(
                selected_pointer_event,
                bytes(lexeme.pointer.block_cells),
                population=f"persistent/retrieval/motor_{lexeme_index}/pointer_equality",
                races=races,
            )
            fact_link_equality = _reader._event_equality_signal(
                bytes.fromhex(selected_fact.answer_lexeme_sha256),
                bytes.fromhex(lexeme.sha256),
                population=f"persistent/retrieval/motor_{lexeme_index}/fact_link_equality",
                races=races,
            )
            motor_signals.append(
                _reader._conjunction_signal(
                    (pointer_equality, fact_link_equality, motor_control),
                    population=f"persistent/retrieval/motor_{lexeme_index}/verification",
                    races=races,
                )
            )
        motor = _reader._physical_competition(
            population="persistent/retrieval/lexical_motor",
            signals=motor_signals,
            races=races,
        )
        if motor.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                relation_binding_races=relation_binding_races,
                relation_cleanup_races=relation_cleanup_races,
                stage="motor",
                reason=motor.trace.stop_reason,
            )
        selected_lexeme = self._lexemes[motor.column]
        document_schema = next(
            schema
            for schema in schemas
            if schema.key_sha256 == selected_fact.document_schema_key_sha256
        )
        return self._decision(
            decision="EMIT",
            answer_occurrence=None,
            memory_fact=selected_fact,
            memory_lexeme=selected_lexeme,
            answer_source="memory",
            schema_sha256=document_schema.sha256,
            completed_hops=1,
            stop_stage=None,
            stop_reason=None,
            races=races,
            relation_binding_races=relation_binding_races,
            relation_cleanup_races=relation_cleanup_races,
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
        relation_binding_races: list[SparseTransformRaceTrace],
        relation_cleanup_races: list[SparseCleanupRaceTrace],
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
            relation_binding_races=relation_binding_races,
            relation_cleanup_races=relation_cleanup_races,
            stage=stage,
            reason=stop.trace.stop_reason,
        )

    def _stop(
        self,
        *,
        query_sha256: str,
        races: list[RelationalRaceTrace],
        relation_binding_races: list[SparseTransformRaceTrace],
        relation_cleanup_races: list[SparseCleanupRaceTrace],
        stage: str,
        reason: str | None,
    ) -> PersistentDocumentDecision:
        if reason is None:
            raise RuntimeError("physical STOP lost its race reason")
        return self._decision(
            decision="STOP",
            answer_occurrence=None,
            memory_fact=None,
            memory_lexeme=None,
            answer_source=None,
            schema_sha256=None,
            completed_hops=0,
            stop_stage=stage,
            stop_reason=f"{stage}:{reason}",
            races=races,
            relation_binding_races=relation_binding_races,
            relation_cleanup_races=relation_cleanup_races,
            query_sha256=query_sha256,
        )

    def _decision(
        self,
        *,
        decision: str,
        answer_occurrence: OpaqueTokenOccurrence | None,
        memory_fact: _ConsolidatedFact | None,
        memory_lexeme: _RetainedLexeme | None,
        answer_source: str | None,
        schema_sha256: str | None,
        completed_hops: int,
        stop_stage: str | None,
        stop_reason: str | None,
        races: list[RelationalRaceTrace],
        relation_binding_races: list[SparseTransformRaceTrace],
        relation_cleanup_races: list[SparseCleanupRaceTrace],
        query_sha256: str,
    ) -> PersistentDocumentDecision:
        race_tuple = tuple(races)
        binding_race_tuple = tuple(relation_binding_races)
        cleanup_race_tuple = tuple(relation_cleanup_races)
        physical_cell_steps = (
            sum(race.physical_cell_steps for race in race_tuple)
            + sum(race.physical_cell_steps for race in binding_race_tuple)
            + sum(race.physical_cell_steps for race in cleanup_race_tuple)
        )
        if (answer_occurrence is None) == (memory_fact is None) and decision == "EMIT":
            raise RuntimeError("EMIT must select exactly one live or memory answer")
        if (memory_fact is None) != (memory_lexeme is None):
            raise RuntimeError("memory fact and lexeme must be selected together")
        if memory_fact is not None and memory_lexeme is not None:
            if memory_fact.answer_lexeme_sha256 != memory_lexeme.sha256:
                raise RuntimeError("memory motor selected an unlinked lexeme")
            answer = memory_lexeme.identity_event
        else:
            answer = None if answer_occurrence is None else answer_occurrence.identity_event
        occurrence_event = None if answer_occurrence is None else answer_occurrence.occurrence_event
        document_id = None if answer_occurrence is None else answer_occurrence.document_id
        provenance_id = None if answer_occurrence is None else answer_occurrence.provenance_id
        if memory_fact is not None:
            document_sha256s = tuple(
                source.document_id_sha256 for source in memory_fact.sources
            )
            provenance_sha256s = tuple(
                source.provenance_id_sha256 for source in memory_fact.sources
            )
        elif answer_occurrence is not None:
            document_sha256s = (
                () if document_id is None else (_bytes_sha256(document_id),)
            )
            provenance_sha256s = (
                () if provenance_id is None else (_bytes_sha256(provenance_id),)
            )
        else:
            document_sha256s = ()
            provenance_sha256s = ()
        memory_fact_sha256 = None if memory_fact is None else memory_fact.sha256
        record = {
            "answer_document_id_hex": None if document_id is None else document_id.hex(),
            "answer_hex": None if answer is None else answer.hex(),
            "answer_occurrence_event_hex": (
                None if occurrence_event is None else occurrence_event.hex()
            ),
            "answer_provenance_id_hex": (None if provenance_id is None else provenance_id.hex()),
            "answer_document_sha256s": document_sha256s,
            "answer_provenance_sha256s": provenance_sha256s,
            "answer_source": answer_source,
            "checkpoint_sha256": self._checkpoint_sha256,
            "completed_hops": completed_hops,
            "decision": decision,
            "physical_cell_steps": physical_cell_steps,
            "memory_fact_sha256": memory_fact_sha256,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in race_tuple),
            "relation_binding_race_sha256s": tuple(race.sha256 for race in binding_race_tuple),
            "relation_cleanup_race_sha256s": tuple(race.sha256 for race in cleanup_race_tuple),
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
            answer_source=answer_source,
            memory_fact_sha256=memory_fact_sha256,
            answer_document_sha256s=document_sha256s,
            answer_provenance_sha256s=provenance_sha256s,
            schema_sha256=schema_sha256,
            completed_hops=completed_hops,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            races=race_tuple,
            relation_binding_races=binding_race_tuple,
            relation_cleanup_races=cleanup_race_tuple,
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
        or _canonical_json(
            _persistent_state_record(
                parent._schemas,
                parent._associations,
                parent._chunks,
                parent._facts,
                parent._lexemes,
            )
        )
        != state_before
    )


def start() -> FrozenPersistentNeuronalDocumentLearner:
    """Return the authenticated empty learner."""

    return _freeze(schemas=(), associations=(), chunks=(), facts=(), lexemes=())


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
    parent_state_before = _canonical_json(
        _persistent_state_record(
            parent._schemas,
            parent._associations,
            parent._chunks,
            parent._facts,
            parent._lexemes,
        )
    )
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
            association_updated=False,
            association_link_count=0,
            delta_chunks_added=0,
            current_episode_count=1,
            parent_episode_replay_count=0,
            parent_mutation_count=parent_mutation_count,
            retained_raw_training_row_count=(
                parent.training_summary.retained_raw_training_row_count
            ),
            learning_cell_steps=0,
            learning_active_synapses=0,
            association_binding_races=(),
            association_cleanup_races=(),
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
            association_updated=False,
            association_link_count=0,
            delta_chunks_added=0,
            current_episode_count=1,
            parent_episode_replay_count=0,
            parent_mutation_count=parent_mutation_count,
            retained_raw_training_row_count=(
                parent.training_summary.retained_raw_training_row_count
            ),
            learning_cell_steps=chunk.learning_cell_steps,
            learning_active_synapses=chunk.learning_active_synapses,
            association_binding_races=(),
            association_cleanup_races=(),
            successor=parent,
        )
    prior_binding_chunk, association_cleanup_races = _shared_binding_chunk(
        parent._chunks,
        chunk,
    )
    chunk = _with_association_probe(chunk, association_cleanup_races)
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
    promoted_now = support_before < _MIN_PROMOTION_SUPPORT <= schema.support_ticks
    associations = parent._associations
    new_associations: list[_SurfaceAssociation] = []
    association_binding_races: list[SparseTransformRaceTrace] = []
    if promoted_now:
        associations, added, binding_race = _append_ordered_association(
            associations,
            query_schema=schema,
            document_schema=schema,
            evidence_chunk_sha256s=tuple(
                evidence_chunk.sha256
                for evidence_chunk in chunks
                if evidence_chunk.schema_key_sha256 == schema.key_sha256
            ),
        )
        if added is not None and binding_race is not None:
            new_associations.append(added)
            association_binding_races.append(binding_race)
    if (
        prior_binding_chunk is not None
        and prior_binding_chunk.schema_key_sha256 != schema.key_sha256
    ):
        prior_schema = next(
            candidate
            for candidate in schemas
            if candidate.key_sha256 == prior_binding_chunk.schema_key_sha256
        )
        bridge_evidence = (prior_binding_chunk.sha256, chunk.sha256)
        for query_schema, document_schema in (
            (prior_schema, schema),
            (schema, prior_schema),
        ):
            associations, added, binding_race = _append_ordered_association(
                associations,
                query_schema=query_schema,
                document_schema=document_schema,
                evidence_chunk_sha256s=bridge_evidence,
            )
            if added is not None and binding_race is not None:
                new_associations.append(added)
                association_binding_races.append(binding_race)
    successor = _freeze(
        schemas=schemas,
        associations=associations,
        chunks=chunks,
        facts=parent._facts,
        lexemes=parent._lexemes,
    )
    parent_mutation_count = _measure_parent_mutation(
        parent,
        checkpoint_before=parent_root,
        state_before=parent_state_before,
    )
    if parent_mutation_count:
        raise RuntimeError("persistent advance mutated its parent")
    if new_associations and any(
        association.learned_equivalence for association in new_associations
    ):
        status = "ASSOCIATED"
    elif promoted_now:
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
        association_updated=bool(new_associations),
        association_link_count=len(new_associations),
        delta_chunks_added=1,
        current_episode_count=1,
        parent_episode_replay_count=0,
        parent_mutation_count=parent_mutation_count,
        retained_raw_training_row_count=(
            successor.training_summary.retained_raw_training_row_count
        ),
        learning_cell_steps=(
            chunk.learning_cell_steps
            + chunk.association_probe_cell_steps
            + sum(race.physical_cell_steps for race in association_binding_races)
        ),
        learning_active_synapses=(
            chunk.learning_active_synapses
            + sum(race.active_synapses for race in association_binding_races)
        ),
        association_binding_races=tuple(association_binding_races),
        association_cleanup_races=association_cleanup_races,
        successor=successor,
    )


def consolidate(
    parent: FrozenPersistentNeuronalDocumentLearner,
    documents: tuple[OpaqueDocument, ...],
    ablation: PersistentDocumentAblation = PersistentDocumentAblation.NONE,
) -> PersistentConsolidationTransition:
    """Consolidate every physical learned-surface match in one neutral episode."""

    if type(parent) is not FrozenPersistentNeuronalDocumentLearner:
        raise TypeError("parent must be one exact frozen persistent learner")
    if type(documents) is not tuple or not 1 <= len(documents) <= 4:
        raise ValueError("documents must contain one to four values")
    if any(not isinstance(document, OpaqueDocument) for document in documents):
        raise TypeError("documents must contain OpaqueDocument values")
    if not isinstance(ablation, PersistentDocumentAblation):
        raise TypeError("ablation must be PersistentDocumentAblation")
    parent.authenticate()
    parent_root = parent.checkpoint_sha256
    parent_state_before = _canonical_json(
        _persistent_state_record(
            parent._schemas,
            parent._associations,
            parent._chunks,
            parent._facts,
            parent._lexemes,
        )
    )
    neutral_episode_sha256 = _digest(
        _NEUTRAL_EPISODE_NAMESPACE,
        {
            "documents": tuple(
                {
                    "content_sha256": _bytes_sha256(document.content),
                    "document_id_sha256": _bytes_sha256(document.document_id),
                    "provenance_id_sha256": _bytes_sha256(document.provenance_id),
                }
                for document in documents
            )
        },
    )
    tape = OpaqueDocumentEventTape.observe(b"_", documents)
    edges = _reader._all_edges(tape)
    races: list[RelationalRaceTrace] = []
    binding_races: list[SparseTransformRaceTrace] = []
    frame_control = _reader._control_signal(
        population="persistent/consolidation/document_frame_control",
        enabled=True,
        races=races,
    )
    write_control = _reader._control_signal(
        population="persistent/consolidation/write_control",
        enabled=ablation is not PersistentDocumentAblation.CONSOLIDATION_OFF,
        races=races,
    )
    document_schema_keys = {
        association.document_schema_key_sha256 for association in parent._associations
    }
    document_schemas = tuple(
        schema for schema in parent._schemas if schema.key_sha256 in document_schema_keys
    )
    candidates: list[tuple[_ConsolidatedFact, _RetainedLexeme]] = []
    for schema_index, schema in enumerate(document_schemas):
        for edge_index, edge in enumerate(edges):
            trace_start = len(races)
            frame_signal = parent._document_frame_signal(
                schema,
                edge,
                schema_index=schema_index,
                edge_index=edge_index,
                control=frame_control,
                races=races,
            )
            write_signal = _reader._conjunction_signal(
                (frame_signal, write_control),
                population=(
                    f"persistent/consolidation/schema_{schema_index}/edge_{edge_index}/"
                    "write_gate"
                ),
                races=races,
            )
            write_choice = _reader._physical_competition(
                population=(
                    f"persistent/consolidation/schema_{schema_index}/edge_{edge_index}/"
                    "fact_write"
                ),
                signals=(write_signal,),
                races=races,
            )
            if write_choice.column is None:
                continue
            fact, lexeme, fact_binding_races = _make_consolidated_fact(
                parent_checkpoint_sha256=parent_root,
                neutral_episode_sha256=neutral_episode_sha256,
                document_schema=schema,
                edge=edge,
                consolidation_races=tuple(races[trace_start:]),
            )
            candidates.append((fact, lexeme))
            binding_races.extend(fact_binding_races)

    facts = list(parent._facts)
    lexemes = list(parent._lexemes)
    changed_fact_indices: set[int] = set()
    for candidate, lexeme in candidates:
        lexeme_index = next(
            (
                index
                for index, retained in enumerate(lexemes)
                if retained.pointer.sha256 == lexeme.pointer.sha256
            ),
            None,
        )
        if lexeme_index is None:
            lexemes.append(lexeme)
        elif lexemes[lexeme_index] != lexeme:
            raise RuntimeError("one lexical pointer mapped to different retained bytes")
        fact_index = next(
            (
                index
                for index, retained in enumerate(facts)
                if retained.key_pointer.sha256 == candidate.key_pointer.sha256
                and retained.fact_pointer.sha256 == candidate.fact_pointer.sha256
            ),
            None,
        )
        if fact_index is None:
            if len(facts) == _MAX_CONSOLIDATED_FACTS:
                raise PersistentNeuronalDocumentLearnerError(
                    "consolidated fact capacity exhausted"
                )
            facts.append(candidate)
            changed_fact_indices.add(len(facts) - 1)
            continue
        merged, changed = _merge_fact_source(facts[fact_index], candidate)
        if changed:
            facts[fact_index] = merged
            changed_fact_indices.add(fact_index)

    updated = bool(changed_fact_indices)
    if updated:
        successor = _freeze(
            schemas=parent._schemas,
            associations=parent._associations,
            chunks=parent._chunks,
            facts=tuple(facts),
            lexemes=tuple(lexemes),
        )
        status = "STORED"
        stop_stage = None
        stop_reason = None
    else:
        successor = parent
        if candidates:
            status = "NO_UPDATE"
            stop_stage = None
            stop_reason = None
        else:
            inactive = _reader._control_signal(
                population="persistent/consolidation/no_fact_input",
                enabled=False,
                races=races,
            )
            stopped = _reader._physical_competition(
                population="persistent/consolidation/tonic_stop",
                signals=(inactive,),
                races=races,
            )
            status = "STOP"
            stop_stage = "consolidation"
            stop_reason = f"consolidation:{stopped.trace.stop_reason}"
    parent_mutation_count = _measure_parent_mutation(
        parent,
        checkpoint_before=parent_root,
        state_before=parent_state_before,
    )
    if parent_mutation_count:
        raise RuntimeError("persistent consolidation mutated its parent")
    return PersistentConsolidationTransition(
        status=status,
        updated=updated,
        parent_checkpoint_sha256=parent_root,
        successor_checkpoint_sha256=successor.checkpoint_sha256,
        fact_sha256s=tuple(facts[index].sha256 for index in sorted(changed_fact_indices)),
        facts_added=(
            successor.training_summary.consolidated_fact_count
            - parent.training_summary.consolidated_fact_count
        ),
        current_episode_count=1,
        parent_episode_replay_count=0,
        parent_mutation_count=parent_mutation_count,
        retained_raw_document_count=successor.training_summary.retained_raw_document_count,
        retained_fact_lexical_byte_count=(
            successor.training_summary.retained_fact_lexical_byte_count
        ),
        consolidation_cell_steps=(
            sum(race.physical_cell_steps for race in races)
            + sum(race.physical_cell_steps for race in binding_races)
        ),
        consolidation_active_synapses=sum(
            race.active_synapses for race in binding_races
        ),
        stop_stage=stop_stage,
        stop_reason=stop_reason,
        races=tuple(races),
        binding_races=tuple(binding_races),
        successor=successor,
    )
