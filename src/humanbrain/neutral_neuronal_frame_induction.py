"""Neutral neuronal induction of reusable two-slot language frames.

The public seam accepts only opaque documents, a raw query, and causal
ablations.  Complete sentences and lexical payloads are never retained.
Consolidation keeps sparse frame and filler pointers plus opaque provenance
roots.  Repeated positional structure promotes through a physical
support-versus-STOP race; independently shared filler pairs then drive a
physical SAME-versus-SWAPPED orientation race.

At frozen inference, the implementation enumerates every compatible one-gap
surface without choosing one.  Sparse cleanup populations identify learned
frames, bound frame routes recover the associated surface, filler cleanup
checks the live alignment, and one final GLIF population emits only a unique
lexeme.  Competing completions collide into STOP.

This is evidence for neutral structural recurrence and shared-binding transfer,
not broad semantics, truth, question intent, or general language understanding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from itertools import combinations
from typing import Self

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.opaque_document_event_tape import (
    OpaqueDocument,
    OpaqueDocumentEventTape,
    OpaqueSentenceOccurrence,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
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
    "FrozenNeutralNeuronalFrameNetwork",
    "NeutralFrameAblation",
    "NeutralFrameDecision",
    "NeutralFrameRaceTrace",
    "NeutralFrameTrainingSummary",
    "start",
]


CLAIM_BOUNDARY = (
    "neutral two-slot structural recurrence and shared-binding SAME/SWAPPED "
    "transfer from opaque documents; token boundaries, positional slot "
    "enumeration, and exact lexeme emission remain scaffolding"
)

_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_DRIVE_SCORE = 2
_TONIC_STOP_SCORE = 1

_CHECKPOINT_NAMESPACE = b"humanbrain.neutral-frame.checkpoint.v1\0"
_DECISION_NAMESPACE = b"humanbrain.neutral-frame.decision.v1\0"
_EVIDENCE_NAMESPACE = b"humanbrain.neutral-frame.evidence.v1\0"
_FRAME_NAMESPACE = b"humanbrain.neutral-frame.structure.v1\0"
_LEXEME_NAMESPACE = b"humanbrain.neutral-frame.lexeme.v1\0"
_PAIR_NAMESPACE = b"humanbrain.neutral-frame.pair.v1\0"
_PROVENANCE_NAMESPACE = b"humanbrain.neutral-frame.provenance.v1\0"
_DOCUMENT_NAMESPACE = b"humanbrain.neutral-frame.document.v1\0"
_ROUTE_NAMESPACE = b"humanbrain.neutral-frame.route.v1\0"
_RACE_NAMESPACE = b"humanbrain.neutral-frame.race.v1\0"

_FRAME_EVENT = b"\x00neutral-frame/structure/v1\0"
_LEXEME_EVENT = b"\x00neutral-frame/lexeme/v1\0"
_CONSTRUCTION_TOKEN = object()
_SUBSTRATE = SparseNeuronalRoleFiller()
_SAME_ORIENTATION_POINTER = _SUBSTRATE.project(
    b"\x00neutral-frame/orientation/same/v1"
)
_SWAPPED_ORIENTATION_POINTER = _SUBSTRATE.project(
    b"\x00neutral-frame/orientation/swapped/v1"
)


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


def _bytes_root(namespace: bytes, value: bytes) -> str:
    return sha256(namespace + len(value).to_bytes(8, "little") + value).hexdigest()


def _pointer_record(pointer: SparseBlockPointer) -> dict[str, object]:
    return {"block_cells": pointer.block_cells, "sha256": pointer.sha256}


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(_RACE_STEPS)
    crossing = population.parameters.threshold_inf_relative_mv / (voltage - threshold)
    currents: list[float] = []
    for score in range(_DRIVE_SCORE + 1):
        step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing[step])
        upper = float(crossing[step - 1])
        if not lower < upper:
            raise RuntimeError("neutral-frame GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    checks = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score
        for score, result in enumerate(checks)
    ):
        raise RuntimeError("neutral-frame GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class NeutralFrameAblation(StrEnum):
    NONE = "none"
    FRAME_PROMOTION_OFF = "frame_promotion_off"
    ASSOCIATION_LEARNING_OFF = "association_learning_off"
    FRAME_MATCH_OFF = "frame_match_off"
    READOUT_OFF = "readout_off"
    ALIGNMENT_OFF = "alignment_off"
    MOTOR_OFF = "motor_off"


@dataclass(frozen=True, slots=True)
class NeutralFrameRaceTrace:
    population: str
    candidate_cell_ids: tuple[int, ...]
    candidate_scores: tuple[int, ...]
    tonic_stop_cell_id: int
    winner_cell_id: int | None
    spike_step: int
    simultaneous_spike_candidates: int
    stopped: bool
    stop_reason: str | None
    collision_stop_spike_step: int | None
    physical_cell_steps: int
    active_synapses: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _RaceChoice:
    column: int | None
    trace: NeutralFrameRaceTrace


def _physical_choice(
    *,
    population: str,
    scores: tuple[int, ...],
    candidate_cell_ids: tuple[int, ...] | None = None,
) -> _RaceChoice:
    if any(type(score) is not int or not 0 <= score <= _DRIVE_SCORE for score in scores):
        raise ValueError("neutral-frame score left its calibrated range")
    cells = tuple(range(len(scores))) if candidate_cell_ids is None else candidate_cell_ids
    if len(cells) != len(scores) or len(set(cells)) != len(cells):
        raise ValueError("candidate cells must be unique and aligned")
    all_scores = (*scores, _TONIC_STOP_SCORE)
    result = _CELL_RACE.race_many(
        _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)][None, :],
        require_unique=False,
    )[0]
    physical_steps = len(all_scores) * _RACE_STEPS
    active_synapses = 1 + sum(score > 0 for score in scores)
    collision_step: int | None = None
    stop_reason: str | None = None
    column: int | None = result.winner_cell
    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            _SCORE_CURRENTS[np.asarray([[_TONIC_STOP_SCORE]], dtype=np.intp)],
            require_unique=False,
        )[0]
        collision_step = relay.spike_step
        physical_steps += _RACE_STEPS
        active_synapses += 1
        stop_reason = "collision"
        column = None
    elif result.winner_cell == len(scores):
        stop_reason = "tonic_stop"
        column = None
    winner_cell = None if column is None else cells[column]
    record = {
        "active_synapses": active_synapses,
        "candidate_cell_ids": cells,
        "candidate_scores": scores,
        "collision_stop_spike_step": collision_step,
        "physical_cell_steps": physical_steps,
        "population": population,
        "simultaneous_spike_candidates": result.simultaneous_spike_candidates,
        "spike_step": result.spike_step,
        "stop_reason": stop_reason,
        "stopped": column is None,
        "tonic_stop_cell_id": len(scores),
        "winner_cell_id": winner_cell,
    }
    trace = NeutralFrameRaceTrace(
        **record,
        sha256=_digest(_RACE_NAMESPACE, record),
    )
    return _RaceChoice(column=column, trace=trace)


def _support_signal(
    *,
    population: str,
    support_roots: tuple[str, ...],
    enabled: bool,
    races: list[NeutralFrameRaceTrace],
) -> bool:
    """Threshold distinct synaptic support against the tonic STOP cell."""

    score = min(len(set(support_roots)), _DRIVE_SCORE) if enabled else 0
    choice = _physical_choice(population=population, scores=(score,))
    races.append(choice.trace)
    return choice.column == 0


def _conjunction_signal(
    *,
    population: str,
    inputs: tuple[bool, ...],
    races: list[NeutralFrameRaceTrace],
) -> bool:
    """Route all upstream spikes through one fixed all-input dendrite."""

    choice = _physical_choice(
        population=population,
        scores=(_DRIVE_SCORE if inputs and all(inputs) else 0,),
    )
    races.append(choice.trace)
    return choice.column == 0


@dataclass(frozen=True, slots=True)
class _Lexeme:
    value: bytes | None
    value_sha256: str
    pointer: SparseBlockPointer


def _lexeme(value: bytes, *, retain_value: bool) -> _Lexeme:
    if not value:
        raise ValueError("frame fillers must be non-empty")
    return _Lexeme(
        value=value if retain_value else None,
        value_sha256=_bytes_root(_LEXEME_NAMESPACE, value),
        pointer=_SUBSTRATE.project(_LEXEME_EVENT + value),
    )


def _lexeme_record(value: _Lexeme) -> dict[str, object]:
    return {
        "pointer": _pointer_record(value.pointer),
        "value_sha256": value.value_sha256,
    }


@dataclass(frozen=True, slots=True)
class _FrameEvidence:
    frame_key_sha256: str
    frame_pointer: SparseBlockPointer
    token_count: int
    slot_indices: tuple[int, int]
    filler0: _Lexeme
    filler1: _Lexeme
    forward_pair_pointer: SparseBlockPointer
    reverse_pair_pointer: SparseBlockPointer
    provenance_root_sha256: str
    document_root_sha256: str
    binding_trace_sha256s: tuple[str, str]
    sha256: str


def _evidence_record(
    evidence: _FrameEvidence,
    *,
    include_sha256: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "binding_trace_sha256s": evidence.binding_trace_sha256s,
        "document_root_sha256": evidence.document_root_sha256,
        "filler0": _lexeme_record(evidence.filler0),
        "filler1": _lexeme_record(evidence.filler1),
        "forward_pair_pointer": _pointer_record(evidence.forward_pair_pointer),
        "frame_key_sha256": evidence.frame_key_sha256,
        "frame_pointer": _pointer_record(evidence.frame_pointer),
        "provenance_root_sha256": evidence.provenance_root_sha256,
        "reverse_pair_pointer": _pointer_record(evidence.reverse_pair_pointer),
        "slot_indices": evidence.slot_indices,
        "token_count": evidence.token_count,
    }
    if include_sha256:
        record["sha256"] = evidence.sha256
    return record


@dataclass(frozen=True, slots=True)
class _Frame:
    key_sha256: str
    pointer: SparseBlockPointer
    token_count: int
    slot_indices: tuple[int, int]
    evidence: tuple[_FrameEvidence, ...]
    promoted: bool
    promotion_trace_sha256: str


def _frame_record(frame: _Frame) -> dict[str, object]:
    return {
        "evidence": tuple(
            _evidence_record(item, include_sha256=True) for item in frame.evidence
        ),
        "key_sha256": frame.key_sha256,
        "pointer": _pointer_record(frame.pointer),
        "promoted": frame.promoted,
        "promotion_trace_sha256": frame.promotion_trace_sha256,
        "slot_indices": frame.slot_indices,
        "token_count": frame.token_count,
    }


@dataclass(frozen=True, slots=True)
class _FrameRoute:
    source_frame_key_sha256: str
    target_frame_key_sha256: str
    orientation_pointer: SparseBlockPointer
    link_pointer: SparseBlockPointer
    support_root_sha256s: tuple[str, ...]
    orientation_trace_sha256: str
    binding_trace_sha256: str
    sha256: str


def _route_record(route: _FrameRoute, *, include_sha256: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "binding_trace_sha256": route.binding_trace_sha256,
        "link_pointer": _pointer_record(route.link_pointer),
        "orientation_pointer": _pointer_record(route.orientation_pointer),
        "orientation_trace_sha256": route.orientation_trace_sha256,
        "source_frame_key_sha256": route.source_frame_key_sha256,
        "support_root_sha256s": route.support_root_sha256s,
        "target_frame_key_sha256": route.target_frame_key_sha256,
    }
    if include_sha256:
        record["sha256"] = route.sha256
    return record


@dataclass(frozen=True, slots=True)
class NeutralFrameTrainingSummary:
    consolidation_episode_count: int
    retained_evidence_count: int
    promoted_frame_count: int
    directional_route_count: int
    retained_raw_document_count: int
    retained_full_sentence_count: int
    retained_lexical_byte_count: int
    learning_physical_cell_steps: int
    learning_active_synapses: int


@dataclass(frozen=True, slots=True)
class NeutralFrameDecision:
    status: str
    emitted: bytes | None
    stopped: bool
    stop_stage: str | None
    stop_reason: str | None
    producer_checkpoint_sha256: str
    support_path_count: int
    cell_races: tuple[NeutralFrameRaceTrace, ...]
    cleanup_races: tuple[SparseCleanupRaceTrace, ...]
    transform_races: tuple[SparseTransformRaceTrace, ...]
    inference_physical_cell_steps: int
    inference_active_synapses: int
    sha256: str


def _state_record(
    *,
    frames: tuple[_Frame, ...],
    routes: tuple[_FrameRoute, ...],
    summary: NeutralFrameTrainingSummary,
) -> dict[str, object]:
    return {
        "frames": tuple(_frame_record(frame) for frame in frames),
        "routes": tuple(_route_record(route, include_sha256=True) for route in routes),
        "summary": {
            "consolidation_episode_count": summary.consolidation_episode_count,
            "directional_route_count": summary.directional_route_count,
            "learning_active_synapses": summary.learning_active_synapses,
            "learning_physical_cell_steps": summary.learning_physical_cell_steps,
            "promoted_frame_count": summary.promoted_frame_count,
            "retained_evidence_count": summary.retained_evidence_count,
            "retained_full_sentence_count": summary.retained_full_sentence_count,
            "retained_lexical_byte_count": summary.retained_lexical_byte_count,
            "retained_raw_document_count": summary.retained_raw_document_count,
        },
    }


@dataclass(frozen=True, slots=True, init=False)
class FrozenNeutralNeuronalFrameNetwork:
    """Immutable neutral frame checkpoint with read-only neuronal inference."""

    _frames: tuple[_Frame, ...]
    _routes: tuple[_FrameRoute, ...]
    checkpoint_sha256: str
    training_summary: NeutralFrameTrainingSummary

    def __init__(
        self,
        token: object,
        *,
        frames: tuple[_Frame, ...],
        routes: tuple[_FrameRoute, ...],
        checkpoint_sha256: str,
        training_summary: NeutralFrameTrainingSummary,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use start() and consolidate()")
        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_routes", routes)
        object.__setattr__(self, "checkpoint_sha256", checkpoint_sha256)
        object.__setattr__(self, "training_summary", training_summary)

    def authenticate(self) -> None:
        expected = _digest(
            _CHECKPOINT_NAMESPACE,
            _state_record(
                frames=self._frames,
                routes=self._routes,
                summary=self.training_summary,
            ),
        )
        if expected != self.checkpoint_sha256:
            raise ValueError("neutral-frame checkpoint custody changed")

    def consolidate(
        self,
        documents: tuple[OpaqueDocument, ...],
        *,
        ablation: NeutralFrameAblation = NeutralFrameAblation.NONE,
    ) -> Self:
        """Return a new checkpoint after one neutral document episode."""

        if type(self) is not FrozenNeutralNeuronalFrameNetwork:
            raise TypeError("network must be one exact frozen checkpoint")
        if type(documents) is not tuple or not documents:
            raise ValueError("documents must be a non-empty tuple")
        if any(not isinstance(document, OpaqueDocument) for document in documents):
            raise TypeError("documents must contain OpaqueDocument values")
        if not isinstance(ablation, NeutralFrameAblation):
            raise TypeError("ablation must be NeutralFrameAblation")
        self.authenticate()

        observed, binding_races = _observe_evidence(documents)
        by_root = {
            evidence.sha256: evidence
            for frame in self._frames
            for evidence in frame.evidence
        }
        for evidence in observed:
            by_root.setdefault(evidence.sha256, evidence)
        all_evidence = tuple(by_root[root] for root in sorted(by_root))
        frames, promotion_races = _promote_frames(
            all_evidence,
            ablation=ablation,
        )
        routes, orientation_races, route_binding_races = _learn_routes(
            frames,
            ablation=ablation,
        )
        all_cell_races = (*promotion_races, *orientation_races)
        all_binding_races = (*binding_races, *route_binding_races)
        return _freeze(
            frames=frames,
            routes=routes,
            consolidation_episode_count=(
                self.training_summary.consolidation_episode_count + 1
            ),
            learning_physical_cell_steps=(
                self.training_summary.learning_physical_cell_steps
                + sum(race.physical_cell_steps for race in all_cell_races)
                + sum(race.physical_cell_steps for race in all_binding_races)
            ),
            learning_active_synapses=(
                self.training_summary.learning_active_synapses
                + sum(race.active_synapses for race in all_cell_races)
                + sum(race.active_synapses for race in all_binding_races)
            ),
        )

    def respond(
        self,
        query: bytes,
        documents: tuple[OpaqueDocument, ...],
        *,
        ablation: NeutralFrameAblation = NeutralFrameAblation.NONE,
    ) -> NeutralFrameDecision:
        """Infer one unique missing filler without modifying the checkpoint."""

        if not isinstance(query, bytes) or not query:
            raise ValueError("query must be non-empty bytes")
        if type(documents) is not tuple:
            raise TypeError("documents must be a tuple")
        if any(not isinstance(document, OpaqueDocument) for document in documents):
            raise TypeError("documents must contain OpaqueDocument values")
        if not isinstance(ablation, NeutralFrameAblation):
            raise TypeError("ablation must be NeutralFrameAblation")
        self.authenticate()
        return _respond(self, query, documents, ablation=ablation)


def _freeze(
    *,
    frames: tuple[_Frame, ...],
    routes: tuple[_FrameRoute, ...],
    consolidation_episode_count: int,
    learning_physical_cell_steps: int,
    learning_active_synapses: int,
) -> FrozenNeutralNeuronalFrameNetwork:
    ordered_frames = tuple(sorted(frames, key=lambda frame: frame.key_sha256))
    ordered_routes = tuple(
        sorted(
            routes,
            key=lambda route: (
                route.source_frame_key_sha256,
                route.target_frame_key_sha256,
            ),
        )
    )
    summary = NeutralFrameTrainingSummary(
        consolidation_episode_count=consolidation_episode_count,
        retained_evidence_count=sum(len(frame.evidence) for frame in ordered_frames),
        promoted_frame_count=sum(frame.promoted for frame in ordered_frames),
        directional_route_count=len(ordered_routes),
        retained_raw_document_count=0,
        retained_full_sentence_count=0,
        retained_lexical_byte_count=0,
        learning_physical_cell_steps=learning_physical_cell_steps,
        learning_active_synapses=learning_active_synapses,
    )
    checkpoint = _digest(
        _CHECKPOINT_NAMESPACE,
        _state_record(frames=ordered_frames, routes=ordered_routes, summary=summary),
    )
    return FrozenNeutralNeuronalFrameNetwork(
        _CONSTRUCTION_TOKEN,
        frames=ordered_frames,
        routes=ordered_routes,
        checkpoint_sha256=checkpoint,
        training_summary=summary,
    )


def start() -> FrozenNeutralNeuronalFrameNetwork:
    """Return the authenticated empty neutral frame network."""

    return _freeze(
        frames=(),
        routes=(),
        consolidation_episode_count=0,
        learning_physical_cell_steps=0,
        learning_active_synapses=0,
    )


def _sentence_lexemes(
    sentence: OpaqueSentenceOccurrence,
    stream: bytes,
) -> tuple[bytes, ...]:
    return tuple(
        stream[token.byte_start : token.byte_end]
        for token in sentence.token_occurrences
    )


def _frame_descriptor(
    tokens: tuple[bytes, ...],
    slot_indices: tuple[int, int],
) -> bytes:
    literals = tuple(
        (index, token.hex())
        for index, token in enumerate(tokens)
        if index not in slot_indices
    )
    return _canonical_json(
        {
            "literals": literals,
            "slot_indices": slot_indices,
            "token_count": len(tokens),
        }
    )


def _make_evidence(
    *,
    tokens: tuple[bytes, ...],
    slot_indices: tuple[int, int],
    provenance_id: bytes,
    document: OpaqueDocument,
) -> tuple[_FrameEvidence, tuple[SparseTransformRaceTrace, SparseTransformRaceTrace]]:
    descriptor = _frame_descriptor(tokens, slot_indices)
    frame_key = _digest(_FRAME_NAMESPACE, {"descriptor_hex": descriptor.hex()})
    frame_pointer = _SUBSTRATE.project(_FRAME_EVENT + descriptor)
    filler0 = _lexeme(tokens[slot_indices[0]], retain_value=False)
    filler1 = _lexeme(tokens[slot_indices[1]], retain_value=False)
    forward = _SUBSTRATE.bind(filler0.pointer, filler1.pointer)
    reverse = _SUBSTRATE.bind(filler1.pointer, filler0.pointer)
    if forward.pointer is None or reverse.pointer is None:
        raise RuntimeError("unablated pair binding selected tonic STOP")
    provenance_root = _bytes_root(_PROVENANCE_NAMESPACE, provenance_id)
    document_root = _digest(
        _DOCUMENT_NAMESPACE,
        {
            "content_sha256": _bytes_root(_DOCUMENT_NAMESPACE, document.content),
            "document_id_sha256": _bytes_root(
                _DOCUMENT_NAMESPACE,
                document.document_id,
            ),
            "provenance_root_sha256": provenance_root,
        },
    )
    provisional = _FrameEvidence(
        frame_key_sha256=frame_key,
        frame_pointer=frame_pointer,
        token_count=len(tokens),
        slot_indices=slot_indices,
        filler0=filler0,
        filler1=filler1,
        forward_pair_pointer=forward.pointer,
        reverse_pair_pointer=reverse.pointer,
        provenance_root_sha256=provenance_root,
        document_root_sha256=document_root,
        binding_trace_sha256s=(forward.trace.sha256, reverse.trace.sha256),
        sha256="",
    )
    evidence = replace(
        provisional,
        sha256=_digest(
            _EVIDENCE_NAMESPACE,
            _evidence_record(provisional, include_sha256=False),
        ),
    )
    return evidence, (forward.trace, reverse.trace)


def _observe_evidence(
    documents: tuple[OpaqueDocument, ...],
) -> tuple[tuple[_FrameEvidence, ...], tuple[SparseTransformRaceTrace, ...]]:
    tape = OpaqueDocumentEventTape.observe(b"_", documents)
    evidence: list[_FrameEvidence] = []
    races: list[SparseTransformRaceTrace] = []
    for document_sentences in tape.document_sentences:
        for sentence in document_sentences:
            if sentence.document_index is None or sentence.provenance_id is None:
                raise RuntimeError("document sentence lost its opaque custody")
            document = documents[sentence.document_index]
            tokens = _sentence_lexemes(sentence, document.content)
            if len(tokens) < 3:
                continue
            for slot_indices in combinations(range(len(tokens)), 2):
                item, item_races = _make_evidence(
                    tokens=tokens,
                    slot_indices=slot_indices,
                    provenance_id=sentence.provenance_id,
                    document=document,
                )
                evidence.append(item)
                races.extend(item_races)
    ordered = tuple(sorted(evidence, key=lambda item: item.sha256))
    return ordered, tuple(races)


def _promote_frames(
    evidence: tuple[_FrameEvidence, ...],
    *,
    ablation: NeutralFrameAblation,
) -> tuple[tuple[_Frame, ...], tuple[NeutralFrameRaceTrace, ...]]:
    grouped: dict[str, list[_FrameEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.frame_key_sha256, []).append(item)
    frames: list[_Frame] = []
    races: list[NeutralFrameRaceTrace] = []
    for key in sorted(grouped):
        rows = tuple(sorted(grouped[key], key=lambda item: item.sha256))
        first = rows[0]
        if any(
            row.frame_pointer != first.frame_pointer
            or row.token_count != first.token_count
            or row.slot_indices != first.slot_indices
            for row in rows[1:]
        ):
            raise RuntimeError("one frame key mapped to different structures")
        enabled = ablation is not NeutralFrameAblation.FRAME_PROMOTION_OFF
        prefix = f"learning/frame/{key}"
        support_spikes = (
            _support_signal(
                population=f"{prefix}/provenance_support",
                support_roots=tuple(row.provenance_root_sha256 for row in rows),
                enabled=enabled,
                races=races,
            ),
            _support_signal(
                population=f"{prefix}/pair_support",
                support_roots=tuple(row.forward_pair_pointer.sha256 for row in rows),
                enabled=enabled,
                races=races,
            ),
            _support_signal(
                population=f"{prefix}/slot_0_variability",
                support_roots=tuple(row.filler0.value_sha256 for row in rows),
                enabled=enabled,
                races=races,
            ),
            _support_signal(
                population=f"{prefix}/slot_1_variability",
                support_roots=tuple(row.filler1.value_sha256 for row in rows),
                enabled=enabled,
                races=races,
            ),
        )
        promoted = _conjunction_signal(
            population=f"{prefix}/promotion",
            inputs=support_spikes,
            races=races,
        )
        frames.append(
            _Frame(
                key_sha256=key,
                pointer=first.frame_pointer,
                token_count=first.token_count,
                slot_indices=first.slot_indices,
                evidence=rows,
                promoted=promoted,
                promotion_trace_sha256=races[-1].sha256,
            )
        )
    return tuple(frames), tuple(races)


@dataclass(frozen=True, slots=True)
class _OrientationSupport:
    pair_roots: tuple[str, ...]
    left_provenance_roots: tuple[str, ...]
    right_provenance_roots: tuple[str, ...]


def _orientation_support(
    left: _Frame,
    right: _Frame,
    *,
    swapped: bool,
) -> _OrientationSupport:
    by_pair: dict[str, tuple[set[str], set[str]]] = {}
    for left_row in left.evidence:
        for right_row in right.evidence:
            if left_row.provenance_root_sha256 == right_row.provenance_root_sha256:
                continue
            target_pointer = (
                right_row.reverse_pair_pointer
                if swapped
                else right_row.forward_pair_pointer
            )
            if left_row.forward_pair_pointer.sha256 != target_pointer.sha256:
                continue
            left_roots, right_roots = by_pair.setdefault(
                left_row.forward_pair_pointer.sha256,
                (set(), set()),
            )
            left_roots.add(left_row.provenance_root_sha256)
            right_roots.add(right_row.provenance_root_sha256)
    pair_roots: list[str] = []
    left_provenance: set[str] = set()
    right_provenance: set[str] = set()
    for pair_pointer in sorted(by_pair):
        left_roots, right_roots = by_pair[pair_pointer]
        pair_roots.append(
            _digest(
                _PAIR_NAMESPACE,
                {
                    "left_provenance_roots": tuple(sorted(left_roots)),
                    "orientation": "SWAPPED" if swapped else "SAME",
                    "pair_pointer_sha256": pair_pointer,
                    "right_provenance_roots": tuple(sorted(right_roots)),
                },
            )
        )
        left_provenance.update(left_roots)
        right_provenance.update(right_roots)
    return _OrientationSupport(
        pair_roots=tuple(pair_roots),
        left_provenance_roots=tuple(sorted(left_provenance)),
        right_provenance_roots=tuple(sorted(right_provenance)),
    )


def _orientation_signal(
    *,
    population: str,
    support: _OrientationSupport,
    enabled: bool,
    races: list[NeutralFrameRaceTrace],
) -> bool:
    inputs = (
        _support_signal(
            population=f"{population}/pair_support",
            support_roots=support.pair_roots,
            enabled=enabled,
            races=races,
        ),
        _support_signal(
            population=f"{population}/left_provenance_support",
            support_roots=support.left_provenance_roots,
            enabled=enabled,
            races=races,
        ),
        _support_signal(
            population=f"{population}/right_provenance_support",
            support_roots=support.right_provenance_roots,
            enabled=enabled,
            races=races,
        ),
    )
    return _conjunction_signal(
        population=f"{population}/support_conjunction",
        inputs=inputs,
        races=races,
    )


def _make_route(
    *,
    source: _Frame,
    target: _Frame,
    orientation_pointer: SparseBlockPointer,
    support_roots: tuple[str, ...],
    orientation_trace: NeutralFrameRaceTrace,
) -> tuple[_FrameRoute, SparseTransformRaceTrace]:
    binding = _SUBSTRATE.bind(source.pointer, target.pointer)
    if binding.pointer is None:
        raise RuntimeError("unablated frame-route binding selected tonic STOP")
    provisional = _FrameRoute(
        source_frame_key_sha256=source.key_sha256,
        target_frame_key_sha256=target.key_sha256,
        orientation_pointer=orientation_pointer,
        link_pointer=binding.pointer,
        support_root_sha256s=support_roots,
        orientation_trace_sha256=orientation_trace.sha256,
        binding_trace_sha256=binding.trace.sha256,
        sha256="",
    )
    route = _FrameRoute(
        source_frame_key_sha256=provisional.source_frame_key_sha256,
        target_frame_key_sha256=provisional.target_frame_key_sha256,
        orientation_pointer=provisional.orientation_pointer,
        link_pointer=provisional.link_pointer,
        support_root_sha256s=provisional.support_root_sha256s,
        orientation_trace_sha256=provisional.orientation_trace_sha256,
        binding_trace_sha256=provisional.binding_trace_sha256,
        sha256=_digest(
            _ROUTE_NAMESPACE,
            _route_record(provisional, include_sha256=False),
        ),
    )
    return route, binding.trace


def _learn_routes(
    frames: tuple[_Frame, ...],
    *,
    ablation: NeutralFrameAblation,
) -> tuple[
    tuple[_FrameRoute, ...],
    tuple[NeutralFrameRaceTrace, ...],
    tuple[SparseTransformRaceTrace, ...],
]:
    promoted = tuple(frame for frame in frames if frame.promoted)
    routes: list[_FrameRoute] = []
    orientation_races: list[NeutralFrameRaceTrace] = []
    binding_races: list[SparseTransformRaceTrace] = []
    enabled = ablation is not NeutralFrameAblation.ASSOCIATION_LEARNING_OFF
    for left, right in combinations(promoted, 2):
        same_support = _orientation_support(left, right, swapped=False)
        swapped_support = _orientation_support(left, right, swapped=True)
        prefix = f"learning/association/{left.key_sha256}/{right.key_sha256}"
        same_spike = _orientation_signal(
            population=f"{prefix}/same",
            support=same_support,
            enabled=enabled,
            races=orientation_races,
        )
        swapped_spike = _orientation_signal(
            population=f"{prefix}/swapped",
            support=swapped_support,
            enabled=enabled,
            races=orientation_races,
        )
        choice = _physical_choice(
            population=f"{prefix}/orientation",
            scores=(
                _DRIVE_SCORE if same_spike else 0,
                _DRIVE_SCORE if swapped_spike else 0,
            ),
        )
        orientation_races.append(choice.trace)
        if choice.column is None:
            continue
        orientation_pointer = (
            _SAME_ORIENTATION_POINTER
            if choice.column == 0
            else _SWAPPED_ORIENTATION_POINTER
        )
        support = same_support if choice.column == 0 else swapped_support
        for source, target in ((left, right), (right, left)):
            route, binding_race = _make_route(
                source=source,
                target=target,
                orientation_pointer=orientation_pointer,
                support_roots=support.pair_roots,
                orientation_trace=choice.trace,
            )
            routes.append(route)
            binding_races.append(binding_race)
    return tuple(routes), tuple(orientation_races), tuple(binding_races)


@dataclass(frozen=True, slots=True)
class _FullMatch:
    frame: _Frame
    filler0: _Lexeme
    filler1: _Lexeme


@dataclass(frozen=True, slots=True)
class _PartialMatch:
    frame: _Frame
    known_filler: _Lexeme


def _cleanup_match(
    projected: SparseBlockPointer,
    frames: tuple[_Frame, ...],
    *,
    disabled: bool,
) -> tuple[_Frame | None, SparseCleanupRaceTrace | None]:
    if not frames:
        return None, None
    state = SparsePointerSuperposition.empty().accumulate(projected)
    decision = _SUBSTRATE.cleanup(
        state,
        tuple(frame.pointer for frame in frames),
        ablation=(
            SparseRoleFillerAblation.CLEANUP_OFF
            if disabled
            else SparseRoleFillerAblation.NONE
        ),
    )
    if decision.pointer is None:
        return None, decision.trace
    matches = tuple(frame for frame in frames if frame.pointer == decision.pointer)
    return (matches[0] if len(matches) == 1 else None), decision.trace


def _full_matches(
    tape: OpaqueDocumentEventTape,
    documents: tuple[OpaqueDocument, ...],
    frames: tuple[_Frame, ...],
    *,
    disabled: bool,
) -> tuple[tuple[_FullMatch, ...], tuple[SparseCleanupRaceTrace, ...]]:
    matches: list[_FullMatch] = []
    races: list[SparseCleanupRaceTrace] = []
    for sentence_stream in tape.document_sentences:
        for sentence in sentence_stream:
            if sentence.document_index is None:
                continue
            tokens = _sentence_lexemes(
                sentence,
                documents[sentence.document_index].content,
            )
            if len(tokens) < 3:
                continue
            for slots in combinations(range(len(tokens)), 2):
                descriptor = _frame_descriptor(tokens, slots)
                projected = _SUBSTRATE.project(_FRAME_EVENT + descriptor)
                frame, race = _cleanup_match(projected, frames, disabled=disabled)
                if race is not None:
                    races.append(race)
                if frame is None:
                    continue
                matches.append(
                    _FullMatch(
                        frame=frame,
                        filler0=_lexeme(tokens[slots[0]], retain_value=True),
                        filler1=_lexeme(tokens[slots[1]], retain_value=True),
                    )
                )
    return tuple(matches), tuple(races)


def _partial_matches(
    tape: OpaqueDocumentEventTape,
    query: bytes,
    frames: tuple[_Frame, ...],
    *,
    disabled: bool,
) -> tuple[tuple[_PartialMatch, ...], tuple[SparseCleanupRaceTrace, ...]]:
    matches: list[_PartialMatch] = []
    races: list[SparseCleanupRaceTrace] = []
    for sentence in tape.query_sentences:
        observed = _sentence_lexemes(sentence, query)
        full_count = len(observed) + 1
        if full_count < 3:
            continue
        for slots in combinations(range(full_count), 2):
            for missing_slot in (0, 1):
                missing_position = slots[missing_slot]
                expanded: list[bytes] = []
                observed_index = 0
                known_filler: bytes | None = None
                known_slot = 1 - missing_slot
                for position in range(full_count):
                    if position == missing_position:
                        expanded.append(b"\x00missing")
                        continue
                    token = observed[observed_index]
                    observed_index += 1
                    expanded.append(token)
                    if position == slots[known_slot]:
                        known_filler = token
                if known_filler is None:
                    raise RuntimeError("partial frame lost its observed slot")
                descriptor_tokens = tuple(expanded)
                descriptor = _frame_descriptor(descriptor_tokens, slots)
                projected = _SUBSTRATE.project(_FRAME_EVENT + descriptor)
                frame, race = _cleanup_match(projected, frames, disabled=disabled)
                if race is not None:
                    races.append(race)
                if frame is None:
                    continue
                matches.append(
                    _PartialMatch(
                        frame=frame,
                        known_filler=_lexeme(known_filler, retain_value=True),
                    )
                )
    return tuple(matches), tuple(races)


def _decision(
    *,
    network: FrozenNeutralNeuronalFrameNetwork,
    emitted: bytes | None,
    support_path_count: int,
    cell_races: tuple[NeutralFrameRaceTrace, ...],
    cleanup_races: tuple[SparseCleanupRaceTrace, ...],
    transform_races: tuple[SparseTransformRaceTrace, ...],
) -> NeutralFrameDecision:
    motor = cell_races[-1]
    stopped = emitted is None
    record = {
        "cell_race_sha256s": tuple(race.sha256 for race in cell_races),
        "cleanup_race_sha256s": tuple(race.sha256 for race in cleanup_races),
        "emitted_hex": None if emitted is None else emitted.hex(),
        "producer_checkpoint_sha256": network.checkpoint_sha256,
        "status": "STOP" if stopped else "EMIT",
        "stop_reason": None if not stopped else f"motor:{motor.stop_reason}",
        "support_path_count": support_path_count,
        "transform_race_sha256s": tuple(race.sha256 for race in transform_races),
    }
    return NeutralFrameDecision(
        status=record["status"],
        emitted=emitted,
        stopped=stopped,
        stop_stage="motor" if stopped else None,
        stop_reason=record["stop_reason"],
        producer_checkpoint_sha256=network.checkpoint_sha256,
        support_path_count=support_path_count,
        cell_races=cell_races,
        cleanup_races=cleanup_races,
        transform_races=transform_races,
        inference_physical_cell_steps=(
            sum(race.physical_cell_steps for race in cell_races)
            + sum(race.physical_cell_steps for race in cleanup_races)
            + sum(race.physical_cell_steps for race in transform_races)
        ),
        inference_active_synapses=(
            sum(race.active_synapses for race in cell_races)
            + sum(race.active_synapses for race in cleanup_races)
            + sum(race.active_synapses for race in transform_races)
        ),
        sha256=_digest(_DECISION_NAMESPACE, record),
    )


def _respond(
    network: FrozenNeutralNeuronalFrameNetwork,
    query: bytes,
    documents: tuple[OpaqueDocument, ...],
    *,
    ablation: NeutralFrameAblation,
) -> NeutralFrameDecision:
    tape = OpaqueDocumentEventTape.observe(query, documents)
    frames = tuple(frame for frame in network._frames if frame.promoted)
    cleanup_races: list[SparseCleanupRaceTrace] = []
    transform_races: list[SparseTransformRaceTrace] = []
    cell_races: list[NeutralFrameRaceTrace] = []
    frame_disabled = ablation is NeutralFrameAblation.FRAME_MATCH_OFF
    full, full_races = _full_matches(
        tape,
        documents,
        frames,
        disabled=frame_disabled,
    )
    partial, partial_races = _partial_matches(
        tape,
        query,
        frames,
        disabled=frame_disabled,
    )
    cleanup_races.extend(full_races)
    cleanup_races.extend(partial_races)

    frame_by_key = {frame.key_sha256: frame for frame in frames}
    route_by_pair: dict[tuple[str, str], list[_FrameRoute]] = {}
    for route in network._routes:
        route_by_pair.setdefault(
            (
                route.source_frame_key_sha256,
                route.target_frame_key_sha256,
            ),
            [],
        ).append(route)

    support_paths: list[_Lexeme] = []
    for source in full:
        for target in partial:
            routes = route_by_pair.get(
                (source.frame.key_sha256, target.frame.key_sha256),
                (),
            )
            for route in routes:
                readout = _SUBSTRATE.unbind(
                    source.frame.pointer,
                    route.link_pointer,
                    ablation=(
                        SparseRoleFillerAblation.BINDING_OFF
                        if ablation is NeutralFrameAblation.READOUT_OFF
                        else SparseRoleFillerAblation.NONE
                    ),
                )
                transform_races.append(readout.trace)
                if readout.pointer is None:
                    continue
                recovered, recovery_trace = _cleanup_match(
                    readout.pointer,
                    frames,
                    disabled=False,
                )
                if recovery_trace is not None:
                    cleanup_races.append(recovery_trace)
                expected_target = frame_by_key.get(route.target_frame_key_sha256)
                if recovered is None or recovered != expected_target:
                    continue
                orientation_state = SparsePointerSuperposition.empty().accumulate(
                    route.orientation_pointer
                )
                orientation = _SUBSTRATE.cleanup(
                    orientation_state,
                    (_SAME_ORIENTATION_POINTER, _SWAPPED_ORIENTATION_POINTER),
                )
                cleanup_races.append(orientation.trace)
                if orientation.pointer == _SAME_ORIENTATION_POINTER:
                    target_pair = (source.filler0, source.filler1)
                elif orientation.pointer == _SWAPPED_ORIENTATION_POINTER:
                    target_pair = (source.filler1, source.filler0)
                else:
                    continue

                forward_pair = _SUBSTRATE.bind(
                    target_pair[0].pointer,
                    target_pair[1].pointer,
                )
                reverse_pair = _SUBSTRATE.bind(
                    target_pair[1].pointer,
                    target_pair[0].pointer,
                )
                transform_races.extend((forward_pair.trace, reverse_pair.trace))
                if forward_pair.pointer is None or reverse_pair.pointer is None:
                    continue
                alignment_state = SparsePointerSuperposition.empty().accumulate(
                    target.known_filler.pointer
                )
                aligned = _SUBSTRATE.cleanup(
                    alignment_state,
                    (target_pair[0].pointer, target_pair[1].pointer),
                    ablation=(
                        SparseRoleFillerAblation.CLEANUP_OFF
                        if ablation is NeutralFrameAblation.ALIGNMENT_OFF
                        else SparseRoleFillerAblation.NONE
                    ),
                )
                cleanup_races.append(aligned.trace)
                if aligned.pointer is None:
                    continue
                if aligned.pointer == target_pair[0].pointer:
                    pair_link = forward_pair.pointer
                elif aligned.pointer == target_pair[1].pointer:
                    pair_link = reverse_pair.pointer
                else:
                    continue
                opposite = _SUBSTRATE.unbind(aligned.pointer, pair_link)
                transform_races.append(opposite.trace)
                if opposite.pointer is None:
                    continue
                output_state = SparsePointerSuperposition.empty().accumulate(
                    opposite.pointer
                )
                output = _SUBSTRATE.cleanup(
                    output_state,
                    (target_pair[0].pointer, target_pair[1].pointer),
                )
                cleanup_races.append(output.trace)
                if output.pointer is None:
                    continue
                emitted_lexemes = tuple(
                    lexeme for lexeme in target_pair if lexeme.pointer == output.pointer
                )
                unique_lexemes = {
                    lexeme.value_sha256: lexeme for lexeme in emitted_lexemes
                }
                if len(unique_lexemes) == 1:
                    support_paths.append(next(iter(unique_lexemes.values())))

    unique_outputs: dict[str, _Lexeme] = {}
    for lexeme in support_paths:
        existing = unique_outputs.get(lexeme.value_sha256)
        if existing is not None and existing.value != lexeme.value:
            raise RuntimeError("one lexical root mapped to different bytes")
        unique_outputs[lexeme.value_sha256] = lexeme
    ordered_outputs = tuple(unique_outputs[root] for root in sorted(unique_outputs))
    motor = _physical_choice(
        population="response/motor",
        scores=tuple(
            0 if ablation is NeutralFrameAblation.MOTOR_OFF else _DRIVE_SCORE
            for _ in ordered_outputs
        ),
    )
    cell_races.append(motor.trace)
    emitted = None if motor.column is None else ordered_outputs[motor.column].value
    return _decision(
        network=network,
        emitted=emitted,
        support_path_count=len(support_paths),
        cell_races=tuple(cell_races),
        cleanup_races=tuple(cleanup_races),
        transform_races=tuple(transform_races),
    )
