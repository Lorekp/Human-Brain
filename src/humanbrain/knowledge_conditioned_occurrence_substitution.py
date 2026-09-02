"""Frozen neuronal bridge from retained opaque facts to one workspace substitution.

The public inference surface accepts one complete opaque observation and an
ablation only.  A frozen persistent document learner is owned by the bridge and
is invoked internally.  Its physical memory winner is converted into a sealed
pointer broadcast before either factor is allowed to affect the workspace.

Two evaluator-authenticated complete transitions train only a reusable
factor-zero-to-occurrence walk.  A fixed two-state SUBSTITUTE/END cell program
provides actuation scaffolding.  A training-only byte diff verifies the
demonstrated occurrence, but no path, span, replacement, answer, or repaired
workspace is retained.  At inference the selected file is inherited from a
live occurrence cell and the inserted bytes come from the retained lexeme
selected by the fact assembly.

This is a narrow opaque occurrence-substitution mechanism.  It is not general
language understanding, syntax editing, planning, or online growth.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Self

import numpy as np

from humanbrain import persistent_neuronal_document_learner as _memory
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
from humanbrain.opaque_document_event_tape import OpaqueDocument, OpaqueDocumentEventTape
from humanbrain.persistent_neuronal_document_learner import (
    FrozenPersistentNeuronalDocumentLearner,
    PersistentDocumentAblation,
    PersistentDocumentDecision,
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
from humanbrain.unified_opaque_observation_tape import UnifiedOpaqueObservationTape
from humanbrain.variability_aware_segment_binder import (
    VariabilityAwareSegmentBinder,
    VariabilityBinderAblation,
    VariabilityBindingDecision,
    VariabilityRaceTrace,
    VariabilitySegmentDemonstration,
)

__all__ = [
    "CLAIM_BOUNDARY",
    "FrozenKnowledgeConditionedOccurrenceSubstitution",
    "KnowledgeSubstitutionAblation",
    "KnowledgeSubstitutionDecision",
    "KnowledgeSubstitutionExecutionAuthority",
    "KnowledgeSubstitutionExecutionReceipt",
    "KnowledgeSubstitutionRaceTrace",
    "KnowledgeSubstitutionSuccess",
    "KnowledgeSubstitutionTrainingSummary",
    "grow",
]


CLAIM_BOUNDARY = (
    "frozen two-transition transfer from one physically retrieved opaque fact "
    "to one exact lexical workspace substitution; persistent-reader surfaces, "
    "training-only full-state differencing, and fixed SUBSTITUTE/END operations "
    "remain scaffolding"
)

_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_DRIVE_SCORE = 2
_TONIC_STOP_SCORE = 1

_CHECKPOINT_NAMESPACE = b"humanbrain.knowledge-substitution.checkpoint.v1\0"
_OBSERVATION_NAMESPACE = b"humanbrain.knowledge-substitution.observation.v1\0"
_ACTION_NAMESPACE = b"humanbrain.knowledge-substitution.action.v1\0"
_RECEIPT_NAMESPACE = b"humanbrain.knowledge-substitution.receipt.v1\0"
_SUCCESS_NAMESPACE = b"humanbrain.knowledge-substitution.success.v1\0"
_BROADCAST_NAMESPACE = b"humanbrain.knowledge-substitution.broadcast.v1\0"
_OCCURRENCE_NAMESPACE = b"humanbrain.knowledge-substitution.occurrence.v1\0"
_QUERY_NAMESPACE = b"humanbrain.knowledge-substitution.query.v1\0"
_DECISION_NAMESPACE = b"humanbrain.knowledge-substitution.decision.v1\0"
_RACE_NAMESPACE = b"humanbrain.knowledge-substitution.race.v1\0"
_CANONICAL_LEXEME_NAMESPACE = b"humanbrain.opaque-lexeme-pointer.v1\0"

_FILE_START = b"\x00knowledge-substitution/file-start/v1"
_FILE_END = b"\x00knowledge-substitution/file-end/v1"
_POINTER_EVENT = b"\x00knowledge-substitution/pointer/v1\0"
_LEXEME_EVENT = b"\x00knowledge-substitution/lexeme/v1\0"

_RECEIPT_TOKEN = object()
_BROADCAST_TOKEN = object()
_BRIDGE_TOKEN = object()
_SUBSTRATE = SparseNeuronalRoleFiller()


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


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _pointer_record(pointer: SparseBlockPointer) -> dict[str, object]:
    return {"block_cells": pointer.block_cells, "sha256": pointer.sha256}


def _workspace_record(files: tuple[OpaqueWorkspaceFile, ...]) -> tuple[dict[str, str], ...]:
    return tuple({"content_hex": item.content.hex(), "path": item.path} for item in files)


def _canonical_files(value: object, name: str) -> tuple[OpaqueWorkspaceFile, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(not isinstance(item, OpaqueWorkspaceFile) for item in value):
        raise TypeError(f"{name} must contain OpaqueWorkspaceFile values")
    files = tuple(value)
    if len({item.path for item in files}) != len(files):
        raise ValueError(f"{name} paths must be unique")
    return files


def _authenticate_observation(value: object) -> UnifiedOpaqueObservationTape:
    if type(value) is not UnifiedOpaqueObservationTape:
        raise TypeError("observation must be one exact UnifiedOpaqueObservationTape")
    rebuilt = UnifiedOpaqueObservationTape.observe(
        value.reconstruct_request(),
        value.reconstruct_documents(),
        value.reconstruct_workspace(),
        value.reconstruct_tool_events(),
    )
    if rebuilt.stable_hash != value.stable_hash:
        raise ValueError("unified observation custody changed")
    return rebuilt


def _observation_sha256(observation: UnifiedOpaqueObservationTape) -> str:
    return _digest(_OBSERVATION_NAMESPACE, {"stable_hash": observation.stable_hash.hex()})


def _action_sha256(files: tuple[OpaqueWorkspaceFile, ...]) -> str:
    return _digest(_ACTION_NAMESPACE, {"workspace": _workspace_record(files)})


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
            raise RuntimeError("knowledge-substitution GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    checks = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score
        for score, result in enumerate(checks)
    ):
        raise RuntimeError("knowledge-substitution GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class KnowledgeSubstitutionAblation(StrEnum):
    NONE = "none"
    BRIDGE_OFF = "bridge_off"
    UNBIND_OFF = "unbind_off"
    ANCHOR_CLEANUP_OFF = "anchor_cleanup_off"
    PAYLOAD_CLEANUP_OFF = "payload_cleanup_off"
    PROPAGATION_OFF = "propagation_off"
    BOUNDARY_OFF = "boundary_off"
    PROGRAM_OFF = "program_off"
    FINAL_EDIT_OFF = "final_edit_off"


@dataclass(frozen=True, slots=True)
class KnowledgeSubstitutionRaceTrace:
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
    trace: KnowledgeSubstitutionRaceTrace


def _physical_choice(
    *,
    population: str,
    scores: tuple[int, ...],
    candidate_cell_ids: tuple[int, ...] | None = None,
) -> _RaceChoice:
    if any(type(score) is not int or not 0 <= score <= _DRIVE_SCORE for score in scores):
        raise ValueError("physical race scores left their calibrated range")
    cells = tuple(range(len(scores))) if candidate_cell_ids is None else candidate_cell_ids
    if len(cells) != len(scores) or len(set(cells)) != len(cells):
        raise ValueError("physical race candidate cells must be unique and aligned")
    all_scores = (*scores, _TONIC_STOP_SCORE)
    result = _CELL_RACE.race_many(
        _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)][None, :],
        require_unique=False,
    )[0]
    collision_step: int | None = None
    stop_reason: str | None = None
    column: int | None = result.winner_cell
    physical_steps = len(all_scores) * _RACE_STEPS
    active_synapses = 1 + sum(score > 0 for score in scores)
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
    trace = KnowledgeSubstitutionRaceTrace(
        **record,
        sha256=_digest(_RACE_NAMESPACE, record),
    )
    return _RaceChoice(column=column, trace=trace)


@dataclass(frozen=True, slots=True, init=False)
class KnowledgeSubstitutionExecutionReceipt:
    producer_checkpoint_sha256: str
    observation_sha256: str
    action_sha256: str
    evaluator_sha256: str
    outcome: str
    sha256: str

    def __init__(
        self,
        token: object,
        *,
        producer_checkpoint_sha256: str,
        observation_sha256: str,
        action_sha256: str,
        evaluator_sha256: str,
        outcome: str,
        sha256: str,
    ) -> None:
        if token is not _RECEIPT_TOKEN:
            raise TypeError("execution receipts are issued only by their authority")
        object.__setattr__(self, "producer_checkpoint_sha256", producer_checkpoint_sha256)
        object.__setattr__(self, "observation_sha256", observation_sha256)
        object.__setattr__(self, "action_sha256", action_sha256)
        object.__setattr__(self, "evaluator_sha256", evaluator_sha256)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "sha256", sha256)

    def authenticate(self) -> None:
        record = {
            "action_sha256": _require_sha256(self.action_sha256, "action_sha256"),
            "evaluator_sha256": _require_sha256(
                self.evaluator_sha256, "evaluator_sha256"
            ),
            "observation_sha256": _require_sha256(
                self.observation_sha256, "observation_sha256"
            ),
            "outcome": self.outcome,
            "producer_checkpoint_sha256": _require_sha256(
                self.producer_checkpoint_sha256,
                "producer_checkpoint_sha256",
            ),
        }
        if self.outcome != "PASS" or self.sha256 != _digest(_RECEIPT_NAMESPACE, record):
            raise ValueError("execution receipt is not one authentic PASS")


class KnowledgeSubstitutionExecutionAuthority:
    """Run one pinned evaluator before sealing a complete transition receipt."""

    __slots__ = ("_evaluator", "_evaluator_sha256")

    def __init__(self, evaluator: Callable[..., object], evaluator_sha256: str) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self._evaluator = evaluator
        self._evaluator_sha256 = _require_sha256(evaluator_sha256, "evaluator_sha256")

    def execute_and_issue(
        self,
        *,
        producer_checkpoint_sha256: str,
        observation: UnifiedOpaqueObservationTape,
        after_workspace: tuple[OpaqueWorkspaceFile, ...],
    ) -> KnowledgeSubstitutionExecutionReceipt:
        producer = _require_sha256(
            producer_checkpoint_sha256,
            "producer_checkpoint_sha256",
        )
        observed = _authenticate_observation(observation)
        after = _canonical_files(after_workspace, "after_workspace")
        if self._evaluator(observation=observed, after_workspace=after) is not True:
            raise ValueError("frozen evaluator did not PASS the transition")
        observation_root = _observation_sha256(observed)
        action_root = _action_sha256(after)
        record = {
            "action_sha256": action_root,
            "evaluator_sha256": self._evaluator_sha256,
            "observation_sha256": observation_root,
            "outcome": "PASS",
            "producer_checkpoint_sha256": producer,
        }
        receipt = KnowledgeSubstitutionExecutionReceipt(
            _RECEIPT_TOKEN,
            producer_checkpoint_sha256=producer,
            observation_sha256=observation_root,
            action_sha256=action_root,
            evaluator_sha256=self._evaluator_sha256,
            outcome="PASS",
            sha256=_digest(_RECEIPT_NAMESPACE, record),
        )
        receipt.authenticate()
        return receipt


@dataclass(frozen=True, slots=True)
class KnowledgeSubstitutionSuccess:
    """One evaluator-authenticated complete TRAIN transition."""

    observation: UnifiedOpaqueObservationTape
    after_workspace: tuple[OpaqueWorkspaceFile, ...]
    execution_receipt: KnowledgeSubstitutionExecutionReceipt
    provenance_id: bytes
    sha256: str

    @classmethod
    def seal(
        cls,
        *,
        observation: UnifiedOpaqueObservationTape,
        after_workspace: tuple[OpaqueWorkspaceFile, ...],
        execution_receipt: KnowledgeSubstitutionExecutionReceipt,
        provenance_id: bytes,
    ) -> Self:
        observed = _authenticate_observation(observation)
        after = _canonical_files(after_workspace, "after_workspace")
        if not isinstance(execution_receipt, KnowledgeSubstitutionExecutionReceipt):
            raise TypeError("execution_receipt has the wrong type")
        execution_receipt.authenticate()
        if not isinstance(provenance_id, bytes) or not provenance_id:
            raise ValueError("provenance_id must be non-empty bytes")
        observation_root = _observation_sha256(observed)
        action_root = _action_sha256(after)
        if execution_receipt.observation_sha256 != observation_root:
            raise ValueError("receipt does not bind the complete unified observation")
        if execution_receipt.action_sha256 != action_root:
            raise ValueError("receipt does not bind the complete resulting workspace")
        record = {
            "action_sha256": action_root,
            "execution_receipt_sha256": execution_receipt.sha256,
            "observation_sha256": observation_root,
            "provenance_sha256": sha256(provenance_id).hexdigest(),
        }
        return cls(
            observation=observed,
            after_workspace=after,
            execution_receipt=execution_receipt,
            provenance_id=provenance_id,
            sha256=_digest(_SUCCESS_NAMESPACE, record),
        )

    def authenticate(self) -> None:
        observed = _authenticate_observation(self.observation)
        after = _canonical_files(self.after_workspace, "after_workspace")
        if not isinstance(self.execution_receipt, KnowledgeSubstitutionExecutionReceipt):
            raise TypeError("execution_receipt has the wrong type")
        self.execution_receipt.authenticate()
        record = {
            "action_sha256": _action_sha256(after),
            "execution_receipt_sha256": self.execution_receipt.sha256,
            "observation_sha256": _observation_sha256(observed),
            "provenance_sha256": sha256(self.provenance_id).hexdigest(),
        }
        if (
            self.execution_receipt.observation_sha256 != record["observation_sha256"]
            or self.execution_receipt.action_sha256 != record["action_sha256"]
            or self.sha256 != _digest(_SUCCESS_NAMESPACE, record)
        ):
            raise ValueError("successful transition custody changed")


@dataclass(frozen=True, slots=True)
class _WorkspaceOccurrence:
    file_ordinal: int
    path: str
    identity_event: bytes
    occurrence_event: bytes
    pointer: SparseBlockPointer
    pointer_event: bytes
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class _WorkspaceView:
    files: tuple[OpaqueWorkspaceFile, ...]
    events: tuple[bytes, ...]
    occurrences: tuple[_WorkspaceOccurrence, ...]


def _canonical_lexeme_pointer(identity_event: bytes) -> SparseBlockPointer:
    payload = len(identity_event).to_bytes(4, "big") + identity_event
    return _SUBSTRATE.project(_CANONICAL_LEXEME_NAMESPACE + payload)


def _pointer_event(pointer: SparseBlockPointer) -> bytes:
    return _POINTER_EVENT + bytes(pointer.block_cells)


def _workspace_view(observation: UnifiedOpaqueObservationTape) -> _WorkspaceView:
    files = observation.reconstruct_workspace()
    documents = tuple(
        OpaqueDocument(
            document_id=item.path.encode("utf-8"),
            provenance_id=observation.stable_hash + ordinal.to_bytes(4, "big"),
            content=item.content,
        )
        for ordinal, item in enumerate(files)
    )
    if not documents:
        return _WorkspaceView(files=files, events=(), occurrences=())
    lexical = OpaqueDocumentEventTape.observe(b"_", documents)
    events: list[bytes] = []
    occurrences: list[_WorkspaceOccurrence] = []
    for file_ordinal, (item, tokens) in enumerate(
        zip(files, lexical.document_token_occurrences, strict=True)
    ):
        events.append(_FILE_START)
        for token in tokens:
            pointer = _canonical_lexeme_pointer(token.identity_event)
            occurrence_record = {
                "byte_end": token.byte_end,
                "byte_start": token.byte_start,
                "file_ordinal": file_ordinal,
                "identity_hex": token.identity_event.hex(),
                "observation_sha256": observation.stable_hash.hex(),
                "path_sha256": sha256(item.path.encode("utf-8")).hexdigest(),
                "token_index": token.token_index,
            }
            occurrence_event = _LEXEME_EVENT + bytes.fromhex(
                _digest(_OCCURRENCE_NAMESPACE, occurrence_record)
            )
            pointer_identity = _pointer_event(pointer)
            occurrences.append(
                _WorkspaceOccurrence(
                    file_ordinal=file_ordinal,
                    path=item.path,
                    identity_event=token.identity_event,
                    occurrence_event=occurrence_event,
                    pointer=pointer,
                    pointer_event=pointer_identity,
                    byte_start=token.byte_start,
                    byte_end=token.byte_end,
                )
            )
            events.append(pointer_identity)
        events.append(_FILE_END)
    return _WorkspaceView(files=files, events=tuple(events), occurrences=tuple(occurrences))


@dataclass(frozen=True, slots=True, init=False)
class _FactAssemblyBroadcast:
    producer_checkpoint_sha256: str
    observation_sha256: str
    producer_decision_sha256: str
    fact_sha256: str
    schema_sha256: str
    source_provenance_sha256s: tuple[str, ...]
    surface_pointer: SparseBlockPointer
    key_pointer: SparseBlockPointer
    fact_pointer: SparseBlockPointer
    factor_zero_pointer: SparseBlockPointer
    factor_one_pointer: SparseBlockPointer
    lexeme_identity_event: bytes
    lexeme_occurrence_event: bytes
    retained_lexeme_sha256s: tuple[str, ...]
    transform_race_sha256s: tuple[str, ...]
    cleanup_race_sha256s: tuple[str, ...]
    selection_race_sha256s: tuple[str, ...]
    physical_cell_steps: int
    sha256: str

    def __init__(self, token: object, **values: object) -> None:
        if token is not _BROADCAST_TOKEN:
            raise TypeError("fact broadcasts are issued only inside the frozen bridge")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def authenticate(self) -> None:
        for name in (
            "producer_checkpoint_sha256",
            "observation_sha256",
            "producer_decision_sha256",
            "fact_sha256",
            "schema_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for pointer in (
            self.surface_pointer,
            self.key_pointer,
            self.fact_pointer,
            self.factor_zero_pointer,
            self.factor_one_pointer,
        ):
            SparseBlockPointer(pointer.block_cells, pointer.sha256)
        for root in self.retained_lexeme_sha256s:
            _require_sha256(root, "retained_lexeme_sha256")
        record = _broadcast_record(self)
        if self.sha256 != _digest(_BROADCAST_NAMESPACE, record):
            raise ValueError("fact broadcast custody changed")


def _broadcast_record(broadcast: _FactAssemblyBroadcast) -> dict[str, object]:
    return {
        "cleanup_race_sha256s": broadcast.cleanup_race_sha256s,
        "fact_pointer": _pointer_record(broadcast.fact_pointer),
        "fact_sha256": broadcast.fact_sha256,
        "factor_one_pointer": _pointer_record(broadcast.factor_one_pointer),
        "factor_zero_pointer": _pointer_record(broadcast.factor_zero_pointer),
        "key_pointer": _pointer_record(broadcast.key_pointer),
        "lexeme_identity_hex": broadcast.lexeme_identity_event.hex(),
        "lexeme_occurrence_event_hex": broadcast.lexeme_occurrence_event.hex(),
        "observation_sha256": broadcast.observation_sha256,
        "physical_cell_steps": broadcast.physical_cell_steps,
        "producer_checkpoint_sha256": broadcast.producer_checkpoint_sha256,
        "producer_decision_sha256": broadcast.producer_decision_sha256,
        "retained_lexeme_sha256s": broadcast.retained_lexeme_sha256s,
        "schema_sha256": broadcast.schema_sha256,
        "selection_race_sha256s": broadcast.selection_race_sha256s,
        "source_provenance_sha256s": broadcast.source_provenance_sha256s,
        "surface_pointer": _pointer_record(broadcast.surface_pointer),
        "transform_race_sha256s": broadcast.transform_race_sha256s,
    }


@dataclass(frozen=True, slots=True)
class _BroadcastAttempt:
    broadcast: _FactAssemblyBroadcast | None
    producer_decision: PersistentDocumentDecision | None
    transform_races: tuple[SparseTransformRaceTrace, ...]
    cleanup_races: tuple[SparseCleanupRaceTrace, ...]
    selection_races: tuple[KnowledgeSubstitutionRaceTrace, ...]
    physical_cell_steps: int
    stop_reason: str | None


def _factor_zero_lane_pointer(identity_event: bytes) -> SparseBlockPointer:
    # This is the producer's existing first structural factor lane.  The lane
    # name is deliberately contained here; the motor sees only canonical pointers.
    return _memory._project_digest(
        _memory._FACT_FILLER_NAMESPACE + b"subject\0",
        identity_event,
    )


def _issue_broadcast(
    producer: FrozenPersistentNeuronalDocumentLearner,
    observation: UnifiedOpaqueObservationTape,
    ablation: KnowledgeSubstitutionAblation,
) -> _BroadcastAttempt:
    documents = observation.reconstruct_documents()
    if not 1 <= len(documents) <= 4:
        return _BroadcastAttempt(None, None, (), (), (), 0, "document_count")
    producer_ablation = (
        PersistentDocumentAblation.ANSWER_SELECTION_OFF
        if ablation is KnowledgeSubstitutionAblation.BRIDGE_OFF
        else PersistentDocumentAblation.NONE
    )
    decision = producer.respond(
        observation.reconstruct_request(),
        documents,
        producer_ablation,
    )
    if (
        decision.decision != "EMIT"
        or decision.answer_source != "memory"
        or decision.memory_fact_sha256 is None
    ):
        return _BroadcastAttempt(
            None,
            decision,
            (),
            (),
            (),
            decision.physical_cell_steps,
            decision.stop_reason or "memory_fact_required",
        )

    facts = tuple(
        fact for fact in producer._facts if fact.sha256 == decision.memory_fact_sha256
    )
    if len(facts) != 1:
        raise RuntimeError("physical memory decision lost its unique fact")
    fact = facts[0]
    schemas = tuple(
        schema
        for schema in producer._schemas
        if schema.key_sha256 == fact.document_schema_key_sha256
    )
    if len(schemas) != 1:
        raise RuntimeError("physical memory fact lost its surface assembly")
    schema = schemas[0]
    sparse_ablation = (
        SparseRoleFillerAblation.BINDING_OFF
        if ablation is KnowledgeSubstitutionAblation.UNBIND_OFF
        else SparseRoleFillerAblation.NONE
    )
    factor_zero = _SUBSTRATE.unbind(
        schema.surface_pointer,
        fact.key_pointer,
        ablation=sparse_ablation,
    )
    factor_one = _SUBSTRATE.unbind(
        fact.key_pointer,
        fact.fact_pointer,
        ablation=sparse_ablation,
    )
    transform_races = (factor_zero.trace, factor_one.trace)
    if factor_zero.pointer is None or factor_one.pointer is None:
        steps = decision.physical_cell_steps + sum(
            race.physical_cell_steps for race in transform_races
        )
        return _BroadcastAttempt(
            None,
            decision,
            transform_races,
            (),
            (),
            steps,
            "unbinding_stop",
        )

    request_occurrences = observation.request_token_occurrences
    if not request_occurrences:
        return _BroadcastAttempt(
            None,
            decision,
            transform_races,
            (),
            (),
            decision.physical_cell_steps
            + sum(race.physical_cell_steps for race in transform_races),
            "missing_factor_zero_occurrence",
        )
    request_lane_pointers = tuple(
        _factor_zero_lane_pointer(item.identity_event) for item in request_occurrences
    )
    request_cleanup = _SUBSTRATE.cleanup(
        SparsePointerSuperposition.empty().accumulate(factor_zero.pointer),
        request_lane_pointers,
    )
    lexemes = producer._lexemes
    payload_cleanup = _SUBSTRATE.cleanup(
        SparsePointerSuperposition.empty().accumulate(factor_one.pointer),
        tuple(item.pointer for item in lexemes),
        ablation=(
            SparseRoleFillerAblation.CLEANUP_OFF
            if ablation is KnowledgeSubstitutionAblation.PAYLOAD_CLEANUP_OFF
            else SparseRoleFillerAblation.NONE
        ),
    )
    cleanup_races = (request_cleanup.trace, payload_cleanup.trace)
    selection_races: list[KnowledgeSubstitutionRaceTrace] = []
    if request_cleanup.pointer is None or payload_cleanup.pointer is None:
        steps = (
            decision.physical_cell_steps
            + sum(race.physical_cell_steps for race in transform_races)
            + sum(race.physical_cell_steps for race in cleanup_races)
        )
        return _BroadcastAttempt(
            None,
            decision,
            transform_races,
            cleanup_races,
            (),
            steps,
            "factor_cleanup_stop",
        )

    request_choice = _physical_choice(
        population="broadcast/factor_zero_occurrence",
        scores=tuple(
            _DRIVE_SCORE if pointer.sha256 == request_cleanup.pointer.sha256 else 0
            for pointer in request_lane_pointers
        ),
    )
    selection_races.append(request_choice.trace)
    payload_choice = _physical_choice(
        population="broadcast/factor_one_lexeme",
        scores=tuple(
            _DRIVE_SCORE if item.pointer.sha256 == payload_cleanup.pointer.sha256 else 0
            for item in lexemes
        ),
    )
    selection_races.append(payload_choice.trace)
    if request_choice.column is None or payload_choice.column is None:
        steps = (
            decision.physical_cell_steps
            + sum(race.physical_cell_steps for race in transform_races)
            + sum(race.physical_cell_steps for race in cleanup_races)
            + sum(race.physical_cell_steps for race in selection_races)
        )
        return _BroadcastAttempt(
            None,
            decision,
            transform_races,
            cleanup_races,
            tuple(selection_races),
            steps,
            "factor_occurrence_stop",
        )

    factor_zero_identity = request_occurrences[request_choice.column].identity_event
    lexeme = lexemes[payload_choice.column]
    canonical_zero = _canonical_lexeme_pointer(factor_zero_identity)
    canonical_one = _canonical_lexeme_pointer(lexeme.identity_event)
    selection_steps = sum(race.physical_cell_steps for race in selection_races)
    physical_steps = (
        decision.physical_cell_steps
        + sum(race.physical_cell_steps for race in transform_races)
        + sum(race.physical_cell_steps for race in cleanup_races)
        + selection_steps
    )
    values: dict[str, object] = {
        "producer_checkpoint_sha256": producer.checkpoint_sha256,
        "observation_sha256": _observation_sha256(observation),
        "producer_decision_sha256": decision.sha256,
        "fact_sha256": fact.sha256,
        "schema_sha256": schema.sha256,
        "source_provenance_sha256s": tuple(
            source.provenance_id_sha256 for source in fact.sources
        ),
        "surface_pointer": schema.surface_pointer,
        "key_pointer": fact.key_pointer,
        "fact_pointer": fact.fact_pointer,
        "factor_zero_pointer": canonical_zero,
        "factor_one_pointer": canonical_one,
        "lexeme_identity_event": lexeme.identity_event,
        "lexeme_occurrence_event": _LEXEME_EVENT + bytes.fromhex(lexeme.sha256),
        "retained_lexeme_sha256s": tuple(item.sha256 for item in lexemes),
        "transform_race_sha256s": tuple(race.sha256 for race in transform_races),
        "cleanup_race_sha256s": tuple(race.sha256 for race in cleanup_races),
        "selection_race_sha256s": tuple(race.sha256 for race in selection_races),
        "physical_cell_steps": physical_steps,
    }
    provisional = _FactAssemblyBroadcast(_BROADCAST_TOKEN, **values, sha256="")
    broadcast = _FactAssemblyBroadcast(
        _BROADCAST_TOKEN,
        **values,
        sha256=_digest(_BROADCAST_NAMESPACE, _broadcast_record(provisional)),
    )
    broadcast.authenticate()
    return _BroadcastAttempt(
        broadcast,
        decision,
        transform_races,
        cleanup_races,
        tuple(selection_races),
        physical_steps,
        None,
    )


def _single_substitution(
    observation: UnifiedOpaqueObservationTape,
    after_workspace: tuple[OpaqueWorkspaceFile, ...],
    broadcast: _FactAssemblyBroadcast,
) -> _WorkspaceOccurrence:
    before = observation.reconstruct_workspace()
    if tuple(item.path for item in before) != tuple(item.path for item in after_workspace):
        raise ValueError("successful transition must preserve workspace paths and order")
    changed = tuple(
        (old, new)
        for old, new in zip(before, after_workspace, strict=True)
        if old.content != new.content
    )
    if len(changed) != 1:
        raise ValueError("successful transition must change exactly one file")
    old_file, new_file = changed[0]
    view = _workspace_view(observation)
    old_occurrences = tuple(item for item in view.occurrences if item.path == old_file.path)
    after_document = OpaqueDocument(
        document_id=b"after",
        provenance_id=b"authenticated-training-transition",
        content=new_file.content,
    )
    after_tokens = OpaqueDocumentEventTape.observe(
        b"_",
        (after_document,),
    ).document_token_occurrences[0]
    explanations: list[_WorkspaceOccurrence] = []
    for old in old_occurrences:
        for new in after_tokens:
            if (
                new.identity_event == broadcast.lexeme_identity_event
                and old_file.content[: old.byte_start]
                + new.identity_event
                + old_file.content[old.byte_end :]
                == new_file.content
            ):
                explanations.append(old)
    if len(explanations) != 1:
        raise ValueError("full transition lacks one unique fact-grounded substitution")
    return explanations[0]


@dataclass(frozen=True, slots=True)
class KnowledgeSubstitutionTrainingSummary:
    demonstration_count: int
    evidence_root_count: int
    retained_full_transition_count: int
    binder_role_offset: int
    binder_wildcard_hops: int
    fixed_program_synapse_count: int
    sparse_fixed_cells: int
    sparse_fixed_synapses: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class KnowledgeSubstitutionDecision:
    decision: str
    workspace: tuple[OpaqueWorkspaceFile, ...] | None
    emission_events: tuple[bytes, ...]
    stop_stage: str | None
    stop_reason: str | None
    producer_decision_sha256: str | None
    broadcast_sha256: str | None
    binder_decision_sha256: str | None
    transform_races: tuple[SparseTransformRaceTrace, ...]
    cleanup_races: tuple[SparseCleanupRaceTrace, ...]
    binder_races: tuple[VariabilityRaceTrace, ...]
    structural_races: tuple[KnowledgeSubstitutionRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    observation_sha256: str
    sha256: str


class FrozenKnowledgeConditionedOccurrenceSubstitution:
    """One frozen two-transition fact-to-occurrence neuronal bridge."""

    __slots__ = (
        "_binder",
        "_checkpoint_sha256",
        "_evidence_sha256s",
        "_producer",
        "_summary",
    )

    def __init__(
        self,
        token: object,
        *,
        producer: FrozenPersistentNeuronalDocumentLearner,
        binder: VariabilityAwareSegmentBinder,
        evidence_sha256s: tuple[str, str],
        summary: KnowledgeSubstitutionTrainingSummary,
        checkpoint_sha256: str,
    ) -> None:
        if token is not _BRIDGE_TOKEN:
            raise TypeError("use grow()")
        self._producer = producer
        self._binder = binder
        self._evidence_sha256s = evidence_sha256s
        self._summary = summary
        self._checkpoint_sha256 = checkpoint_sha256
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> KnowledgeSubstitutionTrainingSummary:
        self.authenticate()
        return self._summary

    def authenticate(self) -> None:
        self._producer.authenticate()
        if len(self._evidence_sha256s) != 2:
            raise RuntimeError("frozen bridge lost its two evidence roots")
        record = {
            "binder_checkpoint_sha256": self._binder.checkpoint_sha256,
            "evidence_sha256s": self._evidence_sha256s,
            "producer_checkpoint_sha256": self._producer.checkpoint_sha256,
            "program": ("substitute", "end"),
        }
        expected = _digest(_CHECKPOINT_NAMESPACE, record)
        if expected != self._checkpoint_sha256 or self._summary.checkpoint_sha256 != expected:
            raise RuntimeError("frozen bridge checkpoint changed")

    def respond(
        self,
        observation: UnifiedOpaqueObservationTape,
        *,
        ablation: KnowledgeSubstitutionAblation = KnowledgeSubstitutionAblation.NONE,
    ) -> KnowledgeSubstitutionDecision:
        """Run memory, pointer grounding, learned propagation, and EDIT/STOP."""

        self.authenticate()
        observed = _authenticate_observation(observation)
        if not isinstance(ablation, KnowledgeSubstitutionAblation):
            raise TypeError("ablation must be KnowledgeSubstitutionAblation")
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": self._checkpoint_sha256,
                "observation_sha256": _observation_sha256(observed),
            },
        )
        attempt = _issue_broadcast(self._producer, observed, ablation)
        structural = list(attempt.selection_races)
        cleanup = list(attempt.cleanup_races)
        transform = list(attempt.transform_races)
        binder_races: tuple[VariabilityRaceTrace, ...] = ()
        producer_sha = (
            None if attempt.producer_decision is None else attempt.producer_decision.sha256
        )
        if attempt.broadcast is None:
            return self._stop(
                observed,
                query_sha256,
                stage="broadcast",
                reason=attempt.stop_reason or "tonic_stop",
                producer_sha=producer_sha,
                broadcast=None,
                binder=None,
                transform=transform,
                cleanup=cleanup,
                binder_races=binder_races,
                structural=structural,
                prior_steps=attempt.physical_cell_steps,
            )
        broadcast = attempt.broadcast
        view = _workspace_view(observed)
        if not view.occurrences or not view.events:
            return self._stop(
                observed,
                query_sha256,
                stage="workspace",
                reason="missing_occurrences",
                producer_sha=producer_sha,
                broadcast=broadcast,
                binder=None,
                transform=transform,
                cleanup=cleanup,
                binder_races=binder_races,
                structural=structural,
                prior_steps=attempt.physical_cell_steps,
            )
        anchor_cleanup = _SUBSTRATE.cleanup(
            SparsePointerSuperposition.empty().accumulate(broadcast.factor_zero_pointer),
            tuple(item.pointer for item in view.occurrences),
            ablation=(
                SparseRoleFillerAblation.CLEANUP_OFF
                if ablation is KnowledgeSubstitutionAblation.ANCHOR_CLEANUP_OFF
                else SparseRoleFillerAblation.NONE
            ),
        )
        cleanup.append(anchor_cleanup.trace)
        steps = attempt.physical_cell_steps + anchor_cleanup.trace.physical_cell_steps
        if anchor_cleanup.pointer is None:
            return self._stop(
                observed,
                query_sha256,
                stage="anchor_cleanup",
                reason=anchor_cleanup.trace.stop_reason or "tonic_stop",
                producer_sha=producer_sha,
                broadcast=broadcast,
                binder=None,
                transform=transform,
                cleanup=cleanup,
                binder_races=binder_races,
                structural=structural,
                prior_steps=steps,
            )
        binder_ablation = VariabilityBinderAblation.NONE
        if ablation is KnowledgeSubstitutionAblation.PROPAGATION_OFF:
            binder_ablation = VariabilityBinderAblation.PROPAGATION_OFF
        elif ablation is KnowledgeSubstitutionAblation.BOUNDARY_OFF:
            binder_ablation = VariabilityBinderAblation.BOUNDARY_OFF
        binder = self._binder.respond(
            (_pointer_event(anchor_cleanup.pointer),),
            view.events,
            ablation=binder_ablation,
        )
        binder_races = binder.races
        steps += binder.physical_cell_steps
        if binder.decision != "BIND" or binder.role_identity is None:
            return self._stop(
                observed,
                query_sha256,
                stage="propagation",
                reason=binder.stop_reason or "tonic_stop",
                producer_sha=producer_sha,
                broadcast=broadcast,
                binder=binder,
                transform=transform,
                cleanup=cleanup,
                binder_races=binder_races,
                structural=structural,
                prior_steps=steps,
            )
        occurrence_choice = _physical_choice(
            population="motor/occurrence_selection",
            scores=tuple(
                _DRIVE_SCORE if item.pointer_event == binder.role_identity else 0
                for item in view.occurrences
            ),
        )
        structural.append(occurrence_choice.trace)
        steps += occurrence_choice.trace.physical_cell_steps
        if occurrence_choice.column is None:
            return self._stop(
                observed,
                query_sha256,
                stage="occurrence",
                reason=occurrence_choice.trace.stop_reason or "tonic_stop",
                producer_sha=producer_sha,
                broadcast=broadcast,
                binder=binder,
                transform=transform,
                cleanup=cleanup,
                binder_races=binder_races,
                structural=structural,
                prior_steps=steps,
            )
        target = view.occurrences[occurrence_choice.column]
        program_score = (
            0 if ablation is KnowledgeSubstitutionAblation.PROGRAM_OFF else _DRIVE_SCORE
        )
        substitute = _physical_choice(
            population="program/substitute",
            scores=(program_score,),
        )
        end = _physical_choice(
            population="program/end",
            scores=(program_score if substitute.column is not None else 0,),
        )
        structural.extend((substitute.trace, end.trace))
        steps += substitute.trace.physical_cell_steps + end.trace.physical_cell_steps
        complete = substitute.column is not None and end.column is not None
        final = _physical_choice(
            population="action/edit_or_stop",
            scores=(
                _DRIVE_SCORE
                if complete and ablation is not KnowledgeSubstitutionAblation.FINAL_EDIT_OFF
                else 0,
            ),
        )
        structural.append(final.trace)
        steps += final.trace.physical_cell_steps
        if final.column is None:
            return self._decision(
                observed,
                query_sha256,
                decision="STOP",
                workspace=None,
                emissions=(),
                stage="final_action",
                reason=final.trace.stop_reason or "tonic_stop",
                producer_sha=producer_sha,
                broadcast=broadcast,
                binder=binder,
                transform=transform,
                cleanup=cleanup,
                binder_races=binder_races,
                structural=structural,
                physical_steps=steps,
            )
        files = list(view.files)
        source = files[target.file_ordinal]
        if (
            source.path != target.path
            or source.content[target.byte_start : target.byte_end] != target.identity_event
        ):
            raise RuntimeError("winning occurrence custody changed before materialization")
        replacement = (
            source.content[: target.byte_start]
            + broadcast.lexeme_identity_event
            + source.content[target.byte_end :]
        )
        files[target.file_ordinal] = OpaqueWorkspaceFile(source.path, replacement)
        return self._decision(
            observed,
            query_sha256,
            decision="EDIT",
            workspace=tuple(files),
            emissions=(target.occurrence_event, broadcast.lexeme_occurrence_event),
            stage=None,
            reason=None,
            producer_sha=producer_sha,
            broadcast=broadcast,
            binder=binder,
            transform=transform,
            cleanup=cleanup,
            binder_races=binder_races,
            structural=structural,
            physical_steps=steps,
        )

    def _stop(
        self,
        observation: UnifiedOpaqueObservationTape,
        query_sha256: str,
        *,
        stage: str,
        reason: str,
        producer_sha: str | None,
        broadcast: _FactAssemblyBroadcast | None,
        binder: VariabilityBindingDecision | None,
        transform: list[SparseTransformRaceTrace],
        cleanup: list[SparseCleanupRaceTrace],
        binder_races: tuple[VariabilityRaceTrace, ...],
        structural: list[KnowledgeSubstitutionRaceTrace],
        prior_steps: int,
    ) -> KnowledgeSubstitutionDecision:
        final = _physical_choice(population="action/edit_or_stop", scores=(0,))
        structural.append(final.trace)
        return self._decision(
            observation,
            query_sha256,
            decision="STOP",
            workspace=None,
            emissions=(),
            stage=stage,
            reason=f"{stage}:{reason}",
            producer_sha=producer_sha,
            broadcast=broadcast,
            binder=binder,
            transform=transform,
            cleanup=cleanup,
            binder_races=binder_races,
            structural=structural,
            physical_steps=prior_steps + final.trace.physical_cell_steps,
        )

    def _decision(
        self,
        observation: UnifiedOpaqueObservationTape,
        query_sha256: str,
        *,
        decision: str,
        workspace: tuple[OpaqueWorkspaceFile, ...] | None,
        emissions: tuple[bytes, ...],
        stage: str | None,
        reason: str | None,
        producer_sha: str | None,
        broadcast: _FactAssemblyBroadcast | None,
        binder: VariabilityBindingDecision | None,
        transform: list[SparseTransformRaceTrace],
        cleanup: list[SparseCleanupRaceTrace],
        binder_races: tuple[VariabilityRaceTrace, ...],
        structural: list[KnowledgeSubstitutionRaceTrace],
        physical_steps: int,
    ) -> KnowledgeSubstitutionDecision:
        record = {
            "binder_decision_sha256": None if binder is None else binder.sha256,
            "broadcast_sha256": None if broadcast is None else broadcast.sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
            "cleanup_race_sha256s": tuple(race.sha256 for race in cleanup),
            "decision": decision,
            "emission_events_hex": tuple(event.hex() for event in emissions),
            "observation_sha256": _observation_sha256(observation),
            "physical_cell_steps": physical_steps,
            "producer_decision_sha256": producer_sha,
            "query_sha256": query_sha256,
            "stop_reason": reason,
            "stop_stage": stage,
            "structural_race_sha256s": tuple(race.sha256 for race in structural),
            "transform_race_sha256s": tuple(race.sha256 for race in transform),
            "workspace": None if workspace is None else _workspace_record(workspace),
        }
        return KnowledgeSubstitutionDecision(
            decision=decision,
            workspace=workspace,
            emission_events=emissions,
            stop_stage=stage,
            stop_reason=reason,
            producer_decision_sha256=producer_sha,
            broadcast_sha256=None if broadcast is None else broadcast.sha256,
            binder_decision_sha256=None if binder is None else binder.sha256,
            transform_races=tuple(transform),
            cleanup_races=tuple(cleanup),
            binder_races=binder_races,
            structural_races=tuple(structural),
            physical_cell_steps=physical_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            observation_sha256=_observation_sha256(observation),
            sha256=_digest(_DECISION_NAMESPACE, record),
        )


def grow(
    producer: FrozenPersistentNeuronalDocumentLearner,
    successes: Iterable[KnowledgeSubstitutionSuccess],
) -> FrozenKnowledgeConditionedOccurrenceSubstitution:
    """Freeze one bridge from exactly two independent authenticated transitions."""

    if type(producer) is not FrozenPersistentNeuronalDocumentLearner:
        raise TypeError("producer must be one exact frozen persistent learner")
    producer.authenticate()
    rows = tuple(successes)
    if len(rows) != 2:
        raise ValueError("the frozen bridge requires exactly two transitions")
    if any(not isinstance(row, KnowledgeSubstitutionSuccess) for row in rows):
        raise TypeError("successes must contain KnowledgeSubstitutionSuccess values")
    for row in rows:
        row.authenticate()
        if row.execution_receipt.producer_checkpoint_sha256 != producer.checkpoint_sha256:
            raise ValueError("transition receipt belongs to another producer checkpoint")
    if len({row.sha256 for row in rows}) != 2:
        raise ValueError("training transitions must be independent")
    if len({row.provenance_id for row in rows}) != 2:
        raise ValueError("training provenance must be independent")
    if len({_observation_sha256(row.observation) for row in rows}) != 2:
        raise ValueError("training observations must be independent")

    demonstrations: list[VariabilitySegmentDemonstration] = []
    fact_roots: list[str] = []
    changed_paths: list[str] = []
    inserted_identities: list[bytes] = []
    for row in rows:
        attempt = _issue_broadcast(
            producer,
            row.observation,
            KnowledgeSubstitutionAblation.NONE,
        )
        if attempt.broadcast is None:
            raise ValueError("training transition has no physical retained-fact broadcast")
        broadcast = attempt.broadcast
        target = _single_substitution(row.observation, row.after_workspace, broadcast)
        view = _workspace_view(row.observation)
        demonstrations.append(
            VariabilitySegmentDemonstration(
                query_events=(_pointer_event(broadcast.factor_zero_pointer),),
                source_events=view.events,
                demonstrated_role=target.pointer_event,
            )
        )
        fact_roots.append(broadcast.fact_sha256)
        changed_paths.append(target.path)
        inserted_identities.append(broadcast.lexeme_identity_event)
    if len(set(fact_roots)) != 2 or len(set(changed_paths)) != 2:
        raise ValueError("training transitions need distinct facts and changed paths")
    if len(set(inserted_identities)) != 2:
        raise ValueError("training transitions need distinct retained lexemes")

    binder = VariabilityAwareSegmentBinder.grow(demonstrations)
    if binder.training_summary.wildcard_intermediate_hop_count < 1:
        raise ValueError("training transitions must expose at least one variable hop")
    evidence_sha256s = (rows[0].sha256, rows[1].sha256)
    checkpoint = _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "binder_checkpoint_sha256": binder.checkpoint_sha256,
            "evidence_sha256s": evidence_sha256s,
            "producer_checkpoint_sha256": producer.checkpoint_sha256,
            "program": ("substitute", "end"),
        },
    )
    summary = KnowledgeSubstitutionTrainingSummary(
        demonstration_count=2,
        evidence_root_count=2,
        retained_full_transition_count=0,
        binder_role_offset=binder.training_summary.role_offset,
        binder_wildcard_hops=binder.training_summary.wildcard_intermediate_hop_count,
        fixed_program_synapse_count=2,
        sparse_fixed_cells=_SUBSTRATE.counts.fixed_cell_count,
        sparse_fixed_synapses=_SUBSTRATE.counts.fixed_synapse_count,
        checkpoint_sha256=checkpoint,
    )
    return FrozenKnowledgeConditionedOccurrenceSubstitution(
        _BRIDGE_TOKEN,
        producer=producer,
        binder=binder,
        evidence_sha256s=evidence_sha256s,
        summary=summary,
        checkpoint_sha256=checkpoint,
    )
