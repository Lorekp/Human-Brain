"""Bind anonymous occurrence schemas to exact executable aggregate witnesses.

This self-posed bridge composes two frozen mechanisms.  The occurrence
checkpoint recognizes a neutral variable-length physical bundle.  A learned
2x2 uint8 E/H relay maps only its winning physical schema cell onto the
rebased content-addressed schema cells.  The latter cell drives the banked
physical reducer-action race.  The resulting exact local action receipt
authorizes one already-banked count or raw-sum scaffold, and an identity-issued
plan-slice receipt materializes exactly one proof and executable witness.

Frozen build has no append probe, desired output, task/family label, target
action, or schema input.  Carrier allocation, bundle cues, prompt literal
binding, the two reducer scaffolds, typed enumeration, and compilation remain
explicit host scaffolds.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple, cast

import numpy as np
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_content_addressed_segment_schema_v1 as content
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as segmented
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter
from experiments.r3_pilot import public_typed_output_cells as typed_output
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_OCCURRENCE_EXECUTABLE_REDUCER_BRIDGE_V1"
READY = "CAUSAL_OCCURRENCE_EXECUTABLE_REDUCER_BRIDGE_V1_READY"
STOP_OCCURRENCE_SCHEMA = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_OCCURRENCE_SCHEMA"
STOP_STATIC_DEPENDENCY = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_STATIC_DEPENDENCY"
STOP_RELAY_TONIC = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_RELAY_TONIC"
STOP_RELAY_AMBIGUOUS = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_RELAY_AMBIGUOUS"
STOP_CAUSAL_PAIR_CONFLICT = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_CAUSAL_PAIR_CONFLICT"
STOP_ACTION_ZERO = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_ACTION_ZERO"
STOP_ACTION_AMBIGUOUS = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_ACTION_AMBIGUOUS"
STOP_RESIDUAL_BINDING = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_RESIDUAL_BINDING"
STOP_PLAN_SLICE = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_PLAN_SLICE"
STOP_WITNESS_NOT_SINGLETON = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_WITNESS_NOT_SINGLETON"
STOP_WORK_BOUND = "STOP_CAUSAL_OCCURRENCE_EXECUTABLE_WORK_BOUND"

CLAIM_BOUNDARY = (
    "one self-posed learned occurrence-schema to aggregate-reducer composition "
    "materializes exact count and raw-sum witnesses on fresh neutral bundles; "
    "OPEN/EXTEND/EOS cues, node carrier allocation, equality/graph projection, "
    "prompt literal binding, fixed reducer scaffolds, typed enumeration, and "
    "compilation remain explicit, so parsing, arbitrary algorithms, public "
    "coding transfer, and SWE capability remain unproven"
)

OCCURRENCE_SOURCE_COMMIT = "b5be799"
CONTENT_SOURCE_COMMIT = "17c891f"

SCHEMA_CAPACITY = 2
TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
RELAY_STATE_SHAPE = (SCHEMA_CAPACITY, SCHEMA_CAPACITY)
NEW_PERSISTENT_STATE_BYTES = (
    2 * int(np.prod(RELAY_STATE_SHAPE)) * TICK_DTYPE.itemsize
)

# Both upstream modules were defined relative to the same segmented base.  In
# the composite identity map the occurrence range comes first and the five
# content cells are rebased immediately after it.
OCCURRENCE_CELL_OFFSET = segmented.TOTAL_CELL_COUNT
OCCURRENCE_CELL_STOP = occurrence.TOTAL_CELL_COUNT
REBASED_CONTENT_CELL_OFFSET = OCCURRENCE_CELL_STOP
REBASED_CONTENT_SCHEMA_CELL_OFFSET = REBASED_CONTENT_CELL_OFFSET
REBASED_CONTENT_NOVEL_CELL = REBASED_CONTENT_CELL_OFFSET + content.SCHEMA_CAPACITY
REBASED_CONTENT_TONIC_CELL = REBASED_CONTENT_NOVEL_CELL + 1
REBASED_CONTENT_SCRATCH_CELL = REBASED_CONTENT_TONIC_CELL + 1
NEW_CELL_COUNT = 0
TOTAL_CELL_COUNT = occurrence.TOTAL_CELL_COUNT + content.NEW_CELL_COUNT
SEGMENTED_COMPOSED_PERSISTENT_STATE_BYTES = 308
COMPOSED_PERSISTENT_STATE_BYTES = (
    SEGMENTED_COMPOSED_PERSISTENT_STATE_BYTES
    + occurrence.PERSISTENT_STATE_BYTES
    + content.NEW_PERSISTENT_STATE_BYTES
    + NEW_PERSISTENT_STATE_BYTES
)

RELAY_RACE_CELL_COUNT = SCHEMA_CAPACITY + 1
RELAY_RACE_STEPS = output_projection.RACE_STEPS
RELAY_RACE_CELL_STEPS = RELAY_RACE_CELL_COUNT * RELAY_RACE_STEPS
FULL_RELAY_SCORE = content.FULL_SCHEMA_SCORE
ACTION_RACE_CELL_STEPS = content.ACTION_RACE_CELL_STEPS
TRAIN_LESSON_COUNT = 3

DESIRED_READS_DURING_BUILD = 0
PROBE_OUTPUT_READS_DURING_BUILD = 0
DIRECT_OCCURRENCE_ACTION_READS = 0
DIRECT_APPEND_ACTION_READS = 0
SERIALIZED_SCHEMA_ACTION_READS = 0

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DEV_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

if (
    occurrence.SCHEMA_CELL_OFFSET != OCCURRENCE_CELL_OFFSET + occurrence.RELATION_CELL_COUNT
    or OCCURRENCE_CELL_STOP != 69_227
    or content.TOTAL_CELL_COUNT != 69_213
    or TOTAL_CELL_COUNT != 69_232
    or NEW_PERSISTENT_STATE_BYTES != 8
    or COMPOSED_PERSISTENT_STATE_BYTES != 768
):
    raise RuntimeError("occurrence executable composite identity changed")

_NAMESPACE = b"causal-occurrence-executable-reducer-bridge-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_RELAY_RACE_NAMESPACE = _NAMESPACE + b"relay-race\0"
_STATIC_OBSERVATION_NAMESPACE = _NAMESPACE + b"static-observation\0"
_ACTION_RECEIPT_NAMESPACE = _NAMESPACE + b"action-receipt\0"
_TRAIN_WORK_NAMESPACE = _NAMESPACE + b"train-work\0"
_BUILD_WORK_NAMESPACE = _NAMESPACE + b"build-work\0"
_WITNESS_NAMESPACE = _NAMESPACE + b"witness\0"
_CONSTRUCTION_TOKEN = object()
_ACTION_AUTHORITY = object()
_WITNESS_AUTHORITY = object()


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _array_frame(name: str, value: np.ndarray) -> bytes:
    return b"\0".join(
        (
            name.encode("ascii"),
            value.dtype.str.encode("ascii"),
            _canonical_json(value.shape),
            value.tobytes(order="C"),
        )
    )


@dataclass(frozen=True, slots=True)
class CausalOccurrenceExecutableLesson:
    """One neutral bundle paired only with an authenticated physical effect."""

    events: tuple[occurrence.PhysicalOccurrenceEvent, ...]
    span_records: tuple[Mapping[str, object], ...]
    literal_records: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]
    probe_cases: tuple[tuple[list[int]], ...]
    probe_outputs: tuple[typed_output.TypedOutputEncoding, ...]
    event_order: tuple[int, ...] = segmented.CURRICULUM_VALUES


class BridgeTrainingWork(NamedTuple):
    occurrence_physical_cell_steps: int
    static_dependency_physical_cell_steps: int
    relay_race_count: int
    relay_race_cell_steps: int
    action_race_count: int
    action_race_cell_steps: int
    bridge_physical_cell_steps: int
    total_physical_cell_steps: int
    occurrence_work_sha256s: tuple[str, ...]
    static_observation_sha256s: tuple[str, ...]
    relay_race_sha256s: tuple[str, ...]
    action_receipt_sha256s: tuple[str, ...]
    sha256: str


class BridgeTrainingSummary(NamedTuple):
    lesson_count: int
    occurrence_schema_cell_ids: tuple[int, ...]
    static_schema_cell_ids: tuple[int, ...]
    reducer_action_cell_ids: tuple[int, ...]
    learned_cell_pairs: tuple[tuple[int, int], ...]
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    new_persistent_state_bytes: int
    composed_persistent_state_bytes: int
    work: BridgeTrainingWork
    checkpoint_sha256: str


class BridgeBuildWork(NamedTuple):
    occurrence_physical_cell_steps: int
    residual_binding_cell_steps: int
    relay_race_cell_steps: int
    action_race_cell_steps: int
    bridge_physical_cell_steps: int
    prepacked_physical_cell_steps: int
    packed_frontier_build_count: int
    authorized_slice_count: int
    proof_materialization_count: int
    packed_work_sha256: str | None
    sha256: str


class _StaticObservation(NamedTuple):
    schema_cell_id: int
    residual_certificate_sha256: str
    effect_relation: int
    projection_sha256: str
    schema_race_sha256: str
    total_physical_cell_steps: int
    sha256: str


class _RelayDecision(NamedTuple):
    status: str
    static_schema_cell_id: int | None
    raw_scores: tuple[int, ...]
    winner_cell: int
    spike_step: int | None
    simultaneous_spike_candidates: int
    sha256: str


class _ActionReceiptRecord(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    content_checkpoint_sha256: str
    static_schema_cell_id: int
    reducer_action_cell_id: int | None
    action_scores: tuple[int, ...]
    physical_race_sha256: str
    physical_cell_steps: int
    sha256: str


class PhysicalReducerActionReceipt:
    """Opaque exact-object authority issued only by the physical action race."""

    __slots__ = ("_record",)

    def __init__(self, record: _ActionReceiptRecord) -> None:
        if record.authority is not _ACTION_AUTHORITY:
            raise TypeError("reducer action receipt requires private authority")
        self._record = record

    @property
    def status(self) -> str:
        return self._record.status

    @property
    def stop_reason(self) -> str | None:
        return self._record.stop_reason

    @property
    def static_schema_cell_id(self) -> int:
        return self._record.static_schema_cell_id

    @property
    def reducer_action_cell_id(self) -> int | None:
        return self._record.reducer_action_cell_id

    @property
    def sha256(self) -> str:
        return self._record.sha256


class _WitnessReceipt(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    checkpoint_sha256: str
    occurrence_certificate_sha256: str
    occurrence_schema_cell_id: int | None
    relay_race_sha256: str | None
    static_schema_cell_id: int | None
    action_receipt_sha256: str | None
    reducer_action_cell_id: int | None
    residual_certificate_sha256: str | None
    structural_certificate_sha256: str | None
    plan_slice_receipt_sha256: str | None
    exact_proof_count: int
    witness: breadth_v3.PackedGenericTypedWitness | None
    desired_reads_during_build: int
    probe_output_reads_during_build: int
    direct_occurrence_action_reads: int
    direct_append_action_reads: int
    serialized_schema_action_reads: int
    work: BridgeBuildWork
    sha256: str


_ISSUED_ACTION_RECEIPTS: dict[int, PhysicalReducerActionReceipt] = {}
_ISSUED_WITNESS_CERTIFICATES: dict[int, OccurrenceExecutableWitnessCertificate] = {}


class OccurrenceExecutableWitnessCertificate:
    """Opaque exact result containing zero or one desired-free witness."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _WitnessReceipt) -> None:
        if receipt.authority is not _WITNESS_AUTHORITY:
            raise TypeError("executable witness certificate requires private authority")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.status

    @property
    def stop_reason(self) -> str | None:
        return self._receipt.stop_reason

    @property
    def occurrence_schema_cell_id(self) -> int | None:
        return self._receipt.occurrence_schema_cell_id

    @property
    def static_schema_cell_id(self) -> int | None:
        return self._receipt.static_schema_cell_id

    @property
    def reducer_action_cell_id(self) -> int | None:
        return self._receipt.reducer_action_cell_id

    @property
    def exact_proof_count(self) -> int:
        return self._receipt.exact_proof_count

    @property
    def witness(self) -> breadth_v3.PackedGenericTypedWitness | None:
        return self._receipt.witness

    @property
    def work(self) -> BridgeBuildWork:
        return self._receipt.work

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


class OccurrenceExecutableFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _source_ordinal(schema_cell_id: int) -> int:
    ordinal = schema_cell_id - occurrence.SCHEMA_CELL_OFFSET
    if not 0 <= ordinal < SCHEMA_CAPACITY:
        raise OccurrenceExecutableFitStop(
            STOP_OCCURRENCE_SCHEMA,
            "occurrence certificate did not expose one physical schema cell",
        )
    return ordinal


def _static_ordinal(schema_cell_id: int) -> int:
    ordinal = schema_cell_id - REBASED_CONTENT_SCHEMA_CELL_OFFSET
    if not 0 <= ordinal < SCHEMA_CAPACITY:
        raise OccurrenceExecutableFitStop(
            STOP_STATIC_DEPENDENCY, "static schema cell lies outside the frozen map"
        )
    return ordinal


def _recognize_static_schema(
    checkpoint: content.FrozenCausalContentAddressedSegmentSchemaV1,
    lesson: CausalOccurrenceExecutableLesson,
) -> _StaticObservation:
    dependency = checkpoint._segmented_checkpoint
    residual_checkpoint = dependency._append_checkpoint._residual_checkpoint
    observation = append._observe(
        residual_checkpoint,
        lesson.span_records,
        lesson.literal_records,
        lesson.signature_role_record,
        lesson.probe_cases,
        lesson.probe_outputs,
        equality_gate=True,
        literal_role_gate=True,
        order_gate=True,
        upstream_cell_gate=True,
        residual_gate=True,
        residual_cell_gate=True,
        predicate_gate=True,
        effect_gate=True,
        completeness_gate=True,
    )
    if observation.status != append.READY or observation.effect_relation is None:
        raise OccurrenceExecutableFitStop(
            STOP_STATIC_DEPENDENCY, observation.reason or observation.status
        )

    projection = segmented._project_effect_events(
        observation.effect_relation,
        dependency._relation_cells,
        lesson.event_order,
    )
    current = dependency._segmenter
    transitions: list[segmenter.PredictiveSegmentationTransition] = []
    router_roots: list[str] = []
    for event in projection.events:
        transition = segmenter.advance_predictive_segmenter_v1(current, event)
        transitions.append(transition)
        if not transition.accepted_event:
            raise OccurrenceExecutableFitStop(
                STOP_STATIC_DEPENDENCY, "static segmenter rejected a fit event"
            )
        relation = segmented._decision_relation(transition)
        status, action, root = segmented._race_router(
            dependency._excitatory,
            dependency._inhibitory,
            relation,
            cell_gate=True,
        )
        router_roots.append(root)
        if (
            status != segmented.READY
            or action is None
            or action != segmented._required_action(transition)
        ):
            raise OccurrenceExecutableFitStop(
                STOP_STATIC_DEPENDENCY, "static router lost its physical consequence"
            )
        current = transition.successor
    if (
        len(transitions) != segmented.EVENTS_PER_SEGMENT
        or not transitions[-1].segment_completed
        or current.segment_count != dependency._segmenter.segment_count + 1
    ):
        raise OccurrenceExecutableFitStop(
            STOP_STATIC_DEPENDENCY, "static dependency did not complete one segment"
        )

    targets, _target_root = content._route_targets(projection)
    decision = content._race_schema(
        checkpoint._prototype_excitatory,
        checkpoint._prototype_inhibitory,
        checkpoint._valid,
        targets,
        allow_novel=False,
        comparator_gate=True,
        schema_cell_gate=True,
    )
    if decision.status != content.READY or decision.schema_cell_id is None:
        raise OccurrenceExecutableFitStop(
            STOP_STATIC_DEPENDENCY, "content prototype race did not select one schema"
        )
    local_ordinal = decision.schema_cell_id - content.SCHEMA_CELL_OFFSET
    if not 0 <= local_ordinal < SCHEMA_CAPACITY:
        raise OccurrenceExecutableFitStop(
            STOP_STATIC_DEPENDENCY, "content prototype selected a non-schema cell"
        )
    rebased_cell = REBASED_CONTENT_SCHEMA_CELL_OFFSET + local_ordinal
    residual_steps, comparison_steps, completeness_steps, effect_steps = (
        content._observation_cell_steps(observation)
    )
    total_steps = (
        residual_steps
        + comparison_steps
        + completeness_steps
        + effect_steps
        + projection.effect_projection_cell_steps
        + sum(item.physical_cell_steps for item in transitions)
        + len(router_roots) * segmented.ROUTER_RACE_CELL_STEPS
        + content.SCHEMA_RACE_CELL_STEPS
    )
    payload = {
        "content_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "residual_certificate_sha256": observation.residual_certificate.sha256,
        "effect_relation": observation.effect_relation,
        "projection_sha256": projection.sha256,
        "schema_race_sha256": decision.sha256,
        "rebased_schema_cell_id": rebased_cell,
        "total_physical_cell_steps": total_steps,
    }
    return _StaticObservation(
        rebased_cell,
        observation.residual_certificate.sha256,
        observation.effect_relation,
        projection.sha256,
        decision.sha256,
        total_steps,
        _digest(_STATIC_OBSERVATION_NAMESPACE, payload),
    )


def _learn_relay(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    source_ordinal: int,
    target_ordinal: int,
) -> tuple[int, int]:
    excitatory_ticks = 0
    inhibitory_ticks = 0
    for candidate in range(SCHEMA_CAPACITY):
        state = excitatory if candidate == target_ordinal else inhibitory
        before = int(state[source_ordinal, candidate])
        state[source_ordinal, candidate] = min(TICK_CAP, before + 1)
        delta = int(state[source_ordinal, candidate]) - before
        if candidate == target_ordinal:
            excitatory_ticks += delta
        else:
            inhibitory_ticks += delta
    return excitatory_ticks, inhibitory_ticks


def _race_relay(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    occurrence_schema_cell_id: int,
    *,
    cell_gate: bool,
) -> _RelayDecision:
    source = _source_ordinal(occurrence_schema_cell_id)
    raw_scores = tuple(
        int(excitatory[source, target]) - int(inhibitory[source, target])
        for target in range(SCHEMA_CAPACITY)
    )
    candidate_scores = tuple(
        FULL_RELAY_SCORE if cell_gate and score > 0 else 0 for score in raw_scores
    )
    current_scores = (*candidate_scores, output_projection.STOP_SCORE)
    currents = np.asarray(
        [[output_projection._CURRENT_TABLE[score] for score in current_scores]],
        dtype=np.float64,
    )
    race = output_projection._OUTPUT_RACE.race_many(
        currents, require_unique=False
    )[0]
    winner = int(race.winner_cell)
    if race.simultaneous_spike_candidates != 1:
        status = STOP_RELAY_AMBIGUOUS
        static_cell = None
    elif winner < SCHEMA_CAPACITY and candidate_scores[winner] > 0:
        status = READY
        static_cell = REBASED_CONTENT_SCHEMA_CELL_OFFSET + winner
    else:
        status = STOP_RELAY_TONIC
        static_cell = None
    payload = {
        "occurrence_schema_cell_id": occurrence_schema_cell_id,
        "raw_scores": raw_scores,
        "candidate_scores": candidate_scores,
        "cell_gate": cell_gate,
        "winner_cell": winner,
        "spike_step": race.spike_step,
        "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
        "status": status,
        "static_schema_cell_id": static_cell,
        "cell_steps": RELAY_RACE_CELL_STEPS,
    }
    return _RelayDecision(
        status,
        static_cell,
        raw_scores,
        winner,
        race.spike_step,
        race.simultaneous_spike_candidates,
        _digest(_RELAY_RACE_NAMESPACE, payload),
    )


def _action_payload(record: _ActionReceiptRecord) -> dict[str, object]:
    return {
        field: getattr(record, field)
        for field in record._fields
        if field not in {"authority", "sha256"}
    }


def _issue_action_receipt(
    checkpoint: content.FrozenCausalContentAddressedSegmentSchemaV1,
    static_schema_cell_id: int,
    *,
    cell_gate: bool,
) -> PhysicalReducerActionReceipt:
    ordinal = _static_ordinal(static_schema_cell_id)
    status, action, scores, race_sha = content._race_action(
        checkpoint._action_excitatory,
        checkpoint._action_inhibitory,
        ordinal,
        cell_gate=cell_gate,
    )
    if status == content.READY and action is not None:
        local_status = READY
        reason = None
        action_cell = append.REDUCER_ACTION_CELL_OFFSET + action
    elif status == content.STOP_ACTION_ZERO:
        local_status = STOP_ACTION_ZERO
        reason = "physical reducer action population was silent"
        action_cell = None
    else:
        local_status = STOP_ACTION_AMBIGUOUS
        reason = "physical reducer action population was ambiguous"
        action_cell = None
    partial = _ActionReceiptRecord(
        _ACTION_AUTHORITY,
        local_status,
        reason,
        checkpoint.checkpoint_sha256,
        static_schema_cell_id,
        action_cell,
        scores,
        race_sha,
        ACTION_RACE_CELL_STEPS,
        "",
    )
    record = partial._replace(
        sha256=_digest(_ACTION_RECEIPT_NAMESPACE, _action_payload(partial))
    )
    receipt = PhysicalReducerActionReceipt(record)
    _ISSUED_ACTION_RECEIPTS[id(receipt)] = receipt
    return receipt


def _authenticate_action_receipt(
    value: PhysicalReducerActionReceipt,
    checkpoint: content.FrozenCausalContentAddressedSegmentSchemaV1,
) -> _ActionReceiptRecord:
    if (
        type(value) is not PhysicalReducerActionReceipt
        or _ISSUED_ACTION_RECEIPTS.get(id(value)) is not value
    ):
        raise TypeError("exact issued physical reducer action receipt required")
    record = value._record
    if (
        record.authority is not _ACTION_AUTHORITY
        or record.content_checkpoint_sha256 != checkpoint.checkpoint_sha256
        or record.sha256 != _digest(_ACTION_RECEIPT_NAMESPACE, _action_payload(record))
    ):
        raise RuntimeError("physical reducer action receipt authentication failed")
    return record


def _checkpoint_sha256(
    occurrence_checkpoint_sha256: str,
    content_checkpoint_sha256: str,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + occurrence_checkpoint_sha256.encode("ascii")
        + b"\0"
        + content_checkpoint_sha256.encode("ascii")
        + b"\0"
        + _array_frame("relay-excitatory", excitatory)
        + _array_frame("relay-inhibitory", inhibitory)
        + _canonical_json(
            {
                "occurrence_schema_cell_offset": occurrence.SCHEMA_CELL_OFFSET,
                "rebased_content_schema_cell_offset": REBASED_CONTENT_SCHEMA_CELL_OFFSET,
                "total_cell_count": TOTAL_CELL_COUNT,
            }
        )
    ).hexdigest()


class FrozenCausalOccurrenceExecutableReducerBridgeV1:
    """Immutable desired-free bridge from one neutral bundle to one witness."""

    __slots__ = (
        "_checkpoint_sha256",
        "_content_checkpoint",
        "_excitatory",
        "_inhibitory",
        "_occurrence_checkpoint",
        "_summary",
    )

    def __init__(
        self,
        occurrence_checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
        content_checkpoint: content.FrozenCausalContentAddressedSegmentSchemaV1,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: BridgeTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("occurrence executable checkpoints are builder-owned")
        if (
            type(occurrence_checkpoint)
            is not occurrence.FrozenAnonymousOccurrenceBundleSchemaV1
            or type(content_checkpoint)
            is not content.FrozenCausalContentAddressedSegmentSchemaV1
            or excitatory.shape != RELAY_STATE_SHAPE
            or inhibitory.shape != RELAY_STATE_SHAPE
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("occurrence executable checkpoint geometry changed")
        occurrence_checkpoint.authenticate()
        content_checkpoint.authenticate()
        self._occurrence_checkpoint = occurrence_checkpoint
        self._content_checkpoint = content_checkpoint
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha256(
            occurrence_checkpoint.checkpoint_sha256,
            content_checkpoint.checkpoint_sha256,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("occurrence executable training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> BridgeTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        self._occurrence_checkpoint.authenticate()
        self._content_checkpoint.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                self._occurrence_checkpoint.checkpoint_sha256,
                self._content_checkpoint.checkpoint_sha256,
                self._excitatory,
                self._inhibitory,
            )
        ):
            raise RuntimeError("occurrence executable checkpoint authentication failed")

    def build(
        self,
        events: Sequence[occurrence.PhysicalOccurrenceEvent],
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        occurrence_boundary_gate: bool = True,
        occurrence_completion_gate: bool = True,
        occurrence_replay_gate: bool = True,
        occurrence_equality_gate: bool = True,
        occurrence_relation_cell_gate: bool = True,
        occurrence_schema_cell_gate: bool = True,
        occurrence_action_cell_gate: bool = True,
        relay_cell_gate: bool = True,
        reducer_action_cell_gate: bool = True,
        residual_equality_gate: bool = True,
        residual_literal_role_gate: bool = True,
        residual_order_gate: bool = True,
        residual_upstream_cell_gate: bool = True,
        residual_relation_gate: bool = True,
        residual_cell_gate: bool = True,
    ) -> OccurrenceExecutableWitnessCertificate:
        return _build_witness(
            self,
            events,
            span_records,
            literal_records,
            signature_role_record,
            signature=signature,
            cases=cases,
            literals=literals,
            max_cost=max_cost,
            occurrence_boundary_gate=occurrence_boundary_gate,
            occurrence_completion_gate=occurrence_completion_gate,
            occurrence_replay_gate=occurrence_replay_gate,
            occurrence_equality_gate=occurrence_equality_gate,
            occurrence_relation_cell_gate=occurrence_relation_cell_gate,
            occurrence_schema_cell_gate=occurrence_schema_cell_gate,
            occurrence_action_cell_gate=occurrence_action_cell_gate,
            relay_cell_gate=relay_cell_gate,
            reducer_action_cell_gate=reducer_action_cell_gate,
            residual_equality_gate=residual_equality_gate,
            residual_literal_role_gate=residual_literal_role_gate,
            residual_order_gate=residual_order_gate,
            residual_upstream_cell_gate=residual_upstream_cell_gate,
            residual_relation_gate=residual_relation_gate,
            residual_cell_gate=residual_cell_gate,
        )

    def build_from_occurrence_certificate(
        self,
        certificate: occurrence.OccurrenceGraphSchemaCertificate,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        relay_cell_gate: bool = True,
        reducer_action_cell_gate: bool = True,
        residual_equality_gate: bool = True,
        residual_literal_role_gate: bool = True,
        residual_order_gate: bool = True,
        residual_upstream_cell_gate: bool = True,
        residual_relation_gate: bool = True,
        residual_cell_gate: bool = True,
    ) -> OccurrenceExecutableWitnessCertificate:
        """Continue from one exact occurrence certificate without recognition."""

        return _build_witness_from_occurrence_certificate(
            self,
            certificate,
            span_records,
            literal_records,
            signature_role_record,
            signature=signature,
            cases=cases,
            literals=literals,
            max_cost=max_cost,
            relay_cell_gate=relay_cell_gate,
            reducer_action_cell_gate=reducer_action_cell_gate,
            residual_equality_gate=residual_equality_gate,
            residual_literal_role_gate=residual_literal_role_gate,
            residual_order_gate=residual_order_gate,
            residual_upstream_cell_gate=residual_upstream_cell_gate,
            residual_relation_gate=residual_relation_gate,
            residual_cell_gate=residual_cell_gate,
        )


def _training_work_payload(value: BridgeTrainingWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def fit_causal_occurrence_executable_reducer_bridge_v1(
    occurrence_checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
    content_checkpoint: content.FrozenCausalContentAddressedSegmentSchemaV1,
    lessons: tuple[CausalOccurrenceExecutableLesson, ...],
) -> FrozenCausalOccurrenceExecutableReducerBridgeV1:
    """Fit one A/B/A-prime physical occurrence-to-static relay."""

    if (
        type(occurrence_checkpoint)
        is not occurrence.FrozenAnonymousOccurrenceBundleSchemaV1
        or type(content_checkpoint)
        is not content.FrozenCausalContentAddressedSegmentSchemaV1
        or type(lessons) is not tuple
        or len(lessons) != TRAIN_LESSON_COUNT
        or any(type(item) is not CausalOccurrenceExecutableLesson for item in lessons)
    ):
        raise TypeError("bridge fit requires exact checkpoints and three lessons")
    occurrence_checkpoint.authenticate()
    content_checkpoint.authenticate()
    excitatory = np.zeros(RELAY_STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    source_to_target: dict[int, int] = {}
    target_to_source: dict[int, int] = {}
    occurrence_cells: list[int] = []
    static_cells: list[int] = []
    reducer_cells: list[int] = []
    occurrence_work_roots: list[str] = []
    static_roots: list[str] = []
    relay_roots: list[str] = []
    action_roots: list[str] = []
    occurrence_steps = 0
    static_steps = 0
    excitatory_ticks = 0
    inhibitory_ticks = 0

    for lesson in lessons:
        certificate = occurrence_checkpoint.recognize(lesson.events)
        occurrence._authenticate_certificate(certificate)
        source_cell = certificate.schema_cell_id
        if source_cell is None:
            raise OccurrenceExecutableFitStop(
                STOP_OCCURRENCE_SCHEMA,
                "fit bundle did not yield an authenticated occurrence schema cell",
            )
        source = _source_ordinal(source_cell)
        static_observation = _recognize_static_schema(content_checkpoint, lesson)
        target_cell = static_observation.schema_cell_id
        target = _static_ordinal(target_cell)
        if (
            source in source_to_target and source_to_target[source] != target
        ) or (target in target_to_source and target_to_source[target] != source):
            raise OccurrenceExecutableFitStop(
                STOP_CAUSAL_PAIR_CONFLICT,
                "physical occurrence/static pairings are not a stable bijection",
            )
        source_to_target[source] = target
        target_to_source[target] = source
        e_ticks, h_ticks = _learn_relay(excitatory, inhibitory, source, target)
        excitatory_ticks += e_ticks
        inhibitory_ticks += h_ticks
        relay = _race_relay(excitatory, inhibitory, source_cell, cell_gate=True)
        if relay.status != READY or relay.static_schema_cell_id != target_cell:
            raise OccurrenceExecutableFitStop(
                STOP_CAUSAL_PAIR_CONFLICT,
                "updated relay did not reproduce its physical static consequence",
            )
        action_receipt = _issue_action_receipt(
            content_checkpoint, target_cell, cell_gate=True
        )
        action_record = _authenticate_action_receipt(
            action_receipt, content_checkpoint
        )
        if action_record.status != READY or action_record.reducer_action_cell_id is None:
            raise OccurrenceExecutableFitStop(
                action_record.status, action_record.stop_reason or action_record.status
            )
        occurrence_cells.append(source_cell)
        static_cells.append(target_cell)
        reducer_cells.append(action_record.reducer_action_cell_id)
        occurrence_work_roots.append(certificate.work.sha256)
        static_roots.append(static_observation.sha256)
        relay_roots.append(relay.sha256)
        action_roots.append(action_receipt.sha256)
        occurrence_steps += certificate.work.total_physical_cell_steps
        static_steps += static_observation.total_physical_cell_steps

    if (
        tuple(occurrence_cells) != (occurrence_cells[0], occurrence_cells[1], occurrence_cells[0])
        or occurrence_cells[0] == occurrence_cells[1]
        or tuple(static_cells) != (static_cells[0], static_cells[1], static_cells[0])
        or static_cells[0] == static_cells[1]
        or len(source_to_target) != SCHEMA_CAPACITY
        or len(target_to_source) != SCHEMA_CAPACITY
        or len(set(reducer_cells)) != SCHEMA_CAPACITY
    ):
        raise OccurrenceExecutableFitStop(
            STOP_CAUSAL_PAIR_CONFLICT,
            "A/B/A-prime did not learn two reusable occurrence/static/action paths",
        )

    checkpoint_sha = _checkpoint_sha256(
        occurrence_checkpoint.checkpoint_sha256,
        content_checkpoint.checkpoint_sha256,
        excitatory,
        inhibitory,
    )
    bridge_steps = (
        TRAIN_LESSON_COUNT * RELAY_RACE_CELL_STEPS
        + TRAIN_LESSON_COUNT * ACTION_RACE_CELL_STEPS
    )
    work_partial = BridgeTrainingWork(
        occurrence_steps,
        static_steps,
        TRAIN_LESSON_COUNT,
        TRAIN_LESSON_COUNT * RELAY_RACE_CELL_STEPS,
        TRAIN_LESSON_COUNT,
        TRAIN_LESSON_COUNT * ACTION_RACE_CELL_STEPS,
        bridge_steps,
        occurrence_steps + static_steps + bridge_steps,
        tuple(occurrence_work_roots),
        tuple(static_roots),
        tuple(relay_roots),
        tuple(action_roots),
        "",
    )
    work = work_partial._replace(
        sha256=_digest(_TRAIN_WORK_NAMESPACE, _training_work_payload(work_partial))
    )
    summary = BridgeTrainingSummary(
        TRAIN_LESSON_COUNT,
        tuple(occurrence_cells),
        tuple(static_cells),
        tuple(reducer_cells),
        tuple(
            sorted(
                (
                    occurrence.SCHEMA_CELL_OFFSET + source,
                    REBASED_CONTENT_SCHEMA_CELL_OFFSET + target,
                )
                for source, target in source_to_target.items()
            )
        ),
        excitatory_ticks,
        inhibitory_ticks,
        int(np.max(excitatory)),
        int(np.max(inhibitory)),
        NEW_PERSISTENT_STATE_BYTES,
        COMPOSED_PERSISTENT_STATE_BYTES,
        work,
        checkpoint_sha,
    )
    return FrozenCausalOccurrenceExecutableReducerBridgeV1(
        occurrence_checkpoint,
        content_checkpoint,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _build_work_payload(value: BridgeBuildWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _freeze_build_work(
    *,
    occurrence_steps: int,
    residual_steps: int,
    relay_steps: int,
    action_steps: int,
    packed_builds: int,
    authorized_slices: int,
    proof_materializations: int,
    packed_work_sha256: str | None,
) -> BridgeBuildWork:
    bridge_steps = relay_steps + action_steps
    partial = BridgeBuildWork(
        occurrence_steps,
        residual_steps,
        relay_steps,
        action_steps,
        bridge_steps,
        occurrence_steps + residual_steps + bridge_steps,
        packed_builds,
        authorized_slices,
        proof_materializations,
        packed_work_sha256,
        "",
    )
    return partial._replace(
        sha256=_digest(_BUILD_WORK_NAMESPACE, _build_work_payload(partial))
    )


def _witness_payload(receipt: _WitnessReceipt) -> dict[str, object]:
    return {
        field: (
            receipt.witness.sha256
            if field == "witness" and receipt.witness is not None
            else getattr(receipt, field)
        )
        for field in receipt._fields
        if field not in {"authority", "work", "sha256"}
    } | {"work_sha256": receipt.work.sha256}


def _issue_witness(
    checkpoint: FrozenCausalOccurrenceExecutableReducerBridgeV1,
    *,
    status: str,
    stop_reason: str | None,
    occurrence_certificate_sha256: str,
    occurrence_schema_cell_id: int | None,
    relay_race_sha256: str | None,
    static_schema_cell_id: int | None,
    action_receipt_sha256: str | None,
    reducer_action_cell_id: int | None,
    residual_certificate_sha256: str | None,
    structural_certificate_sha256: str | None,
    plan_slice_receipt_sha256: str | None,
    exact_proof_count: int,
    witness: breadth_v3.PackedGenericTypedWitness | None,
    work: BridgeBuildWork,
) -> OccurrenceExecutableWitnessCertificate:
    partial = _WitnessReceipt(
        _WITNESS_AUTHORITY,
        status,
        stop_reason,
        checkpoint.checkpoint_sha256,
        occurrence_certificate_sha256,
        occurrence_schema_cell_id,
        relay_race_sha256,
        static_schema_cell_id,
        action_receipt_sha256,
        reducer_action_cell_id,
        residual_certificate_sha256,
        structural_certificate_sha256,
        plan_slice_receipt_sha256,
        exact_proof_count,
        witness,
        DESIRED_READS_DURING_BUILD,
        PROBE_OUTPUT_READS_DURING_BUILD,
        DIRECT_OCCURRENCE_ACTION_READS,
        DIRECT_APPEND_ACTION_READS,
        SERIALIZED_SCHEMA_ACTION_READS,
        work,
        "",
    )
    receipt = partial._replace(
        sha256=_digest(_WITNESS_NAMESPACE, _witness_payload(partial))
    )
    certificate = OccurrenceExecutableWitnessCertificate(receipt)
    _ISSUED_WITNESS_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _build_witness(
    checkpoint: FrozenCausalOccurrenceExecutableReducerBridgeV1,
    events: Sequence[occurrence.PhysicalOccurrenceEvent],
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    signature: TypedSignature,
    cases: Iterable[tuple[object, ...]],
    literals: Iterable[int],
    max_cost: int,
    occurrence_boundary_gate: bool,
    occurrence_completion_gate: bool,
    occurrence_replay_gate: bool,
    occurrence_equality_gate: bool,
    occurrence_relation_cell_gate: bool,
    occurrence_schema_cell_gate: bool,
    occurrence_action_cell_gate: bool,
    relay_cell_gate: bool,
    reducer_action_cell_gate: bool,
    residual_equality_gate: bool,
    residual_literal_role_gate: bool,
    residual_order_gate: bool,
    residual_upstream_cell_gate: bool,
    residual_relation_gate: bool,
    residual_cell_gate: bool,
) -> OccurrenceExecutableWitnessCertificate:
    checkpoint.authenticate()
    certificate = checkpoint._occurrence_checkpoint.recognize(
        events,
        boundary_gate=occurrence_boundary_gate,
        completion_gate=occurrence_completion_gate,
        replay_gate=occurrence_replay_gate,
        equality_gate=occurrence_equality_gate,
        relation_cell_gate=occurrence_relation_cell_gate,
        schema_cell_gate=occurrence_schema_cell_gate,
        action_cell_gate=occurrence_action_cell_gate,
    )
    occurrence._authenticate_certificate(certificate)
    occurrence_steps = certificate.work.total_physical_cell_steps
    source_cell = certificate.schema_cell_id
    if source_cell is None:
        work = _freeze_build_work(
            occurrence_steps=occurrence_steps,
            residual_steps=0,
            relay_steps=0,
            action_steps=0,
            packed_builds=0,
            authorized_slices=0,
            proof_materializations=0,
            packed_work_sha256=None,
        )
        return _issue_witness(
            checkpoint,
            status=STOP_OCCURRENCE_SCHEMA,
            stop_reason="occurrence mechanism did not issue one physical schema cell",
            occurrence_certificate_sha256=certificate.sha256,
            occurrence_schema_cell_id=None,
            relay_race_sha256=None,
            static_schema_cell_id=None,
            action_receipt_sha256=None,
            reducer_action_cell_id=None,
            residual_certificate_sha256=None,
            structural_certificate_sha256=None,
            plan_slice_receipt_sha256=None,
            exact_proof_count=0,
            witness=None,
            work=work,
        )

    return _build_witness_from_occurrence_certificate(
        checkpoint,
        certificate,
        span_records,
        literal_records,
        signature_role_record,
        signature=signature,
        cases=cases,
        literals=literals,
        max_cost=max_cost,
        relay_cell_gate=relay_cell_gate,
        reducer_action_cell_gate=reducer_action_cell_gate,
        residual_equality_gate=residual_equality_gate,
        residual_literal_role_gate=residual_literal_role_gate,
        residual_order_gate=residual_order_gate,
        residual_upstream_cell_gate=residual_upstream_cell_gate,
        residual_relation_gate=residual_relation_gate,
        residual_cell_gate=residual_cell_gate,
    )


def _build_witness_from_occurrence_certificate(
    checkpoint: FrozenCausalOccurrenceExecutableReducerBridgeV1,
    certificate: occurrence.OccurrenceGraphSchemaCertificate,
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    signature: TypedSignature,
    cases: Iterable[tuple[object, ...]],
    literals: Iterable[int],
    max_cost: int,
    relay_cell_gate: bool,
    reducer_action_cell_gate: bool,
    residual_equality_gate: bool,
    residual_literal_role_gate: bool,
    residual_order_gate: bool,
    residual_upstream_cell_gate: bool,
    residual_relation_gate: bool,
    residual_cell_gate: bool,
) -> OccurrenceExecutableWitnessCertificate:
    checkpoint.authenticate()
    receipt = occurrence._authenticate_certificate(certificate)
    if receipt.checkpoint_sha256 != checkpoint._occurrence_checkpoint.checkpoint_sha256:
        raise ValueError("occurrence certificate checkpoint authority changed")
    binding = occurrence.require_occurrence_graph_schema(certificate)
    occurrence_steps = receipt.work.total_physical_cell_steps
    source_cell = binding.schema_cell_id

    relay = _race_relay(
        checkpoint._excitatory,
        checkpoint._inhibitory,
        source_cell,
        cell_gate=relay_cell_gate,
    )
    if relay.status != READY or relay.static_schema_cell_id is None:
        work = _freeze_build_work(
            occurrence_steps=occurrence_steps,
            residual_steps=0,
            relay_steps=RELAY_RACE_CELL_STEPS,
            action_steps=0,
            packed_builds=0,
            authorized_slices=0,
            proof_materializations=0,
            packed_work_sha256=None,
        )
        return _issue_witness(
            checkpoint,
            status=relay.status,
            stop_reason="learned occurrence-to-static relay abstained",
            occurrence_certificate_sha256=certificate.sha256,
            occurrence_schema_cell_id=source_cell,
            relay_race_sha256=relay.sha256,
            static_schema_cell_id=None,
            action_receipt_sha256=None,
            reducer_action_cell_id=None,
            residual_certificate_sha256=None,
            structural_certificate_sha256=None,
            plan_slice_receipt_sha256=None,
            exact_proof_count=0,
            witness=None,
            work=work,
        )

    action_receipt = _issue_action_receipt(
        checkpoint._content_checkpoint,
        relay.static_schema_cell_id,
        cell_gate=reducer_action_cell_gate,
    )
    action_record = _authenticate_action_receipt(
        action_receipt, checkpoint._content_checkpoint
    )
    if action_record.status != READY or action_record.reducer_action_cell_id is None:
        work = _freeze_build_work(
            occurrence_steps=occurrence_steps,
            residual_steps=0,
            relay_steps=RELAY_RACE_CELL_STEPS,
            action_steps=ACTION_RACE_CELL_STEPS,
            packed_builds=0,
            authorized_slices=0,
            proof_materializations=0,
            packed_work_sha256=None,
        )
        return _issue_witness(
            checkpoint,
            status=action_record.status,
            stop_reason=action_record.stop_reason,
            occurrence_certificate_sha256=certificate.sha256,
            occurrence_schema_cell_id=source_cell,
            relay_race_sha256=relay.sha256,
            static_schema_cell_id=relay.static_schema_cell_id,
            action_receipt_sha256=action_receipt.sha256,
            reducer_action_cell_id=None,
            residual_certificate_sha256=None,
            structural_certificate_sha256=None,
            plan_slice_receipt_sha256=None,
            exact_proof_count=0,
            witness=None,
            work=work,
        )

    residual_checkpoint = (
        checkpoint._content_checkpoint._segmented_checkpoint._append_checkpoint._residual_checkpoint
    )
    residual_certificate = residual_checkpoint.bind(
        span_records,
        literal_records,
        signature_role_record,
        equality_gate=residual_equality_gate,
        literal_role_gate=residual_literal_role_gate,
        order_gate=residual_order_gate,
        upstream_cell_gate=residual_upstream_cell_gate,
        residual_gate=residual_relation_gate,
        cell_gate=residual_cell_gate,
    )
    residual._authenticate_binding(residual_certificate)
    residual_steps = residual_certificate.work.stacked_binding_cell_steps
    try:
        residual_binding = residual.require_residual_literal_route_binding(
            residual_certificate
        )
    except RuntimeError as error:
        work = _freeze_build_work(
            occurrence_steps=occurrence_steps,
            residual_steps=residual_steps,
            relay_steps=RELAY_RACE_CELL_STEPS,
            action_steps=ACTION_RACE_CELL_STEPS,
            packed_builds=0,
            authorized_slices=0,
            proof_materializations=0,
            packed_work_sha256=None,
        )
        return _issue_witness(
            checkpoint,
            status=STOP_RESIDUAL_BINDING,
            stop_reason=str(error),
            occurrence_certificate_sha256=certificate.sha256,
            occurrence_schema_cell_id=source_cell,
            relay_race_sha256=relay.sha256,
            static_schema_cell_id=relay.static_schema_cell_id,
            action_receipt_sha256=action_receipt.sha256,
            reducer_action_cell_id=action_record.reducer_action_cell_id,
            residual_certificate_sha256=residual_certificate.sha256,
            structural_certificate_sha256=None,
            plan_slice_receipt_sha256=None,
            exact_proof_count=0,
            witness=None,
            work=work,
        )

    reducer_action = (
        action_record.reducer_action_cell_id - append.REDUCER_ACTION_CELL_OFFSET
    )
    scaffold = append._reducer_scaffold(residual_binding, reducer_action)
    delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
        signature,
        tuple(cases),
        tuple(literals),
        max_cost,
        scaffold,
    )
    packed_work_sha = delegate.work.sha256
    receipt = delegate.plan_slice_receipt
    if (
        delegate.status != plan_slice.FRONTIER_READY
        or delegate.state.authorized_slice_count != 1
        or receipt is None
    ):
        work = _freeze_build_work(
            occurrence_steps=occurrence_steps,
            residual_steps=residual_steps,
            relay_steps=RELAY_RACE_CELL_STEPS,
            action_steps=ACTION_RACE_CELL_STEPS,
            packed_builds=1,
            authorized_slices=delegate.state.authorized_slice_count,
            proof_materializations=0,
            packed_work_sha256=packed_work_sha,
        )
        return _issue_witness(
            checkpoint,
            status=STOP_PLAN_SLICE,
            stop_reason=delegate.status,
            occurrence_certificate_sha256=certificate.sha256,
            occurrence_schema_cell_id=source_cell,
            relay_race_sha256=relay.sha256,
            static_schema_cell_id=relay.static_schema_cell_id,
            action_receipt_sha256=action_receipt.sha256,
            reducer_action_cell_id=action_record.reducer_action_cell_id,
            residual_certificate_sha256=residual_certificate.sha256,
            structural_certificate_sha256=scaffold.sha256,
            plan_slice_receipt_sha256=None,
            exact_proof_count=0,
            witness=None,
            work=work,
        )

    binding = plan_slice._authenticate_plan_slice_receipt(
        delegate._frontier, receipt
    )
    try:
        forest = plan_slice._materialize_authorized_forest(binding)
    except packed.PackedRelationWorkLimitExceeded:
        work = _freeze_build_work(
            occurrence_steps=occurrence_steps,
            residual_steps=residual_steps,
            relay_steps=RELAY_RACE_CELL_STEPS,
            action_steps=ACTION_RACE_CELL_STEPS,
            packed_builds=1,
            authorized_slices=1,
            proof_materializations=0,
            packed_work_sha256=delegate.work.sha256,
        )
        return _issue_witness(
            checkpoint,
            status=STOP_WORK_BOUND,
            stop_reason="authorized singleton materialization exceeded packed work",
            occurrence_certificate_sha256=certificate.sha256,
            occurrence_schema_cell_id=source_cell,
            relay_race_sha256=relay.sha256,
            static_schema_cell_id=relay.static_schema_cell_id,
            action_receipt_sha256=action_receipt.sha256,
            reducer_action_cell_id=action_record.reducer_action_cell_id,
            residual_certificate_sha256=residual_certificate.sha256,
            structural_certificate_sha256=scaffold.sha256,
            plan_slice_receipt_sha256=receipt.sha256,
            exact_proof_count=0,
            witness=None,
            work=work,
        )

    proof_count = forest.exact_logical_recipe_count
    witnesses = forest.first_two_witnesses
    if proof_count != 1 or len(witnesses) != 1:
        work = _freeze_build_work(
            occurrence_steps=occurrence_steps,
            residual_steps=residual_steps,
            relay_steps=RELAY_RACE_CELL_STEPS,
            action_steps=ACTION_RACE_CELL_STEPS,
            packed_builds=1,
            authorized_slices=1,
            proof_materializations=1,
            packed_work_sha256=delegate.work.sha256,
        )
        return _issue_witness(
            checkpoint,
            status=STOP_WITNESS_NOT_SINGLETON,
            stop_reason="authorized plan slice did not materialize exactly one proof",
            occurrence_certificate_sha256=certificate.sha256,
            occurrence_schema_cell_id=source_cell,
            relay_race_sha256=relay.sha256,
            static_schema_cell_id=relay.static_schema_cell_id,
            action_receipt_sha256=action_receipt.sha256,
            reducer_action_cell_id=action_record.reducer_action_cell_id,
            residual_certificate_sha256=residual_certificate.sha256,
            structural_certificate_sha256=scaffold.sha256,
            plan_slice_receipt_sha256=receipt.sha256,
            exact_proof_count=proof_count,
            witness=None,
            work=work,
        )

    route_witness = binding.route._render_witness(witnesses[0], binding.component)
    witness = cast(
        breadth_v3.PackedGenericTypedWitness,
        delegate._frontier._compile_witness(route_witness),
    )
    work = _freeze_build_work(
        occurrence_steps=occurrence_steps,
        residual_steps=residual_steps,
        relay_steps=RELAY_RACE_CELL_STEPS,
        action_steps=ACTION_RACE_CELL_STEPS,
        packed_builds=1,
        authorized_slices=1,
        proof_materializations=1,
        packed_work_sha256=delegate.work.sha256,
    )
    return _issue_witness(
        checkpoint,
        status=READY,
        stop_reason=None,
        occurrence_certificate_sha256=certificate.sha256,
        occurrence_schema_cell_id=source_cell,
        relay_race_sha256=relay.sha256,
        static_schema_cell_id=relay.static_schema_cell_id,
        action_receipt_sha256=action_receipt.sha256,
        reducer_action_cell_id=action_record.reducer_action_cell_id,
        residual_certificate_sha256=residual_certificate.sha256,
        structural_certificate_sha256=scaffold.sha256,
        plan_slice_receipt_sha256=receipt.sha256,
        exact_proof_count=proof_count,
        witness=witness,
        work=work,
    )


def require_occurrence_executable_witness(
    value: OccurrenceExecutableWitnessCertificate,
) -> breadth_v3.PackedGenericTypedWitness:
    if (
        type(value) is not OccurrenceExecutableWitnessCertificate
        or _ISSUED_WITNESS_CERTIFICATES.get(id(value)) is not value
    ):
        raise TypeError("exact issued occurrence executable certificate required")
    receipt = value._receipt
    if (
        receipt.authority is not _WITNESS_AUTHORITY
        or receipt.work.sha256 != _digest(_BUILD_WORK_NAMESPACE, _build_work_payload(receipt.work))
        or receipt.sha256 != _digest(_WITNESS_NAMESPACE, _witness_payload(receipt))
    ):
        raise RuntimeError("occurrence executable witness certificate changed")
    if (
        receipt.status != READY
        or receipt.exact_proof_count != 1
        or receipt.witness is None
    ):
        raise ValueError(receipt.stop_reason or receipt.status)
    return receipt.witness


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "occurrence_source_commit": OCCURRENCE_SOURCE_COMMIT,
        "content_source_commit": CONTENT_SOURCE_COMMIT,
        "occurrence_cell_range": (OCCURRENCE_CELL_OFFSET, OCCURRENCE_CELL_STOP - 1),
        "rebased_content_cell_range": (
            REBASED_CONTENT_CELL_OFFSET,
            TOTAL_CELL_COUNT - 1,
        ),
        "relay_state_shape": RELAY_STATE_SHAPE,
        "new_persistent_state_bytes": NEW_PERSISTENT_STATE_BYTES,
        "segmented_composed_persistent_state_bytes": (
            SEGMENTED_COMPOSED_PERSISTENT_STATE_BYTES
        ),
        "composed_persistent_state_bytes": COMPOSED_PERSISTENT_STATE_BYTES,
        "total_cell_count": TOTAL_CELL_COUNT,
        "relay_race_cell_steps": RELAY_RACE_CELL_STEPS,
        "action_race_cell_steps": ACTION_RACE_CELL_STEPS,
        "desired_reads_during_build": DESIRED_READS_DURING_BUILD,
        "probe_output_reads_during_build": PROBE_OUTPUT_READS_DURING_BUILD,
        "direct_occurrence_action_reads": DIRECT_OCCURRENCE_ACTION_READS,
        "direct_append_action_reads": DIRECT_APPEND_ACTION_READS,
        "serialized_schema_action_reads": SERIALIZED_SCHEMA_ACTION_READS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", dict(CONFIG_RECORD))


__all__ = [
    "ACTION_RACE_CELL_STEPS",
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "COMPOSED_PERSISTENT_STATE_BYTES",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESIRED_READS_DURING_BUILD",
    "DEV_INPUT_COUNT",
    "DIRECT_APPEND_ACTION_READS",
    "DIRECT_OCCURRENCE_ACTION_READS",
    "G0_INPUT_COUNT",
    "HELDOUT_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "NEW_PERSISTENT_STATE_BYTES",
    "PATH2_INPUT_COUNT",
    "PROBE_OUTPUT_READS_DURING_BUILD",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "READY",
    "REBASED_CONTENT_SCHEMA_CELL_OFFSET",
    "RELAY_RACE_CELL_STEPS",
    "SCHEMA",
    "SERIALIZED_SCHEMA_ACTION_READS",
    "STOP_ACTION_AMBIGUOUS",
    "STOP_ACTION_ZERO",
    "STOP_CAUSAL_PAIR_CONFLICT",
    "STOP_OCCURRENCE_SCHEMA",
    "STOP_PLAN_SLICE",
    "STOP_RELAY_AMBIGUOUS",
    "STOP_RELAY_TONIC",
    "STOP_RESIDUAL_BINDING",
    "STOP_STATIC_DEPENDENCY",
    "STOP_WITNESS_NOT_SINGLETON",
    "STOP_WORK_BOUND",
    "TOTAL_CELL_COUNT",
    "CausalOccurrenceExecutableLesson",
    "FrozenCausalOccurrenceExecutableReducerBridgeV1",
    "OccurrenceExecutableFitStop",
    "OccurrenceExecutableWitnessCertificate",
    "fit_causal_occurrence_executable_reducer_bridge_v1",
    "require_occurrence_executable_witness",
]
