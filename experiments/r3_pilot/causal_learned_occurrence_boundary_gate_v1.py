"""Learn occurrence boundaries from physical completeness and recurrence.

Transport-separated self-posed bundles provide causal action interventions during
fit.  Plasticity receives only the unique physical action whose exact occurrence
consequence is READY.  Frozen inference consumes an authenticated occurrence
completion receipt plus physical root/previous-target equality, never a boundary
label, bundle length, interval change, endpoint range, event-order marker, or
serialized action.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_predictive_occurrence_stream_grouper_v1 as predecessor
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as segmented
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter
from humanbrain.resting_glif_race import CompiledRestingGlifRace, RestingGlifRaceFailure

SCHEMA = "CAUSAL_LEARNED_OCCURRENCE_BOUNDARY_GATE_V1"
READY = "CAUSAL_LEARNED_OCCURRENCE_BOUNDARY_GATE_V1_READY"
STOP_FEATURE_ZERO = "STOP_LEARNED_OCCURRENCE_BOUNDARY_FEATURE_ZERO"
STOP_GATE_ABSTAIN = "STOP_LEARNED_OCCURRENCE_BOUNDARY_GATE_ABSTAIN"
STOP_PREMATURE_NEW = "STOP_LEARNED_OCCURRENCE_BOUNDARY_PREMATURE_NEW"
STOP_ROUTER = "STOP_LEARNED_OCCURRENCE_BOUNDARY_ROUTER"
STOP_REPLAY = "STOP_LEARNED_OCCURRENCE_BOUNDARY_REPLAY"
STOP_OCCURRENCE = "STOP_LEARNED_OCCURRENCE_BOUNDARY_OCCURRENCE"
STOP_CAPACITY = "STOP_LEARNED_OCCURRENCE_BOUNDARY_CAPACITY"
STOP_CONSEQUENCE = "STOP_LEARNED_OCCURRENCE_BOUNDARY_CONSEQUENCE"

CLAIM_BOUNDARY = (
    "self-posed continuous rooted pure-chain/pure-fanout streams only: a learned "
    "four-way physical completeness-by-recurrence gate removes internal grouping "
    "cues for that grammar, while transport separation during fit, initial stream "
    "start, terminal completion, endpoint framing, graph consequence projection, "
    "and the chain/fanout grammar remain explicit; no mixed-topology, parsing, "
    "language, coding, public, sealed, or broad boundary-learning evidence"
)

INCOMPLETE_RECURRENT = 0
INCOMPLETE_NOVEL = 1
COMPLETE_RECURRENT = 2
COMPLETE_NOVEL = 3
FEATURE_COUNT = 4
ACTION_COUNT = 2
CONTINUE_ACTION = segmented.CONTINUE_RELATION
NEW_ACTION = segmented.NEW_RELATION

FEATURE_CELL_OFFSET = predecessor.TOTAL_CELL_COUNT
FEATURE_CELL_COUNT = FEATURE_COUNT
FEATURE_CELL_IDS = tuple(
    FEATURE_CELL_OFFSET + index for index in range(FEATURE_CELL_COUNT)
)
NEW_CELL_COUNT = FEATURE_CELL_COUNT
TOTAL_CELL_COUNT = predecessor.TOTAL_CELL_COUNT + NEW_CELL_COUNT

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
STATE_SHAPE = (FEATURE_COUNT, ACTION_COUNT)
TRAINABLE_STATE_BYTES = 2 * int(np.prod(STATE_SHAPE)) * TICK_DTYPE.itemsize
PERSISTENT_STATE_BYTES = TRAINABLE_STATE_BYTES
RUNTIME_STATE_BYTES = predecessor.RUNTIME_STATE_BYTES
TRAIN_TRANSPORT_COUNT = 5

FEATURE_RACE_STEPS = 30
FEATURE_RACE_CELL_STEPS = FEATURE_CELL_COUNT * FEATURE_RACE_STEPS
GATE_OUTPUT_RACE_CELL_STEPS = 3 * output_projection.RACE_STEPS
CONSEQUENCE_RACE_CELL_STEPS = segmented.ROUTER_RACE_CELL_STEPS
OUTPUT_CURRENT_PA = segmented.OUTPUT_CURRENT_PA
_FEATURE_RACE = CompiledRestingGlifRace.compile(step_count=FEATURE_RACE_STEPS)

BOUNDARY_BOOLEAN_INPUT_COUNT = 0
OPEN_EXTEND_EOS_INPUT_COUNT = 0
HOST_GROUP_LENGTH_PLASTICITY_INPUT_COUNT = 0
INTERVAL_GATE_INPUT_COUNT = 0
SERIALIZED_ACTION_INPUT_COUNT = 0
PUBLIC_TRAIN_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"causal-learned-occurrence-boundary-gate-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_CONSTRUCTION_TOKEN = object()

RawNeutralOccurrenceEvent = predecessor.RawNeutralOccurrenceEvent


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _array_frame(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return (
        name.encode("ascii")
        + b"\0"
        + array.dtype.str.encode("ascii")
        + b"\0"
        + _canonical_json(array.shape)
        + b"\0"
        + memoryview(array).cast("B")
    )


class BoundaryGateFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class BoundaryGateTrainingSummary(NamedTuple):
    transport_count: int
    feature_exposures: tuple[int, ...]
    action_intervention_count: int
    consequence_race_count: int
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    trainable_state_bytes: int
    occurrence_recognition_count: int
    physical_cell_steps: int
    consequence_certificate_sha256s: tuple[str, ...]
    checkpoint_sha256: str


class LearnedBoundaryWork(NamedTuple):
    input_raw_event_count: int
    accepted_raw_event_count: int
    boundary_event_indices: tuple[int, ...]
    feature_race_count: int
    feature_race_cell_steps: int
    gate_output_race_count: int
    gate_output_race_cell_steps: int
    recurrence_equality_comparison_count: int
    recurrence_equality_cell_steps: int
    router_race_count: int
    router_race_cell_steps: int
    trigger_replay_count: int
    trigger_replay_cell_steps: int
    occurrence_recognition_count: int
    occurrence_cell_steps: int
    total_physical_cell_steps: int
    feature_trace_sha256s: tuple[str, ...]
    gate_trace_sha256s: tuple[str, ...]
    occurrence_certificate_sha256s: tuple[str, ...]
    sha256: str


class LearnedBoundaryStreamOutcome(NamedTuple):
    status: str
    stop_reason: str | None
    certificates: tuple[occurrence.OccurrenceGraphSchemaCertificate, ...]
    candidate_certificate: occurrence.OccurrenceGraphSchemaCertificate | None
    accepted_raw_event_count: int
    stop_event_index: int | None
    work: LearnedBoundaryWork
    sha256: str


@dataclass(slots=True)
class _GateLedger:
    feature_race_count: int = 0
    gate_output_race_count: int = 0
    consequence_race_count: int = 0
    action_intervention_count: int = 0
    feature_roots: list[str] | None = None
    gate_roots: list[str] | None = None
    consequence_roots: list[str] | None = None
    boundary_event_indices: list[int] | None = None

    def __post_init__(self) -> None:
        self.feature_roots = []
        self.gate_roots = []
        self.consequence_roots = []
        self.boundary_event_indices = []


def _checkpoint_sha256(
    segmented_sha256: str,
    occurrence_sha256: str,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + segmented_sha256.encode("ascii")
        + occurrence_sha256.encode("ascii")
        + _array_frame("excitatory", excitatory)
        + _array_frame("inhibitory", inhibitory)
    ).hexdigest()


class FrozenLearnedOccurrenceBoundaryGateV1:
    """Immutable physical gate over exact banked router and occurrence states."""

    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_occurrence",
        "_segmented",
        "_summary",
    )

    def __init__(
        self,
        segmented_checkpoint: segmented.FrozenSegmentedAggregateV1,
        occurrence_checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: BoundaryGateTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("learned boundary checkpoints are builder-owned")
        if type(segmented_checkpoint) is not segmented.FrozenSegmentedAggregateV1:
            raise TypeError("exact banked segmented checkpoint required")
        if type(occurrence_checkpoint) is not occurrence.FrozenAnonymousOccurrenceBundleSchemaV1:
            raise TypeError("exact banked occurrence checkpoint required")
        if (
            excitatory.shape != STATE_SHAPE
            or inhibitory.shape != STATE_SHAPE
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("learned boundary state geometry changed")
        segmented_checkpoint.authenticate()
        occurrence_checkpoint.authenticate()
        self._segmented = segmented_checkpoint
        self._occurrence = occurrence_checkpoint
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha256(
            segmented_checkpoint.checkpoint_sha256,
            occurrence_checkpoint.checkpoint_sha256,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("learned boundary training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> BoundaryGateTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def excitatory(self) -> tuple[tuple[int, ...], ...]:
        self.authenticate()
        return tuple(tuple(int(item) for item in row) for row in self._excitatory)

    @property
    def inhibitory(self) -> tuple[tuple[int, ...], ...]:
        self.authenticate()
        return tuple(tuple(int(item) for item in row) for row in self._inhibitory)

    def authenticate(self) -> None:
        self._segmented.authenticate()
        self._occurrence.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                self._segmented.checkpoint_sha256,
                self._occurrence.checkpoint_sha256,
                self._excitatory,
                self._inhibitory,
            )
        ):
            raise RuntimeError("learned boundary checkpoint authentication failed")

    def group(
        self,
        raw_events: Sequence[RawNeutralOccurrenceEvent],
        *,
        completion_cell_gate: bool = True,
        recurrence_cell_gate: bool = True,
        gate_cell_gate: bool = True,
        router_cell_gate: bool = True,
        replay_cell_gate: bool = True,
        boundary_cell_gate: bool = True,
        equality_cell_gate: bool = True,
        relation_cell_gate: bool = True,
        schema_cell_gate: bool = True,
        action_cell_gate: bool = True,
    ) -> LearnedBoundaryStreamOutcome:
        return _group(
            self,
            raw_events,
            completion_cell_gate=completion_cell_gate,
            recurrence_cell_gate=recurrence_cell_gate,
            gate_cell_gate=gate_cell_gate,
            router_cell_gate=router_cell_gate,
            replay_cell_gate=replay_cell_gate,
            boundary_cell_gate=boundary_cell_gate,
            equality_cell_gate=equality_cell_gate,
            relation_cell_gate=relation_cell_gate,
            schema_cell_gate=schema_cell_gate,
            action_cell_gate=action_cell_gate,
        )


def _copy_buffer(source: predecessor._RawBuffer) -> predecessor._RawBuffer:
    target = predecessor._RawBuffer()
    for index in range(source.count):
        target.append(
            RawNeutralOccurrenceEvent(
                tuple(int(value) for value in source.storage[index, 0]),
                tuple(int(value) for value in source.storage[index, 1]),
                segmenter.INTERVAL_CELL_IDS[0],
            )
        )
    return target


def _feature_index(complete: bool, recurrent: bool) -> int:
    if complete:
        return COMPLETE_RECURRENT if recurrent else COMPLETE_NOVEL
    return INCOMPLETE_RECURRENT if recurrent else INCOMPLETE_NOVEL


def _race_feature(
    complete: bool,
    recurrent: bool,
    ledger: _GateLedger,
    *,
    cell_gate: bool,
) -> tuple[str, int | None]:
    expected = _feature_index(complete, recurrent)
    currents = np.zeros(FEATURE_COUNT, dtype=np.float64)
    if cell_gate:
        currents[expected] = OUTPUT_CURRENT_PA
    ledger.feature_race_count += 1
    try:
        race = _FEATURE_RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure:
        status = STOP_FEATURE_ZERO
        winner = None
        spike_step = None
        simultaneous = 0
    else:
        spike_step = race.spike_step
        simultaneous = race.simultaneous_spike_candidates
        if simultaneous != 1 or race.winner_cell != expected:
            status = STOP_FEATURE_ZERO
            winner = None
        else:
            status = READY
            winner = int(race.winner_cell)
    root = _digest(
        _NAMESPACE + b"feature-race\0",
        {
            "complete_cell_id": occurrence.COMPLETION_REPLAY_CELL if complete else None,
            "recurrence_evidence": "PHYSICAL_EQUAL" if recurrent else "PHYSICAL_NOVEL",
            "recurrence_active_lane_count": (
                occurrence.NODE_ACTIVE_CELL_COUNT if recurrent else 0
            ),
            "expected": expected,
            "winner": winner,
            "status": status,
            "spike_step": spike_step,
            "simultaneous": simultaneous,
        },
    )
    assert ledger.feature_roots is not None
    ledger.feature_roots.append(root)
    return status, winner


def _observe_feature(
    occurrence_checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
    buffer: predecessor._RawBuffer,
    event: RawNeutralOccurrenceEvent,
    predecessor_ledger: predecessor._Ledger,
    auxiliary: occurrence._WorkLedger,
    gate_ledger: _GateLedger,
    *,
    completion_cell_gate: bool,
    recurrence_cell_gate: bool,
    gate_cell_gate: bool,
    boundary_cell_gate: bool,
    equality_cell_gate: bool,
    relation_cell_gate: bool,
    schema_cell_gate: bool,
    action_cell_gate: bool,
) -> tuple[str, int | None, occurrence.OccurrenceGraphSchemaCertificate]:
    candidate = predecessor._recognize(
        occurrence_checkpoint,
        buffer,
        predecessor_ledger,
        boundary_cell_gate=boundary_cell_gate,
        completion_cell_gate=completion_cell_gate,
        equality_cell_gate=equality_cell_gate,
        relation_cell_gate=relation_cell_gate,
        schema_cell_gate=schema_cell_gate,
        action_cell_gate=action_cell_gate,
    )
    complete = candidate.status == occurrence.READY
    if complete:
        occurrence.require_occurrence_graph_schema(candidate)
    root_match = occurrence._physical_equal(
        event.source_cells,
        buffer.root_cells(),
        auxiliary,
        category="validation",
        equality_gate=recurrence_cell_gate,
    )
    previous_match = occurrence._physical_equal(
        event.source_cells,
        buffer.previous_target_cells(),
        auxiliary,
        category="validation",
        equality_gate=recurrence_cell_gate,
    )
    status, feature = _race_feature(
        complete,
        root_match or previous_match,
        gate_ledger,
        cell_gate=gate_cell_gate,
    )
    return status, feature, candidate


def _route_continue(
    segmented_checkpoint: segmented.FrozenSegmentedAggregateV1,
    event: RawNeutralOccurrenceEvent,
    buffer: predecessor._RawBuffer,
    ledger: predecessor._Ledger,
    *,
    router_cell_gate: bool,
) -> None:
    accepted, status = predecessor._router_action(
        segmented_checkpoint,
        segmented.CONTINUE_RELATION,
        segmented.EXTEND_ACTION,
        ledger,
        router_cell_gate=router_cell_gate,
    )
    if not accepted:
        raise BoundaryGateFitStop(
            STOP_ROUTER, f"CONTINUE-to-EXTEND router abstained: {status}"
        )
    buffer.append(event)
    ledger.accepted_raw_event_count += 1


def _intervention(
    segmented_checkpoint: segmented.FrozenSegmentedAggregateV1,
    occurrence_checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
    prefix: predecessor._RawBuffer,
    trigger: RawNeutralOccurrenceEvent,
    followup: Sequence[RawNeutralOccurrenceEvent],
    action: int,
    predecessor_ledger: predecessor._Ledger,
    auxiliary: occurrence._WorkLedger,
    gate_ledger: _GateLedger,
) -> occurrence.OccurrenceGraphSchemaCertificate:
    gate_ledger.action_intervention_count += 1
    if action == CONTINUE_ACTION:
        candidate = _copy_buffer(prefix)
        _route_continue(
            segmented_checkpoint,
            trigger,
            candidate,
            predecessor_ledger,
            router_cell_gate=True,
        )
    elif action == NEW_ACTION:
        candidate = predecessor._RawBuffer()
        status, reason = predecessor._start_or_replay(
            segmented_checkpoint,
            trigger,
            candidate,
            predecessor_ledger,
            auxiliary,
            router_cell_gate=True,
            replay_cell_gate=True,
        )
        if status is not None:
            raise BoundaryGateFitStop(status, reason or status)
    else:
        raise RuntimeError("unknown banked boundary action")
    for event in followup:
        _route_continue(
            segmented_checkpoint,
            event,
            candidate,
            predecessor_ledger,
            router_cell_gate=True,
        )
    return predecessor._recognize(
        occurrence_checkpoint,
        candidate,
        predecessor_ledger,
        boundary_cell_gate=True,
        completion_cell_gate=True,
        equality_cell_gate=True,
        relation_cell_gate=True,
        schema_cell_gate=True,
        action_cell_gate=True,
    )


def _race_consequence(
    certificates: tuple[
        occurrence.OccurrenceGraphSchemaCertificate,
        occurrence.OccurrenceGraphSchemaCertificate,
    ],
    ledger: _GateLedger,
) -> int:
    currents = np.asarray(
        [
            OUTPUT_CURRENT_PA if certificate.status == occurrence.READY else 0.0
            for certificate in certificates
        ],
        dtype=np.float64,
    )
    ledger.consequence_race_count += 1
    try:
        race = segmented._ROUTER_RACE.race_many(
            currents[None, :], require_unique=False
        )[0]
    except RestingGlifRaceFailure as exc:
        raise BoundaryGateFitStop(
            STOP_CONSEQUENCE, "both physical action consequences stopped"
        ) from exc
    if race.simultaneous_spike_candidates != 1:
        raise BoundaryGateFitStop(
            STOP_CONSEQUENCE, "physical action consequences were ambiguous"
        )
    winner = int(race.winner_cell)
    root = _digest(
        _NAMESPACE + b"consequence-race\0",
        {
            "certificate_sha256s": tuple(value.sha256 for value in certificates),
            "statuses": tuple(value.status for value in certificates),
            "winner_action_cell_id": (
                segmenter.CONTINUE_CELL_ID
                if winner == CONTINUE_ACTION
                else segmenter.NEW_CELL_ID
            ),
            "spike_step": race.spike_step,
        },
    )
    assert ledger.consequence_roots is not None
    ledger.consequence_roots.append(root)
    return winner


def _update(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    feature: int,
    winner: int,
) -> tuple[int, int]:
    added_e = 0
    added_h = 0
    if excitatory[feature, winner] < TICK_CAP:
        excitatory[feature, winner] += 1
        added_e += 1
    for action in range(ACTION_COUNT):
        if action != winner and inhibitory[feature, action] < TICK_CAP:
            inhibitory[feature, action] += 1
            added_h += 1
    return added_e, added_h


def fit_causal_learned_occurrence_boundary_gate_v1(
    segmented_checkpoint: segmented.FrozenSegmentedAggregateV1,
    occurrence_checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
    transport_bundles: Sequence[Sequence[RawNeutralOccurrenceEvent]],
) -> FrozenLearnedOccurrenceBoundaryGateV1:
    """Learn three observable gate cases from physical action consequences."""

    if type(segmented_checkpoint) is not segmented.FrozenSegmentedAggregateV1:
        raise TypeError("exact banked segmented checkpoint required")
    if type(occurrence_checkpoint) is not occurrence.FrozenAnonymousOccurrenceBundleSchemaV1:
        raise TypeError("exact banked occurrence checkpoint required")
    segmented_checkpoint.authenticate()
    occurrence_checkpoint.authenticate()
    if isinstance(transport_bundles, (str, bytes)) or not isinstance(
        transport_bundles, Sequence
    ):
        raise TypeError("training transports must be a finite sequence")
    bundles = tuple(
        tuple(predecessor._normalize_raw(event) for event in bundle)
        for bundle in transport_bundles
    )
    if len(bundles) != TRAIN_TRANSPORT_COUNT or any(
        len(bundle) < 2 for bundle in bundles
    ):
        raise ValueError("focused fit requires five nontrivial physical transports")
    interval_ids = {
        event.interval_cell_id for bundle in bundles for event in bundle
    }
    if len(interval_ids) != 1:
        raise ValueError("training intervals must remain physically constant")

    excitatory = np.zeros(STATE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    feature_exposures = np.zeros(FEATURE_COUNT, dtype=np.int64)
    added_e = 0
    added_h = 0
    predecessor_ledger = predecessor._Ledger(
        input_raw_event_count=sum(len(bundle) for bundle in bundles)
    )
    auxiliary = occurrence._WorkLedger()
    gate_ledger = _GateLedger()
    trained_buffers: list[predecessor._RawBuffer] = []

    for bundle in bundles:
        buffer = predecessor._RawBuffer()
        status, reason = predecessor._start_or_replay(
            segmented_checkpoint,
            bundle[0],
            buffer,
            predecessor_ledger,
            auxiliary,
            router_cell_gate=True,
            replay_cell_gate=True,
        )
        if status is not None:
            raise BoundaryGateFitStop(status, reason or status)
        for trigger in bundle[1:]:
            status, feature, _candidate = _observe_feature(
                occurrence_checkpoint,
                buffer,
                trigger,
                predecessor_ledger,
                auxiliary,
                gate_ledger,
                completion_cell_gate=True,
                recurrence_cell_gate=True,
                gate_cell_gate=True,
                boundary_cell_gate=True,
                equality_cell_gate=True,
                relation_cell_gate=True,
                schema_cell_gate=True,
                action_cell_gate=True,
            )
            if status != READY or feature is None:
                raise BoundaryGateFitStop(status, "feature population abstained")
            certificates = (
                _intervention(
                    segmented_checkpoint,
                    occurrence_checkpoint,
                    buffer,
                    trigger,
                    (),
                    CONTINUE_ACTION,
                    predecessor_ledger,
                    auxiliary,
                    gate_ledger,
                ),
                _intervention(
                    segmented_checkpoint,
                    occurrence_checkpoint,
                    buffer,
                    trigger,
                    (),
                    NEW_ACTION,
                    predecessor_ledger,
                    auxiliary,
                    gate_ledger,
                ),
            )
            winner = _race_consequence(certificates, gate_ledger)
            tick_e, tick_h = _update(
                excitatory, inhibitory, feature, winner
            )
            added_e += tick_e
            added_h += tick_h
            feature_exposures[feature] += 1
            if winner == CONTINUE_ACTION:
                _route_continue(
                    segmented_checkpoint,
                    trigger,
                    buffer,
                    predecessor_ledger,
                    router_cell_gate=True,
                )
            else:
                raise BoundaryGateFitStop(
                    STOP_CONSEQUENCE,
                    "a within-transport physical consequence selected NEW",
                )
        terminal = predecessor._recognize(
            occurrence_checkpoint,
            buffer,
            predecessor_ledger,
            boundary_cell_gate=True,
            completion_cell_gate=True,
            equality_cell_gate=True,
            relation_cell_gate=True,
            schema_cell_gate=True,
            action_cell_gate=True,
        )
        occurrence.require_occurrence_graph_schema(terminal)
        trained_buffers.append(buffer)

    for index in range(len(bundles) - 1):
        trigger = bundles[index + 1][0]
        followup = (bundles[index + 1][1],)
        status, feature, _candidate = _observe_feature(
            occurrence_checkpoint,
            trained_buffers[index],
            trigger,
            predecessor_ledger,
            auxiliary,
            gate_ledger,
            completion_cell_gate=True,
            recurrence_cell_gate=True,
            gate_cell_gate=True,
            boundary_cell_gate=True,
            equality_cell_gate=True,
            relation_cell_gate=True,
            schema_cell_gate=True,
            action_cell_gate=True,
        )
        if status != READY or feature is None:
            raise BoundaryGateFitStop(status, "boundary feature population abstained")
        certificates = (
            _intervention(
                segmented_checkpoint,
                occurrence_checkpoint,
                trained_buffers[index],
                trigger,
                followup,
                CONTINUE_ACTION,
                predecessor_ledger,
                auxiliary,
                gate_ledger,
            ),
            _intervention(
                segmented_checkpoint,
                occurrence_checkpoint,
                trained_buffers[index],
                trigger,
                followup,
                NEW_ACTION,
                predecessor_ledger,
                auxiliary,
                gate_ledger,
            ),
        )
        winner = _race_consequence(certificates, gate_ledger)
        tick_e, tick_h = _update(excitatory, inhibitory, feature, winner)
        added_e += tick_e
        added_h += tick_h
        feature_exposures[feature] += 1

    checkpoint_sha = _checkpoint_sha256(
        segmented_checkpoint.checkpoint_sha256,
        occurrence_checkpoint.checkpoint_sha256,
        excitatory,
        inhibitory,
    )
    auxiliary_work = occurrence._freeze_work(auxiliary)
    physical_steps = (
        gate_ledger.feature_race_count * FEATURE_RACE_CELL_STEPS
        + gate_ledger.consequence_race_count * CONSEQUENCE_RACE_CELL_STEPS
        + predecessor_ledger.router_race_count * segmented.ROUTER_RACE_CELL_STEPS
        + predecessor_ledger.occurrence_cell_steps
        + auxiliary_work.total_physical_cell_steps
    )
    summary = BoundaryGateTrainingSummary(
        len(bundles),
        tuple(int(value) for value in feature_exposures),
        gate_ledger.action_intervention_count,
        gate_ledger.consequence_race_count,
        added_e,
        added_h,
        int(np.max(excitatory)),
        int(np.max(inhibitory)),
        TRAINABLE_STATE_BYTES,
        predecessor_ledger.occurrence_recognition_count,
        physical_steps,
        tuple(predecessor_ledger.certificate_roots or ()),
        checkpoint_sha,
    )
    return FrozenLearnedOccurrenceBoundaryGateV1(
        segmented_checkpoint,
        occurrence_checkpoint,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _freeze_work(
    predecessor_ledger: predecessor._Ledger,
    auxiliary: occurrence._WorkLedger,
    gate_ledger: _GateLedger,
) -> LearnedBoundaryWork:
    auxiliary_work = occurrence._freeze_work(auxiliary)
    feature_steps = gate_ledger.feature_race_count * FEATURE_RACE_CELL_STEPS
    gate_steps = gate_ledger.gate_output_race_count * GATE_OUTPUT_RACE_CELL_STEPS
    router_steps = (
        predecessor_ledger.router_race_count * segmented.ROUTER_RACE_CELL_STEPS
    )
    total = (
        feature_steps
        + gate_steps
        + router_steps
        + auxiliary_work.total_physical_cell_steps
        + predecessor_ledger.occurrence_cell_steps
    )
    partial = LearnedBoundaryWork(
        predecessor_ledger.input_raw_event_count,
        predecessor_ledger.accepted_raw_event_count,
        tuple(gate_ledger.boundary_event_indices or ()),
        gate_ledger.feature_race_count,
        feature_steps,
        gate_ledger.gate_output_race_count,
        gate_steps,
        auxiliary.validation_equality_comparison_count,
        auxiliary.validation_equality_comparison_count
        * occurrence.EQUALITY_CELL_STEPS_PER_COMPARISON,
        predecessor_ledger.router_race_count,
        router_steps,
        auxiliary.replay_event_count,
        auxiliary.replay_event_count * occurrence.COMPLETION_RACE_CELL_STEPS,
        predecessor_ledger.occurrence_recognition_count,
        predecessor_ledger.occurrence_cell_steps,
        total,
        tuple(gate_ledger.feature_roots or ()),
        tuple(gate_ledger.gate_roots or ()),
        tuple(predecessor_ledger.certificate_roots or ()),
        "",
    )
    payload = {
        field: getattr(partial, field)
        for field in partial._fields
        if field != "sha256"
    }
    return partial._replace(sha256=_digest(_NAMESPACE + b"work\0", payload))


def _outcome(
    status: str,
    reason: str | None,
    certificates: Sequence[occurrence.OccurrenceGraphSchemaCertificate],
    candidate: occurrence.OccurrenceGraphSchemaCertificate | None,
    stop_event_index: int | None,
    predecessor_ledger: predecessor._Ledger,
    auxiliary: occurrence._WorkLedger,
    gate_ledger: _GateLedger,
) -> LearnedBoundaryStreamOutcome:
    frozen = tuple(certificates)
    work = _freeze_work(predecessor_ledger, auxiliary, gate_ledger)
    payload = {
        "status": status,
        "reason": reason,
        "certificates": tuple(value.sha256 for value in frozen),
        "candidate": None if candidate is None else candidate.sha256,
        "accepted": predecessor_ledger.accepted_raw_event_count,
        "stop_event_index": stop_event_index,
        "work_sha256": work.sha256,
    }
    return LearnedBoundaryStreamOutcome(
        status,
        reason,
        frozen,
        candidate,
        predecessor_ledger.accepted_raw_event_count,
        stop_event_index,
        work,
        _digest(_NAMESPACE + b"outcome\0", payload),
    )


def _group(
    checkpoint: FrozenLearnedOccurrenceBoundaryGateV1,
    raw_events: Sequence[RawNeutralOccurrenceEvent],
    *,
    completion_cell_gate: bool,
    recurrence_cell_gate: bool,
    gate_cell_gate: bool,
    router_cell_gate: bool,
    replay_cell_gate: bool,
    boundary_cell_gate: bool,
    equality_cell_gate: bool,
    relation_cell_gate: bool,
    schema_cell_gate: bool,
    action_cell_gate: bool,
) -> LearnedBoundaryStreamOutcome:
    checkpoint.authenticate()
    if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
        raise TypeError("raw occurrence stream must be a finite sequence")
    events = tuple(predecessor._normalize_raw(value) for value in raw_events)
    predecessor_ledger = predecessor._Ledger(input_raw_event_count=len(events))
    auxiliary = occurrence._WorkLedger()
    gate_ledger = _GateLedger()
    buffer = predecessor._RawBuffer()
    completed: list[occurrence.OccurrenceGraphSchemaCertificate] = []
    candidate: occurrence.OccurrenceGraphSchemaCertificate | None = None

    for event_index, event in enumerate(events):
        if buffer.count == 0:
            status, reason = predecessor._start_or_replay(
                checkpoint._segmented,
                event,
                buffer,
                predecessor_ledger,
                auxiliary,
                router_cell_gate=router_cell_gate,
                replay_cell_gate=replay_cell_gate,
            )
            if status is not None:
                mapped = STOP_REPLAY if status == predecessor.STOP_REPLAY else STOP_ROUTER
                return _outcome(
                    mapped,
                    reason,
                    completed,
                    candidate,
                    event_index,
                    predecessor_ledger,
                    auxiliary,
                    gate_ledger,
                )
            continue
        if buffer.count >= predecessor.MAX_BUFFERED_RAW_EDGES:
            return _outcome(
                STOP_CAPACITY,
                "bounded raw occurrence buffer is full",
                completed,
                candidate,
                event_index,
                predecessor_ledger,
                auxiliary,
                gate_ledger,
            )

        feature_status, feature, candidate = _observe_feature(
            checkpoint._occurrence,
            buffer,
            event,
            predecessor_ledger,
            auxiliary,
            gate_ledger,
            completion_cell_gate=completion_cell_gate,
            recurrence_cell_gate=recurrence_cell_gate,
            gate_cell_gate=gate_cell_gate,
            boundary_cell_gate=boundary_cell_gate,
            equality_cell_gate=equality_cell_gate,
            relation_cell_gate=relation_cell_gate,
            schema_cell_gate=schema_cell_gate,
            action_cell_gate=action_cell_gate,
        )
        if feature_status != READY or feature is None:
            return _outcome(
                STOP_FEATURE_ZERO,
                "physical completeness-by-recurrence population emitted zero",
                completed,
                candidate,
                event_index,
                predecessor_ledger,
                auxiliary,
                gate_ledger,
            )
        race = segmenter._race_decision(
            checkpoint._excitatory,
            checkpoint._inhibitory,
            cue_index=feature,
            cue_gate=gate_cell_gate,
            complete_hold=False,
            match_count=0,
            mismatch_count=0,
        )
        gate_ledger.gate_output_race_count += 1
        gate_root = _digest(
            _NAMESPACE + b"gate-output-race\0",
            {
                "event_index": event_index,
                "feature_cell_id": FEATURE_CELL_IDS[feature],
                "decision": race.decision,
                "raw_scores": race.raw_scores,
            },
        )
        assert gate_ledger.gate_roots is not None
        gate_ledger.gate_roots.append(gate_root)
        if race.decision not in ("CONTINUE", "NEW"):
            return _outcome(
                STOP_GATE_ABSTAIN,
                f"learned boundary race abstained with {race.decision}",
                completed,
                candidate,
                event_index,
                predecessor_ledger,
                auxiliary,
                gate_ledger,
            )
        if race.decision == "NEW":
            if candidate.status != occurrence.READY:
                return _outcome(
                    STOP_PREMATURE_NEW,
                    "NEW won without an authentic occurrence-completion receipt",
                    completed,
                    candidate,
                    event_index,
                    predecessor_ledger,
                    auxiliary,
                    gate_ledger,
                )
            occurrence.require_occurrence_graph_schema(candidate)
            completed.append(candidate)
            assert gate_ledger.boundary_event_indices is not None
            gate_ledger.boundary_event_indices.append(event_index)
            buffer.clear()
            status, reason = predecessor._start_or_replay(
                checkpoint._segmented,
                event,
                buffer,
                predecessor_ledger,
                auxiliary,
                router_cell_gate=router_cell_gate,
                replay_cell_gate=replay_cell_gate,
            )
            if status is not None:
                mapped = STOP_REPLAY if status == predecessor.STOP_REPLAY else STOP_ROUTER
                return _outcome(
                    mapped,
                    reason,
                    completed,
                    candidate,
                    event_index,
                    predecessor_ledger,
                    auxiliary,
                    gate_ledger,
                )
            candidate = None
            continue
        try:
            _route_continue(
                checkpoint._segmented,
                event,
                buffer,
                predecessor_ledger,
                router_cell_gate=router_cell_gate,
            )
        except BoundaryGateFitStop as exc:
            return _outcome(
                exc.status,
                exc.reason,
                completed,
                candidate,
                event_index,
                predecessor_ledger,
                auxiliary,
                gate_ledger,
            )

    if buffer.count == 0:
        return _outcome(
            STOP_OCCURRENCE,
            "transport ended without a raw occurrence",
            completed,
            candidate,
            None,
            predecessor_ledger,
            auxiliary,
            gate_ledger,
        )
    candidate = predecessor._recognize(
        checkpoint._occurrence,
        buffer,
        predecessor_ledger,
        boundary_cell_gate=boundary_cell_gate,
        completion_cell_gate=completion_cell_gate,
        equality_cell_gate=equality_cell_gate,
        relation_cell_gate=relation_cell_gate,
        schema_cell_gate=schema_cell_gate,
        action_cell_gate=action_cell_gate,
    )
    if candidate.status != occurrence.READY:
        return _outcome(
            STOP_OCCURRENCE,
            candidate.stop_reason or candidate.status,
            completed,
            candidate,
            None,
            predecessor_ledger,
            auxiliary,
            gate_ledger,
        )
    occurrence.require_occurrence_graph_schema(candidate)
    completed.append(candidate)
    return _outcome(
        READY,
        None,
        completed,
        candidate,
        None,
        predecessor_ledger,
        auxiliary,
        gate_ledger,
    )


__all__ = [
    "ACTION_COUNT",
    "BOUNDARY_BOOLEAN_INPUT_COUNT",
    "CLAIM_BOUNDARY",
    "COMPLETE_NOVEL",
    "COMPLETE_RECURRENT",
    "FEATURE_CELL_IDS",
    "FEATURE_COUNT",
    "HOST_GROUP_LENGTH_PLASTICITY_INPUT_COUNT",
    "INCOMPLETE_NOVEL",
    "INCOMPLETE_RECURRENT",
    "INTERVAL_GATE_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "OPEN_EXTEND_EOS_INPUT_COUNT",
    "PERSISTENT_STATE_BYTES",
    "READY",
    "RUNTIME_STATE_BYTES",
    "SCHEMA",
    "SERIALIZED_ACTION_INPUT_COUNT",
    "STOP_CAPACITY",
    "STOP_CONSEQUENCE",
    "STOP_FEATURE_ZERO",
    "STOP_GATE_ABSTAIN",
    "STOP_OCCURRENCE",
    "STOP_PREMATURE_NEW",
    "STOP_REPLAY",
    "STOP_ROUTER",
    "TOTAL_CELL_COUNT",
    "TRAINABLE_STATE_BYTES",
    "BoundaryGateFitStop",
    "BoundaryGateTrainingSummary",
    "FrozenLearnedOccurrenceBoundaryGateV1",
    "LearnedBoundaryStreamOutcome",
    "LearnedBoundaryWork",
    "RawNeutralOccurrenceEvent",
    "fit_causal_learned_occurrence_boundary_gate_v1",
]
