"""Infer occurrence grouping from raw neutral edge events.

The caller supplies only two physical endpoint assemblies and one banked opaque
interval cell per event.  This module reuses the banked predictive segmenter,
the learned NEW/CONTINUE router, and the anonymous occurrence learner's exact
completion/replay and certificate authorities.  It adds no cells or trainable
state.  In particular, it does not invent a boundary when the banked predictor
collides: cue-independent continuous streams must STOP honestly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as segmented
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter

SCHEMA = "CAUSAL_PREDICTIVE_OCCURRENCE_STREAM_GROUPER_V1"
READY = "CAUSAL_PREDICTIVE_OCCURRENCE_STREAM_GROUPER_V1_READY"
STOP_SEGMENT_AMBIGUITY = "STOP_PREDICTIVE_OCCURRENCE_SEGMENT_AMBIGUITY"
STOP_PREMATURE_NEW = "STOP_PREDICTIVE_OCCURRENCE_PREMATURE_NEW"
STOP_ROUTER = "STOP_PREDICTIVE_OCCURRENCE_ROUTER"
STOP_REPLAY = "STOP_PREDICTIVE_OCCURRENCE_REPLAY"
STOP_OCCURRENCE = "STOP_PREDICTIVE_OCCURRENCE_SCHEMA"
STOP_CAPACITY = "STOP_PREDICTIVE_OCCURRENCE_CAPACITY"

CLAIM_BOUNDARY = (
    "self-posed neutral rooted chain/fanout edge streams only: stream start and "
    "transport termination remain environmental framing; internal grouping is "
    "accepted only when the banked predictive race wins, so interval-neutral "
    "continuous multi-bundle streams currently abstain; no parsing, language, "
    "coding, public, sealed, or broad boundary-learning evidence"
)

NEW_CELL_COUNT = 0
TRAINABLE_STATE_BYTES = 0
MAX_BUFFERED_RAW_EDGES = 6
RAW_EDGE_BUFFER_BYTES = MAX_BUFFERED_RAW_EDGES * 2 * occurrence.NODE_ACTIVE_CELL_COUNT
ACTIVE_ROOT_BYTES = occurrence.NODE_ACTIVE_CELL_COUNT
PREVIOUS_TARGET_BYTES = occurrence.NODE_ACTIVE_CELL_COUNT
RUNTIME_LATCH_BYTES = 4
RUNTIME_STATE_BYTES = (
    RAW_EDGE_BUFFER_BYTES
    + ACTIVE_ROOT_BYTES
    + PREVIOUS_TARGET_BYTES
    + RUNTIME_LATCH_BYTES
)
TOTAL_CELL_COUNT = occurrence.TOTAL_CELL_COUNT

BOUNDARY_BOOLEAN_INPUT_COUNT = 0
OPEN_EXTEND_EOS_INPUT_COUNT = 0
HOST_GROUP_LENGTH_INPUT_COUNT = 0
PUBLIC_TRAIN_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"causal-predictive-occurrence-stream-grouper-v1\0"


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(namespace: bytes, value: object) -> str:
    payload = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(namespace + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RawNeutralOccurrenceEvent:
    """One raw physical edge and one opaque timing cell; no grouping field."""

    source_cells: tuple[int, ...]
    target_cells: tuple[int, ...]
    interval_cell_id: int


class PredictiveOccurrenceGrouperWork(NamedTuple):
    input_raw_event_count: int
    accepted_raw_event_count: int
    segment_race_count: int
    segment_race_cell_steps: int
    recurrence_equality_comparison_count: int
    recurrence_equality_cell_steps: int
    router_race_count: int
    router_race_cell_steps: int
    trigger_replay_count: int
    trigger_replay_cell_steps: int
    occurrence_recognition_count: int
    occurrence_cell_steps: int
    total_physical_cell_steps: int
    segment_trace_sha256s: tuple[str, ...]
    router_trace_sha256s: tuple[str, ...]
    occurrence_certificate_sha256s: tuple[str, ...]
    sha256: str


class PredictiveOccurrenceStreamOutcome(NamedTuple):
    status: str
    stop_reason: str | None
    certificates: tuple[occurrence.OccurrenceGraphSchemaCertificate, ...]
    candidate_certificate: occurrence.OccurrenceGraphSchemaCertificate | None
    accepted_raw_event_count: int
    stop_event_index: int | None
    work: PredictiveOccurrenceGrouperWork
    sha256: str


@dataclass(slots=True)
class _Ledger:
    input_raw_event_count: int = 0
    accepted_raw_event_count: int = 0
    segment_race_count: int = 0
    router_race_count: int = 0
    occurrence_recognition_count: int = 0
    occurrence_cell_steps: int = 0
    segment_roots: list[str] | None = None
    router_roots: list[str] | None = None
    certificate_roots: list[str] | None = None

    def __post_init__(self) -> None:
        self.segment_roots = []
        self.router_roots = []
        self.certificate_roots = []


class _RawBuffer:
    """The exact 60-byte logical runtime state budget."""

    __slots__ = ("control", "previous_target", "root", "storage")

    def __init__(self) -> None:
        self.storage = np.zeros(
            (MAX_BUFFERED_RAW_EDGES, 2, occurrence.NODE_ACTIVE_CELL_COUNT),
            dtype=np.uint8,
        )
        self.root = np.zeros(occurrence.NODE_ACTIVE_CELL_COUNT, dtype=np.uint8)
        self.previous_target = np.zeros(
            occurrence.NODE_ACTIVE_CELL_COUNT, dtype=np.uint8
        )
        # count, active, replay-pending, completed-count
        self.control = np.zeros(RUNTIME_LATCH_BYTES, dtype=np.uint8)

    @property
    def count(self) -> int:
        return int(self.control[0])

    def append(self, event: RawNeutralOccurrenceEvent) -> None:
        index = self.count
        if index >= MAX_BUFFERED_RAW_EDGES:
            raise OverflowError("raw occurrence buffer is full")
        source = np.asarray(event.source_cells, dtype=np.uint8)
        target = np.asarray(event.target_cells, dtype=np.uint8)
        self.storage[index, 0] = source
        self.storage[index, 1] = target
        if index == 0:
            self.root[:] = source
            self.control[1] = 1
        self.previous_target[:] = target
        self.control[0] = index + 1

    def clear(self) -> None:
        self.storage.fill(0)
        self.root.fill(0)
        self.previous_target.fill(0)
        completed = int(self.control[3]) + 1
        self.control.fill(0)
        self.control[3] = min(completed, 255)

    def root_cells(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.root)

    def previous_target_cells(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.previous_target)

    def physical_events(self) -> tuple[occurrence.PhysicalOccurrenceEvent, ...]:
        if self.count == 0:
            raise RuntimeError("cannot close an empty occurrence buffer")
        root = self.root_cells()
        values: list[occurrence.PhysicalOccurrenceEvent] = [
            occurrence.PhysicalOccurrenceEvent(
                occurrence.OPEN_CUE_CELL,
                root,
                root,
            )
        ]
        for index in range(self.count):
            values.append(
                occurrence.PhysicalOccurrenceEvent(
                    occurrence.EXTEND_CUE_CELL,
                    tuple(int(value) for value in self.storage[index, 0]),
                    tuple(int(value) for value in self.storage[index, 1]),
                )
            )
        values.append(
            occurrence.PhysicalOccurrenceEvent(occurrence.EOS_CUE_CELL, (), ())
        )
        return tuple(values)


def _normalize_raw(value: object) -> RawNeutralOccurrenceEvent:
    if type(value) is not RawNeutralOccurrenceEvent:
        raise TypeError("raw stream entries must be exact RawNeutralOccurrenceEvent values")
    if type(value.interval_cell_id) is not int:
        raise ValueError("interval cell must be an exact integer")
    if value.interval_cell_id not in segmenter.INTERVAL_CELL_IDS:
        raise ValueError("interval cell is outside the banked predictor population")
    return RawNeutralOccurrenceEvent(
        occurrence._node_cells(value.source_cells, "source cells"),
        occurrence._node_cells(value.target_cells, "target cells"),
        value.interval_cell_id,
    )


def _work(ledger: _Ledger, auxiliary: occurrence._WorkLedger) -> PredictiveOccurrenceGrouperWork:
    auxiliary_work = occurrence._freeze_work(auxiliary)
    segment_steps = ledger.segment_race_count * segmenter.SEGMENTATION_CELL_STEPS_PER_EVENT
    router_steps = ledger.router_race_count * segmented.ROUTER_RACE_CELL_STEPS
    total = (
        segment_steps
        + router_steps
        + auxiliary_work.total_physical_cell_steps
        + ledger.occurrence_cell_steps
    )
    partial = PredictiveOccurrenceGrouperWork(
        ledger.input_raw_event_count,
        ledger.accepted_raw_event_count,
        ledger.segment_race_count,
        segment_steps,
        auxiliary.validation_equality_comparison_count,
        auxiliary.validation_equality_comparison_count
        * occurrence.EQUALITY_CELL_STEPS_PER_COMPARISON,
        ledger.router_race_count,
        router_steps,
        auxiliary.replay_event_count,
        auxiliary.replay_event_count * occurrence.COMPLETION_RACE_CELL_STEPS,
        ledger.occurrence_recognition_count,
        ledger.occurrence_cell_steps,
        total,
        tuple(ledger.segment_roots or ()),
        tuple(ledger.router_roots or ()),
        tuple(ledger.certificate_roots or ()),
        "",
    )
    payload = {field: getattr(partial, field) for field in partial._fields if field != "sha256"}
    return partial._replace(sha256=_digest(_NAMESPACE + b"work\0", payload))


def _outcome(
    status: str,
    reason: str | None,
    certificates: Sequence[occurrence.OccurrenceGraphSchemaCertificate],
    candidate: occurrence.OccurrenceGraphSchemaCertificate | None,
    stop_event_index: int | None,
    ledger: _Ledger,
    auxiliary: occurrence._WorkLedger,
) -> PredictiveOccurrenceStreamOutcome:
    frozen_certificates = tuple(certificates)
    work = _work(ledger, auxiliary)
    payload = {
        "schema": SCHEMA,
        "status": status,
        "stop_reason": reason,
        "certificates": tuple(value.sha256 for value in frozen_certificates),
        "candidate": None if candidate is None else candidate.sha256,
        "accepted_raw_event_count": ledger.accepted_raw_event_count,
        "stop_event_index": stop_event_index,
        "work_sha256": work.sha256,
    }
    return PredictiveOccurrenceStreamOutcome(
        status,
        reason,
        frozen_certificates,
        candidate,
        ledger.accepted_raw_event_count,
        stop_event_index,
        work,
        _digest(_NAMESPACE + b"outcome\0", payload),
    )


def _router_action(
    checkpoint: segmented.FrozenSegmentedAggregateV1,
    relation: int,
    expected_action: int,
    ledger: _Ledger,
    *,
    router_cell_gate: bool,
) -> tuple[bool, str]:
    status, action, root = segmented._race_router(
        checkpoint._excitatory,
        checkpoint._inhibitory,
        relation,
        cell_gate=router_cell_gate,
    )
    ledger.router_race_count += 1
    assert ledger.router_roots is not None
    ledger.router_roots.append(root)
    return status == segmented.READY and action == expected_action, status


def _recognize(
    checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
    buffer: _RawBuffer,
    ledger: _Ledger,
    *,
    boundary_cell_gate: bool,
    completion_cell_gate: bool,
    equality_cell_gate: bool,
    relation_cell_gate: bool,
    schema_cell_gate: bool,
    action_cell_gate: bool,
) -> occurrence.OccurrenceGraphSchemaCertificate:
    certificate = checkpoint.recognize(
        buffer.physical_events(),
        boundary_gate=boundary_cell_gate,
        completion_gate=completion_cell_gate,
        replay_gate=True,
        equality_gate=equality_cell_gate,
        relation_cell_gate=relation_cell_gate,
        schema_cell_gate=schema_cell_gate,
        action_cell_gate=action_cell_gate,
    )
    receipt = occurrence._authenticate_certificate(certificate)
    if receipt.checkpoint_sha256 != checkpoint.checkpoint_sha256:
        raise RuntimeError("occurrence certificate escaped its exact checkpoint")
    ledger.occurrence_recognition_count += 1
    ledger.occurrence_cell_steps += certificate.work.total_physical_cell_steps
    assert ledger.certificate_roots is not None
    ledger.certificate_roots.append(certificate.sha256)
    return certificate


def _start_or_replay(
    checkpoint: segmented.FrozenSegmentedAggregateV1,
    event: RawNeutralOccurrenceEvent,
    buffer: _RawBuffer,
    ledger: _Ledger,
    auxiliary: occurrence._WorkLedger,
    *,
    router_cell_gate: bool,
    replay_cell_gate: bool,
) -> tuple[str | None, str | None]:
    opened, router_status = _router_action(
        checkpoint,
        segmented.NEW_RELATION,
        segmented.OPEN_ACTION,
        ledger,
        router_cell_gate=router_cell_gate,
    )
    if not opened:
        return STOP_ROUTER, f"NEW-to-OPEN router abstained: {router_status}"
    try:
        occurrence._completion_tick(
            auxiliary, completion_gate=replay_cell_gate, replay=True
        )
    except occurrence.OccurrenceBundleFitStop as exc:
        return STOP_REPLAY, exc.reason
    extended, router_status = _router_action(
        checkpoint,
        segmented.CONTINUE_RELATION,
        segmented.EXTEND_ACTION,
        ledger,
        router_cell_gate=router_cell_gate,
    )
    if not extended:
        return STOP_ROUTER, f"replayed CONTINUE-to-EXTEND abstained: {router_status}"
    buffer.append(event)
    ledger.accepted_raw_event_count += 1
    return None, None


def group_causal_predictive_occurrence_stream_v1(
    segmented_checkpoint: segmented.FrozenSegmentedAggregateV1,
    occurrence_checkpoint: occurrence.FrozenAnonymousOccurrenceBundleSchemaV1,
    raw_events: Sequence[RawNeutralOccurrenceEvent],
    *,
    segmenter_cue_gate: bool = True,
    recurrence_cell_gate: bool = True,
    router_cell_gate: bool = True,
    replay_cell_gate: bool = True,
    boundary_cell_gate: bool = True,
    completion_cell_gate: bool = True,
    equality_cell_gate: bool = True,
    relation_cell_gate: bool = True,
    schema_cell_gate: bool = True,
    action_cell_gate: bool = True,
) -> PredictiveOccurrenceStreamOutcome:
    """Group a finite raw transport stream or return an authentic causal STOP.

    Transport start opens the first candidate and transport termination requests
    completion of only the final candidate.  Neither fact is consulted to place
    an internal boundary.
    """

    if type(segmented_checkpoint) is not segmented.FrozenSegmentedAggregateV1:
        raise TypeError("exact banked segmented checkpoint required")
    if type(occurrence_checkpoint) is not occurrence.FrozenAnonymousOccurrenceBundleSchemaV1:
        raise TypeError("exact banked occurrence checkpoint required")
    segmented_checkpoint.authenticate()
    occurrence_checkpoint.authenticate()
    if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
        raise TypeError("raw occurrence stream must be a finite sequence")

    ledger = _Ledger(input_raw_event_count=len(raw_events))
    auxiliary = occurrence._WorkLedger()
    buffer = _RawBuffer()
    completed: list[occurrence.OccurrenceGraphSchemaCertificate] = []
    candidate: occurrence.OccurrenceGraphSchemaCertificate | None = None

    for event_index, raw in enumerate(raw_events):
        event = _normalize_raw(raw)
        if buffer.count == 0:
            status, reason = _start_or_replay(
                segmented_checkpoint,
                event,
                buffer,
                ledger,
                auxiliary,
                router_cell_gate=router_cell_gate,
                replay_cell_gate=replay_cell_gate,
            )
            if status is not None:
                return _outcome(
                    status,
                    reason,
                    completed,
                    None,
                    event_index,
                    ledger,
                    auxiliary,
                )
            continue

        if buffer.count >= MAX_BUFFERED_RAW_EDGES:
            return _outcome(
                STOP_CAPACITY,
                "bounded raw occurrence buffer is full",
                completed,
                candidate,
                event_index,
                ledger,
                auxiliary,
            )

        candidate = _recognize(
            occurrence_checkpoint,
            buffer,
            ledger,
            boundary_cell_gate=boundary_cell_gate,
            completion_cell_gate=completion_cell_gate,
            equality_cell_gate=equality_cell_gate,
            relation_cell_gate=relation_cell_gate,
            schema_cell_gate=schema_cell_gate,
            action_cell_gate=action_cell_gate,
        )
        complete_hold = candidate.status == occurrence.READY
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
        recurrent = root_match or previous_match
        cue_index = segmenter.INTERVAL_CELL_IDS.index(event.interval_cell_id)
        race = segmenter._race_decision(
            segmented_checkpoint._segmenter._cue_excitatory,
            segmented_checkpoint._segmenter._cue_inhibitory,
            cue_index=cue_index,
            cue_gate=segmenter_cue_gate,
            complete_hold=complete_hold,
            match_count=segmenter.LANES if recurrent else 0,
            mismatch_count=0 if recurrent else segmenter.LANES,
        )
        ledger.segment_race_count += 1
        race_root = _digest(
            _NAMESPACE + b"segment-race\0",
            {
                "event_index": event_index,
                "interval_cell_id": event.interval_cell_id,
                "complete_hold": complete_hold,
                "decision": race.decision,
                "raw_scores": race.raw_scores,
                "match_count": race.match_count,
                "mismatch_count": race.mismatch_count,
            },
        )
        assert ledger.segment_roots is not None
        ledger.segment_roots.append(race_root)

        if race.decision not in ("CONTINUE", "NEW"):
            return _outcome(
                STOP_SEGMENT_AMBIGUITY,
                f"banked predictive race abstained with {race.decision}",
                completed,
                candidate,
                event_index,
                ledger,
                auxiliary,
            )
        if race.decision == "NEW":
            if not complete_hold:
                return _outcome(
                    STOP_PREMATURE_NEW,
                    "NEW won before exact occurrence completion",
                    completed,
                    candidate,
                    event_index,
                    ledger,
                    auxiliary,
                )
            occurrence.require_occurrence_graph_schema(candidate)
            completed.append(candidate)
            buffer.clear()
            status, reason = _start_or_replay(
                segmented_checkpoint,
                event,
                buffer,
                ledger,
                auxiliary,
                router_cell_gate=router_cell_gate,
                replay_cell_gate=replay_cell_gate,
            )
            if status is not None:
                return _outcome(
                    status,
                    reason,
                    completed,
                    candidate,
                    event_index,
                    ledger,
                    auxiliary,
                )
            candidate = None
            continue

        extended, router_status = _router_action(
            segmented_checkpoint,
            segmented.CONTINUE_RELATION,
            segmented.EXTEND_ACTION,
            ledger,
            router_cell_gate=router_cell_gate,
        )
        if not extended:
            return _outcome(
                STOP_ROUTER,
                f"CONTINUE-to-EXTEND router abstained: {router_status}",
                completed,
                candidate,
                event_index,
                ledger,
                auxiliary,
            )
        buffer.append(event)
        ledger.accepted_raw_event_count += 1

    if buffer.count == 0:
        return _outcome(
            STOP_OCCURRENCE,
            "transport ended without a raw occurrence",
            completed,
            candidate,
            None,
            ledger,
            auxiliary,
        )
    candidate = _recognize(
        occurrence_checkpoint,
        buffer,
        ledger,
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
            ledger,
            auxiliary,
        )
    occurrence.require_occurrence_graph_schema(candidate)
    completed.append(candidate)
    return _outcome(
        READY,
        None,
        completed,
        candidate,
        None,
        ledger,
        auxiliary,
    )


__all__ = [
    "BOUNDARY_BOOLEAN_INPUT_COUNT",
    "CLAIM_BOUNDARY",
    "HOST_GROUP_LENGTH_INPUT_COUNT",
    "MAX_BUFFERED_RAW_EDGES",
    "NEW_CELL_COUNT",
    "OPEN_EXTEND_EOS_INPUT_COUNT",
    "READY",
    "RUNTIME_STATE_BYTES",
    "SCHEMA",
    "STOP_CAPACITY",
    "STOP_OCCURRENCE",
    "STOP_PREMATURE_NEW",
    "STOP_REPLAY",
    "STOP_ROUTER",
    "STOP_SEGMENT_AMBIGUITY",
    "TOTAL_CELL_COUNT",
    "TRAINABLE_STATE_BYTES",
    "PredictiveOccurrenceGrouperWork",
    "PredictiveOccurrenceStreamOutcome",
    "RawNeutralOccurrenceEvent",
    "group_causal_predictive_occurrence_stream_v1",
]
