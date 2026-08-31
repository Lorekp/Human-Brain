"""Learn anonymous occurrence-graph relays from neutral physical event bundles.

The banked segmented aggregate exposes learned OPEN/EXTEND cells.  This module
adds only the next reusable seam: recover complete bundles (including replay of
an OPEN event that completed the preceding bundle), factor physical endpoint
equalities into anonymous edge relations, and let two schema populations recall
generic graph-construction actions.

Training receives no schema/family label and no desired action.  The only
teaching signal is the physical graph consequence ``(root arity, max depth)``
produced by the four generic actions.  Frozen recognition validates a recalled
relay against the observed physical edges, but that validation never chooses an
action.  The result is an opaque occurrence-graph certificate; this module does
not import or manufacture an OccurrenceFactorStaticCertificate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as segmented
from humanbrain.resting_glif_race import (
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
)

SCHEMA = "CAUSAL_ANONYMOUS_OCCURRENCE_BUNDLE_SCHEMA_V1"
READY = "CAUSAL_ANONYMOUS_OCCURRENCE_BUNDLE_SCHEMA_V1_READY"
STOP_BOUNDARY_ZERO = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_BOUNDARY_ZERO"
STOP_BOUNDARY_SEQUENCE = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_BOUNDARY_SEQUENCE"
STOP_REPLAY = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_REPLAY"
STOP_INCOMPLETE = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_INCOMPLETE"
STOP_RELATION_ZERO = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_RELATION_ZERO"
STOP_RELATION_CONFLICT = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_RELATION_CONFLICT"
STOP_ACTION_ZERO = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_ACTION_ZERO"
STOP_ACTION_AMBIGUOUS = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_ACTION_AMBIGUOUS"
STOP_CONSEQUENCE = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_CONSEQUENCE"
STOP_SCHEMA_ZERO = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_SCHEMA_ZERO"
STOP_SCHEMA_AMBIGUOUS = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_SCHEMA_AMBIGUOUS"
STOP_TONIC = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_TONIC"
STOP_CAPACITY = "STOP_CAUSAL_ANONYMOUS_OCCURRENCE_CAPACITY"

CLAIM_BOUNDARY = (
    "self-posed neutral event-bundle evidence only: factorized physical "
    "endpoint equality, learned OPEN/EXTEND/EOS recovery, two anonymous "
    "occurrence schemas, and four generic graph actions transfer chain/fanout "
    "relays across fresh lengths and opaque cell relabeling; physical event "
    "framing, endpoint assemblies, equality wiring, graph-action scaffolds, "
    "and consequence projection remain explicit, so this is not canonical "
    "static inference, parsing, public transfer, SWE, or coding capability"
)

# The only upstream signals used here are the already-banked OPEN/EXTEND cells.
OPEN_CUE_CELL = segmented.OPEN_ACTION_CELL
EXTEND_CUE_CELL = segmented.EXTEND_ACTION_CELL

ROOT_SELF_RELATION = 0
FIRST_ATTACHMENT_RELATION = 1
DESCENDANT_RELATION = 2
SIBLING_RELATION = 3
NONLOCAL_SOURCE_RELATION = 4
UNSEEN_SOURCE_RELATION = 5
REUSED_TARGET_RELATION = 6
SELF_EDGE_RELATION = 7
COMPLETE_RELATION = 8
CONFLICT_RELATION = 9
RELATION_CELL_COUNT = 10

SCHEMA_CAPACITY = 2
NOVEL_SCHEMA = 2
TONIC_SCHEMA = 3
SCHEMA_RACE_CELL_COUNT = 4

OPEN_ROOT_ACTION = 0
ATTACH_DESCENDANT_ACTION = 1
ATTACH_SIBLING_ACTION = 2
COMPLETE_ACTION = 3
ACTION_CELL_COUNT = 4

RELATION_CELL_OFFSET = segmented.TOTAL_CELL_COUNT
SCHEMA_CELL_OFFSET = RELATION_CELL_OFFSET + RELATION_CELL_COUNT
ACTION_CELL_OFFSET = SCHEMA_CELL_OFFSET + SCHEMA_RACE_CELL_COUNT
COMPLETION_REPLAY_CELL = ACTION_CELL_OFFSET + ACTION_CELL_COUNT
EOS_CUE_CELL = COMPLETION_REPLAY_CELL
NEW_CELL_COUNT = 19
TOTAL_CELL_COUNT = segmented.TOTAL_CELL_COUNT + NEW_CELL_COUNT
if COMPLETION_REPLAY_CELL != TOTAL_CELL_COUNT - 1:
    raise RuntimeError("anonymous occurrence cell allocation changed")

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
ACTION_POLICY_SHAPE = (SCHEMA_CAPACITY, RELATION_CELL_COUNT, ACTION_CELL_COUNT)
ACTION_POLICY_STATE_BYTES = 2 * int(np.prod(ACTION_POLICY_SHAPE))
SCHEMA_VALID_STATE_BYTES = SCHEMA_CAPACITY
PROVENANCE_STATE_BYTES = 8
REPLAY_LATCH_STATE_BYTES = 8
PERSISTENT_STATE_BYTES = (
    ACTION_POLICY_STATE_BYTES
    + SCHEMA_VALID_STATE_BYTES
    + PROVENANCE_STATE_BYTES
    + REPLAY_LATCH_STATE_BYTES
)
if PERSISTENT_STATE_BYTES != 178:
    raise RuntimeError("anonymous occurrence persistent-state budget changed")

NODE_CELL_WIDTH = 64
NODE_ACTIVE_CELL_COUNT = 4
MAX_BUNDLE_EVENT_COUNT = 8
TRAIN_BUNDLE_COUNT = 3

BOUNDARY_RACE_STEPS = 20
RELATION_RACE_STEPS = 30
SCHEMA_RACE_STEPS = 30
ACTION_RECALL_RACE_STEPS = 30
ACTION_INTERVENTION_RACE_STEPS = 60
COMPLETION_RACE_STEPS = 30
OUTPUT_CURRENT_PA = 10_000.0
TONIC_CURRENT_PA = 10_000.0
WINNER_CURRENT_PA = 20_000.0
EQUALITY_CELL_STEPS_PER_COMPARISON = 2 * NODE_CELL_WIDTH + 2 + 1

BOUNDARY_RACE_CELL_STEPS = 3 * BOUNDARY_RACE_STEPS
RELATION_RACE_CELL_STEPS = RELATION_CELL_COUNT * RELATION_RACE_STEPS
SCHEMA_RACE_CELL_STEPS = SCHEMA_RACE_CELL_COUNT * SCHEMA_RACE_STEPS
ACTION_RECALL_RACE_CELL_STEPS = ACTION_CELL_COUNT * ACTION_RECALL_RACE_STEPS
ACTION_INTERVENTION_RACE_CELL_STEPS = (
    ACTION_CELL_COUNT * ACTION_INTERVENTION_RACE_STEPS
)
COMPLETION_RACE_CELL_STEPS = COMPLETION_RACE_STEPS

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"causal-anonymous-occurrence-bundle-schema-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_WORK_NAMESPACE = _NAMESPACE + b"work\0"
_CERTIFICATE_NAMESPACE = _NAMESPACE + b"certificate\0"
_BOUNDARY_RACE = CompiledRestingGlifRace.compile(step_count=BOUNDARY_RACE_STEPS)
_RELATION_RACE = CompiledRestingGlifRace.compile(step_count=RELATION_RACE_STEPS)
_SCHEMA_RACE = CompiledRestingGlifRace.compile(step_count=SCHEMA_RACE_STEPS)
_ACTION_RECALL_RACE = CompiledRestingGlifRace.compile(
    step_count=ACTION_RECALL_RACE_STEPS
)
_ACTION_INTERVENTION_RACE = CompiledRestingGlifRace.compile(
    step_count=ACTION_INTERVENTION_RACE_STEPS
)
_COMPLETION_RACE = CompiledRestingGlifRace.compile(
    step_count=COMPLETION_RACE_STEPS
)
_CONSTRUCTION_TOKEN = object()


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
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


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")
    return value


def _node_cells(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be one immutable physical assembly")
    cells = tuple(_integer(cell, label) for cell in value)
    if (
        len(cells) != NODE_ACTIVE_CELL_COUNT
        or cells != tuple(sorted(cells))
        or len(set(cells)) != NODE_ACTIVE_CELL_COUNT
        or any(not 0 <= cell < NODE_CELL_WIDTH for cell in cells)
    ):
        raise ValueError(f"{label} physical assembly changed")
    return cells


@dataclass(frozen=True, slots=True)
class PhysicalOccurrenceEvent:
    """One neutral physical edge plus an upstream OPEN/EXTEND/EOS cue."""

    cue_cell_id: int
    source_cells: tuple[int, ...]
    target_cells: tuple[int, ...]


class OccurrenceBundleFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class OccurrenceBundleWork(NamedTuple):
    input_event_count: int
    bundle_count: int
    boundary_race_count: int
    boundary_race_cell_steps: int
    replay_event_count: int
    replay_cell_steps: int
    relation_equality_comparison_count: int
    consequence_equality_comparison_count: int
    validation_equality_comparison_count: int
    equality_cell_steps: int
    relation_race_count: int
    relation_race_cell_steps: int
    schema_race_count: int
    schema_race_cell_steps: int
    action_recall_race_count: int
    action_recall_race_cell_steps: int
    action_intervention_race_count: int
    action_intervention_race_cell_steps: int
    completion_race_count: int
    completion_race_cell_steps: int
    total_physical_cell_steps: int
    trace_sha256s: tuple[str, ...]
    sha256: str


class OccurrenceBundleTrainingSummary(NamedTuple):
    bundle_count: int
    schema_assignments: tuple[int, ...]
    allocated_schema_count: int
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    action_policy_state_bytes: int
    persistent_state_bytes: int
    source_bundle_sha256s: tuple[str, ...]
    relation_sha256s: tuple[str, ...]
    consequence_sha256s: tuple[str, ...]
    work: OccurrenceBundleWork
    checkpoint_sha256: str


class _CertificateReceipt(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    checkpoint_sha256: str
    bundle_sha256: str
    schema_index: int | None
    schema_cell_id: int | None
    selected_actions: tuple[int, ...]
    relation_cell_ids: tuple[int, ...]
    root_cells: tuple[int, ...]
    source_to_carrier: tuple[tuple[tuple[int, ...], int], ...]
    ordered_edges: tuple[tuple[int, int], ...]
    occupancy: tuple[int, ...]
    root_arity: int | None
    max_depth: int | None
    provenance_sha256: str
    work: OccurrenceBundleWork
    sha256: str


_CERTIFICATE_AUTHORITY = object()
_ISSUED_CERTIFICATES: dict[int, OccurrenceGraphSchemaCertificate] = {}


class OccurrenceGraphSchemaCertificate:
    """Opaque exact-object result for one frozen bundle recognition."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _CertificateReceipt) -> None:
        if receipt.authority is not _CERTIFICATE_AUTHORITY:
            raise TypeError("occurrence graph certificate requires private authority")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.status

    @property
    def stop_reason(self) -> str | None:
        return self._receipt.stop_reason

    @property
    def checkpoint_sha256(self) -> str:
        return self._receipt.checkpoint_sha256

    @property
    def schema_index(self) -> int | None:
        return self._receipt.schema_index

    @property
    def schema_cell_id(self) -> int | None:
        return self._receipt.schema_cell_id

    @property
    def selected_actions(self) -> tuple[int, ...]:
        return self._receipt.selected_actions

    @property
    def relation_cell_ids(self) -> tuple[int, ...]:
        return self._receipt.relation_cell_ids

    @property
    def root_cells(self) -> tuple[int, ...]:
        return self._receipt.root_cells

    @property
    def source_to_carrier(self) -> tuple[tuple[tuple[int, ...], int], ...]:
        return self._receipt.source_to_carrier

    @property
    def ordered_edges(self) -> tuple[tuple[int, int], ...]:
        return self._receipt.ordered_edges

    @property
    def occupancy(self) -> tuple[int, ...]:
        return self._receipt.occupancy

    @property
    def root_arity(self) -> int | None:
        return self._receipt.root_arity

    @property
    def max_depth(self) -> int | None:
        return self._receipt.max_depth

    @property
    def provenance_sha256(self) -> str:
        return self._receipt.provenance_sha256

    @property
    def work(self) -> OccurrenceBundleWork:
        return self._receipt.work

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


class OccurrenceGraphSchemaBinding(NamedTuple):
    schema_index: int
    schema_cell_id: int
    selected_actions: tuple[int, ...]
    relation_cell_ids: tuple[int, ...]
    root_cells: tuple[int, ...]
    source_to_carrier: tuple[tuple[tuple[int, ...], int], ...]
    ordered_edges: tuple[tuple[int, int], ...]
    occupancy: tuple[int, ...]
    root_arity: int
    max_depth: int
    provenance_sha256: str
    certificate_sha256: str


class _CollectedBundle(NamedTuple):
    events: tuple[PhysicalOccurrenceEvent, ...]
    sha256: str


class _GraphView(NamedTuple):
    root_cells: tuple[int, ...]
    source_to_carrier: tuple[tuple[tuple[int, ...], int], ...]
    ordered_edges: tuple[tuple[int, int], ...]
    occupancy: tuple[int, ...]
    root_arity: int
    max_depth: int


class _RelationTrace(NamedTuple):
    status: str
    stop_reason: str | None
    relations: tuple[int, ...]
    relation_cell_ids: tuple[int, ...]
    sha256: str


@dataclass(slots=True)
class _WorkLedger:
    input_event_count: int = 0
    bundle_count: int = 0
    boundary_race_count: int = 0
    replay_event_count: int = 0
    relation_equality_comparison_count: int = 0
    consequence_equality_comparison_count: int = 0
    validation_equality_comparison_count: int = 0
    relation_race_count: int = 0
    schema_race_count: int = 0
    action_recall_race_count: int = 0
    action_intervention_race_count: int = 0
    completion_race_count: int = 0
    trace_sha256s: list[str] | None = None

    def __post_init__(self) -> None:
        if self.trace_sha256s is None:
            self.trace_sha256s = []


def _work_payload(value: OccurrenceBundleWork) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field != "sha256"
    }


def _freeze_work(ledger: _WorkLedger) -> OccurrenceBundleWork:
    equality_count = (
        ledger.relation_equality_comparison_count
        + ledger.consequence_equality_comparison_count
        + ledger.validation_equality_comparison_count
    )
    boundary_steps = ledger.boundary_race_count * BOUNDARY_RACE_CELL_STEPS
    replay_steps = ledger.replay_event_count * COMPLETION_RACE_CELL_STEPS
    equality_steps = equality_count * EQUALITY_CELL_STEPS_PER_COMPARISON
    relation_steps = ledger.relation_race_count * RELATION_RACE_CELL_STEPS
    schema_steps = ledger.schema_race_count * SCHEMA_RACE_CELL_STEPS
    recall_steps = ledger.action_recall_race_count * ACTION_RECALL_RACE_CELL_STEPS
    intervention_steps = (
        ledger.action_intervention_race_count
        * ACTION_INTERVENTION_RACE_CELL_STEPS
    )
    completion_steps = ledger.completion_race_count * COMPLETION_RACE_CELL_STEPS
    total = (
        boundary_steps
        + replay_steps
        + equality_steps
        + relation_steps
        + schema_steps
        + recall_steps
        + intervention_steps
        + completion_steps
    )
    partial = OccurrenceBundleWork(
        ledger.input_event_count,
        ledger.bundle_count,
        ledger.boundary_race_count,
        boundary_steps,
        ledger.replay_event_count,
        replay_steps,
        ledger.relation_equality_comparison_count,
        ledger.consequence_equality_comparison_count,
        ledger.validation_equality_comparison_count,
        equality_steps,
        ledger.relation_race_count,
        relation_steps,
        ledger.schema_race_count,
        schema_steps,
        ledger.action_recall_race_count,
        recall_steps,
        ledger.action_intervention_race_count,
        intervention_steps,
        ledger.completion_race_count,
        completion_steps,
        total,
        tuple(ledger.trace_sha256s or ()),
        "",
    )
    return partial._replace(sha256=_digest(_WORK_NAMESPACE, _work_payload(partial)))


def _normalize_event(value: object) -> PhysicalOccurrenceEvent:
    if type(value) is not PhysicalOccurrenceEvent:
        raise TypeError("stream entries must be exact PhysicalOccurrenceEvent values")
    cue = _integer(value.cue_cell_id, "cue cell")
    if cue == EOS_CUE_CELL:
        if value.source_cells or value.target_cells:
            raise ValueError("EOS cannot carry an edge")
        return value
    if cue not in (OPEN_CUE_CELL, EXTEND_CUE_CELL):
        raise ValueError("event cue is not the banked OPEN/EXTEND cell")
    return PhysicalOccurrenceEvent(
        cue,
        _node_cells(value.source_cells, "source cells"),
        _node_cells(value.target_cells, "target cells"),
    )


def _race_cue(
    event: PhysicalOccurrenceEvent,
    ledger: _WorkLedger,
    *,
    boundary_gate: bool,
) -> int:
    local = {
        OPEN_CUE_CELL: 0,
        EXTEND_CUE_CELL: 1,
        EOS_CUE_CELL: 2,
    }[event.cue_cell_id]
    currents = np.zeros(3, dtype=np.float64)
    if boundary_gate:
        currents[local] = OUTPUT_CURRENT_PA
    ledger.input_event_count += 1
    ledger.boundary_race_count += 1
    try:
        race = _BOUNDARY_RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure as exc:
        raise OccurrenceBundleFitStop(
            STOP_BOUNDARY_ZERO, "OPEN/EXTEND/EOS population emitted zero"
        ) from exc
    if race.simultaneous_spike_candidates != 1 or race.winner_cell != local:
        raise OccurrenceBundleFitStop(
            STOP_BOUNDARY_SEQUENCE, "boundary population did not preserve its cue"
        )
    root = _digest(
        _NAMESPACE + b"boundary-race\0",
        {
            "cue_cell_id": event.cue_cell_id,
            "winner": int(race.winner_cell),
            "spike_step": race.spike_step,
        },
    )
    assert ledger.trace_sha256s is not None
    ledger.trace_sha256s.append(root)
    return local


def _completion_tick(
    ledger: _WorkLedger,
    *,
    completion_gate: bool,
    replay: bool,
) -> None:
    currents = np.asarray(
        [OUTPUT_CURRENT_PA if completion_gate else 0.0], dtype=np.float64
    )
    if replay:
        ledger.replay_event_count += 1
    else:
        ledger.completion_race_count += 1
    try:
        race = _COMPLETION_RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure as exc:
        status = STOP_REPLAY if replay else STOP_INCOMPLETE
        raise OccurrenceBundleFitStop(
            status, "completion/replay cell emitted zero"
        ) from exc
    if race.simultaneous_spike_candidates != 1 or race.winner_cell != 0:
        raise OccurrenceBundleFitStop(
            STOP_REPLAY if replay else STOP_INCOMPLETE,
            "completion/replay cell became ambiguous",
        )
    root = _digest(
        _NAMESPACE + b"completion-race\0",
        {"replay": replay, "spike_step": race.spike_step},
    )
    assert ledger.trace_sha256s is not None
    ledger.trace_sha256s.append(root)


def _bundle(events: Sequence[PhysicalOccurrenceEvent]) -> _CollectedBundle:
    frozen = tuple(events)
    if not 3 <= len(frozen) <= MAX_BUNDLE_EVENT_COUNT:
        raise OccurrenceBundleFitStop(
            STOP_INCOMPLETE,
            "a reusable relay needs 3..8 physical edge events",
        )
    return _CollectedBundle(
        frozen,
        _digest(
            _NAMESPACE + b"bundle\0",
            tuple(
                (event.cue_cell_id, event.source_cells, event.target_cells)
                for event in frozen
            ),
        ),
    )


def _collect_bundles(
    values: Sequence[PhysicalOccurrenceEvent],
    ledger: _WorkLedger,
    *,
    boundary_gate: bool,
    completion_gate: bool,
    replay_gate: bool,
) -> tuple[_CollectedBundle, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("physical event stream must be a sequence")
    active: list[PhysicalOccurrenceEvent] = []
    completed: list[_CollectedBundle] = []
    for raw in values:
        event = _normalize_event(raw)
        cue = _race_cue(event, ledger, boundary_gate=boundary_gate)
        if cue == 2:
            if not active:
                raise OccurrenceBundleFitStop(
                    STOP_BOUNDARY_SEQUENCE, "EOS arrived without an active bundle"
                )
            _completion_tick(ledger, completion_gate=completion_gate, replay=False)
            completed.append(_bundle(active))
            active = []
            continue
        if cue == 0:
            if active:
                _completion_tick(
                    ledger, completion_gate=completion_gate, replay=False
                )
                completed.append(_bundle(active))
                if not replay_gate:
                    raise OccurrenceBundleFitStop(
                        STOP_REPLAY,
                        "OPEN completed one bundle but its triggering event was not replayed",
                    )
                _completion_tick(
                    ledger, completion_gate=completion_gate, replay=True
                )
            active = [event]
            continue
        if not active:
            raise OccurrenceBundleFitStop(
                STOP_BOUNDARY_SEQUENCE, "EXTEND arrived without OPEN"
            )
        active.append(event)
    if active:
        raise OccurrenceBundleFitStop(
            STOP_INCOMPLETE, "stream ended without OPEN-triggered completion or EOS"
        )
    if not completed:
        raise OccurrenceBundleFitStop(STOP_INCOMPLETE, "stream contained no bundle")
    ledger.bundle_count = len(completed)
    return tuple(completed)


def _physical_equal(
    left: tuple[int, ...],
    right: tuple[int, ...],
    ledger: _WorkLedger,
    *,
    category: str,
    equality_gate: bool,
) -> bool:
    if category == "relation":
        ledger.relation_equality_comparison_count += 1
    elif category == "consequence":
        ledger.consequence_equality_comparison_count += 1
    elif category == "validation":
        ledger.validation_equality_comparison_count += 1
    else:
        raise RuntimeError("unknown physical equality category")
    if not equality_gate:
        return False
    phase_spikes = np.ones(2, dtype=np.bool_)
    membrane = np.zeros(NODE_CELL_WIDTH, dtype=np.uint8)
    membrane[np.asarray(left, dtype=np.intp)] += 1
    membrane[np.asarray(right, dtype=np.intp)] += 1
    coincidences = np.greater_equal(membrane, 2)
    return bool(np.all(phase_spikes)) and (
        int(np.count_nonzero(coincidences)) >= NODE_ACTIVE_CELL_COUNT
    )


def _race_relation(
    relation: int,
    ledger: _WorkLedger,
    *,
    relation_cell_gate: bool,
) -> None:
    currents = np.zeros(RELATION_CELL_COUNT, dtype=np.float64)
    if relation_cell_gate:
        currents[relation] = OUTPUT_CURRENT_PA
    ledger.relation_race_count += 1
    try:
        race = _RELATION_RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure as exc:
        raise OccurrenceBundleFitStop(
            STOP_RELATION_ZERO, "factorized relation population emitted zero"
        ) from exc
    if race.simultaneous_spike_candidates != 1 or race.winner_cell != relation:
        raise OccurrenceBundleFitStop(
            STOP_RELATION_ZERO, "factorized relation population changed its winner"
        )
    root = _digest(
        _NAMESPACE + b"relation-race\0",
        {"relation": relation, "spike_step": race.spike_step},
    )
    assert ledger.trace_sha256s is not None
    ledger.trace_sha256s.append(root)


def _relations(
    bundle: _CollectedBundle,
    ledger: _WorkLedger,
    *,
    equality_gate: bool,
    relation_cell_gate: bool,
) -> _RelationTrace:
    root = bundle.events[0].source_cells
    previous_target = bundle.events[0].target_cells
    seen: set[tuple[int, ...]] = set()
    relations: list[int] = []
    unsupported = False
    for ordinal, event in enumerate(bundle.events):
        target_equal_source = _physical_equal(
            event.source_cells,
            event.target_cells,
            ledger,
            category="relation",
            equality_gate=equality_gate,
        )
        source_equal_root = _physical_equal(
            event.source_cells,
            root if ordinal else event.target_cells,
            ledger,
            category="relation",
            equality_gate=equality_gate,
        )
        source_equal_previous = _physical_equal(
            event.source_cells,
            previous_target if ordinal else event.target_cells,
            ledger,
            category="relation",
            equality_gate=equality_gate,
        )
        if ordinal == 0:
            relation = (
                ROOT_SELF_RELATION
                if event.cue_cell_id == OPEN_CUE_CELL and target_equal_source
                else CONFLICT_RELATION
            )
            seen.add(event.source_cells)
        elif event.cue_cell_id != EXTEND_CUE_CELL:
            relation = CONFLICT_RELATION
        elif target_equal_source:
            relation = SELF_EDGE_RELATION
        elif event.target_cells in seen:
            relation = REUSED_TARGET_RELATION
        elif ordinal == 1 and source_equal_root and source_equal_previous:
            relation = FIRST_ATTACHMENT_RELATION
        elif source_equal_previous and not source_equal_root:
            relation = DESCENDANT_RELATION
        elif source_equal_root and not source_equal_previous:
            relation = SIBLING_RELATION
        elif event.source_cells in seen:
            relation = NONLOCAL_SOURCE_RELATION
        else:
            relation = UNSEEN_SOURCE_RELATION
        _race_relation(relation, ledger, relation_cell_gate=relation_cell_gate)
        relations.append(relation)
        if relation >= NONLOCAL_SOURCE_RELATION:
            unsupported = True
        seen.add(event.source_cells)
        seen.add(event.target_cells)
        previous_target = event.target_cells
    _race_relation(COMPLETE_RELATION, ledger, relation_cell_gate=relation_cell_gate)
    relations.append(COMPLETE_RELATION)
    status = STOP_RELATION_CONFLICT if unsupported else READY
    stop_reason = (
        "physical edge relations left the rooted chain/fanout support"
        if unsupported
        else None
    )
    payload = {
        "bundle_sha256": bundle.sha256,
        "relations": tuple(relations),
        "relation_cell_ids": tuple(
            RELATION_CELL_OFFSET + relation for relation in relations
        ),
        "status": status,
    }
    return _RelationTrace(
        status,
        stop_reason,
        tuple(relations),
        tuple(RELATION_CELL_OFFSET + relation for relation in relations),
        _digest(_NAMESPACE + b"relations\0", payload),
    )


def _graph_view(bundle: _CollectedBundle) -> _GraphView:
    carrier_by_source: dict[tuple[int, ...], int] = {}
    ordered_sources: list[tuple[int, ...]] = []

    def carrier(cells: tuple[int, ...]) -> int:
        existing = carrier_by_source.get(cells)
        if existing is not None:
            return existing
        index = len(ordered_sources)
        carrier_by_source[cells] = index
        ordered_sources.append(cells)
        return index

    root_cells = bundle.events[0].source_cells
    root = carrier(root_cells)
    if root != 0:
        raise RuntimeError("root carrier allocation changed")
    edges: list[tuple[int, int]] = []
    depth = {0: 0}
    for event in bundle.events[1:]:
        source = carrier(event.source_cells)
        target = carrier(event.target_cells)
        edges.append((source, target))
        depth[target] = depth.get(source, 0) + 1
    occupancy = [0] * len(ordered_sources)
    for source, _target in edges:
        occupancy[source] += 1
    return _GraphView(
        root_cells,
        tuple((cells, index) for index, cells in enumerate(ordered_sources)),
        tuple(edges),
        tuple(occupancy),
        occupancy[0],
        max(depth.values()),
    )


def _consequence_cells(value: int) -> tuple[int, ...]:
    if not 0 <= value <= MAX_BUNDLE_EVENT_COUNT:
        raise ValueError("graph consequence left the bounded byte projection")
    start = NODE_ACTIVE_CELL_COUNT * value
    return tuple(range(start, start + NODE_ACTIVE_CELL_COUNT))


def _race_intervention(action: int, ledger: _WorkLedger) -> None:
    currents = np.zeros(ACTION_CELL_COUNT, dtype=np.float64)
    currents[action] = OUTPUT_CURRENT_PA
    ledger.action_intervention_race_count += 1
    race = _ACTION_INTERVENTION_RACE.race_many(
        currents[None, :], require_unique=False
    )[0]
    if race.simultaneous_spike_candidates != 1 or race.winner_cell != action:
        raise OccurrenceBundleFitStop(
            STOP_CONSEQUENCE, "generic graph action failed its physical race"
        )
    root = _digest(
        _NAMESPACE + b"action-intervention\0",
        {"action": action, "spike_step": race.spike_step},
    )
    assert ledger.trace_sha256s is not None
    ledger.trace_sha256s.append(root)


def _causal_actions(
    bundle: _CollectedBundle,
    graph: _GraphView,
    ledger: _WorkLedger,
) -> tuple[tuple[int, ...], str]:
    attachment_count = len(bundle.events) - 1
    observed = (graph.root_arity, graph.max_depth)
    candidates = {
        ATTACH_DESCENDANT_ACTION: (1, attachment_count),
        ATTACH_SIBLING_ACTION: (attachment_count, 1),
    }
    passing: list[int] = []
    candidate_payload: list[tuple[int, tuple[int, int], bool]] = []
    for action, consequence in candidates.items():
        matches = tuple(
            _physical_equal(
                _consequence_cells(candidate),
                _consequence_cells(actual),
                ledger,
                category="consequence",
                equality_gate=True,
            )
            for candidate, actual in zip(consequence, observed, strict=True)
        )
        passed = all(matches)
        candidate_payload.append((action, consequence, passed))
        if passed:
            passing.append(action)
    if len(passing) != 1:
        raise OccurrenceBundleFitStop(
            STOP_CONSEQUENCE,
            "physical root-arity/depth consequence did not identify one relay",
        )
    relay = passing[0]
    actions = (
        OPEN_ROOT_ACTION,
        *(relay for _ in range(attachment_count)),
        COMPLETE_ACTION,
    )
    for action in actions:
        _race_intervention(action, ledger)
    root = _digest(
        _NAMESPACE + b"causal-actions\0",
        {
            "bundle_sha256": bundle.sha256,
            "observed_consequence": observed,
            "candidate_consequences": tuple(candidate_payload),
            "selected_actions": actions,
        },
    )
    return actions, root


def _race_action(
    scores: np.ndarray,
    ledger: _WorkLedger,
    *,
    action_cell_gate: bool,
) -> tuple[str, int | None, str]:
    currents = np.asarray(
        [
            max(0, int(score)) * OUTPUT_CURRENT_PA if action_cell_gate else 0.0
            for score in scores
        ],
        dtype=np.float64,
    )
    ledger.action_recall_race_count += 1
    try:
        race = _ACTION_RECALL_RACE.race_many(
            currents[None, :], require_unique=False
        )[0]
    except RestingGlifRaceFailure:
        status = STOP_ACTION_ZERO
        action = None
        spike_step = None
        simultaneous = 0
    else:
        spike_step = race.spike_step
        simultaneous = race.simultaneous_spike_candidates
        if simultaneous != 1:
            status = STOP_ACTION_AMBIGUOUS
            action = None
        else:
            status = READY
            action = int(race.winner_cell)
    root = _digest(
        _NAMESPACE + b"action-recall\0",
        {
            "scores": tuple(int(value) for value in scores),
            "gate": action_cell_gate,
            "status": status,
            "action": action,
            "spike_step": spike_step,
            "simultaneous": simultaneous,
        },
    )
    assert ledger.trace_sha256s is not None
    ledger.trace_sha256s.append(root)
    return status, action, root


def _recall_schema(
    schema_index: int,
    relations: tuple[int, ...],
    bundle: _CollectedBundle,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    ledger: _WorkLedger,
    *,
    action_cell_gate: bool,
) -> tuple[bool, tuple[int, ...], str]:
    actions: list[int] = []
    action_roots: list[str] = []
    for relation in relations:
        scores = excitatory[schema_index, relation].astype(
            np.int16
        ) - inhibitory[schema_index, relation].astype(np.int16)
        status, action, root = _race_action(
            scores, ledger, action_cell_gate=action_cell_gate
        )
        action_roots.append(root)
        if status != READY or action is None:
            return False, tuple(actions), _digest(
                _NAMESPACE + b"schema-recall\0",
                {
                    "schema_index": schema_index,
                    "status": status,
                    "action_roots": tuple(action_roots),
                },
            )
        actions.append(action)
    valid = (
        actions[0] == OPEN_ROOT_ACTION
        and actions[-1] == COMPLETE_ACTION
        and len(actions) == len(bundle.events) + 1
    )
    if valid:
        root = bundle.events[0].source_cells
        previous_target = bundle.events[0].target_cells
        for event, action in zip(bundle.events[1:], actions[1:-1], strict=True):
            if action == ATTACH_DESCENDANT_ACTION:
                predicted_source = previous_target
            elif action == ATTACH_SIBLING_ACTION:
                predicted_source = root
            else:
                valid = False
                break
            if not _physical_equal(
                predicted_source,
                event.source_cells,
                ledger,
                category="validation",
                equality_gate=True,
            ):
                valid = False
                break
            previous_target = event.target_cells
    recall_root = _digest(
        _NAMESPACE + b"schema-recall\0",
        {
            "schema_index": schema_index,
            "valid": valid,
            "actions": tuple(actions),
            "action_roots": tuple(action_roots),
            "bundle_sha256": bundle.sha256,
        },
    )
    return valid, tuple(actions), recall_root


def _race_schema(
    matches: Sequence[int],
    valid: np.ndarray,
    ledger: _WorkLedger,
    *,
    allow_novel: bool,
    schema_cell_gate: bool,
) -> tuple[str, int | None, bool, str]:
    currents = np.zeros(SCHEMA_RACE_CELL_COUNT, dtype=np.float64)
    if schema_cell_gate:
        for schema_index in matches:
            currents[schema_index] = WINNER_CURRENT_PA
        free = tuple(int(index) for index in np.flatnonzero(valid == 0))
        if not matches and allow_novel and free:
            currents[NOVEL_SCHEMA] = WINNER_CURRENT_PA
        currents[TONIC_SCHEMA] = TONIC_CURRENT_PA
    ledger.schema_race_count += 1
    try:
        race = _SCHEMA_RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure:
        status = STOP_SCHEMA_ZERO
        schema_index = None
        novel = False
        spike_step = None
        simultaneous = 0
    else:
        spike_step = race.spike_step
        simultaneous = race.simultaneous_spike_candidates
        winner = int(race.winner_cell)
        if simultaneous != 1:
            status = STOP_SCHEMA_AMBIGUOUS
            schema_index = None
            novel = False
        elif winner == TONIC_SCHEMA:
            status = STOP_TONIC
            schema_index = None
            novel = False
        elif winner == NOVEL_SCHEMA:
            free = tuple(int(index) for index in np.flatnonzero(valid == 0))
            if not allow_novel or not free:
                status = STOP_CAPACITY
                schema_index = None
                novel = False
            else:
                status = READY
                schema_index = free[0]
                novel = True
        elif winner < SCHEMA_CAPACITY:
            status = READY
            schema_index = winner
            novel = False
        else:
            raise RuntimeError("schema race left its four-cell capacity")
    root = _digest(
        _NAMESPACE + b"schema-race\0",
        {
            "matches": tuple(matches),
            "valid": tuple(int(value) for value in valid),
            "allow_novel": allow_novel,
            "gate": schema_cell_gate,
            "status": status,
            "schema_index": schema_index,
            "novel": novel,
            "spike_step": spike_step,
            "simultaneous": simultaneous,
        },
    )
    assert ledger.trace_sha256s is not None
    ledger.trace_sha256s.append(root)
    return status, schema_index, novel, root


def _update_policy(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    schema_index: int,
    relations: tuple[int, ...],
    actions: tuple[int, ...],
) -> tuple[int, int]:
    if len(relations) != len(actions):
        raise RuntimeError("relation/action evidence changed length")
    excitatory_ticks = 0
    inhibitory_ticks = 0
    for relation, passed in zip(relations, actions, strict=True):
        for action in range(ACTION_CELL_COUNT):
            target = excitatory if action == passed else inhibitory
            prior = int(target[schema_index, relation, action])
            target[schema_index, relation, action] = min(TICK_CAP, prior + 1)
            if action == passed:
                excitatory_ticks += 1
            else:
                inhibitory_ticks += 1
    return excitatory_ticks, inhibitory_ticks


def _checkpoint_sha256(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    valid: np.ndarray,
    provenance: np.ndarray,
    replay_latch: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + segmented.CONFIG_SHA256.encode("ascii")
        + b"\0"
        + _array_frame("action-excitatory", excitatory)
        + _array_frame("action-inhibitory", inhibitory)
        + _array_frame("schema-valid", valid)
        + _array_frame("provenance", provenance)
        + _array_frame("replay-latch", replay_latch)
    ).hexdigest()


class FrozenAnonymousOccurrenceBundleSchemaV1:
    """Immutable two-schema checkpoint with one physical-event interface."""

    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_provenance",
        "_replay_latch",
        "_summary",
        "_valid",
    )

    def __init__(
        self,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        valid: np.ndarray,
        provenance: np.ndarray,
        replay_latch: np.ndarray,
        summary: OccurrenceBundleTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("anonymous occurrence checkpoints are builder-owned")
        if (
            excitatory.shape != ACTION_POLICY_SHAPE
            or inhibitory.shape != ACTION_POLICY_SHAPE
            or valid.shape != (SCHEMA_CAPACITY,)
            or provenance.shape != (PROVENANCE_STATE_BYTES,)
            or replay_latch.shape != (REPLAY_LATCH_STATE_BYTES,)
            or any(
                value.dtype != TICK_DTYPE
                for value in (
                    excitatory,
                    inhibitory,
                    valid,
                    provenance,
                    replay_latch,
                )
            )
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
            or np.any(valid > 1)
        ):
            raise ValueError("anonymous occurrence state geometry changed")
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._valid = np.array(valid, copy=True, order="C")
        self._provenance = np.array(provenance, copy=True, order="C")
        self._replay_latch = np.array(replay_latch, copy=True, order="C")
        for value in (
            self._excitatory,
            self._inhibitory,
            self._valid,
            self._provenance,
            self._replay_latch,
        ):
            value.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha256(
            self._excitatory,
            self._inhibitory,
            self._valid,
            self._provenance,
            self._replay_latch,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("anonymous occurrence training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> OccurrenceBundleTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            self._excitatory.shape,
            self._inhibitory.shape,
            self._valid.shape,
            self._provenance.shape,
            self._replay_latch.shape,
        )

    def authenticate(self) -> None:
        arrays = (
            self._excitatory,
            self._inhibitory,
            self._valid,
            self._provenance,
            self._replay_latch,
        )
        if any(value.flags.writeable for value in arrays) or (
            self._checkpoint_sha256
            != _checkpoint_sha256(
                self._excitatory,
                self._inhibitory,
                self._valid,
                self._provenance,
                self._replay_latch,
            )
        ):
            raise RuntimeError("anonymous occurrence checkpoint authentication failed")

    def recognize(
        self,
        events: Sequence[PhysicalOccurrenceEvent],
        *,
        boundary_gate: bool = True,
        completion_gate: bool = True,
        replay_gate: bool = True,
        equality_gate: bool = True,
        relation_cell_gate: bool = True,
        schema_cell_gate: bool = True,
        action_cell_gate: bool = True,
    ) -> OccurrenceGraphSchemaCertificate:
        """Recognize exactly one complete bundle without a desired output."""

        self.authenticate()
        ledger = _WorkLedger()
        try:
            bundles = _collect_bundles(
                events,
                ledger,
                boundary_gate=boundary_gate,
                completion_gate=completion_gate,
                replay_gate=replay_gate,
            )
            if len(bundles) != 1:
                raise OccurrenceBundleFitStop(
                    STOP_INCOMPLETE, "recognize requires exactly one complete bundle"
                )
        except OccurrenceBundleFitStop as exc:
            return _issue_certificate(
                self,
                status=exc.status,
                stop_reason=exc.reason,
                bundle=None,
                relation_trace=None,
                graph=None,
                schema_index=None,
                selected_actions=(),
                ledger=ledger,
            )
        bundle = bundles[0]
        try:
            relation_trace = _relations(
                bundle,
                ledger,
                equality_gate=equality_gate,
                relation_cell_gate=relation_cell_gate,
            )
        except OccurrenceBundleFitStop as exc:
            return _issue_certificate(
                self,
                status=exc.status,
                stop_reason=exc.reason,
                bundle=bundle,
                relation_trace=None,
                graph=None,
                schema_index=None,
                selected_actions=(),
                ledger=ledger,
            )
        graph = _graph_view(bundle)
        if relation_trace.status != READY:
            return _issue_certificate(
                self,
                status=relation_trace.status,
                stop_reason=relation_trace.stop_reason,
                bundle=bundle,
                relation_trace=relation_trace,
                graph=graph,
                schema_index=None,
                selected_actions=(),
                ledger=ledger,
            )
        matches: list[int] = []
        actions_by_schema: dict[int, tuple[int, ...]] = {}
        for schema_index in np.flatnonzero(self._valid):
            index = int(schema_index)
            matched, actions, _root = _recall_schema(
                index,
                relation_trace.relations,
                bundle,
                self._excitatory,
                self._inhibitory,
                ledger,
                action_cell_gate=action_cell_gate,
            )
            if matched:
                matches.append(index)
                actions_by_schema[index] = actions
        if not action_cell_gate:
            return _issue_certificate(
                self,
                status=STOP_ACTION_ZERO,
                stop_reason="generic graph action population was disabled",
                bundle=bundle,
                relation_trace=relation_trace,
                graph=graph,
                schema_index=None,
                selected_actions=(),
                ledger=ledger,
            )
        status, schema_index, _novel, _race_root = _race_schema(
            matches,
            self._valid,
            ledger,
            allow_novel=False,
            schema_cell_gate=schema_cell_gate,
        )
        actions = () if schema_index is None else actions_by_schema[schema_index]
        return _issue_certificate(
            self,
            status=status,
            stop_reason=None if status == READY else "tonic/schema race abstained",
            bundle=bundle,
            relation_trace=relation_trace,
            graph=graph,
            schema_index=schema_index,
            selected_actions=actions,
            ledger=ledger,
        )


def _certificate_payload(receipt: _CertificateReceipt) -> dict[str, object]:
    return {
        field: getattr(receipt, field)
        for field in receipt._fields
        if field not in {"authority", "work", "sha256"}
    } | {"work_sha256": receipt.work.sha256}


def _issue_certificate(
    checkpoint: FrozenAnonymousOccurrenceBundleSchemaV1,
    *,
    status: str,
    stop_reason: str | None,
    bundle: _CollectedBundle | None,
    relation_trace: _RelationTrace | None,
    graph: _GraphView | None,
    schema_index: int | None,
    selected_actions: tuple[int, ...],
    ledger: _WorkLedger,
) -> OccurrenceGraphSchemaCertificate:
    work = _freeze_work(ledger)
    provenance_root = _digest(
        _NAMESPACE + b"certificate-provenance\0",
        {
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "bundle_sha256": None if bundle is None else bundle.sha256,
            "training_provenance": tuple(int(value) for value in checkpoint._provenance),
        },
    )
    partial = _CertificateReceipt(
        _CERTIFICATE_AUTHORITY,
        status,
        stop_reason,
        checkpoint.checkpoint_sha256,
        "" if bundle is None else bundle.sha256,
        schema_index,
        None if schema_index is None else SCHEMA_CELL_OFFSET + schema_index,
        selected_actions,
        () if relation_trace is None else relation_trace.relation_cell_ids,
        () if graph is None else graph.root_cells,
        () if graph is None else graph.source_to_carrier,
        () if graph is None else graph.ordered_edges,
        () if graph is None else graph.occupancy,
        None if graph is None else graph.root_arity,
        None if graph is None else graph.max_depth,
        provenance_root,
        work,
        "",
    )
    receipt = partial._replace(
        sha256=_digest(_CERTIFICATE_NAMESPACE, _certificate_payload(partial))
    )
    certificate = OccurrenceGraphSchemaCertificate(receipt)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _authenticate_certificate(
    value: OccurrenceGraphSchemaCertificate,
) -> _CertificateReceipt:
    if (
        type(value) is not OccurrenceGraphSchemaCertificate
        or _ISSUED_CERTIFICATES.get(id(value)) is not value
    ):
        raise TypeError("exact issued occurrence graph certificate required")
    receipt = value._receipt
    if (
        receipt.authority is not _CERTIFICATE_AUTHORITY
        or receipt.work.sha256 != _digest(_WORK_NAMESPACE, _work_payload(receipt.work))
        or receipt.sha256
        != _digest(_CERTIFICATE_NAMESPACE, _certificate_payload(receipt))
    ):
        raise RuntimeError("occurrence graph certificate authentication failed")
    return receipt


def require_occurrence_graph_schema(
    value: OccurrenceGraphSchemaCertificate,
) -> OccurrenceGraphSchemaBinding:
    receipt = _authenticate_certificate(value)
    if (
        receipt.status != READY
        or receipt.schema_index is None
        or receipt.schema_cell_id is None
        or receipt.root_arity is None
        or receipt.max_depth is None
    ):
        raise ValueError(receipt.stop_reason or receipt.status)
    return OccurrenceGraphSchemaBinding(
        receipt.schema_index,
        receipt.schema_cell_id,
        receipt.selected_actions,
        receipt.relation_cell_ids,
        receipt.root_cells,
        receipt.source_to_carrier,
        receipt.ordered_edges,
        receipt.occupancy,
        receipt.root_arity,
        receipt.max_depth,
        receipt.provenance_sha256,
        receipt.sha256,
    )


def fit_causal_anonymous_occurrence_bundle_schema_v1(
    events: Sequence[PhysicalOccurrenceEvent],
    *,
    boundary_gate: bool = True,
    completion_gate: bool = True,
    replay_gate: bool = True,
    equality_gate: bool = True,
    relation_cell_gate: bool = True,
    schema_cell_gate: bool = True,
    action_cell_gate: bool = True,
) -> FrozenAnonymousOccurrenceBundleSchemaV1:
    """Fit two anonymous relay schemas from one A/B/A-prime physical stream."""

    ledger = _WorkLedger()
    bundles = _collect_bundles(
        events,
        ledger,
        boundary_gate=boundary_gate,
        completion_gate=completion_gate,
        replay_gate=replay_gate,
    )
    if len(bundles) != TRAIN_BUNDLE_COUNT:
        raise OccurrenceBundleFitStop(
            STOP_INCOMPLETE, "focused fit requires exactly three complete bundles"
        )
    excitatory = np.zeros(ACTION_POLICY_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    valid = np.zeros(SCHEMA_CAPACITY, dtype=TICK_DTYPE)
    provenance = np.zeros(PROVENANCE_STATE_BYTES, dtype=TICK_DTYPE)
    assignments: list[int] = []
    relation_roots: list[str] = []
    consequence_roots: list[str] = []
    excitatory_ticks = 0
    inhibitory_ticks = 0
    for bundle in bundles:
        relation_trace = _relations(
            bundle,
            ledger,
            equality_gate=equality_gate,
            relation_cell_gate=relation_cell_gate,
        )
        if relation_trace.status != READY:
            raise OccurrenceBundleFitStop(
                relation_trace.status,
                relation_trace.stop_reason or relation_trace.status,
            )
        graph = _graph_view(bundle)
        causal_actions, consequence_root = _causal_actions(bundle, graph, ledger)
        matches: list[int] = []
        for schema_index in np.flatnonzero(valid):
            matched, recalled, _root = _recall_schema(
                int(schema_index),
                relation_trace.relations,
                bundle,
                excitatory,
                inhibitory,
                ledger,
                action_cell_gate=action_cell_gate,
            )
            if matched and recalled == causal_actions:
                matches.append(int(schema_index))
        status, schema_index, novel, _schema_root = _race_schema(
            matches,
            valid,
            ledger,
            allow_novel=True,
            schema_cell_gate=schema_cell_gate,
        )
        if status != READY or schema_index is None:
            raise OccurrenceBundleFitStop(status, "schema race did not assign a relay")
        if novel:
            valid[schema_index] = 1
        added_e, added_h = _update_policy(
            excitatory,
            inhibitory,
            schema_index,
            relation_trace.relations,
            causal_actions,
        )
        excitatory_ticks += added_e
        inhibitory_ticks += added_h
        assignments.append(schema_index)
        relation_roots.append(relation_trace.sha256)
        consequence_roots.append(consequence_root)
        provenance_bytes = sha256(
            provenance.tobytes()
            + bundle.sha256.encode("ascii")
            + bytes((schema_index,))
        ).digest()[:PROVENANCE_STATE_BYTES]
        provenance[:] = np.frombuffer(provenance_bytes, dtype=np.uint8)
    replay_latch = np.asarray(
        [
            0,
            0,
            len(bundles),
            ledger.replay_event_count,
            ledger.input_event_count,
            int(np.count_nonzero(valid)),
            0,
            0,
        ],
        dtype=TICK_DTYPE,
    )
    checkpoint_sha = _checkpoint_sha256(
        excitatory, inhibitory, valid, provenance, replay_latch
    )
    work = _freeze_work(ledger)
    summary = OccurrenceBundleTrainingSummary(
        len(bundles),
        tuple(assignments),
        int(np.count_nonzero(valid)),
        excitatory_ticks,
        inhibitory_ticks,
        int(np.max(excitatory)),
        int(np.max(inhibitory)),
        ACTION_POLICY_STATE_BYTES,
        PERSISTENT_STATE_BYTES,
        tuple(bundle.sha256 for bundle in bundles),
        tuple(relation_roots),
        tuple(consequence_roots),
        work,
        checkpoint_sha,
    )
    return FrozenAnonymousOccurrenceBundleSchemaV1(
        excitatory,
        inhibitory,
        valid,
        provenance,
        replay_latch,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "segmented_aggregate_config_sha256": segmented.CONFIG_SHA256,
        "open_cue_cell": OPEN_CUE_CELL,
        "extend_cue_cell": EXTEND_CUE_CELL,
        "eos_cue_cell": EOS_CUE_CELL,
        "new_cells": NEW_CELL_COUNT,
        "total_cells": TOTAL_CELL_COUNT,
        "relation_cells": RELATION_CELL_COUNT,
        "schema_race_cells": SCHEMA_RACE_CELL_COUNT,
        "action_cells": ACTION_CELL_COUNT,
        "completion_replay_cells": 1,
        "action_policy_state_bytes": ACTION_POLICY_STATE_BYTES,
        "persistent_state_bytes": PERSISTENT_STATE_BYTES,
        "train_bundle_count": TRAIN_BUNDLE_COUNT,
        "max_bundle_event_count": MAX_BUNDLE_EVENT_COUNT,
        "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
        "canonical_panel_spend_count": CANONICAL_PANEL_SPEND_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "path2_input_count": PATH2_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", dict(CONFIG_RECORD))

__all__ = [
    "ACTION_CELL_COUNT",
    "ACTION_CELL_OFFSET",
    "ACTION_POLICY_STATE_BYTES",
    "ATTACH_DESCENDANT_ACTION",
    "ATTACH_SIBLING_ACTION",
    "C1_INPUT_COUNT",
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "COMPLETE_ACTION",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESCENDANT_RELATION",
    "DEV_INPUT_COUNT",
    "EOS_CUE_CELL",
    "EXTEND_CUE_CELL",
    "FIRST_ATTACHMENT_RELATION",
    "G0_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "OPEN_CUE_CELL",
    "OPEN_ROOT_ACTION",
    "PATH2_INPUT_COUNT",
    "PERSISTENT_STATE_BYTES",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "READY",
    "RELATION_CELL_COUNT",
    "RELATION_CELL_OFFSET",
    "SCHEMA_CAPACITY",
    "SCHEMA_CELL_OFFSET",
    "SIBLING_RELATION",
    "STOP_ACTION_ZERO",
    "STOP_RELATION_ZERO",
    "STOP_REPLAY",
    "STOP_SCHEMA_ZERO",
    "STOP_TONIC",
    "TOTAL_CELL_COUNT",
    "FrozenAnonymousOccurrenceBundleSchemaV1",
    "OccurrenceBundleFitStop",
    "OccurrenceBundleTrainingSummary",
    "OccurrenceBundleWork",
    "OccurrenceGraphSchemaBinding",
    "OccurrenceGraphSchemaCertificate",
    "PhysicalOccurrenceEvent",
    "fit_causal_anonymous_occurrence_bundle_schema_v1",
    "require_occurrence_graph_schema",
]
