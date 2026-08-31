"""Causal predictive chunking for continuous physical byte assemblies.

The input surface is deliberately smaller than a token or event frame: a stream
contains only BEGIN, sequential twelve-cell value assemblies, END, and one
independent physical provenance assembly.  Boundaries are recovered by a race
between two interventions.  CONTINUE must explain the trigger from the prior
predictor state; CUT_REPLAY must reset, replay that same trigger, and explain the
shared one-event future.  Neither byte values nor caller supplied spans enter
that race.

This is a bounded V1.  It learns deterministic recurrent transitions and must
STOP at an unknown or ambiguous causal feature.  It is not a general language
segmenter.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from experiments.g0 import common_event_cells_v1 as physical_cells
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from experiments.r3_pilot import common_factorized_context_routed_memory_v2 as router

TRACE_LANES = 12
INPUT_LANE_BANK_CELLS = 65_536
INPUT_CELL_OFFSET = 7_000_000
PROVENANCE_CELL_OFFSET = 8_000_000
BOUNDARY_CELL_OFFSET = 9_000_000

MAX_STREAM_EVENTS = 96
MAX_PREDICTOR_EVIDENCE = 2_048
MAX_PREDICTOR_ASSOCIATIONS = 256
MAX_ADMISSION_EVIDENCE = 512
MAX_ADMITTED_CHUNKS = router.MAX_FACTORS
TICK_CAP = 4
PREDICTOR_TONIC = 1
BOUNDARY_TONIC = 1
ADMISSION_TONIC = 1
HYPOTHESIS_MATCH_CURRENT = 2
HYPOTHESIS_TONIC_CURRENT = 1
HYPOTHESIS_RACE_STEPS = 2

CONTINUE = 0
CUT_REPLAY = 1
ACTION_COUNT = 2
FEATURE_COUNT = 4

RECALLED = "RECALLED_CAUSAL_PREDICTIVE_CHUNKS"
STOP_PREDICTOR_GATE = "STOP_CAUSAL_CHUNK_PREDICTOR_GATE"
STOP_PLASTICITY_GATE = "STOP_CAUSAL_CHUNK_PLASTICITY_GATE"
STOP_PREDICTOR_UNKNOWN = "STOP_CAUSAL_CHUNK_PREDICTOR_UNKNOWN"
STOP_PREDICTOR_AMBIGUOUS = "STOP_CAUSAL_CHUNK_PREDICTOR_AMBIGUOUS"
STOP_COMPARATOR_GATE = "STOP_CAUSAL_CHUNK_COMPARATOR_GATE"
STOP_HYPOTHESIS_ACTION_GATE = "STOP_CAUSAL_CHUNK_HYPOTHESIS_ACTION_GATE"
STOP_BOUNDARY_UNIDENTIFIABLE = "STOP_CAUSAL_CHUNK_BOUNDARY_UNIDENTIFIABLE"
STOP_BOUNDARY_AMBIGUOUS = "STOP_CAUSAL_CHUNK_BOUNDARY_AMBIGUOUS"
STOP_RESET_GATE = "STOP_CAUSAL_CHUNK_RESET_GATE"
STOP_REPLAY_GATE = "STOP_CAUSAL_CHUNK_REPLAY_GATE"
STOP_ADMISSION_GATE = "STOP_CAUSAL_CHUNK_ADMISSION_GATE"
STOP_CHUNK_UNKNOWN = "STOP_CAUSAL_CHUNK_UNKNOWN"
STOP_CHUNK_AMBIGUOUS = "STOP_CAUSAL_CHUNK_AMBIGUOUS"
STOP_TRACE_COLLISION = "STOP_CAUSAL_CHUNK_TRACE_COLLISION"

PUBLIC_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0

STREAM_BEGIN_CELLS = tuple(BOUNDARY_CELL_OFFSET + lane for lane in range(TRACE_LANES))
STREAM_END_CELLS = tuple(
    BOUNDARY_CELL_OFFSET + TRACE_LANES + lane for lane in range(TRACE_LANES)
)
FEATURE_CELLS = tuple(
    BOUNDARY_CELL_OFFSET + 2 * TRACE_LANES + feature
    for feature in range(FEATURE_COUNT)
)
ACTION_CELLS = tuple(
    BOUNDARY_CELL_OFFSET + 2 * TRACE_LANES + FEATURE_COUNT + action
    for action in range(ACTION_COUNT)
)
TONIC_CELL = BOUNDARY_CELL_OFFSET + 2 * TRACE_LANES + FEATURE_COUNT + ACTION_COUNT
RESET_CELL = TONIC_CELL + 1
REPLAY_CELL = TONIC_CELL + 2
CUT_INHIBITORY_CELL = REPLAY_CELL + 1
HYPOTHESIS_CELLS = tuple(
    tuple(
        CUT_INHIBITORY_CELL + 1 + action * TRACE_LANES + lane
        for lane in range(TRACE_LANES)
    )
    for action in range(ACTION_COUNT)
)
HYPOTHESIS_RACE_CELL_COUNT = (
    sum(len(row) for row in HYPOTHESIS_CELLS)
    + len(ACTION_CELLS)
    + 2  # CUT inhibitory consequence and tonic competitor.
)
HYPOTHESIS_RACE_CELL_STEPS = HYPOTHESIS_RACE_CELL_COUNT * HYPOTHESIS_RACE_STEPS
BOUNDARY_MEMORY_RACE_CELL_STEPS = (
    FEATURE_COUNT + ACTION_COUNT + 1
) * HYPOTHESIS_RACE_STEPS
FIXED_CHUNKER_CELL_COUNT = (
    len(FEATURE_CELLS)
    + len(ACTION_CELLS)
    + 4
    + sum(len(row) for row in HYPOTHESIS_CELLS)
)

_CONFIG_NAMESPACE = b"human-brain/common-causal-predictive-byte-chunker-v1/config\x00"
_CHECKPOINT_NAMESPACE = (
    b"human-brain/common-causal-predictive-byte-chunker-v1/checkpoint\x00"
)
_STREAM_NAMESPACE = b"human-brain/physical-continuous-byte-stream-v1\x00"
_TRACE_NAMESPACE = b"human-brain/causal-chunk-trace-recurrence-v1\x00"
_CHUNK_ROOT_NAMESPACE = b"human-brain/causal-chunk-collision-certificate-v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


CONFIG_RECORD = {
    "input": "begin/sequential-physical-value-assemblies/end/provenance",
    "trace_lanes": TRACE_LANES,
    "max_stream_events": MAX_STREAM_EVENTS,
    "predictor_tonic": PREDICTOR_TONIC,
    "boundary_tonic": BOUNDARY_TONIC,
    "admission_tonic": ADMISSION_TONIC,
    "hypothesis_match_current": HYPOTHESIS_MATCH_CURRENT,
    "hypothesis_tonic_current": HYPOTHESIS_TONIC_CURRENT,
    "hypothesis_race_steps": HYPOTHESIS_RACE_STEPS,
    "cut_continuity_inhibitory_cell": CUT_INHIBITORY_CELL,
    "tick_cap": TICK_CAP,
    "actions": 2,
    "features": 4,
    "lookahead_events": 1,
    "caller_factor_tuple": False,
    "caller_span_or_position": False,
    "host_route_label": False,
    "surprise_threshold": False,
    "parent_replay": 0,
    "restricted_inputs": (
        PUBLIC_INPUT_COUNT,
        DEV_INPUT_COUNT,
        M0_INPUT_COUNT,
        C1_INPUT_COUNT,
        G0_INPUT_COUNT,
        PATH2_INPUT_COUNT,
        PROTECTED_INPUT_COUNT,
        HELDOUT_INPUT_COUNT,
    ),
}
CONFIG_SHA256 = sha256(_CONFIG_NAMESPACE + _canonical_json(CONFIG_RECORD)).hexdigest()


def _validate_assembly(name: str, cells: tuple[int, ...], offset: int) -> None:
    if (
        type(cells) is not tuple
        or len(cells) != TRACE_LANES
        or any(type(cell) is not int for cell in cells)
    ):
        raise TypeError(f"{name} must be one exact twelve-cell physical assembly")
    if len(set(cells)) != TRACE_LANES:
        raise ValueError(f"{name} contains duplicate physical cells")
    for lane, cell in enumerate(cells):
        start = offset + lane * INPUT_LANE_BANK_CELLS
        if not start <= cell < start + INPUT_LANE_BANK_CELLS:
            raise ValueError(f"{name} left its physical lane bank")


def _stream_digest(
    begin_cells: tuple[int, ...],
    events: tuple[tuple[int, ...], ...],
    end_cells: tuple[int, ...],
    provenance_cells: tuple[int, ...],
) -> str:
    digest = sha256()
    digest.update(_STREAM_NAMESPACE)
    for assembly in (begin_cells, *events, end_cells, provenance_cells):
        digest.update(np.asarray(assembly, dtype="<i8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalContinuousByteStream:
    begin_cells: tuple[int, ...]
    events: tuple[tuple[int, ...], ...]
    end_cells: tuple[int, ...]
    provenance_cells: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        self.authenticate()

    def authenticate(self) -> None:
        if self.begin_cells != STREAM_BEGIN_CELLS or self.end_cells != STREAM_END_CELLS:
            raise ValueError("whole-stream physical boundary assembly changed")
        if type(self.events) is not tuple or not 2 <= len(self.events) <= MAX_STREAM_EVENTS:
            raise ValueError("stream must contain one bounded continuous event sequence")
        for event in self.events:
            _validate_assembly("stream event", event, INPUT_CELL_OFFSET)
        _validate_assembly(
            "stream provenance", self.provenance_cells, PROVENANCE_CELL_OFFSET
        )
        expected = _stream_digest(
            self.begin_cells, self.events, self.end_cells, self.provenance_cells
        )
        if type(self.sha256) is not str or self.sha256 != expected:
            raise ValueError("continuous physical stream authentication failed")


def seal_physical_continuous_byte_stream(
    begin_cells: tuple[int, ...],
    events: tuple[tuple[int, ...], ...],
    end_cells: tuple[int, ...],
    provenance_cells: tuple[int, ...],
) -> PhysicalContinuousByteStream:
    """Authenticate one caller-visible stream without deriving spans or factors."""

    return PhysicalContinuousByteStream(
        begin_cells=begin_cells,
        events=events,
        end_cells=end_cells,
        provenance_cells=provenance_cells,
        sha256=_stream_digest(begin_cells, events, end_cells, provenance_cells),
    )


@dataclass(frozen=True, slots=True)
class PhysicalContinuousRouteLesson:
    stream: PhysicalContinuousByteStream
    exact_observation_events: tuple[bytes, ...]
    semantic_input_frame: filler_projection.CommonOneByteFillerFrame
    relation_cells: tuple[int, ...]
    evidence_context_cells: tuple[int, ...]
    evidence_source_cells: tuple[int, ...]
    observed_output_cells: tuple[int, ...]
    teacher_gate: bool = True
    cell_gate: bool = True
    comparator_gate: bool = True

    def __post_init__(self) -> None:
        if type(self.stream) is not PhysicalContinuousByteStream:
            raise TypeError("route lesson stream type changed")
        self.stream.authenticate()
        if any(
            type(gate) is not bool
            for gate in (self.teacher_gate, self.cell_gate, self.comparator_gate)
        ):
            raise TypeError("route lesson gates must be exact booleans")


@dataclass(frozen=True, slots=True)
class PredictorAssociation:
    source_cells: tuple[int, ...]
    target_cells: tuple[int, ...]
    independent_stream_support: int
    admitted: bool


@dataclass(frozen=True, slots=True)
class CausalChunkTrainingSummary:
    mode: str
    predictor_stream_count: int
    new_route_lesson_count: int
    total_route_lesson_count: int
    predictor_evidence_count: int
    predictor_duplicate_nonvote_count: int
    predictor_association_count: int
    recurrent_association_count: int
    boundary_update_count: int
    boundary_unidentifiable_count: int
    admission_evidence_count: int
    admission_duplicate_nonvote_count: int
    admitted_chunk_count: int
    trace_collision_count: int
    routed_lesson_count: int
    predictor_cell_steps: int
    comparator_cell_steps: int
    hypothesis_cell_steps: int
    replay_cell_steps: int
    admission_cell_steps: int
    route_learning_cell_steps: int
    fixed_chunker_cell_count: int
    predictor_match_cell_count: int
    admission_match_cell_count: int
    added_chunker_cell_count: int
    chunker_checkpoint_array_bytes: int
    router_checkpoint_array_bytes: int
    total_checkpoint_array_bytes: int
    raw_stream_retention_count: int
    caller_span_retention_count: int
    parent_lesson_replay_count: int
    memory_checkpoint_sha256: str
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    router_checkpoint_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class CausalChunkRecallWork:
    event_assembly_reads: int
    predictor_equality_reads: int
    comparator_cell_steps: int
    hypothesis_cell_steps: int
    replay_cell_steps: int
    trace_recurrence_cell_steps: int
    admission_equality_reads: int
    router_cell_steps: int

    @property
    def total_cell_steps(self) -> int:
        return (
            self.comparator_cell_steps
            + self.hypothesis_cell_steps
            + self.replay_cell_steps
            + self.trace_recurrence_cell_steps
            + self.router_cell_steps
        )


@dataclass(frozen=True, slots=True)
class CausalPredictiveChunkRecall:
    status: str
    cut_offsets: tuple[int, ...]
    admitted_chunk_traces: tuple[tuple[int, ...], ...]
    boundary_feature_cells: tuple[int, ...]
    boundary_action_cells: tuple[int, ...]
    cut_inhibitory_consequence_spikes: tuple[bool, ...]
    routed_recall: router.FactorizedContextRecall | None
    work: CausalChunkRecallWork
    memory_checkpoint_sha256: str
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    router_checkpoint_sha256: str
    checkpoint_sha256: str
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "cut_offsets": self.cut_offsets,
            "admitted_chunk_traces": self.admitted_chunk_traces,
            "boundary_feature_cells": self.boundary_feature_cells,
            "boundary_action_cells": self.boundary_action_cells,
            "cut_inhibitory_consequence_spikes": (
                self.cut_inhibitory_consequence_spikes
            ),
            "routed_recall": (
                None if self.routed_recall is None else self.routed_recall.record()
            ),
            "work": dataclasses.asdict(self.work)
            | {"total_cell_steps": self.work.total_cell_steps},
            "memory_checkpoint_sha256": self.memory_checkpoint_sha256,
            "episodic_checkpoint_sha256": self.episodic_checkpoint_sha256,
            "semantic_checkpoint_sha256": self.semantic_checkpoint_sha256,
            "router_checkpoint_sha256": self.router_checkpoint_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "caller_factor_tuple_input": False,
            "caller_span_or_position_input": False,
            "host_route_label_input": False,
            "host_slot_order_authority": False,
            "surprise_threshold_authority": False,
            "raw_stream_replay": False,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _Prediction:
    status: str
    target_cells: tuple[int, ...] | None
    equality_reads: int
    cell_steps: int


@dataclass(frozen=True, slots=True)
class _HypothesisRace:
    status: str
    excitatory_currents: tuple[int, int]
    inhibitory_currents: tuple[int, int]
    net_currents: tuple[int, int]
    action_spikes: tuple[bool, bool]
    cut_inhibitory_spike: bool
    selected_action: int | None
    selected_action_cell: int | None
    cell_steps: int


@dataclass(frozen=True, slots=True)
class _Segment:
    status: str
    cuts: tuple[int, ...]
    traces: tuple[tuple[int, ...], ...]
    chunk_roots: tuple[bytes, ...]
    feature_cells: tuple[int, ...]
    action_cells: tuple[int, ...]
    cut_inhibitory_spikes: tuple[bool, ...]
    predictor_reads: int
    comparator_steps: int
    hypothesis_steps: int
    replay_steps: int
    trace_steps: int


def _freeze(array: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _chunk_root(events: tuple[tuple[int, ...], ...]) -> bytes:
    digest = sha256()
    digest.update(_CHUNK_ROOT_NAMESPACE)
    for event in events:
        digest.update(np.asarray(event, dtype="<i8").tobytes())
    return digest.digest()


def _trace_seed() -> tuple[int, ...]:
    return tuple(
        physical_cells.TRACE_CELL_OFFSET
        + lane * physical_cells.FEATURE_BANK_CELLS
        for lane in range(TRACE_LANES)
    )


def _advance_trace(
    state: tuple[int, ...], event: tuple[int, ...]
) -> tuple[int, ...]:
    result: list[int] = []
    state_bytes = np.asarray(state, dtype="<i8").tobytes()
    event_bytes = np.asarray(event, dtype="<i8").tobytes()
    for lane in range(TRACE_LANES):
        digest = sha256()
        digest.update(_TRACE_NAMESPACE)
        digest.update(bytes((lane,)))
        digest.update(state_bytes)
        digest.update(event_bytes)
        local = int.from_bytes(digest.digest()[:4], "little")
        result.append(
            physical_cells.TRACE_CELL_OFFSET
            + lane * physical_cells.FEATURE_BANK_CELLS
            + local % physical_cells.FEATURE_BANK_CELLS
        )
    return tuple(result)


def _array_frame(name: str, array: np.ndarray) -> bytes:
    name_bytes = name.encode("ascii")
    dtype_bytes = array.dtype.str.encode("ascii")
    return b"".join(
        (
            len(name_bytes).to_bytes(2, "little"),
            name_bytes,
            len(dtype_bytes).to_bytes(2, "little"),
            dtype_bytes,
            np.asarray(array.shape, dtype="<u8").tobytes(),
            array.tobytes(order="C"),
        )
    )


def _checkpoint_sha256(
    routed: router.FrozenCommonFactorizedContextRoutedMemoryV2,
    summary: CausalChunkTrainingSummary,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> str:
    record = dataclasses.asdict(summary)
    record["checkpoint_sha256"] = ""
    digest = sha256()
    digest.update(_CHECKPOINT_NAMESPACE)
    digest.update(CONFIG_SHA256.encode("ascii"))
    digest.update(routed.checkpoint_sha256.encode("ascii"))
    digest.update(_canonical_json(record))
    for name, array in arrays:
        digest.update(_array_frame(name, array))
    return digest.hexdigest()


def _fit_predictor(
    streams: tuple[PhysicalContinuousByteStream, ...],
    *,
    predictor_gate: bool,
    plasticity_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    evidence: list[tuple[tuple[int, ...], tuple[int, ...], bytes]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...], bytes]] = set()
    duplicate = 0
    if predictor_gate and plasticity_gate:
        for stream in streams:
            stream.authenticate()
            root = bytes.fromhex(stream.sha256)
            within_stream: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
            for source, target in zip(stream.events, stream.events[1:], strict=False):
                within_stream.add((source, target))
            for source, target in within_stream:
                key = (source, target, root)
                if key in seen:
                    duplicate += 1
                    continue
                seen.add(key)
                evidence.append(key)
    if len(evidence) > MAX_PREDICTOR_EVIDENCE:
        raise ValueError("predictor evidence capacity exceeded")
    supports: dict[tuple[tuple[int, ...], tuple[int, ...]], set[bytes]] = {}
    for source, target, root in evidence:
        supports.setdefault((source, target), set()).add(root)
    associations = sorted(supports)
    if len(associations) > MAX_PREDICTOR_ASSOCIATIONS:
        raise ValueError("predictor association capacity exceeded")
    return (
        np.asarray([row[0] for row in evidence], dtype=np.int32).reshape(
            -1, TRACE_LANES
        ),
        np.asarray([row[1] for row in evidence], dtype=np.int32).reshape(
            -1, TRACE_LANES
        ),
        np.asarray([list(row[2]) for row in evidence], dtype=np.uint8).reshape(-1, 32),
        np.asarray([row[0] for row in associations], dtype=np.int32).reshape(
            -1, TRACE_LANES
        ),
        np.asarray([row[1] for row in associations], dtype=np.int32).reshape(
            -1, TRACE_LANES
        ),
        np.asarray(
            [min(TICK_CAP, len(supports[row])) for row in associations],
            dtype=np.uint8,
        ),
        duplicate,
    )


def _predict(
    sources: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
    source_cells: tuple[int, ...],
    *,
    predictor_gate: bool,
    comparator_gate: bool,
) -> _Prediction:
    if not predictor_gate:
        return _Prediction(STOP_PREDICTOR_GATE, None, 0, 0)
    if not comparator_gate:
        return _Prediction(STOP_COMPARATOR_GATE, None, 0, 0)
    equality_reads = int(sources.shape[0] * TRACE_LANES)
    matched = np.all(sources == np.asarray(source_cells, dtype=np.int32), axis=1)
    recurrent = np.flatnonzero(matched & (support > PREDICTOR_TONIC))
    unique_targets = {tuple(int(value) for value in targets[row]) for row in recurrent}
    if len(unique_targets) == 1:
        return _Prediction(
            RECALLED,
            next(iter(unique_targets)),
            equality_reads,
            int(sources.shape[0] * (TRACE_LANES + 1)),
        )
    status = STOP_PREDICTOR_UNKNOWN if not unique_targets else STOP_PREDICTOR_AMBIGUOUS
    return _Prediction(
        status,
        None,
        equality_reads,
        int(sources.shape[0] * (TRACE_LANES + 1)),
    )


def _feature(
    sources: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
    previous: tuple[int, ...],
    trigger: tuple[int, ...],
    future: tuple[int, ...] | None,
    *,
    predictor_gate: bool,
    comparator_gate: bool,
) -> tuple[int, _Prediction, _Prediction | None, bool, bool]:
    past = _predict(
        sources,
        targets,
        support,
        previous,
        predictor_gate=predictor_gate,
        comparator_gate=comparator_gate,
    )
    future_prediction = (
        None
        if future is None
        else _predict(
            sources,
            targets,
            support,
            trigger,
            predictor_gate=predictor_gate,
            comparator_gate=comparator_gate,
        )
    )
    past_match = past.status == RECALLED and past.target_cells == trigger
    future_match = (
        future is not None
        and future_prediction is not None
        and future_prediction.status == RECALLED
        and future_prediction.target_cells == future
    )
    return (
        (2 if past_match else 0) + (1 if future_match else 0),
        past,
        future_prediction,
        past_match,
        bool(future_match),
    )


def _hypothesis_race(
    *,
    past_match_spike: bool,
    future_match_spike: bool,
    hypothesis_action_gate: bool,
) -> _HypothesisRace:
    """Race physical CONTINUE and CUT_REPLAY consequences against tonic.

    A future match excites CUT_REPLAY, but a simultaneous continuity match
    drives the dedicated CUT inhibitory cell.  Thus feature 3 is resolved by
    current cancellation at CUT, never by action ordering or host priority.
    """

    excitatory = (
        HYPOTHESIS_MATCH_CURRENT if past_match_spike else 0,
        HYPOTHESIS_MATCH_CURRENT if future_match_spike else 0,
    )
    inhibitory = (
        0,
        HYPOTHESIS_MATCH_CURRENT if past_match_spike else 0,
    )
    net = tuple(
        max(0, excitation - inhibition)
        for excitation, inhibition in zip(excitatory, inhibitory, strict=True)
    )
    spikes = tuple(
        bool(hypothesis_action_gate and current > HYPOTHESIS_TONIC_CURRENT)
        for current in net
    )
    active = tuple(index for index, spike in enumerate(spikes) if spike)
    if not hypothesis_action_gate:
        status = STOP_HYPOTHESIS_ACTION_GATE
        selected = None
    elif len(active) == 1:
        status = RECALLED
        selected = active[0]
    elif active:
        status = STOP_BOUNDARY_AMBIGUOUS
        selected = None
    else:
        status = STOP_BOUNDARY_UNIDENTIFIABLE
        selected = None
    return _HypothesisRace(
        status=status,
        excitatory_currents=excitatory,
        inhibitory_currents=inhibitory,
        net_currents=net,
        action_spikes=(bool(spikes[0]), bool(spikes[1])),
        cut_inhibitory_spike=past_match_spike,
        selected_action=selected,
        selected_action_cell=None if selected is None else ACTION_CELLS[selected],
        cell_steps=HYPOTHESIS_RACE_CELL_STEPS,
    )


def _learn_boundary_and_segment(
    stream: PhysicalContinuousByteStream,
    sources: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    seen_updates: set[tuple[int, int, bytes]],
    *,
    predictor_gate: bool,
    comparator_gate: bool,
    hypothesis_action_gate: bool,
    reset_gate: bool,
    replay_gate: bool,
    plasticity_gate: bool,
) -> tuple[_Segment, int, int]:
    stream.authenticate()
    root = bytes.fromhex(stream.sha256)
    chunks: list[tuple[tuple[int, ...], ...]] = []
    current: list[tuple[int, ...]] = [stream.events[0]]
    state = _advance_trace(_trace_seed(), stream.events[0])
    cuts: list[int] = []
    feature_cells: list[int] = []
    action_cells: list[int] = []
    cut_inhibitory_spikes: list[bool] = []
    reads = comparator_steps = hypothesis_steps = replay_steps = 0
    updates = unresolved = 0
    for offset in range(1, len(stream.events)):
        trigger = stream.events[offset]
        future = stream.events[offset + 1] if offset + 1 < len(stream.events) else None
        feature, past, future_prediction, past_match, future_match = _feature(
            sources,
            targets,
            support,
            stream.events[offset - 1],
            trigger,
            future,
            predictor_gate=predictor_gate,
            comparator_gate=comparator_gate,
        )
        reads += past.equality_reads + (
            0 if future_prediction is None else future_prediction.equality_reads
        )
        comparator_steps += past.cell_steps + (
            0 if future_prediction is None else future_prediction.cell_steps
        )
        if not predictor_gate:
            return _empty_segment(STOP_PREDICTOR_GATE, reads, comparator_steps), updates, unresolved
        if not comparator_gate:
            return (
                _empty_segment(STOP_COMPARATOR_GATE, reads, comparator_steps),
                updates,
                unresolved,
            )
        feature_cells.append(FEATURE_CELLS[feature])
        hypothesis = _hypothesis_race(
            past_match_spike=past_match,
            future_match_spike=future_match,
            hypothesis_action_gate=hypothesis_action_gate,
        )
        hypothesis_steps += hypothesis.cell_steps
        if hypothesis.selected_action is None:
            unresolved += 1
            return (
                _empty_segment(
                    hypothesis.status, reads, comparator_steps, hypothesis_steps
                ),
                updates,
                unresolved,
            )
        action = hypothesis.selected_action
        action_cells.append(int(hypothesis.selected_action_cell))
        cut_inhibitory_spikes.append(hypothesis.cut_inhibitory_spike)
        update_key = (feature, action, root)
        if plasticity_gate and update_key not in seen_updates:
            seen_updates.add(update_key)
            excitatory[feature, action] = min(
                TICK_CAP, int(excitatory[feature, action]) + 1
            )
            rival = 1 - action
            inhibitory[feature, rival] = min(
                TICK_CAP, int(inhibitory[feature, rival]) + 1
            )
            updates += 1
        if action == CONTINUE:
            current.append(trigger)
            state = _advance_trace(state, trigger)
            continue
        if not reset_gate:
            return _empty_segment(STOP_RESET_GATE, reads, comparator_steps), updates, unresolved
        if not replay_gate:
            return _empty_segment(STOP_REPLAY_GATE, reads, comparator_steps), updates, unresolved
        chunks.append(tuple(current))
        cuts.append(offset)
        current = [trigger]
        state = _advance_trace(_trace_seed(), trigger)
        replay_steps += TRACE_LANES + 2
    chunks.append(tuple(current))
    traces = tuple(_trace_chunk(chunk) for chunk in chunks)
    roots = tuple(_chunk_root(chunk) for chunk in chunks)
    trace_steps = sum(len(chunk) * TRACE_LANES for chunk in chunks)
    return (
        _Segment(
            RECALLED,
            tuple(cuts),
            traces,
            roots,
            tuple(feature_cells),
            tuple(action_cells),
            tuple(cut_inhibitory_spikes),
            reads,
            comparator_steps,
            hypothesis_steps,
            replay_steps,
            trace_steps,
        ),
        updates,
        unresolved,
    )


def _empty_segment(
    status: str,
    reads: int,
    comparator_steps: int,
    hypothesis_steps: int = 0,
) -> _Segment:
    return _Segment(
        status,
        (),
        (),
        (),
        (),
        (),
        (),
        reads,
        comparator_steps,
        hypothesis_steps,
        0,
        0,
    )


def _trace_chunk(
    chunk: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    state = _trace_seed()
    for event in chunk:
        state = _advance_trace(state, event)
    return state


def _boundary_action(
    excitatory: np.ndarray, inhibitory: np.ndarray, feature: int
) -> tuple[str, int | None]:
    raw = np.maximum(
        excitatory[feature].astype(np.int16) - inhibitory[feature].astype(np.int16),
        0,
    )
    active = np.flatnonzero(raw > BOUNDARY_TONIC)
    if active.size == 1:
        return RECALLED, int(active[0])
    if active.size > 1:
        return STOP_BOUNDARY_AMBIGUOUS, None
    return STOP_BOUNDARY_UNIDENTIFIABLE, None


def _segment_recall(
    stream: PhysicalContinuousByteStream,
    sources: np.ndarray,
    targets: np.ndarray,
    support: np.ndarray,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    *,
    predictor_gate: bool,
    comparator_gate: bool,
    hypothesis_action_gate: bool,
    reset_gate: bool,
    replay_gate: bool,
) -> _Segment:
    stream.authenticate()
    chunks: list[tuple[tuple[int, ...], ...]] = []
    current: list[tuple[int, ...]] = [stream.events[0]]
    cuts: list[int] = []
    feature_cells: list[int] = []
    action_cells: list[int] = []
    cut_inhibitory_spikes: list[bool] = []
    reads = comparator_steps = hypothesis_steps = replay_steps = 0
    for offset in range(1, len(stream.events)):
        trigger = stream.events[offset]
        future = stream.events[offset + 1] if offset + 1 < len(stream.events) else None
        feature, past, future_prediction, past_match, future_match = _feature(
            sources,
            targets,
            support,
            stream.events[offset - 1],
            trigger,
            future,
            predictor_gate=predictor_gate,
            comparator_gate=comparator_gate,
        )
        reads += past.equality_reads + (
            0 if future_prediction is None else future_prediction.equality_reads
        )
        comparator_steps += past.cell_steps + (
            0 if future_prediction is None else future_prediction.cell_steps
        )
        if not predictor_gate:
            return _empty_segment(STOP_PREDICTOR_GATE, reads, comparator_steps)
        if not comparator_gate:
            return _empty_segment(STOP_COMPARATOR_GATE, reads, comparator_steps)
        feature_cells.append(FEATURE_CELLS[feature])
        hypothesis = _hypothesis_race(
            past_match_spike=past_match,
            future_match_spike=future_match,
            hypothesis_action_gate=hypothesis_action_gate,
        )
        hypothesis_steps += hypothesis.cell_steps
        if hypothesis.selected_action is None:
            return _empty_segment(
                hypothesis.status, reads, comparator_steps, hypothesis_steps
            )
        status, action = _boundary_action(excitatory, inhibitory, feature)
        hypothesis_steps += BOUNDARY_MEMORY_RACE_CELL_STEPS
        if action is None:
            return _empty_segment(status, reads, comparator_steps, hypothesis_steps)
        if action != hypothesis.selected_action:
            return _empty_segment(
                STOP_BOUNDARY_AMBIGUOUS, reads, comparator_steps, hypothesis_steps
            )
        action_cells.append(int(hypothesis.selected_action_cell))
        cut_inhibitory_spikes.append(hypothesis.cut_inhibitory_spike)
        if action == CONTINUE:
            current.append(trigger)
            continue
        if not reset_gate:
            return _empty_segment(STOP_RESET_GATE, reads, comparator_steps, hypothesis_steps)
        if not replay_gate:
            return _empty_segment(STOP_REPLAY_GATE, reads, comparator_steps, hypothesis_steps)
        chunks.append(tuple(current))
        cuts.append(offset)
        current = [trigger]
        replay_steps += TRACE_LANES + 2
    chunks.append(tuple(current))
    traces = tuple(_trace_chunk(chunk) for chunk in chunks)
    roots = tuple(_chunk_root(chunk) for chunk in chunks)
    return _Segment(
        RECALLED,
        tuple(cuts),
        traces,
        roots,
        tuple(feature_cells),
        tuple(action_cells),
        tuple(cut_inhibitory_spikes),
        reads,
        comparator_steps,
        hypothesis_steps,
        replay_steps,
        sum(len(chunk) * TRACE_LANES for chunk in chunks),
    )


def _router_lesson(
    lesson: PhysicalContinuousRouteLesson,
    factors: tuple[tuple[int, ...], ...],
) -> router.PhysicalFactorRouteLesson:
    return router.PhysicalFactorRouteLesson(
        factor_cells=factors,
        exact_observation_events=lesson.exact_observation_events,
        semantic_input_frame=lesson.semantic_input_frame,
        relation_cells=lesson.relation_cells,
        evidence_context_cells=lesson.evidence_context_cells,
        evidence_source_cells=lesson.evidence_source_cells,
        observed_output_cells=lesson.observed_output_cells,
        teacher_gate=lesson.teacher_gate,
        cell_gate=lesson.cell_gate,
        comparator_gate=lesson.comparator_gate,
    )


class FrozenCommonCausalPredictiveByteChunkerV1:
    """Immutable predictor, boundary lattice, chunk admissions, and V2 router."""

    __slots__ = (
        "_admission_chunk_roots",
        "_admission_evidence_roots",
        "_admission_evidence_traces",
        "_admission_support",
        "_admitted_traces",
        "_boundary_excitatory",
        "_boundary_inhibitory",
        "_checkpoint_sha256",
        "_predictor_evidence_roots",
        "_predictor_evidence_sources",
        "_predictor_evidence_targets",
        "_predictor_sources",
        "_predictor_support",
        "_predictor_targets",
        "_router",
        "_summary",
    )

    def __init__(
        self,
        *,
        routed: router.FrozenCommonFactorizedContextRoutedMemoryV2,
        arrays: tuple[np.ndarray, ...],
        summary: CausalChunkTrainingSummary,
    ) -> None:
        self._router = routed
        (
            predictor_evidence_sources,
            predictor_evidence_targets,
            predictor_evidence_roots,
            predictor_sources,
            predictor_targets,
            predictor_support,
            boundary_excitatory,
            boundary_inhibitory,
            admission_evidence_traces,
            admission_evidence_roots,
            admitted_traces,
            admission_support,
            admission_chunk_roots,
        ) = arrays
        self._predictor_evidence_sources = _freeze(predictor_evidence_sources, np.int32)
        self._predictor_evidence_targets = _freeze(predictor_evidence_targets, np.int32)
        self._predictor_evidence_roots = _freeze(predictor_evidence_roots, np.uint8)
        self._predictor_sources = _freeze(predictor_sources, np.int32)
        self._predictor_targets = _freeze(predictor_targets, np.int32)
        self._predictor_support = _freeze(predictor_support, np.uint8)
        self._boundary_excitatory = _freeze(boundary_excitatory, np.uint8)
        self._boundary_inhibitory = _freeze(boundary_inhibitory, np.uint8)
        self._admission_evidence_traces = _freeze(admission_evidence_traces, np.int32)
        self._admission_evidence_roots = _freeze(admission_evidence_roots, np.uint8)
        self._admitted_traces = _freeze(admitted_traces, np.int32)
        self._admission_support = _freeze(admission_support, np.uint8)
        self._admission_chunk_roots = _freeze(admission_chunk_roots, np.uint8)
        self._summary = summary
        self._checkpoint_sha256 = _checkpoint_sha256(
            self._router, self._summary, self._checkpoint_arrays()
        )
        self.authenticate()

    def _checkpoint_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("predictor-evidence-sources", self._predictor_evidence_sources),
            ("predictor-evidence-targets", self._predictor_evidence_targets),
            ("predictor-evidence-roots", self._predictor_evidence_roots),
            ("predictor-sources", self._predictor_sources),
            ("predictor-targets", self._predictor_targets),
            ("predictor-support", self._predictor_support),
            ("boundary-excitatory", self._boundary_excitatory),
            ("boundary-inhibitory", self._boundary_inhibitory),
            ("admission-evidence-traces", self._admission_evidence_traces),
            ("admission-evidence-roots", self._admission_evidence_roots),
            ("admitted-traces", self._admitted_traces),
            ("admission-support", self._admission_support),
            ("admission-chunk-roots", self._admission_chunk_roots),
        )

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def memory_checkpoint_sha256(self) -> str:
        return self._router.memory_checkpoint_sha256

    @property
    def episodic_checkpoint_sha256(self) -> str:
        return self._router.episodic_checkpoint_sha256

    @property
    def semantic_checkpoint_sha256(self) -> str:
        return self._router.semantic_checkpoint_sha256

    @property
    def router_checkpoint_sha256(self) -> str:
        return self._router.checkpoint_sha256

    @property
    def training_summary(self) -> CausalChunkTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def router_training_summary(self) -> router.FactorRouterTrainingSummary:
        self.authenticate()
        return self._router.training_summary

    @property
    def predictor_associations(self) -> tuple[PredictorAssociation, ...]:
        self.authenticate()
        return tuple(
            PredictorAssociation(
                tuple(int(value) for value in self._predictor_sources[row]),
                tuple(int(value) for value in self._predictor_targets[row]),
                int(self._predictor_support[row]),
                bool(self._predictor_support[row] > PREDICTOR_TONIC),
            )
            for row in range(self._predictor_sources.shape[0])
        )

    @property
    def admitted_chunk_traces(self) -> tuple[tuple[int, ...], ...]:
        self.authenticate()
        return tuple(tuple(int(value) for value in row) for row in self._admitted_traces)

    def authenticate(self) -> None:
        self._router.authenticate()
        arrays = self._checkpoint_arrays()
        if any(array.flags.writeable for _, array in arrays):
            raise RuntimeError("causal chunker checkpoint became mutable")
        if self._predictor_sources.shape != self._predictor_targets.shape:
            raise RuntimeError("predictor association geometry changed")
        if self._predictor_sources.shape[1:] != (TRACE_LANES,):
            raise RuntimeError("predictor lane geometry changed")
        if self._boundary_excitatory.shape != (FEATURE_COUNT, ACTION_COUNT):
            raise RuntimeError("boundary E/H geometry changed")
        if self._boundary_inhibitory.shape != (FEATURE_COUNT, ACTION_COUNT):
            raise RuntimeError("boundary E/H geometry changed")
        if self._admitted_traces.shape[1:] != (TRACE_LANES,):
            raise RuntimeError("admission geometry changed")
        if len({tuple(row) for row in self._admitted_traces.tolist()}) != len(
            self._admitted_traces
        ):
            raise RuntimeError("duplicate admitted physical trace")
        array_bytes = sum(array.nbytes for _, array in arrays)
        summary = self._summary
        if (
            summary.predictor_evidence_count != self._predictor_evidence_roots.shape[0]
            or summary.predictor_association_count != self._predictor_support.shape[0]
            or summary.recurrent_association_count
            != int(np.count_nonzero(self._predictor_support > PREDICTOR_TONIC))
            or summary.admission_evidence_count
            != self._admission_evidence_roots.shape[0]
            or summary.admitted_chunk_count != self._admitted_traces.shape[0]
            or summary.chunker_checkpoint_array_bytes != array_bytes
            or summary.router_checkpoint_array_bytes
            != self._router.training_summary.checkpoint_array_bytes
            or summary.total_checkpoint_array_bytes
            != array_bytes + self._router.training_summary.checkpoint_array_bytes
            or summary.fixed_chunker_cell_count != FIXED_CHUNKER_CELL_COUNT
            or summary.predictor_match_cell_count != self._predictor_sources.shape[0]
            or summary.admission_match_cell_count != self._admitted_traces.shape[0]
            or summary.added_chunker_cell_count
            != FIXED_CHUNKER_CELL_COUNT
            + self._predictor_sources.shape[0]
            + self._admitted_traces.shape[0]
            or summary.raw_stream_retention_count != 0
            or summary.caller_span_retention_count != 0
            or summary.parent_lesson_replay_count != 0
            or summary.router_checkpoint_sha256 != self._router.checkpoint_sha256
            or summary.memory_checkpoint_sha256 != self._router.memory_checkpoint_sha256
            or summary.episodic_checkpoint_sha256
            != self._router.episodic_checkpoint_sha256
            or summary.semantic_checkpoint_sha256
            != self._router.semantic_checkpoint_sha256
            or _checkpoint_sha256(self._router, summary, arrays)
            != self._checkpoint_sha256
        ):
            raise RuntimeError("causal chunker checkpoint authentication failed")

    def _match_admissions(
        self,
        traces: tuple[tuple[int, ...], ...],
        roots: tuple[bytes, ...],
        *,
        admission_gate: bool,
    ) -> tuple[str, tuple[tuple[int, ...], ...], int]:
        if not admission_gate:
            return STOP_ADMISSION_GATE, (), 0
        selected: list[tuple[int, ...]] = []
        reads = 0
        for trace, root in zip(traces, roots, strict=True):
            reads += int(self._admitted_traces.shape[0] * TRACE_LANES)
            matches = np.flatnonzero(
                np.all(
                    self._admitted_traces == np.asarray(trace, dtype=np.int32), axis=1
                )
                & (self._admission_support > ADMISSION_TONIC)
            )
            if matches.size == 0:
                return STOP_CHUNK_UNKNOWN, (), reads
            if matches.size != 1:
                return STOP_CHUNK_AMBIGUOUS, (), reads
            row = int(matches[0])
            if bytes(self._admission_chunk_roots[row]) != root:
                return STOP_TRACE_COLLISION, (), reads
            selected.append(trace)
        if len(set(selected)) != len(selected):
            return STOP_CHUNK_AMBIGUOUS, (), reads
        return RECALLED, tuple(selected), reads

    def recall(
        self,
        stream: PhysicalContinuousByteStream,
        exact_observation_events: tuple[bytes, ...],
        semantic_input_frame: filler_projection.CommonOneByteFillerFrame,
        relation_cells: tuple[int, ...],
        *,
        predictor_gate: bool = True,
        comparator_gate: bool = True,
        hypothesis_action_gate: bool = True,
        reset_gate: bool = True,
        replay_gate: bool = True,
        admission_gate: bool = True,
        router_gate: bool = True,
    ) -> CausalPredictiveChunkRecall:
        self.authenticate()
        if any(
            type(gate) is not bool
            for gate in (
                predictor_gate,
                comparator_gate,
                hypothesis_action_gate,
                reset_gate,
                replay_gate,
                admission_gate,
                router_gate,
            )
        ):
            raise TypeError("recall gates must be exact booleans")
        segment = _segment_recall(
            stream,
            self._predictor_sources,
            self._predictor_targets,
            self._predictor_support,
            self._boundary_excitatory,
            self._boundary_inhibitory,
            predictor_gate=predictor_gate,
            comparator_gate=comparator_gate,
            hypothesis_action_gate=hypothesis_action_gate,
            reset_gate=reset_gate,
            replay_gate=replay_gate,
        )
        admitted_status = segment.status
        admitted: tuple[tuple[int, ...], ...] = ()
        admission_reads = 0
        routed_recall = None
        if segment.status == RECALLED:
            admitted_status, admitted, admission_reads = self._match_admissions(
                segment.traces, segment.chunk_roots, admission_gate=admission_gate
            )
        if admitted_status == RECALLED:
            routed_recall = self._router.recall(
                admitted,
                exact_observation_events,
                semantic_input_frame,
                relation_cells,
                router_gate=router_gate,
            )
            status = (
                RECALLED if routed_recall.status == router.RECALLED else routed_recall.status
            )
        else:
            status = admitted_status
        work = CausalChunkRecallWork(
            event_assembly_reads=len(stream.events) * TRACE_LANES,
            predictor_equality_reads=segment.predictor_reads,
            comparator_cell_steps=segment.comparator_steps,
            hypothesis_cell_steps=segment.hypothesis_steps,
            replay_cell_steps=segment.replay_steps,
            trace_recurrence_cell_steps=segment.trace_steps,
            admission_equality_reads=admission_reads,
            router_cell_steps=(
                0 if routed_recall is None else routed_recall.work.router_cell_steps
            ),
        )
        record = {
            "status": status,
            "cuts": segment.cuts,
            "traces": admitted,
            "features": segment.feature_cells,
            "actions": segment.action_cells,
            "cut_inhibition": segment.cut_inhibitory_spikes,
            "routed": None if routed_recall is None else routed_recall.record(),
            "work": dataclasses.asdict(work),
            "checkpoint": self._checkpoint_sha256,
        }
        return CausalPredictiveChunkRecall(
            status=status,
            cut_offsets=segment.cuts,
            admitted_chunk_traces=admitted,
            boundary_feature_cells=segment.feature_cells,
            boundary_action_cells=segment.action_cells,
            cut_inhibitory_consequence_spikes=segment.cut_inhibitory_spikes,
            routed_recall=routed_recall,
            work=work,
            memory_checkpoint_sha256=self.memory_checkpoint_sha256,
            episodic_checkpoint_sha256=self.episodic_checkpoint_sha256,
            semantic_checkpoint_sha256=self.semantic_checkpoint_sha256,
            router_checkpoint_sha256=self.router_checkpoint_sha256,
            checkpoint_sha256=self._checkpoint_sha256,
            sha256=sha256(_canonical_json(record)).hexdigest(),
        )


def _validate_inputs(
    predictor_streams: tuple[PhysicalContinuousByteStream, ...],
    route_lessons: tuple[PhysicalContinuousRouteLesson, ...],
) -> None:
    if type(predictor_streams) is not tuple or any(
        type(stream) is not PhysicalContinuousByteStream for stream in predictor_streams
    ):
        raise TypeError("predictor streams must be one exact tuple")
    if type(route_lessons) is not tuple or any(
        type(lesson) is not PhysicalContinuousRouteLesson for lesson in route_lessons
    ):
        raise TypeError("route lessons must be one exact tuple")
    for stream in predictor_streams:
        stream.authenticate()
    for lesson in route_lessons:
        lesson.stream.authenticate()


def fit_common_causal_predictive_byte_chunker_v1(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    predictor_streams: tuple[PhysicalContinuousByteStream, ...],
    route_lessons: tuple[PhysicalContinuousRouteLesson, ...],
    *,
    predictor_gate: bool = True,
    comparator_gate: bool = True,
    hypothesis_action_gate: bool = True,
    reset_gate: bool = True,
    replay_gate: bool = True,
    plasticity_gate: bool = True,
    admission_gate: bool = True,
) -> FrozenCommonCausalPredictiveByteChunkerV1:
    """Fit recurrence, causal cuts, chunk admissions, then the public router."""

    if type(child) is not memory.FrozenCommonEpisodicTransformMemoryV2:
        raise TypeError("child must be the exact frozen V2 composite")
    if any(
        type(gate) is not bool
        for gate in (
            predictor_gate,
            comparator_gate,
            hypothesis_action_gate,
            reset_gate,
            replay_gate,
            plasticity_gate,
            admission_gate,
        )
    ):
        raise TypeError("fit gates must be exact booleans")
    if not predictor_gate:
        raise RuntimeError(STOP_PREDICTOR_GATE)
    if not comparator_gate:
        raise RuntimeError(STOP_COMPARATOR_GATE)
    if not hypothesis_action_gate:
        raise RuntimeError(STOP_HYPOTHESIS_ACTION_GATE)
    if not reset_gate:
        raise RuntimeError(STOP_RESET_GATE)
    if not replay_gate:
        raise RuntimeError(STOP_REPLAY_GATE)
    if not plasticity_gate:
        raise RuntimeError(STOP_PLASTICITY_GATE)
    if not admission_gate:
        raise RuntimeError(STOP_ADMISSION_GATE)
    _validate_inputs(predictor_streams, route_lessons)
    child.authenticate()
    all_streams = predictor_streams + tuple(lesson.stream for lesson in route_lessons)
    predictor = _fit_predictor(
        all_streams, predictor_gate=predictor_gate, plasticity_gate=plasticity_gate
    )
    (
        evidence_sources,
        evidence_targets,
        evidence_roots,
        sources,
        targets,
        support,
        predictor_duplicates,
    ) = predictor
    excitatory = np.zeros((FEATURE_COUNT, ACTION_COUNT), dtype=np.uint8)
    inhibitory = np.zeros((FEATURE_COUNT, ACTION_COUNT), dtype=np.uint8)
    seen_boundary: set[tuple[int, int, bytes]] = set()
    boundary_updates = boundary_unidentifiable = 0
    predictor_steps = comparator_steps = hypothesis_steps = replay_steps = 0
    segments: dict[int, _Segment] = {}
    for ordinal, stream in enumerate(all_streams):
        segment, updates, unresolved = _learn_boundary_and_segment(
            stream,
            sources,
            targets,
            support,
            excitatory,
            inhibitory,
            seen_boundary,
            predictor_gate=predictor_gate,
            comparator_gate=comparator_gate,
            hypothesis_action_gate=hypothesis_action_gate,
            reset_gate=reset_gate,
            replay_gate=replay_gate,
            plasticity_gate=plasticity_gate,
        )
        boundary_updates += updates
        boundary_unidentifiable += unresolved
        predictor_steps += segment.predictor_reads
        comparator_steps += segment.comparator_steps
        hypothesis_steps += segment.hypothesis_steps
        replay_steps += segment.replay_steps
        if ordinal >= len(predictor_streams):
            segments[ordinal - len(predictor_streams)] = segment

    admission_rows: list[tuple[tuple[int, ...], bytes, bytes]] = []
    admission_seen: set[tuple[tuple[int, ...], bytes]] = set()
    admission_duplicates = 0
    collision_certificates: dict[tuple[int, ...], bytes] = {}
    if admission_gate and plasticity_gate:
        for ordinal, lesson in enumerate(route_lessons):
            segment = segments[ordinal]
            if segment.status != RECALLED:
                continue
            provenance = bytes.fromhex(lesson.stream.sha256)
            for trace, chunk_root in zip(
                segment.traces, segment.chunk_roots, strict=True
            ):
                prior = collision_certificates.setdefault(trace, chunk_root)
                if prior != chunk_root:
                    raise RuntimeError(STOP_TRACE_COLLISION)
                key = (trace, provenance)
                if key in admission_seen:
                    admission_duplicates += 1
                    continue
                admission_seen.add(key)
                admission_rows.append((trace, provenance, chunk_root))
    if len(admission_rows) > MAX_ADMISSION_EVIDENCE:
        raise ValueError("chunk admission evidence capacity exceeded")
    admission_supports: dict[tuple[int, ...], set[bytes]] = {}
    chunk_roots: dict[tuple[int, ...], bytes] = {}
    for trace, provenance, chunk_root in admission_rows:
        admission_supports.setdefault(trace, set()).add(provenance)
        prior = chunk_roots.setdefault(trace, chunk_root)
        if prior != chunk_root:
            raise RuntimeError(STOP_TRACE_COLLISION)
    admitted = sorted(
        trace
        for trace, roots in admission_supports.items()
        if len(roots) > ADMISSION_TONIC
    )
    if len(admitted) > MAX_ADMITTED_CHUNKS:
        raise ValueError("admitted chunk capacity exceeded")

    routed_lessons: list[router.PhysicalFactorRouteLesson] = []
    if admission_gate and plasticity_gate:
        admitted_set = set(admitted)
        for ordinal, lesson in enumerate(route_lessons):
            segment = segments[ordinal]
            if segment.status == RECALLED and all(
                trace in admitted_set for trace in segment.traces
            ):
                routed_lessons.append(_router_lesson(lesson, segment.traces))
    if not routed_lessons:
        raise RuntimeError(STOP_CHUNK_UNKNOWN)
    routed = router.fit_common_factorized_context_routed_memory_v2(
        child, tuple(routed_lessons)
    )
    arrays = (
        evidence_sources,
        evidence_targets,
        evidence_roots,
        sources,
        targets,
        support,
        excitatory,
        inhibitory,
        np.asarray([row[0] for row in admission_rows], dtype=np.int32).reshape(
            -1, TRACE_LANES
        ),
        np.asarray([list(row[1]) for row in admission_rows], dtype=np.uint8).reshape(
            -1, 32
        ),
        np.asarray(admitted, dtype=np.int32).reshape(-1, TRACE_LANES),
        np.asarray(
            [min(TICK_CAP, len(admission_supports[trace])) for trace in admitted],
            dtype=np.uint8,
        ),
        np.asarray([list(chunk_roots[trace]) for trace in admitted], dtype=np.uint8).reshape(
            -1, 32
        ),
    )
    array_bytes = sum(array.nbytes for array in arrays)
    summary = CausalChunkTrainingSummary(
        mode="INITIAL_CAUSAL_CHUNK_FIT",
        predictor_stream_count=len(predictor_streams),
        new_route_lesson_count=len(route_lessons),
        total_route_lesson_count=len(route_lessons),
        predictor_evidence_count=evidence_roots.shape[0],
        predictor_duplicate_nonvote_count=predictor_duplicates,
        predictor_association_count=support.shape[0],
        recurrent_association_count=int(np.count_nonzero(support > PREDICTOR_TONIC)),
        boundary_update_count=boundary_updates,
        boundary_unidentifiable_count=boundary_unidentifiable,
        admission_evidence_count=len(admission_rows),
        admission_duplicate_nonvote_count=admission_duplicates,
        admitted_chunk_count=len(admitted),
        trace_collision_count=0,
        routed_lesson_count=len(routed_lessons),
        predictor_cell_steps=predictor_steps,
        comparator_cell_steps=comparator_steps,
        hypothesis_cell_steps=hypothesis_steps,
        replay_cell_steps=replay_steps,
        admission_cell_steps=len(admission_rows) * (TRACE_LANES + 1),
        route_learning_cell_steps=routed.training_summary.router_learning_cell_steps,
        fixed_chunker_cell_count=FIXED_CHUNKER_CELL_COUNT,
        predictor_match_cell_count=support.shape[0],
        admission_match_cell_count=len(admitted),
        added_chunker_cell_count=FIXED_CHUNKER_CELL_COUNT + support.shape[0] + len(admitted),
        chunker_checkpoint_array_bytes=array_bytes,
        router_checkpoint_array_bytes=routed.training_summary.checkpoint_array_bytes,
        total_checkpoint_array_bytes=array_bytes
        + routed.training_summary.checkpoint_array_bytes,
        raw_stream_retention_count=0,
        caller_span_retention_count=0,
        parent_lesson_replay_count=0,
        memory_checkpoint_sha256=routed.memory_checkpoint_sha256,
        episodic_checkpoint_sha256=routed.episodic_checkpoint_sha256,
        semantic_checkpoint_sha256=routed.semantic_checkpoint_sha256,
        router_checkpoint_sha256=routed.checkpoint_sha256,
        checkpoint_sha256="",
    )
    named = tuple(
        (name, _freeze(array, array.dtype))
        for name, array in zip(
            (
                "predictor-evidence-sources",
                "predictor-evidence-targets",
                "predictor-evidence-roots",
                "predictor-sources",
                "predictor-targets",
                "predictor-support",
                "boundary-excitatory",
                "boundary-inhibitory",
                "admission-evidence-traces",
                "admission-evidence-roots",
                "admitted-traces",
                "admission-support",
                "admission-chunk-roots",
            ),
            arrays,
            strict=True,
        )
    )
    summary = dataclasses.replace(
        summary, checkpoint_sha256=_checkpoint_sha256(routed, summary, named)
    )
    return FrozenCommonCausalPredictiveByteChunkerV1(
        routed=routed, arrays=arrays, summary=summary
    )


def advance_common_causal_predictive_byte_chunker_v1(
    parent: FrozenCommonCausalPredictiveByteChunkerV1,
    new_route_lessons: tuple[PhysicalContinuousRouteLesson, ...],
) -> FrozenCommonCausalPredictiveByteChunkerV1:
    """Revise only route evidence; predictor, boundaries, and chunks stay frozen."""

    if type(parent) is not FrozenCommonCausalPredictiveByteChunkerV1:
        raise TypeError("parent must be one exact frozen causal chunker")
    if type(new_route_lessons) is not tuple or any(
        type(lesson) is not PhysicalContinuousRouteLesson for lesson in new_route_lessons
    ):
        raise TypeError("new route lessons must be one exact tuple")
    parent.authenticate()
    routed_lessons: list[router.PhysicalFactorRouteLesson] = []
    for lesson in new_route_lessons:
        segment = _segment_recall(
            lesson.stream,
            parent._predictor_sources,
            parent._predictor_targets,
            parent._predictor_support,
            parent._boundary_excitatory,
            parent._boundary_inhibitory,
            predictor_gate=True,
            comparator_gate=True,
            hypothesis_action_gate=True,
            reset_gate=True,
            replay_gate=True,
        )
        if segment.status != RECALLED:
            raise ValueError(f"new route lesson did not causally segment: {segment.status}")
        match_status, admitted, _ = parent._match_admissions(
            segment.traces, segment.chunk_roots, admission_gate=True
        )
        if match_status != RECALLED:
            raise ValueError(f"new route lesson used an unadmitted chunk: {match_status}")
        routed_lessons.append(_router_lesson(lesson, admitted))
    routed = router.advance_common_factorized_context_routed_memory_v2(
        parent._router, tuple(routed_lessons)
    )
    arrays = tuple(array for _, array in parent._checkpoint_arrays())
    array_bytes = sum(array.nbytes for array in arrays)
    summary = dataclasses.replace(
        parent.training_summary,
        mode="REPLAY_FREE_ROUTE_REVISION",
        new_route_lesson_count=len(new_route_lessons),
        total_route_lesson_count=parent.training_summary.total_route_lesson_count
        + len(new_route_lessons),
        routed_lesson_count=parent.training_summary.routed_lesson_count
        + len(routed_lessons),
        route_learning_cell_steps=routed.training_summary.router_learning_cell_steps,
        router_checkpoint_array_bytes=routed.training_summary.checkpoint_array_bytes,
        total_checkpoint_array_bytes=array_bytes
        + routed.training_summary.checkpoint_array_bytes,
        router_checkpoint_sha256=routed.checkpoint_sha256,
        checkpoint_sha256="",
    )
    named = tuple((name, array) for name, array in parent._checkpoint_arrays())
    summary = dataclasses.replace(
        summary, checkpoint_sha256=_checkpoint_sha256(routed, summary, named)
    )
    successor = FrozenCommonCausalPredictiveByteChunkerV1(
        routed=routed, arrays=arrays, summary=summary
    )
    parent.authenticate()
    return successor


__all__ = [
    "ACTION_CELLS",
    "CONFIG_SHA256",
    "CONTINUE",
    "CUT_INHIBITORY_CELL",
    "CUT_REPLAY",
    "FEATURE_CELLS",
    "FIXED_CHUNKER_CELL_COUNT",
    "INPUT_CELL_OFFSET",
    "INPUT_LANE_BANK_CELLS",
    "PROVENANCE_CELL_OFFSET",
    "RECALLED",
    "STOP_ADMISSION_GATE",
    "STOP_BOUNDARY_AMBIGUOUS",
    "STOP_BOUNDARY_UNIDENTIFIABLE",
    "STOP_CHUNK_AMBIGUOUS",
    "STOP_CHUNK_UNKNOWN",
    "STOP_COMPARATOR_GATE",
    "STOP_HYPOTHESIS_ACTION_GATE",
    "STOP_PLASTICITY_GATE",
    "STOP_PREDICTOR_AMBIGUOUS",
    "STOP_PREDICTOR_GATE",
    "STOP_PREDICTOR_UNKNOWN",
    "STOP_REPLAY_GATE",
    "STOP_RESET_GATE",
    "STOP_TRACE_COLLISION",
    "STREAM_BEGIN_CELLS",
    "STREAM_END_CELLS",
    "PhysicalContinuousByteStream",
    "PhysicalContinuousRouteLesson",
    "advance_common_causal_predictive_byte_chunker_v1",
    "fit_common_causal_predictive_byte_chunker_v1",
    "seal_physical_continuous_byte_stream",
]
