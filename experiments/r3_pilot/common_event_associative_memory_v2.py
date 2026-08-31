"""Replay-free exact-event extension over one frozen common V1 memory.

The V1 checkpoint remains the immutable legacy bank.  New physical lessons are
stored in one cumulative content-addressed matrix: twelve projected cue-cell
addresses, four authenticated output-cell addresses, and one 32-byte
provenance root per episode.  Recall activates every delta match cell, compares
its equality-arrival latency with one tonic cell, and binary-ORs all exact delta
spikes with the public active output cells of the legacy checkpoint before one
final ``CommonEventCells.race_output``.

The implementation never reads V1 synapses or metadata, never decodes a child
token, and never chooses a shard or winning episode on the host.  Contradictory
exact episodes therefore reach the final physical race together and STOP.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as projection
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCOPE = "COMMON_EVENT_ASSOCIATIVE_MEMORY_V2"
STATUS = "SELF_POSED_REPLAY_FREE_EXACT_EPISODIC_EXTENSION"
CLAIM_BOUNDARY = (
    "self-posed-exact-opaque-event-association-extension-only-no-semantic-"
    "compression-language-coding-corpus-or-protected-transfer-evidence"
)

TRACE_LANES = projection.TRACE_LANES
TRACE_CELLS_PER_LANE = projection.FEATURE_BANK_CELLS
TRACE_CELL_COUNT = TRACE_LANES * TRACE_CELLS_PER_LANE
TRACE_CELL_OFFSET = associative.TOTAL_CELL_COUNT
BINARY_OR_RELAY_CELL_COUNT = projection.OUTPUT_DIGIT_CELL_COUNT
BINARY_OR_RELAY_CELL_OFFSET = TRACE_CELL_OFFSET + TRACE_CELL_COUNT
MATCH_TONIC_CELL_ID = BINARY_OR_RELAY_CELL_OFFSET + BINARY_OR_RELAY_CELL_COUNT
MATCH_CELL_OFFSET = MATCH_TONIC_CELL_ID + 1

CUE_DTYPE = np.dtype(np.int32)
OUTPUT_DTYPE = np.dtype(np.int32)
PROVENANCE_DTYPE = np.dtype(np.uint8)
CUE_ARRAY_BYTES_PER_EPISODE = TRACE_LANES * CUE_DTYPE.itemsize
OUTPUT_ARRAY_BYTES_PER_EPISODE = (
    projection.OUTPUT_DIGIT_LANES * OUTPUT_DTYPE.itemsize
)
PROVENANCE_ARRAY_BYTES_PER_EPISODE = 32
DELTA_ARRAY_BYTES_PER_EPISODE = (
    CUE_ARRAY_BYTES_PER_EPISODE
    + OUTPUT_ARRAY_BYTES_PER_EPISODE
    + PROVENANCE_ARRAY_BYTES_PER_EPISODE
)
INHERITED_V1_CHECKPOINT_ARRAY_BYTES = associative.CHECKPOINT_ARRAY_BYTES

MATCH_SCORE = TRACE_LANES
MATCH_TONIC_SCORE = MATCH_SCORE - 1
RACE_STEPS = 600
ZERO_SCORE_SPIKE_STEP = 568
MATCH_TONIC_SPIKE_STEP = ZERO_SCORE_SPIKE_STEP - MATCH_TONIC_SCORE
RELAY_ACTIVE_SCORE = 2 * projection.STOP_SCORE

GENERATOR_EXECUTION_COUNT = 0
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_EVALUATION_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_TRACE_MULTIPLIERS = (389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457)
_TRACE_POSITION_GAINS = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)
_TRACE_BIASES = (23, 53, 83, 113, 151, 191, 227, 269, 311, 359, 409, 461)

_SOURCE_PATH = Path(__file__).resolve()
_TRACE_NAMESPACE = b"common-event-associative-memory-v2-trace\0"
_PROVENANCE_NAMESPACE = b"common-event-associative-memory-v2-provenance\0"
_CHECKPOINT_NAMESPACE = b"common-event-associative-memory-v2-checkpoint\0"
_RECALL_NAMESPACE = b"common-event-associative-memory-v2-recall\0"
_CONFIG_NAMESPACE = b"common-event-associative-memory-v2-config\0"
_CONSTRUCTION_TOKEN = object()


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _array_frame(name: str, array: np.ndarray) -> bytes:
    header = _canonical_json(
        {
            "name": name,
            "dtype": array.dtype.str,
            "shape": tuple(int(value) for value in array.shape),
        }
    )
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return (
        len(header).to_bytes(8, "little")
        + header
        + len(payload).to_bytes(8, "little")
        + payload
    )


def _checkpoint_payload(
    metadata: bytes,
    base_checkpoint_sha256: str,
    cue_cells: np.ndarray,
    output_cells: np.ndarray,
    provenance_roots: np.ndarray,
) -> bytes:
    return (
        _CHECKPOINT_NAMESPACE
        + len(metadata).to_bytes(8, "little")
        + metadata
        + bytes.fromhex(base_checkpoint_sha256)
        + _array_frame("delta-cue-cells", cue_cells)
        + _array_frame("delta-output-cells", output_cells)
        + _array_frame("delta-provenance-roots", provenance_roots)
    )


def _validate_observation_events(events: tuple[bytes, ...]) -> None:
    if (
        type(events) is not tuple
        or not 1 <= len(events) <= associative.MAX_OBSERVATION_EVENTS
        or any(
            type(event) is not bytes
            or not 1 <= len(event) <= projection.MAX_EVENT_BYTES
            for event in events
        )
    ):
        raise ValueError("observations must be one bounded tuple of opaque byte events")


def _validate_output_cells(output_cells: tuple[int, ...]) -> None:
    if (
        type(output_cells) is not tuple
        or len(output_cells) != projection.OUTPUT_DIGIT_LANES
        or any(type(cell) is not int for cell in output_cells)
    ):
        raise ValueError("teacher_output_cells must be four exact physical cell IDs")
    for lane, cell in enumerate(output_cells):
        lane_start = (
            projection.OUTPUT_CELL_OFFSET + lane * projection.OUTPUT_DIGIT_RADIX
        )
        if not lane_start <= cell < lane_start + projection.OUTPUT_DIGIT_RADIX:
            raise ValueError("teacher output cells must be in authenticated lane order")


def _build_match_current_table() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(MATCH_SCORE + 1):
        first_step = ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("V2 match-cell fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        row.spike_step != ZERO_SCORE_SPIKE_STEP - score
        for score, row in enumerate(controls)
    ):
        raise RuntimeError("V2 match-cell fitted-GLIF calibration changed")
    return table, race


_MATCH_CURRENT_TABLE, _MATCH_RACE = _build_match_current_table()


@dataclass(frozen=True, slots=True)
class PhysicalCommonAssociationLessonV2:
    observation_events: tuple[bytes, ...]
    teacher_output_cells: tuple[int, ...]
    teacher_gate: bool = True
    cell_gate: bool = True

    def __post_init__(self) -> None:
        _validate_observation_events(self.observation_events)
        _validate_output_cells(self.teacher_output_cells)
        if type(self.teacher_gate) is not bool or type(self.cell_gate) is not bool:
            raise ValueError("teacher_gate and cell_gate must be exact booleans")


@dataclass(frozen=True, slots=True)
class CommonAssociationProvenanceV2:
    match_cell_id: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CommonAssociationTrainingSummaryV2:
    mode: str
    parent_checkpoint_sha256: str | None
    base_checkpoint_sha256: str
    legacy_base_count: int
    delta_matrix_count: int
    retained_v2_parent_count: int
    parent_lesson_replay_count: int
    delta_episode_count: int
    delta_match_cell_count: int
    fixed_trace_cell_count: int
    fixed_binary_or_relay_cell_count: int
    fixed_match_tonic_cell_count: int
    total_v2_cell_count: int
    cue_array_shape: tuple[int, int]
    output_array_shape: tuple[int, int]
    provenance_array_shape: tuple[int, int]
    cue_array_bytes: int
    output_array_bytes: int
    provenance_array_bytes: int
    delta_checkpoint_array_bytes: int
    inherited_v1_checkpoint_array_bytes: int
    canonical_metadata_bytes: int
    canonical_metadata_sha256: str
    checkpoint_sha256: str

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class CommonAssociationTransitionV2:
    status: str
    accepted: bool
    parent_checkpoint_sha256: str
    successor_checkpoint_sha256: str
    delta_episode_count_before: int
    delta_episode_count_after: int
    cue_addresses_added: int
    output_addresses_added: int
    provenance_bytes_added: int
    match_cells_added: int
    parent_lesson_replay_count: int
    successor: FrozenCommonEventAssociativeMemoryV2

    def record(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "successor"
        }


@dataclass(frozen=True, slots=True)
class CommonAssociationRecallWorkV2:
    projection_work: int
    second_order_trace_transitions: int
    base_recall_work: int
    match_synaptic_equality_reads: int
    match_equality_arrivals: int
    match_and_tonic_cell_steps: int
    binary_or_synaptic_equality_reads: int
    binary_or_relay_cell_steps: int
    output_cell_steps: int

    @property
    def total(self) -> int:
        return sum(dataclasses.asdict(self).values())

    def record(self) -> dict[str, int]:
        return {**dataclasses.asdict(self), "total": self.total}


@dataclass(frozen=True, slots=True)
class CommonAssociationRecallV2:
    status: str
    output: projection.CommonOutputDecision
    observation_event_count: int
    event_frame_sha256s: tuple[str, ...]
    cue_cells: tuple[int, ...]
    base_active_output_cells: tuple[int, ...]
    delta_match_arrival_counts: tuple[int, ...]
    delta_match_spike_steps: tuple[int, ...]
    delta_match_spikes: tuple[bool, ...]
    binary_or_relay_spikes: tuple[bool, ...]
    base_checkpoint_sha256: str
    checkpoint_sha256: str
    checkpoint_authentication_count: int
    work: CommonAssociationRecallWorkV2
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "output": self.output.record(),
            "observation_event_count": self.observation_event_count,
            "event_frame_sha256s": self.event_frame_sha256s,
            "cue_cells": self.cue_cells,
            "base_active_output_cells": self.base_active_output_cells,
            "delta_match_arrival_counts": self.delta_match_arrival_counts,
            "delta_match_spike_steps": self.delta_match_spike_steps,
            "delta_match_spikes": self.delta_match_spikes,
            "binary_or_relay_spikes": self.binary_or_relay_spikes,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_authentication_count": self.checkpoint_authentication_count,
            "work": self.work.record(),
            "raw_observation_bytes_retained": False,
            "child_token_or_status_used_for_selection": False,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class _ProjectedCueV2:
    cue_cells: tuple[int, ...]
    event_frame_sha256s: tuple[str, ...]
    projection_work: int
    second_order_trace_transitions: int


def _project_cue(
    cells: projection.CommonEventCells,
    observation_events: tuple[bytes, ...],
) -> _ProjectedCueV2:
    _validate_observation_events(observation_events)
    frames = tuple(cells.project(event) for event in observation_events)
    state = tuple(
        (lane * 47 + 17) % TRACE_CELLS_PER_LANE for lane in range(TRACE_LANES)
    )
    for ordinal, frame in enumerate(frames):
        state = tuple(
            (
                state[lane] * _TRACE_MULTIPLIERS[lane]
                + frame.trace_cells[lane]
                + (ordinal + 1) * _TRACE_POSITION_GAINS[lane]
                + _TRACE_BIASES[lane]
            )
            % TRACE_CELLS_PER_LANE
            for lane in range(TRACE_LANES)
        )
    cue_cells = tuple(
        TRACE_CELL_OFFSET + lane * TRACE_CELLS_PER_LANE + state[lane]
        for lane in range(TRACE_LANES)
    )
    return _ProjectedCueV2(
        cue_cells=cue_cells,
        event_frame_sha256s=tuple(frame.sha256 for frame in frames),
        projection_work=sum(frame.work.total for frame in frames),
        second_order_trace_transitions=len(frames) * TRACE_LANES,
    )


def _authenticate_teacher_output(
    cells: projection.CommonEventCells,
    output_cells: tuple[int, ...],
) -> None:
    _validate_output_cells(output_cells)
    rows = [
        [0 for _digit in range(projection.OUTPUT_DIGIT_RADIX)]
        for _lane in range(projection.OUTPUT_DIGIT_LANES)
    ]
    for lane, cell in enumerate(output_cells):
        lane_start = (
            projection.OUTPUT_CELL_OFFSET + lane * projection.OUTPUT_DIGIT_RADIX
        )
        rows[lane][cell - lane_start] = RELAY_ACTIVE_SCORE
    authenticated = cells.race_output(tuple(tuple(row) for row in rows))
    if authenticated.active_cells != output_cells:
        raise ValueError("teacher output cells failed the common physical race")


def _episode_provenance_root(
    base_checkpoint_sha256: str,
    projected: _ProjectedCueV2,
    output_cells: tuple[int, ...],
) -> bytes:
    payload = (
        _PROVENANCE_NAMESPACE
        + bytes.fromhex(base_checkpoint_sha256)
        + len(projected.event_frame_sha256s).to_bytes(2, "little")
        + b"".join(bytes.fromhex(root) for root in projected.event_frame_sha256s)
        + np.asarray(projected.cue_cells, dtype=CUE_DTYPE).tobytes(order="C")
        + np.asarray(output_cells, dtype=OUTPUT_DTYPE).tobytes(order="C")
    )
    return sha256(payload).digest()


def _validate_delta_arrays(
    cue_cells: np.ndarray,
    output_cells: np.ndarray,
    provenance_roots: np.ndarray,
) -> None:
    episode_count = cue_cells.shape[0] if cue_cells.ndim == 2 else -1
    expectations = (
        (cue_cells, (episode_count, TRACE_LANES), CUE_DTYPE),
        (
            output_cells,
            (episode_count, projection.OUTPUT_DIGIT_LANES),
            OUTPUT_DTYPE,
        ),
        (provenance_roots, (episode_count, 32), PROVENANCE_DTYPE),
    )
    if episode_count < 0 or any(
        array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous
        for array, shape, dtype in expectations
    ):
        raise ValueError("V2 cumulative delta matrix geometry changed")

    if episode_count:
        cue_lane_starts = TRACE_CELL_OFFSET + np.arange(
            TRACE_LANES, dtype=np.int64
        ) * TRACE_CELLS_PER_LANE
        if np.any(cue_cells < cue_lane_starts) or np.any(
            cue_cells >= cue_lane_starts + TRACE_CELLS_PER_LANE
        ):
            raise ValueError("V2 cue cells escaped their physical trace lanes")
        output_lane_starts = projection.OUTPUT_CELL_OFFSET + np.arange(
            projection.OUTPUT_DIGIT_LANES, dtype=np.int64
        ) * projection.OUTPUT_DIGIT_RADIX
        if np.any(output_cells < output_lane_starts) or np.any(
            output_cells >= output_lane_starts + projection.OUTPUT_DIGIT_RADIX
        ):
            raise ValueError("V2 output cells escaped authenticated lane order")


class FrozenCommonEventAssociativeMemoryV2:
    """One immutable V1 base plus one cumulative append-only delta matrix."""

    __slots__ = (
        "_base",
        "_cells",
        "_checkpoint_sha256",
        "_cue_cells",
        "_metadata",
        "_output_cells",
        "_provenance_roots",
        "_training_summary",
    )

    def __init__(
        self,
        *,
        base: associative.FrozenCommonEventAssociativeMemory,
        cue_cells: np.ndarray,
        output_cells: np.ndarray,
        provenance_roots: np.ndarray,
        metadata: bytes,
        training_summary: CommonAssociationTrainingSummaryV2,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("frozen V2 common memories are builder-owned")
        if type(base) is not associative.FrozenCommonEventAssociativeMemory:
            raise TypeError("base must be one exact frozen V1 common memory")
        _validate_delta_arrays(cue_cells, output_cells, provenance_roots)
        base.authenticate()
        self._base = base
        self._cells = projection.CommonEventCells()
        self._cue_cells = np.array(cue_cells, copy=True, order="C")
        self._output_cells = np.array(output_cells, copy=True, order="C")
        self._provenance_roots = np.array(provenance_roots, copy=True, order="C")
        for array in (
            self._cue_cells,
            self._output_cells,
            self._provenance_roots,
        ):
            array.setflags(write=False)
        self._metadata = bytes(metadata)
        self._training_summary = training_summary
        self._checkpoint_sha256 = sha256(
            _checkpoint_payload(
                self._metadata,
                self._base.checkpoint_sha256,
                self._cue_cells,
                self._output_cells,
                self._provenance_roots,
            )
        ).hexdigest()
        self._validate_summary()
        self.authenticate()

    def _validate_summary(self) -> None:
        summary = self._training_summary
        episode_count = self._cue_cells.shape[0]
        if (
            summary.base_checkpoint_sha256 != self._base.checkpoint_sha256
            or summary.delta_episode_count != episode_count
            or summary.delta_match_cell_count != episode_count
            or summary.cue_array_shape != self._cue_cells.shape
            or summary.output_array_shape != self._output_cells.shape
            or summary.provenance_array_shape != self._provenance_roots.shape
            or summary.cue_array_bytes != self._cue_cells.nbytes
            or summary.output_array_bytes != self._output_cells.nbytes
            or summary.provenance_array_bytes != self._provenance_roots.nbytes
            or summary.delta_checkpoint_array_bytes
            != self._cue_cells.nbytes
            + self._output_cells.nbytes
            + self._provenance_roots.nbytes
            or summary.canonical_metadata_bytes != len(self._metadata)
            or summary.canonical_metadata_sha256 != sha256(self._metadata).hexdigest()
            or summary.checkpoint_sha256 != self._checkpoint_sha256
        ):
            raise RuntimeError("V2 common-memory summary identity changed")

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def base_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._base.checkpoint_sha256

    @property
    def training_summary(self) -> CommonAssociationTrainingSummaryV2:
        self.authenticate()
        return self._training_summary

    @property
    def provenance(self) -> tuple[CommonAssociationProvenanceV2, ...]:
        self.authenticate()
        return tuple(
            CommonAssociationProvenanceV2(
                match_cell_id=MATCH_CELL_OFFSET + ordinal,
                sha256=bytes(root).hex(),
            )
            for ordinal, root in enumerate(self._provenance_roots)
        )

    @property
    def learned_state_shapes(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        self.authenticate()
        return (
            self._cue_cells.shape,
            self._output_cells.shape,
            self._provenance_roots.shape,
        )

    def authenticate(self) -> None:
        self._base.authenticate()
        _validate_delta_arrays(
            self._cue_cells,
            self._output_cells,
            self._provenance_roots,
        )
        observed = sha256(
            _checkpoint_payload(
                self._metadata,
                self._base.checkpoint_sha256,
                self._cue_cells,
                self._output_cells,
                self._provenance_roots,
            )
        ).hexdigest()
        if observed != self._checkpoint_sha256:
            raise RuntimeError("V2 common-memory checkpoint authentication failed")
        self._validate_summary()

    def recall(
        self,
        observation_events: tuple[bytes, ...],
        *,
        cell_gate: bool = True,
    ) -> CommonAssociationRecallV2:
        self.authenticate()
        _validate_observation_events(observation_events)
        if type(cell_gate) is not bool:
            raise ValueError("cell_gate must be an exact boolean")

        if cell_gate:
            projected = _project_cue(self._cells, observation_events)
            base_recall = self._base.recall(observation_events)
            base_active_output_cells = base_recall.output.active_cells
            base_recall_work = base_recall.work.total
            if self._cue_cells.shape[0]:
                active = np.asarray(projected.cue_cells, dtype=CUE_DTYPE)
                arrivals = np.count_nonzero(
                    self._cue_cells == active,
                    axis=1,
                ).astype(np.int16)
                races = _MATCH_RACE.race_many(
                    _MATCH_CURRENT_TABLE[arrivals][:, None],
                    require_unique=False,
                )
                spike_steps = np.fromiter(
                    (race.spike_step for race in races),
                    dtype=np.int16,
                    count=len(races),
                )
                match_spikes = spike_steps < MATCH_TONIC_SPIKE_STEP
            else:
                arrivals = np.empty((0,), dtype=np.int16)
                spike_steps = np.empty((0,), dtype=np.int16)
                match_spikes = np.empty((0,), dtype=np.bool_)
        else:
            projected = _ProjectedCueV2((), (), 0, 0)
            base_active_output_cells = ()
            base_recall_work = 0
            arrivals = np.empty((0,), dtype=np.int16)
            spike_steps = np.empty((0,), dtype=np.int16)
            match_spikes = np.empty((0,), dtype=np.bool_)

        candidate_cells = np.arange(
            projection.OUTPUT_CELL_OFFSET,
            projection.OUTPUT_CELL_OFFSET + projection.OUTPUT_DIGIT_CELL_COUNT,
            dtype=OUTPUT_DTYPE,
        )
        base_active = np.asarray(base_active_output_cells, dtype=OUTPUT_DTYPE)
        if base_active.size:
            base_relay_spikes = np.any(
                base_active[:, None] == candidate_cells[None, :],
                axis=0,
            )
        else:
            base_relay_spikes = np.zeros(
                (projection.OUTPUT_DIGIT_CELL_COUNT,), dtype=np.bool_
            )
        if match_spikes.size:
            delta_relay_spikes = np.any(
                match_spikes[:, None, None]
                & (
                    self._output_cells[:, :, None]
                    == candidate_cells[None, None, :]
                ),
                axis=(0, 1),
            )
        else:
            delta_relay_spikes = np.zeros(
                (projection.OUTPUT_DIGIT_CELL_COUNT,), dtype=np.bool_
            )
        relay_spikes = base_relay_spikes | delta_relay_spikes
        raw = np.where(relay_spikes, RELAY_ACTIVE_SCORE, 0).reshape(
            projection.OUTPUT_DIGIT_LANES,
            projection.OUTPUT_DIGIT_RADIX,
        )
        output = self._cells.race_output(
            tuple(tuple(int(value) for value in row) for row in raw)
        )
        if not cell_gate:
            recall_status = "STOP_CELL_OFF"
        elif output.decision == "TOKEN":
            recall_status = "RECALLED"
        elif output.stop_reason == "COLLISION":
            recall_status = "STOP_AMBIGUOUS_COLLISION"
        else:
            recall_status = "STOP_UNKNOWN"

        episode_count = self._cue_cells.shape[0] if cell_gate else 0
        work = CommonAssociationRecallWorkV2(
            projection_work=projected.projection_work,
            second_order_trace_transitions=projected.second_order_trace_transitions,
            base_recall_work=base_recall_work,
            match_synaptic_equality_reads=episode_count * TRACE_LANES,
            match_equality_arrivals=int(np.sum(arrivals, dtype=np.int64)),
            match_and_tonic_cell_steps=(episode_count + int(cell_gate)) * RACE_STEPS,
            binary_or_synaptic_equality_reads=(
                projection.OUTPUT_DIGIT_CELL_COUNT
                * (
                    len(base_active_output_cells)
                    + episode_count * projection.OUTPUT_DIGIT_LANES
                )
            ),
            binary_or_relay_cell_steps=(
                BINARY_OR_RELAY_CELL_COUNT if cell_gate else 0
            ),
            output_cell_steps=output.physical_cell_steps,
        )
        record = {
            "status": recall_status,
            "output": output.record(),
            "observation_event_count": len(observation_events),
            "event_frame_sha256s": projected.event_frame_sha256s,
            "cue_cells": projected.cue_cells,
            "base_active_output_cells": base_active_output_cells,
            "delta_match_arrival_counts": tuple(int(value) for value in arrivals),
            "delta_match_spike_steps": tuple(int(value) for value in spike_steps),
            "delta_match_spikes": tuple(bool(value) for value in match_spikes),
            "binary_or_relay_spikes": tuple(bool(value) for value in relay_spikes),
            "base_checkpoint_sha256": self._base.checkpoint_sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_authentication_count": 1,
            "work": work.record(),
        }
        return CommonAssociationRecallV2(
            status=recall_status,
            output=output,
            observation_event_count=len(observation_events),
            event_frame_sha256s=projected.event_frame_sha256s,
            cue_cells=projected.cue_cells,
            base_active_output_cells=base_active_output_cells,
            delta_match_arrival_counts=tuple(int(value) for value in arrivals),
            delta_match_spike_steps=tuple(int(value) for value in spike_steps),
            delta_match_spikes=tuple(bool(value) for value in match_spikes),
            binary_or_relay_spikes=tuple(bool(value) for value in relay_spikes),
            base_checkpoint_sha256=self._base.checkpoint_sha256,
            checkpoint_sha256=self._checkpoint_sha256,
            checkpoint_authentication_count=1,
            work=work,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )


def _freeze(
    *,
    base: associative.FrozenCommonEventAssociativeMemory,
    cue_cells: np.ndarray,
    output_cells: np.ndarray,
    provenance_roots: np.ndarray,
    mode: str,
    parent_checkpoint_sha256: str | None,
) -> FrozenCommonEventAssociativeMemoryV2:
    _validate_delta_arrays(cue_cells, output_cells, provenance_roots)
    base.authenticate()
    episode_count = cue_cells.shape[0]
    metadata_record = {
        "schema": "common-event-associative-memory-v2-checkpoint",
        "scope": SCOPE,
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "config_sha256": CONFIG_SHA256,
        "mode": mode,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "base_checkpoint_sha256": base.checkpoint_sha256,
        "legacy_base_count": 1,
        "delta_matrix_count": 1,
        "retained_v2_parent_count": 0,
        "parent_lesson_replay_count": 0,
        "delta_episode_count": episode_count,
        "cue_array_shape": cue_cells.shape,
        "output_array_shape": output_cells.shape,
        "provenance_array_shape": provenance_roots.shape,
        "cue_array_bytes": cue_cells.nbytes,
        "output_array_bytes": output_cells.nbytes,
        "provenance_array_bytes": provenance_roots.nbytes,
        "delta_checkpoint_array_bytes": (
            cue_cells.nbytes + output_cells.nbytes + provenance_roots.nbytes
        ),
        "inherited_v1_checkpoint_array_bytes": INHERITED_V1_CHECKPOINT_ARRAY_BYTES,
        "raw_event_retention": False,
        "child_token_or_status_used_for_selection": False,
        "generator_corpus_path2_dev_m0_protected_g0_eval_coding_inputs": (
            GENERATOR_EXECUTION_COUNT,
            CORPUS_INPUT_COUNT,
            PATH2_INPUT_COUNT,
            DEV_INPUT_COUNT,
            M0_INPUT_COUNT,
            PROTECTED_INPUT_COUNT,
            G0_EVALUATION_INPUT_COUNT,
            CODING_INPUT_COUNT,
        ),
    }
    metadata = _canonical_json(metadata_record)
    checkpoint_sha256 = sha256(
        _checkpoint_payload(
            metadata,
            base.checkpoint_sha256,
            cue_cells,
            output_cells,
            provenance_roots,
        )
    ).hexdigest()
    summary = CommonAssociationTrainingSummaryV2(
        mode=mode,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        base_checkpoint_sha256=base.checkpoint_sha256,
        legacy_base_count=1,
        delta_matrix_count=1,
        retained_v2_parent_count=0,
        parent_lesson_replay_count=0,
        delta_episode_count=episode_count,
        delta_match_cell_count=episode_count,
        fixed_trace_cell_count=TRACE_CELL_COUNT,
        fixed_binary_or_relay_cell_count=BINARY_OR_RELAY_CELL_COUNT,
        fixed_match_tonic_cell_count=1,
        total_v2_cell_count=MATCH_CELL_OFFSET + episode_count,
        cue_array_shape=cue_cells.shape,
        output_array_shape=output_cells.shape,
        provenance_array_shape=provenance_roots.shape,
        cue_array_bytes=cue_cells.nbytes,
        output_array_bytes=output_cells.nbytes,
        provenance_array_bytes=provenance_roots.nbytes,
        delta_checkpoint_array_bytes=(
            cue_cells.nbytes + output_cells.nbytes + provenance_roots.nbytes
        ),
        inherited_v1_checkpoint_array_bytes=INHERITED_V1_CHECKPOINT_ARRAY_BYTES,
        canonical_metadata_bytes=len(metadata),
        canonical_metadata_sha256=sha256(metadata).hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonEventAssociativeMemoryV2(
        base=base,
        cue_cells=cue_cells,
        output_cells=output_cells,
        provenance_roots=provenance_roots,
        metadata=metadata,
        training_summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def upgrade_common_event_associative_memory_v2(
    base_v1: associative.FrozenCommonEventAssociativeMemory,
) -> FrozenCommonEventAssociativeMemoryV2:
    """Authenticate one V1 checkpoint and attach an empty cumulative delta."""

    if type(base_v1) is not associative.FrozenCommonEventAssociativeMemory:
        raise TypeError("base_v1 must be one exact frozen V1 common memory")
    base_v1.authenticate()
    return _freeze(
        base=base_v1,
        cue_cells=np.empty((0, TRACE_LANES), dtype=CUE_DTYPE),
        output_cells=np.empty(
            (0, projection.OUTPUT_DIGIT_LANES), dtype=OUTPUT_DTYPE
        ),
        provenance_roots=np.empty((0, 32), dtype=PROVENANCE_DTYPE),
        mode="UPGRADE_ONE_LEGACY_V1_BASE",
        parent_checkpoint_sha256=None,
    )


def advance_common_event_association_v2(
    state: FrozenCommonEventAssociativeMemoryV2,
    lesson: PhysicalCommonAssociationLessonV2,
) -> CommonAssociationTransitionV2:
    """Append one physical exact episode without replaying any prior lesson."""

    if type(state) is not FrozenCommonEventAssociativeMemoryV2:
        raise TypeError("state must be one exact frozen V2 common memory")
    if type(lesson) is not PhysicalCommonAssociationLessonV2:
        raise TypeError("lesson must be one exact physical V2 lesson")
    state.authenticate()
    parent_root = state.checkpoint_sha256
    before = state.training_summary.delta_episode_count

    if not lesson.teacher_gate or not lesson.cell_gate:
        status = "STOP_TEACHER_OFF" if not lesson.teacher_gate else "STOP_CELL_OFF"
        return CommonAssociationTransitionV2(
            status=status,
            accepted=False,
            parent_checkpoint_sha256=parent_root,
            successor_checkpoint_sha256=parent_root,
            delta_episode_count_before=before,
            delta_episode_count_after=before,
            cue_addresses_added=0,
            output_addresses_added=0,
            provenance_bytes_added=0,
            match_cells_added=0,
            parent_lesson_replay_count=0,
            successor=state,
        )

    projected = _project_cue(state._cells, lesson.observation_events)
    _authenticate_teacher_output(state._cells, lesson.teacher_output_cells)
    provenance_root = _episode_provenance_root(
        state._base.checkpoint_sha256,
        projected,
        lesson.teacher_output_cells,
    )
    cue_cells = np.concatenate(
        (
            state._cue_cells,
            np.asarray(projected.cue_cells, dtype=CUE_DTYPE)[None, :],
        ),
        axis=0,
    )
    output_cells = np.concatenate(
        (
            state._output_cells,
            np.asarray(lesson.teacher_output_cells, dtype=OUTPUT_DTYPE)[None, :],
        ),
        axis=0,
    )
    provenance_roots = np.concatenate(
        (
            state._provenance_roots,
            np.frombuffer(provenance_root, dtype=PROVENANCE_DTYPE)[None, :],
        ),
        axis=0,
    )
    successor = _freeze(
        base=state._base,
        cue_cells=cue_cells,
        output_cells=output_cells,
        provenance_roots=provenance_roots,
        mode="APPEND_ONE_REPLAY_FREE_DELTA_EPISODE",
        parent_checkpoint_sha256=parent_root,
    )
    state.authenticate()
    if state.checkpoint_sha256 != parent_root:
        raise RuntimeError("V2 parent checkpoint changed during append")
    return CommonAssociationTransitionV2(
        status="APPENDED",
        accepted=True,
        parent_checkpoint_sha256=parent_root,
        successor_checkpoint_sha256=successor.checkpoint_sha256,
        delta_episode_count_before=before,
        delta_episode_count_after=before + 1,
        cue_addresses_added=TRACE_LANES,
        output_addresses_added=projection.OUTPUT_DIGIT_LANES,
        provenance_bytes_added=PROVENANCE_ARRAY_BYTES_PER_EPISODE,
        match_cells_added=1,
        parent_lesson_replay_count=0,
        successor=successor,
    )


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "projection_geometry_sha256": projection.GEOMETRY_SHA256,
    "legacy_v1_geometry_sha256": associative.GEOMETRY_SHA256,
    "legacy_base_count": 1,
    "delta_matrix_count": 1,
    "retained_v2_parent_count": 0,
    "parent_lesson_replay_count": 0,
    "trace_lanes": TRACE_LANES,
    "trace_cells_per_lane": TRACE_CELLS_PER_LANE,
    "match_score": MATCH_SCORE,
    "match_tonic_score": MATCH_TONIC_SCORE,
    "binary_or_relay_cell_count": BINARY_OR_RELAY_CELL_COUNT,
    "cue_array_bytes_per_episode": CUE_ARRAY_BYTES_PER_EPISODE,
    "output_array_bytes_per_episode": OUTPUT_ARRAY_BYTES_PER_EPISODE,
    "provenance_array_bytes_per_episode": PROVENANCE_ARRAY_BYTES_PER_EPISODE,
    "delta_array_bytes_per_episode": DELTA_ARRAY_BYTES_PER_EPISODE,
    "inherited_v1_checkpoint_array_bytes": INHERITED_V1_CHECKPOINT_ARRAY_BYTES,
    "child_token_or_status_used_for_selection": False,
    "host_shard_iteration_or_ranking_count": 0,
    "raw_byte_or_digit_decode_after_projection_count": 0,
    "generator_corpus_path2_dev_m0_protected_g0_eval_coding_inputs": (
        GENERATOR_EXECUTION_COUNT,
        CORPUS_INPUT_COUNT,
        PATH2_INPUT_COUNT,
        DEV_INPUT_COUNT,
        M0_INPUT_COUNT,
        PROTECTED_INPUT_COUNT,
        G0_EVALUATION_INPUT_COUNT,
        CODING_INPUT_COUNT,
    ),
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)
SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "BINARY_OR_RELAY_CELL_COUNT",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "CUE_ARRAY_BYTES_PER_EPISODE",
    "DELTA_ARRAY_BYTES_PER_EPISODE",
    "DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA",
    "INHERITED_V1_CHECKPOINT_ARRAY_BYTES",
    "MATCH_CELL_OFFSET",
    "MATCH_SCORE",
    "MATCH_TONIC_SCORE",
    "OUTPUT_ARRAY_BYTES_PER_EPISODE",
    "PROVENANCE_ARRAY_BYTES_PER_EPISODE",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "TRACE_CELL_COUNT",
    "CommonAssociationProvenanceV2",
    "CommonAssociationRecallV2",
    "CommonAssociationRecallWorkV2",
    "CommonAssociationTrainingSummaryV2",
    "CommonAssociationTransitionV2",
    "FrozenCommonEventAssociativeMemoryV2",
    "PhysicalCommonAssociationLessonV2",
    "advance_common_event_association_v2",
    "upgrade_common_event_associative_memory_v2",
]
