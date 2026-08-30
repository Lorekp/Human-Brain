"""Bounded domain-neutral associative memory over common G0 event cells.

The memory learns ordered opaque-event -> one-byte consequence associations by
local teacher-gated tick updates.  Recall delegates every publication to the
factorized fitted-GLIF output race in :mod:`common_event_cells_v1`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from experiments.g0 import common_event_cells_v1 as projection

SCOPE = "G0_COMMON_EVENT_ASSOCIATIVE_MEMORY_V1"
STATUS = "MECHANISM_SLICE_ONLY_NO_G0_COHORT_OR_RESULT"

MAX_OBSERVATION_EVENTS = 8
MEMORY_TRACE_LANES = 12
MEMORY_TRACE_CELLS_PER_LANE = 4_096
MEMORY_TRACE_CELL_COUNT = MEMORY_TRACE_LANES * MEMORY_TRACE_CELLS_PER_LANE
MEMORY_TRACE_CELL_OFFSET = projection.TOTAL_CELL_COUNT
TEACHER_CELL_COUNT = projection.OUTPUT_DIGIT_CELL_COUNT
TEACHER_CELL_OFFSET = MEMORY_TRACE_CELL_OFFSET + MEMORY_TRACE_CELL_COUNT
TOTAL_CELL_COUNT = TEACHER_CELL_OFFSET + TEACHER_CELL_COUNT

TICK_CAP = 4
EXCITATORY_ARRIVALS = 2
INHIBITORY_ARRIVALS = 1
TICK_DTYPE = np.dtype(np.uint8)
SYNAPSE_SHAPE = (projection.OUTPUT_DIGIT_CELL_COUNT, MEMORY_TRACE_CELL_COUNT)
CHECKPOINT_ARRAY_BYTES = 2 * int(np.prod(SYNAPSE_SHAPE)) * TICK_DTYPE.itemsize

_TRACE_MULTIPLIERS = (313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383)
_TRACE_POSITION_GAINS = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)
_TRACE_BIASES = (19, 47, 79, 109, 149, 181, 223, 263, 307, 353, 401, 449)

_TRACE_NAMESPACE = b"humanbrain.g0.common-event-associative-memory-v1.trace\0"
_CHECKPOINT_NAMESPACE = b"humanbrain.g0.common-event-associative-memory-v1.checkpoint\0"
_RECALL_NAMESPACE = b"humanbrain.g0.common-event-associative-memory-v1.recall\0"
_GEOMETRY_NAMESPACE = b"humanbrain.g0.common-event-associative-memory-v1.geometry\0"
_CONSTRUCTION_TOKEN = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
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
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> bytes:
    return (
        _CHECKPOINT_NAMESPACE
        + len(metadata).to_bytes(8, "little")
        + metadata
        + _array_frame("excitatory", excitatory)
        + _array_frame("inhibitory", inhibitory)
    )


def _validate_observation_events(events: tuple[bytes, ...]) -> None:
    if (
        type(events) is not tuple
        or not 1 <= len(events) <= MAX_OBSERVATION_EVENTS
        or any(
            type(event) is not bytes
            or not 1 <= len(event) <= projection.MAX_EVENT_BYTES
            for event in events
        )
    ):
        raise ValueError("observations must be one bounded tuple of opaque byte events")


@dataclass(frozen=True, slots=True)
class CommonAssociationLesson:
    observation_events: tuple[bytes, ...]
    consequence_byte: bytes
    cell_gate: bool = True
    teacher_gate: bool = True

    def __post_init__(self) -> None:
        _validate_observation_events(self.observation_events)
        if type(self.consequence_byte) is not bytes or len(self.consequence_byte) != 1:
            raise ValueError("consequence_byte must contain exactly one opaque byte")
        if type(self.cell_gate) is not bool or type(self.teacher_gate) is not bool:
            raise ValueError("cell and teacher gates must be exact booleans")


@dataclass(frozen=True, slots=True)
class AssociationTrainingWork:
    projection_work: int
    delayed_trace_transitions: int
    teacher_cell_spikes: int
    excitatory_update_events: int
    inhibitory_update_events: int
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int

    @property
    def total(self) -> int:
        return sum(dataclasses.asdict(self).values())

    def record(self) -> dict[str, int]:
        return {**dataclasses.asdict(self), "total": self.total}


@dataclass(frozen=True, slots=True)
class AssociationTrainingSummary:
    lesson_count: int
    cell_on_lesson_count: int
    cell_off_lesson_count: int
    teacher_on_lesson_count: int
    teacher_off_lesson_count: int
    work: AssociationTrainingWork
    checkpoint_array_bytes: int
    checkpoint_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "lesson_count": self.lesson_count,
            "cell_on_lesson_count": self.cell_on_lesson_count,
            "cell_off_lesson_count": self.cell_off_lesson_count,
            "teacher_on_lesson_count": self.teacher_on_lesson_count,
            "teacher_off_lesson_count": self.teacher_off_lesson_count,
            "work": self.work.record(),
            "checkpoint_array_bytes": self.checkpoint_array_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True, slots=True)
class RecallWork:
    projection_work: int
    delayed_trace_transitions: int
    synaptic_tick_reads: int
    output_cell_steps: int

    @property
    def total(self) -> int:
        return sum(dataclasses.asdict(self).values())

    def record(self) -> dict[str, int]:
        return {**dataclasses.asdict(self), "total": self.total}


@dataclass(frozen=True, slots=True)
class CommonAssociationRecall:
    status: str
    output: projection.CommonOutputDecision
    observation_event_count: int
    event_frame_sha256s: tuple[str, ...]
    delayed_trace_cells: tuple[int, ...]
    checkpoint_sha256: str
    checkpoint_authentication_count: int
    work: RecallWork
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "output": self.output.record(),
            "observation_event_count": self.observation_event_count,
            "event_frame_sha256s": self.event_frame_sha256s,
            "delayed_trace_cells": self.delayed_trace_cells,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_authentication_count": self.checkpoint_authentication_count,
            "work": self.work.record(),
            "raw_observation_bytes_retained": False,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class _OrderedTrace:
    local_cells: tuple[int, ...]
    physical_cells: tuple[int, ...]
    event_frame_sha256s: tuple[str, ...]
    projection_work: int
    delayed_trace_transitions: int
    sha256: str


def _ordered_trace(
    cells: projection.CommonEventCells,
    observation_events: tuple[bytes, ...],
    *,
    cell_gate: bool,
) -> _OrderedTrace:
    _validate_observation_events(observation_events)
    frames = tuple(cells.project(event) for event in observation_events)
    state = tuple(
        (lane * 41 + 13) % MEMORY_TRACE_CELLS_PER_LANE
        for lane in range(MEMORY_TRACE_LANES)
    )
    for ordinal, frame in enumerate(frames):
        state = tuple(
            (
                state[lane] * _TRACE_MULTIPLIERS[lane]
                + frame.trace_cells[lane]
                + (ordinal + 1) * _TRACE_POSITION_GAINS[lane]
                + _TRACE_BIASES[lane]
            )
            % MEMORY_TRACE_CELLS_PER_LANE
            for lane in range(MEMORY_TRACE_LANES)
        )
    local = (
        tuple(
            lane * MEMORY_TRACE_CELLS_PER_LANE + state[lane]
            for lane in range(MEMORY_TRACE_LANES)
        )
        if cell_gate
        else ()
    )
    physical = tuple(MEMORY_TRACE_CELL_OFFSET + cell for cell in local)
    record = {
        "event_frame_sha256s": tuple(frame.sha256 for frame in frames),
        "local_cells": local,
        "physical_cells": physical,
        "cell_gate": cell_gate,
        "ordered_event_count": len(frames),
    }
    return _OrderedTrace(
        local_cells=local,
        physical_cells=physical,
        event_frame_sha256s=record["event_frame_sha256s"],
        projection_work=sum(frame.work.total for frame in frames),
        delayed_trace_transitions=len(frames) * MEMORY_TRACE_LANES,
        sha256=_digest(_TRACE_NAMESPACE, record),
    )


def _token_digits(consequence_byte: bytes) -> tuple[int, ...]:
    value = consequence_byte[0]
    digits = [0] * projection.OUTPUT_DIGIT_LANES
    for lane in range(projection.OUTPUT_DIGIT_LANES - 1, -1, -1):
        digits[lane] = value % projection.OUTPUT_DIGIT_RADIX
        value //= projection.OUTPUT_DIGIT_RADIX
    return tuple(digits)


def _teacher_spike_cells(consequence_byte: bytes) -> tuple[int, ...]:
    return tuple(
        TEACHER_CELL_OFFSET + lane * projection.OUTPUT_DIGIT_RADIX + digit
        for lane, digit in enumerate(_token_digits(consequence_byte))
    )


def _increment_ticks(
    array: np.ndarray,
    row: int,
    active_cells: tuple[int, ...],
    arrivals: int,
) -> int:
    active = np.asarray(active_cells, dtype=np.intp)
    previous = array[row, active].astype(np.uint16)
    updated = np.minimum(previous + arrivals, TICK_CAP)
    array[row, active] = updated.astype(TICK_DTYPE)
    return int(np.sum(updated - previous, dtype=np.uint64))


GEOMETRY_RECORD = {
    "scope": SCOPE,
    "projection_geometry_sha256": projection.GEOMETRY_SHA256,
    "maximum_observation_events": MAX_OBSERVATION_EVENTS,
    "memory_trace_lanes": MEMORY_TRACE_LANES,
    "memory_trace_cells_per_lane": MEMORY_TRACE_CELLS_PER_LANE,
    "memory_trace_cell_count": MEMORY_TRACE_CELL_COUNT,
    "active_memory_trace_cells": MEMORY_TRACE_LANES,
    "teacher_cell_count": TEACHER_CELL_COUNT,
    "shared_output_cell_count": projection.OUTPUT_CELL_COUNT,
    "synapse_shape": SYNAPSE_SHAPE,
    "tick_dtype": TICK_DTYPE.str,
    "tick_cap": TICK_CAP,
    "checkpoint_array_bytes": CHECKPOINT_ARRAY_BYTES,
    "total_projection_memory_output_cells": TOTAL_CELL_COUNT,
    "domain_decoder_count": 0,
    "domain_specific_primitive_count": 0,
}
GEOMETRY_SHA256 = _digest(_GEOMETRY_NAMESPACE, GEOMETRY_RECORD)


class FrozenCommonEventAssociativeMemory:
    """Immutable local synapses plus teacher-free physical recall."""

    __slots__ = (
        "_cells",
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_metadata",
        "_training_summary",
    )

    def __init__(
        self,
        *,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: bytes,
        training_summary: AssociationTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("frozen common memory is builder-owned")
        for array in (excitatory, inhibitory):
            if (
                array.shape != SYNAPSE_SHAPE
                or array.dtype != TICK_DTYPE
                or not array.flags.c_contiguous
                or np.any(array > TICK_CAP)
            ):
                raise ValueError("common associative-memory tick state changed")
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._metadata = bytes(metadata)
        self._training_summary = training_summary
        self._cells = projection.CommonEventCells()
        self._checkpoint_sha256 = sha256(
            _checkpoint_payload(self._metadata, self._excitatory, self._inhibitory)
        ).hexdigest()
        if self._checkpoint_sha256 != training_summary.checkpoint_sha256:
            raise RuntimeError("common associative-memory checkpoint identity changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> AssociationTrainingSummary:
        self.authenticate()
        return self._training_summary

    def authenticate(self) -> None:
        observed = sha256(
            _checkpoint_payload(self._metadata, self._excitatory, self._inhibitory)
        ).hexdigest()
        if observed != self._checkpoint_sha256:
            raise RuntimeError("common associative-memory checkpoint authentication failed")

    def recall(self, observation_events: tuple[bytes, ...]) -> CommonAssociationRecall:
        self.authenticate()
        trace = _ordered_trace(self._cells, observation_events, cell_gate=True)
        active = np.asarray(trace.local_cells, dtype=np.intp)
        raw = np.sum(
            self._excitatory[:, active].astype(np.int16)
            - self._inhibitory[:, active].astype(np.int16),
            axis=1,
            dtype=np.int64,
        ).reshape(projection.OUTPUT_DIGIT_LANES, projection.OUTPUT_DIGIT_RADIX)
        raw_scores = tuple(tuple(int(value) for value in row) for row in raw)
        output = self._cells.race_output(raw_scores)
        if output.decision == "TOKEN":
            status = "RECALLED"
        elif output.stop_reason == "COLLISION":
            status = "STOP_AMBIGUOUS_COLLISION"
        else:
            status = "STOP_UNKNOWN"
        work = RecallWork(
            projection_work=trace.projection_work,
            delayed_trace_transitions=trace.delayed_trace_transitions,
            synaptic_tick_reads=(
                projection.OUTPUT_DIGIT_CELL_COUNT * len(trace.local_cells) * 2
            ),
            output_cell_steps=output.physical_cell_steps,
        )
        record = {
            "status": status,
            "output": output.record(),
            "observation_event_count": len(observation_events),
            "event_frame_sha256s": trace.event_frame_sha256s,
            "delayed_trace_cells": trace.physical_cells,
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_authentication_count": 1,
            "work": work.record(),
        }
        return CommonAssociationRecall(
            status=status,
            output=output,
            observation_event_count=len(observation_events),
            event_frame_sha256s=trace.event_frame_sha256s,
            delayed_trace_cells=trace.physical_cells,
            checkpoint_sha256=self._checkpoint_sha256,
            checkpoint_authentication_count=1,
            work=work,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )


def fit_common_event_associations(
    lessons: tuple[CommonAssociationLesson, ...],
) -> FrozenCommonEventAssociativeMemory:
    """Run one ordered local-plasticity pass and freeze its exact tick state."""

    if type(lessons) is not tuple or any(
        type(lesson) is not CommonAssociationLesson for lesson in lessons
    ):
        raise TypeError("lessons must be an exact tuple of CommonAssociationLesson values")
    cells = projection.CommonEventCells()
    excitatory = np.zeros(SYNAPSE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros(SYNAPSE_SHAPE, dtype=TICK_DTYPE)
    projection_work = 0
    delayed_transitions = 0
    teacher_spikes = 0
    excitatory_events = 0
    inhibitory_events = 0
    committed_excitatory = 0
    committed_inhibitory = 0

    for lesson in lessons:
        trace = _ordered_trace(
            cells,
            lesson.observation_events,
            cell_gate=lesson.cell_gate,
        )
        projection_work += trace.projection_work
        delayed_transitions += trace.delayed_trace_transitions
        if not lesson.teacher_gate:
            continue
        teacher_cells = _teacher_spike_cells(lesson.consequence_byte)
        teacher_spikes += len(teacher_cells)
        if not trace.local_cells:
            continue
        for teacher_cell in teacher_cells:
            target_row = teacher_cell - TEACHER_CELL_OFFSET
            lane, target_digit = divmod(target_row, projection.OUTPUT_DIGIT_RADIX)
            committed = _increment_ticks(
                excitatory,
                target_row,
                trace.local_cells,
                EXCITATORY_ARRIVALS,
            )
            committed_excitatory += committed
            excitatory_events += int(committed > 0)
            for rival_digit in range(projection.OUTPUT_DIGIT_RADIX):
                if rival_digit == target_digit:
                    continue
                rival_row = lane * projection.OUTPUT_DIGIT_RADIX + rival_digit
                committed = _increment_ticks(
                    inhibitory,
                    rival_row,
                    trace.local_cells,
                    INHIBITORY_ARRIVALS,
                )
                committed_inhibitory += committed
                inhibitory_events += int(committed > 0)

    work = AssociationTrainingWork(
        projection_work=projection_work,
        delayed_trace_transitions=delayed_transitions,
        teacher_cell_spikes=teacher_spikes,
        excitatory_update_events=excitatory_events,
        inhibitory_update_events=inhibitory_events,
        committed_excitatory_ticks=committed_excitatory,
        committed_inhibitory_ticks=committed_inhibitory,
    )
    metadata_record = {
        "schema": "g0-common-event-associative-memory-v1-checkpoint",
        "scope": SCOPE,
        "status": STATUS,
        "projection_geometry_sha256": projection.GEOMETRY_SHA256,
        "memory_geometry_sha256": GEOMETRY_SHA256,
        "array_shape": SYNAPSE_SHAPE,
        "array_dtype": TICK_DTYPE.str,
        "tick_cap": TICK_CAP,
        "lesson_count": len(lessons),
        "cell_on_lesson_count": sum(lesson.cell_gate for lesson in lessons),
        "cell_off_lesson_count": sum(not lesson.cell_gate for lesson in lessons),
        "teacher_on_lesson_count": sum(lesson.teacher_gate for lesson in lessons),
        "teacher_off_lesson_count": sum(not lesson.teacher_gate for lesson in lessons),
        "training_work": work.record(),
        "domain_decoder_count": 0,
        "protected_input_count": 0,
        "adjudicative_seed_count": 0,
    }
    metadata = _canonical_json(metadata_record)
    checkpoint_sha256 = sha256(
        _checkpoint_payload(metadata, excitatory, inhibitory)
    ).hexdigest()
    summary = AssociationTrainingSummary(
        lesson_count=len(lessons),
        cell_on_lesson_count=metadata_record["cell_on_lesson_count"],
        cell_off_lesson_count=metadata_record["cell_off_lesson_count"],
        teacher_on_lesson_count=metadata_record["teacher_on_lesson_count"],
        teacher_off_lesson_count=metadata_record["teacher_off_lesson_count"],
        work=work,
        checkpoint_array_bytes=CHECKPOINT_ARRAY_BYTES,
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonEventAssociativeMemory(
        excitatory=excitatory,
        inhibitory=inhibitory,
        metadata=metadata,
        training_summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "CHECKPOINT_ARRAY_BYTES",
    "GEOMETRY_RECORD",
    "GEOMETRY_SHA256",
    "MAX_OBSERVATION_EVENTS",
    "MEMORY_TRACE_CELL_COUNT",
    "SCOPE",
    "STATUS",
    "SYNAPSE_SHAPE",
    "TICK_CAP",
    "TOTAL_CELL_COUNT",
    "AssociationTrainingSummary",
    "CommonAssociationLesson",
    "CommonAssociationRecall",
    "FrozenCommonEventAssociativeMemory",
    "fit_common_event_associations",
]
