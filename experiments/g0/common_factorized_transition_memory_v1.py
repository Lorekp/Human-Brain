"""Factorized common-cell transition memory for the pre-cohort G0 mechanism lane.

The module learns two independent physical legs -- context and intervention --
from shared CommonEventCells assemblies.  It publishes only when both legs
support the same factorized byte through the existing fitted-GLIF output race.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise

import numpy as np
from experiments.g0 import common_event_cells_v1 as projection
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCOPE = "G0_COMMON_FACTORIZED_TRANSITION_MEMORY_V1"
STATUS = "QUARANTINED_MECHANISM_ONLY_NO_G0_COHORT_OR_RESULT"

CONTEXT_ROLE = 0
INTERVENTION_ROLE = 1
ROLE_COUNT = 2
MAX_CONTEXT_EVENTS = 4

FACTOR_LANES = 12
FACTOR_CELLS_PER_LANE = 32_768
FACTOR_CELL_COUNT = FACTOR_LANES * FACTOR_CELLS_PER_LANE
FACTOR_CELL_OFFSET = projection.TOTAL_CELL_COUNT
MAX_FRAME_FACTOR_BRANCHES = 3 + 3 * projection.MAX_EVENT_BYTES
MAX_CONTEXT_FACTOR_BRANCHES = (
    MAX_CONTEXT_EVENTS * MAX_FRAME_FACTOR_BRANCHES + MAX_CONTEXT_EVENTS - 1
)
MAX_INTERVENTION_FACTOR_BRANCHES = MAX_FRAME_FACTOR_BRANCHES
BRANCH_SELECTOR_POPULATION_COUNT = ROLE_COUNT * projection.OUTPUT_DIGIT_CELL_COUNT
BRANCH_SELECTOR_CELL_OFFSET = FACTOR_CELL_OFFSET + FACTOR_CELL_COUNT
BRANCH_SELECTOR_CELL_COUNT = projection.OUTPUT_DIGIT_CELL_COUNT * (
    MAX_CONTEXT_FACTOR_BRANCHES + 1 + MAX_INTERVENTION_FACTOR_BRANCHES + 1
)
MEMORY_GATE_CELL_COUNT = 2
MEMORY_GATE_CELL_OFFSET = BRANCH_SELECTOR_CELL_OFFSET + BRANCH_SELECTOR_CELL_COUNT
TEACHER_CELL_COUNT = projection.OUTPUT_DIGIT_CELL_COUNT
TEACHER_CELL_OFFSET = MEMORY_GATE_CELL_OFFSET + MEMORY_GATE_CELL_COUNT
TOTAL_CELL_COUNT = TEACHER_CELL_OFFSET + TEACHER_CELL_COUNT

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
EXCITATORY_ARRIVALS = 1
INHIBITORY_ARRIVALS = 1
BRANCH_MIN_SCORE = -(FACTOR_LANES * TICK_CAP)
BRANCH_MAX_SCORE = FACTOR_LANES * TICK_CAP
BRANCH_RACE_STEPS = projection.RACE_STEPS
BRANCH_ZERO_SCORE_SPIKE_STEP = projection.ZERO_SCORE_SPIKE_STEP
SYNAPSE_SHAPE = (
    ROLE_COUNT,
    projection.OUTPUT_DIGIT_CELL_COUNT,
    FACTOR_CELL_COUNT,
)
CHECKPOINT_ARRAY_BYTES = 2 * int(np.prod(SYNAPSE_SHAPE)) * TICK_DTYPE.itemsize

_FACTOR_NAMESPACE = b"humanbrain.g0.common-factorized-transition-v1.factor\0"
_CHECKPOINT_NAMESPACE = b"humanbrain.g0.common-factorized-transition-v1.checkpoint\0"
_RECALL_NAMESPACE = b"humanbrain.g0.common-factorized-transition-v1.recall\0"
_GEOMETRY_NAMESPACE = b"humanbrain.g0.common-factorized-transition-v1.geometry\0"
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
    return len(header).to_bytes(8, "little") + header + len(payload).to_bytes(8, "little") + payload


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


def _validate_transition_input(
    context_events: tuple[bytes, ...], intervention_event: bytes
) -> None:
    if (
        type(context_events) is not tuple
        or not 1 <= len(context_events) <= MAX_CONTEXT_EVENTS
        or any(
            type(event) is not bytes or not 1 <= len(event) <= projection.MAX_EVENT_BYTES
            for event in context_events
        )
        or type(intervention_event) is not bytes
        or not 1 <= len(intervention_event) <= projection.MAX_EVENT_BYTES
    ):
        raise ValueError("transition inputs must be bounded opaque byte events")


@dataclass(frozen=True, slots=True)
class CommonTransitionLesson:
    context_events: tuple[bytes, ...]
    intervention_event: bytes
    consequence_byte: bytes
    cell_gate: bool = True
    teacher_gate: bool = True

    def __post_init__(self) -> None:
        _validate_transition_input(self.context_events, self.intervention_event)
        if type(self.consequence_byte) is not bytes or len(self.consequence_byte) != 1:
            raise ValueError("consequence_byte must contain exactly one opaque byte")
        if type(self.cell_gate) is not bool or type(self.teacher_gate) is not bool:
            raise ValueError("cell and teacher gates must be exact booleans")


@dataclass(frozen=True, slots=True)
class TransitionTrainingWork:
    projection_work: int
    common_assembly_cell_reads: int
    factor_cell_emissions: int
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
class TransitionTrainingSummary:
    lesson_count: int
    cell_on_lesson_count: int
    cell_off_lesson_count: int
    teacher_on_lesson_count: int
    teacher_off_lesson_count: int
    context_factor_branch_count: int
    intervention_factor_branch_count: int
    work: TransitionTrainingWork
    checkpoint_array_bytes: int
    checkpoint_sha256: str

    def record(self) -> dict[str, object]:
        record = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in {"work"}
        }
        record["work"] = self.work.record()
        return record


@dataclass(frozen=True, slots=True)
class TransitionRecallWork:
    projection_work: int
    common_assembly_cell_reads: int
    factor_cell_emissions: int
    synaptic_tick_reads: int
    branch_selector_race_decisions: int
    branch_selector_cell_steps: int
    coincidence_cell_reads: int
    output_cell_steps: int

    @property
    def total(self) -> int:
        return sum(dataclasses.asdict(self).values())

    def record(self) -> dict[str, int]:
        return {**dataclasses.asdict(self), "total": self.total}


@dataclass(frozen=True, slots=True)
class CommonTransitionRecall:
    status: str
    output: projection.CommonOutputDecision
    context_event_count: int
    context_factor_branch_count: int
    intervention_factor_branch_count: int
    context_support_scores: tuple[int, ...]
    intervention_support_scores: tuple[int, ...]
    context_branch_spike_steps: tuple[int, ...]
    intervention_branch_spike_steps: tuple[int, ...]
    context_branch_simultaneous_spike_candidates: tuple[int, ...]
    intervention_branch_simultaneous_spike_candidates: tuple[int, ...]
    context_branch_zero_stop_count: int
    intervention_branch_zero_stop_count: int
    coincidence_scores: tuple[tuple[int, ...], ...]
    memory_reset: bool
    checkpoint_sha256: str
    checkpoint_authentication_count: int
    work: TransitionRecallWork
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in {"output", "work", "sha256"}
        }
        record["output"] = self.output.record()
        record["work"] = self.work.record()
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class _FactorBranch:
    local_cells: tuple[int, ...]
    physical_cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PhysicalBranchSupport:
    scores: tuple[int, ...]
    spike_steps: tuple[int, ...]
    simultaneous_spike_candidates: tuple[int, ...]
    zero_stop_count: int
    physical_cell_steps: int


@dataclass(frozen=True, slots=True)
class _ProjectedTransition:
    context_branches: tuple[_FactorBranch, ...]
    intervention_branches: tuple[_FactorBranch, ...]
    context_frame_sha256s: tuple[str, ...]
    intervention_frame_sha256: str
    projection_work: int
    common_assembly_cell_reads: int
    factor_cell_emissions: int


def _factor_branch(kind: str, source_cells: tuple[int, ...]) -> _FactorBranch:
    if (
        type(kind) is not str
        or not kind
        or type(source_cells) is not tuple
        or not source_cells
        or any(type(cell) is not int or cell < 0 for cell in source_cells)
    ):
        raise ValueError("factor branches require one named physical source assembly")
    payload = b"".join(cell.to_bytes(4, "big") for cell in source_cells)
    local = tuple(
        lane * FACTOR_CELLS_PER_LANE
        + int.from_bytes(
            sha256(
                _FACTOR_NAMESPACE + kind.encode("ascii") + b"\0" + payload + lane.to_bytes(2, "big")
            ).digest()[:4],
            "big",
        )
        % FACTOR_CELLS_PER_LANE
        for lane in range(FACTOR_LANES)
    )
    return _FactorBranch(
        local_cells=local,
        physical_cells=tuple(FACTOR_CELL_OFFSET + cell for cell in local),
    )


def _frame_factor_branches(
    frame: projection.CommonEventFrame,
) -> tuple[_FactorBranch, ...]:
    source_rows: list[tuple[str, tuple[int, ...]]] = [
        ("boundary-start", frame.start_cells),
        ("boundary-end", frame.end_cells),
        ("event-recurrent", frame.trace_cells),
    ]
    for emission in frame.byte_emissions:
        source_rows.extend(
            (
                ("value", emission.value_cells),
                ("position", emission.position_cells),
                ("position-value-binding", emission.binding_cells),
            )
        )
    unique = tuple(sorted(set(source_rows)))
    return tuple(_factor_branch(kind, cells) for kind, cells in unique)


def _project_transition(
    cells: projection.CommonEventCells,
    context_events: tuple[bytes, ...],
    intervention_event: bytes,
    *,
    cell_gate: bool,
) -> _ProjectedTransition:
    _validate_transition_input(context_events, intervention_event)
    context_frames = tuple(cells.project(event) for event in context_events)
    intervention_frame = cells.project(intervention_event)
    context_branches = [
        branch for frame in context_frames for branch in _frame_factor_branches(frame)
    ]
    for left, right in pairwise(context_frames):
        context_branches.append(
            _factor_branch(
                "ordered-immediate-event-transition",
                (*left.trace_cells, *right.trace_cells),
            )
        )
    intervention_branches = list(_frame_factor_branches(intervention_frame))

    def unique_branches(rows: list[_FactorBranch]) -> tuple[_FactorBranch, ...]:
        by_cells = {row.local_cells: row for row in rows}
        return tuple(by_cells[key] for key in sorted(by_cells))

    context = unique_branches(context_branches) if cell_gate else ()
    intervention = unique_branches(intervention_branches) if cell_gate else ()
    branch_count = len(context_branches) + len(intervention_branches)
    return _ProjectedTransition(
        context_branches=context,
        intervention_branches=intervention,
        context_frame_sha256s=tuple(frame.sha256 for frame in context_frames),
        intervention_frame_sha256=intervention_frame.sha256,
        projection_work=(
            sum(frame.work.total for frame in context_frames) + intervention_frame.work.total
        ),
        common_assembly_cell_reads=branch_count * projection.ASSEMBLY_ACTIVE_CELLS,
        factor_cell_emissions=branch_count * FACTOR_LANES,
    )


def _teacher_spike_cells(consequence_byte: bytes) -> tuple[int, ...]:
    value = consequence_byte[0]
    digits = [0] * projection.OUTPUT_DIGIT_LANES
    for lane in range(projection.OUTPUT_DIGIT_LANES - 1, -1, -1):
        digits[lane] = value % projection.OUTPUT_DIGIT_RADIX
        value //= projection.OUTPUT_DIGIT_RADIX
    return tuple(
        TEACHER_CELL_OFFSET + lane * projection.OUTPUT_DIGIT_RADIX + digit
        for lane, digit in enumerate(digits)
    )


def _active_factor_cells(branches: tuple[_FactorBranch, ...]) -> tuple[int, ...]:
    return tuple(sorted({cell for branch in branches for cell in branch.local_cells}))


def _increment_ticks(
    array: np.ndarray,
    role: int,
    row: int,
    active_cells: tuple[int, ...],
    arrivals: int,
) -> int:
    active = np.asarray(active_cells, dtype=np.intp)
    previous = array[role, row, active].astype(np.uint16)
    updated = np.minimum(previous + arrivals, TICK_CAP)
    array[role, row, active] = updated.astype(TICK_DTYPE)
    return int(np.sum(updated - previous, dtype=np.uint64))


def _build_branch_current_table() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    unit_voltage, unit_voltage_threshold = population.compile_resting_response(BRANCH_RACE_STEPS)
    spike_gain = unit_voltage - unit_voltage_threshold
    crossing_current = population.parameters.threshold_inf_relative_mv / spike_gain
    currents: list[float] = []
    for score in range(BRANCH_MAX_SCORE + 1):
        first_step = BRANCH_ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("branch-selector fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=BRANCH_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        row.spike_step != BRANCH_ZERO_SCORE_SPIKE_STEP - score for score, row in enumerate(controls)
    ):
        raise RuntimeError("branch-selector fitted-GLIF latency calibration changed")
    return table, race


_BRANCH_CURRENT_TABLE, _BRANCH_RACE = _build_branch_current_table()
BRANCH_CURRENT_TABLE_SHA256 = sha256(
    _array_frame("branch-current-table", _BRANCH_CURRENT_TABLE)
).hexdigest()


def _physical_branch_support(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    role: int,
    branches: tuple[_FactorBranch, ...],
) -> _PhysicalBranchSupport:
    if branches:
        branch_cells = np.asarray(tuple(branch.local_cells for branch in branches), dtype=np.intp)
        net = excitatory[role][:, branch_cells].astype(np.int16) - inhibitory[role][
            :, branch_cells
        ].astype(np.int16)
        branch_scores = np.sum(net, axis=2, dtype=np.int64)
        injected = _BRANCH_CURRENT_TABLE[np.clip(branch_scores, 0, BRANCH_MAX_SCORE)]
    else:
        injected = np.empty((projection.OUTPUT_DIGIT_CELL_COUNT, 0), dtype=np.float64)
    tonic_zero = np.full(
        (projection.OUTPUT_DIGIT_CELL_COUNT, 1),
        _BRANCH_CURRENT_TABLE[0],
        dtype=np.float64,
    )
    currents = np.ascontiguousarray(
        np.concatenate((injected, tonic_zero), axis=1), dtype=np.float64
    )
    races = _BRANCH_RACE.race_many(currents, require_unique=False)
    scores = tuple(BRANCH_ZERO_SCORE_SPIKE_STEP - race.spike_step for race in races)
    if any(not 0 <= score <= BRANCH_MAX_SCORE for score in scores):
        raise RuntimeError("physical branch-selector latency left calibrated score range")
    return _PhysicalBranchSupport(
        scores=scores,
        spike_steps=tuple(race.spike_step for race in races),
        simultaneous_spike_candidates=tuple(race.simultaneous_spike_candidates for race in races),
        zero_stop_count=sum(score == 0 for score in scores),
        physical_cell_steps=(
            projection.OUTPUT_DIGIT_CELL_COUNT * (len(branches) + 1) * BRANCH_RACE_STEPS
        ),
    )


GEOMETRY_RECORD = {
    "scope": SCOPE,
    "common_projection_geometry_sha256": projection.GEOMETRY_SHA256,
    "roles": ("context", "intervention"),
    "maximum_context_events": MAX_CONTEXT_EVENTS,
    "factor_lanes": FACTOR_LANES,
    "factor_cells_per_lane": FACTOR_CELLS_PER_LANE,
    "factor_cell_count": FACTOR_CELL_COUNT,
    "active_cells_per_factor": FACTOR_LANES,
    "factor_sources": (
        "value-assembly",
        "position-assembly",
        "position-value-binding-assembly",
        "boundary-assembly",
        "event-recurrent-assembly",
        "ordered-immediate-event-transition",
    ),
    "branch_selector": {
        "mechanism": "independent-fitted-glif-first-spike-max-pool",
        "authoritative_observable": "first-spike-latency-only",
        "host_branch_winner_identity_used": False,
        "population_count": BRANCH_SELECTOR_POPULATION_COUNT,
        "maximum_context_branches_per_population": MAX_CONTEXT_FACTOR_BRANCHES,
        "maximum_intervention_branches_per_population": (MAX_INTERVENTION_FACTOR_BRANCHES),
        "tonic_zero_cells_per_population": 1,
        "cell_count": BRANCH_SELECTOR_CELL_COUNT,
        "race_steps": BRANCH_RACE_STEPS,
        "zero_score_spike_step": BRANCH_ZERO_SCORE_SPIKE_STEP,
        "minimum_score": BRANCH_MIN_SCORE,
        "maximum_score": BRANCH_MAX_SCORE,
        "current_table_sha256": BRANCH_CURRENT_TABLE_SHA256,
    },
    "role_coincidence": "candidate-current-is-weaker-nonnegative-role-leg",
    "synapse_shape": SYNAPSE_SHAPE,
    "tick_dtype": TICK_DTYPE.str,
    "tick_cap": TICK_CAP,
    "checkpoint_array_bytes": CHECKPOINT_ARRAY_BYTES,
    "memory_gate_cell_count": MEMORY_GATE_CELL_COUNT,
    "teacher_cell_count": TEACHER_CELL_COUNT,
    "total_cell_count": TOTAL_CELL_COUNT,
    "whole_trace_lookup_count": 0,
    "host_nearest_neighbor_count": 0,
    "domain_decoder_count": 0,
}
GEOMETRY_SHA256 = _digest(_GEOMETRY_NAMESPACE, GEOMETRY_RECORD)


class FrozenCommonFactorizedTransitionMemory:
    """Immutable factor synapses with teacher-free two-leg recall."""

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
        training_summary: TransitionTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("frozen factorized transition memory is builder-owned")
        for array in (excitatory, inhibitory):
            if (
                array.shape != SYNAPSE_SHAPE
                or array.dtype != TICK_DTYPE
                or not array.flags.c_contiguous
                or np.any(array > TICK_CAP)
            ):
                raise ValueError("factorized transition tick state changed")
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
            raise RuntimeError("factorized transition checkpoint identity changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> TransitionTrainingSummary:
        self.authenticate()
        return self._training_summary

    def authenticate(self) -> None:
        observed = sha256(
            _checkpoint_payload(self._metadata, self._excitatory, self._inhibitory)
        ).hexdigest()
        if observed != self._checkpoint_sha256:
            raise RuntimeError("factorized transition checkpoint authentication failed")

    def recall(
        self,
        context_events: tuple[bytes, ...],
        intervention_event: bytes,
        *,
        memory_reset: bool = False,
    ) -> CommonTransitionRecall:
        if type(memory_reset) is not bool:
            raise TypeError("memory_reset must be an exact bool")
        self.authenticate()
        projected = _project_transition(
            self._cells,
            context_events,
            intervention_event,
            cell_gate=True,
        )
        context_branches = () if memory_reset else projected.context_branches
        context_support = _physical_branch_support(
            self._excitatory,
            self._inhibitory,
            CONTEXT_ROLE,
            context_branches,
        )
        intervention_support = _physical_branch_support(
            self._excitatory,
            self._inhibitory,
            INTERVENTION_ROLE,
            projected.intervention_branches,
        )
        coincidence_flat = tuple(
            min(max(context, 0), max(intervention, 0))
            for context, intervention in zip(
                context_support.scores,
                intervention_support.scores,
                strict=True,
            )
        )
        coincidence = tuple(
            coincidence_flat[
                lane * projection.OUTPUT_DIGIT_RADIX : (lane + 1) * projection.OUTPUT_DIGIT_RADIX
            ]
            for lane in range(projection.OUTPUT_DIGIT_LANES)
        )
        output = self._cells.race_output(coincidence)
        if output.decision == "TOKEN":
            status = "RECALLED"
        elif output.stop_reason == "COLLISION":
            status = "STOP_AMBIGUOUS_COLLISION"
        else:
            status = "STOP_UNKNOWN"
        scored_branch_count = len(context_branches) + len(projected.intervention_branches)
        work = TransitionRecallWork(
            projection_work=projected.projection_work,
            common_assembly_cell_reads=projected.common_assembly_cell_reads,
            factor_cell_emissions=projected.factor_cell_emissions,
            synaptic_tick_reads=(
                scored_branch_count * projection.OUTPUT_DIGIT_CELL_COUNT * FACTOR_LANES * 2
            ),
            branch_selector_race_decisions=BRANCH_SELECTOR_POPULATION_COUNT,
            branch_selector_cell_steps=(
                context_support.physical_cell_steps + intervention_support.physical_cell_steps
            ),
            coincidence_cell_reads=projection.OUTPUT_DIGIT_CELL_COUNT * ROLE_COUNT,
            output_cell_steps=output.physical_cell_steps,
        )
        record = {
            "status": status,
            "output": output.record(),
            "context_event_count": len(context_events),
            "context_factor_branch_count": len(context_branches),
            "intervention_factor_branch_count": len(projected.intervention_branches),
            "context_support_scores": context_support.scores,
            "intervention_support_scores": intervention_support.scores,
            "context_branch_spike_steps": context_support.spike_steps,
            "intervention_branch_spike_steps": intervention_support.spike_steps,
            "context_branch_simultaneous_spike_candidates": (
                context_support.simultaneous_spike_candidates
            ),
            "intervention_branch_simultaneous_spike_candidates": (
                intervention_support.simultaneous_spike_candidates
            ),
            "context_branch_zero_stop_count": context_support.zero_stop_count,
            "intervention_branch_zero_stop_count": (intervention_support.zero_stop_count),
            "coincidence_scores": coincidence,
            "memory_reset": memory_reset,
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_authentication_count": 1,
            "work": work.record(),
        }
        return CommonTransitionRecall(
            status=status,
            output=output,
            context_event_count=len(context_events),
            context_factor_branch_count=len(context_branches),
            intervention_factor_branch_count=len(projected.intervention_branches),
            context_support_scores=context_support.scores,
            intervention_support_scores=intervention_support.scores,
            context_branch_spike_steps=context_support.spike_steps,
            intervention_branch_spike_steps=intervention_support.spike_steps,
            context_branch_simultaneous_spike_candidates=(
                context_support.simultaneous_spike_candidates
            ),
            intervention_branch_simultaneous_spike_candidates=(
                intervention_support.simultaneous_spike_candidates
            ),
            context_branch_zero_stop_count=context_support.zero_stop_count,
            intervention_branch_zero_stop_count=(intervention_support.zero_stop_count),
            coincidence_scores=coincidence,
            memory_reset=memory_reset,
            checkpoint_sha256=self._checkpoint_sha256,
            checkpoint_authentication_count=1,
            work=work,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )


def fit_common_factorized_transitions(
    lessons: tuple[CommonTransitionLesson, ...],
) -> FrozenCommonFactorizedTransitionMemory:
    """Run one local teacher-gated pass and freeze exact factor synapses."""

    if type(lessons) is not tuple or any(
        type(lesson) is not CommonTransitionLesson for lesson in lessons
    ):
        raise TypeError("lessons must be an exact tuple of CommonTransitionLesson")
    cells = projection.CommonEventCells()
    excitatory = np.zeros(SYNAPSE_SHAPE, dtype=TICK_DTYPE)
    inhibitory = np.zeros(SYNAPSE_SHAPE, dtype=TICK_DTYPE)
    projection_work = 0
    common_reads = 0
    factor_emissions = 0
    teacher_spikes = 0
    excitatory_events = 0
    inhibitory_events = 0
    committed_excitatory = 0
    committed_inhibitory = 0
    context_branch_count = 0
    intervention_branch_count = 0

    for lesson in lessons:
        projected = _project_transition(
            cells,
            lesson.context_events,
            lesson.intervention_event,
            cell_gate=lesson.cell_gate,
        )
        projection_work += projected.projection_work
        common_reads += projected.common_assembly_cell_reads
        factor_emissions += projected.factor_cell_emissions
        context_branch_count += len(projected.context_branches)
        intervention_branch_count += len(projected.intervention_branches)
        if not lesson.teacher_gate:
            continue
        teacher_cells = _teacher_spike_cells(lesson.consequence_byte)
        teacher_spikes += len(teacher_cells)
        active_by_role = (
            _active_factor_cells(projected.context_branches),
            _active_factor_cells(projected.intervention_branches),
        )
        if not all(active_by_role):
            continue
        for teacher_cell in teacher_cells:
            target_row = teacher_cell - TEACHER_CELL_OFFSET
            lane, target_digit = divmod(target_row, projection.OUTPUT_DIGIT_RADIX)
            for role, active_cells in enumerate(active_by_role):
                committed = _increment_ticks(
                    excitatory,
                    role,
                    target_row,
                    active_cells,
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
                        role,
                        rival_row,
                        active_cells,
                        INHIBITORY_ARRIVALS,
                    )
                    committed_inhibitory += committed
                    inhibitory_events += int(committed > 0)

    work = TransitionTrainingWork(
        projection_work=projection_work,
        common_assembly_cell_reads=common_reads,
        factor_cell_emissions=factor_emissions,
        teacher_cell_spikes=teacher_spikes,
        excitatory_update_events=excitatory_events,
        inhibitory_update_events=inhibitory_events,
        committed_excitatory_ticks=committed_excitatory,
        committed_inhibitory_ticks=committed_inhibitory,
    )
    metadata_record = {
        "schema": "g0-common-factorized-transition-memory-v1-checkpoint",
        "scope": SCOPE,
        "status": STATUS,
        "projection_geometry_sha256": projection.GEOMETRY_SHA256,
        "transition_geometry_sha256": GEOMETRY_SHA256,
        "array_shape": SYNAPSE_SHAPE,
        "array_dtype": TICK_DTYPE.str,
        "tick_cap": TICK_CAP,
        "lesson_count": len(lessons),
        "cell_on_lesson_count": sum(lesson.cell_gate for lesson in lessons),
        "cell_off_lesson_count": sum(not lesson.cell_gate for lesson in lessons),
        "teacher_on_lesson_count": sum(lesson.teacher_gate for lesson in lessons),
        "teacher_off_lesson_count": sum(not lesson.teacher_gate for lesson in lessons),
        "context_factor_branch_count": context_branch_count,
        "intervention_factor_branch_count": intervention_branch_count,
        "training_work": work.record(),
        "whole_trace_lookup_count": 0,
        "host_nearest_neighbor_count": 0,
        "protected_input_count": 0,
        "adjudicative_seed_count": 0,
    }
    metadata = _canonical_json(metadata_record)
    checkpoint_sha256 = sha256(_checkpoint_payload(metadata, excitatory, inhibitory)).hexdigest()
    summary = TransitionTrainingSummary(
        lesson_count=len(lessons),
        cell_on_lesson_count=metadata_record["cell_on_lesson_count"],
        cell_off_lesson_count=metadata_record["cell_off_lesson_count"],
        teacher_on_lesson_count=metadata_record["teacher_on_lesson_count"],
        teacher_off_lesson_count=metadata_record["teacher_off_lesson_count"],
        context_factor_branch_count=context_branch_count,
        intervention_factor_branch_count=intervention_branch_count,
        work=work,
        checkpoint_array_bytes=CHECKPOINT_ARRAY_BYTES,
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonFactorizedTransitionMemory(
        excitatory=excitatory,
        inhibitory=inhibitory,
        metadata=metadata,
        training_summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "BRANCH_CURRENT_TABLE_SHA256",
    "BRANCH_SELECTOR_CELL_COUNT",
    "BRANCH_SELECTOR_POPULATION_COUNT",
    "BRANCH_ZERO_SCORE_SPIKE_STEP",
    "CHECKPOINT_ARRAY_BYTES",
    "FACTOR_CELL_COUNT",
    "GEOMETRY_RECORD",
    "GEOMETRY_SHA256",
    "MAX_CONTEXT_EVENTS",
    "SCOPE",
    "STATUS",
    "SYNAPSE_SHAPE",
    "TICK_CAP",
    "TOTAL_CELL_COUNT",
    "CommonTransitionLesson",
    "CommonTransitionRecall",
    "FrozenCommonFactorizedTransitionMemory",
    "TransitionTrainingSummary",
    "fit_common_factorized_transitions",
]
