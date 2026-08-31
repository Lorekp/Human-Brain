"""Route two frozen memory views through learned reusable context factors.

The child is one immutable ``FrozenCommonEpisodicTransformMemoryV2``.  A
lesson supplies an unordered tuple of twelve-cell physical factor assemblies.
Only individual assemblies are banked; complete context combinations are not
part of the trainable state.  Public exact and semantic child recalls compete
against an observed four-cell consequence, so no route label is supplied by
the host.

At recall, every factor is recovered by a fitted-GLIF equality race.  Its local
E/H route evidence is summed commutatively with the other recovered factors and
the two routes compete against a tonic abstention cell.  The winning route gates
only public child ``active_cells`` through fixed relays into a shared physical
output race.  Unknown factors, route collisions, and selected-child STOPs never
fall back.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCOPE = "COMMON_FACTORIZED_CONTEXT_ROUTED_MEMORY_V2"
STATUS = "SELF_POSED_CONSEQUENCE_TRAINED_FACTORIZED_CONTEXT_RETRIEVAL"
CLAIM_BOUNDARY = (
    "self-posed-replay-free-routing-between-two-banked-one-byte-memory-views-"
    "from-unordered-compositions-of-previously-learned-opaque-physical-factors-"
    "with-explicit-factor-framing-only-no-factor-discovery-language-coding-"
    "corpus-protected-transfer-or-benchmark-evidence"
)

EXACT_ROUTE = 0
SEMANTIC_ROUTE = 1
ROUTE_COUNT = 2

RECALLED = "RECALLED"
STOP_CELL_OFF = "STOP_FACTORIZED_CONTEXT_CELL_OFF"
STOP_FACTOR_GATE = "STOP_FACTORIZED_CONTEXT_FACTOR_GATE_OFF"
STOP_FACTOR_UNKNOWN = "STOP_FACTORIZED_CONTEXT_FACTOR_UNKNOWN"
STOP_FACTOR_AMBIGUOUS = "STOP_FACTORIZED_CONTEXT_FACTOR_AMBIGUOUS"
STOP_ROUTE_UNSUPPORTED = "STOP_FACTORIZED_CONTEXT_ROUTE_UNSUPPORTED"
STOP_ROUTE_AMBIGUOUS = "STOP_FACTORIZED_CONTEXT_ROUTE_AMBIGUOUS"
STOP_OUTPUT_RELAY = "STOP_FACTORIZED_CONTEXT_OUTPUT_RELAY_OFF"
STOP_SELECTED_CHILD = "STOP_FACTORIZED_CONTEXT_SELECTED_CHILD"
STOP_CONSEQUENCE_NOT_SINGLETON = (
    "STOP_FACTORIZED_CONTEXT_CONSEQUENCE_NOT_SINGLETON"
)
STOP_COMPARATOR_OFF = "STOP_FACTORIZED_CONTEXT_COMPARATOR_OFF"
STOP_TEACHER_OFF = "STOP_FACTORIZED_CONTEXT_TEACHER_OFF"

ACCEPTED_VOTE = "ACCEPTED_VOTE"
DUPLICATE_NO_VOTE = "DUPLICATE_NO_VOTE"
DEPENDENT_NO_VOTE = "DEPENDENT_NO_VOTE"

MAX_FACTORS = 32
MAX_FACTORS_PER_CONTEXT = 4
MAX_PROVENANCE_ROWS = 512
TRACE_LANES = output_projection.TRACE_LANES
OUTPUT_LANES = output_projection.OUTPUT_DIGIT_LANES
OUTPUT_RADIX = output_projection.OUTPUT_DIGIT_RADIX
OUTPUT_CANDIDATE_COUNT = OUTPUT_LANES * OUTPUT_RADIX

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
CUE_DTYPE = np.dtype(np.int32)
PROVENANCE_INDEX_DTYPE = np.dtype(np.int16)
PROVENANCE_CELL_DTYPE = np.dtype(np.int32)
PROVENANCE_CODE_DTYPE = np.dtype(np.uint8)
PROVENANCE_ROOT_DTYPE = np.dtype(np.uint8)

FACTOR_STATE_BYTES = TRACE_LANES * CUE_DTYPE.itemsize + (
    2 * ROUTE_COUNT * TICK_DTYPE.itemsize
)
PROVENANCE_BYTES_PER_ROW = (
    PROVENANCE_INDEX_DTYPE.itemsize
    + 3 * TRACE_LANES * PROVENANCE_CELL_DTYPE.itemsize
    + 2 * PROVENANCE_CODE_DTYPE.itemsize
    + 32 * PROVENANCE_ROOT_DTYPE.itemsize
)

RACE_STEPS = 600
ZERO_SCORE_SPIKE_STEP = 568
FACTOR_MATCH_SCORE = TRACE_LANES
FACTOR_TONIC_SCORE = FACTOR_MATCH_SCORE - 2
FACTOR_LEARNING_TONIC_SCORE = FACTOR_MATCH_SCORE - 1
CONSEQUENCE_MATCH_SCORE = OUTPUT_LANES
CONSEQUENCE_TONIC_SCORE = CONSEQUENCE_MATCH_SCORE - 1
ROUTE_TONIC_SCORE = 1
MAX_RACE_SCORE = MAX_FACTORS_PER_CONTEXT * TICK_CAP
RELAY_ACTIVE_SCORE = 2 * output_projection.STOP_SCORE

CONSEQUENCE_EQUALITY_CELL_COUNT = ROUTE_COUNT * OUTPUT_LANES
CONSEQUENCE_INTEGRATOR_CELL_COUNT = ROUTE_COUNT
CONSEQUENCE_TONIC_CELL_ID = (
    CONSEQUENCE_EQUALITY_CELL_COUNT + CONSEQUENCE_INTEGRATOR_CELL_COUNT
)
ROUTE_ACTION_CELL_OFFSET = CONSEQUENCE_TONIC_CELL_ID + 1
ROUTE_ACTION_CELL_COUNT = ROUTE_COUNT
ROUTE_TONIC_CELL_ID = ROUTE_ACTION_CELL_OFFSET + ROUTE_ACTION_CELL_COUNT
OUTPUT_AND_RELAY_CELL_OFFSET = ROUTE_TONIC_CELL_ID + 1
OUTPUT_AND_RELAY_CELL_COUNT = ROUTE_COUNT * OUTPUT_CANDIDATE_COUNT
OUTPUT_OR_RELAY_CELL_OFFSET = (
    OUTPUT_AND_RELAY_CELL_OFFSET + OUTPUT_AND_RELAY_CELL_COUNT
)
OUTPUT_OR_RELAY_CELL_COUNT = OUTPUT_CANDIDATE_COUNT
FACTOR_TONIC_CELL_ID = OUTPUT_OR_RELAY_CELL_OFFSET + OUTPUT_OR_RELAY_CELL_COUNT
FACTOR_MATCH_CELL_OFFSET = FACTOR_TONIC_CELL_ID + 1
FIXED_ROUTER_CELL_COUNT = FACTOR_MATCH_CELL_OFFSET

CONSEQUENCE_CELL_STEPS = (
    CONSEQUENCE_EQUALITY_CELL_COUNT
    + CONSEQUENCE_INTEGRATOR_CELL_COUNT
    + 1
) * RACE_STEPS
ROUTE_RACE_CELL_STEPS = (ROUTE_ACTION_CELL_COUNT + 1) * RACE_STEPS
OUTPUT_AND_RELAY_CELL_STEPS = OUTPUT_AND_RELAY_CELL_COUNT * RACE_STEPS
OUTPUT_OR_RELAY_CELL_STEPS = OUTPUT_OR_RELAY_CELL_COUNT * RACE_STEPS

PUBLIC_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0
CORPUS_INPUT_COUNT = 0
GENERATOR_EXECUTION_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_SOURCE_PATH = Path(__file__).resolve()
_CONFIG_NAMESPACE = b"common-factorized-context-router-v2-config\0"
_CHECKPOINT_NAMESPACE = b"common-factorized-context-router-v2-checkpoint\0"
_PROVENANCE_NAMESPACE = b"common-factorized-context-router-v2-provenance\0"
_RECALL_NAMESPACE = b"common-factorized-context-router-v2-recall\0"
_CONSTRUCTION_TOKEN = object()
_OUTPUT_CELLS = output_projection.CommonEventCells()
_FILLER_CELLS = filler_projection.CommonOneByteFillerCells()


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
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


def _build_score_current_table() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(RACE_STEPS)
    crossing = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(MAX_RACE_SCORE + 1):
        first_step = ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing[first_step])
        upper = float(crossing[first_step - 1])
        if not lower < upper:
            raise RuntimeError("factorized router fitted-GLIF interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        row.spike_step != ZERO_SCORE_SPIKE_STEP - score
        for score, row in enumerate(controls)
    ):
        raise RuntimeError("factorized router fitted-GLIF calibration changed")
    return table, race


_SCORE_CURRENT_TABLE, _SCORE_RACE = _build_score_current_table()

CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "child_config_sha256": memory.CONFIG_SHA256,
    "child_access_surface": (
        "authenticate",
        "checkpoint roots",
        "recall_episode",
        "recall_semantic",
    ),
    "child_token_or_status_used_for_selection": False,
    "host_route_label_input": False,
    "host_factor_order_authority": False,
    "selected_child_fallback": False,
    "factor_encoding": "unordered-twelve-lane-physical-cell-assemblies",
    "maximum_factors": MAX_FACTORS,
    "maximum_factors_per_context": MAX_FACTORS_PER_CONTEXT,
    "maximum_provenance_rows": MAX_PROVENANCE_ROWS,
    "tick_cap": TICK_CAP,
    "race_steps": RACE_STEPS,
    "factor_match_score": FACTOR_MATCH_SCORE,
    "factor_tonic_score": FACTOR_TONIC_SCORE,
    "factor_learning_tonic_score": FACTOR_LEARNING_TONIC_SCORE,
    "route_tonic_score": ROUTE_TONIC_SCORE,
    "fixed_router_cell_count": FIXED_ROUTER_CELL_COUNT,
    "factor_state_bytes": FACTOR_STATE_BYTES,
    "provenance_bytes_per_row": PROVENANCE_BYTES_PER_ROW,
    "replay_count": 0,
    "retained_parent_count": 0,
    "raw_lesson_retention_count": 0,
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
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)


def _validate_event(name: str, event: bytes) -> None:
    if type(event) is not bytes or not 1 <= len(event) <= output_projection.MAX_EVENT_BYTES:
        raise ValueError(f"{name} must be one nonempty bounded bytes event")


def _canonical_factor_cells(
    factors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if (
        type(factors) is not tuple
        or not 1 <= len(factors) <= MAX_FACTORS_PER_CONTEXT
    ):
        raise TypeError("factor_cells must be one bounded nonempty exact tuple")
    for cells in factors:
        _validate_trace_cells("factor_cells", cells)
    if len(set(factors)) != len(factors):
        raise ValueError("factor context contains a duplicate physical assembly")
    return tuple(sorted(factors))


def _validate_observation_events(events: tuple[bytes, ...]) -> None:
    if type(events) is not tuple or not events:
        raise TypeError("exact_observation_events must be one nonempty exact tuple")
    for event in events:
        _validate_event("exact observation", event)


def _validate_trace_cells(name: str, cells: tuple[int, ...]) -> None:
    if (
        type(cells) is not tuple
        or len(cells) != TRACE_LANES
        or any(type(cell) is not int for cell in cells)
    ):
        raise TypeError(f"{name} must be an exact twelve-lane physical trace")
    for lane, cell in enumerate(cells):
        start = (
            output_projection.TRACE_CELL_OFFSET
            + lane * output_projection.FEATURE_BANK_CELLS
        )
        if not start <= cell < start + output_projection.FEATURE_BANK_CELLS:
            raise ValueError(f"{name} physical trace lane changed")


def _validate_output_cells(cells: tuple[int, ...]) -> None:
    if (
        type(cells) is not tuple
        or len(cells) != OUTPUT_LANES
        or any(type(cell) is not int for cell in cells)
    ):
        raise TypeError("observed output must be four exact physical cells")
    for lane, cell in enumerate(cells):
        start = output_projection.OUTPUT_CELL_OFFSET + lane * OUTPUT_RADIX
        if not start <= cell < start + OUTPUT_RADIX:
            raise ValueError("observed output physical lane changed")


@dataclass(frozen=True, slots=True)
class PhysicalFactorRouteLesson:
    factor_cells: tuple[tuple[int, ...], ...]
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
        _canonical_factor_cells(self.factor_cells)
        _validate_observation_events(self.exact_observation_events)
        _FILLER_CELLS.authenticate_frame(self.semantic_input_frame)
        _validate_trace_cells("relation_cells", self.relation_cells)
        _validate_trace_cells("evidence_context_cells", self.evidence_context_cells)
        _validate_trace_cells("evidence_source_cells", self.evidence_source_cells)
        if self.evidence_context_cells == self.evidence_source_cells:
            raise ValueError("evidence context and source must be physically distinct")
        _validate_output_cells(self.observed_output_cells)
        if any(
            type(gate) is not bool
            for gate in (self.teacher_gate, self.cell_gate, self.comparator_gate)
        ):
            raise TypeError("lesson gates must be exact booleans")


@dataclass(frozen=True, slots=True)
class FactorRouteProvenance:
    factor_index: int
    target_route: int
    disposition: str
    relation_cells: tuple[int, ...]
    evidence_context_cells: tuple[int, ...]
    evidence_source_cells: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class FactorRouteLearningStep:
    lesson_ordinal: int
    status: str
    factor_indices: tuple[int, ...]
    factor_dispositions: tuple[str, ...]
    target_route: int | None
    committed_factor_vote_count: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int
    router_cell_steps: int


@dataclass(frozen=True, slots=True)
class FactorRouterTrainingSummary:
    mode: str
    new_lesson_count: int
    total_lesson_count: int
    teacher_off_count: int
    cell_off_count: int
    comparator_off_count: int
    consequence_not_singleton_count: int
    accepted_factor_vote_count: int
    duplicate_nonvote_count: int
    dependent_nonvote_count: int
    factor_count: int
    provenance_count: int
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    factor_assembly_reads: int
    router_learning_cell_steps: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int
    trainable_state_bytes: int
    provenance_state_bytes: int
    checkpoint_array_bytes: int
    fixed_router_cell_count: int
    factor_match_cell_count: int
    added_cell_count: int
    parent_lesson_replay_count: int
    retained_router_parent_count: int
    raw_lesson_retention_count: int
    child_mutation_count: int
    memory_checkpoint_sha256: str
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    last_steps: tuple[FactorRouteLearningStep, ...]
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class FactorRouterRecallWork:
    factor_assembly_reads: int
    factor_match_equality_reads: int
    factor_match_equality_arrivals: int
    factor_match_cell_steps: int
    route_cell_steps: int
    output_and_relay_cell_steps: int
    output_or_relay_cell_steps: int
    output_race_cell_steps: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int

    @property
    def router_cell_steps(self) -> int:
        return (
            self.factor_match_cell_steps
            + self.route_cell_steps
            + self.output_and_relay_cell_steps
            + self.output_or_relay_cell_steps
            + self.output_race_cell_steps
        )

    def record(self) -> dict[str, int]:
        return {**dataclasses.asdict(self), "router_cell_steps": self.router_cell_steps}


@dataclass(frozen=True, slots=True)
class FactorizedContextRecall:
    status: str
    output: output_projection.CommonOutputDecision
    factor_cells: tuple[tuple[int, ...], ...]
    factor_indices: tuple[int | None, ...]
    factor_match_arrival_counts: tuple[tuple[int, ...], ...]
    route_raw_scores: tuple[int, int]
    selected_route: int | None
    selected_route_cell_id: int | None
    route_spike_step: int
    route_simultaneous_spike_count: int
    exact_child_active_cells: tuple[int, ...]
    semantic_child_active_cells: tuple[int, ...]
    output_and_relay_spikes: tuple[bool, ...]
    output_or_relay_spikes: tuple[bool, ...]
    work: FactorRouterRecallWork
    memory_checkpoint_sha256: str
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    checkpoint_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in {"output", "work", "sha256"}
        }
        record["output"] = self.output.record()
        record["work"] = self.work.record()
        record["child_token_or_status_used_for_selection"] = False
        record["selected_child_fallback"] = False
        record["host_factor_order_authority"] = False
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class _FactorMatch:
    status: str
    factor_index: int | None
    novel: bool
    arrivals: tuple[int, ...]
    spike_step: int
    simultaneous_spike_count: int
    equality_reads: int
    equality_arrivals: int
    cell_steps: int


@dataclass(frozen=True, slots=True)
class _RouteRace:
    status: str
    raw_scores: tuple[int, int]
    selected_route: int | None
    spike_step: int
    simultaneous_spike_count: int


@dataclass(frozen=True, slots=True)
class _ConsequenceRace:
    target_route: int | None
    equality_spikes: tuple[tuple[bool, ...], ...]
    integrator_arrivals: tuple[int, int]
    spike_step: int
    simultaneous_spike_count: int
    cell_steps: int


@dataclass(frozen=True, slots=True)
class _BatchTotals:
    teacher_off_count: int
    cell_off_count: int
    comparator_off_count: int
    consequence_not_singleton_count: int
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    factor_assembly_reads: int
    router_learning_cell_steps: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int
    steps: tuple[FactorRouteLearningStep, ...]


def _factor_match(
    cue_cells: np.ndarray,
    active_trace: tuple[int, ...],
    *,
    cell_gate: bool,
    tonic_score: int,
) -> _FactorMatch:
    count = cue_cells.shape[0]
    if count:
        active = np.asarray(active_trace, dtype=CUE_DTYPE)
        arrivals_array = np.count_nonzero(cue_cells == active, axis=1).astype(
            np.int16
        )
    else:
        arrivals_array = np.empty((0,), dtype=np.int16)
    scores = arrivals_array if cell_gate else np.zeros_like(arrivals_array)
    currents = np.concatenate(
        (
            _SCORE_CURRENT_TABLE[scores],
            np.asarray([_SCORE_CURRENT_TABLE[tonic_score]]),
        )
    )
    race = _SCORE_RACE.race_many(currents[None, :], require_unique=False)[0]
    if race.simultaneous_spike_candidates != 1:
        status = STOP_FACTOR_AMBIGUOUS
        factor_index = None
        novel = False
    elif race.winner_cell == count:
        status = STOP_FACTOR_UNKNOWN
        factor_index = None
        novel = True
    else:
        status = RECALLED
        factor_index = race.winner_cell
        novel = False
    return _FactorMatch(
        status=status,
        factor_index=factor_index,
        novel=novel,
        arrivals=tuple(int(value) for value in arrivals_array),
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_candidates,
        equality_reads=count * TRACE_LANES,
        equality_arrivals=int(np.sum(arrivals_array, dtype=np.int64)),
        cell_steps=(count + 1) * RACE_STEPS,
    )


def _route_race(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    factor_indices: tuple[int, ...],
    *,
    router_gate: bool,
) -> _RouteRace:
    if factor_indices:
        local = np.maximum(
            excitatory[np.asarray(factor_indices, dtype=np.intp)].astype(np.int16)
            - inhibitory[np.asarray(factor_indices, dtype=np.intp)].astype(np.int16),
            0,
        )
        raw_array = np.sum(local, axis=0, dtype=np.int16)
    else:
        raw_array = np.zeros((ROUTE_COUNT,), dtype=np.int16)
    scores = raw_array if router_gate else np.zeros_like(raw_array)
    currents = np.concatenate(
        (
            _SCORE_CURRENT_TABLE[scores],
            np.asarray([_SCORE_CURRENT_TABLE[ROUTE_TONIC_SCORE]]),
        )
    )
    race = _SCORE_RACE.race_many(currents[None, :], require_unique=False)[0]
    if race.simultaneous_spike_candidates != 1:
        status = STOP_ROUTE_AMBIGUOUS
        selected = None
    elif race.winner_cell == ROUTE_COUNT:
        status = STOP_ROUTE_UNSUPPORTED
        selected = None
    else:
        status = RECALLED
        selected = race.winner_cell
    return _RouteRace(
        status=status,
        raw_scores=(int(raw_array[0]), int(raw_array[1])),
        selected_route=selected,
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_candidates,
    )


def _consequence_race(
    exact_active_cells: tuple[int, ...],
    semantic_active_cells: tuple[int, ...],
    observed_output_cells: tuple[int, ...],
    *,
    comparator_gate: bool,
) -> _ConsequenceRace:
    candidates = (exact_active_cells, semantic_active_cells)
    equality_currents = np.asarray(
        [
            _SCORE_CURRENT_TABLE[1]
            if comparator_gate and observed in active
            else 0.0
            for active in candidates
            for observed in observed_output_cells
        ],
        dtype=np.float64,
    )
    equality_spikes = _SCORE_RACE.accumulated_spike_mask(equality_currents).reshape(
        ROUTE_COUNT, OUTPUT_LANES
    )
    arrivals = np.count_nonzero(equality_spikes, axis=1).astype(np.int16)
    currents = np.concatenate(
        (
            _SCORE_CURRENT_TABLE[arrivals],
            np.asarray([_SCORE_CURRENT_TABLE[CONSEQUENCE_TONIC_SCORE]]),
        )
    )
    race = _SCORE_RACE.race_many(currents[None, :], require_unique=False)[0]
    target = (
        race.winner_cell
        if race.simultaneous_spike_candidates == 1
        and race.winner_cell < ROUTE_COUNT
        and arrivals[race.winner_cell] == CONSEQUENCE_MATCH_SCORE
        else None
    )
    return _ConsequenceRace(
        target_route=target,
        equality_spikes=tuple(
            tuple(bool(value) for value in row) for row in equality_spikes
        ),
        integrator_arrivals=(int(arrivals[0]), int(arrivals[1])),
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_candidates,
        cell_steps=CONSEQUENCE_CELL_STEPS,
    )


def _freeze_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    frozen = np.ascontiguousarray(value, dtype=dtype)
    frozen.setflags(write=False)
    return frozen


def _checkpoint_sha256(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    summary: FactorRouterTrainingSummary,
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> str:
    summary_record = dataclasses.asdict(summary)
    summary_record["checkpoint_sha256"] = None
    digest = sha256()
    digest.update(_CHECKPOINT_NAMESPACE)
    digest.update(CONFIG_SHA256.encode("ascii"))
    digest.update(child.checkpoint_sha256.encode("ascii"))
    digest.update(child.episodic_checkpoint_sha256.encode("ascii"))
    digest.update(child.semantic_checkpoint_sha256.encode("ascii"))
    digest.update(_canonical_json(summary_record))
    for name, array in arrays:
        digest.update(_array_frame(name, array))
    return digest.hexdigest()


class FrozenCommonFactorizedContextRoutedMemoryV2:
    """One immutable dual-view child plus factor-local route evidence."""

    __slots__ = (
        "_checkpoint_sha256",
        "_cue_cells",
        "_dispositions",
        "_evidence_context_cells",
        "_evidence_source_cells",
        "_excitatory",
        "_inhibitory",
        "_memory",
        "_provenance_factor_indices",
        "_provenance_relation_cells",
        "_provenance_roots",
        "_provenance_routes",
        "_summary",
    )

    def __init__(
        self,
        *,
        child: memory.FrozenCommonEpisodicTransformMemoryV2,
        cue_cells: np.ndarray,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        provenance_factor_indices: np.ndarray,
        provenance_relation_cells: np.ndarray,
        evidence_context_cells: np.ndarray,
        evidence_source_cells: np.ndarray,
        provenance_routes: np.ndarray,
        dispositions: np.ndarray,
        provenance_roots: np.ndarray,
        summary: FactorRouterTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("factorized context memories are builder-owned")
        if type(child) is not memory.FrozenCommonEpisodicTransformMemoryV2:
            raise TypeError("child must be the exact frozen V2 composite")
        if type(summary) is not FactorRouterTrainingSummary:
            raise TypeError("factorized router summary type changed")
        self._memory = child
        self._cue_cells = _freeze_array(cue_cells, CUE_DTYPE)
        self._excitatory = _freeze_array(excitatory, TICK_DTYPE)
        self._inhibitory = _freeze_array(inhibitory, TICK_DTYPE)
        self._provenance_factor_indices = _freeze_array(
            provenance_factor_indices, PROVENANCE_INDEX_DTYPE
        )
        self._provenance_relation_cells = _freeze_array(
            provenance_relation_cells, PROVENANCE_CELL_DTYPE
        )
        self._evidence_context_cells = _freeze_array(
            evidence_context_cells, PROVENANCE_CELL_DTYPE
        )
        self._evidence_source_cells = _freeze_array(
            evidence_source_cells, PROVENANCE_CELL_DTYPE
        )
        self._provenance_routes = _freeze_array(
            provenance_routes, PROVENANCE_CODE_DTYPE
        )
        self._dispositions = _freeze_array(dispositions, PROVENANCE_CODE_DTYPE)
        self._provenance_roots = _freeze_array(
            provenance_roots, PROVENANCE_ROOT_DTYPE
        )
        self._summary = summary
        self._checkpoint_sha256 = _checkpoint_sha256(
            self._memory, self._summary, self._checkpoint_arrays()
        )
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise RuntimeError("factorized router checkpoint identity changed")
        self.authenticate()

    def _checkpoint_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("factor-cue-cells", self._cue_cells),
            ("route-excitatory", self._excitatory),
            ("route-inhibitory", self._inhibitory),
            ("provenance-factor-indices", self._provenance_factor_indices),
            ("provenance-relation-cells", self._provenance_relation_cells),
            ("provenance-context-cells", self._evidence_context_cells),
            ("provenance-source-cells", self._evidence_source_cells),
            ("provenance-routes", self._provenance_routes),
            ("provenance-dispositions", self._dispositions),
            ("provenance-roots", self._provenance_roots),
        )

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def memory_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._memory.checkpoint_sha256

    @property
    def episodic_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._memory.episodic_checkpoint_sha256

    @property
    def semantic_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._memory.semantic_checkpoint_sha256

    @property
    def training_summary(self) -> FactorRouterTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def provenance(self) -> tuple[FactorRouteProvenance, ...]:
        self.authenticate()
        labels = {1: ACCEPTED_VOTE, 2: DUPLICATE_NO_VOTE, 3: DEPENDENT_NO_VOTE}
        return tuple(
            FactorRouteProvenance(
                factor_index=int(self._provenance_factor_indices[row]),
                target_route=int(self._provenance_routes[row]),
                disposition=labels[int(self._dispositions[row])],
                relation_cells=tuple(
                    int(value) for value in self._provenance_relation_cells[row]
                ),
                evidence_context_cells=tuple(
                    int(value) for value in self._evidence_context_cells[row]
                ),
                evidence_source_cells=tuple(
                    int(value) for value in self._evidence_source_cells[row]
                ),
                sha256=bytes(self._provenance_roots[row]).hex(),
            )
            for row in range(self._provenance_roots.shape[0])
        )

    @property
    def learned_state_shapes(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        self.authenticate()
        return tuple((name, array.shape) for name, array in self._checkpoint_arrays())

    def authenticate(self) -> None:
        self._memory.authenticate()
        arrays = self._checkpoint_arrays()
        if any(array.flags.writeable for _, array in arrays):
            raise RuntimeError("factorized router checkpoint array became mutable")
        factor_count = self._cue_cells.shape[0]
        provenance_count = self._provenance_roots.shape[0]
        expected_shapes = (
            (factor_count, TRACE_LANES),
            (factor_count, ROUTE_COUNT),
            (factor_count, ROUTE_COUNT),
            (provenance_count,),
            (provenance_count, TRACE_LANES),
            (provenance_count, TRACE_LANES),
            (provenance_count, TRACE_LANES),
            (provenance_count,),
            (provenance_count,),
            (provenance_count, 32),
        )
        if tuple(array.shape for _, array in arrays) != expected_shapes:
            raise RuntimeError("factorized router checkpoint shape changed")
        if factor_count > MAX_FACTORS or provenance_count > MAX_PROVENANCE_ROWS:
            raise RuntimeError("factorized router checkpoint capacity changed")
        if factor_count != len({tuple(row) for row in self._cue_cells.tolist()}):
            raise RuntimeError("factorized router contains duplicate factor assemblies")
        if np.any(self._excitatory > TICK_CAP) or np.any(
            self._inhibitory > TICK_CAP
        ):
            raise RuntimeError("factorized router tick cap changed")
        if provenance_count and (
            np.any(self._provenance_factor_indices < 0)
            or np.any(self._provenance_factor_indices >= factor_count)
            or np.any(self._provenance_routes >= ROUTE_COUNT)
            or np.any(self._dispositions < 1)
            or np.any(self._dispositions > 3)
        ):
            raise RuntimeError("factorized router provenance address changed")
        trainable_bytes = sum(array.nbytes for _, array in arrays[:3])
        provenance_bytes = sum(array.nbytes for _, array in arrays[3:])
        accepted = int(np.count_nonzero(self._dispositions == 1))
        duplicate = int(np.count_nonzero(self._dispositions == 2))
        dependent = int(np.count_nonzero(self._dispositions == 3))
        summary = self._summary
        if (
            summary.memory_checkpoint_sha256 != self._memory.checkpoint_sha256
            or summary.episodic_checkpoint_sha256
            != self._memory.episodic_checkpoint_sha256
            or summary.semantic_checkpoint_sha256
            != self._memory.semantic_checkpoint_sha256
            or summary.factor_count != factor_count
            or summary.provenance_count != provenance_count
            or summary.accepted_factor_vote_count != accepted
            or summary.duplicate_nonvote_count != duplicate
            or summary.dependent_nonvote_count != dependent
            or summary.trainable_state_bytes != trainable_bytes
            or summary.provenance_state_bytes != provenance_bytes
            or summary.checkpoint_array_bytes != trainable_bytes + provenance_bytes
            or summary.fixed_router_cell_count != FIXED_ROUTER_CELL_COUNT
            or summary.factor_match_cell_count != factor_count
            or summary.added_cell_count != FIXED_ROUTER_CELL_COUNT + factor_count
            or summary.parent_lesson_replay_count != 0
            or summary.retained_router_parent_count != 0
            or summary.raw_lesson_retention_count != 0
            or summary.child_mutation_count != 0
            or _checkpoint_sha256(self._memory, summary, arrays)
            != self._checkpoint_sha256
        ):
            raise RuntimeError("factorized router checkpoint authentication failed")

    def recall(
        self,
        factor_cells: tuple[tuple[int, ...], ...],
        exact_observation_events: tuple[bytes, ...],
        semantic_input_frame: filler_projection.CommonOneByteFillerFrame,
        relation_cells: tuple[int, ...],
        *,
        cell_gate: bool = True,
        factor_gate: bool = True,
        router_gate: bool = True,
        output_relay_gate: bool = True,
        semantic_relation_gate: bool = True,
    ) -> FactorizedContextRecall:
        self.authenticate()
        factors = _canonical_factor_cells(factor_cells)
        _validate_observation_events(exact_observation_events)
        _FILLER_CELLS.authenticate_frame(semantic_input_frame)
        _validate_trace_cells("relation_cells", relation_cells)
        if any(
            type(gate) is not bool
            for gate in (
                cell_gate,
                factor_gate,
                router_gate,
                output_relay_gate,
                semantic_relation_gate,
            )
        ):
            raise TypeError("recall gates must be exact booleans")

        child_root = self._memory.checkpoint_sha256
        exact_root = self._memory.episodic_checkpoint_sha256
        semantic_root = self._memory.semantic_checkpoint_sha256
        matches = tuple(
            _factor_match(
                self._cue_cells,
                cells,
                cell_gate=cell_gate and factor_gate,
                tonic_score=FACTOR_TONIC_SCORE,
            )
            for cells in factors
        )
        all_recalled = all(match.status == RECALLED for match in matches)
        matched_indices = tuple(
            int(match.factor_index)
            for match in matches
            if match.factor_index is not None
        )
        exact_recall = self._memory.recall_episode(
            exact_observation_events, cell_gate=cell_gate
        )
        semantic_recall = self._memory.recall_semantic(
            semantic_input_frame,
            relation_cells,
            cell_gate=cell_gate,
            relation_gate=semantic_relation_gate,
        )
        route = _route_race(
            self._excitatory,
            self._inhibitory,
            matched_indices if all_recalled else (),
            router_gate=cell_gate and factor_gate and router_gate and all_recalled,
        )

        child_active = (
            exact_recall.output.active_cells,
            semantic_recall.output.active_cells,
        )
        output_candidates = tuple(
            output_projection.OUTPUT_CELL_OFFSET + candidate
            for candidate in range(OUTPUT_CANDIDATE_COUNT)
        )
        route_spikes = tuple(
            bool(
                cell_gate
                and factor_gate
                and router_gate
                and route.selected_route == candidate_route
            )
            for candidate_route in range(ROUTE_COUNT)
        )
        and_spikes = tuple(
            bool(
                output_relay_gate
                and route_spikes[candidate_route]
                and output_cell in child_active[candidate_route]
            )
            for candidate_route in range(ROUTE_COUNT)
            for output_cell in output_candidates
        )
        or_spikes = tuple(
            any(
                and_spikes[candidate_route * OUTPUT_CANDIDATE_COUNT + candidate]
                for candidate_route in range(ROUTE_COUNT)
            )
            for candidate in range(OUTPUT_CANDIDATE_COUNT)
        )
        raw_scores = np.where(
            np.asarray(or_spikes, dtype=np.bool_), RELAY_ACTIVE_SCORE, 0
        ).reshape(OUTPUT_LANES, OUTPUT_RADIX)
        output = _OUTPUT_CELLS.race_output(
            tuple(tuple(int(value) for value in row) for row in raw_scores)
        )

        if not cell_gate:
            recall_status = STOP_CELL_OFF
        elif not factor_gate:
            recall_status = STOP_FACTOR_GATE
        elif any(match.status == STOP_FACTOR_AMBIGUOUS for match in matches):
            recall_status = STOP_FACTOR_AMBIGUOUS
        elif any(match.status == STOP_FACTOR_UNKNOWN for match in matches):
            recall_status = STOP_FACTOR_UNKNOWN
        elif route.status != RECALLED:
            recall_status = route.status
        elif not output_relay_gate:
            recall_status = STOP_OUTPUT_RELAY
        elif output.decision == "TOKEN":
            recall_status = RECALLED
        else:
            recall_status = STOP_SELECTED_CHILD

        work = FactorRouterRecallWork(
            factor_assembly_reads=len(factors) * TRACE_LANES,
            factor_match_equality_reads=sum(match.equality_reads for match in matches),
            factor_match_equality_arrivals=sum(
                match.equality_arrivals for match in matches
            ),
            factor_match_cell_steps=sum(match.cell_steps for match in matches),
            route_cell_steps=ROUTE_RACE_CELL_STEPS,
            output_and_relay_cell_steps=OUTPUT_AND_RELAY_CELL_STEPS,
            output_or_relay_cell_steps=OUTPUT_OR_RELAY_CELL_STEPS,
            output_race_cell_steps=output.physical_cell_steps,
            exact_child_reported_work=exact_recall.work.total,
            semantic_child_cell_steps=semantic_recall.physical_cell_steps,
        )
        record = {
            "status": recall_status,
            "output": output.record(),
            "factor_cells": factors,
            "factor_indices": tuple(match.factor_index for match in matches),
            "factor_match_arrival_counts": tuple(match.arrivals for match in matches),
            "route_raw_scores": route.raw_scores,
            "selected_route": route.selected_route,
            "route_spike_step": route.spike_step,
            "route_simultaneous_spike_count": route.simultaneous_spike_count,
            "exact_child_active_cells": child_active[EXACT_ROUTE],
            "semantic_child_active_cells": child_active[SEMANTIC_ROUTE],
            "output_and_relay_spikes": and_spikes,
            "output_or_relay_spikes": or_spikes,
            "work": work.record(),
            "memory_checkpoint_sha256": child_root,
            "episodic_checkpoint_sha256": exact_root,
            "semantic_checkpoint_sha256": semantic_root,
            "checkpoint_sha256": self._checkpoint_sha256,
            "child_token_or_status_used_for_selection": False,
            "selected_child_fallback": False,
            "host_factor_order_authority": False,
        }
        result = FactorizedContextRecall(
            status=recall_status,
            output=output,
            factor_cells=factors,
            factor_indices=tuple(match.factor_index for match in matches),
            factor_match_arrival_counts=tuple(match.arrivals for match in matches),
            route_raw_scores=route.raw_scores,
            selected_route=route.selected_route,
            selected_route_cell_id=(
                None
                if route.selected_route is None
                else ROUTE_ACTION_CELL_OFFSET + route.selected_route
            ),
            route_spike_step=route.spike_step,
            route_simultaneous_spike_count=route.simultaneous_spike_count,
            exact_child_active_cells=child_active[EXACT_ROUTE],
            semantic_child_active_cells=child_active[SEMANTIC_ROUTE],
            output_and_relay_spikes=and_spikes,
            output_or_relay_spikes=or_spikes,
            work=work,
            memory_checkpoint_sha256=child_root,
            episodic_checkpoint_sha256=exact_root,
            semantic_checkpoint_sha256=semantic_root,
            checkpoint_sha256=self._checkpoint_sha256,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )
        self._memory.authenticate()
        if (
            self._memory.checkpoint_sha256 != child_root
            or self._memory.episodic_checkpoint_sha256 != exact_root
            or self._memory.semantic_checkpoint_sha256 != semantic_root
        ):
            raise RuntimeError("factorized router recall mutated its V2 child")
        return result


def _empty_arrays() -> tuple[np.ndarray, ...]:
    return (
        np.empty((0, TRACE_LANES), dtype=CUE_DTYPE),
        np.empty((0, ROUTE_COUNT), dtype=TICK_DTYPE),
        np.empty((0, ROUTE_COUNT), dtype=TICK_DTYPE),
        np.empty((0,), dtype=PROVENANCE_INDEX_DTYPE),
        np.empty((0, TRACE_LANES), dtype=PROVENANCE_CELL_DTYPE),
        np.empty((0, TRACE_LANES), dtype=PROVENANCE_CELL_DTYPE),
        np.empty((0, TRACE_LANES), dtype=PROVENANCE_CELL_DTYPE),
        np.empty((0,), dtype=PROVENANCE_CODE_DTYPE),
        np.empty((0,), dtype=PROVENANCE_CODE_DTYPE),
        np.empty((0, 32), dtype=PROVENANCE_ROOT_DTYPE),
    )


def _validate_lessons(lessons: tuple[PhysicalFactorRouteLesson, ...]) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not PhysicalFactorRouteLesson for lesson in lessons
    ):
        raise TypeError("lessons must be an exact tuple of PhysicalFactorRouteLesson")


def _apply_lessons(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    arrays: tuple[np.ndarray, ...],
    lessons: tuple[PhysicalFactorRouteLesson, ...],
) -> tuple[tuple[np.ndarray, ...], _BatchTotals]:
    (
        cue_cells,
        excitatory,
        inhibitory,
        provenance_factor_indices,
        provenance_relation_cells,
        evidence_context_cells,
        evidence_source_cells,
        provenance_routes,
        dispositions,
        provenance_roots,
    ) = (np.array(array, copy=True) for array in arrays)
    teacher_off_count = 0
    cell_off_count = 0
    comparator_off_count = 0
    consequence_not_singleton_count = 0
    excitatory_ticks = 0
    inhibitory_ticks = 0
    factor_assembly_reads = 0
    router_cell_steps = 0
    exact_child_work = 0
    semantic_child_steps = 0
    steps: list[FactorRouteLearningStep] = []

    child_root = child.checkpoint_sha256
    exact_root = child.episodic_checkpoint_sha256
    semantic_root = child.semantic_checkpoint_sha256
    for ordinal, lesson in enumerate(lessons):
        if not lesson.teacher_gate:
            teacher_off_count += 1
            steps.append(
                FactorRouteLearningStep(
                    ordinal, STOP_TEACHER_OFF, (), (), None, 0, 0, 0, 0
                )
            )
            continue
        if not lesson.cell_gate:
            cell_off_count += 1
            steps.append(
                FactorRouteLearningStep(
                    ordinal, STOP_CELL_OFF, (), (), None, 0, 0, 0, 0
                )
            )
            continue

        exact_recall = child.recall_episode(
            lesson.exact_observation_events, cell_gate=True
        )
        semantic_recall = child.recall_semantic(
            lesson.semantic_input_frame,
            lesson.relation_cells,
            cell_gate=True,
            relation_gate=True,
        )
        lesson_exact_work = exact_recall.work.total
        lesson_semantic_steps = semantic_recall.physical_cell_steps
        exact_child_work += lesson_exact_work
        semantic_child_steps += lesson_semantic_steps
        consequence = _consequence_race(
            exact_recall.output.active_cells,
            semantic_recall.output.active_cells,
            lesson.observed_output_cells,
            comparator_gate=lesson.comparator_gate,
        )
        router_cell_steps += consequence.cell_steps
        if not lesson.comparator_gate:
            comparator_off_count += 1
            steps.append(
                FactorRouteLearningStep(
                    ordinal,
                    STOP_COMPARATOR_OFF,
                    (),
                    (),
                    None,
                    0,
                    lesson_exact_work,
                    lesson_semantic_steps,
                    consequence.cell_steps,
                )
            )
            continue
        if consequence.target_route is None:
            consequence_not_singleton_count += 1
            steps.append(
                FactorRouteLearningStep(
                    ordinal,
                    STOP_CONSEQUENCE_NOT_SINGLETON,
                    (),
                    (),
                    None,
                    0,
                    lesson_exact_work,
                    lesson_semantic_steps,
                    consequence.cell_steps,
                )
            )
            continue

        factors = _canonical_factor_cells(lesson.factor_cells)
        factor_assembly_reads += len(factors) * TRACE_LANES
        factor_indices: list[int] = []
        dispositions_for_step: list[str] = []
        committed = 0
        lesson_match_steps = 0
        for cells in factors:
            match = _factor_match(
                cue_cells,
                cells,
                cell_gate=True,
                tonic_score=FACTOR_LEARNING_TONIC_SCORE,
            )
            lesson_match_steps += match.cell_steps
            if match.status == STOP_FACTOR_AMBIGUOUS:
                dispositions_for_step.append(STOP_FACTOR_AMBIGUOUS)
                continue
            factor_index = match.factor_index
            if factor_index is None:
                if not match.novel:
                    raise RuntimeError("factor allocation authority changed")
                if cue_cells.shape[0] == MAX_FACTORS:
                    raise RuntimeError("factorized router factor capacity exhausted")
                factor_index = cue_cells.shape[0]
                cue_cells = np.concatenate(
                    (
                        cue_cells,
                        np.asarray([cells], dtype=CUE_DTYPE),
                    ),
                    axis=0,
                )
                excitatory = np.concatenate(
                    (excitatory, np.zeros((1, ROUTE_COUNT), dtype=TICK_DTYPE)),
                    axis=0,
                )
                inhibitory = np.concatenate(
                    (inhibitory, np.zeros((1, ROUTE_COUNT), dtype=TICK_DTYPE)),
                    axis=0,
                )
            factor_indices.append(int(factor_index))

            accepted_rows = np.flatnonzero(
                (provenance_factor_indices == factor_index) & (dispositions == 1)
            )
            relation = np.asarray(
                lesson.relation_cells, dtype=PROVENANCE_CELL_DTYPE
            )
            context = np.asarray(
                lesson.evidence_context_cells, dtype=PROVENANCE_CELL_DTYPE
            )
            source = np.asarray(
                lesson.evidence_source_cells, dtype=PROVENANCE_CELL_DTYPE
            )
            duplicate = any(
                np.array_equal(provenance_relation_cells[row], relation)
                and np.array_equal(evidence_context_cells[row], context)
                and np.array_equal(evidence_source_cells[row], source)
                for row in accepted_rows
            )
            dependent = not duplicate and any(
                np.array_equal(provenance_relation_cells[row], relation)
                or np.array_equal(evidence_context_cells[row], context)
                or np.array_equal(evidence_source_cells[row], source)
                for row in accepted_rows
            )
            if duplicate:
                disposition_code = 2
                disposition_label = DUPLICATE_NO_VOTE
            elif dependent:
                disposition_code = 3
                disposition_label = DEPENDENT_NO_VOTE
            else:
                disposition_code = 1
                disposition_label = ACCEPTED_VOTE
                committed += 1
                target = int(consequence.target_route)
                rival = 1 - target
                before = int(excitatory[factor_index, target])
                excitatory[factor_index, target] = min(TICK_CAP, before + 1)
                excitatory_ticks += int(excitatory[factor_index, target]) - before
                if int(excitatory[factor_index, rival]) > 0:
                    before = int(inhibitory[factor_index, rival])
                    inhibitory[factor_index, rival] = min(TICK_CAP, before + 1)
                    inhibitory_ticks += int(inhibitory[factor_index, rival]) - before
            dispositions_for_step.append(disposition_label)

            provenance_record = {
                "memory_checkpoint_sha256": child_root,
                "episodic_checkpoint_sha256": exact_root,
                "semantic_checkpoint_sha256": semantic_root,
                "factor_cells_sha256": _digest(b"physical-factor-cells\0", cells),
                "factor_cells": cells,
                "factor_index": factor_index,
                "relation_cells": lesson.relation_cells,
                "evidence_context_cells": lesson.evidence_context_cells,
                "evidence_source_cells": lesson.evidence_source_cells,
                "observed_output_cells": lesson.observed_output_cells,
                "exact_child_active_cells": exact_recall.output.active_cells,
                "semantic_child_active_cells": semantic_recall.output.active_cells,
                "consequence_equality_spikes": consequence.equality_spikes,
                "consequence_integrator_arrivals": consequence.integrator_arrivals,
                "target_route": consequence.target_route,
                "disposition": disposition_label,
                "route_excitatory_after": tuple(
                    int(value) for value in excitatory[factor_index]
                ),
                "route_inhibitory_after": tuple(
                    int(value) for value in inhibitory[factor_index]
                ),
                "child_token_or_status_used_for_selection": False,
            }
            provenance_sha = _digest(_PROVENANCE_NAMESPACE, provenance_record)
            if provenance_roots.shape[0] == MAX_PROVENANCE_ROWS:
                raise RuntimeError("factorized router provenance capacity exhausted")
            provenance_factor_indices = np.append(
                provenance_factor_indices,
                np.asarray(factor_index, dtype=PROVENANCE_INDEX_DTYPE),
            )
            provenance_relation_cells = np.concatenate(
                (provenance_relation_cells, relation[None, :]), axis=0
            )
            evidence_context_cells = np.concatenate(
                (evidence_context_cells, context[None, :]), axis=0
            )
            evidence_source_cells = np.concatenate(
                (evidence_source_cells, source[None, :]), axis=0
            )
            provenance_routes = np.append(
                provenance_routes,
                np.asarray(consequence.target_route, dtype=PROVENANCE_CODE_DTYPE),
            )
            dispositions = np.append(
                dispositions,
                np.asarray(disposition_code, dtype=PROVENANCE_CODE_DTYPE),
            )
            provenance_roots = np.concatenate(
                (
                    provenance_roots,
                    np.frombuffer(bytes.fromhex(provenance_sha), dtype=np.uint8)[
                        None, :
                    ],
                ),
                axis=0,
            )

        router_cell_steps += lesson_match_steps
        step_status = (
            ACCEPTED_VOTE
            if committed
            else (
                dispositions_for_step[0]
                if dispositions_for_step
                and len(set(dispositions_for_step)) == 1
                else STOP_FACTOR_AMBIGUOUS
            )
        )
        steps.append(
            FactorRouteLearningStep(
                ordinal,
                step_status,
                tuple(factor_indices),
                tuple(dispositions_for_step),
                consequence.target_route,
                committed,
                lesson_exact_work,
                lesson_semantic_steps,
                consequence.cell_steps + lesson_match_steps,
            )
        )

    child.authenticate()
    if (
        child.checkpoint_sha256 != child_root
        or child.episodic_checkpoint_sha256 != exact_root
        or child.semantic_checkpoint_sha256 != semantic_root
    ):
        raise RuntimeError("factorized router learning mutated its V2 child")
    result_arrays = (
        cue_cells,
        excitatory,
        inhibitory,
        provenance_factor_indices,
        provenance_relation_cells,
        evidence_context_cells,
        evidence_source_cells,
        provenance_routes,
        dispositions,
        provenance_roots,
    )
    totals = _BatchTotals(
        teacher_off_count=teacher_off_count,
        cell_off_count=cell_off_count,
        comparator_off_count=comparator_off_count,
        consequence_not_singleton_count=consequence_not_singleton_count,
        excitatory_update_ticks=excitatory_ticks,
        inhibitory_update_ticks=inhibitory_ticks,
        factor_assembly_reads=factor_assembly_reads,
        router_learning_cell_steps=router_cell_steps,
        exact_child_reported_work=exact_child_work,
        semantic_child_cell_steps=semantic_child_steps,
        steps=tuple(steps),
    )
    return result_arrays, totals


def _freeze(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    arrays: tuple[np.ndarray, ...],
    batch: _BatchTotals,
    *,
    mode: str,
    new_lesson_count: int,
    previous: FactorRouterTrainingSummary | None,
) -> FrozenCommonFactorizedContextRoutedMemoryV2:
    factor_count = arrays[0].shape[0]
    provenance_count = arrays[-1].shape[0]
    trainable_bytes = sum(array.nbytes for array in arrays[:3])
    provenance_bytes = sum(array.nbytes for array in arrays[3:])
    base = previous

    def prior(name: str) -> int:
        return 0 if base is None else int(getattr(base, name))

    summary_record = {
        "mode": mode,
        "new_lesson_count": new_lesson_count,
        "total_lesson_count": prior("total_lesson_count") + new_lesson_count,
        "teacher_off_count": prior("teacher_off_count") + batch.teacher_off_count,
        "cell_off_count": prior("cell_off_count") + batch.cell_off_count,
        "comparator_off_count": prior("comparator_off_count")
        + batch.comparator_off_count,
        "consequence_not_singleton_count": prior(
            "consequence_not_singleton_count"
        )
        + batch.consequence_not_singleton_count,
        "accepted_factor_vote_count": int(np.count_nonzero(arrays[8] == 1)),
        "duplicate_nonvote_count": int(np.count_nonzero(arrays[8] == 2)),
        "dependent_nonvote_count": int(np.count_nonzero(arrays[8] == 3)),
        "factor_count": factor_count,
        "provenance_count": provenance_count,
        "excitatory_update_ticks": prior("excitatory_update_ticks")
        + batch.excitatory_update_ticks,
        "inhibitory_update_ticks": prior("inhibitory_update_ticks")
        + batch.inhibitory_update_ticks,
        "factor_assembly_reads": prior("factor_assembly_reads")
        + batch.factor_assembly_reads,
        "router_learning_cell_steps": prior("router_learning_cell_steps")
        + batch.router_learning_cell_steps,
        "exact_child_reported_work": prior("exact_child_reported_work")
        + batch.exact_child_reported_work,
        "semantic_child_cell_steps": prior("semantic_child_cell_steps")
        + batch.semantic_child_cell_steps,
        "trainable_state_bytes": trainable_bytes,
        "provenance_state_bytes": provenance_bytes,
        "checkpoint_array_bytes": trainable_bytes + provenance_bytes,
        "fixed_router_cell_count": FIXED_ROUTER_CELL_COUNT,
        "factor_match_cell_count": factor_count,
        "added_cell_count": FIXED_ROUTER_CELL_COUNT + factor_count,
        "parent_lesson_replay_count": 0,
        "retained_router_parent_count": 0,
        "raw_lesson_retention_count": 0,
        "child_mutation_count": 0,
        "memory_checkpoint_sha256": child.checkpoint_sha256,
        "episodic_checkpoint_sha256": child.episodic_checkpoint_sha256,
        "semantic_checkpoint_sha256": child.semantic_checkpoint_sha256,
        "last_steps": batch.steps,
    }
    placeholder = FactorRouterTrainingSummary(
        **summary_record, checkpoint_sha256=""
    )
    named_arrays = (
        ("factor-cue-cells", arrays[0]),
        ("route-excitatory", arrays[1]),
        ("route-inhibitory", arrays[2]),
        ("provenance-factor-indices", arrays[3]),
        ("provenance-relation-cells", arrays[4]),
        ("provenance-context-cells", arrays[5]),
        ("provenance-source-cells", arrays[6]),
        ("provenance-routes", arrays[7]),
        ("provenance-dispositions", arrays[8]),
        ("provenance-roots", arrays[9]),
    )
    checkpoint = _checkpoint_sha256(child, placeholder, named_arrays)
    summary = dataclasses.replace(placeholder, checkpoint_sha256=checkpoint)
    return FrozenCommonFactorizedContextRoutedMemoryV2(
        child=child,
        cue_cells=arrays[0],
        excitatory=arrays[1],
        inhibitory=arrays[2],
        provenance_factor_indices=arrays[3],
        provenance_relation_cells=arrays[4],
        evidence_context_cells=arrays[5],
        evidence_source_cells=arrays[6],
        provenance_routes=arrays[7],
        dispositions=arrays[8],
        provenance_roots=arrays[9],
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def fit_common_factorized_context_routed_memory_v2(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    lessons: tuple[PhysicalFactorRouteLesson, ...],
) -> FrozenCommonFactorizedContextRoutedMemoryV2:
    """Fit factor-local route evidence over one frozen dual-view child."""

    if type(child) is not memory.FrozenCommonEpisodicTransformMemoryV2:
        raise TypeError("child must be the exact frozen V2 composite")
    _validate_lessons(lessons)
    child.authenticate()
    arrays, batch = _apply_lessons(child, _empty_arrays(), lessons)
    return _freeze(
        child,
        arrays,
        batch,
        mode="INITIAL_FIT",
        new_lesson_count=len(lessons),
        previous=None,
    )


def advance_common_factorized_context_routed_memory_v2(
    parent: FrozenCommonFactorizedContextRoutedMemoryV2,
    new_lessons: tuple[PhysicalFactorRouteLesson, ...],
) -> FrozenCommonFactorizedContextRoutedMemoryV2:
    """Consume only new factor lessons and return a flat successor."""

    if type(parent) is not FrozenCommonFactorizedContextRoutedMemoryV2:
        raise TypeError("parent must be one exact frozen factorized router")
    _validate_lessons(new_lessons)
    parent.authenticate()
    parent_root = parent.checkpoint_sha256
    child_root = parent.memory_checkpoint_sha256
    exact_root = parent.episodic_checkpoint_sha256
    semantic_root = parent.semantic_checkpoint_sha256
    arrays, batch = _apply_lessons(
        parent._memory,
        tuple(array for _, array in parent._checkpoint_arrays()),
        new_lessons,
    )
    successor = _freeze(
        parent._memory,
        arrays,
        batch,
        mode="REPLAY_FREE_NEW_LESSON_ADVANCE",
        new_lesson_count=len(new_lessons),
        previous=parent.training_summary,
    )
    parent.authenticate()
    if (
        parent.checkpoint_sha256 != parent_root
        or parent.memory_checkpoint_sha256 != child_root
        or parent.episodic_checkpoint_sha256 != exact_root
        or parent.semantic_checkpoint_sha256 != semantic_root
    ):
        raise RuntimeError("factorized router advance mutated its parent or child")
    return successor


SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "ACCEPTED_VOTE",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DEPENDENT_NO_VOTE",
    "DUPLICATE_NO_VOTE",
    "EXACT_ROUTE",
    "FACTOR_STATE_BYTES",
    "FactorRouteLearningStep",
    "FactorRouteProvenance",
    "FactorRouterRecallWork",
    "FactorRouterTrainingSummary",
    "FactorizedContextRecall",
    "FIXED_ROUTER_CELL_COUNT",
    "FrozenCommonFactorizedContextRoutedMemoryV2",
    "MAX_FACTORS",
    "MAX_FACTORS_PER_CONTEXT",
    "PhysicalFactorRouteLesson",
    "PROVENANCE_BYTES_PER_ROW",
    "RECALLED",
    "SEMANTIC_ROUTE",
    "SOURCE_SHA256",
    "STOP_CELL_OFF",
    "STOP_CONSEQUENCE_NOT_SINGLETON",
    "STOP_FACTOR_AMBIGUOUS",
    "STOP_FACTOR_GATE",
    "STOP_FACTOR_UNKNOWN",
    "STOP_OUTPUT_RELAY",
    "STOP_ROUTE_AMBIGUOUS",
    "STOP_ROUTE_UNSUPPORTED",
    "STOP_SELECTED_CHILD",
    "advance_common_factorized_context_routed_memory_v2",
    "fit_common_factorized_context_routed_memory_v2",
]
