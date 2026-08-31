"""Learn which frozen memory view to retrieve from an opaque query context.

The child is one immutable ``FrozenCommonEpisodicTransformMemoryV2``.  Every
training observation invokes its exact and semantic public recalls read-only.
Eight physical equality cells and two four-lane consequence integrators admit a
vote only when exactly one child matches the observed physical output.  A flat
content-addressed query bank then learns EXACT versus SEMANTIC with local E/H
ticks, tonic abstention, independent evidence, provenance, and revision.

Recall never reads either child token or status.  A physical query race and a
physical route race gate candidate output cells through fixed AND/OR relays into
one shared ``CommonEventCells`` output race.  A selected child STOP therefore
stays STOP; the other child is never a fallback.
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

SCOPE = "COMMON_CONTEXT_ROUTED_MEMORY_V1"
STATUS = "SELF_POSED_CONSEQUENCE_TRAINED_CONTEXT_RETRIEVAL"
CLAIM_BOUNDARY = (
    "self-posed-revisable-selection-between-two-banked-one-byte-memory-views-"
    "for-recurring-exact-query-context-traces-only-no-unseen-context-"
    "abstraction-language-coding-corpus-or-benchmark-evidence"
)

EXACT_ROUTE = 0
SEMANTIC_ROUTE = 1
ROUTE_COUNT = 2

RECALLED = "RECALLED"
STOP_CELL_OFF = "STOP_CONTEXT_ROUTER_CELL_OFF"
STOP_QUERY_UNKNOWN = "STOP_CONTEXT_ROUTER_QUERY_UNKNOWN"
STOP_QUERY_AMBIGUOUS = "STOP_CONTEXT_ROUTER_QUERY_AMBIGUOUS"
STOP_ROUTE_UNSUPPORTED = "STOP_CONTEXT_ROUTER_ROUTE_UNSUPPORTED"
STOP_ROUTE_AMBIGUOUS = "STOP_CONTEXT_ROUTER_ROUTE_AMBIGUOUS"
STOP_OUTPUT_RELAY = "STOP_CONTEXT_ROUTER_OUTPUT_RELAY_OFF"
STOP_SELECTED_CHILD = "STOP_CONTEXT_ROUTER_SELECTED_CHILD"
STOP_CONSEQUENCE_NOT_SINGLETON = (
    "STOP_CONTEXT_ROUTER_CONSEQUENCE_NOT_SINGLETON"
)
STOP_COMPARATOR_OFF = "STOP_CONTEXT_ROUTER_COMPARATOR_OFF"
STOP_TEACHER_OFF = "STOP_CONTEXT_ROUTER_TEACHER_OFF"

ACCEPTED_VOTE = "ACCEPTED_VOTE"
DUPLICATE_NO_VOTE = "DUPLICATE_NO_VOTE"
DEPENDENT_NO_VOTE = "DEPENDENT_NO_VOTE"

MAX_QUERY_CONTEXTS = 16
MAX_PROVENANCE_ROWS = 256
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

QUERY_STATE_BYTES_PER_SCHEMA = (
    TRACE_LANES * CUE_DTYPE.itemsize
    + 2 * ROUTE_COUNT * TICK_DTYPE.itemsize
)
PROVENANCE_BYTES_PER_ROW = (
    PROVENANCE_INDEX_DTYPE.itemsize
    + 3 * TRACE_LANES * PROVENANCE_CELL_DTYPE.itemsize
    + 2 * PROVENANCE_CODE_DTYPE.itemsize
    + 32 * PROVENANCE_ROOT_DTYPE.itemsize
)

RACE_STEPS = 600
ZERO_SCORE_SPIKE_STEP = 568
QUERY_MATCH_SCORE = TRACE_LANES
QUERY_TONIC_SCORE = QUERY_MATCH_SCORE - 1
CONSEQUENCE_MATCH_SCORE = OUTPUT_LANES
CONSEQUENCE_TONIC_SCORE = CONSEQUENCE_MATCH_SCORE - 1
ROUTE_TONIC_SCORE = 1
MAX_RACE_SCORE = QUERY_MATCH_SCORE
RELAY_ACTIVE_SCORE = 2 * output_projection.STOP_SCORE

CONSEQUENCE_EQUALITY_CELL_OFFSET = 0
CONSEQUENCE_EQUALITY_CELL_COUNT = ROUTE_COUNT * OUTPUT_LANES
CONSEQUENCE_INTEGRATOR_CELL_OFFSET = (
    CONSEQUENCE_EQUALITY_CELL_OFFSET + CONSEQUENCE_EQUALITY_CELL_COUNT
)
CONSEQUENCE_INTEGRATOR_CELL_COUNT = ROUTE_COUNT
CONSEQUENCE_TONIC_CELL_ID = (
    CONSEQUENCE_INTEGRATOR_CELL_OFFSET + CONSEQUENCE_INTEGRATOR_CELL_COUNT
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
QUERY_TONIC_CELL_ID = OUTPUT_OR_RELAY_CELL_OFFSET + OUTPUT_OR_RELAY_CELL_COUNT
QUERY_MATCH_CELL_OFFSET = QUERY_TONIC_CELL_ID + 1
FIXED_ROUTER_CELL_COUNT = QUERY_MATCH_CELL_OFFSET

CONSEQUENCE_EQUALITY_CELL_STEPS = CONSEQUENCE_EQUALITY_CELL_COUNT * RACE_STEPS
CONSEQUENCE_INTEGRATOR_CELL_STEPS = (
    (CONSEQUENCE_INTEGRATOR_CELL_COUNT + 1) * RACE_STEPS
)
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
_CONFIG_NAMESPACE = b"common-context-routed-memory-v1-config\0"
_CHECKPOINT_NAMESPACE = b"common-context-routed-memory-v1-checkpoint\0"
_PROVENANCE_NAMESPACE = b"common-context-routed-memory-v1-provenance\0"
_RECALL_NAMESPACE = b"common-context-routed-memory-v1-recall\0"
_CONSTRUCTION_TOKEN = object()
_QUERY_CELLS = output_projection.CommonEventCells()
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
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
    crossing_current = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(MAX_RACE_SCORE + 1):
        first_step = ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("context router fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        row.spike_step != ZERO_SCORE_SPIKE_STEP - score
        for score, row in enumerate(controls)
    ):
        raise RuntimeError("context router fitted-GLIF calibration changed")
    return table, race


_SCORE_CURRENT_TABLE, _SCORE_RACE = _build_score_current_table()

CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "child_composite_config_sha256": memory.CONFIG_SHA256,
    "child_access_surface": (
        "authenticate",
        "checkpoint roots",
        "recall_episode",
        "recall_semantic",
    ),
    "child_token_or_status_used_for_selection": False,
    "host_output_selection": False,
    "selected_child_fallback": False,
    "query_context_encoding": "twelve-lane-common-event-trace",
    "routes": ("EXACT", "SEMANTIC"),
    "maximum_query_contexts": MAX_QUERY_CONTEXTS,
    "maximum_provenance_rows": MAX_PROVENANCE_ROWS,
    "tick_cap": TICK_CAP,
    "race_steps": RACE_STEPS,
    "query_match_score": QUERY_MATCH_SCORE,
    "query_tonic_score": QUERY_TONIC_SCORE,
    "consequence_match_score": CONSEQUENCE_MATCH_SCORE,
    "consequence_tonic_score": CONSEQUENCE_TONIC_SCORE,
    "route_tonic_score": ROUTE_TONIC_SCORE,
    "fixed_router_cell_count": FIXED_ROUTER_CELL_COUNT,
    "query_state_bytes_per_schema": QUERY_STATE_BYTES_PER_SCHEMA,
    "provenance_bytes_per_row": PROVENANCE_BYTES_PER_ROW,
    "replay_count": 0,
    "retained_parent_count": 0,
    "raw_lesson_retention_count": 0,
    "public_dev_m0_c1_g0_path2_protected_heldout_inputs": (
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
class PhysicalContextRetrievalLesson:
    query_context_event: bytes
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
        _validate_event("query_context_event", self.query_context_event)
        _validate_observation_events(self.exact_observation_events)
        _FILLER_CELLS.authenticate_frame(self.semantic_input_frame)
        _validate_trace_cells("relation_cells", self.relation_cells)
        _validate_trace_cells("evidence_context_cells", self.evidence_context_cells)
        _validate_trace_cells("evidence_source_cells", self.evidence_source_cells)
        if self.evidence_context_cells == self.evidence_source_cells:
            raise ValueError("evidence context and source must be physically distinct")
        _validate_output_cells(self.observed_output_cells)
        if (
            type(self.teacher_gate) is not bool
            or type(self.cell_gate) is not bool
            or type(self.comparator_gate) is not bool
        ):
            raise TypeError("lesson gates must be exact booleans")


@dataclass(frozen=True, slots=True)
class ContextRouteProvenance:
    query_schema_index: int
    target_route: int
    disposition: str
    relation_cells: tuple[int, ...]
    evidence_context_cells: tuple[int, ...]
    evidence_source_cells: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ContextRouteLearningStep:
    lesson_ordinal: int
    status: str
    query_schema_index: int | None
    target_route: int | None
    vote_committed: bool
    provenance_sha256: str | None
    exact_child_reported_work: int
    semantic_child_cell_steps: int
    router_cell_steps: int


@dataclass(frozen=True, slots=True)
class ContextRouterTrainingSummary:
    mode: str
    new_lesson_count: int
    total_lesson_count: int
    teacher_off_count: int
    cell_off_count: int
    comparator_off_count: int
    consequence_not_singleton_count: int
    accepted_vote_count: int
    duplicate_nonvote_count: int
    dependent_nonvote_count: int
    query_schema_count: int
    provenance_count: int
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    query_projection_work: int
    router_learning_cell_steps: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int
    trainable_state_bytes: int
    provenance_state_bytes: int
    checkpoint_array_bytes: int
    fixed_router_cell_count: int
    query_match_cell_count: int
    added_cell_count: int
    parent_lesson_replay_count: int
    retained_router_parent_count: int
    raw_lesson_retention_count: int
    child_mutation_count: int
    memory_checkpoint_sha256: str
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    last_steps: tuple[ContextRouteLearningStep, ...]
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ContextRouterRecallWork:
    query_projection_work: int
    query_match_synaptic_equality_reads: int
    query_match_equality_arrivals: int
    query_match_cell_steps: int
    route_cell_steps: int
    output_and_relay_cell_steps: int
    output_or_relay_cell_steps: int
    output_race_cell_steps: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int

    @property
    def router_cell_steps(self) -> int:
        return (
            self.query_match_cell_steps
            + self.route_cell_steps
            + self.output_and_relay_cell_steps
            + self.output_or_relay_cell_steps
            + self.output_race_cell_steps
        )

    def record(self) -> dict[str, int]:
        return {**dataclasses.asdict(self), "router_cell_steps": self.router_cell_steps}


@dataclass(frozen=True, slots=True)
class ContextRoutedRecall:
    status: str
    output: output_projection.CommonOutputDecision
    query_context_cells: tuple[int, ...]
    query_schema_index: int | None
    query_schema_cell_id: int | None
    query_match_arrival_counts: tuple[int, ...]
    query_match_spike_step: int
    query_match_simultaneous_spike_count: int
    route_raw_scores: tuple[int, int]
    selected_route: int | None
    selected_route_cell_id: int | None
    route_spike_step: int
    route_simultaneous_spike_count: int
    exact_child_active_cells: tuple[int, ...]
    semantic_child_active_cells: tuple[int, ...]
    output_and_relay_spikes: tuple[bool, ...]
    output_or_relay_spikes: tuple[bool, ...]
    work: ContextRouterRecallWork
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
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class _ContextMatch:
    status: str
    query_schema_index: int | None
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
    query_projection_work: int
    router_learning_cell_steps: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int
    steps: tuple[ContextRouteLearningStep, ...]


def _context_match(
    cue_cells: np.ndarray,
    active_trace: tuple[int, ...],
    *,
    cell_gate: bool,
) -> _ContextMatch:
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
            np.asarray([_SCORE_CURRENT_TABLE[QUERY_TONIC_SCORE]]),
        )
    )
    race = _SCORE_RACE.race_many(currents[None, :], require_unique=False)[0]
    if race.simultaneous_spike_candidates != 1:
        status = STOP_QUERY_AMBIGUOUS
        query_index = None
        novel = False
    elif race.winner_cell == count:
        status = STOP_QUERY_UNKNOWN
        query_index = None
        novel = True
    else:
        status = RECALLED
        query_index = race.winner_cell
        novel = False
    return _ContextMatch(
        status=status,
        query_schema_index=query_index,
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
    query_schema_index: int | None,
    *,
    router_gate: bool,
) -> _RouteRace:
    if query_schema_index is None:
        raw = np.zeros((ROUTE_COUNT,), dtype=np.int16)
    else:
        raw = (
            excitatory[query_schema_index].astype(np.int16)
            - inhibitory[query_schema_index].astype(np.int16)
        )
    scores = np.maximum(raw, 0) if router_gate else np.zeros_like(raw)
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
        raw_scores=(int(raw[0]), int(raw[1])),
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
    candidate_cells = (exact_active_cells, semantic_active_cells)
    equality_currents = np.asarray(
        [
            (
                _SCORE_CURRENT_TABLE[1]
                if comparator_gate and observed in active
                else 0.0
            )
            for active in candidate_cells
            for observed in observed_output_cells
        ],
        dtype=np.float64,
    )
    equality_spikes_array = _SCORE_RACE.accumulated_spike_mask(equality_currents)
    equality_spikes_array = equality_spikes_array.reshape(
        ROUTE_COUNT, OUTPUT_LANES
    )
    arrivals = np.count_nonzero(equality_spikes_array, axis=1).astype(np.int16)
    integrator_currents = np.concatenate(
        (
            _SCORE_CURRENT_TABLE[arrivals],
            np.asarray([_SCORE_CURRENT_TABLE[CONSEQUENCE_TONIC_SCORE]]),
        )
    )
    race = _SCORE_RACE.race_many(
        integrator_currents[None, :], require_unique=False
    )[0]
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
            tuple(bool(value) for value in row) for row in equality_spikes_array
        ),
        integrator_arrivals=(int(arrivals[0]), int(arrivals[1])),
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_candidates,
        cell_steps=(
            CONSEQUENCE_EQUALITY_CELL_STEPS
            + CONSEQUENCE_INTEGRATOR_CELL_STEPS
        ),
    )


def _freeze_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    frozen = np.ascontiguousarray(value, dtype=dtype)
    frozen.setflags(write=False)
    return frozen


def _checkpoint_sha256(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    summary: ContextRouterTrainingSummary,
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


class FrozenCommonContextRoutedMemoryV1:
    """One immutable V2 child plus flat cue/action/provenance arrays."""

    __slots__ = (
        "_checkpoint_sha256",
        "_cue_cells",
        "_dispositions",
        "_evidence_context_cells",
        "_evidence_source_cells",
        "_excitatory",
        "_inhibitory",
        "_memory",
        "_provenance_query_indices",
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
        provenance_query_indices: np.ndarray,
        provenance_relation_cells: np.ndarray,
        evidence_context_cells: np.ndarray,
        evidence_source_cells: np.ndarray,
        provenance_routes: np.ndarray,
        dispositions: np.ndarray,
        provenance_roots: np.ndarray,
        summary: ContextRouterTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("context-routed memories are builder-owned")
        if type(child) is not memory.FrozenCommonEpisodicTransformMemoryV2:
            raise TypeError("child must be the exact frozen V2 composite")
        if type(summary) is not ContextRouterTrainingSummary:
            raise TypeError("context router summary type changed")
        self._memory = child
        self._cue_cells = _freeze_array(cue_cells, CUE_DTYPE)
        self._excitatory = _freeze_array(excitatory, TICK_DTYPE)
        self._inhibitory = _freeze_array(inhibitory, TICK_DTYPE)
        self._provenance_query_indices = _freeze_array(
            provenance_query_indices, PROVENANCE_INDEX_DTYPE
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
            raise RuntimeError("context router checkpoint identity changed")
        self.authenticate()

    def _checkpoint_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("query-cue-cells", self._cue_cells),
            ("route-excitatory", self._excitatory),
            ("route-inhibitory", self._inhibitory),
            ("provenance-query-indices", self._provenance_query_indices),
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
    def training_summary(self) -> ContextRouterTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def provenance(self) -> tuple[ContextRouteProvenance, ...]:
        self.authenticate()
        labels = {
            1: ACCEPTED_VOTE,
            2: DUPLICATE_NO_VOTE,
            3: DEPENDENT_NO_VOTE,
        }
        return tuple(
            ContextRouteProvenance(
                query_schema_index=int(self._provenance_query_indices[row]),
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
            raise RuntimeError("context router checkpoint array became mutable")
        query_count = self._cue_cells.shape[0]
        provenance_count = self._provenance_roots.shape[0]
        expected_shapes = (
            (query_count, TRACE_LANES),
            (query_count, ROUTE_COUNT),
            (query_count, ROUTE_COUNT),
            (provenance_count,),
            (provenance_count, TRACE_LANES),
            (provenance_count, TRACE_LANES),
            (provenance_count, TRACE_LANES),
            (provenance_count,),
            (provenance_count,),
            (provenance_count, 32),
        )
        if tuple(array.shape for _, array in arrays) != expected_shapes:
            raise RuntimeError("context router checkpoint shape changed")
        if query_count > MAX_QUERY_CONTEXTS or provenance_count > MAX_PROVENANCE_ROWS:
            raise RuntimeError("context router checkpoint capacity changed")
        if np.any(self._excitatory > TICK_CAP) or np.any(
            self._inhibitory > TICK_CAP
        ):
            raise RuntimeError("context router tick cap changed")
        if provenance_count and (
            np.any(self._provenance_query_indices < 0)
            or np.any(self._provenance_query_indices >= query_count)
            or np.any(self._provenance_routes >= ROUTE_COUNT)
            or np.any(self._dispositions < 1)
            or np.any(self._dispositions > 3)
        ):
            raise RuntimeError("context router provenance address changed")
        trainable_bytes = (
            self._cue_cells.nbytes
            + self._excitatory.nbytes
            + self._inhibitory.nbytes
        )
        provenance_bytes = sum(array.nbytes for _, array in arrays[3:])
        accepted = int(np.count_nonzero(self._dispositions == 1))
        duplicate = int(np.count_nonzero(self._dispositions == 2))
        dependent = int(np.count_nonzero(self._dispositions == 3))
        if (
            self._summary.memory_checkpoint_sha256
            != self._memory.checkpoint_sha256
            or self._summary.episodic_checkpoint_sha256
            != self._memory.episodic_checkpoint_sha256
            or self._summary.semantic_checkpoint_sha256
            != self._memory.semantic_checkpoint_sha256
            or self._summary.query_schema_count != query_count
            or self._summary.provenance_count != provenance_count
            or self._summary.accepted_vote_count != accepted
            or self._summary.duplicate_nonvote_count != duplicate
            or self._summary.dependent_nonvote_count != dependent
            or self._summary.trainable_state_bytes != trainable_bytes
            or self._summary.provenance_state_bytes != provenance_bytes
            or self._summary.checkpoint_array_bytes
            != trainable_bytes + provenance_bytes
            or self._summary.fixed_router_cell_count != FIXED_ROUTER_CELL_COUNT
            or self._summary.query_match_cell_count != query_count
            or self._summary.added_cell_count
            != FIXED_ROUTER_CELL_COUNT + query_count
            or self._summary.parent_lesson_replay_count != 0
            or self._summary.retained_router_parent_count != 0
            or self._summary.raw_lesson_retention_count != 0
            or self._summary.child_mutation_count != 0
            or _checkpoint_sha256(self._memory, self._summary, arrays)
            != self._checkpoint_sha256
        ):
            raise RuntimeError("context router authentication failed")

    def recall(
        self,
        query_context_event: bytes,
        exact_observation_events: tuple[bytes, ...],
        semantic_input_frame: filler_projection.CommonOneByteFillerFrame,
        relation_cells: tuple[int, ...],
        *,
        cell_gate: bool = True,
        router_gate: bool = True,
        output_relay_gate: bool = True,
        semantic_relation_gate: bool = True,
    ) -> ContextRoutedRecall:
        self.authenticate()
        _validate_event("query_context_event", query_context_event)
        _validate_observation_events(exact_observation_events)
        _FILLER_CELLS.authenticate_frame(semantic_input_frame)
        _validate_trace_cells("relation_cells", relation_cells)
        if any(
            type(gate) is not bool
            for gate in (
                cell_gate,
                router_gate,
                output_relay_gate,
                semantic_relation_gate,
            )
        ):
            raise TypeError("recall gates must be exact booleans")

        child_root = self._memory.checkpoint_sha256
        exact_root = self._memory.episodic_checkpoint_sha256
        semantic_root = self._memory.semantic_checkpoint_sha256
        query_frame = _QUERY_CELLS.project(query_context_event)
        context_match = _context_match(
            self._cue_cells,
            query_frame.trace_cells,
            cell_gate=cell_gate,
        )
        exact_recall = self._memory.recall_episode(
            exact_observation_events,
            cell_gate=cell_gate,
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
            context_match.query_schema_index,
            router_gate=(
                cell_gate
                and router_gate
                and context_match.query_schema_index is not None
            ),
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
        elif context_match.status != RECALLED:
            recall_status = context_match.status
        elif route.status != RECALLED:
            recall_status = route.status
        elif not output_relay_gate:
            recall_status = STOP_OUTPUT_RELAY
        elif output.decision == "TOKEN":
            recall_status = RECALLED
        else:
            recall_status = STOP_SELECTED_CHILD

        work = ContextRouterRecallWork(
            query_projection_work=query_frame.work.total,
            query_match_synaptic_equality_reads=context_match.equality_reads,
            query_match_equality_arrivals=context_match.equality_arrivals,
            query_match_cell_steps=context_match.cell_steps,
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
            "query_context_cells": query_frame.trace_cells,
            "query_schema_index": context_match.query_schema_index,
            "query_schema_cell_id": (
                None
                if context_match.query_schema_index is None
                else QUERY_MATCH_CELL_OFFSET + context_match.query_schema_index
            ),
            "query_match_arrival_counts": context_match.arrivals,
            "query_match_spike_step": context_match.spike_step,
            "query_match_simultaneous_spike_count": (
                context_match.simultaneous_spike_count
            ),
            "route_raw_scores": route.raw_scores,
            "selected_route": route.selected_route,
            "selected_route_cell_id": (
                None
                if route.selected_route is None
                else ROUTE_ACTION_CELL_OFFSET + route.selected_route
            ),
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
        }
        result = ContextRoutedRecall(
            status=recall_status,
            output=output,
            query_context_cells=query_frame.trace_cells,
            query_schema_index=context_match.query_schema_index,
            query_schema_cell_id=(
                None
                if context_match.query_schema_index is None
                else QUERY_MATCH_CELL_OFFSET + context_match.query_schema_index
            ),
            query_match_arrival_counts=context_match.arrivals,
            query_match_spike_step=context_match.spike_step,
            query_match_simultaneous_spike_count=(
                context_match.simultaneous_spike_count
            ),
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
            raise RuntimeError("context router recall mutated its V2 child")
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


def _validate_lessons(lessons: tuple[PhysicalContextRetrievalLesson, ...]) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not PhysicalContextRetrievalLesson for lesson in lessons
    ):
        raise TypeError(
            "lessons must be an exact tuple of PhysicalContextRetrievalLesson"
        )


def _apply_lessons(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    arrays: tuple[np.ndarray, ...],
    lessons: tuple[PhysicalContextRetrievalLesson, ...],
) -> tuple[tuple[np.ndarray, ...], _BatchTotals]:
    (
        cue_cells,
        excitatory,
        inhibitory,
        provenance_query_indices,
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
    query_projection_work = 0
    router_cell_steps = 0
    exact_child_work = 0
    semantic_child_steps = 0
    steps: list[ContextRouteLearningStep] = []

    child_root = child.checkpoint_sha256
    exact_root = child.episodic_checkpoint_sha256
    semantic_root = child.semantic_checkpoint_sha256
    for ordinal, lesson in enumerate(lessons):
        if not lesson.teacher_gate:
            teacher_off_count += 1
            steps.append(
                ContextRouteLearningStep(
                    ordinal,
                    STOP_TEACHER_OFF,
                    None,
                    None,
                    False,
                    None,
                    0,
                    0,
                    0,
                )
            )
            continue
        if not lesson.cell_gate:
            cell_off_count += 1
            steps.append(
                ContextRouteLearningStep(
                    ordinal,
                    STOP_CELL_OFF,
                    None,
                    None,
                    False,
                    None,
                    0,
                    0,
                    0,
                )
            )
            continue

        exact_recall = child.recall_episode(
            lesson.exact_observation_events,
            cell_gate=True,
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
                ContextRouteLearningStep(
                    ordinal,
                    STOP_COMPARATOR_OFF,
                    None,
                    None,
                    False,
                    None,
                    lesson_exact_work,
                    lesson_semantic_steps,
                    consequence.cell_steps,
                )
            )
            continue
        if consequence.target_route is None:
            consequence_not_singleton_count += 1
            steps.append(
                ContextRouteLearningStep(
                    ordinal,
                    STOP_CONSEQUENCE_NOT_SINGLETON,
                    None,
                    None,
                    False,
                    None,
                    lesson_exact_work,
                    lesson_semantic_steps,
                    consequence.cell_steps,
                )
            )
            continue

        query_frame = _QUERY_CELLS.project(lesson.query_context_event)
        query_projection_work += query_frame.work.total
        context_match = _context_match(
            cue_cells,
            query_frame.trace_cells,
            cell_gate=True,
        )
        router_cell_steps += context_match.cell_steps
        if context_match.status == STOP_QUERY_AMBIGUOUS:
            steps.append(
                ContextRouteLearningStep(
                    ordinal,
                    STOP_QUERY_AMBIGUOUS,
                    None,
                    consequence.target_route,
                    False,
                    None,
                    lesson_exact_work,
                    lesson_semantic_steps,
                    consequence.cell_steps + context_match.cell_steps,
                )
            )
            continue
        query_index = context_match.query_schema_index
        if query_index is None:
            if not context_match.novel:
                raise RuntimeError("context router query allocation authority changed")
            if cue_cells.shape[0] == MAX_QUERY_CONTEXTS:
                raise RuntimeError("context router query capacity exhausted")
            query_index = cue_cells.shape[0]
            cue_cells = np.concatenate(
                (
                    cue_cells,
                    np.asarray([query_frame.trace_cells], dtype=CUE_DTYPE),
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

        accepted_rows = np.flatnonzero(
            (provenance_query_indices == query_index) & (dispositions == 1)
        )
        relation = np.asarray(lesson.relation_cells, dtype=PROVENANCE_CELL_DTYPE)
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
            vote_committed = False
        elif dependent:
            disposition_code = 3
            disposition_label = DEPENDENT_NO_VOTE
            vote_committed = False
        else:
            disposition_code = 1
            disposition_label = ACCEPTED_VOTE
            vote_committed = True
            target = consequence.target_route
            rival = 1 - target
            before = int(excitatory[query_index, target])
            excitatory[query_index, target] = min(TICK_CAP, before + 1)
            excitatory_ticks += int(excitatory[query_index, target]) - before
            if int(excitatory[query_index, rival]) > 0:
                before = int(inhibitory[query_index, rival])
                inhibitory[query_index, rival] = min(TICK_CAP, before + 1)
                inhibitory_ticks += int(inhibitory[query_index, rival]) - before

        provenance_record = {
            "memory_checkpoint_sha256": child_root,
            "episodic_checkpoint_sha256": exact_root,
            "semantic_checkpoint_sha256": semantic_root,
            "query_frame_sha256": query_frame.sha256,
            "query_schema_index": query_index,
            "relation_cells": lesson.relation_cells,
            "evidence_context_cells": lesson.evidence_context_cells,
            "evidence_source_cells": lesson.evidence_source_cells,
            "observed_output_cells": lesson.observed_output_cells,
            "exact_child_active_cells": exact_recall.output.active_cells,
            "semantic_child_active_cells": semantic_recall.output.active_cells,
            "consequence_equality_spikes": consequence.equality_spikes,
            "consequence_integrator_arrivals": consequence.integrator_arrivals,
            "consequence_spike_step": consequence.spike_step,
            "consequence_simultaneous_spike_count": (
                consequence.simultaneous_spike_count
            ),
            "target_route": consequence.target_route,
            "disposition": disposition_label,
            "route_excitatory_after": tuple(
                int(value) for value in excitatory[query_index]
            ),
            "route_inhibitory_after": tuple(
                int(value) for value in inhibitory[query_index]
            ),
            "child_token_or_status_used_for_selection": False,
        }
        provenance_sha = _digest(_PROVENANCE_NAMESPACE, provenance_record)
        if provenance_roots.shape[0] == MAX_PROVENANCE_ROWS:
            raise RuntimeError("context router provenance capacity exhausted")
        provenance_query_indices = np.append(
            provenance_query_indices,
            np.asarray(query_index, dtype=PROVENANCE_INDEX_DTYPE),
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
        steps.append(
            ContextRouteLearningStep(
                ordinal,
                disposition_label,
                query_index,
                consequence.target_route,
                vote_committed,
                provenance_sha,
                lesson_exact_work,
                lesson_semantic_steps,
                consequence.cell_steps + context_match.cell_steps,
            )
        )

    child.authenticate()
    if (
        child.checkpoint_sha256 != child_root
        or child.episodic_checkpoint_sha256 != exact_root
        or child.semantic_checkpoint_sha256 != semantic_root
    ):
        raise RuntimeError("context router learning mutated its V2 child")
    result_arrays = (
        cue_cells,
        excitatory,
        inhibitory,
        provenance_query_indices,
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
        query_projection_work=query_projection_work,
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
    previous: ContextRouterTrainingSummary | None,
) -> FrozenCommonContextRoutedMemoryV1:
    query_count = arrays[0].shape[0]
    provenance_count = arrays[-1].shape[0]
    trainable_bytes = arrays[0].nbytes + arrays[1].nbytes + arrays[2].nbytes
    provenance_bytes = sum(array.nbytes for array in arrays[3:])
    base = previous
    summary_record = {
        "mode": mode,
        "new_lesson_count": new_lesson_count,
        "total_lesson_count": (0 if base is None else base.total_lesson_count)
        + new_lesson_count,
        "teacher_off_count": (0 if base is None else base.teacher_off_count)
        + batch.teacher_off_count,
        "cell_off_count": (0 if base is None else base.cell_off_count)
        + batch.cell_off_count,
        "comparator_off_count": (0 if base is None else base.comparator_off_count)
        + batch.comparator_off_count,
        "consequence_not_singleton_count": (
            0 if base is None else base.consequence_not_singleton_count
        )
        + batch.consequence_not_singleton_count,
        "accepted_vote_count": int(np.count_nonzero(arrays[8] == 1)),
        "duplicate_nonvote_count": int(np.count_nonzero(arrays[8] == 2)),
        "dependent_nonvote_count": int(np.count_nonzero(arrays[8] == 3)),
        "query_schema_count": query_count,
        "provenance_count": provenance_count,
        "excitatory_update_ticks": (
            0 if base is None else base.excitatory_update_ticks
        )
        + batch.excitatory_update_ticks,
        "inhibitory_update_ticks": (
            0 if base is None else base.inhibitory_update_ticks
        )
        + batch.inhibitory_update_ticks,
        "query_projection_work": (
            0 if base is None else base.query_projection_work
        )
        + batch.query_projection_work,
        "router_learning_cell_steps": (
            0 if base is None else base.router_learning_cell_steps
        )
        + batch.router_learning_cell_steps,
        "exact_child_reported_work": (
            0 if base is None else base.exact_child_reported_work
        )
        + batch.exact_child_reported_work,
        "semantic_child_cell_steps": (
            0 if base is None else base.semantic_child_cell_steps
        )
        + batch.semantic_child_cell_steps,
        "trainable_state_bytes": trainable_bytes,
        "provenance_state_bytes": provenance_bytes,
        "checkpoint_array_bytes": trainable_bytes + provenance_bytes,
        "fixed_router_cell_count": FIXED_ROUTER_CELL_COUNT,
        "query_match_cell_count": query_count,
        "added_cell_count": FIXED_ROUTER_CELL_COUNT + query_count,
        "parent_lesson_replay_count": 0,
        "retained_router_parent_count": 0,
        "raw_lesson_retention_count": 0,
        "child_mutation_count": 0,
        "memory_checkpoint_sha256": child.checkpoint_sha256,
        "episodic_checkpoint_sha256": child.episodic_checkpoint_sha256,
        "semantic_checkpoint_sha256": child.semantic_checkpoint_sha256,
        "last_steps": batch.steps,
    }
    placeholder = ContextRouterTrainingSummary(
        **summary_record, checkpoint_sha256=""
    )
    named_arrays = (
        ("query-cue-cells", arrays[0]),
        ("route-excitatory", arrays[1]),
        ("route-inhibitory", arrays[2]),
        ("provenance-query-indices", arrays[3]),
        ("provenance-relation-cells", arrays[4]),
        ("provenance-context-cells", arrays[5]),
        ("provenance-source-cells", arrays[6]),
        ("provenance-routes", arrays[7]),
        ("provenance-dispositions", arrays[8]),
        ("provenance-roots", arrays[9]),
    )
    checkpoint = _checkpoint_sha256(child, placeholder, named_arrays)
    summary = dataclasses.replace(placeholder, checkpoint_sha256=checkpoint)
    return FrozenCommonContextRoutedMemoryV1(
        child=child,
        cue_cells=arrays[0],
        excitatory=arrays[1],
        inhibitory=arrays[2],
        provenance_query_indices=arrays[3],
        provenance_relation_cells=arrays[4],
        evidence_context_cells=arrays[5],
        evidence_source_cells=arrays[6],
        provenance_routes=arrays[7],
        dispositions=arrays[8],
        provenance_roots=arrays[9],
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def fit_common_context_routed_memory_v1(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    lessons: tuple[PhysicalContextRetrievalLesson, ...],
) -> FrozenCommonContextRoutedMemoryV1:
    """Fit a flat query router over one already-frozen dual-view child."""

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


def advance_common_context_routed_memory_v1(
    parent: FrozenCommonContextRoutedMemoryV1,
    new_lessons: tuple[PhysicalContextRetrievalLesson, ...],
) -> FrozenCommonContextRoutedMemoryV1:
    """Consume only new lessons and return one flat successor checkpoint."""

    if type(parent) is not FrozenCommonContextRoutedMemoryV1:
        raise TypeError("parent must be one exact frozen context router")
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
        raise RuntimeError("context router advance mutated its parent or child")
    return successor


SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "ACCEPTED_VOTE",
    "C1_INPUT_COUNT",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DEPENDENT_NO_VOTE",
    "DEV_INPUT_COUNT",
    "DUPLICATE_NO_VOTE",
    "EXACT_ROUTE",
    "FIXED_ROUTER_CELL_COUNT",
    "G0_INPUT_COUNT",
    "HELDOUT_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PROVENANCE_BYTES_PER_ROW",
    "PUBLIC_INPUT_COUNT",
    "QUERY_STATE_BYTES_PER_SCHEMA",
    "RECALLED",
    "SEMANTIC_ROUTE",
    "SOURCE_SHA256",
    "STOP_OUTPUT_RELAY",
    "STOP_QUERY_AMBIGUOUS",
    "STOP_QUERY_UNKNOWN",
    "STOP_ROUTE_AMBIGUOUS",
    "STOP_ROUTE_UNSUPPORTED",
    "STOP_SELECTED_CHILD",
    "ContextRouteLearningStep",
    "ContextRouteProvenance",
    "ContextRoutedRecall",
    "ContextRouterRecallWork",
    "ContextRouterTrainingSummary",
    "FrozenCommonContextRoutedMemoryV1",
    "PhysicalContextRetrievalLesson",
    "advance_common_context_routed_memory_v1",
    "fit_common_context_routed_memory_v1",
]
