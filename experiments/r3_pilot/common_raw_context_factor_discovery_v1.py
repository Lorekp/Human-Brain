"""Discover reusable routing factors from authenticated raw context frames.

This layer is deliberately thin.  It projects each physical position/value
binding in a ``CommonEventFrame`` into the same twelve-lane branch geometry
banked by the common factorized transition learner.  The already-banked V2
factor router remains the only owner of consequence-trained route E/H and
independent provenance.  This layer adds only a persistent admission latch:
two independent accepted observations and a unique physical route-versus-tonic
winner are required before a branch can participate in recall.

Recall projects the raw frame again, recovers admitted branches through a
fitted-GLIF equality race, and passes their unordered physical assemblies to
the V2 router.  Unknown or ambiguous branches stop before routing; a selected
child STOP is returned unchanged and never falls back.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.g0 import common_event_cells_v1 as projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from experiments.r3_pilot import common_factorized_context_routed_memory_v2 as router
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCOPE = "COMMON_RAW_CONTEXT_FACTOR_DISCOVERY_V1"
STATUS = "SELF_POSED_RAW_CONTEXT_REUSABLE_FACTOR_DISCOVERY"
CLAIM_BOUNDARY = (
    "self-posed-replay-free-discovery-of-reusable-position-value-binding-"
    "factors-from-authenticated-presegmented-common-event-frames-with-physical-"
    "admission-and-routing-only-no-arbitrary-latent-parsing-language-coding-"
    "corpus-protected-transfer-or-benchmark-evidence"
)

RECALLED = router.RECALLED
STOP_PROJECTION_GATE = "STOP_RAW_CONTEXT_PROJECTION_GATE_OFF"
STOP_FACTOR_GATE = "STOP_RAW_CONTEXT_FACTOR_GATE_OFF"
STOP_FACTOR_UNKNOWN = "STOP_RAW_CONTEXT_FACTOR_UNKNOWN"
STOP_FACTOR_AMBIGUOUS = "STOP_RAW_CONTEXT_FACTOR_AMBIGUOUS"

MAX_RAW_CONTEXT_BYTES = 2
MAX_ADMITTED_FACTORS = router.MAX_FACTORS
TRACE_LANES = projection.TRACE_LANES
TICK_CAP = router.TICK_CAP

CUE_DTYPE = np.dtype(np.int32)
RACE_STEPS = 600
ZERO_SCORE_SPIKE_STEP = 568
BRANCH_MATCH_SCORE = TRACE_LANES
BRANCH_TONIC_SCORE = BRANCH_MATCH_SCORE - 1
ADMISSION_TONIC_SCORE = 1
FIXED_DISCOVERY_CELL_COUNT = 4

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

_SOURCE_PATH = Path(__file__).resolve()
_RAW_FRAME_NAMESPACE = b"humanbrain.g0.common-event-cells-v1.frame\0"
_BRANCH_NAMESPACE = b"common-raw-context-factor-discovery-v1.branch\0"
_CONFIG_NAMESPACE = b"common-raw-context-factor-discovery-v1.config\0"
_CHECKPOINT_NAMESPACE = b"common-raw-context-factor-discovery-v1.checkpoint\0"
_RECALL_NAMESPACE = b"common-raw-context-factor-discovery-v1.recall\0"
_CONSTRUCTION_TOKEN = object()


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
        + array.tobytes(order="C")
    )


def _build_score_current_table() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(RACE_STEPS)
    crossing = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(BRANCH_MATCH_SCORE + 1):
        first_step = ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing[first_step])
        upper = float(crossing[first_step - 1])
        if not lower < upper:
            raise RuntimeError("raw-context fitted-GLIF interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        row.spike_step != ZERO_SCORE_SPIKE_STEP - score
        for score, row in enumerate(controls)
    ):
        raise RuntimeError("raw-context fitted-GLIF calibration changed")
    return table, race


_SCORE_CURRENT_TABLE, _SCORE_RACE = _build_score_current_table()

CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "factor_router_config_sha256": router.CONFIG_SHA256,
    "raw_projection_geometry_sha256": projection.GEOMETRY_SHA256,
    "raw_context_bytes": (1, MAX_RAW_CONTEXT_BYTES),
    "branch_source": "physical-position-value-binding-assemblies",
    "branch_kind_string_input": False,
    "caller_factor_tuple_input": False,
    "host_route_label_input": False,
    "host_source_order_authority": False,
    "whole_context_registry": False,
    "nearest_neighbor_lookup": False,
    "selected_child_fallback": False,
    "minimum_independent_support": 2,
    "branch_tonic_score": BRANCH_TONIC_SCORE,
    "admission_tonic_score": ADMISSION_TONIC_SCORE,
    "maximum_admitted_factors": MAX_ADMITTED_FACTORS,
    "fixed_discovery_cell_count": FIXED_DISCOVERY_CELL_COUNT,
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


def _authenticate_frame(frame: projection.CommonEventFrame) -> None:
    if type(frame) is not projection.CommonEventFrame:
        raise TypeError("raw_context_frame must be one exact CommonEventFrame")
    if not 1 <= frame.byte_count <= MAX_RAW_CONTEXT_BYTES:
        raise ValueError("raw context exceeds the bounded V1 projection")
    record = {
        "byte_count": frame.byte_count,
        "start_cells": frame.start_cells,
        "byte_emissions": tuple(row.record() for row in frame.byte_emissions),
        "end_cells": frame.end_cells,
        "trace_cells": frame.trace_cells,
        "work": frame.work.record(),
    }
    if _digest(_RAW_FRAME_NAMESPACE, record) != frame.sha256:
        raise ValueError("raw CommonEventFrame authentication failed")


def _branch_trace(source_cells: tuple[int, ...]) -> tuple[int, ...]:
    if (
        type(source_cells) is not tuple
        or len(source_cells) != projection.ASSEMBLY_ACTIVE_CELLS
        or any(type(cell) is not int or cell < 0 for cell in source_cells)
    ):
        raise ValueError("branch projection requires one physical assembly")
    payload = b"".join(cell.to_bytes(4, "big") for cell in source_cells)
    return tuple(
        projection.TRACE_CELL_OFFSET
        + lane * projection.FEATURE_BANK_CELLS
        + int.from_bytes(
            sha256(_BRANCH_NAMESPACE + payload + lane.to_bytes(2, "big")).digest()[
                :4
            ],
            "big",
        )
        % projection.FEATURE_BANK_CELLS
        for lane in range(TRACE_LANES)
    )


def _project_branches(
    frame: projection.CommonEventFrame,
) -> tuple[tuple[int, ...], ...]:
    _authenticate_frame(frame)
    branches = tuple(
        _branch_trace(emission.binding_cells) for emission in frame.byte_emissions
    )
    if len(set(branches)) != len(branches):
        raise RuntimeError("raw branch projection collided inside one frame")
    return tuple(sorted(branches))


@dataclass(frozen=True, slots=True)
class PhysicalRawContextRouteLesson:
    raw_context_frame: projection.CommonEventFrame
    exact_observation_events: tuple[bytes, ...]
    semantic_input_frame: filler_projection.CommonOneByteFillerFrame
    relation_cells: tuple[int, ...]
    evidence_context_cells: tuple[int, ...]
    evidence_source_cells: tuple[int, ...]
    observed_output_cells: tuple[int, ...]
    teacher_gate: bool = True
    cell_gate: bool = True
    comparator_gate: bool = True
    projection_gate: bool = True
    plasticity_gate: bool = True

    def __post_init__(self) -> None:
        factors = _project_branches(self.raw_context_frame)
        router.PhysicalFactorRouteLesson(
            factor_cells=factors,
            exact_observation_events=self.exact_observation_events,
            semantic_input_frame=self.semantic_input_frame,
            relation_cells=self.relation_cells,
            evidence_context_cells=self.evidence_context_cells,
            evidence_source_cells=self.evidence_source_cells,
            observed_output_cells=self.observed_output_cells,
            teacher_gate=self.teacher_gate,
            cell_gate=self.cell_gate,
            comparator_gate=self.comparator_gate,
        )
        if any(
            type(gate) is not bool
            for gate in (self.projection_gate, self.plasticity_gate)
        ):
            raise TypeError("raw-context gates must be exact booleans")


@dataclass(frozen=True, slots=True)
class BranchGateMatch:
    status: str
    admitted_index: int | None
    arrival_counts: tuple[int, ...]
    spike_step: int
    simultaneous_spike_count: int
    equality_reads: int
    equality_arrivals: int
    cell_steps: int


@dataclass(frozen=True, slots=True)
class RawContextTrainingSummary:
    mode: str
    new_raw_lesson_count: int
    total_raw_lesson_count: int
    projection_off_count: int
    plasticity_off_count: int
    candidate_branch_count: int
    parent_factor_count: int
    accepted_factor_vote_count: int
    duplicate_nonvote_count: int
    dependent_nonvote_count: int
    route_trainable_state_bytes: int
    provenance_state_bytes: int
    parent_router_learning_cell_steps: int
    admission_probe_count: int
    admitted_factor_count: int
    new_admitted_factor_count: int
    admission_probe_router_cell_steps: int
    admission_probe_exact_child_work: int
    admission_probe_semantic_cell_steps: int
    admission_support_cell_steps: int
    admission_state_bytes: int
    checkpoint_array_bytes: int
    fixed_discovery_cell_count: int
    branch_match_cell_count: int
    added_discovery_cell_count: int
    added_composite_cell_count: int
    parent_lesson_replay_count: int
    retained_parent_count: int
    raw_lesson_retention_count: int
    child_mutation_count: int
    factor_router_checkpoint_sha256: str
    memory_checkpoint_sha256: str
    episodic_checkpoint_sha256: str
    semantic_checkpoint_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class RawContextRecallWork:
    upstream_projection_reported_work: int
    branch_projection_source_reads: int
    branch_match_equality_reads: int
    branch_match_equality_arrivals: int
    branch_match_cell_steps: int
    parent_router_cell_steps: int
    exact_child_reported_work: int
    semantic_child_cell_steps: int

    @property
    def discovery_cell_steps(self) -> int:
        return self.branch_match_cell_steps

    def record(self) -> dict[str, int]:
        return {
            **dataclasses.asdict(self),
            "discovery_cell_steps": self.discovery_cell_steps,
        }


@dataclass(frozen=True, slots=True)
class RawContextFactorRecall:
    status: str
    raw_context_frame_sha256: str
    projected_factor_cells: tuple[tuple[int, ...], ...]
    admitted_factor_indices: tuple[int | None, ...]
    branch_matches: tuple[BranchGateMatch, ...]
    routed_recall: router.FactorizedContextRecall | None
    work: RawContextRecallWork
    checkpoint_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "raw_context_frame_sha256": self.raw_context_frame_sha256,
            "projected_factor_cells": self.projected_factor_cells,
            "admitted_factor_indices": self.admitted_factor_indices,
            "branch_matches": tuple(dataclasses.asdict(row) for row in self.branch_matches),
            "routed_recall": (
                None if self.routed_recall is None else self.routed_recall.record()
            ),
            "work": self.work.record(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "caller_factor_tuple_input": False,
            "host_route_label_input": False,
            "host_source_order_authority": False,
            "whole_context_registry": False,
            "nearest_neighbor_lookup": False,
            "selected_child_fallback": False,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _freeze_array(value: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(value, dtype=CUE_DTYPE)
    frozen.setflags(write=False)
    return frozen


def _support_wins(independent_support: int) -> tuple[bool, int]:
    score = min(max(independent_support, 0), TICK_CAP)
    currents = np.asarray(
        [[
            _SCORE_CURRENT_TABLE[score],
            _SCORE_CURRENT_TABLE[ADMISSION_TONIC_SCORE],
        ]],
        dtype=np.float64,
    )
    race = _SCORE_RACE.race_many(currents, require_unique=False)[0]
    return (
        race.simultaneous_spike_candidates == 1 and race.winner_cell == 0,
        2 * RACE_STEPS,
    )


def _branch_match(
    admitted_cells: np.ndarray,
    branch: tuple[int, ...],
    *,
    factor_gate: bool,
) -> BranchGateMatch:
    count = admitted_cells.shape[0]
    if count:
        active = np.asarray(branch, dtype=CUE_DTYPE)
        arrivals_array = np.count_nonzero(admitted_cells == active, axis=1).astype(
            np.int16
        )
    else:
        arrivals_array = np.empty((0,), dtype=np.int16)
    scores = arrivals_array if factor_gate else np.zeros_like(arrivals_array)
    currents = np.concatenate(
        (
            _SCORE_CURRENT_TABLE[scores],
            np.asarray([_SCORE_CURRENT_TABLE[BRANCH_TONIC_SCORE]]),
        )
    )
    race = _SCORE_RACE.race_many(currents[None, :], require_unique=False)[0]
    if race.simultaneous_spike_candidates != 1:
        status = STOP_FACTOR_AMBIGUOUS
        admitted_index = None
    elif race.winner_cell == count:
        status = STOP_FACTOR_UNKNOWN
        admitted_index = None
    else:
        status = RECALLED
        admitted_index = race.winner_cell
    return BranchGateMatch(
        status=status,
        admitted_index=admitted_index,
        arrival_counts=tuple(int(value) for value in arrivals_array),
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_candidates,
        equality_reads=count * TRACE_LANES,
        equality_arrivals=int(np.sum(arrivals_array, dtype=np.int64)),
        cell_steps=(count + 1) * RACE_STEPS,
    )


def _parent_lesson(
    lesson: PhysicalRawContextRouteLesson,
) -> router.PhysicalFactorRouteLesson:
    enabled = lesson.teacher_gate and lesson.projection_gate and lesson.plasticity_gate
    return router.PhysicalFactorRouteLesson(
        factor_cells=_project_branches(lesson.raw_context_frame),
        exact_observation_events=lesson.exact_observation_events,
        semantic_input_frame=lesson.semantic_input_frame,
        relation_cells=lesson.relation_cells,
        evidence_context_cells=lesson.evidence_context_cells,
        evidence_source_cells=lesson.evidence_source_cells,
        observed_output_cells=lesson.observed_output_cells,
        teacher_gate=enabled,
        cell_gate=lesson.cell_gate,
        comparator_gate=lesson.comparator_gate,
    )


def _candidate_witnesses(
    lessons: tuple[PhysicalRawContextRouteLesson, ...],
) -> dict[tuple[int, ...], PhysicalRawContextRouteLesson]:
    witnesses: dict[tuple[int, ...], PhysicalRawContextRouteLesson] = {}
    for lesson in lessons:
        if not (
            lesson.teacher_gate
            and lesson.cell_gate
            and lesson.comparator_gate
            and lesson.projection_gate
            and lesson.plasticity_gate
        ):
            continue
        for cells in _project_branches(lesson.raw_context_frame):
            witnesses.setdefault(cells, lesson)
    return witnesses


@dataclass(frozen=True, slots=True)
class _AdmissionTotals:
    probes: int
    router_steps: int
    exact_work: int
    semantic_steps: int
    support_steps: int


def _discover_new_admissions(
    factor_router: router.FrozenCommonFactorizedContextRoutedMemoryV2,
    existing: tuple[tuple[int, ...], ...],
    witnesses: dict[tuple[int, ...], PhysicalRawContextRouteLesson],
) -> tuple[tuple[tuple[int, ...], ...], _AdmissionTotals]:
    admitted = set(existing)
    provenance = factor_router.provenance
    probes = 0
    router_steps = 0
    exact_work = 0
    semantic_steps = 0
    support_steps = 0
    for cells in sorted(witnesses):
        if cells in admitted:
            continue
        witness = witnesses[cells]
        probe = factor_router.recall(
            (cells,),
            witness.exact_observation_events,
            witness.semantic_input_frame,
            witness.relation_cells,
            output_relay_gate=False,
        )
        probes += 1
        router_steps += probe.work.router_cell_steps
        exact_work += probe.work.exact_child_reported_work
        semantic_steps += probe.work.semantic_child_cell_steps
        factor_index = probe.factor_indices[0]
        if factor_index is None:
            continue
        independent_support = sum(
            row.factor_index == factor_index and row.disposition == router.ACCEPTED_VOTE
            for row in provenance
        )
        support_won, cell_steps = _support_wins(independent_support)
        support_steps += cell_steps
        route_won = (
            probe.selected_route_cell_id is not None
            and probe.route_simultaneous_spike_count == 1
        )
        if support_won and route_won:
            admitted.add(cells)
    return tuple(sorted(admitted)), _AdmissionTotals(
        probes=probes,
        router_steps=router_steps,
        exact_work=exact_work,
        semantic_steps=semantic_steps,
        support_steps=support_steps,
    )


def _checkpoint_sha256(
    factor_router: router.FrozenCommonFactorizedContextRoutedMemoryV2,
    summary: RawContextTrainingSummary,
    admitted_cells: np.ndarray,
) -> str:
    summary_record = dataclasses.asdict(summary)
    summary_record["checkpoint_sha256"] = None
    digest = sha256()
    digest.update(_CHECKPOINT_NAMESPACE)
    digest.update(CONFIG_SHA256.encode("ascii"))
    digest.update(factor_router.checkpoint_sha256.encode("ascii"))
    digest.update(_canonical_json(summary_record))
    digest.update(_array_frame("admitted-factor-cells", admitted_cells))
    return digest.hexdigest()


class FrozenCommonRawContextFactorDiscoveryV1:
    """One immutable V2 factor router plus raw-branch admission latches."""

    __slots__ = (
        "_admitted_cells",
        "_checkpoint_sha256",
        "_factor_router",
        "_summary",
    )

    def __init__(
        self,
        *,
        factor_router: router.FrozenCommonFactorizedContextRoutedMemoryV2,
        admitted_cells: np.ndarray,
        summary: RawContextTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("raw-context factor discoveries are builder-owned")
        if type(factor_router) is not router.FrozenCommonFactorizedContextRoutedMemoryV2:
            raise TypeError("factor_router must be one exact frozen V2 router")
        if type(summary) is not RawContextTrainingSummary:
            raise TypeError("raw-context training summary type changed")
        self._factor_router = factor_router
        self._admitted_cells = _freeze_array(admitted_cells)
        self._summary = summary
        self._checkpoint_sha256 = _checkpoint_sha256(
            factor_router, summary, self._admitted_cells
        )
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise RuntimeError("raw-context checkpoint identity changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def factor_router_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._factor_router.checkpoint_sha256

    @property
    def memory_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._factor_router.memory_checkpoint_sha256

    @property
    def episodic_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._factor_router.episodic_checkpoint_sha256

    @property
    def semantic_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._factor_router.semantic_checkpoint_sha256

    @property
    def training_summary(self) -> RawContextTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def provenance(self) -> tuple[router.FactorRouteProvenance, ...]:
        self.authenticate()
        return self._factor_router.provenance

    @property
    def admitted_factor_cells(self) -> tuple[tuple[int, ...], ...]:
        self.authenticate()
        return tuple(
            tuple(int(value) for value in row) for row in self._admitted_cells
        )

    def authenticate(self) -> None:
        self._factor_router.authenticate()
        if self._admitted_cells.flags.writeable:
            raise RuntimeError("raw-context admission state became mutable")
        if (
            self._admitted_cells.ndim != 2
            or self._admitted_cells.shape[1] != TRACE_LANES
            or self._admitted_cells.shape[0] > MAX_ADMITTED_FACTORS
        ):
            raise RuntimeError("raw-context admission shape changed")
        rows = tuple(tuple(row) for row in self._admitted_cells.tolist())
        if len(rows) != len(set(rows)) or rows != tuple(sorted(rows)):
            raise RuntimeError("raw-context admission identity changed")
        summary = self._summary
        admission_bytes = self._admitted_cells.nbytes
        parent_summary = self._factor_router.training_summary
        if (
            summary.admitted_factor_count != self._admitted_cells.shape[0]
            or summary.parent_factor_count != parent_summary.factor_count
            or summary.accepted_factor_vote_count
            != parent_summary.accepted_factor_vote_count
            or summary.duplicate_nonvote_count
            != parent_summary.duplicate_nonvote_count
            or summary.dependent_nonvote_count
            != parent_summary.dependent_nonvote_count
            or summary.route_trainable_state_bytes
            != parent_summary.trainable_state_bytes
            or summary.provenance_state_bytes
            != parent_summary.provenance_state_bytes
            or summary.parent_router_learning_cell_steps
            != parent_summary.router_learning_cell_steps
            or summary.admission_state_bytes != admission_bytes
            or summary.checkpoint_array_bytes
            != parent_summary.checkpoint_array_bytes + admission_bytes
            or summary.fixed_discovery_cell_count != FIXED_DISCOVERY_CELL_COUNT
            or summary.branch_match_cell_count != self._admitted_cells.shape[0]
            or summary.added_discovery_cell_count
            != FIXED_DISCOVERY_CELL_COUNT + self._admitted_cells.shape[0]
            or summary.added_composite_cell_count
            != parent_summary.added_cell_count + summary.added_discovery_cell_count
            or summary.parent_lesson_replay_count != 0
            or summary.retained_parent_count != 0
            or summary.raw_lesson_retention_count != 0
            or summary.child_mutation_count != 0
            or summary.factor_router_checkpoint_sha256
            != self._factor_router.checkpoint_sha256
            or summary.memory_checkpoint_sha256
            != self._factor_router.memory_checkpoint_sha256
            or summary.episodic_checkpoint_sha256
            != self._factor_router.episodic_checkpoint_sha256
            or summary.semantic_checkpoint_sha256
            != self._factor_router.semantic_checkpoint_sha256
            or _checkpoint_sha256(self._factor_router, summary, self._admitted_cells)
            != self._checkpoint_sha256
        ):
            raise RuntimeError("raw-context checkpoint authentication failed")

    def recall(
        self,
        raw_context_frame: projection.CommonEventFrame,
        exact_observation_events: tuple[bytes, ...],
        semantic_input_frame: filler_projection.CommonOneByteFillerFrame,
        relation_cells: tuple[int, ...],
        *,
        projection_gate: bool = True,
        factor_gate: bool = True,
        router_gate: bool = True,
        output_relay_gate: bool = True,
        semantic_relation_gate: bool = True,
    ) -> RawContextFactorRecall:
        self.authenticate()
        _authenticate_frame(raw_context_frame)
        if any(
            type(gate) is not bool
            for gate in (
                projection_gate,
                factor_gate,
                router_gate,
                output_relay_gate,
                semantic_relation_gate,
            )
        ):
            raise TypeError("raw-context recall gates must be exact booleans")

        if projection_gate:
            factors = _project_branches(raw_context_frame)
            matches = tuple(
                _branch_match(self._admitted_cells, cells, factor_gate=factor_gate)
                for cells in factors
            )
        else:
            factors = ()
            matches = ()

        routed: router.FactorizedContextRecall | None = None
        if not projection_gate:
            status = STOP_PROJECTION_GATE
        elif not factor_gate:
            status = STOP_FACTOR_GATE
        elif any(match.status == STOP_FACTOR_AMBIGUOUS for match in matches):
            status = STOP_FACTOR_AMBIGUOUS
        elif any(match.status == STOP_FACTOR_UNKNOWN for match in matches):
            status = STOP_FACTOR_UNKNOWN
        else:
            routed = self._factor_router.recall(
                tuple(sorted(factors)),
                exact_observation_events,
                semantic_input_frame,
                relation_cells,
                factor_gate=True,
                router_gate=router_gate,
                output_relay_gate=output_relay_gate,
                semantic_relation_gate=semantic_relation_gate,
            )
            status = routed.status

        work = RawContextRecallWork(
            upstream_projection_reported_work=raw_context_frame.work.total,
            branch_projection_source_reads=len(factors)
            * projection.ASSEMBLY_ACTIVE_CELLS,
            branch_match_equality_reads=sum(row.equality_reads for row in matches),
            branch_match_equality_arrivals=sum(
                row.equality_arrivals for row in matches
            ),
            branch_match_cell_steps=sum(row.cell_steps for row in matches),
            parent_router_cell_steps=(
                0 if routed is None else routed.work.router_cell_steps
            ),
            exact_child_reported_work=(
                0 if routed is None else routed.work.exact_child_reported_work
            ),
            semantic_child_cell_steps=(
                0 if routed is None else routed.work.semantic_child_cell_steps
            ),
        )
        record = {
            "status": status,
            "raw_context_frame_sha256": raw_context_frame.sha256,
            "projected_factor_cells": factors,
            "admitted_factor_indices": tuple(
                row.admitted_index for row in matches
            ),
            "branch_matches": tuple(dataclasses.asdict(row) for row in matches),
            "routed_recall": None if routed is None else routed.record(),
            "work": work.record(),
            "checkpoint_sha256": self._checkpoint_sha256,
            "caller_factor_tuple_input": False,
            "host_route_label_input": False,
            "host_source_order_authority": False,
            "whole_context_registry": False,
            "nearest_neighbor_lookup": False,
            "selected_child_fallback": False,
        }
        result = RawContextFactorRecall(
            status=status,
            raw_context_frame_sha256=raw_context_frame.sha256,
            projected_factor_cells=factors,
            admitted_factor_indices=tuple(row.admitted_index for row in matches),
            branch_matches=matches,
            routed_recall=routed,
            work=work,
            checkpoint_sha256=self._checkpoint_sha256,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )
        self.authenticate()
        return result


def _validate_lessons(
    lessons: tuple[PhysicalRawContextRouteLesson, ...],
) -> None:
    if type(lessons) is not tuple or any(
        type(lesson) is not PhysicalRawContextRouteLesson for lesson in lessons
    ):
        raise TypeError(
            "lessons must be an exact tuple of PhysicalRawContextRouteLesson"
        )


def _freeze(
    factor_router: router.FrozenCommonFactorizedContextRoutedMemoryV2,
    admitted: tuple[tuple[int, ...], ...],
    admission_totals: _AdmissionTotals,
    lessons: tuple[PhysicalRawContextRouteLesson, ...],
    *,
    mode: str,
    previous: RawContextTrainingSummary | None,
    candidate_branch_count: int,
) -> FrozenCommonRawContextFactorDiscoveryV1:
    admitted_array = np.asarray(admitted, dtype=CUE_DTYPE).reshape(
        len(admitted), TRACE_LANES
    )
    parent_summary = factor_router.training_summary
    base = previous

    def prior(name: str) -> int:
        return 0 if base is None else int(getattr(base, name))

    summary_record = {
        "mode": mode,
        "new_raw_lesson_count": len(lessons),
        "total_raw_lesson_count": prior("total_raw_lesson_count") + len(lessons),
        "projection_off_count": prior("projection_off_count")
        + sum(not lesson.projection_gate for lesson in lessons),
        "plasticity_off_count": prior("plasticity_off_count")
        + sum(not lesson.plasticity_gate for lesson in lessons),
        "candidate_branch_count": candidate_branch_count,
        "parent_factor_count": parent_summary.factor_count,
        "accepted_factor_vote_count": parent_summary.accepted_factor_vote_count,
        "duplicate_nonvote_count": parent_summary.duplicate_nonvote_count,
        "dependent_nonvote_count": parent_summary.dependent_nonvote_count,
        "route_trainable_state_bytes": parent_summary.trainable_state_bytes,
        "provenance_state_bytes": parent_summary.provenance_state_bytes,
        "parent_router_learning_cell_steps": parent_summary.router_learning_cell_steps,
        "admission_probe_count": prior("admission_probe_count")
        + admission_totals.probes,
        "admitted_factor_count": len(admitted),
        "new_admitted_factor_count": len(admitted)
        - (0 if base is None else base.admitted_factor_count),
        "admission_probe_router_cell_steps": prior(
            "admission_probe_router_cell_steps"
        )
        + admission_totals.router_steps,
        "admission_probe_exact_child_work": prior(
            "admission_probe_exact_child_work"
        )
        + admission_totals.exact_work,
        "admission_probe_semantic_cell_steps": prior(
            "admission_probe_semantic_cell_steps"
        )
        + admission_totals.semantic_steps,
        "admission_support_cell_steps": prior("admission_support_cell_steps")
        + admission_totals.support_steps,
        "admission_state_bytes": admitted_array.nbytes,
        "checkpoint_array_bytes": parent_summary.checkpoint_array_bytes
        + admitted_array.nbytes,
        "fixed_discovery_cell_count": FIXED_DISCOVERY_CELL_COUNT,
        "branch_match_cell_count": len(admitted),
        "added_discovery_cell_count": FIXED_DISCOVERY_CELL_COUNT + len(admitted),
        "added_composite_cell_count": parent_summary.added_cell_count
        + FIXED_DISCOVERY_CELL_COUNT
        + len(admitted),
        "parent_lesson_replay_count": 0,
        "retained_parent_count": 0,
        "raw_lesson_retention_count": 0,
        "child_mutation_count": 0,
        "factor_router_checkpoint_sha256": factor_router.checkpoint_sha256,
        "memory_checkpoint_sha256": factor_router.memory_checkpoint_sha256,
        "episodic_checkpoint_sha256": factor_router.episodic_checkpoint_sha256,
        "semantic_checkpoint_sha256": factor_router.semantic_checkpoint_sha256,
    }
    placeholder = RawContextTrainingSummary(**summary_record, checkpoint_sha256="")
    checkpoint = _checkpoint_sha256(factor_router, placeholder, admitted_array)
    summary = dataclasses.replace(placeholder, checkpoint_sha256=checkpoint)
    return FrozenCommonRawContextFactorDiscoveryV1(
        factor_router=factor_router,
        admitted_cells=admitted_array,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def fit_common_raw_context_factor_discovery_v1(
    child: memory.FrozenCommonEpisodicTransformMemoryV2,
    lessons: tuple[PhysicalRawContextRouteLesson, ...],
) -> FrozenCommonRawContextFactorDiscoveryV1:
    """Fit route evidence from raw contexts and latch reusable branches."""

    _validate_lessons(lessons)
    child.authenticate()
    child_root = child.checkpoint_sha256
    parent_lessons = tuple(_parent_lesson(lesson) for lesson in lessons)
    factor_router = router.fit_common_factorized_context_routed_memory_v2(
        child, parent_lessons
    )
    witnesses = _candidate_witnesses(lessons)
    admitted, totals = _discover_new_admissions(factor_router, (), witnesses)
    result = _freeze(
        factor_router,
        admitted,
        totals,
        lessons,
        mode="INITIAL_RAW_CONTEXT_FIT",
        previous=None,
        candidate_branch_count=len(witnesses),
    )
    child.authenticate()
    if child.checkpoint_sha256 != child_root:
        raise RuntimeError("raw-context fit mutated its dual-view child")
    return result


def advance_common_raw_context_factor_discovery_v1(
    parent: FrozenCommonRawContextFactorDiscoveryV1,
    new_lessons: tuple[PhysicalRawContextRouteLesson, ...],
) -> FrozenCommonRawContextFactorDiscoveryV1:
    """Consume only new raw lessons and return one flat immutable successor."""

    if type(parent) is not FrozenCommonRawContextFactorDiscoveryV1:
        raise TypeError("parent must be one exact frozen raw-context discovery")
    _validate_lessons(new_lessons)
    parent.authenticate()
    parent_root = parent.checkpoint_sha256
    child_root = parent.memory_checkpoint_sha256
    factor_router = router.advance_common_factorized_context_routed_memory_v2(
        parent._factor_router,
        tuple(_parent_lesson(lesson) for lesson in new_lessons),
    )
    witnesses = _candidate_witnesses(new_lessons)
    admitted, totals = _discover_new_admissions(
        factor_router, parent.admitted_factor_cells, witnesses
    )
    result = _freeze(
        factor_router,
        admitted,
        totals,
        new_lessons,
        mode="REPLAY_FREE_RAW_CONTEXT_ADVANCE",
        previous=parent.training_summary,
        candidate_branch_count=len(witnesses),
    )
    parent.authenticate()
    if (
        parent.checkpoint_sha256 != parent_root
        or parent.memory_checkpoint_sha256 != child_root
    ):
        raise RuntimeError("raw-context advance mutated its parent or child")
    return result
