"""Relation-gated one-byte transform learned from physical filler neurons.

The banked CommonEvent V2 projection exposes four reusable input filler cells,
one per base-four lane.  This module learns local routes from those physical
cells to the unchanged CommonEvent V1 output population.  Relation, context,
source, input, and teacher evidence enter only as authenticated cell addresses.

COPY is the self-posed curriculum used by the focused experiment; the learner
itself can acquire any lane-local filler permutation.  It retains no raw byte,
digit tuple, event name, or whole-byte lookup.
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
from experiments.g0 import common_event_cells_v2 as projection

SCOPE = "COMMON_ONE_BYTE_COPY_RELATION_V2"
STATUS = "SELF_POSED_RELATION_GATED_ONE_BYTE_FILLER_RECOMBINATION"
CLAIM_BOUNDARY = (
    "self-posed-two-context-lane-local-one-byte-filler-recombination-only-"
    "no-language-coding-corpus-or-protected-transfer-evidence"
)

PROJECTION_SOURCE_COMMIT = "6248269"
EXPECTED_PROJECTION_SOURCE_SHA256 = (
    "66270079137d574ad283c9982c0f96828f60332fdc8f8db737c4fefdfb62671a"
)
EXPECTED_PROJECTION_CONFIG_SHA256 = (
    "337f1b0b9c2232924bb7beab8eb0391c13eb4b0154c47712bfbaff4af87e9b23"
)
EXPECTED_PROJECTION_SUBSTRATE_ROOT_SHA256 = (
    "88e1e0509a46472b1a05d150859a907dad277635b3ed95853788758b5d90380a"
)

MAX_RELATIONS = 8
MAX_INDEPENDENT_CONTEXTS = 64
RELATION_ASSEMBLY_CELLS = output_projection.TRACE_LANES
CONTEXT_ASSEMBLY_CELLS = output_projection.TRACE_LANES
SOURCE_ASSEMBLY_CELLS = output_projection.TRACE_LANES
LANES = projection.FILLER_LANES
RADIX = projection.FILLER_RADIX
INPUT_FILLER_CELLS = projection.FILLER_CELL_COUNT
MIN_CONTEXT_ROUTE_OBSERVATIONS = 2

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
POSITIVE_TICK = 1
NEGATIVE_TICK = 1
RELATION_TONIC_SCORE = output_projection.STOP_SCORE
ROUTE_SCORE_GAIN = RELATION_ASSEMBLY_CELLS
MAX_PHYSICAL_SCORE = TICK_CAP * ROUTE_SCORE_GAIN

RELATION_SCHEMA_CELL_OFFSET = projection.TOTAL_CELL_COUNT
RELATION_SCHEMA_CELL_COUNT = MAX_RELATIONS
ROUTE_SYNAPSE_CHANNEL_COUNT = MAX_RELATIONS * LANES * RADIX * RADIX
FUSED_ROUTE_SIMULATED_CELL_COUNT = 0
PROVENANCE_CELL_OFFSET = RELATION_SCHEMA_CELL_OFFSET + RELATION_SCHEMA_CELL_COUNT
PROVENANCE_CELL_COUNT = MAX_INDEPENDENT_CONTEXTS
RELATION_TONIC_CELL_ID = PROVENANCE_CELL_OFFSET + PROVENANCE_CELL_COUNT
NEW_CELL_COUNT = (
    RELATION_SCHEMA_CELL_COUNT
    + PROVENANCE_CELL_COUNT
    + 1
)
REUSED_PROJECTION_CELL_COUNT = projection.TOTAL_CELL_COUNT
REUSED_OUTPUT_CELL_COUNT = output_projection.OUTPUT_CELL_COUNT
REUSED_OUTPUT_TONIC_CELL_ID = output_projection.STOP_CELL_ID
TOTAL_CELL_COUNT = REUSED_PROJECTION_CELL_COUNT + NEW_CELL_COUNT

ROUTE_EH_STATE_BYTES = 2 * ROUTE_SYNAPSE_CHANNEL_COUNT * TICK_DTYPE.itemsize
RELATION_STATE_BYTES = (
    MAX_RELATIONS * RELATION_ASSEMBLY_CELLS * TICK_DTYPE.itemsize
)

GENERATOR_EXECUTION_COUNT = 0
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0
PRIOR_MEMORY_MUTATION_COUNT = 0
PRIOR_SCHEMA_MUTATION_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_PROJECTION_SOURCE_PATH = _ROOT / "experiments/g0/common_event_cells_v2.py"
_CONFIG_NAMESPACE = b"common-one-byte-copy-relation-v2-config\0"
_CHECKPOINT_NAMESPACE = b"common-one-byte-copy-relation-v2-checkpoint\0"
_RECALL_NAMESPACE = b"common-one-byte-copy-relation-v2-recall\0"
_RELATION_STATE_NAMESPACE = b"common-one-byte-copy-relation-v2-state\0"
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
        {"name": name, "dtype": array.dtype.str, "shape": array.shape}
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
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> bytes:
    return (
        _CHECKPOINT_NAMESPACE
        + len(metadata).to_bytes(8, "little")
        + metadata
        + b"".join(_array_frame(name, array) for name, array in arrays)
    )


def _authenticate_dependencies() -> None:
    if (
        sha256(_PROJECTION_SOURCE_PATH.read_bytes()).hexdigest()
        != EXPECTED_PROJECTION_SOURCE_SHA256
        or projection.SOURCE_SHA256 != EXPECTED_PROJECTION_SOURCE_SHA256
        or projection.CONFIG_SHA256 != EXPECTED_PROJECTION_CONFIG_SHA256
        or projection.SUBSTRATE_ROOT_SHA256
        != EXPECTED_PROJECTION_SUBSTRATE_ROOT_SHA256
        or projection.FILLER_CELL_COUNT != 16
        or projection.TOTAL_CELL_COUNT != 65_569
        or output_projection.OUTPUT_CELL_COUNT != 17
        or output_projection.STOP_CELL_ID != REUSED_OUTPUT_TONIC_CELL_ID
        or any(
            (
                projection.LEARNER_EXECUTION_COUNT,
                projection.CORPUS_INPUT_COUNT,
                projection.PATH2_INPUT_COUNT,
                projection.DEV_INPUT_COUNT,
                projection.M0_INPUT_COUNT,
                projection.PROTECTED_INPUT_COUNT,
                projection.G0_INPUT_COUNT,
                projection.CODING_INPUT_COUNT,
            )
        )
    ):
        raise RuntimeError("banked CommonEvent V2 projection identity changed")


def _validate_opaque_assembly(name: str, cells: tuple[int, ...]) -> None:
    if (
        type(cells) is not tuple
        or len(cells) != RELATION_ASSEMBLY_CELLS
        or any(type(cell) is not int for cell in cells)
    ):
        raise TypeError(f"{name} must be an exact physical trace-cell tuple")
    for lane, cell_id in enumerate(cells):
        lane_start = (
            output_projection.TRACE_CELL_OFFSET
            + lane * output_projection.FEATURE_BANK_CELLS
        )
        if not lane_start <= cell_id < lane_start + output_projection.FEATURE_BANK_CELLS:
            raise ValueError(f"{name} physical trace lane changed")


_OUTPUT_CANDIDATES_BY_LANE = tuple(
    tuple(
        output_projection.OUTPUT_CELL_OFFSET + lane * RADIX + candidate
        for candidate in range(RADIX)
    )
    for lane in range(LANES)
)


def _teacher_candidate_indices(
    teacher_output_cells: tuple[int, ...],
) -> tuple[int, ...]:
    if (
        type(teacher_output_cells) is not tuple
        or len(teacher_output_cells) != LANES
        or len(set(teacher_output_cells)) != LANES
        or any(type(cell) is not int for cell in teacher_output_cells)
    ):
        raise TypeError("teacher output must be four distinct physical V1 cells")
    selected: list[int] = []
    for candidates in _OUTPUT_CANDIDATES_BY_LANE:
        local = tuple(cell for cell in teacher_output_cells if cell in candidates)
        if len(local) != 1:
            raise ValueError("teacher output must activate one physical cell per lane")
        selected.append(candidates.index(local[0]))
    if any(
        cell not in {candidate for row in _OUTPUT_CANDIDATES_BY_LANE for candidate in row}
        for cell in teacher_output_cells
    ):
        raise ValueError("teacher output contains a non-output cell")
    return tuple(selected)


def _input_route_indices(
    cells: projection.CommonOneByteFillerCells,
    frame: projection.CommonOneByteFillerFrame,
) -> tuple[int, ...]:
    cells.authenticate_frame(frame)
    routes: list[int] = []
    for lane, cell_id in enumerate(frame.filler_cells):
        lane_start = projection.FILLER_CELL_OFFSET + lane * RADIX
        if not lane_start <= cell_id < lane_start + RADIX:
            raise ValueError("input filler physical lane changed")
        routes.append(cell_id - projection.FILLER_CELL_OFFSET)
    return tuple(routes)


@dataclass(frozen=True, slots=True)
class RelationGatedOneByteLesson:
    input_frame: projection.CommonOneByteFillerFrame
    relation_cells: tuple[int, ...]
    context_cells: tuple[int, ...]
    source_cells: tuple[int, ...]
    teacher_output_cells: tuple[int, ...]
    teacher_gate: bool = True
    cell_gate: bool = True

    def __post_init__(self) -> None:
        if type(self.input_frame) is not projection.CommonOneByteFillerFrame:
            raise TypeError("input_frame must have the exact physical V2 frame type")
        _validate_opaque_assembly("relation_cells", self.relation_cells)
        _validate_opaque_assembly("context_cells", self.context_cells)
        _validate_opaque_assembly("source_cells", self.source_cells)
        if self.context_cells == self.source_cells:
            raise ValueError("context and source assemblies must be physically distinct")
        _teacher_candidate_indices(self.teacher_output_cells)
        if type(self.teacher_gate) is not bool or type(self.cell_gate) is not bool:
            raise TypeError("teacher_gate and cell_gate must be exact booleans")


@dataclass(frozen=True, slots=True)
class ContextConsolidationStep:
    first_lesson_ordinal: int
    status: str
    relation_schema_cell_id: int | None
    provenance_cell_id: int | None
    complete_route_count: int
    minimum_route_observations: int
    relation_support_count: int
    excitatory_update_ticks: int
    inhibitory_update_ticks: int


@dataclass(frozen=True, slots=True)
class PhysicalContextProvenance:
    provenance_cell_id: int
    relation_schema_cell_id: int
    context_cells: tuple[int, ...]
    source_cells: tuple[int, ...]
    route_output_cells: tuple[int, ...]
    minimum_route_observations: int


@dataclass(frozen=True, slots=True)
class RelationGatedTrainingSummary:
    lesson_count: int
    teacher_on_lesson_count: int
    teacher_off_lesson_count: int
    cell_on_lesson_count: int
    cell_off_lesson_count: int
    context_group_count: int
    independent_context_count: int
    dependent_context_count: int
    incomplete_context_count: int
    contradictory_context_count: int
    relation_count: int
    local_physical_update_ticks: int
    relation_excitatory_update_ticks: int
    route_excitatory_update_ticks: int
    route_inhibitory_update_ticks: int
    maximum_excitatory_tick: int
    maximum_inhibitory_tick: int
    synaptic_state_bytes: int
    checkpoint_array_bytes: int
    context_steps: tuple[ContextConsolidationStep, ...]
    provenance: tuple[PhysicalContextProvenance, ...]
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class RelationGatedCopyRecall:
    status: str
    output: output_projection.CommonOutputDecision
    relation_schema_cell_id: int | None
    relation_support_count: int
    provenance_context_count: int
    input_filler_cells: tuple[int, ...]
    relation_cells: tuple[int, ...]
    relation_raw_scores: tuple[int, ...]
    route_raw_scores: tuple[tuple[int, ...], ...]
    relation_spike_step: int | None
    relation_simultaneous_spike_count: int | None
    physical_cell_steps: int
    checkpoint_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        record["output"] = self.output.record()
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class _RelationRace:
    decision: str
    winner_index: int | None
    spike_step: int
    simultaneous_spike_count: int
    physical_cell_steps: int


@dataclass(slots=True)
class _ContextAccumulator:
    first_ordinal: int
    relation_cells: tuple[int, ...]
    context_cells: tuple[int, ...]
    source_cells: tuple[int, ...]
    counts: np.ndarray


def _race_relation(raw_scores: tuple[int, ...]) -> _RelationRace:
    bounded = tuple(max(0, min(score, MAX_PHYSICAL_SCORE)) for score in raw_scores)
    currents = np.asarray(
        [[
            *(output_projection._CURRENT_TABLE[score] for score in bounded),
            output_projection._CURRENT_TABLE[RELATION_TONIC_SCORE],
        ]],
        dtype=np.float64,
    )
    race = output_projection._OUTPUT_RACE.race_many(
        currents,
        require_unique=False,
    )[0]
    if race.simultaneous_spike_candidates != 1:
        decision = "COLLISION"
        winner = None
    elif race.winner_cell == len(raw_scores):
        decision = "TONIC_STOP"
        winner = None
    else:
        decision = "WINNER"
        winner = race.winner_cell
    return _RelationRace(
        decision=decision,
        winner_index=winner,
        spike_step=race.spike_step,
        simultaneous_spike_count=race.simultaneous_spike_candidates,
        physical_cell_steps=currents.shape[1] * output_projection.RACE_STEPS,
    )


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "projection_source_commit": PROJECTION_SOURCE_COMMIT,
    "projection_source_sha256": EXPECTED_PROJECTION_SOURCE_SHA256,
    "projection_config_sha256": EXPECTED_PROJECTION_CONFIG_SHA256,
    "projection_substrate_root_sha256": (
        EXPECTED_PROJECTION_SUBSTRATE_ROOT_SHA256
    ),
    "relation_assembly_cells": RELATION_ASSEMBLY_CELLS,
    "maximum_relations": MAX_RELATIONS,
    "maximum_independent_contexts": MAX_INDEPENDENT_CONTEXTS,
    "lanes": LANES,
    "radix": RADIX,
    "minimum_context_route_observations": MIN_CONTEXT_ROUTE_OBSERVATIONS,
    "tick_dtype": TICK_DTYPE.str,
    "tick_cap": TICK_CAP,
    "relation_tonic_score": RELATION_TONIC_SCORE,
    "route_score_gain": ROUTE_SCORE_GAIN,
    "independence": "relation-local-novel-physical-context-and-source-assemblies",
    "route_synapse_geometry": (
        "relation-by-lane-by-input-filler-by-output-candidate-local-synapses"
    ),
    "route_synapse_channel_count": ROUTE_SYNAPSE_CHANNEL_COUNT,
    "route_simulated_cell_count": FUSED_ROUTE_SIMULATED_CELL_COUNT,
    "route_execution": (
        "direct-fused-synaptic-accumulation-into-reused-v1-output-glif-cells"
    ),
    "plasticity": (
        "teacher-gated-positive-target-and-negative-previously-observed-rival"
    ),
    "raw_byte_digit_name_or_hash_used_after_projection": False,
    "whole_byte_static_result_lookup": False,
    "new_cell_count": NEW_CELL_COUNT,
    "total_cell_count": TOTAL_CELL_COUNT,
    "route_eh_state_bytes": ROUTE_EH_STATE_BYTES,
    "domain_specific_computational_primitive_delta": 0,
    "generator_corpus_path2_dev_m0_protected_g0_coding_inputs": (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ),
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)


class FrozenCommonOneByteCopyRelationV2:
    """Immutable relation-conditioned filler routes and physical provenance."""

    __slots__ = (
        "_cells",
        "_checkpoint_sha256",
        "_context_count",
        "_context_relation_slots",
        "_context_route_targets",
        "_context_sources",
        "_context_subjects",
        "_metadata",
        "_relation_cells",
        "_relation_count",
        "_relation_excitatory",
        "_relation_support_counts",
        "_route_excitatory",
        "_route_inhibitory",
        "_summary",
    )

    def __init__(
        self,
        *,
        relation_cells: np.ndarray,
        relation_excitatory: np.ndarray,
        relation_support_counts: np.ndarray,
        route_excitatory: np.ndarray,
        route_inhibitory: np.ndarray,
        context_relation_slots: np.ndarray,
        context_subjects: np.ndarray,
        context_sources: np.ndarray,
        context_route_targets: np.ndarray,
        relation_count: int,
        context_count: int,
        metadata: bytes,
        summary: RelationGatedTrainingSummary,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("relation-gated one-byte transforms are builder-owned")
        expected = (
            (
                relation_cells,
                (MAX_RELATIONS, RELATION_ASSEMBLY_CELLS),
                np.dtype(np.int32),
            ),
            (
                relation_excitatory,
                (MAX_RELATIONS, RELATION_ASSEMBLY_CELLS),
                TICK_DTYPE,
            ),
            (relation_support_counts, (MAX_RELATIONS,), TICK_DTYPE),
            (
                route_excitatory,
                (MAX_RELATIONS, LANES, RADIX, RADIX),
                TICK_DTYPE,
            ),
            (
                route_inhibitory,
                (MAX_RELATIONS, LANES, RADIX, RADIX),
                TICK_DTYPE,
            ),
            (
                context_relation_slots,
                (MAX_INDEPENDENT_CONTEXTS,),
                np.dtype(np.int8),
            ),
            (
                context_subjects,
                (MAX_INDEPENDENT_CONTEXTS, CONTEXT_ASSEMBLY_CELLS),
                np.dtype(np.int32),
            ),
            (
                context_sources,
                (MAX_INDEPENDENT_CONTEXTS, SOURCE_ASSEMBLY_CELLS),
                np.dtype(np.int32),
            ),
            (
                context_route_targets,
                (MAX_INDEPENDENT_CONTEXTS, INPUT_FILLER_CELLS),
                np.dtype(np.int8),
            ),
        )
        if any(
            array.shape != shape
            or array.dtype != dtype
            or not array.flags.c_contiguous
            for array, shape, dtype in expected
        ):
            raise ValueError("relation-gated route checkpoint geometry changed")
        if not 0 <= relation_count <= MAX_RELATIONS:
            raise ValueError("relation count changed")
        if not 0 <= context_count <= MAX_INDEPENDENT_CONTEXTS:
            raise ValueError("context count changed")
        self._cells = projection.CommonOneByteFillerCells()
        self._relation_count = relation_count
        self._context_count = context_count
        self._metadata = bytes(metadata)
        self._summary = summary
        for name, array in (
            ("_relation_cells", relation_cells),
            ("_relation_excitatory", relation_excitatory),
            ("_relation_support_counts", relation_support_counts),
            ("_route_excitatory", route_excitatory),
            ("_route_inhibitory", route_inhibitory),
            ("_context_relation_slots", context_relation_slots),
            ("_context_subjects", context_subjects),
            ("_context_sources", context_sources),
            ("_context_route_targets", context_route_targets),
        ):
            copied = np.array(array, copy=True, order="C")
            copied.setflags(write=False)
            setattr(self, name, copied)
        self._checkpoint_sha256 = self._calculate_checkpoint_sha256()
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise RuntimeError("relation-gated route checkpoint identity changed")
        self.authenticate()

    def _checkpoint_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("relation-cells", self._relation_cells),
            ("relation-excitatory", self._relation_excitatory),
            ("relation-support-counts", self._relation_support_counts),
            ("route-excitatory", self._route_excitatory),
            ("route-inhibitory", self._route_inhibitory),
            ("context-relation-slots", self._context_relation_slots),
            ("context-subjects", self._context_subjects),
            ("context-sources", self._context_sources),
            ("context-route-targets", self._context_route_targets),
        )

    def _calculate_checkpoint_sha256(self) -> str:
        return sha256(
            _checkpoint_payload(self._metadata, self._checkpoint_arrays())
        ).hexdigest()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> RelationGatedTrainingSummary:
        self.authenticate()
        return self._summary

    def authenticate(self) -> None:
        _authenticate_dependencies()
        if self._calculate_checkpoint_sha256() != self._checkpoint_sha256:
            raise RuntimeError("relation-gated route checkpoint authentication failed")

    def _matching_relation_slot(self, relation_cells: tuple[int, ...]) -> int | None:
        active = np.asarray(relation_cells, dtype=np.int32)
        matches = tuple(
            row
            for row in range(self._relation_count)
            if bool(np.all(self._relation_cells[row] == active))
        )
        if len(matches) > 1:
            raise RuntimeError("relation-gated route duplicated a physical relation")
        return matches[0] if matches else None

    def relation_support_count(self, relation_cells: tuple[int, ...]) -> int:
        self.authenticate()
        _validate_opaque_assembly("relation_cells", relation_cells)
        slot = self._matching_relation_slot(relation_cells)
        return 0 if slot is None else int(self._relation_support_counts[slot])

    def relation_state_sha256(self, relation_cells: tuple[int, ...]) -> str | None:
        """Root one relation's state so unrelated schemas can be compared."""

        self.authenticate()
        _validate_opaque_assembly("relation_cells", relation_cells)
        slot = self._matching_relation_slot(relation_cells)
        if slot is None:
            return None
        context_rows = np.flatnonzero(
            self._context_relation_slots[: self._context_count] == slot
        )
        payload = (
            _array_frame("relation", self._relation_cells[slot : slot + 1])
            + _array_frame(
                "relation-excitatory",
                self._relation_excitatory[slot : slot + 1],
            )
            + _array_frame(
                "route-excitatory",
                self._route_excitatory[slot : slot + 1],
            )
            + _array_frame(
                "route-inhibitory",
                self._route_inhibitory[slot : slot + 1],
            )
            + _array_frame("context-subjects", self._context_subjects[context_rows])
            + _array_frame("context-sources", self._context_sources[context_rows])
            + _array_frame(
                "context-route-targets",
                self._context_route_targets[context_rows],
            )
        )
        return sha256(_RELATION_STATE_NAMESPACE + payload).hexdigest()

    def _stop_recall(
        self,
        *,
        status: str,
        relation_cells: tuple[int, ...],
        input_filler_cells: tuple[int, ...],
        relation_raw_scores: tuple[int, ...] = (),
        relation_race: _RelationRace | None = None,
    ) -> RelationGatedCopyRecall:
        output = self._cells.race_output(((0, 0, 0, 0),) * LANES)
        record = {
            "status": status,
            "output": output.record(),
            "relation_schema_cell_id": None,
            "relation_support_count": 0,
            "provenance_context_count": 0,
            "input_filler_cells": input_filler_cells,
            "relation_cells": relation_cells,
            "relation_raw_scores": relation_raw_scores,
            "route_raw_scores": (),
            "relation_spike_step": (
                None if relation_race is None else relation_race.spike_step
            ),
            "relation_simultaneous_spike_count": (
                None
                if relation_race is None
                else relation_race.simultaneous_spike_count
            ),
            "physical_cell_steps": output.physical_cell_steps
            + (0 if relation_race is None else relation_race.physical_cell_steps),
            "checkpoint_sha256": self._checkpoint_sha256,
        }
        return RelationGatedCopyRecall(
            **{key: value for key, value in record.items() if key != "output"},
            output=output,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )

    def recall(
        self,
        input_frame: projection.CommonOneByteFillerFrame,
        relation_cells: tuple[int, ...],
        *,
        cell_gate: bool = True,
        relation_gate: bool = True,
    ) -> RelationGatedCopyRecall:
        """Recall through physical relation and filler routes only."""

        self.authenticate()
        if type(cell_gate) is not bool or type(relation_gate) is not bool:
            raise TypeError("cell_gate and relation_gate must be exact booleans")
        _validate_opaque_assembly("relation_cells", relation_cells)
        try:
            input_routes = _input_route_indices(self._cells, input_frame)
        except (TypeError, ValueError):
            return self._stop_recall(
                status="STOP_INPUT_AUTHENTICATION",
                relation_cells=relation_cells,
                input_filler_cells=(),
            )
        input_fillers = input_frame.filler_cells
        if not cell_gate:
            return self._stop_recall(
                status="STOP_CELL_OFF",
                relation_cells=relation_cells,
                input_filler_cells=input_fillers,
            )
        if not relation_gate:
            return self._stop_recall(
                status="STOP_RELATION_OFF",
                relation_cells=relation_cells,
                input_filler_cells=input_fillers,
            )

        active_relation = np.asarray(relation_cells, dtype=np.int32)
        relation_scores = tuple(
            int(
                np.sum(
                    np.where(
                        self._relation_cells[row] == active_relation,
                        self._relation_excitatory[row],
                        0,
                    ),
                    dtype=np.int64,
                )
            )
            for row in range(self._relation_count)
        )
        relation_race = _race_relation(relation_scores)
        if relation_race.winner_index is None:
            status = (
                "ABSTAIN_UNKNOWN_RELATION"
                if max(relation_scores, default=0) == 0
                else "ABSTAIN_LOW_SUPPORT"
            )
            return self._stop_recall(
                status=status,
                relation_cells=relation_cells,
                input_filler_cells=input_fillers,
                relation_raw_scores=relation_scores,
                relation_race=relation_race,
            )

        slot = relation_race.winner_index
        route_rows: list[tuple[int, ...]] = []
        conflict = False
        for lane, flat_input in enumerate(input_routes):
            local_input = flat_input - lane * RADIX
            excitatory = self._route_excitatory[slot, lane, local_input]
            inhibitory = self._route_inhibitory[slot, lane, local_input]
            route_rows.append(
                tuple(
                    (int(excitatory[candidate]) - int(inhibitory[candidate]))
                    * ROUTE_SCORE_GAIN
                    for candidate in range(RADIX)
                )
            )
            conflict = conflict or int(np.count_nonzero(excitatory)) > 1
        route_scores = tuple(route_rows)
        output = self._cells.race_output(route_scores)
        status = "READY" if output.decision == "TOKEN" else (
            "ABSTAIN_CONFLICT" if conflict else "ABSTAIN_LOW_SUPPORT"
        )
        support_count = int(self._relation_support_counts[slot])
        provenance_count = int(
            np.count_nonzero(
                self._context_relation_slots[: self._context_count] == slot
            )
        )
        record = {
            "status": status,
            "output": output.record(),
            "relation_schema_cell_id": RELATION_SCHEMA_CELL_OFFSET + slot,
            "relation_support_count": support_count,
            "provenance_context_count": provenance_count,
            "input_filler_cells": input_fillers,
            "relation_cells": relation_cells,
            "relation_raw_scores": relation_scores,
            "route_raw_scores": route_scores,
            "relation_spike_step": relation_race.spike_step,
            "relation_simultaneous_spike_count": (
                relation_race.simultaneous_spike_count
            ),
            "physical_cell_steps": (
                relation_race.physical_cell_steps + output.physical_cell_steps
            ),
            "checkpoint_sha256": self._checkpoint_sha256,
        }
        return RelationGatedCopyRecall(
            **{key: value for key, value in record.items() if key != "output"},
            output=output,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )


def fit_common_one_byte_copy_relation_v2(
    lessons: tuple[RelationGatedOneByteLesson, ...],
) -> FrozenCommonOneByteCopyRelationV2:
    """Learn and freeze local physical routes without reopening host labels."""

    if type(lessons) is not tuple or any(
        type(lesson) is not RelationGatedOneByteLesson for lesson in lessons
    ):
        raise TypeError("lessons must be an exact tuple of physical lessons")
    _authenticate_dependencies()
    cells = projection.CommonOneByteFillerCells()
    groups: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        _ContextAccumulator,
    ] = {}
    local_update_ticks = 0

    for ordinal, lesson in enumerate(lessons):
        input_routes = _input_route_indices(cells, lesson.input_frame)
        teacher_candidates = _teacher_candidate_indices(
            lesson.teacher_output_cells
        )
        if not lesson.teacher_gate or not lesson.cell_gate:
            continue
        key = (lesson.relation_cells, lesson.context_cells, lesson.source_cells)
        accumulator = groups.get(key)
        if accumulator is None:
            accumulator = _ContextAccumulator(
                first_ordinal=ordinal,
                relation_cells=lesson.relation_cells,
                context_cells=lesson.context_cells,
                source_cells=lesson.source_cells,
                counts=np.zeros((INPUT_FILLER_CELLS, RADIX), dtype=TICK_DTYPE),
            )
            groups[key] = accumulator
        for lane, flat_input in enumerate(input_routes):
            candidate = teacher_candidates[lane]
            previous = int(accumulator.counts[flat_input, candidate])
            updated = min(previous + POSITIVE_TICK, TICK_CAP)
            accumulator.counts[flat_input, candidate] = updated
            local_update_ticks += updated - previous

    relation_cells = np.full(
        (MAX_RELATIONS, RELATION_ASSEMBLY_CELLS),
        -1,
        dtype=np.int32,
    )
    relation_excitatory = np.zeros(
        (MAX_RELATIONS, RELATION_ASSEMBLY_CELLS),
        dtype=TICK_DTYPE,
    )
    relation_support_counts = np.zeros(MAX_RELATIONS, dtype=TICK_DTYPE)
    route_excitatory = np.zeros(
        (MAX_RELATIONS, LANES, RADIX, RADIX),
        dtype=TICK_DTYPE,
    )
    route_inhibitory = np.zeros_like(route_excitatory)
    context_relation_slots = np.full(
        MAX_INDEPENDENT_CONTEXTS,
        -1,
        dtype=np.int8,
    )
    context_subjects = np.full(
        (MAX_INDEPENDENT_CONTEXTS, CONTEXT_ASSEMBLY_CELLS),
        -1,
        dtype=np.int32,
    )
    context_sources = np.full(
        (MAX_INDEPENDENT_CONTEXTS, SOURCE_ASSEMBLY_CELLS),
        -1,
        dtype=np.int32,
    )
    context_route_targets = np.full(
        (MAX_INDEPENDENT_CONTEXTS, INPUT_FILLER_CELLS),
        -1,
        dtype=np.int8,
    )
    relation_count = 0
    context_count = 0
    dependent_context_count = 0
    incomplete_context_count = 0
    contradictory_context_count = 0
    relation_update_ticks = 0
    route_e_ticks = 0
    route_h_ticks = 0
    steps: list[ContextConsolidationStep] = []
    provenance: list[PhysicalContextProvenance] = []

    for group in sorted(groups.values(), key=lambda row: row.first_ordinal):
        target_candidates: list[int] = []
        minimum_observations = TICK_CAP
        complete = True
        internally_contradictory = False
        for flat_input in range(INPUT_FILLER_CELLS):
            active = np.flatnonzero(group.counts[flat_input] > 0)
            if len(active) != 1:
                complete = False
                internally_contradictory = internally_contradictory or len(active) > 1
                target_candidates.append(-1)
                continue
            target = int(active[0])
            observations = int(group.counts[flat_input, target])
            minimum_observations = min(minimum_observations, observations)
            if observations < MIN_CONTEXT_ROUTE_OBSERVATIONS:
                complete = False
            target_candidates.append(target)
        complete_route_count = sum(target >= 0 for target in target_candidates)
        if not complete:
            incomplete_context_count += 1
            contradictory_context_count += int(internally_contradictory)
            steps.append(
                ContextConsolidationStep(
                    first_lesson_ordinal=group.first_ordinal,
                    status=(
                        "CONTEXT_CONTRADICTION"
                        if internally_contradictory
                        else "CONTEXT_INCOMPLETE"
                    ),
                    relation_schema_cell_id=None,
                    provenance_cell_id=None,
                    complete_route_count=complete_route_count,
                    minimum_route_observations=(
                        0 if minimum_observations == TICK_CAP else minimum_observations
                    ),
                    relation_support_count=0,
                    excitatory_update_ticks=0,
                    inhibitory_update_ticks=0,
                )
            )
            continue

        active_relation = np.asarray(group.relation_cells, dtype=np.int32)
        matching = tuple(
            row
            for row in range(relation_count)
            if bool(np.all(relation_cells[row] == active_relation))
        )
        if len(matching) > 1:
            raise RuntimeError("physical relation schema duplicated")
        if matching:
            slot = matching[0]
        else:
            if relation_count == MAX_RELATIONS:
                raise RuntimeError("relation schema capacity exhausted")
            slot = relation_count
            relation_count += 1
            relation_cells[slot] = active_relation

        dependent = any(
            int(context_relation_slots[row]) == slot
            and (
                bool(
                    np.all(
                        context_subjects[row]
                        == np.asarray(group.context_cells, dtype=np.int32)
                    )
                )
                or bool(
                    np.all(
                        context_sources[row]
                        == np.asarray(group.source_cells, dtype=np.int32)
                    )
                )
            )
            for row in range(context_count)
        )
        if dependent:
            dependent_context_count += 1
            steps.append(
                ContextConsolidationStep(
                    first_lesson_ordinal=group.first_ordinal,
                    status="DEPENDENT_CONTEXT_IGNORED",
                    relation_schema_cell_id=RELATION_SCHEMA_CELL_OFFSET + slot,
                    provenance_cell_id=None,
                    complete_route_count=complete_route_count,
                    minimum_route_observations=minimum_observations,
                    relation_support_count=int(relation_support_counts[slot]),
                    excitatory_update_ticks=0,
                    inhibitory_update_ticks=0,
                )
            )
            continue
        if context_count == MAX_INDEPENDENT_CONTEXTS:
            raise RuntimeError("physical context provenance capacity exhausted")

        provenance_cell_id = PROVENANCE_CELL_OFFSET + context_count
        context_relation_slots[context_count] = slot
        context_subjects[context_count] = np.asarray(
            group.context_cells,
            dtype=np.int32,
        )
        context_sources[context_count] = np.asarray(
            group.source_cells,
            dtype=np.int32,
        )
        context_route_targets[context_count] = np.asarray(
            target_candidates,
            dtype=np.int8,
        )

        previous_relation = relation_excitatory[slot].astype(np.uint16)
        updated_relation = np.minimum(previous_relation + POSITIVE_TICK, TICK_CAP)
        relation_excitatory[slot] = updated_relation.astype(TICK_DTYPE)
        committed_relation = int(
            np.sum(updated_relation - previous_relation, dtype=np.uint64)
        )
        relation_update_ticks += committed_relation
        relation_support_counts[slot] = min(
            int(relation_support_counts[slot]) + 1,
            TICK_CAP,
        )

        context_e_ticks = 0
        context_h_ticks = 0
        route_output_cells: list[int] = []
        for flat_input, target in enumerate(target_candidates):
            lane = flat_input // RADIX
            local_input = flat_input - lane * RADIX
            previous_targets = tuple(
                int(candidate)
                for candidate in np.flatnonzero(
                    route_excitatory[slot, lane, local_input] > 0
                )
                if int(candidate) != target
            )
            previous_e = int(route_excitatory[slot, lane, local_input, target])
            updated_e = min(previous_e + POSITIVE_TICK, TICK_CAP)
            route_excitatory[slot, lane, local_input, target] = updated_e
            context_e_ticks += updated_e - previous_e
            for rival in previous_targets:
                previous_h = int(
                    route_inhibitory[slot, lane, local_input, rival]
                )
                updated_h = min(previous_h + NEGATIVE_TICK, TICK_CAP)
                route_inhibitory[slot, lane, local_input, rival] = updated_h
                context_h_ticks += updated_h - previous_h
            route_output_cells.append(_OUTPUT_CANDIDATES_BY_LANE[lane][target])
        route_e_ticks += context_e_ticks
        route_h_ticks += context_h_ticks
        support_count = int(relation_support_counts[slot])
        provenance.append(
            PhysicalContextProvenance(
                provenance_cell_id=provenance_cell_id,
                relation_schema_cell_id=RELATION_SCHEMA_CELL_OFFSET + slot,
                context_cells=group.context_cells,
                source_cells=group.source_cells,
                route_output_cells=tuple(route_output_cells),
                minimum_route_observations=minimum_observations,
            )
        )
        steps.append(
            ContextConsolidationStep(
                first_lesson_ordinal=group.first_ordinal,
                status="INDEPENDENT_CONTEXT_CONSOLIDATED",
                relation_schema_cell_id=RELATION_SCHEMA_CELL_OFFSET + slot,
                provenance_cell_id=provenance_cell_id,
                complete_route_count=complete_route_count,
                minimum_route_observations=minimum_observations,
                relation_support_count=support_count,
                excitatory_update_ticks=committed_relation + context_e_ticks,
                inhibitory_update_ticks=context_h_ticks,
            )
        )
        context_count += 1

    arrays = (
        ("relation-cells", relation_cells),
        ("relation-excitatory", relation_excitatory),
        ("relation-support-counts", relation_support_counts),
        ("route-excitatory", route_excitatory),
        ("route-inhibitory", route_inhibitory),
        ("context-relation-slots", context_relation_slots),
        ("context-subjects", context_subjects),
        ("context-sources", context_sources),
        ("context-route-targets", context_route_targets),
    )
    checkpoint_array_bytes = sum(array.nbytes for _, array in arrays)
    metadata_record = {
        "scope": SCOPE,
        "status": STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "config_sha256": CONFIG_SHA256,
        "lesson_count": len(lessons),
        "context_group_count": len(groups),
        "independent_context_count": context_count,
        "relation_count": relation_count,
        "raw_byte_digit_name_hash_or_whole_byte_identity_retained": False,
        "prior_memory_mutation_count": 0,
        "prior_schema_mutation_count": 0,
        "generator_corpus_path2_dev_m0_protected_g0_coding_inputs": (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    }
    metadata = _canonical_json(metadata_record)
    checkpoint_sha256 = sha256(
        _checkpoint_payload(metadata, arrays)
    ).hexdigest()
    summary = RelationGatedTrainingSummary(
        lesson_count=len(lessons),
        teacher_on_lesson_count=sum(lesson.teacher_gate for lesson in lessons),
        teacher_off_lesson_count=sum(not lesson.teacher_gate for lesson in lessons),
        cell_on_lesson_count=sum(lesson.cell_gate for lesson in lessons),
        cell_off_lesson_count=sum(not lesson.cell_gate for lesson in lessons),
        context_group_count=len(groups),
        independent_context_count=context_count,
        dependent_context_count=dependent_context_count,
        incomplete_context_count=incomplete_context_count,
        contradictory_context_count=contradictory_context_count,
        relation_count=relation_count,
        local_physical_update_ticks=local_update_ticks,
        relation_excitatory_update_ticks=relation_update_ticks,
        route_excitatory_update_ticks=route_e_ticks,
        route_inhibitory_update_ticks=route_h_ticks,
        maximum_excitatory_tick=max(
            int(np.max(relation_excitatory, initial=0)),
            int(np.max(route_excitatory, initial=0)),
        ),
        maximum_inhibitory_tick=int(np.max(route_inhibitory, initial=0)),
        synaptic_state_bytes=(
            relation_excitatory.nbytes
            + route_excitatory.nbytes
            + route_inhibitory.nbytes
        ),
        checkpoint_array_bytes=checkpoint_array_bytes,
        context_steps=tuple(steps),
        provenance=tuple(provenance),
        checkpoint_sha256=checkpoint_sha256,
    )
    return FrozenCommonOneByteCopyRelationV2(
        relation_cells=relation_cells,
        relation_excitatory=relation_excitatory,
        relation_support_counts=relation_support_counts,
        route_excitatory=route_excitatory,
        route_inhibitory=route_inhibitory,
        context_relation_slots=context_relation_slots,
        context_subjects=context_subjects,
        context_sources=context_sources,
        context_route_targets=context_route_targets,
        relation_count=relation_count,
        context_count=context_count,
        metadata=metadata,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


_authenticate_dependencies()
SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "CLAIM_BOUNDARY",
    "CODING_INPUT_COUNT",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "CORPUS_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA",
    "FUSED_ROUTE_SIMULATED_CELL_COUNT",
    "G0_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "MAX_INDEPENDENT_CONTEXTS",
    "MAX_RELATIONS",
    "NEW_CELL_COUNT",
    "PATH2_INPUT_COUNT",
    "PRIOR_MEMORY_MUTATION_COUNT",
    "PRIOR_SCHEMA_MUTATION_COUNT",
    "PROTECTED_INPUT_COUNT",
    "REUSED_OUTPUT_TONIC_CELL_ID",
    "REUSED_PROJECTION_CELL_COUNT",
    "ROUTE_EH_STATE_BYTES",
    "ROUTE_SYNAPSE_CHANNEL_COUNT",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "TICK_CAP",
    "TOTAL_CELL_COUNT",
    "ContextConsolidationStep",
    "FrozenCommonOneByteCopyRelationV2",
    "PhysicalContextProvenance",
    "RelationGatedCopyRecall",
    "RelationGatedOneByteLesson",
    "RelationGatedTrainingSummary",
    "fit_common_one_byte_copy_relation_v2",
]
