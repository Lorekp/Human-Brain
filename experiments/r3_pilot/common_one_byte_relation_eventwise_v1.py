"""Eventwise neuronal consolidation for physical one-byte relation lessons.

The banked V2 learner proves sequential checkpoint extension, but its extension
interface still receives a host-grouped complete context.  This successor keeps
one active physical context in immutable state and accepts exactly one event at
a time.  Local evidence, conflict, ready-relay, tonic, poison, and complete-cell
races decide when the context is complete enough to consolidate into a cloned
V2 checkpoint.  No lesson tuple, raw byte, digit tuple, name, or whole-byte
identity is retained or replayed.

Physical context/source equality still supplies the independence check.  This
slice removes host batch grouping and completeness authority; it does not yet
learn context segmentation or source novelty.
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
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as relation_v2

SCOPE = "COMMON_ONE_BYTE_RELATION_EVENTWISE_V1"
STATUS = "EVENTWISE_PERSISTENT_NEURONAL_COMPLETENESS_CONSOLIDATION"
CLAIM_BOUNDARY = (
    "self-posed-single-active-context-eventwise-physical-one-byte-lane-local-"
    "accumulation-and-neuronal-completeness-consolidation-only-host-physical-"
    "address-independence-remains-no-language-coding-corpus-or-protected-"
    "transfer-evidence"
)

V2_SOURCE_COMMIT = "74250ab"
EXPECTED_V2_SOURCE_SHA256 = (
    "e10f6af9993fd17ab5b08fbd90c211ca0df91895863bbc6806ff49c43de24222"
)
EXPECTED_V2_CONFIG_SHA256 = (
    "68b2a7eb100041784531ac0675c5de5a68609df6d49dbb7fd98d3475ac89339c"
)

LANES = relation_v2.LANES
RADIX = relation_v2.RADIX
ROUTES = relation_v2.INPUT_FILLER_CELLS
ASSEMBLY_CELLS = relation_v2.RELATION_ASSEMBLY_CELLS
TICK_DTYPE = np.dtype(np.uint8)
READY_DTYPE = np.dtype(np.int8)
TICK_CAP = relation_v2.TICK_CAP
MIN_ROUTE_OBSERVATIONS = relation_v2.MIN_CONTEXT_ROUTE_OBSERVATIONS
ROUTE_SCORE_GAIN = relation_v2.ROUTE_SCORE_GAIN
TONIC_SCORE = relation_v2.RELATION_TONIC_SCORE
MAX_SCORE = relation_v2.MAX_PHYSICAL_SCORE

EVIDENCE_CELL_OFFSET = relation_v2.TOTAL_CELL_COUNT
EVIDENCE_CELL_COUNT = ROUTES * RADIX
CONFLICT_CELL_OFFSET = EVIDENCE_CELL_OFFSET + EVIDENCE_CELL_COUNT
CONFLICT_CELL_COUNT = ROUTES
READY_RELAY_CELL_OFFSET = CONFLICT_CELL_OFFSET + CONFLICT_CELL_COUNT
READY_RELAY_CELL_COUNT = ROUTES
COMPLETE_CELL_ID = READY_RELAY_CELL_OFFSET + READY_RELAY_CELL_COUNT
COMPLETE_TONIC_CELL_ID = COMPLETE_CELL_ID + 1
ACTIVE_CONTEXT_LATCH_CELL_ID = COMPLETE_TONIC_CELL_ID + 1
POISON_LATCH_CELL_ID = ACTIVE_CONTEXT_LATCH_CELL_ID + 1
NEW_CELL_COUNT = (
    EVIDENCE_CELL_COUNT
    + CONFLICT_CELL_COUNT
    + READY_RELAY_CELL_COUNT
    + 4
)
TOTAL_CELL_COUNT = relation_v2.TOTAL_CELL_COUNT + NEW_CELL_COUNT

EVIDENCE_STATE_BYTES = EVIDENCE_CELL_COUNT * TICK_DTYPE.itemsize
PENDING_ASSEMBLY_STATE_BYTES = 3 * ASSEMBLY_CELLS * np.dtype(np.int32).itemsize
READY_WINNER_STATE_BYTES = READY_RELAY_CELL_COUNT * READY_DTYPE.itemsize
LATCH_STATE_BYTES = 2
PERSISTENT_STATE_BYTES = (
    EVIDENCE_STATE_BYTES
    + PENDING_ASSEMBLY_STATE_BYTES
    + READY_WINNER_STATE_BYTES
    + LATCH_STATE_BYTES
)

ROUTE_RACE_CELL_COUNT = RADIX + 2
ROUTE_RELAY_CELL_COUNT = 1
TOUCHED_ROUTE_COUNT_PER_EVENT = LANES
COMPLETE_RELAY_CELL_COUNT = ROUTES
COMPLETE_RACE_CELL_COUNT = 3
ROUTE_RACE_CELL_STEPS_PER_EVENT = (
    TOUCHED_ROUTE_COUNT_PER_EVENT
    * (ROUTE_RACE_CELL_COUNT + ROUTE_RELAY_CELL_COUNT)
    * output_projection.RACE_STEPS
)
COMPLETE_RACE_CELL_STEPS_PER_EVENT = (
    (COMPLETE_RELAY_CELL_COUNT + COMPLETE_RACE_CELL_COUNT)
    * output_projection.RACE_STEPS
)
PHYSICAL_CELL_STEPS_PER_ACCEPTED_EVENT = (
    ROUTE_RACE_CELL_STEPS_PER_EVENT + COMPLETE_RACE_CELL_STEPS_PER_EVENT
)

GENERATOR_EXECUTION_COUNT = 0
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0
PARENT_LESSON_REPLAY_COUNT = 0
PRIOR_MEMORY_MUTATION_COUNT = 0
PRIOR_SCHEMA_MUTATION_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_V2_SOURCE_PATH = _ROOT / "experiments/r3_pilot/common_one_byte_copy_relation_v2.py"
_CONFIG_NAMESPACE = b"common-one-byte-relation-eventwise-v1-config\0"
_STATE_NAMESPACE = b"common-one-byte-relation-eventwise-v1-state\0"
_TRANSITION_NAMESPACE = b"common-one-byte-relation-eventwise-v1-transition\0"
_CONSTRUCTION_TOKEN = object()
_CELLS = projection.CommonOneByteFillerCells()


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


def _authenticate_dependencies() -> None:
    if (
        sha256(_V2_SOURCE_PATH.read_bytes()).hexdigest()
        != EXPECTED_V2_SOURCE_SHA256
        or relation_v2.SOURCE_SHA256 != EXPECTED_V2_SOURCE_SHA256
        or relation_v2.CONFIG_SHA256 != EXPECTED_V2_CONFIG_SHA256
        or relation_v2.NEW_CELL_COUNT != 73
        or relation_v2.ROUTE_SYNAPSE_CHANNEL_COUNT != 512
        or relation_v2.FUSED_ROUTE_SIMULATED_CELL_COUNT != 0
        or relation_v2.TOTAL_CELL_COUNT != 65_642
    ):
        raise RuntimeError("banked common one-byte relation V2 identity changed")


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "v2_source_commit": V2_SOURCE_COMMIT,
    "v2_source_sha256": EXPECTED_V2_SOURCE_SHA256,
    "v2_config_sha256": EXPECTED_V2_CONFIG_SHA256,
    "public_interface": (
        "start-authenticated-parent",
        "advance-exactly-one-physical-lesson",
    ),
    "active_context_capacity": 1,
    "evidence_geometry": (ROUTES, RADIX),
    "minimum_route_observations": MIN_ROUTE_OBSERVATIONS,
    "route_race_cells": (
        "four-evidence",
        "pair-coincidence-conflict",
        "tonic",
        "ready-relay",
    ),
    "complete_race_cells": (
        "sixteen-ready-relays",
        "complete-integrator",
        "tonic-fifteen",
        "poison",
    ),
    "host_groupby_finalize_or_completeness_flag": False,
    "parent_lesson_replay_count": PARENT_LESSON_REPLAY_COUNT,
    "raw_byte_digit_name_hash_or_whole_byte_identity_retained": False,
    "independence_remaining_scaffold": (
        "host-equality-of-authenticated-physical-context-and-source-assemblies"
    ),
    "new_cell_count": NEW_CELL_COUNT,
    "total_cell_count": TOTAL_CELL_COUNT,
    "persistent_state_bytes": PERSISTENT_STATE_BYTES,
    "physical_cell_steps_per_accepted_event": (
        PHYSICAL_CELL_STEPS_PER_ACCEPTED_EVENT
    ),
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


class FrozenEventwiseOneByteRelationV1:
    """Immutable parent checkpoint plus one persistent physical context."""

    __slots__ = (
        "_accumulator_update_ticks",
        "_active",
        "_checkpoint_sha256",
        "_consolidation_count",
        "_event_count",
        "_evidence",
        "_parent",
        "_pending_context_cells",
        "_pending_event_count",
        "_pending_relation_cells",
        "_pending_source_cells",
        "_physical_cell_steps",
        "_poison_latched",
        "_ready_winners",
    )

    def __init__(
        self,
        *,
        parent: relation_v2.FrozenCommonOneByteCopyRelationV2,
        evidence: np.ndarray,
        ready_winners: np.ndarray,
        pending_relation_cells: tuple[int, ...] | None,
        pending_context_cells: tuple[int, ...] | None,
        pending_source_cells: tuple[int, ...] | None,
        active: bool,
        poison_latched: bool,
        event_count: int,
        pending_event_count: int,
        accumulator_update_ticks: int,
        physical_cell_steps: int,
        consolidation_count: int,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("eventwise relation states are builder-owned")
        if type(parent) is not relation_v2.FrozenCommonOneByteCopyRelationV2:
            raise TypeError("parent must be the exact frozen V2 checkpoint type")
        parent.authenticate()
        if (
            evidence.shape != (ROUTES, RADIX)
            or evidence.dtype != TICK_DTYPE
            or not evidence.flags.c_contiguous
            or np.any(evidence > TICK_CAP)
        ):
            raise ValueError("eventwise evidence geometry changed")
        if (
            ready_winners.shape != (ROUTES,)
            or ready_winners.dtype != READY_DTYPE
            or not ready_winners.flags.c_contiguous
            or np.any(ready_winners < -1)
            or np.any(ready_winners >= RADIX)
        ):
            raise ValueError("eventwise ready-winner geometry changed")
        if type(active) is not bool or type(poison_latched) is not bool:
            raise TypeError("active and poison latches must be exact booleans")
        pending = (
            pending_relation_cells,
            pending_context_cells,
            pending_source_cells,
        )
        if active:
            if any(value is None for value in pending):
                raise ValueError("active context lost a physical assembly")
            relation_v2._validate_opaque_assembly(
                "pending_relation_cells", pending_relation_cells
            )
            relation_v2._validate_opaque_assembly(
                "pending_context_cells", pending_context_cells
            )
            relation_v2._validate_opaque_assembly(
                "pending_source_cells", pending_source_cells
            )
            if pending_context_cells == pending_source_cells:
                raise ValueError("pending context and source must stay distinct")
            if pending_event_count <= 0:
                raise ValueError("active context has no accepted event")
        elif any(value is not None for value in pending) or pending_event_count != 0:
            raise ValueError("inactive state retained pending context authority")
        counters = (
            event_count,
            pending_event_count,
            accumulator_update_ticks,
            physical_cell_steps,
            consolidation_count,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("eventwise counters must be non-negative integers")
        self._parent = parent
        self._pending_relation_cells = pending_relation_cells
        self._pending_context_cells = pending_context_cells
        self._pending_source_cells = pending_source_cells
        self._active = active
        self._poison_latched = poison_latched
        self._event_count = event_count
        self._pending_event_count = pending_event_count
        self._accumulator_update_ticks = accumulator_update_ticks
        self._physical_cell_steps = physical_cell_steps
        self._consolidation_count = consolidation_count
        self._evidence = np.array(evidence, copy=True, order="C")
        self._ready_winners = np.array(ready_winners, copy=True, order="C")
        self._evidence.setflags(write=False)
        self._ready_winners.setflags(write=False)
        self._checkpoint_sha256 = self._calculate_checkpoint_sha256()
        self.authenticate()

    def _metadata_record(self) -> dict[str, object]:
        return {
            "scope": SCOPE,
            "status": STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "config_sha256": CONFIG_SHA256,
            "parent_checkpoint_sha256": self._parent.checkpoint_sha256,
            "pending_relation_cells": self._pending_relation_cells,
            "pending_context_cells": self._pending_context_cells,
            "pending_source_cells": self._pending_source_cells,
            "active": self._active,
            "poison_latched": self._poison_latched,
            "event_count": self._event_count,
            "pending_event_count": self._pending_event_count,
            "accumulator_update_ticks": self._accumulator_update_ticks,
            "physical_cell_steps": self._physical_cell_steps,
            "consolidation_count": self._consolidation_count,
            "parent_lesson_replay_count": 0,
        }

    def _calculate_checkpoint_sha256(self) -> str:
        payload = (
            _STATE_NAMESPACE
            + _canonical_json(self._metadata_record())
            + _array_frame("evidence", self._evidence)
            + _array_frame("ready-winners", self._ready_winners)
        )
        return sha256(payload).hexdigest()

    @property
    def parent(self) -> relation_v2.FrozenCommonOneByteCopyRelationV2:
        self.authenticate()
        return self._parent

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def evidence_counts(self) -> tuple[tuple[int, ...], ...]:
        self.authenticate()
        return tuple(tuple(int(value) for value in row) for row in self._evidence)

    @property
    def route_ready_winners(self) -> tuple[int, ...]:
        self.authenticate()
        return tuple(int(value) for value in self._ready_winners)

    @property
    def active(self) -> bool:
        self.authenticate()
        return self._active

    @property
    def poison_latched(self) -> bool:
        self.authenticate()
        return self._poison_latched

    @property
    def event_count(self) -> int:
        self.authenticate()
        return self._event_count

    @property
    def pending_event_count(self) -> int:
        self.authenticate()
        return self._pending_event_count

    @property
    def accumulator_update_ticks(self) -> int:
        self.authenticate()
        return self._accumulator_update_ticks

    @property
    def physical_cell_steps(self) -> int:
        self.authenticate()
        return self._physical_cell_steps

    @property
    def consolidation_count(self) -> int:
        self.authenticate()
        return self._consolidation_count

    def authenticate(self) -> None:
        _authenticate_dependencies()
        self._parent.authenticate()
        if self._calculate_checkpoint_sha256() != self._checkpoint_sha256:
            raise RuntimeError("eventwise relation state authentication failed")


@dataclass(frozen=True, slots=True)
class EventwiseRelationTransition:
    """Rooted result of consuming one physical lesson."""

    status: str
    successor: FrozenEventwiseOneByteRelationV1
    accepted_event: bool
    consolidated: bool
    complete_route_count: int
    poison_latched: bool
    event_observation_ticks: int
    event_accumulator_update_ticks: int
    event_physical_cell_steps: int
    consolidation_relation_excitatory_ticks: int
    consolidation_route_excitatory_ticks: int
    consolidation_route_inhibitory_ticks: int
    consolidation_provenance_writes: int
    parent_checkpoint_before_sha256: str
    parent_checkpoint_after_sha256: str
    sha256: str

    @property
    def consolidation_work_ticks(self) -> int:
        return (
            self.consolidation_relation_excitatory_ticks
            + self.consolidation_route_excitatory_ticks
            + self.consolidation_route_inhibitory_ticks
            + self.consolidation_provenance_writes
        )


@dataclass(frozen=True, slots=True)
class _ConsolidationResult:
    parent: relation_v2.FrozenCommonOneByteCopyRelationV2
    relation_excitatory_ticks: int
    route_excitatory_ticks: int
    route_inhibitory_ticks: int
    provenance_writes: int


def _new_state(
    *,
    parent: relation_v2.FrozenCommonOneByteCopyRelationV2,
    evidence: np.ndarray,
    ready_winners: np.ndarray,
    pending_relation_cells: tuple[int, ...] | None,
    pending_context_cells: tuple[int, ...] | None,
    pending_source_cells: tuple[int, ...] | None,
    active: bool,
    poison_latched: bool,
    event_count: int,
    pending_event_count: int,
    accumulator_update_ticks: int,
    physical_cell_steps: int,
    consolidation_count: int,
) -> FrozenEventwiseOneByteRelationV1:
    return FrozenEventwiseOneByteRelationV1(
        parent=parent,
        evidence=evidence,
        ready_winners=ready_winners,
        pending_relation_cells=pending_relation_cells,
        pending_context_cells=pending_context_cells,
        pending_source_cells=pending_source_cells,
        active=active,
        poison_latched=poison_latched,
        event_count=event_count,
        pending_event_count=pending_event_count,
        accumulator_update_ticks=accumulator_update_ticks,
        physical_cell_steps=physical_cell_steps,
        consolidation_count=consolidation_count,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _transition(
    *,
    status: str,
    successor: FrozenEventwiseOneByteRelationV1,
    accepted_event: bool,
    consolidated: bool,
    complete_route_count: int,
    poison_latched: bool,
    event_observation_ticks: int,
    event_accumulator_update_ticks: int,
    event_physical_cell_steps: int,
    consolidation_relation_excitatory_ticks: int = 0,
    consolidation_route_excitatory_ticks: int = 0,
    consolidation_route_inhibitory_ticks: int = 0,
    consolidation_provenance_writes: int = 0,
    parent_checkpoint_before_sha256: str,
) -> EventwiseRelationTransition:
    record = {
        "status": status,
        "successor_checkpoint_sha256": successor.checkpoint_sha256,
        "accepted_event": accepted_event,
        "consolidated": consolidated,
        "complete_route_count": complete_route_count,
        "poison_latched": poison_latched,
        "event_observation_ticks": event_observation_ticks,
        "event_accumulator_update_ticks": event_accumulator_update_ticks,
        "event_physical_cell_steps": event_physical_cell_steps,
        "consolidation_relation_excitatory_ticks": (
            consolidation_relation_excitatory_ticks
        ),
        "consolidation_route_excitatory_ticks": (
            consolidation_route_excitatory_ticks
        ),
        "consolidation_route_inhibitory_ticks": (
            consolidation_route_inhibitory_ticks
        ),
        "consolidation_provenance_writes": consolidation_provenance_writes,
        "parent_checkpoint_before_sha256": parent_checkpoint_before_sha256,
        "parent_checkpoint_after_sha256": successor.parent.checkpoint_sha256,
    }
    return EventwiseRelationTransition(
        status=status,
        successor=successor,
        accepted_event=accepted_event,
        consolidated=consolidated,
        complete_route_count=complete_route_count,
        poison_latched=poison_latched,
        event_observation_ticks=event_observation_ticks,
        event_accumulator_update_ticks=event_accumulator_update_ticks,
        event_physical_cell_steps=event_physical_cell_steps,
        consolidation_relation_excitatory_ticks=(
            consolidation_relation_excitatory_ticks
        ),
        consolidation_route_excitatory_ticks=(
            consolidation_route_excitatory_ticks
        ),
        consolidation_route_inhibitory_ticks=(
            consolidation_route_inhibitory_ticks
        ),
        consolidation_provenance_writes=consolidation_provenance_writes,
        parent_checkpoint_before_sha256=parent_checkpoint_before_sha256,
        parent_checkpoint_after_sha256=successor.parent.checkpoint_sha256,
        sha256=_digest(_TRANSITION_NAMESPACE, record),
    )


def _context_independence_stop(
    parent: relation_v2.FrozenCommonOneByteCopyRelationV2,
    relation_cells: tuple[int, ...],
    context_cells: tuple[int, ...],
    source_cells: tuple[int, ...],
) -> str | None:
    slot = parent._matching_relation_slot(relation_cells)
    if slot is None:
        return None
    context = np.asarray(context_cells, dtype=np.int32)
    source = np.asarray(source_cells, dtype=np.int32)
    dependent = False
    for row in range(parent._context_count):
        if int(parent._context_relation_slots[row]) != slot:
            continue
        same_context = bool(np.all(parent._context_subjects[row] == context))
        same_source = bool(np.all(parent._context_sources[row] == source))
        if same_context and same_source:
            return "STOP_DUPLICATE_CONTEXT"
        dependent = dependent or same_context or same_source
    return "STOP_DEPENDENT_CONTEXT" if dependent else None


def _race_route(counts: np.ndarray) -> tuple[int, bool, int]:
    active_candidate_count = int(np.count_nonzero(counts))
    pair_coincidences = (
        active_candidate_count * (active_candidate_count - 1) // 2
    )
    conflict_score = min(pair_coincidences * MAX_SCORE, MAX_SCORE)
    raw_scores = (
        *(int(value) * ROUTE_SCORE_GAIN for value in counts),
        conflict_score,
        TONIC_SCORE,
    )
    currents = np.asarray(
        [[output_projection._CURRENT_TABLE[score] for score in raw_scores]],
        dtype=np.float64,
    )
    race = output_projection._OUTPUT_RACE.race_many(
        currents,
        require_unique=False,
    )[0]
    conflict = active_candidate_count > 1
    winner = -1
    if (
        not conflict
        and race.simultaneous_spike_candidates == 1
        and 0 <= race.winner_cell < RADIX
    ):
        winner = int(race.winner_cell)

    relay_score = 2 * TONIC_SCORE if winner >= 0 else 0
    relay_currents = np.asarray(
        [[output_projection._CURRENT_TABLE[relay_score]]],
        dtype=np.float64,
    )
    output_projection._OUTPUT_RACE.race_many(
        relay_currents,
        require_unique=False,
    )
    physical_steps = (
        ROUTE_RACE_CELL_COUNT + ROUTE_RELAY_CELL_COUNT
    ) * output_projection.RACE_STEPS
    return winner, conflict, physical_steps


def _race_complete(
    ready_winners: np.ndarray,
    *,
    poison_latched: bool,
    complete_cell_gate: bool,
) -> tuple[str, int, int]:
    ready_count = int(np.count_nonzero(ready_winners >= 0))
    relay_currents = np.asarray(
        [
            [output_projection._CURRENT_TABLE[2 * TONIC_SCORE if winner >= 0 else 0]]
            for winner in ready_winners
        ],
        dtype=np.float64,
    )
    output_projection._OUTPUT_RACE.race_many(
        relay_currents,
        require_unique=False,
    )

    complete_score = ready_count if complete_cell_gate else 0
    poison_score = MAX_SCORE if poison_latched else 0
    global_currents = np.asarray(
        [[
            output_projection._CURRENT_TABLE[complete_score],
            output_projection._CURRENT_TABLE[ROUTES - 1],
            output_projection._CURRENT_TABLE[poison_score],
        ]],
        dtype=np.float64,
    )
    race = output_projection._OUTPUT_RACE.race_many(
        global_currents,
        require_unique=False,
    )[0]
    if poison_latched:
        decision = "POISON"
    elif (
        complete_cell_gate
        and race.simultaneous_spike_candidates == 1
        and race.winner_cell == 0
    ):
        decision = "COMPLETE"
    else:
        decision = "PENDING"
    physical_steps = (
        COMPLETE_RELAY_CELL_COUNT + COMPLETE_RACE_CELL_COUNT
    ) * output_projection.RACE_STEPS
    return decision, ready_count, physical_steps


def _consolidate_parent(
    parent: relation_v2.FrozenCommonOneByteCopyRelationV2,
    *,
    relation_cells_value: tuple[int, ...],
    context_cells_value: tuple[int, ...],
    source_cells_value: tuple[int, ...],
    evidence: np.ndarray,
    ready_winners: np.ndarray,
    pending_event_count: int,
) -> _ConsolidationResult:
    parent.authenticate()
    if np.any(ready_winners < 0):
        raise RuntimeError("incomplete eventwise routes reached consolidation")
    if _context_independence_stop(
        parent,
        relation_cells_value,
        context_cells_value,
        source_cells_value,
    ) is not None:
        raise RuntimeError("dependent physical context reached consolidation")

    relation_cells = np.array(parent._relation_cells, copy=True, order="C")
    relation_excitatory = np.array(
        parent._relation_excitatory, copy=True, order="C"
    )
    relation_support_counts = np.array(
        parent._relation_support_counts, copy=True, order="C"
    )
    route_excitatory = np.array(
        parent._route_excitatory, copy=True, order="C"
    )
    route_inhibitory = np.array(
        parent._route_inhibitory, copy=True, order="C"
    )
    context_relation_slots = np.array(
        parent._context_relation_slots, copy=True, order="C"
    )
    context_subjects = np.array(
        parent._context_subjects, copy=True, order="C"
    )
    context_sources = np.array(parent._context_sources, copy=True, order="C")
    context_route_targets = np.array(
        parent._context_route_targets, copy=True, order="C"
    )
    relation_count = parent._relation_count
    context_count = parent._context_count
    parent_summary = parent.training_summary
    parent_checkpoint_sha256 = parent.checkpoint_sha256

    slot = parent._matching_relation_slot(relation_cells_value)
    if slot is None:
        if relation_count == relation_v2.MAX_RELATIONS:
            raise RuntimeError("relation schema capacity exhausted")
        slot = relation_count
        relation_count += 1
        relation_cells[slot] = np.asarray(relation_cells_value, dtype=np.int32)
    if context_count == relation_v2.MAX_INDEPENDENT_CONTEXTS:
        raise RuntimeError("physical context provenance capacity exhausted")

    provenance_cell_id = relation_v2.PROVENANCE_CELL_OFFSET + context_count
    context_relation_slots[context_count] = slot
    context_subjects[context_count] = np.asarray(
        context_cells_value, dtype=np.int32
    )
    context_sources[context_count] = np.asarray(source_cells_value, dtype=np.int32)
    context_route_targets[context_count] = ready_winners

    previous_relation = relation_excitatory[slot].astype(np.uint16)
    updated_relation = np.minimum(
        previous_relation + relation_v2.POSITIVE_TICK,
        TICK_CAP,
    )
    relation_excitatory[slot] = updated_relation.astype(TICK_DTYPE)
    relation_e_ticks = int(
        np.sum(updated_relation - previous_relation, dtype=np.uint64)
    )
    relation_support_counts[slot] = min(
        int(relation_support_counts[slot]) + 1,
        TICK_CAP,
    )

    route_e_ticks = 0
    route_h_ticks = 0
    route_output_cells: list[int] = []
    for flat_input, target_value in enumerate(ready_winners):
        target = int(target_value)
        lane, local_input = divmod(flat_input, RADIX)
        prior_targets = tuple(
            int(candidate)
            for candidate in np.flatnonzero(
                route_excitatory[slot, lane, local_input] > 0
            )
            if int(candidate) != target
        )
        previous_e = int(route_excitatory[slot, lane, local_input, target])
        updated_e = min(previous_e + relation_v2.POSITIVE_TICK, TICK_CAP)
        route_excitatory[slot, lane, local_input, target] = updated_e
        route_e_ticks += updated_e - previous_e
        for rival in prior_targets:
            previous_h = int(route_inhibitory[slot, lane, local_input, rival])
            updated_h = min(previous_h + relation_v2.NEGATIVE_TICK, TICK_CAP)
            route_inhibitory[slot, lane, local_input, rival] = updated_h
            route_h_ticks += updated_h - previous_h
        route_output_cells.append(
            relation_v2._OUTPUT_CANDIDATES_BY_LANE[lane][target]
        )

    minimum_observations = min(
        int(evidence[route, int(target)])
        for route, target in enumerate(ready_winners)
    )
    local_ticks = int(np.sum(evidence, dtype=np.uint64))
    context_count += 1
    provenance = relation_v2.PhysicalContextProvenance(
        provenance_cell_id=provenance_cell_id,
        relation_schema_cell_id=relation_v2.RELATION_SCHEMA_CELL_OFFSET + slot,
        context_cells=context_cells_value,
        source_cells=source_cells_value,
        route_output_cells=tuple(route_output_cells),
        minimum_route_observations=minimum_observations,
    )
    context_step = relation_v2.ContextConsolidationStep(
        first_lesson_ordinal=parent_summary.lesson_count,
        status="EVENTWISE_NEURONAL_CONTEXT_CONSOLIDATED",
        relation_schema_cell_id=relation_v2.RELATION_SCHEMA_CELL_OFFSET + slot,
        provenance_cell_id=provenance_cell_id,
        complete_route_count=ROUTES,
        minimum_route_observations=minimum_observations,
        relation_support_count=int(relation_support_counts[slot]),
        excitatory_update_ticks=relation_e_ticks + route_e_ticks,
        inhibitory_update_ticks=route_h_ticks,
    )

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
    metadata = _canonical_json(
        {
            "scope": relation_v2.SCOPE,
            "status": STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "config_sha256": relation_v2.CONFIG_SHA256,
            "eventwise_config_sha256": CONFIG_SHA256,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_lesson_replay_count": 0,
            "new_event_count": pending_event_count,
            "accepted_new_context_count": 1,
            "relation_count": relation_count,
            "independent_context_count": context_count,
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
    )
    checkpoint_sha256 = sha256(
        relation_v2._checkpoint_payload(metadata, arrays)
    ).hexdigest()
    cumulative_summary = relation_v2.RelationGatedTrainingSummary(
        lesson_count=parent_summary.lesson_count + pending_event_count,
        teacher_on_lesson_count=(
            parent_summary.teacher_on_lesson_count + pending_event_count
        ),
        teacher_off_lesson_count=parent_summary.teacher_off_lesson_count,
        cell_on_lesson_count=(
            parent_summary.cell_on_lesson_count + pending_event_count
        ),
        cell_off_lesson_count=parent_summary.cell_off_lesson_count,
        context_group_count=parent_summary.context_group_count + 1,
        independent_context_count=context_count,
        dependent_context_count=parent_summary.dependent_context_count,
        incomplete_context_count=parent_summary.incomplete_context_count,
        contradictory_context_count=(
            parent_summary.contradictory_context_count + int(route_h_ticks > 0)
        ),
        relation_count=relation_count,
        local_physical_update_ticks=(
            parent_summary.local_physical_update_ticks + local_ticks
        ),
        relation_excitatory_update_ticks=(
            parent_summary.relation_excitatory_update_ticks + relation_e_ticks
        ),
        route_excitatory_update_ticks=(
            parent_summary.route_excitatory_update_ticks + route_e_ticks
        ),
        route_inhibitory_update_ticks=(
            parent_summary.route_inhibitory_update_ticks + route_h_ticks
        ),
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
        context_steps=(*parent_summary.context_steps, context_step),
        provenance=(*parent_summary.provenance, provenance),
        checkpoint_sha256=checkpoint_sha256,
    )
    extension_summary = relation_v2.RelationGatedExtensionSummary(
        status=STATUS,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha256,
        new_lesson_count=pending_event_count,
        teacher_on_lesson_count=pending_event_count,
        teacher_off_lesson_count=0,
        cell_on_lesson_count=pending_event_count,
        cell_off_lesson_count=0,
        active_context_group_count=1,
        accepted_new_context_count=1,
        duplicate_context_count=0,
        dependent_context_count=0,
        incomplete_context_count=0,
        contradictory_context_count=int(route_h_ticks > 0),
        new_relation_count=relation_count - parent._relation_count,
        parent_provenance_count=parent._context_count,
        retained_parent_provenance_count=len(parent_summary.provenance),
        total_provenance_count=context_count,
        physical_observation_ticks=pending_event_count * LANES,
        committed_local_update_ticks=local_ticks,
        relation_excitatory_update_ticks=relation_e_ticks,
        route_excitatory_update_ticks=route_e_ticks,
        route_inhibitory_update_ticks=route_h_ticks,
        parent_checkpoint_array_bytes=parent_summary.checkpoint_array_bytes,
        checkpoint_array_bytes=checkpoint_array_bytes,
        context_steps=(context_step,),
        provenance_added=(provenance,),
    )
    successor = relation_v2.FrozenCommonOneByteCopyRelationV2(
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
        summary=cumulative_summary,
        extension_summary=extension_summary,
        construction_token=relation_v2._CONSTRUCTION_TOKEN,
    )
    parent.authenticate()
    if parent.checkpoint_sha256 != parent_checkpoint_sha256:
        raise RuntimeError("parent changed during eventwise consolidation")
    return _ConsolidationResult(
        parent=successor,
        relation_excitatory_ticks=relation_e_ticks,
        route_excitatory_ticks=route_e_ticks,
        route_inhibitory_ticks=route_h_ticks,
        provenance_writes=1,
    )


def start_eventwise_common_one_byte_relation_v1(
    parent: relation_v2.FrozenCommonOneByteCopyRelationV2,
) -> FrozenEventwiseOneByteRelationV1:
    """Start empty eventwise accumulation from one authenticated frozen parent."""

    _authenticate_dependencies()
    if type(parent) is not relation_v2.FrozenCommonOneByteCopyRelationV2:
        raise TypeError("parent must be the exact frozen V2 checkpoint type")
    parent.authenticate()
    return _new_state(
        parent=parent,
        evidence=np.zeros((ROUTES, RADIX), dtype=TICK_DTYPE),
        ready_winners=np.full((ROUTES,), -1, dtype=READY_DTYPE),
        pending_relation_cells=None,
        pending_context_cells=None,
        pending_source_cells=None,
        active=False,
        poison_latched=False,
        event_count=0,
        pending_event_count=0,
        accumulator_update_ticks=0,
        physical_cell_steps=0,
        consolidation_count=0,
    )


def advance_eventwise_common_one_byte_relation_v1(
    state: FrozenEventwiseOneByteRelationV1,
    lesson: relation_v2.RelationGatedOneByteLesson,
    *,
    accumulator_gate: bool = True,
    complete_cell_gate: bool = True,
) -> EventwiseRelationTransition:
    """Consume exactly one physical event and return its rooted successor."""

    if type(state) is not FrozenEventwiseOneByteRelationV1:
        raise TypeError("state must be the exact frozen eventwise state type")
    if type(lesson) is not relation_v2.RelationGatedOneByteLesson:
        raise TypeError("advance accepts exactly one physical V2 lesson")
    if type(accumulator_gate) is not bool or type(complete_cell_gate) is not bool:
        raise TypeError("eventwise cell gates must be exact booleans")
    state.authenticate()
    parent_before = state.parent.checkpoint_sha256
    try:
        input_routes = relation_v2._input_route_indices(_CELLS, lesson.input_frame)
        teacher_candidates = relation_v2._teacher_candidate_indices(
            lesson.teacher_output_cells
        )
    except (TypeError, ValueError):
        return _transition(
            status="STOP_INPUT_OR_TEACHER_AUTHENTICATION",
            successor=state,
            accepted_event=False,
            consolidated=False,
            complete_route_count=sum(
                winner >= 0 for winner in state.route_ready_winners
            ),
            poison_latched=state.poison_latched,
            event_observation_ticks=0,
            event_accumulator_update_ticks=0,
            event_physical_cell_steps=0,
            parent_checkpoint_before_sha256=parent_before,
        )
    if not lesson.teacher_gate:
        return _transition(
            status="STOP_TEACHER_OFF",
            successor=state,
            accepted_event=False,
            consolidated=False,
            complete_route_count=sum(
                winner >= 0 for winner in state.route_ready_winners
            ),
            poison_latched=state.poison_latched,
            event_observation_ticks=0,
            event_accumulator_update_ticks=0,
            event_physical_cell_steps=0,
            parent_checkpoint_before_sha256=parent_before,
        )
    if not lesson.cell_gate:
        return _transition(
            status="STOP_CELL_OFF",
            successor=state,
            accepted_event=False,
            consolidated=False,
            complete_route_count=sum(
                winner >= 0 for winner in state.route_ready_winners
            ),
            poison_latched=state.poison_latched,
            event_observation_ticks=0,
            event_accumulator_update_ticks=0,
            event_physical_cell_steps=0,
            parent_checkpoint_before_sha256=parent_before,
        )
    if not accumulator_gate:
        return _transition(
            status="STOP_ACCUMULATOR_OFF",
            successor=state,
            accepted_event=False,
            consolidated=False,
            complete_route_count=sum(
                winner >= 0 for winner in state.route_ready_winners
            ),
            poison_latched=state.poison_latched,
            event_observation_ticks=0,
            event_accumulator_update_ticks=0,
            event_physical_cell_steps=0,
            parent_checkpoint_before_sha256=parent_before,
        )

    if state._active:
        if (
            lesson.relation_cells != state._pending_relation_cells
            or lesson.context_cells != state._pending_context_cells
            or lesson.source_cells != state._pending_source_cells
        ):
            return _transition(
                status="STOP_INCOMPLETE_CONTEXT_SWITCH",
                successor=state,
                accepted_event=False,
                consolidated=False,
                complete_route_count=sum(
                    winner >= 0 for winner in state.route_ready_winners
                ),
                poison_latched=state.poison_latched,
                event_observation_ticks=0,
                event_accumulator_update_ticks=0,
                event_physical_cell_steps=0,
                parent_checkpoint_before_sha256=parent_before,
            )
        pending_relation = state._pending_relation_cells
        pending_context = state._pending_context_cells
        pending_source = state._pending_source_cells
    else:
        independence_stop = _context_independence_stop(
            state.parent,
            lesson.relation_cells,
            lesson.context_cells,
            lesson.source_cells,
        )
        if independence_stop is not None:
            return _transition(
                status=independence_stop,
                successor=state,
                accepted_event=False,
                consolidated=False,
                complete_route_count=0,
                poison_latched=False,
                event_observation_ticks=0,
                event_accumulator_update_ticks=0,
                event_physical_cell_steps=0,
                parent_checkpoint_before_sha256=parent_before,
            )
        pending_relation = lesson.relation_cells
        pending_context = lesson.context_cells
        pending_source = lesson.source_cells

    evidence = np.array(state._evidence, copy=True, order="C")
    ready_winners = np.array(state._ready_winners, copy=True, order="C")
    accumulator_ticks = 0
    route_physical_steps = 0
    poison_latched = state._poison_latched
    for lane, flat_input in enumerate(input_routes):
        candidate = teacher_candidates[lane]
        previous = int(evidence[flat_input, candidate])
        updated = min(previous + relation_v2.POSITIVE_TICK, TICK_CAP)
        evidence[flat_input, candidate] = updated
        accumulator_ticks += updated - previous
        winner, conflict, physical_steps = _race_route(evidence[flat_input])
        ready_winners[flat_input] = winner
        poison_latched = poison_latched or conflict
        route_physical_steps += physical_steps

    complete_decision, ready_count, complete_physical_steps = _race_complete(
        ready_winners,
        poison_latched=poison_latched,
        complete_cell_gate=complete_cell_gate,
    )
    event_physical_steps = route_physical_steps + complete_physical_steps
    if event_physical_steps != PHYSICAL_CELL_STEPS_PER_ACCEPTED_EVENT:
        raise RuntimeError("eventwise physical work accounting changed")
    pending_event_count = state._pending_event_count + 1
    provisional = _new_state(
        parent=state.parent,
        evidence=evidence,
        ready_winners=ready_winners,
        pending_relation_cells=pending_relation,
        pending_context_cells=pending_context,
        pending_source_cells=pending_source,
        active=True,
        poison_latched=poison_latched,
        event_count=state._event_count + 1,
        pending_event_count=pending_event_count,
        accumulator_update_ticks=(
            state._accumulator_update_ticks + accumulator_ticks
        ),
        physical_cell_steps=state._physical_cell_steps + event_physical_steps,
        consolidation_count=state._consolidation_count,
    )

    if complete_decision != "COMPLETE":
        if complete_decision == "POISON":
            transition_status = "STOP_CONFLICT_POISON"
        elif not complete_cell_gate and ready_count == ROUTES:
            transition_status = "STOP_COMPLETE_CELL_OFF"
        else:
            transition_status = "PENDING_EVIDENCE"
        return _transition(
            status=transition_status,
            successor=provisional,
            accepted_event=True,
            consolidated=False,
            complete_route_count=ready_count,
            poison_latched=poison_latched,
            event_observation_ticks=LANES,
            event_accumulator_update_ticks=accumulator_ticks,
            event_physical_cell_steps=event_physical_steps,
            parent_checkpoint_before_sha256=parent_before,
        )

    if pending_relation is None or pending_context is None or pending_source is None:
        raise RuntimeError("complete eventwise context lost physical identity")
    consolidated = _consolidate_parent(
        state.parent,
        relation_cells_value=pending_relation,
        context_cells_value=pending_context,
        source_cells_value=pending_source,
        evidence=evidence,
        ready_winners=ready_winners,
        pending_event_count=pending_event_count,
    )
    successor = _new_state(
        parent=consolidated.parent,
        evidence=np.zeros((ROUTES, RADIX), dtype=TICK_DTYPE),
        ready_winners=np.full((ROUTES,), -1, dtype=READY_DTYPE),
        pending_relation_cells=None,
        pending_context_cells=None,
        pending_source_cells=None,
        active=False,
        poison_latched=False,
        event_count=state._event_count + 1,
        pending_event_count=0,
        accumulator_update_ticks=(
            state._accumulator_update_ticks + accumulator_ticks
        ),
        physical_cell_steps=state._physical_cell_steps + event_physical_steps,
        consolidation_count=state._consolidation_count + 1,
    )
    return _transition(
        status="CONSOLIDATED",
        successor=successor,
        accepted_event=True,
        consolidated=True,
        complete_route_count=ready_count,
        poison_latched=False,
        event_observation_ticks=LANES,
        event_accumulator_update_ticks=accumulator_ticks,
        event_physical_cell_steps=event_physical_steps,
        consolidation_relation_excitatory_ticks=(
            consolidated.relation_excitatory_ticks
        ),
        consolidation_route_excitatory_ticks=(
            consolidated.route_excitatory_ticks
        ),
        consolidation_route_inhibitory_ticks=(
            consolidated.route_inhibitory_ticks
        ),
        consolidation_provenance_writes=consolidated.provenance_writes,
        parent_checkpoint_before_sha256=parent_before,
    )


_authenticate_dependencies()
SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "CLAIM_BOUNDARY",
    "CODING_INPUT_COUNT",
    "COMPLETE_RACE_CELL_STEPS_PER_EVENT",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "CORPUS_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA",
    "EVIDENCE_CELL_COUNT",
    "EVIDENCE_STATE_BYTES",
    "G0_INPUT_COUNT",
    "LATCH_STATE_BYTES",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "PARENT_LESSON_REPLAY_COUNT",
    "PATH2_INPUT_COUNT",
    "PERSISTENT_STATE_BYTES",
    "PHYSICAL_CELL_STEPS_PER_ACCEPTED_EVENT",
    "PRIOR_MEMORY_MUTATION_COUNT",
    "PRIOR_SCHEMA_MUTATION_COUNT",
    "PROTECTED_INPUT_COUNT",
    "READY_RELAY_CELL_COUNT",
    "READY_WINNER_STATE_BYTES",
    "ROUTE_RACE_CELL_STEPS_PER_EVENT",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "TOTAL_CELL_COUNT",
    "EventwiseRelationTransition",
    "FrozenEventwiseOneByteRelationV1",
    "advance_eventwise_common_one_byte_relation_v1",
    "start_eventwise_common_one_byte_relation_v1",
]
