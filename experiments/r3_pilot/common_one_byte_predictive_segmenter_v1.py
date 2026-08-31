"""Predictive one-event segmentation without external context/source identity.

The banked eventwise learner accumulates and consolidates one physical context,
but its lesson still carries host-supplied context/source assemblies.  This
successor accepts one timed physical event with no such fields.  Two opaque
interval cells, learned cue E/H synapses, and physical consequence-surprise
comparators drive a CONTINUE/NEW/tonic race.  A NEW winner reserves the next
banked provenance neuron and privately supplies one internal segment assembly
to the reused eventwise learner.  No lesson tuple is retained or replayed.

The first stream segment is the sole calibration segment.  Later NEW decisions
require the preceding segment's neuronal completeness.  This bounded mechanism
does not claim that identical observable streams reveal unobservable latent
boundaries.
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
from experiments.r3_pilot import common_one_byte_relation_eventwise_v1 as eventwise

SCOPE = "COMMON_ONE_BYTE_PREDICTIVE_SEGMENTER_V1"
STATUS = "LEARNED_OPAQUE_INTERVAL_AND_CONSEQUENCE_SURPRISE_SEGMENTATION"
CLAIM_BOUNDARY = (
    "self-posed-complete-single-relation-one-byte-event-stream-segmentation-"
    "from-consequence-surprise-and-learned-opaque-interval-cues-only-no-"
    "external-context-source-boundary-bit-raw-time-or-broad-capability-evidence"
)

EVENTWISE_SOURCE_COMMIT = "65d36ba"
EXPECTED_EVENTWISE_SOURCE_SHA256 = (
    "c7285ed43b65d4403642ae9098ca863a1f50393e089e5a1da43ced4b3d7eafb9"
)
EXPECTED_EVENTWISE_CONFIG_SHA256 = (
    "5db56ed6be811e8f0867e7fadaff92ad082c1d0dfe3cf8436790f09d89d5be9c"
)

LANES = relation_v2.LANES
RADIX = relation_v2.RADIX
ROUTES = relation_v2.INPUT_FILLER_CELLS
INTERVAL_CELL_COUNT = 2
SEGMENT_DECISION_COUNT = 2
CONTINUE_DECISION = 0
NEW_DECISION = 1
PROVENANCE_CAPACITY = 4
TICK_DTYPE = np.dtype(np.uint8)
TARGET_DTYPE = np.dtype(np.int8)
PROVENANCE_DTYPE = np.dtype(np.int32)
TICK_CAP = relation_v2.TICK_CAP
TONIC_SCORE = relation_v2.RELATION_TONIC_SCORE
MAX_SCORE = relation_v2.MAX_PHYSICAL_SCORE
SURPRISE_SCORE_GAIN = relation_v2.ROUTE_SCORE_GAIN
MATCH_SCORE_GAIN = TONIC_SCORE // LANES

INTERVAL_CELL_OFFSET = eventwise.TOTAL_CELL_COUNT
INTERVAL_CELL_IDS = tuple(
    INTERVAL_CELL_OFFSET + index for index in range(INTERVAL_CELL_COUNT)
)
MATCH_COMPARATOR_CELL_OFFSET = INTERVAL_CELL_OFFSET + INTERVAL_CELL_COUNT
MATCH_COMPARATOR_CELL_COUNT = LANES
MISMATCH_COMPARATOR_CELL_OFFSET = (
    MATCH_COMPARATOR_CELL_OFFSET + MATCH_COMPARATOR_CELL_COUNT
)
MISMATCH_COMPARATOR_CELL_COUNT = LANES
CONTINUE_CELL_ID = MISMATCH_COMPARATOR_CELL_OFFSET + MISMATCH_COMPARATOR_CELL_COUNT
NEW_CELL_ID = CONTINUE_CELL_ID + 1
SEGMENT_TONIC_CELL_ID = NEW_CELL_ID + 1
COMPLETE_HOLD_LATCH_CELL_ID = SEGMENT_TONIC_CELL_ID + 1
STREAM_START_LATCH_CELL_ID = COMPLETE_HOLD_LATCH_CELL_ID + 1
NEW_CELL_COUNT = (
    INTERVAL_CELL_COUNT
    + MATCH_COMPARATOR_CELL_COUNT
    + MISMATCH_COMPARATOR_CELL_COUNT
    + SEGMENT_DECISION_COUNT
    + 3
)
TOTAL_CELL_COUNT = eventwise.TOTAL_CELL_COUNT + NEW_CELL_COUNT

COMPARATOR_SYNAPSE_CHANNEL_COUNT = LANES * RADIX * RADIX
CUE_SYNAPSE_CHANNEL_COUNT = INTERVAL_CELL_COUNT * SEGMENT_DECISION_COUNT
CUE_EH_STATE_BYTES = 2 * CUE_SYNAPSE_CHANNEL_COUNT * TICK_DTYPE.itemsize
WITHIN_CUE_STATE_BYTES = INTERVAL_CELL_COUNT * TICK_DTYPE.itemsize
PREVIOUS_TARGET_STATE_BYTES = ROUTES * TARGET_DTYPE.itemsize
INTERNAL_PROVENANCE_STATE_BYTES = (
    PROVENANCE_CAPACITY * PROVENANCE_DTYPE.itemsize
)
SEGMENT_LATCH_STATE_BYTES = 8
ADDED_PERSISTENT_STATE_BYTES = (
    CUE_EH_STATE_BYTES
    + WITHIN_CUE_STATE_BYTES
    + PREVIOUS_TARGET_STATE_BYTES
    + INTERNAL_PROVENANCE_STATE_BYTES
    + SEGMENT_LATCH_STATE_BYTES
)
PERSISTENT_STATE_BYTES = (
    eventwise.PERSISTENT_STATE_BYTES + ADDED_PERSISTENT_STATE_BYTES
)

SEGMENTATION_RACE_CELL_COUNT = NEW_CELL_COUNT
SEGMENTATION_CELL_STEPS_PER_EVENT = (
    SEGMENTATION_RACE_CELL_COUNT * output_projection.RACE_STEPS
)
ROUTED_EVENT_CELL_STEPS = (
    SEGMENTATION_CELL_STEPS_PER_EVENT
    + eventwise.PHYSICAL_CELL_STEPS_PER_ACCEPTED_EVENT
)

EXTERNAL_CONTEXT_SOURCE_INPUT_COUNT = 0
BOUNDARY_BIT_INPUT_COUNT = 0
RAW_ELAPSED_TIME_INPUT_COUNT = 0
PARENT_LESSON_REPLAY_COUNT = 0
GENERATOR_EXECUTION_COUNT = 0
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_EVENTWISE_SOURCE_PATH = (
    _ROOT / "experiments/r3_pilot/common_one_byte_relation_eventwise_v1.py"
)
_CONFIG_NAMESPACE = b"common-one-byte-predictive-segmenter-v1-config\0"
_STATE_NAMESPACE = b"common-one-byte-predictive-segmenter-v1-state\0"
_TRANSITION_NAMESPACE = b"common-one-byte-predictive-segmenter-v1-transition\0"
_CONSTRUCTION_TOKEN = object()
_INPUT_CELLS = projection.CommonOneByteFillerCells()
_OUTPUT_CELLS = output_projection.CommonEventCells()


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


def _internal_assembly(role: bytes, slot: int) -> tuple[int, ...]:
    return _OUTPUT_CELLS.project(
        b"predictive-segmenter-v1/internal/" + role + bytes((slot,))
    ).trace_cells


_INTERNAL_CONTEXT_ASSEMBLIES = tuple(
    _internal_assembly(b"context/", slot) for slot in range(PROVENANCE_CAPACITY)
)
_INTERNAL_SOURCE_ASSEMBLIES = tuple(
    _internal_assembly(b"source/", slot) for slot in range(PROVENANCE_CAPACITY)
)
if len(
    {
        *_INTERNAL_CONTEXT_ASSEMBLIES,
        *_INTERNAL_SOURCE_ASSEMBLIES,
    }
) != 2 * PROVENANCE_CAPACITY:
    raise RuntimeError("internal physical segment assemblies collided")


def _authenticate_dependencies() -> None:
    if (
        sha256(_EVENTWISE_SOURCE_PATH.read_bytes()).hexdigest()
        != EXPECTED_EVENTWISE_SOURCE_SHA256
        or eventwise.SOURCE_SHA256 != EXPECTED_EVENTWISE_SOURCE_SHA256
        or eventwise.CONFIG_SHA256 != EXPECTED_EVENTWISE_CONFIG_SHA256
        or eventwise.TOTAL_CELL_COUNT != 65_742
        or eventwise.NEW_CELL_COUNT != 100
        or eventwise.PARENT_LESSON_REPLAY_COUNT != 0
    ):
        raise RuntimeError("banked eventwise relation identity changed")


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "eventwise_source_commit": EVENTWISE_SOURCE_COMMIT,
    "eventwise_source_sha256": EXPECTED_EVENTWISE_SOURCE_SHA256,
    "eventwise_config_sha256": EXPECTED_EVENTWISE_CONFIG_SHA256,
    "public_interface": (
        "start-authenticated-eventwise-parent",
        "advance-exactly-one-physical-timed-event",
    ),
    "interval_cell_count": INTERVAL_CELL_COUNT,
    "external_context_source_input_count": EXTERNAL_CONTEXT_SOURCE_INPUT_COUNT,
    "boundary_bit_input_count": BOUNDARY_BIT_INPUT_COUNT,
    "raw_elapsed_time_input_count": RAW_ELAPSED_TIME_INPUT_COUNT,
    "first_segment_bootstrap": "single-stream-start-latch-calibration",
    "decision": "physical-continue-new-tonic-race",
    "boundary_evidence": (
        "learned-opaque-interval-cue",
        "physical-route-consequence-surprise",
    ),
    "independence": "new-winner-reserves-next-unused-banked-provenance-cell",
    "new_requires_preceding_neuronal_completeness": True,
    "internal_context_source_assemblies": (
        "private-continuity-plumbing-never-external-segmentation-authority"
    ),
    "comparator_synapse_channel_count": COMPARATOR_SYNAPSE_CHANNEL_COUNT,
    "cue_synapse_channel_count": CUE_SYNAPSE_CHANNEL_COUNT,
    "parent_lesson_replay_count": PARENT_LESSON_REPLAY_COUNT,
    "new_cell_count": NEW_CELL_COUNT,
    "total_cell_count": TOTAL_CELL_COUNT,
    "added_persistent_state_bytes": ADDED_PERSISTENT_STATE_BYTES,
    "persistent_state_bytes": PERSISTENT_STATE_BYTES,
    "segmentation_cell_steps_per_event": SEGMENTATION_CELL_STEPS_PER_EVENT,
    "routed_event_cell_steps": ROUTED_EVENT_CELL_STEPS,
    "latent_boundary_identifiability": (
        "identical-physical-event-consequence-and-interval-streams-have-"
        "identical-state-transitions"
    ),
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


@dataclass(frozen=True, slots=True)
class PhysicalTimedRelationEvent:
    """One physical consequence event with one opaque interval cell."""

    input_frame: projection.CommonOneByteFillerFrame
    relation_cells: tuple[int, ...]
    teacher_output_cells: tuple[int, ...]
    interval_cell_id: int
    teacher_gate: bool = True
    cell_gate: bool = True

    def __post_init__(self) -> None:
        if type(self.input_frame) is not projection.CommonOneByteFillerFrame:
            raise TypeError("input_frame must be the exact physical V2 frame type")
        relation_v2._validate_opaque_assembly("relation_cells", self.relation_cells)
        relation_v2._teacher_candidate_indices(self.teacher_output_cells)
        if type(self.interval_cell_id) is not int:
            raise TypeError("interval cell must be one exact physical cell")
        if self.interval_cell_id not in INTERVAL_CELL_IDS:
            raise ValueError("interval cell is outside the opaque two-cell bank")
        if type(self.teacher_gate) is not bool or type(self.cell_gate) is not bool:
            raise TypeError("teacher_gate and cell_gate must be exact booleans")


class FrozenPredictiveSegmenterV1:
    """Immutable learned segment gate wrapped around one eventwise state."""

    __slots__ = (
        "_calibration",
        "_checkpoint_sha256",
        "_complete_hold",
        "_cue_excitatory",
        "_cue_inhibitory",
        "_event_count",
        "_inner",
        "_internal_provenance_ids",
        "_opening_cue_index",
        "_opening_new_eligible",
        "_physical_cell_steps",
        "_plastic_update_ticks",
        "_previous_route_targets",
        "_reserved_provenance_cell_id",
        "_segment_count",
        "_within_cue_counts",
    )

    def __init__(
        self,
        *,
        inner: eventwise.FrozenEventwiseOneByteRelationV1,
        cue_excitatory: np.ndarray,
        cue_inhibitory: np.ndarray,
        within_cue_counts: np.ndarray,
        previous_route_targets: np.ndarray,
        internal_provenance_ids: np.ndarray,
        complete_hold: bool,
        calibration: bool,
        opening_cue_index: int,
        opening_new_eligible: bool,
        reserved_provenance_cell_id: int,
        segment_count: int,
        event_count: int,
        physical_cell_steps: int,
        plastic_update_ticks: int,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("predictive segmenter states are builder-owned")
        if type(inner) is not eventwise.FrozenEventwiseOneByteRelationV1:
            raise TypeError("inner must be the exact banked eventwise state type")
        inner.authenticate()
        arrays = (
            (
                cue_excitatory,
                (INTERVAL_CELL_COUNT, SEGMENT_DECISION_COUNT),
                TICK_DTYPE,
            ),
            (
                cue_inhibitory,
                (INTERVAL_CELL_COUNT, SEGMENT_DECISION_COUNT),
                TICK_DTYPE,
            ),
            (within_cue_counts, (INTERVAL_CELL_COUNT,), TICK_DTYPE),
            (previous_route_targets, (ROUTES,), TARGET_DTYPE),
            (
                internal_provenance_ids,
                (PROVENANCE_CAPACITY,),
                PROVENANCE_DTYPE,
            ),
        )
        if any(
            array.shape != shape
            or array.dtype != dtype
            or not array.flags.c_contiguous
            for array, shape, dtype in arrays
        ):
            raise ValueError("predictive segmenter state geometry changed")
        if (
            np.any(cue_excitatory > TICK_CAP)
            or np.any(cue_inhibitory > TICK_CAP)
            or np.any(within_cue_counts > TICK_CAP)
            or np.any(previous_route_targets < -1)
            or np.any(previous_route_targets >= RADIX)
        ):
            raise ValueError("predictive segmenter bounded state changed")
        if type(complete_hold) is not bool or type(calibration) is not bool:
            raise TypeError("segment latches must be exact booleans")
        if type(opening_new_eligible) is not bool:
            raise TypeError("opening eligibility must be an exact boolean")
        counters = (
            segment_count,
            event_count,
            physical_cell_steps,
            plastic_update_ticks,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("predictive segmenter counters changed")
        if not 0 <= segment_count <= PROVENANCE_CAPACITY:
            raise ValueError("internal segment capacity changed")
        if opening_cue_index not in (-1, 0, 1):
            raise ValueError("opening cue index changed")
        if reserved_provenance_cell_id < -1:
            raise ValueError("reserved provenance cell changed")
        if inner.active:
            if complete_hold or segment_count == 0 or reserved_provenance_cell_id < 0:
                raise ValueError("active internal segment latch changed")
        elif complete_hold:
            if segment_count == 0 or np.any(previous_route_targets < 0):
                raise ValueError("complete-hold segment lost its route model")
            if reserved_provenance_cell_id != -1:
                raise ValueError("completed segment retained a reservation")
        elif segment_count != 0:
            raise ValueError("inactive segmenter lost its complete-hold latch")
        self._inner = inner
        self._complete_hold = complete_hold
        self._calibration = calibration
        self._opening_cue_index = opening_cue_index
        self._opening_new_eligible = opening_new_eligible
        self._reserved_provenance_cell_id = reserved_provenance_cell_id
        self._segment_count = segment_count
        self._event_count = event_count
        self._physical_cell_steps = physical_cell_steps
        self._plastic_update_ticks = plastic_update_ticks
        for name, array in (
            ("_cue_excitatory", cue_excitatory),
            ("_cue_inhibitory", cue_inhibitory),
            ("_within_cue_counts", within_cue_counts),
            ("_previous_route_targets", previous_route_targets),
            ("_internal_provenance_ids", internal_provenance_ids),
        ):
            copied = np.array(array, copy=True, order="C")
            copied.setflags(write=False)
            setattr(self, name, copied)
        self._checkpoint_sha256 = self._calculate_checkpoint_sha256()
        self.authenticate()

    def _metadata_record(self) -> dict[str, object]:
        return {
            "scope": SCOPE,
            "status": STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "config_sha256": CONFIG_SHA256,
            "inner_checkpoint_sha256": self._inner.checkpoint_sha256,
            "complete_hold": self._complete_hold,
            "calibration": self._calibration,
            "opening_cue_index": self._opening_cue_index,
            "opening_new_eligible": self._opening_new_eligible,
            "reserved_provenance_cell_id": self._reserved_provenance_cell_id,
            "segment_count": self._segment_count,
            "event_count": self._event_count,
            "physical_cell_steps": self._physical_cell_steps,
            "plastic_update_ticks": self._plastic_update_ticks,
            "external_context_source_input_count": 0,
            "parent_lesson_replay_count": 0,
        }

    def _calculate_checkpoint_sha256(self) -> str:
        return sha256(
            _STATE_NAMESPACE
            + _canonical_json(self._metadata_record())
            + _array_frame("cue-excitatory", self._cue_excitatory)
            + _array_frame("cue-inhibitory", self._cue_inhibitory)
            + _array_frame("within-cue-counts", self._within_cue_counts)
            + _array_frame("previous-route-targets", self._previous_route_targets)
            + _array_frame(
                "internal-provenance-ids", self._internal_provenance_ids
            )
        ).hexdigest()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def inner(self) -> eventwise.FrozenEventwiseOneByteRelationV1:
        self.authenticate()
        return self._inner

    @property
    def v2_parent(self) -> relation_v2.FrozenCommonOneByteCopyRelationV2:
        self.authenticate()
        return self._inner.parent

    @property
    def cue_excitatory(self) -> tuple[tuple[int, ...], ...]:
        self.authenticate()
        return tuple(tuple(int(value) for value in row) for row in self._cue_excitatory)

    @property
    def cue_inhibitory(self) -> tuple[tuple[int, ...], ...]:
        self.authenticate()
        return tuple(tuple(int(value) for value in row) for row in self._cue_inhibitory)

    @property
    def internal_provenance_cell_ids(self) -> tuple[int, ...]:
        self.authenticate()
        return tuple(
            int(value) for value in self._internal_provenance_ids if value >= 0
        )

    @property
    def reserved_provenance_cell_id(self) -> int | None:
        self.authenticate()
        return (
            None
            if self._reserved_provenance_cell_id < 0
            else self._reserved_provenance_cell_id
        )

    @property
    def segment_count(self) -> int:
        self.authenticate()
        return self._segment_count

    @property
    def event_count(self) -> int:
        self.authenticate()
        return self._event_count

    @property
    def complete_hold(self) -> bool:
        self.authenticate()
        return self._complete_hold

    @property
    def calibration(self) -> bool:
        self.authenticate()
        return self._calibration

    @property
    def physical_cell_steps(self) -> int:
        self.authenticate()
        return self._physical_cell_steps

    @property
    def plastic_update_ticks(self) -> int:
        self.authenticate()
        return self._plastic_update_ticks

    def authenticate(self) -> None:
        _authenticate_dependencies()
        self._inner.authenticate()
        if self._calculate_checkpoint_sha256() != self._checkpoint_sha256:
            raise RuntimeError("predictive segmenter state authentication failed")


@dataclass(frozen=True, slots=True)
class PredictiveSegmentationTransition:
    status: str
    successor: FrozenPredictiveSegmenterV1
    decision: str
    decision_authority: str
    accepted_event: bool
    segment_started: bool
    segment_completed: bool
    match_count: int
    mismatch_count: int
    decision_raw_scores: tuple[int, int]
    interval_cell_id: int
    cue_plastic_update_ticks: int
    segmentation_physical_cell_steps: int
    inner_physical_cell_steps: int
    provenance_cell_id: int | None
    parent_checkpoint_before_sha256: str
    parent_checkpoint_after_sha256: str
    sha256: str

    @property
    def physical_cell_steps(self) -> int:
        return self.segmentation_physical_cell_steps + self.inner_physical_cell_steps


@dataclass(frozen=True, slots=True)
class _DecisionRace:
    decision: str
    raw_scores: tuple[int, int]
    match_count: int
    mismatch_count: int


def _new_state(
    *,
    inner: eventwise.FrozenEventwiseOneByteRelationV1,
    cue_excitatory: np.ndarray,
    cue_inhibitory: np.ndarray,
    within_cue_counts: np.ndarray,
    previous_route_targets: np.ndarray,
    internal_provenance_ids: np.ndarray,
    complete_hold: bool,
    calibration: bool,
    opening_cue_index: int,
    opening_new_eligible: bool,
    reserved_provenance_cell_id: int,
    segment_count: int,
    event_count: int,
    physical_cell_steps: int,
    plastic_update_ticks: int,
) -> FrozenPredictiveSegmenterV1:
    return FrozenPredictiveSegmenterV1(
        inner=inner,
        cue_excitatory=cue_excitatory,
        cue_inhibitory=cue_inhibitory,
        within_cue_counts=within_cue_counts,
        previous_route_targets=previous_route_targets,
        internal_provenance_ids=internal_provenance_ids,
        complete_hold=complete_hold,
        calibration=calibration,
        opening_cue_index=opening_cue_index,
        opening_new_eligible=opening_new_eligible,
        reserved_provenance_cell_id=reserved_provenance_cell_id,
        segment_count=segment_count,
        event_count=event_count,
        physical_cell_steps=physical_cell_steps,
        plastic_update_ticks=plastic_update_ticks,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _transition(
    *,
    status: str,
    successor: FrozenPredictiveSegmenterV1,
    decision: str,
    decision_authority: str,
    accepted_event: bool,
    segment_started: bool,
    segment_completed: bool,
    match_count: int,
    mismatch_count: int,
    decision_raw_scores: tuple[int, int],
    interval_cell_id: int,
    cue_plastic_update_ticks: int,
    segmentation_physical_cell_steps: int,
    inner_physical_cell_steps: int,
    provenance_cell_id: int | None,
    parent_checkpoint_before_sha256: str,
) -> PredictiveSegmentationTransition:
    record = {
        "status": status,
        "successor_checkpoint_sha256": successor.checkpoint_sha256,
        "decision": decision,
        "decision_authority": decision_authority,
        "accepted_event": accepted_event,
        "segment_started": segment_started,
        "segment_completed": segment_completed,
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "decision_raw_scores": decision_raw_scores,
        "interval_cell_id": interval_cell_id,
        "cue_plastic_update_ticks": cue_plastic_update_ticks,
        "segmentation_physical_cell_steps": segmentation_physical_cell_steps,
        "inner_physical_cell_steps": inner_physical_cell_steps,
        "provenance_cell_id": provenance_cell_id,
        "parent_checkpoint_before_sha256": parent_checkpoint_before_sha256,
        "parent_checkpoint_after_sha256": successor.v2_parent.checkpoint_sha256,
    }
    return PredictiveSegmentationTransition(
        status=status,
        successor=successor,
        decision=decision,
        decision_authority=decision_authority,
        accepted_event=accepted_event,
        segment_started=segment_started,
        segment_completed=segment_completed,
        match_count=match_count,
        mismatch_count=mismatch_count,
        decision_raw_scores=decision_raw_scores,
        interval_cell_id=interval_cell_id,
        cue_plastic_update_ticks=cue_plastic_update_ticks,
        segmentation_physical_cell_steps=segmentation_physical_cell_steps,
        inner_physical_cell_steps=inner_physical_cell_steps,
        provenance_cell_id=provenance_cell_id,
        parent_checkpoint_before_sha256=parent_checkpoint_before_sha256,
        parent_checkpoint_after_sha256=successor.v2_parent.checkpoint_sha256,
        sha256=_digest(_TRANSITION_NAMESPACE, record),
    )


def _interval_index(interval_cell_id: int) -> int:
    return INTERVAL_CELL_IDS.index(interval_cell_id)


def _comparison_counts(
    previous_route_targets: np.ndarray,
    input_routes: tuple[int, ...],
    teacher_candidates: tuple[int, ...],
    *,
    comparator_gate: bool,
) -> tuple[int, int]:
    if not comparator_gate:
        return 0, 0
    matches = 0
    mismatches = 0
    for lane, flat_input in enumerate(input_routes):
        predicted = int(previous_route_targets[flat_input])
        if predicted < 0:
            continue
        teacher = teacher_candidates[lane]
        predicted_cell = relation_v2._OUTPUT_CANDIDATES_BY_LANE[lane][predicted]
        teacher_cell = relation_v2._OUTPUT_CANDIDATES_BY_LANE[lane][teacher]
        if predicted_cell == teacher_cell:
            matches += 1
        else:
            mismatches += 1
    return matches, mismatches


def _race_decision(
    cue_excitatory: np.ndarray,
    cue_inhibitory: np.ndarray,
    *,
    cue_index: int,
    cue_gate: bool,
    complete_hold: bool,
    match_count: int,
    mismatch_count: int,
) -> _DecisionRace:
    if cue_gate:
        cue_scores = tuple(
            max(
                int(cue_excitatory[cue_index, decision])
                - int(cue_inhibitory[cue_index, decision]),
                0,
            )
            * SURPRISE_SCORE_GAIN
            for decision in range(SEGMENT_DECISION_COUNT)
        )
    else:
        cue_scores = (0, 0)
    continue_score = cue_scores[CONTINUE_DECISION]
    new_score = cue_scores[NEW_DECISION]
    if complete_hold:
        continue_score += match_count * MATCH_SCORE_GAIN
        new_score += mismatch_count * SURPRISE_SCORE_GAIN
    bounded = (
        min(continue_score, MAX_SCORE),
        min(new_score, MAX_SCORE),
    )
    currents = np.asarray(
        [[
            output_projection._CURRENT_TABLE[bounded[CONTINUE_DECISION]],
            output_projection._CURRENT_TABLE[bounded[NEW_DECISION]],
            output_projection._CURRENT_TABLE[TONIC_SCORE],
        ]],
        dtype=np.float64,
    )
    race = output_projection._OUTPUT_RACE.race_many(
        currents,
        require_unique=False,
    )[0]
    if race.simultaneous_spike_candidates != 1:
        decision = "COLLISION"
    elif race.winner_cell == 0:
        decision = "CONTINUE"
    elif race.winner_cell == 1:
        decision = "NEW"
    else:
        decision = "TONIC_STOP"
    return _DecisionRace(
        decision=decision,
        raw_scores=bounded,
        match_count=match_count,
        mismatch_count=mismatch_count,
    )


def _reserve_provenance_cell(
    inner: eventwise.FrozenEventwiseOneByteRelationV1,
) -> int:
    context_count = inner.parent._context_count
    if context_count == relation_v2.MAX_INDEPENDENT_CONTEXTS:
        raise RuntimeError("banked provenance capacity exhausted")
    return relation_v2.PROVENANCE_CELL_OFFSET + context_count


def _internal_lesson(
    event: PhysicalTimedRelationEvent,
    *,
    segment_slot: int,
) -> relation_v2.RelationGatedOneByteLesson:
    return relation_v2.RelationGatedOneByteLesson(
        input_frame=event.input_frame,
        relation_cells=event.relation_cells,
        context_cells=_INTERNAL_CONTEXT_ASSEMBLIES[segment_slot],
        source_cells=_INTERNAL_SOURCE_ASSEMBLIES[segment_slot],
        teacher_output_cells=event.teacher_output_cells,
        teacher_gate=True,
        cell_gate=True,
    )


def _targets_from_latest_provenance(
    parent: relation_v2.FrozenCommonOneByteCopyRelationV2,
) -> tuple[int, np.ndarray]:
    provenance = parent.training_summary.provenance[-1]
    targets = np.full((ROUTES,), -1, dtype=TARGET_DTYPE)
    for flat_input, output_cell in enumerate(provenance.route_output_cells):
        lane, _ = divmod(flat_input, RADIX)
        candidates = relation_v2._OUTPUT_CANDIDATES_BY_LANE[lane]
        if output_cell not in candidates:
            raise RuntimeError("internal provenance output lane changed")
        targets[flat_input] = candidates.index(output_cell)
    return provenance.provenance_cell_id, targets


def _apply_completion_plasticity(
    cue_excitatory: np.ndarray,
    cue_inhibitory: np.ndarray,
    within_cue_counts: np.ndarray,
    *,
    opening_cue_index: int,
    opening_new_eligible: bool,
) -> int:
    update_ticks = 0
    for cue_index, count_value in enumerate(within_cue_counts):
        count = int(count_value)
        if count == 0:
            continue
        previous_e = int(cue_excitatory[cue_index, CONTINUE_DECISION])
        updated_e = min(previous_e + count, TICK_CAP)
        cue_excitatory[cue_index, CONTINUE_DECISION] = updated_e
        update_ticks += updated_e - previous_e
        previous_h = int(cue_inhibitory[cue_index, NEW_DECISION])
        updated_h = min(previous_h + count, TICK_CAP)
        cue_inhibitory[cue_index, NEW_DECISION] = updated_h
        update_ticks += updated_h - previous_h
    if opening_new_eligible:
        if opening_cue_index not in (0, 1):
            raise RuntimeError("eligible opening lost its physical cue")
        previous_e = int(cue_excitatory[opening_cue_index, NEW_DECISION])
        updated_e = min(previous_e + 1, TICK_CAP)
        cue_excitatory[opening_cue_index, NEW_DECISION] = updated_e
        update_ticks += updated_e - previous_e
        previous_h = int(cue_inhibitory[opening_cue_index, CONTINUE_DECISION])
        updated_h = min(previous_h + 1, TICK_CAP)
        cue_inhibitory[opening_cue_index, CONTINUE_DECISION] = updated_h
        update_ticks += updated_h - previous_h
    return update_ticks


def start_predictive_segmenter_v1(
    parent: eventwise.FrozenEventwiseOneByteRelationV1,
) -> FrozenPredictiveSegmenterV1:
    """Start one learned segment gate from an inactive authenticated parent."""

    _authenticate_dependencies()
    if type(parent) is not eventwise.FrozenEventwiseOneByteRelationV1:
        raise TypeError("parent must be the exact banked eventwise state type")
    parent.authenticate()
    if parent.active or parent.pending_event_count != 0:
        raise ValueError("predictive segmenter requires an inactive eventwise parent")
    return _new_state(
        inner=parent,
        cue_excitatory=np.zeros(
            (INTERVAL_CELL_COUNT, SEGMENT_DECISION_COUNT), dtype=TICK_DTYPE
        ),
        cue_inhibitory=np.zeros(
            (INTERVAL_CELL_COUNT, SEGMENT_DECISION_COUNT), dtype=TICK_DTYPE
        ),
        within_cue_counts=np.zeros((INTERVAL_CELL_COUNT,), dtype=TICK_DTYPE),
        previous_route_targets=np.full((ROUTES,), -1, dtype=TARGET_DTYPE),
        internal_provenance_ids=np.full(
            (PROVENANCE_CAPACITY,), -1, dtype=PROVENANCE_DTYPE
        ),
        complete_hold=False,
        calibration=True,
        opening_cue_index=-1,
        opening_new_eligible=False,
        reserved_provenance_cell_id=-1,
        segment_count=0,
        event_count=0,
        physical_cell_steps=0,
        plastic_update_ticks=0,
    )


def advance_predictive_segmenter_v1(
    state: FrozenPredictiveSegmenterV1,
    event: PhysicalTimedRelationEvent,
    *,
    cue_gate: bool = True,
    plasticity_gate: bool = True,
    comparator_gate: bool = True,
) -> PredictiveSegmentationTransition:
    """Consume exactly one timed physical event and return a rooted successor."""

    if type(state) is not FrozenPredictiveSegmenterV1:
        raise TypeError("state must be the exact frozen predictive segmenter type")
    if type(event) is not PhysicalTimedRelationEvent:
        raise TypeError("advance accepts exactly one physical timed event")
    if any(type(gate) is not bool for gate in (cue_gate, plasticity_gate, comparator_gate)):
        raise TypeError("segmenter gates must be exact booleans")
    state.authenticate()
    parent_before = state.v2_parent.checkpoint_sha256
    cue_index = _interval_index(event.interval_cell_id)
    try:
        input_routes = relation_v2._input_route_indices(_INPUT_CELLS, event.input_frame)
        teacher_candidates = relation_v2._teacher_candidate_indices(
            event.teacher_output_cells
        )
    except (TypeError, ValueError):
        return _transition(
            status="STOP_EVENT_AUTHENTICATION",
            successor=state,
            decision="STOP",
            decision_authority="AUTHENTICATION",
            accepted_event=False,
            segment_started=False,
            segment_completed=False,
            match_count=0,
            mismatch_count=0,
            decision_raw_scores=(0, 0),
            interval_cell_id=event.interval_cell_id,
            cue_plastic_update_ticks=0,
            segmentation_physical_cell_steps=0,
            inner_physical_cell_steps=0,
            provenance_cell_id=None,
            parent_checkpoint_before_sha256=parent_before,
        )
    if not event.teacher_gate or not event.cell_gate:
        return _transition(
            status="STOP_TEACHER_OFF" if not event.teacher_gate else "STOP_CELL_OFF",
            successor=state,
            decision="STOP",
            decision_authority="PHYSICAL_GATE",
            accepted_event=False,
            segment_started=False,
            segment_completed=False,
            match_count=0,
            mismatch_count=0,
            decision_raw_scores=(0, 0),
            interval_cell_id=event.interval_cell_id,
            cue_plastic_update_ticks=0,
            segmentation_physical_cell_steps=0,
            inner_physical_cell_steps=0,
            provenance_cell_id=None,
            parent_checkpoint_before_sha256=parent_before,
        )

    cue_excitatory = np.array(state._cue_excitatory, copy=True, order="C")
    cue_inhibitory = np.array(state._cue_inhibitory, copy=True, order="C")
    within_cue_counts = np.array(state._within_cue_counts, copy=True, order="C")
    previous_route_targets = np.array(
        state._previous_route_targets, copy=True, order="C"
    )
    internal_provenance_ids = np.array(
        state._internal_provenance_ids, copy=True, order="C"
    )
    decision = "CONTINUE"
    decision_authority = "CALIBRATION"
    match_count = 0
    mismatch_count = 0
    decision_scores = (0, 0)
    segment_started = False
    opening_event = False
    segment_slot = state._segment_count - 1
    segment_count = state._segment_count
    reserved_provenance_cell_id = state._reserved_provenance_cell_id
    opening_cue_index = state._opening_cue_index
    opening_new_eligible = state._opening_new_eligible

    if state._segment_count == 0:
        decision = "NEW"
        decision_authority = "STREAM_START"
        segment_started = True
        opening_event = True
        segment_slot = 0
        segment_count = 1
        reserved_provenance_cell_id = _reserve_provenance_cell(state._inner)
        opening_cue_index = -1
        opening_new_eligible = False
        within_cue_counts.fill(0)
    elif state._inner.active:
        if state._calibration:
            decision = "CONTINUE"
            decision_authority = "CALIBRATION"
        else:
            race = _race_decision(
                cue_excitatory,
                cue_inhibitory,
                cue_index=cue_index,
                cue_gate=cue_gate,
                complete_hold=False,
                match_count=0,
                mismatch_count=0,
            )
            decision = race.decision
            decision_scores = race.raw_scores
            if decision == "NEW":
                return _transition(
                    status="STOP_PREMATURE_NEW",
                    successor=state,
                    decision=decision,
                    decision_authority="LEARNED_CUE",
                    accepted_event=False,
                    segment_started=False,
                    segment_completed=False,
                    match_count=0,
                    mismatch_count=0,
                    decision_raw_scores=decision_scores,
                    interval_cell_id=event.interval_cell_id,
                    cue_plastic_update_ticks=0,
                    segmentation_physical_cell_steps=(
                        SEGMENTATION_CELL_STEPS_PER_EVENT
                    ),
                    inner_physical_cell_steps=0,
                    provenance_cell_id=state.reserved_provenance_cell_id,
                    parent_checkpoint_before_sha256=parent_before,
                )
            if decision != "CONTINUE":
                return _transition(
                    status="ABSTAIN_SEGMENT_AMBIGUITY",
                    successor=state,
                    decision=decision,
                    decision_authority="TONIC_OR_COLLISION",
                    accepted_event=False,
                    segment_started=False,
                    segment_completed=False,
                    match_count=0,
                    mismatch_count=0,
                    decision_raw_scores=decision_scores,
                    interval_cell_id=event.interval_cell_id,
                    cue_plastic_update_ticks=0,
                    segmentation_physical_cell_steps=(
                        SEGMENTATION_CELL_STEPS_PER_EVENT
                    ),
                    inner_physical_cell_steps=0,
                    provenance_cell_id=state.reserved_provenance_cell_id,
                    parent_checkpoint_before_sha256=parent_before,
                )
            decision_authority = "LEARNED_CUE"
    elif state._complete_hold:
        match_count, mismatch_count = _comparison_counts(
            previous_route_targets,
            input_routes,
            teacher_candidates,
            comparator_gate=comparator_gate,
        )
        race = _race_decision(
            cue_excitatory,
            cue_inhibitory,
            cue_index=cue_index,
            cue_gate=cue_gate,
            complete_hold=True,
            match_count=match_count,
            mismatch_count=mismatch_count,
        )
        decision = race.decision
        decision_scores = race.raw_scores
        if decision == "CONTINUE":
            successor = _new_state(
                inner=state._inner,
                cue_excitatory=cue_excitatory,
                cue_inhibitory=cue_inhibitory,
                within_cue_counts=within_cue_counts,
                previous_route_targets=previous_route_targets,
                internal_provenance_ids=internal_provenance_ids,
                complete_hold=True,
                calibration=False,
                opening_cue_index=-1,
                opening_new_eligible=False,
                reserved_provenance_cell_id=-1,
                segment_count=segment_count,
                event_count=state._event_count + 1,
                physical_cell_steps=(
                    state._physical_cell_steps
                    + SEGMENTATION_CELL_STEPS_PER_EVENT
                ),
                plastic_update_ticks=state._plastic_update_ticks,
            )
            return _transition(
                status="SAME_SEGMENT_CONTINUE",
                successor=successor,
                decision=decision,
                decision_authority="LEARNED_CUE",
                accepted_event=True,
                segment_started=False,
                segment_completed=False,
                match_count=match_count,
                mismatch_count=mismatch_count,
                decision_raw_scores=decision_scores,
                interval_cell_id=event.interval_cell_id,
                cue_plastic_update_ticks=0,
                segmentation_physical_cell_steps=SEGMENTATION_CELL_STEPS_PER_EVENT,
                inner_physical_cell_steps=0,
                provenance_cell_id=None,
                parent_checkpoint_before_sha256=parent_before,
            )
        if decision != "NEW":
            return _transition(
                status="ABSTAIN_SEGMENT_AMBIGUITY",
                successor=state,
                decision=decision,
                decision_authority="TONIC_OR_COLLISION",
                accepted_event=False,
                segment_started=False,
                segment_completed=False,
                match_count=match_count,
                mismatch_count=mismatch_count,
                decision_raw_scores=decision_scores,
                interval_cell_id=event.interval_cell_id,
                cue_plastic_update_ticks=0,
                segmentation_physical_cell_steps=SEGMENTATION_CELL_STEPS_PER_EVENT,
                inner_physical_cell_steps=0,
                provenance_cell_id=None,
                parent_checkpoint_before_sha256=parent_before,
            )
        if segment_count == PROVENANCE_CAPACITY:
            return _transition(
                status="STOP_PROVENANCE_CAPACITY",
                successor=state,
                decision=decision,
                decision_authority=(
                    "CONSEQUENCE_SURPRISE" if mismatch_count >= 2 else "LEARNED_CUE"
                ),
                accepted_event=False,
                segment_started=False,
                segment_completed=False,
                match_count=match_count,
                mismatch_count=mismatch_count,
                decision_raw_scores=decision_scores,
                interval_cell_id=event.interval_cell_id,
                cue_plastic_update_ticks=0,
                segmentation_physical_cell_steps=SEGMENTATION_CELL_STEPS_PER_EVENT,
                inner_physical_cell_steps=0,
                provenance_cell_id=None,
                parent_checkpoint_before_sha256=parent_before,
            )
        decision_authority = (
            "CONSEQUENCE_SURPRISE" if mismatch_count >= 2 else "LEARNED_CUE"
        )
        segment_started = True
        opening_event = True
        segment_slot = segment_count
        segment_count += 1
        reserved_provenance_cell_id = _reserve_provenance_cell(state._inner)
        opening_cue_index = cue_index
        opening_new_eligible = plasticity_gate
        within_cue_counts.fill(0)
    else:
        raise RuntimeError("predictive segmenter lost active/complete state")

    inner_transition = eventwise.advance_eventwise_common_one_byte_relation_v1(
        state._inner,
        _internal_lesson(event, segment_slot=segment_slot),
    )
    if not inner_transition.accepted_event:
        raise RuntimeError("internal eventwise learner rejected a routed event")
    if not opening_event and plasticity_gate:
        previous_count = int(within_cue_counts[cue_index])
        within_cue_counts[cue_index] = min(previous_count + 1, TICK_CAP)
    inner = inner_transition.successor
    segment_completed = inner_transition.consolidated
    cue_plastic_ticks = 0
    provenance_cell_id: int | None = reserved_provenance_cell_id
    complete_hold = False
    calibration = state._calibration
    if segment_completed:
        assigned_provenance, previous_route_targets = _targets_from_latest_provenance(
            inner.parent
        )
        if assigned_provenance != reserved_provenance_cell_id:
            raise RuntimeError("reserved internal provenance cell changed")
        if assigned_provenance in internal_provenance_ids:
            raise RuntimeError("internal provenance cell was reused")
        internal_provenance_ids[segment_slot] = assigned_provenance
        cue_plastic_ticks = _apply_completion_plasticity(
            cue_excitatory,
            cue_inhibitory,
            within_cue_counts,
            opening_cue_index=opening_cue_index,
            opening_new_eligible=opening_new_eligible,
        )
        within_cue_counts.fill(0)
        complete_hold = True
        calibration = False
        opening_cue_index = -1
        opening_new_eligible = False
        reserved_provenance_cell_id = -1

    successor = _new_state(
        inner=inner,
        cue_excitatory=cue_excitatory,
        cue_inhibitory=cue_inhibitory,
        within_cue_counts=within_cue_counts,
        previous_route_targets=previous_route_targets,
        internal_provenance_ids=internal_provenance_ids,
        complete_hold=complete_hold,
        calibration=calibration,
        opening_cue_index=opening_cue_index,
        opening_new_eligible=opening_new_eligible,
        reserved_provenance_cell_id=reserved_provenance_cell_id,
        segment_count=segment_count,
        event_count=state._event_count + 1,
        physical_cell_steps=(
            state._physical_cell_steps
            + SEGMENTATION_CELL_STEPS_PER_EVENT
            + inner_transition.event_physical_cell_steps
        ),
        plastic_update_ticks=(
            state._plastic_update_ticks + cue_plastic_ticks
        ),
    )
    if segment_completed:
        status = (
            "STREAM_START_SEGMENT_COMPLETE"
            if decision_authority == "STREAM_START"
            else "SEGMENT_COMPLETE"
        )
    elif segment_started:
        status = (
            "STREAM_START_SEGMENT_PENDING"
            if decision_authority == "STREAM_START"
            else "NEW_SEGMENT_PENDING"
        )
    else:
        status = "CONTINUE_SEGMENT_PENDING"
    return _transition(
        status=status,
        successor=successor,
        decision=decision,
        decision_authority=decision_authority,
        accepted_event=True,
        segment_started=segment_started,
        segment_completed=segment_completed,
        match_count=match_count,
        mismatch_count=mismatch_count,
        decision_raw_scores=decision_scores,
        interval_cell_id=event.interval_cell_id,
        cue_plastic_update_ticks=cue_plastic_ticks,
        segmentation_physical_cell_steps=SEGMENTATION_CELL_STEPS_PER_EVENT,
        inner_physical_cell_steps=inner_transition.event_physical_cell_steps,
        provenance_cell_id=provenance_cell_id,
        parent_checkpoint_before_sha256=parent_before,
    )


_authenticate_dependencies()
SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()

__all__ = [
    "ADDED_PERSISTENT_STATE_BYTES",
    "BOUNDARY_BIT_INPUT_COUNT",
    "CLAIM_BOUNDARY",
    "CODING_INPUT_COUNT",
    "COMPARATOR_SYNAPSE_CHANNEL_COUNT",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "CORPUS_INPUT_COUNT",
    "CUE_EH_STATE_BYTES",
    "CUE_SYNAPSE_CHANNEL_COUNT",
    "DEV_INPUT_COUNT",
    "EXTERNAL_CONTEXT_SOURCE_INPUT_COUNT",
    "G0_INPUT_COUNT",
    "INTERVAL_CELL_COUNT",
    "INTERVAL_CELL_IDS",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "PARENT_LESSON_REPLAY_COUNT",
    "PATH2_INPUT_COUNT",
    "PERSISTENT_STATE_BYTES",
    "PROTECTED_INPUT_COUNT",
    "PROVENANCE_CAPACITY",
    "RAW_ELAPSED_TIME_INPUT_COUNT",
    "ROUTED_EVENT_CELL_STEPS",
    "SCOPE",
    "SEGMENTATION_CELL_STEPS_PER_EVENT",
    "SOURCE_SHA256",
    "STATUS",
    "TOTAL_CELL_COUNT",
    "FrozenPredictiveSegmenterV1",
    "PhysicalTimedRelationEvent",
    "PredictiveSegmentationTransition",
    "advance_predictive_segmenter_v1",
    "start_predictive_segmenter_v1",
]
