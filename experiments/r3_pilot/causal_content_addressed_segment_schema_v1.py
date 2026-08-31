"""Consolidate repeated causal segments into two reusable schema cells.

The banked segmented aggregate supplies learned prompt relations, a predictive
event boundary, and OPEN/EXTEND routing.  This successor observes the append
effect *before* the append checkpoint's reducer-action race, learns two local
route-target prototypes, and content-addresses later completed segments back to
those prototypes.  Only the winning schema cell reaches a new schema-to-reducer
E/H map.  The append checkpoint's direct reducer decision is never read.

This is a self-posed two-schema experiment.  Balanced one-byte framing,
interval cues, predicate/addition/reducer scaffolds, typed decoding, and final
compilation remain explicit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import causal_segmented_aggregate_stream_v1 as segmented
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as relation_v2
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter
from experiments.r3_pilot import public_typed_output_cells as typed_output
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.resting_glif_race import RestingGlifRaceFailure
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1"
READY = "CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_READY"
STOP_DEPENDENCY = "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_DEPENDENCY"
STOP_PROJECTION = "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_PROJECTION"
STOP_SEGMENT = "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_SEGMENT"
STOP_SCHEMA_TONIC = "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_SCHEMA_TONIC"
STOP_SCHEMA_AMBIGUOUS = (
    "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_SCHEMA_AMBIGUOUS"
)
STOP_ACTION_ZERO = "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_ACTION_ZERO"
STOP_ACTION_AMBIGUOUS = (
    "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_ACTION_AMBIGUOUS"
)
STOP_ACTION_CONSEQUENCE = (
    "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_ACTION_CONSEQUENCE"
)
STOP_PROBE_FINAL_OVERLAP = (
    "STOP_CAUSAL_CONTENT_ADDRESSED_SEGMENT_SCHEMA_V1_PROBE_FINAL_OVERLAP"
)

CLAIM_BOUNDARY = (
    "two self-posed causal one-byte transformation schemas are learned from "
    "physical outcomes, reused across A/B/A-prime/B-prime segments, and alone "
    "choose one of two explicit aggregate reducer scaffolds; prompt grouping, "
    "balanced byte framing, interval cues, predicate/addition/reducer "
    "implementations, typed decoding, and compilation remain explicit, so "
    "arbitrary segmentation, language understanding, public static inference, "
    "and coding capability remain unproven"
)

SCHEMA_CAPACITY = 2
ROUTES = segmenter.ROUTES
RADIX = segmenter.RADIX
TICK_DTYPE = np.dtype(np.uint8)
PROVENANCE_DTYPE = np.dtype(np.int32)
TICK_CAP = 4

SCHEMA_CELL_OFFSET = segmented.TOTAL_CELL_COUNT
SCHEMA_CELL_COUNT = SCHEMA_CAPACITY
NOVEL_CELL = SCHEMA_CELL_OFFSET + SCHEMA_CELL_COUNT
TONIC_STOP_CELL = NOVEL_CELL + 1
SCRATCH_COMPLETE_CELL = TONIC_STOP_CELL + 1
NEW_CELL_COUNT = SCHEMA_CELL_COUNT + 3
TOTAL_CELL_COUNT = segmented.TOTAL_CELL_COUNT + NEW_CELL_COUNT

PROTOTYPE_STATE_BYTES = (
    2 * SCHEMA_CAPACITY * ROUTES * RADIX * TICK_DTYPE.itemsize
)
ACTION_STATE_BYTES = (
    2
    * SCHEMA_CAPACITY
    * append.REDUCER_ACTION_CELL_COUNT
    * TICK_DTYPE.itemsize
)
VALID_AND_PROVENANCE_STATE_BYTES = (
    SCHEMA_CAPACITY * TICK_DTYPE.itemsize
    + SCHEMA_CAPACITY * PROVENANCE_DTYPE.itemsize
)
NEW_PERSISTENT_STATE_BYTES = (
    PROTOTYPE_STATE_BYTES
    + ACTION_STATE_BYTES
    + VALID_AND_PROVENANCE_STATE_BYTES
)
if TOTAL_CELL_COUNT > 69_213 or NEW_PERSISTENT_STATE_BYTES > 280:
    raise RuntimeError("content-addressed schema budget changed")

SCHEMA_RACE_CELL_COUNT = SCHEMA_CAPACITY + 2
SCHEMA_RACE_CELL_STEPS = SCHEMA_RACE_CELL_COUNT * output_projection.RACE_STEPS
ACTION_RACE_CELL_STEPS = append.ACTION_RACE_CELL_STEPS
FULL_SCHEMA_SCORE = ROUTES
NOVEL_SCORE = FULL_SCHEMA_SCORE

DESIRED_READS_DURING_BUILD = 0
DIRECT_APPEND_ACTION_READS = 0

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"causal-content-addressed-segment-schema-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_TARGET_NAMESPACE = _NAMESPACE + b"route-targets\0"
_SCHEMA_RACE_NAMESPACE = _NAMESPACE + b"schema-race\0"
_ACTION_RACE_NAMESPACE = _NAMESPACE + b"action-race\0"
_WORK_NAMESPACE = _NAMESPACE + b"work\0"
_STATE_NAMESPACE = _NAMESPACE + b"state\0"
_SELECTION_NAMESPACE = _NAMESPACE + b"selection\0"
_CONSTRUCTION_TOKEN = object()


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
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


class SchemaFitAudit(NamedTuple):
    episode_ordinal: int
    effect_relation: int
    schema_index: int
    disposition: str
    route_targets_sha256: str
    schema_race_sha256: str
    outcome_action: int
    outcome_race_sha256: str


class ContentAddressedSchemaTrainingSummary(NamedTuple):
    episode_count: int
    segmented_checkpoint_sha256: str
    persistent_schema_count: int
    episodic_segment_count: int
    novel_schema_count: int
    reused_segment_count: int
    schema_assignments: tuple[int, ...]
    effect_relations: tuple[int, ...]
    provenance_cell_ids: tuple[int, ...]
    prototype_excitatory_ticks: int
    prototype_inhibitory_ticks: int
    action_excitatory_ticks: int
    action_inhibitory_ticks: int
    residual_binding_cell_steps: int
    probe_comparison_cell_steps: int
    completeness_cell_steps: int
    effect_race_cell_steps: int
    effect_projection_cell_steps: int
    schema_race_cell_steps: int
    packed_authorization_count: int
    outcome_race_count: int
    direct_append_action_reads: int
    new_persistent_state_bytes: int
    fit_audits: tuple[SchemaFitAudit, ...]
    control: str
    checkpoint_sha256: str


class ContentAddressedSchemaWork(NamedTuple):
    residual_binding_cell_steps: int
    probe_comparison_cell_steps: int
    completeness_cell_steps: int
    effect_race_cell_steps: int
    input_projection_work: int
    effect_projection_cell_steps: int
    segmenter_cell_steps: int
    router_race_cell_steps: int
    schema_race_cell_steps: int
    action_race_cell_steps: int
    packed_authorization_count: int
    outcome_race_count: int
    total_physical_cell_steps: int
    projection_sha256: str | None
    transition_sha256s: tuple[str, ...]
    router_race_sha256s: tuple[str, ...]
    schema_race_sha256: str | None
    action_race_sha256: str | None
    outcome_race_sha256: str | None
    sha256: str


class ContentAddressedSchemaFrontierState(NamedTuple):
    status: str
    checkpoint_sha256: str
    residual_certificate_sha256: str
    probe_sha256: str
    segmenter_start_sha256: str
    segmenter_end_sha256: str
    effect_relation: int | None
    schema_index: int | None
    schema_cell_id: int | None
    schema_scores: tuple[int, ...]
    schema_action: int | None
    action_scores: tuple[int, ...]
    outcome_action: int | None
    router_actions: tuple[int, ...]
    scaffold_sha256: str | None
    probe_frontier_sha256: str | None
    delegated_frontier_sha256: str | None
    probe_packed_build_count: int
    final_packed_build_count: int
    desired_reads_during_build: int
    direct_append_action_reads: int
    persistent_schema_count: int
    work: ContentAddressedSchemaWork
    sha256: str


class ContentAddressedSchemaSelection(NamedTuple):
    status: str
    frontier_sha256: str
    delegated: plan_slice.PackedPromptPlanSliceSelection | None
    sha256: str


class ContentAddressedSchemaFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class _SchemaDecision(NamedTuple):
    status: str
    schema_index: int | None
    schema_cell_id: int | None
    raw_scores: tuple[int, ...]
    novel_score: int
    winner_cell: int
    spike_step: int
    simultaneous_spike_candidates: int
    sha256: str


def _route_targets(
    projection: segmented._ProjectedEffectEvents,
) -> tuple[np.ndarray, str]:
    targets = np.full((ROUTES,), -1, dtype=np.int8)
    for event in projection.events:
        input_routes = relation_v2._input_route_indices(
            segmented._INPUT_CELLS, event.input_frame
        )
        teacher_targets = relation_v2._teacher_candidate_indices(
            event.teacher_output_cells
        )
        for lane, route in enumerate(input_routes):
            target = teacher_targets[lane]
            if targets[route] not in (-1, target):
                raise ContentAddressedSchemaFitStop(
                    STOP_PROJECTION, "one physical route received conflicting targets"
                )
            targets[route] = target
    if np.any(targets < 0):
        raise ContentAddressedSchemaFitStop(
            STOP_PROJECTION, "balanced segment did not cover all physical routes"
        )
    root = sha256(
        _TARGET_NAMESPACE + _array_frame("route-targets", targets)
    ).hexdigest()
    targets.setflags(write=False)
    return targets, root


def _prototype_scores(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    valid: np.ndarray,
    targets: np.ndarray,
    *,
    comparator_gate: bool,
    schema_cell_gate: bool,
) -> tuple[int, ...]:
    if not comparator_gate or not schema_cell_gate:
        return (0,) * SCHEMA_CAPACITY
    scores: list[int] = []
    for schema_index in range(SCHEMA_CAPACITY):
        if not valid[schema_index]:
            scores.append(0)
            continue
        score = sum(
            int(excitatory[schema_index, route, int(targets[route])])
            - int(inhibitory[schema_index, route, int(targets[route])])
            for route in range(ROUTES)
        )
        scores.append(score)
    return tuple(scores)


def _race_schema(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    valid: np.ndarray,
    targets: np.ndarray,
    *,
    allow_novel: bool,
    comparator_gate: bool,
    schema_cell_gate: bool,
) -> _SchemaDecision:
    scores = _prototype_scores(
        excitatory,
        inhibitory,
        valid,
        targets,
        comparator_gate=comparator_gate,
        schema_cell_gate=schema_cell_gate,
    )
    full_matches = tuple(
        index
        for index, score in enumerate(scores)
        if valid[index] and score >= FULL_SCHEMA_SCORE
    )
    novel_score = (
        NOVEL_SCORE
        if allow_novel
        and schema_cell_gate
        and comparator_gate
        and not full_matches
        and int(np.count_nonzero(valid)) < SCHEMA_CAPACITY
        else 0
    )
    current_scores = (
        *(
            min(max(score, 0), output_projection.MAX_OUTPUT_SCORE)
            for score in scores
        ),
        novel_score,
        output_projection.STOP_SCORE,
    )
    currents = np.asarray(
        [[output_projection._CURRENT_TABLE[score] for score in current_scores]],
        dtype=np.float64,
    )
    race = output_projection._OUTPUT_RACE.race_many(
        currents, require_unique=False
    )[0]
    winner = int(race.winner_cell)
    if race.simultaneous_spike_candidates != 1:
        status = STOP_SCHEMA_AMBIGUOUS
        schema_index = None
        schema_cell_id = None
    elif winner < SCHEMA_CAPACITY and winner in full_matches:
        status = READY
        schema_index = winner
        schema_cell_id = SCHEMA_CELL_OFFSET + winner
    elif winner == SCHEMA_CAPACITY and novel_score:
        status = "NOVEL"
        schema_index = None
        schema_cell_id = NOVEL_CELL
    else:
        status = STOP_SCHEMA_TONIC
        schema_index = None
        schema_cell_id = None
    payload = {
        "scores": scores,
        "valid": tuple(int(value) for value in valid),
        "allow_novel": allow_novel,
        "comparator_gate": comparator_gate,
        "schema_cell_gate": schema_cell_gate,
        "novel_score": novel_score,
        "status": status,
        "schema_index": schema_index,
        "winner_cell": winner,
        "spike_step": race.spike_step,
        "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
        "cell_steps": SCHEMA_RACE_CELL_STEPS,
    }
    return _SchemaDecision(
        status,
        schema_index,
        schema_cell_id,
        scores,
        novel_score,
        winner,
        race.spike_step,
        race.simultaneous_spike_candidates,
        _digest(_SCHEMA_RACE_NAMESPACE, payload),
    )


def _learn_prototype(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    schema_index: int,
    targets: np.ndarray,
) -> tuple[int, int]:
    excitatory_ticks = 0
    inhibitory_ticks = 0
    for route in range(ROUTES):
        winner = int(targets[route])
        for target in range(RADIX):
            state = excitatory if target == winner else inhibitory
            before = int(state[schema_index, route, target])
            state[schema_index, route, target] = min(TICK_CAP, before + 1)
            delta = int(state[schema_index, route, target]) - before
            if target == winner:
                excitatory_ticks += delta
            else:
                inhibitory_ticks += delta
    return excitatory_ticks, inhibitory_ticks


def _learn_action(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    schema_index: int,
    action: int,
) -> tuple[int, int]:
    excitatory_ticks = 0
    inhibitory_ticks = 0
    for candidate in range(append.REDUCER_ACTION_CELL_COUNT):
        state = excitatory if candidate == action else inhibitory
        before = int(state[schema_index, candidate])
        state[schema_index, candidate] = min(TICK_CAP, before + 1)
        delta = int(state[schema_index, candidate]) - before
        if candidate == action:
            excitatory_ticks += delta
        else:
            inhibitory_ticks += delta
    return excitatory_ticks, inhibitory_ticks


def _race_action(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    schema_index: int,
    *,
    cell_gate: bool,
) -> tuple[str, int | None, tuple[int, ...], str]:
    scores = tuple(
        int(excitatory[schema_index, action])
        - int(inhibitory[schema_index, action])
        for action in range(append.REDUCER_ACTION_CELL_COUNT)
    )
    currents = np.asarray(
        [
            max(score, 0) * append.OUTPUT_CURRENT_PA if cell_gate else 0.0
            for score in scores
        ],
        dtype=np.float64,
    )
    try:
        race = append._ACTION_RACE.race_many(
            currents[None, :], require_unique=False
        )[0]
    except RestingGlifRaceFailure:
        status = STOP_ACTION_ZERO
        action = None
        spike_step = None
        simultaneous = 0
    else:
        spike_step = race.spike_step
        simultaneous = race.simultaneous_spike_candidates
        if simultaneous != 1:
            status = STOP_ACTION_AMBIGUOUS
            action = None
        else:
            status = READY
            action = int(race.winner_cell)
    payload = {
        "schema_index": schema_index,
        "scores": scores,
        "cell_gate": cell_gate,
        "status": status,
        "action": action,
        "spike_step": spike_step,
        "simultaneous_spike_candidates": simultaneous,
        "cell_steps": ACTION_RACE_CELL_STEPS,
    }
    return status, action, scores, _digest(_ACTION_RACE_NAMESPACE, payload)


def _causal_outcome_action(
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
    binding: residual.ResidualLiteralRouteBinding,
    desired: tuple[typed_output.TypedOutputEncoding, ...],
) -> tuple[int, str, int]:
    outputs, authorizations = append._authorize_reducers(frontier, binding)
    keys = ("schema-reducer-cell-0", "schema-reducer-cell-1")
    race = semantic_race.select_exact_semantic_vector(keys, outputs, desired)
    if race.status != semantic_race.SELECTED or race.winner_key is None:
        raise ContentAddressedSchemaFitStop(
            STOP_ACTION_CONSEQUENCE,
            "two reducer interventions did not produce one physical PASS",
        )
    return keys.index(race.winner_key), race.sha256, authorizations


def _observation_cell_steps(
    observation: append._AppendObservation,
) -> tuple[int, int, int, int]:
    residual_steps = observation.residual_certificate.work.stacked_binding_cell_steps
    comparison_steps = (
        observation.comparison_count * literal_role.EQUALITY_CELL_STEPS_PER_COMPARISON
    )
    completeness_steps = (
        observation.completeness_spikes * append.COMPLETENESS_CELL_STEPS
    )
    effect_steps = (
        append.SEMANTIC_RACE_CELL_STEPS
        if observation.effect_relation is not None
        else 0
    )
    return residual_steps, comparison_steps, completeness_steps, effect_steps


def _checkpoint_sha256(
    dependency_sha256: str,
    prototype_excitatory: np.ndarray,
    prototype_inhibitory: np.ndarray,
    action_excitatory: np.ndarray,
    action_inhibitory: np.ndarray,
    valid: np.ndarray,
    provenance_ids: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + dependency_sha256.encode("ascii")
        + b"\0"
        + _array_frame("prototype-excitatory", prototype_excitatory)
        + _array_frame("prototype-inhibitory", prototype_inhibitory)
        + _array_frame("action-excitatory", action_excitatory)
        + _array_frame("action-inhibitory", action_inhibitory)
        + _array_frame("valid", valid)
        + _array_frame("provenance-ids", provenance_ids)
    ).hexdigest()


class FrozenCausalContentAddressedSegmentSchemaV1:
    """Immutable two-schema prototype and reducer-policy checkpoint."""

    __slots__ = (
        "_action_excitatory",
        "_action_inhibitory",
        "_checkpoint_sha256",
        "_prototype_excitatory",
        "_prototype_inhibitory",
        "_provenance_ids",
        "_segmented_checkpoint",
        "_summary",
        "_valid",
    )

    def __init__(
        self,
        segmented_checkpoint: segmented.FrozenSegmentedAggregateV1,
        prototype_excitatory: np.ndarray,
        prototype_inhibitory: np.ndarray,
        action_excitatory: np.ndarray,
        action_inhibitory: np.ndarray,
        valid: np.ndarray,
        provenance_ids: np.ndarray,
        summary: ContentAddressedSchemaTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("content-addressed schema checkpoints are builder-owned")
        if type(segmented_checkpoint) is not segmented.FrozenSegmentedAggregateV1:
            raise TypeError("exact segmented aggregate checkpoint required")
        segmented_checkpoint.authenticate()
        prototype_shape = (SCHEMA_CAPACITY, ROUTES, RADIX)
        action_shape = (SCHEMA_CAPACITY, append.REDUCER_ACTION_CELL_COUNT)
        arrays = (
            (prototype_excitatory, prototype_shape, TICK_DTYPE),
            (prototype_inhibitory, prototype_shape, TICK_DTYPE),
            (action_excitatory, action_shape, TICK_DTYPE),
            (action_inhibitory, action_shape, TICK_DTYPE),
            (valid, (SCHEMA_CAPACITY,), TICK_DTYPE),
            (provenance_ids, (SCHEMA_CAPACITY,), PROVENANCE_DTYPE),
        )
        if any(
            array.shape != shape or array.dtype != dtype
            for array, shape, dtype in arrays
        ):
            raise ValueError("content-addressed schema state geometry changed")
        if (
            np.any(prototype_excitatory > TICK_CAP)
            or np.any(prototype_inhibitory > TICK_CAP)
            or np.any(action_excitatory > TICK_CAP)
            or np.any(action_inhibitory > TICK_CAP)
            or np.any(valid > 1)
        ):
            raise ValueError("content-addressed schema bounded state changed")
        self._segmented_checkpoint = segmented_checkpoint
        self._prototype_excitatory = np.array(
            prototype_excitatory, copy=True, order="C"
        )
        self._prototype_inhibitory = np.array(
            prototype_inhibitory, copy=True, order="C"
        )
        self._action_excitatory = np.array(action_excitatory, copy=True, order="C")
        self._action_inhibitory = np.array(action_inhibitory, copy=True, order="C")
        self._valid = np.array(valid, copy=True, order="C")
        self._provenance_ids = np.array(provenance_ids, copy=True, order="C")
        for array in (
            self._prototype_excitatory,
            self._prototype_inhibitory,
            self._action_excitatory,
            self._action_inhibitory,
            self._valid,
            self._provenance_ids,
        ):
            array.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha256(
            segmented_checkpoint.checkpoint_sha256,
            self._prototype_excitatory,
            self._prototype_inhibitory,
            self._action_excitatory,
            self._action_inhibitory,
            self._valid,
            self._provenance_ids,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("content-addressed schema summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> ContentAddressedSchemaTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            self._prototype_excitatory.shape,
            self._prototype_inhibitory.shape,
            self._action_excitatory.shape,
            self._action_inhibitory.shape,
            self._valid.shape,
            self._provenance_ids.shape,
        )

    def authenticate(self) -> None:
        self._segmented_checkpoint.authenticate()
        arrays = (
            self._prototype_excitatory,
            self._prototype_inhibitory,
            self._action_excitatory,
            self._action_inhibitory,
            self._valid,
            self._provenance_ids,
        )
        if (
            any(array.flags.writeable for array in arrays)
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                self._segmented_checkpoint.checkpoint_sha256,
                self._prototype_excitatory,
                self._prototype_inhibitory,
                self._action_excitatory,
                self._action_inhibitory,
                self._valid,
                self._provenance_ids,
            )
        ):
            raise RuntimeError("content-addressed schema authentication failed")

    def build(
        self,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        probe_cases: tuple[tuple[list[int]], ...],
        probe_outputs: Sequence[typed_output.TypedOutputEncoding],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        event_order: tuple[int, ...] = segmented.CURRICULUM_VALUES,
        schema_cell_gate: bool = True,
        schema_comparator_gate: bool = True,
        comparator_gate: bool = True,
        action_cell_gate: bool = True,
        cue_gate: bool = True,
        router_cell_gate: bool = True,
        equality_gate: bool = True,
        literal_role_gate: bool = True,
        order_gate: bool = True,
        upstream_cell_gate: bool = True,
        residual_gate: bool = True,
        residual_cell_gate: bool = True,
        predicate_gate: bool = True,
        effect_gate: bool = True,
        completeness_gate: bool = True,
    ) -> ContentAddressedSchemaFrontierV1:
        return ContentAddressedSchemaFrontierV1._build(
            self,
            span_records,
            literal_records,
            signature_role_record,
            probe_cases,
            probe_outputs,
            signature=signature,
            cases=cases,
            literals=literals,
            max_cost=max_cost,
            event_order=event_order,
            schema_cell_gate=schema_cell_gate,
            schema_comparator_gate=schema_comparator_gate,
            comparator_gate=comparator_gate,
            action_cell_gate=action_cell_gate,
            cue_gate=cue_gate,
            router_cell_gate=router_cell_gate,
            equality_gate=equality_gate,
            literal_role_gate=literal_role_gate,
            order_gate=order_gate,
            upstream_cell_gate=upstream_cell_gate,
            residual_gate=residual_gate,
            residual_cell_gate=residual_cell_gate,
            predicate_gate=predicate_gate,
            effect_gate=effect_gate,
            completeness_gate=completeness_gate,
        )


def fit_causal_content_addressed_segment_schema_v1(
    segmented_checkpoint: segmented.FrozenSegmentedAggregateV1,
    episodes: tuple[append.CausalAppendEffectEpisode, ...],
    *,
    signature: TypedSignature,
    probe_cases: tuple[tuple[list[int]], ...],
    literals: tuple[int, ...],
    max_cost: int,
) -> FrozenCausalContentAddressedSegmentSchemaV1:
    """Learn two schemas and require the third segment to reuse the first."""

    if type(segmented_checkpoint) is not segmented.FrozenSegmentedAggregateV1:
        raise TypeError("exact segmented aggregate checkpoint required")
    if (
        type(episodes) is not tuple
        or len(episodes) != segmented.TRAIN_SEGMENT_COUNT
        or any(type(item) is not append.CausalAppendEffectEpisode for item in episodes)
    ):
        raise TypeError("content-addressed fit requires exactly three episodes")
    segmented_checkpoint.authenticate()
    if segmented.TOTAL_CELL_COUNT != 69_208:
        raise RuntimeError("segmented aggregate cell identity changed")
    residual_checkpoint = segmented_checkpoint._append_checkpoint._residual_checkpoint
    residual_checkpoint.authenticate()
    training_frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
        signature, probe_cases, literals, max_cost
    )
    if training_frontier.status != breadth_v4.FRONTIER_READY:
        raise ContentAddressedSchemaFitStop(
            STOP_DEPENDENCY, "training reducer frontier is not ready"
        )

    prototype_excitatory = np.zeros(
        (SCHEMA_CAPACITY, ROUTES, RADIX), dtype=TICK_DTYPE
    )
    prototype_inhibitory = np.zeros_like(prototype_excitatory)
    action_excitatory = np.zeros(
        (SCHEMA_CAPACITY, append.REDUCER_ACTION_CELL_COUNT), dtype=TICK_DTYPE
    )
    action_inhibitory = np.zeros_like(action_excitatory)
    valid = np.zeros((SCHEMA_CAPACITY,), dtype=TICK_DTYPE)
    provenance_ids = np.full(
        (SCHEMA_CAPACITY,), -1, dtype=PROVENANCE_DTYPE
    )
    dependency_provenance = segmented_checkpoint._segmenter.internal_provenance_cell_ids
    if len(dependency_provenance) < SCHEMA_CAPACITY:
        raise ContentAddressedSchemaFitStop(
            STOP_DEPENDENCY, "segmented checkpoint has fewer than two provenance cells"
        )

    audits: list[SchemaFitAudit] = []
    assignments: list[int] = []
    effect_relations: list[int] = []
    prototype_e_ticks = 0
    prototype_h_ticks = 0
    action_e_ticks = 0
    action_h_ticks = 0
    residual_steps = 0
    comparison_steps = 0
    completeness_steps = 0
    effect_steps = 0
    projection_steps = 0
    packed_authorizations = 0
    schema_count = 0

    for ordinal, episode in enumerate(episodes):
        observation = append._observe(
            residual_checkpoint,
            episode.span_records,
            episode.literal_records,
            episode.signature_role_record,
            probe_cases,
            episode.probe_outputs,
            equality_gate=True,
            literal_role_gate=True,
            order_gate=True,
            upstream_cell_gate=True,
            residual_gate=True,
            residual_cell_gate=True,
            predicate_gate=True,
            effect_gate=True,
            completeness_gate=True,
        )
        if (
            observation.status != append.READY
            or observation.effect_relation is None
            or observation.residual_binding is None
        ):
            raise ContentAddressedSchemaFitStop(
                STOP_DEPENDENCY, observation.reason or observation.status
            )
        projection = segmented._project_effect_events(
            observation.effect_relation,
            segmented_checkpoint._relation_cells,
            segmented.CURRICULUM_VALUES,
        )
        targets, target_root = _route_targets(projection)
        decision = _race_schema(
            prototype_excitatory,
            prototype_inhibitory,
            valid,
            targets,
            allow_novel=True,
            comparator_gate=True,
            schema_cell_gate=True,
        )
        disposition: str
        if decision.status == "NOVEL":
            if schema_count == SCHEMA_CAPACITY:
                raise ContentAddressedSchemaFitStop(
                    STOP_SCHEMA_TONIC, "schema capacity exhausted"
                )
            schema_index = schema_count
            schema_count += 1
            valid[schema_index] = 1
            provenance_ids[schema_index] = dependency_provenance[schema_index]
            e_ticks, h_ticks = _learn_prototype(
                prototype_excitatory,
                prototype_inhibitory,
                schema_index,
                targets,
            )
            prototype_e_ticks += e_ticks
            prototype_h_ticks += h_ticks
            disposition = "NOVEL"
        elif decision.status == READY and decision.schema_index is not None:
            schema_index = decision.schema_index
            disposition = "REUSED"
        else:
            raise ContentAddressedSchemaFitStop(
                decision.status, "schema race did not yield novel or reusable state"
            )

        desired = append._probe_outputs(episode.probe_outputs)
        outcome_action, outcome_root, authorizations = _causal_outcome_action(
            training_frontier, observation.residual_binding, desired
        )
        packed_authorizations += authorizations
        if disposition == "NOVEL":
            e_ticks, h_ticks = _learn_action(
                action_excitatory,
                action_inhibitory,
                schema_index,
                outcome_action,
            )
            action_e_ticks += e_ticks
            action_h_ticks += h_ticks
        else:
            action_status, learned_action, _scores, _root = _race_action(
                action_excitatory,
                action_inhibitory,
                schema_index,
                cell_gate=True,
            )
            if action_status != READY or learned_action != outcome_action:
                raise ContentAddressedSchemaFitStop(
                    STOP_ACTION_CONSEQUENCE,
                    "reused schema did not preserve its causal reducer consequence",
                )

        r_steps, c_steps, complete_steps, e_steps = _observation_cell_steps(
            observation
        )
        residual_steps += r_steps
        comparison_steps += c_steps
        completeness_steps += complete_steps
        effect_steps += e_steps
        projection_steps += projection.effect_projection_cell_steps
        assignments.append(schema_index)
        effect_relations.append(observation.effect_relation)
        audits.append(
            SchemaFitAudit(
                ordinal,
                observation.effect_relation,
                schema_index,
                disposition,
                target_root,
                decision.sha256,
                outcome_action,
                outcome_root,
            )
        )

    if (
        schema_count != SCHEMA_CAPACITY
        or tuple(assignments) != (assignments[0], assignments[1], assignments[0])
        or assignments[0] == assignments[1]
        or tuple(item.disposition for item in audits) != ("NOVEL", "NOVEL", "REUSED")
        or len(set(effect_relations)) != SCHEMA_CAPACITY
        or effect_relations[0] != effect_relations[2]
    ):
        raise ContentAddressedSchemaFitStop(
            STOP_PROJECTION, "A/B/A-prime did not consolidate into exactly two schemas"
        )

    checkpoint_sha = _checkpoint_sha256(
        segmented_checkpoint.checkpoint_sha256,
        prototype_excitatory,
        prototype_inhibitory,
        action_excitatory,
        action_inhibitory,
        valid,
        provenance_ids,
    )
    summary = ContentAddressedSchemaTrainingSummary(
        len(episodes),
        segmented_checkpoint.checkpoint_sha256,
        schema_count,
        len(episodes),
        schema_count,
        len(episodes) - schema_count,
        tuple(assignments),
        tuple(effect_relations),
        tuple(int(value) for value in provenance_ids),
        prototype_e_ticks,
        prototype_h_ticks,
        action_e_ticks,
        action_h_ticks,
        residual_steps,
        comparison_steps,
        completeness_steps,
        effect_steps,
        projection_steps,
        len(episodes) * SCHEMA_RACE_CELL_STEPS,
        packed_authorizations,
        len(episodes),
        DIRECT_APPEND_ACTION_READS,
        NEW_PERSISTENT_STATE_BYTES,
        tuple(audits),
        "ENABLED",
        checkpoint_sha,
    )
    return FrozenCausalContentAddressedSegmentSchemaV1(
        segmented_checkpoint,
        prototype_excitatory,
        prototype_inhibitory,
        action_excitatory,
        action_inhibitory,
        valid,
        provenance_ids,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _controlled_checkpoint(
    checkpoint: FrozenCausalContentAddressedSegmentSchemaV1,
    *,
    prototype_off: bool,
    derange_actions: bool,
    control: str,
) -> FrozenCausalContentAddressedSegmentSchemaV1:
    checkpoint.authenticate()
    prototype_excitatory = np.array(
        checkpoint._prototype_excitatory, copy=True, order="C"
    )
    prototype_inhibitory = np.array(
        checkpoint._prototype_inhibitory, copy=True, order="C"
    )
    action_excitatory = np.array(
        checkpoint._action_excitatory, copy=True, order="C"
    )
    action_inhibitory = np.array(
        checkpoint._action_inhibitory, copy=True, order="C"
    )
    valid = np.array(checkpoint._valid, copy=True, order="C")
    provenance_ids = np.array(checkpoint._provenance_ids, copy=True, order="C")
    persistent_count = checkpoint.training_summary.persistent_schema_count
    if prototype_off:
        prototype_excitatory.fill(0)
        prototype_inhibitory.fill(0)
        valid.fill(0)
        provenance_ids.fill(-1)
        persistent_count = 0
    if derange_actions:
        action_excitatory = action_excitatory[:, ::-1].copy(order="C")
        action_inhibitory = action_inhibitory[:, ::-1].copy(order="C")
    checkpoint_sha = _checkpoint_sha256(
        checkpoint._segmented_checkpoint.checkpoint_sha256,
        prototype_excitatory,
        prototype_inhibitory,
        action_excitatory,
        action_inhibitory,
        valid,
        provenance_ids,
    )
    summary = checkpoint.training_summary._replace(
        persistent_schema_count=persistent_count,
        control=control,
        checkpoint_sha256=checkpoint_sha,
    )
    return FrozenCausalContentAddressedSegmentSchemaV1(
        checkpoint._segmented_checkpoint,
        prototype_excitatory,
        prototype_inhibitory,
        action_excitatory,
        action_inhibitory,
        valid,
        provenance_ids,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def freeze_prototype_off_control(
    checkpoint: FrozenCausalContentAddressedSegmentSchemaV1,
) -> FrozenCausalContentAddressedSegmentSchemaV1:
    """Return the exact checkpoint with learned schema prototypes silenced."""

    return _controlled_checkpoint(
        checkpoint,
        prototype_off=True,
        derange_actions=False,
        control="PROTOTYPE_OFF",
    )


def freeze_deranged_action_control(
    checkpoint: FrozenCausalContentAddressedSegmentSchemaV1,
) -> FrozenCausalContentAddressedSegmentSchemaV1:
    """Swap the learned schema-to-reducer columns without changing prototypes."""

    return _controlled_checkpoint(
        checkpoint,
        prototype_off=False,
        derange_actions=True,
        control="ACTION_DERANGED",
    )


def _work_payload(value: ContentAddressedSchemaWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _state_payload(value: ContentAddressedSchemaFrontierState) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field not in {"work", "sha256"}
    } | {"work_sha256": value.work.sha256}


class ContentAddressedSchemaFrontierV1:
    """Desired-free frontier whose reducer is authorized by a schema cell."""

    __slots__ = (
        "_checkpoint",
        "_delegate",
        "_end_segmenter",
        "_observation",
        "_probe_frontier",
        "_state",
    )

    def __init__(self) -> None:
        raise TypeError("use FrozenCausalContentAddressedSegmentSchemaV1.build")

    @classmethod
    def _build(
        cls,
        checkpoint: FrozenCausalContentAddressedSegmentSchemaV1,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        probe_cases: tuple[tuple[list[int]], ...],
        probe_outputs: Sequence[typed_output.TypedOutputEncoding],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        event_order: tuple[int, ...],
        schema_cell_gate: bool,
        schema_comparator_gate: bool,
        comparator_gate: bool,
        action_cell_gate: bool,
        cue_gate: bool,
        router_cell_gate: bool,
        equality_gate: bool,
        literal_role_gate: bool,
        order_gate: bool,
        upstream_cell_gate: bool,
        residual_gate: bool,
        residual_cell_gate: bool,
        predicate_gate: bool,
        effect_gate: bool,
        completeness_gate: bool,
    ) -> ContentAddressedSchemaFrontierV1:
        checkpoint.authenticate()
        residual_checkpoint = (
            checkpoint._segmented_checkpoint._append_checkpoint._residual_checkpoint
        )
        observation = append._observe(
            residual_checkpoint,
            span_records,
            literal_records,
            signature_role_record,
            probe_cases,
            probe_outputs,
            equality_gate=equality_gate,
            literal_role_gate=literal_role_gate,
            order_gate=order_gate,
            upstream_cell_gate=upstream_cell_gate,
            residual_gate=residual_gate,
            residual_cell_gate=residual_cell_gate,
            predicate_gate=predicate_gate,
            effect_gate=effect_gate,
            completeness_gate=completeness_gate,
        )
        dependency = checkpoint._segmented_checkpoint
        start = dependency._segmenter
        current = start
        status = observation.status
        projection: segmented._ProjectedEffectEvents | None = None
        transitions: list[segmenter.PredictiveSegmentationTransition] = []
        router_actions: list[int] = []
        router_roots: list[str] = []
        schema_decision: _SchemaDecision | None = None
        schema_action: int | None = None
        action_scores: tuple[int, ...] = ()
        action_root: str | None = None
        outcome_action: int | None = None
        outcome_root: str | None = None
        packed_authorizations = 0
        probe_frontier: breadth_v4.PackedGenericTypedTrainBreadthV4 | None = None
        delegate: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None = None
        scaffold_sha: str | None = None
        frozen_literals = tuple(literals)
        frozen_cases = segmented._normalize_final_cases(cases)

        if (
            status == append.READY
            and observation.effect_relation is not None
            and observation.residual_binding is not None
        ):
            projection = segmented._project_effect_events(
                observation.effect_relation,
                dependency._relation_cells,
                event_order,
            )
            for event in projection.events:
                transition = segmenter.advance_predictive_segmenter_v1(
                    current,
                    event,
                    cue_gate=cue_gate,
                    comparator_gate=comparator_gate,
                )
                transitions.append(transition)
                if not transition.accepted_event:
                    status = transition.status
                    break
                relation = segmented._decision_relation(transition)
                router_status, action, router_root = segmented._race_router(
                    dependency._excitatory,
                    dependency._inhibitory,
                    relation,
                    cell_gate=router_cell_gate,
                )
                router_roots.append(router_root)
                if router_status != segmented.READY or action is None:
                    status = router_status
                    break
                required = segmented._required_action(transition)
                if action != required:
                    status = segmented.STOP_ROUTER_CONSEQUENCE
                    break
                router_actions.append(action)
                current = transition.successor

            if status == append.READY:
                if (
                    len(transitions) != segmented.EVENTS_PER_SEGMENT
                    or not transitions[-1].segment_completed
                    or current.segment_count != start.segment_count + 1
                ):
                    status = STOP_SEGMENT
                else:
                    targets, _target_root = _route_targets(projection)
                    schema_decision = _race_schema(
                        checkpoint._prototype_excitatory,
                        checkpoint._prototype_inhibitory,
                        checkpoint._valid,
                        targets,
                        allow_novel=False,
                        comparator_gate=schema_comparator_gate,
                        schema_cell_gate=schema_cell_gate,
                    )
                    status = schema_decision.status

            if (
                status == READY
                and schema_decision is not None
                and schema_decision.schema_index is not None
            ):
                action_status, schema_action, action_scores, action_root = _race_action(
                    checkpoint._action_excitatory,
                    checkpoint._action_inhibitory,
                    schema_decision.schema_index,
                    cell_gate=action_cell_gate,
                )
                status = action_status

            if status == READY and schema_action is not None:
                probe_frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
                    signature, probe_cases, frozen_literals, max_cost
                )
                if probe_frontier.status != breadth_v4.FRONTIER_READY:
                    status = STOP_ACTION_CONSEQUENCE
                else:
                    desired_probe = append._probe_outputs(probe_outputs)
                    try:
                        outcome_action, outcome_root, packed_authorizations = (
                            _causal_outcome_action(
                                probe_frontier,
                                observation.residual_binding,
                                desired_probe,
                            )
                        )
                    except ContentAddressedSchemaFitStop:
                        status = STOP_ACTION_CONSEQUENCE
                    else:
                        if outcome_action != schema_action:
                            status = STOP_ACTION_CONSEQUENCE

            if status == READY and schema_action is not None:
                probe_keys = {tuple(row[0]) for row in probe_cases}
                if any(
                    key is not None and key in probe_keys
                    for key in map(segmented._list_case_key, frozen_cases)
                ):
                    status = STOP_PROBE_FINAL_OVERLAP
                else:
                    scaffold = append._reducer_scaffold(
                        observation.residual_binding, schema_action
                    )
                    scaffold_sha = scaffold.sha256
                    delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
                        signature,
                        frozen_cases,
                        frozen_literals,
                        max_cost,
                        scaffold,
                    )
                    status = delegate.status

        (
            residual_steps,
            comparison_steps,
            completeness_steps,
            effect_steps,
        ) = _observation_cell_steps(observation)
        projection_steps = (
            0 if projection is None else projection.effect_projection_cell_steps
        )
        segment_steps = sum(item.physical_cell_steps for item in transitions)
        router_steps = len(router_roots) * segmented.ROUTER_RACE_CELL_STEPS
        schema_steps = 0 if schema_decision is None else SCHEMA_RACE_CELL_STEPS
        action_steps = 0 if action_root is None else ACTION_RACE_CELL_STEPS
        total_steps = (
            residual_steps
            + comparison_steps
            + completeness_steps
            + effect_steps
            + projection_steps
            + segment_steps
            + router_steps
            + schema_steps
            + action_steps
        )
        partial_work = ContentAddressedSchemaWork(
            residual_steps,
            comparison_steps,
            completeness_steps,
            effect_steps,
            0 if projection is None else projection.input_projection_work,
            projection_steps,
            segment_steps,
            router_steps,
            schema_steps,
            action_steps,
            packed_authorizations,
            0 if outcome_root is None else 1,
            total_steps,
            None if projection is None else projection.sha256,
            tuple(item.sha256 for item in transitions),
            tuple(router_roots),
            None if schema_decision is None else schema_decision.sha256,
            action_root,
            outcome_root,
            "",
        )
        work = partial_work._replace(
            sha256=_digest(_WORK_NAMESPACE, _work_payload(partial_work))
        )
        partial_state = ContentAddressedSchemaFrontierState(
            status,
            checkpoint.checkpoint_sha256,
            observation.residual_certificate.sha256,
            observation.probe_sha256,
            start.checkpoint_sha256,
            current.checkpoint_sha256,
            observation.effect_relation,
            None if schema_decision is None else schema_decision.schema_index,
            None if schema_decision is None else schema_decision.schema_cell_id,
            () if schema_decision is None else schema_decision.raw_scores,
            schema_action,
            action_scores,
            outcome_action,
            tuple(router_actions),
            scaffold_sha,
            None if probe_frontier is None else probe_frontier.sha256,
            None if delegate is None else delegate.sha256,
            0 if probe_frontier is None else 1,
            0 if delegate is None else 1,
            DESIRED_READS_DURING_BUILD,
            DIRECT_APPEND_ACTION_READS,
            checkpoint.training_summary.persistent_schema_count,
            work,
            "",
        )
        state = partial_state._replace(
            sha256=_digest(_STATE_NAMESPACE, _state_payload(partial_state))
        )
        instance = object.__new__(cls)
        instance._checkpoint = checkpoint
        instance._observation = observation
        instance._end_segmenter = current
        instance._probe_frontier = probe_frontier
        instance._delegate = delegate
        instance._state = state
        instance._authenticate()
        return instance

    @property
    def state(self) -> ContentAddressedSchemaFrontierState:
        self._authenticate()
        return self._state

    @property
    def status(self) -> str:
        self._authenticate()
        return self._state.status

    @property
    def sha256(self) -> str:
        self._authenticate()
        return self._state.sha256

    def _authenticate(self) -> None:
        self._checkpoint.authenticate()
        residual._authenticate_binding(self._observation.residual_certificate)
        self._end_segmenter.authenticate()
        if (
            self._state.checkpoint_sha256 != self._checkpoint.checkpoint_sha256
            or self._state.residual_certificate_sha256
            != self._observation.residual_certificate.sha256
            or self._state.segmenter_end_sha256
            != self._end_segmenter.checkpoint_sha256
            or self._state.probe_frontier_sha256
            != (
                None
                if self._probe_frontier is None
                else self._probe_frontier.sha256
            )
            or self._state.delegated_frontier_sha256
            != (None if self._delegate is None else self._delegate.sha256)
            or self._state.probe_packed_build_count
            != (0 if self._probe_frontier is None else 1)
            or self._state.final_packed_build_count
            != (0 if self._delegate is None else 1)
            or self._state.desired_reads_during_build != 0
            or self._state.direct_append_action_reads != 0
            or self._state.work.sha256
            != _digest(_WORK_NAMESPACE, _work_payload(self._state.work))
            or self._state.sha256
            != _digest(_STATE_NAMESPACE, _state_payload(self._state))
        ):
            raise RuntimeError("content-addressed schema frontier authentication failed")

    def select(self, desired: Sequence[object]) -> ContentAddressedSchemaSelection:
        self._authenticate()
        delegated = None if self._delegate is None else self._delegate.select(desired)
        status = self._state.status if delegated is None else delegated.status
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "delegated_sha256": None if delegated is None else delegated.sha256,
        }
        return ContentAddressedSchemaSelection(
            status,
            self._state.sha256,
            delegated,
            _digest(_SELECTION_NAMESPACE, payload),
        )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "segmented_config_sha256": segmented.CONFIG_SHA256,
        "schema_capacity": SCHEMA_CAPACITY,
        "routes": ROUTES,
        "radix": RADIX,
        "new_cells": NEW_CELL_COUNT,
        "total_cells": TOTAL_CELL_COUNT,
        "prototype_state_bytes": PROTOTYPE_STATE_BYTES,
        "action_state_bytes": ACTION_STATE_BYTES,
        "valid_and_provenance_state_bytes": VALID_AND_PROVENANCE_STATE_BYTES,
        "new_persistent_state_bytes": NEW_PERSISTENT_STATE_BYTES,
        "schema_race_cell_steps": SCHEMA_RACE_CELL_STEPS,
        "action_race_cell_steps": ACTION_RACE_CELL_STEPS,
        "direct_append_action_reads": DIRECT_APPEND_ACTION_READS,
        "desired_reads_during_build": DESIRED_READS_DURING_BUILD,
        "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
        "canonical_panel_spend_count": CANONICAL_PANEL_SPEND_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "path2_input_count": PATH2_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
    }
)
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", dict(CONFIG_RECORD))

__all__ = [
    "ACTION_STATE_BYTES",
    "C1_INPUT_COUNT",
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESIRED_READS_DURING_BUILD",
    "DEV_INPUT_COUNT",
    "DIRECT_APPEND_ACTION_READS",
    "G0_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "NEW_PERSISTENT_STATE_BYTES",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PROTOTYPE_STATE_BYTES",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "READY",
    "SCHEMA_CAPACITY",
    "STOP_ACTION_CONSEQUENCE",
    "STOP_SCHEMA_TONIC",
    "TOTAL_CELL_COUNT",
    "ContentAddressedSchemaFitStop",
    "ContentAddressedSchemaFrontierState",
    "ContentAddressedSchemaFrontierV1",
    "ContentAddressedSchemaSelection",
    "ContentAddressedSchemaTrainingSummary",
    "ContentAddressedSchemaWork",
    "FrozenCausalContentAddressedSegmentSchemaV1",
    "fit_causal_content_addressed_segment_schema_v1",
    "freeze_deranged_action_control",
    "freeze_prototype_off_control",
]
