"""Compose learned aggregate semantics with learned predictive segmentation.

The append-effect checkpoint already owns literal-role, operand-order,
residual-literal, and reducer authority.  This module projects its two anonymous
effect cells into the banked one-byte identity/shift geometry, lets the banked
predictive segmenter discover request boundaries, and learns only the missing
two-way routing seam: NEW/CONTINUE to OPEN/EXTEND.

The three visible probe outputs are task evidence.  Final desired outputs are
absent from :meth:`FrozenSegmentedAggregateV1.build` and appear only in
``select`` on the returned frontier.
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
from experiments.g0 import common_event_cells_v2 as input_projection
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter
from experiments.r3_pilot import public_typed_output_cells as typed_output
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.resting_glif_race import (
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1"
READY = "CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_READY"
STOP_DEPENDENCY = "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_DEPENDENCY"
STOP_PROJECTION = "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_PROJECTION"
STOP_SEGMENT = "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_SEGMENT"
STOP_ROUTER_ZERO = "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_ROUTER_ZERO"
STOP_ROUTER_AMBIGUOUS = (
    "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_ROUTER_AMBIGUOUS"
)
STOP_ROUTER_CONSEQUENCE = (
    "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_ROUTER_CONSEQUENCE"
)
STOP_ROUTER_INCOMPLETE = (
    "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_ROUTER_INCOMPLETE"
)
STOP_PROBE_FINAL_OVERLAP = (
    "STOP_CAUSAL_SEGMENTED_AGGREGATE_STREAM_V1_PROBE_FINAL_OVERLAP"
)

CLAIM_BOUNDARY = (
    "one self-posed List[Int]-to-Int stream composition: banked anonymous "
    "literal role, operand order, residual route, append-effect reducer, and "
    "predictive segmentation feed one learned two-cell OPEN/EXTEND router and "
    "one desired-free packed frontier; prompt/probe grouping, balanced byte "
    "framing, token timing, predicate/addition/reducer scaffolds, typed decoding, "
    "and compilation remain explicit, so general parsing, arbitrary algorithms, "
    "public transfer, and coding capability remain unproven"
)

CONTINUE_RELATION = segmenter.CONTINUE_DECISION
NEW_RELATION = segmenter.NEW_DECISION
OPEN_ACTION = 0
EXTEND_ACTION = 1
ROUTER_ACTION_CELL_COUNT = 2
ROUTER_ACTION_CELL_OFFSET = append.TOTAL_CELL_COUNT + segmenter.TOTAL_CELL_COUNT
OPEN_ACTION_CELL = ROUTER_ACTION_CELL_OFFSET + OPEN_ACTION
EXTEND_ACTION_CELL = ROUTER_ACTION_CELL_OFFSET + EXTEND_ACTION
NEW_CELL_COUNT = ROUTER_ACTION_CELL_COUNT
TOTAL_CELL_COUNT = (
    append.TOTAL_CELL_COUNT + segmenter.TOTAL_CELL_COUNT + NEW_CELL_COUNT
)

TICK_DTYPE = np.dtype(np.uint8)
TICK_CAP = 4
TRAINABLE_STATE_BYTES = (
    2
    * segmenter.SEGMENT_DECISION_COUNT
    * ROUTER_ACTION_CELL_COUNT
    * TICK_DTYPE.itemsize
)
ROUTER_RACE_STEPS = 30
ROUTER_RACE_CELL_STEPS = ROUTER_ACTION_CELL_COUNT * ROUTER_RACE_STEPS
OUTPUT_CURRENT_PA = 10_000.0
_ROUTER_RACE = CompiledRestingGlifRace.compile(step_count=ROUTER_RACE_STEPS)

CURRICULUM_VALUES = (0x1E, 0x63, 0xB4, 0xC9, 0x27, 0x78, 0x8D, 0xD2)
EVENTS_PER_SEGMENT = len(CURRICULUM_VALUES)
TRAIN_SEGMENT_COUNT = 3
DESIRED_READS_DURING_BUILD = 0

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"causal-segmented-aggregate-stream-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_WORK_NAMESPACE = _NAMESPACE + b"work\0"
_STATE_NAMESPACE = _NAMESPACE + b"state\0"
_SELECTION_NAMESPACE = _NAMESPACE + b"selection\0"
_INPUT_CELLS = input_projection.CommonOneByteFillerCells()
_OUTPUT_CELLS = output_projection.CommonEventCells()
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


def _digits(value: int) -> tuple[int, ...]:
    return tuple(
        (
            value
            // (
                input_projection.FILLER_RADIX
                ** (input_projection.FILLER_LANES - lane - 1)
            )
        )
        % input_projection.FILLER_RADIX
        for lane in range(input_projection.FILLER_LANES)
    )


def _shift_fillers(value: int) -> int:
    shifted = 0
    for digit in _digits(value):
        shifted = (
            shifted * input_projection.FILLER_RADIX
            + (digit + 1) % input_projection.FILLER_RADIX
        )
    return shifted


def _teacher_decision(value: int) -> output_projection.CommonOutputDecision:
    digits = _digits(value)
    decision = _OUTPUT_CELLS.race_output(
        tuple(
            tuple(24 if candidate == digit else 0 for candidate in range(4))
            for digit in digits
        )
    )
    if decision.decision != "TOKEN" or decision.token != value:
        raise RuntimeError("balanced effect projection did not emit its physical byte")
    return decision


class _ProjectedEffectEvents(NamedTuple):
    events: tuple[segmenter.PhysicalTimedRelationEvent, ...]
    input_projection_work: int
    effect_projection_cell_steps: int
    frame_sha256s: tuple[str, ...]
    teacher_sha256s: tuple[str, ...]
    sha256: str


def _project_effect_events(
    effect_relation: int,
    relation_cells: tuple[int, ...],
    event_order: tuple[int, ...],
) -> _ProjectedEffectEvents:
    if effect_relation not in (
        append.PREDICATE_EFFECT_RELATION,
        append.PAYLOAD_EFFECT_RELATION,
    ):
        raise ValueError("effect relation left the banked two-cell capacity")
    if type(event_order) is not tuple or sorted(event_order) != sorted(
        CURRICULUM_VALUES
    ):
        raise ValueError("event order must be one exact curriculum permutation")
    frames = tuple(
        _INPUT_CELLS.project_one_byte(bytes((value,))) for value in event_order
    )
    teacher_values = tuple(
        value
        if effect_relation == append.PREDICATE_EFFECT_RELATION
        else _shift_fillers(value)
        for value in event_order
    )
    teachers = tuple(_teacher_decision(value) for value in teacher_values)
    opening, within = segmenter.INTERVAL_CELL_IDS
    events = tuple(
        segmenter.PhysicalTimedRelationEvent(
            input_frame=frame,
            relation_cells=relation_cells,
            teacher_output_cells=teacher.active_cells,
            interval_cell_id=opening if ordinal == 0 else within,
        )
        for ordinal, (frame, teacher) in enumerate(
            zip(frames, teachers, strict=True)
        )
    )
    payload = {
        "effect_relation_cell": append.EFFECT_RELATION_CELL_OFFSET
        + effect_relation,
        "event_order": event_order,
        "frame_sha256s": tuple(frame.sha256 for frame in frames),
        "teacher_sha256s": tuple(teacher.sha256 for teacher in teachers),
        "interval_cell_ids": tuple(event.interval_cell_id for event in events),
    }
    return _ProjectedEffectEvents(
        events,
        sum(frame.projection_work_total for frame in frames),
        sum(teacher.physical_cell_steps for teacher in teachers),
        tuple(frame.sha256 for frame in frames),
        tuple(teacher.sha256 for teacher in teachers),
        _digest(_NAMESPACE + b"effect-events\0", payload),
    )


class RouterInterventionAudit(NamedTuple):
    decision_relation: int
    pass_action: int
    transition_sha256: str
    outcome_race_sha256: str
    outcome_race_cell_steps: int


class SegmentedAggregateTrainingSummary(NamedTuple):
    episode_count: int
    append_checkpoint_sha256: str
    segmenter_start_sha256: str
    segmenter_checkpoint_sha256: str
    append_binding_cell_steps: int
    input_projection_work: int
    effect_projection_cell_steps: int
    segmenter_cell_steps: int
    router_outcome_race_count: int
    router_outcome_race_cell_steps: int
    excitatory_update_ticks: int
    inhibitory_update_ticks: int
    trainable_state_bytes: int
    effect_relations: tuple[int, ...]
    projection_sha256s: tuple[str, ...]
    transition_sha256s: tuple[str, ...]
    router_audits: tuple[RouterInterventionAudit, ...]
    checkpoint_sha256: str


class SegmentedAggregateWork(NamedTuple):
    append_binding_cell_steps: int
    input_projection_work: int
    effect_projection_cell_steps: int
    segmenter_cell_steps: int
    router_race_cell_steps: int
    total_physical_cell_steps: int
    projection_sha256: str | None
    transition_sha256s: tuple[str, ...]
    router_race_sha256s: tuple[str, ...]
    append_work_sha256: str
    sha256: str


class SegmentedAggregateFrontierState(NamedTuple):
    status: str
    checkpoint_sha256: str
    append_certificate_sha256: str
    segmenter_start_sha256: str
    segmenter_end_sha256: str
    effect_relation: int | None
    selected_action: int | None
    router_actions: tuple[int, ...]
    scaffold_sha256: str | None
    delegated_frontier_sha256: str | None
    packed_build_count: int
    desired_reads_during_build: int
    work: SegmentedAggregateWork
    sha256: str


class SegmentedAggregateSelection(NamedTuple):
    status: str
    frontier_sha256: str
    delegated: plan_slice.PackedPromptPlanSliceSelection | None
    sha256: str


class SegmentedAggregateFitStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _decision_relation(
    transition: segmenter.PredictiveSegmentationTransition,
) -> int:
    if transition.decision == "CONTINUE":
        return CONTINUE_RELATION
    if transition.decision == "NEW":
        return NEW_RELATION
    raise SegmentedAggregateFitStop(
        STOP_SEGMENT, f"segmenter returned {transition.decision}"
    )


def _required_action(
    transition: segmenter.PredictiveSegmentationTransition,
) -> int:
    if not transition.accepted_event:
        raise SegmentedAggregateFitStop(
            STOP_SEGMENT, f"segmenter stopped at {transition.status}"
        )
    return OPEN_ACTION if transition.segment_started else EXTEND_ACTION


def _physical_router_outcome(
    transition: segmenter.PredictiveSegmentationTransition,
) -> RouterInterventionAudit:
    relation = _decision_relation(transition)
    pass_action = _required_action(transition)
    currents = np.asarray(
        [OUTPUT_CURRENT_PA if action == pass_action else 0.0 for action in range(2)],
        dtype=np.float64,
    )
    race = _ROUTER_RACE.race_many(currents[None, :], require_unique=False)[0]
    if race.simultaneous_spike_candidates != 1 or race.winner_cell != pass_action:
        raise SegmentedAggregateFitStop(
            STOP_ROUTER_CONSEQUENCE,
            "OPEN/EXTEND interventions did not produce one physical PASS",
        )
    payload = {
        "decision_relation": relation,
        "pass_action": pass_action,
        "transition_sha256": transition.sha256,
        "winner_cell": race.winner_cell,
        "winner_spike_step": race.spike_step,
        "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
        "cell_steps": ROUTER_RACE_CELL_STEPS,
    }
    return RouterInterventionAudit(
        relation,
        pass_action,
        transition.sha256,
        _digest(_NAMESPACE + b"router-outcome\0", payload),
        ROUTER_RACE_CELL_STEPS,
    )


def _learn_router_policy(
    audits: Sequence[RouterInterventionAudit],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    labels: list[list[int]] = [[], []]
    for audit in audits:
        labels[audit.decision_relation].append(audit.pass_action)
    if any(not values for values in labels):
        raise SegmentedAggregateFitStop(
            STOP_ROUTER_INCOMPLETE, "both segment decisions need causal support"
        )
    if any(any(value != values[0] for value in values) for values in labels):
        raise SegmentedAggregateFitStop(
            STOP_ROUTER_CONSEQUENCE, "one segment decision has conflicting actions"
        )
    if {values[0] for values in labels} != {OPEN_ACTION, EXTEND_ACTION}:
        raise SegmentedAggregateFitStop(
            STOP_ROUTER_INCOMPLETE, "both router actions need causal support"
        )
    excitatory = np.zeros((2, 2), dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    excitatory_ticks = 0
    inhibitory_ticks = 0
    for relation, values in enumerate(labels):
        pass_action = values[0]
        for action in range(2):
            target = excitatory if action == pass_action else inhibitory
            target[relation, action] = 1
            if action == pass_action:
                excitatory_ticks += 1
            else:
                inhibitory_ticks += 1
    return excitatory, inhibitory, excitatory_ticks, inhibitory_ticks


def _checkpoint_sha256(
    append_sha256: str,
    segmenter_sha256: str,
    relation_cells: tuple[int, ...],
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + append_sha256.encode("ascii")
        + b"\0"
        + segmenter_sha256.encode("ascii")
        + b"\0"
        + _canonical_json(relation_cells)
        + b"\0"
        + _array_frame("router-excitatory", excitatory)
        + _array_frame("router-inhibitory", inhibitory)
    ).hexdigest()


def _race_router(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    relation: int,
    *,
    cell_gate: bool,
) -> tuple[str, int | None, str]:
    scores = excitatory[relation].astype(np.int16) - inhibitory[relation].astype(
        np.int16
    )
    currents = np.asarray(
        [
            max(0, int(score)) * OUTPUT_CURRENT_PA if cell_gate else 0.0
            for score in scores
        ],
        dtype=np.float64,
    )
    try:
        race = _ROUTER_RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure:
        status = STOP_ROUTER_ZERO
        action = None
        spike_step = None
        simultaneous_spike_candidates = 0
    else:
        spike_step = race.spike_step
        simultaneous_spike_candidates = race.simultaneous_spike_candidates
        if simultaneous_spike_candidates != 1:
            status = STOP_ROUTER_AMBIGUOUS
            action = None
        else:
            status = READY
            action = int(race.winner_cell)
    payload = {
        "relation": relation,
        "scores": tuple(int(value) for value in scores),
        "cell_gate": cell_gate,
        "status": status,
        "action": action,
        "spike_step": spike_step,
        "simultaneous_spike_candidates": simultaneous_spike_candidates,
        "cell_steps": ROUTER_RACE_CELL_STEPS,
    }
    return status, action, _digest(_NAMESPACE + b"router-race\0", payload)


class FrozenSegmentedAggregateV1:
    """Immutable composite checkpoint with one deep desired-free build seam."""

    __slots__ = (
        "_append_checkpoint",
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_relation_cells",
        "_segmenter",
        "_summary",
    )

    def __init__(
        self,
        append_checkpoint: append.FrozenCausalAppendEffectReducerV1,
        frozen_segmenter: segmenter.FrozenPredictiveSegmenterV1,
        relation_cells: tuple[int, ...],
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        summary: SegmentedAggregateTrainingSummary,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("segmented aggregate checkpoints are builder-owned")
        if type(append_checkpoint) is not append.FrozenCausalAppendEffectReducerV1:
            raise TypeError("exact banked append-effect checkpoint required")
        if type(frozen_segmenter) is not segmenter.FrozenPredictiveSegmenterV1:
            raise TypeError("exact banked predictive segmenter required")
        if (
            excitatory.shape != (2, 2)
            or inhibitory.shape != (2, 2)
            or excitatory.dtype != TICK_DTYPE
            or inhibitory.dtype != TICK_DTYPE
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("router state geometry changed")
        append_checkpoint.authenticate()
        frozen_segmenter.authenticate()
        self._append_checkpoint = append_checkpoint
        self._segmenter = frozen_segmenter
        self._relation_cells = tuple(relation_cells)
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._checkpoint_sha256 = _checkpoint_sha256(
            append_checkpoint.checkpoint_sha256,
            frozen_segmenter.checkpoint_sha256,
            self._relation_cells,
            self._excitatory,
            self._inhibitory,
        )
        self._summary = summary
        if summary.checkpoint_sha256 != self._checkpoint_sha256:
            raise RuntimeError("segmented aggregate training summary changed")
        self.authenticate()

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> SegmentedAggregateTrainingSummary:
        self.authenticate()
        return self._summary

    @property
    def segmenter_checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._segmenter.checkpoint_sha256

    @property
    def learned_state_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._excitatory.shape, self._inhibitory.shape

    def authenticate(self) -> None:
        self._append_checkpoint.authenticate()
        self._segmenter.authenticate()
        if (
            self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or self._checkpoint_sha256
            != _checkpoint_sha256(
                self._append_checkpoint.checkpoint_sha256,
                self._segmenter.checkpoint_sha256,
                self._relation_cells,
                self._excitatory,
                self._inhibitory,
            )
        ):
            raise RuntimeError("segmented aggregate checkpoint authentication failed")

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
        event_order: tuple[int, ...] = CURRICULUM_VALUES,
        cue_gate: bool = True,
        comparator_gate: bool = True,
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
        append_cell_gate: bool = True,
    ) -> SegmentedAggregateFrontierV1:
        """Build one target frontier without reading its final desired outputs."""

        self.authenticate()
        certificate = self._append_checkpoint.bind(
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
            cell_gate=append_cell_gate,
        )
        return SegmentedAggregateFrontierV1._build(
            self,
            certificate,
            signature=signature,
            cases=cases,
            literals=literals,
            max_cost=max_cost,
            event_order=event_order,
            cue_gate=cue_gate,
            comparator_gate=comparator_gate,
            router_cell_gate=router_cell_gate,
        )


def fit_causal_segmented_aggregate_v1(
    append_checkpoint: append.FrozenCausalAppendEffectReducerV1,
    initial_segmenter: segmenter.FrozenPredictiveSegmenterV1,
    episodes: tuple[append.CausalAppendEffectEpisode, ...],
    *,
    probe_cases: tuple[tuple[list[int]], ...],
    relation_cells: tuple[int, ...],
) -> FrozenSegmentedAggregateV1:
    """Fit the two-cell router from three ordinary effect segments."""

    if type(append_checkpoint) is not append.FrozenCausalAppendEffectReducerV1:
        raise TypeError("exact banked append-effect checkpoint required")
    if type(initial_segmenter) is not segmenter.FrozenPredictiveSegmenterV1:
        raise TypeError("exact banked predictive segmenter required")
    if (
        type(episodes) is not tuple
        or len(episodes) != TRAIN_SEGMENT_COUNT
        or any(type(item) is not append.CausalAppendEffectEpisode for item in episodes)
    ):
        raise TypeError("segmented aggregate fit requires exactly three episodes")
    append_checkpoint.authenticate()
    initial_segmenter.authenticate()
    if initial_segmenter.segment_count != 0:
        raise ValueError("segmented aggregate fit requires an unused segmenter")

    current = initial_segmenter
    certificates: list[append.AppendEffectReducerCertificate] = []
    projections: list[_ProjectedEffectEvents] = []
    transitions: list[segmenter.PredictiveSegmentationTransition] = []
    audit_by_relation: dict[int, RouterInterventionAudit] = {}
    effect_relations: list[int] = []
    for episode in episodes:
        certificate = append_checkpoint.bind(
            episode.span_records,
            episode.literal_records,
            episode.signature_role_record,
            probe_cases,
            episode.probe_outputs,
        )
        if certificate.status != append.READY:
            raise SegmentedAggregateFitStop(
                STOP_DEPENDENCY, certificate.stop_reason or certificate.status
            )
        binding = append.require_append_effect_reducer_binding(certificate)
        projection = _project_effect_events(
            binding.effect_relation, relation_cells, CURRICULUM_VALUES
        )
        episode_transitions: list[segmenter.PredictiveSegmentationTransition] = []
        for event in projection.events:
            transition = segmenter.advance_predictive_segmenter_v1(current, event)
            if not transition.accepted_event:
                raise SegmentedAggregateFitStop(
                    STOP_SEGMENT, f"balanced projection stopped at {transition.status}"
                )
            current = transition.successor
            episode_transitions.append(transition)
            if transition.decision_authority not in {"CALIBRATION", "STREAM_START"}:
                relation = _decision_relation(transition)
                audit = _physical_router_outcome(transition)
                prior = audit_by_relation.get(relation)
                if prior is not None and prior.pass_action != audit.pass_action:
                    raise SegmentedAggregateFitStop(
                        STOP_ROUTER_CONSEQUENCE,
                        "one physical decision produced conflicting router consequences",
                    )
                audit_by_relation.setdefault(relation, audit)
        if not episode_transitions[-1].segment_completed:
            raise SegmentedAggregateFitStop(
                STOP_PROJECTION,
                "fixed balanced projection did not complete in eight events",
            )
        certificates.append(certificate)
        projections.append(projection)
        transitions.extend(episode_transitions)
        effect_relations.append(binding.effect_relation)

    if current.segment_count != TRAIN_SEGMENT_COUNT or not current.complete_hold:
        raise SegmentedAggregateFitStop(
            STOP_SEGMENT, "three physical episodes did not freeze three segments"
        )
    audits = tuple(audit_by_relation[index] for index in range(2))
    excitatory, inhibitory, excitatory_ticks, inhibitory_ticks = (
        _learn_router_policy(audits)
    )
    checkpoint_sha = _checkpoint_sha256(
        append_checkpoint.checkpoint_sha256,
        current.checkpoint_sha256,
        relation_cells,
        excitatory,
        inhibitory,
    )
    summary = SegmentedAggregateTrainingSummary(
        len(episodes),
        append_checkpoint.checkpoint_sha256,
        initial_segmenter.checkpoint_sha256,
        current.checkpoint_sha256,
        sum(certificate.work.stacked_binding_cell_steps for certificate in certificates),
        sum(item.input_projection_work for item in projections),
        sum(item.effect_projection_cell_steps for item in projections),
        sum(item.physical_cell_steps for item in transitions),
        len(audits),
        sum(item.outcome_race_cell_steps for item in audits),
        excitatory_ticks,
        inhibitory_ticks,
        TRAINABLE_STATE_BYTES,
        tuple(effect_relations),
        tuple(item.sha256 for item in projections),
        tuple(item.sha256 for item in transitions),
        audits,
        checkpoint_sha,
    )
    return FrozenSegmentedAggregateV1(
        append_checkpoint,
        current,
        relation_cells,
        excitatory,
        inhibitory,
        summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _work_payload(value: SegmentedAggregateWork) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _state_payload(value: SegmentedAggregateFrontierState) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field not in {"work", "sha256"}
    } | {"work_sha256": value.work.sha256}


def _normalize_final_cases(
    values: Iterable[tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("final cases must be an iterable of argument tuples")
    frozen = tuple(tuple(row) for row in values)
    if not frozen:
        raise ValueError("final cases cannot be empty")
    return frozen


def _list_case_key(row: tuple[object, ...]) -> tuple[int, ...] | None:
    if len(row) != 1 or type(row[0]) is not list:
        return None
    if any(type(item) is not int for item in row[0]):
        return None
    return tuple(row[0])


class SegmentedAggregateFrontierV1:
    """Authenticated desired-free frontier produced by one composite checkpoint."""

    __slots__ = (
        "_append_certificate",
        "_checkpoint",
        "_delegate",
        "_end_segmenter",
        "_state",
    )

    def __init__(self) -> None:
        raise TypeError("use FrozenSegmentedAggregateV1.build")

    @classmethod
    def _build(
        cls,
        checkpoint: FrozenSegmentedAggregateV1,
        certificate: append.AppendEffectReducerCertificate,
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        event_order: tuple[int, ...],
        cue_gate: bool,
        comparator_gate: bool,
        router_cell_gate: bool,
    ) -> SegmentedAggregateFrontierV1:
        receipt = append._authenticate_binding(certificate)
        if receipt.checkpoint_sha256 != checkpoint._append_checkpoint.checkpoint_sha256:
            raise RuntimeError("append certificate belongs to another checkpoint")
        start = checkpoint._segmenter
        current = start
        projection: _ProjectedEffectEvents | None = None
        transitions: list[segmenter.PredictiveSegmentationTransition] = []
        router_actions: list[int] = []
        router_roots: list[str] = []
        delegate: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None = None
        scaffold_sha: str | None = None
        effect_relation: int | None = receipt.effect_relation
        selected_action: int | None = receipt.selected_action
        status = certificate.status

        if status == append.READY:
            binding = append.require_append_effect_reducer_binding(certificate)
            projection = _project_effect_events(
                binding.effect_relation, checkpoint._relation_cells, event_order
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
                relation = _decision_relation(transition)
                router_status, action, router_sha = _race_router(
                    checkpoint._excitatory,
                    checkpoint._inhibitory,
                    relation,
                    cell_gate=router_cell_gate,
                )
                router_roots.append(router_sha)
                if router_status != READY or action is None:
                    status = router_status
                    break
                required = _required_action(transition)
                if action != required:
                    status = STOP_ROUTER_CONSEQUENCE
                    break
                router_actions.append(action)
                current = transition.successor

            if status == append.READY:
                if (
                    len(transitions) != EVENTS_PER_SEGMENT
                    or not transitions[-1].segment_completed
                    or current.segment_count != start.segment_count + 1
                ):
                    status = STOP_SEGMENT
                else:
                    binding = append.require_append_effect_reducer_binding(certificate)
                    frozen_cases = _normalize_final_cases(cases)
                    probe_keys = {row[0] for row in binding.probe_cases}
                    if any(
                        key is not None and key in probe_keys
                        for key in map(_list_case_key, frozen_cases)
                    ):
                        status = STOP_PROBE_FINAL_OVERLAP
                    else:
                        upstream = residual.require_residual_literal_route_binding(
                            receipt.residual_certificate
                        )
                        scaffold = append._reducer_scaffold(
                            upstream, binding.selected_action
                        )
                        scaffold_sha = scaffold.sha256
                        delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
                            signature,
                            frozen_cases,
                            tuple(literals),
                            max_cost,
                            scaffold,
                        )
                        status = delegate.status

        append_steps = certificate.work.stacked_binding_cell_steps
        effect_steps = 0 if projection is None else projection.effect_projection_cell_steps
        segment_steps = sum(item.physical_cell_steps for item in transitions)
        router_steps = len(router_roots) * ROUTER_RACE_CELL_STEPS
        partial_work = SegmentedAggregateWork(
            append_steps,
            0 if projection is None else projection.input_projection_work,
            effect_steps,
            segment_steps,
            router_steps,
            append_steps + effect_steps + segment_steps + router_steps,
            None if projection is None else projection.sha256,
            tuple(item.sha256 for item in transitions),
            tuple(router_roots),
            certificate.work.sha256,
            "",
        )
        work = partial_work._replace(
            sha256=_digest(_WORK_NAMESPACE, _work_payload(partial_work))
        )
        partial_state = SegmentedAggregateFrontierState(
            status,
            checkpoint.checkpoint_sha256,
            certificate.sha256,
            start.checkpoint_sha256,
            current.checkpoint_sha256,
            effect_relation,
            selected_action,
            tuple(router_actions),
            scaffold_sha,
            None if delegate is None else delegate.sha256,
            0 if delegate is None else 1,
            DESIRED_READS_DURING_BUILD,
            work,
            "",
        )
        state = partial_state._replace(
            sha256=_digest(_STATE_NAMESPACE, _state_payload(partial_state))
        )
        instance = object.__new__(cls)
        instance._checkpoint = checkpoint
        instance._append_certificate = certificate
        instance._end_segmenter = current
        instance._delegate = delegate
        instance._state = state
        instance._authenticate()
        return instance

    @property
    def state(self) -> SegmentedAggregateFrontierState:
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
        append._authenticate_binding(self._append_certificate)
        self._end_segmenter.authenticate()
        if (
            self._state.checkpoint_sha256 != self._checkpoint.checkpoint_sha256
            or self._state.append_certificate_sha256
            != self._append_certificate.sha256
            or self._state.segmenter_end_sha256
            != self._end_segmenter.checkpoint_sha256
            or self._state.delegated_frontier_sha256
            != (None if self._delegate is None else self._delegate.sha256)
            or self._state.packed_build_count != (0 if self._delegate is None else 1)
            or self._state.desired_reads_during_build != 0
            or self._state.work.sha256
            != _digest(_WORK_NAMESPACE, _work_payload(self._state.work))
            or self._state.sha256
            != _digest(_STATE_NAMESPACE, _state_payload(self._state))
        ):
            raise RuntimeError("segmented aggregate frontier authentication failed")

    def select(self, desired: Sequence[object]) -> SegmentedAggregateSelection:
        self._authenticate()
        delegated = None if self._delegate is None else self._delegate.select(desired)
        status = self._state.status if delegated is None else delegated.status
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "delegated_sha256": None if delegated is None else delegated.sha256,
        }
        return SegmentedAggregateSelection(
            status,
            self._state.sha256,
            delegated,
            _digest(_SELECTION_NAMESPACE, payload),
        )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "append_config_sha256": append.CONFIG_SHA256,
        "segmenter_config_sha256": segmenter.CONFIG_SHA256,
        "curriculum_values": CURRICULUM_VALUES,
        "events_per_segment": EVENTS_PER_SEGMENT,
        "train_segment_count": TRAIN_SEGMENT_COUNT,
        "new_cells": NEW_CELL_COUNT,
        "total_cells": TOTAL_CELL_COUNT,
        "router_state_bytes": TRAINABLE_STATE_BYTES,
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
    "C1_INPUT_COUNT",
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "CONTINUE_RELATION",
    "CURRICULUM_VALUES",
    "DESIRED_READS_DURING_BUILD",
    "DEV_INPUT_COUNT",
    "EVENTS_PER_SEGMENT",
    "EXTEND_ACTION",
    "G0_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "NEW_RELATION",
    "OPEN_ACTION",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "READY",
    "ROUTER_RACE_CELL_STEPS",
    "SCHEMA",
    "STOP_ROUTER_ZERO",
    "TOTAL_CELL_COUNT",
    "TRAINABLE_STATE_BYTES",
    "FrozenSegmentedAggregateV1",
    "SegmentedAggregateFitStop",
    "SegmentedAggregateFrontierState",
    "SegmentedAggregateFrontierV1",
    "SegmentedAggregateSelection",
    "SegmentedAggregateTrainingSummary",
    "SegmentedAggregateWork",
    "fit_causal_segmented_aggregate_v1",
]
