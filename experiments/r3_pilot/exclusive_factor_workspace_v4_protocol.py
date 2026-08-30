"""Data-only freeze for the exclusive prompt-factor workspace V4.

This module contains no learner, learned arrays, fitting operation, factor
resolution operation, or composition execution.  It freezes the inputs and
physical contract that a separately implemented successor must satisfy.

The eventual module has a deep two-operation interface: build one frozen
workspace, then resolve authenticated span batches through it.  Those names
are authenticated metadata here, not executable callables.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from experiments.r3_pilot import (
    prompt_factor_multispan_event_bridge_v3 as v3,
)
from experiments.r3_pilot import (
    public_train_prompt_factor_compositional_bridge_v2 as v2,
)
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

PROTOCOL_SCOPE = "SELF_POSED_EXCLUSIVE_FACTOR_WORKSPACE_V4_PROTOCOL"
PROTOCOL_STATUS = "FROZEN_DATA_ONLY_NOT_RUN"
FIT = "FIT"
HELDOUT = "HELDOUT"

FACTOR_REGISTRY = v2.FACTOR_REGISTRY
FACTOR_COUNT = len(FACTOR_REGISTRY)
TAUGHT_FACTORS = (
    "add",
    "absolute",
    "maximum",
    "subtract",
    "less-than",
    "sum",
    "minimum",
    "map",
    "count",
)
TRIADS = (
    ("add", "absolute", "maximum"),
    ("subtract", "less-than", "sum"),
    ("minimum", "map", "count"),
)

GROUP_COUNT = 18
LESSON_COUNT = 54
FIT_COUNT = 36
HELDOUT_COUNT = 18
NEUTRAL_SPAN_COUNT = 6
COMPOSITION_CASE_COUNT = 18
DUPLICATE_CONTROL_COUNT = 9
OVERLAP_CONTROL_COUNT = 3
CAPACITY_CONTROL_COUNT = 3
WTA_ELECTRICAL_CONTROL_COUNT = 5

INPUT_BANK_COUNT = 3
INPUT_BANK_CELL_COUNT = 512
INPUT_CELL_COUNT = INPUT_BANK_COUNT * INPUT_BANK_CELL_COUNT
ASSEMBLY_ACTIVE_CELL_COUNT = 12
MAX_SPAN_TOKENS = 4
MAX_ACTIVE_INPUT_CELLS = (
    MAX_SPAN_TOKENS * ASSEMBLY_ACTIVE_CELL_COUNT
    + MAX_SPAN_TOKENS * ASSEMBLY_ACTIVE_CELL_COUNT
    + (MAX_SPAN_TOKENS - 1) * ASSEMBLY_ACTIVE_CELL_COUNT
)

TICK_DTYPE = "uint8"
TICK_CAP = 4
SCORE_MIN = -MAX_ACTIVE_INPUT_CELLS * TICK_CAP
SCORE_MAX = MAX_ACTIVE_INPUT_CELLS * TICK_CAP
RAW_TO_CURRENT_SCORE_RULE = "current_score=max(raw_score,0)"
STOP_SCORE = ASSEMBLY_ACTIVE_CELL_COUNT
RACE_STEPS = 600
ZERO_SCORE_SPIKE_STEP = 568
MIN_SCORE_SPIKE_STEP = ZERO_SCORE_SPIKE_STEP - SCORE_MAX
TEACHER_CLAMP_CURRENT_PA = 5_000.0
RELAY_COPY_CURRENT_PA = 5_000.0
BASKET_INHIBITION_CURRENT_PA = -5_000.0
BASKET_DELAY_TICKS = 1
FACTOR_RELAY_DELAY_TICKS = 3
STOP_RELAY_DELAY_TICKS = 1
COLLISION_DETECTOR_DELAY_TICKS = 1
COLLISION_RELAY_DELAY_TICKS = 1
COLLISION_STOP_DELAY_TICKS = 2
EVENT_CAPACITY = 2

PUBLIC_INPUT_COUNT = 0
CANONICAL_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
LEARNING_RUN_COUNT = 0
COMPOSITION_RESULT_COUNT = 0

_ATOMIC_LESSON_NAMESPACE = b"exclusive-factor-workspace-v4-atomic-lesson\0"
_ATOMIC_INPUT_NAMESPACE = b"exclusive-factor-workspace-v4-atomic-inputs\0"
_CURRICULUM_NAMESPACE = b"exclusive-factor-workspace-v4-curriculum\0"
_SPLIT_NAMESPACE = b"exclusive-factor-workspace-v4-split\0"
_V3_REFERENCE_NAMESPACE = b"exclusive-factor-workspace-v4-v3-reference\0"
_NEUTRAL_NAMESPACE = b"exclusive-factor-workspace-v4-neutral\0"
_COMPOSITION_CASE_NAMESPACE = b"exclusive-factor-workspace-v4-composition-case\0"
_COMPOSITION_NAMESPACE = b"exclusive-factor-workspace-v4-composition\0"
_CONTROL_NAMESPACE = b"exclusive-factor-workspace-v4-control\0"
_LATENCY_NAMESPACE = b"exclusive-factor-workspace-v4-latency\0"
_INTERFACE_NAMESPACE = b"exclusive-factor-workspace-v4-interface\0"
_WTA_NAMESPACE = b"exclusive-factor-workspace-v4-wta\0"
_WTA_ELECTRICAL_CONTROL_NAMESPACE = (
    b"exclusive-factor-workspace-v4-wta-electrical-control\0"
)
_WTA_ELECTRICAL_CONTROLS_NAMESPACE = (
    b"exclusive-factor-workspace-v4-wta-electrical-controls\0"
)
_LOCAL_RULE_NAMESPACE = b"exclusive-factor-workspace-v4-local-rule\0"
_PROTOCOL_NAMESPACE = b"exclusive-factor-workspace-v4-protocol\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _current_score(raw_score: int) -> int:
    if type(raw_score) is not int or not SCORE_MIN <= raw_score <= SCORE_MAX:
        raise ValueError("raw factor score lies outside the frozen V4 domain")
    return max(raw_score, 0)


def _freeze_record(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_record(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_record(item) for item in value)
    return value


MARKER_TOKEN = "m"
_ROW_SPECS = (
    (FIT, 0),
    (FIT, 1),
    (FIT, 2),
    (FIT, 3),
    (HELDOUT, 0),
    (HELDOUT, 1),
)


def _lesson_tokens(
    row_index: int,
    target: str,
    distractor_1: str,
    distractor_2: str,
) -> tuple[str, str, str, str]:
    if row_index == 0:
        return (target, distractor_1, distractor_2, MARKER_TOKEN)
    if row_index == 1:
        return (distractor_1, target, MARKER_TOKEN, distractor_2)
    if row_index == 2:
        return (distractor_2, MARKER_TOKEN, target, distractor_1)
    if row_index == 3:
        return (MARKER_TOKEN, distractor_2, distractor_1, target)
    if row_index == 4:
        return (target, distractor_1, MARKER_TOKEN, distractor_2)
    if row_index == 5:
        return (distractor_1, target, distractor_2, MARKER_TOKEN)
    raise ValueError("row_index lies outside the frozen V4 curriculum")


@dataclass(frozen=True, slots=True)
class AtomicLesson:
    group_index: int
    triad_index: int
    row_index: int
    target_index: int
    tokens: tuple[str, str, str, str]
    factor: str
    split: str
    target_slot: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "group_index": self.group_index,
            "triad_index": self.triad_index,
            "row_index": self.row_index,
            "target_index": self.target_index,
            "tokens": self.tokens,
            "factor": self.factor,
            "split": self.split,
            "target_slot": self.target_slot,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class AtomicContrastGroup:
    group_index: int
    triad_index: int
    row_index: int
    split: str
    lessons: tuple[AtomicLesson, AtomicLesson, AtomicLesson]


@dataclass(frozen=True, slots=True)
class NeutralSpan:
    neutral_index: int
    tokens: tuple[str, str, str, str]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "neutral_index": self.neutral_index,
            "tokens": self.tokens,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SpanPlacement:
    start_token: int
    stop_token: int
    tokens: tuple[str, str, str, str]
    kind: str
    source_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "start_token": self.start_token,
            "stop_token": self.stop_token,
            "tokens": self.tokens,
            "kind": self.kind,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompositionCase:
    case_index: int
    triad_index: int
    left_factor: str
    right_factor: str
    spans: tuple[SpanPlacement, SpanPlacement, SpanPlacement]
    expected_factors: tuple[str, str]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "case_index": self.case_index,
            "triad_index": self.triad_index,
            "left_factor": self.left_factor,
            "right_factor": self.right_factor,
            "spans": tuple(span.record() for span in self.spans),
            "expected_factors": self.expected_factors,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ProtocolFactorEvent:
    factor: str
    start_token: int
    stop_token: int
    event_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "factor": self.factor,
            "start_token": self.start_token,
            "stop_token": self.stop_token,
            "event_sha256": self.event_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompositorControl:
    control_index: int
    kind: str
    events: tuple[ProtocolFactorEvent, ...]
    expected_status: str
    expected_factors: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "control_index": self.control_index,
            "kind": self.kind,
            "events": tuple(event.record() for event in self.events),
            "expected_status": self.expected_status,
            "expected_factors": self.expected_factors,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class LatencyEntry:
    score: int
    current_pa: float
    first_spike_step: int

    def record(self) -> dict[str, object]:
        return {
            "score": self.score,
            "current_pa_hex": self.current_pa.hex(),
            "first_spike_step": self.first_spike_step,
        }


@dataclass(frozen=True, slots=True)
class WTAElectricalControl:
    control_index: int
    name: str
    principal_raw_scores: tuple[int, ...]
    principal_current_scores: tuple[int, ...]
    principal_currents_pa: tuple[float, ...]
    principal_first_spike_step: int
    principal_first_spike_tier: tuple[int, ...]
    simultaneous_spike_candidates: int
    basket_inhibition_step: int
    stop_relay_step: int | None
    collision_detection_step: int | None
    collision_stop_step: int | None
    factor_relay_steps: tuple[tuple[int, int], ...]
    expected_decision_step: int
    expected_decision: str
    expected_factor: str | None
    full_circuit_execution_deferred: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "control_index": self.control_index,
            "name": self.name,
            "principal_raw_scores": self.principal_raw_scores,
            "principal_current_scores": self.principal_current_scores,
            "principal_currents_pa_hex": tuple(
                current.hex() for current in self.principal_currents_pa
            ),
            "principal_first_spike_step": self.principal_first_spike_step,
            "principal_first_spike_tier": self.principal_first_spike_tier,
            "simultaneous_spike_candidates": self.simultaneous_spike_candidates,
            "basket_inhibition_step": self.basket_inhibition_step,
            "stop_relay_step": self.stop_relay_step,
            "collision_detection_step": self.collision_detection_step,
            "collision_stop_step": self.collision_stop_step,
            "factor_relay_steps": self.factor_relay_steps,
            "expected_decision_step": self.expected_decision_step,
            "expected_decision": self.expected_decision,
            "expected_factor": self.expected_factor,
            "full_circuit_execution_deferred": self.full_circuit_execution_deferred,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenExclusiveFactorProtocol:
    scope: str
    status: str
    atomic_groups: tuple[AtomicContrastGroup, ...]
    fit_lesson_sha256s: tuple[str, ...]
    heldout_lesson_sha256s: tuple[str, ...]
    neutral_spans: tuple[NeutralSpan, ...]
    composition_cases: tuple[CompositionCase, ...]
    duplicate_controls: tuple[CompositorControl, ...]
    overlap_controls: tuple[CompositorControl, ...]
    capacity_controls: tuple[CompositorControl, ...]
    latency_table: tuple[LatencyEntry, ...]
    wta_electrical_controls: tuple[WTAElectricalControl, ...]
    stop_current_pa: float
    interface_sha256: str
    atomic_input_sha256: str
    curriculum_sha256: str
    split_sha256: str
    v3_reference_input_sha256: str
    neutral_sha256: str
    composition_sha256: str
    duplicate_controls_sha256: str
    overlap_controls_sha256: str
    capacity_controls_sha256: str
    glif_sha256: str
    latency_table_sha256: str
    wta_topology_sha256: str
    wta_electrical_controls_sha256: str
    local_rule_sha256: str
    protocol_sha256: str
    public_input_count: int
    canonical_input_count: int
    protected_input_count: int
    learning_run_count: int
    composition_result_count: int

    @property
    def lessons(self) -> tuple[AtomicLesson, ...]:
        return tuple(lesson for group in self.atomic_groups for lesson in group.lessons)

    def root_record(self, *, include_protocol_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "interface_sha256": self.interface_sha256,
            "atomic_input_sha256": self.atomic_input_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "split_sha256": self.split_sha256,
            "v3_reference_input_sha256": self.v3_reference_input_sha256,
            "neutral_sha256": self.neutral_sha256,
            "composition_sha256": self.composition_sha256,
            "duplicate_controls_sha256": self.duplicate_controls_sha256,
            "overlap_controls_sha256": self.overlap_controls_sha256,
            "capacity_controls_sha256": self.capacity_controls_sha256,
            "glif_sha256": self.glif_sha256,
            "latency_table_sha256": self.latency_table_sha256,
            "wta_topology_sha256": self.wta_topology_sha256,
            "wta_electrical_controls_sha256": self.wta_electrical_controls_sha256,
            "local_rule_sha256": self.local_rule_sha256,
            "group_count": len(self.atomic_groups),
            "lesson_count": len(self.lessons),
            "fit_count": len(self.fit_lesson_sha256s),
            "heldout_count": len(self.heldout_lesson_sha256s),
            "neutral_span_count": len(self.neutral_spans),
            "composition_case_count": len(self.composition_cases),
            "duplicate_control_count": len(self.duplicate_controls),
            "overlap_control_count": len(self.overlap_controls),
            "capacity_control_count": len(self.capacity_controls),
            "wta_electrical_control_count": len(self.wta_electrical_controls),
            "public_input_count": self.public_input_count,
            "canonical_input_count": self.canonical_input_count,
            "protected_input_count": self.protected_input_count,
            "learning_run_count": self.learning_run_count,
            "composition_result_count": self.composition_result_count,
            "learner_state_constructed": False,
            "composition_executed": False,
            "execution_surface_present": False,
            "learner_electrical_preflight_executed": False,
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


_INTERFACE_RECORD = {
    "module": "FrozenExclusiveFactorBridge",
    "operations": (
        {
            "name": "build_frozen_exclusive_factor_bridge",
            "inputs": ("FrozenExclusiveFactorProtocol",),
            "output": "FrozenExclusiveFactorBridge",
            "atomic_failure": (
                "return an authenticated terminal workspace whose resolve operation "
                "refuses before inspecting its prompt"
            ),
        },
        {
            "name": "FrozenExclusiveFactorBridge.resolve",
            "inputs": ("AuthenticatedSpanBatch",),
            "output": "ExclusiveFactorResolution",
            "invariants": (
                "process every authenticated span",
                "no caller factor selection, union, filter, or winner tie-break",
                "at most two physical factor events",
            ),
        },
    ),
    "protocol_module_exposes_operations": False,
}
INTERFACE_SHA256 = _namespaced_sha256(_INTERFACE_NAMESPACE, _INTERFACE_RECORD)
INTERFACE_RECORD = _freeze_record(_INTERFACE_RECORD)


_LOCAL_RULE_BASE_RECORD = {
    "training_epochs": 4,
    "tick_dtype": TICK_DTYPE,
    "tick_cap": TICK_CAP,
    "state_shape": (FACTOR_COUNT, INPUT_CELL_COUNT),
    "state_arrays": ("excitatory_ticks", "inhibitory_ticks"),
    "target_miss": "increment active target excitatory ticks",
    "unique_wrong_factor": (
        "increment active target excitatory and active physical-winner inhibitory ticks"
    ),
    "collision_or_stop": "increment active target excitatory ticks only",
    "correct_unique_factor": "no update",
    "update_inputs": (
        "active presynaptic fitted cells",
        "physical factor or STOP relay event",
        "one-hot teacher clamp",
    ),
    "forbidden": (
        "independent binary factor errors",
        "heldout updates",
        "host negative-label generation",
        "retry",
        "parameter sweep",
    ),
}


def _build_local_rule_record(
    lessons: tuple[AtomicLesson, ...],
    fit_ids: tuple[str, ...],
) -> tuple[object, str]:
    fit_id_set = set(fit_ids)
    ordered_fit = tuple(
        sorted(
            (lesson for lesson in lessons if lesson.sha256 in fit_id_set),
            key=lambda lesson: (lesson.tokens, lesson.factor, lesson.sha256),
        )
    )
    if len(ordered_fit) != FIT_COUNT:
        raise RuntimeError("exclusive V4 FIT schedule shape changed")
    record = dict(_LOCAL_RULE_BASE_RECORD)
    record["fit_schedule"] = {
        "sort_key": ("canonical_tokens", "factor", "lesson_sha256"),
        "sort_direction": "ascending",
        "shuffle": False,
        "lesson_sha256s": tuple(lesson.sha256 for lesson in ordered_fit),
        "causal_order_source": "this lesson_sha256s tuple only",
        "split_fit_ids_semantics": "membership only; never causal iteration order",
        "epoch_count": record["training_epochs"],
        "epoch_policy": "repeat this exact FIT order unchanged every epoch",
        "heldout_scheduled": False,
    }
    local_rule_sha = _namespaced_sha256(_LOCAL_RULE_NAMESPACE, record)
    return _freeze_record(record), local_rule_sha


_WTA_TOPOLOGY_RECORD = {
    "principal_factor_cells": FACTOR_COUNT,
    "principal_stop_cells": 1,
    "basket_inhibitory_cells": 1,
    "collision_cells": 1,
    "delayed_factor_relays": FACTOR_COUNT,
    "stop_relays": 1,
    "candidate_count": FACTOR_COUNT + 1,
    "stop_candidate_index": FACTOR_COUNT,
    "raw_score_domain": (SCORE_MIN, SCORE_MAX),
    "raw_to_current_score_rule": RAW_TO_CURRENT_SCORE_RULE,
    "current_score_domain": (0, SCORE_MAX),
    "stop_score": STOP_SCORE,
    "score_acceptance": "unique factor score >= stop_score + 1",
    "tie_semantics": "any leading tie routes through the physical collision path to STOP",
    "basket_delay_ticks": BASKET_DELAY_TICKS,
    "factor_relay_delay_ticks": FACTOR_RELAY_DELAY_TICKS,
    "stop_relay_delay_ticks": STOP_RELAY_DELAY_TICKS,
    "collision_detector_delay_ticks": COLLISION_DETECTOR_DELAY_TICKS,
    "collision_relay_delay_ticks": COLLISION_RELAY_DELAY_TICKS,
    "collision_stop_delay_ticks": COLLISION_STOP_DELAY_TICKS,
    "tick_phase_precedence": (
        "scheduled basket inhibition",
        "principal threshold evaluation",
        "collision and decision relay arrivals",
    ),
    "basket_precedes_principal_threshold_on_equal_step": True,
    "teacher_clamp_current_pa_hex": TEACHER_CLAMP_CURRENT_PA.hex(),
    "relay_copy_current_pa_hex": RELAY_COPY_CURRENT_PA.hex(),
    "basket_inhibition_current_pa_hex": BASKET_INHIBITION_CURRENT_PA.hex(),
    "event_capacity": EVENT_CAPACITY,
    "race_adapter": "CompiledRestingGlifRace.race_many",
    "batched_current_shape": ("authenticated_span_count", FACTOR_COUNT + 1),
    "batch_authentication": (
        "one checkpoint authentication before and one after the complete span batch"
    ),
    "per_span_checkpoint_hashing": False,
    "race_require_unique": False,
    "winner_cell_use": (
        "permitted only when simultaneous_spike_candidates == 1; otherwise fixed "
        "collision/STOP projection consumes the spike tier"
    ),
    "host_argmax_or_tie_break": False,
    "electrical_control_scope": (
        "principal 25-way race_many tiers are executed in this freeze; basket, "
        "collision, and relay execution is deferred to learner electrical preflight"
    ),
    "compositor": (
        "fixed admission, exact-duplicate, overlap, position-order, one-hot-copy, "
        "and overflow STOP races"
    ),
}
WTA_TOPOLOGY_SHA256 = _namespaced_sha256(_WTA_NAMESPACE, _WTA_TOPOLOGY_RECORD)
WTA_TOPOLOGY_RECORD = _freeze_record(_WTA_TOPOLOGY_RECORD)


def _build_atomic_curriculum() -> tuple[
    tuple[AtomicContrastGroup, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    groups: list[AtomicContrastGroup] = []
    fit_ids: list[str] = []
    heldout_ids: list[str] = []
    group_index = 0
    for triad_index, triad in enumerate(TRIADS):
        for row_index, (split, target_slot) in enumerate(_ROW_SPECS):
            lessons: list[AtomicLesson] = []
            for target_index, target in enumerate(triad):
                distractor_1 = triad[(target_index + 1) % len(triad)]
                distractor_2 = triad[(target_index + 2) % len(triad)]
                tokens = _lesson_tokens(
                    row_index,
                    target,
                    distractor_1,
                    distractor_2,
                )
                record = {
                    "group_index": group_index,
                    "triad_index": triad_index,
                    "row_index": row_index,
                    "target_index": target_index,
                    "tokens": tokens,
                    "factor": target,
                    "split": split,
                    "target_slot": target_slot,
                }
                lesson = AtomicLesson(
                    **record,
                    sha256=_namespaced_sha256(_ATOMIC_LESSON_NAMESPACE, record),
                )
                lessons.append(lesson)
                (fit_ids if split == FIT else heldout_ids).append(lesson.sha256)
            groups.append(
                AtomicContrastGroup(
                    group_index=group_index,
                    triad_index=triad_index,
                    row_index=row_index,
                    split=split,
                    lessons=tuple(lessons),
                )
            )
            group_index += 1
    return tuple(groups), tuple(fit_ids), tuple(heldout_ids)


def _validate_atomic_curriculum(
    groups: tuple[AtomicContrastGroup, ...],
    fit_ids: tuple[str, ...],
    heldout_ids: tuple[str, ...],
) -> tuple[AtomicLesson, ...]:
    lessons = tuple(lesson for group in groups for lesson in group.lessons)
    if (
        len(groups) != GROUP_COUNT
        or len(lessons) != LESSON_COUNT
        or len(fit_ids) != FIT_COUNT
        or len(heldout_ids) != HELDOUT_COUNT
        or len({lesson.sha256 for lesson in lessons}) != LESSON_COUNT
        or set(fit_ids) & set(heldout_ids)
        or set(fit_ids) | set(heldout_ids) != {lesson.sha256 for lesson in lessons}
    ):
        raise RuntimeError("exclusive V4 atomic curriculum shape changed")
    if tuple(factor for triad in TRIADS for factor in triad) != TAUGHT_FACTORS:
        raise RuntimeError("exclusive V4 taught-factor order changed")
    if any(factor not in FACTOR_REGISTRY for factor in TAUGHT_FACTORS):
        raise RuntimeError("exclusive V4 curriculum contains an unknown factor")

    for group in groups:
        if (
            len({tuple(sorted(lesson.tokens)) for lesson in group.lessons}) != 1
            or len({lesson.factor for lesson in group.lessons}) != 3
            or {lesson.split for lesson in group.lessons} != {group.split}
            or any(lesson.tokens[lesson.target_slot] != lesson.factor for lesson in group.lessons)
        ):
            raise RuntimeError("exclusive V4 matched contrast geometry changed")

    fit_set = set(fit_ids)
    heldout_set = set(heldout_ids)
    for factor in TAUGHT_FACTORS:
        if (
            sum(lesson.factor == factor and lesson.sha256 in fit_set for lesson in lessons) != 4
            or sum(
                lesson.factor == factor and lesson.sha256 in heldout_set
                for lesson in lessons
            )
            != 2
        ):
            raise RuntimeError("exclusive V4 factor capacity changed")
        fit_slots = {
            lesson.target_slot
            for lesson in lessons
            if lesson.factor == factor and lesson.sha256 in fit_set
        }
        heldout_slots = {
            lesson.target_slot
            for lesson in lessons
            if lesson.factor == factor and lesson.sha256 in heldout_set
        }
        if not heldout_slots <= fit_slots:
            raise RuntimeError("heldout factor placement lacks FIT support")

    fit_inputs = {lesson.tokens for lesson in lessons if lesson.sha256 in fit_set}
    heldout_inputs = {lesson.tokens for lesson in lessons if lesson.sha256 in heldout_set}
    if fit_inputs & heldout_inputs:
        raise RuntimeError("exclusive V4 FIT and heldout inputs overlap")

    fit_features = {
        factor: tuple(set() for _bank in range(3)) for factor in TAUGHT_FACTORS
    }
    for lesson in lessons:
        if lesson.sha256 not in fit_set:
            continue
        for destination, source in zip(
            fit_features[lesson.factor],
            v2._feature_keys(lesson.tokens),
            strict=True,
        ):
            destination.update(source)
    for lesson in lessons:
        if lesson.sha256 not in heldout_set:
            continue
        token_keys, slot_keys, transition_keys = v2._feature_keys(lesson.tokens)
        factor_features = fit_features[lesson.factor]
        if (
            not token_keys <= factor_features[0]
            or not slot_keys <= factor_features[1]
            or len(transition_keys & factor_features[2]) < 2
        ):
            raise RuntimeError(
                "exclusive V4 heldout feature support changed"
            )

    v3_groups, _v3_split = v3.self_posed_local_span_curriculum()
    v3_inputs = {
        lesson.teacher_span.canonical_tokens
        for group in v3_groups
        for lesson in group.lessons
    }
    if {lesson.tokens for lesson in lessons} & v3_inputs:
        raise RuntimeError("exclusive V4 input duplicates a spent V3 input")
    return lessons


def _build_neutral_spans() -> tuple[NeutralSpan, ...]:
    token_rows = (
        ("quiet", "idle", "plain", "blank"),
        ("blank", "plain", "idle", "quiet"),
        ("silent", "calm", "still", "empty"),
        ("empty", "still", "calm", "silent"),
        ("neutral", "unused", "vacant", "spare"),
        ("spare", "vacant", "unused", "neutral"),
    )
    rows: list[NeutralSpan] = []
    for neutral_index, tokens in enumerate(token_rows):
        record = {"neutral_index": neutral_index, "tokens": tokens}
        rows.append(
            NeutralSpan(
                neutral_index=neutral_index,
                tokens=tokens,
                sha256=_namespaced_sha256(_NEUTRAL_NAMESPACE, record),
            )
        )
    return tuple(rows)


def _lesson_lookup(
    lessons: tuple[AtomicLesson, ...],
) -> dict[tuple[int, int, str], AtomicLesson]:
    return {
        (lesson.triad_index, lesson.row_index, lesson.factor): lesson
        for lesson in lessons
    }


def _build_composition_cases(
    lessons: tuple[AtomicLesson, ...],
    neutrals: tuple[NeutralSpan, ...],
) -> tuple[CompositionCase, ...]:
    lookup = _lesson_lookup(lessons)
    cases: list[CompositionCase] = []
    for triad_index, triad in enumerate(TRIADS):
        ordered_pairs = tuple(
            (left, right)
            for left in triad
            for right in triad
            if left != right
        )
        for pair_index, (left_factor, right_factor) in enumerate(ordered_pairs):
            case_index = len(cases)
            left_row, right_row = ((4, 5) if pair_index % 2 == 0 else (5, 4))
            left = lookup[(triad_index, left_row, left_factor)]
            right = lookup[(triad_index, right_row, right_factor)]
            neutral = neutrals[case_index % len(neutrals)]
            spans = (
                SpanPlacement(0, 4, left.tokens, "factor", left.sha256),
                SpanPlacement(4, 8, neutral.tokens, "neutral", neutral.sha256),
                SpanPlacement(8, 12, right.tokens, "factor", right.sha256),
            )
            record = {
                "case_index": case_index,
                "triad_index": triad_index,
                "left_factor": left_factor,
                "right_factor": right_factor,
                "spans": tuple(span.record() for span in spans),
                "expected_factors": (left_factor, right_factor),
            }
            cases.append(
                CompositionCase(
                    case_index=case_index,
                    triad_index=triad_index,
                    left_factor=left_factor,
                    right_factor=right_factor,
                    spans=spans,
                    expected_factors=(left_factor, right_factor),
                    sha256=_namespaced_sha256(
                        _COMPOSITION_CASE_NAMESPACE,
                        record,
                    ),
                )
            )
    return tuple(cases)


def _event(factor: str, start: int, stop: int) -> ProtocolFactorEvent:
    record = {"factor": factor, "start_token": start, "stop_token": stop}
    return ProtocolFactorEvent(
        factor=factor,
        start_token=start,
        stop_token=stop,
        event_sha256=_namespaced_sha256(_CONTROL_NAMESPACE, record),
    )


def _control(
    control_index: int,
    kind: str,
    events: tuple[ProtocolFactorEvent, ...],
    expected_status: str,
    expected_factors: tuple[str, ...],
) -> CompositorControl:
    record = {
        "control_index": control_index,
        "kind": kind,
        "events": tuple(event.record() for event in events),
        "expected_status": expected_status,
        "expected_factors": expected_factors,
    }
    return CompositorControl(
        control_index=control_index,
        kind=kind,
        events=events,
        expected_status=expected_status,
        expected_factors=expected_factors,
        sha256=_namespaced_sha256(_CONTROL_NAMESPACE, record),
    )


def _build_controls() -> tuple[
    tuple[CompositorControl, ...],
    tuple[CompositorControl, ...],
    tuple[CompositorControl, ...],
]:
    duplicates = tuple(
        _control(
            factor_index,
            "exact-duplicate",
            (_event(factor, 0, 4), _event(factor, 0, 4)),
            "ONE_EVENT",
            (factor,),
        )
        for factor_index, factor in enumerate(TAUGHT_FACTORS)
    )
    overlaps = tuple(
        _control(
            triad_index,
            "distinct-overlap",
            (_event(triad[0], 0, 4), _event(triad[1], 2, 6)),
            "STOP_OVERLAPPING_FACTOR_EVENTS",
            (),
        )
        for triad_index, triad in enumerate(TRIADS)
    )
    capacities = tuple(
        _control(
            triad_index,
            "capacity-three",
            tuple(
                _event(factor, factor_index * 4, (factor_index + 1) * 4)
                for factor_index, factor in enumerate(triad)
            ),
            "STOP_EVENT_CAPACITY_2",
            (),
        )
        for triad_index, triad in enumerate(TRIADS)
    )
    return duplicates, overlaps, capacities


def _glif_record(parameters: object) -> dict[str, object]:
    return {
        "model": type(parameters).__name__,
        "parameters": dataclasses.asdict(parameters),
        "race_steps": RACE_STEPS,
        "latency_construction": (
            "midpoint of the exact constant-current interval whose first fitted-GLIF5 "
            "crossing occurs at the requested step"
        ),
    }


def _build_latency_table() -> tuple[tuple[LatencyEntry, ...], str, str]:
    population = AdaptiveGlifPopulation(1)
    unit_voltage, unit_voltage_threshold = population.compile_resting_response(
        RACE_STEPS
    )
    spike_gain = unit_voltage - unit_voltage_threshold
    crossing_current = (
        population.parameters.threshold_inf_relative_mv / spike_gain
    )
    entries: list[LatencyEntry] = []
    for score in range(SCORE_MAX + 1):
        first_step = ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("fitted-GLIF5 latency interval is not increasing")
        entries.append(
            LatencyEntry(
                score=score,
                current_pa=(lower + upper) / 2.0,
                first_spike_step=first_step,
            )
        )

    currents = np.asarray([entry.current_pa for entry in entries], dtype=np.float64)
    races = CompiledRestingGlifRace.compile(step_count=RACE_STEPS).race_many(
        currents[:, None]
    )
    if any(
        race.spike_step != entry.first_spike_step
        for entry, race in zip(entries, races, strict=True)
    ):
        raise RuntimeError("fitted-GLIF5 latency table failed exact replay")
    if any(race.simultaneous_spike_candidates != 1 for race in races):
        raise RuntimeError("single-cell fitted-GLIF5 calibration became non-unique")

    glif_sha = _namespaced_sha256(
        _LATENCY_NAMESPACE,
        _glif_record(population.parameters),
    )
    table_sha = _namespaced_sha256(
        _LATENCY_NAMESPACE,
        {
            "glif_sha256": glif_sha,
            "raw_score_min": SCORE_MIN,
            "raw_score_max": SCORE_MAX,
            "raw_to_current_score_rule": RAW_TO_CURRENT_SCORE_RULE,
            "current_score_min": 0,
            "current_score_max": SCORE_MAX,
            "stop_score": STOP_SCORE,
            "entries": tuple(entry.record() for entry in entries),
        },
    )
    return tuple(entries), glif_sha, table_sha


def _build_wta_electrical_controls(
    latency_table: tuple[LatencyEntry, ...],
) -> tuple[WTAElectricalControl, ...]:
    factor_zeroes = (0,) * FACTOR_COUNT
    negative_one = (-1,) + (0,) * (FACTOR_COUNT - 1)
    factor_stop_tie = (STOP_SCORE,) + (0,) * (FACTOR_COUNT - 1)
    two_factor_margin = (STOP_SCORE + 1, STOP_SCORE + 1) + (0,) * (
        FACTOR_COUNT - 2
    )
    unique_factor_margin = (STOP_SCORE + 1,) + (0,) * (FACTOR_COUNT - 1)
    specs = (
        ("empty-stop", factor_zeroes, (FACTOR_COUNT,), "STOP", None),
        (
            "negative-one-clamps-to-empty-stop",
            negative_one,
            (FACTOR_COUNT,),
            "STOP",
            None,
        ),
        (
            "factor-stop-score-12-tie",
            factor_stop_tie,
            (0, FACTOR_COUNT),
            "STOP",
            None,
        ),
        (
            "two-factor-score-13-collision",
            two_factor_margin,
            (0, 1),
            "STOP",
            None,
        ),
        (
            "unique-factor-score-13-accepted",
            unique_factor_margin,
            (0,),
            "FACTOR",
            FACTOR_REGISTRY[0],
        ),
    )
    raw_score_rows = tuple(
        (*factor_raw_scores, STOP_SCORE)
        for _, factor_raw_scores, *_ in specs
    )
    current_score_rows = tuple(
        tuple(_current_score(raw_score) for raw_score in raw_scores)
        for raw_scores in raw_score_rows
    )
    current_rows = tuple(
        tuple(latency_table[score].current_pa for score in scores)
        for scores in current_score_rows
    )
    races = CompiledRestingGlifRace.compile(step_count=RACE_STEPS).race_many(
        np.asarray(current_rows, dtype=np.float64),
        require_unique=False,
    )

    controls: list[WTAElectricalControl] = []
    for control_index, (spec, raw_scores, current_scores, currents, race) in enumerate(
        zip(
            specs,
            raw_score_rows,
            current_score_rows,
            current_rows,
            races,
            strict=True,
        )
    ):
        name, _factor_raw_scores, expected_tier, expected_decision, expected_factor = spec
        tier = tuple(
            cell_index
            for cell_index, score in enumerate(current_scores)
            if latency_table[score].first_spike_step == race.spike_step
        )
        if (
            tier != expected_tier
            or race.simultaneous_spike_candidates != len(tier)
        ):
            raise RuntimeError("exclusive V4 principal electrical tier changed")

        principal_step = race.spike_step
        basket_step = principal_step + BASKET_DELAY_TICKS
        is_collision = len(tier) > 1
        stop_relay_step = (
            principal_step + STOP_RELAY_DELAY_TICKS
            if tier == (FACTOR_COUNT,)
            else None
        )
        collision_detection_step = (
            principal_step + COLLISION_DETECTOR_DELAY_TICKS
            if is_collision
            else None
        )
        collision_stop_step = (
            principal_step + COLLISION_STOP_DELAY_TICKS
            if is_collision
            else None
        )
        if (
            collision_detection_step is not None
            and collision_stop_step
            != collision_detection_step + COLLISION_RELAY_DELAY_TICKS
        ):
            raise RuntimeError("exclusive V4 collision schedule changed")
        factor_relay_steps = tuple(
            (cell_index, principal_step + FACTOR_RELAY_DELAY_TICKS)
            for cell_index in tier
            if not is_collision and cell_index < FACTOR_COUNT
        )
        stop_arrivals = tuple(
            step
            for step in (stop_relay_step, collision_stop_step)
            if step is not None
        )
        if stop_arrivals:
            decision_step = min(stop_arrivals)
            decision = "STOP"
            factor = None
        elif len(factor_relay_steps) == 1:
            decision_step = factor_relay_steps[0][1]
            decision = "FACTOR"
            factor = FACTOR_REGISTRY[factor_relay_steps[0][0]]
        else:
            raise RuntimeError("exclusive V4 electrical control has no decision")
        if decision != expected_decision or factor != expected_factor:
            raise RuntimeError("exclusive V4 electrical decision changed")

        record = {
            "control_index": control_index,
            "name": name,
            "principal_raw_scores": raw_scores,
            "principal_current_scores": current_scores,
            "principal_currents_pa_hex": tuple(
                current.hex() for current in currents
            ),
            "principal_first_spike_step": principal_step,
            "principal_first_spike_tier": tier,
            "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
            "basket_inhibition_step": basket_step,
            "stop_relay_step": stop_relay_step,
            "collision_detection_step": collision_detection_step,
            "collision_stop_step": collision_stop_step,
            "factor_relay_steps": factor_relay_steps,
            "expected_decision_step": decision_step,
            "expected_decision": decision,
            "expected_factor": factor,
            "full_circuit_execution_deferred": True,
        }
        controls.append(
            WTAElectricalControl(
                control_index=control_index,
                name=name,
                principal_raw_scores=raw_scores,
                principal_current_scores=current_scores,
                principal_currents_pa=currents,
                principal_first_spike_step=principal_step,
                principal_first_spike_tier=tier,
                simultaneous_spike_candidates=race.simultaneous_spike_candidates,
                basket_inhibition_step=basket_step,
                stop_relay_step=stop_relay_step,
                collision_detection_step=collision_detection_step,
                collision_stop_step=collision_stop_step,
                factor_relay_steps=factor_relay_steps,
                expected_decision_step=decision_step,
                expected_decision=decision,
                expected_factor=factor,
                full_circuit_execution_deferred=True,
                sha256=_namespaced_sha256(
                    _WTA_ELECTRICAL_CONTROL_NAMESPACE,
                    record,
                ),
            )
        )
    return tuple(controls)


def _records_sha256(namespace: bytes, rows: tuple[object, ...]) -> str:
    return _namespaced_sha256(namespace, rows)


def _build_protocol() -> tuple[FrozenExclusiveFactorProtocol, object, str]:
    groups, fit_ids, heldout_ids = _build_atomic_curriculum()
    lessons = _validate_atomic_curriculum(groups, fit_ids, heldout_ids)
    local_rule_record, local_rule_sha = _build_local_rule_record(
        lessons,
        fit_ids,
    )
    neutrals = _build_neutral_spans()
    if len(neutrals) != NEUTRAL_SPAN_COUNT or len(
        {row.sha256 for row in neutrals}
    ) != len(neutrals):
        raise RuntimeError("exclusive V4 neutral-span panel changed")
    if {row.tokens for row in neutrals} & {lesson.tokens for lesson in lessons}:
        raise RuntimeError("exclusive V4 neutral span duplicates an atomic input")

    composition = _build_composition_cases(lessons, neutrals)
    duplicates, overlaps, capacities = _build_controls()
    if (
        len(composition) != COMPOSITION_CASE_COUNT
        or len(duplicates) != DUPLICATE_CONTROL_COUNT
        or len(overlaps) != OVERLAP_CONTROL_COUNT
        or len(capacities) != CAPACITY_CONTROL_COUNT
    ):
        raise RuntimeError("exclusive V4 composition protocol shape changed")

    v3_groups, _v3_split = v3.self_posed_local_span_curriculum()
    v3_inputs = tuple(
        sorted(
            lesson.teacher_span.canonical_tokens
            for group in v3_groups
            for lesson in group.lessons
        )
    )
    atomic_input_sha = _records_sha256(
        _ATOMIC_INPUT_NAMESPACE,
        tuple(sorted(lesson.tokens for lesson in lessons)),
    )
    curriculum_sha = _records_sha256(
        _CURRICULUM_NAMESPACE,
        tuple(
            lesson.record(include_sha256=False)
            for lesson in sorted(lessons, key=lambda row: row.sha256)
        ),
    )
    split_sha = _namespaced_sha256(
        _SPLIT_NAMESPACE,
        {"fit": tuple(sorted(fit_ids)), "heldout": tuple(sorted(heldout_ids))},
    )
    v3_reference_sha = _records_sha256(_V3_REFERENCE_NAMESPACE, v3_inputs)
    neutral_sha = _records_sha256(
        _NEUTRAL_NAMESPACE,
        tuple(row.record(include_sha256=False) for row in neutrals),
    )
    composition_sha = _records_sha256(
        _COMPOSITION_NAMESPACE,
        tuple(row.record(include_sha256=False) for row in composition),
    )
    duplicate_sha = _records_sha256(
        _CONTROL_NAMESPACE + b"duplicate\0",
        tuple(row.record(include_sha256=False) for row in duplicates),
    )
    overlap_sha = _records_sha256(
        _CONTROL_NAMESPACE + b"overlap\0",
        tuple(row.record(include_sha256=False) for row in overlaps),
    )
    capacity_sha = _records_sha256(
        _CONTROL_NAMESPACE + b"capacity\0",
        tuple(row.record(include_sha256=False) for row in capacities),
    )
    latency, glif_sha, latency_sha = _build_latency_table()
    wta_electrical_controls = _build_wta_electrical_controls(latency)
    if (
        len(wta_electrical_controls) != WTA_ELECTRICAL_CONTROL_COUNT
        or len({row.sha256 for row in wta_electrical_controls})
        != WTA_ELECTRICAL_CONTROL_COUNT
    ):
        raise RuntimeError("exclusive V4 electrical control panel changed")
    wta_electrical_controls_sha = _records_sha256(
        _WTA_ELECTRICAL_CONTROLS_NAMESPACE,
        tuple(
            row.record(include_sha256=False) for row in wta_electrical_controls
        ),
    )

    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": PROTOCOL_STATUS,
        "interface_sha256": INTERFACE_SHA256,
        "atomic_input_sha256": atomic_input_sha,
        "curriculum_sha256": curriculum_sha,
        "split_sha256": split_sha,
        "v3_reference_input_sha256": v3_reference_sha,
        "neutral_sha256": neutral_sha,
        "composition_sha256": composition_sha,
        "duplicate_controls_sha256": duplicate_sha,
        "overlap_controls_sha256": overlap_sha,
        "capacity_controls_sha256": capacity_sha,
        "glif_sha256": glif_sha,
        "latency_table_sha256": latency_sha,
        "wta_topology_sha256": WTA_TOPOLOGY_SHA256,
        "wta_electrical_controls_sha256": wta_electrical_controls_sha,
        "local_rule_sha256": local_rule_sha,
        "group_count": len(groups),
        "lesson_count": len(lessons),
        "fit_count": len(fit_ids),
        "heldout_count": len(heldout_ids),
        "neutral_span_count": len(neutrals),
        "composition_case_count": len(composition),
        "duplicate_control_count": len(duplicates),
        "overlap_control_count": len(overlaps),
        "capacity_control_count": len(capacities),
        "wta_electrical_control_count": len(wta_electrical_controls),
        "public_input_count": PUBLIC_INPUT_COUNT,
        "canonical_input_count": CANONICAL_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "composition_result_count": COMPOSITION_RESULT_COUNT,
        "learner_state_constructed": False,
        "composition_executed": False,
        "execution_surface_present": False,
        "learner_electrical_preflight_executed": False,
    }
    protocol_sha = _namespaced_sha256(_PROTOCOL_NAMESPACE, root_without_protocol)
    protocol = FrozenExclusiveFactorProtocol(
        scope=PROTOCOL_SCOPE,
        status=PROTOCOL_STATUS,
        atomic_groups=groups,
        fit_lesson_sha256s=fit_ids,
        heldout_lesson_sha256s=heldout_ids,
        neutral_spans=neutrals,
        composition_cases=composition,
        duplicate_controls=duplicates,
        overlap_controls=overlaps,
        capacity_controls=capacities,
        latency_table=latency,
        wta_electrical_controls=wta_electrical_controls,
        stop_current_pa=latency[STOP_SCORE].current_pa,
        interface_sha256=INTERFACE_SHA256,
        atomic_input_sha256=atomic_input_sha,
        curriculum_sha256=curriculum_sha,
        split_sha256=split_sha,
        v3_reference_input_sha256=v3_reference_sha,
        neutral_sha256=neutral_sha,
        composition_sha256=composition_sha,
        duplicate_controls_sha256=duplicate_sha,
        overlap_controls_sha256=overlap_sha,
        capacity_controls_sha256=capacity_sha,
        glif_sha256=glif_sha,
        latency_table_sha256=latency_sha,
        wta_topology_sha256=WTA_TOPOLOGY_SHA256,
        wta_electrical_controls_sha256=wta_electrical_controls_sha,
        local_rule_sha256=local_rule_sha,
        protocol_sha256=protocol_sha,
        public_input_count=PUBLIC_INPUT_COUNT,
        canonical_input_count=CANONICAL_INPUT_COUNT,
        protected_input_count=PROTECTED_INPUT_COUNT,
        learning_run_count=LEARNING_RUN_COUNT,
        composition_result_count=COMPOSITION_RESULT_COUNT,
    )
    if protocol.root_record(include_protocol_sha256=False) != root_without_protocol:
        raise RuntimeError("exclusive V4 protocol root construction diverged")
    if any(
        not _is_sha256(value)
        for name, value in dataclasses.asdict(protocol).items()
        if name.endswith("sha256")
    ):
        raise RuntimeError("exclusive V4 protocol contains an invalid root hash")
    return protocol, local_rule_record, local_rule_sha


PROTOCOL, LOCAL_RULE_RECORD, LOCAL_RULE_SHA256 = _build_protocol()


__all__ = [
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "BASKET_DELAY_TICKS",
    "CANONICAL_INPUT_COUNT",
    "CAPACITY_CONTROL_COUNT",
    "COLLISION_DETECTOR_DELAY_TICKS",
    "COLLISION_RELAY_DELAY_TICKS",
    "COLLISION_STOP_DELAY_TICKS",
    "COMPOSITION_CASE_COUNT",
    "COMPOSITION_RESULT_COUNT",
    "DUPLICATE_CONTROL_COUNT",
    "EVENT_CAPACITY",
    "FACTOR_COUNT",
    "FACTOR_REGISTRY",
    "FACTOR_RELAY_DELAY_TICKS",
    "FIT",
    "FIT_COUNT",
    "GROUP_COUNT",
    "HELDOUT",
    "HELDOUT_COUNT",
    "INTERFACE_RECORD",
    "INTERFACE_SHA256",
    "LEARNING_RUN_COUNT",
    "LESSON_COUNT",
    "LOCAL_RULE_RECORD",
    "LOCAL_RULE_SHA256",
    "MARKER_TOKEN",
    "MAX_ACTIVE_INPUT_CELLS",
    "MIN_SCORE_SPIKE_STEP",
    "NEUTRAL_SPAN_COUNT",
    "OVERLAP_CONTROL_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PROTOCOL_STATUS",
    "PUBLIC_INPUT_COUNT",
    "RACE_STEPS",
    "RAW_TO_CURRENT_SCORE_RULE",
    "SCORE_MAX",
    "SCORE_MIN",
    "STOP_RELAY_DELAY_TICKS",
    "STOP_SCORE",
    "TAUGHT_FACTORS",
    "TICK_CAP",
    "TICK_DTYPE",
    "TRIADS",
    "WTA_ELECTRICAL_CONTROL_COUNT",
    "WTA_TOPOLOGY_RECORD",
    "WTA_TOPOLOGY_SHA256",
    "ZERO_SCORE_SPIKE_STEP",
    "AtomicContrastGroup",
    "AtomicLesson",
    "CompositionCase",
    "CompositorControl",
    "FrozenExclusiveFactorProtocol",
    "LatencyEntry",
    "NeutralSpan",
    "ProtocolFactorEvent",
    "SpanPlacement",
    "WTAElectricalControl",
]
