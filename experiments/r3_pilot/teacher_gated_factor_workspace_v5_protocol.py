"""Data-only freeze for the teacher-gated factor workspace V5.

This module exposes immutable protocol records only.  It does not allocate
learner state, fit a learner, resolve a span, or execute composition.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4
from experiments.r3_pilot import prompt_factor_multispan_event_bridge_v3 as v3
from humanbrain.resting_glif_race import CompiledRestingGlifRace

PROTOCOL_SCOPE = "SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_PROTOCOL"
PROTOCOL_STATUS = "FROZEN_DATA_ONLY_NOT_RUN"
FIT = "FIT"
HELDOUT = "HELDOUT"

FACTOR_REGISTRY = v4.FACTOR_REGISTRY
FACTOR_COUNT = len(FACTOR_REGISTRY)
TRIADS = (
    ("add", "less-than", "count"),
    ("absolute", "sum", "minimum"),
    ("maximum", "subtract", "map"),
)
TAUGHT_FACTORS = tuple(factor for triad in TRIADS for factor in triad)
CUE_TOKENS = MappingProxyType({"NEXT": "cue-next-v5", "PREV": "cue-prev-v5"})
TEMPLATE_SPECS = (
    (FIT, "NEXT", 0, 1),
    (FIT, "PREV", 1, 0),
    (FIT, "NEXT", 2, 3),
    (FIT, "PREV", 3, 2),
    (HELDOUT, "PREV", 0, 3),
    (HELDOUT, "NEXT", 3, 0),
)

GROUP_COUNT = 18
LESSON_COUNT = 54
FIT_COUNT = 36
HELDOUT_COUNT = 18

LEXICAL_CELL_COUNT = 512
ASSEMBLY_ACTIVE_CELL_COUNT = 12
ASSEMBLY_MAX_PAIRWISE_OVERLAP = 3

SELECTOR_ADDRESS_COUNT = 8
SELECTOR_CANDIDATE_COUNT = 9
SELECTOR_STOP_CANDIDATE_INDEX = 8
SELECTOR_STOP_SCORE = v4.STOP_SCORE
SELECTOR_ADDRESS_SCORE = v4.STOP_SCORE + 1
SELECTOR_ELECTRICAL_CONTROL_COUNT = 10

NEUTRAL_SPAN_COUNT = 6
COMPOSITION_CASE_COUNT = 18
DUPLICATE_CONTROL_COUNT = 12
OVERLAP_CONTROL_COUNT = 6
CAPACITY_CONTROL_COUNT = 3
EVENT_CAPACITY = 2

TICK_DTYPE = "uint8"
TICK_CAP = 4
FUTURE_STATE_SHAPE = (FACTOR_COUNT, LEXICAL_CELL_COUNT)
FUTURE_STATE_BYTES = FACTOR_COUNT * LEXICAL_CELL_COUNT * 2
SCORE_MIN = -ASSEMBLY_ACTIVE_CELL_COUNT * TICK_CAP
SCORE_MAX = ASSEMBLY_ACTIVE_CELL_COUNT * TICK_CAP
RAW_TO_CURRENT_SCORE_RULE = v4.RAW_TO_CURRENT_SCORE_RULE
STOP_SCORE = v4.STOP_SCORE
RANDOM_LABEL_SEED = 50_005

PUBLIC_INPUT_COUNT = 0
CANONICAL_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
LEARNING_RUN_COUNT = 0
COMPOSITION_RESULT_COUNT = 0

_ATOMIC_LESSON_NAMESPACE = b"teacher-gated-factor-workspace-v5-atomic-lesson\0"
_ATOMIC_INPUT_NAMESPACE = b"teacher-gated-factor-workspace-v5-atomic-inputs\0"
_CURRICULUM_NAMESPACE = b"teacher-gated-factor-workspace-v5-curriculum\0"
_SPLIT_NAMESPACE = b"teacher-gated-factor-workspace-v5-split\0"
_SPENT_INPUT_NAMESPACE = b"teacher-gated-factor-workspace-v5-spent-inputs\0"
_LEXICAL_ASSEMBLY_NAMESPACE = b"teacher-gated-factor-workspace-v5-lexical-assembly\0"
_PROJECTION_NAMESPACE = b"teacher-gated-factor-workspace-v5-projection\0"
_SELECTOR_ADDRESS_NAMESPACE = b"teacher-gated-factor-workspace-v5-selector-address\0"
_SELECTOR_TOPOLOGY_NAMESPACE = b"teacher-gated-factor-workspace-v5-selector-topology\0"
_SELECTOR_CONTROL_NAMESPACE = b"teacher-gated-factor-workspace-v5-selector-control\0"
_SELECTOR_CONTROLS_NAMESPACE = b"teacher-gated-factor-workspace-v5-selector-controls\0"
_SELECTOR_CIRCUIT_NAMESPACE = b"teacher-gated-factor-workspace-v5-selector-circuit\0"
_SELECTOR_FULL_PREFLIGHT_NAMESPACE = b"teacher-gated-factor-workspace-v5-selector-full-preflight\0"
_V4_WTA_REFERENCE_NAMESPACE = b"teacher-gated-factor-workspace-v5-v4-wta-reference\0"
_LOCAL_RULE_NAMESPACE = b"teacher-gated-factor-workspace-v5-local-rule\0"
_CONTROL_PANEL_NAMESPACE = b"teacher-gated-factor-workspace-v5-control-panel\0"
_ATOMIC_GATE_NAMESPACE = b"teacher-gated-factor-workspace-v5-atomic-gate\0"
_TEACHER_SCOUT_CIRCUIT_NAMESPACE = b"teacher-gated-factor-workspace-v5-teacher-scout-circuit\0"
_ELECTRICAL_PREFLIGHT_CONTRACT_NAMESPACE = (
    b"teacher-gated-factor-workspace-v5-electrical-preflight-contract\0"
)
_NEUTRAL_NAMESPACE = b"teacher-gated-factor-workspace-v5-neutral\0"
_COMPOSITION_CASE_NAMESPACE = b"teacher-gated-factor-workspace-v5-composition-case\0"
_COMPOSITION_NAMESPACE = b"teacher-gated-factor-workspace-v5-composition\0"
_COMPOSITOR_CONTROL_NAMESPACE = b"teacher-gated-factor-workspace-v5-compositor-control\0"
_COMPOSITOR_INGRESS_BUFFER_NAMESPACE = (
    b"teacher-gated-factor-workspace-v5-compositor-ingress-buffer\0"
)
_COMPOSITOR_ADMISSION_NAMESPACE = b"teacher-gated-factor-workspace-v5-compositor-admission\0"
_COMPOSITOR_EOS_STATUS_NAMESPACE = b"teacher-gated-factor-workspace-v5-compositor-eos-status\0"
_COMPOSITOR_CIRCUIT_NAMESPACE = b"teacher-gated-factor-workspace-v5-compositor-circuit\0"
_COMPOSITION_EXECUTION_NAMESPACE = b"teacher-gated-factor-workspace-v5-composition-execution\0"
_FUTURE_AUTHENTICATION_NAMESPACE = b"teacher-gated-factor-workspace-v5-future-authentication\0"
_DIRECT_READOUT_CHANNEL_NAMESPACE = b"teacher-gated-factor-workspace-v5-direct-readout-channel\0"
_PERSISTENT_LATCH_PRIMITIVE_NAMESPACE = (
    b"teacher-gated-factor-workspace-v5-persistent-latch-primitive\0"
)
_PROTOCOL_NAMESPACE = b"teacher-gated-factor-workspace-v5-protocol\0"


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


def _freeze_record(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_record(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_record(item) for item in value)
    return value


class _NoLeakStorageReadoutPopulation:
    """Two-compartment physical latch with nondestructive gated reads."""

    __slots__ = ("_readout_pa", "_storage_pa")

    def __init__(self, size: int) -> None:
        if type(size) is not int or size < 1:
            raise ValueError("persistent latch population requires a positive size")
        self._storage_pa = np.zeros(size, dtype=np.float64)
        self._readout_pa = np.zeros(size, dtype=np.float64)

    @property
    def storage_charge_pa(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._storage_pa)

    def step(
        self,
        *,
        set_currents_pa: np.ndarray,
        read_gate_currents_pa: np.ndarray,
        direct_readout_currents_pa: np.ndarray,
        reset_currents_pa: np.ndarray,
    ) -> np.ndarray:
        set_currents = np.asarray(set_currents_pa, dtype=np.float64)
        read_currents = np.asarray(read_gate_currents_pa, dtype=np.float64)
        direct_currents = np.asarray(direct_readout_currents_pa, dtype=np.float64)
        reset_currents = np.asarray(reset_currents_pa, dtype=np.float64)
        expected_shape = self._storage_pa.shape
        if (
            set_currents.shape != expected_shape
            or read_currents.shape != expected_shape
            or direct_currents.shape != expected_shape
            or reset_currents.shape != expected_shape
            or not np.all(np.isfinite(set_currents))
            or not np.all(np.isfinite(read_currents))
            or not np.all(np.isfinite(direct_currents))
            or not np.all(np.isfinite(reset_currents))
        ):
            raise ValueError("persistent latch current shape changed")
        full = float(v4.RELAY_COPY_CURRENT_PA)
        inhibition = float(v4.BASKET_INHIBITION_CURRENT_PA)
        if (
            np.any((set_currents != 0.0) & (set_currents != full))
            or np.any((read_currents != 0.0) & (read_currents != full))
            or np.any((direct_currents != 0.0) & (direct_currents != 2.0 * full))
            or np.any((reset_currents != 0.0) & (reset_currents != inhibition))
        ):
            raise ValueError("persistent latch current domain changed")

        reset = reset_currents == inhibition
        self._storage_pa[reset] = 0.0
        self._readout_pa[reset] = 0.0

        active = ~reset
        storage_snapshot = self._storage_pa.copy()
        gated_storage = np.where(
            active & (read_currents > 0.0),
            storage_snapshot,
            0.0,
        )
        self._readout_pa[active] += (
            read_currents[active] + gated_storage[active] + direct_currents[active]
        )
        fired = active & (self._readout_pa >= 2.0 * float(v4.RELAY_COPY_CURRENT_PA))
        self._readout_pa.fill(0.0)

        self._storage_pa[active] = np.minimum(
            full,
            self._storage_pa[active] + set_currents[active],
        )
        return fired


_DIRECT_READOUT_CHANNEL_RECORD = {
    "purpose": "two-principal collision coincidence without aliasing a storage read gate",
    "direct_synaptic_input_current_pa_hex": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
    "input_current_domain_pa_hex": (
        (0.0).hex(),
        float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
    ),
    "readout_threshold_pa_hex": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
    "principal_input_count": 2,
    "current_per_principal_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    "storage_coupling": False,
    "read_gate_alias": False,
    "invalid_or_overdrive_rejected_before_state_mutation": True,
}
DIRECT_READOUT_CHANNEL_SHA256 = _namespaced_sha256(
    _DIRECT_READOUT_CHANNEL_NAMESPACE,
    _DIRECT_READOUT_CHANNEL_RECORD,
)
DIRECT_READOUT_CHANNEL_RECORD = _freeze_record(_DIRECT_READOUT_CHANNEL_RECORD)


_PERSISTENT_LATCH_PRIMITIVE_RECORD = {
    "direct_readout_channel_sha256": DIRECT_READOUT_CHANNEL_SHA256,
    "cell_model": "two-compartment no-leak storage plus gated readout",
    "storage_set_current_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    "storage_charge_cap_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    "storage_to_readout_current_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    "read_gate_current_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    "set_current_domain_pa_hex": ((0.0).hex(), float(v4.RELAY_COPY_CURRENT_PA).hex()),
    "read_gate_current_domain_pa_hex": ((0.0).hex(), float(v4.RELAY_COPY_CURRENT_PA).hex()),
    "direct_readout_current_domain_pa_hex": (
        (0.0).hex(),
        float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
    ),
    "readout_threshold_pa_hex": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
    "reset_inhibition_pa_hex": float(v4.BASKET_INHIBITION_CURRENT_PA).hex(),
    "idle_leak_pa_hex": (0.0).hex(),
    "set_saturation": "storage charge clamps at 5000 pA",
    "read_semantics": (
        "a 5000 pA gate couples the unchanged 5000 pA storage charge into a "
        "separate 10000 pA readout; the readout resets after every observation"
    ),
    "direct_readout_semantics": (
        "a separate domain-restricted 10000 pA direct synaptic channel represents "
        "exactly two simultaneous 5000 pA principal inputs; it never aliases a read gate"
    ),
    "reset_semantics": (
        "-5000 pA reset inhibition followed by an explicit zero clamp clears both "
        "compartments and dominates simultaneous set/read/direct inputs"
    ),
    "operation_precedence": (
        "reset inhibition and zero clamp dominate the addressed storage/readout",
        "read the pre-set storage snapshot through a separate ephemeral compartment",
        "add only an explicit 0-or-10000 pA direct multi-principal readout input",
        "clear every readout membrane after observation",
        "apply set current to storage and clamp at the finite charge cap",
        "retain storage charge exactly across an idle tick",
    ),
    "mandatory_preflight_controls": {
        "idle_persistence": {
            "set_tick": 0,
            "idle_tick_count": 1024,
            "read_tick": 1025,
            "expected_storage_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "expected_readout_spikes": 1,
        },
        "nondestructive_repeated_read": {
            "read_ticks": (0, 1),
            "expected_readout_spikes": (1, 1),
            "expected_final_storage_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        },
        "reset_dominance": {
            "simultaneous_set_read_reset": True,
            "expected_readout_spikes": 0,
            "expected_final_storage_pa_hex": (0.0).hex(),
        },
        "current_domain_rejection": {
            "invalid_set_current_pa_hex": (2_500.0).hex(),
            "invalid_read_gate_current_pa_hex": (10_000.0).hex(),
            "invalid_direct_readout_current_pa_hex": (15_000.0).hex(),
            "expected_mutation_before_error": False,
        },
        "direct_two_principal_collision": {
            "direct_readout_current_pa_hex": (10_000.0).hex(),
            "expected_storage_pa_hex": (0.0).hex(),
            "expected_readout_spikes": 1,
        },
    },
    "nondestructive_repeated_read": True,
    "host_boolean_or_semantic_state": False,
    "learner_state": False,
}
PERSISTENT_LATCH_PRIMITIVE_SHA256 = _namespaced_sha256(
    _PERSISTENT_LATCH_PRIMITIVE_NAMESPACE,
    _PERSISTENT_LATCH_PRIMITIVE_RECORD,
)
PERSISTENT_LATCH_PRIMITIVE_RECORD = _freeze_record(_PERSISTENT_LATCH_PRIMITIVE_RECORD)


@dataclass(frozen=True, slots=True)
class AtomicLesson:
    group_index: int
    triad_index: int
    template_index: int
    target_index: int
    direction: str
    cue_token: str
    cue_slot: int
    selected_slot: int
    tokens: tuple[str, str, str, str]
    factor: str
    split: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "group_index": self.group_index,
            "triad_index": self.triad_index,
            "template_index": self.template_index,
            "target_index": self.target_index,
            "direction": self.direction,
            "cue_token": self.cue_token,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
            "tokens": self.tokens,
            "factor": self.factor,
            "split": self.split,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class AtomicContrastGroup:
    group_index: int
    triad_index: int
    template_index: int
    direction: str
    cue_slot: int
    selected_slot: int
    split: str
    lessons: tuple[AtomicLesson, AtomicLesson, AtomicLesson]


@dataclass(frozen=True, slots=True)
class LexicalAssembly:
    token: str
    salt: int
    cells: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "token": self.token,
            "salt": self.salt,
            "cells": self.cells,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SelectorAddress:
    address_index: int
    direction: str
    cue_token: str
    cue_slot: int
    phase_cell: int
    selected_slot: int
    selected_slot_latch_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "address_index": self.address_index,
            "direction": self.direction,
            "cue_token": self.cue_token,
            "cue_slot": self.cue_slot,
            "phase_cell": self.phase_cell,
            "selected_slot": self.selected_slot,
            "selected_slot_latch_cell": self.selected_slot_latch_cell,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SelectorElectricalControl:
    control_index: int
    name: str
    cue_events: tuple[tuple[str, int], ...]
    address_event_cells: tuple[int, ...]
    principal_current_scores: tuple[int, ...]
    principal_currents_pa: tuple[float, ...]
    principal_first_spike_step: int
    principal_first_spike_tier: tuple[int, ...]
    simultaneous_spike_candidates: int
    decision: str
    selected_slot: int | None
    full_circuit_execution_deferred: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "control_index": self.control_index,
            "name": self.name,
            "cue_events": self.cue_events,
            "address_event_cells": self.address_event_cells,
            "principal_current_scores": self.principal_current_scores,
            "principal_currents_pa_hex": tuple(
                current.hex() for current in self.principal_currents_pa
            ),
            "principal_first_spike_step": self.principal_first_spike_step,
            "principal_first_spike_tier": self.principal_first_spike_tier,
            "simultaneous_spike_candidates": self.simultaneous_spike_candidates,
            "decision": self.decision,
            "selected_slot": self.selected_slot,
            "full_circuit_execution_deferred": self.full_circuit_execution_deferred,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


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
    expected_eos_status_latch: str | None
    expected_factors: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "control_index": self.control_index,
            "kind": self.kind,
            "events": tuple(event.record() for event in self.events),
            "expected_status": self.expected_status,
            "expected_eos_status_latch": self.expected_eos_status_latch,
            "expected_factors": self.expected_factors,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenTeacherGatedFactorProtocol:
    scope: str
    status: str
    atomic_groups: tuple[AtomicContrastGroup, ...]
    fit_lesson_sha256s: tuple[str, ...]
    heldout_lesson_sha256s: tuple[str, ...]
    lexical_assemblies: tuple[LexicalAssembly, ...]
    selector_electrical_controls: tuple[SelectorElectricalControl, ...]
    factor_wta_electrical_controls: tuple[v4.WTAElectricalControl, ...]
    neutral_spans: tuple[NeutralSpan, ...]
    composition_cases: tuple[CompositionCase, ...]
    duplicate_controls: tuple[CompositorControl, ...]
    overlap_controls: tuple[CompositorControl, ...]
    capacity_controls: tuple[CompositorControl, ...]
    atomic_input_sha256: str
    curriculum_sha256: str
    split_sha256: str
    spent_input_reference_sha256: str
    projection_sha256: str
    direct_readout_channel_sha256: str
    persistent_latch_primitive_sha256: str
    selector_topology_sha256: str
    selector_electrical_controls_sha256: str
    selector_circuit_sha256: str
    selector_full_preflight_sha256: str
    v4_wta_reference_sha256: str
    local_rule_sha256: str
    control_panel_sha256: str
    atomic_gate_sha256: str
    teacher_scout_circuit_sha256: str
    electrical_preflight_contract_sha256: str
    neutral_sha256: str
    composition_sha256: str
    duplicate_controls_sha256: str
    overlap_controls_sha256: str
    capacity_controls_sha256: str
    compositor_ingress_buffer_sha256: str
    compositor_admission_sha256: str
    compositor_eos_status_sha256: str
    compositor_circuit_sha256: str
    composition_execution_sha256: str
    future_authentication_sha256: str
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
            "atomic_input_sha256": self.atomic_input_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "split_sha256": self.split_sha256,
            "spent_input_reference_sha256": self.spent_input_reference_sha256,
            "projection_sha256": self.projection_sha256,
            "direct_readout_channel_sha256": self.direct_readout_channel_sha256,
            "persistent_latch_primitive_sha256": (self.persistent_latch_primitive_sha256),
            "selector_topology_sha256": self.selector_topology_sha256,
            "selector_electrical_controls_sha256": (self.selector_electrical_controls_sha256),
            "selector_circuit_sha256": self.selector_circuit_sha256,
            "selector_full_preflight_sha256": (self.selector_full_preflight_sha256),
            "v4_wta_reference_sha256": self.v4_wta_reference_sha256,
            "local_rule_sha256": self.local_rule_sha256,
            "control_panel_sha256": self.control_panel_sha256,
            "atomic_gate_sha256": self.atomic_gate_sha256,
            "teacher_scout_circuit_sha256": self.teacher_scout_circuit_sha256,
            "electrical_preflight_contract_sha256": (self.electrical_preflight_contract_sha256),
            "neutral_sha256": self.neutral_sha256,
            "composition_sha256": self.composition_sha256,
            "duplicate_controls_sha256": self.duplicate_controls_sha256,
            "overlap_controls_sha256": self.overlap_controls_sha256,
            "capacity_controls_sha256": self.capacity_controls_sha256,
            "compositor_ingress_buffer_sha256": (self.compositor_ingress_buffer_sha256),
            "compositor_admission_sha256": self.compositor_admission_sha256,
            "compositor_eos_status_sha256": self.compositor_eos_status_sha256,
            "compositor_circuit_sha256": self.compositor_circuit_sha256,
            "composition_execution_sha256": self.composition_execution_sha256,
            "future_authentication_sha256": self.future_authentication_sha256,
            "group_count": len(self.atomic_groups),
            "lesson_count": len(self.lessons),
            "fit_count": len(self.fit_lesson_sha256s),
            "heldout_count": len(self.heldout_lesson_sha256s),
            "lexical_assembly_count": len(self.lexical_assemblies),
            "selector_electrical_control_count": len(self.selector_electrical_controls),
            "factor_wta_electrical_control_count": len(self.factor_wta_electrical_controls),
            "neutral_span_count": len(self.neutral_spans),
            "composition_case_count": len(self.composition_cases),
            "duplicate_control_count": len(self.duplicate_controls),
            "overlap_control_count": len(self.overlap_controls),
            "capacity_control_count": len(self.capacity_controls),
            "public_input_count": self.public_input_count,
            "canonical_input_count": self.canonical_input_count,
            "protected_input_count": self.protected_input_count,
            "learning_run_count": self.learning_run_count,
            "composition_result_count": self.composition_result_count,
            "learner_state_constructed": False,
            "learning_executed": False,
            "composition_executed": False,
            "execution_surface_present": False,
            "private_mechanism_primitive_present": True,
            "private_mechanism_primitive_instantiated_at_import": False,
            "learner_or_full_circuit_execution_surface_present": False,
            "selector_principal_controls_executed": True,
            "factor_principal_controls_executed": True,
            "full_selected_content_circuit_executed": False,
            "full_factor_circuit_executed": False,
            "teacher_scout_circuit_executed": False,
            "compositor_circuit_executed": False,
            "composition_attempt_count": 0,
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


def _build_selector_addresses() -> tuple[SelectorAddress, ...]:
    rows: list[SelectorAddress] = []
    for direction in ("NEXT", "PREV"):
        delta = 1 if direction == "NEXT" else -1
        for cue_slot in range(4):
            address_index = len(rows)
            selected_slot = (cue_slot + delta) % 4
            record = {
                "address_index": address_index,
                "direction": direction,
                "cue_token": CUE_TOKENS[direction],
                "cue_slot": cue_slot,
                "phase_cell": cue_slot,
                "selected_slot": selected_slot,
                "selected_slot_latch_cell": selected_slot,
            }
            rows.append(
                SelectorAddress(
                    **record,
                    sha256=_namespaced_sha256(_SELECTOR_ADDRESS_NAMESPACE, record),
                )
            )
    return tuple(rows)


SELECTOR_ADDRESSES = _build_selector_addresses()

_SELECTOR_TOPOLOGY_RECORD = {
    "direct_readout_channel_sha256": DIRECT_READOUT_CHANNEL_SHA256,
    "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
    "phase_cells": 4,
    "coincidence_address_cells": SELECTOR_ADDRESS_COUNT,
    "address_event_counter_cells": 1,
    "address_current_storage_compartments": SELECTOR_ADDRESS_COUNT,
    "address_current_readout_compartments": SELECTOR_ADDRESS_COUNT,
    "address_candidates": tuple(row.record(include_sha256=False) for row in SELECTOR_ADDRESSES),
    "selected_slot_latch_cells": 4,
    "selected_slot_storage_compartments": 4,
    "selected_slot_readout_compartments": 4,
    "selected_slot_latch_recurrence": (
        "none: a unique relay sets no-leak storage; the matching replay phase "
        "nondestructively gates its separate readout compartment"
    ),
    "lexical_replay_relay_cells": LEXICAL_CELL_COUNT,
    "lexical_replay_rule": (
        "replay every lexical assembly; the matching phase plus selected-slot latch "
        "disinhibits only its 12-cell trace"
    ),
    "transition_bank_cells": 0,
    "principal_address_cells": SELECTOR_ADDRESS_COUNT,
    "principal_stop_cells": 1,
    "basket_inhibitory_cells": 1,
    "collision_cells": 1,
    "delayed_address_relays": SELECTOR_ADDRESS_COUNT,
    "stop_relays": 1,
    "address_candidate_count": SELECTOR_CANDIDATE_COUNT,
    "stop_candidate_index": SELECTOR_STOP_CANDIDATE_INDEX,
    "address_score": SELECTOR_ADDRESS_SCORE,
    "stop_score": SELECTOR_STOP_SCORE,
    "zero_address_events": "STOP",
    "multiple_address_events": "COLLISION_STOP",
    "tie_semantics": "any leading address tie enters the physical collision STOP path",
    "basket_delay_ticks": v4.BASKET_DELAY_TICKS,
    "address_relay_delay_ticks": v4.FACTOR_RELAY_DELAY_TICKS,
    "stop_relay_delay_ticks": v4.STOP_RELAY_DELAY_TICKS,
    "collision_detector_delay_ticks": v4.COLLISION_DETECTOR_DELAY_TICKS,
    "collision_relay_delay_ticks": v4.COLLISION_RELAY_DELAY_TICKS,
    "collision_stop_delay_ticks": v4.COLLISION_STOP_DELAY_TICKS,
    "tick_phase_precedence": (
        "scheduled basket inhibition",
        "principal threshold evaluation",
        "collision and address/STOP relay arrivals",
    ),
    "winner_cell_use": (
        "permitted only for a single address tier; zero/tied tiers are consumed by "
        "the physical STOP/collision projection"
    ),
    "host_selected_token_materialization": False,
    "host_slot_selection_or_tie_break": False,
    "full_selected_content_circuit_deferred": True,
    "full_selector_circuit_deferred": True,
    "selector_principal_control_adapter": "CompiledRestingGlifRace.race_many",
    "selector_principal_control_shape": (
        SELECTOR_ELECTRICAL_CONTROL_COUNT,
        SELECTOR_CANDIDATE_COUNT,
    ),
    "latency_table_sha256": v4.PROTOCOL.latency_table_sha256,
    "electrical_control_scope": (
        "the 9-way principal tiers execute in this protocol; phase/coincidence, "
        "basket/collision/address relays, recurrent latches, and lexical replay "
        "execute in learner electrical preflight"
    ),
}
SELECTOR_TOPOLOGY_SHA256 = _namespaced_sha256(
    _SELECTOR_TOPOLOGY_NAMESPACE,
    _SELECTOR_TOPOLOGY_RECORD,
)
SELECTOR_TOPOLOGY_RECORD = _freeze_record(_SELECTOR_TOPOLOGY_RECORD)


_SELECTOR_SCAN_TICKS = (1, 2, 3, 4)
_SELECTOR_ADDRESS_SCAN_CLOSE_TICK = 5
_SELECTOR_PRINCIPAL_START_TICK = 6
_SELECTOR_UNIQUE_PRINCIPAL_SPIKE_TICK = (
    _SELECTOR_PRINCIPAL_START_TICK
    + v4.PROTOCOL.latency_table[SELECTOR_ADDRESS_SCORE].first_spike_step
)
_SELECTOR_UNIQUE_ADDRESS_RELAY_TICK = (
    _SELECTOR_UNIQUE_PRINCIPAL_SPIKE_TICK + v4.FACTOR_RELAY_DELAY_TICKS
)
_SELECTOR_CONTENT_PHASE_OFFSETS = (1, 3, 5, 7)
_SELECTOR_CONTENT_GATE_OFFSETS = (2, 4, 6, 8)
_SELECTOR_CONTENT_CLOSE_RESET_OFFSET = 9

_SELECTOR_CIRCUIT_RECORD = {
    "direct_readout_channel_sha256": DIRECT_READOUT_CHANNEL_SHA256,
    "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
    "cell_model": (
        "rooted no-leak storage/readout compartments and custom coincidence/reset "
        "neurons around the frozen fitted-GLIF address principals"
    ),
    "cell_counts": {
        "phase": 4,
        "address_coincidence": SELECTOR_ADDRESS_COUNT,
        "address_current_latch": SELECTOR_ADDRESS_COUNT,
        "address_current_storage_compartment": SELECTOR_ADDRESS_COUNT,
        "address_current_readout_compartment": SELECTOR_ADDRESS_COUNT,
        "address_event_counter": 1,
        "address_event_counter_storage_compartment": 1,
        "address_event_counter_readout_compartment": 1,
        "address_principal": SELECTOR_ADDRESS_COUNT,
        "stop_principal": 1,
        "basket": 1,
        "collision": 1,
        "collision_storage_compartment": 1,
        "collision_readout_compartment": 1,
        "address_relay": SELECTOR_ADDRESS_COUNT,
        "stop_relay": 1,
        "selected_slot_latch": 4,
        "selected_slot_storage_compartment": 4,
        "selected_slot_readout_compartment": 4,
        "lexical_replay_relay": LEXICAL_CELL_COUNT,
        "reset": 1,
        "transition_bank": 0,
    },
    "current_pa_hex": {
        "full": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "reset_inhibition": float(v4.BASKET_INHIBITION_CURRENT_PA).hex(),
        "address_coincidence_threshold": float(
            v4.RELAY_COPY_CURRENT_PA * (ASSEMBLY_ACTIVE_CELL_COUNT + 1)
        ).hex(),
        "address_event_counter_threshold": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "persistent_storage_set": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "persistent_read_gate": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "persistent_readout_threshold": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "direct_two_principal_collision_readout": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "selected_slot_latch_threshold": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "lexical_replay_threshold": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "inactive_address_principal": v4.PROTOCOL.latency_table[0].current_pa.hex(),
        "active_address_principal": v4.PROTOCOL.latency_table[
            SELECTOR_ADDRESS_SCORE
        ].current_pa.hex(),
        "tonic_stop_principal": v4.PROTOCOL.latency_table[SELECTOR_STOP_SCORE].current_pa.hex(),
    },
    "thresholds_pa_hex": {
        "phase": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "address_coincidence": float(
            v4.RELAY_COPY_CURRENT_PA * (ASSEMBLY_ACTIVE_CELL_COUNT + 1)
        ).hex(),
        "address_current_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
        "address_event_counter": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "selected_slot_latch": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "lexical_replay_relay": float(v4.RELAY_COPY_CURRENT_PA * 2).hex(),
        "reset": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    },
    "address_coincidence_rule": (
        "each of 12 matching cue-assembly axons plus the one-hot phase axon "
        "contributes 5000 pA, so only the exact 13-input conjunction reaches the "
        "65000 pA address threshold"
    ),
    "address_false_positive_bound": (
        "a noncue assembly overlaps in at most 3 cells; even with the phase axon "
        "its maximum address drive is 20000 pA"
    ),
    "address_per_event_reset_rule": (
        "each cue event is evaluated serially inside its scan tick; immediately "
        "after its address spike is copied, -5000 pA hard-reset inhibition and a "
        "zero clamp clear all eight address coincidence membranes"
    ),
    "address_event_counter_rule": (
        "each serial address spike first gates a nondestructive read of the "
        "seen-address storage and then sets that storage to 5000 pA; the first "
        "event is silent and every later event reaches the 10000 pA readout"
    ),
    "event_counter_operation_order": (
        "read pre-event seen-address storage",
        "emit repeat spike only when an earlier address event was stored",
        "set/saturate seen-address storage for the current event",
        "copy a repeat spike into collision storage",
    ),
    "same_address_collision_rule": (
        "the second same-address event sets 5000 pA no-leak collision storage; "
        "one later leading principal supplies the 5000 pA read gate at the frozen "
        "collision tick, reaching the 10000 pA readout threshold"
    ),
    "multi_address_collision_rule": (
        "two leading principals supply the separately domain-restricted 10000 pA "
        "direct synaptic input to the collision readout; a stored repeat flag may "
        "also gate its storage but cannot change the STOP result"
    ),
    "address_current_latch_rule": (
        "an address coincidence sets its same-index storage to 5000 pA; at every "
        "principal tick from 6 through 605 a 5000 pA gate nondestructively reads "
        "that storage, and each readout spike opens the fixed score-13 current "
        "synapse into only the same-index fitted-GLIF principal for that tick"
    ),
    "address_current_principal_read_window": (6, 605, "inclusive"),
    "address_current_principal_read_count": v4.RACE_STEPS,
    "tonic_stop_current_window": (6, 605, "inclusive"),
    "decision_rule": (
        "zero address events route through direct tonic STOP; two or more events "
        "or a multi-address leading tier route through collision STOP; exactly one "
        "event and one leading address relay may set a selected-slot latch"
    ),
    "within_tick_precedence": (
        "reset inhibition and scheduled external assembly spikes",
        "phase plus cue-assembly address coincidence for each cue event",
        "address event-count/current latches then per-event address hard reset",
        "principal, basket, collision, and delayed-relay threshold evaluation",
        "selected-slot latch and lexical replay relay threshold evaluation",
    ),
    "lesson_reset_tick": 0,
    "scan_ticks": _SELECTOR_SCAN_TICKS,
    "scan_slot_order": (0, 1, 2, 3),
    "address_scan_close_tick": _SELECTOR_ADDRESS_SCAN_CLOSE_TICK,
    "principal_start_tick": _SELECTOR_PRINCIPAL_START_TICK,
    "principal_step_count": v4.RACE_STEPS,
    "unique_address_principal_spike_tick": _SELECTOR_UNIQUE_PRINCIPAL_SPIKE_TICK,
    "unique_address_relay_tick": _SELECTOR_UNIQUE_ADDRESS_RELAY_TICK,
    "selected_slot_latch_set_current_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    "selected_slot_latch_rule": (
        "hard-reset all four storage/readout pairs after address scan; only the "
        "unique delayed address relay sets one 5000 pA storage compartment; during "
        "replay only the matching phase supplies the 5000 pA nondestructive read "
        "gate to its separate 10000 pA readout"
    ),
    "selected_latches_reset_between_scan_and_relay": True,
    "phase_to_latch_axons_enabled_during_scan": False,
    "phase_to_latch_axons_enabled_during_replay": True,
    "content_phase_offsets": _SELECTOR_CONTENT_PHASE_OFFSETS,
    "content_gate_offsets": _SELECTOR_CONTENT_GATE_OFFSETS,
    "content_replay_slot_order": (0, 1, 2, 3),
    "content_close_reset_offset": _SELECTOR_CONTENT_CLOSE_RESET_OFFSET,
    "lexical_relay_rule": (
        "one tick after a replay phase makes the selected latch spike, the held "
        "token assembly contributes 5000 pA on its 12 active axons and the latch "
        "gate contributes 5000 pA to all 512 relays; exactly 12 reach 10000 pA"
    ),
    "nonselected_lexical_drive_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    "lexical_relay_phase_reset": (
        "after each slot gate tick, hard reset all 512 lexical relays before the next replay phase"
    ),
    "reset_rule": (
        "the one-hot reset neuron delivers -5000 pA to every phase, cue, address, "
        "persistent storage/readout, latch, and lexical-relay compartment; an "
        "explicit zero clamp then clears every lesson-local membrane"
    ),
    "host_selected_token_materialization": False,
    "host_slot_selection_or_tie_break": False,
    "full_circuit_execution_deferred": True,
}
SELECTOR_CIRCUIT_SHA256 = _namespaced_sha256(
    _SELECTOR_CIRCUIT_NAMESPACE,
    _SELECTOR_CIRCUIT_RECORD,
)
SELECTOR_CIRCUIT_RECORD = _freeze_record(_SELECTOR_CIRCUIT_RECORD)


def _build_selector_electrical_controls() -> tuple[SelectorElectricalControl, ...]:
    specs: list[
        tuple[
            str,
            tuple[tuple[str, int], ...],
            tuple[int, ...],
            str,
            int | None,
        ]
    ] = []
    for address in SELECTOR_ADDRESSES:
        specs.append(
            (
                f"{address.direction.lower()}-slot-{address.cue_slot}",
                ((address.direction, address.cue_slot),),
                (address.address_index,),
                "SELECTED_SLOT",
                address.selected_slot,
            )
        )
    specs.extend(
        (
            ("no-cue-stop", (), (), "STOP", None),
            (
                "incompatible-two-cue-collision-stop",
                (("NEXT", 0), ("PREV", 0)),
                (0, 4),
                "COLLISION_STOP",
                None,
            ),
        )
    )
    score_rows = tuple(
        (
            *(
                SELECTOR_ADDRESS_SCORE if cell in address_events else 0
                for cell in range(SELECTOR_ADDRESS_COUNT)
            ),
            SELECTOR_STOP_SCORE,
        )
        for _name, _cues, address_events, _decision, _slot in specs
    )
    table = v4.PROTOCOL.latency_table
    current_rows = tuple(tuple(table[score].current_pa for score in row) for row in score_rows)
    races = CompiledRestingGlifRace.compile(step_count=v4.RACE_STEPS).race_many(
        np.asarray(current_rows, dtype=np.float64),
        require_unique=False,
    )
    controls: list[SelectorElectricalControl] = []
    for control_index, (specification, scores, currents, race) in enumerate(
        zip(specs, score_rows, current_rows, races, strict=True)
    ):
        name, cue_events, address_events, decision, selected_slot = specification
        tier = tuple(
            cell
            for cell, score in enumerate(scores)
            if table[score].first_spike_step == race.spike_step
        )
        expected_tier = address_events if address_events else (SELECTOR_STOP_CANDIDATE_INDEX,)
        if tier != expected_tier or race.simultaneous_spike_candidates != len(tier):
            raise RuntimeError("teacher-gated V5 selector principal tier changed")
        record = {
            "control_index": control_index,
            "name": name,
            "cue_events": cue_events,
            "address_event_cells": address_events,
            "principal_current_scores": scores,
            "principal_currents_pa_hex": tuple(current.hex() for current in currents),
            "principal_first_spike_step": race.spike_step,
            "principal_first_spike_tier": tier,
            "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
            "decision": decision,
            "selected_slot": selected_slot,
            "full_circuit_execution_deferred": True,
        }
        controls.append(
            SelectorElectricalControl(
                control_index=control_index,
                name=name,
                cue_events=cue_events,
                address_event_cells=address_events,
                principal_current_scores=scores,
                principal_currents_pa=currents,
                principal_first_spike_step=race.spike_step,
                principal_first_spike_tier=tier,
                simultaneous_spike_candidates=race.simultaneous_spike_candidates,
                decision=decision,
                selected_slot=selected_slot,
                full_circuit_execution_deferred=True,
                sha256=_namespaced_sha256(_SELECTOR_CONTROL_NAMESPACE, record),
            )
        )
    return tuple(controls)


def _selector_content_trace(
    *,
    slot_tokens: tuple[str, str, str, str],
    selected_slot: int,
    decision_tick: int,
    assembly_by_token: dict[str, LexicalAssembly],
) -> dict[str, object]:
    if (
        len(slot_tokens) != 4
        or type(selected_slot) is not int
        or not 0 <= selected_slot < 4
        or any(token not in assembly_by_token for token in slot_tokens)
    ):
        raise ValueError("selector content fixture changed")
    phase_ticks = tuple(decision_tick + offset for offset in _SELECTOR_CONTENT_PHASE_OFFSETS)
    gate_ticks = tuple(decision_tick + offset for offset in _SELECTOR_CONTENT_GATE_OFFSETS)
    replay_events: list[dict[str, object]] = []
    for slot, (token, phase_tick, gate_tick) in enumerate(
        zip(slot_tokens, phase_ticks, gate_ticks, strict=True)
    ):
        assembly = assembly_by_token[token]
        replay_events.append(
            {
                "slot": slot,
                "phase_tick": phase_tick,
                "gate_tick": gate_tick,
                "token": token,
                "assembly_sha256": assembly.sha256,
                "assembly_cells": assembly.cells,
                "relay_spike_cells": (assembly.cells if slot == selected_slot else ()),
            }
        )
    selected_token = slot_tokens[selected_slot]
    selected_assembly = assembly_by_token[selected_token]
    return {
        "slot_tokens": slot_tokens,
        "selected_token": selected_token,
        "content_phase_ticks": phase_ticks,
        "content_gate_ticks": gate_ticks,
        "content_replay_events": tuple(replay_events),
        "latch_spike_tick": phase_ticks[selected_slot],
        "selected_lexical_relay_tick": gate_ticks[selected_slot],
        "selected_lexical_relay_cells": selected_assembly.cells,
        "replay_close_reset_tick": (decision_tick + _SELECTOR_CONTENT_CLOSE_RESET_OFFSET),
    }


def _build_selector_full_preflight_record(
    assemblies: tuple[LexicalAssembly, ...],
    controls: tuple[SelectorElectricalControl, ...],
) -> tuple[object, str]:
    assembly_by_token = {assembly.token: assembly for assembly in assemblies}
    address_by_index = {address.address_index: address for address in SELECTOR_ADDRESSES}
    probe_tokens = tuple(
        sorted(
            token
            for token in assembly_by_token
            if token not in TAUGHT_FACTORS and token not in CUE_TOKENS.values()
        )
    )
    if len(probe_tokens) < 3:
        raise RuntimeError("selector physical content probe panel changed")
    rows: list[dict[str, object]] = []
    phase_spikes = tuple(
        (tick, slot) for tick, slot in zip(_SELECTOR_SCAN_TICKS, range(4), strict=True)
    )
    for control in controls:
        principal_tick = _SELECTOR_PRINCIPAL_START_TICK + control.principal_first_spike_step
        basket_tick = principal_tick + v4.BASKET_DELAY_TICKS
        collision_tick = (
            principal_tick + v4.COLLISION_DETECTOR_DELAY_TICKS
            if control.simultaneous_spike_candidates > 1
            else None
        )
        if control.decision == "SELECTED_SLOT":
            decision_tick = principal_tick + v4.FACTOR_RELAY_DELAY_TICKS
        elif control.decision == "STOP":
            decision_tick = principal_tick + v4.STOP_RELAY_DELAY_TICKS
        else:
            decision_tick = principal_tick + v4.COLLISION_STOP_DELAY_TICKS
        cue_assembly_events = tuple(
            {
                "tick": _SELECTOR_SCAN_TICKS[cue_slot],
                "direction": direction,
                "cue_slot": cue_slot,
                "serial_event_index": event_index,
                "token": CUE_TOKENS[direction],
                "assembly_sha256": assembly_by_token[CUE_TOKENS[direction]].sha256,
                "assembly_cells": assembly_by_token[CUE_TOKENS[direction]].cells,
            }
            for event_index, (direction, cue_slot) in enumerate(control.cue_events)
        )
        counter_events = tuple(
            {
                "tick": _SELECTOR_SCAN_TICKS[cue_slot],
                "serial_event_index": event_index,
                "read_pre_event_storage": True,
                "repeat_readout_spike": event_index > 0,
                "set_storage_after_read": True,
            }
            for event_index, (_direction, cue_slot) in enumerate(control.cue_events)
        )
        collision_storage_set_tick = counter_events[1]["tick"] if len(counter_events) > 1 else None
        address_spikes = tuple(
            (
                _SELECTOR_SCAN_TICKS[address_by_index[cell].cue_slot],
                cell,
            )
            for cell in control.address_event_cells
        )
        if control.selected_slot is None:
            slot_tokens: tuple[str, ...] = ()
            selected_token = None
            selected_cells: tuple[int, ...] = ()
            latch_set_tick = None
            latch_spike_tick = None
            replay_phase_ticks: tuple[int, ...] = ()
            replay_gate_ticks: tuple[int, ...] = ()
            content_replay_events: tuple[dict[str, object], ...] = ()
            selected_relay_tick = None
            replay_close_reset_tick = decision_tick + 1
        else:
            address = address_by_index[control.control_index]
            physical_slots: list[str | None] = [None, None, None, None]
            physical_slots[address.cue_slot] = address.cue_token
            fill_slots = [slot for slot, token in enumerate(physical_slots) if token is None]
            for probe_index, slot in enumerate(fill_slots):
                physical_slots[slot] = probe_tokens[
                    (control.control_index * 3 + probe_index) % len(probe_tokens)
                ]
            if any(token is None for token in physical_slots):
                raise RuntimeError("selector physical slot fixture changed")
            slot_tokens = tuple(physical_slots)  # type: ignore[assignment]
            latch_set_tick = decision_tick
            content_trace = _selector_content_trace(
                slot_tokens=slot_tokens,
                selected_slot=control.selected_slot,
                decision_tick=decision_tick,
                assembly_by_token=assembly_by_token,
            )
            selected_token = content_trace["selected_token"]
            selected_cells = content_trace["selected_lexical_relay_cells"]
            replay_phase_ticks = content_trace["content_phase_ticks"]
            replay_gate_ticks = content_trace["content_gate_ticks"]
            content_replay_events = content_trace["content_replay_events"]
            latch_spike_tick = content_trace["latch_spike_tick"]
            selected_relay_tick = content_trace["selected_lexical_relay_tick"]
            replay_close_reset_tick = content_trace["replay_close_reset_tick"]
        rows.append(
            {
                "control_sha256": control.sha256,
                "name": control.name,
                "cue_events": control.cue_events,
                "address_event_count": len(control.cue_events),
                "phase_spikes": phase_spikes,
                "cue_assembly_events": cue_assembly_events,
                "event_counter_read_then_set_events": counter_events,
                "collision_storage_set_tick": collision_storage_set_tick,
                "collision_storage_charge_pa_hex": (
                    float(v4.RELAY_COPY_CURRENT_PA).hex()
                    if collision_storage_set_tick is not None
                    else (0.0).hex()
                ),
                "collision_read_gate_tick": (principal_tick + v4.COLLISION_DETECTOR_DELAY_TICKS),
                "address_coincidence_spikes": address_spikes,
                "address_hard_reset_count": len(control.cue_events),
                "address_current_latches_set": control.address_event_cells,
                "principal_start_tick": _SELECTOR_PRINCIPAL_START_TICK,
                "principal_first_spike_tick": principal_tick,
                "principal_first_spike_tier": control.principal_first_spike_tier,
                "basket_tick": basket_tick,
                "collision_tick": collision_tick,
                "decision_tick": decision_tick,
                "decision": control.decision,
                "selected_slot": control.selected_slot,
                "slot_tokens": slot_tokens,
                "selected_token": selected_token,
                "latch_set_tick": latch_set_tick,
                "latch_spike_tick": latch_spike_tick,
                "content_phase_ticks": replay_phase_ticks,
                "content_gate_ticks": replay_gate_ticks,
                "content_replay_events": content_replay_events,
                "selected_lexical_relay_tick": selected_relay_tick,
                "selected_lexical_relay_cells": selected_cells,
                "replay_close_reset_tick": replay_close_reset_tick,
                "post_reset_active_cell_count": 0,
            }
        )
    base = controls[0]
    principal_tick = _SELECTOR_PRINCIPAL_START_TICK + base.principal_first_spike_step
    same_address_decision_tick = principal_tick + v4.COLLISION_STOP_DELAY_TICKS
    repeated_cue_assembly = assembly_by_token[CUE_TOKENS["NEXT"]]
    rows.append(
        {
            "control_sha256": base.sha256,
            "name": "same-address-double-cue-collision-stop",
            "cue_events": (("NEXT", 0), ("NEXT", 0)),
            "address_event_count": 2,
            "phase_spikes": phase_spikes,
            "cue_assembly_events": (
                {
                    "tick": 1,
                    "direction": "NEXT",
                    "cue_slot": 0,
                    "serial_event_index": 0,
                    "token": CUE_TOKENS["NEXT"],
                    "assembly_sha256": repeated_cue_assembly.sha256,
                    "assembly_cells": repeated_cue_assembly.cells,
                },
                {
                    "tick": 1,
                    "direction": "NEXT",
                    "cue_slot": 0,
                    "serial_event_index": 1,
                    "token": CUE_TOKENS["NEXT"],
                    "assembly_sha256": repeated_cue_assembly.sha256,
                    "assembly_cells": repeated_cue_assembly.cells,
                },
            ),
            "event_counter_read_then_set_events": (
                {
                    "tick": 1,
                    "serial_event_index": 0,
                    "read_pre_event_storage": True,
                    "repeat_readout_spike": False,
                    "set_storage_after_read": True,
                },
                {
                    "tick": 1,
                    "serial_event_index": 1,
                    "read_pre_event_storage": True,
                    "repeat_readout_spike": True,
                    "set_storage_after_read": True,
                },
            ),
            "collision_storage_set_tick": 1,
            "collision_storage_charge_pa_hex": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "collision_read_gate_tick": (principal_tick + v4.COLLISION_DETECTOR_DELAY_TICKS),
            "address_coincidence_spikes": ((1, 0), (1, 0)),
            "address_hard_reset_count": 2,
            "address_current_latches_set": (0,),
            "principal_start_tick": _SELECTOR_PRINCIPAL_START_TICK,
            "principal_first_spike_tick": principal_tick,
            "principal_first_spike_tier": (0,),
            "basket_tick": principal_tick + v4.BASKET_DELAY_TICKS,
            "collision_tick": principal_tick + v4.COLLISION_DETECTOR_DELAY_TICKS,
            "decision_tick": same_address_decision_tick,
            "decision": "COLLISION_STOP",
            "selected_slot": None,
            "slot_tokens": (),
            "selected_token": None,
            "latch_set_tick": None,
            "latch_spike_tick": None,
            "content_phase_ticks": (),
            "content_gate_ticks": (),
            "content_replay_events": (),
            "selected_lexical_relay_tick": None,
            "selected_lexical_relay_cells": (),
            "replay_close_reset_tick": same_address_decision_tick + 1,
            "post_reset_active_cell_count": 0,
        }
    )
    record = {
        "selector_circuit_sha256": SELECTOR_CIRCUIT_SHA256,
        "principal_controls_sha256": _namespaced_sha256(
            _SELECTOR_CONTROLS_NAMESPACE,
            tuple(control.record(include_sha256=False) for control in controls),
        ),
        "control_count": len(rows),
        "required_pass_count": len(rows),
        "controls": tuple(rows),
        "required_observations": (
            "phase and cue-assembly axon spikes",
            "per-event address coincidence/hard-reset and persistent event count",
            "held address-current latches",
            "complete principal tier",
            "basket, collision, and delayed decision relay spikes",
            "post-scan selected-latch reset, unique set, replay-phase spike, and reset",
            "exact selected 12-cell lexical relay trace and zero nonselected trace",
            "zero active lesson-local cells after close/reset",
        ),
        "causal_controls": {
            "circuit_off": "zero decision, latch, and lexical-relay spikes",
            "phase_leg_off": "cue assemblies spike but no address cell may spike",
            "cue_leg_off": "all phase cells spike but no address cell may spike",
            "latch_persistent_charge_off": (
                "selected latch may receive its set input but cannot retain the "
                "subthreshold charge until replay"
            ),
            "lexical_gate_off": (
                "selector decision survives but lexical-relay spike count is zero"
            ),
        },
        "execution_deferred": True,
        "must_execute_in_learner_before_state_allocation": True,
    }
    return _freeze_record(record), _namespaced_sha256(
        _SELECTOR_FULL_PREFLIGHT_NAMESPACE,
        record,
    )


def _build_v4_wta_reference() -> tuple[object, str]:
    controls = v4.PROTOCOL.wta_electrical_controls
    current_rows = np.asarray(
        [control.principal_currents_pa for control in controls],
        dtype=np.float64,
    )
    races = CompiledRestingGlifRace.compile(step_count=v4.RACE_STEPS).race_many(
        current_rows,
        require_unique=False,
    )
    observations = tuple(
        {
            "control_sha256": control.sha256,
            "name": control.name,
            "principal_first_spike_step": race.spike_step,
            "principal_first_spike_tier": control.principal_first_spike_tier,
            "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
        }
        for control, race in zip(controls, races, strict=True)
    )
    if any(
        observation["principal_first_spike_step"] != control.principal_first_spike_step
        or observation["simultaneous_spike_candidates"] != control.simultaneous_spike_candidates
        for observation, control in zip(observations, controls, strict=True)
    ):
        raise RuntimeError("teacher-gated V5 changed a passed V4 principal control")
    record = {
        "latency_table_sha256": v4.PROTOCOL.latency_table_sha256,
        "wta_topology_sha256": v4.PROTOCOL.wta_topology_sha256,
        "wta_electrical_controls_sha256": (v4.PROTOCOL.wta_electrical_controls_sha256),
        "factor_registry": FACTOR_REGISTRY,
        "factor_cells": FACTOR_COUNT,
        "stop_candidate_index": FACTOR_COUNT,
        "raw_score_domain": (SCORE_MIN, SCORE_MAX),
        "raw_to_current_score_rule": RAW_TO_CURRENT_SCORE_RULE,
        "current_score_domain": (0, SCORE_MAX),
        "v4_latency_table_domain": (0, v4.SCORE_MAX),
        "stop_score": STOP_SCORE,
        "principal_control_adapter": "CompiledRestingGlifRace.race_many",
        "principal_control_shape": (len(controls), FACTOR_COUNT + 1),
        "principal_control_observations": observations,
        "reused_v4_learner_state": False,
        "full_factor_circuit_deferred": True,
        "factor_circuit_deferred_scope": (
            "basket, collision, delayed factor and STOP relays execute in learner "
            "electrical preflight"
        ),
    }
    return _freeze_record(record), _namespaced_sha256(
        _V4_WTA_REFERENCE_NAMESPACE,
        record,
    )


V4_WTA_REFERENCE_RECORD, V4_WTA_REFERENCE_SHA256 = _build_v4_wta_reference()

_FUTURE_STATE_RECORD = {
    "tick_dtype": TICK_DTYPE,
    "tick_cap": TICK_CAP,
    "state_shape": FUTURE_STATE_SHAPE,
    "state_arrays": ("excitatory_ticks", "inhibitory_ticks"),
    "array_count": 2,
    "total_state_bytes": FUTURE_STATE_BYTES,
    "raw_score_domain": (SCORE_MIN, SCORE_MAX),
    "raw_to_current_score_rule": RAW_TO_CURRENT_SCORE_RULE,
    "stop_score": STOP_SCORE,
    "score_source": "sum(E-H) over the selected 12-cell lexical trace only",
    "learner_arrays_allocated_in_protocol": False,
}
FUTURE_STATE_RECORD = _freeze_record(_FUTURE_STATE_RECORD)


def _build_local_rule_record(
    lessons: tuple[AtomicLesson, ...],
    fit_ids: tuple[str, ...],
) -> tuple[object, str]:
    fit_set = set(fit_ids)
    ordered_fit = tuple(
        sorted(
            (lesson for lesson in lessons if lesson.sha256 in fit_set),
            key=lambda lesson: (lesson.tokens, lesson.factor, lesson.sha256),
        )
    )
    if len(ordered_fit) != FIT_COUNT:
        raise RuntimeError("teacher-gated V5 FIT schedule shape changed")
    record = {
        "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
        "future_state": _FUTURE_STATE_RECORD,
        "teacher_population_cells": FACTOR_COUNT,
        "teacher_population_rule": "one-hot physical target clamp",
        "teacher_clamp_current_pa_hex": v4.TEACHER_CLAMP_CURRENT_PA.hex(),
        "teacher_after_autonomous_prediction": True,
        "scout_population_cells": FACTOR_COUNT,
        "scout_basket_suppressed": False,
        "scout_window_close": "physical STOP-reference first spike",
        "selected_trace_cell_count": ASSEMBLY_ACTIVE_CELL_COUNT,
        "target_not_autonomous_unique": (
            "increment selected-trace E ticks on the local teacher-target row"
        ),
        "non_target_scout": (
            "only when the target is not autonomous unique, every non-target scout "
            "firing no later than STOP increments its selected-trace H ticks"
        ),
        "correct_unique_target": "suppress the complete atomic E/H commit",
        "atomic_commit": (
            "derive all local deltas from one pre-update snapshot and commit them "
            "simultaneously only after the complete spike trace"
        ),
        "fit_pass_count": 1,
        "epochs": None,
        "parameter_sweep": False,
        "retry": False,
        "fit_schedule": {
            "sort_key": ("canonical_tokens", "factor", "lesson_sha256"),
            "sort_direction": "ascending",
            "shuffle": False,
            "repeat": False,
            "lesson_sha256s": tuple(lesson.sha256 for lesson in ordered_fit),
            "causal_order_source": "this lesson_sha256s tuple only",
            "heldout_scheduled": False,
        },
        "forbidden": (
            "host factor selection",
            "host negative-label generation",
            "heldout updates",
            "epoch loop",
            "parameter sweep",
            "retry",
        ),
    }
    return _freeze_record(record), _namespaced_sha256(_LOCAL_RULE_NAMESPACE, record)


def _random_label_mapping() -> tuple[tuple[str, str], ...]:
    ranked = tuple(
        sorted(
            TAUGHT_FACTORS,
            key=lambda factor: sha256(
                b"teacher-gated-factor-workspace-v5-random-label\0"
                + RANDOM_LABEL_SEED.to_bytes(8, "little")
                + factor.encode("ascii")
            ).digest(),
        )
    )
    mapping = tuple(
        (factor, ranked[(index + 1) % len(ranked)]) for index, factor in enumerate(ranked)
    )
    if any(source == target for source, target in mapping):
        raise RuntimeError("teacher-gated V5 random-label mapping has a fixed point")
    return mapping


def _build_control_panel(local_rule_record: object) -> tuple[object, str]:
    schedule = local_rule_record["fit_schedule"]
    record = {
        "selector_electrical_control_count": SELECTOR_ELECTRICAL_CONTROL_COUNT,
        "factor_wta_electrical_control_count": len(v4.PROTOCOL.wta_electrical_controls),
        "two_rival_scratch": {
            "selected_factor": TRIADS[0][0],
            "teacher_target": TRIADS[0][0],
            "rival_factors": TRIADS[0][1:],
            "rival_seed_ticks_per_selected_cell": 2,
            "rival_raw_scores": (24, 24),
            "expected_target_e_update_events": 1,
            "expected_inhibitory_update_events": 2,
            "expected_target_e_ticks": 12,
            "expected_rival_h_ticks": 24,
            "full_teacher_scout_circuit_deferred": True,
        },
        "nonselected_distractor_scratch": {
            "selected_factor": TRIADS[0][0],
            "nonselected_distractor": TRIADS[0][1],
            "wrong_head_seeded_only_on_distractor_trace": True,
            "wrong_head_ticks_per_distractor_cell": 2,
            "maximum_selected_trace_overlap": ASSEMBLY_MAX_PAIRWISE_OVERLAP,
            "maximum_wrong_selected_score": 6,
            "expected_rival_events": 0,
            "expected_target_e_ticks": 12,
            "full_teacher_scout_circuit_deferred": True,
        },
        "teacher_off": {
            "fit_schedule_sha256s": schedule["lesson_sha256s"],
            "teacher_gain": 0,
            "expected_checkpoint_nonzero_bytes": 0,
            "expected_stop_count": LESSON_COUNT,
            "expected_factor_event_count": 0,
        },
        "cell_off": {
            "treatment_checkpoint_source": "frozen treatment checkpoint",
            "content_relay_gain": 0,
            "expected_stop_count": LESSON_COUNT,
            "expected_factor_event_count": 0,
        },
        "random_label_permutation": {
            "seed": RANDOM_LABEL_SEED,
            "algorithm": "SHA-256 rank then one-position cyclic rotation",
            "mapping": _random_label_mapping(),
            "fixed_point_count": 0,
            "expected_permuted_exact": LESSON_COUNT,
            "expected_true_exact": 0,
            "learner_execution_deferred": True,
        },
        "silent_rows": {
            "factors": tuple(factor for factor in FACTOR_REGISTRY if factor not in TAUGHT_FACTORS),
            "factor_count": FACTOR_COUNT - len(TAUGHT_FACTORS),
            "expected_nonzero_bytes": 0,
        },
        "treatment_inference": {
            "current_matrix_shape": (LESSON_COUNT, FACTOR_COUNT + 1),
            "race_adapter": "CompiledRestingGlifRace.race_many",
            "race_many_call_count": 1,
            "checkpoint_authentications": 2,
            "per_span_checkpoint_hashing": False,
            "winner_cell_on_nonunique_tier": "ignored",
            "learner_execution_deferred": True,
        },
    }
    return _freeze_record(record), _namespaced_sha256(_CONTROL_PANEL_NAMESPACE, record)


_FACTOR_PRINCIPAL_START_TICK = (
    _SELECTOR_UNIQUE_ADDRESS_RELAY_TICK + _SELECTOR_CONTENT_CLOSE_RESET_OFFSET + 1
)
_SCOUT_STOP_CLOSE_TICK = (
    _FACTOR_PRINCIPAL_START_TICK + v4.PROTOCOL.latency_table[STOP_SCORE].first_spike_step
)
_AUTONOMOUS_RELAY_CAPTURE_CLOSE_TICK = _SCOUT_STOP_CLOSE_TICK + v4.FACTOR_RELAY_DELAY_TICKS
_TEACHER_SPIKE_TICK = _AUTONOMOUS_RELAY_CAPTURE_CLOSE_TICK + 1
_AUTONOMOUS_AND_SCOUT_READ_TICK = _TEACHER_SPIKE_TICK + 1
_TARGET_MATCH_TICK = _AUTONOMOUS_AND_SCOUT_READ_TICK + 1
_UPDATE_RELAY_TICK = _TARGET_MATCH_TICK + 1
_ATOMIC_DELTA_COMMIT_TICK = _UPDATE_RELAY_TICK + 1
_LESSON_RESET_TICK = _ATOMIC_DELTA_COMMIT_TICK + 1


def _build_teacher_scout_circuit_record(
    lessons: tuple[AtomicLesson, ...],
    local_rule_record: object,
    local_rule_sha256: str,
    control_panel_sha256: str,
) -> tuple[object, str]:
    lesson_by_sha = {lesson.sha256: lesson for lesson in lessons}
    schedule_rows: list[dict[str, object]] = []
    for schedule_index, lesson_sha256 in enumerate(
        local_rule_record["fit_schedule"]["lesson_sha256s"]
    ):
        lesson = lesson_by_sha[lesson_sha256]
        schedule_rows.append(
            {
                "schedule_index": schedule_index,
                "lesson_sha256": lesson.sha256,
                "factor": lesson.factor,
                "factor_cell": FACTOR_REGISTRY.index(lesson.factor),
                "direction": lesson.direction,
                "cue_slot": lesson.cue_slot,
                "selected_slot": lesson.selected_slot,
                "selector_control_name": (f"{lesson.direction.lower()}-slot-{lesson.cue_slot}"),
                "selected_trace_set_tick": (
                    _SELECTOR_UNIQUE_ADDRESS_RELAY_TICK
                    + _SELECTOR_CONTENT_GATE_OFFSETS[lesson.selected_slot]
                ),
                "selected_trace_valid_through_tick": _ATOMIC_DELTA_COMMIT_TICK,
                "factor_principal_start_tick": _FACTOR_PRINCIPAL_START_TICK,
                "scout_stop_close_tick": _SCOUT_STOP_CLOSE_TICK,
                "autonomous_relay_capture_close_tick": (_AUTONOMOUS_RELAY_CAPTURE_CLOSE_TICK),
                "teacher_spike_tick": _TEACHER_SPIKE_TICK,
                "autonomous_and_scout_read_tick": (_AUTONOMOUS_AND_SCOUT_READ_TICK),
                "target_match_tick": _TARGET_MATCH_TICK,
                "update_relay_tick": _UPDATE_RELAY_TICK,
                "atomic_delta_commit_tick": _ATOMIC_DELTA_COMMIT_TICK,
                "lesson_reset_tick": _LESSON_RESET_TICK,
            }
        )
    record = {
        "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
        "selector_circuit_sha256": SELECTOR_CIRCUIT_SHA256,
        "v4_factor_latency_table_sha256": v4.PROTOCOL.latency_table_sha256,
        "v4_factor_wta_topology_sha256": v4.PROTOCOL.wta_topology_sha256,
        "local_rule_sha256": local_rule_sha256,
        "control_panel_sha256": control_panel_sha256,
        "cell_models": {
            "factor_principal_and_scout": "AdaptiveGlifPopulation",
            "factor_glif_sha256": v4.PROTOCOL.glif_sha256,
            "teacher_latch_match_update_reset": (
                "custom current-integrating threshold/reset neurons"
            ),
        },
        "cell_counts": {
            "selected_trace_latch": LEXICAL_CELL_COUNT,
            "selected_trace_storage_compartment": LEXICAL_CELL_COUNT,
            "selected_trace_readout_compartment": LEXICAL_CELL_COUNT,
            "factor_principal": FACTOR_COUNT,
            "stop_reference": 1,
            "autonomous_factor_result_latch": FACTOR_COUNT,
            "autonomous_result_storage_compartment": FACTOR_COUNT,
            "autonomous_result_readout_compartment": FACTOR_COUNT,
            "scout": FACTOR_COUNT,
            "scout_eligibility_latch": FACTOR_COUNT,
            "scout_eligibility_storage_compartment": FACTOR_COUNT,
            "scout_eligibility_readout_compartment": FACTOR_COUNT,
            "teacher": FACTOR_COUNT,
            "target_match": FACTOR_COUNT,
            "target_e_update_relay": FACTOR_COUNT,
            "rival_h_update_relay": FACTOR_COUNT,
            "reset": 1,
        },
        "current_pa_hex": {
            "full": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "half": float(v4.RELAY_COPY_CURRENT_PA / 2.0).hex(),
            "inhibition": float(v4.BASKET_INHIBITION_CURRENT_PA).hex(),
            "teacher_clamp": float(v4.TEACHER_CLAMP_CURRENT_PA).hex(),
            "persistent_storage_set": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "persistent_read_gate": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "persistent_storage_coupling": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        },
        "threshold_pa_hex": {
            "teacher": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "selected_trace_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
            "autonomous_factor_result_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
            "scout_eligibility_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
            "target_match": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "target_e_update_relay": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "rival_h_update_relay": float(v4.RELAY_COPY_CURRENT_PA).hex(),
            "reset": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        },
        "factor_principal_start_tick": _FACTOR_PRINCIPAL_START_TICK,
        "scout_window": (
            _FACTOR_PRINCIPAL_START_TICK,
            _SCOUT_STOP_CLOSE_TICK,
            "inclusive",
        ),
        "scout_stop_close_tick": _SCOUT_STOP_CLOSE_TICK,
        "autonomous_relay_capture_close_tick": (_AUTONOMOUS_RELAY_CAPTURE_CLOSE_TICK),
        "teacher_spike_tick": _TEACHER_SPIKE_TICK,
        "autonomous_and_scout_read_tick": _AUTONOMOUS_AND_SCOUT_READ_TICK,
        "target_match_tick": _TARGET_MATCH_TICK,
        "update_relay_tick": _UPDATE_RELAY_TICK,
        "selected_trace_read_and_atomic_commit_tick": (_ATOMIC_DELTA_COMMIT_TICK),
        "atomic_delta_commit_tick": _ATOMIC_DELTA_COMMIT_TICK,
        "lesson_reset_tick": _LESSON_RESET_TICK,
        "selected_trace_valid_through_tick": _ATOMIC_DELTA_COMMIT_TICK,
        "persistent_latch_rule": (
            "lexical relays, unique autonomous factor relays, and eligible scout "
            "spikes each set only the 5000 pA no-leak storage compartment; a later "
            "5000 pA gate reads the separate 10000 pA compartment without changing "
            "stored charge"
        ),
        "latch_read_is_nondestructive": True,
        "factor_current_rule": (
            "sum uint8 E-H over exactly the selected 12-cell eligibility trace, "
            "clamp raw score at zero, then index the frozen V4 latency table"
        ),
        "autonomous_capture_rule": (
            "the complete V4 basket/collision/delayed-relay circuit latches only a "
            "unique factor relay; STOP or collision never sets a factor result latch"
        ),
        "winner_cell_on_nonunique_tier": "ignored",
        "scout_basket_suppressed": False,
        "scout_rule": (
            "24 unsuppressed GLIF scouts receive the same factor currents; every "
            "factor scout with a first spike no later than the tonic score-12 STOP "
            "reference sets its local eligibility latch"
        ),
        "teacher_rule": (
            "after autonomous relay capture closes, exactly one target teacher cell "
            "receives the 5000 pA clamp and spikes"
        ),
        "teacher_scout_tick_schedule": (
            "tick 1134: spike one-hot teacher after every autonomous relay closes",
            "tick 1135: nondestructively read autonomous-result and scout storage",
            "tick 1136: evaluate same-index teacher/result target match",
            "tick 1137: evaluate target-E and rival-H update relays",
            "tick 1138: read selected trace and atomically commit snapshot deltas",
            "tick 1139: reset every lesson-local storage and readout compartment",
        ),
        "physical_delays": {
            "teacher_to_target_match_ticks": 2,
            "autonomous_readout_to_target_match_ticks": 1,
            "teacher_to_update_relay_ticks": 3,
            "scout_readout_to_update_relay_ticks": 2,
            "target_match_to_update_relay_ticks": 1,
            "update_relay_to_commit_ticks": 1,
        },
        "target_match_rule": (
            "2500 pA from the target teacher plus 2500 pA from the same-index "
            "autonomous factor-result latch closes the shared atomic miss gate"
        ),
        "atomic_miss_gate_rule": (
            "a matching autonomous unique target latch inhibits every E and H "
            "update relay; STOP, collision, or a unique wrong factor leaves the "
            "teacher-opened miss gate active"
        ),
        "target_e_rule": (
            "the target-E relay receives 5000 pA from its delayed teacher spike "
            "and -5000 pA from a matching target-match spike; if it fires, its "
            "delayed output increments the 12 selected-trace uint8 E ticks once, "
            "saturating at 4"
        ),
        "rival_h_rule": (
            "each rival-H relay receives 2500 pA from the delayed teacher-open "
            "broadcast plus 2500 pA from its scout readout, -5000 pA from the "
            "same-index target teacher, and -5000 pA from target match; every relay "
            "that fires increments its 12 selected-trace uint8 H ticks once, "
            "saturating at 4"
        ),
        "teacher_same_index_inhibition": (
            "the one-hot teacher contributes -5000 pA only to its same-index H "
            "relay; other latched scout relays remain local and unsuppressed"
        ),
        "correct_unique_no_update_condition": (
            "matching autonomous unique target latch suppresses the complete atomic "
            "E/H commit, irrespective of later nonleading scout spikes"
        ),
        "precommit_snapshot_rule": (
            "compute every E/H delta from the same immutable selected-trace, E, H, "
            "autonomous-result, and scout-latch snapshot; then commit simultaneously"
        ),
        "teacher_off_disables_plastic_commit": True,
        "atomic_commit_precedence": (
            "freeze autonomous factor/STOP relay and scout traces",
            "spike one-hot teacher target",
            "read autonomous-result and scout storage nondestructively",
            "evaluate target match before update relays",
            "evaluate target-E and rival-H relays from the frozen snapshot",
            "read selected-trace storage and commit all E/H deltas simultaneously",
            "reset every lesson-local population and eligibility latch",
        ),
        "reset_scope": (
            "selector phase/cue/address/current latches",
            "selected-slot and lexical replay relays",
            "selected-trace eligibility latches",
            "factor principals, basket, collision, and decision relays",
            "autonomous result, scouts, scout latches, teacher, match, and updates",
        ),
        "fit_lesson_schedule": tuple(schedule_rows),
        "preflight_controls": {
            "score_12_target_stop_tie": (
                "collision STOP, miss gate open, one target E relay, zero non-target "
                "H relays when all rivals are below STOP"
            ),
            "score_24_correct_unique": (
                "unique target relay closes the atomic miss gate; all E/H update "
                "relays remain silent even if a later scout spikes"
            ),
            "two_rival_scratch": (
                "target E relay spikes once; two non-target scout H relays spike "
                "once each; commit 12 E and 24 H ticks"
            ),
            "nonselected_distractor_scratch": (
                "distractor-only synapses receive no selected trace and emit zero "
                "scout/H events; target commits 12 E ticks"
            ),
            "teacher_off": (
                "teacher, E, and H update relays emit zero spikes and checkpoint "
                "state stays all-zero"
            ),
            "cell_off": (
                "selected trace remains empty, all factor decisions are STOP, and "
                "no E/H cell changes"
            ),
        },
        "full_circuit_execution_deferred": True,
    }
    return _freeze_record(record), _namespaced_sha256(
        _TEACHER_SCOUT_CIRCUIT_NAMESPACE,
        record,
    )


_ATOMIC_GATE_RECORD = {
    "status": "PREREGISTERED_EXPECTATION_FALSIFIER_NOT_RUN",
    "required_fit_unique_exact": FIT_COUNT,
    "required_heldout_unique_exact": HELDOUT_COUNT,
    "required_atomic_unique_exact": LESSON_COUNT,
    "allowed_stop_count": 0,
    "allowed_collision_count": 0,
    "allowed_wrong_factor_count": 0,
    "required_checkpoint_immutable": True,
    "required_causal_controls_exact": True,
    "expected_treatment_telemetry": {
        "excitatory_update_events": 18,
        "inhibitory_update_events": 0,
        "committed_excitatory_ticks": 216,
        "excitatory_max_ticks": 2,
        "inhibitory_max_ticks": 0,
    },
    "expected_treatment_telemetry_derivation": (
        "each factor's first selected-trace presentation raises score 0 to 12; "
        "the second breaks the score-12 STOP tie by raising it to 24; later "
        "presentations are unique; pairwise lexical overlap <=3 bounds any "
        "wrong learned head at score 6"
    ),
    "expectation_status": "analytical falsifier, never learner evidence",
    "composition_attempts_on_any_failure": 0,
    "no_retry_or_tuning": True,
}
ATOMIC_GATE_SHA256 = _namespaced_sha256(_ATOMIC_GATE_NAMESPACE, _ATOMIC_GATE_RECORD)
ATOMIC_GATE_RECORD = _freeze_record(_ATOMIC_GATE_RECORD)


def _build_neutral_spans() -> tuple[NeutralSpan, ...]:
    token_rows = (
        ("v5-quiet", "v5-idle", "v5-plain", "v5-blank"),
        ("v5-blank", "v5-plain", "v5-idle", "v5-quiet"),
        ("v5-silent", "v5-calm", "v5-still", "v5-empty"),
        ("v5-empty", "v5-still", "v5-calm", "v5-silent"),
        ("v5-neutral", "v5-unused", "v5-vacant", "v5-spare"),
        ("v5-spare", "v5-vacant", "v5-unused", "v5-neutral"),
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


def _build_composition_cases(
    lessons: tuple[AtomicLesson, ...],
    neutrals: tuple[NeutralSpan, ...],
) -> tuple[CompositionCase, ...]:
    lookup = {
        (lesson.triad_index, lesson.template_index, lesson.factor): lesson for lesson in lessons
    }
    cases: list[CompositionCase] = []
    for triad_index, triad in enumerate(TRIADS):
        ordered_pairs = tuple((left, right) for left in triad for right in triad if left != right)
        for pair_index, (left_factor, right_factor) in enumerate(ordered_pairs):
            case_index = len(cases)
            left_template, right_template = (4, 5) if pair_index % 2 == 0 else (5, 4)
            left = lookup[(triad_index, left_template, left_factor)]
            right = lookup[(triad_index, right_template, right_factor)]
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
        event_sha256=_namespaced_sha256(_COMPOSITOR_CONTROL_NAMESPACE, record),
    )


def _compositor_control(
    control_index: int,
    kind: str,
    events: tuple[ProtocolFactorEvent, ...],
    expected_status: str,
    expected_eos_status_latch: str | None,
    expected_factors: tuple[str, ...],
) -> CompositorControl:
    record = {
        "control_index": control_index,
        "kind": kind,
        "events": tuple(event.record() for event in events),
        "expected_status": expected_status,
        "expected_eos_status_latch": expected_eos_status_latch,
        "expected_factors": expected_factors,
    }
    return CompositorControl(
        control_index=control_index,
        kind=kind,
        events=events,
        expected_status=expected_status,
        expected_eos_status_latch=expected_eos_status_latch,
        expected_factors=expected_factors,
        sha256=_namespaced_sha256(_COMPOSITOR_CONTROL_NAMESPACE, record),
    )


def _build_compositor_controls() -> tuple[
    tuple[CompositorControl, ...],
    tuple[CompositorControl, ...],
    tuple[CompositorControl, ...],
]:
    exact_duplicates = tuple(
        _compositor_control(
            factor_index,
            "exact-duplicate",
            (_event(factor, 0, 4), _event(factor, 0, 4)),
            "ONE_EVENT",
            None,
            (factor,),
        )
        for factor_index, factor in enumerate(TAUGHT_FACTORS)
    )
    duplicate_after_full = tuple(
        _compositor_control(
            len(exact_duplicates) + triad_index,
            "exact-duplicate-after-full",
            (
                _event(triad[0], 0, 4),
                _event(triad[1], 4, 8),
                _event(triad[0], 0, 4),
            ),
            "TWO_EVENTS",
            None,
            (triad[0], triad[1]),
        )
        for triad_index, triad in enumerate(TRIADS)
    )
    duplicates = (*exact_duplicates, *duplicate_after_full)
    distinct_overlaps = tuple(
        _compositor_control(
            triad_index,
            "distinct-overlap",
            (_event(triad[0], 0, 4), _event(triad[1], 2, 6)),
            "STOP_OVERLAPPING_FACTOR_EVENTS",
            "overlap",
            (),
        )
        for triad_index, triad in enumerate(TRIADS)
    )
    overlap_after_full = tuple(
        _compositor_control(
            len(distinct_overlaps) + triad_index,
            "distinct-overlap-after-full",
            (
                _event(triad[0], 0, 4),
                _event(triad[1], 4, 8),
                _event(triad[2], 2, 6),
            ),
            "STOP_OVERLAPPING_FACTOR_EVENTS",
            "overlap",
            (),
        )
        for triad_index, triad in enumerate(TRIADS)
    )
    overlaps = (*distinct_overlaps, *overlap_after_full)
    capacities = tuple(
        _compositor_control(
            triad_index,
            "capacity-three",
            tuple(
                _event(factor, factor_index * 4, (factor_index + 1) * 4)
                for factor_index, factor in enumerate(triad)
            ),
            "STOP_EVENT_CAPACITY_2",
            "capacity",
            (),
        )
        for triad_index, triad in enumerate(TRIADS)
    )
    return duplicates, overlaps, capacities


class _SequentialFirstFreeSlotAdmissionPopulation:
    """Staged first-free encoder over a frozen pre-event occupancy snapshot."""

    __slots__ = ()

    def evaluate(
        self,
        *,
        occupied_storage_pa: np.ndarray,
        admission_block_storage_pa: np.ndarray,
    ) -> np.ndarray:
        occupied = np.asarray(occupied_storage_pa, dtype=np.float64)
        block = np.asarray(admission_block_storage_pa, dtype=np.float64)
        full = float(v4.RELAY_COPY_CURRENT_PA)
        if (
            occupied.shape != (EVENT_CAPACITY,)
            or block.shape != (1,)
            or not np.all(np.isfinite(occupied))
            or not np.all(np.isfinite(block))
            or np.any((occupied != 0.0) & (occupied != full))
            or np.any((block != 0.0) & (block != full))
            or tuple(occupied) not in ((0.0, 0.0), (full, 0.0), (full, full))
        ):
            raise ValueError("compositor admission storage domain changed")

        slot_zero_current = full - occupied[0] - block[0]
        slot_zero_fired = slot_zero_current >= full
        slot_one_current = occupied[0] - occupied[1] - block[0]
        slot_one_fired = slot_one_current >= full
        return np.asarray((slot_zero_fired, slot_one_fired), dtype=np.bool_)


_COMPOSITOR_BOUNDARY_CELL_COUNT = 13
_COMPOSITOR_TOKEN_POSITION_CELL_COUNT = 12
_COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT = (
    FACTOR_COUNT
    + _COMPOSITOR_BOUNDARY_CELL_COUNT
    + _COMPOSITOR_BOUNDARY_CELL_COUNT
    + _COMPOSITOR_TOKEN_POSITION_CELL_COUNT
)
_COMPOSITOR_DUPLICATE_KEY_CELL_COUNT = (
    EVENT_CAPACITY
    * FACTOR_COUNT
    * _COMPOSITOR_BOUNDARY_CELL_COUNT
    * _COMPOSITOR_BOUNDARY_CELL_COUNT
)
_COMPOSITOR_OVERLAP_COINCIDENCE_CELL_COUNT = EVENT_CAPACITY * _COMPOSITOR_TOKEN_POSITION_CELL_COUNT
_COMPOSITOR_SLOT_COPY_OUTPUT_CELL_COUNT = EVENT_CAPACITY * (
    _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT + 1
)


class _CompositorPhaseRequestPopulation:
    """Ready-gated one-hot next-ingress versus EOS request cells."""

    __slots__ = ()

    def evaluate(
        self,
        *,
        request_spikes: np.ndarray,
        event_ready_storage_pa: np.ndarray,
    ) -> np.ndarray:
        requests = np.asarray(request_spikes)
        ready = np.asarray(event_ready_storage_pa, dtype=np.float64)
        full = float(v4.RELAY_COPY_CURRENT_PA)
        if (
            requests.shape != (2,)
            or requests.dtype != np.bool_
            or np.count_nonzero(requests) != 1
            or ready.shape != (1,)
            or not np.all(np.isfinite(ready))
            or np.any((ready != 0.0) & (ready != full))
        ):
            raise ValueError("compositor phase-request domain changed")
        return requests.copy() if ready[0] == full else np.zeros(2, dtype=np.bool_)


class _BufferedIngressSlotCopyPopulation:
    """Half-current coincidence cells for buffered identity-to-slot copy."""

    __slots__ = ()

    def evaluate(
        self,
        *,
        ingress_readout_spikes: np.ndarray,
        admitted_slot_spikes: np.ndarray,
    ) -> np.ndarray:
        ingress = np.asarray(ingress_readout_spikes)
        admitted = np.asarray(admitted_slot_spikes)
        if (
            ingress.shape != (_COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,)
            or admitted.shape != (EVENT_CAPACITY,)
            or ingress.dtype != np.bool_
            or admitted.dtype != np.bool_
            or np.count_nonzero(admitted) > 1
        ):
            raise ValueError("compositor buffered-copy spike domain changed")

        factor = ingress[:FACTOR_COUNT]
        start = ingress[FACTOR_COUNT : FACTOR_COUNT + _COMPOSITOR_BOUNDARY_CELL_COUNT]
        stop_offset = FACTOR_COUNT + _COMPOSITOR_BOUNDARY_CELL_COUNT
        stop = ingress[stop_offset : stop_offset + _COMPOSITOR_BOUNDARY_CELL_COUNT]
        footprint = ingress[stop_offset + _COMPOSITOR_BOUNDARY_CELL_COUNT :]
        if not np.any(ingress):
            if np.any(admitted):
                raise ValueError("compositor buffered-copy admitted empty ingress")
            return np.zeros(
                (EVENT_CAPACITY, _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT + 1),
                dtype=np.bool_,
            )
        if (
            np.count_nonzero(factor) != 1
            or np.count_nonzero(start) != 1
            or np.count_nonzero(stop) != 1
        ):
            raise ValueError("compositor buffered-copy event one-hot domain changed")
        start_index = int(np.flatnonzero(start)[0])
        stop_index = int(np.flatnonzero(stop)[0])
        expected_footprint = np.zeros(
            _COMPOSITOR_TOKEN_POSITION_CELL_COUNT,
            dtype=np.bool_,
        )
        if start_index >= stop_index or stop_index > _COMPOSITOR_TOKEN_POSITION_CELL_COUNT:
            raise ValueError("compositor buffered-copy event boundary domain changed")
        expected_footprint[start_index:stop_index] = True
        if not np.array_equal(footprint, expected_footprint):
            raise ValueError("compositor buffered-copy footprint domain changed")
        if not np.any(admitted):
            return np.zeros(
                (EVENT_CAPACITY, _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT + 1),
                dtype=np.bool_,
            )

        half = float(v4.RELAY_COPY_CURRENT_PA / 2.0)
        threshold = float(v4.RELAY_COPY_CURRENT_PA)
        identity_currents = (
            ingress.astype(np.float64)[None, :] * half + admitted.astype(np.float64)[:, None] * half
        )
        identity_copy = identity_currents >= threshold
        ingress_factor_leg = float(np.any(ingress[:FACTOR_COUNT])) * half
        occupied_currents = admitted.astype(np.float64) * half + ingress_factor_leg
        occupied_copy = (occupied_currents >= threshold)[:, None]
        return np.concatenate((identity_copy, occupied_copy), axis=1)


class _AbsorbingCompositorPoisonPopulation:
    """Persistent one-hot first-cause poison plus a case-wide ingress halt."""

    __slots__ = ("_halt", "_status")

    def __init__(self) -> None:
        self._status = _NoLeakStorageReadoutPopulation(2)
        self._halt = _NoLeakStorageReadoutPopulation(1)

    @property
    def status_charge_pa(self) -> tuple[float, float]:
        return self._status.storage_charge_pa  # type: ignore[return-value]

    @property
    def ingress_allowed(self) -> bool:
        return self._halt.storage_charge_pa == (0.0,)

    def step(
        self,
        *,
        overlap_decision: bool,
        capacity_decision: bool,
        case_reset: bool = False,
    ) -> tuple[tuple[bool, bool], bool]:
        if not all(
            isinstance(value, (bool, np.bool_))
            for value in (overlap_decision, capacity_decision, case_reset)
        ):
            raise ValueError("compositor poison decision domain changed")
        if overlap_decision and capacity_decision and self.ingress_allowed and not case_reset:
            raise ValueError("compositor poison decisions lost one-hot precedence")

        status_zeros = np.zeros(2, dtype=np.float64)
        halt_zeros = np.zeros(1, dtype=np.float64)
        if case_reset:
            self._status.step(
                set_currents_pa=status_zeros,
                read_gate_currents_pa=status_zeros,
                direct_readout_currents_pa=status_zeros,
                reset_currents_pa=np.full(2, -5_000.0),
            )
            self._halt.step(
                set_currents_pa=halt_zeros,
                read_gate_currents_pa=halt_zeros,
                direct_readout_currents_pa=halt_zeros,
                reset_currents_pa=np.full(1, -5_000.0),
            )
        elif self.ingress_allowed:
            status_set = np.asarray(
                (5_000.0 if overlap_decision else 0.0, 5_000.0 if capacity_decision else 0.0)
            )
            halt_set = np.asarray((5_000.0 if overlap_decision or capacity_decision else 0.0,))
            self._status.step(
                set_currents_pa=status_set,
                read_gate_currents_pa=status_zeros,
                direct_readout_currents_pa=status_zeros,
                reset_currents_pa=status_zeros,
            )
            self._halt.step(
                set_currents_pa=halt_set,
                read_gate_currents_pa=halt_zeros,
                direct_readout_currents_pa=halt_zeros,
                reset_currents_pa=halt_zeros,
            )

        status = tuple(charge == 5_000.0 for charge in self.status_charge_pa)
        return status, not self.ingress_allowed  # type: ignore[return-value]


_COMPOSITOR_INGRESS_BUFFER_RECORD = {
    "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
    "cell_model": "ephemeral exact-domain no-leak storage/readout compartments",
    "identity_widths": {
        "factor": FACTOR_COUNT,
        "start": _COMPOSITOR_BOUNDARY_CELL_COUNT,
        "stop": _COMPOSITOR_BOUNDARY_CELL_COUNT,
        "footprint": _COMPOSITOR_TOKEN_POSITION_CELL_COUNT,
    },
    "ingress_identity_storage_compartments": _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
    "ingress_identity_readout_compartments": _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
    "pre_event_occupancy_snapshot_storage_compartments": EVENT_CAPACITY,
    "pre_event_occupancy_snapshot_readout_compartments": EVENT_CAPACITY,
    "external_ingress_attempt_tick": -1,
    "case_poison_halt_read_tick": -1,
    "ingress_gate_tick": 0,
    "ingress_gate_cells": _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
    "ingress_gate_current_legs_pa_hex": {
        "identity": (2_500.0).hex(),
        "event_ready": (2_500.0).hex(),
        "case_poison_halt": (-5_000.0).hex(),
    },
    "ingress_gate_threshold_pa_hex": (5_000.0).hex(),
    "capture_tick": 1,
    "stored_slot_nondestructive_read_tick": 0,
    "occupancy_snapshot_set_tick": 1,
    "occupancy_snapshot_read_ticks": {
        "capacity": 3,
        "slot_zero_admission": 5,
        "slot_one_admission": 6,
    },
    "ingress_copy_read_tick": 8,
    "copy_coincidence_tick": 9,
    "admitted_slot_storage_set_tick": 10,
    "ephemeral_reset_tick": 10,
    "event_ready_rearm_tick": 10,
    "next_ingress_or_eos_request_tick": 11,
    "selected_phase_gate_tick": 12,
    "event_gate_stride_ticks": 12,
    "capture_rule": (
        "the -1 external factor/start/stop/footprint spikes, -1 event-ready readout, "
        "and -1 case-halt readout meet at 62 +0 ingress gates; permitted +0 gate "
        "spikes set 62 exact ingress storage compartments and drive comparisons only "
        "at +1; no host replay is permitted"
    ),
    "occupancy_snapshot_rule": (
        "the +0 nondestructive occupied-slot readouts set two no-leak snapshot "
        "storage compartments at +1; separately gated snapshot readouts drive +4 "
        "capacity, +6 slot-zero admission, and +7 slot-one admission"
    ),
    "copy_rule": (
        "at +8 the admitted-slot relays and a nondestructive ingress-buffer read "
        "emit to 126 half-current copy-coincidence cells at +9; each occupied-copy "
        "cell requires both its admitted-slot leg and any one-hot ingress-factor leg, "
        "and those outputs alone set chosen slot storage at +10"
    ),
    "duplicate_no_write_rule": (
        "an exact-duplicate decision sets the admission block, so neither admitted-slot "
        "relay nor any +9 copy-coincidence cell can fire; +10 clears only ephemeral state"
    ),
    "copy_input_domain_rule": (
        "ingress is independently either all-empty or exactly one-hot factor/start/stop "
        "with nonempty contiguous [start, stop) footprint; admission is independently "
        "zero-or-one-hot; admitted plus empty ingress and every malformed ingress fail, "
        "while valid ingress plus no admission and both-empty emit no copy"
    ),
    "same_microtick_spike_influence": False,
    "serialization_rule": (
        "an accepted +0 factor gate spike clears event-ready storage at +1; the +9 "
        "completion relay resets ephemeral state and rearms ready at +10, so the next "
        "ready read carries exactly one ingress-or-EOS request at +11 and the selected "
        "ingress gate or EOS read is +12"
    ),
}
COMPOSITOR_INGRESS_BUFFER_SHA256 = _namespaced_sha256(
    _COMPOSITOR_INGRESS_BUFFER_NAMESPACE,
    _COMPOSITOR_INGRESS_BUFFER_RECORD,
)
COMPOSITOR_INGRESS_BUFFER_RECORD = _freeze_record(_COMPOSITOR_INGRESS_BUFFER_RECORD)


_COMPOSITOR_ADMISSION_RECORD = {
    "compositor_ingress_buffer_sha256": COMPOSITOR_INGRESS_BUFFER_SHA256,
    "cell_model": "staged current-integrating first-free-slot encoder",
    "slot_count": EVENT_CAPACITY,
    "occupied_storage_domain_pa_hex": ((0.0).hex(), (5_000.0).hex()),
    "canonical_occupancy_snapshots_pa_hex": (
        ((0.0).hex(), (0.0).hex()),
        ((5_000.0).hex(), (0.0).hex()),
        ((5_000.0).hex(), (5_000.0).hex()),
    ),
    "admission_block_storage_domain_pa_hex": ((0.0).hex(), (5_000.0).hex()),
    "base_drive_pa_hex": (5_000.0).hex(),
    "occupied_inhibition_pa_hex": (-5_000.0).hex(),
    "block_inhibition_pa_hex": (-5_000.0).hex(),
    "slot_zero_occupied_excitation_pa_hex": (5_000.0).hex(),
    "slot_one_occupied_inhibition_pa_hex": (-5_000.0).hex(),
    "threshold_pa_hex": (5_000.0).hex(),
    "admission_block_set_tick": 5,
    "slot_zero_evaluation_tick": 6,
    "slot_one_evaluation_tick": 7,
    "admitted_slot_relay_tick": 8,
    "ingress_copy_read_tick": 8,
    "copy_coincidence_tick": 9,
    "admitted_slot_storage_set_tick": 10,
    "slot_zero_rule": (
        "at +6 base drive is inhibited by slot-zero pre-event occupancy and a "
        "classification spike emitted no later than +5"
    ),
    "slot_one_rule": (
        "at +7 slot-zero pre-event occupancy supplies the only excitatory drive, "
        "while slot-one pre-event occupancy and the +6 admission-block readout inhibit"
    ),
    "storage_set_rule": (
        "slot-zero and slot-one admission spikes converge on separate +8 selected-slot "
        "relays; +8 ingress readout and selected-slot spikes meet only at +9 copy "
        "coincidences, whose spikes set storage only at +10"
    ),
    "same_microtick_spike_influence": False,
    "mandatory_preflight_controls": {
        "both_empty": {
            "occupied_storage_pa_hex": ((0.0).hex(), (0.0).hex()),
            "admission_block_storage_pa_hex": (0.0).hex(),
            "expected_admission_spikes": (True, False),
        },
        "slot_zero_occupied_slot_one_empty": {
            "occupied_storage_pa_hex": ((5_000.0).hex(), (0.0).hex()),
            "admission_block_storage_pa_hex": (0.0).hex(),
            "expected_admission_spikes": (False, True),
        },
        "blocked": {
            "occupied_storage_pa_hex": ((0.0).hex(), (0.0).hex()),
            "admission_block_storage_pa_hex": (5_000.0).hex(),
            "expected_admission_spikes": (False, False),
        },
    },
}
COMPOSITOR_ADMISSION_SHA256 = _namespaced_sha256(
    _COMPOSITOR_ADMISSION_NAMESPACE,
    _COMPOSITOR_ADMISSION_RECORD,
)
COMPOSITOR_ADMISSION_RECORD = _freeze_record(_COMPOSITOR_ADMISSION_RECORD)


class _CompositorEosStatusPopulation:
    """Five-way causal EOS status candidates and poison-gated factor outputs."""

    __slots__ = ()

    def evaluate_status(
        self,
        *,
        occupied_storage_pa: np.ndarray,
        poison_storage_pa: np.ndarray,
    ) -> np.ndarray:
        occupied = np.asarray(occupied_storage_pa, dtype=np.float64)
        poison = np.asarray(poison_storage_pa, dtype=np.float64)
        full = float(v4.RELAY_COPY_CURRENT_PA)
        if (
            occupied.shape != (EVENT_CAPACITY,)
            or poison.shape != (2,)
            or not np.all(np.isfinite(occupied))
            or not np.all(np.isfinite(poison))
            or np.any((occupied != 0.0) & (occupied != full))
            or np.any((poison != 0.0) & (poison != full))
            or tuple(occupied) not in ((0.0, 0.0), (full, 0.0), (full, full))
            or np.count_nonzero(poison) > 1
            or (np.any(poison) and np.any(occupied))
        ):
            raise ValueError("compositor EOS storage domain changed")

        occupied_zero, occupied_one = occupied
        overlap_poison, capacity_poison = poison
        poison_inhibition = overlap_poison + capacity_poison
        candidate_currents = np.asarray(
            (
                full - occupied_zero - occupied_one - poison_inhibition,
                occupied_zero - occupied_one - poison_inhibition,
                (occupied_zero + occupied_one) / 2.0 - poison_inhibition,
                overlap_poison - capacity_poison,
                capacity_poison - overlap_poison,
            )
        )
        status = candidate_currents >= full
        if np.count_nonzero(status) != 1:
            raise RuntimeError("compositor EOS status is not one-hot")
        return status

    def evaluate_factor_outputs(
        self,
        *,
        factor_readout_spikes: np.ndarray,
        status_spikes: np.ndarray,
    ) -> np.ndarray:
        factors = np.asarray(factor_readout_spikes)
        status = np.asarray(status_spikes)
        if (
            factors.shape != (EVENT_CAPACITY, FACTOR_COUNT)
            or status.shape != (5,)
            or factors.dtype != np.bool_
            or status.dtype != np.bool_
            or np.count_nonzero(status) != 1
        ):
            raise ValueError("compositor EOS factor/status spike domain changed")
        factor_row_counts = tuple(int(np.count_nonzero(row)) for row in factors)
        expected_row_counts = (1, 0) if status[1] else (1, 1) if status[2] else (0, 0)
        if factor_row_counts != expected_row_counts:
            raise ValueError("compositor EOS factor cardinality changed")
        slot_enable = np.zeros(EVENT_CAPACITY, dtype=np.bool_)
        if status[1]:
            slot_enable[0] = True
        elif status[2]:
            slot_enable[:] = True
        return factors & slot_enable[:, None]


_COMPOSITOR_EOS_STATUS_RECORD = {
    "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
    "status_order": (
        "ZERO_EVENTS",
        "ONE_EVENT",
        "TWO_EVENTS",
        "STOP_OVERLAPPING_FACTOR_EVENTS",
        "STOP_EVENT_CAPACITY_2",
    ),
    "status_candidate_cells": 5,
    "status_output_relays": 5,
    "factor_copy_relays": EVENT_CAPACITY * FACTOR_COUNT,
    "microticks": {
        "nondestructive_slot_factor_occupied_and_poison_read": 0,
        "ordinary_count_and_poison_status_candidates": 1,
        "one_hot_status_and_poison_gated_factor_outputs": 2,
        "case_reset_relay": 3,
        "complete_case_reset": 4,
        "next_case_event_ready_initialization": 5,
    },
    "threshold_pa_hex": (5_000.0).hex(),
    "poison_inhibition_pa_hex": (-5_000.0).hex(),
    "ordinary_status_rules": {
        "zero": "+5000 base minus each occupied and poison readout",
        "one": "+5000 from slot-zero occupied minus slot-one occupied and poison",
        "two": "+2500 from each occupied slot minus poison",
    },
    "poison_status_rules": {
        "overlap": "+5000 overlap readout minus capacity readout",
        "capacity": "+5000 capacity readout minus overlap readout",
    },
    "factor_output_rule": (
        "+0 factor readouts and the +1 ONE/TWO candidate spikes meet at +2 factor "
        "copy relays; ONE requires row cardinalities 1/0, TWO requires 1/1, and "
        "ZERO/overlap/capacity require 0/0 and enable no factor relay"
    ),
    "same_microtick_spike_influence": False,
    "causal_reset_rule": (
        "the +2 status/output observation drives the +3 case-reset relay, whose spike "
        "delivers reset current to every scoped compartment only at +4; an independent "
        "case-setup phase sets event-ready at +5 for a first read no earlier than +6"
    ),
    "case_phase_entry_rule": (
        "after any event completes at event +10, a ready-gated one-hot EOS request at "
        "+11 selects EOS local read +0 only at global +12 and excludes ingress; for "
        "ZERO_EVENTS, case setup initializes ready and the exclusive EOS request reads "
        "empty occupied/poison storage without any accepted ingress"
    ),
    "mandatory_outcomes": {
        "zero": ((0.0, 0.0), (0.0, 0.0), (True, False, False, False, False)),
        "one": ((5_000.0, 0.0), (0.0, 0.0), (False, True, False, False, False)),
        "two": ((5_000.0, 5_000.0), (0.0, 0.0), (False, False, True, False, False)),
        "overlap": ((0.0, 0.0), (5_000.0, 0.0), (False, False, False, True, False)),
        "capacity": ((0.0, 0.0), (0.0, 5_000.0), (False, False, False, False, True)),
    },
    "complete_reset_scope": (
        "accepted slot storage/readout",
        "ingress identity storage/readout",
        "pre-event occupancy snapshot storage/readout",
        "overlap/capacity poison storage/readout",
        "case poison-halt storage/readout",
        "event-ready storage/readout",
        "admission-block storage/readout",
        "comparison, OR, decision, admission, copy, EOS candidate/output relays",
    ),
}
COMPOSITOR_EOS_STATUS_SHA256 = _namespaced_sha256(
    _COMPOSITOR_EOS_STATUS_NAMESPACE,
    _COMPOSITOR_EOS_STATUS_RECORD,
)
COMPOSITOR_EOS_STATUS_RECORD = _freeze_record(_COMPOSITOR_EOS_STATUS_RECORD)

_COMPOSITOR_CIRCUIT_RECORD = {
    "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
    "compositor_ingress_buffer_sha256": COMPOSITOR_INGRESS_BUFFER_SHA256,
    "compositor_admission_sha256": COMPOSITOR_ADMISSION_SHA256,
    "compositor_eos_status_sha256": COMPOSITOR_EOS_STATUS_SHA256,
    "cell_model": "custom current-integrating threshold/reset neurons",
    "event_capacity": EVENT_CAPACITY,
    "event_slot_count": EVENT_CAPACITY,
    "factor_cells_per_slot": FACTOR_COUNT,
    "boundary_cells": _COMPOSITOR_BOUNDARY_CELL_COUNT,
    "token_position_cells": _COMPOSITOR_TOKEN_POSITION_CELL_COUNT,
    "duplicate_key_cells": _COMPOSITOR_DUPLICATE_KEY_CELL_COUNT,
    "case_persistent_compartments": {
        "slot_factor_storage": EVENT_CAPACITY * FACTOR_COUNT,
        "slot_factor_readout": EVENT_CAPACITY * FACTOR_COUNT,
        "slot_start_storage": (EVENT_CAPACITY * _COMPOSITOR_BOUNDARY_CELL_COUNT),
        "slot_start_readout": (EVENT_CAPACITY * _COMPOSITOR_BOUNDARY_CELL_COUNT),
        "slot_stop_storage": EVENT_CAPACITY * _COMPOSITOR_BOUNDARY_CELL_COUNT,
        "slot_stop_readout": EVENT_CAPACITY * _COMPOSITOR_BOUNDARY_CELL_COUNT,
        "slot_footprint_storage": (EVENT_CAPACITY * _COMPOSITOR_TOKEN_POSITION_CELL_COUNT),
        "slot_footprint_readout": (EVENT_CAPACITY * _COMPOSITOR_TOKEN_POSITION_CELL_COUNT),
        "slot_occupied_storage": EVENT_CAPACITY,
        "slot_occupied_readout": EVENT_CAPACITY,
        "overlap_poison_storage": 1,
        "overlap_poison_readout": 1,
        "capacity_poison_storage": 1,
        "capacity_poison_readout": 1,
        "case_poison_halt_storage": 1,
        "case_poison_halt_readout": 1,
        "event_ready_storage": 1,
        "event_ready_readout": 1,
    },
    "event_ephemeral_no_leak_compartments": {
        "ingress_identity_storage": _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
        "ingress_identity_readout": _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
        "pre_event_occupancy_snapshot_storage": EVENT_CAPACITY,
        "pre_event_occupancy_snapshot_readout": EVENT_CAPACITY,
        "admission_block_storage": 1,
        "admission_block_readout": 1,
    },
    "transient_population_counts": {
        "ingress_gate_cells": _COMPOSITOR_INGRESS_IDENTITY_CELL_COUNT,
        "duplicate_key_coincidence_cells": _COMPOSITOR_DUPLICATE_KEY_CELL_COUNT,
        "overlap_coincidence_cells": _COMPOSITOR_OVERLAP_COINCIDENCE_CELL_COUNT,
        "duplicate_or_relay": 1,
        "overlap_or_relay": 1,
        "duplicate_decision_relay": 1,
        "distinct_overlap_decision_relay": 1,
        "capacity_decision_relay": 1,
        "poison_slot_reset_relay": 1,
        "slot_zero_admission_cell": 1,
        "slot_one_admission_cell": 1,
        "admitted_slot_relays": EVENT_CAPACITY,
        "slot_copy_coincidence_cells": _COMPOSITOR_SLOT_COPY_OUTPUT_CELL_COUNT,
        "ephemeral_reset_relay": 1,
        "eos_status_candidate_cells": 5,
        "eos_status_output_relays": 5,
        "case_reset_relay": 1,
        "ingress_or_eos_phase_request_cells": 2,
    },
    "eos_status_relays": {
        "zero": 1,
        "one": 1,
        "two": 1,
        "overlap": 1,
        "capacity": 1,
    },
    "output_factor_relays": EVENT_CAPACITY * FACTOR_COUNT,
    "current_pa_hex": {
        "full": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "half": float(v4.RELAY_COPY_CURRENT_PA / 2.0).hex(),
        "duplicate_key_leg": float(v4.RELAY_COPY_CURRENT_PA / 5.0).hex(),
        "reset_inhibition": float(v4.BASKET_INHIBITION_CURRENT_PA).hex(),
        "persistent_storage_set": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "persistent_read_gate": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "persistent_storage_coupling": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    },
    "threshold_pa_hex": {
        "factor_start_stop_footprint_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
        "occupied_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
        "overlap_poison_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
        "capacity_poison_readout": float(2.0 * v4.RELAY_COPY_CURRENT_PA).hex(),
        "duplicate_key": float(v4.RELAY_COPY_CURRENT_PA * 6.0 / 5.0).hex(),
        "overlap_coincidence": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "duplicate_or": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "overlap_or": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "duplicate_decision": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "distinct_overlap_decision": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "capacity": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "admission": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "copy_coincidence": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "ingress_gate": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "output_or_stop_relay": float(v4.RELAY_COPY_CURRENT_PA).hex(),
        "reset": float(v4.RELAY_COPY_CURRENT_PA).hex(),
    },
    "event_input": (
        "at -1 one delayed physical factor relay, one-hot start/stop boundary spikes, "
        "and one spike for every covered token position meet the event-ready and "
        "case-poison halt readouts at 62 explicitly counted +0 gate cells"
    ),
    "causal_delay_rule": (
        "every emitted spike may influence only a threshold or storage compartment "
        "at a strictly later rooted microtick"
    ),
    "same_microtick_spike_influence": False,
    "event_microticks": {
        "ingress_attempt_and_case_poison_halt_read": -1,
        "ingress_gate_and_slot_nondestructive_read": 0,
        "ingress_capture_equality_intersection_and_occupancy_snapshot_set": 1,
        "duplicate_and_overlap_or_relays": 2,
        "duplicate_and_distinct_overlap_decisions": 3,
        "capacity_decision": 4,
        "poison_block_set_and_poison_slot_reset_relay": 5,
        "poison_slot_reset_arrival_and_slot_zero_admission": 6,
        "slot_one_admission": 7,
        "admitted_slot_relays_and_ingress_copy_read": 8,
        "copy_coincidence_and_ephemeral_reset_relay": 9,
        "slot_storage_set_ephemeral_reset_and_ready_rearm": 10,
    },
    "state_timing": {
        "stored_slot_read_tick": 0,
        "stored_slot_read_is_nondestructive": True,
        "case_poison_halt_read_tick": -1,
        "event_ready_read_tick": -1,
        "ingress_gate_tick": 0,
        "ingress_capture_tick": 1,
        "event_ready_clear_tick": 1,
        "occupancy_snapshot_set_tick": 1,
        "poison_storage_set_ticks": {"overlap": 5, "capacity": 5},
        "case_poison_halt_set_tick": 5,
        "admission_block_set_tick": 5,
        "poison_slot_reset_relay_tick": 5,
        "poison_slot_reset_tick": 6,
        "admission_block_read_tick": 6,
        "ingress_copy_read_tick": 8,
        "copy_coincidence_tick": 9,
        "ephemeral_reset_relay_tick": 9,
        "slot_storage_set_tick": 10,
        "ephemeral_reset_tick": 10,
        "event_ready_rearm_tick": 10,
        "next_ingress_or_eos_request_tick": 11,
        "selected_phase_gate_tick": 12,
        "event_gate_stride_ticks": 12,
    },
    "causal_delivery_edges": (
        "-1 ingress attempts plus ready/halt readouts -> +0 ingress gate cells",
        "+0 permitted ingress and slot readout spikes -> +1 capture/equality/intersection",
        "+0 accepted factor gate spike -> +1 event-ready storage clear",
        "+0 occupied-slot readout spikes -> +1 occupancy snapshot storage",
        "+1 equality/intersection spikes -> +2 duplicate/overlap OR relays",
        "+2 OR spikes -> +3 duplicate/distinct-overlap decision relays",
        "+3 decisions and +3 occupancy snapshot readouts -> +4 capacity cell",
        "+3 duplicate/overlap or +4 capacity spikes -> +5 status/block storage",
        "+3/+4 poison decisions -> +5 poison-slot reset relay",
        "+5 poison-slot reset spike -> +6 accepted-slot storage reset",
        "+3/+4 classification spikes -> +6 slot-zero admission inhibition",
        "+5 occupancy snapshot readout -> +6 slot-zero admission cell",
        "+6 occupancy and admission-block readouts -> +7 slot-one admission cell",
        "+6/+7 admission spikes -> +8 selected-slot relays",
        "+8 selected-slot and ingress readout spikes -> +9 copy-coincidence cells",
        "+8 completion phase -> +9 ephemeral-reset relay",
        "+9 copy spikes -> +10 admitted slot storage",
        "+9 ephemeral-reset spike -> +10 ephemeral reset and event-ready rearm",
        "+10 ready state -> +11 one-hot ingress-or-EOS request -> +12 selected gate",
    ),
    "eos_microticks": {
        "nondestructive_slot_factor_occupied_and_poison_read": 0,
        "ordinary_count_and_poison_status_candidates": 1,
        "one_hot_status_and_poison_gated_factor_outputs": 2,
        "case_reset_relay": 3,
        "complete_case_reset": 4,
        "next_case_event_ready_initialization": 5,
    },
    "classification_precedence": (
        "exact duplicate decision",
        "distinct overlap decision inhibited by the prior duplicate OR spike",
        "capacity decision inhibited by either prior classification decision",
        "persistent one-hot status and admission block",
        "slot-zero admission",
        "slot-one admission from the frozen pre-event occupancy snapshot",
        "delayed buffered-identity copy",
    ),
    "duplicate_precedes_overlap_and_capacity": True,
    "duplicate_detector_rule": (
        "each slot owns 24x13x13 key cells; the matching key receives six 1000 pA "
        "legs from stored and ingress factor/start/stop one-hots, so only exact "
        "identity reaches the 6000 pA threshold at +1; all 8112 key cells converge "
        "on one explicitly counted +2 duplicate OR relay"
    ),
    "exact_duplicate_rule": (
        "same factor, start, and stop key emits the +3 duplicate decision, sets only "
        "the +5 admission block, preserves accepted slots, and creates no copy spike"
    ),
    "duplicate_precedence_current_rule": (
        "the +2 duplicate OR supplies +5000 pA to the +3 duplicate decision and "
        "-5000 pA to the +3 overlap decision; the +3 duplicate decision supplies "
        "-5000 pA to +4 capacity and, through rooted delays, blocks both admissions"
    ),
    "overlap_detector_rule": (
        "each covered position contributes 2500 pA from stored occupancy and "
        "2500 pA from ingress occupancy; any distinct shared position reaches "
        "one of 24 per-slot +1 coincidence thresholds and all converge on the "
        "separately counted +2 overlap OR relay"
    ),
    "overlap_rule": (
        "at +3 the overlap OR drive survives only when duplicate inhibition is absent; "
        "that decision sets only overlap poison plus admission block at +5 and resets "
        "all accepted slots at +6"
    ),
    "capacity_detector_rule": (
        "at +4 each frozen pre-event occupied snapshot contributes 2500 pA, while "
        "either +3 duplicate or distinct-overlap decision contributes -5000 pA; only "
        "a third nonduplicate nonoverlapping event reaches the 5000 pA threshold"
    ),
    "capacity_rule": (
        "a third nonduplicate, nonoverlapping event sets only capacity poison plus "
        "admission block at +5 and resets every accepted slot at +6"
    ),
    "poison_status_mutual_exclusion_rule": (
        "duplicate inhibition prevents the distinct-overlap decision and both prior "
        "decisions inhibit capacity; therefore overlap and capacity set axons are "
        "physically one-hot before either persistent status storage is addressed"
    ),
    "absorbing_first_poison_rule": (
        "the first overlap or capacity decision sets its one-hot status and the case-"
        "poison halt storage at +5; before every later attempted event, a -1 gated "
        "halt readout inhibits all +0 ingress gates, so no later capture, comparison, "
        "classification, admission, cross-set, or slot refill can occur"
    ),
    "event_serialization_rule": (
        "event-ready storage is initialized to 5000 pA at case setup; its -1 readout "
        "and each -1 identity spike contribute 2500 pA to the +0 ingress gate, while "
        "case halt contributes -5000 pA; acceptance clears ready at +1, and only the "
        "+9 completion-relay spike rearms it at +10 for the earliest +11 attempt/+12 gate"
    ),
    "minimum_event_gate_stride_ticks": 12,
    "case_phase_serialization": {
        "post_event_completion_tick": 10,
        "ready_gated_one_hot_request_tick": 11,
        "selected_next_ingress_or_eos_gate_tick": 12,
        "ingress_and_eos_requests_mutually_exclusive": True,
        "request_domain": "exactly one of next-ingress or EOS",
        "invalid_request_rule": (
            "zero or two request spikes fail closed before either selected gate and "
            "before any persistent or ephemeral state mutation"
        ),
        "request_gate_rule": (
            "the exact one-hot request and event-ready readout emitted at +11 drive "
            "only the corresponding +12 ingress or EOS gate; without ready both remain silent"
        ),
        "zero_event_path": (
            "case setup initializes ready; with no ingress accepted, one EOS request "
            "selects the EOS read gate over empty occupied/poison storage"
        ),
        "early_eos_rule": (
            "before ready rearm, EOS request gates remain silent and cannot inspect "
            "ingress, snapshot, block, poison-reset, admission, or copy transients"
        ),
    },
    "mandatory_absorbing_poison_controls": {
        "overlap_first_then_capacity_pressure": {
            "event_start_stop_pairs": ((0, 4), (2, 6), (0, 4), (4, 8), (8, 12)),
            "first_poison_event_index": 1,
            "expected_status_latches": (True, False),
            "expected_eos_status": "STOP_OVERLAPPING_FACTOR_EVENTS",
            "expected_factors": (),
        },
        "capacity_first_then_overlap_pressure": {
            "event_start_stop_pairs": ((0, 4), (4, 8), (8, 12), (0, 4), (2, 6)),
            "first_poison_event_index": 2,
            "expected_status_latches": (False, True),
            "expected_eos_status": "STOP_EVENT_CAPACITY_2",
            "expected_factors": (),
        },
    },
    "admission_rule": (
        "the rooted staged admission primitive reads only the frozen pre-event "
        "occupancy snapshot and admission classification; +6 chooses empty slot zero, "
        "+7 chooses slot one only when slot zero was previously occupied, and neither "
        "stage can write storage directly"
    ),
    "admission_current_rule": (
        "+6 slot zero receives +5000 pA base drive minus its 5000 pA occupancy and "
        "classification inhibition; +7 slot one receives +5000 pA only from slot-zero "
        "pre-event occupancy, minus slot-one occupancy and the +6 block readout"
    ),
    "buffered_copy_rule": (
        "the +1 captured ingress identity remains in 62 no-leak storage compartments through "
        "classification; only a +8 selected-slot relay and +8 ingress readout can "
        "coincide at +9, and only the resulting +9 spikes set slot storage at +10"
    ),
    "ordering_rule": (
        "factor relays arrive in physical left-to-right span scan order; accepted "
        "slot order is output order"
    ),
    "one_hot_copy_rule": (
        "at physical EOS, a 5000 pA read gate nondestructively reads each accepted "
        "slot's single factor storage; only its corresponding 24-way output relay "
        "fires, and duplicate collapse never creates another"
    ),
    "stop_rule": (
        "overlap or capacity sets its own one-hot 5000 pA poison storage and the "
        "admission block at +5; prior decision spikes hard-reset every accepted slot "
        "at +6, before any admission or copy output, and EOS reads both status latches"
    ),
    "eos_status_rule": (
        "the rooted five-way EOS circuit reads occupied and poison storage at +0, "
        "forms mutually exclusive ordinary-count/poison candidates at +1, emits "
        "exactly one status relay at +2, and permits factor outputs only for ONE/TWO"
    ),
    "persistent_state_rule": (
        "the two accepted event-slot storage groups, both one-hot poison-status, and "
        "case-poison halt "
        "storage compartments have exactly zero leak across arbitrary nondestructive "
        "reads; ingress, occupancy snapshot, and admission-block storage are explicit "
        "event-ephemeral no-leak compartments reset at +10, event-ready rearms only "
        "after that reset, and the EOS +3 reset relay clears all at +4"
    ),
    "no_partial_output_rule": (
        "factor output relays require both +0 factor readout and a +1 ONE/TWO status "
        "candidate at +2; ZERO/overlap/capacity cannot enable factors, and the +3 "
        "reset relay fires only after the one-hot status/output observation and reset "
        "current arrives one microtick later"
    ),
    "host_factor_selection_union_filter_sort": False,
    "host_equality_overlap_capacity_or_sort": False,
    "host_ingress_replay_or_occupancy_state": False,
    "host_winner_or_tie_break": False,
    "full_circuit_execution_deferred": True,
}
COMPOSITOR_CIRCUIT_SHA256 = _namespaced_sha256(
    _COMPOSITOR_CIRCUIT_NAMESPACE,
    _COMPOSITOR_CIRCUIT_RECORD,
)
COMPOSITOR_CIRCUIT_RECORD = _freeze_record(_COMPOSITOR_CIRCUIT_RECORD)


def _lesson_tokens(
    triad: tuple[str, str, str],
    template_index: int,
    target_index: int,
) -> tuple[str, str, str, str]:
    _split, direction, cue_slot, selected_slot = TEMPLATE_SPECS[template_index]
    target = triad[target_index]
    distractors = [
        triad[(target_index + 1) % len(triad)],
        triad[(target_index + 2) % len(triad)],
    ]
    if template_index % 2:
        distractors.reverse()
    tokens: list[str | None] = [None, None, None, None]
    tokens[cue_slot] = CUE_TOKENS[direction]
    tokens[selected_slot] = target
    remaining_slots = [slot for slot, token in enumerate(tokens) if token is None]
    for slot, distractor in zip(remaining_slots, distractors, strict=True):
        tokens[slot] = distractor
    if any(token is None for token in tokens):
        raise RuntimeError("teacher-gated V5 lesson geometry is incomplete")
    return tuple(tokens)  # type: ignore[return-value]


def _build_atomic_curriculum() -> tuple[
    tuple[AtomicContrastGroup, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    groups: list[AtomicContrastGroup] = []
    fit_ids: list[str] = []
    heldout_ids: list[str] = []
    for triad_index, triad in enumerate(TRIADS):
        for template_index, (split, direction, cue_slot, selected_slot) in enumerate(
            TEMPLATE_SPECS
        ):
            group_index = len(groups)
            lessons: list[AtomicLesson] = []
            for target_index, factor in enumerate(triad):
                tokens = _lesson_tokens(triad, template_index, target_index)
                record = {
                    "group_index": group_index,
                    "triad_index": triad_index,
                    "template_index": template_index,
                    "target_index": target_index,
                    "direction": direction,
                    "cue_token": CUE_TOKENS[direction],
                    "cue_slot": cue_slot,
                    "selected_slot": selected_slot,
                    "tokens": tokens,
                    "factor": factor,
                    "split": split,
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
                    template_index=template_index,
                    direction=direction,
                    cue_slot=cue_slot,
                    selected_slot=selected_slot,
                    split=split,
                    lessons=tuple(lessons),
                )
            )
    return tuple(groups), tuple(fit_ids), tuple(heldout_ids)


def _spent_inputs() -> tuple[tuple[str, str, str, str], ...]:
    v3_groups, _ = v3.self_posed_local_span_curriculum()
    rows = {lesson.teacher_span.canonical_tokens for group in v3_groups for lesson in group.lessons}
    rows.update(lesson.tokens for lesson in v4.PROTOCOL.lessons)
    return tuple(sorted(rows))


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
        raise RuntimeError("teacher-gated V5 atomic curriculum shape changed")
    if len(set(TAUGHT_FACTORS)) != 9 or any(
        factor not in FACTOR_REGISTRY for factor in TAUGHT_FACTORS
    ):
        raise RuntimeError("teacher-gated V5 taught-factor registry changed")
    for group in groups:
        if (
            len(group.lessons) != 3
            or len({tuple(sorted(lesson.tokens)) for lesson in group.lessons}) != 1
            or {lesson.factor for lesson in group.lessons} != set(TRIADS[group.triad_index])
            or any(
                lesson.tokens[lesson.cue_slot] != lesson.cue_token
                or lesson.tokens[lesson.selected_slot] != lesson.factor
                for lesson in group.lessons
            )
        ):
            raise RuntimeError("teacher-gated V5 matched contrast geometry changed")

    fit_set = set(fit_ids)
    heldout_set = set(heldout_ids)
    selected_fit = Counter(
        (lesson.factor, lesson.selected_slot) for lesson in lessons if lesson.sha256 in fit_set
    )
    unselected_fit = Counter(
        (factor, slot)
        for lesson in lessons
        if lesson.sha256 in fit_set
        for slot, factor in enumerate(lesson.tokens)
        if factor in TAUGHT_FACTORS and slot != lesson.selected_slot
    )
    if selected_fit != Counter(
        {(factor, slot): 1 for factor in TAUGHT_FACTORS for slot in range(4)}
    ) or unselected_fit != Counter(
        {(factor, slot): 2 for factor in TAUGHT_FACTORS for slot in range(4)}
    ):
        raise RuntimeError("teacher-gated V5 factor-slot marginals changed")
    if Counter(lesson.factor for lesson in lessons if lesson.sha256 in fit_set) != Counter(
        {factor: 4 for factor in TAUGHT_FACTORS}
    ) or Counter(lesson.factor for lesson in lessons if lesson.sha256 in heldout_set) != Counter(
        {factor: 2 for factor in TAUGHT_FACTORS}
    ):
        raise RuntimeError("teacher-gated V5 factor capacity changed")

    fit_inputs = {lesson.tokens for lesson in lessons if lesson.sha256 in fit_set}
    heldout_inputs = {lesson.tokens for lesson in lessons if lesson.sha256 in heldout_set}
    if fit_inputs & heldout_inputs or (fit_inputs | heldout_inputs) & set(_spent_inputs()):
        raise RuntimeError("teacher-gated V5 input is not exact-disjoint")
    fit_cue_tokens = {lesson.cue_token for lesson in lessons if lesson.sha256 in fit_set}
    fit_cue_slots = {lesson.cue_slot for lesson in lessons if lesson.sha256 in fit_set}
    fit_selected_slots = {lesson.selected_slot for lesson in lessons if lesson.sha256 in fit_set}
    if any(
        lesson.cue_token not in fit_cue_tokens
        or lesson.cue_slot not in fit_cue_slots
        or lesson.selected_slot not in fit_selected_slots
        or lesson.factor not in TAUGHT_FACTORS
        for lesson in lessons
        if lesson.sha256 in heldout_set
    ):
        raise RuntimeError("teacher-gated V5 heldout component support changed")
    return lessons


def _candidate_cells(token: str, salt: int) -> tuple[int, ...]:
    selected: set[int] = set()
    counter = 0
    while len(selected) < ASSEMBLY_ACTIVE_CELL_COUNT:
        digest = sha256()
        digest.update(_LEXICAL_ASSEMBLY_NAMESPACE)
        digest.update(token.encode("utf-8"))
        digest.update(salt.to_bytes(4, "little"))
        digest.update(counter.to_bytes(8, "little"))
        block = digest.digest()
        for offset in range(0, len(block), 4):
            selected.add(int.from_bytes(block[offset : offset + 4], "little") % LEXICAL_CELL_COUNT)
            if len(selected) == ASSEMBLY_ACTIVE_CELL_COUNT:
                break
        counter += 1
    return tuple(sorted(selected))


def _build_lexical_assemblies(tokens: tuple[str, ...]) -> tuple[LexicalAssembly, ...]:
    rows: list[LexicalAssembly] = []
    for token in sorted(set(tokens)):
        for salt in range(65_536):
            cells = _candidate_cells(token, salt)
            if all(
                len(set(cells) & set(row.cells)) <= ASSEMBLY_MAX_PAIRWISE_OVERLAP for row in rows
            ):
                record = {"token": token, "salt": salt, "cells": cells}
                rows.append(
                    LexicalAssembly(
                        token=token,
                        salt=salt,
                        cells=cells,
                        sha256=_namespaced_sha256(
                            _LEXICAL_ASSEMBLY_NAMESPACE,
                            record,
                        ),
                    )
                )
                break
        else:
            raise RuntimeError("teacher-gated V5 lexical overlap bound exhausted")
    return tuple(rows)


def _build_composition_execution_record(
    composition: tuple[CompositionCase, ...],
    duplicates: tuple[CompositorControl, ...],
    overlaps: tuple[CompositorControl, ...],
    capacities: tuple[CompositorControl, ...],
    *,
    atomic_gate_sha256: str,
    composition_sha256: str,
    duplicate_controls_sha256: str,
    overlap_controls_sha256: str,
    capacity_controls_sha256: str,
) -> tuple[object, str]:
    control_order = (*duplicates, *overlaps, *capacities)
    record = {
        "atomic_gate_sha256": atomic_gate_sha256,
        "compositor_circuit_sha256": COMPOSITOR_CIRCUIT_SHA256,
        "composition_sha256": composition_sha256,
        "duplicate_controls_sha256": duplicate_controls_sha256,
        "overlap_controls_sha256": overlap_controls_sha256,
        "capacity_controls_sha256": capacity_controls_sha256,
        "atomic_prerequisite": "54/54 unique exact",
        "atomic_failure_status": (
            "STOP_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_ATOMIC_READINESS"
        ),
        "atomic_failure_attempt_counts": {
            "composition": 0,
            "duplicate": 0,
            "overlap": 0,
            "capacity": 0,
        },
        "composition_input_inspection_on_atomic_failure": False,
        "composition_case_sha256s": tuple(case.sha256 for case in composition),
        "composition_expected_factors": tuple(case.expected_factors for case in composition),
        "composition_execution_order": "ascending case_index only",
        "composition_batch_complete_before_gate": True,
        "required_composition_attempt_count": COMPOSITION_CASE_COUNT,
        "required_composition_exact": COMPOSITION_CASE_COUNT,
        "allowed_composition_stop_collision_wrong_count": 0,
        "composition_failure_status": (
            "STOP_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_COMPOSITION"
        ),
        "composition_failure_control_attempt_count": 0,
        "control_execution_order": tuple(control.sha256 for control in control_order),
        "control_panel_batch_order": (
            "9 exact-duplicate then 3 duplicate-after-full controls in ascending control_index",
            "3 distinct-overlap then 3 overlap-after-full controls in ascending control_index",
            "3 capacity-three controls in ascending control_index",
        ),
        "control_batch_complete_before_gate": True,
        "required_control_attempt_count": (
            DUPLICATE_CONTROL_COUNT + OVERLAP_CONTROL_COUNT + CAPACITY_CONTROL_COUNT
        ),
        "required_control_exact": {
            "duplicate": DUPLICATE_CONTROL_COUNT,
            "overlap": OVERLAP_CONTROL_COUNT,
            "capacity": CAPACITY_CONTROL_COUNT,
        },
        "control_failure_status": (
            "STOP_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_COMPOSITOR_CONTROLS"
        ),
        "pass_status": "PASS_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5",
        "no_partial_result_acceptance": True,
        "retry": False,
        "parameter_tuning": False,
        "execution_deferred": True,
    }
    return _freeze_record(record), _namespaced_sha256(
        _COMPOSITION_EXECUTION_NAMESPACE,
        record,
    )


def _build_electrical_preflight_contract_record(
    *,
    selector_full_preflight_sha256: str,
    teacher_scout_circuit_sha256: str,
    control_panel_sha256: str,
) -> tuple[object, str]:
    record = {
        "direct_readout_channel_sha256": DIRECT_READOUT_CHANNEL_SHA256,
        "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
        "selector_circuit_sha256": SELECTOR_CIRCUIT_SHA256,
        "selector_full_preflight_sha256": selector_full_preflight_sha256,
        "v4_wta_reference_sha256": V4_WTA_REFERENCE_SHA256,
        "v4_full_factor_controls_sha256": (v4.PROTOCOL.wta_electrical_controls_sha256),
        "teacher_scout_circuit_sha256": teacher_scout_circuit_sha256,
        "control_panel_sha256": control_panel_sha256,
        "must_pass_before_state_allocation": True,
        "execution_order": (
            "authenticate complete V5 protocol",
            "execute persistent-latch idle, repeated-read, reset, exact-domain, and "
            "direct-collision controls",
            "execute all 11 full selector controls and causal interventions",
            "execute all 5 complete passed-V4 factor WTA controls",
            "execute teacher/scout scratch, teacher-off, and cell-off controls",
            "authenticate the complete observed electrical trace root",
            "allocate uint8 E/H learner arrays",
        ),
        "selector_principal_controls_already_executed": (SELECTOR_ELECTRICAL_CONTROL_COUNT),
        "persistent_latch_controls_already_executed": 0,
        "factor_principal_controls_already_executed": len(v4.PROTOCOL.wta_electrical_controls),
        "already_executed_scope": (
            "principal race_many first-spike tiers only; they are mechanism evidence "
            "and not the mandatory learner full-circuit preflight"
        ),
        "required_observed_trace_roots": (
            "persistent-latch idle/repeated-read/reset trace",
            "selector phase/cue/address/latch/content trace",
            "selector basket/collision/delayed-relay trace",
            "factor principal/basket/collision/delayed-relay trace",
            "teacher/scout/match/update/reset trace",
            "circuit-off and causal-intervention traces",
        ),
        "expected_metadata_is_not_observed_evidence": True,
        "on_any_failure": (
            "return STOP_SELF_POSED_TEACHER_GATED_FACTOR_WORKSPACE_V5_"
            "ELECTRICAL_PREFLIGHT before allocating learner arrays"
        ),
        "retry": False,
        "parameter_tuning": False,
        "full_circuit_execution_deferred": True,
    }
    return _freeze_record(record), _namespaced_sha256(
        _ELECTRICAL_PREFLIGHT_CONTRACT_NAMESPACE,
        record,
    )


def _build_future_authentication_record(
    *,
    roots: dict[str, str],
    fit_schedule: tuple[str, ...],
    composition_execution_sha256: str,
) -> tuple[object, str]:
    control_batch_count = DUPLICATE_CONTROL_COUNT + OVERLAP_CONTROL_COUNT + CAPACITY_CONTROL_COUNT
    record = {
        "hash_algorithm": "SHA-256",
        "canonical_encoding": (
            "UTF-8 canonical JSON with sorted keys, compact separators, and no NaN; "
            "ndarrays use name/dtype/shape/length-framed C-order bytes"
        ),
        "future_deep_interface": (
            {
                "operation": "build_frozen_teacher_gated_factor_bridge",
                "input": "FrozenTeacherGatedFactorProtocol",
                "output": "FrozenTeacherGatedFactorBridge",
            },
            {
                "operation": "FrozenTeacherGatedFactorBridge.resolve",
                "input": "AuthenticatedSpanBatch",
                "output": "TeacherGatedFactorResolution",
            },
        ),
        "protocol_module_exposes_operations": False,
        "checkpoint_namespace": "teacher-gated-factor-workspace-v5-checkpoint",
        "checkpoint_root_dependencies": tuple(sorted(roots.items())),
        "checkpoint_metadata": {
            "protocol_root_record": "complete authenticated PROTOCOL.root_record()",
            "fit_schedule_sha256s": fit_schedule,
            "state_dtype": TICK_DTYPE,
            "state_shape": FUTURE_STATE_SHAPE,
            "tick_cap": TICK_CAP,
            "state_bytes": FUTURE_STATE_BYTES,
            "electrical_trace_root_required": True,
            "projection_panel_root_required": True,
            "factor_registry_root_required": True,
            "exact_update_telemetry_required": True,
            "composition_and_control_attempt_counts_at_freeze": 0,
        },
        "checkpoint_arrays": (
            ("excitatory_ticks", TICK_DTYPE, FUTURE_STATE_SHAPE),
            ("inhibitory_ticks", TICK_DTYPE, FUTURE_STATE_SHAPE),
        ),
        "checkpoint_freeze": (
            "copy both arrays, mark them read-only, frame their exact bytes, hash "
            "metadata plus E then H, and reject any later byte or metadata change"
        ),
        "atomic_batch": {
            "current_shape": (LESSON_COUNT, FACTOR_COUNT + 1),
            "race_many_calls": 1,
            "checkpoint_authentications": 2,
            "per_span_hashing": False,
        },
        "atomic_result_schema": (
            "protocol/checkpoint/electrical/projection/factor/local-rule roots",
            "ordered 54 lesson IDs, raw score rows, principal tiers, relay traces, "
            "decisions, expected factors, and exact update telemetry",
            "atomic pass/mismatch counts and zero composition/control attempts",
        ),
        "atomic_result_gate": (
            "authenticate the result root, then require exactly 54 unique correct "
            "factor relays with zero STOP/collision/wrong decisions"
        ),
        "composition_execution_sha256": composition_execution_sha256,
        "composition_batch_checkpoint_authentications": 2,
        "composition_result_schema": (
            "authenticated atomic-result and checkpoint roots, ordered 18 case IDs, "
            "observed physical event/STOP traces, expected ordered factors, exact "
            "pass count, and all control attempt counts"
        ),
        "control_batch_control_count": control_batch_count,
        "control_batch_checkpoint_authentications": 2,
        "compositor_control_result_schema": (
            "authenticated composition-result/checkpoint/compositor roots, ordered "
            "21 control IDs and physical duplicate/overlap/capacity, one-hot status, "
            "and exact EOS output traces"
        ),
        "result_hash_order": (
            "checkpoint",
            "atomic-result",
            "composition-result",
            "compositor-control-result",
        ),
        "authentication_sequence": (
            "authenticate protocol before electrical preflight",
            "authenticate full observed electrical root before state allocation",
            "freeze and authenticate checkpoint after the one FIT pass",
            "authenticate checkpoint immediately before and after the one 54-row batch",
            "authenticate atomic result before opening composition",
            "authenticate checkpoint immediately before and after the 18-case batch",
            "authenticate composition result before opening compositor controls",
            (
                "authenticate checkpoint immediately before and after the "
                f"{control_batch_count}-control batch"
            ),
            "authenticate compositor-control result before returning PASS",
        ),
        "terminal_resolve_sequence": (
            "authenticate protocol root",
            "authenticate frozen checkpoint",
            "authenticate atomic result",
            "inspect the 54-case atomic gate and exact downstream attempt counts",
            "on atomic STOP prove composition/control results absent and all downstream "
            "attempts zero, then raise before reading any requested batch field",
            "on atomic PASS authenticate the composition result",
            "inspect the exact 18-case composition gate and compositor-control attempt count",
            "on composition STOP prove the control result absent and control attempts "
            "zero, then raise before reading any requested batch field",
            "on composition PASS authenticate the compositor-control result",
            f"inspect the exact {control_batch_count}-control compositor gate",
            "on compositor-control STOP raise before reading any requested batch field",
            "only after all three gates PASS authenticate and inspect the requested batch",
        ),
        "terminal_gate_cardinalities": {
            "atomic": LESSON_COUNT,
            "composition": COMPOSITION_CASE_COUNT,
            "compositor_control": control_batch_count,
        },
        "terminal_result_absence_rules": {
            "atomic_stop": (
                "composition and compositor-control results must both be absent; "
                "composition and all compositor-control attempt counts must be zero"
            ),
            "composition_stop": (
                "the compositor-control result must be absent and every compositor-"
                "control attempt count must be zero"
            ),
            "all_other_missing_results": "fail closed before requested-batch access",
        },
        "terminal_batch_access_barrier": (
            "before all three authenticated gates PASS, do not read requested batch "
            "type, length, hash, iterator, or contents"
        ),
        "winner_cell_on_nonunique_tier": "ignored",
        "no_per_lesson_or_per_span_checkpoint_hashing": True,
        "future_hashes_are_schema_bound_not_run_results": True,
        "execution_deferred": True,
    }
    return _freeze_record(record), _namespaced_sha256(
        _FUTURE_AUTHENTICATION_NAMESPACE,
        record,
    )


def _build_protocol() -> tuple[
    FrozenTeacherGatedFactorProtocol,
    object,
    str,
    object,
    str,
    object,
    str,
    object,
    str,
    object,
    str,
    object,
    str,
    object,
    str,
]:
    groups, fit_ids, heldout_ids = _build_atomic_curriculum()
    lessons = _validate_atomic_curriculum(groups, fit_ids, heldout_ids)
    neutrals = _build_neutral_spans()
    if (
        len(neutrals) != NEUTRAL_SPAN_COUNT
        or len({neutral.sha256 for neutral in neutrals}) != NEUTRAL_SPAN_COUNT
        or {neutral.tokens for neutral in neutrals}
        & ({lesson.tokens for lesson in lessons} | set(_spent_inputs()))
    ):
        raise RuntimeError("teacher-gated V5 neutral-span panel changed")
    assemblies = _build_lexical_assemblies(
        tuple(token for lesson in lessons for token in lesson.tokens)
        + tuple(token for neutral in neutrals for token in neutral.tokens)
    )
    atomic_input_sha = _namespaced_sha256(
        _ATOMIC_INPUT_NAMESPACE,
        tuple(sorted(lesson.tokens for lesson in lessons)),
    )
    curriculum_sha = _namespaced_sha256(
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
    spent_reference_sha = _namespaced_sha256(
        _SPENT_INPUT_NAMESPACE,
        _spent_inputs(),
    )
    projection_sha = _namespaced_sha256(
        _PROJECTION_NAMESPACE,
        {
            "cell_count": LEXICAL_CELL_COUNT,
            "active_cell_count": ASSEMBLY_ACTIVE_CELL_COUNT,
            "max_pairwise_overlap": ASSEMBLY_MAX_PAIRWISE_OVERLAP,
            "assemblies": tuple(assembly.record(include_sha256=False) for assembly in assemblies),
        },
    )
    selector_controls = _build_selector_electrical_controls()
    if (
        len(selector_controls) != SELECTOR_ELECTRICAL_CONTROL_COUNT
        or len({control.sha256 for control in selector_controls})
        != SELECTOR_ELECTRICAL_CONTROL_COUNT
    ):
        raise RuntimeError("teacher-gated V5 selector control panel changed")
    selector_controls_sha = _namespaced_sha256(
        _SELECTOR_CONTROLS_NAMESPACE,
        tuple(control.record(include_sha256=False) for control in selector_controls),
    )
    selector_full_preflight_record, selector_full_preflight_sha = (
        _build_selector_full_preflight_record(assemblies, selector_controls)
    )
    local_rule_record, local_rule_sha = _build_local_rule_record(lessons, fit_ids)
    control_panel_record, control_panel_sha = _build_control_panel(local_rule_record)
    teacher_scout_record, teacher_scout_sha = _build_teacher_scout_circuit_record(
        lessons,
        local_rule_record,
        local_rule_sha,
        control_panel_sha,
    )
    electrical_preflight_record, electrical_preflight_sha = (
        _build_electrical_preflight_contract_record(
            selector_full_preflight_sha256=selector_full_preflight_sha,
            teacher_scout_circuit_sha256=teacher_scout_sha,
            control_panel_sha256=control_panel_sha,
        )
    )
    composition = _build_composition_cases(lessons, neutrals)
    duplicates, overlaps, capacities = _build_compositor_controls()
    if (
        len(composition) != COMPOSITION_CASE_COUNT
        or len(duplicates) != DUPLICATE_CONTROL_COUNT
        or len(overlaps) != OVERLAP_CONTROL_COUNT
        or len(capacities) != CAPACITY_CONTROL_COUNT
    ):
        raise RuntimeError("teacher-gated V5 compositor preregistration changed")
    neutral_sha = _namespaced_sha256(
        _NEUTRAL_NAMESPACE,
        tuple(neutral.record(include_sha256=False) for neutral in neutrals),
    )
    composition_sha = _namespaced_sha256(
        _COMPOSITION_NAMESPACE,
        tuple(case.record(include_sha256=False) for case in composition),
    )
    duplicate_sha = _namespaced_sha256(
        _COMPOSITOR_CONTROL_NAMESPACE + b"duplicate\0",
        tuple(control.record(include_sha256=False) for control in duplicates),
    )
    overlap_sha = _namespaced_sha256(
        _COMPOSITOR_CONTROL_NAMESPACE + b"overlap\0",
        tuple(control.record(include_sha256=False) for control in overlaps),
    )
    capacity_sha = _namespaced_sha256(
        _COMPOSITOR_CONTROL_NAMESPACE + b"capacity\0",
        tuple(control.record(include_sha256=False) for control in capacities),
    )
    composition_execution_record, composition_execution_sha = _build_composition_execution_record(
        composition,
        duplicates,
        overlaps,
        capacities,
        atomic_gate_sha256=ATOMIC_GATE_SHA256,
        composition_sha256=composition_sha,
        duplicate_controls_sha256=duplicate_sha,
        overlap_controls_sha256=overlap_sha,
        capacity_controls_sha256=capacity_sha,
    )
    future_authentication_record, future_authentication_sha = _build_future_authentication_record(
        roots={
            "atomic_input_sha256": atomic_input_sha,
            "curriculum_sha256": curriculum_sha,
            "split_sha256": split_sha,
            "spent_input_reference_sha256": spent_reference_sha,
            "projection_sha256": projection_sha,
            "direct_readout_channel_sha256": DIRECT_READOUT_CHANNEL_SHA256,
            "persistent_latch_primitive_sha256": (PERSISTENT_LATCH_PRIMITIVE_SHA256),
            "selector_topology_sha256": SELECTOR_TOPOLOGY_SHA256,
            "selector_electrical_controls_sha256": selector_controls_sha,
            "selector_circuit_sha256": SELECTOR_CIRCUIT_SHA256,
            "selector_full_preflight_sha256": selector_full_preflight_sha,
            "v4_wta_reference_sha256": V4_WTA_REFERENCE_SHA256,
            "local_rule_sha256": local_rule_sha,
            "control_panel_sha256": control_panel_sha,
            "atomic_gate_sha256": ATOMIC_GATE_SHA256,
            "teacher_scout_circuit_sha256": teacher_scout_sha,
            "electrical_preflight_contract_sha256": electrical_preflight_sha,
            "neutral_sha256": neutral_sha,
            "composition_sha256": composition_sha,
            "duplicate_controls_sha256": duplicate_sha,
            "overlap_controls_sha256": overlap_sha,
            "capacity_controls_sha256": capacity_sha,
            "compositor_ingress_buffer_sha256": COMPOSITOR_INGRESS_BUFFER_SHA256,
            "compositor_admission_sha256": COMPOSITOR_ADMISSION_SHA256,
            "compositor_eos_status_sha256": COMPOSITOR_EOS_STATUS_SHA256,
            "compositor_circuit_sha256": COMPOSITOR_CIRCUIT_SHA256,
            "composition_execution_sha256": composition_execution_sha,
        },
        fit_schedule=tuple(local_rule_record["fit_schedule"]["lesson_sha256s"]),
        composition_execution_sha256=composition_execution_sha,
    )
    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": PROTOCOL_STATUS,
        "atomic_input_sha256": atomic_input_sha,
        "curriculum_sha256": curriculum_sha,
        "split_sha256": split_sha,
        "spent_input_reference_sha256": spent_reference_sha,
        "projection_sha256": projection_sha,
        "direct_readout_channel_sha256": DIRECT_READOUT_CHANNEL_SHA256,
        "persistent_latch_primitive_sha256": PERSISTENT_LATCH_PRIMITIVE_SHA256,
        "selector_topology_sha256": SELECTOR_TOPOLOGY_SHA256,
        "selector_electrical_controls_sha256": selector_controls_sha,
        "selector_circuit_sha256": SELECTOR_CIRCUIT_SHA256,
        "selector_full_preflight_sha256": selector_full_preflight_sha,
        "v4_wta_reference_sha256": V4_WTA_REFERENCE_SHA256,
        "local_rule_sha256": local_rule_sha,
        "control_panel_sha256": control_panel_sha,
        "atomic_gate_sha256": ATOMIC_GATE_SHA256,
        "teacher_scout_circuit_sha256": teacher_scout_sha,
        "electrical_preflight_contract_sha256": electrical_preflight_sha,
        "neutral_sha256": neutral_sha,
        "composition_sha256": composition_sha,
        "duplicate_controls_sha256": duplicate_sha,
        "overlap_controls_sha256": overlap_sha,
        "capacity_controls_sha256": capacity_sha,
        "compositor_ingress_buffer_sha256": COMPOSITOR_INGRESS_BUFFER_SHA256,
        "compositor_admission_sha256": COMPOSITOR_ADMISSION_SHA256,
        "compositor_eos_status_sha256": COMPOSITOR_EOS_STATUS_SHA256,
        "compositor_circuit_sha256": COMPOSITOR_CIRCUIT_SHA256,
        "composition_execution_sha256": composition_execution_sha,
        "future_authentication_sha256": future_authentication_sha,
        "group_count": len(groups),
        "lesson_count": len(lessons),
        "fit_count": len(fit_ids),
        "heldout_count": len(heldout_ids),
        "lexical_assembly_count": len(assemblies),
        "selector_electrical_control_count": len(selector_controls),
        "factor_wta_electrical_control_count": len(v4.PROTOCOL.wta_electrical_controls),
        "neutral_span_count": len(neutrals),
        "composition_case_count": len(composition),
        "duplicate_control_count": len(duplicates),
        "overlap_control_count": len(overlaps),
        "capacity_control_count": len(capacities),
        "public_input_count": PUBLIC_INPUT_COUNT,
        "canonical_input_count": CANONICAL_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "composition_result_count": COMPOSITION_RESULT_COUNT,
        "learner_state_constructed": False,
        "learning_executed": False,
        "composition_executed": False,
        "execution_surface_present": False,
        "private_mechanism_primitive_present": True,
        "private_mechanism_primitive_instantiated_at_import": False,
        "learner_or_full_circuit_execution_surface_present": False,
        "selector_principal_controls_executed": True,
        "factor_principal_controls_executed": True,
        "full_selected_content_circuit_executed": False,
        "full_factor_circuit_executed": False,
        "teacher_scout_circuit_executed": False,
        "compositor_circuit_executed": False,
        "composition_attempt_count": 0,
    }
    protocol = FrozenTeacherGatedFactorProtocol(
        scope=PROTOCOL_SCOPE,
        status=PROTOCOL_STATUS,
        atomic_groups=groups,
        fit_lesson_sha256s=fit_ids,
        heldout_lesson_sha256s=heldout_ids,
        lexical_assemblies=assemblies,
        selector_electrical_controls=selector_controls,
        factor_wta_electrical_controls=v4.PROTOCOL.wta_electrical_controls,
        neutral_spans=neutrals,
        composition_cases=composition,
        duplicate_controls=duplicates,
        overlap_controls=overlaps,
        capacity_controls=capacities,
        atomic_input_sha256=atomic_input_sha,
        curriculum_sha256=curriculum_sha,
        split_sha256=split_sha,
        spent_input_reference_sha256=spent_reference_sha,
        projection_sha256=projection_sha,
        direct_readout_channel_sha256=DIRECT_READOUT_CHANNEL_SHA256,
        persistent_latch_primitive_sha256=PERSISTENT_LATCH_PRIMITIVE_SHA256,
        selector_topology_sha256=SELECTOR_TOPOLOGY_SHA256,
        selector_electrical_controls_sha256=selector_controls_sha,
        selector_circuit_sha256=SELECTOR_CIRCUIT_SHA256,
        selector_full_preflight_sha256=selector_full_preflight_sha,
        v4_wta_reference_sha256=V4_WTA_REFERENCE_SHA256,
        local_rule_sha256=local_rule_sha,
        control_panel_sha256=control_panel_sha,
        atomic_gate_sha256=ATOMIC_GATE_SHA256,
        teacher_scout_circuit_sha256=teacher_scout_sha,
        electrical_preflight_contract_sha256=electrical_preflight_sha,
        neutral_sha256=neutral_sha,
        composition_sha256=composition_sha,
        duplicate_controls_sha256=duplicate_sha,
        overlap_controls_sha256=overlap_sha,
        capacity_controls_sha256=capacity_sha,
        compositor_ingress_buffer_sha256=COMPOSITOR_INGRESS_BUFFER_SHA256,
        compositor_admission_sha256=COMPOSITOR_ADMISSION_SHA256,
        compositor_eos_status_sha256=COMPOSITOR_EOS_STATUS_SHA256,
        compositor_circuit_sha256=COMPOSITOR_CIRCUIT_SHA256,
        composition_execution_sha256=composition_execution_sha,
        future_authentication_sha256=future_authentication_sha,
        protocol_sha256=_namespaced_sha256(
            _PROTOCOL_NAMESPACE,
            root_without_protocol,
        ),
        public_input_count=PUBLIC_INPUT_COUNT,
        canonical_input_count=CANONICAL_INPUT_COUNT,
        protected_input_count=PROTECTED_INPUT_COUNT,
        learning_run_count=LEARNING_RUN_COUNT,
        composition_result_count=COMPOSITION_RESULT_COUNT,
    )
    if protocol.root_record(include_protocol_sha256=False) != root_without_protocol:
        raise RuntimeError("teacher-gated V5 protocol root construction diverged")
    if any(
        not _is_sha256(value)
        for name, value in dataclasses.asdict(protocol).items()
        if name.endswith("sha256")
    ):
        raise RuntimeError("teacher-gated V5 protocol contains an invalid root hash")
    return (
        protocol,
        local_rule_record,
        local_rule_sha,
        control_panel_record,
        control_panel_sha,
        selector_full_preflight_record,
        selector_full_preflight_sha,
        teacher_scout_record,
        teacher_scout_sha,
        electrical_preflight_record,
        electrical_preflight_sha,
        composition_execution_record,
        composition_execution_sha,
        future_authentication_record,
        future_authentication_sha,
    )


(
    PROTOCOL,
    LOCAL_RULE_RECORD,
    LOCAL_RULE_SHA256,
    CONTROL_PANEL_RECORD,
    CONTROL_PANEL_SHA256,
    SELECTOR_FULL_PREFLIGHT_RECORD,
    SELECTOR_FULL_PREFLIGHT_SHA256,
    TEACHER_SCOUT_CIRCUIT_RECORD,
    TEACHER_SCOUT_CIRCUIT_SHA256,
    ELECTRICAL_PREFLIGHT_CONTRACT_RECORD,
    ELECTRICAL_PREFLIGHT_CONTRACT_SHA256,
    COMPOSITION_EXECUTION_RECORD,
    COMPOSITION_EXECUTION_SHA256,
    FUTURE_AUTHENTICATION_RECORD,
    FUTURE_AUTHENTICATION_SHA256,
) = _build_protocol()


__all__ = [
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "ASSEMBLY_MAX_PAIRWISE_OVERLAP",
    "ATOMIC_GATE_RECORD",
    "ATOMIC_GATE_SHA256",
    "CANONICAL_INPUT_COUNT",
    "CAPACITY_CONTROL_COUNT",
    "COMPOSITION_CASE_COUNT",
    "COMPOSITION_EXECUTION_RECORD",
    "COMPOSITION_EXECUTION_SHA256",
    "COMPOSITION_RESULT_COUNT",
    "COMPOSITOR_ADMISSION_RECORD",
    "COMPOSITOR_ADMISSION_SHA256",
    "COMPOSITOR_CIRCUIT_RECORD",
    "COMPOSITOR_CIRCUIT_SHA256",
    "COMPOSITOR_EOS_STATUS_RECORD",
    "COMPOSITOR_EOS_STATUS_SHA256",
    "COMPOSITOR_INGRESS_BUFFER_RECORD",
    "COMPOSITOR_INGRESS_BUFFER_SHA256",
    "CONTROL_PANEL_RECORD",
    "CONTROL_PANEL_SHA256",
    "CUE_TOKENS",
    "DIRECT_READOUT_CHANNEL_RECORD",
    "DIRECT_READOUT_CHANNEL_SHA256",
    "DUPLICATE_CONTROL_COUNT",
    "ELECTRICAL_PREFLIGHT_CONTRACT_RECORD",
    "ELECTRICAL_PREFLIGHT_CONTRACT_SHA256",
    "EVENT_CAPACITY",
    "FACTOR_COUNT",
    "FACTOR_REGISTRY",
    "FIT",
    "FIT_COUNT",
    "FUTURE_AUTHENTICATION_RECORD",
    "FUTURE_AUTHENTICATION_SHA256",
    "FUTURE_STATE_BYTES",
    "FUTURE_STATE_RECORD",
    "FUTURE_STATE_SHAPE",
    "GROUP_COUNT",
    "HELDOUT",
    "HELDOUT_COUNT",
    "LEARNING_RUN_COUNT",
    "LESSON_COUNT",
    "LEXICAL_CELL_COUNT",
    "LOCAL_RULE_RECORD",
    "LOCAL_RULE_SHA256",
    "NEUTRAL_SPAN_COUNT",
    "OVERLAP_CONTROL_COUNT",
    "PERSISTENT_LATCH_PRIMITIVE_RECORD",
    "PERSISTENT_LATCH_PRIMITIVE_SHA256",
    "PROTECTED_INPUT_COUNT",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PROTOCOL_STATUS",
    "PUBLIC_INPUT_COUNT",
    "RANDOM_LABEL_SEED",
    "RAW_TO_CURRENT_SCORE_RULE",
    "SCORE_MAX",
    "SCORE_MIN",
    "SELECTOR_ADDRESSES",
    "SELECTOR_ADDRESS_COUNT",
    "SELECTOR_CIRCUIT_RECORD",
    "SELECTOR_CIRCUIT_SHA256",
    "SELECTOR_ELECTRICAL_CONTROL_COUNT",
    "SELECTOR_FULL_PREFLIGHT_RECORD",
    "SELECTOR_FULL_PREFLIGHT_SHA256",
    "SELECTOR_TOPOLOGY_RECORD",
    "SELECTOR_TOPOLOGY_SHA256",
    "STOP_SCORE",
    "TAUGHT_FACTORS",
    "TEACHER_SCOUT_CIRCUIT_RECORD",
    "TEACHER_SCOUT_CIRCUIT_SHA256",
    "TEMPLATE_SPECS",
    "TICK_CAP",
    "TICK_DTYPE",
    "TRIADS",
    "V4_WTA_REFERENCE_RECORD",
    "V4_WTA_REFERENCE_SHA256",
    "AtomicContrastGroup",
    "AtomicLesson",
    "CompositionCase",
    "CompositorControl",
    "FrozenTeacherGatedFactorProtocol",
    "LexicalAssembly",
    "NeutralSpan",
    "ProtocolFactorEvent",
    "SelectorAddress",
    "SelectorElectricalControl",
    "SpanPlacement",
]
