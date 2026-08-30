"""Fresh data-only role-bound factor-membership workspace V4 protocol.

V3 remains terminal.  Its scorer discarded selector geometry and its exclusive
factor race could not represent an unordered co-target set.  V4 freezes a fresh
matched cohort in which only a physical content x selector-role conjunction can
identify the teacher target, followed by simultaneous factor-membership relays
and the unchanged capacity-four exact-set collector.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.r3_pilot import public_train_factor_workspace_v3_protocol as v3
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5

PROTOCOL_SCOPE = "PUBLIC_TRAIN_FACTOR_WORKSPACE_V4_PROTOCOL"
PROTOCOL_STATUS = "FROZEN_DATA_ONLY_NOT_RUN"

FIT = v3.FIT
HELDOUT = v3.HELDOUT
FACTOR_REGISTRY = v3.FACTOR_REGISTRY
ACTIVE_FACTORS = v3.ACTIVE_FACTORS
DORMANT_FACTORS = v3.DORMANT_FACTORS
FACTOR_SET_CLASSES = v3.FACTOR_SET_CLASSES
FACTOR_COUNT = len(FACTOR_REGISTRY)
FACTOR_SET_CLASS_COUNT = len(FACTOR_SET_CLASSES)
EVENT_CAPACITY = v3.EVENT_CAPACITY

LEXICAL_CELL_COUNT = v3.LEXICAL_CELL_COUNT
ASSEMBLY_ACTIVE_CELL_COUNT = v3.ASSEMBLY_ACTIVE_CELL_COUNT
SELECTOR_DIRECTION_CELL_COUNT = 2
SELECTOR_CUE_ROLE_CELL_COUNT = 4
SELECTOR_SELECTED_ROLE_CELL_COUNT = 4
SELECTOR_ROLE_CELL_COUNT = (
    SELECTOR_DIRECTION_CELL_COUNT + SELECTOR_CUE_ROLE_CELL_COUNT + SELECTOR_SELECTED_ROLE_CELL_COUNT
)
CONJUNCTION_CELL_COUNT = LEXICAL_CELL_COUNT * SELECTOR_ROLE_CELL_COUNT
REPRESENTATION_CELL_COUNT = LEXICAL_CELL_COUNT + CONJUNCTION_CELL_COUNT
ACTIVE_ROLE_CELL_COUNT = 3
ACTIVE_CONJUNCTION_CELL_COUNT = ASSEMBLY_ACTIVE_CELL_COUNT * ACTIVE_ROLE_CELL_COUNT
ACTIVE_REPRESENTATION_CELL_COUNT = ASSEMBLY_ACTIVE_CELL_COUNT + ACTIVE_CONJUNCTION_CELL_COUNT
MEMBERSHIP_SCORE_THRESHOLD = ASSEMBLY_ACTIVE_CELL_COUNT
TICK_CAP = v5.TICK_CAP
TICK_DTYPE = v5.TICK_DTYPE
FUTURE_STATE_SHAPE = (FACTOR_COUNT, REPRESENTATION_CELL_COUNT)
FUTURE_STATE_BYTES = FACTOR_COUNT * REPRESENTATION_CELL_COUNT * 2

CARRIER_PAIR_COUNT = FACTOR_SET_CLASS_COUNT // 2
TEMPLATE_COUNT = 6
LESSON_COUNT = FACTOR_SET_CLASS_COUNT * TEMPLATE_COUNT
FIT_LESSON_COUNT = FACTOR_SET_CLASS_COUNT * 4
HELDOUT_LESSON_COUNT = FACTOR_SET_CLASS_COUNT * 2
FIT_INPUTS_PER_CLASS = 4
HELDOUT_INPUTS_PER_CLASS = 2
FRESH_COHORT_SEED = 2_026_083_004
HELDOUT_MIN_SUPPORTED_REPRESENTATION_CELLS = ASSEMBLY_ACTIVE_CELL_COUNT * 2

CANONICAL_PROMPT_APPEARANCE_COUNT = v3.CANONICAL_PROMPT_APPEARANCE_COUNT
CANONICAL_SEMANTIC_IDENTITY_COUNT = v3.CANONICAL_SEMANTIC_IDENTITY_COUNT
CANONICAL_FIT_IDENTITY_COUNT = v3.CANONICAL_FIT_IDENTITY_COUNT
CANONICAL_HELDOUT_IDENTITY_COUNT = v3.CANONICAL_HELDOUT_IDENTITY_COUNT

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_COUNT = 0
PROGRAM_ARTIFACT_COUNT = 0
PROPERTY_ROW_COUNT = 0
FAMILY_IDENTIFIER_COUNT = 0
TASK_IDENTIFIER_COUNT = 0
LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
HELDOUT_TREATMENT_RESULT_COUNT = 0

V3_PROTOCOL_COMMIT = "398818f"
V3_PROTOCOL_SOURCE_SHA256 = "30f9393169caa57313250cc9323fe064ae2f9b5a1ce1970df41f5177da2410e7"
V3_PROTOCOL_SHA256 = "1722424f3cb540ed152d0035f916f85b9c7a8e9306242ecfe026b39b9685fdec"
V3_LEARNER_COMMIT = "640d3d7"
V3_LEARNER_SOURCE_SHA256 = "c1761eafa575cd9ce732da0f0582eca7bdd024bf8c58c3d12229d687543afe22"
V3_LEARNER_TEST_SHA256 = "3612dc1f73f39fe5fea217cde2b8be2065b7048efdc92bf682e005e4a85611ab"
V3_RESULT_SHA256 = "af73e68a8897611c7ae34003acbb58fe18f6d8184fa5eb9fe8f770b170d1612f"
V3_FIT_REPLAY_SHA256 = "9d581cb8f005390ea148db747e352d3cc81a08c950419db0f516b294dddbe1c1"

_DEPENDENCY_NAMESPACE = b"public-train-factor-workspace-v4-dependencies\0"
_LOCALIZATION_NAMESPACE = b"public-train-factor-workspace-v4-localization\0"
_CLASS_NAMESPACE = b"public-train-factor-workspace-v4-binding-class\0"
_TEMPLATE_NAMESPACE = b"public-train-factor-workspace-v4-template\0"
_LESSON_NAMESPACE = b"public-train-factor-workspace-v4-fresh-lesson\0"
_COHORT_NAMESPACE = b"public-train-factor-workspace-v4-fresh-cohort\0"
_IDENTIFIABILITY_NAMESPACE = b"public-train-factor-workspace-v4-identifiability\0"
_ROLE_TOPOLOGY_NAMESPACE = b"public-train-factor-workspace-v4-role-topology\0"
_MEMBERSHIP_TOPOLOGY_NAMESPACE = b"public-train-factor-workspace-v4-membership-topology\0"
_LOCAL_RULE_NAMESPACE = b"public-train-factor-workspace-v4-local-rule\0"
_CONTROL_NAMESPACE = b"public-train-factor-workspace-v4-control\0"
_SURFACE_NAMESPACE = b"public-train-factor-workspace-v4-surface\0"
_PROTOCOL_NAMESPACE = b"public-train-factor-workspace-v4-protocol\0"


def _plain_record(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return {str(key): _plain_record(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _plain_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain_record(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain_record(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _freeze_record(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_record(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_record(item) for item in value)
    return value


def _factor_set_record(factors: tuple[str, ...]) -> dict[str, object]:
    if (
        type(factors) is not tuple
        or not factors
        or len(factors) > EVENT_CAPACITY
        or len(factors) != len(set(factors))
        or factors != tuple(factor for factor in FACTOR_REGISTRY if factor in frozenset(factors))
    ):
        raise ValueError("V4 teacher factor set is not exact and registry ordered")
    return {
        "factor_registry_sha256": v3.v2.manifest_v1.FACTOR_REGISTRY_SHA256,
        "factors": factors,
        "cardinality": len(factors),
        "set_semantics": True,
    }


def _v3_causal_source_sha256(causal_input: v3.CausalSelectorInputKey) -> str:
    return _namespaced_sha256(
        b"public-train-factor-workspace-v3-causal-input\0",
        causal_input.record(),
    )


@dataclass(frozen=True, slots=True)
class V3FitCollisionLocalization:
    status: str
    fit_attempt_count: int
    fit_exact_count: int
    fit_collision_count: int
    heldout_treatment_attempt_count: int
    multilabel_equal_score_collision_count: int
    once_seen_stop_tie_count: int
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int
    excitatory_max_ticks: int
    inhibitory_max_ticks: int
    scorer_feature_cells: int
    causal_key_geometry_discarded_by_scorer: bool
    add_seed_overlap_counts: tuple[tuple[int, ...], ...]
    mismatch_causal_source_sha256s: tuple[str, ...]
    repair: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "fit_attempt_count": self.fit_attempt_count,
            "fit_exact_count": self.fit_exact_count,
            "fit_collision_count": self.fit_collision_count,
            "heldout_treatment_attempt_count": self.heldout_treatment_attempt_count,
            "multilabel_equal_score_collision_count": (self.multilabel_equal_score_collision_count),
            "once_seen_stop_tie_count": self.once_seen_stop_tie_count,
            "committed_excitatory_ticks": self.committed_excitatory_ticks,
            "committed_inhibitory_ticks": self.committed_inhibitory_ticks,
            "excitatory_max_ticks": self.excitatory_max_ticks,
            "inhibitory_max_ticks": self.inhibitory_max_ticks,
            "scorer_feature_cells": self.scorer_feature_cells,
            "causal_key_geometry_discarded_by_scorer": (
                self.causal_key_geometry_discarded_by_scorer
            ),
            "add_seed_overlap_counts": self.add_seed_overlap_counts,
            "mismatch_causal_source_sha256s": self.mismatch_causal_source_sha256s,
            "repair": self.repair,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RoleBindingClass:
    class_index: int
    pair_index: int
    direction: str
    carrier_assembly_cells: tuple[int, ...]
    target_factors: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_role": {
                "direction": self.direction,
                "carrier_assembly_cells": self.carrier_assembly_cells,
            },
            "teacher_target": _factor_set_record(self.target_factors),
            "audit_only": {
                "class_index": self.class_index,
                "pair_index": self.pair_index,
            },
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RoleBindingTemplate:
    template_index: int
    split: str
    cue_slot: int
    neutral_variant: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "template_index": self.template_index,
            "split": self.split,
            "cue_slot": self.cue_slot,
            "neutral_variant": self.neutral_variant,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RoleBoundCausalInputKey:
    direction: str
    cue_slot: int
    selected_slot: int
    slot_assembly_cells: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    selector_role_cells: tuple[int, int, int]
    representation_spike_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.direction not in ("NEXT", "PREV")
            or not 0 <= self.cue_slot < 4
            or self.selected_slot
            != ((self.cue_slot + 1) % 4 if self.direction == "NEXT" else (self.cue_slot - 1) % 4)
            or len(self.slot_assembly_cells) != 4
            or self.slot_assembly_cells[self.cue_slot] != v3.CUE_ASSEMBLY_CELLS[self.direction]
        ):
            raise ValueError("V4 selector geometry changed")
        carrier = self.slot_assembly_cells[self.selected_slot]
        free_slots = tuple(
            slot for slot in range(4) if slot not in (self.cue_slot, self.selected_slot)
        )
        free_assemblies = tuple(self.slot_assembly_cells[slot] for slot in free_slots)
        if (
            len(carrier) != ASSEMBLY_ACTIVE_CELL_COUNT
            or len(set(carrier)) != ASSEMBLY_ACTIVE_CELL_COUNT
            or tuple(sorted(carrier)) != carrier
            or free_assemblies
            not in (
                (v3.NEUTRAL_SLOT_ASSEMBLIES[0],) * 2,
                (v3.NEUTRAL_SLOT_ASSEMBLIES[1],) * 2,
            )
        ):
            raise ValueError("V4 carrier or fresh neutral context changed")
        expected_roles = _selector_role_cells(
            self.direction,
            self.cue_slot,
            self.selected_slot,
        )
        expected_representation = _representation_spike_cells(carrier, expected_roles)
        if (
            self.selector_role_cells != expected_roles
            or self.representation_spike_cells != expected_representation
        ):
            raise ValueError("V4 role-bound representation changed")

    @property
    def carrier_assembly_cells(self) -> tuple[int, ...]:
        return self.slot_assembly_cells[self.selected_slot]

    def full_input_record(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
            "slot_assembly_cells": self.slot_assembly_cells,
        }

    def representation_record(self) -> dict[str, object]:
        return {
            "selector_role_cells": self.selector_role_cells,
            "representation_spike_cells": self.representation_spike_cells,
        }

    def record(self) -> dict[str, object]:
        return {
            "physical_selector_input": self.full_input_record(),
            "physical_role_bound_projection": self.representation_record(),
        }


@dataclass(frozen=True, slots=True)
class RoleBoundLesson:
    lesson_index: int
    class_index: int
    template_index: int
    split: str
    causal_input: RoleBoundCausalInputKey
    target_factors: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_input": self.causal_input.record(),
            "teacher_target": _factor_set_record(self.target_factors),
            "audit_only": {
                "lesson_index": self.lesson_index,
                "class_index": self.class_index,
                "template_index": self.template_index,
                "split": self.split,
            },
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RoleBindingIdentifiabilityProof:
    lesson_count: int
    unique_full_input_count: int
    unique_role_bound_projection_count: int
    full_keys_with_multiple_target_sets: int
    role_bound_keys_with_multiple_target_sets: int
    content_only_keys_with_multiple_target_sets: int
    direction_only_keys_with_multiple_target_sets: int
    fresh_v3_full_input_overlap_count: int
    fresh_v3_lesson_sha_overlap_count: int
    min_heldout_supported_representation_cells: int
    total_function: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "lesson_count": self.lesson_count,
            "unique_full_input_count": self.unique_full_input_count,
            "unique_role_bound_projection_count": self.unique_role_bound_projection_count,
            "full_keys_with_multiple_target_sets": self.full_keys_with_multiple_target_sets,
            "role_bound_keys_with_multiple_target_sets": (
                self.role_bound_keys_with_multiple_target_sets
            ),
            "content_only_keys_with_multiple_target_sets": (
                self.content_only_keys_with_multiple_target_sets
            ),
            "direction_only_keys_with_multiple_target_sets": (
                self.direction_only_keys_with_multiple_target_sets
            ),
            "fresh_v3_full_input_overlap_count": self.fresh_v3_full_input_overlap_count,
            "fresh_v3_lesson_sha_overlap_count": self.fresh_v3_lesson_sha_overlap_count,
            "min_heldout_supported_representation_cells": (
                self.min_heldout_supported_representation_cells
            ),
            "total_function": self.total_function,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FutureCausalControl:
    control_index: int
    name: str
    intervention: str
    expected_status: str
    expected_factor_membership_count: int
    expected_exact: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "control_index": self.control_index,
            "name": self.name,
            "intervention": self.intervention,
            "expected_status": self.expected_status,
            "expected_factor_membership_count": self.expected_factor_membership_count,
            "expected_exact": self.expected_exact,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainFactorWorkspaceV4Protocol:
    scope: str
    status: str
    canonical_identities: tuple[object, ...]
    canonical_fit_prompt_input_sha256s: tuple[str, ...]
    canonical_heldout_prompt_input_sha256s: tuple[str, ...]
    v3_localization: V3FitCollisionLocalization
    binding_classes: tuple[RoleBindingClass, ...]
    templates: tuple[RoleBindingTemplate, ...]
    lessons: tuple[RoleBoundLesson, ...]
    identifiability_proof: RoleBindingIdentifiabilityProof
    future_controls: tuple[FutureCausalControl, ...]
    dependency_sha256: str
    localization_sha256: str
    cohort_sha256: str
    identifiability_sha256: str
    role_binding_topology_sha256: str
    membership_topology_sha256: str
    local_rule_sha256: str
    controls_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str

    def root_record(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "localization_sha256": self.localization_sha256,
            "cohort_sha256": self.cohort_sha256,
            "identifiability_sha256": self.identifiability_sha256,
            "role_binding_topology_sha256": self.role_binding_topology_sha256,
            "membership_topology_sha256": self.membership_topology_sha256,
            "local_rule_sha256": self.local_rule_sha256,
            "controls_sha256": self.controls_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "canonical_prompt_appearance_count": CANONICAL_PROMPT_APPEARANCE_COUNT,
            "canonical_semantic_identity_count": len(self.canonical_identities),
            "canonical_fit_identity_count": len(self.canonical_fit_prompt_input_sha256s),
            "canonical_heldout_identity_count": len(self.canonical_heldout_prompt_input_sha256s),
            "factor_set_class_count": len(self.binding_classes),
            "carrier_pair_count": CARRIER_PAIR_COUNT,
            "lesson_count": len(self.lessons),
            "fit_lesson_count": sum(row.split == FIT for row in self.lessons),
            "heldout_lesson_count": sum(row.split == HELDOUT for row in self.lessons),
            "representation_cell_count": REPRESENTATION_CELL_COUNT,
            "event_capacity": EVENT_CAPACITY,
            "learner_array_count": LEARNER_ARRAY_COUNT,
            "learning_run_count": LEARNING_RUN_COUNT,
            "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
            "heldout_treatment_result_count": HELDOUT_TREATMENT_RESULT_COUNT,
            "protocol_sha256": self.protocol_sha256,
        }


def _authenticate_dependencies() -> None:
    protocol_path = Path(v3.__file__).resolve()
    learner_path = protocol_path.with_name("public_train_factor_workspace_v3.py")
    learner_test_path = (
        protocol_path.parents[2] / "tests" / "test_public_train_factor_workspace_v3.py"
    )
    if (
        sha256(protocol_path.read_bytes()).hexdigest() != V3_PROTOCOL_SOURCE_SHA256
        or v3.PROTOCOL.protocol_sha256 != V3_PROTOCOL_SHA256
        or v3._build_protocol() != v3.PROTOCOL
        or sha256(learner_path.read_bytes()).hexdigest() != V3_LEARNER_SOURCE_SHA256
        or sha256(learner_test_path.read_bytes()).hexdigest() != V3_LEARNER_TEST_SHA256
        or any(
            (
                v3.DEV_INPUT_COUNT,
                v3.PROTECTED_INPUT_COUNT,
                v3.DESIRED_OUTPUT_COUNT,
                v3.PROGRAM_ARTIFACT_COUNT,
                v3.PROPERTY_ROW_COUNT,
                v3.FAMILY_IDENTIFIER_COUNT,
                v3.TASK_IDENTIFIER_COUNT,
                v3.LEARNER_ARRAY_COUNT,
                v3.LEARNING_RUN_COUNT,
                v3.ELECTRICAL_EXECUTION_COUNT,
            )
        )
    ):
        raise RuntimeError("PUBLIC_TRAIN factor workspace V3 dependency changed")


DEPENDENCY_RECORD = _freeze_record(
    {
        "v3_protocol_commit": V3_PROTOCOL_COMMIT,
        "v3_protocol_source_sha256": V3_PROTOCOL_SOURCE_SHA256,
        "v3_protocol_sha256": V3_PROTOCOL_SHA256,
        "v3_learner_commit": V3_LEARNER_COMMIT,
        "v3_learner_source_sha256": V3_LEARNER_SOURCE_SHA256,
        "v3_learner_test_sha256": V3_LEARNER_TEST_SHA256,
        "v3_result_sha256": V3_RESULT_SHA256,
        "v3_fit_replay_sha256": V3_FIT_REPLAY_SHA256,
        "v5_protocol_sha256": v5.PROTOCOL.protocol_sha256,
        "v5_local_rule_sha256": v5.LOCAL_RULE_SHA256,
        "factor_registry_sha256": v3.v2.manifest_v1.FACTOR_REGISTRY_SHA256,
        "canonical_inventory_sha256": v3.v2.PROTOCOL.canonical_inventory_sha256,
        "canonical_split_sha256": v3.v2.PROTOCOL.canonical_split_sha256,
        "capacity_four_compositor_topology_sha256": v3.v2.COMPOSITOR_TOPOLOGY_SHA256,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(_DEPENDENCY_NAMESPACE, dict(DEPENDENCY_RECORD))


_V3_MISMATCH_CAUSAL_SOURCES = (
    "6ec8784c8728ad194e45b147ea5162579706d64d85296ed70c328b8405eb6fef",
    "d09efe147fa0945e471ef0aadb2fb270036c3b6372fd386c08ffbb628699684b",
    "3826ab0c0e08946b1eab60724a547810ef99ad0b8f518b1d291e8022604af669",
    "bf8b9bc2ce83e285151da27ad635c08f018e4090fa4f85c8208c40e64ed4790e",
    "735cadb43deb17b6b3e0ba435f0e7fee2b7790267df46be2d1c802f3392f2097",
)


def _build_v3_localization() -> V3FitCollisionLocalization:
    mismatch_rows = tuple(
        lesson
        for lesson in v3.PROTOCOL.lessons
        if _v3_causal_source_sha256(lesson.causal_input) in frozenset(_V3_MISMATCH_CAUSAL_SOURCES)
    )
    add_class = next(
        row for row in v3.PROTOCOL.factor_set_classes if row.target_factors == ("add",)
    )
    overlaps = tuple(
        tuple(len(set(left) & set(right)) for right in add_class.physical_seed_assemblies)
        for left in add_class.physical_seed_assemblies
    )
    if (
        len(mismatch_rows) != 5
        or any(row.split != FIT for row in mismatch_rows)
        or Counter(row.target_factors for row in mismatch_rows)
        != Counter({("minimum", "maximum"): 4, ("add",): 1})
        or overlaps != ((12, 0, 1), (0, 12, 0), (1, 0, 12))
    ):
        raise RuntimeError("V3 FIT-only collision localization changed")
    record = {
        "status": "STOP_PUBLIC_TRAIN_FACTOR_WORKSPACE_V3_FIT_READINESS",
        "fit_attempt_count": 48,
        "fit_exact_count": 43,
        "fit_collision_count": 5,
        "heldout_treatment_attempt_count": 0,
        "multilabel_equal_score_collision_count": 4,
        "once_seen_stop_tie_count": 1,
        "committed_excitatory_ticks": 396,
        "committed_inhibitory_ticks": 0,
        "excitatory_max_ticks": 2,
        "inhibitory_max_ticks": 0,
        "scorer_feature_cells": LEXICAL_CELL_COUNT,
        "causal_key_geometry_discarded_by_scorer": True,
        "add_seed_overlap_counts": overlaps,
        "mismatch_causal_source_sha256s": _V3_MISMATCH_CAUSAL_SOURCES,
        "repair": (
            "bind selected content to distributed selector roles and replace exclusive "
            "factor ordering with simultaneous locally learned factor membership"
        ),
    }
    return V3FitCollisionLocalization(
        **record,
        sha256=_namespaced_sha256(_LOCALIZATION_NAMESPACE, record),
    )


def _selector_role_cells(direction: str, cue_slot: int, selected_slot: int) -> tuple[int, int, int]:
    direction_cell = 0 if direction == "NEXT" else 1
    return (
        direction_cell,
        SELECTOR_DIRECTION_CELL_COUNT + cue_slot,
        SELECTOR_DIRECTION_CELL_COUNT + SELECTOR_CUE_ROLE_CELL_COUNT + selected_slot,
    )


def _representation_spike_cells(
    carrier_cells: tuple[int, ...],
    role_cells: tuple[int, int, int],
) -> tuple[int, ...]:
    conjunctions = tuple(
        LEXICAL_CELL_COUNT + role_cell * LEXICAL_CELL_COUNT + lexical_cell
        for role_cell in role_cells
        for lexical_cell in carrier_cells
    )
    return tuple(sorted((*carrier_cells, *conjunctions)))


def _carrier_assemblies() -> tuple[tuple[int, ...], ...]:
    candidates = tuple(
        sorted(
            {
                assembly
                for factor_class in v3.PROTOCOL.factor_set_classes
                for assembly in factor_class.physical_seed_assemblies
            }
        )
    )
    rotation = FRESH_COHORT_SEED % len(candidates)
    rotated = candidates[rotation:] + candidates[:rotation]
    carriers = rotated[:CARRIER_PAIR_COUNT]
    forbidden = frozenset((*v3.CUE_ASSEMBLY_CELLS.values(), *v3.NEUTRAL_SLOT_ASSEMBLIES))
    if len(carriers) != CARRIER_PAIR_COUNT or any(carrier in forbidden for carrier in carriers):
        raise RuntimeError("V4 fresh carrier selection changed")
    return carriers


def _build_binding_classes() -> tuple[RoleBindingClass, ...]:
    carriers = _carrier_assemblies()
    rows: list[RoleBindingClass] = []
    for class_index, target_factors in enumerate(FACTOR_SET_CLASSES):
        pair_index = class_index // 2
        direction = "NEXT" if class_index % 2 == 0 else "PREV"
        record = {
            "causal_role": {
                "direction": direction,
                "carrier_assembly_cells": carriers[pair_index],
            },
            "teacher_target": _factor_set_record(target_factors),
            "audit_only": {"class_index": class_index, "pair_index": pair_index},
        }
        rows.append(
            RoleBindingClass(
                class_index=class_index,
                pair_index=pair_index,
                direction=direction,
                carrier_assembly_cells=carriers[pair_index],
                target_factors=target_factors,
                sha256=_namespaced_sha256(_CLASS_NAMESPACE, record),
            )
        )
    return tuple(rows)


_TEMPLATE_SPECS = (
    (FIT, 0, 0),
    (FIT, 0, 1),
    (FIT, 1, 0),
    (FIT, 1, 1),
    (HELDOUT, 2, 0),
    (HELDOUT, 3, 1),
)


def _build_templates() -> tuple[RoleBindingTemplate, ...]:
    rows: list[RoleBindingTemplate] = []
    for template_index, (split, cue_slot, neutral_variant) in enumerate(_TEMPLATE_SPECS):
        record = {
            "template_index": template_index,
            "split": split,
            "cue_slot": cue_slot,
            "neutral_variant": neutral_variant,
        }
        rows.append(
            RoleBindingTemplate(
                template_index=template_index,
                split=split,
                cue_slot=cue_slot,
                neutral_variant=neutral_variant,
                sha256=_namespaced_sha256(_TEMPLATE_NAMESPACE, record),
            )
        )
    return tuple(rows)


def _causal_input(
    binding_class: RoleBindingClass,
    template: RoleBindingTemplate,
) -> RoleBoundCausalInputKey:
    selected_slot = (
        (template.cue_slot + 1) % 4
        if binding_class.direction == "NEXT"
        else (template.cue_slot - 1) % 4
    )
    slots: list[tuple[int, ...] | None] = [None, None, None, None]
    slots[template.cue_slot] = v3.CUE_ASSEMBLY_CELLS[binding_class.direction]
    slots[selected_slot] = binding_class.carrier_assembly_cells
    neutral = v3.NEUTRAL_SLOT_ASSEMBLIES[template.neutral_variant]
    for slot in range(4):
        if slots[slot] is None:
            slots[slot] = neutral
    role_cells = _selector_role_cells(
        binding_class.direction,
        template.cue_slot,
        selected_slot,
    )
    return RoleBoundCausalInputKey(
        direction=binding_class.direction,
        cue_slot=template.cue_slot,
        selected_slot=selected_slot,
        slot_assembly_cells=(
            tuple(slots[0] or ()),
            tuple(slots[1] or ()),
            tuple(slots[2] or ()),
            tuple(slots[3] or ()),
        ),
        selector_role_cells=role_cells,
        representation_spike_cells=_representation_spike_cells(
            binding_class.carrier_assembly_cells,
            role_cells,
        ),
    )


def _build_lessons(
    classes: tuple[RoleBindingClass, ...],
    templates: tuple[RoleBindingTemplate, ...],
) -> tuple[RoleBoundLesson, ...]:
    rows: list[RoleBoundLesson] = []
    for binding_class in classes:
        for template in templates:
            causal_input = _causal_input(binding_class, template)
            lesson_index = len(rows)
            record = {
                "causal_input": causal_input.record(),
                "teacher_target": _factor_set_record(binding_class.target_factors),
                "audit_only": {
                    "lesson_index": lesson_index,
                    "class_index": binding_class.class_index,
                    "template_index": template.template_index,
                    "split": template.split,
                },
            }
            rows.append(
                RoleBoundLesson(
                    lesson_index=lesson_index,
                    class_index=binding_class.class_index,
                    template_index=template.template_index,
                    split=template.split,
                    causal_input=causal_input,
                    target_factors=binding_class.target_factors,
                    sha256=_namespaced_sha256(_LESSON_NAMESPACE, record),
                )
            )
    return tuple(rows)


def _conflict_count(mapping: dict[object, set[tuple[str, ...]]]) -> int:
    return sum(len(targets) > 1 for targets in mapping.values())


def _build_identifiability_proof(
    lessons: tuple[RoleBoundLesson, ...],
) -> RoleBindingIdentifiabilityProof:
    full: dict[object, set[tuple[str, ...]]] = defaultdict(set)
    role_bound: dict[object, set[tuple[str, ...]]] = defaultdict(set)
    content_only: dict[object, set[tuple[str, ...]]] = defaultdict(set)
    direction_only: dict[object, set[tuple[str, ...]]] = defaultdict(set)
    for lesson in lessons:
        full[_canonical_json(lesson.causal_input.full_input_record())].add(lesson.target_factors)
        role_bound[_canonical_json(lesson.causal_input.representation_record())].add(
            lesson.target_factors
        )
        content_only[lesson.causal_input.carrier_assembly_cells].add(lesson.target_factors)
        direction_only[lesson.causal_input.direction].add(lesson.target_factors)

    v3_full_inputs = {
        _canonical_json(lesson.causal_input.record()) for lesson in v3.PROTOCOL.lessons
    }
    full_overlap = sum(
        _canonical_json(lesson.causal_input.full_input_record()) in v3_full_inputs
        for lesson in lessons
    )
    v3_lesson_shas = {lesson.sha256 for lesson in v3.PROTOCOL.lessons}
    lesson_sha_overlap = len({lesson.sha256 for lesson in lessons} & v3_lesson_shas)

    fit_by_class = defaultdict(list)
    for lesson in lessons:
        if lesson.split == FIT:
            fit_by_class[lesson.class_index].append(lesson)
    heldout_support: list[int] = []
    for lesson in lessons:
        if lesson.split != HELDOUT:
            continue
        supported = set().union(
            *(
                set(row.causal_input.representation_spike_cells)
                for row in fit_by_class[lesson.class_index]
            )
        )
        heldout_support.append(len(set(lesson.causal_input.representation_spike_cells) & supported))

    record = {
        "lesson_count": len(lessons),
        "unique_full_input_count": len(full),
        "unique_role_bound_projection_count": len(role_bound),
        "full_keys_with_multiple_target_sets": _conflict_count(full),
        "role_bound_keys_with_multiple_target_sets": _conflict_count(role_bound),
        "content_only_keys_with_multiple_target_sets": _conflict_count(content_only),
        "direction_only_keys_with_multiple_target_sets": _conflict_count(direction_only),
        "fresh_v3_full_input_overlap_count": full_overlap,
        "fresh_v3_lesson_sha_overlap_count": lesson_sha_overlap,
        "min_heldout_supported_representation_cells": min(heldout_support),
        "total_function": _conflict_count(full) == 0 and _conflict_count(role_bound) == 0,
    }
    if (
        record["lesson_count"] != LESSON_COUNT
        or record["unique_full_input_count"] != LESSON_COUNT
        or record["unique_role_bound_projection_count"] != 48
        or record["full_keys_with_multiple_target_sets"] != 0
        or record["role_bound_keys_with_multiple_target_sets"] != 0
        or record["content_only_keys_with_multiple_target_sets"] != CARRIER_PAIR_COUNT
        or record["direction_only_keys_with_multiple_target_sets"] != 2
        or full_overlap != 0
        or lesson_sha_overlap != 0
        or min(heldout_support) < HELDOUT_MIN_SUPPORTED_REPRESENTATION_CELLS
    ):
        raise RuntimeError("V4 role-binding identifiability proof changed")
    return RoleBindingIdentifiabilityProof(
        **record,
        sha256=_namespaced_sha256(_IDENTIFIABILITY_NAMESPACE, record),
    )


ROLE_BINDING_TOPOLOGY_RECORD = _freeze_record(
    {
        "lexical_content_cells": LEXICAL_CELL_COUNT,
        "selector_role_cells": SELECTOR_ROLE_CELL_COUNT,
        "selector_role_decomposition": {
            "direction_cells": SELECTOR_DIRECTION_CELL_COUNT,
            "cue_slot_cells": SELECTOR_CUE_ROLE_CELL_COUNT,
            "selected_slot_cells": SELECTOR_SELECTED_ROLE_CELL_COUNT,
        },
        "active_role_cells_per_input": ACTIVE_ROLE_CELL_COUNT,
        "content_role_conjunction_cells": CONJUNCTION_CELL_COUNT,
        "conjunction_rule": (
            "one lexical relay plus one selector-role relay excite one physical AND cell"
        ),
        "representation_cells": REPRESENTATION_CELL_COUNT,
        "active_content_cells": ASSEMBLY_ACTIVE_CELL_COUNT,
        "active_conjunction_cells": ACTIVE_CONJUNCTION_CELL_COUNT,
        "active_representation_cells": ACTIVE_REPRESENTATION_CELL_COUNT,
        "projection_uses_factor_label": False,
        "projection_uses_hash_or_audit_identity": False,
        "projection_uses_lesson_or_class_ordinal": False,
        "domain_specific_primitive_delta": 0,
    }
)
ROLE_BINDING_TOPOLOGY_SHA256 = _namespaced_sha256(
    _ROLE_TOPOLOGY_NAMESPACE,
    dict(ROLE_BINDING_TOPOLOGY_RECORD),
)

MEMBERSHIP_TOPOLOGY_RECORD = _freeze_record(
    {
        "factor_registry_sha256": v3.v2.manifest_v1.FACTOR_REGISTRY_SHA256,
        "factor_membership_principal_cells": FACTOR_COUNT,
        "membership_score_threshold": MEMBERSHIP_SCORE_THRESHOLD,
        "membership_relays_are_independent": True,
        "simultaneous_distinct_factor_relays_are_legal": True,
        "factor_factor_collision_poison": False,
        "tonic_stop_rule": "STOP fires only when the any-factor detector is silent",
        "factor_stop_equal_score_collision": False,
        "set_collection": "simultaneous factor relays set physical occupancy cells",
        "event_capacity": EVENT_CAPACITY,
        "same_factor_duplicate": "absorbing duplicate poison",
        "fifth_distinct_factor": "absorbing capacity poison",
        "candidate_admission": "exact physical occupancy and cardinality equality",
        "factor_set_permutation_invariant": True,
        "host_union_or_filter": False,
        "ordered_factor_role_present": False,
    }
)
MEMBERSHIP_TOPOLOGY_SHA256 = _namespaced_sha256(
    _MEMBERSHIP_TOPOLOGY_NAMESPACE,
    dict(MEMBERSHIP_TOPOLOGY_RECORD),
)

LOCAL_RULE_RECORD = _freeze_record(
    {
        "fit_pass_count": 1,
        "fit_lesson_count": FIT_LESSON_COUNT,
        "teacher_target": "one simultaneous exact factor-set clamp",
        "excitatory_update": (
            "teacher factor x active role-bound representation coincidence commits locally"
        ),
        "inhibitory_update": (
            "nonteacher membership scout x active representation coincidence commits locally"
        ),
        "co_target_inhibition": "every clamped co-target physically inhibits its H update relay",
        "teacher_off": "zero committed ticks",
        "cell_off": "zero committed ticks and zero memberships",
        "shuffled_teacher": "fresh fixed derangement must have zero true exact sets",
        "tick_dtype": TICK_DTYPE,
        "tick_cap": TICK_CAP,
        "state_shape": FUTURE_STATE_SHAPE,
        "state_bytes": FUTURE_STATE_BYTES,
        "event_capacity_unchanged": EVENT_CAPACITY,
        "retune_after_result": False,
    }
)
LOCAL_RULE_SHA256 = _namespaced_sha256(_LOCAL_RULE_NAMESPACE, dict(LOCAL_RULE_RECORD))


_CONTROL_SPECS = (
    ("zero-membership-stop", "all scores zero", "STOP_ZERO", 0, False),
    ("below-support-stop", "one factor score 11", "STOP_ZERO", 0, False),
    ("once-exposed-membership", "one factor score 12", "READY_FACTOR_SET", 1, True),
    ("two-membership-unordered-set", "two factors score 12", "READY_FACTOR_SET", 2, True),
    ("exact-four-membership", "four factors score 12", "READY_FACTOR_SET", 4, True),
    ("fifth-membership-poison", "five factors score 12", "STOP_EVENT_CAPACITY_4", 5, False),
    (
        "duplicate-relay-poison",
        "one factor relay injected twice",
        "STOP_DUPLICATE_FACTOR",
        1,
        False,
    ),
    ("role-binding-off", "suppress every conjunction relay", "STOP_CONTENT_ONLY_ALIAS", 0, False),
    (
        "direction-role-off",
        "suppress both direction-role relays",
        "STOP_ROLE_CAUSAL_ALIAS",
        0,
        False,
    ),
    ("teacher-off", "teacher clamp gain zero", "STOP_ZERO", 0, False),
    ("cell-off", "content and role relay gain zero", "STOP_ZERO", 0, False),
    ("shuffled-teacher", "fixed 13-factor derangement", "STOP_SHUFFLED_LABEL", 0, False),
    ("factor-set-permutation", "reverse candidate factor order", "READY_FACTOR_SET", 2, True),
)


def _build_controls() -> tuple[FutureCausalControl, ...]:
    rows: list[FutureCausalControl] = []
    for index, (name, intervention, status, count, exact) in enumerate(_CONTROL_SPECS):
        record = {
            "control_index": index,
            "name": name,
            "intervention": intervention,
            "expected_status": status,
            "expected_factor_membership_count": count,
            "expected_exact": exact,
        }
        rows.append(
            FutureCausalControl(
                control_index=index,
                name=name,
                intervention=intervention,
                expected_status=status,
                expected_factor_membership_count=count,
                expected_exact=exact,
                sha256=_namespaced_sha256(_CONTROL_NAMESPACE, record),
            )
        )
    return tuple(rows)


SURFACE_BOUNDARY_RECORD = _freeze_record(
    {
        "canonical_prompt_appearance_count": CANONICAL_PROMPT_APPEARANCE_COUNT,
        "canonical_semantic_identity_count": CANONICAL_SEMANTIC_IDENTITY_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_count": DESIRED_OUTPUT_COUNT,
        "program_artifact_count": PROGRAM_ARTIFACT_COUNT,
        "property_row_count": PROPERTY_ROW_COUNT,
        "family_identifier_count": FAMILY_IDENTIFIER_COUNT,
        "task_identifier_count": TASK_IDENTIFIER_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "heldout_treatment_result_count": HELDOUT_TREATMENT_RESULT_COUNT,
        "raw_prompt_text_count": 0,
        "data_only": True,
        "learner_or_inference_interface_present": False,
        "execution_surface_present": False,
    }
)
SURFACE_BOUNDARY_SHA256 = _namespaced_sha256(
    _SURFACE_NAMESPACE,
    dict(SURFACE_BOUNDARY_RECORD),
)


def _build_protocol() -> FrozenPublicTrainFactorWorkspaceV4Protocol:
    _authenticate_dependencies()
    localization = _build_v3_localization()
    classes = _build_binding_classes()
    templates = _build_templates()
    lessons = _build_lessons(classes, templates)
    proof = _build_identifiability_proof(lessons)
    controls = _build_controls()

    balance = Counter((lesson.target_factors, lesson.split) for lesson in lessons)
    if (
        len(classes) != FACTOR_SET_CLASS_COUNT
        or len(lessons) != LESSON_COUNT
        or any(
            balance[(factor_set, FIT)] != FIT_INPUTS_PER_CLASS
            or balance[(factor_set, HELDOUT)] != HELDOUT_INPUTS_PER_CLASS
            for factor_set in FACTOR_SET_CLASSES
        )
    ):
        raise RuntimeError("V4 fresh cohort balance changed")
    cohort_record = {
        "fresh_cohort_seed": FRESH_COHORT_SEED,
        "binding_classes": tuple(row.record() for row in classes),
        "templates": tuple(row.record() for row in templates),
        "lessons": tuple(row.record() for row in lessons),
        "balance": {
            "factor_set_class_count": FACTOR_SET_CLASS_COUNT,
            "carrier_pair_count": CARRIER_PAIR_COUNT,
            "fit_inputs_per_class": FIT_INPUTS_PER_CLASS,
            "heldout_inputs_per_class": HELDOUT_INPUTS_PER_CLASS,
            "fit_lesson_count": FIT_LESSON_COUNT,
            "heldout_lesson_count": HELDOUT_LESSON_COUNT,
            "lesson_count": LESSON_COUNT,
        },
    }
    cohort_sha = _namespaced_sha256(_COHORT_NAMESPACE, cohort_record)
    controls_sha = _namespaced_sha256(
        _CONTROL_NAMESPACE + b"panel\0",
        tuple(row.record() for row in controls),
    )
    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": PROTOCOL_STATUS,
        "dependency_sha256": DEPENDENCY_SHA256,
        "localization_sha256": localization.sha256,
        "cohort_sha256": cohort_sha,
        "identifiability_sha256": proof.sha256,
        "role_binding_topology_sha256": ROLE_BINDING_TOPOLOGY_SHA256,
        "membership_topology_sha256": MEMBERSHIP_TOPOLOGY_SHA256,
        "local_rule_sha256": LOCAL_RULE_SHA256,
        "controls_sha256": controls_sha,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "canonical_prompt_appearance_count": CANONICAL_PROMPT_APPEARANCE_COUNT,
        "canonical_semantic_identity_count": CANONICAL_SEMANTIC_IDENTITY_COUNT,
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "factor_set_class_count": FACTOR_SET_CLASS_COUNT,
        "carrier_pair_count": CARRIER_PAIR_COUNT,
        "lesson_count": LESSON_COUNT,
        "fit_lesson_count": FIT_LESSON_COUNT,
        "heldout_lesson_count": HELDOUT_LESSON_COUNT,
        "representation_cell_count": REPRESENTATION_CELL_COUNT,
        "event_capacity": EVENT_CAPACITY,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "heldout_treatment_result_count": HELDOUT_TREATMENT_RESULT_COUNT,
    }
    return FrozenPublicTrainFactorWorkspaceV4Protocol(
        scope=PROTOCOL_SCOPE,
        status=PROTOCOL_STATUS,
        canonical_identities=v3.PROTOCOL.canonical_identities,
        canonical_fit_prompt_input_sha256s=v3.PROTOCOL.canonical_fit_prompt_input_sha256s,
        canonical_heldout_prompt_input_sha256s=v3.PROTOCOL.canonical_heldout_prompt_input_sha256s,
        v3_localization=localization,
        binding_classes=classes,
        templates=templates,
        lessons=lessons,
        identifiability_proof=proof,
        future_controls=controls,
        dependency_sha256=DEPENDENCY_SHA256,
        localization_sha256=localization.sha256,
        cohort_sha256=cohort_sha,
        identifiability_sha256=proof.sha256,
        role_binding_topology_sha256=ROLE_BINDING_TOPOLOGY_SHA256,
        membership_topology_sha256=MEMBERSHIP_TOPOLOGY_SHA256,
        local_rule_sha256=LOCAL_RULE_SHA256,
        controls_sha256=controls_sha,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        protocol_sha256=_namespaced_sha256(_PROTOCOL_NAMESPACE, root_without_protocol),
    )


PROTOCOL = _build_protocol()


__all__ = [
    "ACTIVE_CONJUNCTION_CELL_COUNT",
    "ACTIVE_FACTORS",
    "ACTIVE_REPRESENTATION_CELL_COUNT",
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "CANONICAL_FIT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_IDENTITY_COUNT",
    "CANONICAL_PROMPT_APPEARANCE_COUNT",
    "CANONICAL_SEMANTIC_IDENTITY_COUNT",
    "CARRIER_PAIR_COUNT",
    "CONJUNCTION_CELL_COUNT",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "DORMANT_FACTORS",
    "ELECTRICAL_EXECUTION_COUNT",
    "EVENT_CAPACITY",
    "FACTOR_REGISTRY",
    "FACTOR_SET_CLASSES",
    "FACTOR_SET_CLASS_COUNT",
    "FIT",
    "FIT_INPUTS_PER_CLASS",
    "FIT_LESSON_COUNT",
    "FRESH_COHORT_SEED",
    "FUTURE_STATE_BYTES",
    "FUTURE_STATE_SHAPE",
    "HELDOUT",
    "HELDOUT_INPUTS_PER_CLASS",
    "HELDOUT_LESSON_COUNT",
    "HELDOUT_MIN_SUPPORTED_REPRESENTATION_CELLS",
    "HELDOUT_TREATMENT_RESULT_COUNT",
    "LEARNER_ARRAY_COUNT",
    "LEARNING_RUN_COUNT",
    "LESSON_COUNT",
    "LOCAL_RULE_RECORD",
    "LOCAL_RULE_SHA256",
    "MEMBERSHIP_SCORE_THRESHOLD",
    "MEMBERSHIP_TOPOLOGY_RECORD",
    "MEMBERSHIP_TOPOLOGY_SHA256",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PROTOCOL_STATUS",
    "REPRESENTATION_CELL_COUNT",
    "ROLE_BINDING_TOPOLOGY_RECORD",
    "ROLE_BINDING_TOPOLOGY_SHA256",
    "SELECTOR_ROLE_CELL_COUNT",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "TEMPLATE_COUNT",
    "V3_LEARNER_COMMIT",
    "V3_LEARNER_SOURCE_SHA256",
    "V3_LEARNER_TEST_SHA256",
    "V3_PROTOCOL_COMMIT",
    "V3_PROTOCOL_SHA256",
    "V3_PROTOCOL_SOURCE_SHA256",
    "V3_RESULT_SHA256",
    "FrozenPublicTrainFactorWorkspaceV4Protocol",
    "FutureCausalControl",
    "RoleBindingClass",
    "RoleBindingIdentifiabilityProof",
    "RoleBindingTemplate",
    "RoleBoundCausalInputKey",
    "RoleBoundLesson",
    "V3FitCollisionLocalization",
]
