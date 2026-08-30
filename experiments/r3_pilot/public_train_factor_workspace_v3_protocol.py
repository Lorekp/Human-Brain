"""Fresh, data-only PUBLIC_TRAIN factor-set workspace V3 protocol.

V2 is an immutable dependency and remains stopped.  V3 removes the impossible
singleton-label ontology: the learner-visible key contains only selector
geometry and four physical assembly rows, while the teacher publishes one exact
factor set.  Identical causal keys must therefore denote one total target.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.r3_pilot import public_train_factor_workspace_v2_protocol as v2
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5

PROTOCOL_SCOPE = "PUBLIC_TRAIN_FACTOR_WORKSPACE_V3_PROTOCOL"
PROTOCOL_STATUS = "FROZEN_DATA_ONLY_NOT_RUN"

FIT = v2.FIT
HELDOUT = v2.HELDOUT
FACTOR_REGISTRY = v2.FACTOR_REGISTRY
ACTIVE_FACTORS = v2.ACTIVE_FACTORS
DORMANT_FACTORS = v2.DORMANT_FACTORS

FACTOR_SET_CLASSES = (
    ("minimum", "maximum"),
    ("add",),
    ("absolute",),
    ("less-than",),
    ("map",),
    ("sum",),
    ("count",),
    ("filter",),
    ("reverse",),
    ("class:ascii-decimal-digits",),
    ("class:ascii-lowercase",),
    ("class:english-vowels",),
)
FACTOR_SET_CLASS_COUNT = 12
TEMPLATE_COUNT = 6
LESSON_COUNT = FACTOR_SET_CLASS_COUNT * TEMPLATE_COUNT
FIT_LESSON_COUNT = FACTOR_SET_CLASS_COUNT * 4
HELDOUT_LESSON_COUNT = FACTOR_SET_CLASS_COUNT * 2
FIT_INPUTS_PER_CLASS = 4
HELDOUT_INPUTS_PER_CLASS = 2

CANONICAL_PROMPT_APPEARANCE_COUNT = v2.PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT
CANONICAL_SEMANTIC_IDENTITY_COUNT = v2.CANONICAL_SEMANTIC_IDENTITY_COUNT
CANONICAL_FIT_IDENTITY_COUNT = v2.CANONICAL_FIT_IDENTITY_COUNT
CANONICAL_HELDOUT_IDENTITY_COUNT = v2.CANONICAL_HELDOUT_IDENTITY_COUNT

LEXICAL_CELL_COUNT = v2.LEXICAL_CELL_COUNT
ASSEMBLY_ACTIVE_CELL_COUNT = v2.ASSEMBLY_ACTIVE_CELL_COUNT
EVENT_CAPACITY = v2.EVENT_CAPACITY

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

V2_PROTOCOL_COMMIT = "e04953f"
V2_PROTOCOL_SOURCE_SHA256 = "1931f1748d0943a35be7ba348a5460ef010656cc1865fd6a92434e206b3cc4cd"
V2_PROTOCOL_SHA256 = "54aca3e9f9f7ae0bf1b1743510938522400305c64530db9b3f1467b241ee3171"

_DEPENDENCY_NAMESPACE = b"public-train-factor-workspace-v3-dependencies\0"
_FACTOR_SET_NAMESPACE = b"public-train-factor-workspace-v3-factor-set\0"
_CLASS_NAMESPACE = b"public-train-factor-workspace-v3-class\0"
_TEMPLATE_NAMESPACE = b"public-train-factor-workspace-v3-template\0"
_LESSON_NAMESPACE = b"public-train-factor-workspace-v3-lesson\0"
_COLLAPSE_NAMESPACE = b"public-train-factor-workspace-v3-v2-collapse\0"
_CURRICULUM_NAMESPACE = b"public-train-factor-workspace-v3-curriculum\0"
_IDENTIFIABILITY_NAMESPACE = b"public-train-factor-workspace-v3-identifiability\0"
_TEACHER_RULE_NAMESPACE = b"public-train-factor-workspace-v3-teacher-rule\0"
_SURFACE_NAMESPACE = b"public-train-factor-workspace-v3-surface\0"
_PROTOCOL_NAMESPACE = b"public-train-factor-workspace-v3-protocol\0"


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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _factor_set_record(factors: tuple[str, ...]) -> dict[str, object]:
    if (
        type(factors) is not tuple
        or not factors
        or len(factors) > EVENT_CAPACITY
        or len(set(factors)) != len(factors)
        or any(factor not in FACTOR_REGISTRY for factor in factors)
        or factors != tuple(factor for factor in FACTOR_REGISTRY if factor in set(factors))
    ):
        raise ValueError("V3 factor set is not registry-ordered and exact")
    return {
        "factor_registry_sha256": v2.manifest_v1.FACTOR_REGISTRY_SHA256,
        "factors": factors,
        "cardinality": len(factors),
        "set_semantics": True,
    }


def _factor_set_sha256(factors: tuple[str, ...]) -> str:
    return _namespaced_sha256(_FACTOR_SET_NAMESPACE, _factor_set_record(factors))


_V5_ASSEMBLY_BY_TOKEN = {
    assembly.token: assembly.cells for assembly in v5.PROTOCOL.lexical_assemblies
}
CUE_ASSEMBLY_CELLS = MappingProxyType(
    {direction: _V5_ASSEMBLY_BY_TOKEN[token] for direction, token in v5.CUE_TOKENS.items()}
)
NEUTRAL_SLOT_ASSEMBLIES = tuple(
    _V5_ASSEMBLY_BY_TOKEN[token] for token in v5.PROTOCOL.neutral_spans[0].tokens[:2]
)


@dataclass(frozen=True, slots=True)
class CausalSelectorInputKey:
    """The complete learner-visible selector input, with no audit metadata."""

    direction: str
    cue_slot: int
    selected_slot: int
    slot_assembly_cells: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]

    def __post_init__(self) -> None:
        if (
            self.direction not in ("NEXT", "PREV")
            or type(self.cue_slot) is not int
            or type(self.selected_slot) is not int
            or not 0 <= self.cue_slot < 4
            or not 0 <= self.selected_slot < 4
            or self.cue_slot == self.selected_slot
            or self.selected_slot
            != ((self.cue_slot + 1) % 4 if self.direction == "NEXT" else (self.cue_slot - 1) % 4)
            or type(self.slot_assembly_cells) is not tuple
            or len(self.slot_assembly_cells) != 4
            or any(
                type(cells) is not tuple
                or len(cells) != ASSEMBLY_ACTIVE_CELL_COUNT
                or len(set(cells)) != ASSEMBLY_ACTIVE_CELL_COUNT
                or tuple(sorted(cells)) != cells
                or any(
                    type(cell) is not int or not 0 <= cell < LEXICAL_CELL_COUNT for cell in cells
                )
                for cells in self.slot_assembly_cells
            )
            or self.slot_assembly_cells[self.cue_slot] != CUE_ASSEMBLY_CELLS[self.direction]
        ):
            raise ValueError("V3 causal selector input key changed")
        free_slots = tuple(
            slot for slot in range(4) if slot not in (self.cue_slot, self.selected_slot)
        )
        if tuple(self.slot_assembly_cells[slot] for slot in free_slots) != (
            NEUTRAL_SLOT_ASSEMBLIES
        ):
            raise ValueError("V3 causal selector neutral assemblies changed")

    @property
    def selected_assembly_cells(self) -> tuple[int, ...]:
        return self.slot_assembly_cells[self.selected_slot]

    def record(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
            "slot_assembly_cells": self.slot_assembly_cells,
        }


@dataclass(frozen=True, slots=True)
class FactorSetClass:
    class_index: int
    target_factors: tuple[str, ...]
    physical_seed_assemblies: tuple[tuple[int, ...], ...]
    audit_source_profile_sha256s: tuple[tuple[str, ...], ...]
    factor_set_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "class_index": self.class_index,
            "teacher_target": _factor_set_record(self.target_factors),
            "physical_seed_assemblies": self.physical_seed_assemblies,
            "audit_only": {
                "source_v2_profile_sha256s": self.audit_source_profile_sha256s,
            },
            "factor_set_sha256": self.factor_set_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SelectorTemplate:
    template_index: int
    split: str
    direction: str
    cue_slot: int
    selected_slot: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "template_index": self.template_index,
            "split": self.split,
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FactorSetLesson:
    lesson_index: int
    class_index: int
    template_index: int
    split: str
    causal_input: CausalSelectorInputKey
    target_factors: tuple[str, ...]
    factor_set_sha256: str
    audit_seed_index: int
    audit_source_profile_sha256s: tuple[str, ...]
    sha256: str

    def causal_input_record(self) -> dict[str, object]:
        return self.causal_input.record()

    def teacher_target_record(self) -> dict[str, object]:
        return _factor_set_record(self.target_factors)

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_input": self.causal_input_record(),
            "teacher_target": self.teacher_target_record(),
            "factor_set_sha256": self.factor_set_sha256,
            "audit_only": {
                "lesson_index": self.lesson_index,
                "class_index": self.class_index,
                "template_index": self.template_index,
                "split": self.split,
                "seed_index": self.audit_seed_index,
                "source_v2_profile_sha256s": self.audit_source_profile_sha256s,
            },
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class V2AliasCollapse:
    physical_assembly_cells: tuple[int, ...]
    v2_factor_labels: tuple[str, ...]
    v2_lesson_indices: tuple[int, ...]
    v2_lesson_sha256s: tuple[str, ...]
    v3_target_factors: tuple[str, ...]
    v3_lesson_indices: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "physical_assembly_cells": self.physical_assembly_cells,
            "v3_teacher_target": _factor_set_record(self.v3_target_factors),
            "audit_only": {
                "v2_factor_labels": self.v2_factor_labels,
                "v2_lesson_indices": self.v2_lesson_indices,
                "v2_lesson_sha256s": self.v2_lesson_sha256s,
                "v3_lesson_indices": self.v3_lesson_indices,
            },
            "collapse_semantics": (
                "one physical trace teaches one exact registry-ordered factor set"
            ),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CausalIdentifiabilityProof:
    lesson_count: int
    unique_causal_input_key_count: int
    factor_set_class_count: int
    fit_lesson_count: int
    heldout_lesson_count: int
    fit_inputs_per_class: int
    heldout_inputs_per_class: int
    keys_with_multiple_target_sets: int
    total_function: bool
    excluded_noncausal_fields: tuple[str, ...]
    causal_binding_sha256: str
    v2_alias_collapse_count: int
    v2_alias_collapse_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "lesson_count": self.lesson_count,
            "unique_causal_input_key_count": self.unique_causal_input_key_count,
            "factor_set_class_count": self.factor_set_class_count,
            "fit_lesson_count": self.fit_lesson_count,
            "heldout_lesson_count": self.heldout_lesson_count,
            "fit_inputs_per_class": self.fit_inputs_per_class,
            "heldout_inputs_per_class": self.heldout_inputs_per_class,
            "keys_with_multiple_target_sets": self.keys_with_multiple_target_sets,
            "total_function": self.total_function,
            "excluded_noncausal_fields": self.excluded_noncausal_fields,
            "causal_binding_sha256": self.causal_binding_sha256,
            "v2_alias_collapse_count": self.v2_alias_collapse_count,
            "v2_alias_collapse_sha256": self.v2_alias_collapse_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainFactorWorkspaceV3Protocol:
    scope: str
    status: str
    canonical_identities: tuple[v2.CanonicalIdentityGroup, ...]
    canonical_fit_prompt_input_sha256s: tuple[str, ...]
    canonical_heldout_prompt_input_sha256s: tuple[str, ...]
    factor_set_classes: tuple[FactorSetClass, ...]
    selector_templates: tuple[SelectorTemplate, ...]
    lessons: tuple[FactorSetLesson, ...]
    v2_alias_collapses: tuple[V2AliasCollapse, ...]
    identifiability_proof: CausalIdentifiabilityProof
    masked_wta_controls: tuple[v2.MaskedWTAControl, ...]
    compositor_controls: tuple[v2.FactorSetCompositorControl, ...]
    dependency_sha256: str
    curriculum_sha256: str
    alias_collapse_sha256: str
    identifiability_sha256: str
    teacher_rule_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str

    def root_record(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "v2_protocol_sha256": V2_PROTOCOL_SHA256,
            "canonical_inventory_sha256": v2.PROTOCOL.canonical_inventory_sha256,
            "canonical_split_sha256": v2.PROTOCOL.canonical_split_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "alias_collapse_sha256": self.alias_collapse_sha256,
            "identifiability_sha256": self.identifiability_sha256,
            "teacher_rule_sha256": self.teacher_rule_sha256,
            "masked_wta_topology_sha256": v2.MASKED_WTA_TOPOLOGY_SHA256,
            "masked_wta_controls_sha256": v2.PROTOCOL.masked_wta_controls_sha256,
            "compositor_topology_sha256": v2.COMPOSITOR_TOPOLOGY_SHA256,
            "compositor_controls_sha256": v2.PROTOCOL.compositor_controls_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "canonical_prompt_appearance_count": CANONICAL_PROMPT_APPEARANCE_COUNT,
            "canonical_semantic_identity_count": len(self.canonical_identities),
            "canonical_fit_identity_count": len(self.canonical_fit_prompt_input_sha256s),
            "canonical_heldout_identity_count": len(self.canonical_heldout_prompt_input_sha256s),
            "active_factor_count": len(ACTIVE_FACTORS),
            "factor_set_class_count": len(self.factor_set_classes),
            "template_count": len(self.selector_templates),
            "lesson_count": len(self.lessons),
            "fit_lesson_count": sum(row.split == FIT for row in self.lessons),
            "heldout_lesson_count": sum(row.split == HELDOUT for row in self.lessons),
            "event_capacity": EVENT_CAPACITY,
            "learner_array_count": LEARNER_ARRAY_COUNT,
            "learning_run_count": LEARNING_RUN_COUNT,
            "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
            "protocol_sha256": self.protocol_sha256,
        }


def _authenticate_dependencies() -> None:
    source_path = Path(v2.__file__).resolve()
    if (
        sha256(source_path.read_bytes()).hexdigest() != V2_PROTOCOL_SOURCE_SHA256
        or v2.PROTOCOL.protocol_sha256 != V2_PROTOCOL_SHA256
        or v2._build_protocol()[0] != v2.PROTOCOL
        or v2.PROTOCOL.root_record() != v2._build_protocol()[0].root_record()
        or v2.PROTOCOL.status != v2.PROTOCOL_STATUS
        or any(
            (
                v2.DEV_INPUT_COUNT,
                v2.PROTECTED_INPUT_COUNT,
                v2.DESIRED_OUTPUT_COUNT,
                v2.PROGRAM_ARTIFACT_COUNT,
                v2.PROPERTY_ROW_COUNT,
                v2.FAMILY_IDENTIFIER_COUNT,
                v2.LEARNER_ARRAY_COUNT,
                v2.LEARNING_RUN_COUNT,
                v2.ELECTRICAL_EXECUTION_COUNT,
            )
        )
    ):
        raise RuntimeError("PUBLIC_TRAIN factor workspace V2 dependency changed")


DEPENDENCY_RECORD = _freeze_record(
    {
        "v2_protocol_commit": V2_PROTOCOL_COMMIT,
        "v2_protocol_source_sha256": V2_PROTOCOL_SOURCE_SHA256,
        "v2_protocol_sha256": V2_PROTOCOL_SHA256,
        "v2_dependency_sha256": v2.DEPENDENCY_SHA256,
        "factor_registry_sha256": v2.manifest_v1.FACTOR_REGISTRY_SHA256,
        "input_projection_sha256": v2.manifest_v1.INPUT_PROJECTION_SHA256,
        "selector_full_preflight_sha256": v5.SELECTOR_FULL_PREFLIGHT_SHA256,
        "canonical_inventory_sha256": v2.PROTOCOL.canonical_inventory_sha256,
        "canonical_split_sha256": v2.PROTOCOL.canonical_split_sha256,
        "masked_wta_topology_sha256": v2.MASKED_WTA_TOPOLOGY_SHA256,
        "compositor_topology_sha256": v2.COMPOSITOR_TOPOLOGY_SHA256,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(
    _DEPENDENCY_NAMESPACE,
    dict(DEPENDENCY_RECORD),
)


_TEMPLATE_SPECS = (
    (FIT, "NEXT", 0, 1),
    (FIT, "PREV", 1, 0),
    (FIT, "NEXT", 1, 2),
    (FIT, "PREV", 2, 1),
    (HELDOUT, "NEXT", 3, 0),
    (HELDOUT, "PREV", 0, 3),
)


def _build_templates() -> tuple[SelectorTemplate, ...]:
    rows: list[SelectorTemplate] = []
    for template_index, (split, direction, cue_slot, selected_slot) in enumerate(_TEMPLATE_SPECS):
        record = {
            "template_index": template_index,
            "split": split,
            "direction": direction,
            "cue_slot": cue_slot,
            "selected_slot": selected_slot,
        }
        rows.append(
            SelectorTemplate(
                template_index=template_index,
                split=split,
                direction=direction,
                cue_slot=cue_slot,
                selected_slot=selected_slot,
                sha256=_namespaced_sha256(_TEMPLATE_NAMESPACE, record),
            )
        )
    return tuple(rows)


def _build_factor_set_classes() -> tuple[FactorSetClass, ...]:
    profiles_by_cells: dict[tuple[int, ...], list[v2.SpanFeatureProfile]] = defaultdict(list)
    for profile in v2.PROTOCOL.span_feature_profiles:
        profiles_by_cells[profile.assembly_cells].append(profile)

    rows: list[FactorSetClass] = []
    for class_index, factors in enumerate(FACTOR_SET_CLASSES):
        factor_set = set(factors)
        physical_seeds: list[tuple[int, ...]] = []
        source_profile_rows: list[tuple[str, ...]] = []
        for cells, profiles in sorted(profiles_by_cells.items()):
            observed_factors = {profile.factor for profile in profiles}
            if (len(factors) == 2 and factor_set <= observed_factors) or (
                len(factors) == 1 and factors[0] in observed_factors and len(observed_factors) == 1
            ):
                physical_seeds.append(cells)
                source_profile_rows.append(
                    tuple(
                        profile.sha256
                        for profile in sorted(
                            profiles,
                            key=lambda row: (
                                FACTOR_REGISTRY.index(row.factor),
                                row.profile_index,
                            ),
                        )
                        if profile.factor in factor_set
                    )
                )
        if not physical_seeds or any(not row for row in source_profile_rows):
            raise RuntimeError("V3 factor-set class has no physical seed")
        class_record = {
            "class_index": class_index,
            "teacher_target": _factor_set_record(factors),
            "physical_seed_assemblies": tuple(physical_seeds),
            "audit_only": {
                "source_v2_profile_sha256s": tuple(source_profile_rows),
            },
            "factor_set_sha256": _factor_set_sha256(factors),
        }
        rows.append(
            FactorSetClass(
                class_index=class_index,
                target_factors=factors,
                physical_seed_assemblies=tuple(physical_seeds),
                audit_source_profile_sha256s=tuple(source_profile_rows),
                factor_set_sha256=_factor_set_sha256(factors),
                sha256=_namespaced_sha256(_CLASS_NAMESPACE, class_record),
            )
        )
    if len(rows) != FACTOR_SET_CLASS_COUNT or Counter(
        factor for row in rows for factor in row.target_factors
    ) != Counter(ACTIVE_FACTORS):
        raise RuntimeError("V3 factor-set class coverage changed")
    return tuple(rows)


def _causal_input(
    template: SelectorTemplate,
    selected_assembly_cells: tuple[int, ...],
) -> CausalSelectorInputKey:
    slots: list[tuple[int, ...] | None] = [None, None, None, None]
    slots[template.cue_slot] = CUE_ASSEMBLY_CELLS[template.direction]
    slots[template.selected_slot] = selected_assembly_cells
    free_slots = [
        slot for slot in range(4) if slot not in (template.cue_slot, template.selected_slot)
    ]
    for slot, neutral_cells in zip(
        free_slots,
        NEUTRAL_SLOT_ASSEMBLIES,
        strict=True,
    ):
        slots[slot] = neutral_cells
    if any(cells is None for cells in slots):
        raise RuntimeError("V3 selector input did not close all four slots")
    return CausalSelectorInputKey(
        direction=template.direction,
        cue_slot=template.cue_slot,
        selected_slot=template.selected_slot,
        slot_assembly_cells=(
            tuple(slots[0] or ()),
            tuple(slots[1] or ()),
            tuple(slots[2] or ()),
            tuple(slots[3] or ()),
        ),
    )


def _build_lessons(
    classes: tuple[FactorSetClass, ...],
    templates: tuple[SelectorTemplate, ...],
) -> tuple[FactorSetLesson, ...]:
    rows: list[FactorSetLesson] = []
    lesson_index = 0
    for factor_class in classes:
        for template in templates:
            seed_index = template.template_index % len(factor_class.physical_seed_assemblies)
            causal_input = _causal_input(
                template,
                factor_class.physical_seed_assemblies[seed_index],
            )
            lesson_record = {
                "causal_input": causal_input.record(),
                "teacher_target": _factor_set_record(factor_class.target_factors),
                "factor_set_sha256": factor_class.factor_set_sha256,
                "audit_only": {
                    "lesson_index": lesson_index,
                    "class_index": factor_class.class_index,
                    "template_index": template.template_index,
                    "split": template.split,
                    "seed_index": seed_index,
                    "source_v2_profile_sha256s": (
                        factor_class.audit_source_profile_sha256s[seed_index]
                    ),
                },
            }
            rows.append(
                FactorSetLesson(
                    lesson_index=lesson_index,
                    class_index=factor_class.class_index,
                    template_index=template.template_index,
                    split=template.split,
                    causal_input=causal_input,
                    target_factors=factor_class.target_factors,
                    factor_set_sha256=factor_class.factor_set_sha256,
                    audit_seed_index=seed_index,
                    audit_source_profile_sha256s=(
                        factor_class.audit_source_profile_sha256s[seed_index]
                    ),
                    sha256=_namespaced_sha256(_LESSON_NAMESPACE, lesson_record),
                )
            )
            lesson_index += 1
    return tuple(rows)


def _build_v2_alias_collapses(
    lessons: tuple[FactorSetLesson, ...],
) -> tuple[V2AliasCollapse, ...]:
    v2_rows_by_cells: dict[
        tuple[int, ...],
        list[v2.CounterfactualFactorLesson],
    ] = defaultdict(list)
    for group in v2.PROTOCOL.counterfactual_groups:
        for lesson in group.lessons:
            v2_rows_by_cells[lesson.target_assembly_cells].append(lesson)

    rows: list[V2AliasCollapse] = []
    for cells, v2_rows in sorted(v2_rows_by_cells.items()):
        labels = tuple(
            factor for factor in FACTOR_REGISTRY if factor in {row.factor for row in v2_rows}
        )
        if len(labels) < 2:
            continue
        ordered_v2_rows = tuple(sorted(v2_rows, key=lambda row: row.lesson_index))
        v3_rows = tuple(
            lesson
            for lesson in lessons
            if lesson.causal_input.selected_assembly_cells == cells
            and lesson.target_factors == labels
        )
        collapse_record = {
            "physical_assembly_cells": cells,
            "v3_teacher_target": _factor_set_record(labels),
            "audit_only": {
                "v2_factor_labels": labels,
                "v2_lesson_indices": tuple(row.lesson_index for row in ordered_v2_rows),
                "v2_lesson_sha256s": tuple(row.sha256 for row in ordered_v2_rows),
                "v3_lesson_indices": tuple(row.lesson_index for row in v3_rows),
            },
            "collapse_semantics": (
                "one physical trace teaches one exact registry-ordered factor set"
            ),
        }
        rows.append(
            V2AliasCollapse(
                physical_assembly_cells=cells,
                v2_factor_labels=labels,
                v2_lesson_indices=tuple(row.lesson_index for row in ordered_v2_rows),
                v2_lesson_sha256s=tuple(row.sha256 for row in ordered_v2_rows),
                v3_target_factors=labels,
                v3_lesson_indices=tuple(row.lesson_index for row in v3_rows),
                sha256=_namespaced_sha256(_COLLAPSE_NAMESPACE, collapse_record),
            )
        )
    if (
        len(rows) != 2
        or any(row.v2_factor_labels != ("minimum", "maximum") for row in rows)
        or {row.v2_lesson_indices for row in rows} != {(1, 2, 30, 39, 59), (22, 31, 51, 67, 68)}
        or any(len(row.v3_lesson_indices) != 3 for row in rows)
    ):
        raise RuntimeError("V3 did not explicitly collapse both V2 aliases")
    return tuple(rows)


def _build_identifiability_proof(
    lessons: tuple[FactorSetLesson, ...],
    collapses: tuple[V2AliasCollapse, ...],
) -> CausalIdentifiabilityProof:
    targets_by_key: dict[CausalSelectorInputKey, set[tuple[str, ...]]] = defaultdict(set)
    for lesson in lessons:
        targets_by_key[lesson.causal_input].add(lesson.target_factors)
    conflicts = sum(len(targets) != 1 for targets in targets_by_key.values())
    balance = Counter((lesson.target_factors, lesson.split) for lesson in lessons)
    if (
        len(lessons) != LESSON_COUNT
        or len(targets_by_key) != LESSON_COUNT
        or conflicts
        or any(
            balance[(factor_set, FIT)] != FIT_INPUTS_PER_CLASS
            or balance[(factor_set, HELDOUT)] != HELDOUT_INPUTS_PER_CLASS
            for factor_set in FACTOR_SET_CLASSES
        )
    ):
        raise RuntimeError("V3 stripped causal input is not a balanced total function")

    causal_bindings = tuple(
        {
            "causal_input": lesson.causal_input.record(),
            "teacher_target": _factor_set_record(lesson.target_factors),
        }
        for lesson in lessons
    )
    causal_binding_sha = _namespaced_sha256(
        _IDENTIFIABILITY_NAMESPACE + b"bindings\0",
        causal_bindings,
    )
    collapse_sha = _namespaced_sha256(
        _COLLAPSE_NAMESPACE + b"panel\0",
        tuple(row.record() for row in collapses),
    )
    excluded = (
        "factor_label",
        "profile_sha256",
        "lesson_id",
        "lesson_sha256",
        "input_sha256",
        "audit_hash",
        "split",
        "class_index",
        "template_index",
    )
    proof_record = {
        "lesson_count": len(lessons),
        "unique_causal_input_key_count": len(targets_by_key),
        "factor_set_class_count": FACTOR_SET_CLASS_COUNT,
        "fit_lesson_count": sum(row.split == FIT for row in lessons),
        "heldout_lesson_count": sum(row.split == HELDOUT for row in lessons),
        "fit_inputs_per_class": FIT_INPUTS_PER_CLASS,
        "heldout_inputs_per_class": HELDOUT_INPUTS_PER_CLASS,
        "keys_with_multiple_target_sets": conflicts,
        "total_function": conflicts == 0,
        "excluded_noncausal_fields": excluded,
        "causal_binding_sha256": causal_binding_sha,
        "v2_alias_collapse_count": len(collapses),
        "v2_alias_collapse_sha256": collapse_sha,
    }
    return CausalIdentifiabilityProof(
        lesson_count=len(lessons),
        unique_causal_input_key_count=len(targets_by_key),
        factor_set_class_count=FACTOR_SET_CLASS_COUNT,
        fit_lesson_count=sum(row.split == FIT for row in lessons),
        heldout_lesson_count=sum(row.split == HELDOUT for row in lessons),
        fit_inputs_per_class=FIT_INPUTS_PER_CLASS,
        heldout_inputs_per_class=HELDOUT_INPUTS_PER_CLASS,
        keys_with_multiple_target_sets=conflicts,
        total_function=conflicts == 0,
        excluded_noncausal_fields=excluded,
        causal_binding_sha256=causal_binding_sha,
        v2_alias_collapse_count=len(collapses),
        v2_alias_collapse_sha256=collapse_sha,
        sha256=_namespaced_sha256(_IDENTIFIABILITY_NAMESPACE, proof_record),
    )


TEACHER_RULE_RECORD = _freeze_record(
    {
        "teacher_target": "one exact registry-ordered factor set per causal input",
        "minimum_maximum_rule": (
            "the shared canonical assembly teaches {minimum, maximum} in one lesson"
        ),
        "physical_publication": (
            "repeated masked factor/STOP races publish every distinct target factor"
        ),
        "event_capacity": EVENT_CAPACITY,
        "candidate_admission": "exact physical factor-set equality",
        "ordered_publication_is_audit_only": True,
        "same_factor_duplicate": "absorbing duplicate poison",
        "fifth_distinct_factor": "absorbing capacity poison",
        "different_factor_overlap": "legal",
        "different_factor_nesting": "legal",
        "host_union_or_filter": False,
    }
)
TEACHER_RULE_SHA256 = _namespaced_sha256(
    _TEACHER_RULE_NAMESPACE,
    dict(TEACHER_RULE_RECORD),
)

SURFACE_BOUNDARY_RECORD = _freeze_record(
    {
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
        "execution_surface_present": False,
        "learner_or_inference_interface_present": False,
        "deployment_checkpoint_present": False,
        "data_only": True,
    }
)
SURFACE_BOUNDARY_SHA256 = _namespaced_sha256(
    _SURFACE_NAMESPACE,
    dict(SURFACE_BOUNDARY_RECORD),
)


def _build_protocol() -> FrozenPublicTrainFactorWorkspaceV3Protocol:
    _authenticate_dependencies()
    classes = _build_factor_set_classes()
    templates = _build_templates()
    lessons = _build_lessons(classes, templates)
    collapses = _build_v2_alias_collapses(lessons)
    proof = _build_identifiability_proof(lessons, collapses)

    curriculum_record = {
        "factor_set_classes": tuple(row.record() for row in classes),
        "selector_templates": tuple(row.record() for row in templates),
        "lessons": tuple(row.record() for row in lessons),
        "balance": {
            "factor_set_class_count": FACTOR_SET_CLASS_COUNT,
            "fit_inputs_per_class": FIT_INPUTS_PER_CLASS,
            "heldout_inputs_per_class": HELDOUT_INPUTS_PER_CLASS,
            "fit_lesson_count": FIT_LESSON_COUNT,
            "heldout_lesson_count": HELDOUT_LESSON_COUNT,
            "lesson_count": LESSON_COUNT,
        },
        "v2_lesson_sha_overlap_count": len(
            {row.sha256 for row in lessons}
            & {
                lesson.sha256
                for group in v2.PROTOCOL.counterfactual_groups
                for lesson in group.lessons
            }
        ),
    }
    curriculum_sha = _namespaced_sha256(_CURRICULUM_NAMESPACE, curriculum_record)
    alias_collapse_sha = proof.v2_alias_collapse_sha256

    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": PROTOCOL_STATUS,
        "dependency_sha256": DEPENDENCY_SHA256,
        "v2_protocol_sha256": V2_PROTOCOL_SHA256,
        "canonical_inventory_sha256": v2.PROTOCOL.canonical_inventory_sha256,
        "canonical_split_sha256": v2.PROTOCOL.canonical_split_sha256,
        "curriculum_sha256": curriculum_sha,
        "alias_collapse_sha256": alias_collapse_sha,
        "identifiability_sha256": proof.sha256,
        "teacher_rule_sha256": TEACHER_RULE_SHA256,
        "masked_wta_topology_sha256": v2.MASKED_WTA_TOPOLOGY_SHA256,
        "masked_wta_controls_sha256": v2.PROTOCOL.masked_wta_controls_sha256,
        "compositor_topology_sha256": v2.COMPOSITOR_TOPOLOGY_SHA256,
        "compositor_controls_sha256": v2.PROTOCOL.compositor_controls_sha256,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "canonical_prompt_appearance_count": CANONICAL_PROMPT_APPEARANCE_COUNT,
        "canonical_semantic_identity_count": CANONICAL_SEMANTIC_IDENTITY_COUNT,
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "active_factor_count": len(ACTIVE_FACTORS),
        "factor_set_class_count": FACTOR_SET_CLASS_COUNT,
        "template_count": TEMPLATE_COUNT,
        "lesson_count": LESSON_COUNT,
        "fit_lesson_count": FIT_LESSON_COUNT,
        "heldout_lesson_count": HELDOUT_LESSON_COUNT,
        "event_capacity": EVENT_CAPACITY,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
    }
    return FrozenPublicTrainFactorWorkspaceV3Protocol(
        scope=PROTOCOL_SCOPE,
        status=PROTOCOL_STATUS,
        canonical_identities=v2.PROTOCOL.canonical_identities,
        canonical_fit_prompt_input_sha256s=(v2.PROTOCOL.canonical_fit_prompt_input_sha256s),
        canonical_heldout_prompt_input_sha256s=(v2.PROTOCOL.canonical_heldout_prompt_input_sha256s),
        factor_set_classes=classes,
        selector_templates=templates,
        lessons=lessons,
        v2_alias_collapses=collapses,
        identifiability_proof=proof,
        masked_wta_controls=v2.PROTOCOL.masked_wta_controls,
        compositor_controls=v2.PROTOCOL.compositor_controls,
        dependency_sha256=DEPENDENCY_SHA256,
        curriculum_sha256=curriculum_sha,
        alias_collapse_sha256=alias_collapse_sha,
        identifiability_sha256=proof.sha256,
        teacher_rule_sha256=TEACHER_RULE_SHA256,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        protocol_sha256=_namespaced_sha256(
            _PROTOCOL_NAMESPACE,
            root_without_protocol,
        ),
    )


PROTOCOL = _build_protocol()


__all__ = [
    "ACTIVE_FACTORS",
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "CANONICAL_FIT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_IDENTITY_COUNT",
    "CANONICAL_PROMPT_APPEARANCE_COUNT",
    "CANONICAL_SEMANTIC_IDENTITY_COUNT",
    "CUE_ASSEMBLY_CELLS",
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
    "HELDOUT",
    "HELDOUT_INPUTS_PER_CLASS",
    "HELDOUT_LESSON_COUNT",
    "LEARNER_ARRAY_COUNT",
    "LEARNING_RUN_COUNT",
    "LESSON_COUNT",
    "NEUTRAL_SLOT_ASSEMBLIES",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PROTOCOL_STATUS",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "TEACHER_RULE_RECORD",
    "TEACHER_RULE_SHA256",
    "TEMPLATE_COUNT",
    "V2_PROTOCOL_COMMIT",
    "V2_PROTOCOL_SHA256",
    "V2_PROTOCOL_SOURCE_SHA256",
    "CausalIdentifiabilityProof",
    "CausalSelectorInputKey",
    "FactorSetClass",
    "FactorSetLesson",
    "FrozenPublicTrainFactorWorkspaceV3Protocol",
    "SelectorTemplate",
    "V2AliasCollapse",
]
