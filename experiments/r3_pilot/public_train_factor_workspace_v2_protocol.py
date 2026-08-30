"""Frozen data-only protocol for the public-TRAIN factor workspace V2.

This module preregisters data and physical contracts only.  It contains no
learner arrays, fitting path, inference path, desired outputs, programs,
property rows, task/family identifiers, or DEV/protected inputs.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from experiments.r3_pilot import public_train_prompt_factor_manifest_v1 as manifest_v1
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5

PROTOCOL_SCOPE = "PUBLIC_TRAIN_FACTOR_WORKSPACE_V2_PROTOCOL"
PROTOCOL_STATUS = "FROZEN_DATA_ONLY_NOT_RUN"

FIT = "FIT"
HELDOUT = "HELDOUT"

FACTOR_REGISTRY = manifest_v1.FACTOR_REGISTRY
FACTOR_COUNT = len(FACTOR_REGISTRY)
ACTIVE_FACTORS = (
    "add",
    "minimum",
    "maximum",
    "absolute",
    "less-than",
    "map",
    "sum",
    "count",
    "filter",
    "reverse",
    "class:ascii-decimal-digits",
    "class:ascii-lowercase",
    "class:english-vowels",
)
DORMANT_FACTORS = tuple(factor for factor in FACTOR_REGISTRY if factor not in ACTIVE_FACTORS)
RARE_FIT_FACTORS = (
    "absolute",
    "filter",
    "reverse",
    "class:ascii-decimal-digits",
    "class:ascii-lowercase",
    "class:english-vowels",
)

PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT = 90
CANONICAL_SEMANTIC_IDENTITY_COUNT = 42
CANONICAL_FIT_IDENTITY_COUNT = 24
CANONICAL_HELDOUT_IDENTITY_COUNT = 18
COUNTERFACTUAL_GROUP_COUNT = 30
COUNTERFACTUAL_LESSON_COUNT = 78
COUNTERFACTUAL_FIT_COUNT = 52
COUNTERFACTUAL_HELDOUT_COUNT = 26
ROOTED_TEMPLATE_COUNT = 6
GROUP_SIZES = (3, 3, 3, 2, 2)
FACTOR_ROTATION_STRIDE = 5

LEXICAL_CELL_COUNT = 512
ASSEMBLY_ACTIVE_CELL_COUNT = 12
EVENT_CAPACITY = 4
MASKED_WTA_PRINCIPAL_COUNT = FACTOR_COUNT + 1
MASKED_WTA_STOP_INDEX = FACTOR_COUNT
CARDINALITY_CELL_COUNT = EVENT_CAPACITY + 1

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_COUNT = 0
PROGRAM_ARTIFACT_COUNT = 0
PROPERTY_ROW_COUNT = 0
FAMILY_IDENTIFIER_COUNT = 0
LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0

PROMPT_SUBSTRATE_COMMIT = "c89a3c6"
PROMPT_MANIFEST_COMMIT = "4cc4a1a"
V5_PROTOCOL_COMMIT = "1326ddb"
V5_LEARNER_COMMIT = "5cb01cc"

PROMPT_SUBSTRATE_SOURCE_SHA256 = "74a123d88707525b63ff50021b3c7e5638f93ed8ed3dd3b82d20724bb3f1d662"
PROMPT_MANIFEST_SOURCE_SHA256 = "95aa4af55c9cd9d6fa841e203beccf2dce8a67fe41bf3e15d1fe1dfeacb67c78"
V5_PROTOCOL_SOURCE_SHA256 = "d62fe76444ef48f0b58723f0cede1b6ca76e86c78623eb43c07f38be8e8c8746"
V5_LEARNER_SOURCE_SHA256 = "c76292164b43586b9400f8277f1c2aa71b2a136f113f36bc3f89712f0ca6895d"

_CANONICAL_INVENTORY_NAMESPACE = b"public-train-factor-workspace-v2-canonical\0"
_CANONICAL_SPLIT_NAMESPACE = b"public-train-factor-workspace-v2-split\0"
_SPAN_PROFILE_NAMESPACE = b"public-train-factor-workspace-v2-span-profile\0"
_TEMPLATE_NAMESPACE = b"public-train-factor-workspace-v2-template\0"
_COUNTERFACTUAL_INPUT_NAMESPACE = b"public-train-factor-workspace-v2-input\0"
_COUNTERFACTUAL_LESSON_NAMESPACE = b"public-train-factor-workspace-v2-lesson\0"
_COUNTERFACTUAL_GROUP_NAMESPACE = b"public-train-factor-workspace-v2-group\0"
_COUNTERFACTUAL_CURRICULUM_NAMESPACE = b"public-train-factor-workspace-v2-curriculum\0"
_FACTOR_SET_NAMESPACE = b"public-train-factor-workspace-v2-factor-set\0"
_ORDERED_PUBLICATION_NAMESPACE = b"public-train-factor-workspace-v2-publication\0"
_EVENT_NAMESPACE = b"public-train-factor-workspace-v2-event\0"
_CONTROL_NAMESPACE = b"public-train-factor-workspace-v2-control\0"
_DEPENDENCY_NAMESPACE = b"public-train-factor-workspace-v2-dependencies\0"
_MASKED_WTA_TOPOLOGY_NAMESPACE = b"public-train-factor-workspace-v2-masked-wta\0"
_COMPOSITOR_TOPOLOGY_NAMESPACE = b"public-train-factor-workspace-v2-compositor\0"
_ELECTRICAL_PREFLIGHT_NAMESPACE = b"public-train-factor-workspace-v2-preflight\0"
_SURFACE_BOUNDARY_NAMESPACE = b"public-train-factor-workspace-v2-surface\0"
_PROTOCOL_NAMESPACE = b"public-train-factor-workspace-v2-protocol\0"


def _plain_record(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_record(item) for key, item in value.items()}
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
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_record(item) for key, item in value.items()})
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
    factor_set = frozenset(factors)
    if len(factor_set) != len(factors) or not factor_set <= set(FACTOR_REGISTRY):
        raise ValueError("factor set requires unique registered factors")
    occupancy = tuple(int(factor in factor_set) for factor in FACTOR_REGISTRY)
    return {
        "factor_registry_sha256": manifest_v1.FACTOR_REGISTRY_SHA256,
        "occupancy": occupancy,
        "cardinality": len(factor_set),
    }


def _factor_set_sha256(factors: tuple[str, ...]) -> str:
    return _namespaced_sha256(_FACTOR_SET_NAMESPACE, _factor_set_record(factors))


@dataclass(frozen=True, slots=True)
class CanonicalIdentityGroup:
    prompt_input_sha256: str
    appearance_count: int
    target_factors: tuple[str, ...]
    factor_set_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "prompt_input_sha256": self.prompt_input_sha256,
            "appearance_count": self.appearance_count,
            "target_factors": self.target_factors,
            "target_order_semantics": "prompt-span-audit-only",
            "factor_set_sha256": self.factor_set_sha256,
        }


@dataclass(frozen=True, slots=True)
class SpanFeatureProfile:
    factor: str
    profile_index: int
    span_identity_sha256: str
    tokens: tuple[str, ...]
    assembly_cells: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "factor": self.factor,
            "profile_index": self.profile_index,
            "span_identity_sha256": self.span_identity_sha256,
            "tokens": self.tokens,
            "assembly_cells": self.assembly_cells,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RootedCounterfactualTemplate:
    template_index: int
    split: str
    direction: str
    cue_slot: int
    selected_slot: int
    rotation: int
    factor_groups: tuple[tuple[str, ...], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "template_index": self.template_index,
            "split": self.split,
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
            "rotation": self.rotation,
            "factor_groups": self.factor_groups,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CounterfactualFactorLesson:
    lesson_index: int
    template_index: int
    group_index: int
    target_index: int
    split: str
    direction: str
    cue_slot: int
    selected_slot: int
    input_slots: tuple[str, str, str, str]
    input_sha256: str
    factor: str
    target_span_profile_sha256: str
    target_span_identity_sha256: str
    target_assembly_cells: tuple[int, ...]
    sha256: str

    def input_record(self) -> dict[str, object]:
        return {
            "input_projection_sha256": manifest_v1.INPUT_PROJECTION_SHA256,
            "selector_preflight_sha256": v5.SELECTOR_FULL_PREFLIGHT_SHA256,
            "input_slots": self.input_slots,
        }

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "lesson_index": self.lesson_index,
            "template_index": self.template_index,
            "group_index": self.group_index,
            "target_index": self.target_index,
            "split": self.split,
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
            "input_slots": self.input_slots,
            "input_sha256": self.input_sha256,
            "factor": self.factor,
            "target_span_profile_sha256": self.target_span_profile_sha256,
            "target_span_identity_sha256": self.target_span_identity_sha256,
            "target_assembly_cells": self.target_assembly_cells,
            "teacher_target_count": 1,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CounterfactualContrastGroup:
    group_index: int
    template_index: int
    split: str
    direction: str
    cue_slot: int
    selected_slot: int
    factors: tuple[str, ...]
    lessons: tuple[CounterfactualFactorLesson, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "group_index": self.group_index,
            "template_index": self.template_index,
            "split": self.split,
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
            "factors": self.factors,
            "lessons": tuple(lesson.record() for lesson in self.lessons),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class MaskedWTAControl:
    control_index: int
    name: str
    principal_tiers: tuple[tuple[str, ...], ...]
    expected_status: str
    expected_factor_publication: tuple[str, ...]
    expected_masked_factors: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "control_index": self.control_index,
            "name": self.name,
            "principal_tiers": self.principal_tiers,
            "expected_status": self.expected_status,
            "expected_factor_publication": self.expected_factor_publication,
            "expected_masked_factors": self.expected_masked_factors,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ProtocolFactorEvent:
    factor: str
    start_token: int
    stop_token: int
    masked_rank: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "factor": self.factor,
            "start_token": self.start_token,
            "stop_token": self.stop_token,
            "masked_rank": self.masked_rank,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FactorSetCompositorControl:
    control_index: int
    name: str
    events: tuple[ProtocolFactorEvent, ...]
    reset_before: bool
    expected_status: str
    expected_public_factors: tuple[str, ...]
    expected_pre_poison_factors: tuple[str, ...]
    expected_factor_set_sha256: str | None
    ordered_publication_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "control_index": self.control_index,
            "name": self.name,
            "events": tuple(event.record() for event in self.events),
            "reset_before": self.reset_before,
            "expected_status": self.expected_status,
            "expected_public_factors": self.expected_public_factors,
            "expected_pre_poison_factors": self.expected_pre_poison_factors,
            "expected_factor_set_sha256": self.expected_factor_set_sha256,
            "ordered_publication_sha256": self.ordered_publication_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainFactorWorkspaceV2Protocol:
    scope: str
    status: str
    canonical_identities: tuple[CanonicalIdentityGroup, ...]
    canonical_fit_prompt_input_sha256s: tuple[str, ...]
    canonical_heldout_prompt_input_sha256s: tuple[str, ...]
    span_feature_profiles: tuple[SpanFeatureProfile, ...]
    rooted_templates: tuple[RootedCounterfactualTemplate, ...]
    counterfactual_groups: tuple[CounterfactualContrastGroup, ...]
    masked_wta_controls: tuple[MaskedWTAControl, ...]
    compositor_controls: tuple[FactorSetCompositorControl, ...]
    dependency_sha256: str
    canonical_inventory_sha256: str
    canonical_split_sha256: str
    active_factor_coverage_sha256: str
    span_feature_profiles_sha256: str
    counterfactual_curriculum_sha256: str
    rare_factor_support_sha256: str
    masked_wta_topology_sha256: str
    masked_wta_controls_sha256: str
    compositor_topology_sha256: str
    compositor_controls_sha256: str
    electrical_preflight_contract_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str

    @property
    def counterfactual_lessons(self) -> tuple[CounterfactualFactorLesson, ...]:
        return tuple(lesson for group in self.counterfactual_groups for lesson in group.lessons)

    def root_record(self, *, include_protocol_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "canonical_inventory_sha256": self.canonical_inventory_sha256,
            "canonical_split_sha256": self.canonical_split_sha256,
            "active_factor_coverage_sha256": self.active_factor_coverage_sha256,
            "span_feature_profiles_sha256": self.span_feature_profiles_sha256,
            "counterfactual_curriculum_sha256": self.counterfactual_curriculum_sha256,
            "rare_factor_support_sha256": self.rare_factor_support_sha256,
            "masked_wta_topology_sha256": self.masked_wta_topology_sha256,
            "masked_wta_controls_sha256": self.masked_wta_controls_sha256,
            "compositor_topology_sha256": self.compositor_topology_sha256,
            "compositor_controls_sha256": self.compositor_controls_sha256,
            "electrical_preflight_contract_sha256": (self.electrical_preflight_contract_sha256),
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "public_train_prompt_appearance_count": PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
            "canonical_semantic_identity_count": len(self.canonical_identities),
            "canonical_fit_identity_count": len(self.canonical_fit_prompt_input_sha256s),
            "canonical_heldout_identity_count": len(self.canonical_heldout_prompt_input_sha256s),
            "active_factor_count": len(ACTIVE_FACTORS),
            "rooted_template_count": len(self.rooted_templates),
            "counterfactual_group_count": len(self.counterfactual_groups),
            "counterfactual_lesson_count": len(self.counterfactual_lessons),
            "counterfactual_fit_count": sum(
                lesson.split == FIT for lesson in self.counterfactual_lessons
            ),
            "counterfactual_heldout_count": sum(
                lesson.split == HELDOUT for lesson in self.counterfactual_lessons
            ),
            "event_capacity": EVENT_CAPACITY,
            "learner_array_count": LEARNER_ARRAY_COUNT,
            "learning_run_count": LEARNING_RUN_COUNT,
            "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
            "learner_state_constructed": False,
            "learning_executed": False,
            "electrical_controls_executed": False,
            "execution_surface_present": False,
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


_CANONICAL_IDENTITY_SPECS = (
    ("0b1b35530cf1cd008dbf1f06c9b12b5bdb4f7f21116800eb41f145565e7218db", 1, ("minimum", "maximum")),
    ("44d4833bf08a64132d580a3b4177eb7a519a5e332437276c4cfcaa90acdaa6c6", 1, ("minimum", "maximum")),
    ("819f05c7981087f36c02d8fb77aaa262f814bbf3f3f054ffed470053fa081613", 3, ("minimum", "maximum")),
    ("92574fe36c00ab573f262c9e2e7fb6efe8b70334a63f2febb7a6e3513a97412c", 3, ("minimum", "maximum")),
    ("c1822ecc1eed61b2fd345a3da91d8ece36e8df89fe9ddef05c7ec0cadac8f416", 2, ("minimum", "maximum")),
    ("1ea63f6a9e51f0ba7c53332f6bb6afa257c1e0c9e6ac8d0d7a2c6c3105a0350a", 1, ("add",)),
    ("3c6d723ca3674df3a00a2bb3209ac91c532d6556154a302d133dd790d58b3417", 1, ("add",)),
    ("93b50c88e382800341b4df95eb2c266d21e049033e7a124d4aa355bc9f1e3830", 1, ("add",)),
    ("9c37e4ec14e765097f2ed3b9ce639758611b401d128112b478ef1ed8e17d5db1", 2, ("add",)),
    ("a69c93d115f0c21b0fd15393205ab280f3d6a94d71a1e24ca2fc10d9aa69de70", 1, ("add",)),
    ("ada96e9f27df9673cf5403da639d2c00dc7601404eb521d1e75c7a3e1f727981", 1, ("add",)),
    ("b0e11966bc4d8401403bdd8911f88d5ca2fca5cd089cb632a4f2dca3412b1057", 1, ("add",)),
    ("c13c6983dcc883f18221f1bb498804b3bfd328aab5e9f801553f2d0735b36f5f", 1, ("add",)),
    ("ef74c4faa42df9fe13830226617d93ea9a0ce426b57657babfd57aad5c1c34e8", 1, ("add",)),
    ("bb19ef6d812310fc149c7674e202255f4deec260ae6815b455849183b0c241b9", 10, ("maximum",)),
    ("8ae6fe3144d91c4eba12deaae79953a391399c21a2612d1edca63d840c4f2b3c", 1, ("add", "map")),
    ("a5384e5aaf19583709419e4d18a1c9e36f0b41fdcd5257f6b8ade3c545a7d453", 3, ("add", "map")),
    ("ca5b99de021d754bd33999d7f59b18a2e2fd4132afb1559404ad9b8b890e750b", 1, ("add", "map")),
    ("cce640ee89e2c02c5b3749e1bdde528d822b825fad77d65d964c0d041a927628", 3, ("add", "map")),
    ("f81c12c1435d040d5fdd6977ca98269f184ac6ba3f0513c6c715d12b371835f1", 2, ("add", "map")),
    ("e673f87f51571c9b3b11d4e81aac52a993e11a6ca9d7155e57e43b129c41c1f5", 10, ("absolute", "map")),
    (
        "2b98f16e11011af3a4a43e467fd032b38191eed195ee4dc397e462848b12686c",
        3,
        ("minimum", "maximum", "map", "sum"),
    ),
    (
        "449c4e8e502b9e0eb9454c4f85d6618cdcd768ad03976d3d28486727b05cb596",
        1,
        ("minimum", "maximum", "map", "sum"),
    ),
    (
        "89772cfe0acde474c4bd561aff4620d70d2f99a29b7ad595bfd38f50d01178ee",
        2,
        ("minimum", "maximum", "map", "sum"),
    ),
    (
        "90c8125972221dd26bfcda51f78ad12bfa2678a7ef8e2712959ad837eaeddbff",
        1,
        ("minimum", "maximum", "map", "sum"),
    ),
    (
        "9dc96b833e02ced1f9bd681ecdd3c2d5e982eec8fcee427a772569dcb419b2e0",
        1,
        ("minimum", "maximum", "map", "sum"),
    ),
    (
        "e8d84713fea7c2d0057368bea1e75aa3dd2eeaa1c685e9dc675de41428dd288d",
        1,
        ("minimum", "maximum", "map", "sum"),
    ),
    (
        "fc3c43134eeaea4c920f1f31f15e04da1081b3ea5fb29c492d3058961c518224",
        1,
        ("minimum", "maximum", "map", "sum"),
    ),
    (
        "0eb123c8a8a5d799a359dd1a5cd8a0f36f8a30cffbccaf2b1c9fa676035a3a1c",
        1,
        ("add", "map", "count", "less-than"),
    ),
    (
        "17bffc18f60edf862f6ac6b141fbdbc8522b30d3b78a699fe229f6d13fafca48",
        1,
        ("add", "map", "count", "less-than"),
    ),
    (
        "53b970f9f2bc8cb840b6d1d13e71977c62b9cb203888374cc9fb170bac05a796",
        1,
        ("add", "map", "count", "less-than"),
    ),
    (
        "78e892eb3ea2d16eca25293096ab61793a051cbbc8d30a72c250e5b2b753f0dc",
        1,
        ("add", "map", "count", "less-than"),
    ),
    (
        "b54c3f40cf170b70da6fb073b2bd825bcca12b9508dcb12900a29ae8ef2c295d",
        1,
        ("add", "map", "count", "less-than"),
    ),
    (
        "d67e923ca43060115634a1b26ea63b0c745723b963e45d48ab320848f1dd27c6",
        2,
        ("add", "map", "count", "less-than"),
    ),
    (
        "e99695895a675fd842c92a151fd8a06f257dfd1a8410f5f04178cc6a84187684",
        1,
        ("add", "map", "count", "less-than"),
    ),
    (
        "ead6b56f7281b339824ae963e9c7de1764d743b862fbb7a98bac84d094b35ef6",
        2,
        ("add", "map", "count", "less-than"),
    ),
    (
        "fccd2b5c2b1538c0f6374c4d26faff3f75722773fd9ad9ebd7922a5f3381e0dd",
        2,
        ("filter", "class:ascii-decimal-digits"),
    ),
    (
        "c11291cc43f2cd38d2491b6ec4a4da3f45fe97f8b792fe01029d15835002b0e8",
        4,
        ("filter", "class:ascii-lowercase"),
    ),
    (
        "915f0c59431bced6f285bd3e276776652d8fa375f6ed6c3b5fd30a4a6753b6ff",
        4,
        ("filter", "class:english-vowels"),
    ),
    (
        "86a085f40de0125e2e9246fc1f0e50ca1dfcc15cabbaa2cf05dbf2eb2f91348b",
        3,
        ("filter", "class:ascii-decimal-digits", "reverse"),
    ),
    (
        "88a0aab410711043014c050ec9a4f1fb31628d5d7838a81c5169dbb2bf307bbd",
        2,
        ("filter", "class:ascii-lowercase", "reverse"),
    ),
    (
        "cf798e16f3e6e7ef6dc6466c3e915cb6587e4c1c04c695800050a7ac3d8dffed",
        5,
        ("filter", "class:english-vowels", "reverse"),
    ),
)

CANONICAL_HELDOUT_PROMPT_INPUT_SHA256S = (
    "0eb123c8a8a5d799a359dd1a5cd8a0f36f8a30cffbccaf2b1c9fa676035a3a1c",
    "17bffc18f60edf862f6ac6b141fbdbc8522b30d3b78a699fe229f6d13fafca48",
    "2b98f16e11011af3a4a43e467fd032b38191eed195ee4dc397e462848b12686c",
    "449c4e8e502b9e0eb9454c4f85d6618cdcd768ad03976d3d28486727b05cb596",
    "86a085f40de0125e2e9246fc1f0e50ca1dfcc15cabbaa2cf05dbf2eb2f91348b",
    "88a0aab410711043014c050ec9a4f1fb31628d5d7838a81c5169dbb2bf307bbd",
    "915f0c59431bced6f285bd3e276776652d8fa375f6ed6c3b5fd30a4a6753b6ff",
    "c11291cc43f2cd38d2491b6ec4a4da3f45fe97f8b792fe01029d15835002b0e8",
    "cf798e16f3e6e7ef6dc6466c3e915cb6587e4c1c04c695800050a7ac3d8dffed",
    "e673f87f51571c9b3b11d4e81aac52a993e11a6ca9d7155e57e43b129c41c1f5",
    "fccd2b5c2b1538c0f6374c4d26faff3f75722773fd9ad9ebd7922a5f3381e0dd",
    "0b1b35530cf1cd008dbf1f06c9b12b5bdb4f7f21116800eb41f145565e7218db",
    "1ea63f6a9e51f0ba7c53332f6bb6afa257c1e0c9e6ac8d0d7a2c6c3105a0350a",
    "3c6d723ca3674df3a00a2bb3209ac91c532d6556154a302d133dd790d58b3417",
    "44d4833bf08a64132d580a3b4177eb7a519a5e332437276c4cfcaa90acdaa6c6",
    "53b970f9f2bc8cb840b6d1d13e71977c62b9cb203888374cc9fb170bac05a796",
    "78e892eb3ea2d16eca25293096ab61793a051cbbc8d30a72c250e5b2b753f0dc",
    "819f05c7981087f36c02d8fb77aaa262f814bbf3f3f054ffed470053fa081613",
)

_SPAN_PROFILE_SPECS = (
    (
        "minimum",
        "bf72cd5d37d2b59fb1a830f921221d507bb1a2d4d15c1ccd62f0fab3d3c5f214",
        ("limited", "to", "the", "inclusive"),
        (54, 67, 91, 99, 114, 157, 162, 216, 224, 231, 254, 329),
    ),
    (
        "minimum",
        "f41a2c3ef24cfacf12c8b2a14aa203daac3a3d42fcd208c7b9fa050a02147e95",
        ("limit", "every", "integer", "to"),
        (49, 53, 72, 272, 325, 362, 391, 443, 455, 467, 483, 507),
    ),
    (
        "maximum",
        "bf72cd5d37d2b59fb1a830f921221d507bb1a2d4d15c1ccd62f0fab3d3c5f214",
        ("limited", "to", "the", "inclusive"),
        (54, 67, 91, 99, 114, 157, 162, 216, 224, 231, 254, 329),
    ),
    (
        "maximum",
        "c67cdc2b7aec3ab6d63257d9c5999da06c714ee6098fd5b5f6005a4bcf2d4909",
        ("nonnegative",),
        (132, 137, 190, 241, 257, 311, 316, 384, 437, 444, 478, 484),
    ),
    (
        "maximum",
        "f41a2c3ef24cfacf12c8b2a14aa203daac3a3d42fcd208c7b9fa050a02147e95",
        ("limit", "every", "integer", "to"),
        (49, 53, 72, 272, 325, 362, 391, 443, 455, 467, 483, 507),
    ),
    (
        "add",
        "255faf9892026d7850becb6156a4069d898b9da907b2359155e1088dbe1db9b7",
        ("add", "<INT>", "to", "each"),
        (22, 155, 216, 220, 226, 257, 260, 299, 317, 318, 339, 426),
    ),
    (
        "add",
        "a4147c76eed7bce453f000c4fcb991bc80be0df15a701b3d6264fa7847ca9eac",
        ("plus", "the", "fixed", "offset"),
        (47, 59, 111, 150, 163, 204, 210, 295, 299, 383, 440, 470),
    ),
    (
        "add",
        "e08d64f75992b81ef58e54ba2d4bf51df863e3636307b13ca4eea0aa2aa77236",
        ("added", "to", "every", "integer"),
        (37, 45, 86, 194, 286, 307, 330, 389, 393, 469, 490, 503),
    ),
    (
        "map",
        "478d27879ed45783338669e00ab9d3cf4ec9670bd73ee35f9c80af5715abda65",
        ("every", "integer"),
        (2, 28, 55, 59, 121, 140, 285, 292, 300, 373, 382, 387),
    ),
    (
        "map",
        "ffa0729974ba5c52670bd910332eabf2dc250dd9cbf4fdf15f40b21dff7d6fd7",
        ("each", "integer"),
        (25, 35, 81, 104, 120, 171, 199, 313, 337, 375, 474, 499),
    ),
    (
        "absolute",
        "af7971cc565cc4b27e4812c3dfd7c8c030ef524fe240ca181db7bf34a6dbfab5",
        ("absolute", "value", "of", "each"),
        (25, 51, 102, 157, 167, 230, 325, 333, 339, 413, 414, 481),
    ),
    (
        "filter",
        "8fb3306e7280b623d902ce69e03747c5c1760e3e9f0dda2ad79b3f9bdeb0726f",
        ("keep", "only"),
        (90, 114, 128, 202, 212, 269, 308, 354, 377, 400, 450, 504),
    ),
    (
        "class:ascii-lowercase",
        "7e3df3de6ced2f5dd05e4336a5c8a8cca1acdef23371d054d7cbc304dc8feffb",
        ("lowercase", "ascii", "letters"),
        (104, 115, 159, 163, 168, 207, 231, 250, 260, 366, 370, 485),
    ),
    (
        "class:english-vowels",
        "ef293a0d14a7c84e41ab268d9b56bfe680c048b6a6b554f520881ccf89d4f4bd",
        ("ascii", "vowels"),
        (15, 18, 176, 225, 233, 253, 307, 325, 363, 373, 447, 472),
    ),
    (
        "class:ascii-decimal-digits",
        "6a017054daf00dd0036af326f9e4c1ea67faae753c3e2580535c513f20ea72b5",
        ("ascii", "decimal", "digits"),
        (107, 126, 189, 270, 316, 343, 374, 384, 434, 451, 483, 506),
    ),
    (
        "sum",
        "d3a63ed53f038b87a65202bd5d21b5d02fa06b16d56059423d9b1263401dccc8",
        ("return", "their", "sum"),
        (31, 89, 98, 150, 251, 277, 293, 371, 373, 385, 497, 499),
    ),
    (
        "count",
        "af8b9568f27a201399d1028b21c0ddacbcab9758ff3484c7319297c8599e8bbe",
        ("count", "results"),
        (3, 43, 44, 70, 155, 166, 247, 251, 297, 342, 376, 492),
    ),
    (
        "less-than",
        "8d71231c9cfb399092b1f20ce4cdedf0334ed0414bcb966c1d48392d6c556ff5",
        ("strictly", "greater", "than"),
        (37, 46, 77, 150, 197, 243, 262, 264, 333, 384, 402, 507),
    ),
    (
        "reverse",
        "301fd9b2d43f09666fb9e7f5845714cf70eb80587b32b94e137c230fb34e31da",
        ("reverse", "the", "retained", "characters"),
        (66, 73, 138, 162, 163, 215, 231, 261, 294, 321, 348, 436),
    ),
)

_EXPECTED_FACTOR_COVERAGE = {
    "add": (30, 22),
    "minimum": (20, 12),
    "maximum": (30, 13),
    "absolute": (10, 1),
    "less-than": (10, 8),
    "map": (40, 21),
    "sum": (10, 7),
    "count": (10, 8),
    "filter": (20, 6),
    "reverse": (10, 3),
    "class:ascii-decimal-digits": (5, 2),
    "class:ascii-lowercase": (6, 2),
    "class:english-vowels": (9, 2),
}

_TEMPLATE_SPECS = (
    (FIT, "NEXT", 0, 1),
    (FIT, "PREV", 1, 0),
    (FIT, "NEXT", 2, 3),
    (FIT, "PREV", 3, 2),
    (HELDOUT, "PREV", 0, 3),
    (HELDOUT, "NEXT", 3, 0),
)


def _build_canonical_inventory() -> tuple[CanonicalIdentityGroup, ...]:
    rows = tuple(
        CanonicalIdentityGroup(
            prompt_input_sha256=identity,
            appearance_count=appearance_count,
            target_factors=target_factors,
            factor_set_sha256=_factor_set_sha256(tuple(dict.fromkeys(target_factors))),
        )
        for identity, appearance_count, target_factors in _CANONICAL_IDENTITY_SPECS
    )
    if (
        len(rows) != CANONICAL_SEMANTIC_IDENTITY_COUNT
        or len({row.prompt_input_sha256 for row in rows}) != len(rows)
        or sum(row.appearance_count for row in rows) != PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT
        or any(not _is_sha256(row.prompt_input_sha256) for row in rows)
    ):
        raise RuntimeError("public-TRAIN V2 canonical identity inventory changed")
    return rows


def _build_span_profiles() -> tuple[SpanFeatureProfile, ...]:
    counts: Counter[str] = Counter()
    rows: list[SpanFeatureProfile] = []
    for factor, identity, tokens, cells in _SPAN_PROFILE_SPECS:
        profile_index = counts[factor]
        counts[factor] += 1
        record = {
            "factor": factor,
            "profile_index": profile_index,
            "span_identity_sha256": identity,
            "tokens": tokens,
            "assembly_cells": cells,
        }
        rows.append(
            SpanFeatureProfile(
                factor=factor,
                profile_index=profile_index,
                span_identity_sha256=identity,
                tokens=tokens,
                assembly_cells=cells,
                sha256=_namespaced_sha256(_SPAN_PROFILE_NAMESPACE, record),
            )
        )
    if (
        set(counts) != set(ACTIVE_FACTORS)
        or any(len(row.assembly_cells) != ASSEMBLY_ACTIVE_CELL_COUNT for row in rows)
        or any(tuple(sorted(set(row.assembly_cells))) != row.assembly_cells for row in rows)
        or any(
            not all(0 <= cell < LEXICAL_CELL_COUNT for cell in row.assembly_cells) for row in rows
        )
    ):
        raise RuntimeError("public-TRAIN V2 canonical span profiles changed")
    return tuple(rows)


def _build_templates() -> tuple[RootedCounterfactualTemplate, ...]:
    rows: list[RootedCounterfactualTemplate] = []
    for template_index, (split, direction, cue_slot, selected_slot) in enumerate(_TEMPLATE_SPECS):
        rotation = (template_index * FACTOR_ROTATION_STRIDE) % len(ACTIVE_FACTORS)
        rotated = tuple(
            ACTIVE_FACTORS[(rotation + offset) % len(ACTIVE_FACTORS)]
            for offset in range(len(ACTIVE_FACTORS))
        )
        factor_groups: list[tuple[str, ...]] = []
        cursor = 0
        for size in GROUP_SIZES:
            factor_groups.append(rotated[cursor : cursor + size])
            cursor += size
        record = {
            "template_index": template_index,
            "split": split,
            "direction": direction,
            "cue_slot": cue_slot,
            "selected_slot": selected_slot,
            "rotation": rotation,
            "factor_groups": tuple(factor_groups),
        }
        rows.append(
            RootedCounterfactualTemplate(
                template_index=template_index,
                split=split,
                direction=direction,
                cue_slot=cue_slot,
                selected_slot=selected_slot,
                rotation=rotation,
                factor_groups=tuple(factor_groups),
                sha256=_namespaced_sha256(_TEMPLATE_NAMESPACE, record),
            )
        )
    return tuple(rows)


def _build_counterfactual_curriculum(
    templates: tuple[RootedCounterfactualTemplate, ...],
    profiles: tuple[SpanFeatureProfile, ...],
) -> tuple[CounterfactualContrastGroup, ...]:
    profiles_by_factor: dict[str, list[SpanFeatureProfile]] = defaultdict(list)
    for profile in profiles:
        profiles_by_factor[profile.factor].append(profile)

    groups: list[CounterfactualContrastGroup] = []
    lesson_index = 0
    group_index = 0
    for template in templates:
        for factors in template.factor_groups:
            candidate_profiles = tuple(
                profiles_by_factor[factor][
                    template.template_index % len(profiles_by_factor[factor])
                ]
                for factor in factors
            )
            lessons: list[CounterfactualFactorLesson] = []
            for target_index, (factor, target_profile) in enumerate(
                zip(factors, candidate_profiles, strict=True)
            ):
                slots: list[str | None] = [None, None, None, None]
                slots[template.cue_slot] = f"cue:{template.direction.lower()}:v2"
                slots[template.selected_slot] = f"span:{target_profile.sha256}"
                remaining_relays = sorted(
                    f"span:{profile.sha256}"
                    for profile in candidate_profiles
                    if profile.factor != factor
                )
                while len(remaining_relays) < 2:
                    remaining_relays.append("neutral:factor-workspace-v2")
                free_slots = [slot for slot, value in enumerate(slots) if value is None]
                for slot, relay in zip(free_slots, remaining_relays, strict=True):
                    slots[slot] = relay
                input_slots = tuple(slots)
                if not all(type(value) is str for value in input_slots):
                    raise RuntimeError("counterfactual four-slot input was not closed")
                typed_input_slots = (
                    str(input_slots[0]),
                    str(input_slots[1]),
                    str(input_slots[2]),
                    str(input_slots[3]),
                )
                input_record = {
                    "input_projection_sha256": manifest_v1.INPUT_PROJECTION_SHA256,
                    "selector_preflight_sha256": v5.SELECTOR_FULL_PREFLIGHT_SHA256,
                    "input_slots": typed_input_slots,
                }
                input_sha = _namespaced_sha256(_COUNTERFACTUAL_INPUT_NAMESPACE, input_record)
                lesson_record = {
                    "lesson_index": lesson_index,
                    "template_index": template.template_index,
                    "group_index": group_index,
                    "target_index": target_index,
                    "split": template.split,
                    "direction": template.direction,
                    "cue_slot": template.cue_slot,
                    "selected_slot": template.selected_slot,
                    "input_slots": typed_input_slots,
                    "input_sha256": input_sha,
                    "factor": factor,
                    "target_span_profile_sha256": target_profile.sha256,
                    "target_span_identity_sha256": target_profile.span_identity_sha256,
                    "target_assembly_cells": target_profile.assembly_cells,
                    "teacher_target_count": 1,
                }
                lessons.append(
                    CounterfactualFactorLesson(
                        lesson_index=lesson_index,
                        template_index=template.template_index,
                        group_index=group_index,
                        target_index=target_index,
                        split=template.split,
                        direction=template.direction,
                        cue_slot=template.cue_slot,
                        selected_slot=template.selected_slot,
                        input_slots=typed_input_slots,
                        input_sha256=input_sha,
                        factor=factor,
                        target_span_profile_sha256=target_profile.sha256,
                        target_span_identity_sha256=target_profile.span_identity_sha256,
                        target_assembly_cells=target_profile.assembly_cells,
                        sha256=_namespaced_sha256(
                            _COUNTERFACTUAL_LESSON_NAMESPACE,
                            lesson_record,
                        ),
                    )
                )
                lesson_index += 1
            group_record = {
                "group_index": group_index,
                "template_index": template.template_index,
                "split": template.split,
                "direction": template.direction,
                "cue_slot": template.cue_slot,
                "selected_slot": template.selected_slot,
                "factors": factors,
                "lessons": tuple(lesson.record() for lesson in lessons),
            }
            groups.append(
                CounterfactualContrastGroup(
                    group_index=group_index,
                    template_index=template.template_index,
                    split=template.split,
                    direction=template.direction,
                    cue_slot=template.cue_slot,
                    selected_slot=template.selected_slot,
                    factors=factors,
                    lessons=tuple(lessons),
                    sha256=_namespaced_sha256(
                        _COUNTERFACTUAL_GROUP_NAMESPACE,
                        group_record,
                    ),
                )
            )
            group_index += 1
    return tuple(groups)


def _validate_counterfactual_curriculum(
    groups: tuple[CounterfactualContrastGroup, ...],
) -> object:
    lessons = tuple(lesson for group in groups for lesson in group.lessons)
    fit = tuple(lesson for lesson in lessons if lesson.split == FIT)
    heldout = tuple(lesson for lesson in lessons if lesson.split == HELDOUT)
    if (
        len(groups) != COUNTERFACTUAL_GROUP_COUNT
        or len(lessons) != COUNTERFACTUAL_LESSON_COUNT
        or len(fit) != COUNTERFACTUAL_FIT_COUNT
        or len(heldout) != COUNTERFACTUAL_HELDOUT_COUNT
        or len({lesson.input_sha256 for lesson in lessons}) != len(lessons)
        or {lesson.input_sha256 for lesson in lessons}
        & {identity for identity, _count, _factors in _CANONICAL_IDENTITY_SPECS}
        or Counter(lesson.factor for lesson in fit)
        != Counter({factor: 4 for factor in ACTIVE_FACTORS})
        or Counter(lesson.factor for lesson in heldout)
        != Counter({factor: 2 for factor in ACTIVE_FACTORS})
    ):
        raise RuntimeError("public-TRAIN V2 counterfactual capacity changed")
    for group in groups:
        if (
            len(group.lessons) not in (2, 3)
            or len({tuple(sorted(lesson.input_slots)) for lesson in group.lessons}) != 1
            or {lesson.factor for lesson in group.lessons} != set(group.factors)
            or any(
                lesson.input_slots[lesson.cue_slot] != f"cue:{lesson.direction.lower()}:v2"
                for lesson in group.lessons
            )
            or any(
                lesson.input_slots[lesson.selected_slot]
                != f"span:{lesson.target_span_profile_sha256}"
                for lesson in group.lessons
            )
        ):
            raise RuntimeError("public-TRAIN V2 matched contrast geometry changed")

    fit_profiles: dict[str, set[str]] = defaultdict(set)
    fit_cells: dict[str, set[int]] = defaultdict(set)
    for lesson in fit:
        fit_profiles[lesson.factor].add(lesson.target_span_identity_sha256)
        fit_cells[lesson.factor].update(lesson.target_assembly_cells)
    for lesson in heldout:
        if (
            lesson.target_span_identity_sha256 not in fit_profiles[lesson.factor]
            or not set(lesson.target_assembly_cells) <= fit_cells[lesson.factor]
        ):
            raise RuntimeError("public-TRAIN V2 heldout physical feature support failed")

    rare_support = {
        factor: tuple(
            sorted(
                {lesson.target_span_identity_sha256 for lesson in fit if lesson.factor == factor}
            )
        )
        for factor in RARE_FIT_FACTORS
    }
    if any(len(identities) != 1 for identities in rare_support.values()):
        raise RuntimeError("public-TRAIN V2 rare-factor FIT span support changed")
    return _freeze_record(
        {
            "fit_factor_counts": dict(Counter(lesson.factor for lesson in fit)),
            "heldout_factor_counts": dict(Counter(lesson.factor for lesson in heldout)),
            "rare_factor_fit_span_identities": rare_support,
            "heldout_profile_identity_in_fit": True,
            "heldout_active_cells_subset_of_fit": True,
            "full_input_identity_disjoint": True,
            "canonical_identity_disjoint": True,
            "claim_boundary": (
                "context and selector recombination over FIT-supported canonical span profiles"
            ),
        }
    )


def _masked_control(
    index: int,
    name: str,
    tiers: tuple[tuple[str, ...], ...],
    status: str,
    publication: tuple[str, ...],
    masked: tuple[str, ...],
) -> MaskedWTAControl:
    record = {
        "control_index": index,
        "name": name,
        "principal_tiers": tiers,
        "expected_status": status,
        "expected_factor_publication": publication,
        "expected_masked_factors": masked,
    }
    return MaskedWTAControl(
        control_index=index,
        name=name,
        principal_tiers=tiers,
        expected_status=status,
        expected_factor_publication=publication,
        expected_masked_factors=masked,
        sha256=_namespaced_sha256(_CONTROL_NAMESPACE + b"masked\0", record),
    )


def _build_masked_wta_controls() -> tuple[MaskedWTAControl, ...]:
    return (
        _masked_control(0, "empty-stop", (("STOP",),), "STOP", (), ()),
        _masked_control(1, "one-then-stop", (("add",), ("STOP",)), "READY", ("add",), ("add",)),
        _masked_control(
            2,
            "same-span-two-labels",
            (("minimum",), ("maximum",), ("STOP",)),
            "READY",
            ("minimum", "maximum"),
            ("minimum", "maximum"),
        ),
        _masked_control(
            3,
            "same-span-reversed-physical-rank",
            (("maximum",), ("minimum",), ("STOP",)),
            "READY",
            ("maximum", "minimum"),
            ("maximum", "minimum"),
        ),
        _masked_control(
            4, "principal-collision-stop", (("minimum", "maximum"),), "COLLISION_STOP", (), ()
        ),
        _masked_control(
            5,
            "exact-four-then-stop",
            (("minimum",), ("maximum",), ("map",), ("sum",), ("STOP",)),
            "READY",
            ("minimum", "maximum", "map", "sum"),
            ("minimum", "maximum", "map", "sum"),
        ),
        _masked_control(
            6,
            "fifth-relay-handed-to-capacity-poison",
            (("add",), ("map",), ("count",), ("less-than",), ("sum",)),
            "FIFTH_EVENT",
            ("add", "map", "count", "less-than", "sum"),
            ("add", "map", "count", "less-than", "sum"),
        ),
        _masked_control(
            7, "reset-clears-mask", (("add",), ("STOP",)), "READY_AFTER_RESET", ("add",), ("add",)
        ),
    )


def _event(factor: str, start: int, stop: int, rank: int) -> ProtocolFactorEvent:
    record = {
        "factor": factor,
        "start_token": start,
        "stop_token": stop,
        "masked_rank": rank,
    }
    return ProtocolFactorEvent(
        factor=factor,
        start_token=start,
        stop_token=stop,
        masked_rank=rank,
        sha256=_namespaced_sha256(_EVENT_NAMESPACE, record),
    )


def _compositor_control(
    index: int,
    name: str,
    events: tuple[ProtocolFactorEvent, ...],
    *,
    reset_before: bool,
    status: str,
    public_factors: tuple[str, ...],
    pre_poison_factors: tuple[str, ...] = (),
) -> FactorSetCompositorControl:
    factor_set_sha = (
        _factor_set_sha256(tuple(dict.fromkeys(public_factors))) if public_factors else None
    )
    publication_sha = _namespaced_sha256(
        _ORDERED_PUBLICATION_NAMESPACE,
        tuple(event.record() for event in events),
    )
    record = {
        "control_index": index,
        "name": name,
        "events": tuple(event.record() for event in events),
        "reset_before": reset_before,
        "expected_status": status,
        "expected_public_factors": public_factors,
        "expected_pre_poison_factors": pre_poison_factors,
        "expected_factor_set_sha256": factor_set_sha,
        "ordered_publication_sha256": publication_sha,
    }
    return FactorSetCompositorControl(
        control_index=index,
        name=name,
        events=events,
        reset_before=reset_before,
        expected_status=status,
        expected_public_factors=public_factors,
        expected_pre_poison_factors=pre_poison_factors,
        expected_factor_set_sha256=factor_set_sha,
        ordered_publication_sha256=publication_sha,
        sha256=_namespaced_sha256(_CONTROL_NAMESPACE + b"compositor\0", record),
    )


def _build_compositor_controls() -> tuple[FactorSetCompositorControl, ...]:
    canonical_four = (
        _event("minimum", 6, 10, 0),
        _event("maximum", 6, 10, 1),
        _event("map", 7, 9, 0),
        _event("sum", 15, 18, 0),
    )
    permuted_four = (
        canonical_four[3],
        canonical_four[2],
        canonical_four[1],
        canonical_four[0],
    )
    return (
        _compositor_control(
            0, "zero", (), reset_before=True, status="STOP_ZERO", public_factors=()
        ),
        _compositor_control(
            1,
            "one",
            (_event("add", 8, 12, 0),),
            reset_before=True,
            status="READY_FACTOR_SET",
            public_factors=("add",),
        ),
        _compositor_control(
            2,
            "same-span-different-factor-overlap",
            (_event("minimum", 8, 12, 0), _event("maximum", 8, 12, 1)),
            reset_before=True,
            status="READY_FACTOR_SET",
            public_factors=("minimum", "maximum"),
        ),
        _compositor_control(
            3,
            "nested-exact-four-prompt-order",
            canonical_four,
            reset_before=True,
            status="READY_FACTOR_SET",
            public_factors=("minimum", "maximum", "map", "sum"),
        ),
        _compositor_control(
            4,
            "nested-exact-four-permuted-order",
            permuted_four,
            reset_before=True,
            status="READY_FACTOR_SET",
            public_factors=("sum", "map", "maximum", "minimum"),
        ),
        _compositor_control(
            5,
            "same-factor-duplicate",
            (_event("add", 6, 10, 0), _event("add", 12, 16, 0)),
            reset_before=True,
            status="STOP_DUPLICATE_FACTOR",
            public_factors=(),
            pre_poison_factors=("add",),
        ),
        _compositor_control(
            6,
            "fifth-distinct-factor",
            (
                _event("add", 1, 2, 0),
                _event("map", 2, 3, 0),
                _event("count", 3, 4, 0),
                _event("less-than", 4, 5, 0),
                _event("sum", 5, 6, 0),
            ),
            reset_before=True,
            status="STOP_EVENT_CAPACITY_4",
            public_factors=(),
            pre_poison_factors=("add", "map", "count", "less-than"),
        ),
        _compositor_control(
            7,
            "post-poison-reset-exact-four",
            canonical_four,
            reset_before=True,
            status="READY_FACTOR_SET",
            public_factors=("minimum", "maximum", "map", "sum"),
        ),
    )


def _authenticate_dependencies() -> None:
    if (
        manifest_v1.INPUT_PROJECTION_SHA256
        != "2108f4864cfbbcd3f44be96ad7b55d95a89d8d4899c824ec913204b5318320d6"
        or manifest_v1.FACTOR_REGISTRY_SHA256
        != "f7df546c7c89c20d48c79c737dab8fcb101d9c861cb041c0c39654268ecb28c6"
        or manifest_v1.EXPECTED_LESSON_MANIFEST_SHA256
        != "139e678a6001f5264d8e6ba44609e2b97082cb8b7c443462326c370d4fc5e774"
        or manifest_v1.EXPECTED_PREFLIGHT_PAYLOAD_SHA256
        != "c4dd1d14b57c9359691f8077742f41b2fe4277e6cca414b5b2f60f94871459cd"
        or v5.PROTOCOL.protocol_sha256
        != "587bd68a293ca2116570cf4e5ee7a860bbdd9928138982ae43748c4f87a252c1"
        or v5.SELECTOR_FULL_PREFLIGHT_SHA256
        != "b4d6cceae873edcd6e886ddb7f41a01e7ed659f6929029dcf6f77828c33ae073"
        or v5.V4_WTA_REFERENCE_SHA256
        != "fdd313d438e26a64769e7e8889260673d7703d9d870006cf62b326af3d741404"
    ):
        raise RuntimeError("public-TRAIN V2 dependency root changed")


_authenticate_dependencies()

DEPENDENCY_RECORD = _freeze_record(
    {
        "prompt_substrate_commit": PROMPT_SUBSTRATE_COMMIT,
        "prompt_substrate_source_sha256": PROMPT_SUBSTRATE_SOURCE_SHA256,
        "prompt_manifest_commit": PROMPT_MANIFEST_COMMIT,
        "prompt_manifest_source_sha256": PROMPT_MANIFEST_SOURCE_SHA256,
        "prompt_input_projection_sha256": manifest_v1.INPUT_PROJECTION_SHA256,
        "prompt_factor_registry_sha256": manifest_v1.FACTOR_REGISTRY_SHA256,
        "prompt_capsule_raw_sha256": manifest_v1.CAPSULE_RAW_SHA256,
        "prompt_capsule_payload_sha256": manifest_v1.CAPSULE_PAYLOAD_SHA256,
        "prompt_lesson_manifest_sha256": manifest_v1.EXPECTED_LESSON_MANIFEST_SHA256,
        "prompt_preflight_payload_sha256": manifest_v1.EXPECTED_PREFLIGHT_PAYLOAD_SHA256,
        "v5_protocol_commit": V5_PROTOCOL_COMMIT,
        "v5_protocol_source_sha256": V5_PROTOCOL_SOURCE_SHA256,
        "v5_protocol_sha256": v5.PROTOCOL.protocol_sha256,
        "v5_learner_commit": V5_LEARNER_COMMIT,
        "v5_learner_source_sha256": V5_LEARNER_SOURCE_SHA256,
        "v5_selector_full_preflight_sha256": v5.SELECTOR_FULL_PREFLIGHT_SHA256,
        "v5_wta_reference_sha256": v5.V4_WTA_REFERENCE_SHA256,
        "dependency_access_scope": "public-TRAIN prompt/signature and self-posed V5 only",
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(
    _DEPENDENCY_NAMESPACE,
    dict(DEPENDENCY_RECORD),
)

MASKED_WTA_TOPOLOGY_RECORD = _freeze_record(
    {
        "factor_registry_sha256": manifest_v1.FACTOR_REGISTRY_SHA256,
        "v5_wta_reference_sha256": v5.V4_WTA_REFERENCE_SHA256,
        "v5_selector_full_preflight_sha256": v5.SELECTOR_FULL_PREFLIGHT_SHA256,
        "principal_factor_cells": FACTOR_COUNT,
        "principal_stop_cells": 1,
        "principal_count": MASKED_WTA_PRINCIPAL_COUNT,
        "tonic_stop_principal_index": MASKED_WTA_STOP_INDEX,
        "mask_storage_cells": FACTOR_COUNT,
        "mask_readout_cells": FACTOR_COUNT,
        "basket_cells": 1,
        "collision_cells": 1,
        "delayed_factor_relays": FACTOR_COUNT,
        "delayed_stop_relays": 1,
        "reset_cells": 1,
        "maximum_factor_events_before_stop_or_poison": EVENT_CAPACITY + 1,
        "unique_factor_rule": (
            "publish one delayed factor relay, set its persistent mask, and repeat the "
            "same span competition"
        ),
        "masked_factor_rule": (
            "a persisted factor mask inhibits that principal on every later race for the same span"
        ),
        "span_stop_rule": "the unique tonic STOP relay closes the current span",
        "collision_rule": "any simultaneous first tier closes the span without an event",
        "same_span_multi_label_supported": True,
        "ordered_publication_semantics": "physical-audit-only",
        "full_circuit_execution_deferred": True,
    }
)
MASKED_WTA_TOPOLOGY_SHA256 = _namespaced_sha256(
    _MASKED_WTA_TOPOLOGY_NAMESPACE,
    dict(MASKED_WTA_TOPOLOGY_RECORD),
)

COMPOSITOR_TOPOLOGY_RECORD = _freeze_record(
    {
        "factor_registry_sha256": manifest_v1.FACTOR_REGISTRY_SHA256,
        "event_capacity": EVENT_CAPACITY,
        "ingress_factor_cells": FACTOR_COUNT,
        "occupancy_storage_cells": FACTOR_COUNT,
        "occupancy_readout_cells": FACTOR_COUNT,
        "pre_event_occupancy_snapshot_cells": FACTOR_COUNT,
        "duplicate_factor_coincidence_cells": FACTOR_COUNT,
        "cardinality_storage_cells": CARDINALITY_CELL_COUNT,
        "cardinality_readout_cells": CARDINALITY_CELL_COUNT,
        "cardinality_transition_cells": EVENT_CAPACITY,
        "ordered_publication_slots": EVENT_CAPACITY,
        "duplicate_poison_cells": 1,
        "capacity_poison_cells": 1,
        "overlap_detector_cells": 0,
        "overlap_poison_cells": 0,
        "candidate_admission_present_absent_equality_cells": FACTOR_COUNT * 2,
        "candidate_admission_cardinality_equality_cells": CARDINALITY_CELL_COUNT,
        "candidate_admission_all_match_cells": 1,
        "different_factor_overlap_and_nesting_allowed": True,
        "same_factor_duplicate_rule": (
            "incoming factor plus its occupied snapshot sets duplicate poison and performs no write"
        ),
        "fifth_distinct_factor_rule": (
            "a novel factor arriving while cardinality four sets capacity poison and "
            "performs no write"
        ),
        "poison_output_rule": "poison exposes no partial factor set",
        "factor_set_admission_rule": (
            "physical 24-cell occupancy and one-hot cardinality must equal the "
            "compiler-owned rooted candidate factor set"
        ),
        "ordered_publication_rule": (
            "retain start, stop, masked rank, and factor relay in ingress order for audit"
        ),
        "factor_set_semantics_ignore_ingress_order": True,
        "host_order_sort": False,
        "host_set_union": False,
        "host_span_filter": False,
        "full_circuit_execution_deferred": True,
    }
)
COMPOSITOR_TOPOLOGY_SHA256 = _namespaced_sha256(
    _COMPOSITOR_TOPOLOGY_NAMESPACE,
    dict(COMPOSITOR_TOPOLOGY_RECORD),
)

SURFACE_BOUNDARY_RECORD = _freeze_record(
    {
        "public_train_prompt_appearance_count": PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
        "canonical_semantic_identity_count": CANONICAL_SEMANTIC_IDENTITY_COUNT,
        "raw_public_prompt_text_count": 0,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_count": DESIRED_OUTPUT_COUNT,
        "program_artifact_count": PROGRAM_ARTIFACT_COUNT,
        "property_row_count": PROPERTY_ROW_COUNT,
        "family_identifier_count": FAMILY_IDENTIFIER_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "learner_or_inference_interface_present": False,
        "execution_deferred_to_separately_named_learner": True,
    }
)
SURFACE_BOUNDARY_SHA256 = _namespaced_sha256(
    _SURFACE_BOUNDARY_NAMESPACE,
    dict(SURFACE_BOUNDARY_RECORD),
)


def _build_protocol() -> tuple[
    FrozenPublicTrainFactorWorkspaceV2Protocol,
    object,
    str,
    object,
    str,
    object,
    str,
    object,
    str,
]:
    canonical = _build_canonical_inventory()
    canonical_ids = tuple(row.prompt_input_sha256 for row in canonical)
    heldout_set = set(CANONICAL_HELDOUT_PROMPT_INPUT_SHA256S)
    canonical_fit = tuple(identity for identity in canonical_ids if identity not in heldout_set)
    if (
        len(heldout_set) != CANONICAL_HELDOUT_IDENTITY_COUNT
        or not heldout_set <= set(canonical_ids)
        or len(canonical_fit) != CANONICAL_FIT_IDENTITY_COUNT
    ):
        raise RuntimeError("public-TRAIN V2 canonical 24/18 split changed")

    appearance_counts: Counter[str] = Counter()
    distinct_counts: Counter[str] = Counter()
    for row in canonical:
        for factor in set(row.target_factors):
            appearance_counts[factor] += row.appearance_count
            distinct_counts[factor] += 1
    coverage = {
        factor: {
            "appearance_count": appearance_counts[factor],
            "distinct_identity_count": distinct_counts[factor],
        }
        for factor in ACTIVE_FACTORS
    }
    if {
        factor: (row["appearance_count"], row["distinct_identity_count"])
        for factor, row in coverage.items()
    } != _EXPECTED_FACTOR_COVERAGE:
        raise RuntimeError("public-TRAIN V2 active-factor coverage changed")
    coverage_record = _freeze_record(
        {
            "active_factors": ACTIVE_FACTORS,
            "dormant_factors": DORMANT_FACTORS,
            "coverage": coverage,
        }
    )
    coverage_sha = _namespaced_sha256(
        _CANONICAL_INVENTORY_NAMESPACE + b"coverage\0",
        {
            "active_factors": ACTIVE_FACTORS,
            "dormant_factors": DORMANT_FACTORS,
            "coverage": coverage,
        },
    )

    canonical_inventory_sha = _namespaced_sha256(
        _CANONICAL_INVENTORY_NAMESPACE,
        tuple(row.record() for row in canonical),
    )
    split_record = _freeze_record(
        {
            "canonical_inventory_sha256": canonical_inventory_sha,
            "fit_prompt_input_sha256s": canonical_fit,
            "heldout_prompt_input_sha256s": CANONICAL_HELDOUT_PROMPT_INPUT_SHA256S,
            "identity_not_appearance_split": True,
            "multiplicity_is_audit_only": True,
            "final_refit_identity_order": canonical_ids,
            "final_refit_only_after_initial_frozen_gate": True,
            "final_refit_is_not_heldout_evidence": True,
        }
    )
    split_sha = _namespaced_sha256(_CANONICAL_SPLIT_NAMESPACE, dict(split_record))

    profiles = _build_span_profiles()
    profiles_sha = _namespaced_sha256(
        _SPAN_PROFILE_NAMESPACE + b"registry\0",
        tuple(profile.record(include_sha256=False) for profile in profiles),
    )
    templates = _build_templates()
    groups = _build_counterfactual_curriculum(templates, profiles)
    rare_support_record = _validate_counterfactual_curriculum(groups)
    rare_support_sha = _namespaced_sha256(
        _COUNTERFACTUAL_CURRICULUM_NAMESPACE + b"support\0",
        dict(rare_support_record),
    )
    curriculum_sha = _namespaced_sha256(
        _COUNTERFACTUAL_CURRICULUM_NAMESPACE,
        {
            "templates": tuple(template.record() for template in templates),
            "groups": tuple(group.record() for group in groups),
        },
    )

    masked_controls = _build_masked_wta_controls()
    masked_controls_sha = _namespaced_sha256(
        _CONTROL_NAMESPACE + b"masked-panel\0",
        tuple(control.record(include_sha256=False) for control in masked_controls),
    )
    compositor_controls = _build_compositor_controls()
    compositor_controls_sha = _namespaced_sha256(
        _CONTROL_NAMESPACE + b"compositor-panel\0",
        tuple(control.record(include_sha256=False) for control in compositor_controls),
    )
    canonical_order = compositor_controls[3]
    permuted_order = compositor_controls[4]
    if (
        canonical_order.expected_factor_set_sha256 != permuted_order.expected_factor_set_sha256
        or canonical_order.ordered_publication_sha256 == permuted_order.ordered_publication_sha256
    ):
        raise RuntimeError("public-TRAIN V2 factor-set permutation invariant failed")

    electrical_record = _freeze_record(
        {
            "dependency_sha256": DEPENDENCY_SHA256,
            "masked_wta_topology_sha256": MASKED_WTA_TOPOLOGY_SHA256,
            "masked_wta_controls_sha256": masked_controls_sha,
            "compositor_topology_sha256": COMPOSITOR_TOPOLOGY_SHA256,
            "compositor_controls_sha256": compositor_controls_sha,
            "execution_order": (
                "authenticate dependencies, canonical inventory, split, and curriculum",
                "execute the complete inherited V5 selector and factor-WTA controls",
                "execute all repeated-masked-WTA controls",
                "execute all capacity-four factor-set compositor controls",
                "authenticate every observed control root",
                "allocate learner arrays only after every control passes",
            ),
            "required_masked_wta_control_count": len(masked_controls),
            "required_compositor_control_count": len(compositor_controls),
            "controls_before_learner_state_allocation": True,
            "execution_deferred": True,
            "learner_state_constructed": False,
        }
    )
    electrical_sha = _namespaced_sha256(
        _ELECTRICAL_PREFLIGHT_NAMESPACE,
        dict(electrical_record),
    )

    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": PROTOCOL_STATUS,
        "dependency_sha256": DEPENDENCY_SHA256,
        "canonical_inventory_sha256": canonical_inventory_sha,
        "canonical_split_sha256": split_sha,
        "active_factor_coverage_sha256": coverage_sha,
        "span_feature_profiles_sha256": profiles_sha,
        "counterfactual_curriculum_sha256": curriculum_sha,
        "rare_factor_support_sha256": rare_support_sha,
        "masked_wta_topology_sha256": MASKED_WTA_TOPOLOGY_SHA256,
        "masked_wta_controls_sha256": masked_controls_sha,
        "compositor_topology_sha256": COMPOSITOR_TOPOLOGY_SHA256,
        "compositor_controls_sha256": compositor_controls_sha,
        "electrical_preflight_contract_sha256": electrical_sha,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "public_train_prompt_appearance_count": PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
        "canonical_semantic_identity_count": len(canonical),
        "canonical_fit_identity_count": len(canonical_fit),
        "canonical_heldout_identity_count": len(CANONICAL_HELDOUT_PROMPT_INPUT_SHA256S),
        "active_factor_count": len(ACTIVE_FACTORS),
        "rooted_template_count": len(templates),
        "counterfactual_group_count": len(groups),
        "counterfactual_lesson_count": sum(len(group.lessons) for group in groups),
        "counterfactual_fit_count": COUNTERFACTUAL_FIT_COUNT,
        "counterfactual_heldout_count": COUNTERFACTUAL_HELDOUT_COUNT,
        "event_capacity": EVENT_CAPACITY,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "learner_state_constructed": False,
        "learning_executed": False,
        "electrical_controls_executed": False,
        "execution_surface_present": False,
    }
    protocol = FrozenPublicTrainFactorWorkspaceV2Protocol(
        scope=PROTOCOL_SCOPE,
        status=PROTOCOL_STATUS,
        canonical_identities=canonical,
        canonical_fit_prompt_input_sha256s=canonical_fit,
        canonical_heldout_prompt_input_sha256s=CANONICAL_HELDOUT_PROMPT_INPUT_SHA256S,
        span_feature_profiles=profiles,
        rooted_templates=templates,
        counterfactual_groups=groups,
        masked_wta_controls=masked_controls,
        compositor_controls=compositor_controls,
        dependency_sha256=DEPENDENCY_SHA256,
        canonical_inventory_sha256=canonical_inventory_sha,
        canonical_split_sha256=split_sha,
        active_factor_coverage_sha256=coverage_sha,
        span_feature_profiles_sha256=profiles_sha,
        counterfactual_curriculum_sha256=curriculum_sha,
        rare_factor_support_sha256=rare_support_sha,
        masked_wta_topology_sha256=MASKED_WTA_TOPOLOGY_SHA256,
        masked_wta_controls_sha256=masked_controls_sha,
        compositor_topology_sha256=COMPOSITOR_TOPOLOGY_SHA256,
        compositor_controls_sha256=compositor_controls_sha,
        electrical_preflight_contract_sha256=electrical_sha,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        protocol_sha256=_namespaced_sha256(_PROTOCOL_NAMESPACE, root_without_protocol),
    )
    return (
        protocol,
        coverage_record,
        coverage_sha,
        split_record,
        split_sha,
        rare_support_record,
        rare_support_sha,
        electrical_record,
        electrical_sha,
    )


(
    PROTOCOL,
    ACTIVE_FACTOR_COVERAGE_RECORD,
    ACTIVE_FACTOR_COVERAGE_SHA256,
    CANONICAL_SPLIT_RECORD,
    CANONICAL_SPLIT_SHA256,
    RARE_FACTOR_SUPPORT_RECORD,
    RARE_FACTOR_SUPPORT_SHA256,
    ELECTRICAL_PREFLIGHT_CONTRACT_RECORD,
    ELECTRICAL_PREFLIGHT_CONTRACT_SHA256,
) = _build_protocol()


__all__ = [
    "ACTIVE_FACTORS",
    "ACTIVE_FACTOR_COVERAGE_RECORD",
    "ACTIVE_FACTOR_COVERAGE_SHA256",
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "CANONICAL_FIT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_PROMPT_INPUT_SHA256S",
    "CANONICAL_SEMANTIC_IDENTITY_COUNT",
    "CANONICAL_SPLIT_RECORD",
    "CANONICAL_SPLIT_SHA256",
    "CARDINALITY_CELL_COUNT",
    "COMPOSITOR_TOPOLOGY_RECORD",
    "COMPOSITOR_TOPOLOGY_SHA256",
    "COUNTERFACTUAL_FIT_COUNT",
    "COUNTERFACTUAL_GROUP_COUNT",
    "COUNTERFACTUAL_HELDOUT_COUNT",
    "COUNTERFACTUAL_LESSON_COUNT",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "DORMANT_FACTORS",
    "ELECTRICAL_PREFLIGHT_CONTRACT_RECORD",
    "ELECTRICAL_PREFLIGHT_CONTRACT_SHA256",
    "EVENT_CAPACITY",
    "FACTOR_COUNT",
    "FACTOR_REGISTRY",
    "FIT",
    "GROUP_SIZES",
    "HELDOUT",
    "MASKED_WTA_PRINCIPAL_COUNT",
    "MASKED_WTA_STOP_INDEX",
    "MASKED_WTA_TOPOLOGY_RECORD",
    "MASKED_WTA_TOPOLOGY_SHA256",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PROTOCOL_STATUS",
    "PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT",
    "RARE_FACTOR_SUPPORT_RECORD",
    "RARE_FACTOR_SUPPORT_SHA256",
    "RARE_FIT_FACTORS",
    "ROOTED_TEMPLATE_COUNT",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "CanonicalIdentityGroup",
    "CounterfactualContrastGroup",
    "CounterfactualFactorLesson",
    "FactorSetCompositorControl",
    "FrozenPublicTrainFactorWorkspaceV2Protocol",
    "MaskedWTAControl",
    "ProtocolFactorEvent",
    "RootedCounterfactualTemplate",
    "SpanFeatureProfile",
]
