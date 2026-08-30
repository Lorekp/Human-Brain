"""Frozen public-TRAIN structural annotation capsule V1.

This data-only capsule projects the compiler-owned public-TRAIN semantic
teachers into the eight generic relay channels frozen by
``PUBLIC_TRAIN_PROMPT_SELECTION_FREEZE_V1_PROTOCOL``.  Learner-facing rows
contain only a physical causal key and abstract relay labels.  Full signatures,
role-normalized prompt-input identities, and compiler source identities remain
authentication metadata and never become neural evidence.

The module hashes, but never imports or executes, the pinned task-generator
source.  It does not construct tasks, desired rows, property rows, executable
programs, source, ASTs, candidates, proofs, learners, or electrical executions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.r3_pilot import public_train_prompt_factor_manifest_v1 as prompt_manifest
from experiments.r3_pilot import public_train_prompt_selection_freeze_v1_protocol as selection_v1

PROTOCOL_SCOPE = "PUBLIC_TRAIN_STRUCTURAL_ANNOTATION_CAPSULE_V1"
PASS_IDENTIFIABLE = "PASS_PUBLIC_TRAIN_STRUCTURAL_ANNOTATION_CAPSULE_V1_IDENTIFIABLE"
STOP_TOTAL_FUNCTION = "STOP_PUBLIC_TRAIN_STRUCTURAL_ANNOTATION_CAPSULE_V1_CAUSAL_KEY_TOTAL_FUNCTION"
STOP_CAUSAL_KEY_TARGET_ALIAS = (
    "STOP_PUBLIC_TRAIN_STRUCTURAL_ANNOTATION_CAPSULE_V1_CAUSAL_KEY_TARGET_ALIAS"
)

PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT = 90
CANONICAL_CAUSAL_IDENTITY_COUNT = 42
CANONICAL_FIT_IDENTITY_COUNT = 24
CANONICAL_HELDOUT_IDENTITY_COUNT = 18
COMPILER_TEACHER_PROJECTION_COUNT = 13

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_READ_COUNT = 0
PROGRAM_ARTIFACT_EXPORT_COUNT = 0
EXECUTABLE_SOURCE_EXPORT_COUNT = 0
AST_ARTIFACT_EXPORT_COUNT = 0
PROPERTY_ROW_READ_COUNT = 0
TASK_IDENTIFIER_EXPORT_COUNT = 0
FAMILY_IDENTIFIER_EXPORT_COUNT = 0
ENTRYPOINT_SPELLING_NEURAL_EVIDENCE_COUNT = 0
AUDIT_HASH_NEURAL_EVIDENCE_COUNT = 0
CANDIDATE_RANKING_COUNT = 0
LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
CANDIDATE_COUNT = 0
PROOF_COUNT = 0

EXPECTED_SELECTION_PROTOCOL_SHA256 = (
    "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"
)
COMPILER_TEACHER_SOURCE_COMMIT = "0121fd279b3dd9dda59776a31b9fe7a12f774217"
COMPILER_TEACHER_SOURCE_PATH = "src/humanbrain/simple_swe/tasks.py"
COMPILER_TEACHER_SOURCE_SHA256 = "b0edfc2eedc25335bcb2318a7b13026f6ab7affc29643af76e16f4c6b57240c3"

_ROOT = Path(__file__).resolve().parents[2]
_RELAY_TARGET_NAMESPACE = b"public-train-structural-annotation-v1-relay-target\0"
_TARGET_SET_NAMESPACE = b"public-train-structural-annotation-v1-target-set\0"
_ANNOTATION_NAMESPACE = b"public-train-structural-annotation-v1-annotation\0"
_PROJECTION_NAMESPACE = b"public-train-structural-annotation-v1-projection\0"
_AUDIT_NAMESPACE = b"public-train-structural-annotation-v1-audit\0"
_DEPENDENCY_NAMESPACE = b"public-train-structural-annotation-v1-dependency\0"
_SURFACE_NAMESPACE = b"public-train-structural-annotation-v1-surface\0"
_CAPSULE_NAMESPACE = b"public-train-structural-annotation-v1-capsule\0"


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


@dataclass(frozen=True, slots=True)
class AbstractRelayLabel:
    channel: str
    fields: object
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "channel": self.channel,
            "fields": self.fields,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class StructuralAnnotation:
    """One learner-facing total-function row."""

    causal_key_sha256: str
    relay_labels: tuple[AbstractRelayLabel, ...]
    target_set_sha256: str
    sha256: str

    def learner_record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_key_sha256": self.causal_key_sha256,
            "relay_labels": tuple(label.record() for label in self.relay_labels),
            "target_set_sha256": self.target_set_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class StructuralIdentifiabilityAudit:
    status: str
    prompt_appearance_count: int
    duplicate_prompt_appearance_count: int
    canonical_causal_identity_count: int
    exact_target_set_count: int
    missing_causal_key_count: int
    extra_causal_key_count: int
    appearance_target_mismatch_count: int
    causal_key_target_collision_count: int
    relay_channel_counts: tuple[tuple[str, int], ...]
    total_function_completed: bool
    collision_audit_completed: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "prompt_appearance_count": self.prompt_appearance_count,
            "duplicate_prompt_appearance_count": self.duplicate_prompt_appearance_count,
            "canonical_causal_identity_count": self.canonical_causal_identity_count,
            "exact_target_set_count": self.exact_target_set_count,
            "missing_causal_key_count": self.missing_causal_key_count,
            "extra_causal_key_count": self.extra_causal_key_count,
            "appearance_target_mismatch_count": self.appearance_target_mismatch_count,
            "causal_key_target_collision_count": self.causal_key_target_collision_count,
            "relay_channel_counts": self.relay_channel_counts,
            "identical_causal_key_rule": (
                "one physical causal key must map to one exact relay target set"
            ),
            "total_function_completed": self.total_function_completed,
            "collision_audit_completed": self.collision_audit_completed,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainStructuralAnnotationCapsuleV1:
    scope: str
    status: str
    annotations: tuple[StructuralAnnotation, ...]
    identifiability_audit: StructuralIdentifiabilityAudit
    dependency_sha256: str
    teacher_projection_sha256: str
    identifiability_audit_sha256: str
    surface_boundary_sha256: str
    capsule_sha256: str

    def root_record(self, *, include_capsule_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "teacher_projection_sha256": self.teacher_projection_sha256,
            "identifiability_audit_sha256": self.identifiability_audit_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "prompt_appearance_count": self.identifiability_audit.prompt_appearance_count,
            "canonical_causal_identity_count": len(self.annotations),
            "exact_target_set_count": len(self.annotations),
            "relay_label_count": sum(len(row.relay_labels) for row in self.annotations),
            "learner_array_count": LEARNER_ARRAY_COUNT,
            "learning_run_count": LEARNING_RUN_COUNT,
            "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "proof_count": PROOF_COUNT,
            "learner_or_execution_interface_present": False,
        }
        if include_capsule_sha256:
            record["capsule_sha256"] = self.capsule_sha256
        return record


_CHANNELS = tuple(channel.channel for channel in selection_v1.RELAY_TARGET_VOCABULARY)
_CHANNEL_ORDER = {channel: index for index, channel in enumerate(_CHANNELS)}
_REQUIRED_FIELDS = {
    channel.channel: frozenset(channel.required_fields)
    for channel in selection_v1.RELAY_TARGET_VOCABULARY
}


def _relay(channel: str, **fields: object) -> AbstractRelayLabel:
    required = _REQUIRED_FIELDS.get(channel)
    if required is None or set(fields) != required:
        raise RuntimeError(f"relay channel {channel!r} fields changed")
    record = {"channel": channel, "fields": fields}
    return AbstractRelayLabel(
        channel=channel,
        fields=_freeze_record(fields),
        sha256=_namespaced_sha256(_RELAY_TARGET_NAMESPACE, record),
    )


def _root(production: str, family: str, input_type: str, output_type: str) -> AbstractRelayLabel:
    return _relay(
        "root-production-family",
        root_production=production,
        root_family=family,
        input_type=input_type,
        output_type=output_type,
    )


def _edge(parent: str, ordinal: int, child: str) -> AbstractRelayLabel:
    return _relay(
        "ordered-parent-child-edge",
        parent_slot=parent,
        child_ordinal=ordinal,
        child_slot=child,
    )


def _orientation(operator: str, ordinal: int, source_role: str) -> AbstractRelayLabel:
    return _relay(
        "argument-operand-orientation",
        operator_slot=operator,
        operand_ordinal=ordinal,
        source_role=source_role,
    )


def _nesting(outer: str, inner: str, depth: int) -> AbstractRelayLabel:
    return _relay(
        "nesting-relation",
        outer_slot=outer,
        inner_slot=inner,
        nesting_depth=depth,
    )


def _parameter(semantic_slot: str) -> AbstractRelayLabel:
    return _relay(
        "parameter-role-binding",
        signature_parameter_role="ARG0",
        semantic_argument_slot=semantic_slot,
    )


def _literal_slot(source: int | str, slot: str) -> AbstractRelayLabel:
    return _relay(
        "literal-slot-binding",
        literal_source_ordinal=source,
        semantic_literal_slot=slot,
    )


def _literal_copy(slot: str, role: str) -> AbstractRelayLabel:
    return _relay(
        "literal-value-copy",
        semantic_literal_slot=slot,
        typed_output_copy_role=role,
    )


def _agreement() -> AbstractRelayLabel:
    return _relay(
        "terminal-root-eclass-agreement",
        root_handle_role="prompt-selected-root-production",
        eclass_handle_role="authoritative-retained-root-eclass",
        exact_identity_required=True,
    )


def _target_set(*labels: AbstractRelayLabel) -> tuple[AbstractRelayLabel, ...]:
    ordered = tuple(
        sorted(
            labels,
            key=lambda label: (
                _CHANNEL_ORDER[label.channel],
                _canonical_json(label.fields),
            ),
        )
    )
    if len({label.sha256 for label in ordered}) != len(ordered):
        raise RuntimeError("structural teacher emitted a duplicate relay label")
    return ordered


def _scalar_clamp() -> tuple[AbstractRelayLabel, ...]:
    return _target_set(
        _root("minimum", "scalar-extremum", "Int", "Int"),
        _edge("root", 0, "lower-extremum"),
        _edge("root", 1, "upper-literal"),
        _edge("lower-extremum", 0, "parameter-arg0"),
        _edge("lower-extremum", 1, "lower-literal"),
        _orientation("root", 0, "nested-maximum"),
        _orientation("root", 1, "upper-bound-literal"),
        _orientation("lower-extremum", 0, "signature-parameter-ARG0"),
        _orientation("lower-extremum", 1, "lower-bound-literal"),
        _nesting("root", "lower-extremum", 1),
        _parameter("parameter-arg0"),
        _literal_slot(0, "lower-literal"),
        _literal_copy("lower-literal", "copy-typed-int-prompt-literal-0"),
        _literal_slot(1, "upper-literal"),
        _literal_copy("upper-literal", "copy-typed-int-prompt-literal-1"),
        _agreement(),
    )


def _scalar_add() -> tuple[AbstractRelayLabel, ...]:
    return _target_set(
        _root("add", "scalar-additive", "Int", "Int"),
        _edge("root", 0, "parameter-arg0"),
        _edge("root", 1, "offset-literal"),
        _orientation("root", 0, "signature-parameter-ARG0"),
        _orientation("root", 1, "offset-literal"),
        _parameter("parameter-arg0"),
        _literal_slot(0, "offset-literal"),
        _literal_copy("offset-literal", "copy-typed-int-prompt-literal-0"),
        _agreement(),
    )


def _scalar_nonnegative() -> tuple[AbstractRelayLabel, ...]:
    return _target_set(
        _root("maximum", "scalar-extremum", "Int", "Int"),
        _edge("root", 0, "parameter-arg0"),
        _edge("root", 1, "implicit-zero"),
        _orientation("root", 0, "signature-parameter-ARG0"),
        _orientation("root", 1, "compiler-constant-zero"),
        _parameter("parameter-arg0"),
        _literal_slot("compiler-constant-zero", "implicit-zero"),
        _literal_copy("implicit-zero", "emit-typed-int-zero"),
        _agreement(),
    )


def _list_map_add() -> tuple[AbstractRelayLabel, ...]:
    return _target_set(
        _root("map", "list-map", "List[Int]", "List[Int]"),
        _edge("root", 0, "value-binder"),
        _edge("root", 1, "collection-arg0"),
        _edge("value-binder", 0, "bound-add"),
        _edge("bound-add", 0, "bound-item0"),
        _edge("bound-add", 1, "offset-literal"),
        _orientation("root", 0, "value-binder"),
        _orientation("root", 1, "signature-parameter-ARG0"),
        _orientation("bound-add", 0, "bound-item0"),
        _orientation("bound-add", 1, "offset-literal"),
        _nesting("root", "value-binder", 1),
        _nesting("root", "bound-add", 2),
        _parameter("collection-arg0"),
        _literal_slot(0, "offset-literal"),
        _literal_copy("offset-literal", "copy-typed-int-prompt-literal-0"),
        _agreement(),
    )


def _list_map_absolute() -> tuple[AbstractRelayLabel, ...]:
    return _target_set(
        _root("map", "list-map", "List[Int]", "List[Int]"),
        _edge("root", 0, "value-binder"),
        _edge("root", 1, "collection-arg0"),
        _edge("value-binder", 0, "bound-absolute"),
        _edge("bound-absolute", 0, "bound-item0"),
        _orientation("root", 0, "value-binder"),
        _orientation("root", 1, "signature-parameter-ARG0"),
        _nesting("root", "value-binder", 1),
        _nesting("root", "bound-absolute", 2),
        _parameter("collection-arg0"),
        _agreement(),
    )


def _clamp_sum() -> tuple[AbstractRelayLabel, ...]:
    return _target_set(
        _root("sum", "list-fold", "List[Int]", "Int"),
        _edge("root", 0, "mapped-clamp"),
        _edge("mapped-clamp", 0, "value-binder"),
        _edge("mapped-clamp", 1, "collection-arg0"),
        _edge("value-binder", 0, "upper-extremum"),
        _edge("upper-extremum", 0, "lower-extremum"),
        _edge("upper-extremum", 1, "upper-literal"),
        _edge("lower-extremum", 0, "bound-item0"),
        _edge("lower-extremum", 1, "lower-literal"),
        _orientation("mapped-clamp", 0, "value-binder"),
        _orientation("mapped-clamp", 1, "signature-parameter-ARG0"),
        _orientation("upper-extremum", 0, "nested-maximum"),
        _orientation("upper-extremum", 1, "upper-bound-literal"),
        _orientation("lower-extremum", 0, "bound-item0"),
        _orientation("lower-extremum", 1, "lower-bound-literal"),
        _nesting("root", "mapped-clamp", 1),
        _nesting("root", "value-binder", 2),
        _nesting("root", "upper-extremum", 3),
        _nesting("root", "lower-extremum", 4),
        _parameter("collection-arg0"),
        _literal_slot(0, "lower-literal"),
        _literal_copy("lower-literal", "copy-typed-int-prompt-literal-0"),
        _literal_slot(1, "upper-literal"),
        _literal_copy("upper-literal", "copy-typed-int-prompt-literal-1"),
        _agreement(),
    )


def _add_then_count() -> tuple[AbstractRelayLabel, ...]:
    return _target_set(
        _root("count", "list-count", "List[Int]", "Int"),
        _edge("root", 0, "predicate-binder"),
        _edge("root", 1, "collection-arg0"),
        _edge("predicate-binder", 0, "greater-than-as-less-than"),
        _edge("greater-than-as-less-than", 0, "threshold-literal"),
        _edge("greater-than-as-less-than", 1, "bound-add"),
        _edge("bound-add", 0, "bound-item0"),
        _edge("bound-add", 1, "offset-literal"),
        _orientation("root", 0, "predicate-binder"),
        _orientation("root", 1, "signature-parameter-ARG0"),
        _orientation("greater-than-as-less-than", 0, "threshold-literal"),
        _orientation("greater-than-as-less-than", 1, "nested-add"),
        _orientation("bound-add", 0, "bound-item0"),
        _orientation("bound-add", 1, "offset-literal"),
        _nesting("root", "predicate-binder", 1),
        _nesting("root", "greater-than-as-less-than", 2),
        _nesting("root", "bound-add", 3),
        _parameter("collection-arg0"),
        _literal_slot(0, "offset-literal"),
        _literal_copy("offset-literal", "copy-typed-int-prompt-literal-0"),
        _literal_slot(1, "threshold-literal"),
        _literal_copy("threshold-literal", "copy-typed-int-prompt-literal-1"),
        _agreement(),
    )


def _string_filter(class_label: str, *, reverse: bool) -> tuple[AbstractRelayLabel, ...]:
    root_production = "reverse" if reverse else "filter"
    root_family = "string-transform" if reverse else "string-filter"
    labels: list[AbstractRelayLabel] = [
        _root(root_production, root_family, "String", "String"),
    ]
    filter_slot = "filtered-string" if reverse else "root"
    if reverse:
        labels.extend(
            (
                _edge("root", 0, filter_slot),
                _nesting("root", filter_slot, 1),
            )
        )
    labels.extend(
        (
            _edge(filter_slot, 0, "predicate-binder"),
            _edge(filter_slot, 1, "string-arg0"),
            _edge("predicate-binder", 0, "contains-class"),
            _edge("contains-class", 0, "ascii-class-literal"),
            _edge("contains-class", 1, "bound-char0"),
            _orientation(filter_slot, 0, "predicate-binder"),
            _orientation(filter_slot, 1, "signature-parameter-ARG0"),
            _orientation("contains-class", 0, "compiler-ascii-class"),
            _orientation("contains-class", 1, "bound-char0"),
            _nesting("root", "predicate-binder", 2 if reverse else 1),
            _nesting("root", "contains-class", 3 if reverse else 2),
            _parameter("string-arg0"),
            _literal_slot("compiler-ascii-class", "ascii-class-literal"),
            _literal_copy(
                "ascii-class-literal",
                f"emit-ascii-class-{class_label}",
            ),
            _agreement(),
        )
    )
    return _target_set(*labels)


_SCALAR_CLAMP_INPUTS = (
    "0b1b35530cf1cd008dbf1f06c9b12b5bdb4f7f21116800eb41f145565e7218db",
    "44d4833bf08a64132d580a3b4177eb7a519a5e332437276c4cfcaa90acdaa6c6",
    "819f05c7981087f36c02d8fb77aaa262f814bbf3f3f054ffed470053fa081613",
    "92574fe36c00ab573f262c9e2e7fb6efe8b70334a63f2febb7a6e3513a97412c",
    "c1822ecc1eed61b2fd345a3da91d8ece36e8df89fe9ddef05c7ec0cadac8f416",
)
_SCALAR_ADD_INPUTS = (
    "1ea63f6a9e51f0ba7c53332f6bb6afa257c1e0c9e6ac8d0d7a2c6c3105a0350a",
    "3c6d723ca3674df3a00a2bb3209ac91c532d6556154a302d133dd790d58b3417",
    "93b50c88e382800341b4df95eb2c266d21e049033e7a124d4aa355bc9f1e3830",
    "9c37e4ec14e765097f2ed3b9ce639758611b401d128112b478ef1ed8e17d5db1",
    "a69c93d115f0c21b0fd15393205ab280f3d6a94d71a1e24ca2fc10d9aa69de70",
    "ada96e9f27df9673cf5403da639d2c00dc7601404eb521d1e75c7a3e1f727981",
    "b0e11966bc4d8401403bdd8911f88d5ca2fca5cd089cb632a4f2dca3412b1057",
    "c13c6983dcc883f18221f1bb498804b3bfd328aab5e9f801553f2d0735b36f5f",
    "ef74c4faa42df9fe13830226617d93ea9a0ce426b57657babfd57aad5c1c34e8",
)
_SCALAR_NONNEGATIVE_INPUTS = ("bb19ef6d812310fc149c7674e202255f4deec260ae6815b455849183b0c241b9",)
_LIST_MAP_ADD_INPUTS = (
    "8ae6fe3144d91c4eba12deaae79953a391399c21a2612d1edca63d840c4f2b3c",
    "a5384e5aaf19583709419e4d18a1c9e36f0b41fdcd5257f6b8ade3c545a7d453",
    "ca5b99de021d754bd33999d7f59b18a2e2fd4132afb1559404ad9b8b890e750b",
    "cce640ee89e2c02c5b3749e1bdde528d822b825fad77d65d964c0d041a927628",
    "f81c12c1435d040d5fdd6977ca98269f184ac6ba3f0513c6c715d12b371835f1",
)
_LIST_MAP_ABSOLUTE_INPUTS = ("e673f87f51571c9b3b11d4e81aac52a993e11a6ca9d7155e57e43b129c41c1f5",)
_CLAMP_SUM_INPUTS = (
    "2b98f16e11011af3a4a43e467fd032b38191eed195ee4dc397e462848b12686c",
    "449c4e8e502b9e0eb9454c4f85d6618cdcd768ad03976d3d28486727b05cb596",
    "89772cfe0acde474c4bd561aff4620d70d2f99a29b7ad595bfd38f50d01178ee",
    "90c8125972221dd26bfcda51f78ad12bfa2678a7ef8e2712959ad837eaeddbff",
    "9dc96b833e02ced1f9bd681ecdd3c2d5e982eec8fcee427a772569dcb419b2e0",
    "e8d84713fea7c2d0057368bea1e75aa3dd2eeaa1c685e9dc675de41428dd288d",
    "fc3c43134eeaea4c920f1f31f15e04da1081b3ea5fb29c492d3058961c518224",
)
_ADD_COUNT_INPUTS = (
    "0eb123c8a8a5d799a359dd1a5cd8a0f36f8a30cffbccaf2b1c9fa676035a3a1c",
    "17bffc18f60edf862f6ac6b141fbdbc8522b30d3b78a699fe229f6d13fafca48",
    "53b970f9f2bc8cb840b6d1d13e71977c62b9cb203888374cc9fb170bac05a796",
    "78e892eb3ea2d16eca25293096ab61793a051cbbc8d30a72c250e5b2b753f0dc",
    "b54c3f40cf170b70da6fb073b2bd825bcca12b9508dcb12900a29ae8ef2c295d",
    "d67e923ca43060115634a1b26ea63b0c745723b963e45d48ab320848f1dd27c6",
    "e99695895a675fd842c92a151fd8a06f257dfd1a8410f5f04178cc6a84187684",
    "ead6b56f7281b339824ae963e9c7de1764d743b862fbb7a98bac84d094b35ef6",
)


Teacher = Callable[[], tuple[AbstractRelayLabel, ...]]


def _fixed_string_teacher(class_label: str, *, reverse: bool) -> Teacher:
    return lambda: _string_filter(class_label, reverse=reverse)


_COMPILER_TEACHER_PROJECTIONS: tuple[tuple[tuple[str, ...], Teacher, int], ...] = (
    (_SCALAR_CLAMP_INPUTS, _scalar_clamp, 2),
    (_SCALAR_ADD_INPUTS, _scalar_add, 1),
    (_SCALAR_NONNEGATIVE_INPUTS, _scalar_nonnegative, 0),
    (_LIST_MAP_ADD_INPUTS, _list_map_add, 1),
    (_LIST_MAP_ABSOLUTE_INPUTS, _list_map_absolute, 0),
    (_CLAMP_SUM_INPUTS, _clamp_sum, 2),
    (_ADD_COUNT_INPUTS, _add_then_count, 2),
    (
        ("fccd2b5c2b1538c0f6374c4d26faff3f75722773fd9ad9ebd7922a5f3381e0dd",),
        _fixed_string_teacher("ascii-decimal-digits", reverse=False),
        0,
    ),
    (
        ("c11291cc43f2cd38d2491b6ec4a4da3f45fe97f8b792fe01029d15835002b0e8",),
        _fixed_string_teacher("ascii-lowercase", reverse=False),
        0,
    ),
    (
        ("915f0c59431bced6f285bd3e276776652d8fa375f6ed6c3b5fd30a4a6753b6ff",),
        _fixed_string_teacher("english-vowels", reverse=False),
        0,
    ),
    (
        ("86a085f40de0125e2e9246fc1f0e50ca1dfcc15cabbaa2cf05dbf2eb2f91348b",),
        _fixed_string_teacher("ascii-decimal-digits", reverse=True),
        0,
    ),
    (
        ("88a0aab410711043014c050ec9a4f1fb31628d5d7838a81c5169dbb2bf307bbd",),
        _fixed_string_teacher("ascii-lowercase", reverse=True),
        0,
    ),
    (
        ("cf798e16f3e6e7ef6dc6466c3e915cb6587e4c1c04c695800050a7ac3d8dffed",),
        _fixed_string_teacher("english-vowels", reverse=True),
        0,
    ),
)


def _authenticate_dependencies() -> None:
    if selection_v1.PROTOCOL.protocol_sha256 != EXPECTED_SELECTION_PROTOCOL_SHA256:
        raise RuntimeError("spent prompt-selection protocol root changed")
    teacher_source = _ROOT / COMPILER_TEACHER_SOURCE_PATH
    if sha256(teacher_source.read_bytes()).hexdigest() != COMPILER_TEACHER_SOURCE_SHA256:
        raise RuntimeError("compiler-owned public-TRAIN semantic teacher source changed")
    if len(_COMPILER_TEACHER_PROJECTIONS) != COMPILER_TEACHER_PROJECTION_COUNT:
        raise RuntimeError("compiler teacher projection count changed")


_authenticate_dependencies()

DEPENDENCY_RECORD = _freeze_record(
    {
        "selection_protocol_scope": selection_v1.PROTOCOL.scope,
        "selection_protocol_status": selection_v1.PROTOCOL.status,
        "selection_protocol_sha256": selection_v1.PROTOCOL.protocol_sha256,
        "signature_bindings_sha256": selection_v1.PROTOCOL.signature_bindings_sha256,
        "canonical_inputs_sha256": selection_v1.PROTOCOL.canonical_inputs_sha256,
        "relay_target_vocabulary_sha256": (selection_v1.PROTOCOL.relay_target_vocabulary_sha256),
        "compiler_teacher_source_commit": COMPILER_TEACHER_SOURCE_COMMIT,
        "compiler_teacher_source_path": COMPILER_TEACHER_SOURCE_PATH,
        "compiler_teacher_source_sha256": COMPILER_TEACHER_SOURCE_SHA256,
        "compiler_teacher_source_bytes_hashed_only": True,
        "compiler_teacher_module_imported": False,
        "task_generator_executed": False,
        "oracle_executed": False,
        "private_rows_materialized": False,
        "dependency_metadata_is_not_learner_evidence": True,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(_DEPENDENCY_NAMESPACE, dict(DEPENDENCY_RECORD))


def _build_projection() -> tuple[
    tuple[StructuralAnnotation, ...],
    StructuralIdentifiabilityAudit,
    str,
]:
    canonical = selection_v1.PROTOCOL.canonical_inputs
    canonical_by_prompt = {row.prompt_input_sha256: row for row in canonical}
    expected_prompt_inputs = set(canonical_by_prompt)

    teachers: dict[str, tuple[Teacher, int]] = {}
    for prompt_inputs, teacher, literal_count in _COMPILER_TEACHER_PROJECTIONS:
        for prompt_input_sha256 in prompt_inputs:
            incumbent = teachers.setdefault(
                prompt_input_sha256,
                (teacher, literal_count),
            )
            if incumbent != (teacher, literal_count):
                raise RuntimeError("compiler teacher prompt-input registry aliases")

    prompt_inputs = prompt_manifest.load_public_train_prompt_inputs()
    if len(prompt_inputs) != PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT:
        raise RuntimeError("public-TRAIN prompt appearance count changed")
    live_literal_counts: dict[str, int] = {}
    for binding, prompt_input in zip(
        selection_v1.PROTOCOL.signature_bindings,
        prompt_inputs,
        strict=True,
    ):
        if binding.prompt_input_sha256 != prompt_input.sha256:
            raise RuntimeError("signature binding and prompt-input order changed")
        literal_count = len(prompt_input.signed_literals)
        incumbent = live_literal_counts.setdefault(prompt_input.sha256, literal_count)
        if incumbent != literal_count:
            raise RuntimeError("identical prompt input has divergent literal arity")

    missing_prompt_inputs = expected_prompt_inputs - set(teachers)
    extra_prompt_inputs = set(teachers) - expected_prompt_inputs
    annotations_by_causal_key: dict[str, list[StructuralAnnotation]] = defaultdict(list)
    annotation_by_prompt: dict[str, StructuralAnnotation] = {}
    for row in canonical:
        teacher_spec = teachers.get(row.prompt_input_sha256)
        if teacher_spec is None:
            continue
        teacher, expected_literal_count = teacher_spec
        if live_literal_counts.get(row.prompt_input_sha256) != expected_literal_count:
            raise RuntimeError("compiler teacher literal-source arity changed")
        labels = teacher()
        numeric_literal_sources = {
            label.fields["literal_source_ordinal"]
            for label in labels
            if label.channel == "literal-slot-binding"
            and type(label.fields["literal_source_ordinal"]) is int
        }
        if any(
            source < 0 or source >= expected_literal_count for source in numeric_literal_sources
        ):
            raise RuntimeError("compiler teacher references an absent physical literal event")
        target_set_sha = _namespaced_sha256(
            _TARGET_SET_NAMESPACE,
            tuple(label.record() for label in labels),
        )
        record = {
            "causal_key_sha256": row.causal_key_sha256,
            "relay_labels": tuple(label.record() for label in labels),
            "target_set_sha256": target_set_sha,
        }
        annotation = StructuralAnnotation(
            causal_key_sha256=row.causal_key_sha256,
            relay_labels=labels,
            target_set_sha256=target_set_sha,
            sha256=_namespaced_sha256(_ANNOTATION_NAMESPACE, record),
        )
        annotation_by_prompt[row.prompt_input_sha256] = annotation
        annotations_by_causal_key[row.causal_key_sha256].append(annotation)

    annotations = tuple(
        annotation_by_prompt[row.prompt_input_sha256]
        for row in canonical
        if row.prompt_input_sha256 in annotation_by_prompt
    )
    causal_collisions = sum(
        len({item.target_set_sha256 for item in items}) > 1
        for items in annotations_by_causal_key.values()
    )
    appearance_mismatches = sum(
        binding.prompt_input_sha256 not in annotation_by_prompt
        or annotation_by_prompt[binding.prompt_input_sha256].causal_key_sha256
        != binding.causal_key.sha256
        for binding in selection_v1.PROTOCOL.signature_bindings
    )
    exact_causal_keys = {row.causal_key_sha256 for row in canonical}
    published_causal_keys = {row.causal_key_sha256 for row in annotations}
    missing_causal_keys = exact_causal_keys - published_causal_keys
    extra_causal_keys = published_causal_keys - exact_causal_keys
    if missing_prompt_inputs or extra_prompt_inputs or missing_causal_keys or extra_causal_keys:
        status = STOP_TOTAL_FUNCTION
    elif causal_collisions or appearance_mismatches:
        status = STOP_CAUSAL_KEY_TARGET_ALIAS
    else:
        status = PASS_IDENTIFIABLE

    channel_counts = Counter(
        label.channel for annotation in annotations for label in annotation.relay_labels
    )
    if status == PASS_IDENTIFIABLE and set(channel_counts) != set(_CHANNELS):
        raise RuntimeError("structural teacher projection lost a relay channel")
    channel_rows = tuple((channel, channel_counts[channel]) for channel in _CHANNELS)
    audit_without_sha = {
        "status": status,
        "prompt_appearance_count": len(selection_v1.PROTOCOL.signature_bindings),
        "duplicate_prompt_appearance_count": (
            len(selection_v1.PROTOCOL.signature_bindings) - len(canonical)
        ),
        "canonical_causal_identity_count": len(canonical),
        "exact_target_set_count": len(annotations),
        "missing_causal_key_count": len(missing_causal_keys),
        "extra_causal_key_count": len(extra_causal_keys),
        "appearance_target_mismatch_count": appearance_mismatches,
        "causal_key_target_collision_count": causal_collisions,
        "relay_channel_counts": channel_rows,
        "identical_causal_key_rule": (
            "one physical causal key must map to one exact relay target set"
        ),
        "total_function_completed": status == PASS_IDENTIFIABLE,
        "collision_audit_completed": status == PASS_IDENTIFIABLE,
    }
    audit = StructuralIdentifiabilityAudit(
        status=status,
        prompt_appearance_count=len(selection_v1.PROTOCOL.signature_bindings),
        duplicate_prompt_appearance_count=(
            len(selection_v1.PROTOCOL.signature_bindings) - len(canonical)
        ),
        canonical_causal_identity_count=len(canonical),
        exact_target_set_count=len(annotations),
        missing_causal_key_count=len(missing_causal_keys),
        extra_causal_key_count=len(extra_causal_keys),
        appearance_target_mismatch_count=appearance_mismatches,
        causal_key_target_collision_count=causal_collisions,
        relay_channel_counts=channel_rows,
        total_function_completed=status == PASS_IDENTIFIABLE,
        collision_audit_completed=status == PASS_IDENTIFIABLE,
        sha256=_namespaced_sha256(_AUDIT_NAMESPACE, audit_without_sha),
    )
    projection_sha = _namespaced_sha256(
        _PROJECTION_NAMESPACE,
        tuple(annotation.learner_record() for annotation in annotations),
    )
    return annotations, audit, projection_sha


ANNOTATIONS, IDENTIFIABILITY_AUDIT, TEACHER_PROJECTION_SHA256 = _build_projection()

SURFACE_BOUNDARY_RECORD = _freeze_record(
    {
        "public_train_prompt_appearance_count": PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
        "canonical_causal_identity_count": CANONICAL_CAUSAL_IDENTITY_COUNT,
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "compiler_teacher_projection_count": COMPILER_TEACHER_PROJECTION_COUNT,
        "learner_facing_field_names": (
            "causal_key_sha256",
            "relay_labels",
            "target_set_sha256",
            "sha256",
        ),
        "raw_prompt_text_export_count": 0,
        "full_typed_signature_export_count": 0,
        "role_normalized_prompt_input_sha256_export_count": 0,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_read_count": DESIRED_OUTPUT_READ_COUNT,
        "program_artifact_export_count": PROGRAM_ARTIFACT_EXPORT_COUNT,
        "executable_source_export_count": EXECUTABLE_SOURCE_EXPORT_COUNT,
        "ast_artifact_export_count": AST_ARTIFACT_EXPORT_COUNT,
        "property_row_read_count": PROPERTY_ROW_READ_COUNT,
        "task_identifier_export_count": TASK_IDENTIFIER_EXPORT_COUNT,
        "family_identifier_export_count": FAMILY_IDENTIFIER_EXPORT_COUNT,
        "entrypoint_spelling_neural_evidence_count": (ENTRYPOINT_SPELLING_NEURAL_EVIDENCE_COUNT),
        "audit_hash_neural_evidence_count": AUDIT_HASH_NEURAL_EVIDENCE_COUNT,
        "candidate_ranking_count": CANDIDATE_RANKING_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "proof_count": PROOF_COUNT,
        "learner_or_execution_interface_present": False,
    }
)
SURFACE_BOUNDARY_SHA256 = _namespaced_sha256(
    _SURFACE_NAMESPACE,
    dict(SURFACE_BOUNDARY_RECORD),
)


def _build_capsule() -> FrozenPublicTrainStructuralAnnotationCapsuleV1:
    root_without_capsule = {
        "scope": PROTOCOL_SCOPE,
        "status": IDENTIFIABILITY_AUDIT.status,
        "dependency_sha256": DEPENDENCY_SHA256,
        "teacher_projection_sha256": TEACHER_PROJECTION_SHA256,
        "identifiability_audit_sha256": IDENTIFIABILITY_AUDIT.sha256,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "prompt_appearance_count": IDENTIFIABILITY_AUDIT.prompt_appearance_count,
        "canonical_causal_identity_count": len(ANNOTATIONS),
        "exact_target_set_count": len(ANNOTATIONS),
        "relay_label_count": sum(len(row.relay_labels) for row in ANNOTATIONS),
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "proof_count": PROOF_COUNT,
        "learner_or_execution_interface_present": False,
    }
    return FrozenPublicTrainStructuralAnnotationCapsuleV1(
        scope=PROTOCOL_SCOPE,
        status=IDENTIFIABILITY_AUDIT.status,
        annotations=ANNOTATIONS,
        identifiability_audit=IDENTIFIABILITY_AUDIT,
        dependency_sha256=DEPENDENCY_SHA256,
        teacher_projection_sha256=TEACHER_PROJECTION_SHA256,
        identifiability_audit_sha256=IDENTIFIABILITY_AUDIT.sha256,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        capsule_sha256=_namespaced_sha256(_CAPSULE_NAMESPACE, root_without_capsule),
    )


CAPSULE = _build_capsule()


__all__ = [
    "ANNOTATIONS",
    "CANONICAL_CAUSAL_IDENTITY_COUNT",
    "CANONICAL_FIT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_IDENTITY_COUNT",
    "CAPSULE",
    "COMPILER_TEACHER_PROJECTION_COUNT",
    "COMPILER_TEACHER_SOURCE_COMMIT",
    "COMPILER_TEACHER_SOURCE_PATH",
    "COMPILER_TEACHER_SOURCE_SHA256",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "IDENTIFIABILITY_AUDIT",
    "PASS_IDENTIFIABLE",
    "PROTOCOL_SCOPE",
    "PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT",
    "STOP_CAUSAL_KEY_TARGET_ALIAS",
    "STOP_TOTAL_FUNCTION",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "TEACHER_PROJECTION_SHA256",
    "AbstractRelayLabel",
    "FrozenPublicTrainStructuralAnnotationCapsuleV1",
    "StructuralAnnotation",
    "StructuralIdentifiabilityAudit",
]
