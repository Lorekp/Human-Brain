"""Fresh data-only public prompt-selection V5 protocol.

The predecessor V4 target projection can describe abstract structural labels,
but its annotations do not publish the occurrence-local physical identities
needed by the banked occurrence-slot mechanism.  This module therefore performs
only a FIT-roster identifiability audit.  It never turns legacy slot spelling,
relay order, or child ordinals into a node identity.

The result is an immutable STOP ledger and an explicit contract for the next
compiler-owned annotation capsule.  No learner is imported, allocated, or run;
the curriculum is intentionally empty.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.r3_pilot import (
    public_train_list_int_to_int_prompt_input_capsule_v1 as input_capsule,
)
from experiments.r3_pilot import (
    public_train_prompt_selection_freeze_v1_protocol as selection_v1,
)
from experiments.r3_pilot import (
    public_train_structural_annotation_capsule_v1 as annotations_v1,
)

PROTOCOL_SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_V5_PROTOCOL"
STOP_IDENTIFIABILITY = (
    "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V5_PROTOCOL_"
    "OCCURRENCE_LOCAL_TARGETS_UNIDENTIFIABLE"
)
STOP_DEPENDENCY = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V5_PROTOCOL_DEPENDENCY"
FIT = "FIT"

V4_PROTOCOL_SOURCE_PATH = (
    "experiments/r3_pilot/public_train_prompt_selection_v4_protocol.py"
)
V4_PROTOCOL_SOURCE_SHA256 = (
    "3473d2b46eed8c974dfe3b30b3d075614ac4966a5b4e25f135202d2ad986b1f6"
)
V4_PROTOCOL_SHA256 = "97f8ea272aac11188a0c5f31355085fddcf69eed602b0140eabc2f90d4920918"
SELECTION_V1_PROTOCOL_SHA256 = (
    "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"
)
ANNOTATION_CAPSULE_SHA256 = (
    "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf"
)
ANNOTATION_SOURCE_PATH = (
    "experiments/r3_pilot/public_train_structural_annotation_capsule_v1.py"
)
ANNOTATION_SOURCE_SHA256 = (
    "d2e977b3aff2e496307168ed64e85c2ebbc03579224bf58b0fd175f0ed096c88"
)
INPUT_CAPSULE_SOURCE_PATH = (
    "experiments/r3_pilot/public_train_list_int_to_int_prompt_input_capsule_v1.py"
)
INPUT_CAPSULE_SOURCE_SHA256 = (
    "cef67a95765e268425e23cc9c270f8cdc161723584135c7b29e208b8ec9a4433"
)
INPUT_ARTIFACT_SHA256 = input_capsule.ARTIFACT_SHA256
INPUT_CONTENT_ROOT = input_capsule.CONTENT_ROOT

FIT_IDENTITY_COUNT = 9
FIT_APPEARANCE_COUNT = 12
EXCLUDED_HELDOUT_IDENTITY_COUNT = 6

LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
CURRICULUM_LESSON_COUNT = 0
PANEL_SPEND_COUNT = 0
V4_WEIGHT_READ_COUNT = 0
V4_CHECKPOINT_READ_COUNT = 0
HELDOUT_TARGET_READ_COUNT = 0
HELDOUT_INPUT_EVENT_READ_COUNT = 0
TASK_FIELD_COUNT = 0
FAMILY_FIELD_COUNT = 0
SOURCE_ORDINAL_FIELD_COUNT = 0
ENTRYPOINT_FIELD_COUNT = 0
PROGRAM_FIELD_COUNT = 0
DESIRED_FIELD_COUNT = 0
PROPERTY_FIELD_COUNT = 0
DEV_FIELD_COUNT = 0
PROTECTED_FIELD_COUNT = 0

NODE_CARRIER_ACTIVE_CELL_COUNT = 12
FACTOR_KINDS = ("literal", "role")
PHYSICAL_TYPES = ("Int", "List[Int]")
RELATION_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge")

MISSING_REQUIREMENTS = (
    "stable-node-occurrence-carrier",
    "edge-parent-child-carriers",
    "parent-factor-slot-occupancy",
    "occurrence-local-raw-span-or-typed-literal-binding",
    "ordered-generic-factor-slot-records",
    "physical-relation-slot-value-events",
    "complete-domains-and-literal-copy-groups",
)

FORBIDDEN_FIELDS = (
    "task",
    "family",
    "source_ordinal",
    "entrypoint",
    "program",
    "desired",
    "property",
    "dev",
    "heldout",
    "protected",
)

_ROOT = Path(__file__).resolve().parents[2]
_DEPENDENCY_NAMESPACE = b"public-train-prompt-selection-v5-dependency\0"
_SUCCESSOR_NAMESPACE = b"public-train-prompt-selection-v5-successor-contract\0"
_FIT_ROW_NAMESPACE = b"public-train-prompt-selection-v5-fit-row\0"
_ROSTER_NAMESPACE = b"public-train-prompt-selection-v5-fit-roster\0"
_STOP_NAMESPACE = b"public-train-prompt-selection-v5-stop-ledger\0"
_SURFACE_NAMESPACE = b"public-train-prompt-selection-v5-surface\0"
_PROTOCOL_NAMESPACE = b"public-train-prompt-selection-v5-protocol\0"


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _file_sha256(relative_path: str) -> str:
    return sha256((_ROOT / relative_path).read_bytes()).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class SuccessorAnnotationContract:
    """Exact offline-teacher surface required before a fresh learner exists."""

    scope: str
    event_fields: tuple[str, ...]
    relation_event_fields: tuple[str, ...]
    domain_fields: tuple[str, ...]
    literal_copy_group_fields: tuple[str, ...]
    factor_kinds: tuple[str, ...]
    physical_types: tuple[str, ...]
    relation_operators: tuple[str, ...]
    node_carrier_active_cell_count: int
    invariants: tuple[str, ...]
    forbidden_fields: tuple[str, ...]
    sha256: str


def _build_successor_contract() -> SuccessorAnnotationContract:
    record = {
        "scope": "FIT_ONLY_COMPILER_OWNED_STRUCTURAL_EVENT_TARGETS_V5",
        "event_fields": (
            "compiler_event_sha256",
            "event_kind_cells",
            "node_occurrence_cells",
            "edge_parent_occurrence_cells",
            "edge_child_occurrence_cells",
            "edge_child_position_cells",
            "parent_factor_slot_occupancy_cells",
            "occurrence_source_kind_cells",
            "occurrence_source_event_sha256",
            "ordered_factor_slots",
            "relation_events",
        ),
        "relation_event_fields": (
            "left_factor_slot_cells",
            "operator_cells",
            "right_factor_slot_cells",
            "right_value_event_cells",
        ),
        "domain_fields": (
            "node_occurrence_cells",
            "factor_slot_cells",
            "complete_allowed_value_event_cells",
        ),
        "literal_copy_group_fields": (
            "copy_group_cells",
            "terminal_node_occurrence_cells",
            "terminal_factor_slot_cells",
        ),
        "factor_kinds": FACTOR_KINDS,
        "physical_types": PHYSICAL_TYPES,
        "relation_operators": RELATION_OPERATORS,
        "node_carrier_active_cell_count": NODE_CARRIER_ACTIVE_CELL_COUNT,
        "invariants": (
            "emit-all-targets-atomically-at-structural-event-production",
            "reuse-one-distributed-node-cue-across-edge-orientation-parameter-literal-copy",
            "bind-every-edge-endpoint-directly-never-by-slot-spelling-or-order",
            "bind-every-source-to-one-authenticated-raw-span-or-typed-literal-event",
            "order-factor-slots-explicitly-as-kind-and-physical-type-records",
            "publish-physical-assemblies-for-all-six-relation-operators",
            "publish-complete-constrained-domains-not-observed-subsets",
            "partition-every-terminal-literal-factor-into-one-copy-group",
            "compiler-targets-are-offline-teacher-data-never-inference-evidence",
        ),
        "forbidden_fields": FORBIDDEN_FIELDS,
    }
    return SuccessorAnnotationContract(
        **record,
        sha256=_digest(_SUCCESSOR_NAMESPACE, record),
    )


SUCCESSOR_ANNOTATION_CONTRACT = _build_successor_contract()


@dataclass(frozen=True, slots=True)
class FitIdentityAudit:
    """Schema-level audit only; no semantic label value is retained or replayed."""

    input_root_sha256: str
    multiplicity: int
    annotation_sha256: str
    target_set_sha256: str
    raw_span_event_count: int
    typed_literal_event_count: int
    legacy_relay_label_count: int
    legacy_edge_label_count: int
    legacy_child_ordinal_field_count: int
    legacy_orientation_label_count: int
    legacy_parameter_label_count: int
    legacy_literal_slot_label_count: int
    legacy_literal_copy_label_count: int
    explicit_stable_node_occurrence_count: int
    explicit_edge_carrier_record_count: int
    explicit_parent_factor_slot_occupancy_count: int
    explicit_occurrence_local_source_binding_count: int
    explicit_generic_factor_slot_record_count: int
    explicit_relation_slot_value_event_count: int
    explicit_complete_domain_record_count: int
    explicit_literal_copy_group_count: int
    missing_requirements: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class IdentifiabilityStopLedger:
    status: str
    fit_identity_count: int
    fit_appearance_count: int
    excluded_heldout_identity_count: int
    input_annotation_join_count: int
    missing_input_annotation_count: int
    duplicate_input_root_count: int
    target_class_count: int
    causal_key_target_conflict_count: int
    raw_span_event_count: int
    typed_literal_event_count: int
    legacy_relay_label_count: int
    legacy_edge_label_count: int
    legacy_edge_endpoint_reference_count: int
    legacy_child_ordinal_field_count: int
    legacy_orientation_label_count: int
    legacy_parameter_label_count: int
    legacy_literal_slot_label_count: int
    legacy_literal_copy_label_count: int
    structural_event_target_count: int
    stable_node_occurrence_count: int
    edge_carrier_record_count: int
    parent_factor_slot_occupancy_count: int
    occurrence_local_source_binding_count: int
    generic_factor_slot_record_count: int
    relation_slot_value_event_count: int
    complete_domain_record_count: int
    literal_copy_group_count: int
    missing_requirement_identity_counts: tuple[tuple[str, int], ...]
    fit_roster_sha256: str
    successor_contract_sha256: str
    curriculum_lesson_count: int
    learner_array_count: int
    learning_run_count: int
    electrical_execution_count: int
    panel_spend_count: int
    sha256: str


def _verify_dependencies() -> None:
    if (
        _file_sha256(V4_PROTOCOL_SOURCE_PATH) != V4_PROTOCOL_SOURCE_SHA256
        or _file_sha256(ANNOTATION_SOURCE_PATH) != ANNOTATION_SOURCE_SHA256
        or _file_sha256(INPUT_CAPSULE_SOURCE_PATH) != INPUT_CAPSULE_SOURCE_SHA256
        or selection_v1.PROTOCOL.protocol_sha256 != SELECTION_V1_PROTOCOL_SHA256
        or annotations_v1.CAPSULE.capsule_sha256 != ANNOTATION_CAPSULE_SHA256
        or input_capsule.ARTIFACT_SHA256 != INPUT_ARTIFACT_SHA256
        or input_capsule.CONTENT_ROOT != INPUT_CONTENT_ROOT
    ):
        raise RuntimeError(STOP_DEPENDENCY)


_verify_dependencies()

DEPENDENCY_RECORD = _freeze(
    {
        "v4_protocol_source_path": V4_PROTOCOL_SOURCE_PATH,
        "v4_protocol_source_sha256": V4_PROTOCOL_SOURCE_SHA256,
        "v4_protocol_sha256": V4_PROTOCOL_SHA256,
        "v4_module_imported": False,
        "v4_weights_or_checkpoint_read": False,
        "selection_v1_protocol_sha256": SELECTION_V1_PROTOCOL_SHA256,
        "annotation_capsule_sha256": ANNOTATION_CAPSULE_SHA256,
        "annotation_source_sha256": ANNOTATION_SOURCE_SHA256,
        "input_artifact_sha256": INPUT_ARTIFACT_SHA256,
        "input_content_root": INPUT_CONTENT_ROOT,
        "compiler_annotations_are_offline_teacher_data_only": True,
    }
)
DEPENDENCY_SHA256 = _digest(_DEPENDENCY_NAMESPACE, DEPENDENCY_RECORD)


def _fit_identity_audits() -> tuple[FitIdentityAudit, ...]:
    capsule = input_capsule.load_input_capsule()
    split_by_root = {
        row.causal_key_sha256: row.split
        for row in selection_v1.PROTOCOL.canonical_inputs
    }
    fit_roots = tuple(
        sorted(
            root
            for root in input_capsule.EXPECTED_INPUT_ROOTS
            if split_by_root.get(root) == FIT
        )
    )
    excluded_roots = tuple(
        root
        for root in input_capsule.EXPECTED_INPUT_ROOTS
        if split_by_root.get(root) != FIT
    )
    if (
        len(fit_roots) != FIT_IDENTITY_COUNT
        or len(excluded_roots) != EXCLUDED_HELDOUT_IDENTITY_COUNT
    ):
        raise RuntimeError(STOP_DEPENDENCY)

    annotations_by_root = {
        row.causal_key_sha256: row for row in annotations_v1.CAPSULE.annotations
    }
    rows: list[FitIdentityAudit] = []
    for root in fit_roots:
        frozen_input = capsule.lookup(root)
        annotation = annotations_by_root.get(root)
        if annotation is None:
            raise RuntimeError(STOP_DEPENDENCY)
        channel_counts = Counter(label.channel for label in annotation.relay_labels)
        record = {
            "input_root_sha256": root,
            "multiplicity": frozen_input.multiplicity,
            "annotation_sha256": annotation.sha256,
            "target_set_sha256": annotation.target_set_sha256,
            "raw_span_event_count": len(frozen_input.raw_span_records),
            "typed_literal_event_count": len(frozen_input.typed_literal_records),
            "legacy_relay_label_count": len(annotation.relay_labels),
            "legacy_edge_label_count": channel_counts["ordered-parent-child-edge"],
            "legacy_child_ordinal_field_count": channel_counts[
                "ordered-parent-child-edge"
            ],
            "legacy_orientation_label_count": channel_counts[
                "argument-operand-orientation"
            ],
            "legacy_parameter_label_count": channel_counts[
                "parameter-role-binding"
            ],
            "legacy_literal_slot_label_count": channel_counts[
                "literal-slot-binding"
            ],
            "legacy_literal_copy_label_count": channel_counts[
                "literal-value-copy"
            ],
            # The legacy schema has none of these exact fields.  Counts remain
            # zero: never synthesize them from semantic strings or sequence.
            "explicit_stable_node_occurrence_count": 0,
            "explicit_edge_carrier_record_count": 0,
            "explicit_parent_factor_slot_occupancy_count": 0,
            "explicit_occurrence_local_source_binding_count": 0,
            "explicit_generic_factor_slot_record_count": 0,
            "explicit_relation_slot_value_event_count": 0,
            "explicit_complete_domain_record_count": 0,
            "explicit_literal_copy_group_count": 0,
            "missing_requirements": MISSING_REQUIREMENTS,
        }
        rows.append(
            FitIdentityAudit(
                **record,
                sha256=_digest(_FIT_ROW_NAMESPACE, record),
            )
        )
    if sum(row.multiplicity for row in rows) != FIT_APPEARANCE_COUNT:
        raise RuntimeError(STOP_DEPENDENCY)
    return tuple(rows)


FIT_IDENTITY_AUDITS = _fit_identity_audits()
FIT_ROSTER_SHA256 = _digest(
    _ROSTER_NAMESPACE,
    tuple(row.sha256 for row in FIT_IDENTITY_AUDITS),
)


def _build_stop_ledger() -> IdentifiabilityStopLedger:
    rows = FIT_IDENTITY_AUDITS
    record = {
        "status": STOP_IDENTIFIABILITY,
        "fit_identity_count": len(rows),
        "fit_appearance_count": sum(row.multiplicity for row in rows),
        "excluded_heldout_identity_count": EXCLUDED_HELDOUT_IDENTITY_COUNT,
        "input_annotation_join_count": len(rows),
        "missing_input_annotation_count": 0,
        "duplicate_input_root_count": len(rows)
        - len({row.input_root_sha256 for row in rows}),
        "target_class_count": len({row.target_set_sha256 for row in rows}),
        "causal_key_target_conflict_count": 0,
        "raw_span_event_count": sum(row.raw_span_event_count for row in rows),
        "typed_literal_event_count": sum(
            row.typed_literal_event_count for row in rows
        ),
        "legacy_relay_label_count": sum(
            row.legacy_relay_label_count for row in rows
        ),
        "legacy_edge_label_count": sum(row.legacy_edge_label_count for row in rows),
        "legacy_edge_endpoint_reference_count": 2
        * sum(row.legacy_edge_label_count for row in rows),
        "legacy_child_ordinal_field_count": sum(
            row.legacy_child_ordinal_field_count for row in rows
        ),
        "legacy_orientation_label_count": sum(
            row.legacy_orientation_label_count for row in rows
        ),
        "legacy_parameter_label_count": sum(
            row.legacy_parameter_label_count for row in rows
        ),
        "legacy_literal_slot_label_count": sum(
            row.legacy_literal_slot_label_count for row in rows
        ),
        "legacy_literal_copy_label_count": sum(
            row.legacy_literal_copy_label_count for row in rows
        ),
        "structural_event_target_count": 0,
        "stable_node_occurrence_count": 0,
        "edge_carrier_record_count": 0,
        "parent_factor_slot_occupancy_count": 0,
        "occurrence_local_source_binding_count": 0,
        "generic_factor_slot_record_count": 0,
        "relation_slot_value_event_count": 0,
        "complete_domain_record_count": 0,
        "literal_copy_group_count": 0,
        "missing_requirement_identity_counts": tuple(
            (requirement, len(rows)) for requirement in MISSING_REQUIREMENTS
        ),
        "fit_roster_sha256": FIT_ROSTER_SHA256,
        "successor_contract_sha256": SUCCESSOR_ANNOTATION_CONTRACT.sha256,
        "curriculum_lesson_count": CURRICULUM_LESSON_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "panel_spend_count": PANEL_SPEND_COUNT,
    }
    return IdentifiabilityStopLedger(
        **record,
        sha256=_digest(_STOP_NAMESPACE, record),
    )


STOP_LEDGER = _build_stop_ledger()

SURFACE_BOUNDARY_RECORD = _freeze(
    {
        "data_only": True,
        "curriculum_lesson_count": CURRICULUM_LESSON_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "panel_spend_count": PANEL_SPEND_COUNT,
        "v4_weight_read_count": V4_WEIGHT_READ_COUNT,
        "v4_checkpoint_read_count": V4_CHECKPOINT_READ_COUNT,
        "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
        "heldout_input_event_read_count": HELDOUT_INPUT_EVENT_READ_COUNT,
        "task_field_count": TASK_FIELD_COUNT,
        "family_field_count": FAMILY_FIELD_COUNT,
        "source_ordinal_field_count": SOURCE_ORDINAL_FIELD_COUNT,
        "entrypoint_field_count": ENTRYPOINT_FIELD_COUNT,
        "program_field_count": PROGRAM_FIELD_COUNT,
        "desired_field_count": DESIRED_FIELD_COUNT,
        "property_field_count": PROPERTY_FIELD_COUNT,
        "dev_field_count": DEV_FIELD_COUNT,
        "protected_field_count": PROTECTED_FIELD_COUNT,
        "neural_evidence_fields": (),
        "legacy_slot_spelling_used_as_identity": False,
        "legacy_event_order_used_as_identity": False,
        "legacy_child_ordinal_used_to_mint_identity": False,
    }
)
SURFACE_BOUNDARY_SHA256 = _digest(_SURFACE_NAMESPACE, SURFACE_BOUNDARY_RECORD)


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainPromptSelectionV5Protocol:
    scope: str
    status: str
    dependency_sha256: str
    successor_contract_sha256: str
    fit_roster_sha256: str
    stop_ledger_sha256: str
    surface_boundary_sha256: str
    fit_identity_audits: tuple[FitIdentityAudit, ...]
    curriculum: tuple[object, ...]
    protocol_sha256: str

    def root_record(self, *, include_protocol_sha256: bool = True) -> dict[str, object]:
        record = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "successor_contract_sha256": self.successor_contract_sha256,
            "fit_roster_sha256": self.fit_roster_sha256,
            "stop_ledger_sha256": self.stop_ledger_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "fit_identity_count": len(self.fit_identity_audits),
            "curriculum_lesson_count": len(self.curriculum),
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


def _build_protocol() -> FrozenPublicTrainPromptSelectionV5Protocol:
    record = {
        "scope": PROTOCOL_SCOPE,
        "status": STOP_IDENTIFIABILITY,
        "dependency_sha256": DEPENDENCY_SHA256,
        "successor_contract_sha256": SUCCESSOR_ANNOTATION_CONTRACT.sha256,
        "fit_roster_sha256": FIT_ROSTER_SHA256,
        "stop_ledger_sha256": STOP_LEDGER.sha256,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "fit_identity_count": len(FIT_IDENTITY_AUDITS),
        "curriculum_lesson_count": 0,
    }
    return FrozenPublicTrainPromptSelectionV5Protocol(
        scope=PROTOCOL_SCOPE,
        status=STOP_IDENTIFIABILITY,
        dependency_sha256=DEPENDENCY_SHA256,
        successor_contract_sha256=SUCCESSOR_ANNOTATION_CONTRACT.sha256,
        fit_roster_sha256=FIT_ROSTER_SHA256,
        stop_ledger_sha256=STOP_LEDGER.sha256,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        fit_identity_audits=FIT_IDENTITY_AUDITS,
        curriculum=(),
        protocol_sha256=_digest(_PROTOCOL_NAMESPACE, record),
    )


PROTOCOL = _build_protocol()


__all__ = [
    "CURRICULUM_LESSON_COUNT",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "FIT_IDENTITY_AUDITS",
    "FIT_ROSTER_SHA256",
    "FORBIDDEN_FIELDS",
    "MISSING_REQUIREMENTS",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "RELATION_OPERATORS",
    "STOP_IDENTIFIABILITY",
    "STOP_LEDGER",
    "SUCCESSOR_ANNOTATION_CONTRACT",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "FitIdentityAudit",
    "FrozenPublicTrainPromptSelectionV5Protocol",
    "IdentifiabilityStopLedger",
    "SuccessorAnnotationContract",
]
