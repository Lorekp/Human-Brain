"""Compiler-owned FIT-only occurrence/factor annotation capsule V1.

This data-only capsule attempts to translate the nine public-TRAIN
``List[Int] -> Int`` FIT identities into the complete physical target contract
frozen by :mod:`public_train_prompt_selection_v5_protocol` and structurally
exercised by the banked self-posed occurrence-factor V6 mechanism.

Compiler labels are used only while constructing offline teacher targets.  The
published capsule contains counts and physical-plan roots, never task/family
identity or semantic labels as neural evidence.  Translation stops atomically:
the public annotations identify every raw lexical span, but do not identify the
``NEXT/PREV`` selector geometry required to make one exact
``RawSpanObservation``.  The implementation enumerates that missing 8-way
space and refuses to choose by source order or a fixed index.  No learner,
checkpoint, panel, or target artifact is allocated.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from experiments.r3_pilot import (
    public_train_list_int_to_int_prompt_input_capsule_v1 as input_capsule,
)
from experiments.r3_pilot import public_train_prompt_selection_freeze_v1_protocol as selection_v1
from experiments.r3_pilot import public_train_prompt_selection_v5_protocol as prompt_v5
from experiments.r3_pilot import public_train_prompt_semantic_bridge as prompt_bridge
from experiments.r3_pilot import public_train_structural_annotation_capsule_v1 as structural_v1
from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.typed_program_types import TypedSignature

PROTOCOL_SCOPE = "PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V1"
STOP_RAW_SELECTOR = (
    "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V1_"
    "RAW_SPAN_SELECTOR_GEOMETRY_UNAVAILABLE"
)
STOP_DEPENDENCY = (
    "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V1_DEPENDENCY"
)

FIT = "FIT"
FIT_IDENTITY_COUNT = 9
FIT_APPEARANCE_COUNT = 12
TARGET_CLASS_COUNT = 2

RAW_SELECTOR_DIRECTIONS = ("NEXT", "PREV")
RAW_SELECTOR_CUE_SLOTS = (0, 1, 2, 3)
RAW_SELECTOR_GEOMETRY_COUNT = 8
EXPECTED_RAW_OBSERVATION_COUNT = 59

EVENT_ROOT = 0
EVENT_SOURCE = 1
EVENT_EDGE = 2
EVENT_FACTOR = 3
EVENT_RELATION = 4
EVENT_DOMAIN = 5
EVENT_COPY = 6
EVENT_SIGNATURE = 7
EVENT_KIND_COUNT = 8

FACTOR_KINDS = ("literal", "role")
PHYSICAL_TYPES = ("Int", "List[Int]")
RELATION_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge")
RHS_SLOT = 0
RHS_VALUE = 1

NODE_CARRIER_CELL_COUNT = 256
NODE_CARRIER_ACTIVE_CELL_COUNT = 12
COPY_GROUP_CELL_COUNT = 64
COPY_GROUP_ACTIVE_CELL_COUNT = 8
TYPE_CELL_COUNT = 48
TYPE_ACTIVE_CELL_COUNT = 8

LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
PANEL_SPEND_COUNT = 0
V4_WEIGHT_READ_COUNT = 0
V4_CHECKPOINT_READ_COUNT = 0
V5_WEIGHT_READ_COUNT = 0
V5_CHECKPOINT_READ_COUNT = 0
HELDOUT_TARGET_READ_COUNT = 0
HELDOUT_INPUT_EVENT_READ_COUNT = 0
ARTIFACT_EXPORT_COUNT = 0
CURRICULUM_LESSON_COUNT = 0
EMITTED_TARGET_RECORD_COUNT = 0

TASK_FIELD_COUNT = 0
FAMILY_FIELD_COUNT = 0
SOURCE_ORDINAL_NEURAL_FIELD_COUNT = 0
ENTRYPOINT_NEURAL_FIELD_COUNT = 0
PROGRAM_FIELD_COUNT = 0
DESIRED_FIELD_COUNT = 0
PROPERTY_FIELD_COUNT = 0
DEV_FIELD_COUNT = 0
PROTECTED_FIELD_COUNT = 0

PROMPT_V5_SOURCE_PATH = (
    "experiments/r3_pilot/public_train_prompt_selection_v5_protocol.py"
)
PROMPT_V5_SOURCE_SHA256 = (
    "6246cc8f4df35d8bc8760de91c87b9dab6ee229111e37832036732373215732b"
)
PROMPT_V5_PROTOCOL_SHA256 = (
    "c6f5bf2951576a05304b4870a2dbc3fc1921cdb62c4e992fd2af176d63baa408"
)
SUCCESSOR_CONTRACT_SHA256 = (
    "8764290fa5a1c3fda2badc06fbf400df462c14bf603bae128fdfb1394fc14cbd"
)
SELECTION_V1_PROTOCOL_SHA256 = (
    "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"
)
STRUCTURAL_ANNOTATION_CAPSULE_SHA256 = (
    "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf"
)
INPUT_ARTIFACT_SHA256 = (
    "0e90318d1c1d3bd365f9812fb3c43aa5f0806dbfa5a1e81cb527c8e7f2c20dcf"
)
INPUT_CONTENT_ROOT = (
    "a7ff53818cf4e83c823e6b23f5547e0e7ee1fd30ed73dd136e3672063ea7aca8"
)
PROMPT_SOURCE_SHA256 = (
    "0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289"
)
PROMPT_OBSERVER_SOURCE_PATH = (
    "experiments/r3_pilot/public_train_prompt_semantic_bridge.py"
)
PROMPT_OBSERVER_SOURCE_SHA256 = (
    "74a123d88707525b63ff50021b3c7e5638f93ed8ed3dd3b82d20724bb3f1d662"
)
PLAN_SLICE_SOURCE_PATH = (
    "experiments/r3_pilot/typed_semantic_packed_prompt_authorized_plan_slice_v1.py"
)
PLAN_SLICE_SOURCE_SHA256 = (
    "012cd2a52794a74d966223d50c1e7c645e5aeb150f4326bde531c8cf81f29576"
)
OCCURRENCE_FACTOR_V6_COMMIT = "b277dc781260fc09c110186c131b3ee8d179ac83"
OCCURRENCE_FACTOR_V6_SOURCE_PATH = (
    "experiments/r3_pilot/prompt_occurrence_factor_relays_v6.py"
)
OCCURRENCE_FACTOR_V6_SOURCE_SHA256 = (
    "b266d3e851828aef701a9e0b0a59b6e47df74da4349cd873dd4653394f74d269"
)

_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_SOURCE_PATH = Path(__file__).with_name(
    "public_train_prompt_signature_capsule_v1.json"
)
_NAMESPACE = b"public-train-fit-occurrence-factor-annotation-capsule-v1\0"
_DEPENDENCY_NAMESPACE = _NAMESPACE + b"dependency\0"
_VOCABULARY_NAMESPACE = _NAMESPACE + b"physical-vocabulary\0"
_SPAN_EVENT_NAMESPACE = _NAMESPACE + b"raw-span-event\0"
_LITERAL_EVENT_NAMESPACE = _NAMESPACE + b"typed-literal-event\0"
_PARTIAL_TARGET_NAMESPACE = _NAMESPACE + b"partial-target\0"
_IDENTITY_AUDIT_NAMESPACE = _NAMESPACE + b"identity-audit\0"
_AUDIT_SET_NAMESPACE = _NAMESPACE + b"audit-set\0"
_STOP_NAMESPACE = _NAMESPACE + b"stop-ledger\0"
_SURFACE_NAMESPACE = _NAMESPACE + b"surface\0"
_SUCCESSOR_NAMESPACE = _NAMESPACE + b"raw-source-successor-contract\0"
_CAPSULE_NAMESPACE = _NAMESPACE + b"capsule\0"


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
    result = sha256()
    result.update(namespace)
    result.update(_canonical_json(value))
    return result.hexdigest()


def _file_sha256(relative_path: str) -> str:
    return sha256((_ROOT / relative_path).read_bytes()).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


def _sparse_cells(
    label: str,
    key: object,
    *,
    width: int,
    count: int,
) -> tuple[int, ...]:
    selected: set[int] = set()
    counter = 0
    while len(selected) < count:
        digest = sha256()
        digest.update(_NAMESPACE)
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(key))
        digest.update(counter.to_bytes(4, "little"))
        block = digest.digest()
        for start in range(0, len(block), 4):
            selected.add(int.from_bytes(block[start : start + 4], "little") % width)
            if len(selected) == count:
                break
        counter += 1
    return tuple(sorted(selected))


def _allocate_disjoint(
    labels: tuple[str, ...],
    *,
    width: int,
    count: int,
    namespace: str,
) -> tuple[tuple[int, ...], ...]:
    used: set[int] = set()
    rows: list[tuple[int, ...]] = []
    for label in labels:
        selected: set[int] = set()
        counter = 0
        while len(selected) < count:
            for cell in _sparse_cells(
                namespace,
                (label, counter),
                width=width,
                count=count,
            ):
                if cell not in used:
                    selected.add(cell)
                    if len(selected) == count:
                        break
            counter += 1
        frozen = tuple(sorted(selected))
        used.update(frozen)
        rows.append(frozen)
    return tuple(rows)


TYPE_ASSEMBLIES = _allocate_disjoint(
    PHYSICAL_TYPES,
    width=TYPE_CELL_COUNT,
    count=TYPE_ACTIVE_CELL_COUNT,
    namespace="physical-type",
)
PHYSICAL_VOCABULARY_RECORD = _freeze(
    {
        "event_kind_cells": tuple(range(EVENT_KIND_COUNT)),
        "factor_kind_cells": tuple(range(len(FACTOR_KINDS))),
        "physical_type_assemblies": TYPE_ASSEMBLIES,
        "relation_operator_cells": tuple(range(len(RELATION_OPERATORS))),
        "rhs_kind_cells": (RHS_SLOT, RHS_VALUE),
        "raw_selector_directions": len(RAW_SELECTOR_DIRECTIONS),
        "raw_selector_cue_slots": len(RAW_SELECTOR_CUE_SLOTS),
        "semantic_labels_are_not_neural_inputs": True,
    }
)
PHYSICAL_VOCABULARY_SHA256 = _digest(
    _VOCABULARY_NAMESPACE, PHYSICAL_VOCABULARY_RECORD
)

RAW_SOURCE_SUCCESSOR_CONTRACT_RECORD = _freeze(
    {
        "scope": "PUBLIC_TRAIN_FIT_RAW_OCCURRENCE_SOURCE_ANNOTATION_V2",
        "required_record_count": EXPECTED_RAW_OBSERVATION_COUNT,
        "record_fields": (
            "teacher_join_input_root_sha256",
            "compiler_event_sha256",
            "node_occurrence_cells",
            "physical_span_event_sha256",
            "raw_span_assembly_cells",
            "direction",
            "cue_slot",
            "selected_slot",
        ),
        "offline_only_fields": (
            "teacher_join_input_root_sha256",
            "compiler_event_sha256",
        ),
        "invariants": (
            "emit-with-the-compiler-owned-structural-event",
            "publish-exactly-one-record-for-every-raw-backed-node-occurrence",
            "reuse-the-same-12-cell-node-occurrence-carrier",
            "direction-is-NEXT-or-PREV",
            "cue-slot-is-an-integer-from-zero-through-three",
            "selected-slot-is-the-directional-neighbor-of-cue-slot",
            "raw-span-assembly-is-the-authenticated-12-cell-public-observation",
            "teacher-join-and-compiler-event-hashes-are-never-neural-inputs",
        ),
        "forbidden_derivations": (
            "task",
            "family",
            "source-ordinal",
            "entrypoint",
            "program",
            "desired",
            "property",
            "dev",
            "heldout",
            "protected",
            "source-order",
            "fixed-index",
            "audit-hash-as-neural-evidence",
        ),
        "allocation_condition": "all-59-records-atomically-identifiable",
    }
)
RAW_SOURCE_SUCCESSOR_CONTRACT_SHA256 = _digest(
    _SUCCESSOR_NAMESPACE, RAW_SOURCE_SUCCESSOR_CONTRACT_RECORD
)


@dataclass(frozen=True, slots=True)
class _Relation:
    left_slot: int
    operator: str
    right_slot: int | None = None
    right_value: int | None = None


@dataclass(frozen=True, slots=True)
class _Node:
    slot: str
    parent: str | None
    child_position: int | None
    factors: tuple[tuple[str, str], ...]
    parent_occupancy: tuple[int, ...]
    source_kind: str
    source_key: str
    relations: tuple[_Relation, ...] = ()


@dataclass(frozen=True, slots=True)
class _Blueprint:
    root_production: str
    nodes: tuple[_Node, ...]
    source_factors: tuple[str, ...]
    expected_channel_counts: tuple[tuple[str, int], ...]


_LITERAL_INT = ("literal", "Int")
_ROLE_INT = ("role", "Int")
_ROLE_LIST_INT = ("role", "List[Int]")

_COUNT_BLUEPRINT = _Blueprint(
    root_production="count",
    nodes=(
        _Node(
            "root",
            None,
            None,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT, _ROLE_LIST_INT),
            (),
            "raw",
            "count",
        ),
        _Node(
            "predicate-binder",
            "root",
            0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2),
            "raw",
            "less-than",
        ),
        _Node(
            "greater-than-as-less-than",
            "predicate-binder",
            0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2),
            "raw",
            "less-than",
        ),
        _Node(
            "threshold-literal",
            "greater-than-as-less-than",
            0,
            (_LITERAL_INT,),
            (0,),
            "literal",
            "threshold-literal",
        ),
        _Node(
            "bound-add",
            "greater-than-as-less-than",
            1,
            (_LITERAL_INT, _ROLE_INT),
            (1, 2),
            "raw",
            "add",
            (_Relation(0, "ne", right_value=0),),
        ),
        _Node(
            "bound-item0",
            "bound-add",
            0,
            (_ROLE_INT,),
            (1,),
            "raw",
            "add",
        ),
        _Node(
            "offset-literal",
            "bound-add",
            1,
            (_LITERAL_INT,),
            (0,),
            "literal",
            "offset-literal",
        ),
        _Node(
            "collection-arg0",
            "root",
            1,
            (_ROLE_LIST_INT,),
            (3,),
            "raw",
            "map",
        ),
    ),
    source_factors=("add", "map", "count", "less-than"),
    expected_channel_counts=(
        ("root-production-family", 1),
        ("ordered-parent-child-edge", 7),
        ("argument-operand-orientation", 6),
        ("nesting-relation", 3),
        ("parameter-role-binding", 1),
        ("literal-slot-binding", 2),
        ("literal-value-copy", 2),
        ("terminal-root-eclass-agreement", 1),
    ),
)

_CLAMP_SUM_BLUEPRINT = _Blueprint(
    root_production="sum",
    nodes=(
        _Node(
            "root",
            None,
            None,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT, _ROLE_LIST_INT),
            (),
            "raw",
            "sum",
        ),
        _Node(
            "mapped-clamp",
            "root",
            0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT, _ROLE_LIST_INT),
            (0, 1, 2, 3),
            "raw",
            "map",
        ),
        _Node(
            "value-binder",
            "mapped-clamp",
            0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2),
            "raw",
            "map",
        ),
        _Node(
            "upper-extremum",
            "value-binder",
            0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2),
            "raw",
            "minimum",
            (_Relation(0, "le", right_slot=1),),
        ),
        _Node(
            "lower-extremum",
            "upper-extremum",
            0,
            (_LITERAL_INT, _ROLE_INT),
            (0, 2),
            "raw",
            "maximum",
        ),
        _Node(
            "bound-item0",
            "lower-extremum",
            0,
            (_ROLE_INT,),
            (1,),
            "raw",
            "maximum",
        ),
        _Node(
            "lower-literal",
            "lower-extremum",
            1,
            (_LITERAL_INT,),
            (0,),
            "literal",
            "lower-literal",
        ),
        _Node(
            "upper-literal",
            "upper-extremum",
            1,
            (_LITERAL_INT,),
            (1,),
            "literal",
            "upper-literal",
        ),
        _Node(
            "collection-arg0",
            "mapped-clamp",
            1,
            (_ROLE_LIST_INT,),
            (3,),
            "raw",
            "map",
        ),
    ),
    source_factors=("minimum", "maximum", "map", "sum"),
    expected_channel_counts=(
        ("root-production-family", 1),
        ("ordered-parent-child-edge", 8),
        ("argument-operand-orientation", 6),
        ("nesting-relation", 4),
        ("parameter-role-binding", 1),
        ("literal-slot-binding", 2),
        ("literal-value-copy", 2),
        ("terminal-root-eclass-agreement", 1),
    ),
)

_BLUEPRINTS = {
    _COUNT_BLUEPRINT.root_production: _COUNT_BLUEPRINT,
    _CLAMP_SUM_BLUEPRINT.root_production: _CLAMP_SUM_BLUEPRINT,
}


@dataclass(frozen=True, slots=True)
class FitIdentityAnnotationAudit:
    input_root_sha256: str
    multiplicity: int
    target_set_sha256: str
    topology_node_count: int
    edge_record_count: int
    factor_slot_record_count: int
    relation_record_count: int
    domain_record_count: int
    literal_copy_group_count: int
    signature_record_count: int
    potential_lesson_count: int
    stable_node_carrier_count: int
    stable_node_carrier_conflict_count: int
    exact_physical_span_join_count: int
    exact_typed_literal_event_join_count: int
    required_raw_observation_count: int
    raw_selector_candidate_count: int
    exact_raw_observation_join_count: int
    multiple_physical_span_candidate_count: int
    source_order_choice_count: int
    fixed_index_choice_count: int
    translation_conflict_count: int
    partial_physical_target_sha256: str
    missing_fields: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class OccurrenceFactorAnnotationStopLedger:
    status: str
    fit_identity_count: int
    fit_appearance_count: int
    target_class_count: int
    complete_identity_count: int
    stopped_identity_count: int
    topology_node_count: int
    edge_record_count: int
    factor_slot_record_count: int
    relation_record_count: int
    relation_operator_vocabulary_count: int
    domain_record_count: int
    literal_copy_group_count: int
    signature_record_count: int
    potential_lesson_count: int
    stable_node_carrier_count: int
    stable_node_carrier_conflict_count: int
    exact_physical_span_join_count: int
    exact_typed_literal_event_join_count: int
    required_raw_observation_count: int
    raw_selector_candidate_count: int
    exact_raw_observation_join_count: int
    missing_raw_direction_count: int
    missing_raw_cue_slot_count: int
    missing_raw_selected_slot_count: int
    multiple_physical_span_candidate_count: int
    source_order_choice_count: int
    fixed_index_choice_count: int
    translation_conflict_count: int
    emitted_target_record_count: int
    curriculum_lesson_count: int
    learner_array_count: int
    learning_run_count: int
    electrical_execution_count: int
    panel_spend_count: int
    artifact_export_count: int
    identity_audit_set_sha256: str
    sha256: str


def _verify_dependencies() -> None:
    if (
        _file_sha256(PROMPT_V5_SOURCE_PATH) != PROMPT_V5_SOURCE_SHA256
        or prompt_v5.PROTOCOL.protocol_sha256 != PROMPT_V5_PROTOCOL_SHA256
        or prompt_v5.SUCCESSOR_ANNOTATION_CONTRACT.sha256
        != SUCCESSOR_CONTRACT_SHA256
        or selection_v1.PROTOCOL.protocol_sha256 != SELECTION_V1_PROTOCOL_SHA256
        or structural_v1.CAPSULE.capsule_sha256
        != STRUCTURAL_ANNOTATION_CAPSULE_SHA256
        or input_capsule.ARTIFACT_SHA256 != INPUT_ARTIFACT_SHA256
        or input_capsule.CONTENT_ROOT != INPUT_CONTENT_ROOT
        or sha256(_PROMPT_SOURCE_PATH.read_bytes()).hexdigest()
        != PROMPT_SOURCE_SHA256
        or _file_sha256(PROMPT_OBSERVER_SOURCE_PATH)
        != PROMPT_OBSERVER_SOURCE_SHA256
        or _file_sha256(PLAN_SLICE_SOURCE_PATH) != PLAN_SLICE_SOURCE_SHA256
        or _file_sha256(OCCURRENCE_FACTOR_V6_SOURCE_PATH)
        != OCCURRENCE_FACTOR_V6_SOURCE_SHA256
    ):
        raise RuntimeError(STOP_DEPENDENCY)


_verify_dependencies()

DEPENDENCY_RECORD = _freeze(
    {
        "prompt_v5_source_sha256": PROMPT_V5_SOURCE_SHA256,
        "prompt_v5_protocol_sha256": PROMPT_V5_PROTOCOL_SHA256,
        "successor_contract_sha256": SUCCESSOR_CONTRACT_SHA256,
        "selection_v1_protocol_sha256": SELECTION_V1_PROTOCOL_SHA256,
        "structural_annotation_capsule_sha256": (
            STRUCTURAL_ANNOTATION_CAPSULE_SHA256
        ),
        "input_artifact_sha256": INPUT_ARTIFACT_SHA256,
        "input_content_root": INPUT_CONTENT_ROOT,
        "prompt_source_sha256": PROMPT_SOURCE_SHA256,
        "prompt_observer_source_sha256": PROMPT_OBSERVER_SOURCE_SHA256,
        "plan_slice_source_sha256": PLAN_SLICE_SOURCE_SHA256,
        "occurrence_factor_v6_commit": OCCURRENCE_FACTOR_V6_COMMIT,
        "occurrence_factor_v6_source_sha256": (
            OCCURRENCE_FACTOR_V6_SOURCE_SHA256
        ),
        "occurrence_factor_v6_module_imported": False,
        "compiler_annotations_are_offline_teacher_data_only": True,
    }
)
DEPENDENCY_SHA256 = _digest(_DEPENDENCY_NAMESPACE, DEPENDENCY_RECORD)


def _list_int_to_int(signature: TypedSignature) -> bool:
    if len(signature.parameters) != 1:
        return False
    return (
        signature.parameters[0].type.to_record()
        == {
            "constructor": "List",
            "arguments": [{"constructor": "Int", "arguments": []}],
        }
        and signature.result.to_record()
        == {"constructor": "Int", "arguments": []}
    )


def _observed_fit_appearances() -> Mapping[str, tuple[object, ...]]:
    raw = _PROMPT_SOURCE_PATH.read_bytes()
    source = json.loads(raw)
    if (
        set(source) != {"records", "schema", "scope"}
        or len(source["records"]) != 90
    ):
        raise RuntimeError(STOP_DEPENDENCY)
    canonical_by_prompt = {
        row.prompt_input_sha256: row
        for row in selection_v1.PROTOCOL.canonical_inputs
    }
    grouped: dict[str, list[object]] = defaultdict(list)
    for raw_record in source["records"]:
        signature = TypedSignature.from_record(raw_record["signature"])
        if not _list_int_to_int(signature):
            continue
        observed = prompt_bridge.observe_public_train_prompt(
            raw_record["prompt"], signature
        )
        canonical = canonical_by_prompt.get(observed.sha256)
        if canonical is None:
            raise RuntimeError(STOP_DEPENDENCY)
        if canonical.split == FIT:
            grouped[canonical.causal_key_sha256].append(observed)
    frozen = {root: tuple(rows) for root, rows in grouped.items()}
    if len(frozen) != FIT_IDENTITY_COUNT or sum(map(len, frozen.values())) != FIT_APPEARANCE_COUNT:
        raise RuntimeError(STOP_DEPENDENCY)
    return MappingProxyType(frozen)


def _root_blueprint(annotation: object) -> _Blueprint:
    root_labels = tuple(
        label
        for label in annotation.relay_labels
        if label.channel == "root-production-family"
    )
    if len(root_labels) != 1:
        raise RuntimeError(STOP_DEPENDENCY)
    root_production = str(root_labels[0].fields["root_production"])
    blueprint = _BLUEPRINTS.get(root_production)
    if blueprint is None:
        raise RuntimeError(STOP_DEPENDENCY)
    return blueprint


def _validate_structure(annotation: object, blueprint: _Blueprint) -> None:
    channel_counts = Counter(label.channel for label in annotation.relay_labels)
    observed_counts = tuple(
        (channel, channel_counts[channel])
        for channel, _ in blueprint.expected_channel_counts
    )
    if observed_counts != blueprint.expected_channel_counts:
        raise RuntimeError(STOP_DEPENDENCY)
    expected_edges = {
        (node.parent, node.child_position, node.slot)
        for node in blueprint.nodes
        if node.parent is not None
    }
    actual_edges = {
        (
            str(label.fields["parent_slot"]),
            int(label.fields["child_ordinal"]),
            str(label.fields["child_slot"]),
        )
        for label in annotation.relay_labels
        if label.channel == "ordered-parent-child-edge"
    }
    if actual_edges != expected_edges:
        raise RuntimeError(STOP_DEPENDENCY)

    literal_slots = {
        str(label.fields["semantic_literal_slot"])
        for label in annotation.relay_labels
        if label.channel == "literal-slot-binding"
    }
    copy_slots = {
        str(label.fields["semantic_literal_slot"])
        for label in annotation.relay_labels
        if label.channel == "literal-value-copy"
    }
    expected_literals = {
        node.slot for node in blueprint.nodes if node.source_kind == "literal"
    }
    if literal_slots != expected_literals or copy_slots != expected_literals:
        raise RuntimeError(STOP_DEPENDENCY)


def _physical_span_record(span: object) -> dict[str, object]:
    return {
        "start_token": span.start_token,
        "stop_token": span.stop_token,
        "first_spike_step": span.first_spike_step,
        "assembly_cells": tuple(span.assembly_cells),
        "spike_cells": tuple(span.spike_cells),
    }


def _typed_literal_record(literal: object) -> dict[str, object]:
    return {
        "canonical_token_index": literal.canonical_token_index,
        "typed_output_active_cells": tuple(literal.typed_output_active_cells),
    }


def _build_identity_audit(
    root: str,
    appearances: tuple[object, ...],
    frozen_input: object,
    canonical: object,
    annotation: object,
    factor_annotation: object,
) -> FitIdentityAnnotationAudit:
    blueprint = _root_blueprint(annotation)
    _validate_structure(annotation, blueprint)
    if {target.factor for target in factor_annotation.targets} != set(
        blueprint.source_factors
    ):
        raise RuntimeError(STOP_DEPENDENCY)
    if any(observed.sha256 != canonical.prompt_input_sha256 for observed in appearances):
        raise RuntimeError(STOP_DEPENDENCY)

    exemplar = appearances[0]
    if any(
        tuple(_physical_span_record(span) for span in observed.spans)
        != tuple(_physical_span_record(span) for span in exemplar.spans)
        or tuple(_typed_literal_record(item) for item in observed.signed_literals)
        != tuple(_typed_literal_record(item) for item in exemplar.signed_literals)
        for observed in appearances[1:]
    ):
        raise RuntimeError(STOP_DEPENDENCY)

    frozen_span_records = tuple(_plain(row) for row in frozen_input.raw_span_records)
    frozen_literal_records = tuple(
        _plain(row) for row in frozen_input.typed_literal_records
    )
    observed_span_records = tuple(_physical_span_record(span) for span in exemplar.spans)
    observed_literal_records = tuple(
        _typed_literal_record(item) for item in exemplar.signed_literals
    )
    if (
        _plain(observed_span_records) != _plain(frozen_span_records)
        or _plain(observed_literal_records) != _plain(frozen_literal_records)
    ):
        raise RuntimeError(STOP_DEPENDENCY)

    literal_binding_by_slot: dict[str, int] = {}
    for label in annotation.relay_labels:
        if label.channel != "literal-slot-binding":
            continue
        source = label.fields["literal_source_ordinal"]
        slot = str(label.fields["semantic_literal_slot"])
        if type(source) is not int or slot in literal_binding_by_slot:
            raise RuntimeError(STOP_DEPENDENCY)
        literal_binding_by_slot[slot] = source

    factor_target_by_name = {target.factor: target for target in factor_annotation.targets}
    if len(factor_target_by_name) != len(factor_annotation.targets):
        raise RuntimeError(STOP_DEPENDENCY)

    carriers = {
        node.slot: _sparse_cells(
            "stable-node-carrier",
            (root, node.slot),
            width=NODE_CARRIER_CELL_COUNT,
            count=NODE_CARRIER_ACTIVE_CELL_COUNT,
        )
        for node in blueprint.nodes
    }
    carrier_conflicts = len(carriers) - len(set(carriers.values()))

    source_records: list[dict[str, object]] = []
    exact_span_joins = 0
    exact_literal_joins = 0
    raw_required = 0
    multiple_span_candidates = 0
    for node in blueprint.nodes:
        if node.source_kind == "raw":
            raw_required += 1
            target = factor_target_by_name[node.source_key]
            candidates = tuple(
                span
                for span in exemplar.spans
                if span.start_token == target.start_token
                and span.stop_token == target.stop_token
                and span.identity_sha256 == target.span_identity_sha256
            )
            multiple_span_candidates += int(len(candidates) > 1)
            if len(candidates) != 1:
                raise RuntimeError(STOP_DEPENDENCY)
            span_record = _physical_span_record(candidates[0])
            exact_span_joins += 1
            source_records.append(
                {
                    "occurrence": blueprint.nodes.index(node),
                    "carrier_cells": carriers[node.slot],
                    "source_kind_cell": 0,
                    "physical_span_event_sha256": _digest(
                        _SPAN_EVENT_NAMESPACE, span_record
                    ),
                    "physical_event_cells": span_record["spike_cells"],
                    "raw_observation_direction": None,
                    "raw_observation_cue_slot": None,
                    "raw_observation_selected_slot": None,
                    "authorized_geometry_count": 0,
                    "candidate_geometry_count": RAW_SELECTOR_GEOMETRY_COUNT,
                }
            )
        else:
            source = literal_binding_by_slot.get(node.source_key)
            if source is None or not 0 <= source < len(exemplar.signed_literals):
                raise RuntimeError(STOP_DEPENDENCY)
            literal = exemplar.signed_literals[source]
            literal_record = _typed_literal_record(literal)
            matches = tuple(
                record
                for record in frozen_literal_records
                if _plain(record) == _plain(literal_record)
            )
            if len(matches) != 1:
                raise RuntimeError(STOP_DEPENDENCY)
            exact_literal_joins += 1
            source_records.append(
                {
                    "occurrence": blueprint.nodes.index(node),
                    "carrier_cells": carriers[node.slot],
                    "source_kind_cell": 1,
                    "typed_literal_event_sha256": _digest(
                        _LITERAL_EVENT_NAMESPACE, literal_record
                    ),
                    "physical_event_cells": literal_record[
                        "typed_output_active_cells"
                    ],
                }
            )

    slot_index = {node.slot: index for index, node in enumerate(blueprint.nodes)}
    edges = tuple(
        (
            carriers[node.parent],
            carriers[node.slot],
            node.child_position,
            node.parent_occupancy,
        )
        for node in blueprint.nodes
        if node.parent is not None
    )
    factor_slots = tuple(
        (
            carriers[node.slot],
            local_slot,
            FACTOR_KINDS.index(kind),
            TYPE_ASSEMBLIES[PHYSICAL_TYPES.index(value_type)],
        )
        for node in blueprint.nodes
        for local_slot, (kind, value_type) in enumerate(node.factors)
    )
    relations = tuple(
        (
            carriers[node.slot],
            relation.left_slot,
            RELATION_OPERATORS.index(relation.operator),
            RHS_SLOT if relation.right_slot is not None else RHS_VALUE,
            relation.right_slot,
            (
                ()
                if relation.right_value is None
                else output_cells.TypedOutputCellBank.encode(
                    relation.right_value,
                    declared_type=output_cells.OutputType.INT,
                ).active_global_cell_indices()
            ),
        )
        for node in blueprint.nodes
        for relation in node.relations
    )
    literal_nodes = tuple(
        node for node in blueprint.nodes if node.source_kind == "literal"
    )
    literal_source_records = {
        record["occurrence"]: record
        for record in source_records
        if record["source_kind_cell"] == 1
    }
    domains = tuple(
        (
            carriers[node.slot],
            0,
            TYPE_ASSEMBLIES[PHYSICAL_TYPES.index("Int")],
            literal_source_records[slot_index[node.slot]]["physical_event_cells"],
        )
        for node in literal_nodes
    )
    copy_groups = tuple(
        (
            _sparse_cells(
                "literal-copy-group",
                (root, group_index),
                width=COPY_GROUP_CELL_COUNT,
                count=COPY_GROUP_ACTIVE_CELL_COUNT,
            ),
            ((carriers[node.slot], 0),),
        )
        for group_index, node in enumerate(literal_nodes)
    )
    signatures = (
        (carriers["root"], 0, 0, TYPE_ASSEMBLIES[PHYSICAL_TYPES.index("List[Int]")]),
        (carriers["root"], 1, 0, TYPE_ASSEMBLIES[PHYSICAL_TYPES.index("Int")]),
    )
    potential_lesson_count = (
        len(blueprint.nodes)
        + len(edges)
        + len(factor_slots)
        + len(relations)
        + len(domains)
        + len(copy_groups)
        + len(signatures)
    )
    partial_target = {
        "root_carrier_cells": carriers["root"],
        "source_records": tuple(source_records),
        "edges": edges,
        "factor_slots": factor_slots,
        "relations": relations,
        "domains": domains,
        "literal_copy_groups": copy_groups,
        "signature_relays": signatures,
        "potential_lesson_count": potential_lesson_count,
        "semantic_labels_present": False,
        "audit_hashes_are_neural_inputs": False,
    }
    record = {
        "input_root_sha256": root,
        "multiplicity": frozen_input.multiplicity,
        "target_set_sha256": annotation.target_set_sha256,
        "topology_node_count": len(blueprint.nodes),
        "edge_record_count": len(edges),
        "factor_slot_record_count": len(factor_slots),
        "relation_record_count": len(relations),
        "domain_record_count": len(domains),
        "literal_copy_group_count": len(copy_groups),
        "signature_record_count": len(signatures),
        "potential_lesson_count": potential_lesson_count,
        "stable_node_carrier_count": len(carriers),
        "stable_node_carrier_conflict_count": carrier_conflicts,
        "exact_physical_span_join_count": exact_span_joins,
        "exact_typed_literal_event_join_count": exact_literal_joins,
        "required_raw_observation_count": raw_required,
        "raw_selector_candidate_count": raw_required
        * RAW_SELECTOR_GEOMETRY_COUNT,
        "exact_raw_observation_join_count": 0,
        "multiple_physical_span_candidate_count": multiple_span_candidates,
        "source_order_choice_count": 0,
        "fixed_index_choice_count": 0,
        "translation_conflict_count": carrier_conflicts,
        "partial_physical_target_sha256": _digest(
            _PARTIAL_TARGET_NAMESPACE, partial_target
        ),
        "missing_fields": (
            "raw_observation.direction",
            "raw_observation.cue_slot",
            "raw_observation.selected_slot",
        ),
    }
    return FitIdentityAnnotationAudit(
        **record,
        sha256=_digest(_IDENTITY_AUDIT_NAMESPACE, record),
    )


def _build_identity_audits() -> tuple[FitIdentityAnnotationAudit, ...]:
    appearances_by_root = _observed_fit_appearances()
    input_state = input_capsule.load_input_capsule()
    canonical_by_root = {
        row.causal_key_sha256: row
        for row in selection_v1.PROTOCOL.canonical_inputs
        if row.split == FIT
    }
    annotation_by_root = {
        row.causal_key_sha256: row for row in structural_v1.CAPSULE.annotations
    }
    factor_by_prompt = {
        row.prompt_input_sha256: row
        for row in selection_v1.PROTOCOL.factor_annotations
    }
    audits: list[FitIdentityAnnotationAudit] = []
    for root in sorted(appearances_by_root):
        canonical = canonical_by_root.get(root)
        annotation = annotation_by_root.get(root)
        if canonical is None or annotation is None:
            raise RuntimeError(STOP_DEPENDENCY)
        audits.append(
            _build_identity_audit(
                root,
                appearances_by_root[root],
                input_state.lookup(root),
                canonical,
                annotation,
                factor_by_prompt[canonical.prompt_input_sha256],
            )
        )
    return tuple(audits)


FIT_IDENTITY_AUDITS = _build_identity_audits()
IDENTITY_AUDIT_SET_SHA256 = _digest(
    _AUDIT_SET_NAMESPACE, tuple(row.sha256 for row in FIT_IDENTITY_AUDITS)
)


def _aggregate_carrier_conflict_count() -> int:
    annotation_by_root = {
        row.causal_key_sha256: row for row in structural_v1.CAPSULE.annotations
    }
    carriers = tuple(
        _sparse_cells(
            "stable-node-carrier",
            (audit.input_root_sha256, node.slot),
            width=NODE_CARRIER_CELL_COUNT,
            count=NODE_CARRIER_ACTIVE_CELL_COUNT,
        )
        for audit in FIT_IDENTITY_AUDITS
        for node in _root_blueprint(annotation_by_root[audit.input_root_sha256]).nodes
    )
    if len(carriers) != sum(row.topology_node_count for row in FIT_IDENTITY_AUDITS):
        raise RuntimeError(STOP_DEPENDENCY)
    return len(carriers) - len(set(carriers))


AGGREGATE_STABLE_NODE_CARRIER_CONFLICT_COUNT = _aggregate_carrier_conflict_count()


def _build_stop_ledger() -> OccurrenceFactorAnnotationStopLedger:
    rows = FIT_IDENTITY_AUDITS
    record = {
        "status": STOP_RAW_SELECTOR,
        "fit_identity_count": len(rows),
        "fit_appearance_count": sum(row.multiplicity for row in rows),
        "target_class_count": len({row.target_set_sha256 for row in rows}),
        "complete_identity_count": 0,
        "stopped_identity_count": len(rows),
        "topology_node_count": sum(row.topology_node_count for row in rows),
        "edge_record_count": sum(row.edge_record_count for row in rows),
        "factor_slot_record_count": sum(
            row.factor_slot_record_count for row in rows
        ),
        "relation_record_count": sum(row.relation_record_count for row in rows),
        "relation_operator_vocabulary_count": len(RELATION_OPERATORS),
        "domain_record_count": sum(row.domain_record_count for row in rows),
        "literal_copy_group_count": sum(
            row.literal_copy_group_count for row in rows
        ),
        "signature_record_count": sum(
            row.signature_record_count for row in rows
        ),
        "potential_lesson_count": sum(
            row.potential_lesson_count for row in rows
        ),
        "stable_node_carrier_count": sum(
            row.stable_node_carrier_count for row in rows
        ),
        "stable_node_carrier_conflict_count": (
            AGGREGATE_STABLE_NODE_CARRIER_CONFLICT_COUNT
        ),
        "exact_physical_span_join_count": sum(
            row.exact_physical_span_join_count for row in rows
        ),
        "exact_typed_literal_event_join_count": sum(
            row.exact_typed_literal_event_join_count for row in rows
        ),
        "required_raw_observation_count": sum(
            row.required_raw_observation_count for row in rows
        ),
        "raw_selector_candidate_count": sum(
            row.raw_selector_candidate_count for row in rows
        ),
        "exact_raw_observation_join_count": sum(
            row.exact_raw_observation_join_count for row in rows
        ),
        "missing_raw_direction_count": sum(
            row.required_raw_observation_count for row in rows
        ),
        "missing_raw_cue_slot_count": sum(
            row.required_raw_observation_count for row in rows
        ),
        "missing_raw_selected_slot_count": sum(
            row.required_raw_observation_count for row in rows
        ),
        "multiple_physical_span_candidate_count": sum(
            row.multiple_physical_span_candidate_count for row in rows
        ),
        "source_order_choice_count": sum(
            row.source_order_choice_count for row in rows
        ),
        "fixed_index_choice_count": sum(
            row.fixed_index_choice_count for row in rows
        ),
        "translation_conflict_count": AGGREGATE_STABLE_NODE_CARRIER_CONFLICT_COUNT,
        "emitted_target_record_count": EMITTED_TARGET_RECORD_COUNT,
        "curriculum_lesson_count": CURRICULUM_LESSON_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "panel_spend_count": PANEL_SPEND_COUNT,
        "artifact_export_count": ARTIFACT_EXPORT_COUNT,
        "identity_audit_set_sha256": IDENTITY_AUDIT_SET_SHA256,
    }
    return OccurrenceFactorAnnotationStopLedger(
        **record,
        sha256=_digest(_STOP_NAMESPACE, record),
    )


STOP_LEDGER = _build_stop_ledger()

SURFACE_BOUNDARY_RECORD = _freeze(
    {
        "data_only": True,
        "compiler_semantic_labels_are_offline_only": True,
        "semantic_label_neural_input_count": 0,
        "audit_hash_neural_input_count": 0,
        "task_field_count": TASK_FIELD_COUNT,
        "family_field_count": FAMILY_FIELD_COUNT,
        "source_ordinal_neural_field_count": SOURCE_ORDINAL_NEURAL_FIELD_COUNT,
        "entrypoint_neural_field_count": ENTRYPOINT_NEURAL_FIELD_COUNT,
        "program_field_count": PROGRAM_FIELD_COUNT,
        "desired_field_count": DESIRED_FIELD_COUNT,
        "property_field_count": PROPERTY_FIELD_COUNT,
        "dev_field_count": DEV_FIELD_COUNT,
        "protected_field_count": PROTECTED_FIELD_COUNT,
        "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
        "heldout_input_event_read_count": HELDOUT_INPUT_EVENT_READ_COUNT,
        "v4_weight_read_count": V4_WEIGHT_READ_COUNT,
        "v4_checkpoint_read_count": V4_CHECKPOINT_READ_COUNT,
        "v5_weight_read_count": V5_WEIGHT_READ_COUNT,
        "v5_checkpoint_read_count": V5_CHECKPOINT_READ_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "panel_spend_count": PANEL_SPEND_COUNT,
        "artifact_export_count": ARTIFACT_EXPORT_COUNT,
        "emitted_target_record_count": EMITTED_TARGET_RECORD_COUNT,
        "source_order_choice_count": 0,
        "fixed_index_choice_count": 0,
    }
)
SURFACE_BOUNDARY_SHA256 = _digest(
    _SURFACE_NAMESPACE, SURFACE_BOUNDARY_RECORD
)


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV1:
    scope: str
    status: str
    dependency_sha256: str
    successor_contract_sha256: str
    physical_vocabulary_sha256: str
    identity_audit_set_sha256: str
    stop_ledger_sha256: str
    surface_boundary_sha256: str
    raw_source_successor_contract_sha256: str
    fit_identity_audits: tuple[FitIdentityAnnotationAudit, ...]
    capsule_sha256: str

    def root_record(self, *, include_capsule_sha256: bool = True) -> dict[str, object]:
        record = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "successor_contract_sha256": self.successor_contract_sha256,
            "physical_vocabulary_sha256": self.physical_vocabulary_sha256,
            "identity_audit_set_sha256": self.identity_audit_set_sha256,
            "stop_ledger_sha256": self.stop_ledger_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "raw_source_successor_contract_sha256": (
                self.raw_source_successor_contract_sha256
            ),
            "fit_identity_count": len(self.fit_identity_audits),
            "emitted_target_record_count": EMITTED_TARGET_RECORD_COUNT,
            "artifact_export_count": ARTIFACT_EXPORT_COUNT,
        }
        if include_capsule_sha256:
            record["capsule_sha256"] = self.capsule_sha256
        return record


def _build_capsule() -> FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV1:
    record = {
        "scope": PROTOCOL_SCOPE,
        "status": STOP_RAW_SELECTOR,
        "dependency_sha256": DEPENDENCY_SHA256,
        "successor_contract_sha256": SUCCESSOR_CONTRACT_SHA256,
        "physical_vocabulary_sha256": PHYSICAL_VOCABULARY_SHA256,
        "identity_audit_set_sha256": IDENTITY_AUDIT_SET_SHA256,
        "stop_ledger_sha256": STOP_LEDGER.sha256,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "raw_source_successor_contract_sha256": (
            RAW_SOURCE_SUCCESSOR_CONTRACT_SHA256
        ),
        "fit_identity_count": len(FIT_IDENTITY_AUDITS),
        "emitted_target_record_count": EMITTED_TARGET_RECORD_COUNT,
        "artifact_export_count": ARTIFACT_EXPORT_COUNT,
    }
    return FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV1(
        scope=PROTOCOL_SCOPE,
        status=STOP_RAW_SELECTOR,
        dependency_sha256=DEPENDENCY_SHA256,
        successor_contract_sha256=SUCCESSOR_CONTRACT_SHA256,
        physical_vocabulary_sha256=PHYSICAL_VOCABULARY_SHA256,
        identity_audit_set_sha256=IDENTITY_AUDIT_SET_SHA256,
        stop_ledger_sha256=STOP_LEDGER.sha256,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        raw_source_successor_contract_sha256=(
            RAW_SOURCE_SUCCESSOR_CONTRACT_SHA256
        ),
        fit_identity_audits=FIT_IDENTITY_AUDITS,
        capsule_sha256=_digest(_CAPSULE_NAMESPACE, record),
    )


CAPSULE = _build_capsule()


__all__ = [
    "AGGREGATE_STABLE_NODE_CARRIER_CONFLICT_COUNT",
    "ARTIFACT_EXPORT_COUNT",
    "CAPSULE",
    "CURRICULUM_LESSON_COUNT",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "EMITTED_TARGET_RECORD_COUNT",
    "FIT_IDENTITY_AUDITS",
    "IDENTITY_AUDIT_SET_SHA256",
    "PHYSICAL_VOCABULARY_RECORD",
    "PHYSICAL_VOCABULARY_SHA256",
    "PROTOCOL_SCOPE",
    "RAW_SELECTOR_GEOMETRY_COUNT",
    "RAW_SOURCE_SUCCESSOR_CONTRACT_RECORD",
    "RAW_SOURCE_SUCCESSOR_CONTRACT_SHA256",
    "STOP_LEDGER",
    "STOP_RAW_SELECTOR",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "FitIdentityAnnotationAudit",
    "FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV1",
    "OccurrenceFactorAnnotationStopLedger",
]
