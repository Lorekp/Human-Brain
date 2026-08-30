"""Producer-owned FIT occurrence/factor annotation capsule V2.

V1 proved that the public FIT compiler surface identifies the retained
occurrence graphs, factors, relations, literal domains, copy groups, typed
signatures, and exact physical source spans, but not a selector geometry for
raw sources.  V2 is a fresh teacher cohort: selector geometry is authored by
the raw structural-event producer and emitted atomically with that event.

The published neural surface is operator-name free.  Compiler labels and
causal roots are used only to construct and authenticate offline FIT teacher
records.  Held-out rows, desired values, programs, checkpoints, and learned
weights are neither read nor emitted.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_annotation_capsule_v1 as predecessor_v1,
)
from experiments.r3_pilot import (
    public_train_list_int_to_int_prompt_input_capsule_v1 as input_capsule,
)
from experiments.r3_pilot import (
    public_train_prompt_selection_freeze_v1_protocol as selection_v1,
)
from experiments.r3_pilot import (
    public_train_structural_annotation_capsule_v1 as structural_v1,
)
from experiments.r3_pilot import public_typed_output_cells as output_cells

PROTOCOL_SCOPE = "PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V2"
READY = "READY_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V2"
STOP_DEPENDENCY = (
    "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V2_DEPENDENCY"
)
STOP_INCOMPLETE = (
    "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V2_INCOMPLETE"
)
FIT = "FIT"

PREDECESSOR_SOURCE_PATH = (
    "experiments/r3_pilot/"
    "public_train_fit_occurrence_factor_annotation_capsule_v1.py"
)
PREDECESSOR_SOURCE_SHA256 = (
    "86fa1e23e2d9c2bdab3751353932168ec95e561af64f35535221eec5b0f7bbc8"
)
PREDECESSOR_COMMIT = "e2cc14a05354c7f80b8dea34234d45c8596f3209"
PREDECESSOR_CAPSULE_SHA256 = (
    "f0e1bf69b3943c7c09fa7f7322d5a91d3246db651e8ce898030b54ae6393235b"
)
PREDECESSOR_STOP_SHA256 = (
    "39f02d04249bbb6cb52c2d36f76a72f705ce5871f4899568377784243325e7de"
)
PREDECESSOR_AUDIT_SET_SHA256 = (
    "aa40aed56c2e43b89cce44cd9fa1d7ac5ba883b76b4641263b78e71def4a3669"
)
PREDECESSOR_SUCCESSOR_CONTRACT_SHA256 = (
    "fb11a848948255a8c98b10d1b50ad28e494da2082b8eab990c88de00c9c145bc"
)
PREDECESSOR_STATUS = (
    "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_ANNOTATION_CAPSULE_V1_"
    "RAW_SPAN_SELECTOR_GEOMETRY_UNAVAILABLE"
)

SELECTION_PROTOCOL_SHA256 = (
    "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"
)
STRUCTURAL_CAPSULE_SHA256 = (
    "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf"
)
INPUT_ARTIFACT_SHA256 = (
    "0e90318d1c1d3bd365f9812fb3c43aa5f0806dbfa5a1e81cb527c8e7f2c20dcf"
)
INPUT_CONTENT_ROOT = (
    "a7ff53818cf4e83c823e6b23f5547e0e7ee1fd30ed73dd136e3672063ea7aca8"
)

FIT_IDENTITY_COUNT = 9
FIT_APPEARANCE_COUNT = 12
TARGET_SHAPE_COUNT = 2
EXPECTED_SOURCE_COUNT = 77
EXPECTED_RAW_SOURCE_COUNT = 59
EXPECTED_LITERAL_SOURCE_COUNT = 18
EXPECTED_EDGE_COUNT = 68
EXPECTED_FACTOR_COUNT = 164
EXPECTED_RELATION_COUNT = 9
EXPECTED_DOMAIN_COUNT = 18
EXPECTED_COPY_GROUP_COUNT = 18
EXPECTED_SIGNATURE_COUNT = 18
EXPECTED_TARGET_RECORD_COUNT = 372

EVENT_SOURCE = 1
EVENT_EDGE = 2
EVENT_FACTOR = 3
EVENT_RELATION = 4
EVENT_DOMAIN = 5
EVENT_COPY = 6
EVENT_SIGNATURE = 7
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
RAW_SPAN_ACTIVE_CELL_COUNT = 12

LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
CHECKPOINT_READ_COUNT = 0
WEIGHT_READ_COUNT = 0
HELDOUT_TARGET_READ_COUNT = 0
HELDOUT_INPUT_LOOKUP_COUNT = 0
PANEL_SPEND_COUNT = 0
DESIRED_READ_COUNT = 0
PROGRAM_FIELD_COUNT = 0
TASK_FIELD_COUNT = 0
FAMILY_FIELD_COUNT = 0
SOURCE_ORDER_GEOMETRY_CHOICE_COUNT = 0
FIXED_CANDIDATE_GEOMETRY_CHOICE_COUNT = 0
HASH_GEOMETRY_CHOICE_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]
_NAMESPACE = b"public-train-fit-occurrence-factor-annotation-capsule-v2\0"
_DEPENDENCY_NAMESPACE = _NAMESPACE + b"dependency\0"
_TYPE_NAMESPACE = _NAMESPACE + b"type\0"
_SPAN_NAMESPACE = _NAMESPACE + b"span-event\0"
_LITERAL_NAMESPACE = _NAMESPACE + b"literal-event\0"
_PRODUCER_NAMESPACE = _NAMESPACE + b"producer-provenance\0"
_EVENT_NAMESPACE = _NAMESPACE + b"compiler-event\0"
_TARGET_NAMESPACE = _NAMESPACE + b"identity-target\0"
_TARGET_SET_NAMESPACE = _NAMESPACE + b"target-set\0"
_SURFACE_NAMESPACE = _NAMESPACE + b"surface\0"
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


def _sparse_cells(
    label: str,
    key: object,
    *,
    width: int,
    count: int,
) -> tuple[int, ...]:
    """Allocate a stable physical identity; the key itself is never emitted."""

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


def _allocate_disjoint_types() -> tuple[tuple[int, ...], ...]:
    used: set[int] = set()
    rows: list[tuple[int, ...]] = []
    for type_name in PHYSICAL_TYPES:
        selected: set[int] = set()
        counter = 0
        while len(selected) < TYPE_ACTIVE_CELL_COUNT:
            candidates = _sparse_cells(
                "physical-type",
                (type_name, counter),
                width=TYPE_CELL_COUNT,
                count=TYPE_ACTIVE_CELL_COUNT,
            )
            selected.update(cell for cell in candidates if cell not in used)
            counter += 1
        frozen = tuple(sorted(selected))[:TYPE_ACTIVE_CELL_COUNT]
        used.update(frozen)
        rows.append(frozen)
    return tuple(rows)


TYPE_ASSEMBLIES = _allocate_disjoint_types()
TYPE_VOCABULARY_SHA256 = _digest(_TYPE_NAMESPACE, TYPE_ASSEMBLIES)


@dataclass(frozen=True, slots=True)
class _RelationSpec:
    left_slot: int
    operator: str
    right_slot: int | None = None
    right_value: int | None = None


@dataclass(frozen=True, slots=True)
class _NodeSpec:
    key: str
    parent: str | None
    child_position: int | None
    factors: tuple[tuple[str, str], ...]
    parent_occupancy: tuple[int, ...]
    source_kind: str
    source_key: str
    selector: tuple[str, int, int] | None
    relations: tuple[_RelationSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class _TopologySpec:
    compiler_root_label: str
    nodes: tuple[_NodeSpec, ...]
    source_factors: tuple[str, ...]
    expected_channel_counts: tuple[tuple[str, int], ...]


_LITERAL_INT = ("literal", "Int")
_ROLE_INT = ("role", "Int")
_ROLE_LIST_INT = ("role", "List[Int]")

# Selector triples are direct producer fields, not indices into an enumerated
# candidate list.  Repeated lexical spans receive distinct occurrence-local
# geometry without consulting source order, hashes, or semantic outputs.
_COUNT_TOPOLOGY = _TopologySpec(
    compiler_root_label="count",
    nodes=(
        _NodeSpec(
            "root", None, None,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT, _ROLE_LIST_INT),
            (), "raw", "count", ("NEXT", 0, 1),
        ),
        _NodeSpec(
            "predicate-binder", "root", 0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2), "raw", "less-than", ("PREV", 1, 0),
        ),
        _NodeSpec(
            "greater-than-as-less-than", "predicate-binder", 0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2), "raw", "less-than", ("NEXT", 1, 2),
        ),
        _NodeSpec(
            "threshold-literal", "greater-than-as-less-than", 0,
            (_LITERAL_INT,), (0,), "literal", "threshold-literal", None,
        ),
        _NodeSpec(
            "bound-add", "greater-than-as-less-than", 1,
            (_LITERAL_INT, _ROLE_INT), (1, 2), "raw", "add",
            ("PREV", 2, 1), (_RelationSpec(0, "ne", right_value=0),),
        ),
        _NodeSpec(
            "bound-item0", "bound-add", 0,
            (_ROLE_INT,), (1,), "raw", "add", ("NEXT", 2, 3),
        ),
        _NodeSpec(
            "offset-literal", "bound-add", 1,
            (_LITERAL_INT,), (0,), "literal", "offset-literal", None,
        ),
        _NodeSpec(
            "collection-arg0", "root", 1,
            (_ROLE_LIST_INT,), (3,), "raw", "map", ("PREV", 3, 2),
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

_CLAMP_SUM_TOPOLOGY = _TopologySpec(
    compiler_root_label="sum",
    nodes=(
        _NodeSpec(
            "root", None, None,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT, _ROLE_LIST_INT),
            (), "raw", "sum", ("NEXT", 0, 1),
        ),
        _NodeSpec(
            "mapped-clamp", "root", 0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT, _ROLE_LIST_INT),
            (0, 1, 2, 3), "raw", "map", ("PREV", 1, 0),
        ),
        _NodeSpec(
            "value-binder", "mapped-clamp", 0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2), "raw", "map", ("NEXT", 1, 2),
        ),
        _NodeSpec(
            "upper-extremum", "value-binder", 0,
            (_LITERAL_INT, _LITERAL_INT, _ROLE_INT),
            (0, 1, 2), "raw", "minimum", ("PREV", 2, 1),
            (_RelationSpec(0, "le", right_slot=1),),
        ),
        _NodeSpec(
            "lower-extremum", "upper-extremum", 0,
            (_LITERAL_INT, _ROLE_INT), (0, 2), "raw", "maximum",
            ("NEXT", 2, 3),
        ),
        _NodeSpec(
            "bound-item0", "lower-extremum", 0,
            (_ROLE_INT,), (1,), "raw", "maximum", ("PREV", 3, 2),
        ),
        _NodeSpec(
            "lower-literal", "lower-extremum", 1,
            (_LITERAL_INT,), (0,), "literal", "lower-literal", None,
        ),
        _NodeSpec(
            "upper-literal", "upper-extremum", 1,
            (_LITERAL_INT,), (1,), "literal", "upper-literal", None,
        ),
        _NodeSpec(
            "collection-arg0", "mapped-clamp", 1,
            (_ROLE_LIST_INT,), (3,), "raw", "map", ("NEXT", 3, 0),
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

_TOPOLOGIES = {
    _COUNT_TOPOLOGY.compiler_root_label: _COUNT_TOPOLOGY,
    _CLAMP_SUM_TOPOLOGY.compiler_root_label: _CLAMP_SUM_TOPOLOGY,
}


@dataclass(frozen=True, slots=True)
class RawSpanObservationTarget:
    physical_span_event_sha256: str
    assembly_cells: tuple[int, ...]
    direction: str
    cue_slot: int
    selected_slot: int
    producer_provenance_sha256: str
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            "assembly_cells": self.assembly_cells,
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
        }


@dataclass(frozen=True, slots=True)
class RawSourceTarget:
    event_kind_cell: int
    node_occurrence_cells: tuple[int, ...]
    observation: RawSpanObservationTarget
    compiler_event_sha256: str
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            "event_kind_cell": self.event_kind_cell,
            "node_occurrence_cells": self.node_occurrence_cells,
            "source_kind_cell": 0,
            "raw_span_observation": self.observation.neural_record(),
        }


@dataclass(frozen=True, slots=True)
class LiteralSourceTarget:
    event_kind_cell: int
    node_occurrence_cells: tuple[int, ...]
    typed_value_event_cells: tuple[int, ...]
    typed_value_type_cells: tuple[int, ...]
    physical_literal_event_sha256: str
    compiler_event_sha256: str
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            "event_kind_cell": self.event_kind_cell,
            "node_occurrence_cells": self.node_occurrence_cells,
            "source_kind_cell": 1,
            "typed_value_event_cells": self.typed_value_event_cells,
            "typed_value_type_cells": self.typed_value_type_cells,
        }


SourceTarget = RawSourceTarget | LiteralSourceTarget


@dataclass(frozen=True, slots=True)
class EdgeTarget:
    event_kind_cell: int
    parent_occurrence_cells: tuple[int, ...]
    child_occurrence_cells: tuple[int, ...]
    child_position_cell: int
    parent_factor_slot_occupancy_cells: tuple[int, ...]
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "sha256"
        }


@dataclass(frozen=True, slots=True)
class FactorSlotTarget:
    event_kind_cell: int
    node_occurrence_cells: tuple[int, ...]
    factor_slot_cell: int
    factor_kind_cell: int
    physical_type_cells: tuple[int, ...]
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "sha256"
        }


@dataclass(frozen=True, slots=True)
class RelationTarget:
    event_kind_cell: int
    node_occurrence_cells: tuple[int, ...]
    left_factor_slot_cell: int
    operator_cell: int
    rhs_kind_cell: int
    right_factor_slot_cell: int | None
    right_typed_value_event_cells: tuple[int, ...]
    right_typed_value_type_cells: tuple[int, ...]
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "sha256"
        }


@dataclass(frozen=True, slots=True)
class DomainTarget:
    event_kind_cell: int
    node_occurrence_cells: tuple[int, ...]
    factor_slot_cell: int
    physical_type_cells: tuple[int, ...]
    complete_allowed_value_event_cells: tuple[tuple[int, ...], ...]
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "sha256"
        }


@dataclass(frozen=True, slots=True)
class LiteralCopyGroupTarget:
    event_kind_cell: int
    copy_group_cells: tuple[int, ...]
    terminal_occurrence_slot_cells: tuple[
        tuple[tuple[int, ...], int], ...
    ]
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "sha256"
        }


@dataclass(frozen=True, slots=True)
class SignatureTarget:
    event_kind_cell: int
    root_occurrence_cells: tuple[int, ...]
    signature_role_cell: int
    signature_position_cell: int
    physical_type_cells: tuple[int, ...]
    sha256: str

    def neural_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "sha256"
        }


@dataclass(frozen=True, slots=True)
class FitOccurrenceFactorTarget:
    teacher_join_sha256: str
    multiplicity: int
    root_occurrence_cells: tuple[int, ...]
    sources: tuple[SourceTarget, ...]
    edges: tuple[EdgeTarget, ...]
    factor_slots: tuple[FactorSlotTarget, ...]
    relations: tuple[RelationTarget, ...]
    domains: tuple[DomainTarget, ...]
    literal_copy_groups: tuple[LiteralCopyGroupTarget, ...]
    signatures: tuple[SignatureTarget, ...]
    sha256: str

    def neural_record(self) -> dict[str, object]:
        """Return only physical teacher events, excluding every audit hash."""

        return {
            "root_occurrence_cells": self.root_occurrence_cells,
            "sources": tuple(row.neural_record() for row in self.sources),
            "edges": tuple(row.neural_record() for row in self.edges),
            "factor_slots": tuple(row.neural_record() for row in self.factor_slots),
            "relations": tuple(row.neural_record() for row in self.relations),
            "domains": tuple(row.neural_record() for row in self.domains),
            "literal_copy_groups": tuple(
                row.neural_record() for row in self.literal_copy_groups
            ),
            "signatures": tuple(row.neural_record() for row in self.signatures),
        }


@dataclass(frozen=True, slots=True)
class CapsuleCounts:
    fit_identity_count: int
    fit_appearance_count: int
    target_shape_count: int
    source_count: int
    raw_source_count: int
    literal_source_count: int
    edge_count: int
    factor_slot_count: int
    relation_count: int
    domain_count: int
    literal_copy_group_count: int
    signature_count: int
    emitted_target_record_count: int
    carrier_conflict_count: int
    missing_raw_observation_count: int
    producer_geometry_conflict_count: int


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV2:
    scope: str
    status: str
    predecessor_capsule_sha256: str
    predecessor_stop_sha256: str
    predecessor_successor_contract_sha256: str
    dependency_sha256: str
    type_vocabulary_sha256: str
    target_set_sha256: str
    surface_boundary_sha256: str
    counts: CapsuleCounts
    targets: tuple[FitOccurrenceFactorTarget, ...]
    stop_reason: str | None
    capsule_sha256: str

    def neural_records(self) -> tuple[dict[str, object], ...]:
        if self.status != READY:
            return ()
        return tuple(target.neural_record() for target in self.targets)


class _CapsuleStop(RuntimeError):
    pass


def _verify_dependencies() -> str:
    checks = (
        _file_sha256(PREDECESSOR_SOURCE_PATH) == PREDECESSOR_SOURCE_SHA256,
        predecessor_v1.CAPSULE.status == PREDECESSOR_STATUS,
        predecessor_v1.CAPSULE.capsule_sha256 == PREDECESSOR_CAPSULE_SHA256,
        predecessor_v1.STOP_LEDGER.sha256 == PREDECESSOR_STOP_SHA256,
        predecessor_v1.IDENTITY_AUDIT_SET_SHA256
        == PREDECESSOR_AUDIT_SET_SHA256,
        predecessor_v1.RAW_SOURCE_SUCCESSOR_CONTRACT_SHA256
        == PREDECESSOR_SUCCESSOR_CONTRACT_SHA256,
        predecessor_v1.EMITTED_TARGET_RECORD_COUNT == 0,
        predecessor_v1.LEARNER_ARRAY_COUNT == 0,
        predecessor_v1.LEARNING_RUN_COUNT == 0,
        selection_v1.PROTOCOL.protocol_sha256 == SELECTION_PROTOCOL_SHA256,
        structural_v1.CAPSULE.capsule_sha256 == STRUCTURAL_CAPSULE_SHA256,
        input_capsule.ARTIFACT_SHA256 == INPUT_ARTIFACT_SHA256,
        input_capsule.CONTENT_ROOT == INPUT_CONTENT_ROOT,
    )
    if not all(checks):
        raise _CapsuleStop(STOP_DEPENDENCY)
    record = {
        "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
        "predecessor_commit": PREDECESSOR_COMMIT,
        "predecessor_capsule_sha256": PREDECESSOR_CAPSULE_SHA256,
        "predecessor_stop_sha256": PREDECESSOR_STOP_SHA256,
        "predecessor_audit_set_sha256": PREDECESSOR_AUDIT_SET_SHA256,
        "predecessor_successor_contract_sha256": (
            PREDECESSOR_SUCCESSOR_CONTRACT_SHA256
        ),
        "selection_protocol_sha256": SELECTION_PROTOCOL_SHA256,
        "structural_capsule_sha256": STRUCTURAL_CAPSULE_SHA256,
        "input_artifact_sha256": INPUT_ARTIFACT_SHA256,
        "input_content_root": INPUT_CONTENT_ROOT,
        "predecessor_targets_consumed": False,
        "predecessor_private_helpers_consumed": False,
        "v4_v5_checkpoint_reads": 0,
    }
    return _digest(_DEPENDENCY_NAMESPACE, record)


def _root_topology(annotation: object) -> _TopologySpec:
    labels = tuple(
        label
        for label in annotation.relay_labels
        if label.channel == "root-production-family"
    )
    if len(labels) != 1:
        raise _CapsuleStop(STOP_INCOMPLETE)
    topology = _TOPOLOGIES.get(str(labels[0].fields["root_production"]))
    if topology is None:
        raise _CapsuleStop(STOP_INCOMPLETE)
    return topology


def _validate_topology(annotation: object, topology: _TopologySpec) -> None:
    channel_counts = Counter(label.channel for label in annotation.relay_labels)
    if tuple(
        (channel, channel_counts[channel])
        for channel, _ in topology.expected_channel_counts
    ) != topology.expected_channel_counts:
        raise _CapsuleStop(STOP_INCOMPLETE)
    expected_edges = {
        (node.parent, node.child_position, node.key)
        for node in topology.nodes
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
    expected_literals = {
        node.key for node in topology.nodes if node.source_kind == "literal"
    }
    observed_literals = {
        str(label.fields["semantic_literal_slot"])
        for label in annotation.relay_labels
        if label.channel == "literal-slot-binding"
    }
    observed_copies = {
        str(label.fields["semantic_literal_slot"])
        for label in annotation.relay_labels
        if label.channel == "literal-value-copy"
    }
    if (
        actual_edges != expected_edges
        or observed_literals != expected_literals
        or observed_copies != expected_literals
    ):
        raise _CapsuleStop(STOP_INCOMPLETE)


def _event_hash(record: object) -> str:
    return _digest(_EVENT_NAMESPACE, record)


def _with_sha(cls: type, namespace: bytes, record: dict[str, object]) -> object:
    return cls(**record, sha256=_digest(namespace, record))


def _build_identity_target(
    *,
    root: str,
    frozen_input: object,
    canonical: object,
    annotation: object,
    factor_annotation: object,
) -> FitOccurrenceFactorTarget:
    topology = _root_topology(annotation)
    _validate_topology(annotation, topology)
    factors_by_name = {target.factor: target for target in factor_annotation.targets}
    if set(factors_by_name) != set(topology.source_factors):
        raise _CapsuleStop(STOP_INCOMPLETE)

    literal_ordinals: dict[str, int] = {}
    for label in annotation.relay_labels:
        if label.channel != "literal-slot-binding":
            continue
        key = str(label.fields["semantic_literal_slot"])
        ordinal = label.fields["literal_source_ordinal"]
        if type(ordinal) is not int or key in literal_ordinals:
            raise _CapsuleStop(STOP_INCOMPLETE)
        literal_ordinals[key] = ordinal

    carriers = {
        node.key: _sparse_cells(
            "stable-node-occurrence",
            (root, node.key),
            width=NODE_CARRIER_CELL_COUNT,
            count=NODE_CARRIER_ACTIVE_CELL_COUNT,
        )
        for node in topology.nodes
    }
    if len(set(carriers.values())) != len(carriers):
        raise _CapsuleStop(STOP_INCOMPLETE)

    sources: list[SourceTarget] = []
    for node in topology.nodes:
        carrier = carriers[node.key]
        if node.source_kind == "raw":
            if node.selector is None:
                raise _CapsuleStop(STOP_INCOMPLETE)
            direction, cue_slot, selected_slot = node.selector
            expected_selected = (
                (cue_slot + 1) % 4 if direction == "NEXT" else (cue_slot - 1) % 4
            )
            if (
                direction not in ("NEXT", "PREV")
                or not 0 <= cue_slot < 4
                or selected_slot != expected_selected
            ):
                raise _CapsuleStop(STOP_INCOMPLETE)
            factor_target = factors_by_name[node.source_key]
            candidates = tuple(
                span
                for span in frozen_input.raw_span_records
                if int(span["start_token"]) == factor_target.start_token
                and int(span["stop_token"]) == factor_target.stop_token
            )
            if len(candidates) != 1:
                raise _CapsuleStop(STOP_INCOMPLETE)
            span = candidates[0]
            assembly = tuple(int(cell) for cell in span["assembly_cells"])
            spikes = tuple(int(cell) for cell in span["spike_cells"])
            if (
                len(assembly) != RAW_SPAN_ACTIVE_CELL_COUNT
                or len(set(assembly)) != len(assembly)
                or assembly != spikes
            ):
                raise _CapsuleStop(STOP_INCOMPLETE)
            span_event_record = {
                "physical_span_channel_sha256": (
                    frozen_input.physical_span_events_sha256
                ),
                "span_identity_sha256": factor_target.span_identity_sha256,
                "start_token": factor_target.start_token,
                "stop_token": factor_target.stop_token,
                "assembly_cells": assembly,
                "first_spike_step": int(span["first_spike_step"]),
            }
            physical_span_event_sha = _digest(_SPAN_NAMESPACE, span_event_record)
            provenance_record = {
                "teacher_join_root": root,
                "producer_node_key": node.key,
                "physical_span_event_sha256": physical_span_event_sha,
                "node_occurrence_cells": carrier,
                "direction": direction,
                "cue_slot": cue_slot,
                "selected_slot": selected_slot,
                "geometry_is_direct_producer_state": True,
                "candidate_enumeration_count": 0,
            }
            producer_sha = _digest(_PRODUCER_NAMESPACE, provenance_record)
            observation_record = {
                "physical_span_event_sha256": physical_span_event_sha,
                "assembly_cells": assembly,
                "direction": direction,
                "cue_slot": cue_slot,
                "selected_slot": selected_slot,
                "producer_provenance_sha256": producer_sha,
            }
            observation = _with_sha(
                RawSpanObservationTarget,
                _PRODUCER_NAMESPACE,
                observation_record,
            )
            compiler_event_record = {
                "event_kind_cell": EVENT_SOURCE,
                "node_occurrence_cells": carrier,
                "source_kind_cell": 0,
                "observation_sha256": observation.sha256,
            }
            compiler_sha = _event_hash(compiler_event_record)
            source_record = {
                "event_kind_cell": EVENT_SOURCE,
                "node_occurrence_cells": carrier,
                "observation": observation,
                "compiler_event_sha256": compiler_sha,
            }
            sources.append(
                _with_sha(RawSourceTarget, _EVENT_NAMESPACE, source_record)
            )
        else:
            if node.selector is not None:
                raise _CapsuleStop(STOP_INCOMPLETE)
            ordinal = literal_ordinals.get(node.source_key)
            if ordinal is None or not 0 <= ordinal < len(frozen_input.typed_literal_records):
                raise _CapsuleStop(STOP_INCOMPLETE)
            literal = frozen_input.typed_literal_records[ordinal]
            active_cells = tuple(
                int(cell) for cell in literal["typed_output_active_cells"]
            )
            literal_event_record = {
                "typed_literal_channel_sha256": (
                    frozen_input.typed_literal_events_sha256
                ),
                "canonical_token_index": int(literal["canonical_token_index"]),
                "typed_value_event_cells": active_cells,
                "physical_type_cells": TYPE_ASSEMBLIES[0],
            }
            literal_event_sha = _digest(_LITERAL_NAMESPACE, literal_event_record)
            compiler_event_record = {
                "event_kind_cell": EVENT_SOURCE,
                "node_occurrence_cells": carrier,
                "source_kind_cell": 1,
                "physical_literal_event_sha256": literal_event_sha,
                "typed_value_event_cells": active_cells,
            }
            compiler_sha = _event_hash(compiler_event_record)
            source_record = {
                "event_kind_cell": EVENT_SOURCE,
                "node_occurrence_cells": carrier,
                "typed_value_event_cells": active_cells,
                "typed_value_type_cells": TYPE_ASSEMBLIES[0],
                "physical_literal_event_sha256": literal_event_sha,
                "compiler_event_sha256": compiler_sha,
            }
            sources.append(
                _with_sha(LiteralSourceTarget, _EVENT_NAMESPACE, source_record)
            )

    edges = tuple(
        _with_sha(
            EdgeTarget,
            _EVENT_NAMESPACE,
            {
                "event_kind_cell": EVENT_EDGE,
                "parent_occurrence_cells": carriers[node.parent],
                "child_occurrence_cells": carriers[node.key],
                "child_position_cell": node.child_position,
                "parent_factor_slot_occupancy_cells": node.parent_occupancy,
            },
        )
        for node in topology.nodes
        if node.parent is not None
    )
    factor_slots = tuple(
        _with_sha(
            FactorSlotTarget,
            _EVENT_NAMESPACE,
            {
                "event_kind_cell": EVENT_FACTOR,
                "node_occurrence_cells": carriers[node.key],
                "factor_slot_cell": slot,
                "factor_kind_cell": FACTOR_KINDS.index(kind),
                "physical_type_cells": TYPE_ASSEMBLIES[PHYSICAL_TYPES.index(type_name)],
            },
        )
        for node in topology.nodes
        for slot, (kind, type_name) in enumerate(node.factors)
    )
    relations: list[RelationTarget] = []
    for node in topology.nodes:
        for relation in node.relations:
            rhs_kind = RHS_SLOT if relation.right_slot is not None else RHS_VALUE
            if relation.right_value is None:
                right_value_cells: tuple[int, ...] = ()
                right_type_cells: tuple[int, ...] = ()
            else:
                encoding = output_cells.TypedOutputCellBank.encode(
                    relation.right_value,
                    declared_type=output_cells.OutputType.INT,
                )
                if not encoding.supported:
                    raise _CapsuleStop(STOP_INCOMPLETE)
                right_value_cells = encoding.active_global_cell_indices()
                right_type_cells = TYPE_ASSEMBLIES[0]
            relation_record = {
                "event_kind_cell": EVENT_RELATION,
                "node_occurrence_cells": carriers[node.key],
                "left_factor_slot_cell": relation.left_slot,
                "operator_cell": RELATION_OPERATORS.index(relation.operator),
                "rhs_kind_cell": rhs_kind,
                "right_factor_slot_cell": relation.right_slot,
                "right_typed_value_event_cells": right_value_cells,
                "right_typed_value_type_cells": right_type_cells,
            }
            relations.append(
                _with_sha(RelationTarget, _EVENT_NAMESPACE, relation_record)
            )

    source_by_carrier = {source.node_occurrence_cells: source for source in sources}
    literal_nodes = tuple(
        node for node in topology.nodes if node.source_kind == "literal"
    )
    domains: list[DomainTarget] = []
    copy_groups: list[LiteralCopyGroupTarget] = []
    for group_index, node in enumerate(literal_nodes):
        source = source_by_carrier[carriers[node.key]]
        if not isinstance(source, LiteralSourceTarget):
            raise _CapsuleStop(STOP_INCOMPLETE)
        domains.append(
            _with_sha(
                DomainTarget,
                _EVENT_NAMESPACE,
                {
                    "event_kind_cell": EVENT_DOMAIN,
                    "node_occurrence_cells": carriers[node.key],
                    "factor_slot_cell": 0,
                    "physical_type_cells": TYPE_ASSEMBLIES[0],
                    "complete_allowed_value_event_cells": (
                        source.typed_value_event_cells,
                    ),
                },
            )
        )
        copy_groups.append(
            _with_sha(
                LiteralCopyGroupTarget,
                _EVENT_NAMESPACE,
                {
                    "event_kind_cell": EVENT_COPY,
                    "copy_group_cells": _sparse_cells(
                        "literal-copy-group",
                        (root, node.key, group_index),
                        width=COPY_GROUP_CELL_COUNT,
                        count=COPY_GROUP_ACTIVE_CELL_COUNT,
                    ),
                    "terminal_occurrence_slot_cells": ((carriers[node.key], 0),),
                },
            )
        )

    signatures = (
        _with_sha(
            SignatureTarget,
            _EVENT_NAMESPACE,
            {
                "event_kind_cell": EVENT_SIGNATURE,
                "root_occurrence_cells": carriers["root"],
                "signature_role_cell": 0,
                "signature_position_cell": 0,
                "physical_type_cells": TYPE_ASSEMBLIES[1],
            },
        ),
        _with_sha(
            SignatureTarget,
            _EVENT_NAMESPACE,
            {
                "event_kind_cell": EVENT_SIGNATURE,
                "root_occurrence_cells": carriers["root"],
                "signature_role_cell": 1,
                "signature_position_cell": 0,
                "physical_type_cells": TYPE_ASSEMBLIES[0],
            },
        ),
    )
    teacher_join_record = {
        "input_root_sha256": root,
        "canonical_sha256": canonical.sha256,
        "factor_annotation_sha256": factor_annotation.sha256,
        "structural_annotation_sha256": annotation.sha256,
        "predecessor_audit_set_sha256": PREDECESSOR_AUDIT_SET_SHA256,
        "compiler_fields_are_offline_only": True,
    }
    teacher_join_sha = _digest(_TARGET_NAMESPACE, teacher_join_record)
    target_record = {
        "teacher_join_sha256": teacher_join_sha,
        "multiplicity": frozen_input.multiplicity,
        "root_occurrence_cells": carriers["root"],
        "sources": tuple(sources),
        "edges": edges,
        "factor_slots": factor_slots,
        "relations": tuple(relations),
        "domains": tuple(domains),
        "literal_copy_groups": tuple(copy_groups),
        "signatures": signatures,
    }
    return _with_sha(FitOccurrenceFactorTarget, _TARGET_NAMESPACE, target_record)


def _build_targets() -> tuple[FitOccurrenceFactorTarget, ...]:
    state = input_capsule.load_input_capsule()
    authorized_input_roots = {row.input_root for row in state.inputs}
    canonical_rows = tuple(
        row
        for row in selection_v1.PROTOCOL.canonical_inputs
        if row.split == FIT and row.causal_key_sha256 in authorized_input_roots
    )
    if (
        len(canonical_rows) != FIT_IDENTITY_COUNT
        or sum(row.appearance_count for row in canonical_rows) != FIT_APPEARANCE_COUNT
    ):
        raise _CapsuleStop(STOP_INCOMPLETE)
    canonical_by_root = {row.causal_key_sha256: row for row in canonical_rows}
    annotations = {
        row.causal_key_sha256: row for row in structural_v1.CAPSULE.annotations
    }
    factor_annotations = {
        row.prompt_input_sha256: row
        for row in selection_v1.PROTOCOL.factor_annotations
    }
    targets = tuple(
        _build_identity_target(
            root=root,
            frozen_input=state.lookup(root),
            canonical=canonical_by_root[root],
            annotation=annotations[root],
            factor_annotation=factor_annotations[
                canonical_by_root[root].prompt_input_sha256
            ],
        )
        for root in sorted(canonical_by_root)
    )
    return targets


def _count_targets(
    targets: tuple[FitOccurrenceFactorTarget, ...],
) -> CapsuleCounts:
    sources = tuple(source for target in targets for source in target.sources)
    raw = tuple(source for source in sources if isinstance(source, RawSourceTarget))
    literals = tuple(
        source for source in sources if isinstance(source, LiteralSourceTarget)
    )
    carriers = tuple(source.node_occurrence_cells for source in sources)
    edges = tuple(row for target in targets for row in target.edges)
    factors = tuple(row for target in targets for row in target.factor_slots)
    relations = tuple(row for target in targets for row in target.relations)
    domains = tuple(row for target in targets for row in target.domains)
    copies = tuple(row for target in targets for row in target.literal_copy_groups)
    signatures = tuple(row for target in targets for row in target.signatures)
    emitted = (
        len(sources)
        + len(edges)
        + len(factors)
        + len(relations)
        + len(domains)
        + len(copies)
        + len(signatures)
    )
    geometry_conflicts = sum(
        observation.selected_slot
        != (
            (observation.cue_slot + 1) % 4
            if observation.direction == "NEXT"
            else (observation.cue_slot - 1) % 4
        )
        for observation in (source.observation for source in raw)
    )
    shape_counts = {len(target.sources) for target in targets}
    counts = CapsuleCounts(
        fit_identity_count=len(targets),
        fit_appearance_count=sum(target.multiplicity for target in targets),
        target_shape_count=len(shape_counts),
        source_count=len(sources),
        raw_source_count=len(raw),
        literal_source_count=len(literals),
        edge_count=len(edges),
        factor_slot_count=len(factors),
        relation_count=len(relations),
        domain_count=len(domains),
        literal_copy_group_count=len(copies),
        signature_count=len(signatures),
        emitted_target_record_count=emitted,
        carrier_conflict_count=len(carriers) - len(set(carriers)),
        missing_raw_observation_count=0,
        producer_geometry_conflict_count=geometry_conflicts,
    )
    expected = CapsuleCounts(
        fit_identity_count=FIT_IDENTITY_COUNT,
        fit_appearance_count=FIT_APPEARANCE_COUNT,
        target_shape_count=TARGET_SHAPE_COUNT,
        source_count=EXPECTED_SOURCE_COUNT,
        raw_source_count=EXPECTED_RAW_SOURCE_COUNT,
        literal_source_count=EXPECTED_LITERAL_SOURCE_COUNT,
        edge_count=EXPECTED_EDGE_COUNT,
        factor_slot_count=EXPECTED_FACTOR_COUNT,
        relation_count=EXPECTED_RELATION_COUNT,
        domain_count=EXPECTED_DOMAIN_COUNT,
        literal_copy_group_count=EXPECTED_COPY_GROUP_COUNT,
        signature_count=EXPECTED_SIGNATURE_COUNT,
        emitted_target_record_count=EXPECTED_TARGET_RECORD_COUNT,
        carrier_conflict_count=0,
        missing_raw_observation_count=0,
        producer_geometry_conflict_count=0,
    )
    if counts != expected:
        raise _CapsuleStop(STOP_INCOMPLETE)
    return counts


SURFACE_BOUNDARY_RECORD = {
    "fit_only": True,
    "heldout_input_lookup_count": HELDOUT_INPUT_LOOKUP_COUNT,
    "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
    "learner_array_count": LEARNER_ARRAY_COUNT,
    "learning_run_count": LEARNING_RUN_COUNT,
    "checkpoint_read_count": CHECKPOINT_READ_COUNT,
    "weight_read_count": WEIGHT_READ_COUNT,
    "panel_spend_count": PANEL_SPEND_COUNT,
    "desired_read_count": DESIRED_READ_COUNT,
    "program_field_count": PROGRAM_FIELD_COUNT,
    "task_field_count": TASK_FIELD_COUNT,
    "family_field_count": FAMILY_FIELD_COUNT,
    "source_order_geometry_choice_count": SOURCE_ORDER_GEOMETRY_CHOICE_COUNT,
    "fixed_candidate_geometry_choice_count": (
        FIXED_CANDIDATE_GEOMETRY_CHOICE_COUNT
    ),
    "hash_geometry_choice_count": HASH_GEOMETRY_CHOICE_COUNT,
    "compiler_names_are_offline_teacher_fields_only": True,
    "audit_hashes_are_excluded_from_neural_records": True,
    "geometry_is_direct_raw_event_producer_state": True,
    "authority_order_is_noncausal": True,
    "canonical_or_public_transfer_claimed": False,
}
SURFACE_BOUNDARY_SHA256 = _digest(_SURFACE_NAMESPACE, SURFACE_BOUNDARY_RECORD)


def _stopped_capsule(
    reason: str,
    dependency_sha256: str,
) -> FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV2:
    counts = CapsuleCounts(*(0 for _ in range(16)))
    record = {
        "scope": PROTOCOL_SCOPE,
        "status": reason,
        "predecessor_capsule_sha256": PREDECESSOR_CAPSULE_SHA256,
        "predecessor_stop_sha256": PREDECESSOR_STOP_SHA256,
        "predecessor_successor_contract_sha256": (
            PREDECESSOR_SUCCESSOR_CONTRACT_SHA256
        ),
        "dependency_sha256": dependency_sha256,
        "type_vocabulary_sha256": TYPE_VOCABULARY_SHA256,
        "target_set_sha256": _digest(_TARGET_SET_NAMESPACE, ()),
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "counts": counts,
        "targets": (),
        "stop_reason": reason,
    }
    return FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV2(
        **record,
        capsule_sha256=_digest(_CAPSULE_NAMESPACE, record),
    )


def _build_capsule() -> FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV2:
    dependency_sha = "0" * 64
    try:
        dependency_sha = _verify_dependencies()
        targets = _build_targets()
        counts = _count_targets(targets)
    except _CapsuleStop as error:
        return _stopped_capsule(str(error), dependency_sha)
    target_set_sha = _digest(
        _TARGET_SET_NAMESPACE,
        tuple(target.sha256 for target in targets),
    )
    record = {
        "scope": PROTOCOL_SCOPE,
        "status": READY,
        "predecessor_capsule_sha256": PREDECESSOR_CAPSULE_SHA256,
        "predecessor_stop_sha256": PREDECESSOR_STOP_SHA256,
        "predecessor_successor_contract_sha256": (
            PREDECESSOR_SUCCESSOR_CONTRACT_SHA256
        ),
        "dependency_sha256": dependency_sha,
        "type_vocabulary_sha256": TYPE_VOCABULARY_SHA256,
        "target_set_sha256": target_set_sha,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "counts": counts,
        "targets": targets,
        "stop_reason": None,
    }
    return FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV2(
        **record,
        capsule_sha256=_digest(_CAPSULE_NAMESPACE, record),
    )


CAPSULE = _build_capsule()


def require_ready_annotation_capsule_v2(
    candidate: object,
) -> tuple[FitOccurrenceFactorTarget, ...]:
    """Authenticate the exact issued authority before exposing FIT targets."""

    if candidate is not CAPSULE or CAPSULE.status != READY:
        raise RuntimeError(CAPSULE.stop_reason or STOP_INCOMPLETE)
    expected = _build_capsule()
    if expected.capsule_sha256 != CAPSULE.capsule_sha256:
        raise RuntimeError(STOP_DEPENDENCY)
    return CAPSULE.targets


__all__ = [
    "CAPSULE",
    "READY",
    "STOP_DEPENDENCY",
    "STOP_INCOMPLETE",
    "CapsuleCounts",
    "DomainTarget",
    "EdgeTarget",
    "FactorSlotTarget",
    "FitOccurrenceFactorTarget",
    "FrozenPublicTrainFitOccurrenceFactorAnnotationCapsuleV2",
    "LiteralCopyGroupTarget",
    "LiteralSourceTarget",
    "RawSourceTarget",
    "RawSpanObservationTarget",
    "RelationTarget",
    "SignatureTarget",
    "require_ready_annotation_capsule_v2",
]
