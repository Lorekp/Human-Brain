"""Session-native semantic partitions for the packed scalar frontier.

V1 proved that symbolic proof families can precede concrete witnesses, but it
still invoked public relation algebra once for every child-semantic tuple.  V2
keeps one private, session-owned workset: child partitions are imported and
rebased once, constraints are pushed into shared DAG traversal, and only shared
partition/group roots are published.  A selected group becomes a public
relation and proof slice only after the desired-vector race.

This remains the deliberately small scalar mechanism grammar from V1.  It is
not the full typed semantic synthesis grammar or capability evidence.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import NamedTuple

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from humanbrain.typed_program import FactorAction

FRONTIER_READY = "PACKED_SCALAR_FRONTIER_V2_READY"
STOP_WORK_BOUND = "STOP_PACKED_SCALAR_FRONTIER_V2_WORK_BOUND"
SELECTED = "SELECTED"
STOP_NO_EXACT = "STOP_NO_EXACT_SEMANTIC_VECTOR"
STOP_AMBIGUOUS = "STOP_AMBIGUOUS_VISIBLE_SPEC"

MAX_RECIPE_RELATION_WORK = 30_000
MAX_PHYSICAL_PREDICTIONS = 30_000
MAX_PHYSICAL_RACES = 80_000

_ROLE_SLOT = packed.TypedSlot("role", "Int", "argument", 0)
_LITERAL_SLOT = packed.TypedSlot("literal", "Int", "literal", 0)
_NAMESPACE = b"typed-semantic-packed-frontier-v2\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


def _slot_record(slot: packed.TypedSlot) -> tuple[object, ...]:
    return slot.to_record()


def _vector_record(
    values: Sequence[output_cells.TypedOutputEncoding],
) -> tuple[str, ...]:
    return tuple(value.sha256 for value in values)


def _vector_sha256(values: Sequence[output_cells.TypedOutputEncoding]) -> str:
    return _digest("physical-vector", _vector_record(values))


def _eclass_id(values: Sequence[output_cells.TypedOutputEncoding]) -> str:
    return _digest("e-class-id", _vector_record(values))


def _family_identity(production: str, child_family_ids: Sequence[str]) -> str:
    return _digest(
        "family-id",
        {"production": production, "children": tuple(child_family_ids)},
    )


def _relation_identity(
    schema: Sequence[packed.TypedSlot],
    root_id: str | None,
) -> str:
    return _digest(
        "ephemeral-relation",
        {
            "schema": tuple(_slot_record(slot) for slot in schema),
            "root_id": root_id,
        },
    )


def _constraint_record(constraint: packed.SlotConstraint) -> tuple[object, ...]:
    return (
        constraint.left.to_record(),
        constraint.operator,
        None if constraint.right_slot is None else constraint.right_slot.to_record(),
        constraint.right_value,
    )


class _FrontierSession(packed.PackedRelationSession):
    """One relation ledger for raw input, workset, frontier, and proof work."""

    __slots__ = ()

    def charge_frontier(self, event: str) -> None:
        if type(event) is not str or not event:
            raise ValueError("frontier event must be a nonempty string")
        self._charge(f"frontier:{event}")

    def charge_workset(self, event: str) -> None:
        if type(event) is not str or not event:
            raise ValueError("workset event must be a nonempty string")
        self._charge(f"workset:{event}")


@dataclass(frozen=True, slots=True)
class PackedPartitionHandle:
    """Opaque capability for one live session-owned semantic partition."""

    session_nonce: str
    generation: int
    partition_id: str
    family_id: str
    schema: tuple[packed.TypedSlot, ...]
    root_id: str | None
    relation_identity: str
    registry_digest: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedSemanticGroupRef:
    """Published witness-free reference to one semantic partition root."""

    partition_id: str
    family_id: str
    group_id: str
    schema: tuple[packed.TypedSlot, ...]
    root_id: str | None
    child_semantic_refs: tuple[tuple[str, ...], ...]
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    physical_output_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedNormalizationRecord:
    normalization_id: str
    source_family_id: str
    target_family_id: str
    retained_source_slots: tuple[packed.TypedSlot, ...]
    target_slots: tuple[packed.TypedSlot, ...]
    required_constants: tuple[tuple[packed.TypedSlot, int], ...]
    source_root_id: str
    recovered_root_id: str
    target_root_id: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedPublishedPartition:
    partition_id: str
    family_id: str
    production: str
    inside_cost: int
    child_family_ids: tuple[str, ...]
    child_slot_indices: tuple[tuple[int, ...], ...]
    schema: tuple[packed.TypedSlot, ...]
    root_id: str | None
    relation_identity: str
    generation: int
    registry_digest: str
    constraints: tuple[tuple[object, ...], ...]
    normalization_id: str | None
    normalization: PackedNormalizationRecord | None
    groups: tuple[PackedSemanticGroupRef, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedFrozenWorkset:
    arena_generation: int
    topology_node_count: int
    topology_sha256: str
    partitions: tuple[PackedPublishedPartition, ...]
    registry_digest: str
    sha256: str


class PackedFamily(NamedTuple):
    family_id: str
    production: str
    inside_cost: int
    child_family_ids: tuple[str, ...]
    child_slot_indices: tuple[tuple[int, ...], ...]
    schema: tuple[packed.TypedSlot, ...]
    partition_sha256: str
    sha256: str


class PackedENode(NamedTuple):
    enode_id: str
    family_id: str
    child_semantic_refs: tuple[tuple[str, ...], ...]
    group_ref: PackedSemanticGroupRef
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    physical_output_sha256: str
    sha256: str


class PackedEClass(NamedTuple):
    eclass_id: str
    minimum_inside_cost: int
    enode_ids: tuple[str, ...]
    group_refs: tuple[PackedSemanticGroupRef, ...]
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    sha256: str


class _AuthenticatedComponents(NamedTuple):
    owner_token: object
    frozen_sha256: str
    families: object
    enodes: object
    eclasses: object
    family_sha256s: tuple[str, ...]
    enode_sha256s: tuple[str, ...]
    eclass_sha256s: tuple[str, ...]
    group_to_enode: tuple[tuple[str, str], ...]
    enode_to_eclass: tuple[tuple[str, str], ...]
    root_eclass_ids: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedWitness:
    family_id: str
    binding: tuple[int, ...]
    source: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedFrontierWorkAudit:
    relation: packed.PackedRelationWorkAudit
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    concrete_proof_witnesses: int
    materialized_witnesses: int
    bound_name: str | None
    bound_attempted: int | None
    agenda_trace: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedFrontierState:
    status: str
    case_arguments: tuple[int, ...]
    literals: tuple[int, ...]
    requested_max_ast_cost: int
    maximum_completed_ast_cost: int
    operator_checkpoint_sha256: str
    frozen_workset: PackedFrozenWorkset | None
    families: tuple[PackedFamily, ...]
    enodes: tuple[PackedENode, ...]
    eclasses: tuple[PackedEClass, ...]
    root_eclass_ids: tuple[str, ...]
    component_capsule_sha256: str | None
    build_work_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedFrontierSelection:
    status: str
    frontier_sha256: str
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[PackedWitness, ...]
    selected_proof_forest_sha256: str | None
    race: semantic_race.TypedSemanticVectorRaceAudit | None
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    work: PackedFrontierWorkAudit
    sha256: str


class _PhysicalWorkBoundExceeded(RuntimeError):
    def __init__(self, name: str, attempted: int) -> None:
        super().__init__(f"{name} limit exceeded at {attempted}")
        self.name = name
        self.attempted = attempted


@dataclass(frozen=True, slots=True)
class _FamilyPlan:
    family_id: str
    production: str
    inside_cost: int
    child_family_ids: tuple[str, ...]
    child_target_slots: tuple[tuple[packed.TypedSlot, ...], ...]
    child_slot_indices: tuple[tuple[int, ...], ...]
    schema: tuple[packed.TypedSlot, ...]
    constraints: tuple[packed.SlotConstraint, ...]
    normalization_id: str | None = None


@dataclass(slots=True)
class _MutableSemanticGroup:
    root_id: str
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    child_refs: list[tuple[str, ...]]


@dataclass(slots=True)
class _PartitionRecord:
    handle: PackedPartitionHandle
    plan: _FamilyPlan
    root_id: str | None
    groups: tuple[_MutableSemanticGroup, ...]
    normalization: PackedNormalizationRecord | None


@dataclass(frozen=True, slots=True)
class _AuthenticatedFrozen:
    frozen_sha256: str
    owner_token: object
    arena: object
    group_index: object
    root_index: object


@dataclass(frozen=True, slots=True)
class _PublicationAuthentication:
    arena: object
    group_index: object
    root_index: object
    topology_node_count: int
    topology_sha256: str


def _handle_payload(handle: PackedPartitionHandle) -> dict[str, object]:
    return {
        "session_nonce": handle.session_nonce,
        "generation": handle.generation,
        "partition_id": handle.partition_id,
        "family_id": handle.family_id,
        "schema": tuple(_slot_record(slot) for slot in handle.schema),
        "root_id": handle.root_id,
        "relation_identity": handle.relation_identity,
        "registry_digest": handle.registry_digest,
    }


def _group_payload(group: PackedSemanticGroupRef) -> dict[str, object]:
    return {
        "partition_id": group.partition_id,
        "family_id": group.family_id,
        "group_id": group.group_id,
        "schema": tuple(_slot_record(slot) for slot in group.schema),
        "root_id": group.root_id,
        "child_semantic_refs": group.child_semantic_refs,
        "physical_outputs": _vector_record(group.physical_outputs),
        "physical_output_sha256": group.physical_output_sha256,
    }


def _normalization_payload(
    normalization: PackedNormalizationRecord,
) -> dict[str, object]:
    return {
        "normalization_id": normalization.normalization_id,
        "source_family_id": normalization.source_family_id,
        "target_family_id": normalization.target_family_id,
        "retained_source_slots": tuple(
            slot.to_record() for slot in normalization.retained_source_slots
        ),
        "target_slots": tuple(slot.to_record() for slot in normalization.target_slots),
        "required_constants": tuple(
            (slot.to_record(), value) for slot, value in normalization.required_constants
        ),
        "source_root_id": normalization.source_root_id,
        "recovered_root_id": normalization.recovered_root_id,
        "target_root_id": normalization.target_root_id,
    }


def _partition_payload(partition: PackedPublishedPartition) -> dict[str, object]:
    return {
        "partition_id": partition.partition_id,
        "family_id": partition.family_id,
        "production": partition.production,
        "inside_cost": partition.inside_cost,
        "child_family_ids": partition.child_family_ids,
        "child_slot_indices": partition.child_slot_indices,
        "schema": tuple(_slot_record(slot) for slot in partition.schema),
        "root_id": partition.root_id,
        "relation_identity": partition.relation_identity,
        "generation": partition.generation,
        "registry_digest": partition.registry_digest,
        "constraints": partition.constraints,
        "normalization_id": partition.normalization_id,
        "normalization_sha256": (
            None if partition.normalization is None else partition.normalization.sha256
        ),
        "groups": tuple(item.sha256 for item in partition.groups),
    }


def _frozen_payload(frozen: PackedFrozenWorkset) -> dict[str, object]:
    return {
        "arena_generation": frozen.arena_generation,
        "topology_node_count": frozen.topology_node_count,
        "topology_sha256": frozen.topology_sha256,
        "partitions": tuple(item.sha256 for item in frozen.partitions),
        "registry_digest": frozen.registry_digest,
    }


class _SemanticPartitionWorkset:
    """Private deep module for one session-owned semantic partition arena."""

    __slots__ = (
        "_arena",
        "_authenticated",
        "_component_capsule",
        "_frozen",
        "_generation",
        "_nonce",
        "_owner_token",
        "_records",
        "_sealed",
        "session",
    )

    def __init__(self, session: _FrontierSession) -> None:
        if type(session) is not _FrontierSession:
            raise TypeError("workset requires the enclosing frontier session")
        self.session = session
        self._arena = packed._Arena()
        self._records: dict[str, _PartitionRecord] = {}
        self._nonce = secrets.token_hex(16)
        self._generation = 0
        self._sealed = False
        self._frozen: PackedFrozenWorkset | None = None
        self._authenticated: _AuthenticatedFrozen | None = None
        self._component_capsule: _AuthenticatedComponents | None = None
        self._owner_token = object()

    def _operation(self, name: str) -> None:
        self.session._begin(f"workset-{name}")

    def _ensure_open(self) -> None:
        if self._sealed:
            raise RuntimeError("semantic partition workset is sealed")

    def _import_relation(self, relation: packed.PackedBindingRelation) -> None:
        self.session.charge_workset("import")
        validated = self.session._authenticate_relation(relation)
        for node_id in sorted(validated.arena.nodes):
            self.session.charge_workset("import-node")
            node = validated.arena.node(node_id)
            incumbent = self._arena.nodes.setdefault(node_id, node)
            if incumbent != node:
                raise RuntimeError("semantic workset node substitution")

    def _make(
        self,
        slot: packed.TypedSlot,
        edges: Iterable[tuple[int, str | None]],
    ) -> str | None:
        return self.session._make(self._arena, slot, edges)

    def _read_node_edges(
        self,
        node_id: str,
    ) -> tuple[packed.DecisionNode, tuple[tuple[int, str], ...]]:
        return self.session._read_node_edges(self._arena, node_id)

    def _union_roots(
        self,
        schema: tuple[packed.TypedSlot, ...],
        left_id: str | None,
        right_id: str | None,
        level: int = 0,
    ) -> str | None:
        if left_id is None:
            return right_id
        if right_id is None or left_id == right_id:
            return left_id
        if level == len(schema):
            return packed._TERMINAL_ID
        left_node, left_edges = self._read_node_edges(left_id)
        right_node, right_edges = self._read_node_edges(right_id)
        if left_node.slot != schema[level] or right_node.slot != schema[level]:
            raise RuntimeError("semantic group union schema changed")
        merged: list[tuple[int, str | None]] = []
        left_index = 0
        right_index = 0
        while left_index < len(left_edges) or right_index < len(right_edges):
            if right_index == len(right_edges) or (
                left_index < len(left_edges)
                and left_edges[left_index][0] < right_edges[right_index][0]
            ):
                merged.append(left_edges[left_index])
                left_index += 1
            elif left_index == len(left_edges) or (
                right_edges[right_index][0] < left_edges[left_index][0]
            ):
                merged.append(right_edges[right_index])
                right_index += 1
            else:
                value = left_edges[left_index][0]
                merged.append(
                    (
                        value,
                        self._union_roots(
                            schema,
                            left_edges[left_index][1],
                            right_edges[right_index][1],
                            level + 1,
                        ),
                    )
                )
                left_index += 1
                right_index += 1
        return self._make(schema[level], merged)

    def _rebase_partition(
        self,
        record: _PartitionRecord,
        target_slots: tuple[packed.TypedSlot, ...],
    ) -> tuple[str | None, tuple[str, ...]]:
        if len(record.handle.schema) != len(target_slots):
            raise RuntimeError("child partition rebase arity changed")
        memo: dict[tuple[str, int], str] = {}

        def visit(node_id: str, level: int) -> str:
            if node_id == packed._TERMINAL_ID:
                return packed._TERMINAL_ID
            self.session.charge_workset("rebase-cache-lookup")
            key = (node_id, level)
            cached = memo.get(key)
            if cached is not None:
                return cached
            node, edges = self._read_node_edges(node_id)
            if node.slot != record.handle.schema[level]:
                raise RuntimeError("child partition schema changed during rebase")
            root = self._make(
                target_slots[level],
                ((value, visit(child, level + 1)) for value, child in edges),
            )
            if root is None:
                raise RuntimeError("nonempty child partition disappeared during rebase")
            memo[key] = root
            return root

        def rebase(root_id: str | None) -> str | None:
            if root_id is None:
                return None
            return visit(root_id, 0)

        family_root = rebase(record.root_id)
        group_roots = tuple(rebase(group.root_id) for group in record.groups)
        if any(root is None for root in group_roots):
            raise RuntimeError("semantic child group became empty during rebase")
        return family_root, tuple(root for root in group_roots if root is not None)

    def _product_roots(
        self,
        child_schemas: tuple[tuple[packed.TypedSlot, ...], ...],
        child_roots: tuple[str | None, ...],
        output_schema: tuple[packed.TypedSlot, ...],
        constraints: tuple[packed.SlotConstraint, ...],
    ) -> str | None:
        if any(root is None for root in child_roots):
            return None
        owners: dict[packed.TypedSlot, int] = {}
        for child_index, schema in enumerate(child_schemas):
            for slot in schema:
                self.session.charge_workset("schema-owner-insert")
                if slot in owners:
                    raise RuntimeError("semantic product schemas overlap")
                owners[slot] = child_index
        if tuple(sorted(owners, key=lambda item: item.sort_key)) != output_schema:
            raise RuntimeError("semantic product output schema changed")

        def visit(
            level: int,
            roots: tuple[str, ...],
            child_levels: tuple[int, ...],
            binding: dict[packed.TypedSlot, int],
        ) -> str | None:
            if level == len(output_schema):
                if any(root != packed._TERMINAL_ID for root in roots):
                    return None
                for constraint in constraints:
                    self.session.charge_workset("constraint-decision")
                    if constraint.evaluate(binding) is not True:
                        return None
                return packed._TERMINAL_ID
            slot = output_schema[level]
            self.session.charge_workset("schema-owner-lookup")
            owner = owners[slot]
            node, edges = self._read_node_edges(roots[owner])
            if node.slot != slot or child_schemas[owner][child_levels[owner]] != slot:
                raise RuntimeError("semantic product traversal schema changed")
            proposed: list[tuple[int, str | None]] = []
            for value, child in edges:
                binding[slot] = value
                accepted = True
                for constraint in constraints:
                    self.session.charge_workset("constraint-decision")
                    if constraint.evaluate(binding) is False:
                        accepted = False
                        break
                if accepted:
                    next_roots = list(roots)
                    next_roots[owner] = child
                    next_levels = list(child_levels)
                    next_levels[owner] += 1
                    proposed.append(
                        (
                            value,
                            visit(
                                level + 1,
                                tuple(next_roots),
                                tuple(next_levels),
                                binding,
                            ),
                        )
                    )
                binding.pop(slot)
            return self._make(slot, proposed)

        return visit(
            0,
            tuple(root for root in child_roots if root is not None),
            tuple(0 for _schema in child_schemas),
            {},
        )

    def _project_normalization(
        self,
        source_schema: tuple[packed.TypedSlot, ...],
        source_root: str,
        retained_source_slots: tuple[packed.TypedSlot, ...],
        target_slots: tuple[packed.TypedSlot, ...],
        required_constants: tuple[tuple[packed.TypedSlot, int], ...],
    ) -> str:
        if len(retained_source_slots) != len(target_slots):
            raise RuntimeError("normalization projection arity changed")
        retained_index = {slot: index for index, slot in enumerate(retained_source_slots)}
        required = dict(required_constants)

        def visit(node_id: str, level: int) -> str | None:
            if node_id == packed._TERMINAL_ID:
                return packed._TERMINAL_ID
            node, edges = self._read_node_edges(node_id)
            if level >= len(source_schema) or node.slot != source_schema[level]:
                raise RuntimeError("normalization source schema changed")
            selected_edges: list[tuple[int, str | None]] = []
            for value, child in edges:
                self.session.charge_workset("normalization-constant-decision")
                if node.slot in required and value != required[node.slot]:
                    continue
                selected_edges.append((value, visit(child, level + 1)))
            if node.slot in retained_index:
                return self._make(
                    target_slots[retained_index[node.slot]],
                    selected_edges,
                )
            projected: str | None = None
            for _value, child in selected_edges:
                self.session.charge_workset("normalization-project-merge")
                projected = self._union_roots(
                    target_slots,
                    projected,
                    child,
                    sum(1 for slot in source_schema[:level] if slot in retained_index),
                )
            return projected

        recovered = visit(source_root, 0)
        if recovered is None:
            raise RuntimeError("normalization projection recovered no proof")
        return recovered

    def _record_partition(
        self,
        plan: _FamilyPlan,
        root_id: str | None,
        groups: tuple[_MutableSemanticGroup, ...],
        relation_identity: str,
        normalization: PackedNormalizationRecord | None = None,
    ) -> PackedPartitionHandle:
        self.session.charge_workset("registry-lookup")
        if plan.family_id in self._records:
            raise RuntimeError("duplicate semantic family partition")
        self._generation += 1
        registry_digest = _digest(
            "workset-registry-entry",
            {
                "generation": self._generation,
                "family_id": plan.family_id,
                "production": plan.production,
                "inside_cost": plan.inside_cost,
                "child_family_ids": plan.child_family_ids,
                "child_slot_indices": plan.child_slot_indices,
                "schema": tuple(_slot_record(slot) for slot in plan.schema),
                "root_id": root_id,
                "relation_identity": relation_identity,
                "constraints": tuple(
                    _constraint_record(constraint) for constraint in plan.constraints
                ),
                "normalization_id": plan.normalization_id,
                "normalization_sha256": (None if normalization is None else normalization.sha256),
                "groups": tuple(
                    {
                        "root_id": group.root_id,
                        "output": _vector_sha256(group.physical_outputs),
                        "refs": self._canonical_child_refs(group.child_refs),
                    }
                    for group in groups
                ),
            },
        )
        partition_id = _digest(
            "partition-id",
            {"family_id": plan.family_id, "generation": self._generation},
        )
        partial = PackedPartitionHandle(
            session_nonce=self._nonce,
            generation=self._generation,
            partition_id=partition_id,
            family_id=plan.family_id,
            schema=plan.schema,
            root_id=root_id,
            relation_identity=relation_identity,
            registry_digest=registry_digest,
            sha256="",
        )
        handle = PackedPartitionHandle(
            session_nonce=partial.session_nonce,
            generation=partial.generation,
            partition_id=partial.partition_id,
            family_id=partial.family_id,
            schema=partial.schema,
            root_id=partial.root_id,
            relation_identity=partial.relation_identity,
            registry_digest=partial.registry_digest,
            sha256=_digest("partition-handle", _handle_payload(partial)),
        )
        self.session.charge_workset("registry-insert")
        self._records[plan.family_id] = _PartitionRecord(
            handle,
            plan,
            root_id,
            groups,
            normalization,
        )
        return handle

    def _require_handle(self, handle: PackedPartitionHandle) -> _PartitionRecord:
        self.session.charge_workset("handle-lookup")
        if type(handle) is not PackedPartitionHandle:
            raise TypeError("semantic partition handle is invalid")
        if handle.session_nonce != self._nonce:
            raise RuntimeError("foreign semantic partition handle")
        if handle.sha256 != _digest("partition-handle", _handle_payload(handle)):
            raise RuntimeError("semantic partition handle authentication failed")
        self.session.charge_workset("registry-lookup")
        record = self._records.get(handle.family_id)
        if record is None or record.handle != handle:
            raise RuntimeError("stale or substituted semantic partition handle")
        if (
            handle.root_id != record.root_id
            or handle.schema != record.plan.schema
            or handle.relation_identity != _relation_identity(record.plan.schema, record.root_id)
        ):
            raise RuntimeError("semantic partition relation identity changed")
        return record

    def _canonical_child_refs(
        self,
        refs: Sequence[tuple[str, ...]],
    ) -> tuple[tuple[str, ...], ...]:
        unique: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for reference in refs:
            self.session.charge_workset("child-ref-dedup-lookup")
            if reference not in seen:
                self.session.charge_workset("child-ref-insert")
                seen.add(reference)
                unique.append(reference)
        for _reference in unique:
            self.session.charge_workset("child-ref-order-read")
        return tuple(sorted(unique))

    def seed_family(
        self,
        plan: _FamilyPlan,
        relation: packed.PackedBindingRelation,
        output_for_value: Callable[[int], tuple[output_cells.TypedOutputEncoding, ...]],
    ) -> PackedPartitionHandle:
        self._operation("seed-family")
        self._ensure_open()
        if plan.child_family_ids or len(plan.schema) != 1:
            raise ValueError("seed family must be a one-slot terminal")
        if relation.slots != plan.schema or relation.root_id is None:
            raise RuntimeError("seed relation schema or root changed")
        self._import_relation(relation)
        node, edges = self._read_node_edges(relation.root_id)
        if node.slot != plan.schema[0]:
            raise RuntimeError("seed relation slot changed")
        grouped: dict[str, _MutableSemanticGroup] = {}
        for value, child in edges:
            self.session.charge_workset("semantic-group-read")
            if child != packed._TERMINAL_ID:
                raise RuntimeError("seed domain has unexpected depth")
            outputs = output_for_value(value)
            output_sha = _vector_sha256(outputs)
            singleton = self._make(plan.schema[0], ((value, child),))
            if singleton is None:
                raise RuntimeError("seed semantic group is empty")
            self.session.charge_workset("semantic-group-lookup")
            incumbent = grouped.get(output_sha)
            if incumbent is None:
                self.session.charge_workset("semantic-group-insert")
                grouped[output_sha] = _MutableSemanticGroup(singleton, outputs, [])
            else:
                self.session.charge_workset("group-merge")
                merged = self._union_roots(plan.schema, incumbent.root_id, singleton)
                if merged is None:
                    raise RuntimeError("seed semantic group merge became empty")
                incumbent.root_id = merged
        groups = tuple(grouped[key] for key in sorted(grouped))
        return self._record_partition(
            plan,
            relation.root_id,
            groups,
            _relation_identity(plan.schema, relation.root_id),
        )

    def apply_family(
        self,
        plan: _FamilyPlan,
        child_handles: tuple[PackedPartitionHandle, ...],
        predict: Callable[
            [str, tuple[tuple[output_cells.TypedOutputEncoding, ...], ...]],
            tuple[output_cells.TypedOutputEncoding, ...],
        ],
    ) -> PackedPartitionHandle:
        self._operation("apply-family")
        self._ensure_open()
        if len(child_handles) != len(plan.child_family_ids):
            raise ValueError("semantic family child arity changed")
        records: list[_PartitionRecord] = []
        rebased_family_roots: list[str | None] = []
        rebased_group_roots: list[tuple[str, ...]] = []
        for expected_family, targets, handle in zip(
            plan.child_family_ids,
            plan.child_target_slots,
            child_handles,
            strict=True,
        ):
            record = self._require_handle(handle)
            if record.plan.family_id != expected_family:
                raise RuntimeError("semantic family child substitution")
            family_root, group_roots = self._rebase_partition(record, targets)
            records.append(record)
            rebased_family_roots.append(family_root)
            rebased_group_roots.append(group_roots)
        expected_root = self._product_roots(
            plan.child_target_slots,
            tuple(rebased_family_roots),
            plan.schema,
            plan.constraints,
        )
        grouped: dict[str, _MutableSemanticGroup] = {}
        normalization_source: str | None = None
        group_ranges = tuple(range(len(record.groups)) for record in records)
        for indices in product(*group_ranges):
            self.session.charge_workset("semantic-tuple-read")
            child_roots = tuple(
                rebased_group_roots[index][group_index] for index, group_index in enumerate(indices)
            )
            tuple_root = self._product_roots(
                plan.child_target_slots,
                child_roots,
                plan.schema,
                plan.constraints,
            )
            if tuple_root is None:
                if plan.normalization_id is not None:
                    if len(records) != 2 or len(plan.child_target_slots[1]) != 1:
                        raise RuntimeError("normalization child shape changed")
                    right_root = child_roots[1]
                    right_node, right_edges = self._read_node_edges(right_root)
                    if right_node.slot != plan.child_target_slots[1][0] or right_edges != (
                        (0, packed._TERMINAL_ID),
                    ):
                        raise RuntimeError("normalization did not isolate exact right zero")
                    raw_root = self._product_roots(
                        plan.child_target_slots,
                        child_roots,
                        plan.schema,
                        (),
                    )
                    if raw_root is None:
                        raise RuntimeError("normalization source tuple is empty")
                    self.session.charge_workset("normalization-source-merge")
                    normalization_source = self._union_roots(
                        plan.schema,
                        normalization_source,
                        raw_root,
                    )
                continue
            self.session.charge_workset("supported-semantic-tuple")
            child_outputs = tuple(
                records[index].groups[group_index].physical_outputs
                for index, group_index in enumerate(indices)
            )
            outputs = predict(plan.production, child_outputs)
            output_sha = _vector_sha256(outputs)
            refs = tuple(
                f"{records[index].plan.family_id}:{_eclass_id(child_outputs[index])}"
                for index in range(len(records))
            )
            self.session.charge_workset("semantic-group-lookup")
            incumbent = grouped.get(output_sha)
            if incumbent is None:
                self.session.charge_workset("semantic-group-insert")
                grouped[output_sha] = _MutableSemanticGroup(tuple_root, outputs, [refs])
            else:
                self.session.charge_workset("group-merge")
                merged = self._union_roots(plan.schema, incumbent.root_id, tuple_root)
                if merged is None:
                    raise RuntimeError("semantic output group merge became empty")
                incumbent.root_id = merged
                self.session.charge_workset("child-ref-append")
                incumbent.child_refs.append(refs)
        normalization: PackedNormalizationRecord | None = None
        if plan.normalization_id is not None:
            if normalization_source is None or records[0].root_id is None:
                raise RuntimeError("declared proof normalization did not fire")
            required_constants = ((plan.child_target_slots[1][0], 0),)
            recovered_root = self._project_normalization(
                plan.schema,
                normalization_source,
                plan.child_target_slots[0],
                records[0].plan.schema,
                required_constants,
            )
            if recovered_root != records[0].root_id:
                raise RuntimeError("x + 0 normalization did not recover its child")
            partial_normalization = PackedNormalizationRecord(
                normalization_id=plan.normalization_id,
                source_family_id=plan.family_id,
                target_family_id=records[0].plan.family_id,
                retained_source_slots=plan.child_target_slots[0],
                target_slots=records[0].plan.schema,
                required_constants=required_constants,
                source_root_id=normalization_source,
                recovered_root_id=recovered_root,
                target_root_id=records[0].root_id,
                sha256="",
            )
            normalization = PackedNormalizationRecord(
                normalization_id=partial_normalization.normalization_id,
                source_family_id=partial_normalization.source_family_id,
                target_family_id=partial_normalization.target_family_id,
                retained_source_slots=partial_normalization.retained_source_slots,
                target_slots=partial_normalization.target_slots,
                required_constants=partial_normalization.required_constants,
                source_root_id=partial_normalization.source_root_id,
                recovered_root_id=partial_normalization.recovered_root_id,
                target_root_id=partial_normalization.target_root_id,
                sha256=_digest(
                    "normalization-record",
                    _normalization_payload(partial_normalization),
                ),
            )
        groups = tuple(grouped[key] for key in sorted(grouped))
        coverage: str | None = None
        for group in groups:
            self.session.charge_workset("coverage-merge")
            coverage = self._union_roots(plan.schema, coverage, group.root_id)
        if coverage != expected_root:
            raise RuntimeError("semantic groups do not cover the family partition")
        return self._record_partition(
            plan,
            expected_root,
            groups,
            _relation_identity(plan.schema, expected_root),
            normalization,
        )

    def _authenticate_topology(
        self,
        root_specs: Sequence[tuple[str | None, tuple[packed.TypedSlot, ...]]],
    ) -> tuple[packed._Arena, dict[tuple[object, ...], str], int, str]:
        """Capture and authenticate all published roots in one global pass."""

        retained = packed._Arena()
        captured: dict[str, packed.DecisionNode] = {}
        topology_records: list[tuple[object, ...]] = []
        validated: set[tuple[str, int, tuple[tuple[object, ...], ...]]] = set()
        root_index: dict[tuple[object, ...], str] = {}

        for root_id, schema in root_specs:
            self.session.charge_workset("published-root-reference")
            schema_record = tuple(slot.to_record() for slot in schema)
            root_key = (root_id, schema_record)
            relation_identity = _relation_identity(schema, root_id)
            incumbent_identity = root_index.setdefault(root_key, relation_identity)
            if incumbent_identity != relation_identity:
                raise RuntimeError("published root identity collision")
            if root_id is None:
                continue
            if type(root_id) is not str or len(root_id) != 64:
                raise RuntimeError("published root identity changed")
            stack = [(root_id, 0)]
            while stack:
                node_id, level = stack.pop()
                if node_id == packed._TERMINAL_ID:
                    if level != len(schema):
                        raise RuntimeError("published root reached terminal at wrong depth")
                    continue
                memo_key = (node_id, level, schema_record)
                if memo_key in validated:
                    continue
                self.session.charge_workset("published-root-lookup")
                if level >= len(schema):
                    raise RuntimeError("published root exceeds schema")
                node = captured.get(node_id)
                if node is None:
                    node, edges = self._read_node_edges(node_id)
                    if type(node) is not packed.DecisionNode:
                        raise RuntimeError("published workset node changed")
                    if (
                        node.node_id != node_id
                        or node.node_id != packed._node_id(node.slot, edges)
                        or tuple(value for value, _child in edges)
                        != tuple(sorted({value for value, _child in edges}))
                    ):
                        raise RuntimeError("published workset node authentication failed")
                    for edge in edges:
                        if (
                            type(edge) is not tuple
                            or len(edge) != 2
                            or type(edge[0]) is not int
                            or type(edge[1]) is not str
                            or len(edge[1]) != 64
                        ):
                            raise RuntimeError("published workset edge changed")
                    incumbent = retained.nodes.setdefault(node_id, node)
                    if incumbent is not node:
                        raise RuntimeError("published workset node substitution")
                    captured[node_id] = node
                    topology_records.append((node.node_id, node.slot.to_record(), edges))
                else:
                    edges = node.edges
                if node.slot != schema[level]:
                    raise RuntimeError("published root schema changed")
                validated.add(memo_key)
                for _value, child in reversed(edges):
                    stack.append((child, level + 1))

        return (
            retained,
            root_index,
            len(captured),
            _digest("published-topology", tuple(topology_records)),
        )

    def publish(self) -> PackedFrozenWorkset:
        self._operation("publish")
        self._ensure_open()
        partitions: list[PackedPublishedPartition] = []
        for family_id in sorted(
            self._records,
            key=lambda item: (
                self._records[item].plan.inside_cost,
                item,
            ),
        ):
            record = self._require_handle(self._records[family_id].handle)
            group_refs: list[PackedSemanticGroupRef] = []
            for group in record.groups:
                self.session.charge_workset("publication-group")
                output_sha = _vector_sha256(group.physical_outputs)
                group_id = _digest(
                    "semantic-group-id",
                    {
                        "partition_id": record.handle.partition_id,
                        "family_id": family_id,
                        "root_id": group.root_id,
                        "physical_output_sha256": output_sha,
                    },
                )
                partial_group = PackedSemanticGroupRef(
                    partition_id=record.handle.partition_id,
                    family_id=family_id,
                    group_id=group_id,
                    schema=record.plan.schema,
                    root_id=group.root_id,
                    child_semantic_refs=self._canonical_child_refs(group.child_refs),
                    physical_outputs=group.physical_outputs,
                    physical_output_sha256=output_sha,
                    sha256="",
                )
                group_refs.append(
                    PackedSemanticGroupRef(
                        partition_id=partial_group.partition_id,
                        family_id=partial_group.family_id,
                        group_id=partial_group.group_id,
                        schema=partial_group.schema,
                        root_id=partial_group.root_id,
                        child_semantic_refs=partial_group.child_semantic_refs,
                        physical_outputs=partial_group.physical_outputs,
                        physical_output_sha256=partial_group.physical_output_sha256,
                        sha256=_digest("published-semantic-group", _group_payload(partial_group)),
                    )
                )
            partial_partition = PackedPublishedPartition(
                partition_id=record.handle.partition_id,
                family_id=family_id,
                production=record.plan.production,
                inside_cost=record.plan.inside_cost,
                child_family_ids=record.plan.child_family_ids,
                child_slot_indices=record.plan.child_slot_indices,
                schema=record.plan.schema,
                root_id=record.root_id,
                relation_identity=record.handle.relation_identity,
                generation=record.handle.generation,
                registry_digest=record.handle.registry_digest,
                constraints=tuple(
                    _constraint_record(constraint) for constraint in record.plan.constraints
                ),
                normalization_id=record.plan.normalization_id,
                normalization=record.normalization,
                groups=tuple(group_refs),
                sha256="",
            )
            partitions.append(
                PackedPublishedPartition(
                    partition_id=partial_partition.partition_id,
                    family_id=partial_partition.family_id,
                    production=partial_partition.production,
                    inside_cost=partial_partition.inside_cost,
                    child_family_ids=partial_partition.child_family_ids,
                    child_slot_indices=partial_partition.child_slot_indices,
                    schema=partial_partition.schema,
                    root_id=partial_partition.root_id,
                    relation_identity=partial_partition.relation_identity,
                    generation=partial_partition.generation,
                    registry_digest=partial_partition.registry_digest,
                    constraints=partial_partition.constraints,
                    normalization_id=partial_partition.normalization_id,
                    normalization=partial_partition.normalization,
                    groups=partial_partition.groups,
                    sha256=_digest(
                        "published-partition",
                        _partition_payload(partial_partition),
                    ),
                )
            )
        registry_digest = _digest(
            "published-registry",
            tuple((partition.partition_id, partition.registry_digest) for partition in partitions),
        )
        publication = self.authenticate(tuple(partitions), registry_digest)
        partial = PackedFrozenWorkset(
            arena_generation=self._generation,
            topology_node_count=publication.topology_node_count,
            topology_sha256=publication.topology_sha256,
            partitions=tuple(partitions),
            registry_digest=registry_digest,
            sha256="",
        )
        frozen = PackedFrozenWorkset(
            arena_generation=partial.arena_generation,
            topology_node_count=partial.topology_node_count,
            topology_sha256=partial.topology_sha256,
            partitions=partial.partitions,
            registry_digest=partial.registry_digest,
            sha256=_digest("frozen-workset", _frozen_payload(partial)),
        )
        self._frozen = frozen
        self._sealed = True
        self._authenticated = _AuthenticatedFrozen(
            frozen_sha256=frozen.sha256,
            owner_token=self._owner_token,
            arena=publication.arena,
            group_index=publication.group_index,
            root_index=publication.root_index,
        )
        return frozen

    def authenticate(
        self,
        partitions: tuple[PackedPublishedPartition, ...],
        registry_digest: str,
    ) -> _PublicationAuthentication:
        self._operation("authenticate-publication")
        self._ensure_open()
        if type(partitions) is not tuple or type(registry_digest) is not str:
            raise RuntimeError("published workset metadata changed")
        root_specs: list[tuple[str | None, tuple[packed.TypedSlot, ...]]] = []
        for partition in partitions:
            if type(partition) is not PackedPublishedPartition:
                raise RuntimeError("published partition type changed")
            root_specs.append((partition.root_id, partition.schema))
            if partition.normalization is not None:
                root_specs.extend(
                    (
                        (partition.normalization.source_root_id, partition.schema),
                        (
                            partition.normalization.recovered_root_id,
                            partition.normalization.target_slots,
                        ),
                        (
                            partition.normalization.target_root_id,
                            partition.normalization.target_slots,
                        ),
                    )
                )
            root_specs.extend((group.root_id, group.schema) for group in partition.groups)
        arena, root_index, topology_node_count, topology_sha256 = self._authenticate_topology(
            root_specs
        )
        group_index: dict[str, PackedSemanticGroupRef] = {}
        partition_records: list[tuple[str, str]] = []
        semantic_pairs: set[tuple[str, str]] = set()
        expected_order = tuple(
            sorted(
                self._records,
                key=lambda item: (self._records[item].plan.inside_cost, item),
            )
        )
        if tuple(partition.family_id for partition in partitions) != expected_order:
            raise RuntimeError("published partition order changed")
        for partition in partitions:
            self.session.charge_workset("publication-authentication")
            self.session.charge_workset("registry-lookup")
            record = self._records.get(partition.family_id)
            if (
                record is None
                or partition.sha256 != _digest("published-partition", _partition_payload(partition))
                or partition.relation_identity
                != _relation_identity(partition.schema, partition.root_id)
                or partition.partition_id != record.handle.partition_id
                or partition.generation != record.handle.generation
                or partition.registry_digest != record.handle.registry_digest
                or partition.constraints
                != tuple(_constraint_record(constraint) for constraint in record.plan.constraints)
                or partition.normalization_id != record.plan.normalization_id
                or partition.normalization is not record.normalization
            ):
                raise RuntimeError("published partition authentication failed")
            normalization = partition.normalization
            if normalization is not None:
                if len(record.plan.child_family_ids) != 2:
                    raise RuntimeError("published normalization authentication failed")
                self.session.charge_workset("registry-lookup")
                target_record = self._records.get(normalization.target_family_id)
                expected_required = ((record.plan.child_target_slots[1][0], 0),)
                if (
                    normalization.sha256
                    != _digest(
                        "normalization-record",
                        _normalization_payload(normalization),
                    )
                    or normalization.normalization_id != partition.normalization_id
                    or normalization.source_family_id != partition.family_id
                    or normalization.target_family_id != record.plan.child_family_ids[0]
                    or target_record is None
                    or normalization.retained_source_slots != record.plan.child_target_slots[0]
                    or normalization.target_slots != target_record.plan.schema
                    or normalization.required_constants != expected_required
                    or normalization.recovered_root_id != normalization.target_root_id
                    or normalization.target_root_id != target_record.root_id
                ):
                    raise RuntimeError("published normalization authentication failed")
            partition_records.append((partition.partition_id, partition.registry_digest))
            for group in partition.groups:
                self.session.charge_workset("publication-group-authentication")
                for output in group.physical_outputs:
                    output_cells.TypedOutputCellBank.validate(output)
                if (
                    group.partition_id != partition.partition_id
                    or group.family_id != partition.family_id
                    or group.schema != partition.schema
                    or group.physical_output_sha256 != _vector_sha256(group.physical_outputs)
                    or group.sha256 != _digest("published-semantic-group", _group_payload(group))
                ):
                    raise RuntimeError("published semantic group authentication failed")
                self.session.charge_workset("published-group-index-insert")
                incumbent = group_index.setdefault(group.group_id, group)
                if incumbent is not group:
                    raise RuntimeError("published semantic group identity collision")
                semantic_pairs.add((group.family_id, _eclass_id(group.physical_outputs)))
        for partition in partitions:
            for group in partition.groups:
                child_ids = partition.child_family_ids
                if not child_ids and group.child_semantic_refs:
                    raise RuntimeError("published terminal group gained child semantics")
                if child_ids and not group.child_semantic_refs:
                    raise RuntimeError("published operator group lost child semantics")
                for refs in group.child_semantic_refs:
                    self.session.charge_workset("published-child-ref-authentication")
                    if len(refs) != len(child_ids):
                        raise RuntimeError("published child semantic arity changed")
                    for reference, child_id in zip(refs, child_ids, strict=True):
                        prefix = f"{child_id}:"
                        if (
                            not reference.startswith(prefix)
                            or (child_id, reference.removeprefix(prefix)) not in semantic_pairs
                        ):
                            raise RuntimeError("published child semantic ref changed")
        if registry_digest != _digest("published-registry", tuple(partition_records)):
            raise RuntimeError("published workset registry changed")
        return _PublicationAuthentication(
            arena=arena,
            group_index=group_index,
            root_index=root_index,
            topology_node_count=topology_node_count,
            topology_sha256=topology_sha256,
        )

    def require_retained(
        self,
        frozen: PackedFrozenWorkset,
    ) -> _AuthenticatedFrozen:
        """Authenticate retained publication identities without topology replay."""

        self._operation("authenticate-retained")
        authenticated = self._authenticated
        if (
            type(frozen) is not PackedFrozenWorkset
            or frozen is not self._frozen
            or authenticated is None
            or authenticated is not self._authenticated
            or authenticated.owner_token is not self._owner_token
            or authenticated.frozen_sha256 != frozen.sha256
            or frozen.arena_generation != self._generation
            or frozen.sha256 != _digest("frozen-workset", _frozen_payload(frozen))
        ):
            raise RuntimeError("retained published workset authentication failed")
        for partition in frozen.partitions:
            self.session.charge_workset("retained-partition-authentication")
            if partition.sha256 != _digest("published-partition", _partition_payload(partition)):
                raise RuntimeError("retained published partition authentication failed")
            if partition.normalization is not None and (
                partition.normalization.sha256
                != _digest(
                    "normalization-record",
                    _normalization_payload(partition.normalization),
                )
            ):
                raise RuntimeError("retained normalization changed")
            for group in partition.groups:
                self.session.charge_workset("retained-group-authentication")
                if group.sha256 != _digest("published-semantic-group", _group_payload(group)):
                    raise RuntimeError("retained published group changed")
        return authenticated

    def materialize_group_relations(
        self,
        frozen: PackedFrozenWorkset,
        authenticated: _AuthenticatedFrozen,
        groups: tuple[PackedSemanticGroupRef, ...],
    ) -> tuple[packed.PackedBindingRelation, ...]:
        self._operation("materialize-selected-groups")
        if (
            authenticated is not self._authenticated
            or authenticated.owner_token is not self._owner_token
            or authenticated.frozen_sha256 != frozen.sha256
            or frozen is not self._frozen
            or type(authenticated.arena) is not packed._Arena
            or type(authenticated.group_index) is not dict
            or type(authenticated.root_index) is not dict
        ):
            raise RuntimeError("stale or foreign published workset authentication")
        relations: list[packed.PackedBindingRelation] = []
        for group in groups:
            self.session.charge_workset("selected-group-lookup")
            incumbent = authenticated.group_index.get(group.group_id)
            if incumbent is not group:
                raise RuntimeError("selected semantic group was substituted")
            root_key = (
                group.root_id,
                tuple(slot.to_record() for slot in group.schema),
            )
            if authenticated.root_index.get(root_key) != _relation_identity(
                group.schema,
                group.root_id,
            ):
                raise RuntimeError("selected semantic group root was substituted")
            relations.append(self.session._finish(group.schema, group.root_id, authenticated.arena))
        return tuple(relations)


def _family_payload(family: PackedFamily) -> dict[str, object]:
    return {
        "family_id": family.family_id,
        "production": family.production,
        "inside_cost": family.inside_cost,
        "child_family_ids": family.child_family_ids,
        "child_slot_indices": family.child_slot_indices,
        "schema": tuple(_slot_record(slot) for slot in family.schema),
        "partition_sha256": family.partition_sha256,
    }


def _enode_payload(enode: PackedENode) -> dict[str, object]:
    return {
        "enode_id": enode.enode_id,
        "family_id": enode.family_id,
        "child_semantic_refs": enode.child_semantic_refs,
        "group_ref_sha256": enode.group_ref.sha256,
        "physical_outputs": _vector_record(enode.physical_outputs),
        "physical_output_sha256": enode.physical_output_sha256,
    }


def _eclass_payload(eclass: PackedEClass) -> dict[str, object]:
    return {
        "eclass_id": eclass.eclass_id,
        "minimum_inside_cost": eclass.minimum_inside_cost,
        "enode_ids": eclass.enode_ids,
        "group_refs": tuple(item.sha256 for item in eclass.group_refs),
        "physical_outputs": _vector_record(eclass.physical_outputs),
    }


def _component_capsule_payload(
    capsule: _AuthenticatedComponents,
) -> dict[str, object]:
    return {
        "frozen_sha256": capsule.frozen_sha256,
        "family_sha256s": capsule.family_sha256s,
        "enode_sha256s": capsule.enode_sha256s,
        "eclass_sha256s": capsule.eclass_sha256s,
        "group_to_enode": capsule.group_to_enode,
        "enode_to_eclass": capsule.enode_to_eclass,
        "root_eclass_ids": capsule.root_eclass_ids,
    }


def _work_audit(
    relation: packed.PackedRelationWorkAudit,
    physical_predictions: int,
    prediction_cache_hits: int,
    physical_races: int,
    concrete_proof_witnesses: int,
    bound_name: str | None,
    bound_attempted: int | None,
    agenda_trace: tuple[tuple[str, str], ...],
) -> PackedFrontierWorkAudit:
    event_counts = dict(relation.event_counts)
    materialized = event_counts.get("frontier:materialize-witness", 0)
    payload = {
        "relation_event_sha256": relation.event_sha256,
        "relation_completed": relation.completed_units,
        "relation_attempted": relation.attempted_units,
        "physical_predictions": physical_predictions,
        "prediction_cache_hits": prediction_cache_hits,
        "physical_races": physical_races,
        "concrete_proof_witnesses": concrete_proof_witnesses,
        "materialized_witnesses": materialized,
        "bound_name": bound_name,
        "bound_attempted": bound_attempted,
        "agenda_trace": agenda_trace,
    }
    return PackedFrontierWorkAudit(
        relation=relation,
        physical_predictions=physical_predictions,
        prediction_cache_hits=prediction_cache_hits,
        physical_races=physical_races,
        concrete_proof_witnesses=concrete_proof_witnesses,
        materialized_witnesses=materialized,
        bound_name=bound_name,
        bound_attempted=bound_attempted,
        agenda_trace=agenda_trace,
        sha256=_digest("work", payload),
    )


class _Builder:
    def __init__(
        self,
        supplied_case_arguments: Iterable[int],
        supplied_literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
    ) -> None:
        self.case_arguments: tuple[int, ...] = ()
        self.supplied_case_arguments = supplied_case_arguments
        self.literals: tuple[int, ...] = ()
        self.supplied_literals = supplied_literals
        self.max_ast_cost = max_ast_cost
        self.operator = operator
        self.session = _FrontierSession(work_limit=MAX_RECIPE_RELATION_WORK)
        self.workset = _SemanticPartitionWorkset(self.session)
        self._prediction_cache: dict[
            tuple[str, str, str | None],
            output_cells.TypedOutputEncoding,
        ] = {}
        self.physical_predictions = 0
        self.prediction_cache_hits = 0
        self.physical_races = 0
        self.bound_name: str | None = None
        self.bound_attempted: int | None = None
        self.agenda_trace: list[tuple[str, str]] = []
        self.maximum_completed_ast_cost = 0
        self.normalization_count = 0
        self.authenticated_components: _AuthenticatedComponents | None = None

    def _charge(self, event: str) -> None:
        self.session.charge_frontier(event)

    def _record(self, kind: str, identity: str) -> None:
        self._charge("trace-record")
        self.agenda_trace.append((kind, identity))

    def _predict_case(
        self,
        production: str,
        left: output_cells.TypedOutputEncoding,
        right: output_cells.TypedOutputEncoding | None = None,
    ) -> output_cells.TypedOutputEncoding:
        key = (production, left.sha256, None if right is None else right.sha256)
        self.session.charge_workset("prediction-cache-lookup")
        cached = self._prediction_cache.get(key)
        if cached is not None:
            output_cells.TypedOutputCellBank.validate(cached)
            self.prediction_cache_hits += 1
            return cached
        attempted_prediction = self.physical_predictions + 1
        if attempted_prediction > MAX_PHYSICAL_PREDICTIONS:
            raise _PhysicalWorkBoundExceeded("physical-predictions", attempted_prediction)
        reserved_races = 1 if production == "absolute" else (2 if production == "add" else 4)
        attempted_race = self.physical_races + reserved_races
        if attempted_race > MAX_PHYSICAL_RACES:
            raise _PhysicalWorkBoundExceeded("physical-races", attempted_race)
        self.physical_predictions = attempted_prediction
        self._record("prediction", _digest("prediction-key", key))
        if right is None:
            action = FactorAction((), "APPLY1", production, "SELECTED", "NONE", None)
            prediction = self.operator.predict(left, action)
        else:
            action = FactorAction((), "APPLY2", production, "SELECTED", "SCOPE0", None)
            prediction = self.operator.predict(left, action, scopes=(right,))
        if len(prediction.races) != reserved_races:
            raise RuntimeError("physical operator race reservation drifted")
        self.physical_races += len(prediction.races)
        output = output_cells.TypedOutputCellBank.validate(prediction.output)
        self.session.charge_workset("prediction-cache-insert")
        incumbent = self._prediction_cache.setdefault(key, output)
        if incumbent != output:
            raise RuntimeError("physical prediction cache identity collision")
        return output

    def _predict_vector(
        self,
        production: str,
        children: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if len(children) == 1:
            return tuple(self._predict_case(production, value) for value in children[0])
        if len(children) == 2:
            return tuple(
                self._predict_case(production, left, right)
                for left, right in zip(children[0], children[1], strict=True)
            )
        raise ValueError("only unary and binary semantic families are supported")

    @staticmethod
    def _new_slot(
        source: packed.TypedSlot,
        occurrences: dict[tuple[str, str, str], int],
    ) -> packed.TypedSlot:
        key = (source.kind, source.value_type, source.name)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        return packed.TypedSlot(*key, occurrence)

    def _operator_plan(
        self,
        production: str,
        children: tuple[PackedPartitionHandle, ...],
        *,
        constraint_kind: str | None = None,
    ) -> _FamilyPlan:
        occurrences: dict[tuple[str, str, str], int] = {}
        targets: list[tuple[packed.TypedSlot, ...]] = []
        for child in children:
            self._charge("plan-child")
            targets.append(tuple(self._new_slot(slot, occurrences) for slot in child.schema))
        schema = tuple(
            sorted(
                (slot for child_targets in targets for slot in child_targets),
                key=lambda slot: slot.sort_key,
            )
        )
        positions = {slot: index for index, slot in enumerate(schema)}
        indices = tuple(
            tuple(positions[slot] for slot in child_targets) for child_targets in targets
        )
        constraints: tuple[packed.SlotConstraint, ...] = ()
        normalization_id: str | None = None
        if constraint_kind == "nonzero-right":
            constraints = (packed.SlotConstraint.value(targets[1][0], "ne", 0),)
            normalization_id = "add-right-zero-v2"
        elif constraint_kind == "ordered-bounds":
            constraints = (packed.SlotConstraint.slots(targets[0][0], "le", targets[1][0]),)
        elif constraint_kind is not None:
            raise ValueError("unknown semantic family constraint")
        family_id = _family_identity(
            production,
            tuple(child.family_id for child in children),
        )
        return _FamilyPlan(
            family_id=family_id,
            production=production,
            inside_cost=sum(
                self.workset._require_handle(child).plan.inside_cost for child in children
            )
            + 1,
            child_family_ids=tuple(child.family_id for child in children),
            child_target_slots=tuple(targets),
            child_slot_indices=indices,
            schema=schema,
            constraints=constraints,
            normalization_id=normalization_id,
        )

    def _terminal_families(
        self,
    ) -> tuple[PackedPartitionHandle, PackedPartitionHandle]:
        arguments: list[int] = []
        for value in self.supplied_case_arguments:
            self._charge("argument-value-read")
            if type(value) is not int:
                raise ValueError("case_arguments must contain exact integers")
            arguments.append(value)
        if not arguments:
            raise ValueError("case_arguments must contain exact integers")
        self.case_arguments = tuple(arguments)
        role_relation = self.session.domain(_ROLE_SLOT, (0,))

        accepted_literals: list[int] = []

        def observed_literals() -> Iterable[int]:
            for value in self.supplied_literals:
                yield value
                self._charge("literal-canonicalization")
                accepted_literals.append(value)

        literal_relation = self.session.domain(_LITERAL_SLOT, observed_literals())
        if not accepted_literals:
            raise ValueError("literals must contain at least one exact integer")
        self.literals = tuple(sorted(set(accepted_literals)))

        role_plan = _FamilyPlan(
            family_id=_family_identity("role", ()),
            production="role",
            inside_cost=1,
            child_family_ids=(),
            child_target_slots=(),
            child_slot_indices=(),
            schema=(_ROLE_SLOT,),
            constraints=(),
        )

        def role_outputs(_value: int) -> tuple[output_cells.TypedOutputEncoding, ...]:
            encoded: list[output_cells.TypedOutputEncoding] = []
            for argument in self.case_arguments:
                self._charge("terminal-output")
                encoded.append(
                    output_cells.TypedOutputCellBank.encode(
                        argument,
                        declared_type=output_cells.OutputType.INT,
                    )
                )
            return tuple(encoded)

        role = self.workset.seed_family(role_plan, role_relation, role_outputs)
        self._record("family-primary", role.family_id)

        literal_plan = _FamilyPlan(
            family_id=_family_identity("literal", ()),
            production="literal",
            inside_cost=1,
            child_family_ids=(),
            child_target_slots=(),
            child_slot_indices=(),
            schema=(_LITERAL_SLOT,),
            constraints=(),
        )

        def literal_outputs(value: int) -> tuple[output_cells.TypedOutputEncoding, ...]:
            encoded: list[output_cells.TypedOutputEncoding] = []
            for _argument in self.case_arguments:
                self._charge("terminal-output")
                encoded.append(
                    output_cells.TypedOutputCellBank.encode(
                        value,
                        declared_type=output_cells.OutputType.INT,
                    )
                )
            return tuple(encoded)

        literal = self.workset.seed_family(
            literal_plan,
            literal_relation,
            literal_outputs,
        )
        self._record("family-primary", literal.family_id)
        self.maximum_completed_ast_cost = 1
        return role, literal

    def _operator_family(
        self,
        production: str,
        children: tuple[PackedPartitionHandle, ...],
        *,
        constraint_kind: str | None = None,
    ) -> PackedPartitionHandle:
        plan = self._operator_plan(
            production,
            children,
            constraint_kind=constraint_kind,
        )
        self._record("family-primary", plan.family_id)
        self._record("semantic-expand", plan.family_id)
        handle = self.workset.apply_family(plan, children, self._predict_vector)
        if plan.normalization_id is not None:
            self.normalization_count += 1
        self.maximum_completed_ast_cost = max(
            self.maximum_completed_ast_cost,
            plan.inside_cost,
        )
        return handle

    def _components(
        self,
        frozen: PackedFrozenWorkset,
    ) -> tuple[tuple[PackedFamily, ...], tuple[PackedENode, ...], tuple[PackedEClass, ...]]:
        families: list[PackedFamily] = []
        enodes: list[PackedENode] = []
        group_owner: dict[str, str] = {}
        enode_owner: dict[str, str] = {}
        family_sha256s: list[str] = []
        enode_sha256s: list[str] = []
        eclass_sha256s: list[str] = []
        group_to_enode: list[tuple[str, str]] = []
        enode_to_eclass: list[tuple[str, str]] = []
        eclass_enode_ids: dict[str, list[str]] = {}
        eclass_group_refs: dict[str, list[PackedSemanticGroupRef]] = {}
        eclass_min_cost: dict[str, int] = {}
        eclass_outputs: dict[
            str,
            tuple[output_cells.TypedOutputEncoding, ...],
        ] = {}
        for partition in frozen.partitions:
            self._charge("publish-family")
            partial_family = PackedFamily(
                family_id=partition.family_id,
                production=partition.production,
                inside_cost=partition.inside_cost,
                child_family_ids=partition.child_family_ids,
                child_slot_indices=partition.child_slot_indices,
                schema=partition.schema,
                partition_sha256=partition.sha256,
                sha256="",
            )
            family = PackedFamily(
                family_id=partial_family.family_id,
                production=partial_family.production,
                inside_cost=partial_family.inside_cost,
                child_family_ids=partial_family.child_family_ids,
                child_slot_indices=partial_family.child_slot_indices,
                schema=partial_family.schema,
                partition_sha256=partial_family.partition_sha256,
                sha256=_digest("family", _family_payload(partial_family)),
            )
            families.append(family)
            family_sha256s.append(family.sha256)
            for group in partition.groups:
                self._charge("publish-e-node")
                if group.group_id in group_owner:
                    raise RuntimeError("published group was consumed twice")
                enode_id = _digest(
                    "e-node-id",
                    {
                        "family_id": family.family_id,
                        "group_ref_sha256": group.sha256,
                        "physical_output_sha256": group.physical_output_sha256,
                    },
                )
                partial_enode = PackedENode(
                    enode_id=enode_id,
                    family_id=family.family_id,
                    child_semantic_refs=group.child_semantic_refs,
                    group_ref=group,
                    physical_outputs=group.physical_outputs,
                    physical_output_sha256=group.physical_output_sha256,
                    sha256="",
                )
                enode = PackedENode(
                    enode_id=partial_enode.enode_id,
                    family_id=partial_enode.family_id,
                    child_semantic_refs=partial_enode.child_semantic_refs,
                    group_ref=partial_enode.group_ref,
                    physical_outputs=partial_enode.physical_outputs,
                    physical_output_sha256=partial_enode.physical_output_sha256,
                    sha256=_digest("e-node", _enode_payload(partial_enode)),
                )
                if enode.enode_id in enode_owner:
                    raise RuntimeError("packed e-node identity was consumed twice")
                enodes.append(enode)
                enode_sha256s.append(enode.sha256)
                group_owner[group.group_id] = enode.enode_id
                group_to_enode.append((group.group_id, enode.enode_id))
                eclass_id = _eclass_id(enode.physical_outputs)
                enode_owner[enode.enode_id] = eclass_id
                enode_to_eclass.append((enode.enode_id, eclass_id))
                self._charge("e-class-lookup")
                eclass_enode_ids.setdefault(eclass_id, []).append(enode.enode_id)
                eclass_group_refs.setdefault(eclass_id, []).append(enode.group_ref)
                prior_cost = eclass_min_cost.get(eclass_id)
                if prior_cost is None or family.inside_cost < prior_cost:
                    eclass_min_cost[eclass_id] = family.inside_cost
                self._charge("e-class-output-lookup")
                incumbent = eclass_outputs.setdefault(eclass_id, enode.physical_outputs)
                if incumbent != enode.physical_outputs:
                    raise RuntimeError("global semantic e-class identity collision")
        eclasses: list[PackedEClass] = []
        root_eclass_ids: list[str] = []
        finalized_enodes = 0
        for eclass_id, selected_ids in eclass_enode_ids.items():
            self._charge("publish-e-class")
            selected_group_refs = eclass_group_refs[eclass_id]
            finalized_enodes += len(selected_ids)
            partial = PackedEClass(
                eclass_id=eclass_id,
                minimum_inside_cost=eclass_min_cost[eclass_id],
                enode_ids=tuple(selected_ids),
                group_refs=tuple(selected_group_refs),
                physical_outputs=eclass_outputs[eclass_id],
                sha256="",
            )
            eclass = PackedEClass(
                eclass_id=partial.eclass_id,
                minimum_inside_cost=partial.minimum_inside_cost,
                enode_ids=partial.enode_ids,
                group_refs=partial.group_refs,
                physical_outputs=partial.physical_outputs,
                sha256=_digest("e-class", _eclass_payload(partial)),
            )
            eclasses.append(eclass)
            eclass_sha256s.append(eclass.sha256)
            root_eclass_ids.append(eclass.eclass_id)
        frozen_families = tuple(families)
        frozen_enodes = tuple(enodes)
        frozen_eclasses = tuple(eclasses)
        canonical_group_to_enode = tuple(group_to_enode)
        canonical_enode_to_eclass = tuple(enode_to_eclass)
        if (
            len(group_owner) != len(frozen_enodes)
            or len(enode_owner) != len(frozen_enodes)
            or len(canonical_group_to_enode) != len(frozen_enodes)
            or len(canonical_enode_to_eclass) != len(frozen_enodes)
            or finalized_enodes != len(frozen_enodes)
            or len(frozen_eclasses) != len(eclass_enode_ids)
        ):
            raise RuntimeError("packed component publication is not bijective")
        partial_capsule = _AuthenticatedComponents(
            owner_token=self.workset._owner_token,
            frozen_sha256=frozen.sha256,
            families=frozen_families,
            enodes=frozen_enodes,
            eclasses=frozen_eclasses,
            family_sha256s=tuple(family_sha256s),
            enode_sha256s=tuple(enode_sha256s),
            eclass_sha256s=tuple(eclass_sha256s),
            group_to_enode=canonical_group_to_enode,
            enode_to_eclass=canonical_enode_to_eclass,
            root_eclass_ids=tuple(root_eclass_ids),
            sha256="",
        )
        self.authenticated_components = _AuthenticatedComponents(
            owner_token=partial_capsule.owner_token,
            frozen_sha256=partial_capsule.frozen_sha256,
            families=partial_capsule.families,
            enodes=partial_capsule.enodes,
            eclasses=partial_capsule.eclasses,
            family_sha256s=partial_capsule.family_sha256s,
            enode_sha256s=partial_capsule.enode_sha256s,
            eclass_sha256s=partial_capsule.eclass_sha256s,
            group_to_enode=partial_capsule.group_to_enode,
            enode_to_eclass=partial_capsule.enode_to_eclass,
            root_eclass_ids=partial_capsule.root_eclass_ids,
            sha256=_digest(
                "authenticated-components",
                _component_capsule_payload(partial_capsule),
            ),
        )
        self.workset._component_capsule = self.authenticated_components
        return (
            frozen_families,
            frozen_enodes,
            frozen_eclasses,
        )

    def build(self) -> tuple[PackedFrontierState, _SemanticPartitionWorkset]:
        try:
            role, literal = self._terminal_families()
            if self.max_ast_cost >= 2:
                self._operator_family("absolute", (role,))
            if self.max_ast_cost >= 3:
                self._operator_family("add", (role, role))
                self._operator_family(
                    "add",
                    (role, literal),
                    constraint_kind=("nonzero-right" if 0 in self.literals else None),
                )
                extremum = self._operator_family("maximum", (role, literal))
            else:
                extremum = None
            if self.max_ast_cost >= 5 and extremum is not None:
                extremum = self._operator_family(
                    "minimum",
                    (extremum, literal),
                    constraint_kind="ordered-bounds",
                )
            for cost, production in ((7, "maximum"), (9, "minimum")):
                if self.max_ast_cost >= cost and extremum is not None:
                    extremum = self._operator_family(production, (extremum, literal))
            frozen = self.workset.publish()
            families, enodes, eclasses = self._components(frozen)
            capsule = self.authenticated_components
            if capsule is None:
                raise RuntimeError("packed component capsule was not published")
            self._charge("frontier-seal")
            build_work = self.work_audit()
            payload = {
                "status": FRONTIER_READY,
                "case_arguments": self.case_arguments,
                "literals": self.literals,
                "requested_max_ast_cost": self.max_ast_cost,
                "maximum_completed_ast_cost": self.maximum_completed_ast_cost,
                "operator_checkpoint_sha256": self.operator.checkpoint_sha256,
                "frozen_workset_sha256": frozen.sha256,
                "component_capsule_sha256": capsule.sha256,
                "root_eclass_ids": capsule.root_eclass_ids,
                "build_work_sha256": build_work.sha256,
            }
            return (
                PackedFrontierState(
                    status=FRONTIER_READY,
                    case_arguments=self.case_arguments,
                    literals=self.literals,
                    requested_max_ast_cost=self.max_ast_cost,
                    maximum_completed_ast_cost=self.maximum_completed_ast_cost,
                    operator_checkpoint_sha256=self.operator.checkpoint_sha256,
                    frozen_workset=frozen,
                    families=families,
                    enodes=enodes,
                    eclasses=eclasses,
                    root_eclass_ids=capsule.root_eclass_ids,
                    component_capsule_sha256=capsule.sha256,
                    build_work_sha256=build_work.sha256,
                    sha256=_digest("frontier", payload),
                ),
                self.workset,
            )
        except packed.PackedRelationWorkLimitExceeded as error:
            self.bound_name = "recipe-relation-work"
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset
        except _PhysicalWorkBoundExceeded as error:
            self.bound_name = error.name
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset

    def _stopped_state(self) -> PackedFrontierState:
        work = self.work_audit()
        payload = {
            "status": STOP_WORK_BOUND,
            "case_arguments": self.case_arguments,
            "literals": self.literals,
            "requested_max_ast_cost": self.max_ast_cost,
            "maximum_completed_ast_cost": self.maximum_completed_ast_cost,
            "operator_checkpoint_sha256": self.operator.checkpoint_sha256,
            "frozen_workset_sha256": None,
            "component_capsule_sha256": None,
            "root_eclass_ids": (),
            "build_work_sha256": work.sha256,
        }
        return PackedFrontierState(
            status=STOP_WORK_BOUND,
            case_arguments=self.case_arguments,
            literals=self.literals,
            requested_max_ast_cost=self.max_ast_cost,
            maximum_completed_ast_cost=self.maximum_completed_ast_cost,
            operator_checkpoint_sha256=self.operator.checkpoint_sha256,
            frozen_workset=None,
            families=(),
            enodes=(),
            eclasses=(),
            root_eclass_ids=(),
            component_capsule_sha256=None,
            build_work_sha256=work.sha256,
            sha256=_digest("frontier", payload),
        )

    def work_audit(self) -> PackedFrontierWorkAudit:
        return _work_audit(
            self.session.audit,
            self.physical_predictions,
            self.prediction_cache_hits,
            self.physical_races,
            0,
            self.bound_name,
            self.bound_attempted,
            tuple(self.agenda_trace),
        )


def _require_authenticated_components(
    workset: _SemanticPartitionWorkset,
    state: PackedFrontierState,
    capsule: _AuthenticatedComponents | None,
) -> None:
    workset.session.charge_frontier("authenticate-component-capsule")
    frozen = state.frozen_workset
    if (
        frozen is None
        or type(capsule) is not _AuthenticatedComponents
        or capsule is not workset._component_capsule
        or capsule.owner_token is not workset._owner_token
        or capsule.frozen_sha256 != frozen.sha256
        or capsule.families is not state.families
        or capsule.enodes is not state.enodes
        or capsule.eclasses is not state.eclasses
        or capsule.root_eclass_ids is not state.root_eclass_ids
        or capsule.sha256 != state.component_capsule_sha256
    ):
        raise RuntimeError("retained packed component authentication failed")


class PackedScalarFrontier:
    """One sealed frontier retaining its sole cumulative workset session."""

    __slots__ = (
        "_agenda_trace",
        "_authenticated_components",
        "_bound_attempted",
        "_bound_name",
        "_concrete_proof_witnesses",
        "_operator",
        "_physical_predictions",
        "_physical_races",
        "_prediction_cache_hits",
        "_state",
        "_workset",
    )

    def __init__(
        self,
        state: PackedFrontierState,
        workset: _SemanticPartitionWorkset,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
        builder: _Builder,
    ) -> None:
        self._state = state
        self._workset = workset
        self._authenticated_components = builder.authenticated_components
        self._operator = operator
        self._physical_predictions = builder.physical_predictions
        self._prediction_cache_hits = builder.prediction_cache_hits
        self._physical_races = builder.physical_races
        self._concrete_proof_witnesses = 0
        self._bound_name = builder.bound_name
        self._bound_attempted = builder.bound_attempted
        self._agenda_trace = tuple(builder.agenda_trace)

    @classmethod
    def build(
        cls,
        *,
        case_arguments: Iterable[int],
        literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor | None = None,
    ) -> PackedScalarFrontier:
        if isinstance(case_arguments, (str, bytes)):
            raise TypeError("case_arguments must be an iterable of exact integers")
        if isinstance(literals, (str, bytes)):
            raise TypeError("literals must be an iterable of exact integers")
        if type(max_ast_cost) is not int or not 1 <= max_ast_cost <= 9:
            raise ValueError("max_ast_cost must be in [1, 9]")
        frozen_operator = (
            operator_successor.train_public_typed_operator_successor()[0]
            if operator is None
            else operator
        )
        if not isinstance(frozen_operator, operator_successor.FrozenTypedOperatorSuccessor):
            raise TypeError("operator must be a FrozenTypedOperatorSuccessor")
        frozen_operator.clear_effect_memo()
        before = frozen_operator.live_checkpoint_sha256
        if before != frozen_operator.checkpoint_sha256:
            raise RuntimeError("typed operator checkpoint changed before frontier build")
        builder = _Builder(
            case_arguments,
            literals,
            max_ast_cost,
            frozen_operator,
        )
        state, workset = builder.build()
        if frozen_operator.live_checkpoint_sha256 != before:
            raise RuntimeError("typed operator checkpoint changed during frontier build")
        return cls(state, workset, frozen_operator, builder)

    @property
    def state(self) -> PackedFrontierState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def families(self) -> tuple[PackedFamily, ...]:
        return self._state.families

    @property
    def enodes(self) -> tuple[PackedENode, ...]:
        return self._state.enodes

    @property
    def eclasses(self) -> tuple[PackedEClass, ...]:
        return self._state.eclasses

    @property
    def work(self) -> PackedFrontierWorkAudit:
        return _work_audit(
            self._workset.session.audit,
            self._physical_predictions,
            self._prediction_cache_hits,
            self._physical_races,
            self._concrete_proof_witnesses,
            self._bound_name,
            self._bound_attempted,
            self._agenda_trace,
        )

    def _state_payload(self) -> dict[str, object]:
        return {
            "status": self._state.status,
            "case_arguments": self._state.case_arguments,
            "literals": self._state.literals,
            "requested_max_ast_cost": self._state.requested_max_ast_cost,
            "maximum_completed_ast_cost": self._state.maximum_completed_ast_cost,
            "operator_checkpoint_sha256": self._state.operator_checkpoint_sha256,
            "frozen_workset_sha256": (
                None if self._state.frozen_workset is None else self._state.frozen_workset.sha256
            ),
            "component_capsule_sha256": self._state.component_capsule_sha256,
            "root_eclass_ids": self._state.root_eclass_ids,
            "build_work_sha256": self._state.build_work_sha256,
        }

    def _authenticate(self) -> _AuthenticatedFrozen:
        if self._state.status != FRONTIER_READY or self._state.frozen_workset is None:
            raise RuntimeError("only a complete packed frontier can be authenticated")
        if self._operator.live_checkpoint_sha256 != self._state.operator_checkpoint_sha256:
            raise RuntimeError("packed frontier operator checkpoint changed")
        authenticated = self._workset.require_retained(self._state.frozen_workset)
        _require_authenticated_components(
            self._workset,
            self._state,
            self._authenticated_components,
        )
        if self._state.sha256 != _digest("frontier", self._state_payload()):
            raise RuntimeError("packed frontier authentication failed")
        return authenticated

    def _authenticate_stopped(self) -> None:
        if self._state.status != STOP_WORK_BOUND:
            raise RuntimeError("packed frontier has an unknown terminal status")
        if any(
            (
                self._state.frozen_workset,
                self._state.families,
                self._state.enodes,
                self._state.eclasses,
                self._state.root_eclass_ids,
            )
        ):
            raise RuntimeError("stopped packed frontier exposed partial topology")
        if self._operator.live_checkpoint_sha256 != self._state.operator_checkpoint_sha256:
            raise RuntimeError("stopped packed frontier operator checkpoint changed")
        if self._state.build_work_sha256 != self.work.sha256:
            raise RuntimeError("stopped packed frontier work prefix changed")
        if self._state.sha256 != _digest("frontier", self._state_payload()):
            raise RuntimeError("stopped packed frontier authentication failed")

    def _materialize(self, witness: packed.ProofWitness) -> PackedWitness:
        self._workset.session.charge_frontier("materialize-witness")
        families: dict[str, PackedFamily] = {}
        for item in self._state.families:
            self._workset.session.charge_frontier("materialize-family-index")
            families[item.family_id] = item

        def render(family_id: str, binding: tuple[int, ...]) -> str:
            self._workset.session.charge_frontier("materialize-family-lookup")
            family = families[family_id]
            if len(binding) != len(family.schema):
                raise RuntimeError("proof witness binding arity changed")
            if family.production == "role":
                return "x"
            if family.production == "literal":
                return str(binding[0])
            children: list[str] = []
            for child_id, indices in zip(
                family.child_family_ids,
                family.child_slot_indices,
                strict=True,
            ):
                child_binding = tuple(binding[index] for index in indices)
                children.append(render(child_id, child_binding))
            if family.production == "absolute":
                return f"abs({children[0]})"
            if family.production == "add":
                return f"({children[0]} + {children[1]})"
            symbols = {"minimum": "min", "maximum": "max"}
            return f"{symbols[family.production]}({children[0]}, {children[1]})"

        source = render(witness.family_id, witness.binding)
        payload = {
            "family_id": witness.family_id,
            "binding": witness.binding,
            "source": source,
        }
        return PackedWitness(
            family_id=witness.family_id,
            binding=witness.binding,
            source=source,
            sha256=_digest("witness", payload),
        )

    def _selection(
        self,
        status: str,
        *,
        selected_eclass_id: str | None = None,
        exact_proof_count: int = 0,
        witnesses: tuple[PackedWitness, ...] = (),
        selected_proof_forest_sha256: str | None = None,
        race: semantic_race.TypedSemanticVectorRaceAudit | None = None,
        checkpoint_before: str | None = None,
    ) -> PackedFrontierSelection:
        before = (
            self._operator.live_checkpoint_sha256
            if checkpoint_before is None
            else checkpoint_before
        )
        after = self._operator.live_checkpoint_sha256
        work = self.work
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "selected_eclass_id": selected_eclass_id,
            "exact_proof_count": exact_proof_count,
            "witnesses": tuple(item.sha256 for item in witnesses),
            "selected_proof_forest_sha256": selected_proof_forest_sha256,
            "race_sha256": None if race is None else race.sha256,
            "checkpoint_before_sha256": before,
            "checkpoint_after_sha256": after,
            "work_sha256": work.sha256,
        }
        return PackedFrontierSelection(
            status=status,
            frontier_sha256=self._state.sha256,
            selected_eclass_id=selected_eclass_id,
            exact_proof_count=exact_proof_count,
            witnesses=witnesses,
            selected_proof_forest_sha256=selected_proof_forest_sha256,
            race=race,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
            work=work,
            sha256=_digest("selection", payload),
        )

    def select(self, desired_outputs: Sequence[int]) -> PackedFrontierSelection:
        if self._state.status != FRONTIER_READY:
            self._authenticate_stopped()
            return self._selection(STOP_WORK_BOUND)
        try:
            authenticated = self._authenticate()
            if isinstance(desired_outputs, (str, bytes)):
                raise TypeError("desired_outputs must be exact integers")
            desired_values: list[int] = []
            for value in desired_outputs:
                self._workset.session.charge_frontier("desired-value-read")
                if type(value) is not int:
                    raise ValueError("desired_outputs must align with frontier cases")
                desired_values.append(value)
            if len(desired_values) != len(self._state.case_arguments):
                raise ValueError("desired_outputs must align with frontier cases")
            desired = tuple(
                output_cells.TypedOutputCellBank.encode(
                    value,
                    declared_type=output_cells.OutputType.INT,
                )
                for value in desired_values
            )
            before = self._operator.live_checkpoint_sha256
            attempted_races = self._physical_races + 1
            if attempted_races > MAX_PHYSICAL_RACES:
                raise _PhysicalWorkBoundExceeded("physical-races", attempted_races)
            self._physical_races = attempted_races
            race = semantic_race.select_exact_semantic_vector(
                self._state.root_eclass_ids,
                tuple(item.physical_outputs for item in self._state.eclasses),
                desired,
            )
            if self._operator.live_checkpoint_sha256 != before:
                raise RuntimeError("operator checkpoint changed during selection")
            if race.status == semantic_race.STOP_NO_EXACT:
                return self._selection(STOP_NO_EXACT, race=race, checkpoint_before=before)
            if race.status != semantic_race.SELECTED or race.winner_key is None:
                raise RuntimeError("unique e-class bank returned a physical tie")
            eclass: PackedEClass | None = None
            for item in self._state.eclasses:
                self._workset.session.charge_frontier("winner-e-class-lookup")
                if item.eclass_id == race.winner_key:
                    eclass = item
                    break
            if eclass is None:
                raise RuntimeError("physical winner references an unknown e-class")
            frozen = self._state.frozen_workset
            if frozen is None:
                raise RuntimeError("selected frontier lost its published workset")
            relations = self._workset.materialize_group_relations(
                frozen,
                authenticated,
                eclass.group_refs,
            )
            slices = tuple(
                self._workset.session.family_slice(group.family_id, relation)
                for group, relation in zip(eclass.group_refs, relations, strict=True)
            )
            forest = self._workset.session.proof_forest(slices, saturation_limit=2)
            self._workset.session.authenticate_forest(forest)
            self._concrete_proof_witnesses = len(forest.first_two_witnesses)
            proof_count = self._workset.session.forest_exact_count(forest)
            proof_witnesses = self._workset.session.forest_first_witnesses(forest)
            if proof_count == 1:
                witnesses = (self._materialize(proof_witnesses[0]),)
                return self._selection(
                    SELECTED,
                    selected_eclass_id=eclass.eclass_id,
                    exact_proof_count=proof_count,
                    witnesses=witnesses,
                    selected_proof_forest_sha256=forest.sha256,
                    race=race,
                    checkpoint_before=before,
                )
            witnesses = tuple(self._materialize(item) for item in proof_witnesses[:2])
            return self._selection(
                STOP_AMBIGUOUS,
                selected_eclass_id=eclass.eclass_id,
                exact_proof_count=proof_count,
                witnesses=witnesses,
                selected_proof_forest_sha256=forest.sha256,
                race=race,
                checkpoint_before=before,
            )
        except packed.PackedRelationWorkLimitExceeded as error:
            self._bound_name = "recipe-relation-work"
            self._bound_attempted = error.attempted
            return self._selection(STOP_WORK_BOUND)
        except _PhysicalWorkBoundExceeded as error:
            self._bound_name = error.name
            self._bound_attempted = error.attempted
            return self._selection(STOP_WORK_BOUND)


__all__ = [
    "FRONTIER_READY",
    "MAX_PHYSICAL_PREDICTIONS",
    "MAX_PHYSICAL_RACES",
    "MAX_RECIPE_RELATION_WORK",
    "SELECTED",
    "STOP_AMBIGUOUS",
    "STOP_NO_EXACT",
    "STOP_WORK_BOUND",
    "PackedEClass",
    "PackedENode",
    "PackedFamily",
    "PackedFrontierSelection",
    "PackedFrontierState",
    "PackedFrontierWorkAudit",
    "PackedFrozenWorkset",
    "PackedPartitionHandle",
    "PackedPublishedPartition",
    "PackedScalarFrontier",
    "PackedSemanticGroupRef",
    "PackedWitness",
]
