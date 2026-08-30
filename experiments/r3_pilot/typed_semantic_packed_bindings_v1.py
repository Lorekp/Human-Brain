"""Packed finite binding relations and proof forests for semantic V2.

This module is deliberately below the semantic-frontier seam.  It provides a
small immutable relational substrate for factorized literal/role bindings, but
does not enumerate typed programs, evaluate operators, inspect desired values,
or select a recipe.

Relations are reduced, content-addressed multi-valued decision diagrams.  A
decision node names one typed slot and has sorted value-to-child edges.  Equal
suffixes are hash-consed, so independent finite domains form a product without
materializing every row.  Proof forests add a canonical family identity above
the relation: identical bindings in different families remain proof-distinct
until an explicit ``ProofNormalizationProjection`` maps one family into the
other.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal


SLOT_KINDS = ("literal", "role")
CONSTRAINT_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge")
DEFAULT_SATURATION_LIMIT = 2**31 - 1

_TERMINAL_ID = sha256(b"typed-semantic-packed-bindings-v1-terminal\0").hexdigest()
_NODE_NAMESPACE = b"typed-semantic-packed-bindings-v1-node\0"
_RELATION_NAMESPACE = b"typed-semantic-packed-bindings-v1-relation\0"
_SLICE_NAMESPACE = b"typed-semantic-packed-bindings-v1-family-slice\0"
_FOREST_NAMESPACE = b"typed-semantic-packed-bindings-v1-proof-forest\0"


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


@dataclass(frozen=True, slots=True)
class TypedSlot:
    """One typed binding position; ``occurrence`` preserves multiplicity."""

    kind: Literal["literal", "role"]
    value_type: str
    name: str
    occurrence: int = 0

    def __post_init__(self) -> None:
        if self.kind not in SLOT_KINDS:
            raise ValueError("slot kind must be literal or role")
        if type(self.value_type) is not str or not self.value_type:
            raise ValueError("slot value_type must be a nonempty string")
        if type(self.name) is not str or not self.name:
            raise ValueError("slot name must be a nonempty string")
        if type(self.occurrence) is not int or self.occurrence < 0:
            raise ValueError("slot occurrence must be a nonnegative integer")

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (self.kind, self.value_type, self.name, self.occurrence)

    def to_record(self) -> tuple[object, ...]:
        return (self.kind, self.value_type, self.name, self.occurrence)


@dataclass(frozen=True, slots=True)
class DecisionNode:
    """One content-addressed decision in a finite relation."""

    node_id: str
    slot: TypedSlot
    edges: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if type(self.node_id) is not str or len(self.node_id) != 64:
            raise ValueError("decision node_id must be a SHA-256")
        if not isinstance(self.slot, TypedSlot):
            raise TypeError("decision node slot must be TypedSlot")
        if (
            type(self.edges) is not tuple
            or not self.edges
            or any(
                type(value) is not int
                or type(child) is not str
                or len(child) != 64
                for value, child in self.edges
            )
        ):
            raise ValueError("decision edges must be nonempty integer/SHA pairs")
        values = tuple(value for value, _child in self.edges)
        if values != tuple(sorted(set(values))):
            raise ValueError("decision edges must have unique sorted values")
        expected = _node_id(self.slot, self.edges)
        if self.node_id != expected:
            raise ValueError("decision node content hash is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "slot": self.slot.to_record(),
            "edges": self.edges,
        }


def _node_id(slot: TypedSlot, edges: tuple[tuple[int, str], ...]) -> str:
    return _namespaced_sha256(
        _NODE_NAMESPACE,
        {"slot": slot.to_record(), "edges": edges},
    )


@dataclass(frozen=True, slots=True)
class RelationAllocation:
    """Actual reachable decision storage charged to one relation."""

    decision_nodes: int
    decision_edges: int

    @property
    def charged_units(self) -> int:
        return self.decision_nodes + self.decision_edges


class _Arena:
    """Private content-addressed node arena used while composing relations."""

    def __init__(self, nodes: Iterable[DecisionNode] = ()) -> None:
        self.nodes: dict[str, DecisionNode] = {}
        for node in nodes:
            incumbent = self.nodes.setdefault(node.node_id, node)
            if incumbent != node:
                raise RuntimeError("decision-node hash collision")

    def node(self, node_id: str) -> DecisionNode:
        if node_id == _TERMINAL_ID:
            raise ValueError("terminal has no decision-node record")
        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise ValueError("relation references an unknown decision node") from error

    def make(
        self,
        slot: TypedSlot,
        edges: Iterable[tuple[int, str | None]],
    ) -> str | None:
        canonical = tuple(
            (value, child)
            for value, child in sorted(edges)
            if child is not None
        )
        if not canonical:
            return None
        node_id = _node_id(slot, canonical)
        node = DecisionNode(node_id, slot, canonical)
        incumbent = self.nodes.setdefault(node_id, node)
        if incumbent != node:
            raise RuntimeError("decision-node hash collision")
        return node_id


def _canonical_slots(slots: Sequence[TypedSlot]) -> tuple[TypedSlot, ...]:
    if isinstance(slots, (str, bytes)):
        raise TypeError("slots must be a sequence of TypedSlot values")
    supplied = tuple(slots)
    if not all(isinstance(slot, TypedSlot) for slot in supplied):
        raise TypeError("slots must contain TypedSlot values")
    if len(set(supplied)) != len(supplied):
        raise ValueError("relation slots must be unique")
    return tuple(sorted(supplied, key=lambda slot: slot.sort_key))


def _reachable_nodes(
    root_id: str | None,
    arena: _Arena,
) -> tuple[DecisionNode, ...]:
    if root_id is None or root_id == _TERMINAL_ID:
        return ()
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id == _TERMINAL_ID or node_id in seen:
            continue
        node = arena.node(node_id)
        seen.add(node_id)
        stack.extend(child for _value, child in node.edges)
    return tuple(sorted((arena.node(node_id) for node_id in seen), key=lambda node: node.node_id))


def _relation_payload(
    slots: tuple[TypedSlot, ...],
    root_id: str | None,
    nodes: tuple[DecisionNode, ...],
) -> dict[str, object]:
    return {
        "schema": "typed-semantic-packed-binding-relation-v1",
        "slots": tuple(slot.to_record() for slot in slots),
        "root_id": root_id,
        "terminal_id": _TERMINAL_ID,
        "nodes": tuple(node.to_record() for node in nodes),
        "allocation": {
            "decision_nodes": len(nodes),
            "decision_edges": sum(len(node.edges) for node in nodes),
        },
    }


@dataclass(frozen=True, slots=True)
class SlotConstraint:
    """A deterministic binary selection predicate over slots or a constant."""

    left: TypedSlot
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    right_slot: TypedSlot | None = None
    right_value: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.left, TypedSlot):
            raise TypeError("constraint left operand must be a TypedSlot")
        if self.operator not in CONSTRAINT_OPERATORS:
            raise ValueError("unsupported relation constraint operator")
        if (self.right_slot is None) == (self.right_value is None):
            raise ValueError("constraint needs exactly one right operand")
        if self.right_slot is not None and not isinstance(self.right_slot, TypedSlot):
            raise TypeError("constraint right slot must be TypedSlot")
        if self.right_value is not None and type(self.right_value) is not int:
            raise TypeError("constraint right value must be an exact integer")

    @classmethod
    def value(
        cls,
        left: TypedSlot,
        operator: Literal["eq", "ne", "lt", "le", "gt", "ge"],
        right: int,
    ) -> SlotConstraint:
        return cls(left, operator, right_value=right)

    @classmethod
    def slots(
        cls,
        left: TypedSlot,
        operator: Literal["eq", "ne", "lt", "le", "gt", "ge"],
        right: TypedSlot,
    ) -> SlotConstraint:
        return cls(left, operator, right_slot=right)

    @property
    def required_slots(self) -> frozenset[TypedSlot]:
        return frozenset(
            (self.left,)
            if self.right_slot is None
            else (self.left, self.right_slot)
        )

    def evaluate(self, binding: dict[TypedSlot, int]) -> bool | None:
        if not self.required_slots <= binding.keys():
            return None
        left = binding[self.left]
        right = (
            binding[self.right_slot]
            if self.right_slot is not None
            else self.right_value
        )
        if self.operator == "eq":
            return left == right
        if self.operator == "ne":
            return left != right
        if self.operator == "lt":
            return left < right
        if self.operator == "le":
            return left <= right
        if self.operator == "gt":
            return left > right
        if self.operator == "ge":
            return left >= right
        raise RuntimeError("constraint operator validation drifted")


@dataclass(frozen=True, slots=True)
class PackedBindingRelation:
    """Immutable finite relation represented by a reduced decision DAG."""

    slots: tuple[TypedSlot, ...]
    root_id: str | None
    nodes: tuple[DecisionNode, ...]
    allocation: RelationAllocation
    sha256: str

    def __post_init__(self) -> None:
        if self.slots != _canonical_slots(self.slots):
            raise ValueError("relation slots are not in canonical order")
        if type(self.nodes) is not tuple or tuple(
            sorted(self.nodes, key=lambda node: node.node_id)
        ) != self.nodes:
            raise ValueError("relation nodes must be a sorted tuple")
        arena = _Arena(self.nodes)
        reachable = _reachable_nodes(self.root_id, arena)
        if reachable != self.nodes:
            raise ValueError("relation contains unreachable or missing nodes")
        if self.root_id == _TERMINAL_ID and self.slots:
            raise ValueError("nonempty slot schema cannot have terminal root")
        if self.root_id not in (None, _TERMINAL_ID):
            self._validate_levels(arena, self.root_id, 0, set())
        expected_allocation = RelationAllocation(
            len(self.nodes), sum(len(node.edges) for node in self.nodes)
        )
        if self.allocation != expected_allocation:
            raise ValueError("relation allocation audit is invalid")
        expected_sha = _namespaced_sha256(
            _RELATION_NAMESPACE,
            _relation_payload(self.slots, self.root_id, self.nodes),
        )
        if self.sha256 != expected_sha:
            raise ValueError("relation SHA-256 is invalid")

    def _validate_levels(
        self,
        arena: _Arena,
        node_id: str,
        level: int,
        seen: set[tuple[str, int]],
    ) -> None:
        key = (node_id, level)
        if key in seen:
            return
        seen.add(key)
        if node_id == _TERMINAL_ID:
            if level != len(self.slots):
                raise ValueError("terminal appears before the final slot")
            return
        if level >= len(self.slots):
            raise ValueError("decision appears after the final slot")
        node = arena.node(node_id)
        if node.slot != self.slots[level]:
            raise ValueError("decision slots do not follow the relation schema")
        for _value, child in node.edges:
            self._validate_levels(arena, child, level + 1, seen)

    @classmethod
    def _finish(
        cls,
        slots: tuple[TypedSlot, ...],
        root_id: str | None,
        arena: _Arena,
    ) -> PackedBindingRelation:
        nodes = _reachable_nodes(root_id, arena)
        allocation = RelationAllocation(
            len(nodes), sum(len(node.edges) for node in nodes)
        )
        payload = _relation_payload(slots, root_id, nodes)
        return cls(
            slots=slots,
            root_id=root_id,
            nodes=nodes,
            allocation=allocation,
            sha256=_namespaced_sha256(_RELATION_NAMESPACE, payload),
        )

    @classmethod
    def empty(cls, slots: Sequence[TypedSlot] = ()) -> PackedBindingRelation:
        canonical = _canonical_slots(slots)
        return cls._finish(canonical, None, _Arena())

    @classmethod
    def from_bindings(
        cls,
        slots: Sequence[TypedSlot],
        bindings: Iterable[Sequence[int]],
    ) -> PackedBindingRelation:
        supplied_slots = tuple(slots)
        canonical = _canonical_slots(supplied_slots)
        source_index = {slot: index for index, slot in enumerate(supplied_slots)}
        canonical_indices = tuple(source_index[slot] for slot in canonical)
        rows: set[tuple[int, ...]] = set()
        for supplied in bindings:
            row = tuple(supplied)
            if len(row) != len(supplied_slots) or any(type(value) is not int for value in row):
                raise ValueError("binding rows must align with slots and contain exact integers")
            rows.add(tuple(row[index] for index in canonical_indices))
        if not canonical:
            return cls._finish(canonical, _TERMINAL_ID if rows else None, _Arena())
        arena = _Arena()

        def build(level: int, selected: tuple[tuple[int, ...], ...]) -> str | None:
            if level == len(canonical):
                return _TERMINAL_ID
            grouped: dict[int, list[tuple[int, ...]]] = {}
            for row in selected:
                grouped.setdefault(row[level], []).append(row)
            return arena.make(
                canonical[level],
                (
                    (value, build(level + 1, tuple(group)))
                    for value, group in sorted(grouped.items())
                ),
            )

        ordered_rows = tuple(sorted(rows))
        root = build(0, ordered_rows) if ordered_rows else None
        return cls._finish(canonical, root, arena)

    @classmethod
    def domain(
        cls,
        slot: TypedSlot,
        values: Iterable[int],
    ) -> PackedBindingRelation:
        return cls.from_bindings((slot,), ((value,) for value in values))

    @property
    def exact_count(self) -> int:
        arena = _Arena(self.nodes)
        memo: dict[str, int] = {_TERMINAL_ID: 1}

        def count(node_id: str) -> int:
            if node_id in memo:
                return memo[node_id]
            total = sum(count(child) for _value, child in arena.node(node_id).edges)
            memo[node_id] = total
            return total

        return 0 if self.root_id is None else count(self.root_id)

    def saturating_count(self, limit: int = DEFAULT_SATURATION_LIMIT) -> int:
        if type(limit) is not int or limit < 1:
            raise ValueError("saturation limit must be a positive integer")
        arena = _Arena(self.nodes)
        memo: dict[str, int] = {_TERMINAL_ID: 1}

        def count(node_id: str) -> int:
            if node_id in memo:
                return memo[node_id]
            total = 0
            for _value, child in arena.node(node_id).edges:
                total = min(limit, total + count(child))
                if total == limit:
                    break
            memo[node_id] = total
            return total

        return 0 if self.root_id is None else count(self.root_id)

    def first_bindings(self, limit: int = 2) -> tuple[tuple[int, ...], ...]:
        if type(limit) is not int or limit < 0:
            raise ValueError("binding limit must be a nonnegative integer")
        if limit == 0 or self.root_id is None:
            return ()
        arena = _Arena(self.nodes)
        results: list[tuple[int, ...]] = []

        def visit(node_id: str, prefix: tuple[int, ...]) -> None:
            if len(results) >= limit:
                return
            if node_id == _TERMINAL_ID:
                results.append(prefix)
                return
            for value, child in arena.node(node_id).edges:
                visit(child, (*prefix, value))
                if len(results) >= limit:
                    return

        visit(self.root_id, ())
        return tuple(results)

    def canonical_bytes(self) -> bytes:
        return _canonical_json(_relation_payload(self.slots, self.root_id, self.nodes))

    def authenticate(self) -> bool:
        expected = _namespaced_sha256(
            _RELATION_NAMESPACE,
            _relation_payload(self.slots, self.root_id, self.nodes),
        )
        if expected != self.sha256:
            raise RuntimeError("packed binding relation authentication failed")
        return True

    def _require_same_schema(self, other: PackedBindingRelation) -> None:
        if not isinstance(other, PackedBindingRelation):
            raise TypeError("relation operation requires PackedBindingRelation")
        if self.slots != other.slots:
            raise ValueError("relation schemas do not match")

    def _set_operation(
        self,
        other: PackedBindingRelation,
        operation: Literal["union", "difference"],
    ) -> PackedBindingRelation:
        self._require_same_schema(other)
        arena = _Arena((*self.nodes, *other.nodes))
        memo: dict[tuple[str | None, str | None, int], str | None] = {}

        def apply(
            left_id: str | None,
            right_id: str | None,
            level: int,
        ) -> str | None:
            key = (left_id, right_id, level)
            if key in memo:
                return memo[key]
            if operation == "union":
                if left_id is None:
                    return right_id
                if right_id is None or left_id == right_id:
                    return left_id
            else:
                if left_id is None or left_id == right_id:
                    return None
                if right_id is None:
                    return left_id
            if level == len(self.slots):
                result = _TERMINAL_ID if operation == "union" else None
                memo[key] = result
                return result
            left_edges = dict(arena.node(left_id).edges)
            right_edges = dict(arena.node(right_id).edges)
            values = (
                sorted(set(left_edges) | set(right_edges))
                if operation == "union"
                else sorted(left_edges)
            )
            result = arena.make(
                self.slots[level],
                (
                    (
                        value,
                        apply(
                            left_edges.get(value),
                            right_edges.get(value),
                            level + 1,
                        ),
                    )
                    for value in values
                ),
            )
            memo[key] = result
            return result

        root = apply(self.root_id, other.root_id, 0)
        return self._finish(self.slots, root, arena)

    def union(self, other: PackedBindingRelation) -> PackedBindingRelation:
        return self._set_operation(other, "union")

    def difference(self, other: PackedBindingRelation) -> PackedBindingRelation:
        return self._set_operation(other, "difference")

    def join(self, other: PackedBindingRelation) -> PackedBindingRelation:
        """Natural join; disjoint schemas are a canonical Cartesian product."""

        if not isinstance(other, PackedBindingRelation):
            raise TypeError("join requires PackedBindingRelation")
        output_slots = _canonical_slots(
            tuple(set(self.slots).union(other.slots))
        )
        if self.root_id is None or other.root_id is None:
            return self.empty(output_slots)
        arena = _Arena((*self.nodes, *other.nodes))
        left_positions = {slot: index for index, slot in enumerate(self.slots)}
        right_positions = {slot: index for index, slot in enumerate(other.slots)}
        memo: dict[
            tuple[int, int, str, int, str], str | None
        ] = {}

        def apply(
            output_level: int,
            left_level: int,
            left_id: str,
            right_level: int,
            right_id: str,
        ) -> str | None:
            key = (output_level, left_level, left_id, right_level, right_id)
            if key in memo:
                return memo[key]
            if output_level == len(output_slots):
                result = (
                    _TERMINAL_ID
                    if left_id == _TERMINAL_ID and right_id == _TERMINAL_ID
                    else None
                )
                memo[key] = result
                return result
            slot = output_slots[output_level]
            left_consumes = slot in left_positions
            right_consumes = slot in right_positions
            left_edges = (
                dict(arena.node(left_id).edges) if left_consumes else None
            )
            right_edges = (
                dict(arena.node(right_id).edges) if right_consumes else None
            )
            if left_edges is not None and right_edges is not None:
                values = sorted(set(left_edges) & set(right_edges))
            elif left_edges is not None:
                values = sorted(left_edges)
            elif right_edges is not None:
                values = sorted(right_edges)
            else:
                raise RuntimeError("join output slot belongs to neither input")
            result = arena.make(
                slot,
                (
                    (
                        value,
                        apply(
                            output_level + 1,
                            left_level + int(left_consumes),
                            left_edges[value] if left_edges is not None else left_id,
                            right_level + int(right_consumes),
                            right_edges[value] if right_edges is not None else right_id,
                        ),
                    )
                    for value in values
                ),
            )
            memo[key] = result
            return result

        root = apply(0, 0, self.root_id, 0, other.root_id)
        return self._finish(output_slots, root, arena)

    def cartesian_product(
        self,
        other: PackedBindingRelation,
    ) -> PackedBindingRelation:
        if set(self.slots) & set(other.slots):
            raise ValueError("Cartesian product requires disjoint slot schemas")
        return self.join(other)

    def select(
        self,
        constraints: Sequence[SlotConstraint],
    ) -> PackedBindingRelation:
        if isinstance(constraints, (str, bytes)):
            raise TypeError("constraints must be a sequence")
        checks = tuple(constraints)
        if not all(isinstance(item, SlotConstraint) for item in checks):
            raise TypeError("constraints must contain SlotConstraint values")
        schema = set(self.slots)
        if any(not item.required_slots <= schema for item in checks):
            raise ValueError("constraint references a slot outside the relation")
        if self.root_id is None or not checks:
            return self
        source = _Arena(self.nodes)
        output = _Arena()

        def visit(
            node_id: str,
            level: int,
            binding: dict[TypedSlot, int],
        ) -> str | None:
            if node_id == _TERMINAL_ID:
                return _TERMINAL_ID
            node = source.node(node_id)
            edges: list[tuple[int, str | None]] = []
            for value, child in node.edges:
                binding[node.slot] = value
                outcomes = tuple(check.evaluate(binding) for check in checks)
                if False not in outcomes:
                    edges.append((value, visit(child, level + 1, binding)))
                binding.pop(node.slot)
            return output.make(self.slots[level], edges)

        root = visit(self.root_id, 0, {})
        return self._finish(self.slots, root, output)

    def project_distinct(
        self,
        retained_slots: Sequence[TypedSlot],
    ) -> PackedBindingRelation:
        retained = _canonical_slots(retained_slots)
        if not set(retained) <= set(self.slots):
            raise ValueError("projection retains a slot outside the relation")
        if retained == self.slots:
            return self
        source = _Arena(self.nodes)
        output = _Arena()
        retained_set = set(retained)
        new_level = {slot: index for index, slot in enumerate(retained)}
        memo: dict[tuple[str, int], str | None] = {}

        def union_ids(
            left: str | None,
            right: str | None,
            level: int,
        ) -> str | None:
            if left is None:
                return right
            if right is None or left == right:
                return left
            if level == len(retained):
                return _TERMINAL_ID
            left_edges = dict(output.node(left).edges)
            right_edges = dict(output.node(right).edges)
            return output.make(
                retained[level],
                (
                    (
                        value,
                        union_ids(
                            left_edges.get(value),
                            right_edges.get(value),
                            level + 1,
                        ),
                    )
                    for value in sorted(set(left_edges) | set(right_edges))
                ),
            )

        def visit(node_id: str, level: int) -> str | None:
            key = (node_id, level)
            if key in memo:
                return memo[key]
            if node_id == _TERMINAL_ID:
                result: str | None = _TERMINAL_ID
            else:
                node = source.node(node_id)
                if node.slot in retained_set:
                    result = output.make(
                        node.slot,
                        ((value, visit(child, level + 1)) for value, child in node.edges),
                    )
                else:
                    result = None
                    output_level = sum(
                        1 for slot in self.slots[:level] if slot in retained_set
                    )
                    for _value, child in node.edges:
                        result = union_ids(
                            result,
                            visit(child, level + 1),
                            output_level,
                        )
            memo[key] = result
            return result

        root = None if self.root_id is None else visit(self.root_id, 0)
        return self._finish(retained, root, output)

    def _rename_slots(
        self,
        target_slots: tuple[TypedSlot, ...],
    ) -> PackedBindingRelation:
        if len(target_slots) != len(self.slots):
            raise ValueError("slot rename must preserve arity")
        if target_slots != _canonical_slots(target_slots):
            raise ValueError("renamed slots must preserve canonical order")
        source = _Arena(self.nodes)
        output = _Arena()
        memo: dict[tuple[str, int], str] = {}

        def visit(node_id: str, level: int) -> str:
            if node_id == _TERMINAL_ID:
                return _TERMINAL_ID
            key = (node_id, level)
            if key in memo:
                return memo[key]
            node = source.node(node_id)
            renamed = output.make(
                target_slots[level],
                ((value, visit(child, level + 1)) for value, child in node.edges),
            )
            if renamed is None:
                raise RuntimeError("nonempty rename lost all decision edges")
            memo[key] = renamed
            return renamed

        root = None if self.root_id is None else visit(self.root_id, 0)
        return self._finish(target_slots, root, output)


@dataclass(frozen=True, slots=True)
class ProofNormalizationProjection:
    """Explicit proof rewrite from one family/schema into another."""

    normalization_id: str
    source_family_id: str
    target_family_id: str
    retained_source_slots: tuple[TypedSlot, ...]
    target_slots: tuple[TypedSlot, ...]
    required_constants: tuple[tuple[TypedSlot, int], ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("normalization_id", self.normalization_id),
            ("source_family_id", self.source_family_id),
            ("target_family_id", self.target_family_id),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{field_name} must be a nonempty string")
        if self.retained_source_slots != _canonical_slots(self.retained_source_slots):
            raise ValueError("normalization source slots must be canonical")
        if self.target_slots != _canonical_slots(self.target_slots):
            raise ValueError("normalization target slots must be canonical")
        if len(self.retained_source_slots) != len(self.target_slots):
            raise ValueError("normalization projection must preserve retained arity")
        constants = tuple(sorted(self.required_constants, key=lambda item: item[0].sort_key))
        if constants != self.required_constants or len({slot for slot, _ in constants}) != len(constants):
            raise ValueError("normalization constants must be uniquely sorted")
        if any(not isinstance(slot, TypedSlot) or type(value) is not int for slot, value in constants):
            raise TypeError("normalization constants must be typed slots and integers")


@dataclass(frozen=True, slots=True)
class PackedFamilySlice:
    """A canonical proof family paired with its distinct finite bindings."""

    family_id: str
    relation: PackedBindingRelation
    sha256: str

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise ValueError("family_id must be a nonempty string")
        if not isinstance(self.relation, PackedBindingRelation):
            raise TypeError("family relation must be PackedBindingRelation")
        expected = _namespaced_sha256(
            _SLICE_NAMESPACE,
            {"family_id": self.family_id, "relation_sha256": self.relation.sha256},
        )
        if self.sha256 != expected:
            raise ValueError("family slice SHA-256 is invalid")

    @classmethod
    def build(
        cls,
        family_id: str,
        relation: PackedBindingRelation,
    ) -> PackedFamilySlice:
        digest = _namespaced_sha256(
            _SLICE_NAMESPACE,
            {"family_id": family_id, "relation_sha256": relation.sha256},
        )
        return cls(family_id, relation, digest)

    def normalize(
        self,
        projection: ProofNormalizationProjection,
    ) -> PackedFamilySlice:
        if not isinstance(projection, ProofNormalizationProjection):
            raise TypeError("normalization requires ProofNormalizationProjection")
        if self.family_id != projection.source_family_id:
            raise ValueError("normalization source family does not match")
        schema = set(self.relation.slots)
        required_slots = {slot for slot, _value in projection.required_constants}
        if not set(projection.retained_source_slots) | required_slots <= schema:
            raise ValueError("normalization references slots outside the relation")
        constraints = tuple(
            SlotConstraint.value(slot, "eq", value)
            for slot, value in projection.required_constants
        )
        selected = self.relation.select(constraints)
        projected = selected.project_distinct(projection.retained_source_slots)
        renamed = projected._rename_slots(projection.target_slots)
        return PackedFamilySlice.build(projection.target_family_id, renamed)


@dataclass(frozen=True, slots=True)
class ProofWitness:
    family_id: str
    binding: tuple[int, ...]


def _forest_payload(
    slices: tuple[PackedFamilySlice, ...],
    saturation_limit: int,
    exact_count: int,
    saturating_count: int,
    first_two: tuple[ProofWitness, ...],
) -> dict[str, object]:
    return {
        "schema": "typed-semantic-packed-proof-forest-v1",
        "slices": tuple(
            {
                "family_id": item.family_id,
                "relation_sha256": item.relation.sha256,
                "slice_sha256": item.sha256,
            }
            for item in slices
        ),
        "saturation_limit": saturation_limit,
        "exact_logical_recipe_count": exact_count,
        "saturating_logical_recipe_count": saturating_count,
        "first_two_witnesses": tuple(
            (item.family_id, item.binding) for item in first_two
        ),
    }


@dataclass(frozen=True, slots=True)
class PackedProofForest:
    """Merkle-authenticated OR forest of canonical proof-family slices."""

    slices: tuple[PackedFamilySlice, ...]
    saturation_limit: int
    exact_logical_recipe_count: int
    saturating_logical_recipe_count: int
    first_two_witnesses: tuple[ProofWitness, ...]
    sha256: str

    def __post_init__(self) -> None:
        if type(self.saturation_limit) is not int or self.saturation_limit < 1:
            raise ValueError("forest saturation limit must be positive")
        if tuple(sorted(self.slices, key=lambda item: item.family_id)) != self.slices:
            raise ValueError("forest slices must be sorted by family")
        if len({item.family_id for item in self.slices}) != len(self.slices):
            raise ValueError("forest may contain only one slice per family")
        exact = sum(item.relation.exact_count for item in self.slices)
        saturated = min(self.saturation_limit, exact)
        witnesses = tuple(
            sorted(
                (
                    ProofWitness(item.family_id, binding)
                    for item in self.slices
                    for binding in item.relation.first_bindings(2)
                ),
                key=lambda item: (item.family_id, item.binding),
            )[:2]
        )
        if self.exact_logical_recipe_count != exact:
            raise ValueError("forest exact logical count is invalid")
        if self.saturating_logical_recipe_count != saturated:
            raise ValueError("forest saturating logical count is invalid")
        if self.first_two_witnesses != witnesses:
            raise ValueError("forest first witnesses are invalid")
        expected = _namespaced_sha256(
            _FOREST_NAMESPACE,
            _forest_payload(
                self.slices,
                self.saturation_limit,
                exact,
                saturated,
                witnesses,
            ),
        )
        if self.sha256 != expected:
            raise ValueError("proof forest SHA-256 is invalid")

    @classmethod
    def build(
        cls,
        slices: Iterable[PackedFamilySlice] = (),
        *,
        saturation_limit: int = DEFAULT_SATURATION_LIMIT,
    ) -> PackedProofForest:
        if type(saturation_limit) is not int or saturation_limit < 1:
            raise ValueError("forest saturation limit must be positive")
        grouped: dict[str, PackedBindingRelation] = {}
        for item in slices:
            if not isinstance(item, PackedFamilySlice):
                raise TypeError("forest requires PackedFamilySlice values")
            incumbent = grouped.get(item.family_id)
            grouped[item.family_id] = (
                item.relation if incumbent is None else incumbent.union(item.relation)
            )
        canonical = tuple(
            PackedFamilySlice.build(family_id, relation)
            for family_id, relation in sorted(grouped.items())
        )
        exact = sum(item.relation.exact_count for item in canonical)
        saturated = min(saturation_limit, exact)
        witnesses = tuple(
            sorted(
                (
                    ProofWitness(item.family_id, binding)
                    for item in canonical
                    for binding in item.relation.first_bindings(2)
                ),
                key=lambda item: (item.family_id, item.binding),
            )[:2]
        )
        digest = _namespaced_sha256(
            _FOREST_NAMESPACE,
            _forest_payload(
                canonical,
                saturation_limit,
                exact,
                saturated,
                witnesses,
            ),
        )
        return cls(
            slices=canonical,
            saturation_limit=saturation_limit,
            exact_logical_recipe_count=exact,
            saturating_logical_recipe_count=saturated,
            first_two_witnesses=witnesses,
            sha256=digest,
        )

    def union(self, other: PackedProofForest) -> PackedProofForest:
        if not isinstance(other, PackedProofForest):
            raise TypeError("forest union requires PackedProofForest")
        if self.saturation_limit != other.saturation_limit:
            raise ValueError("forest saturation limits do not match")
        return self.build(
            (*self.slices, *other.slices),
            saturation_limit=self.saturation_limit,
        )

    @property
    def allocation(self) -> RelationAllocation:
        return RelationAllocation(
            sum(item.relation.allocation.decision_nodes for item in self.slices),
            sum(item.relation.allocation.decision_edges for item in self.slices),
        )

    def canonical_bytes(self) -> bytes:
        return _canonical_json(
            _forest_payload(
                self.slices,
                self.saturation_limit,
                self.exact_logical_recipe_count,
                self.saturating_logical_recipe_count,
                self.first_two_witnesses,
            )
        )

    def authenticate(self) -> bool:
        expected = _namespaced_sha256(
            _FOREST_NAMESPACE,
            _forest_payload(
                self.slices,
                self.saturation_limit,
                self.exact_logical_recipe_count,
                self.saturating_logical_recipe_count,
                self.first_two_witnesses,
            ),
        )
        if expected != self.sha256:
            raise RuntimeError("packed proof forest authentication failed")
        return True


__all__ = [
    "CONSTRAINT_OPERATORS",
    "DEFAULT_SATURATION_LIMIT",
    "SLOT_KINDS",
    "DecisionNode",
    "PackedBindingRelation",
    "PackedFamilySlice",
    "PackedProofForest",
    "ProofNormalizationProjection",
    "ProofWitness",
    "RelationAllocation",
    "SlotConstraint",
    "TypedSlot",
]
