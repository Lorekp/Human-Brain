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
_CACHE_ENTRY_NAMESPACE = b"typed-semantic-packed-bindings-v1-cache-entry\0"


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
        if type(self.edges) is not tuple or not self.edges:
            raise ValueError("decision edges must be a nonempty tuple")

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


@dataclass(frozen=True, slots=True)
class PackedRelationWorkAudit:
    """Exact cumulative topology work performed by one relation session."""

    limit: int
    completed_units: int
    attempted_units: int
    denied_event: str | None
    operation_lookups: int
    cache_hits: int
    domain_value_reads: int
    binding_row_reads: int
    decision_node_reads: int
    decision_edge_reads: int
    decision_node_proposals: int
    decision_edge_proposals: int
    proof_unions: int
    event_counts: tuple[tuple[str, int], ...]
    event_sha256: str


class PackedRelationWorkLimitExceeded(RuntimeError):
    """Raised immediately before executing work unit ``limit + 1``."""

    def __init__(
        self,
        *,
        limit: int,
        completed: int,
        attempted: int,
        event: str,
    ) -> None:
        super().__init__(
            f"packed relation work limit {limit} exceeded by {event!r}"
        )
        self.limit = limit
        self.completed = completed
        self.attempted = attempted
        self.event = event


@dataclass(frozen=True, slots=True)
class _AuthenticatedCacheEntry:
    operation_key: str
    result_type: str
    result_identity: str
    value: object
    sha256: str

    @staticmethod
    def payload(
        operation_key: str,
        result_type: str,
        result_identity: str,
    ) -> dict[str, str]:
        return {
            "operation_key": operation_key,
            "result_type": result_type,
            "result_identity": result_identity,
        }


class _Arena:
    """Private content-addressed node arena used while composing relations."""

    def __init__(self) -> None:
        self.nodes: dict[str, DecisionNode] = {}

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
        edges: tuple[tuple[int, str], ...],
    ) -> str | None:
        if not edges:
            return None
        node_id = _node_id(slot, edges)
        node = DecisionNode(node_id, slot, edges)
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
        if type(self.slots) is not tuple or not all(
            isinstance(slot, TypedSlot) for slot in self.slots
        ):
            raise ValueError("relation slots must be a tuple")
        if type(self.nodes) is not tuple:
            raise ValueError("relation nodes must be a tuple")
        if not isinstance(self.allocation, RelationAllocation):
            raise TypeError("relation allocation must be RelationAllocation")
        if type(self.sha256) is not str or len(self.sha256) != 64:
            raise ValueError("relation SHA-256 must be a digest")


@dataclass(frozen=True, slots=True)
class _ValidatedRelation:
    """Ephemeral proof that one live relation was traversed in this session."""

    arena: _Arena
    canonical_bytes: bytes

class PackedRelationSession:
    """The sole execution seam for packed relation algebra.

    A work unit is charged before every operation lookup, supplied domain
    value or concrete row is processed, live decision node or edge is read,
    output decision node or edge is proposed, and proof-family union is
    executed.  Cached values are recursively authenticated on every hit.
    """

    __slots__ = (
        "_attempted_units",
        "_cache",
        "_cache_hits",
        "_completed_units",
        "_denied_event",
        "_event_counts",
        "_event_digest",
        "limit",
    )

    def __init__(self, *, work_limit: int) -> None:
        if type(work_limit) is not int or work_limit < 1:
            raise ValueError("packed relation work_limit must be positive")
        self.limit = work_limit
        self._completed_units = 0
        self._attempted_units = 0
        self._denied_event: str | None = None
        self._event_counts: dict[str, int] = {}
        self._event_digest = sha256()
        self._event_digest.update(b"typed-semantic-packed-relation-work-v1\0")
        self._cache: dict[str, _AuthenticatedCacheEntry] = {}
        self._cache_hits = 0

    def _charge(self, event: str) -> None:
        attempted = self._completed_units + 1
        self._attempted_units = attempted
        if attempted > self.limit:
            self._denied_event = event
            raise PackedRelationWorkLimitExceeded(
                limit=self.limit,
                completed=self._completed_units,
                attempted=attempted,
                event=event,
            )
        self._completed_units = attempted
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        self._event_digest.update(event.encode("utf-8"))
        self._event_digest.update(b"\0")

    def _begin(self, operation: str) -> None:
        self._charge(f"operation:{operation}")

    def _proof_union(self) -> None:
        self._charge("proof-union")

    @property
    def audit(self) -> PackedRelationWorkAudit:
        counts = tuple(sorted(self._event_counts.items()))
        return PackedRelationWorkAudit(
            limit=self.limit,
            completed_units=self._completed_units,
            attempted_units=self._attempted_units,
            denied_event=self._denied_event,
            operation_lookups=sum(
                count for event, count in counts if event.startswith("operation:")
            ),
            cache_hits=self._cache_hits,
            domain_value_reads=dict(counts).get("domain-value-read", 0),
            binding_row_reads=dict(counts).get("binding-row-read", 0),
            decision_node_reads=dict(counts).get("decision-node-read", 0),
            decision_edge_reads=dict(counts).get("decision-edge-read", 0),
            decision_node_proposals=dict(counts).get(
                "decision-node-proposal", 0
            ),
            decision_edge_proposals=dict(counts).get(
                "decision-edge-proposal", 0
            ),
            proof_unions=dict(counts).get("proof-union", 0),
            event_counts=counts,
            event_sha256=self._event_digest.hexdigest(),
        )

    @staticmethod
    def _key(operation: str, payload: object) -> str:
        return _namespaced_sha256(
            b"typed-semantic-packed-relation-session-cache-v1\0",
            {"operation": operation, "payload": payload},
        )

    @staticmethod
    def _type_identity(value_type: type) -> str:
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @staticmethod
    def _cache_entry_digest(
        operation_key: str,
        result_type: str,
        result_identity: str,
    ) -> str:
        return _namespaced_sha256(
            _CACHE_ENTRY_NAMESPACE,
            _AuthenticatedCacheEntry.payload(
                operation_key,
                result_type,
                result_identity,
            ),
        )

    def _authenticate_value(self, value: object) -> None:
        if type(value) is PackedBindingRelation:
            self._authenticate_relation(value)
            return
        if type(value) is PackedFamilySlice:
            self._authenticate_slice(value)
            return
        if type(value) is PackedProofForest:
            self._authenticate_forest(value)
            return
        raise RuntimeError("packed session value has an unsupported result type")

    @staticmethod
    def _result_identity(value: object) -> str:
        identity = getattr(value, "sha256", None)
        if type(identity) is not str or len(identity) != 64:
            raise RuntimeError("packed session result identity is invalid")
        return identity

    def _cached(self, key: str, expected_type: type) -> object | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if type(entry) is not _AuthenticatedCacheEntry:
            raise RuntimeError("packed relation cache entry is unauthenticated")
        expected_type_identity = self._type_identity(expected_type)
        expected_entry_digest = self._cache_entry_digest(
            key,
            expected_type_identity,
            entry.result_identity,
        )
        if (
            entry.operation_key != key
            or entry.result_type != expected_type_identity
            or entry.sha256 != expected_entry_digest
            or type(entry.value) is not expected_type
        ):
            raise RuntimeError("packed relation cache entry authentication failed")
        self._authenticate_value(entry.value)
        if self._result_identity(entry.value) != entry.result_identity:
            raise RuntimeError("packed relation cache result identity changed")
        self._cache_hits += 1
        return entry.value

    def _store(self, key: str, value: object) -> object:
        self._authenticate_value(value)
        result_type = self._type_identity(type(value))
        result_identity = self._result_identity(value)
        entry = _AuthenticatedCacheEntry(
            operation_key=key,
            result_type=result_type,
            result_identity=result_identity,
            value=value,
            sha256=self._cache_entry_digest(key, result_type, result_identity),
        )
        incumbent = self._cache.setdefault(key, entry)
        if incumbent != entry:
            raise RuntimeError("packed relation cache identity collision")
        return incumbent.value

    def _read_node(self, arena: _Arena, node_id: str) -> DecisionNode:
        self._charge("decision-node-read")
        return arena.node(node_id)

    def _read_edges(
        self,
        arena: _Arena,
        node_id: str,
    ) -> tuple[tuple[int, str], ...]:
        _node, edges = self._read_node_edges(arena, node_id)
        return edges

    def _read_node_edges(
        self,
        arena: _Arena,
        node_id: str,
    ) -> tuple[DecisionNode, tuple[tuple[int, str], ...]]:
        node = self._read_node(arena, node_id)
        edges = node.edges
        for _index in range(len(edges)):
            self._charge("decision-edge-read")
        return node, edges

    def _merge_arenas(self, *sources: _Arena) -> _Arena:
        merged = _Arena()
        for source in sources:
            for node_id in sorted(source.nodes):
                self._charge("decision-node-read")
                node = source.node(node_id)
                incumbent = merged.nodes.setdefault(node_id, node)
                if incumbent != node:
                    raise RuntimeError("decision-node hash collision")
        return merged

    def _make(
        self,
        arena: _Arena,
        slot: TypedSlot,
        edges: Iterable[tuple[int, str | None]],
    ) -> str | None:
        proposed: list[tuple[int, str | None]] = []
        for edge in edges:
            self._charge("decision-edge-proposal")
            proposed.append(edge)
        self._charge("decision-node-proposal")
        canonical = tuple(
            (value, child)
            for value, child in sorted(proposed)
            if child is not None
        )
        return arena.make(slot, canonical)

    @staticmethod
    def _digest_canonical(namespace: bytes, canonical: bytes) -> str:
        digest = sha256()
        digest.update(namespace)
        digest.update(canonical)
        return digest.hexdigest()

    def _authenticate_relation(
        self,
        relation: PackedBindingRelation,
    ) -> _ValidatedRelation:
        try:
            if type(relation) is not PackedBindingRelation:
                raise TypeError("relation operation requires PackedBindingRelation")
            if relation.slots != _canonical_slots(relation.slots):
                raise ValueError("relation slots are not canonical")
            if type(relation.root_id) is not str and relation.root_id is not None:
                raise ValueError("relation root must be a SHA-256 or None")
            if relation.root_id is not None and len(relation.root_id) != 64:
                raise ValueError("relation root must be a SHA-256 or None")
            if type(relation.nodes) is not tuple:
                raise TypeError("relation nodes must be a tuple")
            if type(relation.sha256) is not str or len(relation.sha256) != 64:
                raise ValueError("relation SHA-256 is invalid")
            if type(relation.allocation) is not RelationAllocation:
                raise TypeError("relation allocation is invalid")
            if (
                type(relation.allocation.decision_nodes) is not int
                or relation.allocation.decision_nodes < 0
                or type(relation.allocation.decision_edges) is not int
                or relation.allocation.decision_edges < 0
            ):
                raise ValueError("relation allocation counts are invalid")

            arena = _Arena()
            node_records: dict[str, dict[str, object]] = {}
            supplied_ids: list[str] = []
            edge_total = 0
            for node_index in range(len(relation.nodes)):
                self._charge("decision-node-read")
                node = relation.nodes[node_index]
                if type(node) is not DecisionNode:
                    raise TypeError("relation node record is invalid")
                node_id = node.node_id
                slot = node.slot
                edges = node.edges
                if type(node_id) is not str or len(node_id) != 64:
                    raise ValueError("decision node_id must be a SHA-256")
                if not isinstance(slot, TypedSlot):
                    raise TypeError("decision node slot must be TypedSlot")
                if type(edges) is not tuple or not edges:
                    raise ValueError("decision edges must be a nonempty tuple")
                captured_edges: list[tuple[int, str]] = []
                for edge_index in range(len(edges)):
                    self._charge("decision-edge-read")
                    edge = edges[edge_index]
                    if type(edge) is not tuple or len(edge) != 2:
                        raise ValueError("decision edge record is invalid")
                    value, child = edge
                    if (
                        type(value) is not int
                        or type(child) is not str
                        or len(child) != 64
                    ):
                        raise ValueError("decision edge value or child is invalid")
                    captured_edges.append((value, child))
                captured = tuple(captured_edges)
                values = tuple(value for value, _child in captured)
                if values != tuple(sorted(set(values))):
                    raise ValueError("decision edges are not uniquely sorted")
                if node_id != _node_id(slot, captured):
                    raise ValueError("decision node content hash is invalid")
                incumbent = arena.nodes.setdefault(node_id, node)
                if incumbent is not node:
                    raise ValueError("relation contains duplicate decision nodes")
                supplied_ids.append(node_id)
                edge_total += len(captured)
                node_records[node_id] = {
                    "node_id": node_id,
                    "slot": slot.to_record(),
                    "edges": captured,
                }
            if tuple(supplied_ids) != tuple(sorted(set(supplied_ids))):
                raise ValueError("relation nodes are not uniquely sorted")

            seen_levels: dict[str, int] = {}
            if relation.root_id == _TERMINAL_ID:
                if relation.slots:
                    raise ValueError("terminal root cannot skip relation slots")
            elif relation.root_id is not None:
                stack = [(relation.root_id, 0)]
                while stack:
                    node_id, level = stack.pop()
                    if node_id == _TERMINAL_ID:
                        if level != len(relation.slots):
                            raise ValueError("terminal reached at the wrong level")
                        continue
                    prior_level = seen_levels.get(node_id)
                    if prior_level is not None:
                        if prior_level != level:
                            raise ValueError("decision node is reused at two levels")
                        continue
                    if level >= len(relation.slots):
                        raise ValueError("decision node exceeds relation schema")
                    node, edges = self._read_node_edges(arena, node_id)
                    if node.slot != relation.slots[level]:
                        raise ValueError("decision node slot order is invalid")
                    seen_levels[node_id] = level
                    for _value, child in reversed(edges):
                        if child != _TERMINAL_ID and child not in arena.nodes:
                            raise ValueError("decision edge references an unknown node")
                        stack.append((child, level + 1))
            if set(supplied_ids) != set(seen_levels):
                raise ValueError("relation contains unreachable decision nodes")

            if (
                relation.allocation.decision_nodes != len(supplied_ids)
                or relation.allocation.decision_edges != edge_total
            ):
                raise ValueError("relation allocation is invalid")
            payload = {
                "schema": "typed-semantic-packed-binding-relation-v1",
                "slots": tuple(slot.to_record() for slot in relation.slots),
                "root_id": relation.root_id,
                "terminal_id": _TERMINAL_ID,
                "nodes": tuple(node_records[node_id] for node_id in supplied_ids),
                "allocation": {
                    "decision_nodes": len(supplied_ids),
                    "decision_edges": edge_total,
                },
            }
            canonical = _canonical_json(payload)
            if relation.sha256 != self._digest_canonical(
                _RELATION_NAMESPACE,
                canonical,
            ):
                raise ValueError("relation SHA-256 is invalid")
            return _ValidatedRelation(arena=arena, canonical_bytes=canonical)
        except PackedRelationWorkLimitExceeded:
            raise
        except Exception as error:
            raise RuntimeError("packed relation authentication failed") from error

    def _finish(
        self,
        slots: tuple[TypedSlot, ...],
        root_id: str | None,
        arena: _Arena,
    ) -> PackedBindingRelation:
        if slots != _canonical_slots(slots):
            raise ValueError("relation slots are not canonical")
        if root_id is not None and (type(root_id) is not str or len(root_id) != 64):
            raise ValueError("relation root must be a SHA-256 or None")
        seen_levels: dict[str, int] = {}
        captured_nodes: dict[str, DecisionNode] = {}
        node_records: dict[str, dict[str, object]] = {}
        edge_total = 0
        if root_id == _TERMINAL_ID:
            if slots:
                raise ValueError("terminal root cannot skip relation slots")
        elif root_id is not None:
            stack = [(root_id, 0)]
            while stack:
                node_id, level = stack.pop()
                if node_id == _TERMINAL_ID:
                    if level != len(slots):
                        raise ValueError("terminal reached at the wrong level")
                    continue
                prior_level = seen_levels.get(node_id)
                if prior_level is not None:
                    if prior_level != level:
                        raise ValueError("decision node is reused at two levels")
                    continue
                if level >= len(slots):
                    raise ValueError("decision node exceeds relation schema")
                node, edges = self._read_node_edges(arena, node_id)
                if node.slot != slots[level]:
                    raise ValueError("decision node slot order is invalid")
                seen_levels[node_id] = level
                captured_nodes[node_id] = node
                edge_total += len(edges)
                node_records[node_id] = {
                    "node_id": node.node_id,
                    "slot": node.slot.to_record(),
                    "edges": edges,
                }
                for _value, child in reversed(edges):
                    if child != _TERMINAL_ID and child not in arena.nodes:
                        raise ValueError("decision edge references an unknown node")
                    stack.append((child, level + 1))
        node_ids = tuple(sorted(captured_nodes))
        nodes = tuple(captured_nodes[node_id] for node_id in node_ids)
        allocation = RelationAllocation(len(nodes), edge_total)
        payload = {
            "schema": "typed-semantic-packed-binding-relation-v1",
            "slots": tuple(slot.to_record() for slot in slots),
            "root_id": root_id,
            "terminal_id": _TERMINAL_ID,
            "nodes": tuple(node_records[node_id] for node_id in node_ids),
            "allocation": {
                "decision_nodes": allocation.decision_nodes,
                "decision_edges": allocation.decision_edges,
            },
        }
        canonical = _canonical_json(payload)
        digest = self._digest_canonical(_RELATION_NAMESPACE, canonical)
        return PackedBindingRelation(slots, root_id, nodes, allocation, digest)

    def _require_relation(
        self,
        value: object,
    ) -> tuple[PackedBindingRelation, _ValidatedRelation]:
        if type(value) is not PackedBindingRelation:
            raise TypeError("relation operation requires PackedBindingRelation")
        return value, self._authenticate_relation(value)

    def _exact_count(
        self,
        relation: PackedBindingRelation,
        validated: _ValidatedRelation,
    ) -> int:
        memo: dict[str, int] = {}

        def visit(node_id: str | None) -> int:
            if node_id is None:
                return 0
            if node_id == _TERMINAL_ID:
                return 1
            if node_id in memo:
                return memo[node_id]
            total = sum(
                visit(child)
                for _value, child in self._read_edges(validated.arena, node_id)
            )
            memo[node_id] = total
            return total

        return visit(relation.root_id)

    def _saturating_count(
        self,
        relation: PackedBindingRelation,
        validated: _ValidatedRelation,
        limit: int,
    ) -> int:
        memo: dict[str, int] = {}

        def visit(node_id: str | None) -> int:
            if node_id is None:
                return 0
            if node_id == _TERMINAL_ID:
                return 1
            if node_id in memo:
                return memo[node_id]
            total = 0
            for _value, child in self._read_edges(validated.arena, node_id):
                total = min(limit, total + visit(child))
                if total == limit:
                    break
            memo[node_id] = total
            return total

        return visit(relation.root_id)

    def _first_bindings(
        self,
        relation: PackedBindingRelation,
        validated: _ValidatedRelation,
        limit: int,
    ) -> tuple[tuple[int, ...], ...]:
        memo: dict[tuple[str, int], tuple[tuple[int, ...], ...]] = {}

        def visit(node_id: str | None, remaining: int) -> tuple[tuple[int, ...], ...]:
            if remaining <= 0 or node_id is None:
                return ()
            if node_id == _TERMINAL_ID:
                return ((),)
            memo_key = (node_id, remaining)
            if memo_key in memo:
                return memo[memo_key]
            rows: list[tuple[int, ...]] = []
            for value, child in self._read_edges(validated.arena, node_id):
                for suffix in visit(child, remaining - len(rows)):
                    rows.append((value, *suffix))
                    if len(rows) == remaining:
                        result = tuple(rows)
                        memo[memo_key] = result
                        return result
            result = tuple(rows)
            memo[memo_key] = result
            return result

        return visit(relation.root_id, limit)

    def authenticate_relation(self, relation: PackedBindingRelation) -> bool:
        self._begin("authenticate-relation")
        self._require_relation(relation)
        return True

    def exact_count(self, relation: PackedBindingRelation) -> int:
        self._begin("exact-count")
        relation, validated = self._require_relation(relation)
        return self._exact_count(relation, validated)

    def saturating_count(
        self,
        relation: PackedBindingRelation,
        limit: int = DEFAULT_SATURATION_LIMIT,
    ) -> int:
        self._begin("saturating-count")
        if type(limit) is not int or limit < 1:
            raise ValueError("saturating count limit must be positive")
        relation, validated = self._require_relation(relation)
        return self._saturating_count(relation, validated, limit)

    def first_bindings(
        self,
        relation: PackedBindingRelation,
        limit: int = 2,
    ) -> tuple[tuple[int, ...], ...]:
        self._begin("first-bindings")
        if type(limit) is not int or limit < 0:
            raise ValueError("binding witness limit must be nonnegative")
        relation, validated = self._require_relation(relation)
        return self._first_bindings(relation, validated, limit)

    def canonical_bytes(self, relation: PackedBindingRelation) -> bytes:
        self._begin("canonical-bytes")
        _relation, validated = self._require_relation(relation)
        return validated.canonical_bytes

    def empty(
        self,
        slots: Sequence[TypedSlot] = (),
    ) -> PackedBindingRelation:
        self._begin("empty")
        canonical = _canonical_slots(slots)
        key = self._key("empty", tuple(slot.to_record() for slot in canonical))
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        return self._store(
            key,
            self._finish(canonical, None, _Arena()),
        )  # type: ignore[return-value]

    def from_bindings(
        self,
        slots: Sequence[TypedSlot],
        bindings: Iterable[Sequence[int]],
    ) -> PackedBindingRelation:
        self._begin("from-bindings")
        supplied_slots = tuple(slots)
        canonical = _canonical_slots(supplied_slots)
        source_index = {slot: index for index, slot in enumerate(supplied_slots)}
        canonical_indices = tuple(source_index[slot] for slot in canonical)
        rows: set[tuple[int, ...]] = set()
        for supplied in bindings:
            self._charge("binding-row-read")
            row = tuple(supplied)
            if len(row) != len(supplied_slots) or any(
                type(value) is not int for value in row
            ):
                raise ValueError(
                    "binding rows must align with slots and contain exact integers"
                )
            rows.add(tuple(row[index] for index in canonical_indices))
        ordered_rows = tuple(sorted(rows))
        key = self._key(
            "from-bindings",
            {
                "slots": tuple(slot.to_record() for slot in canonical),
                "rows": ordered_rows,
            },
        )
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        if not canonical:
            root = _TERMINAL_ID if ordered_rows else None
            result = self._finish(canonical, root, _Arena())
            return self._store(key, result)  # type: ignore[return-value]
        arena = _Arena()

        def build(level: int, selected: tuple[tuple[int, ...], ...]) -> str | None:
            if level == len(canonical):
                return _TERMINAL_ID
            grouped: dict[int, list[tuple[int, ...]]] = {}
            for row in selected:
                grouped.setdefault(row[level], []).append(row)
            return self._make(
                arena,
                canonical[level],
                (
                    (value, build(level + 1, tuple(group)))
                    for value, group in sorted(grouped.items())
                ),
            )

        root = build(0, ordered_rows) if ordered_rows else None
        result = self._finish(canonical, root, arena)
        return self._store(key, result)  # type: ignore[return-value]

    def domain(
        self,
        slot: TypedSlot,
        values: Iterable[int],
    ) -> PackedBindingRelation:
        self._begin("domain")
        if not isinstance(slot, TypedSlot):
            raise TypeError("domain slot must be TypedSlot")
        supplied_values: list[int] = []
        for value in values:
            self._charge("domain-value-read")
            if type(value) is not int:
                raise ValueError("domain values must be exact integers")
            supplied_values.append(value)
        supplied = tuple(supplied_values)
        canonical_values = tuple(sorted(set(supplied)))
        key = self._key(
            "domain",
            {"slot": slot.to_record(), "values": canonical_values},
        )
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        arena = _Arena()
        root = (
            self._make(
                arena,
                slot,
                ((value, _TERMINAL_ID) for value in canonical_values),
            )
            if canonical_values
            else None
        )
        result = self._finish((slot,), root, arena)
        return self._store(key, result)  # type: ignore[return-value]

    def _set_operation(
        self,
        left: PackedBindingRelation,
        left_validated: _ValidatedRelation,
        right: PackedBindingRelation,
        right_validated: _ValidatedRelation,
        operation: Literal["union", "difference"],
    ) -> PackedBindingRelation:
        if left.slots != right.slots:
            raise ValueError("relation schemas do not match")
        arena = self._merge_arenas(
            left_validated.arena,
            right_validated.arena,
        )
        memo: dict[tuple[str | None, str | None, int], str | None] = {}

        def apply(
            left_id: str | None,
            right_id: str | None,
            level: int,
        ) -> str | None:
            memo_key = (left_id, right_id, level)
            if memo_key in memo:
                return memo[memo_key]
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
            if level == len(left.slots):
                result = _TERMINAL_ID if operation == "union" else None
                memo[memo_key] = result
                return result
            left_edges = dict(self._read_edges(arena, left_id))
            right_edges = dict(self._read_edges(arena, right_id))
            values = (
                sorted(set(left_edges) | set(right_edges))
                if operation == "union"
                else sorted(left_edges)
            )
            result = self._make(
                arena,
                left.slots[level],
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
            memo[memo_key] = result
            return result

        root = apply(left.root_id, right.root_id, 0)
        return self._finish(left.slots, root, arena)

    def union(
        self,
        left: PackedBindingRelation,
        right: PackedBindingRelation,
    ) -> PackedBindingRelation:
        self._begin("union")
        left, left_validated = self._require_relation(left)
        right, right_validated = self._require_relation(right)
        ordered = tuple(sorted((left.sha256, right.sha256)))
        key = self._key("union", ordered)
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        result = self._set_operation(
            left,
            left_validated,
            right,
            right_validated,
            "union",
        )
        return self._store(key, result)  # type: ignore[return-value]

    def difference(
        self,
        left: PackedBindingRelation,
        right: PackedBindingRelation,
    ) -> PackedBindingRelation:
        self._begin("difference")
        left, left_validated = self._require_relation(left)
        right, right_validated = self._require_relation(right)
        key = self._key("difference", (left.sha256, right.sha256))
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        result = self._set_operation(
            left,
            left_validated,
            right,
            right_validated,
            "difference",
        )
        return self._store(key, result)  # type: ignore[return-value]

    def _join_impl(
        self,
        left: PackedBindingRelation,
        left_validated: _ValidatedRelation,
        right: PackedBindingRelation,
        right_validated: _ValidatedRelation,
    ) -> PackedBindingRelation:
        output_slots = _canonical_slots(tuple(set(left.slots).union(right.slots)))
        if left.root_id is None or right.root_id is None:
            return self._finish(output_slots, None, _Arena())
        arena = self._merge_arenas(
            left_validated.arena,
            right_validated.arena,
        )
        left_positions = {slot: index for index, slot in enumerate(left.slots)}
        right_positions = {slot: index for index, slot in enumerate(right.slots)}
        memo: dict[tuple[int, int, str, int, str], str | None] = {}

        def apply(
            output_level: int,
            left_level: int,
            left_id: str,
            right_level: int,
            right_id: str,
        ) -> str | None:
            memo_key = (
                output_level,
                left_level,
                left_id,
                right_level,
                right_id,
            )
            if memo_key in memo:
                return memo[memo_key]
            if output_level == len(output_slots):
                result = (
                    _TERMINAL_ID
                    if left_id == _TERMINAL_ID and right_id == _TERMINAL_ID
                    else None
                )
                memo[memo_key] = result
                return result
            slot = output_slots[output_level]
            left_consumes = slot in left_positions
            right_consumes = slot in right_positions
            left_edges = (
                dict(self._read_edges(arena, left_id)) if left_consumes else None
            )
            right_edges = (
                dict(self._read_edges(arena, right_id)) if right_consumes else None
            )
            if left_edges is not None and right_edges is not None:
                values = sorted(set(left_edges) & set(right_edges))
            elif left_edges is not None:
                values = sorted(left_edges)
            elif right_edges is not None:
                values = sorted(right_edges)
            else:
                raise RuntimeError("join output slot belongs to neither input")
            result = self._make(
                arena,
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
            memo[memo_key] = result
            return result

        root = apply(0, 0, left.root_id, 0, right.root_id)
        return self._finish(output_slots, root, arena)

    def join(
        self,
        left: PackedBindingRelation,
        right: PackedBindingRelation,
    ) -> PackedBindingRelation:
        self._begin("join")
        left, left_validated = self._require_relation(left)
        right, right_validated = self._require_relation(right)
        key = self._key("join", tuple(sorted((left.sha256, right.sha256))))
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        result = self._join_impl(
            left,
            left_validated,
            right,
            right_validated,
        )
        return self._store(key, result)  # type: ignore[return-value]

    def cartesian_product(
        self,
        left: PackedBindingRelation,
        right: PackedBindingRelation,
    ) -> PackedBindingRelation:
        self._begin("cartesian-product")
        left, left_validated = self._require_relation(left)
        right, right_validated = self._require_relation(right)
        if set(left.slots) & set(right.slots):
            raise ValueError("Cartesian product requires disjoint slot schemas")
        key = self._key(
            "cartesian-product",
            tuple(sorted((left.sha256, right.sha256))),
        )
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        result = self._join_impl(
            left,
            left_validated,
            right,
            right_validated,
        )
        return self._store(key, result)  # type: ignore[return-value]

    @staticmethod
    def _constraint_record(constraint: SlotConstraint) -> tuple[object, ...]:
        return (
            constraint.left.to_record(),
            constraint.operator,
            None if constraint.right_slot is None else constraint.right_slot.to_record(),
            constraint.right_value,
        )

    def select(
        self,
        relation: PackedBindingRelation,
        constraints: Sequence[SlotConstraint],
    ) -> PackedBindingRelation:
        self._begin("select")
        relation, validated = self._require_relation(relation)
        if isinstance(constraints, (str, bytes)):
            raise TypeError("constraints must be a sequence")
        checks = tuple(constraints)
        if not all(isinstance(item, SlotConstraint) for item in checks):
            raise TypeError("constraints must contain SlotConstraint values")
        schema = set(relation.slots)
        if any(not item.required_slots <= schema for item in checks):
            raise ValueError("constraint references a slot outside the relation")
        key = self._key(
            "select",
            {
                "relation": relation.sha256,
                "constraints": tuple(self._constraint_record(item) for item in checks),
            },
        )
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        if relation.root_id is None or not checks:
            return self._store(key, relation)  # type: ignore[return-value]
        source = validated.arena
        output = _Arena()

        def visit(
            node_id: str,
            level: int,
            binding: dict[TypedSlot, int],
        ) -> str | None:
            if node_id == _TERMINAL_ID:
                return _TERMINAL_ID
            node, source_edges = self._read_node_edges(source, node_id)
            edges: list[tuple[int, str | None]] = []
            for value, child in source_edges:
                binding[node.slot] = value
                outcomes = tuple(check.evaluate(binding) for check in checks)
                if False not in outcomes:
                    edges.append((value, visit(child, level + 1, binding)))
                binding.pop(node.slot)
            return self._make(output, relation.slots[level], edges)

        root = visit(relation.root_id, 0, {})
        result = self._finish(relation.slots, root, output)
        return self._store(key, result)  # type: ignore[return-value]

    def project_distinct(
        self,
        relation: PackedBindingRelation,
        retained_slots: Sequence[TypedSlot],
    ) -> PackedBindingRelation:
        self._begin("project-distinct")
        relation, validated = self._require_relation(relation)
        retained = _canonical_slots(retained_slots)
        if not set(retained) <= set(relation.slots):
            raise ValueError("projection retains a slot outside the relation")
        key = self._key(
            "project-distinct",
            {
                "relation": relation.sha256,
                "retained": tuple(slot.to_record() for slot in retained),
            },
        )
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        if retained == relation.slots:
            return self._store(key, relation)  # type: ignore[return-value]
        source = validated.arena
        output = _Arena()
        retained_set = set(retained)
        memo: dict[tuple[str, int], str | None] = {}

        def union_ids(
            left_id: str | None,
            right_id: str | None,
            level: int,
        ) -> str | None:
            if left_id is None:
                return right_id
            if right_id is None or left_id == right_id:
                return left_id
            if level == len(retained):
                return _TERMINAL_ID
            left_edges = dict(self._read_edges(output, left_id))
            right_edges = dict(self._read_edges(output, right_id))
            return self._make(
                output,
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
            memo_key = (node_id, level)
            if memo_key in memo:
                return memo[memo_key]
            if node_id == _TERMINAL_ID:
                result: str | None = _TERMINAL_ID
            else:
                node, edges = self._read_node_edges(source, node_id)
                if node.slot in retained_set:
                    result = self._make(
                        output,
                        node.slot,
                        ((value, visit(child, level + 1)) for value, child in edges),
                    )
                else:
                    result = None
                    output_level = sum(
                        1
                        for slot in relation.slots[:level]
                        if slot in retained_set
                    )
                    for _value, child in edges:
                        result = union_ids(
                            result,
                            visit(child, level + 1),
                            output_level,
                        )
            memo[memo_key] = result
            return result

        root = None if relation.root_id is None else visit(relation.root_id, 0)
        result = self._finish(retained, root, output)
        return self._store(key, result)  # type: ignore[return-value]

    def rename(
        self,
        relation: PackedBindingRelation,
        target_slots: Sequence[TypedSlot],
    ) -> PackedBindingRelation:
        self._begin("rename")
        relation, validated = self._require_relation(relation)
        target = tuple(target_slots)
        if len(target) != len(relation.slots):
            raise ValueError("slot rename must preserve arity")
        if target != _canonical_slots(target):
            raise ValueError("renamed slots must preserve canonical order")
        key = self._key(
            "rename",
            {
                "relation": relation.sha256,
                "target": tuple(slot.to_record() for slot in target),
            },
        )
        cached = self._cached(key, PackedBindingRelation)
        if cached is not None:
            return cached  # type: ignore[return-value]
        source = validated.arena
        output = _Arena()
        memo: dict[tuple[str, int], str] = {}

        def visit(node_id: str, level: int) -> str:
            if node_id == _TERMINAL_ID:
                return _TERMINAL_ID
            memo_key = (node_id, level)
            if memo_key in memo:
                return memo[memo_key]
            edges = self._read_edges(source, node_id)
            renamed = self._make(
                output,
                target[level],
                ((value, visit(child, level + 1)) for value, child in edges),
            )
            if renamed is None:
                raise RuntimeError("nonempty rename lost all decision edges")
            memo[memo_key] = renamed
            return renamed

        root = None if relation.root_id is None else visit(relation.root_id, 0)
        result = self._finish(target, root, output)
        return self._store(key, result)  # type: ignore[return-value]

    def _authenticate_slice(
        self,
        family_slice: PackedFamilySlice,
    ) -> _ValidatedRelation:
        try:
            if type(family_slice) is not PackedFamilySlice:
                raise TypeError("family value must be PackedFamilySlice")
            if type(family_slice.family_id) is not str or not family_slice.family_id:
                raise ValueError("family_id must be a nonempty string")
            relation, validated = self._require_relation(family_slice.relation)
            expected = _namespaced_sha256(
                _SLICE_NAMESPACE,
                {
                    "family_id": family_slice.family_id,
                    "relation_sha256": relation.sha256,
                },
            )
            if family_slice.sha256 != expected:
                raise ValueError("family slice SHA-256 is invalid")
            return validated
        except PackedRelationWorkLimitExceeded:
            raise
        except Exception as error:
            raise RuntimeError("packed family slice authentication failed") from error

    def family_slice(
        self,
        family_id: str,
        relation: PackedBindingRelation,
    ) -> PackedFamilySlice:
        self._begin("family-slice")
        if type(family_id) is not str or not family_id:
            raise ValueError("family_id must be a nonempty string")
        relation, _validated = self._require_relation(relation)
        key = self._key(
            "family-slice",
            {"family_id": family_id, "relation_sha256": relation.sha256},
        )
        cached = self._cached(key, PackedFamilySlice)
        if cached is not None:
            return cached  # type: ignore[return-value]
        digest = _namespaced_sha256(
            _SLICE_NAMESPACE,
            {"family_id": family_id, "relation_sha256": relation.sha256},
        )
        return self._store(
            key,
            PackedFamilySlice(family_id, relation, digest),
        )  # type: ignore[return-value]

    def authenticate_slice(self, family_slice: PackedFamilySlice) -> bool:
        self._begin("authenticate-slice")
        self._authenticate_slice(family_slice)
        return True

    def _forest_observation(
        self,
        forest: PackedProofForest,
    ) -> tuple[
        int,
        int,
        tuple[ProofWitness, ...],
        RelationAllocation,
        bytes,
    ]:
        if type(forest) is not PackedProofForest:
            raise TypeError("forest value must be PackedProofForest")
        if type(forest.saturation_limit) is not int or forest.saturation_limit < 1:
            raise ValueError("forest saturation limit must be positive")
        if (
            type(forest.exact_logical_recipe_count) is not int
            or forest.exact_logical_recipe_count < 0
            or type(forest.saturating_logical_recipe_count) is not int
            or forest.saturating_logical_recipe_count < 0
        ):
            raise ValueError("forest logical counts are invalid")
        if type(forest.slices) is not tuple or not all(
            type(item) is PackedFamilySlice for item in forest.slices
        ):
            raise TypeError("forest slices must contain PackedFamilySlice values")
        if tuple(sorted(forest.slices, key=lambda item: item.family_id)) != forest.slices:
            raise ValueError("forest slices must be sorted by family")
        if len({item.family_id for item in forest.slices}) != len(forest.slices):
            raise ValueError("forest may contain only one slice per family")
        if type(forest.first_two_witnesses) is not tuple or not all(
            type(item) is ProofWitness
            and type(item.family_id) is str
            and bool(item.family_id)
            and type(item.binding) is tuple
            and all(type(value) is int for value in item.binding)
            for item in forest.first_two_witnesses
        ):
            raise TypeError("forest witnesses are invalid")

        exact = 0
        node_total = 0
        edge_total = 0
        witnesses: list[ProofWitness] = []
        for item in forest.slices:
            validated = self._authenticate_slice(item)
            exact += self._exact_count(item.relation, validated)
            node_total += item.relation.allocation.decision_nodes
            edge_total += item.relation.allocation.decision_edges
            witnesses.extend(
                ProofWitness(item.family_id, binding)
                for binding in self._first_bindings(item.relation, validated, 2)
            )
        first_two = tuple(
            sorted(witnesses, key=lambda item: (item.family_id, item.binding))[:2]
        )
        saturated = min(forest.saturation_limit, exact)
        if forest.exact_logical_recipe_count != exact:
            raise ValueError("forest exact logical count is invalid")
        if forest.saturating_logical_recipe_count != saturated:
            raise ValueError("forest saturating logical count is invalid")
        if forest.first_two_witnesses != first_two:
            raise ValueError("forest first witnesses are invalid")
        payload = _forest_payload(
            forest.slices,
            forest.saturation_limit,
            exact,
            saturated,
            first_two,
        )
        canonical = _canonical_json(payload)
        expected = self._digest_canonical(_FOREST_NAMESPACE, canonical)
        if forest.sha256 != expected:
            raise ValueError("proof forest SHA-256 is invalid")
        return (
            exact,
            saturated,
            first_two,
            RelationAllocation(node_total, edge_total),
            canonical,
        )

    def _authenticate_forest(
        self,
        forest: PackedProofForest,
    ) -> tuple[
        int,
        int,
        tuple[ProofWitness, ...],
        RelationAllocation,
        bytes,
    ]:
        try:
            return self._forest_observation(forest)
        except PackedRelationWorkLimitExceeded:
            raise
        except Exception as error:
            raise RuntimeError("packed proof forest authentication failed") from error

    def proof_forest(
        self,
        slices: Iterable[PackedFamilySlice] = (),
        *,
        saturation_limit: int = DEFAULT_SATURATION_LIMIT,
    ) -> PackedProofForest:
        self._begin("proof-forest-build")
        if type(saturation_limit) is not int or saturation_limit < 1:
            raise ValueError("forest saturation limit must be positive")
        supplied = tuple(slices)
        for item in supplied:
            self._authenticate_slice(item)
        cache_key = self._key(
            "proof-forest-build",
            {
                "slices": tuple(
                    sorted((item.family_id, item.sha256) for item in supplied)
                ),
                "saturation_limit": saturation_limit,
            },
        )
        cached = self._cached(cache_key, PackedProofForest)
        if cached is not None:
            return cached  # type: ignore[return-value]

        grouped: dict[str, PackedBindingRelation] = {}
        for item in supplied:
            incumbent = grouped.get(item.family_id)
            if incumbent is None:
                grouped[item.family_id] = item.relation
            else:
                self._proof_union()
                grouped[item.family_id] = self.union(incumbent, item.relation)
        canonical = tuple(
            self.family_slice(family_id, relation)
            for family_id, relation in sorted(grouped.items())
        )

        exact = 0
        witnesses: list[ProofWitness] = []
        for item in canonical:
            validated = self._authenticate_slice(item)
            exact += self._exact_count(item.relation, validated)
            witnesses.extend(
                ProofWitness(item.family_id, binding)
                for binding in self._first_bindings(item.relation, validated, 2)
            )
        saturated = min(saturation_limit, exact)
        first_two = tuple(
            sorted(witnesses, key=lambda item: (item.family_id, item.binding))[:2]
        )
        payload = _forest_payload(
            canonical,
            saturation_limit,
            exact,
            saturated,
            first_two,
        )
        digest = self._digest_canonical(
            _FOREST_NAMESPACE,
            _canonical_json(payload),
        )
        result = PackedProofForest(
            slices=canonical,
            saturation_limit=saturation_limit,
            exact_logical_recipe_count=exact,
            saturating_logical_recipe_count=saturated,
            first_two_witnesses=first_two,
            sha256=digest,
        )
        return self._store(cache_key, result)  # type: ignore[return-value]

    def union_forests(
        self,
        left: PackedProofForest,
        right: PackedProofForest,
    ) -> PackedProofForest:
        self._begin("proof-forest-union")
        self._authenticate_forest(left)
        self._authenticate_forest(right)
        if left.saturation_limit != right.saturation_limit:
            raise ValueError("forest saturation limits do not match")
        key = self._key(
            "proof-forest-union",
            tuple(sorted((left.sha256, right.sha256))),
        )
        cached = self._cached(key, PackedProofForest)
        if cached is not None:
            return cached  # type: ignore[return-value]
        self._proof_union()
        result = self.proof_forest(
            (*left.slices, *right.slices),
            saturation_limit=left.saturation_limit,
        )
        return self._store(key, result)  # type: ignore[return-value]

    def authenticate_forest(self, forest: PackedProofForest) -> bool:
        self._begin("authenticate-forest")
        self._authenticate_forest(forest)
        return True

    def forest_exact_count(self, forest: PackedProofForest) -> int:
        self._begin("forest-exact-count")
        exact, _saturated, _witnesses, _allocation, _canonical = (
            self._authenticate_forest(forest)
        )
        return exact

    def forest_saturating_count(self, forest: PackedProofForest) -> int:
        self._begin("forest-saturating-count")
        _exact, saturated, _witnesses, _allocation, _canonical = (
            self._authenticate_forest(forest)
        )
        return saturated

    def forest_first_witnesses(
        self,
        forest: PackedProofForest,
    ) -> tuple[ProofWitness, ...]:
        self._begin("forest-first-witnesses")
        _exact, _saturated, witnesses, _allocation, _canonical = (
            self._authenticate_forest(forest)
        )
        return witnesses

    def forest_allocation(self, forest: PackedProofForest) -> RelationAllocation:
        self._begin("forest-allocation")
        _exact, _saturated, _witnesses, allocation, _canonical = (
            self._authenticate_forest(forest)
        )
        return allocation

    def forest_canonical_bytes(self, forest: PackedProofForest) -> bytes:
        self._begin("forest-canonical-bytes")
        _exact, _saturated, _witnesses, _allocation, canonical = (
            self._authenticate_forest(forest)
        )
        return canonical

    def normalize(
        self,
        family_slice: PackedFamilySlice,
        projection: ProofNormalizationProjection,
    ) -> PackedFamilySlice:
        self._begin("normalize")
        if not isinstance(family_slice, PackedFamilySlice):
            raise TypeError("normalization requires PackedFamilySlice")
        if not isinstance(projection, ProofNormalizationProjection):
            raise TypeError("normalization requires ProofNormalizationProjection")
        self._authenticate_slice(family_slice)
        if family_slice.family_id != projection.source_family_id:
            raise ValueError("normalization source family does not match")
        schema = set(family_slice.relation.slots)
        required_slots = {slot for slot, _value in projection.required_constants}
        if not set(projection.retained_source_slots) | required_slots <= schema:
            raise ValueError("normalization references slots outside the relation")
        key = self._key(
            "normalize",
            {
                "slice": family_slice.sha256,
                "projection": projection.to_record(),
            },
        )
        cached = self._cached(key, PackedFamilySlice)
        if cached is not None:
            return cached  # type: ignore[return-value]
        constraints = tuple(
            SlotConstraint.value(slot, "eq", value)
            for slot, value in projection.required_constants
        )
        selected = self.select(family_slice.relation, constraints)
        projected = self.project_distinct(
            selected, projection.retained_source_slots
        )
        renamed = self.rename(projected, projection.target_slots)
        result = self.family_slice(projection.target_family_id, renamed)
        return self._store(key, result)  # type: ignore[return-value]


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
        constants = tuple(
            sorted(self.required_constants, key=lambda item: item[0].sort_key)
        )
        if constants != self.required_constants or len(
            {slot for slot, _ in constants}
        ) != len(constants):
            raise ValueError("normalization constants must be uniquely sorted")
        if any(
            not isinstance(slot, TypedSlot) or type(value) is not int
            for slot, value in constants
        ):
            raise TypeError("normalization constants must be typed slots and integers")

    def to_record(self) -> dict[str, object]:
        return {
            "normalization_id": self.normalization_id,
            "source_family_id": self.source_family_id,
            "target_family_id": self.target_family_id,
            "retained_source_slots": tuple(
                slot.to_record() for slot in self.retained_source_slots
            ),
            "target_slots": tuple(slot.to_record() for slot in self.target_slots),
            "required_constants": tuple(
                (slot.to_record(), value)
                for slot, value in self.required_constants
            ),
        }


@dataclass(frozen=True, slots=True)
class PackedFamilySlice:
    """A canonical proof family paired with its distinct finite bindings."""

    family_id: str
    relation: PackedBindingRelation
    sha256: str

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise ValueError("family_id must be a nonempty string")
        if type(self.relation) is not PackedBindingRelation:
            raise TypeError("family relation must be PackedBindingRelation")
        if type(self.sha256) is not str or len(self.sha256) != 64:
            raise ValueError("family slice SHA-256 must be a digest")


@dataclass(frozen=True, slots=True)
class ProofWitness:
    family_id: str
    binding: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise ValueError("witness family_id must be a nonempty string")
        if type(self.binding) is not tuple or any(
            type(value) is not int for value in self.binding
        ):
            raise TypeError("witness binding must contain exact integers")


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
        if type(self.slices) is not tuple or not all(
            type(item) is PackedFamilySlice for item in self.slices
        ):
            raise TypeError("forest slices must contain PackedFamilySlice values")
        if type(self.exact_logical_recipe_count) is not int or (
            self.exact_logical_recipe_count < 0
        ):
            raise ValueError("forest exact logical count must be nonnegative")
        if type(self.saturating_logical_recipe_count) is not int or (
            self.saturating_logical_recipe_count < 0
        ):
            raise ValueError("forest saturating count must be nonnegative")
        if type(self.first_two_witnesses) is not tuple or not all(
            type(item) is ProofWitness for item in self.first_two_witnesses
        ):
            raise TypeError("forest witnesses must contain ProofWitness values")
        if type(self.sha256) is not str or len(self.sha256) != 64:
            raise ValueError("proof forest SHA-256 must be a digest")


__all__ = [
    "CONSTRAINT_OPERATORS",
    "DEFAULT_SATURATION_LIMIT",
    "SLOT_KINDS",
    "DecisionNode",
    "PackedBindingRelation",
    "PackedFamilySlice",
    "PackedProofForest",
    "PackedRelationSession",
    "PackedRelationWorkAudit",
    "PackedRelationWorkLimitExceeded",
    "ProofNormalizationProjection",
    "ProofWitness",
    "RelationAllocation",
    "SlotConstraint",
    "TypedSlot",
]
