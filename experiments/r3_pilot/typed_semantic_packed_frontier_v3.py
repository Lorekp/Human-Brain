"""Construction-certified packed scalar frontier.

V2 made semantic partitions session-native and authenticated their reachable
decision topology once.  Its publication path nevertheless walked metadata a
second time, and selection linearly rediscovered the winning e-class before
re-authenticating the selected relation many times.  V3 keeps the same deep
two-operation seam -- ``build`` then ``select`` -- while making publication a
constructional certificate:

* one metered topology traversal captures the retained arena;
* strict immutable publication records make a constant-work retained seal
  sufficient;
* the component capsule owns direct family/e-class indexes; and
* the selected root is counted and witnessed in one metered DAG pass.

The grammar, physical predictions, physical races, winner semantics, explicit
``x + 0 -> x`` proof normalization, and desired-free build are unchanged.  This
is still a self-posed scalar mechanism, not broad typed or capability evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as v2

FRONTIER_READY = "PACKED_SCALAR_FRONTIER_V3_READY"
STOP_WORK_BOUND = "STOP_PACKED_SCALAR_FRONTIER_V3_WORK_BOUND"
SELECTED = v2.SELECTED
STOP_NO_EXACT = v2.STOP_NO_EXACT
STOP_AMBIGUOUS = v2.STOP_AMBIGUOUS

MAX_RECIPE_RELATION_WORK = 30_000
MAX_PHYSICAL_PREDICTIONS = v2.MAX_PHYSICAL_PREDICTIONS
MAX_PHYSICAL_RACES = v2.MAX_PHYSICAL_RACES

_NAMESPACE = b"typed-semantic-packed-frontier-v3\0"
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


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
    # Preserve V2's semantic identity so physical race keys and winners do not
    # change merely because publication became cheaper.
    return v2._eclass_id(values)


class PackedSemanticGroupRef(NamedTuple):
    partition_id: str
    family_id: str
    group_id: str
    schema: tuple[packed.TypedSlot, ...]
    root_id: str | None
    child_semantic_refs: tuple[tuple[str, ...], ...]
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    physical_output_sha256: str
    sha256: str


class PackedNormalizationRecord(NamedTuple):
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


class PackedPublishedPartition(NamedTuple):
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


class PackedFrozenWorkset(NamedTuple):
    arena_generation: int
    topology_node_count: int
    topology_sha256: str
    partitions: tuple[PackedPublishedPartition, ...]
    registry_digest: str
    metadata_sha256: str
    publication_certificate_sha256: str
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
    family_index_records: tuple[tuple[str, str], ...]
    eclass_index_records: tuple[tuple[str, str], ...]
    family_index: object
    eclass_index: object
    sha256: str


class _StrictRetainedArena(NamedTuple):
    """Immutable, unaliased decision topology retained after publication."""

    nodes: Mapping[str, packed.DecisionNode]

    def node(self, node_id: str) -> packed.DecisionNode:
        if node_id == packed._TERMINAL_ID:
            raise ValueError("terminal has no decision-node record")
        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise ValueError("relation references an unknown decision node") from error


class _AuthenticatedFrozen(NamedTuple):
    owner_token: object
    frozen_sha256: str
    certificate_sha256: str
    retained_seal_sha256: str
    arena: _StrictRetainedArena
    group_index: object
    root_index: object


class PackedWitness(NamedTuple):
    family_id: str
    binding: tuple[int, ...]
    source: str
    sha256: str


class PackedFrontierWorkAudit(NamedTuple):
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


class PackedFrontierState(NamedTuple):
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


class PackedFrontierSelection(NamedTuple):
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
        "groups": tuple(group.sha256 for group in partition.groups),
    }


def _frozen_payload(frozen: PackedFrozenWorkset) -> dict[str, object]:
    return {
        "arena_generation": frozen.arena_generation,
        "topology_node_count": frozen.topology_node_count,
        "topology_sha256": frozen.topology_sha256,
        "partitions": tuple(item.sha256 for item in frozen.partitions),
        "registry_digest": frozen.registry_digest,
        "metadata_sha256": frozen.metadata_sha256,
        "publication_certificate_sha256": frozen.publication_certificate_sha256,
    }


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


def _component_capsule_payload(capsule: _AuthenticatedComponents) -> dict[str, object]:
    return {
        "frozen_sha256": capsule.frozen_sha256,
        "family_sha256s": capsule.family_sha256s,
        "enode_sha256s": capsule.enode_sha256s,
        "eclass_sha256s": capsule.eclass_sha256s,
        "group_to_enode": capsule.group_to_enode,
        "enode_to_eclass": capsule.enode_to_eclass,
        "root_eclass_ids": capsule.root_eclass_ids,
        "family_index_records": capsule.family_index_records,
        "eclass_index_records": capsule.eclass_index_records,
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


class _SemanticPartitionWorkset(v2._SemanticPartitionWorkset):
    """Session-owned arena with construction-certified publication metadata."""

    __slots__ = ("_retained_arena",)

    def __init__(self, session: v2._FrontierSession) -> None:
        super().__init__(session)
        self._retained_arena: _StrictRetainedArena | None = None

    @staticmethod
    def _freeze_normalization(
        value: v2.PackedNormalizationRecord | None,
    ) -> PackedNormalizationRecord | None:
        if value is None:
            return None
        if value.sha256 != v2._digest(
            "normalization-record",
            v2._normalization_payload(value),
        ):
            raise RuntimeError("normalization changed before publication")
        partial = PackedNormalizationRecord(
            normalization_id=value.normalization_id,
            source_family_id=value.source_family_id,
            target_family_id=value.target_family_id,
            retained_source_slots=value.retained_source_slots,
            target_slots=value.target_slots,
            required_constants=value.required_constants,
            source_root_id=value.source_root_id,
            recovered_root_id=value.recovered_root_id,
            target_root_id=value.target_root_id,
            sha256="",
        )
        return partial._replace(
            sha256=_digest("normalization-record", _normalization_payload(partial))
        )

    def publish(self) -> PackedFrozenWorkset:
        """Construct metadata and its certificate in the same metered traversal."""

        self._operation("publish")
        self._ensure_open()
        partitions: list[PackedPublishedPartition] = []
        group_index: dict[str, PackedSemanticGroupRef] = {}
        semantic_pairs: set[tuple[str, str]] = set()
        root_specs: list[tuple[str | None, tuple[packed.TypedSlot, ...]]] = []
        expected_order = tuple(
            sorted(
                self._records,
                key=lambda item: (
                    self._records[item].plan.inside_cost,
                    item,
                ),
            )
        )

        for family_id in expected_order:
            self.session.charge_workset("publication-partition")
            record = self._require_handle(self._records[family_id].handle)
            public_partition_id = _digest(
                "partition-id",
                {
                    "family_id": family_id,
                    "generation": record.handle.generation,
                    "relation_identity": record.handle.relation_identity,
                },
            )
            group_refs: list[PackedSemanticGroupRef] = []
            for group in record.groups:
                # This is the sole metadata-group traversal.  Validation and
                # direct-index insertion happen inside the same charged unit.
                self.session.charge_workset("publication-group")
                for output in group.physical_outputs:
                    output_cells.TypedOutputCellBank.validate(output)
                child_refs = self._canonical_child_refs(group.child_refs)
                child_ids = record.plan.child_family_ids
                if not child_ids and child_refs:
                    raise RuntimeError("published terminal group gained child semantics")
                if child_ids and not child_refs:
                    raise RuntimeError("published operator group lost child semantics")
                for refs in child_refs:
                    if len(refs) != len(child_ids):
                        raise RuntimeError("published child semantic arity changed")
                    for reference, child_id in zip(refs, child_ids, strict=True):
                        prefix = f"{child_id}:"
                        if (
                            not reference.startswith(prefix)
                            or (child_id, reference.removeprefix(prefix)) not in semantic_pairs
                        ):
                            raise RuntimeError("published child semantic ref changed")
                output_sha = _vector_sha256(group.physical_outputs)
                group_id = _digest(
                    "semantic-group-id",
                    {
                        "partition_id": public_partition_id,
                        "family_id": family_id,
                        "root_id": group.root_id,
                        "physical_output_sha256": output_sha,
                    },
                )
                partial_group = PackedSemanticGroupRef(
                    partition_id=public_partition_id,
                    family_id=family_id,
                    group_id=group_id,
                    schema=record.plan.schema,
                    root_id=group.root_id,
                    child_semantic_refs=child_refs,
                    physical_outputs=group.physical_outputs,
                    physical_output_sha256=output_sha,
                    sha256="",
                )
                published_group = partial_group._replace(
                    sha256=_digest("published-semantic-group", _group_payload(partial_group))
                )
                self.session.charge_workset("published-group-index-insert")
                incumbent = group_index.setdefault(group_id, published_group)
                if incumbent is not published_group:
                    raise RuntimeError("published semantic group identity collision")
                group_refs.append(published_group)
                semantic_pairs.add((family_id, _eclass_id(group.physical_outputs)))
                root_specs.append((group.root_id, record.plan.schema))

            normalization = self._freeze_normalization(record.normalization)
            if normalization is not None:
                if (
                    normalization.normalization_id != record.plan.normalization_id
                    or normalization.source_family_id != family_id
                    or normalization.target_family_id != record.plan.child_family_ids[0]
                    or normalization.recovered_root_id != normalization.target_root_id
                ):
                    raise RuntimeError("published normalization authentication failed")
                root_specs.extend(
                    (
                        (normalization.source_root_id, record.plan.schema),
                        (normalization.recovered_root_id, normalization.target_slots),
                        (normalization.target_root_id, normalization.target_slots),
                    )
                )
            root_specs.append((record.root_id, record.plan.schema))
            partial_partition = PackedPublishedPartition(
                partition_id=public_partition_id,
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
                    v2._constraint_record(constraint) for constraint in record.plan.constraints
                ),
                normalization_id=record.plan.normalization_id,
                normalization=normalization,
                groups=tuple(group_refs),
                sha256="",
            )
            partitions.append(
                partial_partition._replace(
                    sha256=_digest(
                        "published-partition",
                        _partition_payload(partial_partition),
                    )
                )
            )

        registry_digest = _digest(
            "published-registry",
            tuple((partition.partition_id, partition.registry_digest) for partition in partitions),
        )
        metadata_sha256 = _digest(
            "published-metadata",
            {
                "partitions": tuple(item.sha256 for item in partitions),
                "registry_digest": registry_digest,
            },
        )
        self.session.charge_workset("publication-metadata-seal")
        # Metadata has already been validated while it was constructed.  Only
        # topology remains, and it is still fully traversed and charged once.
        self._operation("authenticate-publication-topology")
        mutable_arena, root_index, topology_node_count, topology_sha256 = (
            self._authenticate_topology(root_specs)
        )
        arena = _StrictRetainedArena(
            nodes=MappingProxyType(dict(mutable_arena.nodes)),
        )
        certificate_sha256 = _digest(
            "publication-certificate",
            {
                "metadata_sha256": metadata_sha256,
                "topology_node_count": topology_node_count,
                "topology_sha256": topology_sha256,
                "arena_generation": self._generation,
            },
        )
        partial = PackedFrozenWorkset(
            arena_generation=self._generation,
            topology_node_count=topology_node_count,
            topology_sha256=topology_sha256,
            partitions=tuple(partitions),
            registry_digest=registry_digest,
            metadata_sha256=metadata_sha256,
            publication_certificate_sha256=certificate_sha256,
            sha256="",
        )
        frozen = partial._replace(sha256=_digest("frozen-workset", _frozen_payload(partial)))
        retained_seal = _digest(
            "retained-publication-seal",
            {
                "frozen_sha256": frozen.sha256,
                "certificate_sha256": certificate_sha256,
            },
        )
        authenticated = _AuthenticatedFrozen(
            owner_token=self._owner_token,
            frozen_sha256=frozen.sha256,
            certificate_sha256=certificate_sha256,
            retained_seal_sha256=retained_seal,
            arena=arena,
            group_index=MappingProxyType(group_index),
            root_index=MappingProxyType(root_index),
        )
        self._retained_arena = arena
        self._frozen = frozen  # type: ignore[assignment]
        self._authenticated = authenticated  # type: ignore[assignment]
        self._sealed = True
        return frozen

    def require_retained(self, frozen: PackedFrozenWorkset) -> _AuthenticatedFrozen:
        """Verify the strict constructional publication seal in constant work."""

        self._operation("authenticate-retained-publication-seal")
        self.session.charge_workset("retained-publication-seal")
        authenticated = self._authenticated
        if type(authenticated) is not _AuthenticatedFrozen:
            raise RuntimeError("retained publication seal is absent")
        expected_seal = _digest(
            "retained-publication-seal",
            {
                "frozen_sha256": frozen.sha256,
                "certificate_sha256": frozen.publication_certificate_sha256,
            },
        )
        if (
            type(frozen) is not PackedFrozenWorkset
            or frozen is not self._frozen
            or authenticated is not self._authenticated
            or authenticated.owner_token is not self._owner_token
            or authenticated.frozen_sha256 != frozen.sha256
            or authenticated.certificate_sha256 != frozen.publication_certificate_sha256
            or authenticated.retained_seal_sha256 != expected_seal
            or frozen.arena_generation != self._generation
            or type(authenticated.arena) is not _StrictRetainedArena
            or authenticated.arena is not self._retained_arena
            or type(authenticated.arena.nodes) is not _MAPPING_PROXY_TYPE
        ):
            raise RuntimeError("retained published workset authentication failed")
        return authenticated

    def _materialize_root_once(
        self,
        schema: tuple[packed.TypedSlot, ...],
        root_id: str | None,
        arena: _StrictRetainedArena,
    ) -> tuple[packed.PackedBindingRelation, int, tuple[tuple[int, ...], ...]]:
        self.session.charge_workset("selected-root-materialization")
        memo: dict[str, tuple[int, tuple[tuple[int, ...], ...]]] = {}
        captured: dict[str, packed.DecisionNode] = {}
        edge_total = 0

        def visit(
            node_id: str | None,
            level: int,
        ) -> tuple[int, tuple[tuple[int, ...], ...]]:
            nonlocal edge_total
            if node_id is None:
                return 0, ()
            if node_id == packed._TERMINAL_ID:
                if level != len(schema):
                    raise RuntimeError("selected root reached terminal at wrong depth")
                return 1, ((),)
            cached = memo.get(node_id)
            if cached is not None:
                return cached
            if level >= len(schema):
                raise RuntimeError("selected root exceeds schema")
            node, edges = self.session._read_node_edges(arena, node_id)
            if type(node) is not packed.DecisionNode:
                raise RuntimeError("selected root node record changed")
            if node.slot != schema[level]:
                raise RuntimeError("selected root schema changed")
            if type(edges) is not tuple or not edges:
                raise RuntimeError("selected root edges changed")
            values: list[int] = []
            for edge in edges:
                if type(edge) is not tuple or len(edge) != 2:
                    raise RuntimeError("selected root edge record changed")
                value, child = edge
                if (
                    type(value) is not int
                    or type(child) is not str
                    or len(child) != 64
                    or (child != packed._TERMINAL_ID and child not in arena.nodes)
                ):
                    raise RuntimeError("selected root edge domain changed")
                values.append(value)
            if tuple(values) != tuple(sorted(set(values))):
                raise RuntimeError("selected root edge ordering changed")
            if node.node_id != node_id or node_id != packed._node_id(node.slot, edges):
                raise RuntimeError("selected root content address changed")
            captured[node_id] = node
            edge_total += len(edges)
            exact = 0
            first_two: list[tuple[int, ...]] = []
            for value, child in edges:
                child_count, suffixes = visit(child, level + 1)
                exact += child_count
                for suffix in suffixes:
                    if len(first_two) < 2:
                        first_two.append((value, *suffix))
            result = (exact, tuple(first_two))
            memo[node_id] = result
            return result

        exact, witnesses = visit(root_id, 0)
        node_ids = tuple(sorted(captured))
        nodes = tuple(captured[node_id] for node_id in node_ids)
        allocation = packed.RelationAllocation(len(nodes), edge_total)
        node_records = tuple(
            {
                "node_id": node.node_id,
                "slot": node.slot.to_record(),
                "edges": node.edges,
            }
            for node in nodes
        )
        payload = {
            "schema": "typed-semantic-packed-binding-relation-v1",
            "slots": tuple(slot.to_record() for slot in schema),
            "root_id": root_id,
            "terminal_id": packed._TERMINAL_ID,
            "nodes": node_records,
            "allocation": {
                "decision_nodes": allocation.decision_nodes,
                "decision_edges": allocation.decision_edges,
            },
        }
        canonical = packed._canonical_json(payload)
        relation = packed.PackedBindingRelation(
            schema,
            root_id,
            nodes,
            allocation,
            self.session._digest_canonical(packed._RELATION_NAMESPACE, canonical),
        )
        return relation, exact, witnesses

    def materialize_selected_forest(
        self,
        frozen: PackedFrozenWorkset,
        authenticated: _AuthenticatedFrozen,
        groups: tuple[PackedSemanticGroupRef, ...],
    ) -> packed.PackedProofForest:
        """Build the selected proof forest with one charged pass per live root."""

        self._operation("materialize-selected-proof")
        if (
            authenticated is not self._authenticated
            or authenticated.owner_token is not self._owner_token
            or authenticated.frozen_sha256 != frozen.sha256
            or frozen is not self._frozen
            or type(authenticated.arena) is not _StrictRetainedArena
            or authenticated.arena is not self._retained_arena
            or type(authenticated.arena.nodes) is not _MAPPING_PROXY_TYPE
            or not isinstance(authenticated.group_index, Mapping)
            or not isinstance(authenticated.root_index, Mapping)
        ):
            raise RuntimeError("stale or foreign published workset authentication")
        slices: list[packed.PackedFamilySlice] = []
        all_witnesses: list[packed.ProofWitness] = []
        exact_total = 0
        seen_families: set[str] = set()
        for group in groups:
            self.session.charge_workset("selected-group-lookup")
            if authenticated.group_index.get(group.group_id) is not group:
                raise RuntimeError("selected semantic group was substituted")
            root_key = (
                group.root_id,
                tuple(slot.to_record() for slot in group.schema),
            )
            if authenticated.root_index.get(root_key) != v2._relation_identity(
                group.schema,
                group.root_id,
            ):
                raise RuntimeError("selected semantic group root was substituted")
            if group.family_id in seen_families:
                raise RuntimeError("selected e-class repeats one proof family")
            seen_families.add(group.family_id)
            relation, exact, bindings = self._materialize_root_once(
                group.schema,
                group.root_id,
                authenticated.arena,
            )
            self.session.charge_workset("selected-proof-slice")
            slice_sha256 = packed._namespaced_sha256(
                packed._SLICE_NAMESPACE,
                {
                    "family_id": group.family_id,
                    "relation_sha256": relation.sha256,
                },
            )
            slices.append(packed.PackedFamilySlice(group.family_id, relation, slice_sha256))
            exact_total += exact
            all_witnesses.extend(
                packed.ProofWitness(group.family_id, binding) for binding in bindings
            )
        canonical_slices = tuple(sorted(slices, key=lambda item: item.family_id))
        first_two = tuple(
            sorted(all_witnesses, key=lambda item: (item.family_id, item.binding))[:2]
        )
        saturated = min(2, exact_total)
        self.session.charge_workset("selected-proof-forest")
        payload = packed._forest_payload(
            canonical_slices,
            2,
            exact_total,
            saturated,
            first_two,
        )
        forest_sha256 = self.session._digest_canonical(
            packed._FOREST_NAMESPACE,
            packed._canonical_json(payload),
        )
        return packed.PackedProofForest(
            slices=canonical_slices,
            saturation_limit=2,
            exact_logical_recipe_count=exact_total,
            saturating_logical_recipe_count=saturated,
            first_two_witnesses=first_two,
            sha256=forest_sha256,
        )


class _Builder(v2._Builder):
    """V2's unchanged grammar with V3 publication and component construction."""

    def __init__(
        self,
        supplied_case_arguments: Iterable[int],
        supplied_literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
    ) -> None:
        super().__init__(
            supplied_case_arguments,
            supplied_literals,
            max_ast_cost,
            operator,
        )
        # The parent constructor has not consumed any work.  Replace its empty
        # session before the first grammar operation so V3's cap remains the
        # sole ledger and monkeypatchable experimental constant.
        self.session = v2._FrontierSession(work_limit=MAX_RECIPE_RELATION_WORK)
        self.workset = _SemanticPartitionWorkset(self.session)
        self.authenticated_components: _AuthenticatedComponents | None = None

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
        family_index: dict[str, PackedFamily] = {}

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
            family = partial_family._replace(
                sha256=_digest("family", _family_payload(partial_family))
            )
            self._charge("family-index-insert")
            incumbent_family = family_index.setdefault(family.family_id, family)
            if incumbent_family is not family:
                raise RuntimeError("packed family identity was consumed twice")
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
                enode = partial_enode._replace(
                    sha256=_digest("e-node", _enode_payload(partial_enode))
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
                incumbent_output = eclass_outputs.setdefault(
                    eclass_id,
                    enode.physical_outputs,
                )
                if incumbent_output != enode.physical_outputs:
                    raise RuntimeError("global semantic e-class identity collision")

        eclasses: list[PackedEClass] = []
        root_eclass_ids: list[str] = []
        eclass_index: dict[str, PackedEClass] = {}
        finalized_enodes = 0
        for eclass_id, selected_ids in eclass_enode_ids.items():
            self._charge("publish-e-class")
            selected_group_refs = eclass_group_refs[eclass_id]
            if len({item.family_id for item in selected_group_refs}) != len(selected_group_refs):
                raise RuntimeError("one e-class repeats a proof family")
            finalized_enodes += len(selected_ids)
            partial = PackedEClass(
                eclass_id=eclass_id,
                minimum_inside_cost=eclass_min_cost[eclass_id],
                enode_ids=tuple(selected_ids),
                group_refs=tuple(selected_group_refs),
                physical_outputs=eclass_outputs[eclass_id],
                sha256="",
            )
            eclass = partial._replace(sha256=_digest("e-class", _eclass_payload(partial)))
            self._charge("eclass-index-insert")
            incumbent_eclass = eclass_index.setdefault(eclass.eclass_id, eclass)
            if incumbent_eclass is not eclass:
                raise RuntimeError("packed e-class identity was consumed twice")
            eclasses.append(eclass)
            eclass_sha256s.append(eclass.sha256)
            root_eclass_ids.append(eclass.eclass_id)

        frozen_families = tuple(families)
        frozen_enodes = tuple(enodes)
        frozen_eclasses = tuple(eclasses)
        canonical_group_to_enode = tuple(group_to_enode)
        canonical_enode_to_eclass = tuple(enode_to_eclass)
        family_index_records = tuple(
            (family.family_id, family.sha256) for family in frozen_families
        )
        eclass_index_records = tuple(
            (eclass.eclass_id, eclass.sha256) for eclass in frozen_eclasses
        )
        if (
            len(group_owner) != len(frozen_enodes)
            or len(enode_owner) != len(frozen_enodes)
            or len(canonical_group_to_enode) != len(frozen_enodes)
            or len(canonical_enode_to_eclass) != len(frozen_enodes)
            or finalized_enodes != len(frozen_enodes)
            or len(frozen_eclasses) != len(eclass_enode_ids)
            or len(family_index) != len(frozen_families)
            or len(eclass_index) != len(frozen_eclasses)
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
            family_index_records=family_index_records,
            eclass_index_records=eclass_index_records,
            family_index=MappingProxyType(family_index),
            eclass_index=MappingProxyType(eclass_index),
            sha256="",
        )
        self.authenticated_components = partial_capsule._replace(
            sha256=_digest(
                "authenticated-components",
                _component_capsule_payload(partial_capsule),
            )
        )
        self.workset._component_capsule = self.authenticated_components
        return frozen_families, frozen_enodes, frozen_eclasses

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
        except v2._PhysicalWorkBoundExceeded as error:
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
) -> _AuthenticatedComponents:
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
        or not isinstance(capsule.family_index, Mapping)
        or not isinstance(capsule.eclass_index, Mapping)
    ):
        raise RuntimeError("retained packed component authentication failed")
    return capsule


class PackedScalarFrontier:
    """Deep build/select module retaining one cumulative relation session."""

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
        if not isinstance(
            frozen_operator,
            operator_successor.FrozenTypedOperatorSuccessor,
        ):
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

    def _authenticate(self) -> tuple[_AuthenticatedFrozen, _AuthenticatedComponents]:
        if self._state.status != FRONTIER_READY or self._state.frozen_workset is None:
            raise RuntimeError("only a complete packed frontier can be authenticated")
        if self._operator.live_checkpoint_sha256 != self._state.operator_checkpoint_sha256:
            raise RuntimeError("packed frontier operator checkpoint changed")
        authenticated = self._workset.require_retained(self._state.frozen_workset)
        capsule = _require_authenticated_components(
            self._workset,
            self._state,
            self._authenticated_components,
        )
        if self._state.sha256 != _digest("frontier", self._state_payload()):
            raise RuntimeError("packed frontier authentication failed")
        return authenticated, capsule

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

    def _materialize(
        self,
        witness: packed.ProofWitness,
        capsule: _AuthenticatedComponents,
    ) -> PackedWitness:
        self._workset.session.charge_frontier("materialize-witness")

        def render(family_id: str, binding: tuple[int, ...]) -> str:
            self._workset.session.charge_frontier("materialize-family-lookup")
            family = capsule.family_index.get(family_id)
            if type(family) is not PackedFamily or family.family_id != family_id:
                raise RuntimeError("proof witness family index changed")
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
            authenticated, capsule = self._authenticate()
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
                raise v2._PhysicalWorkBoundExceeded("physical-races", attempted_races)
            self._physical_races = attempted_races
            race = semantic_race.select_exact_semantic_vector(
                self._state.root_eclass_ids,
                tuple(item.physical_outputs for item in self._state.eclasses),
                desired,
            )
            if self._operator.live_checkpoint_sha256 != before:
                raise RuntimeError("operator checkpoint changed during selection")
            if race.status == semantic_race.STOP_NO_EXACT:
                return self._selection(
                    STOP_NO_EXACT,
                    race=race,
                    checkpoint_before=before,
                )
            if race.status != semantic_race.SELECTED or race.winner_key is None:
                raise RuntimeError("unique e-class bank returned a physical tie")
            self._workset.session.charge_frontier("winner-eclass-index-lookup")
            eclass = capsule.eclass_index.get(race.winner_key)
            if type(eclass) is not PackedEClass or eclass.eclass_id != race.winner_key:
                raise RuntimeError("physical winner references an unknown e-class")
            frozen = self._state.frozen_workset
            if frozen is None:
                raise RuntimeError("selected frontier lost its published workset")
            forest = self._workset.materialize_selected_forest(
                frozen,
                authenticated,
                eclass.group_refs,
            )
            proof_count = forest.exact_logical_recipe_count
            proof_witnesses = forest.first_two_witnesses
            if proof_count < 1 or not proof_witnesses:
                raise RuntimeError("selected physical e-class has no proof")
            witnesses = tuple(self._materialize(item, capsule) for item in proof_witnesses[:2])
            self._concrete_proof_witnesses = len(proof_witnesses)
            if proof_count == 1:
                return self._selection(
                    SELECTED,
                    selected_eclass_id=eclass.eclass_id,
                    exact_proof_count=proof_count,
                    witnesses=witnesses,
                    selected_proof_forest_sha256=forest.sha256,
                    race=race,
                    checkpoint_before=before,
                )
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
            self._concrete_proof_witnesses = 0
            return self._selection(STOP_WORK_BOUND)
        except v2._PhysicalWorkBoundExceeded as error:
            self._bound_name = error.name
            self._bound_attempted = error.attempted
            self._concrete_proof_witnesses = 0
            return self._selection(STOP_WORK_BOUND)


PackedPartitionHandle = v2.PackedPartitionHandle


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
