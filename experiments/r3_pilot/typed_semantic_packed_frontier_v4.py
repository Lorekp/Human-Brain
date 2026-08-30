"""Lazy proof-plan packed scalar frontier.

V3 constructionally certifies its publication and materializes only the
selected proof forest, but it still constructs every operator family's full
decision topology while the desired vector is absent.  This separately named
V4 keeps the banked V3 grammar, physical predictor, race schedule, component
publication, and public ``build`` then ``select`` seam.  Its one replacement is
the semantic-partition workset: operator groups retain content-addressed proof
plans, and only the physically selected group becomes a concrete relation.

Terminal domains remain concrete packed relations.  Operator plans retain the
ordered child occurrence slots, exact constraints, and semantic-tuple
coverage.  Every plan node and edge is charged when inserted; the append-only
registry makes publication sealing constant work, while selected traversal
rechecks each visited content address.  Build remains desired-free and creates
no concrete operator MDD, proof forest, or witness.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
from itertools import product
from types import MappingProxyType
from typing import NamedTuple

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as v3

FRONTIER_READY = v3.FRONTIER_READY
STOP_WORK_BOUND = v3.STOP_WORK_BOUND
SELECTED = v3.SELECTED
STOP_NO_EXACT = v3.STOP_NO_EXACT
STOP_AMBIGUOUS = v3.STOP_AMBIGUOUS

MAX_RECIPE_RELATION_WORK = 30_000
MAX_PHYSICAL_PREDICTIONS = v3.MAX_PHYSICAL_PREDICTIONS
MAX_PHYSICAL_RACES = v3.MAX_PHYSICAL_RACES

_NAMESPACE = b"typed-semantic-packed-frontier-lazy-proof-plan-v4\0"
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


def _constraint_records(
    constraints: Sequence[packed.SlotConstraint],
) -> tuple[tuple[object, ...], ...]:
    return tuple(v2._constraint_record(item) for item in constraints)


class _TupleProofRecipe(NamedTuple):
    schema: tuple[packed.TypedSlot, ...]
    child_root_ids: tuple[str, ...]
    child_schemas: tuple[tuple[packed.TypedSlot, ...], ...]
    child_target_slots: tuple[tuple[packed.TypedSlot, ...], ...]
    constraints: tuple[packed.SlotConstraint, ...]
    domains: tuple[tuple[int, ...], ...]


class _AggregateProofPlan(NamedTuple):
    plan_id: str
    kind: str
    schema: tuple[packed.TypedSlot, ...]
    child_plan_ids: tuple[str, ...]
    recipes: tuple[_TupleProofRecipe, ...]
    domains: tuple[tuple[int, ...], ...]


def _recipe_payload(plan: _TupleProofRecipe) -> dict[str, object]:
    return {
        "schema": tuple(slot.to_record() for slot in plan.schema),
        "child_root_ids": plan.child_root_ids,
        "child_schemas": tuple(
            tuple(slot.to_record() for slot in schema) for schema in plan.child_schemas
        ),
        "child_target_slots": tuple(
            tuple(slot.to_record() for slot in schema) for schema in plan.child_target_slots
        ),
        "constraints": _constraint_records(plan.constraints),
        "domains": plan.domains,
    }


def _aggregate_plan_payload(plan: _AggregateProofPlan) -> dict[str, object]:
    return {
        "kind": plan.kind,
        "schema": tuple(slot.to_record() for slot in plan.schema),
        "child_plan_ids": plan.child_plan_ids,
        "recipes": tuple(_recipe_payload(item) for item in plan.recipes),
        "domains": plan.domains,
    }


def _plan_payload(plan: _AggregateProofPlan) -> dict[str, object]:
    if type(plan) is _AggregateProofPlan:
        return _aggregate_plan_payload(plan)
    raise TypeError("unknown proof-plan node")


class PackedFrozenWorkset(NamedTuple):
    arena_generation: int
    topology_node_count: int
    topology_sha256: str
    plan_node_count: int
    plan_edge_count: int
    plan_recipe_count: int
    plan_registry_sha256: str
    partitions: tuple[v3.PackedPublishedPartition, ...]
    registry_digest: str
    metadata_sha256: str
    publication_certificate_sha256: str
    sha256: str


def _frozen_payload(frozen: PackedFrozenWorkset) -> dict[str, object]:
    return {
        "arena_generation": frozen.arena_generation,
        "topology_node_count": frozen.topology_node_count,
        "topology_sha256": frozen.topology_sha256,
        "plan_node_count": frozen.plan_node_count,
        "plan_edge_count": frozen.plan_edge_count,
        "plan_recipe_count": frozen.plan_recipe_count,
        "plan_registry_sha256": frozen.plan_registry_sha256,
        "partitions": tuple(item.sha256 for item in frozen.partitions),
        "registry_digest": frozen.registry_digest,
        "metadata_sha256": frozen.metadata_sha256,
        "publication_certificate_sha256": frozen.publication_certificate_sha256,
    }


class _AuthenticatedFrozen(NamedTuple):
    owner_token: object
    frozen_sha256: str
    certificate_sha256: str
    retained_seal_sha256: str
    arena: v3._StrictRetainedArena
    plans: Mapping[str, _AggregateProofPlan]
    group_index: object
    root_index: object


class _SemanticPartitionWorkset(v3._SemanticPartitionWorkset):
    """Deep module retaining lazy operator proof plans and concrete terminals."""

    __slots__ = (
        "_plan_edge_count",
        "_plan_recipe_count",
        "_plan_registry_sha256",
        "_plans",
        "_retained_plans",
        "_root_domains",
    )

    def __init__(self, session: v2._FrontierSession) -> None:
        super().__init__(session)
        self._plans: dict[str, _AggregateProofPlan] = {}
        self._root_domains: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._plan_registry_sha256 = _digest("empty-plan-registry", ())
        self._plan_edge_count = 0
        self._plan_recipe_count = 0
        self._retained_plans: Mapping[str, _AggregateProofPlan] | None = None

    def _remember_domains(
        self,
        root_id: str,
        domains: tuple[tuple[int, ...], ...],
    ) -> None:
        self.session.charge_workset("proof-plan-domain-lookup")
        incumbent = self._root_domains.get(root_id)
        if incumbent is None:
            self.session.charge_workset("proof-plan-domain-insert")
            self._root_domains[root_id] = domains
        elif incumbent != domains:
            raise RuntimeError("proof-plan root domain identity changed")

    def _insert_plan(self, plan: _AggregateProofPlan, edge_count: int) -> str:
        self.session.charge_workset("proof-plan-node-lookup")
        incumbent = self._plans.get(plan.plan_id)
        if incumbent is not None:
            if incumbent != plan:
                raise RuntimeError("proof-plan content address collision")
            return plan.plan_id
        self.session.charge_workset("proof-plan-node-insert")
        for _index in range(edge_count):
            self.session.charge_workset("proof-plan-edge-insert")
        self._plans[plan.plan_id] = plan
        self._plan_edge_count += edge_count
        self._plan_registry_sha256 = _digest(
            "plan-registry-step",
            {
                "prior": self._plan_registry_sha256,
                "ordinal": len(self._plans) - 1,
                "plan_id": plan.plan_id,
                "payload": _plan_payload(plan),
            },
        )
        self._root_domains[plan.plan_id] = plan.domains
        return plan.plan_id

    def _tuple_recipe(
        self,
        plan: v2._FamilyPlan,
        child_roots: tuple[str, ...],
        child_schemas: tuple[tuple[packed.TypedSlot, ...], ...],
        domains: tuple[tuple[int, ...], ...],
        constraints: tuple[packed.SlotConstraint, ...] | None = None,
    ) -> _TupleProofRecipe:
        checks = plan.constraints if constraints is None else constraints
        recipe = _TupleProofRecipe(
            schema=plan.schema,
            child_root_ids=child_roots,
            child_schemas=child_schemas,
            child_target_slots=plan.child_target_slots,
            constraints=checks,
            domains=domains,
        )
        self.session.charge_workset("proof-plan-recipe-insert")
        self._plan_recipe_count += 1
        return recipe

    def _aggregate_plan(
        self,
        kind: str,
        schema: tuple[packed.TypedSlot, ...],
        child_plan_ids: Sequence[str] = (),
        recipes: Sequence[_TupleProofRecipe] = (),
    ) -> str:
        if kind not in {"semantic-group", "family-coverage", "normalization-source"}:
            raise ValueError("unknown proof-plan aggregate kind")
        children = tuple(child_plan_ids)
        frozen_recipes = tuple(recipes)
        if bool(children) == bool(frozen_recipes):
            raise RuntimeError("proof-plan aggregate cannot be empty")
        domains: list[tuple[int, ...]] = []
        for index in range(len(schema)):
            values: set[int] = set()
            if children:
                for child_id in children:
                    child_domains = self._root_domains.get(child_id)
                    if child_domains is None or len(child_domains) != len(schema):
                        raise RuntimeError("proof-plan aggregate child schema changed")
                    values.update(child_domains[index])
            else:
                for recipe in frozen_recipes:
                    if recipe.schema != schema or len(recipe.domains) != len(schema):
                        raise RuntimeError("proof-plan recipe schema changed")
                    values.update(recipe.domains[index])
            domains.append(tuple(sorted(values)))
        partial = _AggregateProofPlan(
            plan_id="",
            kind=kind,
            schema=schema,
            child_plan_ids=children,
            recipes=frozen_recipes,
            domains=tuple(domains),
        )
        node = partial._replace(
            plan_id=_digest("aggregate-proof-plan", _aggregate_plan_payload(partial))
        )
        return self._insert_plan(node, len(children))

    def _capture_terminal_domains(self, handle: v2.PackedPartitionHandle) -> None:
        record = self._require_handle(handle)
        roots = (record.root_id, *(group.root_id for group in record.groups))
        for root_id in roots:
            if root_id is None or root_id in self._root_domains:
                continue
            node, edges = self._read_node_edges(root_id)
            if node.slot != record.plan.schema[0]:
                raise RuntimeError("terminal proof root schema changed")
            if any(child != packed._TERMINAL_ID for _value, child in edges):
                raise RuntimeError("terminal proof root gained depth")
            self._remember_domains(root_id, (tuple(value for value, _child in edges),))

    def seed_family(
        self,
        plan: v2._FamilyPlan,
        relation: packed.PackedBindingRelation,
        output_for_value: Callable[[int], tuple[output_cells.TypedOutputEncoding, ...]],
    ) -> v2.PackedPartitionHandle:
        handle = super().seed_family(plan, relation, output_for_value)
        self._capture_terminal_domains(handle)
        return handle

    def _product_domains(
        self,
        plan: v2._FamilyPlan,
        child_roots: tuple[str, ...],
        child_schemas: tuple[tuple[packed.TypedSlot, ...], ...],
        *,
        constraints: tuple[packed.SlotConstraint, ...] | None = None,
    ) -> tuple[tuple[int, ...], ...] | None:
        checks = plan.constraints if constraints is None else constraints
        if len(checks) > 1:
            raise RuntimeError("V4 scalar plan supports one pushed constraint")
        by_slot: dict[packed.TypedSlot, tuple[int, ...]] = {}
        for schema, targets, root_id in zip(
            child_schemas,
            plan.child_target_slots,
            child_roots,
            strict=True,
        ):
            domains = self._root_domains.get(root_id)
            if domains is None or len(domains) != len(schema):
                raise RuntimeError("child proof-plan domains are absent")
            for target, values in zip(targets, domains, strict=True):
                if target in by_slot:
                    raise RuntimeError("proof-plan product slots overlap")
                by_slot[target] = values
        if tuple(sorted(by_slot, key=lambda slot: slot.sort_key)) != plan.schema:
            raise RuntimeError("proof-plan product schema changed")

        for check in checks:
            self.session.charge_workset("proof-plan-constraint-decision")
            left = by_slot.get(check.left)
            if left is None:
                raise RuntimeError("proof-plan constraint lost its left slot")
            if check.right_slot is None:
                if check.operator != "ne" or type(check.right_value) is not int:
                    raise RuntimeError("unsupported scalar value constraint")
                by_slot[check.left] = tuple(value for value in left if value != check.right_value)
            else:
                right = by_slot.get(check.right_slot)
                if right is None:
                    raise RuntimeError("proof-plan constraint lost its right slot")
                if check.operator != "le":
                    raise RuntimeError("unsupported scalar slot constraint")
                if not left or not right:
                    return None
                left_limit = max(right)
                right_limit = min(left)
                by_slot[check.left] = tuple(value for value in left if value <= left_limit)
                by_slot[check.right_slot] = tuple(value for value in right if value >= right_limit)
        domains = tuple(by_slot[slot] for slot in plan.schema)
        if any(not values for values in domains):
            return None
        return domains

    def apply_family(
        self,
        plan: v2._FamilyPlan,
        child_handles: tuple[v2.PackedPartitionHandle, ...],
        predict: Callable[
            [str, tuple[tuple[output_cells.TypedOutputEncoding, ...], ...]],
            tuple[output_cells.TypedOutputEncoding, ...],
        ],
    ) -> v2.PackedPartitionHandle:
        self._operation("apply-family-plan")
        self._ensure_open()
        if len(child_handles) != len(plan.child_family_ids):
            raise ValueError("semantic family child arity changed")
        records: list[v2._PartitionRecord] = []
        for expected_family, handle in zip(
            plan.child_family_ids,
            child_handles,
            strict=True,
        ):
            record = self._require_handle(handle)
            if record.plan.family_id != expected_family:
                raise RuntimeError("semantic family child substitution")
            records.append(record)

        grouped_outputs: dict[str, tuple[output_cells.TypedOutputEncoding, ...]] = {}
        grouped_recipes: dict[str, list[_TupleProofRecipe]] = {}
        normalization_recipes: list[_TupleProofRecipe] = []
        group_ranges = tuple(range(len(record.groups)) for record in records)
        child_schemas = tuple(record.plan.schema for record in records)
        for indices in product(*group_ranges):
            self.session.charge_workset("semantic-tuple-read")
            child_groups = tuple(
                records[index].groups[group_index] for index, group_index in enumerate(indices)
            )
            child_roots = tuple(group.root_id for group in child_groups)
            if any(root is None for root in child_roots):
                raise RuntimeError("nonempty semantic group lost its proof root")
            roots = tuple(root for root in child_roots if root is not None)
            domains = self._product_domains(plan, roots, child_schemas)
            if domains is None:
                if plan.normalization_id is not None:
                    if len(records) != 2 or len(plan.child_target_slots[1]) != 1:
                        raise RuntimeError("normalization child shape changed")
                    raw_domains = self._product_domains(
                        plan,
                        roots,
                        child_schemas,
                        constraints=(),
                    )
                    if raw_domains is None:
                        raise RuntimeError("normalization source tuple is empty")
                    right_values = raw_domains[plan.schema.index(plan.child_target_slots[1][0])]
                    if right_values != (0,):
                        raise RuntimeError("normalization did not isolate exact right zero")
                    normalization_recipes.append(
                        self._tuple_recipe(
                            plan,
                            roots,
                            child_schemas,
                            raw_domains,
                            constraints=(),
                        )
                    )
                continue

            recipe = self._tuple_recipe(
                plan,
                roots,
                child_schemas,
                domains,
            )
            child_outputs = tuple(group.physical_outputs for group in child_groups)
            outputs = predict(plan.production, child_outputs)
            output_sha = v2._vector_sha256(outputs)
            self.session.charge_workset("semantic-plan-group-upsert")
            if output_sha not in grouped_recipes:
                grouped_outputs[output_sha] = outputs
                grouped_recipes[output_sha] = [recipe]
            else:
                grouped_recipes[output_sha].append(recipe)

        groups: list[v2._MutableSemanticGroup] = []
        group_plan_ids: list[str] = []
        for output_sha in sorted(grouped_recipes):
            group_plan_id = self._aggregate_plan(
                "semantic-group",
                plan.schema,
                recipes=grouped_recipes[output_sha],
            )
            group_plan_ids.append(group_plan_id)
            groups.append(
                v2._MutableSemanticGroup(
                    group_plan_id,
                    grouped_outputs[output_sha],
                    [],
                )
            )
        if not groups:
            raise RuntimeError("semantic family produced no supported proof plan")
        family_root = self._aggregate_plan(
            "family-coverage",
            plan.schema,
            group_plan_ids,
        )

        normalization: v2.PackedNormalizationRecord | None = None
        if plan.normalization_id is not None:
            if not normalization_recipes or records[0].root_id is None:
                raise RuntimeError("declared proof normalization did not fire")
            source_root = self._aggregate_plan(
                "normalization-source",
                plan.schema,
                recipes=normalization_recipes,
            )
            if len(normalization_recipes) != len(records[0].groups):
                raise RuntimeError("normalization source did not cover its target family")
            required_constants = ((plan.child_target_slots[1][0], 0),)
            partial = v2.PackedNormalizationRecord(
                normalization_id=plan.normalization_id,
                source_family_id=plan.family_id,
                target_family_id=records[0].plan.family_id,
                retained_source_slots=plan.child_target_slots[0],
                target_slots=records[0].plan.schema,
                required_constants=required_constants,
                source_root_id=source_root,
                recovered_root_id=records[0].root_id,
                target_root_id=records[0].root_id,
                sha256="",
            )
            normalization = v2.PackedNormalizationRecord(
                normalization_id=partial.normalization_id,
                source_family_id=partial.source_family_id,
                target_family_id=partial.target_family_id,
                retained_source_slots=partial.retained_source_slots,
                target_slots=partial.target_slots,
                required_constants=partial.required_constants,
                source_root_id=partial.source_root_id,
                recovered_root_id=partial.recovered_root_id,
                target_root_id=partial.target_root_id,
                sha256=v2._digest(
                    "normalization-record",
                    v2._normalization_payload(partial),
                ),
            )
        return self._record_partition(
            plan,
            family_root,
            tuple(groups),
            v2._relation_identity(plan.schema, family_root),
            normalization,
        )

    def publish(self) -> PackedFrozenWorkset:
        """Freeze construction-certified plans plus terminal decision topology."""

        self._operation("publish-lazy-plan")
        self._ensure_open()
        partitions: list[v3.PackedPublishedPartition] = []
        group_index: dict[str, v3.PackedSemanticGroupRef] = {}
        plan_root_index: dict[tuple[object, ...], str] = {}
        terminal_root_specs: list[tuple[str | None, tuple[packed.TypedSlot, ...]]] = []
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
            group_refs: list[v3.PackedSemanticGroupRef] = []
            for group in record.groups:
                self.session.charge_workset("publication-group")
                for output in group.physical_outputs:
                    output_cells.TypedOutputCellBank.validate(output)
                child_refs = self._canonical_child_refs(group.child_refs)
                child_ids = record.plan.child_family_ids
                if not child_ids and child_refs:
                    raise RuntimeError("published terminal group gained child semantics")
                root_id = group.root_id
                if root_id in self._plans:
                    node = self._plans[root_id]
                    if (
                        type(node) is not _AggregateProofPlan
                        or node.kind != "semantic-group"
                        or node.schema != record.plan.schema
                        or not node.recipes
                    ):
                        raise RuntimeError("published semantic group plan changed")
                    self.session.charge_workset("published-plan-root-index-insert")
                    root_key = (
                        root_id,
                        tuple(slot.to_record() for slot in record.plan.schema),
                    )
                    identity = v2._relation_identity(record.plan.schema, root_id)
                    incumbent_identity = plan_root_index.setdefault(root_key, identity)
                    if incumbent_identity != identity:
                        raise RuntimeError("published plan-root identity collision")
                else:
                    terminal_root_specs.append((root_id, record.plan.schema))
                output_sha = v3._vector_sha256(group.physical_outputs)
                group_id = _digest(
                    "semantic-group-id",
                    {
                        "partition_id": public_partition_id,
                        "family_id": family_id,
                        "root_id": root_id,
                        "physical_output_sha256": output_sha,
                    },
                )
                partial_group = v3.PackedSemanticGroupRef(
                    partition_id=public_partition_id,
                    family_id=family_id,
                    group_id=group_id,
                    schema=record.plan.schema,
                    root_id=root_id,
                    child_semantic_refs=child_refs,
                    physical_outputs=group.physical_outputs,
                    physical_output_sha256=output_sha,
                    sha256="",
                )
                published_group = partial_group._replace(
                    sha256=_digest(
                        "published-semantic-group",
                        v3._group_payload(partial_group),
                    )
                )
                self.session.charge_workset("published-group-index-insert")
                incumbent = group_index.setdefault(group_id, published_group)
                if incumbent is not published_group:
                    raise RuntimeError("published semantic group identity collision")
                group_refs.append(published_group)

            normalization = self._freeze_normalization(record.normalization)
            if normalization is not None and (
                normalization.normalization_id != record.plan.normalization_id
                or normalization.source_family_id != family_id
                or normalization.target_family_id != record.plan.child_family_ids[0]
                or normalization.recovered_root_id != normalization.target_root_id
                or normalization.source_root_id not in self._plans
            ):
                raise RuntimeError("published normalization authentication failed")
            if record.root_id in self._plans:
                family_plan = self._plans[record.root_id]
                if (
                    type(family_plan) is not _AggregateProofPlan
                    or family_plan.kind != "family-coverage"
                    or family_plan.schema != record.plan.schema
                    or family_plan.child_plan_ids != tuple(group.root_id for group in record.groups)
                ):
                    raise RuntimeError("published family coverage plan changed")
            else:
                terminal_root_specs.append((record.root_id, record.plan.schema))
            partial_partition = v3.PackedPublishedPartition(
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
                constraints=_constraint_records(record.plan.constraints),
                normalization_id=record.plan.normalization_id,
                normalization=normalization,
                groups=tuple(group_refs),
                sha256="",
            )
            partitions.append(
                partial_partition._replace(
                    sha256=_digest(
                        "published-partition",
                        v3._partition_payload(partial_partition),
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
                "plan_node_count": len(self._plans),
                "plan_edge_count": self._plan_edge_count,
                "plan_recipe_count": self._plan_recipe_count,
                "plan_registry_sha256": self._plan_registry_sha256,
            },
        )
        self.session.charge_workset("publication-metadata-seal")
        self._operation("authenticate-terminal-topology")
        mutable_arena, terminal_root_index, topology_node_count, topology_sha256 = (
            self._authenticate_topology(terminal_root_specs)
        )
        arena = v3._StrictRetainedArena(
            nodes=MappingProxyType(dict(mutable_arena.nodes)),
        )
        plans = MappingProxyType(dict(self._plans))
        root_index = dict(terminal_root_index)
        for key, identity in plan_root_index.items():
            incumbent = root_index.setdefault(key, identity)
            if incumbent != identity:
                raise RuntimeError("published root index collision")
        certificate_sha256 = _digest(
            "publication-certificate",
            {
                "metadata_sha256": metadata_sha256,
                "topology_node_count": topology_node_count,
                "topology_sha256": topology_sha256,
                "plan_node_count": len(plans),
                "plan_edge_count": self._plan_edge_count,
                "plan_recipe_count": self._plan_recipe_count,
                "plan_registry_sha256": self._plan_registry_sha256,
                "arena_generation": self._generation,
            },
        )
        partial = PackedFrozenWorkset(
            arena_generation=self._generation,
            topology_node_count=topology_node_count,
            topology_sha256=topology_sha256,
            plan_node_count=len(plans),
            plan_edge_count=self._plan_edge_count,
            plan_recipe_count=self._plan_recipe_count,
            plan_registry_sha256=self._plan_registry_sha256,
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
                "plan_registry_sha256": self._plan_registry_sha256,
            },
        )
        authenticated = _AuthenticatedFrozen(
            owner_token=self._owner_token,
            frozen_sha256=frozen.sha256,
            certificate_sha256=certificate_sha256,
            retained_seal_sha256=retained_seal,
            arena=arena,
            plans=plans,
            group_index=MappingProxyType(group_index),
            root_index=MappingProxyType(root_index),
        )
        self._retained_arena = arena
        self._retained_plans = plans
        self._frozen = frozen  # type: ignore[assignment]
        self._authenticated = authenticated  # type: ignore[assignment]
        self._sealed = True
        return frozen

    def require_retained(self, frozen: PackedFrozenWorkset) -> _AuthenticatedFrozen:
        """Authenticate the exact frozen terminal arena and proof-plan map."""

        self._operation("authenticate-retained-plan-publication-seal")
        self.session.charge_workset("retained-plan-publication-seal")
        authenticated = self._authenticated
        expected_seal = _digest(
            "retained-publication-seal",
            {
                "frozen_sha256": frozen.sha256,
                "certificate_sha256": frozen.publication_certificate_sha256,
                "plan_registry_sha256": frozen.plan_registry_sha256,
            },
        )
        if (
            type(frozen) is not PackedFrozenWorkset
            or frozen is not self._frozen
            or type(authenticated) is not _AuthenticatedFrozen
            or authenticated is not self._authenticated
            or authenticated.owner_token is not self._owner_token
            or authenticated.frozen_sha256 != frozen.sha256
            or authenticated.certificate_sha256 != frozen.publication_certificate_sha256
            or authenticated.retained_seal_sha256 != expected_seal
            or frozen.arena_generation != self._generation
            or type(authenticated.arena) is not v3._StrictRetainedArena
            or authenticated.arena is not self._retained_arena
            or type(authenticated.arena.nodes) is not _MAPPING_PROXY_TYPE
            or authenticated.plans is not self._retained_plans
            or type(authenticated.plans) is not _MAPPING_PROXY_TYPE
            or len(authenticated.plans) != frozen.plan_node_count
        ):
            raise RuntimeError("retained lazy proof-plan authentication failed")
        return authenticated

    @staticmethod
    def _validate_plan(node: _AggregateProofPlan, plan_id: str) -> None:
        if type(node) is not _AggregateProofPlan:
            raise RuntimeError("retained proof-plan node type changed")
        expected = _digest("aggregate-proof-plan", _aggregate_plan_payload(node))
        if node.plan_id != plan_id or plan_id != expected:
            raise RuntimeError("retained proof-plan content address changed")

    def _materialize_plan_root(
        self,
        schema: tuple[packed.TypedSlot, ...],
        root_id: str,
        authenticated: _AuthenticatedFrozen,
        memo: dict[str, packed.PackedBindingRelation],
    ) -> packed.PackedBindingRelation:
        cached = memo.get(root_id)
        if cached is not None:
            self.session.charge_workset("selected-plan-cache-hit")
            return cached
        node = authenticated.plans.get(root_id)
        if node is None:
            relation, _exact, _bindings = self._materialize_root_once(
                schema,
                root_id,
                authenticated.arena,
            )
            memo[root_id] = relation
            return relation

        self.session.charge_workset("selected-plan-node")
        self._validate_plan(node, root_id)
        if node.schema != schema:
            raise RuntimeError("selected proof-plan schema changed")
        if node.kind != "semantic-group" or node.child_plan_ids or not node.recipes:
            raise RuntimeError("non-group proof plan reached selection")
        relation: packed.PackedBindingRelation | None = None
        for recipe in node.recipes:
            self.session.charge_workset("selected-plan-recipe")
            if recipe.schema != schema:
                raise RuntimeError("selected proof-plan recipe schema changed")
            renamed_children: list[packed.PackedBindingRelation] = []
            for child_id, child_schema, targets in zip(
                recipe.child_root_ids,
                recipe.child_schemas,
                recipe.child_target_slots,
                strict=True,
            ):
                self.session.charge_workset("selected-plan-edge")
                child = self._materialize_plan_root(
                    child_schema,
                    child_id,
                    authenticated,
                    memo,
                )
                renamed_children.append(self.session.rename(child, targets))
            if not renamed_children:
                raise RuntimeError("selected tuple plan has no children")
            tuple_relation = renamed_children[0]
            for child in renamed_children[1:]:
                tuple_relation = self.session.cartesian_product(tuple_relation, child)
            if recipe.constraints:
                tuple_relation = self.session.select(
                    tuple_relation,
                    recipe.constraints,
                )
            if tuple_relation.slots != schema or tuple_relation.root_id is None:
                raise RuntimeError("selected tuple plan materialized incorrectly")
            relation = (
                tuple_relation if relation is None else self.session.union(relation, tuple_relation)
            )
        if relation is None or relation.root_id is None:
            raise RuntimeError("selected group plan materialized empty")
        memo[root_id] = relation
        return relation

    def materialize_selected_forest(
        self,
        frozen: PackedFrozenWorkset,
        authenticated: _AuthenticatedFrozen,
        groups: tuple[v3.PackedSemanticGroupRef, ...],
    ) -> packed.PackedProofForest:
        """Materialize only the winning semantic groups from retained plans."""

        self._operation("materialize-selected-lazy-proof-plan")
        if (
            authenticated is not self._authenticated
            or authenticated.owner_token is not self._owner_token
            or authenticated.frozen_sha256 != frozen.sha256
            or frozen is not self._frozen
            or authenticated.arena is not self._retained_arena
            or authenticated.plans is not self._retained_plans
            or not isinstance(authenticated.group_index, Mapping)
            or not isinstance(authenticated.root_index, Mapping)
        ):
            raise RuntimeError("stale or foreign lazy proof-plan publication")
        slices: list[packed.PackedFamilySlice] = []
        seen_families: set[str] = set()
        memo: dict[str, packed.PackedBindingRelation] = {}
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
            if group.root_id is None:
                raise RuntimeError("selected semantic group has no proof root")
            seen_families.add(group.family_id)
            relation = self._materialize_plan_root(
                group.schema,
                group.root_id,
                authenticated,
                memo,
            )
            slices.append(self.session.family_slice(group.family_id, relation))
        return self.session.proof_forest(slices, saturation_limit=2)


class _Builder(v3._Builder):
    """Banked V3 scalar grammar over the V4 lazy-plan workset."""

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
        self.session = v2._FrontierSession(work_limit=MAX_RECIPE_RELATION_WORK)
        self.workset = _SemanticPartitionWorkset(self.session)
        self.authenticated_components = None

    def _predict_vector(
        self,
        production: str,
        children: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        """Resolve one semantic tuple through one metered vector-cache probe."""

        if len(children) not in {1, 2}:
            raise ValueError("only unary and binary semantic families are supported")
        width = len(children[0])
        if any(len(child) != width for child in children):
            raise RuntimeError("physical semantic tuple width changed")
        self.session.charge_workset("prediction-vector-cache-lookup")
        outputs: list[output_cells.TypedOutputEncoding] = []
        for case_index in range(width):
            left = children[0][case_index]
            right = None if len(children) == 1 else children[1][case_index]
            key = (
                production,
                left.sha256,
                None if right is None else right.sha256,
            )
            cached = self._prediction_cache.get(key)
            if cached is not None:
                output_cells.TypedOutputCellBank.validate(cached)
                self.prediction_cache_hits += 1
                outputs.append(cached)
                continue
            attempted_prediction = self.physical_predictions + 1
            if attempted_prediction > MAX_PHYSICAL_PREDICTIONS:
                raise v2._PhysicalWorkBoundExceeded(
                    "physical-predictions",
                    attempted_prediction,
                )
            reserved_races = 1 if production == "absolute" else (2 if production == "add" else 4)
            attempted_race = self.physical_races + reserved_races
            if attempted_race > MAX_PHYSICAL_RACES:
                raise v2._PhysicalWorkBoundExceeded("physical-races", attempted_race)
            self.physical_predictions = attempted_prediction
            self._record("prediction", v2._digest("prediction-key", key))
            if right is None:
                action = v2.FactorAction(
                    (),
                    "APPLY1",
                    production,
                    "SELECTED",
                    "NONE",
                    None,
                )
                prediction = self.operator.predict(left, action)
            else:
                action = v2.FactorAction(
                    (),
                    "APPLY2",
                    production,
                    "SELECTED",
                    "SCOPE0",
                    None,
                )
                prediction = self.operator.predict(left, action, scopes=(right,))
            if len(prediction.races) != reserved_races:
                raise RuntimeError("physical operator race reservation drifted")
            self.physical_races += len(prediction.races)
            output = output_cells.TypedOutputCellBank.validate(prediction.output)
            self.session.charge_workset("prediction-cache-insert")
            incumbent = self._prediction_cache.setdefault(key, output)
            if incumbent != output:
                raise RuntimeError("physical prediction cache identity collision")
            outputs.append(output)
        return tuple(outputs)


class PackedScalarFrontier(v3.PackedScalarFrontier):
    """V3's deep public seam backed only by the V4 lazy proof-plan builder."""

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


PackedEClass = v3.PackedEClass
PackedENode = v3.PackedENode
PackedFamily = v3.PackedFamily
PackedFrontierSelection = v3.PackedFrontierSelection
PackedFrontierState = v3.PackedFrontierState
PackedFrontierWorkAudit = v3.PackedFrontierWorkAudit
PackedPartitionHandle = v2.PackedPartitionHandle
PackedPublishedPartition = v3.PackedPublishedPartition
PackedSemanticGroupRef = v3.PackedSemanticGroupRef
PackedWitness = v3.PackedWitness

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
