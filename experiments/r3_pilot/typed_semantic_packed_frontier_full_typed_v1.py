"""Packed-primary full-typed frontier vertical slice.

This separately named slice keeps V3's deep two-operation seam: build a frozen
frontier without desired outputs, then physically select from its authenticated
root bank.  Its deliberately small grammar is nevertheless genuinely
cross-type: a ``List[Int]`` scope, an ``Int`` item role and generic literals
compose through ``less-than`` Bool events, a compiler-only predicate binder,
and a physically accumulated ``count`` Int root.  ``sum`` is the simpler typed
control.

Packed relations are the primary agenda before semantic expansion.  V3 owns
construction-certified publication and selected-only proof materialization;
the typed layer owns immutable root/bound contexts and root-type filtering.
No task, family, prompt, desired output, concrete recipe row, or rendered
witness exists during build.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple, cast

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_synthesizer as typed_semantic
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as v3
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramType, TypedSignature

FRONTIER_READY = "PACKED_FULL_TYPED_FRONTIER_V1_READY"
STOP_WORK_BOUND = "STOP_PACKED_FULL_TYPED_FRONTIER_V1_WORK_BOUND"
SELECTED = "PACKED_FULL_TYPED_FRONTIER_V1_SELECTED"
STOP_NO_EXACT = "STOP_PACKED_FULL_TYPED_FRONTIER_V1_NO_EXACT"
STOP_AMBIGUOUS = "STOP_PACKED_FULL_TYPED_FRONTIER_V1_AMBIGUOUS"

MAX_RECIPE_RELATION_WORK = 30_000
MAX_PHYSICAL_PREDICTIONS = 30_000
MAX_PHYSICAL_RACES = 80_000

_NAMESPACE = b"typed-semantic-packed-frontier-full-typed-v1\0"
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))

_INT = ProgramType.Int()
_BOOL = ProgramType.Bool()
_LIST_INT = ProgramType.List(_INT)

_COLLECTION_SLOT = packed.TypedSlot("role", "List[Int]", "scope", 0)
_ITEM_SLOT = packed.TypedSlot("role", "Int", "item", 0)
_LITERAL_SLOT = packed.TypedSlot("literal", "Int", "integer", 0)

_LESS_THAN = tp.FactorAction((), "APPLY2", "less-than", "SELECTED", "SCOPE0", None)
_ADD = tp.FactorAction((), "APPLY2", "add", "SELECTED", "SCOPE0", None)
_LIST_COPY = typed_semantic._MAP_COPY


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


def _vector_sha256(
    values: Sequence[output_cells.TypedOutputEncoding],
) -> str:
    return _digest("physical-vector", tuple(item.sha256 for item in values))


class TypedContextCertificate(NamedTuple):
    signature_sha256: str
    case_arguments_sha256: str
    root_context_shape: tuple[object, ...]
    bound_context_shape: tuple[object, ...]
    item_offsets: tuple[tuple[int, int], ...]
    flattened_items_sha256: str
    output_projection_sha256: str
    operator_checkpoint_sha256: str
    sha256: str


class TypedFamilyRecord(NamedTuple):
    family_id: str
    production: str
    inside_cost: int
    context_role: str
    environment_type_labels: tuple[str, ...]
    result_type_label: str
    root_candidate: bool
    partition_sha256: str
    sha256: str


class _FamilyTyping(NamedTuple):
    context_role: str
    environment_type_labels: tuple[str, ...]
    result_type_label: str
    root_candidate: bool


class _AuthenticatedTypedCapsule(NamedTuple):
    owner_token: object
    frozen_sha256: str
    component_capsule_sha256: str
    context_certificate_sha256: str
    family_records: tuple[TypedFamilyRecord, ...]
    family_index: object
    root_eclasses: tuple[v3.PackedEClass, ...]
    root_eclass_ids: tuple[str, ...]
    root_ordinal_index: object
    sha256: str


class PackedFullTypedWitness(NamedTuple):
    family_id: str
    binding: tuple[int, ...]
    source: str
    sha256: str


class PackedFullTypedWorkAudit(NamedTuple):
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


class PackedFullTypedState(NamedTuple):
    status: str
    signature: TypedSignature
    case_arguments: tuple[tuple[tuple[int, ...]], ...]
    literals: tuple[int, ...]
    requested_max_ast_cost: int
    maximum_completed_ast_cost: int
    context_certificate: TypedContextCertificate | None
    operator_checkpoint_sha256: str
    frozen_workset: v3.PackedFrozenWorkset | None
    families: tuple[v3.PackedFamily, ...]
    enodes: tuple[v3.PackedENode, ...]
    eclasses: tuple[v3.PackedEClass, ...]
    typed_families: tuple[TypedFamilyRecord, ...]
    root_eclasses: tuple[v3.PackedEClass, ...]
    root_eclass_ids: tuple[str, ...]
    component_capsule_sha256: str | None
    typed_capsule_sha256: str | None
    build_work_sha256: str
    sha256: str


class PackedFullTypedSelection(NamedTuple):
    status: str
    frontier_sha256: str
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[PackedFullTypedWitness, ...]
    selected_proof_forest_sha256: str | None
    race: semantic_race.TypedSemanticVectorRaceAudit | None
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    work: PackedFullTypedWorkAudit
    sha256: str


def _context_payload(value: TypedContextCertificate) -> dict[str, object]:
    return {
        "signature_sha256": value.signature_sha256,
        "case_arguments_sha256": value.case_arguments_sha256,
        "root_context_shape": value.root_context_shape,
        "bound_context_shape": value.bound_context_shape,
        "item_offsets": value.item_offsets,
        "flattened_items_sha256": value.flattened_items_sha256,
        "output_projection_sha256": value.output_projection_sha256,
        "operator_checkpoint_sha256": value.operator_checkpoint_sha256,
    }


def _typed_family_payload(value: TypedFamilyRecord) -> dict[str, object]:
    return {
        "family_id": value.family_id,
        "production": value.production,
        "inside_cost": value.inside_cost,
        "context_role": value.context_role,
        "environment_type_labels": value.environment_type_labels,
        "result_type_label": value.result_type_label,
        "root_candidate": value.root_candidate,
        "partition_sha256": value.partition_sha256,
    }


def _root_eclass_payload(value: v3.PackedEClass) -> dict[str, object]:
    return {
        "eclass_id": value.eclass_id,
        "minimum_inside_cost": value.minimum_inside_cost,
        "enode_ids": value.enode_ids,
        "group_refs": tuple(item.sha256 for item in value.group_refs),
        "physical_outputs": tuple(item.sha256 for item in value.physical_outputs),
    }


def _typed_capsule_payload(value: _AuthenticatedTypedCapsule) -> dict[str, object]:
    root_ordinal_index = cast(Mapping[str, int], value.root_ordinal_index)
    return {
        "frozen_sha256": value.frozen_sha256,
        "component_capsule_sha256": value.component_capsule_sha256,
        "context_certificate_sha256": value.context_certificate_sha256,
        "family_records": tuple(item.sha256 for item in value.family_records),
        "root_eclasses": tuple(item.sha256 for item in value.root_eclasses),
        "root_eclass_ids": value.root_eclass_ids,
        "root_ordinal_index": tuple(sorted(root_ordinal_index.items())),
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
) -> PackedFullTypedWorkAudit:
    materialized = dict(relation.event_counts).get("frontier:materialize-witness", 0)
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
    return PackedFullTypedWorkAudit(
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


class _Builder(v3._Builder):
    """Typed grammar adapter behind V3's packed workset/publication seam."""

    def __init__(
        self,
        signature: TypedSignature,
        supplied_case_arguments: Iterable[tuple[object, ...]],
        supplied_literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
    ) -> None:
        self.signature = signature
        self.supplied_case_arguments = supplied_case_arguments
        self.supplied_literals = supplied_literals
        self.max_ast_cost = max_ast_cost
        self.operator = operator
        self.case_arguments: tuple[tuple[tuple[int, ...]], ...] = ()
        self.flattened_items: tuple[int, ...] = ()
        self.item_offsets: tuple[tuple[int, int], ...] = ()
        self.literals: tuple[int, ...] = ()
        self.context_certificate: TypedContextCertificate | None = None
        self.session = v2._FrontierSession(work_limit=MAX_RECIPE_RELATION_WORK)
        self.workset = v3._SemanticPartitionWorkset(self.session)
        self._prediction_cache: dict[
            tuple[str, str, str | None], output_cells.TypedOutputEncoding
        ] = {}
        self.physical_predictions = 0
        self.prediction_cache_hits = 0
        self.physical_races = 0
        self.bound_name: str | None = None
        self.bound_attempted: int | None = None
        self.agenda_trace: list[tuple[str, str]] = []
        self.maximum_completed_ast_cost = 0
        self.normalization_count = 0
        self.authenticated_components: v3._AuthenticatedComponents | None = None
        self.typed_capsule: _AuthenticatedTypedCapsule | None = None
        self._family_typings: dict[str, _FamilyTyping] = {}

    def _prepare_inputs(self) -> packed.PackedBindingRelation:
        self._charge("signature-read")
        if (
            self.signature.result != _INT
            or len(self.signature.parameters) != 1
            or self.signature.parameters[0].type != _LIST_INT
        ):
            raise ValueError("full-typed V1 requires exactly List[Int] -> Int")

        cases: list[tuple[tuple[int, ...]]] = []
        flattened: list[int] = []
        offsets: list[tuple[int, int]] = []
        physical_rows: list[tuple[object, ...]] = []
        for supplied_row in self.supplied_case_arguments:
            self._charge("case-row-read")
            row = tuple(supplied_row)
            if len(row) != 1 or type(row[0]) is not list:
                raise ValueError("case rows must contain one exact List[Int] argument")
            self._charge("case-list-read")
            supplied_items = cast(list[object], row[0])
            if len(supplied_items) > output_cells.LIST_CAPACITY:
                raise ValueError("case List[Int] exceeds the physical list capacity")
            start = len(flattened)
            items: list[int] = []
            for value in supplied_items:
                self._charge("case-list-item-read")
                if type(value) is not int:
                    raise ValueError("case List[Int] items must be exact integers")
                items.append(value)
                flattened.append(value)
            if not items:
                raise ValueError("full-typed V1 needs nonempty List[Int] cases")
            encoded = output_cells.TypedOutputCellBank.encode(
                items,
                declared_type=output_cells.OutputType.LIST_INT,
            )
            if not encoded.supported:
                raise ValueError("case List[Int] lies outside the physical output bank")
            frozen_items = tuple(items)
            cases.append((frozen_items,))
            physical_rows.append((list(frozen_items),))
            offsets.append((start, len(flattened)))
        if not cases:
            raise ValueError("full-typed V1 needs at least one case")
        self.case_arguments = tuple(cases)
        self.flattened_items = tuple(flattened)
        self.item_offsets = tuple(offsets)

        accepted_literals: list[int] = []

        def observed_literals() -> Iterable[int]:
            for value in self.supplied_literals:
                yield value
                self._charge("literal-canonicalization")
                accepted_literals.append(value)

        literal_relation = self.session.domain(_LITERAL_SLOT, observed_literals())
        if not accepted_literals:
            raise ValueError("full-typed V1 needs at least one integer literal")
        self.literals = tuple(sorted(set(accepted_literals)))

        root_shape = typed_semantic._root_context_shape(tuple(physical_rows))
        bound_shape = typed_semantic._bound_context_shape(
            self.signature,
            tuple(physical_rows),
            _LIST_INT,
        )
        partial = TypedContextCertificate(
            signature_sha256=_digest("signature", self.signature.to_record()),
            case_arguments_sha256=_digest("case-arguments", self.case_arguments),
            root_context_shape=cast(tuple[object, ...], root_shape),
            bound_context_shape=cast(tuple[object, ...], bound_shape),
            item_offsets=self.item_offsets,
            flattened_items_sha256=_digest("flattened-items", self.flattened_items),
            output_projection_sha256=(output_cells.TypedOutputCellBank.projection_sha256()),
            operator_checkpoint_sha256=self.operator.checkpoint_sha256,
            sha256="",
        )
        self.context_certificate = partial._replace(
            sha256=_digest("typed-context-certificate", _context_payload(partial))
        )
        return literal_relation

    def _register_typing(
        self,
        plan: v2._FamilyPlan,
        *,
        context_role: str,
        environment_type_labels: tuple[str, ...],
        result_type_label: str,
        root_candidate: bool,
    ) -> None:
        self._charge("typed-family-index-insert")
        typing = _FamilyTyping(
            context_role,
            environment_type_labels,
            result_type_label,
            root_candidate,
        )
        incumbent = self._family_typings.setdefault(plan.family_id, typing)
        if incumbent != typing:
            raise RuntimeError("typed family identity collision")

    def _terminal_plan(
        self,
        production: str,
        slot: packed.TypedSlot,
        *,
        context_role: str,
        environment_type_labels: tuple[str, ...],
        result_type_label: str,
    ) -> v2._FamilyPlan:
        plan = v2._FamilyPlan(
            family_id=v2._family_identity(production, ()),
            production=production,
            inside_cost=1,
            child_family_ids=(),
            child_target_slots=(),
            child_slot_indices=(),
            schema=(slot,),
            constraints=(),
        )
        self._register_typing(
            plan,
            context_role=context_role,
            environment_type_labels=environment_type_labels,
            result_type_label=result_type_label,
            root_candidate=False,
        )
        return plan

    def _predict_effect(
        self,
        label: str,
        selected: output_cells.TypedOutputEncoding,
        right: output_cells.TypedOutputEncoding | None = None,
    ) -> output_cells.TypedOutputEncoding:
        output_cells.TypedOutputCellBank.validate(selected)
        if right is not None:
            output_cells.TypedOutputCellBank.validate(right)
        key = (label, selected.sha256, None if right is None else right.sha256)
        self.session.charge_workset("prediction-cache-lookup")
        cached = self._prediction_cache.get(key)
        if cached is not None:
            self.prediction_cache_hits += 1
            return output_cells.TypedOutputCellBank.validate(cached)

        attempted_prediction = self.physical_predictions + 1
        if attempted_prediction > MAX_PHYSICAL_PREDICTIONS:
            raise v2._PhysicalWorkBoundExceeded("physical-predictions", attempted_prediction)
        reserved_races = 2 + output_cells.LIST_CAPACITY if label == "list-copy" else 2
        attempted_races = self.physical_races + reserved_races
        if attempted_races > MAX_PHYSICAL_RACES:
            raise v2._PhysicalWorkBoundExceeded("physical-races", attempted_races)
        self.physical_predictions = attempted_prediction
        self._record("prediction", _digest("prediction-key", key))
        if label == "list-copy":
            prediction = self.operator.predict(selected, _LIST_COPY)
        elif label == "less-than" and right is not None:
            prediction = self.operator.predict(selected, _LESS_THAN, scopes=(right,))
        elif label == "add" and right is not None:
            prediction = self.operator.predict(selected, _ADD, scopes=(right,))
        else:
            raise RuntimeError("full-typed physical prediction schema changed")
        if not 1 <= len(prediction.races) <= reserved_races:
            raise RuntimeError("full-typed physical race reservation drifted")
        self.physical_races += len(prediction.races)
        output = output_cells.TypedOutputCellBank.validate(prediction.output)
        self.session.charge_workset("prediction-cache-insert")
        incumbent = self._prediction_cache.setdefault(key, output)
        if incumbent != output:
            raise RuntimeError("full-typed prediction cache identity collision")
        return output

    def _less_than_vector(
        self,
        left: tuple[output_cells.TypedOutputEncoding, ...],
        right: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if len(left) != len(self.flattened_items) or len(right) != len(left):
            raise RuntimeError("bound-context less-than vector changed")
        return tuple(
            self._predict_effect("less-than", lhs, rhs)
            for lhs, rhs in zip(left, right, strict=True)
        )

    def _sum_vector(
        self,
        collections: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if len(collections) != len(self.case_arguments):
            raise RuntimeError("root collection context changed")
        outputs: list[output_cells.TypedOutputEncoding] = []
        for collection in collections:
            copied = self._predict_effect("list-copy", collection)
            values = cast(list[int], output_cells.TypedOutputCellBank.decode(copied))
            accumulator = output_cells.TypedOutputCellBank.encode(
                0, declared_type=output_cells.OutputType.INT
            )
            for value in values:
                item = output_cells.TypedOutputCellBank.encode(
                    value, declared_type=output_cells.OutputType.INT
                )
                accumulator = self._predict_effect("add", accumulator, item)
            outputs.append(accumulator)
        return tuple(outputs)

    def _count_vector(
        self,
        predicate: tuple[output_cells.TypedOutputEncoding, ...],
        collections: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if (
            len(predicate) != len(self.flattened_items)
            or len(collections) != len(self.case_arguments)
            or len(self.item_offsets) != len(collections)
        ):
            raise RuntimeError("count typed contexts changed")
        outputs: list[output_cells.TypedOutputEncoding] = []
        for collection, (start, stop) in zip(collections, self.item_offsets, strict=True):
            copied = self._predict_effect("list-copy", collection)
            values = cast(list[int], output_cells.TypedOutputCellBank.decode(copied))
            if len(values) != stop - start:
                raise RuntimeError("count collection no longer matches bound context")
            accumulator = output_cells.TypedOutputCellBank.encode(
                0, declared_type=output_cells.OutputType.INT
            )
            for event in predicate[start:stop]:
                output_cells.TypedOutputCellBank.validate(event)
                if event.output_type is not output_cells.OutputType.BOOL or not event.supported:
                    raise RuntimeError("count binder lost its supported Bool event")
                increment = output_cells.TypedOutputCellBank.encode(
                    int(cast(bool, output_cells.TypedOutputCellBank.decode(event))),
                    declared_type=output_cells.OutputType.INT,
                )
                accumulator = self._predict_effect("add", accumulator, increment)
            outputs.append(accumulator)
        return tuple(outputs)

    def _predict_vector(
        self,
        production: str,
        children: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if production == "less-than-item-literal":
            return self._less_than_vector(children[0], children[1])
        if production == "less-than-literal-item":
            return self._less_than_vector(children[0], children[1])
        if production == "predicate-binder":
            if len(children) != 1:
                raise RuntimeError("predicate binder arity changed")
            return tuple(output_cells.TypedOutputCellBank.validate(item) for item in children[0])
        if production == "sum":
            return self._sum_vector(children[0])
        if production == "count":
            return self._count_vector(children[0], children[1])
        raise RuntimeError("unknown full-typed family production")

    def _seed_families(
        self,
        literal_relation: packed.PackedBindingRelation,
    ) -> tuple[
        v2.PackedPartitionHandle,
        v2.PackedPartitionHandle,
        v2.PackedPartitionHandle,
    ]:
        collection_plan = self._terminal_plan(
            "collection-role",
            _COLLECTION_SLOT,
            context_role="root",
            environment_type_labels=("List[Int]",),
            result_type_label="List[Int]",
        )
        collection_relation = self.session.domain(_COLLECTION_SLOT, (0,))

        def collection_outputs(
            _value: int,
        ) -> tuple[output_cells.TypedOutputEncoding, ...]:
            return tuple(
                output_cells.TypedOutputCellBank.encode(
                    list(row[0]),
                    declared_type=output_cells.OutputType.LIST_INT,
                )
                for row in self.case_arguments
            )

        collection = self.workset.seed_family(
            collection_plan, collection_relation, collection_outputs
        )
        self._record("family-primary", collection.family_id)

        item_plan = self._terminal_plan(
            "item-role",
            _ITEM_SLOT,
            context_role="bound",
            environment_type_labels=("Int", "List[Int]"),
            result_type_label="Int",
        )
        item_relation = self.session.domain(_ITEM_SLOT, (0,))

        def item_outputs(
            _value: int,
        ) -> tuple[output_cells.TypedOutputEncoding, ...]:
            return tuple(
                output_cells.TypedOutputCellBank.encode(
                    item, declared_type=output_cells.OutputType.INT
                )
                for item in self.flattened_items
            )

        item = self.workset.seed_family(item_plan, item_relation, item_outputs)
        self._record("family-primary", item.family_id)

        literal_plan = self._terminal_plan(
            "literal",
            _LITERAL_SLOT,
            context_role="bound",
            environment_type_labels=("Int", "List[Int]"),
            result_type_label="Int",
        )

        def literal_outputs(
            value: int,
        ) -> tuple[output_cells.TypedOutputEncoding, ...]:
            return tuple(
                output_cells.TypedOutputCellBank.encode(
                    value, declared_type=output_cells.OutputType.INT
                )
                for _item in self.flattened_items
            )

        literal = self.workset.seed_family(literal_plan, literal_relation, literal_outputs)
        self._record("family-primary", literal.family_id)
        self.maximum_completed_ast_cost = 1
        return collection, item, literal

    def _family(
        self,
        production: str,
        children: tuple[v2.PackedPartitionHandle, ...],
        *,
        context_role: str,
        environment_type_labels: tuple[str, ...],
        result_type_label: str,
        root_candidate: bool,
    ) -> v2.PackedPartitionHandle:
        plan = self._operator_plan(production, children)
        self._register_typing(
            plan,
            context_role=context_role,
            environment_type_labels=environment_type_labels,
            result_type_label=result_type_label,
            root_candidate=root_candidate,
        )
        self._record("family-primary", plan.family_id)
        self._record("semantic-expand", plan.family_id)
        handle = self.workset.apply_family(plan, children, self._predict_vector)
        self.maximum_completed_ast_cost = max(self.maximum_completed_ast_cost, plan.inside_cost)
        return handle

    def _typed_publication(
        self,
        frozen: v3.PackedFrozenWorkset,
        families: tuple[v3.PackedFamily, ...],
        enodes: tuple[v3.PackedENode, ...],
        eclasses: tuple[v3.PackedEClass, ...],
    ) -> tuple[tuple[TypedFamilyRecord, ...], tuple[v3.PackedEClass, ...]]:
        component = self.authenticated_components
        context = self.context_certificate
        if component is None or context is None:
            raise RuntimeError("full-typed publication prerequisites are absent")
        partition_by_family = {item.family_id: item for item in frozen.partitions}
        typed_records: list[TypedFamilyRecord] = []
        family_index: dict[str, TypedFamilyRecord] = {}
        for family in families:
            self._charge("typed-family-publication")
            typing = self._family_typings.get(family.family_id)
            partition = partition_by_family.get(family.family_id)
            if typing is None or partition is None:
                raise RuntimeError("published family lost its typed context")
            partial = TypedFamilyRecord(
                family_id=family.family_id,
                production=family.production,
                inside_cost=family.inside_cost,
                context_role=typing.context_role,
                environment_type_labels=typing.environment_type_labels,
                result_type_label=typing.result_type_label,
                root_candidate=typing.root_candidate,
                partition_sha256=partition.sha256,
                sha256="",
            )
            record = partial._replace(
                sha256=_digest("typed-family", _typed_family_payload(partial))
            )
            self._charge("typed-family-public-index-insert")
            incumbent = family_index.setdefault(record.family_id, record)
            if incumbent is not record:
                raise RuntimeError("published typed family identity collision")
            typed_records.append(record)

        root_enodes: dict[str, list[v3.PackedENode]] = {}
        root_family_ids = {
            item.family_id
            for item in typed_records
            if item.root_candidate
            and item.result_type_label == "Int"
            and item.context_role == "root"
        }
        for enode in enodes:
            self._charge("typed-root-filter-read")
            if enode.family_id in root_family_ids:
                root_enodes.setdefault(v3._eclass_id(enode.physical_outputs), []).append(enode)
        full_eclass_index = {item.eclass_id: item for item in eclasses}
        roots: list[v3.PackedEClass] = []
        root_ordinal_index: dict[str, int] = {}
        for eclass_id in sorted(root_enodes):
            self._charge("typed-root-eclass-publication")
            selected = root_enodes[eclass_id]
            full = full_eclass_index[eclass_id]
            partial = v3.PackedEClass(
                eclass_id=eclass_id,
                minimum_inside_cost=min(
                    family_index[item.family_id].inside_cost for item in selected
                ),
                enode_ids=tuple(item.enode_id for item in selected),
                group_refs=tuple(item.group_ref for item in selected),
                physical_outputs=full.physical_outputs,
                sha256="",
            )
            root = partial._replace(
                sha256=_digest("typed-root-eclass", _root_eclass_payload(partial))
            )
            self._charge("typed-root-index-insert")
            ordinal = len(roots)
            incumbent = root_ordinal_index.setdefault(root.eclass_id, ordinal)
            if incumbent != ordinal:
                raise RuntimeError("typed root e-class identity collision")
            roots.append(root)
        if not roots:
            raise RuntimeError("full-typed frontier published no Int roots")

        frozen_records = tuple(typed_records)
        frozen_roots = tuple(roots)
        root_ids = tuple(item.eclass_id for item in frozen_roots)
        partial_capsule = _AuthenticatedTypedCapsule(
            owner_token=self.workset._owner_token,
            frozen_sha256=frozen.sha256,
            component_capsule_sha256=component.sha256,
            context_certificate_sha256=context.sha256,
            family_records=frozen_records,
            family_index=MappingProxyType(family_index),
            root_eclasses=frozen_roots,
            root_eclass_ids=root_ids,
            root_ordinal_index=MappingProxyType(root_ordinal_index),
            sha256="",
        )
        self.typed_capsule = partial_capsule._replace(
            sha256=_digest(
                "authenticated-typed-capsule",
                _typed_capsule_payload(partial_capsule),
            )
        )
        return frozen_records, frozen_roots

    def build(self) -> tuple[PackedFullTypedState, v3._SemanticPartitionWorkset]:
        try:
            literal_relation = self._prepare_inputs()
            collection, item, literal = self._seed_families(literal_relation)
            if self.max_ast_cost >= 2:
                self._family(
                    "sum",
                    (collection,),
                    context_role="root",
                    environment_type_labels=("List[Int]",),
                    result_type_label="Int",
                    root_candidate=True,
                )
            if self.max_ast_cost >= 6:
                item_lt_literal = self._family(
                    "less-than-item-literal",
                    (item, literal),
                    context_role="bound",
                    environment_type_labels=("Int", "List[Int]"),
                    result_type_label="Bool",
                    root_candidate=False,
                )
                literal_lt_item = self._family(
                    "less-than-literal-item",
                    (literal, item),
                    context_role="bound",
                    environment_type_labels=("Int", "List[Int]"),
                    result_type_label="Bool",
                    root_candidate=False,
                )
                for predicate in (item_lt_literal, literal_lt_item):
                    binder = self._family(
                        "predicate-binder",
                        (predicate,),
                        context_role="binder",
                        environment_type_labels=("List[Int]",),
                        result_type_label="Function[Int,Bool]",
                        root_candidate=False,
                    )
                    self._family(
                        "count",
                        (binder, collection),
                        context_role="root",
                        environment_type_labels=("List[Int]",),
                        result_type_label="Int",
                        root_candidate=True,
                    )

            frozen = self.workset.publish()
            families, enodes, eclasses = self._components(frozen)
            typed_families, root_eclasses = self._typed_publication(
                frozen, families, enodes, eclasses
            )
            component = self.authenticated_components
            typed_capsule = self.typed_capsule
            context = self.context_certificate
            if component is None or typed_capsule is None or context is None:
                raise RuntimeError("full-typed frontier publication is incomplete")
            self._charge("frontier-seal")
            work = self.work_audit()
            payload = {
                "status": FRONTIER_READY,
                "signature": self.signature.to_record(),
                "case_arguments": self.case_arguments,
                "literals": self.literals,
                "requested_max_ast_cost": self.max_ast_cost,
                "maximum_completed_ast_cost": self.maximum_completed_ast_cost,
                "context_certificate_sha256": context.sha256,
                "operator_checkpoint_sha256": self.operator.checkpoint_sha256,
                "frozen_workset_sha256": frozen.sha256,
                "component_capsule_sha256": component.sha256,
                "typed_capsule_sha256": typed_capsule.sha256,
                "root_eclass_ids": typed_capsule.root_eclass_ids,
                "build_work_sha256": work.sha256,
            }
            state = PackedFullTypedState(
                status=FRONTIER_READY,
                signature=self.signature,
                case_arguments=self.case_arguments,
                literals=self.literals,
                requested_max_ast_cost=self.max_ast_cost,
                maximum_completed_ast_cost=self.maximum_completed_ast_cost,
                context_certificate=context,
                operator_checkpoint_sha256=self.operator.checkpoint_sha256,
                frozen_workset=frozen,
                families=families,
                enodes=enodes,
                eclasses=eclasses,
                typed_families=typed_families,
                root_eclasses=root_eclasses,
                root_eclass_ids=typed_capsule.root_eclass_ids,
                component_capsule_sha256=component.sha256,
                typed_capsule_sha256=typed_capsule.sha256,
                build_work_sha256=work.sha256,
                sha256=_digest("frontier", payload),
            )
            return state, self.workset
        except packed.PackedRelationWorkLimitExceeded as error:
            self.bound_name = "recipe-relation-work"
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset
        except v2._PhysicalWorkBoundExceeded as error:
            self.bound_name = error.name
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset

    def _stopped_state(self) -> PackedFullTypedState:
        work = self.work_audit()
        payload = {
            "status": STOP_WORK_BOUND,
            "signature": self.signature.to_record(),
            "case_arguments": self.case_arguments,
            "literals": self.literals,
            "requested_max_ast_cost": self.max_ast_cost,
            "maximum_completed_ast_cost": self.maximum_completed_ast_cost,
            "context_certificate_sha256": (
                None if self.context_certificate is None else self.context_certificate.sha256
            ),
            "operator_checkpoint_sha256": self.operator.checkpoint_sha256,
            "frozen_workset_sha256": None,
            "component_capsule_sha256": None,
            "typed_capsule_sha256": None,
            "root_eclass_ids": (),
            "build_work_sha256": work.sha256,
        }
        return PackedFullTypedState(
            status=STOP_WORK_BOUND,
            signature=self.signature,
            case_arguments=self.case_arguments,
            literals=self.literals,
            requested_max_ast_cost=self.max_ast_cost,
            maximum_completed_ast_cost=self.maximum_completed_ast_cost,
            context_certificate=self.context_certificate,
            operator_checkpoint_sha256=self.operator.checkpoint_sha256,
            frozen_workset=None,
            families=(),
            enodes=(),
            eclasses=(),
            typed_families=(),
            root_eclasses=(),
            root_eclass_ids=(),
            component_capsule_sha256=None,
            typed_capsule_sha256=None,
            build_work_sha256=work.sha256,
            sha256=_digest("frontier", payload),
        )

    def work_audit(self) -> PackedFullTypedWorkAudit:
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


class PackedFullTypedFrontier:
    """Deep desired-free build/select module for the bounded typed slice."""

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
        "_typed_capsule",
        "_typed_family_index",
        "_typed_root_ordinal_index",
        "_workset",
    )

    def __init__(
        self,
        state: PackedFullTypedState,
        workset: v3._SemanticPartitionWorkset,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
        builder: _Builder,
    ) -> None:
        self._state = state
        self._workset = workset
        self._operator = operator
        self._authenticated_components = builder.authenticated_components
        self._typed_capsule = builder.typed_capsule
        self._typed_family_index = (
            None if builder.typed_capsule is None else builder.typed_capsule.family_index
        )
        self._typed_root_ordinal_index = (
            None if builder.typed_capsule is None else builder.typed_capsule.root_ordinal_index
        )
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
        signature: TypedSignature,
        case_arguments: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor | None = None,
    ) -> PackedFullTypedFrontier:
        if not isinstance(signature, TypedSignature):
            raise TypeError("full-typed frontier requires a TypedSignature")
        if isinstance(case_arguments, (str, bytes)):
            raise TypeError("case_arguments must be an iterable of typed rows")
        if isinstance(literals, (str, bytes)):
            raise TypeError("literals must be an iterable of exact integers")
        if type(max_ast_cost) is not int or max_ast_cost not in {2, 6}:
            raise ValueError("full-typed V1 max_ast_cost must be 2 or 6")
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
            signature,
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
    def state(self) -> PackedFullTypedState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def families(self) -> tuple[v3.PackedFamily, ...]:
        return self._state.families

    @property
    def enodes(self) -> tuple[v3.PackedENode, ...]:
        return self._state.enodes

    @property
    def eclasses(self) -> tuple[v3.PackedEClass, ...]:
        return self._state.eclasses

    @property
    def root_eclasses(self) -> tuple[v3.PackedEClass, ...]:
        return self._state.root_eclasses

    @property
    def work(self) -> PackedFullTypedWorkAudit:
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
            "signature": self._state.signature.to_record(),
            "case_arguments": self._state.case_arguments,
            "literals": self._state.literals,
            "requested_max_ast_cost": self._state.requested_max_ast_cost,
            "maximum_completed_ast_cost": self._state.maximum_completed_ast_cost,
            "context_certificate_sha256": (
                None
                if self._state.context_certificate is None
                else self._state.context_certificate.sha256
            ),
            "operator_checkpoint_sha256": self._state.operator_checkpoint_sha256,
            "frozen_workset_sha256": (
                None if self._state.frozen_workset is None else self._state.frozen_workset.sha256
            ),
            "component_capsule_sha256": self._state.component_capsule_sha256,
            "typed_capsule_sha256": self._state.typed_capsule_sha256,
            "root_eclass_ids": self._state.root_eclass_ids,
            "build_work_sha256": self._state.build_work_sha256,
        }

    def _authenticate(
        self,
    ) -> tuple[
        v3._AuthenticatedFrozen,
        v3._AuthenticatedComponents,
        _AuthenticatedTypedCapsule,
    ]:
        state = self._state
        frozen = state.frozen_workset
        context = state.context_certificate
        component = self._authenticated_components
        typed_capsule = self._typed_capsule
        if state.status != FRONTIER_READY or frozen is None or context is None:
            raise RuntimeError("only a complete full-typed frontier can be selected")
        if self._operator.live_checkpoint_sha256 != state.operator_checkpoint_sha256:
            raise RuntimeError("full-typed operator checkpoint changed")
        authenticated = self._workset.require_retained(frozen)
        self._workset.session.charge_frontier("authenticate-component-capsule")
        if (
            type(component) is not v3._AuthenticatedComponents
            or component is not self._workset._component_capsule
            or component.owner_token is not self._workset._owner_token
            or component.frozen_sha256 != frozen.sha256
            or component.families is not state.families
            or component.enodes is not state.enodes
            or component.eclasses is not state.eclasses
            or component.sha256 != state.component_capsule_sha256
        ):
            raise RuntimeError("retained packed component authentication failed")
        self._workset.session.charge_frontier("authenticate-typed-capsule")
        if (
            type(typed_capsule) is not _AuthenticatedTypedCapsule
            or typed_capsule.owner_token is not self._workset._owner_token
            or typed_capsule.frozen_sha256 != frozen.sha256
            or typed_capsule.component_capsule_sha256 != component.sha256
            or typed_capsule.context_certificate_sha256 != context.sha256
            or typed_capsule.family_records is not state.typed_families
            or typed_capsule.root_eclasses is not state.root_eclasses
            or typed_capsule.root_eclass_ids is not state.root_eclass_ids
            or typed_capsule.family_index is not self._typed_family_index
            or typed_capsule.root_ordinal_index is not self._typed_root_ordinal_index
            or typed_capsule.sha256 != state.typed_capsule_sha256
            or type(typed_capsule.family_index) is not _MAPPING_PROXY_TYPE
            or type(typed_capsule.root_ordinal_index) is not _MAPPING_PROXY_TYPE
            or typed_capsule.sha256
            != _digest(
                "authenticated-typed-capsule",
                _typed_capsule_payload(typed_capsule),
            )
        ):
            raise RuntimeError("retained typed context authentication failed")
        if (
            context.signature_sha256 != _digest("signature", state.signature.to_record())
            or context.case_arguments_sha256 != _digest("case-arguments", state.case_arguments)
            or context.sha256 != _digest("typed-context-certificate", _context_payload(context))
            or state.sha256 != _digest("frontier", self._state_payload())
        ):
            raise RuntimeError("full-typed frontier authentication failed")
        return authenticated, component, typed_capsule

    def _render_witness(
        self,
        witness: packed.ProofWitness,
        component: v3._AuthenticatedComponents,
    ) -> PackedFullTypedWitness:
        self._workset.session.charge_frontier("materialize-witness")

        def render(family_id: str, binding: tuple[int, ...]) -> tuple[str, ...]:
            self._workset.session.charge_frontier("materialize-family-lookup")
            family = component.family_index.get(family_id)
            if type(family) is not v3.PackedFamily or family.family_id != family_id:
                raise RuntimeError("typed proof family index changed")
            if len(binding) != len(family.schema):
                raise RuntimeError("typed proof binding arity changed")
            if family.production in {"collection-role", "item-role"}:
                if binding != (0,):
                    raise RuntimeError("typed role binding changed")
                return ("variable-0",)
            if family.production == "literal":
                return tp._factor_literal_tokens(binding[0])
            children = tuple(
                render(
                    child_id,
                    tuple(binding[index] for index in indices),
                )
                for child_id, indices in zip(
                    family.child_family_ids,
                    family.child_slot_indices,
                    strict=True,
                )
            )
            if family.production in {
                "less-than-item-literal",
                "less-than-literal-item",
            }:
                return ("less-than", *children[0], *children[1])
            if family.production == "predicate-binder":
                return ("bind", *children[0])
            if family.production == "count":
                return ("count", *children[0], *children[1])
            if family.production == "sum":
                return ("sum", *children[0])
            raise RuntimeError("typed proof production changed")

        tokens = render(witness.family_id, witness.binding)
        source = tp.compile(tp.Program(tokens, self._state.signature)).source
        payload = {
            "family_id": witness.family_id,
            "binding": witness.binding,
            "source": source,
        }
        return PackedFullTypedWitness(
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
        witnesses: tuple[PackedFullTypedWitness, ...] = (),
        selected_proof_forest_sha256: str | None = None,
        race: semantic_race.TypedSemanticVectorRaceAudit | None = None,
        checkpoint_before: str | None = None,
    ) -> PackedFullTypedSelection:
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
        return PackedFullTypedSelection(
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

    def select(self, desired_outputs: Sequence[int]) -> PackedFullTypedSelection:
        if self._state.status != FRONTIER_READY:
            if self._state.sha256 != _digest("frontier", self._state_payload()):
                raise RuntimeError("stopped full-typed frontier authentication failed")
            return self._selection(STOP_WORK_BOUND)
        try:
            authenticated, component, typed_capsule = self._authenticate()
            if isinstance(desired_outputs, (str, bytes)):
                raise TypeError("desired_outputs must be exact integers")
            desired_values: list[int] = []
            for value in desired_outputs:
                self._workset.session.charge_frontier("desired-value-read")
                if type(value) is not int:
                    raise ValueError("desired outputs must align with typed cases")
                desired_values.append(value)
            if len(desired_values) != len(self._state.case_arguments):
                raise ValueError("desired outputs must align with typed cases")
            desired = tuple(
                output_cells.TypedOutputCellBank.encode(
                    value, declared_type=output_cells.OutputType.INT
                )
                for value in desired_values
            )
            before = self._operator.live_checkpoint_sha256
            attempted_races = self._physical_races + 1
            if attempted_races > MAX_PHYSICAL_RACES:
                raise v2._PhysicalWorkBoundExceeded("physical-races", attempted_races)
            self._physical_races = attempted_races
            race = semantic_race.select_exact_semantic_vector(
                typed_capsule.root_eclass_ids,
                tuple(item.physical_outputs for item in typed_capsule.root_eclasses),
                desired,
            )
            if self._operator.live_checkpoint_sha256 != before:
                raise RuntimeError("operator checkpoint changed during typed selection")
            if race.status == semantic_race.STOP_NO_EXACT:
                return self._selection(STOP_NO_EXACT, race=race, checkpoint_before=before)
            if race.status != semantic_race.SELECTED or race.winner_key is None:
                raise RuntimeError("typed root e-class bank returned a physical tie")
            self._workset.session.charge_frontier("winner-eclass-index-lookup")
            root_ordinal_index = cast(Mapping[str, int], typed_capsule.root_ordinal_index)
            ordinal = root_ordinal_index.get(race.winner_key)
            if type(ordinal) is not int or not 0 <= ordinal < len(typed_capsule.root_eclasses):
                raise RuntimeError("typed physical winner references an unknown e-class")
            eclass = typed_capsule.root_eclasses[ordinal]
            if (
                type(eclass) is not v3.PackedEClass
                or typed_capsule.root_eclass_ids[ordinal] != race.winner_key
                or eclass.eclass_id != race.winner_key
            ):
                raise RuntimeError("typed physical winner identity changed")
            frozen = self._state.frozen_workset
            if frozen is None:
                raise RuntimeError("typed selected frontier lost its publication")
            forest = self._workset.materialize_selected_forest(
                frozen,
                authenticated,
                eclass.group_refs,
            )
            proof_count = forest.exact_logical_recipe_count
            if proof_count < 1 or not forest.first_two_witnesses:
                raise RuntimeError("typed selected e-class has no proof")
            witnesses = tuple(
                self._render_witness(item, component) for item in forest.first_two_witnesses[:2]
            )
            self._concrete_proof_witnesses = len(forest.first_two_witnesses)
            status = SELECTED if proof_count == 1 else STOP_AMBIGUOUS
            return self._selection(
                status,
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


__all__ = [
    "FRONTIER_READY",
    "MAX_PHYSICAL_PREDICTIONS",
    "MAX_PHYSICAL_RACES",
    "MAX_RECIPE_RELATION_WORK",
    "SELECTED",
    "STOP_AMBIGUOUS",
    "STOP_NO_EXACT",
    "STOP_WORK_BOUND",
    "PackedFullTypedFrontier",
    "PackedFullTypedSelection",
    "PackedFullTypedState",
    "PackedFullTypedWitness",
    "PackedFullTypedWorkAudit",
    "TypedContextCertificate",
    "TypedFamilyRecord",
]
