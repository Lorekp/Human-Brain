"""Desired-free packed generic typed TRAIN breadth, cross-type successor.

V2 remains the frozen scalar, List-to-List, and String surface.  This separately
named successor adds one signature-derived ``List[Int] -> Int`` route.  Its
generic grammar composes the banked collection, item, literal, sum, count, and
predicate-binder semantics with two bounded compositions:

* maximum then minimum, a value binder, map, and sum;
* add, either less-than orientation, a predicate binder, and count.

All operator families are applied to packed semantic relations before their
physical output tuples are published.  V4 retains content-addressed lazy proof
plans and materializes only the physically selected root proof.  Build observes
only the typed signature, caller cases/literals, and maximum cost; desired
outputs remain exclusive to :meth:`select`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from hashlib import sha256
from typing import NamedTuple, cast

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_full_typed_v1 as full_typed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as packed_v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as packed_v3
from experiments.r3_pilot import typed_semantic_packed_frontier_v4 as packed_v4
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v2 as breadth_v2
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramType, TypedSignature

FRONTIER_READY = "PACKED_GENERIC_TYPED_TRAIN_BREADTH_V3_READY"
STOP_WORK_BOUND = "STOP_PACKED_GENERIC_TYPED_TRAIN_BREADTH_V3_WORK_BOUND"
SELECTED = "PACKED_GENERIC_TYPED_TRAIN_BREADTH_V3_SELECTED"
STOP_NO_EXACT = "STOP_PACKED_GENERIC_TYPED_TRAIN_BREADTH_V3_NO_EXACT"
STOP_AMBIGUOUS = "STOP_PACKED_GENERIC_TYPED_TRAIN_BREADTH_V3_AMBIGUOUS"

MAX_RECIPE_RELATION_WORK = packed_v4.MAX_RECIPE_RELATION_WORK
MAX_PHYSICAL_PREDICTIONS = packed_v4.MAX_PHYSICAL_PREDICTIONS
MAX_PHYSICAL_RACES = packed_v4.MAX_PHYSICAL_RACES

_NAMESPACE = b"packed-generic-typed-train-breadth-v3\0"
_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_MINIMUM = tp.FactorAction((), "APPLY2", "minimum", "SELECTED", "SCOPE0", None)
_MAXIMUM = tp.FactorAction((), "APPLY2", "maximum", "SELECTED", "SCOPE0", None)


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


class _PackedListIntWitness(NamedTuple):
    family_id: str
    binding: tuple[int, ...]
    tokens: tuple[str, ...]
    source: str
    sha256: str


class PackedGenericTypedWitness(NamedTuple):
    family_id: str
    binding: tuple[int, ...]
    program: tp.Program
    source: str
    sha256: str


class PackedGenericTypedState(NamedTuple):
    status: str
    signature: TypedSignature
    case_arguments: tuple[tuple[object, ...], ...]
    literals: tuple[int, ...]
    requested_max_ast_cost: int
    maximum_completed_ast_cost: int
    operator_checkpoint_sha256: str
    root_eclass_ids: tuple[str, ...]
    route_frontier_sha256: str
    route_build_work_sha256: str
    sha256: str


_WorkAudit = packed_v3.PackedFrontierWorkAudit | full_typed.PackedFullTypedWorkAudit


class PackedGenericTypedSelection(NamedTuple):
    status: str
    frontier_sha256: str
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[PackedGenericTypedWitness, ...]
    selected_proof_forest_sha256: str | None
    race: semantic_race.TypedSemanticVectorRaceAudit | None
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    work: _WorkAudit
    route_selection_sha256: str
    sha256: str


def _state_payload(state: PackedGenericTypedState) -> dict[str, object]:
    return {
        "status": state.status,
        "signature": state.signature.to_record(),
        "case_arguments": state.case_arguments,
        "literals": state.literals,
        "requested_max_ast_cost": state.requested_max_ast_cost,
        "maximum_completed_ast_cost": state.maximum_completed_ast_cost,
        "operator_checkpoint_sha256": state.operator_checkpoint_sha256,
        "root_eclass_ids": state.root_eclass_ids,
        "route_frontier_sha256": state.route_frontier_sha256,
        "route_build_work_sha256": state.route_build_work_sha256,
    }


class _ListIntBuilder(full_typed._Builder):
    """One V4 workset for generic List[Int]-to-Int composition."""

    def __init__(
        self,
        signature: TypedSignature,
        supplied_case_arguments: Iterable[tuple[object, ...]],
        supplied_literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
    ) -> None:
        super().__init__(
            signature,
            supplied_case_arguments,
            supplied_literals,
            max_ast_cost,
            operator,
        )
        # The parent has consumed no work.  V4 owns the sole ledger before the
        # first signature, case, or literal event is observed.
        self.session = packed_v2._FrontierSession(work_limit=MAX_RECIPE_RELATION_WORK)
        self.workset = packed_v4._SemanticPartitionWorkset(self.session)
        self.authenticated_components = None

    def _prepare_inputs(self) -> packed.PackedBindingRelation:
        self._charge("signature-read")
        if (
            self.signature.result != _INT
            or len(self.signature.parameters) != 1
            or self.signature.parameters[0].type != _LIST_INT
        ):
            raise ValueError("V3 cross-type route requires exactly List[Int] -> Int")

        cases: list[tuple[tuple[int, ...]]] = []
        flattened: list[int] = []
        offsets: list[tuple[int, int]] = []
        physical_rows: list[tuple[object, ...]] = []
        for supplied_row in self.supplied_case_arguments:
            self._charge("case-row-read")
            row = tuple(supplied_row)
            if len(row) != 1 or type(row[0]) is not list:
                raise ValueError("V3 cross-type rows must contain one exact List[Int]")
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
            raise ValueError("V3 cross-type route needs at least one case")
        self.case_arguments = tuple(cases)
        self.flattened_items = tuple(flattened)
        self.item_offsets = tuple(offsets)

        accepted_literals: list[int] = []

        def observed_literals() -> Iterable[int]:
            for value in self.supplied_literals:
                if type(value) is not int:
                    raise ValueError("V3 cross-type literals must be exact integers")
                yield value
                self._charge("literal-canonicalization")
                accepted_literals.append(value)

        literal_relation = self.session.domain(
            full_typed._LITERAL_SLOT,
            observed_literals(),
        )
        if not accepted_literals:
            raise ValueError("V3 cross-type route needs at least one integer literal")
        self.literals = tuple(sorted(set(accepted_literals)))

        root_shape = full_typed.typed_semantic._root_context_shape(tuple(physical_rows))
        bound_shape = full_typed.typed_semantic._bound_context_shape(
            self.signature,
            tuple(physical_rows),
            _LIST_INT,
        )
        partial = full_typed.TypedContextCertificate(
            signature_sha256=full_typed._digest(
                "signature",
                self.signature.to_record(),
            ),
            case_arguments_sha256=full_typed._digest(
                "case-arguments",
                self.case_arguments,
            ),
            root_context_shape=cast(tuple[object, ...], root_shape),
            bound_context_shape=cast(tuple[object, ...], bound_shape),
            item_offsets=self.item_offsets,
            flattened_items_sha256=full_typed._digest(
                "flattened-items",
                self.flattened_items,
            ),
            output_projection_sha256=output_cells.TypedOutputCellBank.projection_sha256(),
            operator_checkpoint_sha256=self.operator.checkpoint_sha256,
            sha256="",
        )
        self.context_certificate = partial._replace(
            sha256=full_typed._digest(
                "typed-context-certificate",
                full_typed._context_payload(partial),
            )
        )
        return literal_relation

    def _predict_effect(
        self,
        label: str,
        selected: output_cells.TypedOutputEncoding,
        right: output_cells.TypedOutputEncoding | None = None,
    ) -> output_cells.TypedOutputEncoding:
        if label not in {"minimum", "maximum"}:
            return super()._predict_effect(label, selected, right)
        if right is None:
            raise RuntimeError("extremum prediction lost its right operand")
        output_cells.TypedOutputCellBank.validate(selected)
        output_cells.TypedOutputCellBank.validate(right)
        key = (label, selected.sha256, right.sha256)
        self.session.charge_workset("prediction-cache-lookup")
        cached = self._prediction_cache.get(key)
        if cached is not None:
            self.prediction_cache_hits += 1
            return output_cells.TypedOutputCellBank.validate(cached)
        attempted_prediction = self.physical_predictions + 1
        if attempted_prediction > MAX_PHYSICAL_PREDICTIONS:
            raise packed_v2._PhysicalWorkBoundExceeded(
                "physical-predictions",
                attempted_prediction,
            )
        attempted_races = self.physical_races + 4
        if attempted_races > MAX_PHYSICAL_RACES:
            raise packed_v2._PhysicalWorkBoundExceeded(
                "physical-races",
                attempted_races,
            )
        self.physical_predictions = attempted_prediction
        self._record("prediction", _digest("cross-type-prediction-key", key))
        action = _MINIMUM if label == "minimum" else _MAXIMUM
        prediction = self.operator.predict(selected, action, scopes=(right,))
        if len(prediction.races) != 4:
            raise RuntimeError("extremum physical race reservation drifted")
        self.physical_races = attempted_races
        output = output_cells.TypedOutputCellBank.validate(prediction.output)
        self.session.charge_workset("prediction-cache-insert")
        incumbent = self._prediction_cache.setdefault(key, output)
        if incumbent != output:
            raise RuntimeError("V3 cross-type prediction cache identity collision")
        return output

    def _binary_bound_vector(
        self,
        label: str,
        left: tuple[output_cells.TypedOutputEncoding, ...],
        right: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if len(left) != len(self.flattened_items) or len(right) != len(left):
            raise RuntimeError("cross-type bound context changed")
        return tuple(
            self._predict_effect(label, lhs, rhs) for lhs, rhs in zip(left, right, strict=True)
        )

    def _map_vector(
        self,
        body: tuple[output_cells.TypedOutputEncoding, ...],
        collections: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if (
            len(body) != len(self.flattened_items)
            or len(collections) != len(self.case_arguments)
            or len(self.item_offsets) != len(collections)
        ):
            raise RuntimeError("map typed contexts changed")
        outputs: list[output_cells.TypedOutputEncoding] = []
        for collection, (start, stop) in zip(
            collections,
            self.item_offsets,
            strict=True,
        ):
            copied = self._predict_effect("list-copy", collection)
            copied_values = cast(
                list[int],
                output_cells.TypedOutputCellBank.decode(copied),
            )
            if len(copied_values) != stop - start:
                raise RuntimeError("map collection no longer matches its bound context")
            mapped: list[int] = []
            for encoded in body[start:stop]:
                value = output_cells.TypedOutputCellBank.validate(encoded)
                if value.output_type is not output_cells.OutputType.INT or not value.supported:
                    raise RuntimeError("map body lost its supported Int event")
                mapped.append(cast(int, output_cells.TypedOutputCellBank.decode(value)))
            output = output_cells.TypedOutputCellBank.encode(
                mapped,
                declared_type=output_cells.OutputType.LIST_INT,
            )
            outputs.append(output_cells.TypedOutputCellBank.validate(output))
        return tuple(outputs)

    def _predict_vector(
        self,
        production: str,
        children: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if production in {"add", "minimum", "maximum"}:
            return self._binary_bound_vector(production, children[0], children[1])
        if production in {
            "less-than-added-literal",
            "less-than-literal-added",
        }:
            return self._less_than_vector(children[0], children[1])
        if production == "value-binder":
            if len(children) != 1:
                raise RuntimeError("value binder arity changed")
            return tuple(output_cells.TypedOutputCellBank.validate(item) for item in children[0])
        if production == "map":
            return self._map_vector(children[0], children[1])
        return super()._predict_vector(production, children)

    def _family(
        self,
        production: str,
        children: tuple[packed_v2.PackedPartitionHandle, ...],
        *,
        context_role: str,
        environment_type_labels: tuple[str, ...],
        result_type_label: str,
        root_candidate: bool,
        constraint_kind: str | None = None,
    ) -> packed_v2.PackedPartitionHandle:
        plan = self._operator_plan(
            production,
            children,
            constraint_kind=constraint_kind,
        )
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
        if plan.normalization_id is not None:
            self.normalization_count += 1
        self.maximum_completed_ast_cost = max(
            self.maximum_completed_ast_cost,
            plan.inside_cost,
        )
        return handle

    def _predicate_count(
        self,
        predicate: packed_v2.PackedPartitionHandle,
        collection: packed_v2.PackedPartitionHandle,
    ) -> None:
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

    def _extend_before_publish(
        self,
        collection: packed_v2.PackedPartitionHandle,
        item: packed_v2.PackedPartitionHandle,
        literal: packed_v2.PackedPartitionHandle,
    ) -> None:
        """Allow a separately named self-posed roster to grow before sealing."""

        del collection, item, literal

    def build(
        self,
    ) -> tuple[full_typed.PackedFullTypedState, packed_v4._SemanticPartitionWorkset]:
        try:
            literal_relation = self._prepare_inputs()
            collection, item, literal = self._seed_families(literal_relation)
            self._family(
                "sum",
                (collection,),
                context_role="root",
                environment_type_labels=("List[Int]",),
                result_type_label="Int",
                root_candidate=True,
            )

            for children in ((item, literal), (literal, item)):
                predicate = self._family(
                    "less-than-item-literal" if children[0] is item else "less-than-literal-item",
                    children,
                    context_role="bound",
                    environment_type_labels=("Int", "List[Int]"),
                    result_type_label="Bool",
                    root_candidate=False,
                )
                self._predicate_count(predicate, collection)

            added = self._family(
                "add",
                (item, literal),
                context_role="bound",
                environment_type_labels=("Int", "List[Int]"),
                result_type_label="Int",
                root_candidate=False,
                constraint_kind=("nonzero-right" if 0 in self.literals else None),
            )
            for production, children in (
                ("less-than-added-literal", (added, literal)),
                ("less-than-literal-added", (literal, added)),
            ):
                predicate = self._family(
                    production,
                    children,
                    context_role="bound",
                    environment_type_labels=("Int", "List[Int]"),
                    result_type_label="Bool",
                    root_candidate=False,
                )
                self._predicate_count(predicate, collection)

            if self.max_ast_cost >= 9:
                lower_bounded = self._family(
                    "maximum",
                    (item, literal),
                    context_role="bound",
                    environment_type_labels=("Int", "List[Int]"),
                    result_type_label="Int",
                    root_candidate=False,
                )
                bounded = self._family(
                    "minimum",
                    (lower_bounded, literal),
                    context_role="bound",
                    environment_type_labels=("Int", "List[Int]"),
                    result_type_label="Int",
                    root_candidate=False,
                    constraint_kind="ordered-bounds",
                )
                value_binder = self._family(
                    "value-binder",
                    (bounded,),
                    context_role="binder",
                    environment_type_labels=("List[Int]",),
                    result_type_label="Function[Int,Int]",
                    root_candidate=False,
                )
                mapped = self._family(
                    "map",
                    (value_binder, collection),
                    context_role="root",
                    environment_type_labels=("List[Int]",),
                    result_type_label="List[Int]",
                    root_candidate=False,
                )
                self._family(
                    "sum",
                    (mapped,),
                    context_role="root",
                    environment_type_labels=("List[Int]",),
                    result_type_label="Int",
                    root_candidate=True,
                )

            self._extend_before_publish(collection, item, literal)

            frozen = self.workset.publish()
            families, enodes, eclasses = self._components(frozen)
            typed_families, root_eclasses = self._typed_publication(
                frozen,
                families,
                enodes,
                eclasses,
            )
            component = self.authenticated_components
            typed_capsule = self.typed_capsule
            context = self.context_certificate
            if component is None or typed_capsule is None or context is None:
                raise RuntimeError("V3 cross-type publication is incomplete")
            self._charge("frontier-seal")
            work = self.work_audit()
            payload = {
                "status": full_typed.FRONTIER_READY,
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
            state = full_typed.PackedFullTypedState(
                status=full_typed.FRONTIER_READY,
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
                sha256=full_typed._digest("frontier", payload),
            )
            return state, self.workset
        except packed.PackedRelationWorkLimitExceeded as error:
            self.bound_name = "recipe-relation-work"
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset
        except packed_v2._PhysicalWorkBoundExceeded as error:
            self.bound_name = error.name
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset


class _PackedListIntFrontier(full_typed.PackedFullTypedFrontier):
    """Private V4 selected-proof frontier for the cross-type route."""

    @classmethod
    def build(
        cls,
        *,
        signature: TypedSignature,
        case_arguments: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor | None = None,
    ) -> _PackedListIntFrontier:
        if not isinstance(signature, TypedSignature):
            raise TypeError("V3 cross-type frontier requires a TypedSignature")
        if isinstance(case_arguments, (str, bytes, bytearray)):
            raise TypeError("case_arguments must be an iterable of typed rows")
        if isinstance(literals, (str, bytes, bytearray)):
            raise TypeError("literals must be an iterable of exact integers")
        if type(max_ast_cost) is not int or max_ast_cost not in {8, 9}:
            raise ValueError("V3 List[Int] -> Int max_cost must be 8 or 9")
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
            raise RuntimeError("typed operator checkpoint changed before V3 build")
        builder = _ListIntBuilder(
            signature,
            case_arguments,
            literals,
            max_ast_cost,
            frozen_operator,
        )
        state, workset = builder.build()
        if frozen_operator.live_checkpoint_sha256 != before:
            raise RuntimeError("typed operator checkpoint changed during V3 build")
        return cls(state, workset, frozen_operator, builder)

    def _render_witness(
        self,
        witness: packed.ProofWitness,
        component: packed_v3._AuthenticatedComponents,
    ) -> _PackedListIntWitness:
        self._workset.session.charge_frontier("materialize-witness")

        def render(family_id: str, binding: tuple[int, ...]) -> tuple[str, ...]:
            self._workset.session.charge_frontier("materialize-family-lookup")
            family = component.family_index.get(family_id)
            if type(family) is not packed_v3.PackedFamily or family.family_id != family_id:
                raise RuntimeError("V3 cross-type proof family index changed")
            if len(binding) != len(family.schema):
                raise RuntimeError("V3 cross-type proof binding arity changed")
            if family.production in {"collection-role", "item-role"}:
                if binding != (0,):
                    raise RuntimeError("V3 cross-type role binding changed")
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
            binary = {
                "add": "add",
                "minimum": "minimum",
                "maximum": "maximum",
                "less-than-item-literal": "less-than",
                "less-than-literal-item": "less-than",
                "less-than-added-literal": "less-than",
                "less-than-literal-added": "less-than",
            }
            production = binary.get(family.production)
            if production is not None:
                return (production, *children[0], *children[1])
            if family.production == "absolute":
                return ("absolute", *children[0])
            if family.production in {"predicate-binder", "value-binder"}:
                return ("bind", *children[0])
            if family.production == "map":
                return ("map", *children[0], *children[1])
            if family.production == "count":
                return ("count", *children[0], *children[1])
            if family.production == "sum":
                return ("sum", *children[0])
            raise RuntimeError("V3 cross-type proof production changed")

        tokens = render(witness.family_id, witness.binding)
        source = tp.compile(tp.Program(tokens, self._state.signature)).source
        payload = {
            "family_id": witness.family_id,
            "binding": witness.binding,
            "tokens": tokens,
            "source": source,
        }
        return _PackedListIntWitness(
            family_id=witness.family_id,
            binding=witness.binding,
            tokens=tokens,
            source=source,
            sha256=_digest("cross-type-witness", payload),
        )


_Route = breadth_v2.PackedGenericTypedTrainBreadthV2 | _PackedListIntFrontier


def _route_owner_token(route: _Route) -> object:
    if isinstance(route, breadth_v2.PackedGenericTypedTrainBreadthV2):
        return route._packed._workset._owner_token
    return route._workset._owner_token


def _route_build_work_sha256(route: _Route) -> str:
    if isinstance(route, breadth_v2.PackedGenericTypedTrainBreadthV2):
        return route.state.packed_build_work_sha256
    return route.state.build_work_sha256


class _ConstructionReceipt(NamedTuple):
    authority: object
    owner_token: object
    route: _Route
    state: PackedGenericTypedState
    signature: TypedSignature
    sha256: str


def _receipt_payload(receipt: _ConstructionReceipt) -> dict[str, object]:
    return {
        "route_frontier_sha256": receipt.route.sha256,
        "state_sha256": receipt.state.sha256,
        "signature": receipt.signature.to_record(),
    }


def _make_receipt_authority() -> tuple[
    Callable[..., _ConstructionReceipt],
    Callable[[_ConstructionReceipt], bool],
]:
    authority = object()
    issued: dict[int, _ConstructionReceipt] = {}

    def issue(
        *,
        route: _Route,
        state: PackedGenericTypedState,
        signature: TypedSignature,
    ) -> _ConstructionReceipt:
        if (
            type(state) is not PackedGenericTypedState
            or state.signature is not signature
            or state.sha256 != _digest("typed-frontier", _state_payload(state))
            or not isinstance(
                route,
                (breadth_v2.PackedGenericTypedTrainBreadthV2, _PackedListIntFrontier),
            )
            or state.route_frontier_sha256 != route.sha256
            or state.root_eclass_ids != route.state.root_eclass_ids
        ):
            raise RuntimeError("V3 construction receipt inputs diverged")
        partial = _ConstructionReceipt(
            authority=authority,
            owner_token=_route_owner_token(route),
            route=route,
            state=state,
            signature=signature,
            sha256="",
        )
        receipt = partial._replace(
            sha256=_digest("construction-receipt", _receipt_payload(partial))
        )
        issued[id(receipt)] = receipt
        return receipt

    def authenticate(receipt: _ConstructionReceipt) -> bool:
        return bool(
            type(receipt) is _ConstructionReceipt
            and issued.get(id(receipt)) is receipt
            and receipt.authority is authority
            and isinstance(
                receipt.route,
                (breadth_v2.PackedGenericTypedTrainBreadthV2, _PackedListIntFrontier),
            )
            and receipt.owner_token is _route_owner_token(receipt.route)
            and receipt.state.signature is receipt.signature
            and receipt.sha256
            == _digest(
                "construction-receipt",
                _receipt_payload(receipt),
            )
        )

    return issue, authenticate


_issue_construction_receipt, _authenticate_construction_receipt = _make_receipt_authority()


def _status_from_route(status: str) -> str:
    statuses = {
        breadth_v2.FRONTIER_READY: FRONTIER_READY,
        breadth_v2.STOP_WORK_BOUND: STOP_WORK_BOUND,
        breadth_v2.SELECTED: SELECTED,
        breadth_v2.STOP_NO_EXACT: STOP_NO_EXACT,
        breadth_v2.STOP_AMBIGUOUS: STOP_AMBIGUOUS,
        full_typed.FRONTIER_READY: FRONTIER_READY,
        full_typed.STOP_WORK_BOUND: STOP_WORK_BOUND,
        full_typed.SELECTED: SELECTED,
        full_typed.STOP_NO_EXACT: STOP_NO_EXACT,
        full_typed.STOP_AMBIGUOUS: STOP_AMBIGUOUS,
    }
    try:
        return statuses[status]
    except KeyError as error:
        raise RuntimeError("V3 route returned an unknown status") from error


class PackedGenericTypedTrainBreadthV3:
    """One signature-routed desired-free V3 build then select seam."""

    __slots__ = ("_construction_receipt", "_route", "_state")

    def __init__(self) -> None:
        raise TypeError("use PackedGenericTypedTrainBreadthV3.build")

    @classmethod
    def build(
        cls,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
    ) -> PackedGenericTypedTrainBreadthV3:
        if not isinstance(signature, TypedSignature):
            raise TypeError("packed generic breadth V3 requires a TypedSignature")
        cross_type_route = (
            signature.result == _INT
            and len(signature.parameters) == 1
            and signature.parameters[0].type == _LIST_INT
        )
        v2_route = len(signature.parameters) == 1 and (
            (signature.result == _INT and signature.parameters[0].type == _INT)
            or (signature.result == _LIST_INT and signature.parameters[0].type == _LIST_INT)
            or (
                signature.result == ProgramType.String()
                and signature.parameters[0].type == ProgramType.String()
            )
        )
        if not cross_type_route and not v2_route:
            raise ValueError(
                "packed generic breadth V3 requires Int -> Int, List[Int] -> List[Int], "
                "List[Int] -> Int, or String -> String"
            )
        route: _Route
        if cross_type_route:
            route = _PackedListIntFrontier.build(
                signature=signature,
                case_arguments=cases,
                literals=literals,
                max_ast_cost=max_cost,
            )
            route_state = route.state
            accepted_rows = cast(
                tuple[tuple[object, ...], ...],
                route_state.case_arguments,
            )
            build_work_sha256 = route_state.build_work_sha256
        else:
            route = breadth_v2.PackedGenericTypedTrainBreadthV2.build(
                signature,
                cases,
                literals,
                max_cost,
            )
            route_state = route.state
            accepted_rows = cast(
                tuple[tuple[object, ...], ...],
                route_state.case_arguments,
            )
            build_work_sha256 = route_state.packed_build_work_sha256
        partial = PackedGenericTypedState(
            status=_status_from_route(route_state.status),
            signature=signature,
            case_arguments=accepted_rows,
            literals=route_state.literals,
            requested_max_ast_cost=route_state.requested_max_ast_cost,
            maximum_completed_ast_cost=route_state.maximum_completed_ast_cost,
            operator_checkpoint_sha256=route_state.operator_checkpoint_sha256,
            root_eclass_ids=route_state.root_eclass_ids,
            route_frontier_sha256=route.sha256,
            route_build_work_sha256=build_work_sha256,
            sha256="",
        )
        state = partial._replace(sha256=_digest("typed-frontier", _state_payload(partial)))
        instance = object.__new__(cls)
        instance._route = route
        instance._state = state
        instance._construction_receipt = _issue_construction_receipt(
            route=route,
            state=state,
            signature=signature,
        )
        return instance

    @property
    def state(self) -> PackedGenericTypedState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def root_eclass_ids(self) -> tuple[str, ...]:
        return self._state.root_eclass_ids

    @property
    def work(self) -> _WorkAudit:
        return self._route.work

    def _authenticate(self) -> None:
        state = self._state
        route = self._route
        route_state = route.state
        receipt = self._construction_receipt
        if (
            not _authenticate_construction_receipt(receipt)
            or receipt.route is not route
            or receipt.state is not state
            or receipt.signature is not state.signature
            or state.sha256 != _digest("typed-frontier", _state_payload(state))
            or state.status != _status_from_route(route_state.status)
            or state.route_frontier_sha256 != route.sha256
            or state.case_arguments != route_state.case_arguments
            or state.literals != route_state.literals
            or state.requested_max_ast_cost != route_state.requested_max_ast_cost
            or state.maximum_completed_ast_cost != route_state.maximum_completed_ast_cost
            or state.operator_checkpoint_sha256 != route_state.operator_checkpoint_sha256
            or state.root_eclass_ids != route_state.root_eclass_ids
            or state.route_build_work_sha256 != _route_build_work_sha256(route)
        ):
            raise RuntimeError("typed packed V3 frontier authentication failed")
        if (
            isinstance(route, breadth_v2.PackedGenericTypedTrainBreadthV2)
            or route.status == full_typed.FRONTIER_READY
        ):
            route._authenticate()

    def _compile_witness(
        self,
        witness: breadth_v2.PackedGenericTypedWitness | _PackedListIntWitness,
    ) -> PackedGenericTypedWitness:
        if isinstance(witness, breadth_v2.PackedGenericTypedWitness):
            tokens = witness.program.tokens
        elif type(witness) is _PackedListIntWitness:
            tokens = witness.tokens
        else:
            raise RuntimeError("V3 route published an unknown proof witness")
        program = tp.Program(tokens, self._state.signature)
        source = tp.compile(program).source
        if source != witness.source:
            raise RuntimeError("V3 typed proof source changed during publication")
        payload = {
            "route_witness_sha256": witness.sha256,
            "family_id": witness.family_id,
            "binding": witness.binding,
            "program_sha256": program.sha256,
            "source": source,
        }
        return PackedGenericTypedWitness(
            family_id=witness.family_id,
            binding=witness.binding,
            program=program,
            source=source,
            sha256=_digest("typed-witness", payload),
        )

    def select(self, desired: Sequence[object]) -> PackedGenericTypedSelection:
        self._authenticate()
        route_selection = self._route.select(desired)
        status = _status_from_route(route_selection.status)
        if route_selection.frontier_sha256 != self._state.route_frontier_sha256:
            raise RuntimeError("V3 selection crossed its retained route frontier")
        if (
            route_selection.checkpoint_before_sha256 != self._state.operator_checkpoint_sha256
            or route_selection.checkpoint_after_sha256 != self._state.operator_checkpoint_sha256
        ):
            raise RuntimeError("typed operator changed during V3 selection")
        witnesses = tuple(self._compile_witness(witness) for witness in route_selection.witnesses)
        if status == SELECTED and (route_selection.exact_proof_count != 1 or len(witnesses) != 1):
            raise RuntimeError("unique V3 winner lost its selected proof")
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "selected_eclass_id": route_selection.selected_eclass_id,
            "exact_proof_count": route_selection.exact_proof_count,
            "witnesses": tuple(item.sha256 for item in witnesses),
            "selected_proof_forest_sha256": (route_selection.selected_proof_forest_sha256),
            "race_sha256": (None if route_selection.race is None else route_selection.race.sha256),
            "checkpoint_before_sha256": route_selection.checkpoint_before_sha256,
            "checkpoint_after_sha256": route_selection.checkpoint_after_sha256,
            "work_sha256": route_selection.work.sha256,
            "route_selection_sha256": route_selection.sha256,
        }
        return PackedGenericTypedSelection(
            status=status,
            frontier_sha256=self._state.sha256,
            selected_eclass_id=route_selection.selected_eclass_id,
            exact_proof_count=route_selection.exact_proof_count,
            witnesses=witnesses,
            selected_proof_forest_sha256=(route_selection.selected_proof_forest_sha256),
            race=route_selection.race,
            checkpoint_before_sha256=route_selection.checkpoint_before_sha256,
            checkpoint_after_sha256=route_selection.checkpoint_after_sha256,
            work=route_selection.work,
            route_selection_sha256=route_selection.sha256,
            sha256=_digest("typed-selection", payload),
        )


__all__ = [
    "FRONTIER_READY",
    "MAX_PHYSICAL_PREDICTIONS",
    "MAX_PHYSICAL_RACES",
    "MAX_RECIPE_RELATION_WORK",
    "SELECTED",
    "STOP_AMBIGUOUS",
    "STOP_NO_EXACT",
    "STOP_WORK_BOUND",
    "PackedGenericTypedSelection",
    "PackedGenericTypedState",
    "PackedGenericTypedTrainBreadthV3",
    "PackedGenericTypedWitness",
]
