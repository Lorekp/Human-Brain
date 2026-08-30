"""Factorized selected-proof successor for generic typed TRAIN breadth.

V3 remains immutable.  This separately named successor keeps its signature-
routed construction, physical semantic predictions, lazy proof-plan registry,
and public ``build`` then ``select`` interface.  Its sole mechanism change is
inside selected proof publication for ``List[Int] -> Int``: authenticated tuple
recipes with identical schemas and constraints are materialized as an exact
factorized union of Cartesian products.  The least-diverse child occurrence is
factored first, so repeated collection, binder, literal, or nested proof roots
are renamed and joined once while distinct occurrence slots and orientations
remain separate.

Build is still desired-free and proof-free.  Factorization begins only after
the physical desired race identifies one retained e-class, stays on the same
metered packed-relation session, and finishes through the banked family-slice
and proof-forest publication.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable
from typing import NamedTuple, cast

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as packed_v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v4 as packed_v4
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from humanbrain.typed_program_types import ProgramType, TypedSignature

FRONTIER_READY = breadth_v3.FRONTIER_READY
STOP_WORK_BOUND = breadth_v3.STOP_WORK_BOUND
SELECTED = breadth_v3.SELECTED
STOP_NO_EXACT = breadth_v3.STOP_NO_EXACT
STOP_AMBIGUOUS = breadth_v3.STOP_AMBIGUOUS

MAX_RECIPE_RELATION_WORK = breadth_v3.MAX_RECIPE_RELATION_WORK
MAX_PHYSICAL_PREDICTIONS = breadth_v3.MAX_PHYSICAL_PREDICTIONS
MAX_PHYSICAL_RACES = breadth_v3.MAX_PHYSICAL_RACES

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


_Recipe = packed_v4._TupleProofRecipe
_RecipeChild = tuple[
    str,
    tuple[packed.TypedSlot, ...],
    tuple[packed.TypedSlot, ...],
]


class _IssuedRelationReceipt(NamedTuple):
    authority: object
    relation: packed.PackedBindingRelation
    slots: object
    nodes: object
    allocation: object
    validated: packed._ValidatedRelation
    sha256: str


def _relation_receipt_payload(
    receipt: _IssuedRelationReceipt,
) -> dict[str, object]:
    relation = receipt.relation
    return {
        "relation_sha256": relation.sha256,
        "root_id": relation.root_id,
        "slots": tuple(slot.to_record() for slot in relation.slots),
        "allocation": (
            relation.allocation.decision_nodes,
            relation.allocation.decision_edges,
        ),
    }


class _CompactedFrontierSession(packed_v2._FrontierSession):
    """Metered relation session with private selected-only issued receipts."""

    __slots__ = (
        "_compaction_active",
        "_relation_authority",
        "_relation_receipts",
    )

    def __init__(self, *, work_limit: int) -> None:
        super().__init__(work_limit=work_limit)
        self._compaction_active = False
        self._relation_authority = object()
        self._relation_receipts: dict[int, _IssuedRelationReceipt] = {}

    def activate_selected_compaction(self) -> None:
        if not self._compaction_active:
            self.charge_workset("selected-relation-receipt-registry-open")
            self._compaction_active = True

    def _issue_relation_receipt(
        self,
        relation: packed.PackedBindingRelation,
        validated: packed._ValidatedRelation,
    ) -> packed._ValidatedRelation:
        self.charge_workset("selected-relation-receipt-issue")
        partial = _IssuedRelationReceipt(
            authority=self._relation_authority,
            relation=relation,
            slots=relation.slots,
            nodes=relation.nodes,
            allocation=relation.allocation,
            validated=validated,
            sha256="",
        )
        receipt = partial._replace(
            sha256=breadth_v3._digest(
                "selected-relation-receipt",
                _relation_receipt_payload(partial),
            )
        )
        incumbent = self._relation_receipts.setdefault(id(relation), receipt)
        if incumbent is not receipt:
            if incumbent.relation is not relation:
                raise RuntimeError("selected relation receipt identity collision")
            return incumbent.validated
        return validated

    def _authenticate_relation(
        self,
        relation: packed.PackedBindingRelation,
    ) -> packed._ValidatedRelation:
        if self._compaction_active:
            receipt = self._relation_receipts.get(id(relation))
            if receipt is not None:
                self.charge_workset("selected-relation-receipt-authenticate")
                if (
                    type(receipt) is not _IssuedRelationReceipt
                    or receipt.authority is not self._relation_authority
                    or receipt.relation is not relation
                    or receipt.slots is not relation.slots
                    or receipt.nodes is not relation.nodes
                    or receipt.allocation is not relation.allocation
                    or receipt.sha256
                    != breadth_v3._digest(
                        "selected-relation-receipt",
                        _relation_receipt_payload(receipt),
                    )
                ):
                    raise RuntimeError("selected relation receipt authentication failed")
                return receipt.validated
        validated = super()._authenticate_relation(relation)
        if self._compaction_active:
            return self._issue_relation_receipt(relation, validated)
        return validated


def _child_sort_key(child: _RecipeChild) -> tuple[object, ...]:
    root_id, schema, targets = child
    return (
        root_id,
        tuple(slot.to_record() for slot in schema),
        tuple(slot.to_record() for slot in targets),
    )


class _FactorizedSemanticPartitionWorkset(packed_v4._SemanticPartitionWorkset):
    """Exact selected-only union/product factorization over retained recipes."""

    def __init__(self, session: _CompactedFrontierSession) -> None:
        if type(session) is not _CompactedFrontierSession:
            raise TypeError("factorized workset requires its compacted frontier session")
        self.session = session
        self._arena = packed._Arena()
        self._records = {}
        self._nonce = secrets.token_hex(16)
        self._generation = 0
        self._sealed = False
        self._frozen = None
        self._authenticated = None
        self._component_capsule = None
        self._owner_token = object()
        self._retained_arena = None
        self._plans = {}
        self._root_domains = {}
        self._plan_registry_sha256 = packed_v4._digest("empty-plan-registry", ())
        self._plan_edge_count = 0
        self._plan_recipe_count = 0
        self._retained_plans = None

    def _recipe_child(self, recipe: _Recipe, position: int) -> _RecipeChild:
        return (
            recipe.child_root_ids[position],
            recipe.child_schemas[position],
            recipe.child_target_slots[position],
        )

    def _factorized_product_union(
        self,
        recipes: tuple[_Recipe, ...],
        positions: tuple[int, ...],
        authenticated: packed_v4._AuthenticatedFrozen,
        memo: dict[str, packed.PackedBindingRelation],
    ) -> packed.PackedBindingRelation:
        if not recipes or not positions:
            raise RuntimeError("selected factorization lost its tuple recipes")
        position = min(
            positions,
            key=lambda index: (
                len({self._recipe_child(recipe, index) for recipe in recipes}),
                index,
            ),
        )
        self.session.charge_workset("selected-factor-position")
        grouped: dict[_RecipeChild, list[_Recipe]] = {}
        for recipe in recipes:
            self.session.charge_workset("selected-factor-child-lookup")
            child = self._recipe_child(recipe, position)
            grouped.setdefault(child, []).append(recipe)

        remaining = tuple(index for index in positions if index != position)
        relation: packed.PackedBindingRelation | None = None
        for child in sorted(grouped, key=_child_sort_key):
            self.session.charge_workset("selected-factor-branch")
            child_id, child_schema, targets = child
            child_relation = self._materialize_plan_root(
                child_schema,
                child_id,
                authenticated,
                memo,
            )
            branch = self.session.rename(child_relation, targets)
            if remaining:
                suffix = self._factorized_product_union(
                    tuple(grouped[child]),
                    remaining,
                    authenticated,
                    memo,
                )
                branch = self.session.cartesian_product(branch, suffix)
            relation = branch if relation is None else self.session.union(relation, branch)
        if relation is None or relation.root_id is None:
            raise RuntimeError("selected factorization materialized an empty relation")
        return relation

    def _materialize_plan_root(
        self,
        schema: tuple[packed.TypedSlot, ...],
        root_id: str,
        authenticated: packed_v4._AuthenticatedFrozen,
        memo: dict[str, packed.PackedBindingRelation],
    ) -> packed.PackedBindingRelation:
        if (
            authenticated is not self._authenticated
            or authenticated.owner_token is not self._owner_token
            or authenticated.plans is not self._retained_plans
        ):
            raise RuntimeError("factorized selection crossed its retained proof plans")
        session = cast(_CompactedFrontierSession, self.session)
        if type(session) is not _CompactedFrontierSession:
            raise RuntimeError("factorized selection lost its compacted session")
        session.activate_selected_compaction()
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
            raise RuntimeError("selected factorized proof-plan schema changed")
        if node.kind != "semantic-group" or node.child_plan_ids or not node.recipes:
            raise RuntimeError("non-group proof plan reached factorized selection")

        by_shape: dict[tuple[object, ...], list[_Recipe]] = {}
        for recipe in node.recipes:
            self.session.charge_workset("selected-plan-recipe")
            if (
                recipe.schema != schema
                or not recipe.child_root_ids
                or len(recipe.child_root_ids) != len(recipe.child_schemas)
                or len(recipe.child_root_ids) != len(recipe.child_target_slots)
                or len(recipe.domains) != len(schema)
            ):
                raise RuntimeError("selected factorized proof recipe changed")
            key = (
                recipe.child_schemas,
                recipe.child_target_slots,
                recipe.constraints,
            )
            self.session.charge_workset("selected-factor-shape-lookup")
            by_shape.setdefault(key, []).append(recipe)

        relation: packed.PackedBindingRelation | None = None
        for key in sorted(
            by_shape,
            key=lambda item: (
                tuple(
                    tuple(slot.to_record() for slot in child_schema)
                    for child_schema in cast(
                        tuple[tuple[packed.TypedSlot, ...], ...],
                        item[0],
                    )
                ),
                tuple(
                    tuple(slot.to_record() for slot in targets)
                    for targets in cast(
                        tuple[tuple[packed.TypedSlot, ...], ...],
                        item[1],
                    )
                ),
                packed_v4._constraint_records(cast(tuple[packed.SlotConstraint, ...], item[2])),
            ),
        ):
            recipes = tuple(by_shape[key])
            child_count = len(recipes[0].child_root_ids)
            shaped = self._factorized_product_union(
                recipes,
                tuple(range(child_count)),
                authenticated,
                memo,
            )
            constraints = recipes[0].constraints
            if constraints:
                shaped = self.session.select(shaped, constraints)
            if shaped.slots != schema or shaped.root_id is None:
                raise RuntimeError("selected factorized group materialized incorrectly")
            relation = shaped if relation is None else self.session.union(relation, shaped)
        if relation is None or relation.root_id is None:
            raise RuntimeError("selected factorized group materialized empty")
        memo[root_id] = relation
        return relation


class _ListIntBuilder(breadth_v3._ListIntBuilder):
    """V3 generic cross-type construction with one deeper selected-proof module."""

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
        self.session = _CompactedFrontierSession(work_limit=MAX_RECIPE_RELATION_WORK)
        self.workset = _FactorizedSemanticPartitionWorkset(self.session)
        self.authenticated_components = None


class _PackedListIntFrontier(breadth_v3._PackedListIntFrontier):
    """Private factorized selected-proof frontier for ``List[Int] -> Int``."""

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
            raise TypeError("V4 cross-type frontier requires a TypedSignature")
        if isinstance(case_arguments, (str, bytes, bytearray)):
            raise TypeError("case_arguments must be an iterable of typed rows")
        if isinstance(literals, (str, bytes, bytearray)):
            raise TypeError("literals must be an iterable of exact integers")
        if type(max_ast_cost) is not int or max_ast_cost not in {8, 9}:
            raise ValueError("V4 List[Int] -> Int max_cost must be 8 or 9")
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
            raise RuntimeError("typed operator checkpoint changed before V4 build")
        builder = _ListIntBuilder(
            signature,
            case_arguments,
            literals,
            max_ast_cost,
            frozen_operator,
        )
        state, workset = builder.build()
        if frozen_operator.live_checkpoint_sha256 != before:
            raise RuntimeError("typed operator checkpoint changed during V4 build")
        return cls(state, workset, frozen_operator, builder)


class PackedGenericTypedTrainBreadthV4(breadth_v3.PackedGenericTypedTrainBreadthV3):
    """V3's deep interface with factorized cross-type proof publication."""

    __slots__ = ()

    @classmethod
    def build(
        cls,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
    ) -> PackedGenericTypedTrainBreadthV4:
        if not isinstance(signature, TypedSignature):
            raise TypeError("packed generic breadth V4 requires a TypedSignature")
        cross_type_route = (
            signature.result == _INT
            and len(signature.parameters) == 1
            and signature.parameters[0].type == _LIST_INT
        )
        if not cross_type_route:
            return super().build(signature, cases, literals, max_cost)

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
        partial = breadth_v3.PackedGenericTypedState(
            status=breadth_v3._status_from_route(route_state.status),
            signature=signature,
            case_arguments=accepted_rows,
            literals=route_state.literals,
            requested_max_ast_cost=route_state.requested_max_ast_cost,
            maximum_completed_ast_cost=route_state.maximum_completed_ast_cost,
            operator_checkpoint_sha256=route_state.operator_checkpoint_sha256,
            root_eclass_ids=route_state.root_eclass_ids,
            route_frontier_sha256=route.sha256,
            route_build_work_sha256=route_state.build_work_sha256,
            sha256="",
        )
        state = partial._replace(
            sha256=breadth_v3._digest(
                "typed-frontier",
                breadth_v3._state_payload(partial),
            )
        )
        instance = object.__new__(cls)
        instance._route = route
        instance._state = state
        instance._construction_receipt = breadth_v3._issue_construction_receipt(
            route=route,
            state=state,
            signature=signature,
        )
        return instance


PackedGenericTypedSelection = breadth_v3.PackedGenericTypedSelection
PackedGenericTypedState = breadth_v3.PackedGenericTypedState
PackedGenericTypedWitness = breadth_v3.PackedGenericTypedWitness

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
    "PackedGenericTypedTrainBreadthV4",
    "PackedGenericTypedWitness",
]
