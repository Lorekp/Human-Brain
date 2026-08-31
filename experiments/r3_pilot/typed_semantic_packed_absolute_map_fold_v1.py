"""Desired-free physical publication of one absolute-map-sum proof plan.

The banked generic List[Int] V4 route already owns physical item, collection,
map, and sum mechanisms, while its typed operator checkpoint already owns a
fitted unary-absolute population.  This separately named self-posed roster adds
only the missing pre-publication family chain::

    absolute(item) -> value-binder -> map(collection) -> sum

Build receives no desired output, operator choice, task/family identity, or
structural certificate.  An operator-name-free fixed certificate must authorize
exactly one retained proof-plan tree before an identity-bound private receipt
can materialize its singleton witness.  Disabling the absolute population
leaves the ordinary frontier intact but produces an explicit pre-proof STOP.

This is a packed physical mechanism prerequisite, not learned operator choice,
prompt parsing, arbitrary map/fold induction, or coding capability.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from typing import NamedTuple, cast

from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_full_typed_v1 as full_typed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as packed_v2
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from experiments.r3_pilot import typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice
from humanbrain import typed_program
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

SCHEMA = "TYPED_SEMANTIC_PACKED_ABSOLUTE_MAP_FOLD_V1"
READY = f"{SCHEMA}_READY"
STOP_FRONTIER = f"STOP_{SCHEMA}_FRONTIER"
STOP_PLAN_SLICE = f"STOP_{SCHEMA}_PLAN_SLICE"
STOP_WORK_BOUND = f"STOP_{SCHEMA}_WORK_BOUND"
STOP_WITNESS_NOT_SINGLETON = f"STOP_{SCHEMA}_WITNESS_NOT_SINGLETON"

CLAIM_BOUNDARY = (
    "one self-posed desired-free packed publication of a physically predicted "
    "absolute(item) family composed through fixed value-binder, map, and sum "
    "families into one exact cost-6 witness; the fixed family roster, dummy "
    "literal domain, typed contexts, structural certificate, compiler, and "
    "case framing remain explicit, so learned operator choice, arbitrary "
    "map/fold induction, parsing, and coding capability remain unproved"
)

FIXED_MAX_COST = 8
WITNESS_AST_COST = 6
FIXED_DUMMY_LITERALS = (-2, 1, 2)
NEW_CELL_COUNT = 0
NEW_PERSISTENT_STATE_BYTES = 0
DESIRED_READS_DURING_BUILD = 0
OPERATOR_NAME_INPUTS_DURING_BUILD = 0

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_ROLE_INT = ("role", "Int")
_ROLE_LIST_INT = ("role", "List[Int]")
_ABSOLUTE_ACTION = typed_program.FactorAction(
    (), "APPLY1", "absolute", "SELECTED", "NONE", None
)
_NAMESPACE = b"typed-semantic-packed-absolute-map-fold-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: bytes, value: object) -> str:
    return sha256(_NAMESPACE + label + b"\0" + _canonical_json(value)).hexdigest()


def list_int_to_int_signature() -> TypedSignature:
    return TypedSignature(
        "aggregate",
        (ProgramParameter("numbers", _LIST_INT),),
        _INT,
    )


def freeze_absolute_map_fold_scaffold() -> plan_slice.StructuralFactorCertificate:
    """Freeze the operator-name-free sum/map/bind/unary/item occurrence tree."""

    return plan_slice.freeze_structural_factor_certificate(
        (
            plan_slice.StructuralOccurrence(
                None,
                None,
                (_ROLE_INT, _ROLE_LIST_INT),
                (),
            ),
            plan_slice.StructuralOccurrence(
                0,
                0,
                (_ROLE_INT, _ROLE_LIST_INT),
                (0, 1),
            ),
            plan_slice.StructuralOccurrence(1, 0, (_ROLE_INT,), (0,)),
            plan_slice.StructuralOccurrence(2, 0, (_ROLE_INT,), (0,)),
            plan_slice.StructuralOccurrence(3, 0, (_ROLE_INT,), (0,)),
            plan_slice.StructuralOccurrence(1, 1, (_ROLE_LIST_INT,), (1,)),
        )
    )


class AbsoluteMapFoldState(NamedTuple):
    status: str
    stop_reason: str | None
    frontier_sha256: str
    structural_certificate_sha256: str
    authorized_slice_count: int
    plan_slice_receipt_sha256: str | None
    root_eclass_count: int
    desired_reads_during_build: int
    operator_name_inputs_during_build: int
    new_cell_count: int
    new_persistent_state_bytes: int
    build_work_sha256: str
    sha256: str


class AbsoluteMapFoldMaterialization(NamedTuple):
    status: str
    stop_reason: str | None
    publication_sha256: str
    plan_slice_receipt_sha256: str | None
    exact_proof_count: int
    witness: breadth_v3.PackedGenericTypedWitness | None
    relation_completed_units: int
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    desired_reads: int
    sha256: str


def _state_payload(value: AbsoluteMapFoldState) -> dict[str, object]:
    return {field: getattr(value, field) for field in value._fields if field != "sha256"}


def _materialization_payload(
    value: AbsoluteMapFoldMaterialization,
) -> dict[str, object]:
    return {
        field: (
            value.witness.sha256
            if field == "witness" and value.witness is not None
            else getattr(value, field)
        )
        for field in value._fields
        if field != "sha256"
    }


class _AbsoluteListIntBuilder(breadth_v4._ListIntBuilder):
    """The unchanged factorized V4 builder plus one physical ABS family path."""

    def __init__(
        self,
        signature: TypedSignature,
        supplied_case_arguments: Iterable[tuple[object, ...]],
        max_ast_cost: int,
        *,
        absolute_cell_gate: bool,
    ) -> None:
        super().__init__(
            signature,
            supplied_case_arguments,
            FIXED_DUMMY_LITERALS,
            max_ast_cost,
            breadth_v4.operator_successor.train_public_typed_operator_successor()[0],
        )
        if type(absolute_cell_gate) is not bool:
            raise TypeError("absolute_cell_gate must be an exact bool")
        self._absolute_cell_gate = absolute_cell_gate

    def _predict_absolute(
        self,
        selected: output_cells.TypedOutputEncoding,
    ) -> output_cells.TypedOutputEncoding:
        output_cells.TypedOutputCellBank.validate(selected)
        key = ("absolute", selected.sha256, None)
        self.session.charge_workset("prediction-cache-lookup")
        cached = self._prediction_cache.get(key)
        if cached is not None:
            self.prediction_cache_hits += 1
            return output_cells.TypedOutputCellBank.validate(cached)

        attempted_prediction = self.physical_predictions + 1
        if attempted_prediction > full_typed.MAX_PHYSICAL_PREDICTIONS:
            raise packed_v2._PhysicalWorkBoundExceeded(
                "physical-predictions", attempted_prediction
            )
        attempted_races = self.physical_races + 1
        if attempted_races > full_typed.MAX_PHYSICAL_RACES:
            raise packed_v2._PhysicalWorkBoundExceeded(
                "physical-races", attempted_races
            )
        self.physical_predictions = attempted_prediction
        self._record(
            "prediction",
            full_typed._digest("prediction-key", key),
        )
        prediction = self.operator.predict(selected, _ABSOLUTE_ACTION)
        if len(prediction.races) != 1:
            raise RuntimeError("absolute physical race reservation drifted")
        self.physical_races = attempted_races
        output = output_cells.TypedOutputCellBank.validate(prediction.output)
        self.session.charge_workset("prediction-cache-insert")
        incumbent = self._prediction_cache.setdefault(key, output)
        if incumbent != output:
            raise RuntimeError("absolute prediction cache identity collision")
        return output

    def _predict_vector(
        self,
        production: str,
        children: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if production == "absolute":
            if len(children) != 1:
                raise RuntimeError("absolute family arity changed")
            return tuple(self._predict_absolute(item) for item in children[0])
        return super()._predict_vector(production, children)

    def _extend_before_publish(
        self,
        collection: packed_v2.PackedPartitionHandle,
        item: packed_v2.PackedPartitionHandle,
        literal: packed_v2.PackedPartitionHandle,
    ) -> None:
        del literal
        if not self._absolute_cell_gate:
            return
        absolute = self._family(
            "absolute",
            (item,),
            context_role="bound",
            environment_type_labels=("Int", "List[Int]"),
            result_type_label="Int",
            root_candidate=False,
        )
        binder = self._family(
            "value-binder",
            (absolute,),
            context_role="binder",
            environment_type_labels=("List[Int]",),
            result_type_label="Function[Int,Int]",
            root_candidate=False,
        )
        mapped = self._family(
            "map",
            (binder, collection),
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


def _build_frontier(
    signature: TypedSignature,
    cases: Iterable[tuple[object, ...]],
    *,
    absolute_cell_gate: bool,
) -> breadth_v4.PackedGenericTypedTrainBreadthV4:
    if not isinstance(signature, TypedSignature):
        raise TypeError("absolute publication requires a TypedSignature")
    if (
        signature.result != _INT
        or len(signature.parameters) != 1
        or signature.parameters[0].type != _LIST_INT
    ):
        raise ValueError("absolute publication requires exactly List[Int] -> Int")
    if isinstance(cases, (str, bytes, bytearray)):
        raise TypeError("cases must be an iterable of typed rows")

    builder = _AbsoluteListIntBuilder(
        signature,
        cases,
        FIXED_MAX_COST,
        absolute_cell_gate=absolute_cell_gate,
    )
    operator = builder.operator
    operator.clear_effect_memo()
    before = operator.live_checkpoint_sha256
    if before != operator.checkpoint_sha256:
        raise RuntimeError("typed operator checkpoint changed before ABS build")
    state, workset = builder.build()
    if operator.live_checkpoint_sha256 != before:
        raise RuntimeError("typed operator checkpoint changed during ABS build")

    # The exact banked route type is retained so the construction and plan-slice
    # authorities authenticate this publication without widening either seam.
    route = breadth_v4._PackedListIntFrontier(state, workset, operator, builder)
    accepted_rows = cast(tuple[tuple[object, ...], ...], state.case_arguments)
    partial = breadth_v3.PackedGenericTypedState(
        status=breadth_v3._status_from_route(state.status),
        signature=signature,
        case_arguments=accepted_rows,
        literals=state.literals,
        requested_max_ast_cost=state.requested_max_ast_cost,
        maximum_completed_ast_cost=state.maximum_completed_ast_cost,
        operator_checkpoint_sha256=state.operator_checkpoint_sha256,
        root_eclass_ids=state.root_eclass_ids,
        route_frontier_sha256=route.sha256,
        route_build_work_sha256=state.build_work_sha256,
        sha256="",
    )
    outer_state = partial._replace(
        sha256=breadth_v3._digest(
            "typed-frontier",
            breadth_v3._state_payload(partial),
        )
    )
    frontier = object.__new__(breadth_v4.PackedGenericTypedTrainBreadthV4)
    frontier._route = route
    frontier._state = outer_state
    frontier._construction_receipt = breadth_v3._issue_construction_receipt(
        route=route,
        state=outer_state,
        signature=signature,
    )
    return frontier


class PackedPhysicalAbsoluteMapFoldV1:
    """Deep build then desired-free singleton-materialize prerequisite."""

    __slots__ = (
        "_frontier",
        "_materialization",
        "_plan_slice_receipt",
        "_state",
        "_structural_certificate",
    )

    def __init__(self) -> None:
        raise TypeError("use PackedPhysicalAbsoluteMapFoldV1.build")

    @classmethod
    def build(
        cls,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        *,
        absolute_cell_gate: bool = True,
    ) -> PackedPhysicalAbsoluteMapFoldV1:
        frontier = _build_frontier(
            signature,
            cases,
            absolute_cell_gate=absolute_cell_gate,
        )
        certificate = freeze_absolute_map_fold_scaffold()
        receipt: plan_slice.PlanSliceReceipt | None = None
        authorized = 0
        stop_reason: str | None = None
        if frontier.status != breadth_v4.FRONTIER_READY:
            status = STOP_FRONTIER
            stop_reason = frontier.status
        else:
            try:
                slice_status, authorized, receipt, *_roots = plan_slice._bind_certificate(
                    frontier,
                    certificate,
                )
            except packed.PackedRelationWorkLimitExceeded:
                status = STOP_WORK_BOUND
                stop_reason = "absolute plan authorization exceeded packed work"
            else:
                if (
                    slice_status == plan_slice.FRONTIER_READY
                    and authorized == 1
                    and receipt is not None
                ):
                    status = READY
                else:
                    status = STOP_PLAN_SLICE
                    stop_reason = slice_status

        work = frontier.work
        partial = AbsoluteMapFoldState(
            status,
            stop_reason,
            frontier.sha256,
            certificate.sha256,
            authorized,
            None if receipt is None else receipt.sha256,
            len(frontier.root_eclass_ids),
            DESIRED_READS_DURING_BUILD,
            OPERATOR_NAME_INPUTS_DURING_BUILD,
            NEW_CELL_COUNT,
            NEW_PERSISTENT_STATE_BYTES,
            work.sha256,
            "",
        )
        state = partial._replace(
            sha256=_digest(b"publication-state", _state_payload(partial))
        )
        instance = object.__new__(cls)
        instance._frontier = frontier
        instance._structural_certificate = certificate
        instance._plan_slice_receipt = receipt
        instance._state = state
        instance._materialization = None
        instance._authenticate()
        return instance

    @property
    def state(self) -> AbsoluteMapFoldState:
        self._authenticate()
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def work(self) -> full_typed.PackedFullTypedWorkAudit:
        return self._frontier.work

    @property
    def plan_slice_receipt(self) -> plan_slice.PlanSliceReceipt | None:
        self._authenticate()
        return self._plan_slice_receipt

    def _authenticate(self) -> None:
        plan_slice._authenticate_certificate(self._structural_certificate)
        construction_receipt = self._frontier._construction_receipt
        if (
            not breadth_v3._authenticate_construction_receipt(construction_receipt)
            or construction_receipt.route is not self._frontier._route
            or construction_receipt.state is not self._frontier._state
            or construction_receipt.signature is not self._frontier._state.signature
        ):
            raise RuntimeError("absolute frontier construction receipt changed")
        state = self._state
        receipt = self._plan_slice_receipt
        if (
            state.sha256 != _digest(b"publication-state", _state_payload(state))
            or state.frontier_sha256 != self._frontier.sha256
            or state.structural_certificate_sha256
            != self._structural_certificate.sha256
            or state.plan_slice_receipt_sha256
            != (None if receipt is None else receipt.sha256)
            or (self._materialization is None and state.build_work_sha256 != self.work.sha256)
        ):
            raise RuntimeError("absolute map-fold publication authentication failed")
        if receipt is not None:
            plan_slice._authenticate_plan_slice_receipt(self._frontier, receipt)
        result = self._materialization
        if result is not None and result.sha256 != _digest(
            b"materialization", _materialization_payload(result)
        ):
            raise RuntimeError("absolute map-fold materialization changed")

    def materialize(self) -> AbsoluteMapFoldMaterialization:
        self._authenticate()
        existing = self._materialization
        if existing is not None:
            return existing
        receipt = self._plan_slice_receipt
        if self._state.status != READY or receipt is None:
            result = self._freeze_materialization(
                status=self._state.status,
                stop_reason=self._state.stop_reason,
                exact_proof_count=0,
                witness=None,
            )
            self._materialization = result
            return result

        binding = plan_slice._authenticate_plan_slice_receipt(
            self._frontier,
            receipt,
        )
        try:
            forest = plan_slice._materialize_authorized_forest(binding)
        except packed.PackedRelationWorkLimitExceeded:
            result = self._freeze_materialization(
                status=STOP_WORK_BOUND,
                stop_reason="absolute singleton materialization exceeded packed work",
                exact_proof_count=0,
                witness=None,
            )
            self._materialization = result
            return result

        proof_count = forest.exact_logical_recipe_count
        witnesses = forest.first_two_witnesses
        if proof_count != 1 or len(witnesses) != 1:
            result = self._freeze_materialization(
                status=STOP_WITNESS_NOT_SINGLETON,
                stop_reason="absolute plan slice did not materialize one exact proof",
                exact_proof_count=proof_count,
                witness=None,
            )
            self._materialization = result
            return result

        route_witness = binding.route._render_witness(
            witnesses[0],
            binding.component,
        )
        witness = cast(
            breadth_v3.PackedGenericTypedWitness,
            self._frontier._compile_witness(route_witness),
        )
        result = self._freeze_materialization(
            status=READY,
            stop_reason=None,
            exact_proof_count=1,
            witness=witness,
        )
        self._materialization = result
        return result

    def _freeze_materialization(
        self,
        *,
        status: str,
        stop_reason: str | None,
        exact_proof_count: int,
        witness: breadth_v3.PackedGenericTypedWitness | None,
    ) -> AbsoluteMapFoldMaterialization:
        work = self.work
        partial = AbsoluteMapFoldMaterialization(
            status,
            stop_reason,
            self._state.sha256,
            self._state.plan_slice_receipt_sha256,
            exact_proof_count,
            witness,
            work.relation.completed_units,
            work.physical_predictions,
            work.prediction_cache_hits,
            work.physical_races,
            0,
            "",
        )
        return partial._replace(
            sha256=_digest(
                b"materialization",
                _materialization_payload(partial),
            )
        )


def require_absolute_map_fold_witness(
    publication: PackedPhysicalAbsoluteMapFoldV1,
) -> breadth_v3.PackedGenericTypedWitness:
    if type(publication) is not PackedPhysicalAbsoluteMapFoldV1:
        raise TypeError("exact absolute map-fold publication required")
    result = publication.materialize()
    if result.status != READY or result.exact_proof_count != 1 or result.witness is None:
        raise ValueError(result.stop_reason or result.status)
    return result.witness


CONFIG_RECORD = {
    "schema": SCHEMA,
    "fixed_max_cost": FIXED_MAX_COST,
    "witness_ast_cost": WITNESS_AST_COST,
    "fixed_dummy_literals": FIXED_DUMMY_LITERALS,
    "new_cell_count": NEW_CELL_COUNT,
    "new_persistent_state_bytes": NEW_PERSISTENT_STATE_BYTES,
    "desired_reads_during_build": DESIRED_READS_DURING_BUILD,
    "operator_name_inputs_during_build": OPERATOR_NAME_INPUTS_DURING_BUILD,
    "claim_boundary": CLAIM_BOUNDARY,
}
CONFIG_SHA256 = _digest(b"config", CONFIG_RECORD)

__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESIRED_READS_DURING_BUILD",
    "FIXED_DUMMY_LITERALS",
    "FIXED_MAX_COST",
    "NEW_CELL_COUNT",
    "NEW_PERSISTENT_STATE_BYTES",
    "OPERATOR_NAME_INPUTS_DURING_BUILD",
    "READY",
    "STOP_FRONTIER",
    "STOP_PLAN_SLICE",
    "STOP_WITNESS_NOT_SINGLETON",
    "STOP_WORK_BOUND",
    "WITNESS_AST_COST",
    "AbsoluteMapFoldMaterialization",
    "AbsoluteMapFoldState",
    "PackedPhysicalAbsoluteMapFoldV1",
    "freeze_absolute_map_fold_scaffold",
    "list_int_to_int_signature",
    "require_absolute_map_fold_witness",
]
