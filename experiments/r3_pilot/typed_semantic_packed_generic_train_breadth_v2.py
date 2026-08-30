"""Desired-free packed generic typed TRAIN breadth, Layers 1 and 2.

Layer 1 binds the construction-certified packed scalar V3 frontier to an exact
``Int -> Int`` callable signature.  The packed build sees only the typed
signature, visible argument rows, caller-supplied integer literals, and the
maximum AST cost.  Desired outputs enter only through :meth:`select`, after
the retained V3 publication has authenticated its physical root bank.

The deliberately small generic grammar is the V3 scalar grammar: the callable
argument, integer literals, ``absolute``, signed ``add``, ``maximum``, and
``minimum(maximum(...), ...)``.  No prompt, task/family identity, expected
program, desired vector, sandbox result, or corpus input crosses the build
interface.  Only a physically selected proof is translated back into typed
compiler tokens and emitted with the caller's exact entry point and parameter
name.  Layer 2 adds a signature-routed ``List[Int] -> List[Int]`` packed
frontier for generic map-absolute and map-add.  Its bound item/literal
relations and root collection relation remain packed before physical semantic
tuple prediction, and proof materialization remains selection-only.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple, cast

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_full_typed_v1 as full_typed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as packed_v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as packed_v3
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramType, TypedSignature

FRONTIER_READY = "PACKED_GENERIC_TYPED_TRAIN_BREADTH_V2_READY"
STOP_WORK_BOUND = "STOP_PACKED_GENERIC_TYPED_TRAIN_BREADTH_V2_WORK_BOUND"
SELECTED = "PACKED_GENERIC_TYPED_TRAIN_BREADTH_V2_SELECTED"
STOP_NO_EXACT = "STOP_PACKED_GENERIC_TYPED_TRAIN_BREADTH_V2_NO_EXACT"
STOP_AMBIGUOUS = "STOP_PACKED_GENERIC_TYPED_TRAIN_BREADTH_V2_AMBIGUOUS"

MAX_RECIPE_RELATION_WORK = packed_v3.MAX_RECIPE_RELATION_WORK
MAX_PHYSICAL_PREDICTIONS = packed_v3.MAX_PHYSICAL_PREDICTIONS
MAX_PHYSICAL_RACES = packed_v3.MAX_PHYSICAL_RACES

_NAMESPACE = b"packed-generic-typed-train-breadth-v2-layer1\0"
_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_ABSOLUTE = tp.FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None)


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


class PackedGenericTypedWitness(NamedTuple):
    family_id: str
    binding: tuple[int, ...]
    program: tp.Program
    source: str
    sha256: str


class PackedGenericTypedState(NamedTuple):
    status: str
    signature: TypedSignature
    case_arguments: tuple[tuple[int], ...] | tuple[tuple[tuple[int, ...]], ...]
    literals: tuple[int, ...]
    requested_max_ast_cost: int
    maximum_completed_ast_cost: int
    operator_checkpoint_sha256: str
    root_eclass_ids: tuple[str, ...]
    packed_frontier_sha256: str
    packed_build_work_sha256: str
    sha256: str


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
    work: packed_v3.PackedFrontierWorkAudit
    packed_selection_sha256: str
    sha256: str


class _ConstructionReceipt(NamedTuple):
    authority: object
    owner_token: object
    packed_frontier: object
    state: PackedGenericTypedState
    signature: TypedSignature
    sha256: str


def _receipt_payload(receipt: _ConstructionReceipt) -> dict[str, object]:
    frontier = cast(
        packed_v3.PackedScalarFrontier | _PackedListMapFrontier, receipt.packed_frontier
    )
    return {
        "packed_frontier_sha256": frontier.sha256,
        "state_sha256": receipt.state.sha256,
        "signature": receipt.signature.to_record(),
    }


def _make_receipt_authority() -> tuple[
    Callable[..., _ConstructionReceipt], Callable[[_ConstructionReceipt], bool]
]:
    authority = object()
    issued: dict[int, _ConstructionReceipt] = {}

    def issue(
        *,
        packed_frontier: object,
        state: PackedGenericTypedState,
        signature: TypedSignature,
    ) -> _ConstructionReceipt:
        if (
            type(state) is not PackedGenericTypedState
            or state.signature is not signature
            or state.sha256 != _digest("typed-frontier", _state_payload(state))
            or not isinstance(
                packed_frontier,
                (packed_v3.PackedScalarFrontier, _PackedListMapFrontier),
            )
            or state.packed_frontier_sha256 != packed_frontier.sha256
            or state.root_eclass_ids != packed_frontier.state.root_eclass_ids
        ):
            raise RuntimeError("typed packed construction receipt inputs diverged")
        partial = _ConstructionReceipt(
            authority=authority,
            owner_token=packed_frontier._workset._owner_token,
            packed_frontier=packed_frontier,
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
                receipt.packed_frontier,
                (packed_v3.PackedScalarFrontier, _PackedListMapFrontier),
            )
            and receipt.owner_token is receipt.packed_frontier._workset._owner_token
            and receipt.state.signature is receipt.signature
            and receipt.sha256 == _digest("construction-receipt", _receipt_payload(receipt))
        )

    return issue, authenticate


_issue_construction_receipt, _authenticate_construction_receipt = _make_receipt_authority()


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
        "packed_frontier_sha256": state.packed_frontier_sha256,
        "packed_build_work_sha256": state.packed_build_work_sha256,
    }


def _status_from_packed(status: str) -> str:
    statuses = {
        packed_v3.FRONTIER_READY: FRONTIER_READY,
        packed_v3.STOP_WORK_BOUND: STOP_WORK_BOUND,
        packed_v3.SELECTED: SELECTED,
        packed_v3.STOP_NO_EXACT: STOP_NO_EXACT,
        packed_v3.STOP_AMBIGUOUS: STOP_AMBIGUOUS,
    }
    try:
        return statuses[status]
    except KeyError as error:
        raise RuntimeError("packed scalar frontier returned an unknown status") from error


def _expression_tokens(source: str) -> tuple[str, ...]:
    """Translate one authenticated V3 scalar witness into compiler tokens."""

    if type(source) is not str or not source:
        raise TypeError("packed proof source must be a nonempty expression")
    try:
        expression = ast.parse(source, mode="eval").body
    except SyntaxError as error:
        raise RuntimeError("packed proof source is not an expression") from error

    def tokens(node: ast.expr) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            if node.id != "x":
                raise RuntimeError("packed proof introduced an unknown variable")
            return ("variable-0",)
        if isinstance(node, ast.Constant):
            if type(node.value) is not int:
                raise RuntimeError("packed proof introduced a non-integer literal")
            return tp._factor_literal_tokens(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if not isinstance(node.operand, ast.Constant) or type(node.operand.value) is not int:
                raise RuntimeError("packed proof introduced a non-literal unary sign")
            value = node.operand.value
            if isinstance(node.op, ast.USub):
                value = -value
            return tp._factor_literal_tokens(value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ("add", *tokens(node.left), *tokens(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords:
                raise RuntimeError("packed proof introduced keyword arguments")
            if node.func.id == "abs" and len(node.args) == 1:
                return ("absolute", *tokens(node.args[0]))
            productions = {"min": "minimum", "max": "maximum"}
            production = productions.get(node.func.id)
            if production is not None and len(node.args) == 2:
                return (
                    production,
                    *tokens(node.args[0]),
                    *tokens(node.args[1]),
                )
        raise RuntimeError("packed proof lies outside the frozen Layer-1 grammar")

    return tokens(expression)


class _ListMapCapsule(NamedTuple):
    owner_token: object
    frozen_sha256: str
    component_capsule_sha256: str
    context_certificate_sha256: str
    signature_sha256: str
    case_arguments_sha256: str
    root_family_ids: tuple[str, ...]
    root_eclasses: tuple[packed_v3.PackedEClass, ...]
    root_eclass_ids: tuple[str, ...]
    root_ordinal_index: object
    sha256: str


class _PackedListMapState(NamedTuple):
    status: str
    signature: TypedSignature
    case_arguments: tuple[tuple[tuple[int, ...]], ...]
    literals: tuple[int, ...]
    requested_max_ast_cost: int
    maximum_completed_ast_cost: int
    context_certificate: full_typed.TypedContextCertificate | None
    operator_checkpoint_sha256: str
    frozen_workset: packed_v3.PackedFrozenWorkset | None
    families: tuple[packed_v3.PackedFamily, ...]
    enodes: tuple[packed_v3.PackedENode, ...]
    eclasses: tuple[packed_v3.PackedEClass, ...]
    root_eclasses: tuple[packed_v3.PackedEClass, ...]
    root_eclass_ids: tuple[str, ...]
    component_capsule_sha256: str | None
    list_capsule_sha256: str | None
    build_work_sha256: str
    sha256: str


class _PackedListMapWitness(NamedTuple):
    family_id: str
    binding: tuple[int, ...]
    tokens: tuple[str, ...]
    source: str
    sha256: str


class _PackedListMapSelection(NamedTuple):
    status: str
    frontier_sha256: str
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[_PackedListMapWitness, ...]
    selected_proof_forest_sha256: str | None
    race: semantic_race.TypedSemanticVectorRaceAudit | None
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    work: packed_v3.PackedFrontierWorkAudit
    sha256: str


def _list_root_payload(value: packed_v3.PackedEClass) -> dict[str, object]:
    return {
        "eclass_id": value.eclass_id,
        "minimum_inside_cost": value.minimum_inside_cost,
        "enode_ids": value.enode_ids,
        "group_refs": tuple(item.sha256 for item in value.group_refs),
        "physical_outputs": tuple(item.sha256 for item in value.physical_outputs),
    }


def _list_capsule_payload(value: _ListMapCapsule) -> dict[str, object]:
    ordinal_index = cast(Mapping[str, int], value.root_ordinal_index)
    return {
        "frozen_sha256": value.frozen_sha256,
        "component_capsule_sha256": value.component_capsule_sha256,
        "context_certificate_sha256": value.context_certificate_sha256,
        "signature_sha256": value.signature_sha256,
        "case_arguments_sha256": value.case_arguments_sha256,
        "root_family_ids": value.root_family_ids,
        "root_eclasses": tuple(item.sha256 for item in value.root_eclasses),
        "root_eclass_ids": value.root_eclass_ids,
        "root_ordinal_index": tuple(sorted(ordinal_index.items())),
    }


def _list_state_payload(state: _PackedListMapState) -> dict[str, object]:
    return {
        "status": state.status,
        "signature": state.signature.to_record(),
        "case_arguments": state.case_arguments,
        "literals": state.literals,
        "requested_max_ast_cost": state.requested_max_ast_cost,
        "maximum_completed_ast_cost": state.maximum_completed_ast_cost,
        "context_certificate_sha256": (
            None if state.context_certificate is None else state.context_certificate.sha256
        ),
        "operator_checkpoint_sha256": state.operator_checkpoint_sha256,
        "frozen_workset_sha256": (
            None if state.frozen_workset is None else state.frozen_workset.sha256
        ),
        "component_capsule_sha256": state.component_capsule_sha256,
        "list_capsule_sha256": state.list_capsule_sha256,
        "root_eclass_ids": state.root_eclass_ids,
        "build_work_sha256": state.build_work_sha256,
    }


class _ListMapBuilder(full_typed._Builder):
    """List-only grammar adapter retaining the full-typed V3 workset."""

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
        self.list_capsule: _ListMapCapsule | None = None
        self._list_root_family_ids: list[str] = []

    def _prepare_inputs(self) -> packed.PackedBindingRelation:
        self._charge("signature-read")
        if (
            self.signature.result != _LIST_INT
            or len(self.signature.parameters) != 1
            or self.signature.parameters[0].type != _LIST_INT
        ):
            raise ValueError("Layer 2 requires exactly List[Int] -> List[Int]")

        cases: list[tuple[tuple[int, ...]]] = []
        flattened: list[int] = []
        offsets: list[tuple[int, int]] = []
        physical_rows: list[tuple[object, ...]] = []
        for supplied_row in self.supplied_case_arguments:
            self._charge("case-row-read")
            row = tuple(supplied_row)
            if len(row) != 1 or type(row[0]) is not list:
                raise ValueError("Layer-2 rows must contain one exact List[Int]")
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
            raise ValueError("Layer 2 needs at least one case")
        self.case_arguments = tuple(cases)
        self.flattened_items = tuple(flattened)
        self.item_offsets = tuple(offsets)

        accepted_literals: list[int] = []

        def observed_literals() -> Iterable[int]:
            for value in self.supplied_literals:
                yield value
                self._charge("literal-canonicalization")
                accepted_literals.append(value)

        literal_relation = self.session.domain(
            full_typed._LITERAL_SLOT,
            observed_literals(),
        )
        if not accepted_literals:
            raise ValueError("Layer 2 needs at least one integer literal")
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
            output_projection_sha256=(output_cells.TypedOutputCellBank.projection_sha256()),
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
        if label != "absolute":
            return super()._predict_effect(label, selected, right)
        if right is not None:
            raise RuntimeError("absolute prediction gained a right operand")
        output_cells.TypedOutputCellBank.validate(selected)
        key = (label, selected.sha256, None)
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
        attempted_races = self.physical_races + 1
        if attempted_races > MAX_PHYSICAL_RACES:
            raise packed_v2._PhysicalWorkBoundExceeded(
                "physical-races",
                attempted_races,
            )
        self.physical_predictions = attempted_prediction
        self._record("prediction", _digest("list-prediction-key", key))
        prediction = self.operator.predict(selected, _ABSOLUTE)
        if len(prediction.races) != 1:
            raise RuntimeError("absolute physical race reservation drifted")
        self.physical_races = attempted_races
        output = output_cells.TypedOutputCellBank.validate(prediction.output)
        self.session.charge_workset("prediction-cache-insert")
        incumbent = self._prediction_cache.setdefault(key, output)
        if incumbent != output:
            raise RuntimeError("Layer-2 prediction cache identity collision")
        return output

    def _absolute_vector(
        self,
        values: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if len(values) != len(self.flattened_items):
            raise RuntimeError("absolute bound context changed")
        return tuple(self._predict_effect("absolute", value) for value in values)

    def _add_vector(
        self,
        left: tuple[output_cells.TypedOutputEncoding, ...],
        right: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if len(left) != len(self.flattened_items) or len(right) != len(left):
            raise RuntimeError("add bound context changed")
        return tuple(
            self._predict_effect("add", lhs, rhs) for lhs, rhs in zip(left, right, strict=True)
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
        if production == "absolute-item":
            return self._absolute_vector(children[0])
        if production == "add-item-literal":
            return self._add_vector(children[0], children[1])
        if production == "value-binder":
            if len(children) != 1:
                raise RuntimeError("value binder arity changed")
            return tuple(output_cells.TypedOutputCellBank.validate(item) for item in children[0])
        if production == "map":
            return self._map_vector(children[0], children[1])
        raise RuntimeError("unknown Layer-2 family production")

    def _publish_list_roots(
        self,
        frozen: packed_v3.PackedFrozenWorkset,
        enodes: tuple[packed_v3.PackedENode, ...],
        eclasses: tuple[packed_v3.PackedEClass, ...],
    ) -> tuple[packed_v3.PackedEClass, ...]:
        component = self.authenticated_components
        context = self.context_certificate
        if component is None or context is None:
            raise RuntimeError("Layer-2 publication prerequisites are absent")
        root_family_ids = tuple(sorted(set(self._list_root_family_ids)))
        root_enodes: dict[str, list[packed_v3.PackedENode]] = {}
        for enode in enodes:
            self._charge("list-root-filter-read")
            if enode.family_id in root_family_ids:
                root_enodes.setdefault(
                    packed_v3._eclass_id(enode.physical_outputs),
                    [],
                ).append(enode)
        full_index = {item.eclass_id: item for item in eclasses}
        roots: list[packed_v3.PackedEClass] = []
        ordinal_index: dict[str, int] = {}
        for eclass_id in sorted(root_enodes):
            self._charge("list-root-eclass-publication")
            selected = root_enodes[eclass_id]
            full = full_index[eclass_id]
            partial = packed_v3.PackedEClass(
                eclass_id=eclass_id,
                minimum_inside_cost=min(
                    component.family_index[item.family_id].inside_cost for item in selected
                ),
                enode_ids=tuple(item.enode_id for item in selected),
                group_refs=tuple(item.group_ref for item in selected),
                physical_outputs=full.physical_outputs,
                sha256="",
            )
            root = partial._replace(sha256=_digest("list-root-eclass", _list_root_payload(partial)))
            self._charge("list-root-index-insert")
            ordinal = len(roots)
            incumbent = ordinal_index.setdefault(root.eclass_id, ordinal)
            if incumbent != ordinal:
                raise RuntimeError("Layer-2 root e-class identity collision")
            roots.append(root)
        if not roots:
            raise RuntimeError("Layer-2 frontier published no List[Int] roots")
        frozen_roots = tuple(roots)
        root_ids = tuple(item.eclass_id for item in frozen_roots)
        partial_capsule = _ListMapCapsule(
            owner_token=self.workset._owner_token,
            frozen_sha256=frozen.sha256,
            component_capsule_sha256=component.sha256,
            context_certificate_sha256=context.sha256,
            signature_sha256=_digest("list-signature", self.signature.to_record()),
            case_arguments_sha256=_digest(
                "list-case-arguments",
                self.case_arguments,
            ),
            root_family_ids=root_family_ids,
            root_eclasses=frozen_roots,
            root_eclass_ids=root_ids,
            root_ordinal_index=MappingProxyType(ordinal_index),
            sha256="",
        )
        self.list_capsule = partial_capsule._replace(
            sha256=_digest(
                "list-map-capsule",
                _list_capsule_payload(partial_capsule),
            )
        )
        return frozen_roots

    def work_audit(self) -> packed_v3.PackedFrontierWorkAudit:
        return packed_v3._work_audit(
            self.session.audit,
            self.physical_predictions,
            self.prediction_cache_hits,
            self.physical_races,
            0,
            self.bound_name,
            self.bound_attempted,
            tuple(self.agenda_trace),
        )

    def build(self) -> tuple[_PackedListMapState, packed_v3._SemanticPartitionWorkset]:
        try:
            literal_relation = self._prepare_inputs()
            collection, item, literal = self._seed_families(literal_relation)
            absolute = self._family(
                "absolute-item",
                (item,),
                context_role="bound",
                environment_type_labels=("Int", "List[Int]"),
                result_type_label="Int",
                root_candidate=False,
            )
            absolute_binder = self._family(
                "value-binder",
                (absolute,),
                context_role="binder",
                environment_type_labels=("List[Int]",),
                result_type_label="Function[Int,Int]",
                root_candidate=False,
            )
            absolute_map = self._family(
                "map",
                (absolute_binder, collection),
                context_role="root",
                environment_type_labels=("List[Int]",),
                result_type_label="List[Int]",
                root_candidate=True,
            )
            self._list_root_family_ids.append(absolute_map.family_id)
            if self.max_ast_cost >= 6:
                added = self._family(
                    "add-item-literal",
                    (item, literal),
                    context_role="bound",
                    environment_type_labels=("Int", "List[Int]"),
                    result_type_label="Int",
                    root_candidate=False,
                )
                add_binder = self._family(
                    "value-binder",
                    (added,),
                    context_role="binder",
                    environment_type_labels=("List[Int]",),
                    result_type_label="Function[Int,Int]",
                    root_candidate=False,
                )
                add_map = self._family(
                    "map",
                    (add_binder, collection),
                    context_role="root",
                    environment_type_labels=("List[Int]",),
                    result_type_label="List[Int]",
                    root_candidate=True,
                )
                self._list_root_family_ids.append(add_map.family_id)

            frozen = self.workset.publish()
            families, enodes, eclasses = self._components(frozen)
            root_eclasses = self._publish_list_roots(frozen, enodes, eclasses)
            component = self.authenticated_components
            capsule = self.list_capsule
            context = self.context_certificate
            if component is None or capsule is None or context is None:
                raise RuntimeError("Layer-2 publication is incomplete")
            self._charge("frontier-seal")
            work = self.work_audit()
            partial = _PackedListMapState(
                status=packed_v3.FRONTIER_READY,
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
                root_eclasses=root_eclasses,
                root_eclass_ids=capsule.root_eclass_ids,
                component_capsule_sha256=component.sha256,
                list_capsule_sha256=capsule.sha256,
                build_work_sha256=work.sha256,
                sha256="",
            )
            return partial._replace(
                sha256=_digest("list-frontier", _list_state_payload(partial))
            ), self.workset
        except packed.PackedRelationWorkLimitExceeded as error:
            self.bound_name = "recipe-relation-work"
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset
        except packed_v2._PhysicalWorkBoundExceeded as error:
            self.bound_name = error.name
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.workset

    def _stopped_state(self) -> _PackedListMapState:
        work = self.work_audit()
        partial = _PackedListMapState(
            status=packed_v3.STOP_WORK_BOUND,
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
            root_eclasses=(),
            root_eclass_ids=(),
            component_capsule_sha256=None,
            list_capsule_sha256=None,
            build_work_sha256=work.sha256,
            sha256="",
        )
        return partial._replace(sha256=_digest("list-frontier", _list_state_payload(partial)))


class _PackedListMapFrontier:
    """Private selected-proof frontier for the Layer-2 List route."""

    __slots__ = (
        "_agenda_trace",
        "_authenticated_components",
        "_bound_attempted",
        "_bound_name",
        "_concrete_proof_witnesses",
        "_list_capsule",
        "_operator",
        "_physical_predictions",
        "_physical_races",
        "_prediction_cache_hits",
        "_state",
        "_workset",
    )

    def __init__(
        self,
        state: _PackedListMapState,
        workset: packed_v3._SemanticPartitionWorkset,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
        builder: _ListMapBuilder,
    ) -> None:
        self._state = state
        self._workset = workset
        self._operator = operator
        self._authenticated_components = builder.authenticated_components
        self._list_capsule = builder.list_capsule
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
    ) -> _PackedListMapFrontier:
        if max_ast_cost not in {5, 6}:
            raise ValueError("Layer-2 max_cost must be 5 or 6")
        operator = operator_successor.train_public_typed_operator_successor()[0]
        operator.clear_effect_memo()
        before = operator.live_checkpoint_sha256
        if before != operator.checkpoint_sha256:
            raise RuntimeError("typed operator checkpoint changed before Layer-2 build")
        builder = _ListMapBuilder(
            signature,
            case_arguments,
            literals,
            max_ast_cost,
            operator,
        )
        state, workset = builder.build()
        if operator.live_checkpoint_sha256 != before:
            raise RuntimeError("typed operator checkpoint changed during Layer-2 build")
        return cls(state, workset, operator, builder)

    @property
    def state(self) -> _PackedListMapState:
        return self._state

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def work(self) -> packed_v3.PackedFrontierWorkAudit:
        return packed_v3._work_audit(
            self._workset.session.audit,
            self._physical_predictions,
            self._prediction_cache_hits,
            self._physical_races,
            self._concrete_proof_witnesses,
            self._bound_name,
            self._bound_attempted,
            self._agenda_trace,
        )

    def _authenticate(
        self,
    ) -> tuple[
        packed_v3._AuthenticatedFrozen,
        packed_v3._AuthenticatedComponents,
        _ListMapCapsule,
    ]:
        state = self._state
        frozen = state.frozen_workset
        context = state.context_certificate
        component = self._authenticated_components
        capsule = self._list_capsule
        if state.status != packed_v3.FRONTIER_READY or frozen is None or context is None:
            raise RuntimeError("only a complete Layer-2 frontier can be selected")
        if self._operator.live_checkpoint_sha256 != state.operator_checkpoint_sha256:
            raise RuntimeError("Layer-2 operator checkpoint changed")
        authenticated = self._workset.require_retained(frozen)
        self._workset.session.charge_frontier("authenticate-component-capsule")
        if (
            type(component) is not packed_v3._AuthenticatedComponents
            or component is not self._workset._component_capsule
            or component.owner_token is not self._workset._owner_token
            or component.frozen_sha256 != frozen.sha256
            or component.families is not state.families
            or component.enodes is not state.enodes
            or component.eclasses is not state.eclasses
            or component.sha256 != state.component_capsule_sha256
        ):
            raise RuntimeError("retained Layer-2 component authentication failed")
        self._workset.session.charge_frontier("authenticate-list-capsule")
        if (
            type(capsule) is not _ListMapCapsule
            or capsule.owner_token is not self._workset._owner_token
            or capsule.frozen_sha256 != frozen.sha256
            or capsule.component_capsule_sha256 != component.sha256
            or capsule.context_certificate_sha256 != context.sha256
            or capsule.signature_sha256 != _digest("list-signature", state.signature.to_record())
            or capsule.case_arguments_sha256 != _digest("list-case-arguments", state.case_arguments)
            or capsule.root_eclasses is not state.root_eclasses
            or capsule.root_eclass_ids is not state.root_eclass_ids
            or capsule.root_ordinal_index is None
            or type(capsule.root_ordinal_index) is not _MAPPING_PROXY_TYPE
            or capsule.sha256 != state.list_capsule_sha256
            or capsule.sha256 != _digest("list-map-capsule", _list_capsule_payload(capsule))
            or state.sha256 != _digest("list-frontier", _list_state_payload(state))
        ):
            raise RuntimeError("retained Layer-2 context authentication failed")
        return authenticated, component, capsule

    def _render_witness(
        self,
        witness: packed.ProofWitness,
        component: packed_v3._AuthenticatedComponents,
    ) -> _PackedListMapWitness:
        self._workset.session.charge_frontier("materialize-witness")
        family_index = cast(Mapping[str, packed_v3.PackedFamily], component.family_index)

        def render(family_id: str, binding: tuple[int, ...]) -> tuple[str, ...]:
            self._workset.session.charge_frontier("materialize-family-lookup")
            family = family_index.get(family_id)
            if type(family) is not packed_v3.PackedFamily or family.family_id != family_id:
                raise RuntimeError("Layer-2 proof family index changed")
            if len(binding) != len(family.schema):
                raise RuntimeError("Layer-2 proof binding arity changed")
            if family.production in {"collection-role", "item-role"}:
                if binding != (0,):
                    raise RuntimeError("Layer-2 role binding changed")
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
            if family.production == "absolute-item":
                return ("absolute", *children[0])
            if family.production == "add-item-literal":
                return ("add", *children[0], *children[1])
            if family.production == "value-binder":
                return ("bind", *children[0])
            if family.production == "map":
                return ("map", *children[0], *children[1])
            raise RuntimeError("Layer-2 proof production changed")

        tokens = render(witness.family_id, witness.binding)
        source = tp.compile(tp.Program(tokens, self._state.signature)).source
        payload = {
            "family_id": witness.family_id,
            "binding": witness.binding,
            "tokens": tokens,
            "source": source,
        }
        return _PackedListMapWitness(
            family_id=witness.family_id,
            binding=witness.binding,
            tokens=tokens,
            source=source,
            sha256=_digest("list-witness", payload),
        )

    def _selection(
        self,
        status: str,
        *,
        selected_eclass_id: str | None = None,
        exact_proof_count: int = 0,
        witnesses: tuple[_PackedListMapWitness, ...] = (),
        selected_proof_forest_sha256: str | None = None,
        race: semantic_race.TypedSemanticVectorRaceAudit | None = None,
        checkpoint_before: str | None = None,
    ) -> _PackedListMapSelection:
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
        return _PackedListMapSelection(
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
            sha256=_digest("list-selection", payload),
        )

    def select(self, desired_outputs: Sequence[object]) -> _PackedListMapSelection:
        if self._state.status != packed_v3.FRONTIER_READY:
            if self._state.sha256 != _digest(
                "list-frontier",
                _list_state_payload(self._state),
            ):
                raise RuntimeError("stopped Layer-2 frontier authentication failed")
            return self._selection(packed_v3.STOP_WORK_BOUND)
        try:
            authenticated, component, capsule = self._authenticate()
            if isinstance(desired_outputs, (str, bytes, bytearray)):
                raise TypeError("desired outputs must be exact List[Int] values")
            desired: list[output_cells.TypedOutputEncoding] = []
            for supplied in desired_outputs:
                self._workset.session.charge_frontier("desired-value-read")
                if type(supplied) is not list:
                    raise ValueError("desired outputs must align with List[Int] cases")
                values: list[int] = []
                for item in cast(list[object], supplied):
                    self._workset.session.charge_frontier("desired-list-item-read")
                    if type(item) is not int:
                        raise ValueError("desired List[Int] items must be exact integers")
                    values.append(item)
                desired.append(
                    output_cells.TypedOutputCellBank.encode(
                        values,
                        declared_type=output_cells.OutputType.LIST_INT,
                    )
                )
            if len(desired) != len(self._state.case_arguments):
                raise ValueError("desired outputs must align with List[Int] cases")
            before = self._operator.live_checkpoint_sha256
            attempted_races = self._physical_races + 1
            if attempted_races > MAX_PHYSICAL_RACES:
                raise packed_v2._PhysicalWorkBoundExceeded(
                    "physical-races",
                    attempted_races,
                )
            self._physical_races = attempted_races
            race = semantic_race.select_exact_semantic_vector(
                capsule.root_eclass_ids,
                tuple(item.physical_outputs for item in capsule.root_eclasses),
                tuple(desired),
            )
            if self._operator.live_checkpoint_sha256 != before:
                raise RuntimeError("operator checkpoint changed during Layer-2 selection")
            if race.status == semantic_race.STOP_NO_EXACT:
                return self._selection(
                    packed_v3.STOP_NO_EXACT,
                    race=race,
                    checkpoint_before=before,
                )
            if race.status != semantic_race.SELECTED or race.winner_key is None:
                raise RuntimeError("Layer-2 root bank returned a physical tie")
            self._workset.session.charge_frontier("winner-eclass-index-lookup")
            ordinal_index = cast(Mapping[str, int], capsule.root_ordinal_index)
            ordinal = ordinal_index.get(race.winner_key)
            if type(ordinal) is not int or not 0 <= ordinal < len(capsule.root_eclasses):
                raise RuntimeError("Layer-2 winner references an unknown e-class")
            eclass = capsule.root_eclasses[ordinal]
            if (
                capsule.root_eclass_ids[ordinal] != race.winner_key
                or eclass.eclass_id != race.winner_key
            ):
                raise RuntimeError("Layer-2 physical winner identity changed")
            frozen = self._state.frozen_workset
            if frozen is None:
                raise RuntimeError("Layer-2 selected frontier lost its publication")
            forest = self._workset.materialize_selected_forest(
                frozen,
                authenticated,
                eclass.group_refs,
            )
            proof_count = forest.exact_logical_recipe_count
            if proof_count < 1 or not forest.first_two_witnesses:
                raise RuntimeError("Layer-2 selected e-class has no proof")
            witnesses = tuple(
                self._render_witness(item, component) for item in forest.first_two_witnesses[:2]
            )
            self._concrete_proof_witnesses = len(forest.first_two_witnesses)
            status = packed_v3.SELECTED if proof_count == 1 else packed_v3.STOP_AMBIGUOUS
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
            return self._selection(packed_v3.STOP_WORK_BOUND)
        except packed_v2._PhysicalWorkBoundExceeded as error:
            self._bound_name = error.name
            self._bound_attempted = error.attempted
            self._concrete_proof_witnesses = 0
            return self._selection(packed_v3.STOP_WORK_BOUND)


class PackedGenericTypedTrainBreadthV2:
    """Signature-routed desired-free packed frontier for TRAIN breadth."""

    __slots__ = ("_construction_receipt", "_packed", "_state")

    def __init__(self) -> None:
        raise TypeError("use PackedGenericTypedTrainBreadthV2.build")

    @classmethod
    def build(
        cls,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
    ) -> PackedGenericTypedTrainBreadthV2:
        if not isinstance(signature, TypedSignature):
            raise TypeError("packed generic breadth requires a TypedSignature")
        scalar_route = (
            signature.result != _INT
            or len(signature.parameters) != 1
            or signature.parameters[0].type != _INT
        ) is False
        list_route = (
            signature.result == _LIST_INT
            and len(signature.parameters) == 1
            and signature.parameters[0].type == _LIST_INT
        )
        if not scalar_route and not list_route:
            raise ValueError("packed generic breadth requires Int -> Int or List[Int] -> List[Int]")
        if scalar_route and (type(max_cost) is not int or not 1 <= max_cost <= 5):
            raise ValueError("Layer-1 max_cost must be in [1,5]")
        if list_route and (type(max_cost) is not int or max_cost not in {5, 6}):
            raise ValueError("Layer-2 max_cost must be 5 or 6")
        if len(signature.parameters) != 1 or signature.parameters[0].type not in {
            _INT,
            _LIST_INT,
        }:
            raise ValueError("packed generic breadth requires one typed argument")
        if isinstance(cases, (str, bytes, bytearray)):
            raise TypeError("cases must be typed argument rows")
        if isinstance(literals, (str, bytes, bytearray)):
            raise TypeError("literals must be exact integers")
        accepted_rows: tuple[tuple[int], ...] | tuple[tuple[tuple[int, ...]], ...]
        if scalar_route:
            scalar_rows: list[tuple[int]] = []

            def scalar_cases() -> Iterable[int]:
                for supplied_row in cases:
                    if isinstance(supplied_row, (str, bytes, bytearray)):
                        raise TypeError("each case must be one typed argument row")
                    try:
                        row = tuple(supplied_row)
                    except TypeError as error:
                        raise TypeError("each case must be one typed argument row") from error
                    if len(row) != 1 or type(row[0]) is not int:
                        raise ValueError("Layer-1 cases must contain one exact Int")
                    accepted = (row[0],)
                    scalar_rows.append(accepted)
                    yield accepted[0]

            packed_frontier: packed_v3.PackedScalarFrontier | _PackedListMapFrontier = (
                packed_v3.PackedScalarFrontier.build(
                    case_arguments=scalar_cases(),
                    literals=literals,
                    max_ast_cost=max_cost,
                )
            )
            accepted_rows = tuple(scalar_rows)
        else:
            packed_frontier = _PackedListMapFrontier.build(
                signature=signature,
                case_arguments=cases,
                literals=literals,
                max_ast_cost=max_cost,
            )
            accepted_rows = packed_frontier.state.case_arguments
        packed_state = packed_frontier.state
        if scalar_route and tuple(row[0] for row in accepted_rows) != (packed_state.case_arguments):
            raise RuntimeError("typed case rows diverged from the packed frontier")
        if list_route and accepted_rows != packed_state.case_arguments:
            raise RuntimeError("typed List case rows diverged from the packed frontier")
        status = _status_from_packed(packed_state.status)
        partial = PackedGenericTypedState(
            status=status,
            signature=signature,
            case_arguments=accepted_rows,
            literals=packed_state.literals,
            requested_max_ast_cost=packed_state.requested_max_ast_cost,
            maximum_completed_ast_cost=packed_state.maximum_completed_ast_cost,
            operator_checkpoint_sha256=packed_state.operator_checkpoint_sha256,
            root_eclass_ids=packed_state.root_eclass_ids,
            packed_frontier_sha256=packed_state.sha256,
            packed_build_work_sha256=packed_state.build_work_sha256,
            sha256="",
        )
        state = partial._replace(sha256=_digest("typed-frontier", _state_payload(partial)))
        instance = object.__new__(cls)
        instance._packed = packed_frontier
        instance._state = state
        instance._construction_receipt = _issue_construction_receipt(
            packed_frontier=packed_frontier,
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
    def work(self) -> packed_v3.PackedFrontierWorkAudit:
        return self._packed.work

    def _authenticate(self) -> None:
        packed_state = self._packed.state
        state = self._state
        receipt = self._construction_receipt
        if isinstance(self._packed, packed_v3.PackedScalarFrontier):
            case_arguments_match = (
                tuple(row[0] for row in state.case_arguments) == packed_state.case_arguments
            )
        else:
            case_arguments_match = state.case_arguments == packed_state.case_arguments
        if (
            not _authenticate_construction_receipt(receipt)
            or receipt.owner_token is not self._packed._workset._owner_token
            or receipt.packed_frontier is not self._packed
            or receipt.state is not state
            or receipt.signature is not state.signature
            or state.sha256 != _digest("typed-frontier", _state_payload(state))
            or state.status != _status_from_packed(packed_state.status)
            or state.packed_frontier_sha256 != self._packed.sha256
            or not case_arguments_match
            or state.literals != packed_state.literals
            or state.requested_max_ast_cost != packed_state.requested_max_ast_cost
            or state.maximum_completed_ast_cost != packed_state.maximum_completed_ast_cost
            or state.operator_checkpoint_sha256 != packed_state.operator_checkpoint_sha256
            or state.root_eclass_ids != packed_state.root_eclass_ids
            or state.packed_build_work_sha256 != packed_state.build_work_sha256
        ):
            raise RuntimeError("typed packed frontier authentication failed")

    def _compile_witness(
        self,
        witness: packed_v3.PackedWitness | _PackedListMapWitness,
    ) -> PackedGenericTypedWitness:
        tokens = (
            witness.tokens
            if isinstance(witness, _PackedListMapWitness)
            else _expression_tokens(witness.source)
        )
        program = tp.Program(tokens, self._state.signature)
        source = tp.compile(program).source
        if isinstance(witness, _PackedListMapWitness) and source != witness.source:
            raise RuntimeError("Layer-2 typed proof source changed during publication")
        payload = {
            "packed_witness_sha256": witness.sha256,
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
        packed_selection = self._packed.select(desired)
        status = _status_from_packed(packed_selection.status)
        if packed_selection.frontier_sha256 != self._state.packed_frontier_sha256:
            raise RuntimeError("packed selection crossed typed frontier identity")
        if (
            packed_selection.checkpoint_before_sha256 != self._state.operator_checkpoint_sha256
            or packed_selection.checkpoint_after_sha256 != self._state.operator_checkpoint_sha256
        ):
            raise RuntimeError("typed operator changed during selection")
        witnesses = tuple(self._compile_witness(witness) for witness in packed_selection.witnesses)
        if status == SELECTED and (packed_selection.exact_proof_count != 1 or len(witnesses) != 1):
            raise RuntimeError("unique packed winner lost its typed proof")
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "selected_eclass_id": packed_selection.selected_eclass_id,
            "exact_proof_count": packed_selection.exact_proof_count,
            "witnesses": tuple(item.sha256 for item in witnesses),
            "selected_proof_forest_sha256": (packed_selection.selected_proof_forest_sha256),
            "race_sha256": (
                None if packed_selection.race is None else packed_selection.race.sha256
            ),
            "checkpoint_before_sha256": (packed_selection.checkpoint_before_sha256),
            "checkpoint_after_sha256": packed_selection.checkpoint_after_sha256,
            "work_sha256": packed_selection.work.sha256,
            "packed_selection_sha256": packed_selection.sha256,
        }
        return PackedGenericTypedSelection(
            status=status,
            frontier_sha256=self._state.sha256,
            selected_eclass_id=packed_selection.selected_eclass_id,
            exact_proof_count=packed_selection.exact_proof_count,
            witnesses=witnesses,
            selected_proof_forest_sha256=(packed_selection.selected_proof_forest_sha256),
            race=packed_selection.race,
            checkpoint_before_sha256=packed_selection.checkpoint_before_sha256,
            checkpoint_after_sha256=packed_selection.checkpoint_after_sha256,
            work=packed_selection.work,
            packed_selection_sha256=packed_selection.sha256,
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
    "PackedGenericTypedTrainBreadthV2",
    "PackedGenericTypedWitness",
]
