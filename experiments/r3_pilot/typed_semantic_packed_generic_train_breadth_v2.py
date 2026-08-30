"""Desired-free packed generic typed TRAIN breadth, Layer 1.

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
name.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable, Sequence
from hashlib import sha256
from typing import NamedTuple

from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
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
    case_arguments: tuple[tuple[int], ...]
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
    packed_frontier: packed_v3.PackedScalarFrontier
    state: PackedGenericTypedState
    signature: TypedSignature
    sha256: str


def _receipt_payload(receipt: _ConstructionReceipt) -> dict[str, object]:
    return {
        "packed_frontier_sha256": receipt.packed_frontier.sha256,
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
        packed_frontier: packed_v3.PackedScalarFrontier,
        state: PackedGenericTypedState,
        signature: TypedSignature,
    ) -> _ConstructionReceipt:
        if (
            type(state) is not PackedGenericTypedState
            or state.signature is not signature
            or state.sha256 != _digest("typed-frontier", _state_payload(state))
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


class PackedGenericTypedTrainBreadthV2:
    """Deep desired-free ``Int -> Int`` packed frontier for TRAIN breadth."""

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
            raise TypeError("Layer 1 requires a TypedSignature")
        if (
            signature.result != _INT
            or len(signature.parameters) != 1
            or signature.parameters[0].type != _INT
        ):
            raise ValueError("Layer 1 requires exactly Int -> Int")
        if isinstance(cases, (str, bytes, bytearray)):
            raise TypeError("cases must be typed argument rows")
        if isinstance(literals, (str, bytes, bytearray)):
            raise TypeError("literals must be exact integers")
        if type(max_cost) is not int or not 1 <= max_cost <= 5:
            raise ValueError("Layer-1 max_cost must be in [1,5]")

        accepted_rows: list[tuple[int]] = []

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
                accepted_rows.append(accepted)
                yield accepted[0]

        packed_frontier = packed_v3.PackedScalarFrontier.build(
            case_arguments=scalar_cases(),
            literals=literals,
            max_ast_cost=max_cost,
        )
        packed_state = packed_frontier.state
        if tuple(row[0] for row in accepted_rows) != packed_state.case_arguments:
            raise RuntimeError("typed case rows diverged from the packed frontier")
        status = _status_from_packed(packed_state.status)
        partial = PackedGenericTypedState(
            status=status,
            signature=signature,
            case_arguments=tuple(accepted_rows),
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
        if (
            not _authenticate_construction_receipt(receipt)
            or receipt.owner_token is not self._packed._workset._owner_token
            or receipt.packed_frontier is not self._packed
            or receipt.state is not state
            or receipt.signature is not state.signature
            or state.sha256 != _digest("typed-frontier", _state_payload(state))
            or state.status != _status_from_packed(packed_state.status)
            or state.packed_frontier_sha256 != self._packed.sha256
            or tuple(row[0] for row in state.case_arguments) != packed_state.case_arguments
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
        witness: packed_v3.PackedWitness,
    ) -> PackedGenericTypedWitness:
        tokens = _expression_tokens(witness.source)
        program = tp.Program(tokens, self._state.signature)
        source = tp.compile(program).source
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

    def select(self, desired: Sequence[int]) -> PackedGenericTypedSelection:
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
