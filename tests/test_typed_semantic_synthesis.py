from __future__ import annotations

import json

import pytest

from humanbrain import typed_program
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)
from humanbrain.typed_semantic_synthesis import (
    MAX_BRIDGE_AST_COST,
    TypedCandidate,
    TypedPhysicalSemanticVector,
    deduplicate_physical_semantics,
    enumerate_typed_candidate_dag,
)


_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


def _signature(parameter_type: ProgramType, result_type: ProgramType) -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", parameter_type),),
        result=result_type,
    )


def _candidate_with_tokens(dag, tokens: tuple[str, ...]) -> TypedCandidate:
    return next(
        candidate
        for tier in dag.tiers
        for candidate in tier.candidates
        if candidate.program.tokens == tokens
    )


def _test_only_host_vector(
    candidate: TypedCandidate,
    arguments: tuple[tuple[object, ...], ...],
) -> TypedPhysicalSemanticVector:
    """Parity fixture only; production synthesis receives physical bytes."""

    namespace: dict[str, object] = {}
    compiled = typed_program.compile(candidate.program)
    exec(compiled.source, namespace)  # noqa: S102 - trusted compiler output
    function = namespace[candidate.program.signature.entry_point]
    observations = []
    for case_arguments in arguments:
        result = function(*case_arguments)
        normalized = tuple(result) if isinstance(result, list) else result
        observations.append(
            json.dumps(normalized, separators=(",", ":")).encode("utf-8")
        )
    return TypedPhysicalSemanticVector(
        result_type=candidate.program.signature.result,
        observations=tuple(observations),
    )


def test_scalar_fragment_is_cost_ordered_compiler_valid_and_deterministic() -> None:
    signature = _signature(_INT, _INT)
    first = enumerate_typed_candidate_dag(
        signature,
        literals=(3, -2, 3),
        max_ast_cost=MAX_BRIDGE_AST_COST,
    )
    second = enumerate_typed_candidate_dag(
        signature,
        literals=(-2, 3),
        max_ast_cost=MAX_BRIDGE_AST_COST,
    )

    assert first == second
    assert first.literal_bank == (-2, 3)
    assert tuple(tier.ast_cost for tier in first.tiers) == (1, 2, 3, 4, 5, 6)
    assert tuple(len(tier.candidates) for tier in first.tiers) == (3, 3, 18, 0, 0, 0)
    assert all(
        candidate.ast_cost == tier.ast_cost
        for tier in first.tiers
        for candidate in tier.candidates
    )

    add = _candidate_with_tokens(
        first,
        ("add", "variable-0", "integer", "+", "3", "end"),
    )
    node_by_id = {node.node_id: node for node in first.nodes}
    add_node = node_by_id[add.root_node_id]
    assert add.ast_cost == 3
    assert add_node.provenance.production == "add"
    assert tuple(
        node_by_id[operand].provenance.production
        for operand in add_node.provenance.operand_node_ids
    ) == ("variable-0", "integer")
    assert typed_program.compile(add.program).source == (
        "def solve(value):\n    return value + 3\n"
    )

    with pytest.raises(ValueError, match="max_ast_cost"):
        enumerate_typed_candidate_dag(
            signature,
            literals=(0,),
            max_ast_cost=MAX_BRIDGE_AST_COST + 1,
        )


def test_list_map_tiers_retain_bind_body_and_collection_provenance() -> None:
    signature = _signature(_LIST_INT, _LIST_INT)
    dag = enumerate_typed_candidate_dag(
        signature,
        literals=(-2, 4),
        max_ast_cost=MAX_BRIDGE_AST_COST,
    )

    assert tuple(len(tier.candidates) for tier in dag.tiers) == (1, 0, 0, 3, 3, 18)
    _candidate_with_tokens(
        dag,
        ("map", "bind", "variable-0", "variable-0"),
    )
    absolute = _candidate_with_tokens(
        dag,
        ("map", "bind", "absolute", "variable-0", "variable-0"),
    )
    offset = _candidate_with_tokens(
        dag,
        (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            "+",
            "4",
            "end",
            "variable-0",
        ),
    )
    assert (absolute.ast_cost, offset.ast_cost) == (5, 6)

    node_by_id = {node.node_id: node for node in dag.nodes}
    map_node = node_by_id[offset.root_node_id]
    binder = node_by_id[map_node.provenance.operand_node_ids[0]]
    collection = node_by_id[map_node.provenance.operand_node_ids[1]]
    body = node_by_id[binder.provenance.operand_node_ids[0]]
    assert map_node.provenance.production == "map"
    assert binder.provenance.production == "bind"
    assert body.provenance.production == "add"
    assert collection.provenance.production == "variable-0"
    assert binder.result_type_label == "Function[Int,Int]"
    assert body.environment_type_labels == ("Int", "List[Int]")
    assert typed_program.compile(offset.program).source == (
        "def solve(value):\n"
        "    return [item_0 + 4 for item_0 in value]\n"
    )


def test_caller_supplied_physical_vectors_drive_equivalence_not_ranking() -> None:
    signature = _signature(_INT, _INT)
    dag = enumerate_typed_candidate_dag(
        signature,
        literals=(0, 1),
        max_ast_cost=3,
    )
    variable = _candidate_with_tokens(dag, ("variable-0",))
    add_zero = _candidate_with_tokens(
        dag,
        ("add", "variable-0", "integer", "+", "0", "end"),
    )
    subtract_zero = _candidate_with_tokens(
        dag,
        ("subtract", "variable-0", "integer", "+", "0", "end"),
    )
    absolute = _candidate_with_tokens(dag, ("absolute", "variable-0"))
    add_one = _candidate_with_tokens(
        dag,
        ("add", "variable-0", "integer", "+", "1", "end"),
    )
    observed = (variable, add_zero, subtract_zero, absolute, add_one)
    visible_arguments = ((-2,), (3,))
    vectors = {
        candidate.root_node_id: _test_only_host_vector(candidate, visible_arguments)
        for candidate in observed
    }

    result = deduplicate_physical_semantics(dag, vectors)
    identity_class = next(
        equivalence
        for equivalence in result.classes
        if variable.root_node_id in equivalence.member_candidate_ids
    )
    assert result.case_count == 2
    assert result.observed_candidate_count == 5
    assert identity_class.representative_candidate_id == variable.root_node_id
    assert identity_class.member_candidate_ids == (
        variable.root_node_id,
        add_zero.root_node_id,
        subtract_zero.root_node_id,
    )
    assert absolute.root_node_id in result.representative_candidate_ids
    assert add_one.root_node_id in result.representative_candidate_ids

    wrong_type = TypedPhysicalSemanticVector(
        result_type=ProgramType.Bool(),
        observations=(b"same", b"same"),
    )
    with pytest.raises(ValueError, match="does not match"):
        deduplicate_physical_semantics(dag, {variable.root_node_id: wrong_type})


def test_dag_operands_always_precede_their_compound_node() -> None:
    dag = enumerate_typed_candidate_dag(
        _signature(_LIST_INT, _LIST_INT),
        literals=(-1, 2),
    )
    position = {node.node_id: index for index, node in enumerate(dag.nodes)}
    assert all(
        position[operand] < position[node.node_id]
        for node in dag.nodes
        for operand in node.provenance.operand_node_ids
    )
