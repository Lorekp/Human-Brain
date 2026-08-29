from __future__ import annotations

from functools import cache

from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain import typed_program
from humanbrain import typed_semantic_synthesis as synthesis
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


def _signature() -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("values", _LIST_INT),),
        result=_INT,
    )


@cache
def _clamp_model() -> semantic.FrozenTypedSemanticSynthesizer:
    return semantic.build_public_typed_semantic_synthesizer(
        literals=(-5, 5),
        max_ast_cost=9,
    )[0]


@cache
def _count_model() -> semantic.FrozenTypedSemanticSynthesizer:
    return semantic.build_public_typed_semantic_synthesizer(
        literals=(2, 3, 5),
        max_ast_cost=9,
    )[0]


def test_sum_less_than_and_predicate_count_are_generic_compiler_productions() -> None:
    summed = typed_program.Program(("sum", "variable-0"), _signature())
    counted = typed_program.Program(
        (
            "count",
            "bind",
            "less-than",
            "variable-0",
            "integer",
            "+",
            "3",
            "end",
            "variable-0",
        ),
        _signature(),
    )

    assert typed_program.compile(summed).source == (
        "def solve(values):\n    return sum(values)\n"
    )
    assert typed_program.compile(counted).source == (
        "def solve(values):\n"
        "    return sum((1 for item_0 in values if item_0 < 3))\n"
    )


def test_explicit_map_add_predicate_surface_is_enumerable_at_cost_eight() -> None:
    tokens = (
        "count",
        "bind",
        "less-than",
        "integer",
        "+",
        "5",
        "end",
        "add",
        "variable-0",
        "integer",
        "+",
        "2",
        "end",
        "variable-0",
    )
    program = typed_program.Program(tokens, _signature())
    assert typed_program.compile(program).source == (
        "def solve(values):\n"
        "    return sum((1 for item_0 in values if 5 < item_0 + 2))\n"
    )

    dag = synthesis.enumerate_typed_candidate_dag(
        _signature(),
        literals=(2, 5),
        max_ast_cost=8,
    )
    candidate = next(
        candidate
        for candidate in dag.tiers[7].candidates
        if candidate.program.tokens == tokens
    )
    nodes = {node.node_id: node for node in dag.nodes}
    count = nodes[candidate.root_node_id]
    binder = nodes[count.provenance.operand_node_ids[0]]
    predicate = nodes[binder.provenance.operand_node_ids[0]]
    right = nodes[predicate.provenance.operand_node_ids[1]]
    assert candidate.ast_cost == 8
    assert count.provenance.production == "count"
    assert binder.provenance.production == "bind"
    assert predicate.provenance.production == "less-than"
    assert right.provenance.production == "add"


def test_retained_frontier_synthesizes_exact_cost_nine_clamp_map_sum() -> None:
    model = _clamp_model()
    arguments = (([-9, -4, 0, 3],), ([9, -9],), ([-7],), ([7],))
    desired = (-6, 0, -5, 5)

    first = model.synthesize(_signature(), arguments, desired)
    second = model.synthesize(_signature(), arguments, desired)

    assert first.status == semantic.STATUS_DONE
    assert first.program is not None
    assert first.tier_attempts[-1].ast_cost == 9
    assert first.program.tokens == (
        "sum",
        "map",
        "bind",
        "minimum",
        "maximum",
        "variable-0",
        "integer",
        "-",
        "5",
        "end",
        "integer",
        "+",
        "5",
        "end",
        "variable-0",
    )
    assert first == second
    assert first.operator_predictions > 0
    assert first.operator_races > first.operator_predictions
    assert (
        first.checkpoint_before_sha256
        == first.checkpoint_after_sha256
        == model.checkpoint_sha256
    )


def test_predicate_count_retains_bool_binder_body_and_accepts_algebraic_cost_six(
    monkeypatch,
) -> None:
    model = _count_model()
    calls: list[tuple[frozenset[str], synthesis.TypedCandidateDag]] = []
    enumerate_original = synthesis.enumerate_typed_candidate_dag

    def recording_enumerate(*args, **kwargs):
        retained = frozenset(kwargs.get("retained_node_ids") or ())
        dag = enumerate_original(*args, **kwargs)
        calls.append((retained, dag))
        return dag

    monkeypatch.setattr(synthesis, "enumerate_typed_candidate_dag", recording_enumerate)
    arguments = (([-2, 0, 2, 4],), ([3, 4],), ([2],), ([],))
    # This visible behavior was posed as count(x + 2 > 5).  The generic literal
    # algebra may honestly find the equivalent 3 < x program at lower cost.
    decision = model.synthesize(_signature(), arguments, (1, 1, 0, 0))

    assert decision.status == semantic.STATUS_DONE
    assert decision.program is not None
    assert decision.tier_attempts[-1].ast_cost == 6
    assert decision.program.tokens == (
        "count",
        "bind",
        "less-than",
        "integer",
        "+",
        "3",
        "end",
        "variable-0",
        "variable-0",
    )
    selected_dag = next(
        dag
        for _, dag in reversed(calls)
        if any(
            candidate.root_node_id == decision.selected_candidate_id
            for tier in dag.tiers
            for candidate in tier.candidates
        )
    )
    nodes = {node.node_id: node for node in selected_dag.nodes}
    count_node = nodes[decision.selected_candidate_id]
    binder = nodes[count_node.provenance.operand_node_ids[0]]
    predicate = nodes[binder.provenance.operand_node_ids[0]]
    assert predicate.result_type_label == "Bool"
    assert predicate.provenance.production == "less-than"
    assert all(dag.max_ast_cost == model.max_ast_cost for _, dag in calls)
    assert any(
        node.node_id == predicate.node_id and node.ast_cost == 3
        for node in calls[2][1].nodes
    )
    assert predicate.node_id in calls[3][0]
    cost_four_nodes = {node.node_id: node for node in calls[3][1].nodes}
    cost_four_binder = cost_four_nodes[binder.node_id]
    assert cost_four_binder.ast_cost == 4
    assert cost_four_binder.result_type_label == "Function[Int,Bool]"
    assert binder.node_id not in calls[3][0]
    assert all(
        candidate.program.tokens[0] != "count"
        for candidate in calls[3][1].tiers[5].candidates
    )
    assert binder.node_id in calls[4][0]
    assert any(
        candidate.root_node_id == decision.selected_candidate_id
        for candidate in calls[4][1].tiers[5].candidates
    )
    assert any(predicate.node_id in retained for retained, _ in calls)


def test_list_retention_is_independent_of_desired_outputs() -> None:
    model = _count_model()
    arguments = (([-2, 0, 2, 4],), ([3, 4],), ([2],), ([],))
    count = model.synthesize(_signature(), arguments, (1, 1, 0, 0))
    alternate = model.synthesize(_signature(), arguments, (2, 2, 1, 1))
    shared = min(len(count.tier_attempts), len(alternate.tier_attempts))

    assert shared >= 6
    assert tuple(
        attempt.representative_candidate_ids
        for attempt in count.tier_attempts[:shared]
    ) == tuple(
        attempt.representative_candidate_ids
        for attempt in alternate.tier_attempts[:shared]
    )


def test_unsupported_sum_is_pruned_but_its_physical_work_is_counted() -> None:
    model = semantic.build_public_typed_semantic_synthesizer(
        literals=(0,),
        max_ast_cost=2,
    )[0]
    decision = model.synthesize(_signature(), (([64, 64],),), (1,))

    assert decision.status == semantic.STATUS_STOP
    tier = decision.tier_attempts[1]
    assert tier.ast_cost == 2
    assert tier.evaluated_candidates == 2
    assert tier.representative_candidate_ids == ()
    assert decision.candidate_count == 2
    assert decision.operator_predictions == 4
    assert decision.operator_races > 0
    assert (
        decision.checkpoint_before_sha256
        == decision.checkpoint_after_sha256
        == model.checkpoint_sha256
    )
