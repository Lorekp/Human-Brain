from __future__ import annotations

import inspect

import pytest

from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


def _signature(argument: ProgramType, result: ProgramType) -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", argument),),
        result=result,
    )


@pytest.fixture(scope="module")
def model():
    built, _summary = semantic.build_public_typed_semantic_synthesizer(
        literals=(-3, 0, 2),
    )
    return built


def test_physical_bottom_up_synthesis_selects_scalar_offset(model) -> None:
    arguments = ((-4,), (0,), (3,), (8,))
    decision = model.synthesize(
        _signature(_INT, _INT),
        arguments,
        tuple(row[0] + 2 for row in arguments),
    )

    assert decision.status == semantic.STATUS_DONE
    assert decision.program is not None
    assert decision.program.tokens in {
        (
            "add",
            "variable-0",
            "integer",
            "+",
            "2",
            "end",
        ),
        (
            "add",
            "integer",
            "+",
            "2",
            "end",
            "variable-0",
        ),
    }
    assert decision.tier_attempts[-1].ast_cost == 3
    assert decision.tier_attempts[-1].race.winner_key == decision.selected_candidate_id
    assert decision.operator_predictions > 0
    assert decision.operator_races > 0
    assert decision.checkpoint_before_sha256 == decision.checkpoint_after_sha256


@pytest.mark.parametrize(
    ("desired", "expected_token_options", "cost"),
    (
        (
            ([3, 0], [0], [3, 1], []),
            (("map", "bind", "absolute", "variable-0", "variable-0"),),
            5,
        ),
        (
            ([-1, 2], [2], [5, 1], []),
            (
                (
                    "map",
                    "bind",
                    "add",
                    "variable-0",
                    "integer",
                    "+",
                    "2",
                    "end",
                    "variable-0",
                ),
                (
                    "map",
                    "bind",
                    "add",
                    "integer",
                    "+",
                    "2",
                    "end",
                    "variable-0",
                    "variable-0",
                ),
            ),
            6,
        ),
    ),
)
def test_physical_bottom_up_synthesis_selects_map_programs(
    model,
    desired,
    expected_token_options,
    cost,
) -> None:
    arguments = (([-3, 0],), ((0,),), ([3, -1],), ([],))
    decision = model.synthesize(
        _signature(_LIST_INT, _LIST_INT),
        arguments,
        desired,
    )

    assert decision.status == semantic.STATUS_DONE
    assert decision.program.tokens in expected_token_options
    assert decision.tier_attempts[-1].ast_cost == cost
    assert decision.checkpoint_before_sha256 == decision.checkpoint_after_sha256


def test_bounded_tiers_stop_when_literal_is_absent(model) -> None:
    arguments = ((-4,), (0,), (3,), (8,))
    decision = model.synthesize(
        _signature(_INT, _INT),
        arguments,
        tuple(row[0] + 7 for row in arguments),
    )

    assert decision.status == semantic.STATUS_STOP
    assert decision.program is None
    assert decision.source is None
    assert max(
        attempt.ast_cost
        for attempt in decision.tier_attempts
        if attempt.evaluated_candidates
    ) == 3
    assert all(
        attempt.race is None
        or attempt.race.status == semantic.semantic_race.STOP_NO_EXACT
        for attempt in decision.tier_attempts
    )


def test_production_path_contains_no_host_execution_or_ranking() -> None:
    source = inspect.getsource(semantic.FrozenTypedSemanticSynthesizer.synthesize)
    evaluator_source = inspect.getsource(semantic._CaseEvaluator)
    for forbidden in ("exec(", "eval(", "argmax", "argsort", "argpartition"):
        assert forbidden not in source
        assert forbidden not in evaluator_source
