from __future__ import annotations

from functools import cache

import pytest

from experiments.r3_pilot import public_typed_operator_successor as successor
from experiments.r3_pilot import public_typed_output_cells as outputs
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain import typed_program
from humanbrain.typed_program import FactorAction
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature
from humanbrain.typed_semantic_synthesis import enumerate_typed_candidate_dag


_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


def _signature(argument: ProgramType, result: ProgramType) -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", argument),),
        result=result,
    )


@cache
def _operator():
    return successor.train_public_typed_operator_successor()[0]


@cache
def _synthesizer():
    return semantic.build_public_typed_semantic_synthesizer(
        literals=(-5, 0, 5),
        max_ast_cost=5,
    )[0]


def _active_int(encoded: outputs.TypedOutputEncoding) -> int | None:
    cell = int(encoded.int_cells.argmax())
    return None if cell == outputs.INT_UNKNOWN_CELL else outputs.INT_MIN + cell


def test_minimum_and_maximum_compile_as_generic_typed_productions() -> None:
    program = typed_program.Program(
        (
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
        ),
        _signature(_INT, _INT),
    )

    assert typed_program.compile(program).source == (
        "def solve(value):\n    return min(max(value, -5), 5)\n"
    )


@pytest.mark.parametrize(
    ("operator", "left", "right", "expected"),
    (
        ("minimum", -4, 3, -4),
        ("minimum", 3, -4, -4),
        ("maximum", -4, 3, 3),
        ("maximum", 3, -4, 3),
    ),
)
def test_extrema_use_less_than_then_copy_physical_races(
    operator: str,
    left: int,
    right: int,
    expected: int,
) -> None:
    model = _operator()
    before = model.live_checkpoint_sha256
    prediction = model.predict(
        outputs.TypedOutputCellBank.encode(left, declared_type=outputs.OutputType.INT),
        FactorAction((), "APPLY2", operator, "SELECTED", "SCOPE0", None),
        scopes=(
            outputs.TypedOutputCellBank.encode(
                right,
                declared_type=outputs.OutputType.INT,
            ),
        ),
    )

    assert _active_int(prediction.output) == expected
    assert tuple(race.bank for race in prediction.races) == (
        "operator-int-pair",
        "operator-less-than-output",
        "copy-type",
        "copy-int",
    )
    assert all(race.simultaneous_spike_candidates == 1 for race in prediction.races)
    assert model.live_checkpoint_sha256 == before


def test_add_minimum_and_maximum_have_one_commutative_syntax_order() -> None:
    dag = enumerate_typed_candidate_dag(
        _signature(_INT, _INT),
        literals=(0,),
        max_ast_cost=3,
    )
    tokens = {
        candidate.program.tokens
        for candidate in dag.tiers[2].candidates
    }
    variable = ("variable-0",)
    zero = ("integer", "+", "0", "end")
    for production in ("add", "minimum", "maximum"):
        assert (production, *variable, *zero) in tokens
        assert (production, *zero, *variable) not in tokens
    assert ("subtract", *variable, *zero) in tokens
    assert ("subtract", *zero, *variable) in tokens


@pytest.mark.parametrize(
    ("arguments", "desired", "expected_cost"),
    (
        (((-9,), (-2,), (0,), (4,), (11,)), (0, 0, 0, 4, 11), 3),
        (((-9,), (-4,), (0,), (3,), (10,)), (-5, -4, 0, 3, 5), 5),
    ),
)
def test_incremental_physical_retention_synthesizes_scalar_extrema(
    arguments: tuple[tuple[int], ...],
    desired: tuple[int, ...],
    expected_cost: int,
) -> None:
    decision = _synthesizer().synthesize(
        _signature(_INT, _INT),
        arguments,
        desired,
    )

    assert decision.status == semantic.STATUS_DONE
    assert decision.program is not None
    assert decision.tier_attempts[-1].ast_cost == expected_cost
    assert {"minimum", "maximum"} & set(decision.program.tokens)
    assert decision.checkpoint_before_sha256 == decision.checkpoint_after_sha256


def test_desired_outputs_do_not_change_incremental_retention_representatives() -> None:
    model = _synthesizer()
    arguments = ((-9,), (-4,), (0,), (3,), (10,))
    clamp = model.synthesize(
        _signature(_INT, _INT),
        arguments,
        (-5, -4, 0, 3, 5),
    )
    alternate = model.synthesize(
        _signature(_INT, _INT),
        arguments,
        (-8, -1, 2, 7, 11),
    )

    assert tuple(
        attempt.representative_candidate_ids for attempt in clamp.tier_attempts
    ) == tuple(
        attempt.representative_candidate_ids
        for attempt in alternate.tier_attempts[: len(clamp.tier_attempts)]
    )


def test_binder_observations_retain_extrema_for_map_bodies() -> None:
    model = semantic.build_public_typed_semantic_synthesizer(
        literals=(0,),
        max_ast_cost=6,
    )[0]
    arguments = (([-4, 1],), ([0, -2],), ([3],), ([],))
    decision = model.synthesize(
        _signature(_LIST_INT, _LIST_INT),
        arguments,
        ([0, 1], [0, 0], [3], []),
    )

    assert decision.status == semantic.STATUS_DONE
    assert decision.program is not None
    assert decision.tier_attempts[-1].ast_cost == 6
    assert "map" in decision.program.tokens
    assert "maximum" in decision.program.tokens
