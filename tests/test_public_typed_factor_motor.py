import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.r3_pilot import public_typed_factor_motor as motor
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _candidate_bank() -> tuple[tp.FactorAction, ...]:
    return (
        tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None),
        tp.FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None),
        tp.FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 2),
        tp.FactorAction((), "APPLY2", "subtract", "SELECTED", "LITERAL", 2),
        tp.FactorAction((), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None),
        tp.FactorAction((), "APPLY2", "subtract", "SCOPE0", "SCOPE1", None),
        tp.FactorAction((), "APPLY2", "less-than", "SCOPE0", "LITERAL", 2),
        tp.FactorAction((), "APPLY2", "less-than", "LITERAL", "SCOPE0", 2),
        tp.FactorAction((), "MAP_BIND", "map", "BOUND0", "SELECTED", None),
        tp.FactorAction((0, 0), "APPLY1", "absolute", "BOUND0", "NONE", None),
        tp.FactorAction((0, 0), "APPLY2", "add", "BOUND0", "LITERAL", 2),
    )


def _winner(candidate_count: int, cell: int) -> np.ndarray:
    spikes = np.zeros(candidate_count, dtype=np.float64)
    spikes[cell] = 1.0
    return spikes


def _decision(
    projection: motor.FixedFactorMotorProjection,
    action: tp.FactorAction,
) -> motor.FactorMotorDecision:
    return projection.resolve(
        _winner(len(projection.candidate_bank), projection.candidate_bank.index(action))
    )


def test_fixed_projection_has_exactly_six_one_hot_factor_synapses_and_hashes() -> None:
    candidates = _candidate_bank()
    first = motor.FixedFactorMotorProjection.build(candidates)
    second = motor.FixedFactorMotorProjection.build(candidates)

    assert tuple(factor.name for factor in first.factors) == motor.FACTOR_NAMES
    assert len(first.factors) == 6
    assert first.sha256 == second.sha256
    assert first.candidate_bank_sha256 == second.candidate_bank_sha256
    assert len(first.sha256) == len(first.candidate_bank_sha256) == 64
    for factor in first.factors:
        assert factor.weights_pa.shape[1] == len(candidates)
        assert np.all(np.count_nonzero(factor.weights_pa, axis=0) == 1)
        assert not factor.weights_pa.flags.writeable
    assert first.factor("site").labels == ((), (0, 0))
    assert first.factor("literal").labels == (None, 2)
    assert motor.FixedFactorMotorProjection.build(tuple(reversed(candidates))).sha256 != (
        first.sha256
    )


def test_every_planned_target_factor_tuple_round_trips_through_physical_races() -> None:
    projection = motor.FixedFactorMotorProjection.build(_candidate_bank())

    for candidate_cell, source_action in enumerate(projection.candidate_bank):
        decision = projection.resolve(
            _winner(len(projection.candidate_bank), candidate_cell)
        )
        assert decision.source_action is source_action
        assert decision.parity_checked
        assert (
            decision.selected_site,
            decision.selected_schema,
            decision.selected_operator,
            decision.selected_left_source,
            decision.selected_right_source,
            decision.selected_literal,
        ) == (
            source_action.path,
            source_action.schema,
            source_action.operator,
            source_action.left_source,
            source_action.right_source,
            source_action.literal,
        )
        assert tuple(race.factor_name for race in decision.races) == motor.FACTOR_NAMES
        assert all(
            race.simultaneous_spike_candidates == 1
            and race.candidates[race.winner_cell] == race.winner
            for race in decision.races
        )


def test_candidate_winner_must_be_one_unique_physical_spike() -> None:
    projection = motor.FixedFactorMotorProjection.build(_candidate_bank())
    empty = np.zeros(len(projection.candidate_bank), dtype=np.float64)
    tied = empty.copy()
    tied[:2] = 1.0

    with pytest.raises(motor.FactorMotorRaceError, match="exactly one"):
        projection.resolve(empty)
    with pytest.raises(motor.FactorMotorRaceError, match="exactly one"):
        projection.resolve(tied)
    with pytest.raises(ValueError, match="binary spike mask"):
        projection.resolve(np.full(len(projection.candidate_bank), 0.5))


def test_projection_detects_factor_parity_corruption_before_actuation() -> None:
    projection = motor.FixedFactorMotorProjection.build(_candidate_bank())
    source_action = projection.candidate_bank[1]
    operator = projection.factor("operator")
    weights = operator.weights_pa.copy()
    source_column = 1
    absolute_cell = operator.labels.index("absolute")
    add_cell = operator.labels.index("add")
    weights[absolute_cell, source_column] = 0.0
    weights[add_cell, source_column] = 5_000.0
    corrupted_operator = replace(operator, weights_pa=weights)
    corrupted = replace(
        projection,
        factors=tuple(
            corrupted_operator if factor.name == "operator" else factor
            for factor in projection.factors
        ),
    )

    with pytest.raises(motor.FactorMotorParityError, match="source FactorAction"):
        _decision(corrupted, source_action)


def test_only_parity_checked_decisions_reach_compiler_owned_actuator() -> None:
    projection = motor.FixedFactorMotorProjection.build(_candidate_bank())
    integer = ProgramType.Int()
    scalar_signature = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("x", integer),),
        result=integer,
    )
    scalar = tp.Program(("variable-0",), scalar_signature)
    done = _decision(projection, projection.candidate_bank[0])
    absolute = _decision(projection, projection.candidate_bank[1])

    assert motor.actuate_selected_factor_action(scalar, done) is scalar
    absolute_program = motor.actuate_selected_factor_action(scalar, absolute)
    assert tp.compile(absolute_program).source == "def solve(x):\n    return abs(x)\n"
    with pytest.raises(TypeError, match="parity-checked"):
        motor.actuate_selected_factor_action(
            scalar,
            replace(absolute, parity_checked=False),
        )

    list_integer = ProgramType.List(integer)
    list_signature = TypedSignature(
        entry_point="solve_list",
        parameters=(ProgramParameter("xs", list_integer),),
        result=list_integer,
    )
    list_program = tp.Program(("variable-0",), list_signature)
    mapped = motor.actuate_selected_factor_action(
        list_program,
        _decision(projection, projection.candidate_bank[8]),
    )
    absolute_map = motor.actuate_selected_factor_action(
        mapped,
        _decision(projection, projection.candidate_bank[9]),
    )
    add_two_map = motor.actuate_selected_factor_action(
        mapped,
        _decision(projection, projection.candidate_bank[10]),
    )
    assert tp.compile(absolute_map).source == (
        "def solve_list(xs):\n    return [abs(item_0) for item_0 in xs]\n"
    )
    assert tp.compile(add_two_map).source == (
        "def solve_list(xs):\n    return [item_0 + 2 for item_0 in xs]\n"
    )


def test_inference_is_disjoint_from_the_single_actuator_call() -> None:
    source = Path(motor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    actuator_calls: list[tuple[ast.Call, str | None]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_factor_action"
        ):
            continue
        ancestor = parents.get(node)
        while ancestor is not None and not isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            ancestor = parents.get(ancestor)
        actuator_calls.append((node, None if ancestor is None else ancestor.name))

    assert [function_name for _, function_name in actuator_calls] == [
        "actuate_selected_factor_action"
    ]
    call_names = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert {"compile", "exec", "replace_subtree", "FactorAction"}.isdisjoint(
        call_names
    )
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(
        "successor" in module_name
        or "planner" in module_name
        or "scoring" in module_name
        for module_name in imported_modules
    )
