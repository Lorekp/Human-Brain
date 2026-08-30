from __future__ import annotations

from functools import cache

import pytest

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_version_space_v2 as version_space
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


def _signature(argument: ProgramType) -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("values" if argument == _LIST_INT else "x", argument),),
        result=_INT,
    )


@cache
def _scalar_pressure() -> version_space.TypedSemanticVersionSpace:
    return version_space.TypedSemanticVersionSpace.build(
        signature=_signature(_INT),
        case_arguments=((-13,), (-7,), (-2,), (3,), (12,)),
        literals=tuple(range(-9, 10)),
        max_ast_cost=5,
    )


@cache
def _count_pressure() -> version_space.TypedSemanticVersionSpace:
    return version_space.TypedSemanticVersionSpace.build(
        signature=_signature(_LIST_INT),
        case_arguments=(([-8, -1, 0, 3],), ([2, 7],), ([-5],), ([],)),
        literals=tuple(range(-9, 10)),
        max_ast_cost=9,
    )


@cache
def _ambiguous_sum() -> version_space.TypedSemanticVersionSpace:
    return version_space.TypedSemanticVersionSpace.build(
        signature=_signature(_LIST_INT),
        case_arguments=(([-2, 0, 3],), ([1, -1],), ([],), ([4],)),
        literals=(-5, 5),
        max_ast_cost=9,
    )


def _evaluate_form(form: tuple[object, ...], x: int) -> int | bool:
    head = form[0]
    if head == "variable":
        return x
    if head == "literal":
        return int(form[1])
    if head == "boolean":
        return bool(form[1])
    if head == "absolute":
        return abs(int(_evaluate_form(form[1], x)))
    if head == "subtract":
        return int(_evaluate_form(form[1], x)) - int(_evaluate_form(form[2], x))
    if head == "less-than":
        return int(_evaluate_form(form[1], x)) < int(_evaluate_form(form[2], x))
    values = [
        int(_evaluate_form(item, x))
        for item in form[1]
    ]
    if head == "add":
        return sum(values)
    if head == "minimum":
        return min(values)
    if head == "maximum":
        return max(values)
    raise AssertionError(f"unhandled normal form {head!r}")


def test_proof_normalizer_preserves_add_multiplicity_and_exhaustive_parity() -> None:
    variable = ("variable", 0)
    zero = ("literal", 0)
    two = ("literal", 2)
    repeated = version_space._normalize_binary("add", variable, variable)
    repeated_plus_zero = version_space._normalize_binary("add", repeated, zero)
    repeated_subexpressions = version_space._normalize_binary(
        "add",
        version_space._normalize_binary("add", variable, two),
        version_space._normalize_binary("add", variable, two),
    )

    assert repeated == ("add", (variable, variable))
    assert repeated_plus_zero == repeated
    assert repeated_subexpressions == (
        "add",
        (("literal", 4), variable, variable),
    )
    for value in range(-32, 33):
        assert _evaluate_form(repeated, value) == value + value
        assert _evaluate_form(repeated_plus_zero, value) == value + value + 0
        assert _evaluate_form(repeated_subexpressions, value) == (value + 2) * 2

    lower = ("literal", -4)
    upper = ("literal", 6)
    canonical = version_space._normalize_binary(
        "minimum",
        version_space._normalize_binary("maximum", variable, lower),
        upper,
    )
    reversed_clamp = version_space._normalize_binary(
        "maximum",
        version_space._normalize_binary("minimum", variable, upper),
        lower,
    )
    assert reversed_clamp == canonical
    assert version_space._normalize_binary("minimum", variable, variable) == variable
    assert version_space._normalize_absolute(
        version_space._normalize_absolute(variable)
    ) == version_space._normalize_absolute(variable)
    for value in range(-32, 33):
        assert _evaluate_form(canonical, value) == min(max(value, -4), 6)

    python_binary = {
        "add": lambda left, right: left + right,
        "subtract": lambda left, right: left - right,
        "minimum": min,
        "maximum": max,
        "less-than": lambda left, right: left < right,
    }
    for left in range(-9, 10):
        for right in range(-9, 10):
            for production, operation in python_binary.items():
                folded = version_space._normalize_binary(
                    production,
                    ("literal", left),
                    ("literal", right),
                )
                assert _evaluate_form(folded, 0) == operation(left, right)

    for lower_value in range(-3, 4):
        for upper_value in range(lower_value, 4):
            forward = version_space._normalize_binary(
                "minimum",
                version_space._normalize_binary(
                    "maximum", variable, ("literal", lower_value)
                ),
                ("literal", upper_value),
            )
            reverse = version_space._normalize_binary(
                "maximum",
                version_space._normalize_binary(
                    "minimum", variable, ("literal", upper_value)
                ),
                ("literal", lower_value),
            )
            assert reverse == forward
            for value in range(-9, 10):
                assert _evaluate_form(forward, value) == min(
                    max(value, lower_value), upper_value
                )


def test_scalar_pressure_builds_complete_frontier_then_physically_selects_clamp() -> None:
    frontier = _scalar_pressure()
    before_work = frontier.work

    assert frontier.status == version_space.FRONTIER_READY
    assert frontier.work.maximum_admitted_complete_root_cost == 5
    assert frontier.work.admitted_canonical_recipes < (
        version_space.MAX_ADMITTED_CANONICAL_RECIPES
    )
    assert frontier.work.operator_predictions < version_space.MAX_OPERATOR_PREDICTIONS
    assert frontier.work.physical_races < version_space.MAX_PHYSICAL_RACES
    assert frontier.work.operator_skeletons < frontier.work.unique_canonical_recipes
    negative_nine = next(
        recipe
        for recipe in frontier.canonical_recipes
        if recipe.structural_normal_form == ("literal", -9)
    )
    positive_nine = next(
        recipe
        for recipe in frontier.canonical_recipes
        if recipe.structural_normal_form == ("literal", 9)
    )
    assert negative_nine.operator_skeleton_id == positive_nine.operator_skeleton_id

    decision = frontier.select((-4, -4, -2, 3, 6))
    assert decision.status == version_space.SELECTED
    assert decision.race.status == semantic_race.SELECTED
    assert decision.source is not None
    assert "min(" in decision.source and "max(" in decision.source
    assert decision.observation_class is not None
    selected_layers = tuple(
        layer.inside_cost
        for layer in decision.observation_class.cost_layers
        if decision.selected_recipe_id in layer.canonical_recipe_ids
    )
    assert selected_layers == (5,)
    assert decision.checkpoint_before_sha256 == decision.checkpoint_after_sha256

    alternate = frontier.select((-9, -8, -7, -6, -5))
    assert alternate.frontier_sha256 == decision.frontier_sha256
    assert frontier.work == before_work
    assert alternate.checkpoint_before_sha256 == alternate.checkpoint_after_sha256


def test_count_root_is_admitted_at_cost_six_before_later_pressure() -> None:
    frontier = _count_pressure()
    target_form = (
        "count",
        ("less-than", ("variable", 0), ("literal", 2)),
        ("variable", 0),
    )
    target_ids = {
        recipe.canonical_recipe_id
        for recipe in frontier.canonical_recipes
        if recipe.structural_normal_form == target_form
    }
    target_roots = tuple(
        root for root in frontier.root_expansions
        if root.admission.canonical_recipe_id in target_ids
    )

    assert len(target_roots) == 1
    target = target_roots[0]
    assert target.admission.inside_cost == 6
    assert target.admission.minimum_complete_root_cost == 6
    assert target.program.tokens == (
        "count",
        "bind",
        "less-than",
        "variable-0",
        "integer",
        "+",
        "2",
        "end",
        "variable-0",
    )
    target_class = next(
        observation
        for observation in frontier.observation_classes
        if observation.demand_use_role == "root"
        and any(
            target.admission.canonical_recipe_id in layer.canonical_recipe_ids
            for layer in observation.cost_layers
        )
    )
    assert target_class.physical_observation_sha256s == tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        ).sha256
        for value in (3, 0, 1, 0)
    )
    first_later = min(
        admission.ordinal
        for admission in frontier.admissions
        if admission.minimum_complete_root_cost > 6
    )
    assert target.admission.ordinal < first_later
    assert frontier.work.minimum_complete_root_cost == 9
    assert frontier.work.bound_name == "admitted-canonical-recipes"


def test_ambiguity_retains_plain_sum_and_cost_nine_clamp_map_sum() -> None:
    frontier = _ambiguous_sum()
    assert frontier.status == version_space.FRONTIER_READY

    decision = frontier.select((1, 0, 0, 4))
    assert decision.status == version_space.STOP_AMBIGUOUS
    assert decision.selected_recipe_id is None
    assert decision.program is None
    assert decision.race.status == semantic_race.STOP_MULTIPLE_EXACT
    assert len(decision.exact_recipe_ids) >= 2
    assert decision.observation_class is not None
    assert decision.observation_class.saturating_expansion_count == len(
        decision.exact_recipe_ids
    )

    exact = {
        expansion.admission.canonical_recipe_id: expansion
        for expansion in frontier.root_expansions
        if expansion.admission.canonical_recipe_id in decision.exact_recipe_ids
    }
    plain = tuple(
        item for item in exact.values()
        if item.program.tokens == ("sum", "variable-0")
    )
    clamp_maps = tuple(
        item for item in exact.values()
        if item.program.tokens[0] == "sum"
        and "map" in item.program.tokens
        and "minimum" in item.program.tokens
        and "maximum" in item.program.tokens
        and "-" in item.program.tokens
        and "+" in item.program.tokens
    )
    assert len(plain) == 1
    assert len(clamp_maps) == 1
    assert plain[0].admission.inside_cost == 2
    assert clamp_maps[0].admission.inside_cost == 9
    assert {2, 9}.issubset(
        {layer.inside_cost for layer in decision.observation_class.cost_layers}
    )


def test_repeated_rows_audit_actual_effect_calls_and_enforce_both_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = operator_successor.train_public_typed_operator_successor()[0]
    original_predict_effect = (
        operator_successor.FrozenTypedOperatorSuccessor._predict_effect
    )
    actual = {"calls": 0, "races": 0}

    def instrumented_predict_effect(
        self: operator_successor.FrozenTypedOperatorSuccessor,
        selected: output_cells.TypedOutputEncoding,
        action: object,
        scopes: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[
        output_cells.TypedOutputEncoding,
        tuple[operator_successor.OperatorRace, ...],
    ]:
        output, races = original_predict_effect(self, selected, action, scopes)
        actual["calls"] += 1
        actual["races"] += len(races)
        return output, races

    monkeypatch.setattr(
        operator_successor.FrozenTypedOperatorSuccessor,
        "_predict_effect",
        instrumented_predict_effect,
    )
    arguments = ((-3,), (-3,), (2,), (2,))
    build_kwargs = {
        "signature": _signature(_INT),
        "case_arguments": arguments,
        "literals": (-1, 0, 1),
        "max_ast_cost": 3,
        "operator": operator,
    }
    frontier = version_space.TypedSemanticVersionSpace.build(**build_kwargs)

    assert frontier.status == version_space.FRONTIER_READY
    assert actual["calls"] > 0
    assert frontier.work.operator_predictions == actual["calls"]
    assert frontier.work.physical_races == actual["races"]

    actual.update(calls=0, races=0)
    monkeypatch.setattr(version_space, "MAX_OPERATOR_PREDICTIONS", 1)
    prediction_stopped = version_space.TypedSemanticVersionSpace.build(**build_kwargs)
    assert prediction_stopped.status == version_space.STOP_FRONTIER_WORK_BOUND
    assert prediction_stopped.work.bound_name == "operator-predictions"
    assert prediction_stopped.work.operator_predictions == actual["calls"] == 1
    assert prediction_stopped.work.physical_races == actual["races"]

    actual.update(calls=0, races=0)
    monkeypatch.setattr(version_space, "MAX_OPERATOR_PREDICTIONS", 30_000)
    monkeypatch.setattr(version_space, "MAX_PHYSICAL_RACES", 1)
    race_stopped = version_space.TypedSemanticVersionSpace.build(**build_kwargs)
    assert race_stopped.status == version_space.STOP_FRONTIER_WORK_BOUND
    assert race_stopped.work.bound_name == "physical-races"
    assert race_stopped.work.operator_predictions == actual["calls"] == 1
    assert race_stopped.work.physical_races == actual["races"] == 1
    assert actual["races"] <= version_space.MAX_PHYSICAL_RACES


def test_selection_manifest_is_sealed_and_reauthenticated_before_race() -> None:
    frontier = version_space.TypedSemanticVersionSpace.build(
        signature=_signature(_INT),
        case_arguments=((-3,), (2,)),
        literals=(-1, 0, 1),
        max_ast_cost=3,
    )
    recipe_ids = frontier.root_recipe_ids
    outputs = frontier._selection_root_outputs
    assert len(recipe_ids) > 1
    assert frontier.selection_manifest_sha256 == version_space._selection_manifest_digest(
        frontier.signature,
        recipe_ids,
        outputs,
        frontier._selection_root_programs,
    )

    with pytest.raises(AttributeError, match="sealed"):
        frontier._selection_root_recipe_ids = tuple(reversed(recipe_ids))
    with pytest.raises(AttributeError, match="sealed"):
        frontier._selection_root_outputs = (tuple(), *outputs[1:])
    with pytest.raises(TypeError):
        frontier._root_class_by_recipe[recipe_ids[0]] = None

    object.__setattr__(
        frontier,
        "_selection_root_recipe_ids",
        tuple(reversed(recipe_ids)),
    )
    try:
        with pytest.raises(RuntimeError, match="manifest authentication failed"):
            frontier.select((-3, 2))
    finally:
        object.__setattr__(frontier, "_selection_root_recipe_ids", recipe_ids)

    object.__setattr__(
        frontier,
        "_selection_root_outputs",
        (tuple(), *outputs[1:]),
    )
    try:
        with pytest.raises(RuntimeError, match="manifest authentication failed"):
            frontier.select((-3, 2))
    finally:
        object.__setattr__(frontier, "_selection_root_outputs", outputs)

    tampered_signature = TypedSignature(
        entry_point=frontier.signature.entry_point,
        parameters=frontier.signature.parameters,
        result=ProgramType.Bool(),
    )
    original_signature = frontier.signature
    object.__setattr__(frontier, "signature", tampered_signature)
    try:
        with pytest.raises(RuntimeError, match="manifest authentication failed"):
            frontier.select((-3, 2))
    finally:
        object.__setattr__(frontier, "signature", original_signature)

    assert frontier.select((-3, 2)).status in {
        version_space.SELECTED,
        version_space.STOP_AMBIGUOUS,
    }
