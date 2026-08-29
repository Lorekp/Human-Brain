import ast
from pathlib import Path

import pytest
from experiments.r3_pilot import public_typed_prompt_factors as factors
from humanbrain.typed_program import FactorAction
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

MATCH_2 = FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 2)
WRONG_3 = FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 3)
NO_LITERAL = FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None)

DONE = FactorAction((), "DONE", "NONE", "NONE", "NONE", None)
SELECTED_ALIAS = FactorAction(
    (), "APPLY2", "subtract", "SCOPE1", "SELECTED", None
)
A_MINUS_B = FactorAction((), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None)
B_MINUS_A = FactorAction((), "APPLY2", "subtract", "SCOPE0", "SCOPE1", None)
X_LESS_2 = FactorAction((), "APPLY2", "less-than", "SCOPE0", "LITERAL", 2)
TWO_LESS_X = FactorAction((), "APPLY2", "less-than", "LITERAL", "SCOPE0", 2)
X_LESS_3 = FactorAction((), "APPLY2", "less-than", "SCOPE0", "LITERAL", 3)
OUT_OF_RANGE_LITERAL = factors.output_cells.INT_MAX + 1

BINDING_ACTIONS = (
    DONE,
    SELECTED_ALIAS,
    A_MINUS_B,
    B_MINUS_A,
    X_LESS_2,
    TWO_LESS_X,
    X_LESS_3,
)

ABS_LEFT_SEQUENCE = factors.PromptSequenceEvidence(
    first_action=FactorAction(
        (), "APPLY2", "subtract", "SELECTED", "SCOPE0", None
    ),
    first_selected_source="SCOPE1",
    second_action=FactorAction(
        (0,), "APPLY1", "absolute", "SELECTED", "NONE", None
    ),
    second_selected_source="SCOPE1",
)
ABS_RIGHT_SEQUENCE = factors.PromptSequenceEvidence(
    first_action=FactorAction(
        (), "APPLY2", "subtract", "SCOPE1", "SELECTED", None
    ),
    first_selected_source="SCOPE0",
    second_action=FactorAction(
        (1,), "APPLY1", "absolute", "SELECTED", "NONE", None
    ),
    second_selected_source="SCOPE0",
)
FLAT_SUBTRACT_SEQUENCE = factors.PromptSequenceEvidence(
    first_action=A_MINUS_B,
    first_selected_source=None,
    second_action=DONE,
    second_selected_source=None,
)
NONROOT_SEQUENCE = factors.PromptSequenceEvidence(
    first_action=FactorAction(
        (0,), "APPLY2", "subtract", "SELECTED", "SCOPE0", None
    ),
    first_selected_source="SCOPE1",
    second_action=FactorAction(
        (0, 0), "APPLY1", "absolute", "SELECTED", "NONE", None
    ),
    second_selected_source="SCOPE1",
)
MAP_BIND = FactorAction(
    (), "MAP_BIND", "map", "BOUND0", "SELECTED", None
)
MAP_SUBTRACT_TWO_SEQUENCE = factors.PromptSequenceEvidence(
    first_action=MAP_BIND,
    first_selected_source="SCOPE0",
    second_action=FactorAction(
        (0, 0), "APPLY2", "subtract", "SELECTED", "LITERAL", 2
    ),
    second_selected_source="BOUND0",
)
MAP_ADD_THREE_SEQUENCE = factors.PromptSequenceEvidence(
    first_action=MAP_BIND,
    first_selected_source="SCOPE0",
    second_action=FactorAction(
        (0, 0), "APPLY2", "add", "SELECTED", "LITERAL", 3
    ),
    second_selected_source="BOUND0",
)
MAP_SEQUENCE_EVIDENCE = (
    MAP_SUBTRACT_TWO_SEQUENCE,
    MAP_ADD_THREE_SEQUENCE,
)

SEQUENCE_EVIDENCE = (
    ABS_LEFT_SEQUENCE,
    ABS_RIGHT_SEQUENCE,
    FLAT_SUBTRACT_SEQUENCE,
    NONROOT_SEQUENCE,
)


def _signature(*names: str, result: ProgramType | None = None) -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "solve",
        tuple(ProgramParameter(name, integer) for name in names),
        integer if result is None else result,
    )


def _list_signature(name: str = "values") -> TypedSignature:
    list_int = ProgramType.List(ProgramType.Int())
    return TypedSignature(
        "solve",
        (ProgramParameter(name, list_int),),
        list_int,
    )


def test_natural_paraphrases_extract_the_same_physical_literal_goal() -> None:
    first = factors.observe_prompt_factors("Increase the supplied value by 2.")
    second = factors.observe_prompt_factors("Return the input after adding 2 to it.")

    assert first.supported and second.supported
    assert first.literal == second.literal == 2
    assert first.integer_lexemes == second.integer_lexemes == ("2",)
    assert first.distinct_literals == second.distinct_literals == (2,)
    assert first.literal_goal.sha256 == second.literal_goal.sha256
    assert first.literal_goal.output_type is factors.output_cells.OutputType.INT
    assert first.literal_goal.supported
    assert first.sha256 != second.sha256

    first_projection = factors.FixedPromptLiteralProjection.build(first)
    second_projection = factors.FixedPromptLiteralProjection.build(second)
    assert first_projection.sha256 == second_projection.sha256
    assert first_projection.race(MATCH_2).spiked


def test_signed_and_repeated_literals_are_normalized_by_value() -> None:
    signed = factors.observe_prompt_factors("Add -3 to every supplied integer.")
    repeated = factors.observe_prompt_factors(
        "Use offset 2, and keep that same offset 2 for every case."
    )

    assert signed.integer_lexemes == ("-3",)
    assert signed.distinct_literals == (-3,)
    assert signed.literal == -3
    assert signed.supported
    assert factors.FixedPromptLiteralProjection.build(signed).race(
        FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", -3)
    ).spiked

    assert repeated.integer_lexemes == ("2", "2")
    assert repeated.distinct_literals == (2,)
    assert repeated.literal == 2
    assert repeated.supported
    assert len(repeated.literal_encodings) == 1


@pytest.mark.parametrize(
    ("prompt", "status", "distinct"),
    (
        ("Clamp the input between -3 and 2.", factors.STATUS_AMBIGUOUS, (-3, 2)),
        ("Return the absolute value.", factors.STATUS_NO_LITERAL, ()),
        (
            f"Increase the input by {OUT_OF_RANGE_LITERAL}.",
            factors.STATUS_OUT_OF_RANGE,
            (OUT_OF_RANGE_LITERAL,),
        ),
        (
            f"Use 2 for the offset but reject values above {OUT_OF_RANGE_LITERAL}.",
            factors.STATUS_OUT_OF_RANGE,
            (2, OUT_OF_RANGE_LITERAL),
        ),
    ),
)
def test_ambiguous_missing_and_out_of_range_prompts_are_physical_unknown(
    prompt: str,
    status: str,
    distinct: tuple[int, ...],
) -> None:
    observation = factors.observe_prompt_factors(prompt)

    assert observation.status == status
    assert observation.distinct_literals == distinct
    assert observation.literal is None
    assert not observation.supported
    assert not observation.literal_goal.supported
    assert observation.literal_goal.unknown_active
    assert observation.literal_goal.int_cells[factors.output_cells.INT_UNKNOWN_CELL] == 1.0


def test_fixed_literal_projection_matches_only_the_exact_candidate_factor() -> None:
    observation = factors.observe_prompt_factors("Add 2 to the input.")
    projection = factors.FixedPromptLiteralProjection.build(observation)

    matching = projection.race(MATCH_2)
    wrong = projection.race(WRONG_3)
    missing = projection.race(NO_LITERAL)
    unknown_projection = factors.FixedPromptLiteralProjection.build(
        factors.observe_prompt_factors("Return the input unchanged.")
    )

    assert matching.spiked
    assert matching.matched_factor_cells == matching.active_goal_factors == 2
    assert matching.current_pa == 5_000.0
    assert not wrong.spiked and wrong.matched_factor_cells == 1
    assert not missing.spiked
    assert not unknown_projection.race(MATCH_2).spiked


def test_observations_projections_and_races_are_deterministic_and_immutable() -> None:
    first = factors.observe_prompt_factors("Add 2 to the input.")
    second = factors.observe_prompt_factors("Add 2 to the input.")
    first_projection = factors.FixedPromptLiteralProjection.build(first)
    second_projection = factors.FixedPromptLiteralProjection.build(second)
    first_race = first_projection.race(MATCH_2)
    second_race = second_projection.race(MATCH_2)

    assert first.sha256 == second.sha256
    assert first.tokenizer_sha256 == second.tokenizer_sha256
    assert first.output_projection_sha256 == second.output_projection_sha256
    assert first_projection.sha256 == second_projection.sha256
    assert first_race.sha256 == second_race.sha256
    assert not first.literal_goal.cells.flags.writeable
    assert not first_projection.detector.weights.flags.writeable
    assert not first_race.spike_mask.flags.writeable
    with pytest.raises(ValueError):
        first_race.spike_mask[0] = 0.0


@pytest.mark.parametrize(
    ("prompt", "expected"),
    (
        (
            "Given integers a and b, return a minus b.",
            factors.PromptOperandPattern("subtract", "SCOPE1", "SCOPE0", None),
        ),
        (
            "Given integers a and b, return b minus a.",
            factors.PromptOperandPattern("subtract", "SCOPE0", "SCOPE1", None),
        ),
        (
            "Subtract b from a.",
            factors.PromptOperandPattern("subtract", "SCOPE1", "SCOPE0", None),
        ),
        (
            "Return a-b.",
            factors.PromptOperandPattern("subtract", "SCOPE1", "SCOPE0", None),
        ),
    ),
)
def test_two_parameter_prompt_order_maps_to_de_bruijn_sources(
    prompt: str,
    expected: factors.PromptOperandPattern,
) -> None:
    projection = factors.FixedPromptBindingProjection.build(BINDING_ACTIONS)
    decision = projection.project(prompt, _signature("a", "b"))

    assert decision.status == factors.PROJECTION_STATUS_PROJECTED
    assert decision.observation.pattern == expected
    assert decision.physical_winner == expected
    expected_action = A_MINUS_B if expected.left_source == "SCOPE1" else B_MINUS_A
    expected_cell = BINDING_ACTIONS.index(expected_action)
    assert decision.matching_action_cells == (expected_cell,)
    assert decision.action_currents_pa[expected_cell] == 5_000.0
    assert decision.action_currents_pa[BINDING_ACTIONS.index(SELECTED_ALIAS)] == 0.0
    assert decision.simultaneous_spike_candidates == 1


def test_established_first_second_fixture_uses_declared_order_not_names() -> None:
    observation = factors.observe_prompt_bindings(
        "Given integers first and second, return first minus second.",
        _signature("first", "second"),
    )

    assert observation.supported
    assert observation.pattern == factors.PromptOperandPattern(
        "subtract", "SCOPE1", "SCOPE0", None
    )


@pytest.mark.parametrize(
    ("prompt", "expected_action", "left", "right"),
    (
        (
            "Given integer x, return whether x is less than 2.",
            X_LESS_2,
            "SCOPE0",
            "LITERAL",
        ),
        (
            "Given integer x, return whether 2 is less than x.",
            TWO_LESS_X,
            "LITERAL",
            "SCOPE0",
        ),
        ("Return whether x < 2.", X_LESS_2, "SCOPE0", "LITERAL"),
    ),
)
def test_bool_comparison_preserves_literal_orientation(
    prompt: str,
    expected_action: FactorAction,
    left: str,
    right: str,
) -> None:
    projection = factors.FixedPromptBindingProjection.build(BINDING_ACTIONS)
    decision = projection.project(
        prompt,
        _signature("x", result=ProgramType.Bool()),
    )

    assert decision.status == factors.PROJECTION_STATUS_PROJECTED
    assert decision.observation.pattern == factors.PromptOperandPattern(
        "less-than", left, right, 2
    )
    assert decision.matching_action_cells == (BINDING_ACTIONS.index(expected_action),)


@pytest.mark.parametrize(
    ("prompt", "status", "observation_status"),
    (
        (
            "Return a minus b or b minus a.",
            factors.PROJECTION_STATUS_STOP_AMBIGUOUS,
            factors.BINDING_STATUS_AMBIGUOUS,
        ),
        (
            "Return a divided by b.",
            factors.PROJECTION_STATUS_STOP_UNKNOWN,
            factors.BINDING_STATUS_UNKNOWN,
        ),
    ),
)
def test_ambiguous_and_unknown_operand_prompts_physically_stop(
    prompt: str,
    status: str,
    observation_status: str,
) -> None:
    projection = factors.FixedPromptBindingProjection.build(BINDING_ACTIONS)
    decision = projection.project(prompt, _signature("a", "b"))

    assert decision.status == status
    assert decision.observation.status == observation_status
    assert decision.physical_winner == factors.PHYSICAL_STOP
    assert decision.winner_cell == decision.stop_cell
    assert decision.race_currents_pa[decision.stop_cell] == 5_000.0
    assert not any(decision.action_currents_pa)
    assert decision.matching_action_cells == ()
    assert decision.simultaneous_spike_candidates == 1


def test_supported_prompt_absent_from_fixed_factor_bank_physically_stops() -> None:
    projection = factors.FixedPromptBindingProjection.build(BINDING_ACTIONS)
    decision = projection.project(
        "Return whether x is less than 4.",
        _signature("x", result=ProgramType.Bool()),
    )

    assert decision.observation.supported
    assert decision.status == factors.PROJECTION_STATUS_STOP_UNSUPPORTED
    assert decision.physical_winner == factors.PHYSICAL_STOP
    assert decision.winner_cell == decision.stop_cell


def test_tier_with_no_prompt_observable_action_has_a_physical_stop_bank() -> None:
    projection = factors.FixedPromptBindingProjection.build((DONE, SELECTED_ALIAS))
    decision = projection.project(
        "Given integers a and b, return a minus b.",
        _signature("a", "b"),
    )

    assert projection.patterns == ()
    assert projection.race_weights_pa.shape == (1, 1)
    assert projection.action_weights_pa.shape == (2, 0)
    assert decision.status == factors.PROJECTION_STATUS_STOP_UNSUPPORTED
    assert decision.physical_winner == factors.PHYSICAL_STOP
    assert decision.winner_cell == decision.stop_cell == 0


def test_binding_projection_is_readonly_and_checkpoint_stable() -> None:
    projection = factors.FixedPromptBindingProjection.build(BINDING_ACTIONS)
    before = projection.live_checkpoint_sha256
    decision = projection.project(
        "Given integers a and b, return a minus b.",
        _signature("a", "b"),
    )

    assert before == projection.sha256 == projection.live_checkpoint_sha256
    assert decision.checkpoint_before_sha256 == decision.checkpoint_after_sha256
    assert decision.checkpoint_after_sha256 == projection.sha256
    assert not projection.race_weights_pa.flags.writeable
    assert not projection.action_weights_pa.flags.writeable
    with pytest.raises(ValueError):
        projection.action_weights_pa[0, 0] = 1.0


@pytest.mark.parametrize(
    ("prompt", "expected_cell", "nested_side"),
    (
        (
            "Subtract beta from the absolute value of alpha.",
            0,
            "left",
        ),
        ("Return abs(alpha) minus beta.", 0, "left"),
        (
            "Subtract the absolute value of beta from alpha.",
            1,
            "right",
        ),
        ("Return alpha minus abs(beta).", 1, "right"),
    ),
)
def test_compositional_prompt_projects_one_ordered_root_child_sequence(
    prompt: str,
    expected_cell: int,
    nested_side: str,
) -> None:
    projection = factors.FixedPromptSequenceProjection.build(SEQUENCE_EVIDENCE)
    decision = projection.project(prompt, _signature("alpha", "beta"))

    assert decision.status == factors.PROJECTION_STATUS_PROJECTED
    assert decision.matching_sequence_cells == (expected_cell,)
    assert decision.sequence_currents_pa[expected_cell] == 5_000.0
    assert sum(value > 0.0 for value in decision.sequence_currents_pa) == 1
    assert decision.simultaneous_spike_candidates == 1
    assert decision.observation.pattern is not None
    if nested_side == "left":
        nested = decision.observation.pattern.left
        plain = decision.observation.pattern.right
    else:
        nested = decision.observation.pattern.right
        plain = decision.observation.pattern.left
    assert nested.unary_operators == ("absolute",)
    assert plain.unary_operators == ()
    before = projection.sha256
    assert decision.checkpoint_before_sha256 == before
    assert decision.checkpoint_after_sha256 == before
    assert projection.live_checkpoint_sha256 == before
    assert not projection.race_weights_pa.flags.writeable
    assert not projection.sequence_weights_pa.flags.writeable


@pytest.mark.parametrize(
    ("prompt", "expected_cell", "operator", "literal"),
    (
        (
            "Return a new list with two subtracted from every value.",
            0,
            "subtract",
            2,
        ),
        ("Add 3 to each element and return the list.", 1, "add", 3),
        ("Return every element plus 3.", 1, "add", 3),
        ("Return a list with 3 added to every item.", 1, "add", 3),
    ),
)
def test_map_body_prompt_projects_bound_binary_sequence(
    prompt: str,
    expected_cell: int,
    operator: str,
    literal: int,
) -> None:
    projection = factors.FixedPromptSequenceProjection.build(MAP_SEQUENCE_EVIDENCE)
    decision = projection.project(prompt, _list_signature())

    assert decision.status == factors.PROJECTION_STATUS_PROJECTED
    assert decision.matching_sequence_cells == (expected_cell,)
    assert decision.observation.pattern is not None
    assert decision.observation.pattern.container_schema == "MAP_BIND"
    assert decision.observation.pattern.operator == operator
    assert decision.observation.pattern.left == factors.PromptComposedOperand(
        "BOUND0", None, ()
    )
    assert decision.observation.pattern.right == factors.PromptComposedOperand(
        "LITERAL", literal, ()
    )
    assert decision.sequence_currents_pa[expected_cell] == 5_000.0
    assert projection.live_checkpoint_sha256 == projection.sha256


def test_map_body_sequence_does_not_match_a_scalar_binary_prompt() -> None:
    projection = factors.FixedPromptSequenceProjection.build(MAP_SEQUENCE_EVIDENCE)
    decision = projection.project("Return value plus 3.", _signature("value"))

    assert decision.status == factors.PROJECTION_STATUS_STOP_UNSUPPORTED
    assert decision.observation.pattern is not None
    assert decision.observation.pattern.container_schema is None
    assert decision.matching_sequence_cells == ()


def test_conflicting_nested_relations_physically_stop_without_sequence_current(
) -> None:
    projection = factors.FixedPromptSequenceProjection.build(SEQUENCE_EVIDENCE)
    decision = projection.project(
        "Return abs(alpha) minus beta or alpha minus abs(beta).",
        _signature("alpha", "beta"),
    )

    assert decision.status == factors.PROJECTION_STATUS_STOP_AMBIGUOUS
    assert decision.observation.status == factors.BINDING_STATUS_AMBIGUOUS
    assert len(decision.observation.patterns) == 2
    assert decision.physical_winner == factors.PHYSICAL_STOP
    assert decision.winner_cell == decision.stop_cell
    assert decision.race_currents_pa[decision.stop_cell] == 5_000.0
    assert decision.matching_sequence_cells == ()
    assert not any(decision.sequence_currents_pa)


def test_module_has_no_program_output_or_candidate_ranking_seam() -> None:
    source = Path(factors.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    tree = ast.parse(source)
    calls = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))

    for forbidden in (
        "desired_output",
        "task_family",
        "argmax",
        "argpartition",
    ):
        assert forbidden not in lowered
    forbidden_calls = {
        "apply_factor_action",
        "replace_subtree",
        "program_edit_sites",
        "factor_action_candidates",
        "sorted",
        "sort",
        "compile",
        "exec",
    }
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id in forbidden_calls
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr in forbidden_calls
        and not (
            isinstance(call.func.value, ast.Name)
            and call.func.value.id == "re"
            and call.func.attr == "compile"
        )
        for call in calls
    )
    assert "typed_program import factoraction" in lowered
    assert "typed_program import program" not in lowered
