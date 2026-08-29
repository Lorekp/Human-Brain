import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.r3_pilot import public_typed_prompt_factors as factors
from humanbrain.typed_program import FactorAction


MATCH_2 = FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 2)
WRONG_3 = FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 3)
NO_LITERAL = FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None)


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
        ("Increase the input by 21.", factors.STATUS_OUT_OF_RANGE, (21,)),
        (
            "Use 2 for the offset but reject values above 21.",
            factors.STATUS_OUT_OF_RANGE,
            (2, 21),
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
