from humanbrain import typed_program as tp

from experiments.r3_pilot.public_typed_execution_edit_loop import (
    run_public_canary,
)


def test_fixed_seed_cell_edit_loop_records_spent_no_improvement() -> None:
    result = run_public_canary()

    assert result.seed == 17
    assert result.budget == len(result.steps) == 8
    assert result.initial_program.tokens == ("variable-0",)
    assert result.initial_race.winner == "variable-0"
    assert result.initial_score == 0
    assert not result.strict_improvement
    assert result.final_score == result.initial_score == 0
    assert result.final_program.tokens == ("variable-0",)
    assert result.controller_before_sha256 != result.controller_after_sha256
    assert tp.compile(result.final_program).source

    assert all(step.races for step in result.steps)
    assert all(
        race.candidates[race.winner_cell] == race.winner
        and race.spike_step >= 0
        and race.simultaneous_spike_candidates >= 1
        and len(race.candidates) == len(race.currents_pa)
        for step in result.steps
        for race in step.races
    )
    assert all(step.decision == "UNDO" for step in result.steps)
    assert all(step.signed_delta == 0 for step in result.steps)
    assert all(
        step.signed_delta == step.proposal_score - step.before_score
        for step in result.steps
    )
    assert all(
        outcome.arguments and outcome.error is None
        for step in result.steps
        for outcome in step.outcomes
    )
    assert all(not step.local_updates for step in result.steps)
    assert tuple(
        (step.operation, step.literal, step.orientation) for step in result.steps
    ) == (
        ("subtract", -8, "existing-left"),
        ("absolute", None, None),
        ("add", 0, "existing-right"),
        ("subtract", 8, "existing-left"),
        ("absolute", None, None),
        ("add", -4, "existing-right"),
        ("subtract", 7, "existing-left"),
        ("absolute", None, None),
    )
