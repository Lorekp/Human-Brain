from experiments.r3_pilot import diagnostic_task_refreshed_typed_decoder as diagnostic


def test_task_refreshed_decoder_is_a_real_prompt_last_cell_race() -> None:
    result = diagnostic.run_diagnostic()

    assert set(result.curriculum_prompts).isdisjoint(result.target_prompts)
    assert result.training_episodes == 12 * (2 + 6 + 2 + 1)
    for decoded in (*result.fitted, *result.control):
        assert decoded.decisions
        assert decoded.persistent_before == decoded.persistent_after
        assert all(decision.prompt_last for decision in decoded.decisions)
        assert all(
            decision.accepted_token in decision.allowed_tokens
            and decision.winner_cell_id
            == diagnostic.tp._TOKEN_TO_ID[decision.accepted_token]
            for decision in decoded.decisions
        )
    if result.verdict == "FAIL":
        assert any(item.first_mismatch is not None for item in result.fitted)

    print(result.verdict, result.reason)
