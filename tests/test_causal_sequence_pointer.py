from __future__ import annotations

import pytest
from humanbrain.causal_sequence_pointer import (
    CausalSequenceAblation,
    CausalSequenceConfig,
    CausalSequencePointerError,
    CausalSequencePointerModel,
    CausalSequenceTrainingExample,
)
from humanbrain.pointer_binding import SymbolInventory, materialize, pointerize
from humanbrain.tokenizer import TokenizerConfig


def _config(
    ablation: CausalSequenceAblation = CausalSequenceAblation.NONE,
) -> CausalSequenceConfig:
    return CausalSequenceConfig(
        tokenizer=TokenizerConfig(max_phrases=128, min_frequency=2),
        max_observation_order=2,
        max_observation_features=128,
        max_prefix_order=3,
        max_role_cursors=8,
        max_roles=16,
        ablation=ablation,
        random_seed=41,
    )


def _row(
    prompt: str,
    cue: str,
    output: str,
    inventory: SymbolInventory,
    provenance: str,
) -> CausalSequenceTrainingExample:
    return CausalSequenceTrainingExample(
        prompt=prompt,
        output=output,
        inventory=inventory,
        cue_text=cue,
        provenance_id=provenance,
    )


def _composition_rows() -> tuple[CausalSequenceTrainingExample, ...]:
    inventory = SymbolInventory(("left", "right"))
    observations = (
        ("red", "round", "A", "R", "rr-one"),
        ("red", "round", "A", "R", "rr-two"),
        ("red", "square", "A", "S", "rs-one"),
        ("red", "square", "A", "S", "rs-two"),
        ("blue", "round", "B", "R", "br-one"),
        ("blue", "round", "B", "R", "br-two"),
    )
    return tuple(
        _row(
            prompt=f"compose independent observed attributes {witness}",
            cue=f"left {first} right {second} witness {witness}",
            output=f"left:{first_result}:right:{second_result}",
            inventory=inventory,
            provenance=f"composition-{witness}",
        )
        for first, second, first_result, second_result, witness in observations
    )


def _revision_rows() -> tuple[CausalSequenceTrainingExample, ...]:
    inventory = SymbolInventory(("slot",))
    return (
        _row(
            "one episodic observation",
            "switch neutral witness",
            "A",
            inventory,
            "episodic-a",
        ),
        _row(
            "first reusable observation",
            "violet route witness x",
            "B",
            inventory,
            "reusable-b-x",
        ),
        _row(
            "second reusable observation",
            "violet route witness y",
            "B",
            inventory,
            "reusable-b-y",
        ),
    )


def _contrast_config(
    ablation: CausalSequenceAblation = CausalSequenceAblation.NONE,
    *,
    max_contrast_output_bytes: int = 4096,
    max_contrast_candidate_episodes: int = 256,
) -> CausalSequenceConfig:
    return CausalSequenceConfig(
        tokenizer=TokenizerConfig(max_phrases=256, min_frequency=1),
        max_observation_order=2,
        max_observation_features=256,
        max_prefix_order=3,
        max_role_cursors=8,
        max_roles=16,
        max_contrast_output_bytes=max_contrast_output_bytes,
        max_contrast_candidate_episodes=max_contrast_candidate_episodes,
        ablation=ablation,
        random_seed=53,
    )


def _contrast_rows() -> tuple[CausalSequenceTrainingExample, ...]:
    inventory = SymbolInventory(("slot",))
    prompt = "compose two independently observed facets"
    return (
        _row(prompt, "palette amber form round", "aa/rr", inventory, "contrast-base"),
        _row(
            prompt,
            "palette cobalt form round",
            "ULTRAMARINE/rr",
            inventory,
            "contrast-palette",
        ),
        _row(prompt, "palette amber form square", "aa/s", inventory, "contrast-form"),
    )


def _contrast_query() -> tuple[str, str, SymbolInventory]:
    return (
        "compose two independently observed facets",
        "palette cobalt form square",
        SymbolInventory(("live",)),
    )


def test_exact_episode_assembly_physically_replays_long_literal_atoms() -> None:
    inventory = SymbolInventory(("slot",))
    output = "!" * 257
    row = _row(
        "retain one long particular",
        "exact long memory witness",
        output,
        inventory,
        "long-exact",
    )
    model = CausalSequencePointerModel.grow((row,), _config())

    assert model.raw_complete(
        row.prompt,
        inventory,
        cue_text=row.cue_text,
        max_new_tokens=257,
    ) == pointerize(output, inventory)

    trace = model.last_trace
    assert trace is not None and trace.selected_eos and not trace.selected_stop
    assert trace.steps[0].phase == "exact-episode"
    atom_steps = [step for step in trace.steps if step.phase == "exact-atom"]
    assert len(atom_steps) == 257
    assert len(trace.emitted_token_ids) == 257
    assert trace.steps[-1].phase == "exact-eos"
    assert all(len(step.candidate_currents_pa) == 2 for step in trace.steps)
    assert all(
        dependency < step.index for step in trace.steps for dependency in step.depends_on_steps
    )
    model.resolve_pending_neutral(row.prompt, inventory, row.cue_text)


def test_exact_assembly_ablation_physically_stops_with_identical_learning() -> None:
    inventory = SymbolInventory(("slot",))
    row = _row(
        "retain this particular",
        "exact ablation witness",
        "remembered-output",
        inventory,
        "exact-ablation",
    )
    treatment = CausalSequencePointerModel.grow((row,), _config())
    control = CausalSequencePointerModel.grow(
        (row,),
        _config(CausalSequenceAblation.EXACT_ASSEMBLY_OFF),
    )

    assert treatment.training_digest == control.training_digest
    assert treatment.raw_complete(
        row.prompt,
        inventory,
        cue_text=row.cue_text,
        max_new_tokens=64,
    ) == pointerize(row.output, inventory)
    treatment.resolve_pending_neutral(row.prompt, inventory, row.cue_text)

    assert control.raw_complete(
        row.prompt,
        inventory,
        cue_text=row.cue_text,
        max_new_tokens=64,
    ).startswith("<stop>")
    trace = control.last_trace
    assert trace is not None and trace.selected_stop and not trace.selected_eos
    assert len(trace.steps) == 1
    assert trace.steps[0].phase == "exact-episode"
    assert trace.steps[0].winner_operation == "STOP"
    assert len(trace.steps[0].candidate_currents_pa) == 1


def test_contrast_certificates_compose_safe_singletons_with_variable_lengths() -> None:
    rows = _contrast_rows()
    model = CausalSequencePointerModel.grow(rows, _contrast_config())
    prompt, cue, inventory = _contrast_query()

    result = model.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64)

    assert result == "ULTRAMARINE/s"
    assert result not in {row.output for row in rows}
    trace = model.last_trace
    assert trace is not None and trace.selected_eos and not trace.selected_stop
    assert sum(step.phase == "contrast-factor" for step in trace.steps) == 2
    assert sum(step.phase == "contrast-segment" for step in trace.steps) == 2
    assert any(step.phase == "contrast-template-byte" for step in trace.steps)
    assert trace.steps[-1].phase == "contrast-eos"
    assert all(
        dependency < step.index for step in trace.steps for dependency in step.depends_on_steps
    )
    model.resolve_pending_neutral(prompt, inventory, cue)


def test_common_prompt_features_cannot_recruit_a_contrast_pair() -> None:
    inventory = SymbolInventory(("slot",))
    prompt = "the shared prompt must never be contrast evidence"
    rows = (
        _row(prompt, "AAAAAAAA", "left", inventory, "prompt-a"),
        _row(prompt, "BBBBBBBB", "right", inventory, "prompt-b"),
    )
    model = CausalSequencePointerModel.grow(rows, _contrast_config())

    result = model.raw_complete(
        prompt,
        inventory,
        cue_text="BBBBBBBBX",
        max_new_tokens=32,
    )

    assert result.startswith("<stop>")
    trace = model.last_trace
    assert trace is not None and trace.selected_stop
    assert all(not step.phase.startswith("contrast-") for step in trace.steps)


def test_contrast_conflict_physically_stops_instead_of_using_frequency() -> None:
    inventory = SymbolInventory(("slot",))
    prompt = "bind one observed contrast"
    rows = (
        _row(prompt, "signal quiet lane west", "aa/rr", inventory, "conflict-base"),
        _row(prompt, "signal bright lane west", "FIRST/rr", inventory, "conflict-first"),
        _row(prompt, "signal bright lane west", "SECOND/rr", inventory, "conflict-second"),
    )
    model = CausalSequencePointerModel.grow(rows, _contrast_config())

    result = model.raw_complete(
        prompt,
        inventory,
        cue_text="signal bright lane west now",
        max_new_tokens=64,
    )

    assert result.startswith("<stop>")
    trace = model.last_trace
    assert trace is not None and trace.selected_stop
    assert trace.steps[-1].phase == "contrast-conflict"
    assert trace.steps[-1].winner_operation == "STOP"
    assert len(trace.steps[-1].candidate_currents_pa) == 1


def test_contrast_conflict_stops_even_with_two_to_one_replacement_support() -> None:
    inventory = SymbolInventory(("slot",))
    prompt = "bind independently observed switches"
    rows = (
        _row(
            prompt,
            "palette amber form round tone quiet",
            "BASE/tail",
            inventory,
            "u-base",
        ),
        _row(
            prompt,
            "palette cobalt form round tone quiet",
            "FIRST/tail",
            inventory,
            "u-a",
        ),
        _row(
            prompt,
            "palette amber form square tone quiet",
            "FIRST/tail",
            inventory,
            "u-b",
        ),
        _row(
            prompt,
            "palette amber form round tone loud",
            "SECOND/tail",
            inventory,
            "u-c",
        ),
    )
    model = CausalSequencePointerModel.grow(rows, _contrast_config())

    result = model.raw_complete(
        prompt,
        inventory,
        cue_text="palette cobalt form square tone loud",
        max_new_tokens=64,
    )

    assert result.startswith("<stop>")
    trace = model.last_trace
    assert trace is not None and trace.selected_stop
    assert trace.steps[-1].phase == "contrast-conflict"
    assert trace.steps[-1].winner_operation == "STOP"


@pytest.mark.parametrize(
    "ablation, terminal_phase",
    (
        (CausalSequenceAblation.CONTRAST_OFF, "contrast-template"),
        (CausalSequenceAblation.RECURRENCE_OFF, "contrast-segment"),
    ),
)
def test_contrast_controls_share_learned_state_and_physically_stop(
    ablation: CausalSequenceAblation,
    terminal_phase: str,
) -> None:
    rows = _contrast_rows()
    treatment = CausalSequencePointerModel.grow(rows, _contrast_config())
    control = CausalSequencePointerModel.grow(rows, _contrast_config(ablation))
    prompt, cue, inventory = _contrast_query()

    assert treatment.training_digest == control.training_digest
    result = control.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64)

    assert result.startswith("<stop>")
    trace = control.last_trace
    assert trace is not None and trace.selected_stop
    assert trace.steps[-1].phase == terminal_phase
    assert trace.steps[-1].winner_operation == "STOP"


def test_contrast_success_grows_online_and_retains_old_and_new_episodes() -> None:
    rows = _contrast_rows()
    model = CausalSequencePointerModel.grow(rows, _contrast_config())
    prompt, cue, inventory = _contrast_query()
    result = model.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64)
    prior_digest = model.training_digest

    update = model.learn(prompt, result, inventory, cue_text=cue)

    assert update.observed_success_sequence
    assert update.training_digest != prior_digest
    later_prompt = "retrieve the successful composition without replay"
    assert model.raw_complete(later_prompt, inventory, cue_text=cue, max_new_tokens=64) == result
    model.resolve_pending_neutral(later_prompt, inventory, cue)
    old = rows[0]
    assert model.raw_complete(
        old.prompt,
        old.inventory,
        cue_text=old.cue_text,
        max_new_tokens=64,
    ) == pointerize(old.output, old.inventory)
    model.resolve_pending_neutral(old.prompt, old.inventory, old.cue_text)


def test_success_learning_rejects_mismatched_output_without_mutation() -> None:
    rows = _contrast_rows()
    model = CausalSequencePointerModel.grow(rows, _contrast_config())
    prompt, cue, inventory = _contrast_query()
    result = model.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64)
    prior_digest = model.training_digest

    with pytest.raises(
        CausalSequencePointerError,
        match="does not match pending neuronal emission",
    ):
        model.learn(prompt, f"{result}-different", inventory, cue_text=cue)

    assert model.training_digest == prior_digest
    update = model.learn(prompt, result, inventory, cue_text=cue)
    assert update.observed_success_sequence
    assert update.prior_training_digest == prior_digest


def test_contrast_output_bound_rejects_before_pair_diffing() -> None:
    inventory = SymbolInventory(("slot",))
    rows = (
        _row(
            "bounded sensory episode",
            "small cue",
            "12345",
            inventory,
            "oversized-output",
        ),
    )

    with pytest.raises(
        CausalSequencePointerError,
        match="contrast output exceeds configured byte capacity",
    ):
        CausalSequencePointerModel.grow(
            rows,
            _contrast_config(max_contrast_output_bytes=4),
        )


def test_online_contrast_growth_remains_deterministic_under_tiny_candidate_cap() -> None:
    rows = _contrast_rows()
    prompt, cue, inventory = _contrast_query()
    observations: list[object] = []

    for _repeat in range(2):
        model = CausalSequencePointerModel.grow(
            rows,
            _contrast_config(max_contrast_candidate_episodes=2),
        )
        result = model.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64)
        prior_digest = model.training_digest

        update = model.learn(prompt, result, inventory, cue_text=cue)

        assert result == "ULTRAMARINE/s"
        assert update.observed_success_sequence
        assert update.prior_training_digest == prior_digest
        assert update.training_digest == model.training_digest != prior_digest
        assert model._pending is None
        assert max(map(len, model._contrast_episode_postings.values())) <= 2
        assert (
            max(
                len(model._bounded_contrast_candidates(episode.features))
                for episode in model._contrast_episodes.values()
            )
            <= 2
        )

        replay = model.raw_complete(
            prompt,
            inventory,
            cue_text=cue,
            max_new_tokens=64,
        )
        assert replay == result
        model.resolve_pending_neutral(prompt, inventory, cue)
        old = rows[0]
        old_result = model.raw_complete(
            old.prompt,
            old.inventory,
            cue_text=old.cue_text,
            max_new_tokens=64,
        )
        assert old_result == pointerize(old.output, old.inventory)
        model.resolve_pending_neutral(old.prompt, old.inventory, old.cue_text)
        postings = tuple(
            (feature, tuple(posting))
            for feature, posting in sorted(model._contrast_episode_postings.items())
        )
        observations.append((result, replay, old_result, model.training_digest, postings))

    assert observations[0] == observations[1]


def test_contrast_negative_credit_is_local_exact_once_and_nogo_is_causal() -> None:
    rows = _contrast_rows()
    model = CausalSequencePointerModel.grow(rows, _contrast_config())
    prompt, cue, inventory = _contrast_query()
    assert model.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64) == (
        "ULTRAMARINE/s"
    )

    update = model.observe_failure(
        prompt,
        inventory,
        cue_text=cue,
        feedback_text="opaque local consequence 7f21",
    )

    assert sum(edge.startswith("nogo:") for edge in update.changed_synapses) == 1
    with pytest.raises(RuntimeError, match="no pending"):
        model.observe_failure(
            prompt,
            inventory,
            cue_text=cue,
            feedback_text="duplicate consequence",
        )
    assert model.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64).startswith(
        "<stop>"
    )

    unrelated_cue = "palette cobalt form round"
    assert (
        model.raw_complete(
            prompt,
            inventory,
            cue_text=unrelated_cue,
            max_new_tokens=64,
        )
        == "ULTRAMARINE/rr"
    )
    model.resolve_pending_neutral(prompt, inventory, unrelated_cue)

    nogo_off = CausalSequencePointerModel.grow(
        rows,
        _contrast_config(CausalSequenceAblation.NOGO_OFF),
    )
    assert nogo_off.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64) == (
        "ULTRAMARINE/s"
    )
    nogo_off.observe_failure(
        prompt,
        inventory,
        cue_text=cue,
        feedback_text="opaque local consequence 7f21",
    )
    assert nogo_off.raw_complete(prompt, inventory, cue_text=cue, max_new_tokens=64) == (
        "ULTRAMARINE/s"
    )
    nogo_off.resolve_pending_neutral(prompt, inventory, cue)


def test_composes_an_unseen_whole_sequence_in_one_forward_trace() -> None:
    rows = _composition_rows()
    model = CausalSequencePointerModel.grow(rows, _config())
    inventory = SymbolInventory(("source", "target"))
    prompt = "compose independent observed attributes fresh"
    cue = "source blue target square witness fresh"

    result = model.raw_complete(
        prompt,
        inventory,
        cue_text=cue,
        max_new_tokens=64,
    )

    expected = "<@0>:B:<@1>:S"
    assert result == expected
    assert expected not in {pointerize(row.output, row.inventory) for row in rows}
    assert materialize(result, inventory) == "source:B:target:S"
    trace = model.last_trace
    assert trace is not None
    assert trace.selected_eos and not trace.selected_stop
    assert all(
        dependency < step.index for step in trace.steps for dependency in step.depends_on_steps
    )
    prior_digest = model.training_digest
    update = model.learn(
        prompt,
        materialize(expected, inventory),
        inventory,
        cue_text=cue,
    )
    assert update.observed_success_sequence
    assert model.training_digest != prior_digest

    later_prompt = "same learned composition after unrelated prompt change"
    assert (
        model.raw_complete(
            later_prompt,
            inventory,
            cue_text=cue,
            max_new_tokens=64,
        )
        == expected
    )
    model.resolve_pending_neutral(later_prompt, inventory, cue)


def test_recurrence_off_physically_stops() -> None:
    rows = _composition_rows()
    treatment = CausalSequencePointerModel.grow(rows, _config())
    model = CausalSequencePointerModel.grow(
        rows,
        _config(CausalSequenceAblation.RECURRENCE_OFF),
    )
    inventory = SymbolInventory(("source", "target"))

    prompt = "compose independent observed attributes fresh"
    cue = "source blue target square witness fresh"
    assert treatment.training_digest == model.training_digest
    assert (
        treatment.raw_complete(
            prompt,
            inventory,
            cue_text=cue,
            max_new_tokens=64,
        )
        == "<@0>:B:<@1>:S"
    )
    treatment.resolve_pending_neutral(prompt, inventory, cue)

    result = model.raw_complete(
        prompt,
        inventory,
        cue_text=cue,
        max_new_tokens=64,
    )

    assert result.startswith("<stop>")
    trace = model.last_trace
    assert trace is not None and trace.selected_stop
    assert trace.steps[-1].phase == "semantic"
    assert trace.steps[-1].winner_operation == "STOP"
    assert len(trace.steps[-1].candidate_currents_pa) > 1

    # The ablation removes recurrent drive only: an exact episodic trace still
    # has enough local evidence to emit while the novel composition abstains.
    exact = rows[0]
    assert model.raw_complete(
        exact.prompt,
        exact.inventory,
        cue_text=exact.cue_text,
        max_new_tokens=64,
    ) == pointerize(exact.output, exact.inventory)
    model.resolve_pending_neutral(exact.prompt, exact.inventory, exact.cue_text)


def test_balanced_ambiguity_recruits_tonic_stop() -> None:
    inventory = SymbolInventory(("slot",))
    rows = (
        _row("red observation one", "choose red witness x", "A", inventory, "r1"),
        _row("red observation two", "choose red witness y", "A", inventory, "r2"),
        _row("blue observation one", "choose blue witness x", "B", inventory, "b1"),
        _row("blue observation two", "choose blue witness y", "B", inventory, "b2"),
    )
    model = CausalSequencePointerModel.grow(rows, _config())

    result = model.raw_complete(
        "balanced ambiguous observation",
        inventory,
        cue_text="choose red blue witness fresh",
    )

    assert result.startswith("<stop>")
    trace = model.last_trace
    assert trace is not None and trace.selected_stop
    assert trace.steps[-1].winner_operation == "STOP"


def test_negative_receipt_suppresses_only_its_local_exact_episode_once() -> None:
    rows = _revision_rows()
    model = CausalSequencePointerModel.grow(rows, _config())
    inventory = rows[0].inventory
    prompt = rows[0].prompt
    cue = rows[0].cue_text

    assert model.raw_complete(prompt, inventory, cue_text=cue) == "A"
    receipt = "opaque receipt 93d7"
    assert all(receipt not in f"{row.prompt}\n{row.cue_text}\n{row.output}" for row in rows)
    update = model.observe_failure(
        prompt,
        inventory,
        cue_text=cue,
        feedback_text=receipt,
    )
    assert any(edge.startswith("nogo:") for edge in update.changed_synapses)
    with pytest.raises(RuntimeError, match="no pending"):
        model.observe_failure(
            prompt,
            inventory,
            cue_text=cue,
            feedback_text="duplicate receipt",
        )

    assert model.raw_complete(prompt, inventory, cue_text=cue).startswith("<stop>")
    trace = model.last_trace
    assert trace is not None and trace.selected_stop
    assert trace.steps[-1].phase == "exact-episode"
    assert trace.steps[-1].winner_operation == "STOP"
    assert len(trace.steps[-1].candidate_currents_pa) == 1

    unrelated = rows[1]
    assert (
        model.raw_complete(
            unrelated.prompt,
            unrelated.inventory,
            cue_text=unrelated.cue_text,
        )
        == "B"
    )
    model.resolve_pending_neutral(
        unrelated.prompt,
        unrelated.inventory,
        unrelated.cue_text,
    )


def test_nogo_off_repeats_the_failed_contextual_choice() -> None:
    rows = _revision_rows()
    model = CausalSequencePointerModel.grow(
        rows,
        _config(CausalSequenceAblation.NOGO_OFF),
    )
    inventory = rows[0].inventory
    prompt = rows[0].prompt
    cue = rows[0].cue_text

    assert model.raw_complete(prompt, inventory, cue_text=cue) == "A"
    update = model.observe_failure(
        prompt,
        inventory,
        cue_text=cue,
        feedback_text="opaque receipt 93d7",
    )
    assert not any(edge.startswith("nogo:") for edge in update.changed_synapses)
    assert model.raw_complete(prompt, inventory, cue_text=cue) == "A"
    model.resolve_pending_neutral(prompt, inventory, cue)


def test_role_cursor_binds_changed_live_cardinality_before_continuing() -> None:
    inventory_two = SymbolInventory(("a", "b"))
    inventory_three = SymbolInventory(("x", "y", "z"))
    rows = (
        _row(
            "emit every current role first observation",
            "emit all a b witness two",
            "emit(a, b)",
            inventory_two,
            "roles-two",
        ),
        _row(
            "emit every current role second observation",
            "emit all x y z witness three",
            "emit(x, y, z)",
            inventory_three,
            "roles-three",
        ),
    )
    model = CausalSequencePointerModel.grow(rows, _config())
    inventory_four = SymbolInventory(("north", "east", "south", "west"))
    prompt = "emit every current role fresh observation"
    cue = "emit all north east south west witness four"

    result = model.raw_complete(
        prompt,
        inventory_four,
        cue_text=cue,
        max_new_tokens=16,
    )

    assert model.role_cursors
    assert result == "emit(<@0>, <@1>, <@2>, <@3>)"
    assert materialize(result, inventory_four) == "emit(north, east, south, west)"
    trace = model.last_trace
    assert trace is not None
    binds = [step for step in trace.steps if step.phase == "role-bind"]
    boundaries = [step for step in trace.steps if step.phase == "role-boundary"]
    assert [step.winner_operation for step in binds] == [
        "BIND:0",
        "BIND:1",
        "BIND:2",
        "BIND:3",
    ]
    assert [step.winner_operation for step in boundaries] == [
        "CONTINUE",
        "CONTINUE",
        "CONTINUE",
        "EXIT",
    ]
    assert all(len(step.candidate_currents_pa) == 2 for step in binds)
    assert all(len(step.candidate_currents_pa) == 2 for step in boundaries)
    exit_step = boundaries[-1]
    later_semantic = next(
        step for step in trace.steps if step.phase == "semantic" and step.index > exit_step.index
    )
    assert exit_step.index in later_semantic.depends_on_steps
    assert all(
        dependency < step.index for step in trace.steps for dependency in step.depends_on_steps
    )
    model.resolve_pending_neutral(prompt, inventory_four, cue)
