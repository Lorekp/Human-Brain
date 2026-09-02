from __future__ import annotations

import pytest
from humanbrain.causal_sequence_pointer import (
    CausalSequenceAblation,
    CausalSequenceConfig,
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
        "execution succeeded",
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


def test_negative_receipt_revises_only_its_local_branch_and_is_exact_once() -> None:
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

    assert model.raw_complete(prompt, inventory, cue_text=cue) == "B"
    model.resolve_pending_neutral(prompt, inventory, cue)

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
