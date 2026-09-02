from __future__ import annotations

import pytest
from humanbrain.episodic_pointer import (
    CueRoutedPointerCellModel,
    RoutedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import (
    PointerTrainingExample,
    SymbolInventory,
    identifier_inventory,
    materialize,
)

_ACTION_CUE = "Return the payload unchanged."
_ACTION_QUERY_CUE = "Please return the payload unchanged."
_STOP_CUE = "Decline unsupported requests."


def _model() -> tuple[
    CueRoutedPointerCellModel,
    SymbolInventory,
    SymbolInventory,
]:
    anchor_inventory = identifier_inventory(
        "def solve(arg0):\n"
        "    raise NotImplementedError\n"
    )
    renamed_inventory = identifier_inventory(
        "def transform(value):\n"
        "    raise RuntimeError\n"
    )
    examples = (
        RoutedPointerTrainingExample(
            PointerTrainingExample("copy arg0", "arg0!", anchor_inventory),
            _ACTION_CUE,
        ),
        RoutedPointerTrainingExample(
            PointerTrainingExample(
                "decline request",
                "<stop>unsupported</stop>",
                anchor_inventory,
            ),
            _STOP_CUE,
        ),
    )
    model = CueRoutedPointerCellModel.grow(
        examples,
        ModelConfig(
            max_order=256,
            max_phrases=256,
            min_phrase_frequency=1,
        ),
    )
    return model, anchor_inventory, renamed_inventory


def test_public_task_overlap_spikes_route_and_emits_alpha_bound_template() -> None:
    model, _, renamed_inventory = _model()

    raw = model.raw_complete(
        "copy value",
        renamed_inventory,
        cue_text=_ACTION_QUERY_CUE,
        stop_text=("!", "</stop>"),
    )

    trace = model.last_routing_trace
    assert trace is not None
    assert not trace.selected_stop
    assert trace.selected_route == 0
    assert trace.route_scores[0] > trace.route_scores[1]
    assert 0 <= trace.selected_route < model.route_count
    assert trace.spike_time_ms > 0.0
    assert trace.simulated_steps > 0
    assert raw == "<@1>!"
    assert materialize(raw, renamed_inventory) == "value!"


def test_zero_support_selects_learned_stop_by_tonic_competition() -> None:
    model, _, renamed_inventory = _model()

    raw = model.raw_complete(
        "rotate value",
        renamed_inventory,
        cue_text="Rotate aquamarine matrices clockwise.",
        stop_text=("!", "</stop>"),
    )

    trace = model.last_routing_trace
    assert trace is not None
    assert trace.route_scores[0] == 0.0
    assert trace.route_scores[1] > 0.0
    assert trace.selected_route == 1
    assert trace.selected_stop
    assert trace.simulated_steps > 0
    assert raw == "<stop>unsupported</stop>"


def test_failure_excludes_only_the_route_from_the_bound_completion() -> None:
    model, _, renamed_inventory = _model()
    prompt = "copy value"
    digest = model.training_digest

    first = model.raw_complete(
        prompt,
        renamed_inventory,
        cue_text=_ACTION_QUERY_CUE,
        stop_text=("!", "</stop>"),
    )
    first_trace = model.last_routing_trace
    assert first == "<@1>!"
    assert first_trace is not None and not first_trace.selected_stop

    with pytest.raises(
        ValueError,
        match="failure is not bound to the last neuronal completion",
    ):
        model.observe_failure(
            "different request",
            renamed_inventory,
            cue_text=_ACTION_QUERY_CUE,
            feedback_text="failed",
        )
    with pytest.raises(
        ValueError,
        match="failure is not bound to the last neuronal completion",
    ):
        model.observe_failure(
            prompt,
            renamed_inventory,
            cue_text="A different public task cue.",
            feedback_text="failed",
        )
    assert model.failure_events == 0
    assert model.training_digest == digest

    model.observe_failure(
        prompt,
        renamed_inventory,
        cue_text=_ACTION_QUERY_CUE,
        feedback_text="failed",
    )
    assert model.failure_events == 1
    assert model.training_digest == digest

    second = model.raw_complete(
        prompt,
        renamed_inventory,
        cue_text=_ACTION_QUERY_CUE,
        stop_text=("!", "</stop>"),
    )
    second_trace = model.last_routing_trace
    assert second_trace is not None
    assert second_trace.refractory_route == first_trace.selected_route
    assert second_trace.selected_route != first_trace.selected_route
    assert second_trace.selected_stop
    assert second == "<stop>unsupported</stop>"


def test_success_learning_changes_digest_and_binds_fresh_public_cues() -> None:
    model, _, renamed_inventory = _model()
    prompt = "repeat value"
    learning_cue = (
        "Return the payload unchanged while you echo the parcel intact."
    )

    model.raw_complete(
        prompt,
        renamed_inventory,
        cue_text=learning_cue,
        stop_text=("!", "</stop>"),
    )
    selected = model.last_routing_trace
    assert selected is not None and not selected.selected_stop
    digest_before = model.training_digest

    update = model.learn(
        prompt,
        "value!",
        renamed_inventory,
        cue_text=learning_cue,
        strength=2.0,
    )

    assert update.prior_training_digest == digest_before
    assert update.training_digest == model.training_digest
    assert model.training_digest != digest_before
    learned_digest = model.training_digest

    raw = model.raw_complete(
        prompt,
        renamed_inventory,
        cue_text="Please echo the parcel intact.",
        stop_text=("!", "</stop>"),
    )
    learned_trace = model.last_routing_trace
    assert learned_trace is not None
    assert learned_trace.selected_route == selected.selected_route
    assert not learned_trace.selected_stop
    assert learned_trace.route_scores[selected.selected_route] > 0.0
    assert raw == "<@1>!"
    assert materialize(raw, renamed_inventory) == "value!"
    assert model.training_digest == learned_digest
