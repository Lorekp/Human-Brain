from __future__ import annotations

import json
import re

import pytest
from experiments.r4_integrated.factorized_output_binding_fixture_v1 import (
    FRESH_CASES,
    PUBLIC_LESSONS,
)
from experiments.r4_integrated.public_coding_prior_v1 import public_coding_prior_v1
from humanbrain.factorized_pointer import (
    BINDING_OFF,
    FRAGMENT_PROJECTION_OFF,
    STOP_OFF,
    WHOLE_EPISODE_ONLY,
    FactorizedPointerCellModel,
    FactorizedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import SymbolInventory, python_interface_inventory

_LEXICAL_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\s+|.", re.DOTALL)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _cue_text(cue_a: str, cue_b: str) -> str:
    return f"{cue_a} {cue_b}"


def _write_source_output(source: str) -> str:
    payload = {"arguments": {"source": source}, "name": "write_source"}
    return (
        "<tool>"
        + json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "</tool>"
    )


def _equal_modulo_allocated_locals(
    actual: str,
    expected: str,
    allocated: frozenset[str],
) -> bool:
    actual_tokens = tuple(_LEXICAL_TOKEN.findall(actual))
    expected_tokens = tuple(_LEXICAL_TOKEN.findall(expected))
    if len(actual_tokens) != len(expected_tokens):
        return False
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for actual_token, expected_token in zip(
        actual_tokens,
        expected_tokens,
        strict=True,
    ):
        if actual_token == expected_token:
            continue
        if (
            actual_token not in allocated
            or _IDENTIFIER.fullmatch(expected_token) is None
        ):
            return False
        if forward.setdefault(actual_token, expected_token) != expected_token:
            return False
        if reverse.setdefault(expected_token, actual_token) != actual_token:
            return False
    return True


def _model() -> FactorizedPointerCellModel:
    return FactorizedPointerCellModel.grow(
        tuple(
            FactorizedPointerTrainingExample(
                prompt=lesson.lesson_id,
                output=lesson.demonstrated_output,
                inventory=SymbolInventory((lesson.input_symbol,)),
                cue_text=_cue_text(lesson.cue_a, lesson.cue_b),
                provenance_id=lesson.lesson_id,
            )
            for lesson in PUBLIC_LESSONS
        ),
        ModelConfig(max_order=8, max_phrases=64, min_phrase_frequency=2),
    )


def test_fresh_factor_composition_and_frozen_fixture_ablations() -> None:
    model = _model()
    decisions: list[dict[str, object]] = []

    # Freeze every treatment and ablation decision from public observations
    # before opening any scoring-only fixture fields.
    for case in FRESH_CASES:
        observation = case.public_observation()
        prompt = observation["case_id"]
        inventory = SymbolInventory((observation["input_symbol"],))
        cues = _cue_text(observation["cue_a"], observation["cue_b"])

        treatment = model.complete(prompt, inventory, cue_text=cues)
        treatment_trace = model.last_trace
        assert treatment_trace is not None

        whole_episode = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            ablation=WHOLE_EPISODE_ONLY,
        )
        whole_episode_trace = model.last_trace
        assert whole_episode_trace is not None

        binding_off = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            ablation=BINDING_OFF,
        )
        binding_off_trace = model.last_trace
        assert binding_off_trace is not None

        projection_off = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            ablation=FRAGMENT_PROJECTION_OFF,
        )
        projection_off_trace = model.last_trace
        assert projection_off_trace is not None

        decisions.append(
            {
                "case": case,
                "treatment": treatment,
                "treatment_trace": treatment_trace,
                "whole_episode": whole_episode,
                "whole_episode_trace": whole_episode_trace,
                "binding_off": binding_off,
                "binding_off_trace": binding_off_trace,
                "projection_off": projection_off,
                "projection_off_trace": projection_off_trace,
            }
        )

    for decision in decisions:
        case = decision["case"]
        treatment = decision["treatment"]
        treatment_trace = decision["treatment_trace"]
        whole_episode = decision["whole_episode"]
        whole_episode_trace = decision["whole_episode_trace"]
        binding_off = decision["binding_off"]
        binding_off_trace = decision["binding_off_trace"]
        projection_off = decision["projection_off"]
        projection_off_trace = decision["projection_off_trace"]

        assert isinstance(treatment, str)
        assert case.accepts_treatment_output(treatment)
        assert not treatment_trace.selected_stop
        assert treatment_trace.binding_races
        assert treatment_trace.emitted_token_ids
        assert all(
            step.spike_time_ms > 0.0 and step.simulated_steps > 0
            for step in treatment_trace.fragment_steps
        )

        assert isinstance(whole_episode, str)
        assert whole_episode.startswith("<stop>")
        assert whole_episode_trace.selected_stop
        assert len(whole_episode_trace.fragment_steps) == 1

        assert binding_off == case.binding_off_output()
        assert not case.accepts_treatment_output(binding_off)
        assert not binding_off_trace.selected_stop
        assert not binding_off_trace.binding_races

        assert isinstance(projection_off, str)
        assert projection_off.startswith("<stop>")
        assert projection_off_trace.selected_stop

    assert model.grown_branches > 0


def test_unknown_cues_need_the_physical_stop_competitor() -> None:
    model = _model()
    inventory = SymbolInventory(("novel_input",))

    stopped = model.complete(
        "unknown-combination",
        inventory,
        cue_text="opaque unfamiliar",
    )
    stopped_trace = model.last_trace
    assert stopped.startswith("<stop>")
    assert stopped_trace is not None
    assert stopped_trace.selected_stop
    assert stopped_trace.fragment_steps[-1].winner_kind == "stop"

    stop_off = model.complete(
        "unknown-combination-stop-off",
        inventory,
        cue_text="opaque unfamiliar",
        ablation=STOP_OFF,
    )
    stop_off_trace = model.last_trace
    assert not stop_off.startswith("<stop>")
    assert stop_off_trace is not None
    assert not stop_off_trace.selected_stop


def test_fresh_binding_avoids_query_symbols_and_reuses_one_winner() -> None:
    model = _model()
    first = FRESH_CASES[0].public_observation()
    cues = _cue_text(first["cue_a"], first["cue_b"])
    model.complete(
        "first-binding",
        SymbolInventory((first["input_symbol"],)),
        cue_text=cues,
    )
    first_trace = model.last_trace
    assert first_trace is not None
    allocated = first_trace.binding_races[0].symbol

    output = model.complete(
        "binding-conflict",
        SymbolInventory((allocated,)),
        cue_text=cues,
    )
    trace = model.last_trace
    assert trace is not None
    rebound = trace.binding_races[0].symbol
    assert rebound != allocated
    match = re.fullmatch(
        r"let (?P<local>[A-Za-z_][A-Za-z0-9_]*) = "
        r"[A-Za-z_][A-Za-z0-9_]*\((?P<input>[A-Za-z_][A-Za-z0-9_]*)\); "
        r"send [A-Za-z_][A-Za-z0-9_]*\((?P=local)\)",
        output,
    )
    assert match is not None
    assert match.group("input") == allocated
    assert match.group("local") == rebound


def test_failure_feedback_is_context_bound_durable_inhibition() -> None:
    model = _model()
    observation = FRESH_CASES[1].public_observation()
    prompt = observation["case_id"]
    inventory = SymbolInventory((observation["input_symbol"],))
    cues = _cue_text(observation["cue_a"], observation["cue_b"])
    model.complete(prompt, inventory, cue_text=cues)
    prior = model.training_digest

    update = model.observe_failure(
        prompt,
        inventory,
        cue_text=cues,
        feedback_text='{"outcome":"public_failure"}',
    )

    assert update is not None
    assert update.polarity == "failure"
    assert update.synapses_changed > 0
    assert update.prior_training_digest == prior
    assert update.training_digest == model.training_digest
    assert update.training_digest != prior

    with pytest.raises(ValueError, match="last neuronal completion"):
        model.observe_failure(
            "different-query",
            inventory,
            cue_text=cues,
            feedback_text='{"outcome":"public_failure"}',
        )


def test_success_feedback_reinforces_only_the_selected_projection() -> None:
    model = _model()
    observation = FRESH_CASES[2].public_observation()
    prompt = observation["case_id"]
    inventory = SymbolInventory((observation["input_symbol"],))
    cues = _cue_text(observation["cue_a"], observation["cue_b"])
    output = model.complete(prompt, inventory, cue_text=cues)
    prior = model.training_digest

    update = model.learn(
        prompt,
        output,
        inventory,
        cue_text=cues,
    )

    assert update is not None
    assert update.polarity == "success"
    assert update.synapses_changed > 0
    assert update.prior_training_digest == prior
    assert update.training_digest == model.training_digest


def test_variable_width_local_equivalence_allocates_and_reuses_one_symbol() -> None:
    inventory = SymbolInventory(("seed",))
    examples = (
        FactorizedPointerTrainingExample(
            prompt="short example",
            output="begin willow = fuse(seed); emit willow",
            inventory=inventory,
            cue_text="short amber",
            provenance_id="variable-local-short",
        ),
        FactorizedPointerTrainingExample(
            prompt="long example",
            output="begin cedar = extend(seed, seed); audit cedar; emit cedar",
            inventory=inventory,
            cue_text="long cobalt",
            provenance_id="variable-local-long",
        ),
    )
    model = FactorizedPointerCellModel.grow(
        examples,
        ModelConfig(max_order=8, max_phrases=64, min_phrase_frequency=2),
    )

    for example in examples:
        output = model.complete(
            example.prompt,
            inventory,
            cue_text=example.cue_text,
            max_new_tokens=128,
        )
        trace = model.last_trace
        assert trace is not None
        assert trace.terminal_stop
        assert not trace.selected_stop
        assert len(trace.binding_races) == 1
        symbol = trace.binding_races[0].symbol
        assert symbol not in {"seed", "willow", "cedar"}
        assert output.count(symbol) >= 2


def test_all_public_coding_priors_replay_modulo_fresh_local_renaming() -> None:
    cases = public_coding_prior_v1()
    examples = tuple(
        FactorizedPointerTrainingExample(
            prompt=case.public_prompt,
            output=_write_source_output(case.successful_source),
            inventory=python_interface_inventory(
                case.starter_source,
                case.entrypoint,
            ),
            cue_text=case.public_prompt,
            provenance_id=f"public-coding-prior-{index}",
        )
        for index, case in enumerate(cases)
    )
    model = FactorizedPointerCellModel.grow(
        examples,
        ModelConfig(max_order=12, max_phrases=256, min_phrase_frequency=2),
    )

    exact_replays = 0
    for case in cases:
        inventory = python_interface_inventory(
            case.starter_source,
            case.entrypoint,
        )
        expected = _write_source_output(case.successful_source)
        output = model.complete(
            case.public_prompt,
            inventory,
            cue_text=case.public_prompt,
            max_new_tokens=512,
        )
        trace = model.last_trace
        assert trace is not None
        assert trace.terminal_stop
        assert not trace.selected_stop
        assert trace.fragment_steps
        assert all(
            step.spike_time_ms > 0.0 and step.simulated_steps > 0
            for step in trace.fragment_steps
        )
        allocated = frozenset(race.symbol for race in trace.binding_races)
        assert _equal_modulo_allocated_locals(output, expected, allocated)
        if output == expected:
            exact_replays += 1

        assert output.startswith("<tool>") and output.endswith("</tool>")
        payload = json.loads(output[len("<tool>") : -len("</tool>")])
        assert payload["name"] == "write_source"
        compile(payload["arguments"]["source"], "<public-replay>", "exec")

    assert exact_replays == 7
