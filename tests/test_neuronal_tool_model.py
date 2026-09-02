from __future__ import annotations

import inspect
import json
from dataclasses import fields, replace

import humanbrain.neuronal_tool_model as neuronal_tool_module
import pytest
from humanbrain.causal_sequence_pointer import (
    CausalSequenceConfig,
    CausalSequencePointerModel,
)
from humanbrain.neuronal_tool_model import (
    NeuronalToolModel,
    NeuronalToolModelError,
    NeuronalToolTrainingExample,
)
from humanbrain.persistent_tool_learning import (
    ExecutionReceipt,
    ModelEmission,
    ObservationBlock,
    ObservationFrame,
    PersistentToolLearner,
    StopDecision,
    ToolAction,
    ToolAffordance,
)
from humanbrain.pointer_binding import SymbolInventory, materialize, pointerize
from humanbrain.tokenizer import TokenizerConfig

_STOP_TEXT = ("</tool>", "</stop>")


def _config(
    *,
    max_roles: int = 128,
    plasticity: bool = True,
) -> CausalSequenceConfig:
    return CausalSequenceConfig(
        tokenizer=TokenizerConfig(max_phrases=512, min_frequency=1),
        max_observation_order=2,
        max_observation_features=256,
        max_prefix_order=3,
        max_role_cursors=16,
        max_roles=max_roles,
        plasticity=plasticity,
        random_seed=67,
    )


def _row(prompt: str, output: str, provenance: str) -> NeuronalToolTrainingExample:
    return NeuronalToolTrainingExample(prompt, output, provenance)


def _frame(observation: str = "crystal") -> ObservationFrame:
    return ObservationFrame.seal(
        (ObservationBlock.seal("sample", observation),),
        (
            ToolAffordance.seal(
                "measure",
                {
                    "properties": {"probe": {"type": "string"}},
                    "required": ["probe"],
                    "type": "object",
                },
            ),
        ),
    )


def _tool_prompt(frame: ObservationFrame) -> str:
    envelope = {
        "affordances": [
            {"input_schema": json.loads(item.input_schema_json), "name": item.name}
            for item in frame.affordances
        ],
        "observations": [{"kind": item.kind, "text": item.text} for item in frame.blocks],
    }
    encoded = json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        "persistent-tool-learning-v1\n"
        f"{encoded}\n"
        "Reply with exactly one <tool>{canonical JSON}</tool> containing only "
        '"arguments" and "name", or <stop>reason</stop>.'
    )


def _tool_output(probe: str) -> str:
    command = {"arguments": {"probe": probe}, "name": "measure"}
    return "<tool>" + json.dumps(command, separators=(",", ":"), sort_keys=True) + "</tool>"


class _Execution:
    def __init__(self, *, accepted: bool = True, succeeded: bool = True) -> None:
        self.accepted = accepted
        self.succeeded = succeeded
        self.actions: list[ToolAction] = []

    def execute(self, action: ToolAction) -> ExecutionReceipt:
        self.actions.append(action)
        return ExecutionReceipt.seal(
            action,
            accepted=self.accepted,
            succeeded=self.succeeded,
            observations=(ObservationBlock.seal("environment", "measured"),),
        )


def test_tool_transport_keeps_envelope_stable_across_permuted_inventories() -> None:
    public_output = _tool_output("alpha")
    wire = neuronal_tool_module._encode_action_transport(public_output)

    assert wire == (
        '[[84,79,79,76,1],[109,101,97,115,117,114,101],[[[112,114,111,98,101],"alpha"]]]'
    )
    assert all(
        identifier not in wire for identifier in ("tool", "arguments", "name", "measure", "probe")
    )

    learned_inventory = SymbolInventory(("arguments", "name", "measure", "probe", "alpha"))
    permuted_inventory = SymbolInventory(("probe", "measure", "name", "arguments", "omega"))
    pointer_wire = pointerize(wire, learned_inventory)
    assert "<@4>" in pointer_wire

    live_wire = materialize(pointer_wire, permuted_inventory)
    assert neuronal_tool_module._decode_action_transport(live_wire) == _tool_output("omega")
    for raw_output in (
        "plain raw output",
        "<stop>wait</stop>",
        "[1,2]",
        f" {_tool_output('alpha')} ",
    ):
        assert neuronal_tool_module._encode_action_transport(raw_output) == raw_output
        assert neuronal_tool_module._decode_action_transport(raw_output) == raw_output


@pytest.mark.parametrize(
    "malformed_wire",
    (
        "[[84,79,79,76,1],[256],[]]",
        "[[84,79,79,76,1],[195],[]]",
        "[[84.0,79,79,76,1],[109],[]]",
        "[[84,79,79,76,true],[109],[]]",
        "[[84,79,79,76,1],[109],[[[97],1],[[97],2]]]",
        "[[84,79,79,76,1],[109],[]] ",
    ),
)
def test_malformed_tool_transport_fails_closed(
    malformed_wire: str,
) -> None:
    with pytest.raises(NeuronalToolModelError, match="transport is malformed"):
        neuronal_tool_module._decode_action_transport(malformed_wire)


def test_malformed_tool_transport_returns_fail_closed_emission() -> None:
    malformed_wire = "[[84,79,79,76,1]]"
    model = NeuronalToolModel.grow(
        (
            _row(
                "emit",
                malformed_wire,
                "broken-wire",
            ),
        ),
        _config(max_roles=16),
    )

    try:
        emission = model.infer(
            "emit",
            max_new_tokens=32,
            stop_text=_STOP_TEXT,
        )
    except NeuronalToolModelError as error:
        pytest.fail(
            "infer leaked malformed action-transport failure instead of returning "
            f"a fail-closed emission: {error}"
        )

    assert emission.raw_output == malformed_wire
    assert not emission.selected_stop
    trace = model.last_trace
    assert trace is not None
    assert trace.materialized_output_sha256 is None
    assert trace.materialization_error == "invalid_action_transport"
    update = model.resolve_failure(trace.query_sha256, "malformed transport rejected")
    assert update.polarity == "failure"


def test_success_uses_inner_wire_while_public_trace_stays_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    prompt = _tool_prompt(frame)
    public_output = _tool_output("crystal")
    expected_wire = neuronal_tool_module._encode_action_transport(public_output)
    model = NeuronalToolModel.grow(
        (_row(prompt, public_output, "wire-custody"),),
        _config(),
    )
    learned_outputs: list[str] = []
    original_learn = CausalSequencePointerModel.learn

    def record_learn(
        inner: CausalSequencePointerModel,
        inner_prompt: str,
        inner_output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> object:
        learned_outputs.append(inner_output)
        return original_learn(
            inner,
            inner_prompt,
            inner_output,
            inventory,
            cue_text=cue_text,
            strength=strength,
        )

    monkeypatch.setattr(CausalSequencePointerModel, "learn", record_learn)
    result = PersistentToolLearner(model, _Execution()).episode(frame)

    assert result.generation_trace.raw_output == public_output
    assert isinstance(result.decision, ToolAction)
    assert learned_outputs == [expected_wire]
    assert model.last_trace is not None
    assert model.last_trace.query_sha256 == result.generation_trace.model_query_sha256
    assert model.last_trace.materialized_output_sha256 == neuronal_tool_module._digest(
        "humanbrain-neuronal-materialized-output-v1",
        public_output,
    )
    assert model.last_trace.underlying_trace.raw_output != public_output

    prior_digest = model.training_digest
    trace = model.last_trace
    monkeypatch.setattr(
        neuronal_tool_module,
        "_ACTION_TRANSPORT_POLICY",
        "synthetic-tampered-policy",
    )
    assert model.training_digest != prior_digest
    with pytest.raises(ValueError, match="trace is not authentic"):
        replace(trace)


def test_fresh_names_and_variable_cardinality_copy_through_neuronal_races() -> None:
    names = NeuronalToolModel.grow(
        (
            _row("route willow to cedar", "willow=>cedar", "route-a"),
            _row("route amber to quartz", "amber=>quartz", "route-b"),
        ),
        _config(max_roles=32),
    )
    emission = names.infer(
        "route cobalt to ivory",
        max_new_tokens=64,
        stop_text=_STOP_TEXT,
    )
    assert emission.raw_output == "cobalt=>ivory"
    assert not emission.selected_stop
    trace = names.last_trace
    assert trace is not None
    assert trace.inventory_cardinality == 4
    assert len(trace.inventory_sha256) == 64
    assert len(trace.pointer_output_sha256) == 64
    assert len(trace.materialized_output_sha256 or "") == 64
    assert trace.pointer_output_sha256 != trace.materialized_output_sha256
    assert trace.materialization_error is None
    assert len(trace.sha256) == 64
    with pytest.raises(ValueError, match="not authentic"):
        replace(trace, inventory_cardinality=trace.inventory_cardinality + 1)
    first_step = trace.underlying_trace.steps[0]
    tampered_step = replace(
        first_step,
        spike_time_ms=first_step.spike_time_ms + 0.25,
    )
    tampered_underlying = replace(
        trace.underlying_trace,
        steps=(tampered_step, *trace.underlying_trace.steps[1:]),
    )
    with pytest.raises(ValueError, match="not authentic"):
        replace(trace, underlying_trace=tampered_underlying)
    names.resolve_neutral(emission.query_sha256)

    cardinality = NeuronalToolModel.grow(
        (
            _row("repeat alpha beta", "alpha|beta", "repeat-two"),
            _row(
                "repeat cedar birch pine",
                "cedar|birch|pine",
                "repeat-three",
            ),
        ),
        _config(max_roles=32),
    )
    emission = cardinality.infer(
        "repeat moon star cloud rain",
        max_new_tokens=64,
        stop_text=_STOP_TEXT,
    )
    assert emission.raw_output == "moon|star|cloud|rain"
    trace = cardinality.last_trace
    assert trace is not None and not trace.selected_stop
    assert any(step.winner_operation == "ROLE_CURSOR" for step in trace.underlying_trace.steps)
    cardinality.resolve_neutral(emission.query_sha256)


def test_raw_semantic_channel_prevents_same_shape_lexical_collision() -> None:
    rows = (
        _row("signal amber means proceed", "ALLOW", "semantic-allow"),
        _row("signal cobalt means proceed", "DENY", "semantic-deny"),
    )
    model = NeuronalToolModel.grow(rows, _config(max_roles=32))
    states: list[str] = []

    for prompt, expected in (
        (rows[0].prompt, "ALLOW"),
        (rows[1].prompt, "DENY"),
    ):
        emission = model.infer(prompt, max_new_tokens=32, stop_text=_STOP_TEXT)
        assert emission.raw_output == expected
        assert not emission.selected_stop
        trace = model.last_trace
        assert trace is not None
        states.append(trace.underlying_trace.observation_state)
        model.resolve_neutral(emission.query_sha256)

    assert states[0] != states[1]
    singleton = NeuronalToolModel.grow((rows[0],), _config(max_roles=32))
    clone = NeuronalToolModel.grow(rows, _config(max_roles=32))
    assert model.training_digest == clone.training_digest
    assert model.training_digest != singleton.training_digest


@pytest.mark.parametrize(
    ("accepted", "succeeded", "resolution"),
    ((True, True, "success"), (True, False, "failure")),
)
def test_persistent_tool_learner_resolves_generic_execution_consequences(
    accepted: bool,
    succeeded: bool,
    resolution: str,
) -> None:
    frame = _frame()
    prompt = _tool_prompt(frame)
    model = NeuronalToolModel.grow(
        (_row(prompt, _tool_output("crystal"), "tool-action"),),
        _config(),
    )
    execution = _Execution(accepted=accepted, succeeded=succeeded)
    prior_digest = model.training_digest

    result = PersistentToolLearner(model, execution).episode(frame)

    assert result.resolution == resolution
    assert isinstance(result.decision, ToolAction)
    assert len(execution.actions) == 1
    update = model.last_update
    assert update is not None and update.polarity == resolution
    assert update.query_sha256 == result.generation_trace.model_query_sha256
    assert update.consequence_sha256 is not None
    assert len(update.provenance_sha256) == 64
    assert update.underlying_update is not None
    assert model.training_digest != prior_digest
    if resolution == "success":
        replay = model.infer(
            prompt,
            max_new_tokens=512,
            stop_text=_STOP_TEXT,
        )
        assert replay.raw_output == _tool_output("crystal")
        model.resolve_neutral(replay.query_sha256)


def test_explicit_and_physical_stop_close_neutral_custody() -> None:
    frame = _frame()
    prompt = _tool_prompt(frame)
    explicit = NeuronalToolModel.grow(
        (_row(prompt, "<stop>wait for evidence</stop>", "explicit-stop"),),
        _config(),
    )
    explicit_prior = explicit.training_digest

    explicit_result = PersistentToolLearner(explicit, _Execution()).episode(frame)

    assert isinstance(explicit_result.decision, StopDecision)
    assert explicit_result.decision.reason == "wait for evidence"
    assert explicit_result.resolution == "neutral"
    assert explicit.training_digest == explicit_prior
    assert explicit.last_trace is not None and not explicit.last_trace.selected_stop
    assert explicit.last_trace.materialized_output_sha256 is not None
    assert explicit.last_trace.materialization_error is None
    assert explicit.last_update is not None
    assert explicit.last_update.underlying_update is not None

    tonic = NeuronalToolModel.grow(
        (_row("unrelated remembered scene", "remembered", "tonic-base"),),
        _config(),
    )
    tonic_prior = tonic.training_digest

    tonic_result = PersistentToolLearner(tonic, _Execution()).episode(frame)

    assert isinstance(tonic_result.decision, StopDecision)
    assert tonic_result.decision.reason == "tonic_stop"
    assert tonic_result.resolution == "neutral"
    assert tonic.training_digest == tonic_prior
    assert tonic.last_trace is not None and tonic.last_trace.selected_stop
    assert tonic.last_trace.materialized_output_sha256 is None
    assert tonic.last_trace.materialization_error == "physical_stop_not_materialized"
    assert tonic.last_update is not None
    assert tonic.last_update.underlying_update is None


def test_inference_is_read_only_and_query_custody_is_exact_once() -> None:
    model = NeuronalToolModel.grow(
        (_row("echo silver", "silver", "custody"),),
        _config(max_roles=16),
    )
    prior_digest = model.training_digest
    emission = model.infer(
        "echo silver",
        max_new_tokens=32,
        stop_text=_STOP_TEXT,
    )

    assert isinstance(emission, ModelEmission)
    assert model.training_digest == prior_digest
    with pytest.raises(RuntimeError, match="resolve the pending"):
        model.infer("echo silver", max_new_tokens=32, stop_text=_STOP_TEXT)
    with pytest.raises(NeuronalToolModelError, match="does not match"):
        model.resolve_neutral("0" * 64)
    assert model.training_digest == prior_digest

    update = model.resolve_neutral(emission.query_sha256)
    assert update.training_digest == update.prior_training_digest == prior_digest
    with pytest.raises(RuntimeError, match="no pending"):
        model.resolve_neutral(emission.query_sha256)


def test_materialization_error_retains_exact_custody_for_neutral_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = NeuronalToolModel.grow(
        (_row("echo silver", "silver", "materialization"),),
        _config(max_roles=16),
    )
    prior_digest = model.training_digest

    with monkeypatch.context() as patch:

        def reject_materialization(_text: str, _inventory: object) -> str:
            raise ValueError("synthetic invalid pointer output")

        patch.setattr(neuronal_tool_module, "materialize", reject_materialization)
        with pytest.raises(NeuronalToolModelError, match="could not be materialized"):
            model.infer("echo silver", max_new_tokens=32, stop_text=_STOP_TEXT)
        trace = model.last_trace
        assert trace is not None and not trace.selected_stop
        assert trace.materialized_output_sha256 is None
        assert trace.materialization_error == "invalid_pointer_output"
        with pytest.raises(RuntimeError, match="resolve the pending"):
            model.infer("echo silver", max_new_tokens=32, stop_text=_STOP_TEXT)
        model.resolve_neutral(trace.query_sha256)

    assert model.training_digest == prior_digest
    emission = model.infer("echo silver", max_new_tokens=32, stop_text=_STOP_TEXT)
    assert emission.raw_output == "silver"
    model.resolve_neutral(emission.query_sha256)


def test_no_plasticity_control_shares_initial_digest_and_resolves_without_learning() -> None:
    rows = (_row("echo copper", "copper", "control"),)
    plastic = NeuronalToolModel.grow(rows, _config(max_roles=16))
    frozen = NeuronalToolModel.grow(
        rows,
        _config(max_roles=16, plasticity=False),
    )
    assert frozen.training_digest == plastic.training_digest
    prior_digest = frozen.training_digest

    emission = frozen.infer(
        "echo copper",
        max_new_tokens=32,
        stop_text=_STOP_TEXT,
    )
    update = frozen.resolve_success(
        emission.query_sha256,
        "accepted generic consequence",
        1.0,
    )

    assert frozen.training_digest == prior_digest
    assert update.training_digest == update.prior_training_digest == prior_digest
    assert update.underlying_update is not None
    assert not update.underlying_update.observed_success_sequence


def test_lexical_inventory_is_deterministic_bounded_and_not_public_caller_state() -> None:
    rows = (_row("echo alpha alpha", "alpha", "inventory"),)
    first = NeuronalToolModel.grow(rows, _config(max_roles=2))
    second = NeuronalToolModel.grow(rows, _config(max_roles=2))
    assert first.training_digest == second.training_digest

    with pytest.raises(NeuronalToolModelError, match="reserved"):
        first.infer(
            "echo <@0>",
            max_new_tokens=32,
            stop_text=_STOP_TEXT,
        )
    with pytest.raises(NeuronalToolModelError, match="role capacity"):
        first.infer(
            "alpha beta gamma",
            max_new_tokens=32,
            stop_text=_STOP_TEXT,
        )
    emission = first.infer(
        "echo alpha alpha",
        max_new_tokens=32,
        stop_text=_STOP_TEXT,
    )
    assert emission.raw_output == "alpha"
    first.resolve_neutral(emission.query_sha256)

    assert {field.name for field in fields(NeuronalToolTrainingExample)} == {
        "output",
        "prompt",
        "provenance_id",
    }
    infer_parameters = set(inspect.signature(NeuronalToolModel.infer).parameters)
    assert infer_parameters == {"self", "prompt", "max_new_tokens", "stop_text"}
    for method in (
        NeuronalToolModel.infer,
        NeuronalToolModel.resolve_success,
        NeuronalToolModel.resolve_failure,
        NeuronalToolModel.resolve_neutral,
    ):
        parameters = set(inspect.signature(method).parameters)
        assert "inventory" not in parameters
        assert "cue_text" not in parameters


def test_over_cap_training_prompt_fails_before_model_growth() -> None:
    with pytest.raises(NeuronalToolModelError, match="role capacity"):
        NeuronalToolModel.grow(
            (_row("alpha beta gamma", "alpha", "too-many"),),
            _config(max_roles=2),
        )


def test_reserved_pointer_syntax_is_rejected_at_raw_training_boundary() -> None:
    with pytest.raises(NeuronalToolModelError, match="reserved"):
        _row("echo <@0>", "alpha", "reserved-prompt")
    with pytest.raises(NeuronalToolModelError, match="reserved"):
        _row("echo alpha", "<@0>", "reserved-output")
