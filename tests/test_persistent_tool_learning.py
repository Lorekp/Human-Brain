from __future__ import annotations

import json
from dataclasses import replace

import pytest
from humanbrain.persistent_tool_learning import (
    EpisodeResult,
    ExecutionReceipt,
    GenerationTrace,
    ModelEmission,
    ObservationBlock,
    ObservationFrame,
    PersistentToolLearner,
    StopDecision,
    ToolAction,
    ToolAffordance,
)


class FakeModel:
    def __init__(
        self,
        raw_output: str,
        *,
        selected_stop: bool = False,
        learn_on: tuple[str, ...] = ("success", "failure"),
        foreign_prompt: str | None = None,
        infer_fault: str | None = None,
    ) -> None:
        self.raw_output = raw_output
        self.selected_stop = selected_stop
        self.learn_on = learn_on
        self.foreign_prompt = foreign_prompt
        self.infer_fault = infer_fault
        self._digest_version = 0
        self.pending: str | None = None
        self.calls: list[tuple[str, str, str | float | None]] = []
        self.prompt = ""
        self.max_new_tokens: int | None = None

    @property
    def training_digest(self) -> str:
        return f"model-{self._digest_version}"

    def infer(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> ModelEmission:
        assert self.pending is None
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        expected = ModelEmission.seal(
            prompt,
            self.training_digest,
            self.raw_output,
            selected_stop=self.selected_stop,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )
        self.pending = expected.query_sha256
        self.calls.append(("infer", expected.query_sha256, None))
        if self.infer_fault == "raise":
            raise RuntimeError("inference transport failed")
        if self.infer_fault == "invalid":
            return object()  # type: ignore[return-value]
        if self.foreign_prompt is not None:
            return ModelEmission.seal(
                self.foreign_prompt,
                self.training_digest,
                self.raw_output,
                selected_stop=self.selected_stop,
                max_new_tokens=max_new_tokens,
                stop_text=stop_text,
            )
        return expected

    def _resolve(
        self,
        resolution: str,
        query_sha256: str,
        consequence: str | None,
    ) -> None:
        assert query_sha256 == self.pending
        self.calls.append((resolution, query_sha256, consequence))
        self.pending = None
        if resolution in self.learn_on:
            self._digest_version += 1

    def resolve_success(
        self,
        query_sha256: str,
        consequence_text: str,
        strength: float,
    ) -> None:
        assert strength == 1.0
        self._resolve("success", query_sha256, consequence_text)

    def resolve_failure(self, query_sha256: str, consequence_text: str) -> None:
        self._resolve("failure", query_sha256, consequence_text)

    def resolve_neutral(self, query_sha256: str) -> None:
        self._resolve("neutral", query_sha256, None)


class FakeExecution:
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
            observations=(ObservationBlock.seal("environment", "result"),),
        )


def frame() -> ObservationFrame:
    return ObservationFrame.seal(
        (
            ObservationBlock.seal("weather", "cloudy"),
            ObservationBlock.seal("opaque-sensor-77", "alpha\nbeta"),
        ),
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


def assert_one_resolution(model: FakeModel, expected: str) -> None:
    assert model.pending is None
    assert [call[0] for call in model.calls] == ["infer", expected]
    assert model.calls[0][1] == model.calls[1][1]


def test_success_is_bound_end_to_end_and_prompt_has_no_required_ontology() -> None:
    model = FakeModel('<tool>{"arguments":{"probe":"x"},"name":"measure"}</tool>')
    execution = FakeExecution()

    result = PersistentToolLearner(model, execution).episode(frame())

    assert isinstance(result, EpisodeResult)
    assert isinstance(result.generation_trace, GenerationTrace)
    assert isinstance(result.decision, ToolAction)
    assert result.receipt is not None
    assert result.receipt.action_sha256 == result.decision.sha256
    assert result.decision.generation_trace_sha256 == result.generation_trace.sha256
    assert result.resolution == "success"
    assert result.learning_applied
    assert_one_resolution(model, "success")
    assert len(execution.actions) == 1

    prefix, envelope_text, instruction = model.prompt.splitlines()
    envelope = json.loads(envelope_text)
    assert prefix == "persistent-tool-learning-v1"
    assert instruction.startswith("Reply with exactly one")
    assert [item["kind"] for item in envelope["observations"]] == [
        "weather",
        "opaque-sensor-77",
    ]
    assert envelope["affordances"][0]["name"] == "measure"
    assert "starter-source" not in model.prompt
    assert "typed-signature" not in model.prompt


@pytest.mark.parametrize(
    ("raw_output", "selected_stop", "expected_reason"),
    [
        ('<tool>{"arguments":{},"name":"measure"}</tool>', True, "tonic_stop"),
        ("<stop>insufficient evidence</stop>", False, "insufficient evidence"),
    ],
)
def test_tonic_and_explicit_stop_are_neutral(
    raw_output: str,
    selected_stop: bool,
    expected_reason: str,
) -> None:
    model = FakeModel(raw_output, selected_stop=selected_stop)
    execution = FakeExecution()

    result = PersistentToolLearner(model, execution).episode(frame())

    assert isinstance(result.decision, StopDecision)
    assert result.decision.reason == expected_reason
    assert result.decision.generation_trace_sha256 == result.generation_trace.sha256
    assert result.resolution == "neutral"
    assert not result.learning_applied
    assert result.receipt is None
    assert execution.actions == []
    assert_one_resolution(model, "neutral")


@pytest.mark.parametrize(
    ("raw_output", "expected_reason"),
    [
        ("nonsense", "malformed_model_output"),
        ('<tool>{"arguments":{},"name":"missing"}</tool>', "unavailable_tool"),
        ('<tool>{"name":"measure","arguments":{}}</tool>', "malformed_tool_action"),
        ('<tool>{"arguments":{"x":NaN},"name":"measure"}</tool>', "malformed_tool_action"),
        ('<tool>{"arguments":{"x":Infinity},"name":"measure"}</tool>', "malformed_tool_action"),
        ('<tool>{"arguments":{"x":-Infinity},"name":"measure"}</tool>', "malformed_tool_action"),
    ],
)
def test_invalid_generation_is_failure_not_neutral_stop(
    raw_output: str,
    expected_reason: str,
) -> None:
    model = FakeModel(raw_output)
    execution = FakeExecution()

    result = PersistentToolLearner(model, execution).episode(frame())

    assert isinstance(result.decision, StopDecision)
    assert result.decision.reason == expected_reason
    assert result.resolution == "failure"
    assert result.learning_applied
    assert execution.actions == []
    assert_one_resolution(model, "failure")
    consequence = json.loads(str(model.calls[1][2]))
    assert consequence["generation_trace_sha256"] == result.generation_trace.sha256
    assert consequence["reason"] == expected_reason


@pytest.mark.parametrize(
    ("accepted", "succeeded"),
    [(False, False), (True, False)],
)
def test_rejected_or_failed_receipt_resolves_failure(
    accepted: bool,
    succeeded: bool,
) -> None:
    model = FakeModel('<tool>{"arguments":{},"name":"measure"}</tool>')
    execution = FakeExecution(accepted=accepted, succeeded=succeeded)

    result = PersistentToolLearner(model, execution).episode(frame())

    assert result.resolution == "failure"
    assert result.learning_applied
    assert result.receipt is not None
    assert result.receipt.accepted is accepted
    assert result.receipt.succeeded is succeeded
    assert_one_resolution(model, "failure")


def test_resolution_can_be_recorded_without_a_learning_update() -> None:
    model = FakeModel(
        '<tool>{"arguments":{},"name":"measure"}</tool>',
        learn_on=(),
    )

    result = PersistentToolLearner(model, FakeExecution()).episode(frame())

    assert result.resolution == "success"
    assert not result.learning_applied
    assert result.prior_model_digest == result.model_digest
    assert_one_resolution(model, "success")


def test_empty_arbitrary_frame_is_authentic_and_can_stop() -> None:
    empty = ObservationFrame.seal()
    model = FakeModel("<stop>no observations</stop>")

    result = PersistentToolLearner(model, FakeExecution()).episode(empty)

    assert result.resolution == "neutral"
    assert json.loads(model.prompt.splitlines()[1]) == {
        "affordances": [],
        "observations": [],
    }
    assert_one_resolution(model, "neutral")


def test_learner_binds_an_explicit_generation_budget_into_inference() -> None:
    model = FakeModel("<stop>bounded</stop>")
    learner = PersistentToolLearner(model, FakeExecution(), max_new_tokens=4096)

    result = learner.episode(frame())

    assert result.resolution == "neutral"
    assert model.max_new_tokens == 4096


def test_authentication_rejects_nested_and_trace_tampering() -> None:
    block = ObservationBlock.seal("arbitrary", "payload")
    affordance = ToolAffordance.seal("act", {"nested": {"type": "number"}})
    sealed_frame = ObservationFrame.seal((block,), (affordance,))

    with pytest.raises(ValueError, match="observation block"):
        replace(block, text="changed")
    with pytest.raises(ValueError, match="tool affordance"):
        replace(affordance, input_schema_json='{"nested":{"type":"string"}}')
    with pytest.raises(ValueError, match="observation frame"):
        replace(sealed_frame, blocks=())

    model = FakeModel("<stop>done</stop>")
    result = PersistentToolLearner(model, FakeExecution()).episode(sealed_frame)
    with pytest.raises(ValueError, match="generation trace"):
        replace(result.generation_trace, raw_output="<stop>forged</stop>")
    with pytest.raises(ValueError, match="STOP decision"):
        replace(result.decision, generation_trace_sha256="0" * 64)
    with pytest.raises(ValueError, match="episode result"):
        replace(result, prior_model_digest="forged")

    action_model = FakeModel('<tool>{"arguments":{},"name":"act"}</tool>')
    action_result = PersistentToolLearner(action_model, FakeExecution()).episode(sealed_frame)
    assert action_result.receipt is not None
    with pytest.raises(ValueError, match="execution receipt"):
        replace(action_result.receipt, accepted=1)


def test_foreign_query_is_failed_exactly_once_before_rejection() -> None:
    model = FakeModel("<stop>done</stop>", foreign_prompt="different prompt")

    with pytest.raises(ValueError, match="not bound to the prompt query"):
        PersistentToolLearner(model, FakeExecution()).episode(frame())

    assert_one_resolution(model, "failure")
    assert json.loads(str(model.calls[1][2])) == {"protocol_error": "query_binding_failed"}


@pytest.mark.parametrize(
    ("infer_fault", "exception", "message", "protocol_error"),
    [
        ("invalid", TypeError, "must return ModelEmission", "invalid_emission_type"),
        ("raise", RuntimeError, "transport failed", "infer_raised"),
    ],
)
def test_inference_protocol_failure_clears_expected_query_once(
    infer_fault: str,
    exception: type[Exception],
    message: str,
    protocol_error: str,
) -> None:
    model = FakeModel("<stop>done</stop>", infer_fault=infer_fault)

    with pytest.raises(exception, match=message):
        PersistentToolLearner(model, FakeExecution()).episode(frame())

    assert_one_resolution(model, "failure")
    assert json.loads(str(model.calls[1][2])) == {"protocol_error": protocol_error}


def test_inference_is_read_only_until_resolution() -> None:
    class MutatingInference(FakeModel):
        def infer(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            stop_text: tuple[str, ...],
        ) -> ModelEmission:
            emission = super().infer(
                prompt,
                max_new_tokens=max_new_tokens,
                stop_text=stop_text,
            )
            self._digest_version += 1
            return emission

    model = MutatingInference("<stop>done</stop>")

    with pytest.raises(RuntimeError, match="inference mutated"):
        PersistentToolLearner(model, FakeExecution()).episode(frame())

    assert_one_resolution(model, "failure")


def test_unbound_receipt_fails_closed_and_resolves_once() -> None:
    class TamperingExecution(FakeExecution):
        def execute(self, action: ToolAction) -> ExecutionReceipt:
            receipt = super().execute(action)
            object.__setattr__(receipt, "action_sha256", "0" * 64)
            return receipt

    model = FakeModel('<tool>{"arguments":{},"name":"measure"}</tool>')

    with pytest.raises(ValueError, match="execution custody"):
        PersistentToolLearner(model, TamperingExecution()).episode(frame())

    assert_one_resolution(model, "failure")


@pytest.mark.parametrize("mutation", ["action", "receipt"])
def test_executor_cannot_mutate_action_or_receipt_custody(mutation: str) -> None:
    class MutatingCustody(FakeExecution):
        def execute(self, action: ToolAction) -> ExecutionReceipt:
            if mutation == "action":
                receipt = super().execute(action)
                object.__setattr__(action, "tool_name", "changed")
                return receipt
            receipt = ExecutionReceipt.seal(
                action,
                accepted=True,
                succeeded=False,
            )
            object.__setattr__(receipt, "succeeded", True)
            return receipt

    model = FakeModel('<tool>{"arguments":{},"name":"measure"}</tool>')

    with pytest.raises(ValueError, match="execution custody"):
        PersistentToolLearner(model, MutatingCustody()).episode(frame())

    assert_one_resolution(model, "failure")


def test_execution_cannot_mutate_model_before_resolution() -> None:
    model = FakeModel('<tool>{"arguments":{},"name":"measure"}</tool>')

    class MutatingExecution(FakeExecution):
        def execute(self, action: ToolAction) -> ExecutionReceipt:
            receipt = super().execute(action)
            model._digest_version += 1
            return receipt

    with pytest.raises(RuntimeError, match="execution mutated"):
        PersistentToolLearner(model, MutatingExecution()).episode(frame())

    assert_one_resolution(model, "failure")


def test_neutral_resolution_cannot_change_training_digest() -> None:
    model = FakeModel("<stop>done</stop>", learn_on=("neutral",))

    with pytest.raises(RuntimeError, match="neutral resolution mutated"):
        PersistentToolLearner(model, FakeExecution()).episode(frame())

    assert_one_resolution(model, "neutral")
