from __future__ import annotations

import json

import pytest
from humanbrain.episodic_pointer import (
    CueRoutedPointerCellModel,
    RoutedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.persistent_coding import (
    CompletionTrace,
    EpisodeResult,
    ExecutionReceipt,
    ObservationBlock,
    ObservationFrame,
    PersistentCodingLearner,
    StopDecision,
    ToolAction,
    ToolAffordance,
    _entry_point,
    _prompt,
)
from humanbrain.pointer_binding import (
    PointerBoundCellModel,
    PointerTrainingExample,
    identifier_inventory,
    pointerize,
    python_interface_inventory,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _tool_output(source: str) -> str:
    command = {"arguments": {"source": source}, "name": "write_source"}
    return f"<tool>{_canonical_json(command)}</tool>"


class FakeRepositoryExecution:
    def __init__(self, *, succeeded: bool = True) -> None:
        self.actions: list[ToolAction] = []
        self.succeeded = succeeded

    def execute(self, action: ToolAction) -> ExecutionReceipt:
        self.actions.append(action)
        return ExecutionReceipt.seal(
            action,
            accepted=True,
            succeeded=self.succeeded,
            observations=(ObservationBlock.seal("tool-output", "tests passed"),),
        )


def test_entry_point_requires_one_canonical_typed_signature_block() -> None:
    signature = ObservationBlock.seal(
        "typed-signature",
        _canonical_json({"entry_point": "solve", "result": "integer"}),
    )
    frame = ObservationFrame.seal((signature,), ())

    assert _entry_point(frame) == "solve"

    duplicate = ObservationFrame.seal((signature, signature), ())
    with pytest.raises(ValueError, match="exactly one typed-signature"):
        _entry_point(duplicate)

    noncanonical = ObservationFrame.seal(
        (ObservationBlock.seal("typed-signature", '{"entry_point": "solve"}'),),
        (),
    )
    with pytest.raises(ValueError, match="canonical JSON with entry_point"):
        _entry_point(noncanonical)


def test_pointer_completion_custody_and_success_only_learning() -> None:
    starter_source = (
        "def solve(arg0, arg1, arg2):\n"
        "    raise NotImplementedError\n"
    )
    current_source = (
        "def solve(arg0, arg1, arg2):\n"
        "    return arg0\n"
    )
    successful_source = (
        "def solve(arg0, arg1, arg2):\n"
        "    return abs(arg0 - arg1)\n"
    )
    starter_block = ObservationBlock.seal("starter-source", starter_source)
    frame = ObservationFrame.seal(
        blocks=(
            starter_block,
            ObservationBlock.seal(
                "typed-signature",
                _canonical_json({"entry_point": "solve"}),
            ),
            ObservationBlock.seal("current-source", current_source),
            ObservationBlock.seal("public-task", "Return the absolute gap."),
        ),
        affordances=(
            ToolAffordance.seal(
                "write_source",
                {
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                    "type": "object",
                },
            ),
        ),
    )
    prompt = _prompt(frame)
    inventory = python_interface_inventory(starter_source, "solve")
    tool_output = _tool_output(successful_source)
    example = PointerTrainingExample(prompt, tool_output, inventory)
    config = ModelConfig(
        max_order=256,
        max_phrases=256,
        min_phrase_frequency=1,
    )
    model = PointerBoundCellModel.grow((example,), config)
    expected_model = PointerBoundCellModel.grow((example,), config)
    prior_digest = model.training_digest
    execution = FakeRepositoryExecution()

    result = PersistentCodingLearner(model, execution).episode(frame)

    assert isinstance(result, EpisodeResult)
    assert isinstance(result.completion_trace, CompletionTrace)
    assert result.completion_trace.frame_sha256 == frame.sha256
    assert result.completion_trace.model_digest == prior_digest
    assert result.completion_trace.starter_block_sha256 == starter_block.sha256
    assert result.completion_trace.inventory_symbols == inventory.symbols
    assert result.completion_trace.inventory_sha256 == inventory.sha256
    assert result.completion_trace.raw_pointer_output == pointerize(
        tool_output,
        inventory,
    )
    assert result.completion_trace.materialized_output == tool_output
    assert result.completion_trace.materialization_error is None
    assert isinstance(result.decision, ToolAction)
    assert result.decision.completion_trace_sha256 == result.completion_trace.sha256
    assert result.receipt is not None and result.receipt.succeeded
    assert result.receipt.action_sha256 == result.decision.sha256
    assert result.learning_applied
    assert result.prior_model_digest == prior_digest
    assert result.model_digest != prior_digest
    assert result.is_authentic()
    assert execution.actions == [result.decision]

    consequence = _canonical_json(
        {
            "observations": [{"kind": "tool-output", "text": "tests passed"}],
            "receipt_sha256": result.receipt.sha256,
        }
    )
    positive_suffix = f"{tool_output}\n<execution>{consequence}</execution>"
    expected_model.learn(
        prompt,
        positive_suffix,
        inventory,
        cue_text="Return the absolute gap.",
    )
    assert result.model_digest == expected_model.training_digest

    extended_inventory = identifier_inventory(starter_source + "extra_name = 0\n")
    out_of_range_output = _tool_output("extra_name")
    stopped_model = PointerBoundCellModel.grow(
        (PointerTrainingExample(prompt, out_of_range_output, extended_inventory),),
        config,
    )
    stopped_prior_digest = stopped_model.training_digest
    stopped_execution = FakeRepositoryExecution()
    stopped = PersistentCodingLearner(stopped_model, stopped_execution).episode(frame)

    assert isinstance(stopped.decision, StopDecision)
    assert stopped.decision.reason == "pointer_binding_failed_closed"
    assert stopped.completion_trace.materialized_output is None
    assert "pointer index 5" in (stopped.completion_trace.materialization_error or "")
    assert "<@5>" in stopped.completion_trace.raw_pointer_output
    assert stopped.receipt is None
    assert not stopped.learning_applied
    assert stopped.model_digest == stopped_prior_digest
    assert stopped.is_authentic()
    assert stopped_execution.actions == []


def test_failed_receipt_activates_failure_memory_without_learning() -> None:
    starter_source = (
        "def helper(value):\n"
        "    return value\n\n"
        "def solve(arg0, arg1, arg2):\n"
        "    raise NotImplementedError\n"
    )
    frame = ObservationFrame.seal(
        blocks=(
            ObservationBlock.seal("starter-source", starter_source),
            ObservationBlock.seal(
                "typed-signature",
                _canonical_json({"entry_point": "solve"}),
            ),
            ObservationBlock.seal("public-task", "Return the absolute gap."),
        ),
        affordances=(
            ToolAffordance.seal(
                "write_source",
                {
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                    "type": "object",
                },
            ),
        ),
    )
    inventory = python_interface_inventory(starter_source, "solve")
    prompt = _prompt(frame)
    action = RoutedPointerTrainingExample(
        PointerTrainingExample(
            prompt,
            _tool_output(
                "def solve(arg0, arg1, arg2):\n"
                "    return abs(arg0 - arg1)\n"
            ),
            inventory,
        ),
        "Return the absolute gap.",
    )
    stop = RoutedPointerTrainingExample(
        PointerTrainingExample(
            "unsupported request",
            "<stop>uncertain</stop>",
            inventory,
        ),
        "Decline unsupported requests.",
    )
    model = CueRoutedPointerCellModel.grow(
        (action, stop),
        ModelConfig(
            max_order=256,
            max_phrases=256,
            min_phrase_frequency=1,
        ),
    )
    prior_digest = model.training_digest
    execution = FakeRepositoryExecution(succeeded=False)

    result = PersistentCodingLearner(model, execution).episode(frame)

    assert isinstance(result.decision, ToolAction)
    assert result.receipt is not None and not result.receipt.succeeded
    assert model.failure_events == 1
    assert model.training_digest == prior_digest
    assert result.model_digest == prior_digest
    assert not result.learning_applied
    assert result.is_authentic()
    assert execution.actions == [result.decision]
