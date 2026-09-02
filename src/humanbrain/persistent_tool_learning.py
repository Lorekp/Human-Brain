from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol

_MAX_DECISION_TOKENS = 512
_STOP_TEXT = ("</tool>", "</stop>")
_STOP = re.compile(r"<stop>([^<>]+)</stop>", re.DOTALL)
_TOOL = re.compile(r"<tool>(\{.*\})</tool>", re.DOTALL)


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(domain: str, value: object) -> str:
    encoded = _json(value).encode("utf-8")
    return sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()


def _prompt_sha256(prompt: str) -> str:
    return _digest("persistent-tool-prompt-v1", prompt)


def _query_sha256(
    prompt: str,
    model_digest: str,
    max_new_tokens: int,
    stop_text: tuple[str, ...],
) -> str:
    return _digest(
        "persistent-tool-model-query-v1",
        {
            "max_new_tokens": max_new_tokens,
            "model_digest": model_digest,
            "prompt_sha256": _prompt_sha256(prompt),
            "stop_text": stop_text,
        },
    )


@dataclass(frozen=True, slots=True)
class ObservationBlock:
    kind: str
    text: str
    sha256: str

    @classmethod
    def seal(cls, kind: str, text: str) -> ObservationBlock:
        payload = {"kind": kind, "text": text}
        return cls(kind, text, _digest("tool-observation-block-v1", payload))

    def __post_init__(self) -> None:
        payload = {"kind": self.kind, "text": self.text}
        if (
            not isinstance(self.kind, str)
            or not self.kind
            or not isinstance(self.text, str)
            or self.sha256 != _digest("tool-observation-block-v1", payload)
        ):
            raise ValueError("observation block is not authentic")


@dataclass(frozen=True, slots=True)
class ToolAffordance:
    name: str
    input_schema_json: str
    sha256: str

    @classmethod
    def seal(cls, name: str, input_schema: Mapping[str, object]) -> ToolAffordance:
        schema_json = _json(dict(input_schema))
        payload = {"input_schema_json": schema_json, "name": name}
        return cls(name, schema_json, _digest("tool-affordance-v2", payload))

    def __post_init__(self) -> None:
        try:
            schema = json.loads(self.input_schema_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("tool schema must be canonical JSON") from error
        payload = {"input_schema_json": self.input_schema_json, "name": self.name}
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(schema, dict)
            or _json(schema) != self.input_schema_json
            or self.sha256 != _digest("tool-affordance-v2", payload)
        ):
            raise ValueError("tool affordance is not authentic")


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    blocks: tuple[ObservationBlock, ...]
    affordances: tuple[ToolAffordance, ...]
    sha256: str

    @classmethod
    def seal(
        cls,
        blocks: Iterable[ObservationBlock] = (),
        affordances: Iterable[ToolAffordance] = (),
    ) -> ObservationFrame:
        block_tuple = tuple(blocks)
        affordance_tuple = tuple(affordances)
        payload = {
            "affordances": [item.sha256 for item in affordance_tuple],
            "blocks": [item.sha256 for item in block_tuple],
        }
        return cls(
            block_tuple,
            affordance_tuple,
            _digest("tool-observation-frame-v1", payload),
        )

    def __post_init__(self) -> None:
        if not all(isinstance(item, ObservationBlock) for item in self.blocks):
            raise ValueError("frame contains an invalid observation block")
        if not all(isinstance(item, ToolAffordance) for item in self.affordances):
            raise ValueError("frame contains an invalid tool affordance")
        for item in self.blocks:
            ObservationBlock(item.kind, item.text, item.sha256)
        for item in self.affordances:
            ToolAffordance(item.name, item.input_schema_json, item.sha256)
        names = [item.name for item in self.affordances]
        payload = {
            "affordances": [item.sha256 for item in self.affordances],
            "blocks": [item.sha256 for item in self.blocks],
        }
        if len(names) != len(set(names)) or self.sha256 != _digest(
            "tool-observation-frame-v1", payload
        ):
            raise ValueError("observation frame is not authentic")


@dataclass(frozen=True, slots=True)
class ModelEmission:
    query_sha256: str
    raw_output: str
    selected_stop: bool

    @classmethod
    def seal(
        cls,
        prompt: str,
        model_digest: str,
        raw_output: str,
        *,
        selected_stop: bool = False,
        max_new_tokens: int = _MAX_DECISION_TOKENS,
        stop_text: tuple[str, ...] = _STOP_TEXT,
    ) -> ModelEmission:
        if not isinstance(raw_output, str) or not isinstance(selected_stop, bool):
            raise TypeError("model emission has invalid fields")
        return cls(
            _query_sha256(prompt, model_digest, max_new_tokens, stop_text),
            raw_output,
            selected_stop,
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.query_sha256, str)
            or not self.query_sha256
            or not isinstance(self.raw_output, str)
            or not isinstance(self.selected_stop, bool)
        ):
            raise ValueError("model emission is invalid")


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    frame_sha256: str
    prompt_sha256: str
    pre_inference_digest: str
    model_query_sha256: str
    max_new_tokens: int
    raw_output: str
    selected_stop: bool
    sha256: str

    @classmethod
    def _seal(
        cls,
        frame: ObservationFrame,
        prompt: str,
        pre_inference_digest: str,
        emission: ModelEmission,
        *,
        max_new_tokens: int,
    ) -> GenerationTrace:
        expected_query = _query_sha256(
            prompt,
            pre_inference_digest,
            max_new_tokens,
            _STOP_TEXT,
        )
        if emission.query_sha256 != expected_query:
            raise ValueError("model emission is not bound to the prompt query")
        payload = {
            "frame_sha256": frame.sha256,
            "model_query_sha256": emission.query_sha256,
            "max_new_tokens": max_new_tokens,
            "pre_inference_digest": pre_inference_digest,
            "prompt_sha256": _prompt_sha256(prompt),
            "raw_output": emission.raw_output,
            "selected_stop": emission.selected_stop,
        }
        return cls(
            frame.sha256,
            payload["prompt_sha256"],
            pre_inference_digest,
            emission.query_sha256,
            max_new_tokens,
            emission.raw_output,
            emission.selected_stop,
            _digest("generation-trace-v2", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "frame_sha256": self.frame_sha256,
            "model_query_sha256": self.model_query_sha256,
            "max_new_tokens": self.max_new_tokens,
            "pre_inference_digest": self.pre_inference_digest,
            "prompt_sha256": self.prompt_sha256,
            "raw_output": self.raw_output,
            "selected_stop": self.selected_stop,
        }
        if (
            not self.frame_sha256
            or not self.prompt_sha256
            or not self.pre_inference_digest
            or not self.model_query_sha256
            or type(self.max_new_tokens) is not int
            or self.max_new_tokens < 1
            or not isinstance(self.raw_output, str)
            or not isinstance(self.selected_stop, bool)
            or self.sha256 != _digest("generation-trace-v2", payload)
        ):
            raise ValueError("generation trace is not authentic")


@dataclass(frozen=True, slots=True)
class ToolAction:
    frame_sha256: str
    generation_trace_sha256: str
    model_query_sha256: str
    tool_name: str
    arguments_json: str
    sha256: str

    @classmethod
    def _seal(
        cls,
        frame: ObservationFrame,
        trace: GenerationTrace,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> ToolAction:
        if trace.frame_sha256 != frame.sha256:
            raise ValueError("generation trace is not bound to the frame")
        arguments_json = _json(dict(arguments))
        payload = {
            "arguments_json": arguments_json,
            "frame_sha256": frame.sha256,
            "generation_trace_sha256": trace.sha256,
            "model_query_sha256": trace.model_query_sha256,
            "tool_name": tool_name,
        }
        return cls(
            frame.sha256,
            trace.sha256,
            trace.model_query_sha256,
            tool_name,
            arguments_json,
            _digest("tool-action-v3", payload),
        )

    def __post_init__(self) -> None:
        try:
            arguments = json.loads(self.arguments_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("tool arguments must be canonical JSON") from error
        payload = {
            "arguments_json": self.arguments_json,
            "frame_sha256": self.frame_sha256,
            "generation_trace_sha256": self.generation_trace_sha256,
            "model_query_sha256": self.model_query_sha256,
            "tool_name": self.tool_name,
        }
        if (
            not self.frame_sha256
            or not self.generation_trace_sha256
            or not self.model_query_sha256
            or not self.tool_name
            or not isinstance(arguments, dict)
            or _json(arguments) != self.arguments_json
            or self.sha256 != _digest("tool-action-v3", payload)
        ):
            raise ValueError("tool action is not authentic")


@dataclass(frozen=True, slots=True)
class StopDecision:
    frame_sha256: str
    generation_trace_sha256: str
    model_query_sha256: str
    reason: str
    sha256: str

    @classmethod
    def _seal(
        cls,
        frame: ObservationFrame,
        trace: GenerationTrace,
        reason: str,
    ) -> StopDecision:
        if trace.frame_sha256 != frame.sha256:
            raise ValueError("generation trace is not bound to the frame")
        payload = {
            "frame_sha256": frame.sha256,
            "generation_trace_sha256": trace.sha256,
            "model_query_sha256": trace.model_query_sha256,
            "reason": reason,
        }
        return cls(
            frame.sha256,
            trace.sha256,
            trace.model_query_sha256,
            reason,
            _digest("tool-stop-decision-v1", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "frame_sha256": self.frame_sha256,
            "generation_trace_sha256": self.generation_trace_sha256,
            "model_query_sha256": self.model_query_sha256,
            "reason": self.reason,
        }
        if (
            not self.generation_trace_sha256
            or not self.reason
            or self.sha256 != _digest("tool-stop-decision-v1", payload)
        ):
            raise ValueError("STOP decision is not authentic")


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    action_sha256: str
    frame_sha256: str
    accepted: bool
    succeeded: bool
    observations: tuple[ObservationBlock, ...]
    sha256: str

    @classmethod
    def seal(
        cls,
        action: ToolAction,
        *,
        accepted: bool,
        succeeded: bool,
        observations: Iterable[ObservationBlock] = (),
    ) -> ExecutionReceipt:
        blocks = tuple(observations)
        payload = {
            "accepted": accepted,
            "action_sha256": action.sha256,
            "frame_sha256": action.frame_sha256,
            "observations": [item.sha256 for item in blocks],
            "succeeded": succeeded,
        }
        return cls(
            action.sha256,
            action.frame_sha256,
            accepted,
            succeeded,
            blocks,
            _digest("tool-execution-receipt-v1", payload),
        )

    def __post_init__(self) -> None:
        if not all(isinstance(item, ObservationBlock) for item in self.observations):
            raise ValueError("execution receipt contains an invalid observation")
        for item in self.observations:
            ObservationBlock(item.kind, item.text, item.sha256)
        payload = {
            "accepted": self.accepted,
            "action_sha256": self.action_sha256,
            "frame_sha256": self.frame_sha256,
            "observations": [item.sha256 for item in self.observations],
            "succeeded": self.succeeded,
        }
        if (
            type(self.accepted) is not bool
            or type(self.succeeded) is not bool
            or (self.succeeded and not self.accepted)
            or self.sha256 != _digest("tool-execution-receipt-v1", payload)
        ):
            raise ValueError("execution receipt is not authentic")


class ToolExecution(Protocol):
    def execute(self, action: ToolAction) -> ExecutionReceipt: ...


class PersistentToolModel(Protocol):
    @property
    def training_digest(self) -> str: ...

    def infer(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> ModelEmission: ...

    def resolve_success(
        self,
        query_sha256: str,
        consequence_text: str,
        strength: float,
    ) -> object | None: ...

    def resolve_failure(
        self,
        query_sha256: str,
        consequence_text: str,
    ) -> object | None: ...

    def resolve_neutral(self, query_sha256: str) -> object | None: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    generation_trace: GenerationTrace
    decision: ToolAction | StopDecision
    receipt: ExecutionReceipt | None
    resolution: Literal["neutral", "failure", "success"]
    prior_model_digest: str
    model_digest: str
    learning_applied: bool
    sha256: str

    @classmethod
    def _seal(
        cls,
        trace: GenerationTrace,
        decision: ToolAction | StopDecision,
        receipt: ExecutionReceipt | None,
        resolution: Literal["neutral", "failure", "success"],
        prior_model_digest: str,
        model_digest: str,
    ) -> EpisodeResult:
        payload = {
            "decision_sha256": decision.sha256,
            "generation_trace_sha256": trace.sha256,
            "learning_applied": prior_model_digest != model_digest,
            "model_digest": model_digest,
            "prior_model_digest": prior_model_digest,
            "receipt_sha256": None if receipt is None else receipt.sha256,
            "resolution": resolution,
        }
        return cls(
            trace,
            decision,
            receipt,
            resolution,
            prior_model_digest,
            model_digest,
            prior_model_digest != model_digest,
            _digest("persistent-tool-episode-result-v1", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "decision_sha256": self.decision.sha256,
            "generation_trace_sha256": self.generation_trace.sha256,
            "learning_applied": self.learning_applied,
            "model_digest": self.model_digest,
            "prior_model_digest": self.prior_model_digest,
            "receipt_sha256": None if self.receipt is None else self.receipt.sha256,
            "resolution": self.resolution,
        }
        receipt_bound = self.receipt is None or (
            isinstance(self.decision, ToolAction)
            and self.receipt.action_sha256 == self.decision.sha256
            and self.receipt.frame_sha256 == self.decision.frame_sha256
        )
        action_has_receipt = isinstance(self.decision, StopDecision) == (self.receipt is None)
        successful_receipt = (
            self.receipt is not None and self.receipt.accepted and self.receipt.succeeded
        )
        consequence_matches = (
            (self.resolution == "neutral" and isinstance(self.decision, StopDecision))
            or (self.resolution == "success" and successful_receipt)
            or (self.resolution == "failure" and not successful_receipt)
        )
        if (
            self.generation_trace.sha256 != self.decision.generation_trace_sha256
            or self.generation_trace.frame_sha256 != self.decision.frame_sha256
            or self.generation_trace.pre_inference_digest != self.prior_model_digest
            or self.generation_trace.model_query_sha256 != self.decision.model_query_sha256
            or not receipt_bound
            or not action_has_receipt
            or not consequence_matches
            or self.learning_applied != (self.prior_model_digest != self.model_digest)
            or (self.resolution == "neutral" and self.learning_applied)
            or self.sha256 != _digest("persistent-tool-episode-result-v1", payload)
        ):
            raise ValueError("episode result is not authentic")


class PersistentToolLearner:
    def __init__(
        self,
        model: PersistentToolModel,
        execution: ToolExecution,
        *,
        max_new_tokens: int = _MAX_DECISION_TOKENS,
    ) -> None:
        if not model.training_digest:
            raise ValueError("persistent model must expose its training digest")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self._model = model
        self._execution = execution
        self._max_new_tokens = max_new_tokens

    def episode(self, frame: ObservationFrame) -> EpisodeResult:
        if not isinstance(frame, ObservationFrame):
            raise TypeError("episode requires an ObservationFrame")
        ObservationFrame(frame.blocks, frame.affordances, frame.sha256)
        prior_digest = self._model.training_digest
        prompt = prompt_for_frame(frame)
        expected_query = _query_sha256(
            prompt,
            prior_digest,
            self._max_new_tokens,
            _STOP_TEXT,
        )
        try:
            emission = self._model.infer(
                prompt,
                max_new_tokens=self._max_new_tokens,
                stop_text=_STOP_TEXT,
            )
        except Exception as error:
            _cleanup_expected_query(self._model, expected_query, "infer_raised", error)
            raise
        if not isinstance(emission, ModelEmission):
            error = TypeError("model infer must return ModelEmission")
            _cleanup_expected_query(self._model, expected_query, "invalid_emission_type", error)
            raise error
        if self._model.training_digest != prior_digest:
            error = RuntimeError("inference mutated the neuronal model")
            _cleanup_expected_query(self._model, expected_query, "inference_mutated_model", error)
            raise error
        if emission.query_sha256 != expected_query:
            error = ValueError("model emission is not bound to the prompt query")
            _cleanup_expected_query(self._model, expected_query, "query_binding_failed", error)
            raise error
        try:
            trace = GenerationTrace._seal(
                frame,
                prompt,
                prior_digest,
                emission,
                max_new_tokens=self._max_new_tokens,
            )
        except (TypeError, ValueError) as error:
            _cleanup_expected_query(
                self._model, expected_query, "trace_authentication_failed", error
            )
            raise

        decision, resolution = _decode(frame, trace)
        if isinstance(decision, StopDecision):
            if resolution == "neutral":
                self._model.resolve_neutral(trace.model_query_sha256)
                if self._model.training_digest != prior_digest:
                    raise RuntimeError("neutral resolution mutated the neuronal model")
            else:
                self._model.resolve_failure(
                    trace.model_query_sha256,
                    _generation_failure(trace, decision),
                )
            return EpisodeResult._seal(
                trace,
                decision,
                None,
                resolution,
                prior_digest,
                self._model.training_digest,
            )

        try:
            receipt = self._execution.execute(decision)
        except Exception as error:
            _cleanup_expected_query(
                self._model,
                expected_query,
                f"execution_raised:{type(error).__name__}",
                error,
            )
            raise
        try:
            _authenticate_execution_custody(frame, decision, receipt)
        except (AttributeError, TypeError, ValueError) as cause:
            error = ValueError("execution custody is not authentic")
            _cleanup_expected_query(self._model, expected_query, "execution_custody_failed", error)
            raise error from cause
        if self._model.training_digest != prior_digest:
            error = RuntimeError("execution mutated the neuronal model")
            _cleanup_expected_query(self._model, expected_query, "execution_mutated_model", error)
            raise error

        if receipt.accepted and receipt.succeeded:
            self._model.resolve_success(
                trace.model_query_sha256,
                _consequence(decision, receipt),
                1.0,
            )
            resolution = "success"
        else:
            self._model.resolve_failure(
                trace.model_query_sha256,
                _consequence(decision, receipt),
            )
            resolution = "failure"
        return EpisodeResult._seal(
            trace,
            decision,
            receipt,
            resolution,
            prior_digest,
            self._model.training_digest,
        )


def prompt_for_frame(frame: ObservationFrame) -> str:
    """Render the canonical sensory prompt shared by growth and inference."""

    if not isinstance(frame, ObservationFrame):
        raise TypeError("prompt_for_frame requires an ObservationFrame")
    ObservationFrame(frame.blocks, frame.affordances, frame.sha256)
    envelope = {
        "affordances": [
            {"input_schema": json.loads(item.input_schema_json), "name": item.name}
            for item in frame.affordances
        ],
        "observations": [{"kind": item.kind, "text": item.text} for item in frame.blocks],
    }
    return (
        "persistent-tool-learning-v1\n"
        f"{_json(envelope)}\n"
        "Reply with exactly one <tool>{canonical JSON}</tool> containing only "
        '"arguments" and "name", or <stop>reason</stop>.'
    )


def _cleanup_expected_query(
    model: PersistentToolModel,
    expected_query: str,
    protocol_error: str,
    original_error: Exception,
) -> None:
    try:
        model.resolve_failure(
            expected_query,
            _json({"protocol_error": protocol_error}),
        )
    except Exception as cleanup_error:
        original_error.add_note(
            f"expected-query cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        )


def _authenticate_execution_custody(
    frame: ObservationFrame,
    action: ToolAction,
    receipt: object,
) -> None:
    ToolAction(
        action.frame_sha256,
        action.generation_trace_sha256,
        action.model_query_sha256,
        action.tool_name,
        action.arguments_json,
        action.sha256,
    )
    if not isinstance(receipt, ExecutionReceipt):
        raise TypeError("executor must return ExecutionReceipt")
    ExecutionReceipt(
        receipt.action_sha256,
        receipt.frame_sha256,
        receipt.accepted,
        receipt.succeeded,
        receipt.observations,
        receipt.sha256,
    )
    if receipt.action_sha256 != action.sha256 or receipt.frame_sha256 != frame.sha256:
        raise ValueError("execution receipt is not bound to the selected action")


def _decode(
    frame: ObservationFrame,
    trace: GenerationTrace,
) -> tuple[ToolAction | StopDecision, Literal["neutral", "failure"]]:
    if trace.frame_sha256 != frame.sha256:
        raise ValueError("generation trace is not bound to the observation frame")
    if trace.selected_stop:
        return StopDecision._seal(frame, trace, "tonic_stop"), "neutral"
    text = trace.raw_output.strip()
    if match := _STOP.fullmatch(text):
        reason = match.group(1).strip()
        if reason:
            return StopDecision._seal(frame, trace, reason), "neutral"
    match = _TOOL.fullmatch(text)
    if match is None:
        return StopDecision._seal(frame, trace, "malformed_model_output"), "failure"
    try:
        command = json.loads(match.group(1))
        valid = (
            isinstance(command, dict)
            and set(command) == {"arguments", "name"}
            and _json(command) == match.group(1)
            and isinstance(command["name"], str)
            and isinstance(command["arguments"], dict)
        )
    except (json.JSONDecodeError, OverflowError, TypeError, ValueError):
        command = None
        valid = False
    if not valid:
        return StopDecision._seal(frame, trace, "malformed_tool_action"), "failure"
    if command["name"] not in {item.name for item in frame.affordances}:
        return StopDecision._seal(frame, trace, "unavailable_tool"), "failure"
    return (
        ToolAction._seal(frame, trace, command["name"], command["arguments"]),
        "failure",
    )


def _generation_failure(trace: GenerationTrace, decision: StopDecision) -> str:
    return _json(
        {
            "generation_trace_sha256": trace.sha256,
            "raw_output": trace.raw_output,
            "reason": decision.reason,
        }
    )


def _consequence(action: ToolAction, receipt: ExecutionReceipt) -> str:
    return _json(
        {
            "accepted": receipt.accepted,
            "action": {
                "arguments": json.loads(action.arguments_json),
                "name": action.tool_name,
                "sha256": action.sha256,
            },
            "observations": [
                {"kind": item.kind, "text": item.text} for item in receipt.observations
            ],
            "receipt_sha256": receipt.sha256,
            "succeeded": receipt.succeeded,
        }
    )
