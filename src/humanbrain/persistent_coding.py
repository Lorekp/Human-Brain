"""One deep interface for persistent, target-free coding episodes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from humanbrain.pointer_binding import (
    SymbolInventory,
    materialize,
    python_interface_inventory,
)

_MAX_DECISION_TOKENS = 512
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
    return sha256(domain.encode("ascii") + b"\0" + _json(value).encode()).hexdigest()


def _materialize_pointer_output(
    raw_pointer_output: str,
    inventory: SymbolInventory,
) -> tuple[str | None, str | None]:
    try:
        return materialize(raw_pointer_output, inventory), None
    except (TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


@dataclass(frozen=True, slots=True)
class ObservationBlock:
    """One content-authenticated public repository observation."""

    kind: str
    text: str
    sha256: str

    @classmethod
    def seal(cls, kind: str, text: str) -> ObservationBlock:
        payload = {"kind": kind, "text": text}
        return cls(kind, text, _digest("observation-block-v1", payload))

    def __post_init__(self) -> None:
        payload = {"kind": self.kind, "text": self.text}
        if (
            not self.kind
            or not isinstance(self.text, str)
            or self.sha256 != _digest("observation-block-v1", payload)
        ):
            raise ValueError("observation block is not authentic")


@dataclass(frozen=True, slots=True)
class ToolAffordance:
    """A generic tool operation, never a task-specific candidate action."""

    name: str
    input_schema_json: str
    sha256: str

    @classmethod
    def seal(cls, name: str, input_schema: Mapping[str, object]) -> ToolAffordance:
        schema = _json(dict(input_schema))
        payload = {"input_schema_json": schema, "name": name}
        return cls(name, schema, _digest("tool-affordance-v1", payload))

    def __post_init__(self) -> None:
        try:
            schema = json.loads(self.input_schema_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("tool schema must be canonical JSON") from exc
        payload = {"input_schema_json": self.input_schema_json, "name": self.name}
        if (
            not self.name
            or not isinstance(schema, dict)
            or _json(schema) != self.input_schema_json
            or self.sha256 != _digest("tool-affordance-v1", payload)
        ):
            raise ValueError("tool affordance is not authentic")


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    """Authenticated target-free observations and generic affordances."""

    blocks: tuple[ObservationBlock, ...]
    affordances: tuple[ToolAffordance, ...]
    sha256: str

    @classmethod
    def seal(
        cls,
        blocks: Iterable[ObservationBlock],
        affordances: Iterable[ToolAffordance],
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
            _digest("observation-frame-v1", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "affordances": [item.sha256 for item in self.affordances],
            "blocks": [item.sha256 for item in self.blocks],
        }
        names = [item.name for item in self.affordances]
        if (
            not self.blocks
            or len(names) != len(set(names))
            or self.sha256 != _digest("observation-frame-v1", payload)
        ):
            raise ValueError("observation frame is not authentic")


@dataclass(frozen=True, slots=True)
class CompletionTrace:
    """Authenticated custody from neuronal pointers to decoded text."""

    frame_sha256: str
    model_digest: str
    starter_block_sha256: str
    inventory_symbols: tuple[str, ...]
    inventory_sha256: str
    raw_pointer_output: str
    materialized_output: str | None
    materialization_error: str | None
    sha256: str

    @classmethod
    def _seal(
        cls,
        frame: ObservationFrame,
        model_digest: str,
        starter_block: ObservationBlock,
        inventory: SymbolInventory,
        raw_pointer_output: str,
    ) -> CompletionTrace:
        authenticated_starter = _starter_source(frame)
        entry_point = _entry_point(frame)
        if (
            starter_block != authenticated_starter
            or inventory
            != python_interface_inventory(authenticated_starter.text, entry_point)
        ):
            raise ValueError(
                "completion inventory is not derived from the public interface"
            )
        materialized_output, materialization_error = _materialize_pointer_output(
            raw_pointer_output,
            inventory,
        )
        payload = {
            "frame_sha256": frame.sha256,
            "inventory_sha256": inventory.sha256,
            "inventory_symbols": inventory.symbols,
            "materialization_error": materialization_error,
            "materialized_output": materialized_output,
            "model_digest": model_digest,
            "raw_pointer_output": raw_pointer_output,
            "starter_block_sha256": starter_block.sha256,
        }
        return cls(
            frame.sha256,
            model_digest,
            starter_block.sha256,
            inventory.symbols,
            inventory.sha256,
            raw_pointer_output,
            materialized_output,
            materialization_error,
            _digest("completion-trace-v1", payload),
        )

    def __post_init__(self) -> None:
        inventory = SymbolInventory(self.inventory_symbols)
        materialized_output, materialization_error = _materialize_pointer_output(
            self.raw_pointer_output,
            inventory,
        )
        payload = {
            "frame_sha256": self.frame_sha256,
            "inventory_sha256": self.inventory_sha256,
            "inventory_symbols": self.inventory_symbols,
            "materialization_error": self.materialization_error,
            "materialized_output": self.materialized_output,
            "model_digest": self.model_digest,
            "raw_pointer_output": self.raw_pointer_output,
            "starter_block_sha256": self.starter_block_sha256,
        }
        if (
            not self.frame_sha256
            or not self.model_digest
            or not self.starter_block_sha256
            or inventory.sha256 != self.inventory_sha256
            or self.materialized_output != materialized_output
            or self.materialization_error != materialization_error
            or (self.materialized_output is None)
            == (self.materialization_error is None)
            or self.sha256 != _digest("completion-trace-v1", payload)
        ):
            raise ValueError("completion trace is not authentic")


@dataclass(frozen=True, slots=True)
class ToolAction:
    """One generic invocation authored by neuronal model output."""

    frame_sha256: str
    model_digest: str
    completion_trace_sha256: str
    tool_name: str
    arguments_json: str
    sha256: str

    @classmethod
    def _seal(
        cls,
        frame: ObservationFrame,
        completion_trace: CompletionTrace,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> ToolAction:
        if completion_trace.frame_sha256 != frame.sha256:
            raise ValueError("completion trace is not bound to the observation frame")
        arguments_json = _json(dict(arguments))
        payload = {
            "arguments_json": arguments_json,
            "completion_trace_sha256": completion_trace.sha256,
            "frame_sha256": frame.sha256,
            "model_digest": completion_trace.model_digest,
            "tool_name": tool_name,
        }
        return cls(
            frame.sha256,
            completion_trace.model_digest,
            completion_trace.sha256,
            tool_name,
            arguments_json,
            _digest("tool-action-v2", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "arguments_json": self.arguments_json,
            "completion_trace_sha256": self.completion_trace_sha256,
            "frame_sha256": self.frame_sha256,
            "model_digest": self.model_digest,
            "tool_name": self.tool_name,
        }
        if (
            not self.completion_trace_sha256
            or not self.tool_name
            or self.sha256 != _digest("tool-action-v2", payload)
        ):
            raise ValueError("tool action is not authentic")


@dataclass(frozen=True, slots=True)
class StopDecision:
    """Authenticated uncertainty; STOP never crosses the execution seam."""

    frame_sha256: str
    model_digest: str
    reason: str
    sha256: str

    @classmethod
    def _seal(
        cls, frame: ObservationFrame, model_digest: str, reason: str
    ) -> StopDecision:
        payload = {
            "frame_sha256": frame.sha256,
            "model_digest": model_digest,
            "reason": reason,
        }
        return cls(frame.sha256, model_digest, reason, _digest("stop-v1", payload))

    def __post_init__(self) -> None:
        payload = {
            "frame_sha256": self.frame_sha256,
            "model_digest": self.model_digest,
            "reason": self.reason,
        }
        if not self.reason or self.sha256 != _digest("stop-v1", payload):
            raise ValueError("STOP decision is not authentic")


Decision = ToolAction | StopDecision


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Authenticated public consequence of exactly one tool action."""

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
            _digest("execution-receipt-v1", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "accepted": self.accepted,
            "action_sha256": self.action_sha256,
            "frame_sha256": self.frame_sha256,
            "observations": [item.sha256 for item in self.observations],
            "succeeded": self.succeeded,
        }
        if (self.succeeded and not self.accepted) or self.sha256 != _digest(
            "execution-receipt-v1", payload
        ):
            raise ValueError("execution receipt is not authentic")


class RepositoryExecution(Protocol):
    """Seam satisfied by real-workspace and in-memory adapters."""

    def execute(self, action: ToolAction) -> ExecutionReceipt: ...


class PointerLearnerModel(Protocol):
    """Pointer-normalized neuronal model seam, including no-update wrappers."""

    @property
    def training_digest(self) -> str: ...

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str: ...

    def observe_failure(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        feedback_text: str,
    ) -> object | None: ...

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    completion_trace: CompletionTrace
    decision: Decision
    receipt: ExecutionReceipt | None
    prior_model_digest: str
    model_digest: str
    learning_applied: bool
    learning_outcome: str | None
    sha256: str

    @classmethod
    def _seal(
        cls,
        completion_trace: CompletionTrace,
        decision: Decision,
        receipt: ExecutionReceipt | None,
        prior_digest: str,
        model_digest: str,
    ) -> EpisodeResult:
        learned = prior_digest != model_digest
        learning_outcome = (
            None
            if not learned
            else "success"
            if receipt is not None and receipt.accepted and receipt.succeeded
            else "failure"
        )
        payload = {
            "completion_trace": completion_trace.sha256,
            "decision": decision.sha256,
            "learning_applied": learned,
            "learning_outcome": learning_outcome,
            "model_digest": model_digest,
            "prior_model_digest": prior_digest,
            "receipt": None if receipt is None else receipt.sha256,
        }
        return cls(
            completion_trace,
            decision,
            receipt,
            prior_digest,
            model_digest,
            learned,
            learning_outcome,
            _digest("episode-result-v3", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "completion_trace": self.completion_trace.sha256,
            "decision": self.decision.sha256,
            "learning_applied": self.learning_applied,
            "learning_outcome": self.learning_outcome,
            "model_digest": self.model_digest,
            "prior_model_digest": self.prior_model_digest,
            "receipt": None if self.receipt is None else self.receipt.sha256,
        }
        stopped = isinstance(self.decision, StopDecision)
        materialized = self.completion_trace.materialized_output
        explicit_neuronal_stop = (
            stopped
            and self.completion_trace.materialization_error is None
            and isinstance(materialized, str)
            and _STOP.fullmatch(materialized.strip()) is not None
        )
        receipt_is_bound = self.receipt is None or (
            isinstance(self.decision, ToolAction)
            and self.receipt.action_sha256 == self.decision.sha256
            and self.receipt.frame_sha256 == self.completion_trace.frame_sha256
        )
        expected_learning_outcome = (
            None
            if not self.learning_applied
            else "success"
            if self.receipt is not None
            and self.receipt.accepted
            and self.receipt.succeeded
            else "failure"
        )
        if (
            self.completion_trace.frame_sha256 != self.decision.frame_sha256
            or self.completion_trace.model_digest != self.decision.model_digest
            or self.completion_trace.model_digest != self.prior_model_digest
            or (
                isinstance(self.decision, ToolAction)
                and self.decision.completion_trace_sha256
                != self.completion_trace.sha256
            )
            or stopped != (self.receipt is None)
            or not receipt_is_bound
            or self.learning_outcome != expected_learning_outcome
            or self.learning_applied != (self.prior_model_digest != self.model_digest)
            or (
                explicit_neuronal_stop
                and self.learning_applied
            )
            or (
                self.completion_trace.materialization_error is not None
                and (
                    not stopped
                    or self.decision.reason != "pointer_binding_failed_closed"
                )
            )
            or self.sha256 != _digest("episode-result-v3", payload)
        ):
            raise ValueError("episode result is not authentic")

    def is_authentic(self) -> bool:
        payload = {
            "completion_trace": self.completion_trace.sha256,
            "decision": self.decision.sha256,
            "learning_applied": self.learning_applied,
            "learning_outcome": self.learning_outcome,
            "model_digest": self.model_digest,
            "prior_model_digest": self.prior_model_digest,
            "receipt": None if self.receipt is None else self.receipt.sha256,
        }
        return self.sha256 == _digest("episode-result-v3", payload)


class PersistentCodingLearner:
    """Inspect, act once or STOP, then learn from one authenticated consequence."""

    def __init__(
        self,
        model: PointerLearnerModel,
        execution: RepositoryExecution,
    ) -> None:
        if not model.training_digest:
            raise ValueError("pointer learner model must expose its training digest")
        self._model = model
        self._execution = execution

    @property
    def model_digest(self) -> str:
        return self._model.training_digest

    def episode(self, frame: ObservationFrame) -> EpisodeResult:
        """Run one causally ordered target-free coding episode."""

        if not isinstance(frame, ObservationFrame):
            raise TypeError("episode requires an ObservationFrame")
        starter_block = _starter_source(frame)
        entry_point = _entry_point(frame)
        inventory = python_interface_inventory(starter_block.text, entry_point)
        prior_digest = self._model.training_digest
        prompt = _prompt(frame)
        cue_text = _routing_cue_text(frame)
        raw_pointer_output = self._model.raw_complete(
            prompt,
            inventory,
            cue_text=cue_text,
            max_new_tokens=_MAX_DECISION_TOKENS,
            stop_text=("</tool>", "</stop>"),
        )
        if self._model.training_digest != prior_digest:
            raise RuntimeError("inference mutated the neuronal model")

        completion_trace = CompletionTrace._seal(
            frame,
            prior_digest,
            starter_block,
            inventory,
            raw_pointer_output,
        )
        decision = _decode(frame, completion_trace)
        if isinstance(decision, StopDecision):
            materialized = completion_trace.materialized_output
            explicit_neuronal_stop = (
                completion_trace.materialization_error is None
                and isinstance(materialized, str)
                and _STOP.fullmatch(materialized.strip()) is not None
            )
            if not explicit_neuronal_stop:
                self._model.observe_failure(
                    prompt,
                    inventory,
                    cue_text=cue_text,
                    feedback_text=_generation_failure_feedback(completion_trace),
                )
            return EpisodeResult._seal(
                completion_trace,
                decision,
                None,
                prior_digest,
                self._model.training_digest,
            )

        receipt = self._execution.execute(decision)
        if (
            not isinstance(receipt, ExecutionReceipt)
            or receipt.action_sha256 != decision.sha256
            or receipt.frame_sha256 != frame.sha256
        ):
            raise ValueError("execution receipt is not bound to the selected action")
        if self._model.training_digest != prior_digest:
            raise RuntimeError("execution mutated the neuronal model")
        if not receipt.accepted or not receipt.succeeded:
            self._model.observe_failure(
                prompt,
                inventory,
                cue_text=cue_text,
                feedback_text=_failure_feedback(decision, receipt),
            )
            return EpisodeResult._seal(
                completion_trace,
                decision,
                receipt,
                prior_digest,
                self._model.training_digest,
            )

        self._model.learn(
            prompt,
            _positive_suffix(decision, receipt),
            inventory,
            cue_text=cue_text,
        )
        return EpisodeResult._seal(
            completion_trace,
            decision,
            receipt,
            prior_digest,
            self._model.training_digest,
        )


def _starter_source(frame: ObservationFrame) -> ObservationBlock:
    starter_blocks = tuple(
        block for block in frame.blocks if block.kind == "starter-source"
    )
    if len(starter_blocks) != 1:
        raise ValueError("frame must contain exactly one starter-source block")
    starter_block = starter_blocks[0]
    ObservationBlock(
        starter_block.kind,
        starter_block.text,
        starter_block.sha256,
    )
    return starter_block


def _entry_point(frame: ObservationFrame) -> str:
    signature_blocks = tuple(
        block for block in frame.blocks if block.kind == "typed-signature"
    )
    if len(signature_blocks) != 1:
        raise ValueError("frame must contain exactly one typed-signature block")
    signature_block = signature_blocks[0]
    ObservationBlock(
        signature_block.kind,
        signature_block.text,
        signature_block.sha256,
    )
    try:
        signature = json.loads(signature_block.text)
    except json.JSONDecodeError as error:
        raise ValueError("typed-signature block must contain canonical JSON") from error
    if (
        not isinstance(signature, dict)
        or _json(signature) != signature_block.text
        or not isinstance(signature.get("entry_point"), str)
        or not signature["entry_point"]
    ):
        raise ValueError(
            "typed-signature block must be canonical JSON with entry_point"
        )
    return signature["entry_point"]


def _prompt(frame: ObservationFrame) -> str:
    envelope = {
        "affordances": [
            {"input_schema": json.loads(item.input_schema_json), "name": item.name}
            for item in frame.affordances
        ],
        "observations": [
            {"kind": item.kind, "text": item.text} for item in frame.blocks
        ],
    }
    return (
        "persistent-coding-v1\n"
        f"{_json(envelope)}\n"
        "Reply with exactly one <tool>{canonical JSON}</tool> containing only "
        '"arguments" and "name", or <stop>reason</stop>.'
    )


def _routing_cue_text(frame: ObservationFrame) -> str:
    """Expose the bounded public task channel without task-family metadata."""

    task_blocks = tuple(block.text for block in frame.blocks if block.kind == "public-task")
    if len(task_blocks) != 1 or not task_blocks[0]:
        raise ValueError("one non-empty public-task block is required")
    return task_blocks[0]


def _decode(frame: ObservationFrame, completion_trace: CompletionTrace) -> Decision:
    if completion_trace.frame_sha256 != frame.sha256:
        raise ValueError("completion trace is not bound to the observation frame")
    model_digest = completion_trace.model_digest
    if completion_trace.materialization_error is not None:
        return StopDecision._seal(
            frame,
            model_digest,
            "pointer_binding_failed_closed",
        )
    output = completion_trace.materialized_output
    if not isinstance(output, str):
        return StopDecision._seal(frame, model_digest, "malformed_model_output")
    text = output.strip()
    if match := _STOP.fullmatch(text):
        return StopDecision._seal(frame, model_digest, match.group(1).strip())
    match = _TOOL.fullmatch(text)
    if match is None:
        return StopDecision._seal(frame, model_digest, "malformed_model_output")
    try:
        command = json.loads(match.group(1))
    except json.JSONDecodeError:
        command = None
    if (
        not isinstance(command, dict)
        or set(command) != {"arguments", "name"}
        or _json(command) != match.group(1)
        or not isinstance(command["name"], str)
        or not isinstance(command["arguments"], dict)
    ):
        return StopDecision._seal(frame, model_digest, "malformed_tool_action")
    if command["name"] not in {item.name for item in frame.affordances}:
        return StopDecision._seal(frame, model_digest, "unavailable_tool")
    return ToolAction._seal(
        frame,
        completion_trace,
        command["name"],
        command["arguments"],
    )


def _positive_suffix(
    action: ToolAction,
    receipt: ExecutionReceipt,
) -> str:
    action_json = _json(
        {"arguments": json.loads(action.arguments_json), "name": action.tool_name}
    )
    consequence = _json(
        {
            "observations": [
                {"kind": item.kind, "text": item.text}
                for item in receipt.observations
            ],
            "receipt_sha256": receipt.sha256,
        }
    )
    return f"<tool>{action_json}</tool>\n<execution>{consequence}</execution>"


def _failure_feedback(action: ToolAction, receipt: ExecutionReceipt) -> str:
    """Serialize one authenticated negative consequence without interpreting it."""

    return _json(
        {
            "accepted": receipt.accepted,
            "action": {
                "arguments": json.loads(action.arguments_json),
                "tool": action.tool_name,
            },
            "observations": [
                {"kind": observation.kind, "text": observation.text}
                for observation in receipt.observations
            ],
            "succeeded": receipt.succeeded,
        }
    )


def _generation_failure_feedback(completion_trace: CompletionTrace) -> str:
    """Expose malformed generation as failure, distinct from neuronal STOP."""

    return _json(
        {
            "generation_error": completion_trace.materialization_error,
            "raw_pointer_output": completion_trace.raw_pointer_output,
        }
    )
