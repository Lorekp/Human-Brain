"""Prepared V5 integrated continual coding session; importing does not spend it.

The fresh public-TRAIN V5 stream drives one persistent treatment learner, one
independently grown no-update control, and one independently grown no-update
inventory-loop ablation.  All three begin from the same eight public coding
successes.  No evaluator-private label enters learner cognition; hidden policy
and tests open only after a learner action has been sealed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from experiments.r4_integrated.continual_toy_swe_tasks_v5 import (
    TASK_COUNT,
    TRAIN_STREAM_CUSTODY_SHA256,
    TRAIN_STREAM_SHA256,
    ContinualToySweTrainTaskV5,
    LearnerVisibleToySweObservationV5,
    continual_toy_swe_train_stream_v5,
    learner_visible_observation_v5,
    train_stream_custody_v5,
)
from experiments.r4_integrated.public_coding_prior_v1 import (
    PUBLIC_CODING_PRIOR_V1_SHA256,
    SOURCE_CORPUS_SHA256,
    SOURCE_FREEZE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    PublicCodingPriorCaseV1,
    public_coding_prior_v1,
)
from humanbrain import persistent_coding as coding
from humanbrain.factorized_pointer import (
    INVENTORY_LOOP_OFF,
    FactorizedCompletionTrace,
    FactorizedLearningUpdate,
    FactorizedPointerCellModel,
    FactorizedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import SymbolInventory, python_interface_inventory
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.sandbox import (
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)

SCHEMA = "continual-coding-session-v5"
RESERVATION_SCHEMA = "continual-coding-session-reservation-v5"
WRITE_TOOL = "write_source"
MAX_ATTEMPTS = 3
PUBLIC_PRIOR_CASES = 8

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
WORKER_PATH = REPO_ROOT / "src/humanbrain/simple_swe/_worker.py"
OUTPUT_PATH = REPO_ROOT / "benchmarks/r4_integrated_continual_coding_session_v5.json"
FACTORIZED_GATE_PATH = REPO_ROOT / "benchmarks/r4_factorized_output_binding_gate_v1.json"
INVENTORY_GATE_PATH = REPO_ROOT / "benchmarks/r4_variable_cardinality_inventory_gate_v1.json"

_MODEL_CONFIG = {
    "max_order": 12,
    "max_phrases": 256,
    "min_phrase_frequency": 2,
    "random_seed": 7,
}
_FROZEN_DEPENDENCIES = {
    "benchmarks/r4_factorized_output_binding_gate_v1.json": (
        "301e40786f47a9e6ce09d03e7049524faa2bed9d6d928c66dd57cb2e7ce33376"
    ),
    "benchmarks/r4_variable_cardinality_inventory_gate_v1.json": (
        "99da8382989e2fe8008f05fac06de4cdcf415269ad2fa462b311a23a15fc211a"
    ),
    "benchmarks/simple_swe_protocol_v1.json": (
        "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
    ),
    "experiments/r4_integrated/continual_toy_swe_tasks_v5.py": (
        "c73b173265f2d4478fc03a9fffbd6d4d657e4f21aa29e0a47fcb89152ca3d8a4"
    ),
    "experiments/r4_integrated/public_coding_prior_v1.py": (
        "95b0bbae62cc1fc08dd94e0f25e32116c2bbe671c7d5ceb10b50a4617ff0ac5b"
    ),
    "src/humanbrain/cortex.py": (
        "9cf37fc36978c3389a86a2856401f6c528d1a95f491dab0eb1b8662edcd055db"
    ),
    "src/humanbrain/factorized_pointer.py": (
        "d34ed27d64600c0ee8e903425aebd4568fbb0a31beab848f8badc9cb32cbde36"
    ),
    "src/humanbrain/language.py": (
        "f376ebd49e25b327b362eb856de94cddfedc5cc57337699d0160fb1139064f31"
    ),
    "src/humanbrain/persistent_coding.py": (
        "bd62f06b4a4f8d84a0d05a8294f753a603ac1f9af76e950c62f8fabbb366eb1d"
    ),
    "src/humanbrain/pointer_binding.py": (
        "0cf13bd13be981c27d1eb00c1feb3fa08039c543654b17b02fa1b0e07e38171d"
    ),
    "src/humanbrain/simple_swe/_worker.py": (
        "258aaf45decb8ed6635a85b92ee970591932b2cf53b691149a0d66777601c870"
    ),
    "src/humanbrain/tokenizer.py": (
        "08ef114a758006868944ad3091ca66619d3c082086ea2d44d93211d82a9791ce"
    ),
}
_UNKNOWN_CUE = "unseen request with insufficient evidence"
_UNKNOWN_STARTER = (
    "def pending_solution():\n"
    '    """No task evidence has been supplied."""\n'
    "    raise NotImplementedError\n"
)
_STOP = re.compile(r"<stop>([^<>]+)</stop>", re.DOTALL)
_TOOL = re.compile(r"<tool>(\{.*\})</tool>", re.DOTALL)
_PRIVATE_FIELDS = frozenset(
    {
        "attempt_budget",
        "capability_group",
        "expected_action",
        "feedback_pressure",
        "generated_task",
        "private_environment_sha256",
        "private_test_sha256",
        "property_cases",
        "regression_cases",
        "retention_anchor_ordinal",
        "stage",
        "test_sha256",
    }
)


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


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str | None) -> str | None:
    return None if value is None else sha256(value.encode()).hexdigest()


def _assert_frozen_dependencies() -> None:
    for relative_path, expected in _FROZEN_DEPENDENCIES.items():
        actual = _sha256_file(REPO_ROOT / relative_path)
        if actual != expected:
            raise RuntimeError(f"frozen V5 dependency changed: {relative_path}")


@dataclass(frozen=True, slots=True)
class _SandboxRuntime:
    protocol: ProtocolSpec
    executable: Path
    executable_sha256: str
    worker_source: bytes
    worker_sha256: str


def _sandbox_runtime() -> _SandboxRuntime:
    protocol = load_protocol(PROTOCOL_PATH)
    executable = resolve_python_executable()
    identity = python_runtime_identity(executable, protocol.raw["language"])
    worker_sha256 = _sha256_file(WORKER_PATH)
    return _SandboxRuntime(
        protocol=protocol,
        executable=executable,
        executable_sha256=identity["executable_sha256"],
        worker_source=load_worker_snapshot(worker_sha256),
        worker_sha256=worker_sha256,
    )


class ToySweExecutionAdapter:
    """Execute only source carried by a sealed learner ToolAction."""

    def __init__(
        self,
        tasks: tuple[ContinualToySweTrainTaskV5, ...],
        runtime: _SandboxRuntime,
    ) -> None:
        self._tasks = {task.task_sha256: task for task in tasks}
        self._runtime = runtime
        self._task_by_frame: dict[str, ContinualToySweTrainTaskV5] = {}
        self._source_by_task = {
            task.task_sha256: task.public.starter_source for task in tasks
        }
        self._audit_by_action: dict[str, dict[str, object]] = {}
        self.sandbox_executions = 0
        self.policy_feedback_events = 0

    def bind(
        self,
        frame: coding.ObservationFrame,
        task: ContinualToySweTrainTaskV5,
    ) -> None:
        task.authenticate()
        if task.task_sha256 not in self._tasks:
            raise ValueError("frame task is outside the V5 TRAIN stream")
        self._task_by_frame[frame.sha256] = task

    def source(self, task: ContinualToySweTrainTaskV5) -> str:
        return self._source_by_task[task.task_sha256]

    def audit(self, action: coding.ToolAction) -> dict[str, object] | None:
        value = self._audit_by_action.get(action.sha256)
        return None if value is None else dict(value)

    def execute(self, action: coding.ToolAction) -> coding.ExecutionReceipt:
        task = self._task_by_frame.get(action.frame_sha256)
        if task is None:
            self._audit_by_action[action.sha256] = {
                "kind": "tool-rejected",
                "reason": "unbound-frame",
            }
            return coding.ExecutionReceipt.seal(
                action,
                accepted=False,
                succeeded=False,
                observations=(
                    coding.ObservationBlock.seal("tool-error", "unbound frame"),
                ),
            )
        try:
            arguments = json.loads(action.arguments_json)
        except json.JSONDecodeError:
            arguments = None
        if (
            action.tool_name != WRITE_TOOL
            or not isinstance(arguments, dict)
            or set(arguments) != {"source"}
            or not isinstance(arguments["source"], str)
        ):
            self._audit_by_action[action.sha256] = {
                "kind": "tool-rejected",
                "reason": "invalid-write-source-shape",
            }
            return coding.ExecutionReceipt.seal(
                action,
                accepted=False,
                succeeded=False,
                observations=(
                    coding.ObservationBlock.seal(
                        "tool-error",
                        "write_source requires one source string",
                    ),
                ),
            )

        source = arguments["source"]
        source_sha256 = sha256(source.encode()).hexdigest()
        self._source_by_task[task.task_sha256] = source

        # Evaluator policy opens only after this exact action has been sealed.
        if task.private.expected_action == "stop":
            self.policy_feedback_events += 1
            self._audit_by_action[action.sha256] = {
                "kind": "environment-feedback",
                "sealed_action_sha256": action.sha256,
                "source_sha256": source_sha256,
                "sandbox_invoked": False,
                "outcome": "insufficient_evidence",
            }
            return coding.ExecutionReceipt.seal(
                action,
                accepted=True,
                succeeded=False,
                observations=(
                    coding.ObservationBlock.seal(
                        "environment-outcome",
                        "insufficient_evidence",
                    ),
                    coding.ObservationBlock.seal(
                        "environment-detail",
                        "no executable oracle is committed for this request",
                    ),
                ),
            )

        generated = task.private.generated_task
        result = execute_completion(
            source,
            generated.public,
            generated.private,
            self._runtime.protocol.execution,
            python_executable=self._runtime.executable,
            python_executable_sha256=self._runtime.executable_sha256,
            worker_source=self._runtime.worker_source,
            worker_sha256=self._runtime.worker_sha256,
        )
        self.sandbox_executions += 1
        self._audit_by_action[action.sha256] = {
            "kind": "sandbox",
            "sealed_action_sha256": action.sha256,
            "source_sha256": source_sha256,
            "sandbox_invoked": True,
            "outcome": result.outcome,
            "passed": result.passed,
            "sandboxed": result.sandboxed,
            "wall_seconds": result.wall_seconds,
        }
        return coding.ExecutionReceipt.seal(
            action,
            accepted=True,
            succeeded=result.passed,
            observations=(
                coding.ObservationBlock.seal("sandbox-outcome", result.outcome),
                coding.ObservationBlock.seal("sandbox-detail", result.detail),
            ),
        )


class _SessionModel:
    """Expose one factorized model with explicit plasticity and ablation policy."""

    def __init__(
        self,
        model: FactorizedPointerCellModel,
        *,
        plasticity: bool,
        ablation: str | None = None,
        audit_no_update_requests: bool = True,
    ) -> None:
        self._model = model
        self._plasticity = plasticity
        self._ablation = ablation
        self._audit_no_update_requests = audit_no_update_requests
        self.last_update: FactorizedLearningUpdate | None = None
        self.last_update_request: dict[str, str] | None = None

    @property
    def training_digest(self) -> str:
        return self._model.training_digest

    @property
    def grown_branches(self) -> int:
        return self._model.grown_branches

    @property
    def cell_count(self) -> int:
        return self._model.cell_count

    @property
    def vocabulary_size(self) -> int:
        return self._model.vocabulary_size

    @property
    def last_trace(self) -> FactorizedCompletionTrace | None:
        return self._model.last_trace

    def _reset_audit(self) -> None:
        self.last_update = None
        self.last_update_request = None

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 64,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        self._reset_audit()
        return self._model.raw_complete(
            prompt,
            inventory,
            cue_text=cue_text,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
            ablation=self._ablation,
        )

    def complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 64,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        return self.raw_complete(
            prompt,
            inventory,
            cue_text=cue_text,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )

    def observe_failure(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        feedback_text: str,
    ) -> FactorizedLearningUpdate | None:
        if self._plasticity or self._audit_no_update_requests:
            self.last_update_request = {
                "polarity": "failure",
                "evidence_sha256": sha256(feedback_text.encode()).hexdigest(),
            }
        if not self._plasticity:
            return None
        self.last_update = self._model.observe_failure(
            prompt,
            inventory,
            cue_text=cue_text,
            feedback_text=feedback_text,
        )
        return self.last_update

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> FactorizedLearningUpdate | None:
        if self._plasticity or self._audit_no_update_requests:
            self.last_update_request = {
                "polarity": "success",
                "evidence_sha256": sha256(output.encode()).hexdigest(),
            }
        if not self._plasticity:
            return None
        self.last_update = self._model.learn(
            prompt,
            output,
            inventory,
            cue_text=cue_text,
            strength=strength,
        )
        return self.last_update


_WRITE_AFFORDANCE = coding.ToolAffordance.seal(
    WRITE_TOOL,
    {
        "additionalProperties": False,
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
        "type": "object",
    },
)


def _frame_blocks(
    *,
    starter_source: str,
    current_source: str,
    signature: dict[str, object],
    public_prompt: str,
    task_commitment_sha256: str | None = None,
    relative_path: str | None = None,
    previous: coding.ExecutionReceipt | None = None,
) -> tuple[coding.ObservationBlock, ...]:
    blocks = [
        coding.ObservationBlock.seal("starter-source", starter_source),
        coding.ObservationBlock.seal("current-source", current_source),
    ]
    if task_commitment_sha256 is not None:
        blocks.append(
            coding.ObservationBlock.seal(
                "task-commitment",
                task_commitment_sha256,
            )
        )
    if relative_path is not None:
        blocks.append(coding.ObservationBlock.seal("repository-path", relative_path))
    blocks.append(coding.ObservationBlock.seal("typed-signature", _json(signature)))
    if previous is not None:
        blocks.extend(previous.observations)
    blocks.append(coding.ObservationBlock.seal("public-task", public_prompt))
    return tuple(blocks)


def _frame(
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV5,
    previous: coding.ExecutionReceipt | None = None,
) -> coding.ObservationFrame:
    public = learner_visible_observation_v5(task)
    frame = coding.ObservationFrame.seal(
        _frame_blocks(
            starter_source=public.starter_source,
            current_source=adapter.source(task),
            signature=public.signature.to_record(),
            public_prompt=public.prompt,
            task_commitment_sha256=public.task_commitment_sha256,
            relative_path=public.relative_path,
            previous=previous,
        ),
        (_WRITE_AFFORDANCE,),
    )
    adapter.bind(frame, task)
    return frame


def _training_frame(
    starter_source: str,
    entry_point: str,
    public_prompt: str,
) -> coding.ObservationFrame:
    return coding.ObservationFrame.seal(
        _frame_blocks(
            starter_source=starter_source,
            current_source=starter_source,
            signature={"entry_point": entry_point},
            public_prompt=public_prompt,
        ),
        (_WRITE_AFFORDANCE,),
    )


def _bootstrap_examples() -> tuple[FactorizedPointerTrainingExample, ...]:
    prior = public_coding_prior_v1()
    if len(prior) != PUBLIC_PRIOR_CASES:
        raise ValueError("frozen public coding prior must contain exactly eight cases")
    examples = []
    for index, case in enumerate(prior):
        frame = _training_frame(case.starter_source, case.entrypoint, case.public_prompt)
        payload = {
            "arguments": {"source": case.successful_source},
            "name": WRITE_TOOL,
        }
        examples.append(
            FactorizedPointerTrainingExample(
                prompt=coding._prompt(frame),
                output=f"<tool>{_json(payload)}</tool>",
                inventory=python_interface_inventory(
                    case.starter_source,
                    case.entrypoint,
                ),
                cue_text=case.public_prompt,
                provenance_id=(
                    f"public-coding-prior-v1:{index}:"
                    f"{sha256(case.public_prompt.encode()).hexdigest()}"
                ),
            )
        )
    return tuple(examples)


def _bootstrap_corpus_sha256() -> str:
    return _digest(
        "continual-coding-bootstrap-corpus-v5",
        {
            "model_config": _MODEL_CONFIG,
            "examples": [
                {
                    "cue_text": example.cue_text,
                    "inventory_sha256": example.inventory.sha256,
                    "inventory_symbols": example.inventory.symbols,
                    "output": example.output,
                    "prompt": example.prompt,
                    "provenance_id": example.provenance_id,
                }
                for example in _bootstrap_examples()
            ],
        },
    )


def _bootstrap_model() -> FactorizedPointerCellModel:
    return FactorizedPointerCellModel.grow(
        _bootstrap_examples(),
        ModelConfig(**_MODEL_CONFIG),
    )


def _block_record(block: coding.ObservationBlock) -> dict[str, str]:
    return {"kind": block.kind, "text": block.text, "sha256": block.sha256}


def _completion_trace_record(trace: coding.CompletionTrace) -> dict[str, object]:
    return {
        "frame_sha256": trace.frame_sha256,
        "model_digest": trace.model_digest,
        "starter_block_sha256": trace.starter_block_sha256,
        "inventory_symbols": list(trace.inventory_symbols),
        "inventory_sha256": trace.inventory_sha256,
        "raw_pointer_output": trace.raw_pointer_output,
        "raw_pointer_output_sha256": _text_sha256(trace.raw_pointer_output),
        "materialized_output": trace.materialized_output,
        "materialized_output_sha256": _text_sha256(trace.materialized_output),
        "materialization_error": trace.materialization_error,
        "trace_sha256": trace.sha256,
    }


def _factorized_trace_record(trace: FactorizedCompletionTrace) -> dict[str, object]:
    full_steps = [
        {
            "step": step.step,
            "candidate_cells": step.candidate_cells,
            "candidate_scores_hex": tuple(
                float(score).hex() for score in step.candidate_scores
            ),
            "winner_cell": step.winner_cell,
            "winner_kind": step.winner_kind,
            "winner_text": step.winner_text,
            "spike_time_ms_hex": float(step.spike_time_ms).hex(),
            "simulated_steps": step.simulated_steps,
        }
        for step in trace.fragment_steps
    ]
    compact_steps = [
        {
            "step": step["step"],
            "candidate_count": len(step["candidate_cells"]),
            "candidates_sha256": _digest("factorized-step-candidates-v1", step),
            "winner_cell": step["winner_cell"],
            "winner_in_candidates": step["winner_cell"] in step["candidate_cells"],
            "winner_kind": step["winner_kind"],
            "winner_text": step["winner_text"],
            "spike_time_ms_hex": step["spike_time_ms_hex"],
            "simulated_steps": step["simulated_steps"],
        }
        for step in full_steps
    ]
    bindings = [
        {
            "slot": race.slot,
            "winner_cell": race.winner_cell,
            "symbol": race.symbol,
            "candidate_count": race.candidate_count,
            "spike_time_ms_hex": float(race.spike_time_ms).hex(),
            "simulated_steps": race.simulated_steps,
        }
        for race in trace.binding_races
    ]
    inventory_loops = [
        {
            "item_index": race.item_index,
            "candidate_cells": race.candidate_cells,
            "candidate_scores_hex": tuple(
                float(score).hex() for score in race.candidate_scores
            ),
            "winner_cell": race.winner_cell,
            "decision": race.decision,
            "spike_time_ms_hex": float(race.spike_time_ms).hex(),
            "simulated_steps": race.simulated_steps,
        }
        for race in trace.inventory_loop_races
    ]
    full = {
        "query_sha256": trace.query_sha256,
        "model_digest": trace.model_digest,
        "ablation": trace.ablation,
        "cue_words": trace.cue_words,
        "fragment_steps": full_steps,
        "binding_races": bindings,
        "inventory_loop_races": inventory_loops,
        "emitted_fragment_cells": trace.emitted_fragment_cells,
        "emitted_token_ids": trace.emitted_token_ids,
        "selected_stop": trace.selected_stop,
        "terminal_stop": trace.terminal_stop,
        "output": trace.output,
    }
    return {
        "query_sha256": trace.query_sha256,
        "model_digest": trace.model_digest,
        "ablation": trace.ablation,
        "cue_words": list(trace.cue_words),
        "fragment_steps": compact_steps,
        "binding_races": bindings,
        "inventory_loop_races": inventory_loops,
        "emitted_fragment_cells": list(trace.emitted_fragment_cells),
        "emitted_token_ids": list(trace.emitted_token_ids),
        "selected_stop": trace.selected_stop,
        "terminal_stop": trace.terminal_stop,
        "output": trace.output,
        "trace_sha256": _digest("factorized-completion-trace-v2", full),
    }


def _decision_record(decision: coding.Decision) -> dict[str, object]:
    if isinstance(decision, coding.StopDecision):
        return {
            "kind": "stop",
            "frame_sha256": decision.frame_sha256,
            "model_digest": decision.model_digest,
            "reason": decision.reason,
            "decision_sha256": decision.sha256,
        }
    return {
        "kind": "tool",
        "frame_sha256": decision.frame_sha256,
        "model_digest": decision.model_digest,
        "completion_trace_sha256": decision.completion_trace_sha256,
        "tool_name": decision.tool_name,
        "arguments_json": decision.arguments_json,
        "decision_sha256": decision.sha256,
    }


def _receipt_record(
    receipt: coding.ExecutionReceipt | None,
) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "action_sha256": receipt.action_sha256,
        "frame_sha256": receipt.frame_sha256,
        "accepted": receipt.accepted,
        "succeeded": receipt.succeeded,
        "observations": [_block_record(block) for block in receipt.observations],
        "receipt_sha256": receipt.sha256,
    }


def _update_record(
    update: FactorizedLearningUpdate | None,
) -> dict[str, object] | None:
    if update is None:
        return None
    return {
        "prior_training_digest": update.prior_training_digest,
        "training_digest": update.training_digest,
        "polarity": update.polarity,
        "synapses_changed": update.synapses_changed,
    }


def _explicit_stop_text(text: str | None) -> bool:
    return isinstance(text, str) and _STOP.fullmatch(text.strip()) is not None


def _episode_status(result: coding.EpisodeResult, explicit_stop: bool) -> str:
    trace = result.completion_trace
    if trace.materialization_error is not None:
        return "materialization-failure"
    if explicit_stop:
        return "explicit-stop"
    if isinstance(result.decision, coding.StopDecision):
        return f"decoder-stop:{result.decision.reason}"
    receipt = result.receipt
    if receipt is None or not receipt.accepted:
        return "tool-rejected"
    if receipt.succeeded:
        return "passed"
    outcome = next(
        (
            block.text
            for block in receipt.observations
            if block.kind in {"environment-outcome", "sandbox-outcome"}
        ),
        "failed",
    )
    prefix = "environment" if any(
        block.kind == "environment-outcome" for block in receipt.observations
    ) else "sandbox"
    return f"{prefix}:{outcome}"


def _extract_tool_source(output: str | None) -> str | None:
    if not isinstance(output, str):
        return None
    match = _TOOL.fullmatch(output.strip())
    if match is None:
        return None
    try:
        command = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(command, dict)
        or set(command) != {"arguments", "name"}
        or _json(command) != match.group(1)
        or command.get("name") != WRITE_TOOL
        or not isinstance(command.get("arguments"), dict)
        or set(command["arguments"]) != {"source"}
        or not isinstance(command["arguments"]["source"], str)
    ):
        return None
    return command["arguments"]["source"]


def _public_interface_record(
    materialized_output: str | None,
    public: LearnerVisibleToySweObservationV5,
) -> dict[str, object]:
    """Score only the public callable name and ordered public parameter names."""

    expected = (
        public.signature.entry_point,
        *(parameter.name for parameter in public.signature.parameters),
    )
    source = _extract_tool_source(materialized_output)
    observed: tuple[str, ...] | None = None
    error: str | None = None
    if source is None:
        error = "no-canonical-write-source"
    else:
        try:
            observed = python_interface_inventory(
                source,
                public.signature.entry_point,
            ).symbols
        except (TypeError, ValueError) as caught:
            error = type(caught).__name__
    entry_point_exact = observed is not None and observed[:1] == expected[:1]
    parameter_sequence_exact = (
        observed is not None and observed[1:] == expected[1:]
    )
    return {
        "metric": "public-entrypoint-and-ordered-parameter-sequence-exact",
        "expected_symbols": list(expected),
        "observed_symbols": None if observed is None else list(observed),
        "entry_point_exact": entry_point_exact,
        "parameter_sequence_exact": parameter_sequence_exact,
        "exact": entry_point_exact and parameter_sequence_exact,
        "source_sha256": None if source is None else sha256(source.encode()).hexdigest(),
        "error": error,
    }


def _episode_record(
    result: coding.EpisodeResult,
    model: _SessionModel,
    adapter: ToySweExecutionAdapter,
    public: LearnerVisibleToySweObservationV5,
) -> dict[str, object]:
    if not result.is_authentic():
        raise RuntimeError("persistent episode result failed authentication")
    neuronal_trace = model.last_trace
    if neuronal_trace is None:
        raise RuntimeError("factorized completion did not expose its trace")
    completion_trace = result.completion_trace
    if neuronal_trace.output != completion_trace.raw_pointer_output:
        raise RuntimeError("neuronal trace output differs from raw pointer custody")
    if not (
        neuronal_trace.model_digest
        == completion_trace.model_digest
        == result.prior_model_digest
    ):
        raise RuntimeError("neuronal trace is not bound to the pre-plastic model")
    explicit_stop = (
        isinstance(result.decision, coding.StopDecision)
        and completion_trace.materialization_error is None
        and _explicit_stop_text(completion_trace.materialized_output)
    )
    execution = (
        adapter.audit(result.decision)
        if isinstance(result.decision, coding.ToolAction)
        else None
    )
    completion = _completion_trace_record(completion_trace)
    return {
        "status": _episode_status(result, explicit_stop),
        "explicit_neuronal_stop": explicit_stop,
        "materialization_failure": (
            completion_trace.materialization_error is not None
        ),
        "public_interface": _public_interface_record(
            completion_trace.materialized_output,
            public,
        ),
        "raw_pointer_output_sha256": completion["raw_pointer_output_sha256"],
        "materialized_output_sha256": completion["materialized_output_sha256"],
        "factorized_trace": _factorized_trace_record(neuronal_trace),
        "completion_trace": completion,
        "decision": _decision_record(result.decision),
        "receipt": _receipt_record(result.receipt),
        "execution": execution,
        "prior_model_digest": result.prior_model_digest,
        "model_digest": result.model_digest,
        "learning_applied": result.learning_applied,
        "learning_outcome": result.learning_outcome,
        "update_request": (
            None
            if model.last_update_request is None
            else dict(model.last_update_request)
        ),
        "update": _update_record(model.last_update),
        "episode_sha256": result.sha256,
    }


def _run_task(
    learner: coding.PersistentCodingLearner,
    model: _SessionModel,
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV5,
) -> list[dict[str, object]]:
    """Use one public max-three policy without consulting authored budget."""

    frame = _frame(adapter, task)
    attempts: list[dict[str, object]] = []
    for attempt_index in range(MAX_ATTEMPTS):
        result = learner.episode(frame)
        record = _episode_record(result, model, adapter, task.public)
        record["attempt"] = attempt_index + 1
        attempts.append(record)
        if record["status"] == "passed" or record["explicit_neuronal_stop"]:
            break
        if result.receipt is not None:
            frame = _frame(adapter, task, result.receipt)
    return attempts


def _run_loop_off_first_attempt(
    learner: coding.PersistentCodingLearner,
    model: _SessionModel,
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV5,
) -> dict[str, object]:
    """Seal one original-frame ablation decision; never retry or retain feedback."""

    frame = _frame(adapter, task)
    before = model.training_digest
    result = learner.episode(frame)
    record = _episode_record(result, model, adapter, task.public)
    record["attempt"] = 1
    if model.training_digest != before:
        raise RuntimeError("inventory-loop-off no-update arm changed its digest")
    if model.last_update_request is not None or model.last_update is not None:
        raise RuntimeError("inventory-loop-off arm retained environment feedback")
    factorized_trace = record["factorized_trace"]
    if factorized_trace["ablation"] != INVENTORY_LOOP_OFF:
        raise RuntimeError("inventory-loop-off trace lost its causal ablation tag")
    if factorized_trace["inventory_loop_races"]:
        raise RuntimeError("inventory-loop-off trace contained an inventory-loop race")
    return record


class _LocalCollector(ast.NodeVisitor):
    def __init__(self, parameters: frozenset[str]) -> None:
        self.parameters = parameters
        self.locals: list[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        if (
            isinstance(node.ctx, ast.Store)
            and node.id not in self.parameters
            and node.id not in self.locals
        ):
            self.locals.append(node.id)


class _AlphaRename(ast.NodeTransformer):
    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> ast.Name:
        replacement = self.renames.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)


def _alpha_source_key(source: str) -> str | None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    function = tree.body[0]
    parameters = frozenset(argument.arg for argument in function.args.args)
    collector = _LocalCollector(parameters)
    collector.visit(function)
    renames = {
        local: f"_bound_local_{index}" for index, local in enumerate(collector.locals)
    }
    normalized = _AlphaRename(renames).visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _prior_replay_case(
    model: _SessionModel,
    case: PublicCodingPriorCaseV1,
    case_index: int,
) -> dict[str, object]:
    frame = _training_frame(case.starter_source, case.entrypoint, case.public_prompt)
    inventory = python_interface_inventory(case.starter_source, case.entrypoint)
    output = model.complete(
        coding._prompt(frame),
        inventory,
        cue_text=case.public_prompt,
        max_new_tokens=512,
        stop_text=("</tool>", "</stop>"),
    )
    trace = model.last_trace
    if trace is None:
        raise RuntimeError("public-prior replay did not expose a trace")
    expected_payload = {
        "arguments": {"source": case.successful_source},
        "name": WRITE_TOOL,
    }
    expected = f"<tool>{_json(expected_payload)}</tool>"
    source = _extract_tool_source(output)
    return {
        "case_index": case_index,
        "output": output,
        "output_sha256": sha256(output.encode()).hexdigest(),
        "expected_output_sha256": sha256(expected.encode()).hexdigest(),
        "byte_exact": output == expected,
        "valid_write_source": source is not None,
        "source_sha256": None if source is None else sha256(source.encode()).hexdigest(),
        "expected_source_sha256": sha256(case.successful_source.encode()).hexdigest(),
        "alpha_equivalent": (
            source is not None
            and _alpha_source_key(source) == _alpha_source_key(case.successful_source)
        ),
        "terminal_stop": trace.terminal_stop,
        "factorized_trace": _factorized_trace_record(trace),
    }


def _unknown_cue_probe(model: _SessionModel) -> dict[str, object]:
    frame = _training_frame(_UNKNOWN_STARTER, "pending_solution", _UNKNOWN_CUE)
    inventory = python_interface_inventory(_UNKNOWN_STARTER, "pending_solution")
    output = model.complete(
        coding._prompt(frame),
        inventory,
        cue_text=_UNKNOWN_CUE,
        max_new_tokens=512,
        stop_text=("</tool>", "</stop>"),
    )
    trace = model.last_trace
    if trace is None:
        raise RuntimeError("unknown-cue probe did not expose a trace")
    return {
        "cue_sha256": sha256(_UNKNOWN_CUE.encode()).hexdigest(),
        "output": output,
        "explicit_neuronal_stop": _explicit_stop_text(output),
        "factorized_trace": _factorized_trace_record(trace),
    }


def _retention_arm(model: _SessionModel) -> dict[str, object]:
    before = model.training_digest
    cases = [
        _prior_replay_case(model, case, index)
        for index, case in enumerate(public_coding_prior_v1())
    ]
    uncertainty = _unknown_cue_probe(model)
    after = model.training_digest
    if before != after:
        raise RuntimeError("read-only retention replay mutated the model")
    return {
        "model_digest": after,
        "model_digest_unchanged": True,
        "prior_case_count": len(cases),
        "unknown_cue_probe_count": 1,
        "byte_exact": sum(int(case["byte_exact"]) for case in cases),
        "alpha_equivalent": sum(int(case["alpha_equivalent"]) for case in cases),
        "valid_write_source": sum(int(case["valid_write_source"]) for case in cases),
        "terminal_stop": sum(int(case["terminal_stop"]) for case in cases),
        "cases": cases,
        "unknown_cue": uncertainty,
    }


def _retention_checkpoint(
    label: str,
    treatment: _SessionModel,
    control: _SessionModel,
) -> dict[str, object]:
    return {
        "label": label,
        "treatment": _retention_arm(treatment),
        "control": _retention_arm(control),
    }


def _retry_change_record(attempts: list[dict[str, object]]) -> dict[str, object]:
    """Compare adjacent raw and materialized outputs directly, never frame hashes."""

    pairs = []
    for left, right in pairwise(attempts):
        left_completion = left["completion_trace"]
        right_completion = right["completion_trace"]
        raw_changed = (
            left_completion["raw_pointer_output"]
            != right_completion["raw_pointer_output"]
        )
        materialized_changed = (
            left_completion["materialized_output"]
            != right_completion["materialized_output"]
        )
        right_succeeded = right["status"] == "passed"
        if materialized_changed and right_succeeded:
            classification = "changed-recovery"
        elif materialized_changed:
            classification = "changed-unsuccessful"
        elif right_succeeded:
            classification = "identical-later-pass"
        else:
            classification = "repeated-identical-failure"
        pairs.append(
            {
                "from_attempt": left["attempt"],
                "to_attempt": right["attempt"],
                "raw_pointer_output_before_sha256": left[
                    "raw_pointer_output_sha256"
                ],
                "raw_pointer_output_after_sha256": right[
                    "raw_pointer_output_sha256"
                ],
                "raw_pointer_output_changed": raw_changed,
                "materialized_output_before_sha256": left[
                    "materialized_output_sha256"
                ],
                "materialized_output_after_sha256": right[
                    "materialized_output_sha256"
                ],
                "materialized_output_changed": materialized_changed,
                "right_succeeded": right_succeeded,
                "materialized_program_classification": classification,
            }
        )
    return {
        "eligible_adjacent_pairs": len(pairs),
        "raw_pointer_output_changed": any(
            bool(pair["raw_pointer_output_changed"]) for pair in pairs
        ),
        "materialized_output_changed": any(
            bool(pair["materialized_output_changed"]) for pair in pairs
        ),
        "pairs": pairs,
    }


def _arm_score(
    expected_action: str,
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    retry_changes = _retry_change_record(attempts)
    if expected_action == "execute":
        winning = next(
            (
                int(attempt["attempt"])
                for attempt in attempts
                if attempt["status"] == "passed"
            ),
            None,
        )
        observed_terminal = winning
        late_uncertainty = None
    elif expected_action == "stop":
        observed_terminal = next(
            (
                int(attempt["attempt"])
                for attempt in attempts
                if attempt["explicit_neuronal_stop"]
            ),
            None,
        )
        winning = 1 if observed_terminal == 1 else None
        late_uncertainty = (
            observed_terminal
            if observed_terminal is not None and observed_terminal > 1
            else None
        )
    else:
        raise ValueError("unknown evaluator terminal action")
    correct = winning is not None
    if correct:
        failure = None
    elif expected_action == "execute" and any(
        bool(attempt["explicit_neuronal_stop"]) for attempt in attempts
    ):
        failure = "unexpected-explicit-stop"
    elif late_uncertainty is not None:
        failure = "late-explicit-stop"
    elif expected_action == "stop":
        failure = f"missing-explicit-stop:{attempts[-1]['status']}"
    else:
        failure = str(attempts[-1]["status"])
    later_pass = winning is not None and winning > 1
    winning_transition = next(
        (
            pair
            for pair in retry_changes["pairs"]
            if pair["to_attempt"] == winning
        ),
        None,
    )
    winning_transition_classification = (
        None
        if winning_transition is None
        else winning_transition["materialized_program_classification"]
    )
    recovery = (
        later_pass and winning_transition_classification == "changed-recovery"
    )
    later_pass_without_materialized_change = (
        later_pass
        and winning_transition_classification == "identical-later-pass"
    )
    return {
        "correct": correct,
        "winning_attempt": winning,
        "observed_terminal_attempt": observed_terminal,
        "later_pass": later_pass,
        "recovery": recovery,
        "later_pass_without_materialized_change": (
            later_pass_without_materialized_change
        ),
        "winning_transition_classification": winning_transition_classification,
        "late_uncertainty_recovery": late_uncertainty is not None,
        "late_uncertainty_attempt": late_uncertainty,
        "retry_changes": retry_changes,
        "failure": failure,
    }


def _posthoc_score(
    tasks: tuple[ContinualToySweTrainTaskV5, ...],
    task_records: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    stage_scores: dict[str, Counter[str]] = {}
    failures = {
        "treatment": Counter(),
        "control": Counter(),
        "inventory_loop_off": Counter(),
    }
    for task, record in zip(tasks, task_records, strict=True):
        private = task.private
        treatment = _arm_score(private.expected_action, record["attempts"])
        control = _arm_score(private.expected_action, record["control_attempts"])
        loop_off = _arm_score(
            private.expected_action,
            [record["inventory_loop_off_first_attempt"]],
        )
        record.update(
            {
                "stage": private.stage,
                "capability_group": private.capability_group,
                "expected_action": private.expected_action,
                "authored_attempt_budget": private.attempt_budget,
                "authored_budget_used_for_cognition": False,
                "feedback_pressure": private.feedback_pressure,
                "retention_anchor_ordinal": private.retention_anchor_ordinal,
                "private_environment_sha256": private.environment_sha256,
                "score": treatment,
                "control_score": control,
                "inventory_loop_off_score": loop_off,
            }
        )
        counts = stage_scores.setdefault(private.stage, Counter())
        counts["tasks"] += 1
        counts["treatment_correct"] += int(treatment["correct"])
        counts["control_correct"] += int(control["correct"])
        counts["inventory_loop_off_correct"] += int(loop_off["correct"])
        for name, score in (
            ("treatment", treatment),
            ("control", control),
            ("inventory_loop_off", loop_off),
        ):
            if score["failure"] is not None:
                failures[name][str(score["failure"])] += 1
    stage_record = {
        stage: {
            **dict(counts),
            "treatment_rate": counts["treatment_correct"] / counts["tasks"],
            "control_rate": counts["control_correct"] / counts["tasks"],
            "inventory_loop_off_rate": (
                counts["inventory_loop_off_correct"] / counts["tasks"]
            ),
        }
        for stage, counts in sorted(stage_scores.items())
    }
    failure_record = {
        arm: dict(sorted(counter.items())) for arm, counter in failures.items()
    }
    return stage_record, failure_record


def _capability_curve(task_records: list[dict[str, object]]) -> list[dict[str, object]]:
    totals = {
        "treatment_correct": 0,
        "control_correct": 0,
        "inventory_loop_off_correct": 0,
        "treatment_recoveries": 0,
        "control_recoveries": 0,
        "treatment_late_uncertainty_recoveries": 0,
        "control_late_uncertainty_recoveries": 0,
    }
    curve = []
    for index, record in enumerate(task_records):
        score = record["score"]
        control = record["control_score"]
        loop_off = record["inventory_loop_off_score"]
        totals["treatment_correct"] += int(score["correct"])
        totals["control_correct"] += int(control["correct"])
        totals["inventory_loop_off_correct"] += int(loop_off["correct"])
        totals["treatment_recoveries"] += int(score["recovery"])
        totals["control_recoveries"] += int(control["recovery"])
        totals["treatment_late_uncertainty_recoveries"] += int(
            score["late_uncertainty_recovery"]
        )
        totals["control_late_uncertainty_recoveries"] += int(
            control["late_uncertainty_recovery"]
        )
        seen = index + 1
        curve.append(
            {
                "task_index": index,
                "tasks_seen": seen,
                "stage": record["stage"],
                "capability_group": record["capability_group"],
                **totals,
                "treatment_rate": totals["treatment_correct"] / seen,
                "control_rate": totals["control_correct"] / seen,
                "inventory_loop_off_rate": (
                    totals["inventory_loop_off_correct"] / seen
                ),
            }
        )
    return curve


def _delayed_retention_first_attempt_scores(
    task_records: list[dict[str, object]],
) -> dict[str, object]:
    rows = []
    for record in task_records:
        if record["stage"] != "delayed-retention-recheck":
            continue
        expected_action = str(record["expected_action"])
        treatment = _arm_score(expected_action, [record["attempts"][0]])
        control = _arm_score(expected_action, [record["control_attempts"][0]])
        loop_off = _arm_score(
            expected_action,
            [record["inventory_loop_off_first_attempt"]],
        )
        rows.append(
            {
                "task_index": record["task_index"],
                "task_sha256": record["task_sha256"],
                "retention_anchor_ordinal": record["retention_anchor_ordinal"],
                "treatment_first_attempt_correct": treatment["correct"],
                "control_first_attempt_correct": control["correct"],
                "inventory_loop_off_first_attempt_correct": loop_off["correct"],
            }
        )
    ordinals = {int(row["task_index"]) for row in rows}
    if ordinals != {12, 13}:
        raise RuntimeError("V5 delayed-retention first-attempt ordinals changed")
    return {
        "task_count": len(rows),
        "treatment_first_attempt_correct": sum(
            int(row["treatment_first_attempt_correct"]) for row in rows
        ),
        "control_first_attempt_correct": sum(
            int(row["control_first_attempt_correct"]) for row in rows
        ),
        "inventory_loop_off_first_attempt_correct": sum(
            int(row["inventory_loop_off_first_attempt_correct"]) for row in rows
        ),
        "tasks": rows,
    }


def _overall_scores(task_records: list[dict[str, object]]) -> dict[str, object]:
    treatment_correct = sum(int(record["score"]["correct"]) for record in task_records)
    control_correct = sum(
        int(record["control_score"]["correct"]) for record in task_records
    )
    loop_off_correct = sum(
        int(record["inventory_loop_off_score"]["correct"])
        for record in task_records
    )
    treatment_recoveries = sum(
        int(record["score"]["recovery"]) for record in task_records
    )
    control_recoveries = sum(
        int(record["control_score"]["recovery"]) for record in task_records
    )
    return {
        "task_count": len(task_records),
        "treatment_correct": treatment_correct,
        "control_correct": control_correct,
        "inventory_loop_off_correct": loop_off_correct,
        "treatment_minus_control_correct": treatment_correct - control_correct,
        "treatment_minus_inventory_loop_off_correct": (
            treatment_correct - loop_off_correct
        ),
        "treatment_recoveries": treatment_recoveries,
        "control_recoveries": control_recoveries,
        "treatment_minus_control_recoveries": (
            treatment_recoveries - control_recoveries
        ),
    }


def _aggregate_retry_outcomes(
    task_records: list[dict[str, object]],
    score_key: str,
) -> dict[str, object]:
    classifications = Counter(
        {
            "changed-recovery": 0,
            "changed-unsuccessful": 0,
            "identical-later-pass": 0,
            "repeated-identical-failure": 0,
        }
    )
    eligible_retry_tasks = 0
    eligible_adjacent_pairs = 0
    raw_changed_pairs = 0
    materialized_changed_pairs = 0
    for record in task_records:
        retry_changes = record[score_key]["retry_changes"]
        pairs = retry_changes["pairs"]
        if pairs:
            eligible_retry_tasks += 1
        eligible_adjacent_pairs += len(pairs)
        for pair in pairs:
            classifications[pair["materialized_program_classification"]] += 1
            raw_changed_pairs += int(pair["raw_pointer_output_changed"])
            materialized_changed_pairs += int(pair["materialized_output_changed"])
    return {
        "eligible_retry_tasks": eligible_retry_tasks,
        "eligible_adjacent_pairs": eligible_adjacent_pairs,
        "materialized_program_classifications": dict(
            sorted(classifications.items())
        ),
        "raw_pointer_changed_pairs": raw_changed_pairs,
        "materialized_program_changed_pairs": materialized_changed_pairs,
    }


def _public_interface_exactness(
    task_records: list[dict[str, object]],
) -> dict[str, object]:
    arm_records = {
        "treatment_first_attempt": [record["attempts"][0] for record in task_records],
        "control_first_attempt": [
            record["control_attempts"][0] for record in task_records
        ],
        "inventory_loop_off_first_attempt": [
            record["inventory_loop_off_first_attempt"] for record in task_records
        ],
    }
    scores = {
        arm: {
            "exact": sum(
                int(attempt["public_interface"]["exact"])
                for attempt in attempts
            ),
            "total": len(attempts),
        }
        for arm, attempts in arm_records.items()
    }
    scores["treatment_minus_inventory_loop_off"] = (
        scores["treatment_first_attempt"]["exact"]
        - scores["inventory_loop_off_first_attempt"]["exact"]
    )
    scores["control_minus_inventory_loop_off"] = (
        scores["control_first_attempt"]["exact"]
        - scores["inventory_loop_off_first_attempt"]["exact"]
    )
    return {
        "primary_causal_metric": (
            "public-entrypoint-and-ordered-parameter-sequence-exact"
        ),
        "uses_private_tests_or_labels": False,
        "scores": scores,
    }


def _attempt_distribution(
    task_records: list[dict[str, object]],
    key: str,
) -> dict[str, int]:
    if key == "inventory_loop_off_first_attempt":
        attempts = [record[key] for record in task_records]
    else:
        attempts = [attempt for record in task_records for attempt in record[key]]
    return dict(sorted(Counter(str(attempt["status"]) for attempt in attempts).items()))


def _learning_summary(
    task_records: list[dict[str, object]],
    key: str,
) -> dict[str, object]:
    if key == "inventory_loop_off_first_attempt":
        attempts = [record[key] for record in task_records]
    else:
        attempts = [attempt for record in task_records for attempt in record[key]]
    requested = Counter(
        str(attempt["update_request"]["polarity"])
        for attempt in attempts
        if attempt["update_request"] is not None
    )
    applied = Counter(
        str(attempt["update"]["polarity"])
        for attempt in attempts
        if attempt["update"] is not None
    )
    return {
        "requested_polarity": dict(sorted(requested.items())),
        "applied_polarity": dict(sorted(applied.items())),
        "digest_changing_events": sum(
            int(attempt["learning_applied"]) for attempt in attempts
        ),
    }


def _physical_trace_summary(
    task_records: list[dict[str, object]],
    key: str,
) -> dict[str, object]:
    if key == "inventory_loop_off_first_attempt":
        attempts = [record[key] for record in task_records]
    else:
        attempts = [attempt for record in task_records for attempt in record[key]]
    traces = [attempt["factorized_trace"] for attempt in attempts]
    steps = [step for trace in traces for step in trace["fragment_steps"]]
    loops = [race for trace in traces for race in trace["inventory_loop_races"]]
    return {
        "attempt_traces": len(traces),
        "fragment_races": len(steps),
        "fragment_simulated_steps": sum(step["simulated_steps"] for step in steps),
        "all_fragment_races_have_physical_winners": all(
            bool(step["winner_in_candidates"]) and step["simulated_steps"] >= 1
            for step in steps
        ),
        "stop_winner_races": sum(step["winner_kind"] == "stop" for step in steps),
        "inventory_loop_races": len(loops),
        "inventory_loop_simulated_steps": sum(
            race["simulated_steps"] for race in loops
        ),
        "inventory_loop_continue_races": sum(
            race["decision"] == "continue" for race in loops
        ),
        "inventory_loop_exit_races": sum(
            race["decision"] == "exit" for race in loops
        ),
        "all_inventory_loop_races_have_physical_winners": all(
            race["winner_cell"] in race["candidate_cells"]
            and race["simulated_steps"] >= 1
            for race in loops
        ),
        "selected_stop_traces": sum(bool(trace["selected_stop"]) for trace in traces),
        "terminal_stop_traces": sum(bool(trace["terminal_stop"]) for trace in traces),
        "explicit_neuronal_stop_attempts": sum(
            bool(attempt["explicit_neuronal_stop"]) for attempt in attempts
        ),
    }


def _model_identity(
    model: _SessionModel | FactorizedPointerCellModel,
) -> dict[str, object]:
    return {
        "training_digest": model.training_digest,
        "grown_branches": model.grown_branches,
        "cell_count": model.cell_count,
        "vocabulary_size": model.vocabulary_size,
    }


def _execution_custody(runtime: _SandboxRuntime) -> dict[str, str]:
    source_paths = {
        "session_source_sha256": Path(__file__),
        "task_stream_source_sha256": (
            REPO_ROOT / "experiments/r4_integrated/continual_toy_swe_tasks_v5.py"
        ),
        "public_prior_source_sha256": (
            REPO_ROOT / "experiments/r4_integrated/public_coding_prior_v1.py"
        ),
        "persistent_coding_source_sha256": (
            REPO_ROOT / "src/humanbrain/persistent_coding.py"
        ),
        "factorized_pointer_source_sha256": (
            REPO_ROOT / "src/humanbrain/factorized_pointer.py"
        ),
        "pointer_binding_source_sha256": (
            REPO_ROOT / "src/humanbrain/pointer_binding.py"
        ),
        "language_source_sha256": REPO_ROOT / "src/humanbrain/language.py",
        "cortex_source_sha256": REPO_ROOT / "src/humanbrain/cortex.py",
        "tokenizer_source_sha256": REPO_ROOT / "src/humanbrain/tokenizer.py",
        "factorized_gate_artifact_sha256": FACTORIZED_GATE_PATH,
        "variable_cardinality_gate_artifact_sha256": INVENTORY_GATE_PATH,
    }
    custody = {name: _sha256_file(path) for name, path in source_paths.items()}
    custody.update(
        {
            "protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "worker_sha256": runtime.worker_sha256,
            "python_executable_sha256": runtime.executable_sha256,
        }
    )
    return custody


def _corpus_custody() -> dict[str, object]:
    return {
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "train_stream_custody_sha256": TRAIN_STREAM_CUSTODY_SHA256,
        "train_stream_custody": train_stream_custody_v5(),
        "target_task_count": TASK_COUNT,
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "public_prior_cases": PUBLIC_PRIOR_CASES,
        "bootstrap_corpus_sha256": _bootstrap_corpus_sha256(),
        "model_config": dict(_MODEL_CONFIG),
        "source_corpus_sha256": SOURCE_CORPUS_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "target_tasks_present_at_prior_freeze": False,
        "cohort_novel_at_artifact_spend": True,
        "synthetic_stop_supervision_examples": 0,
        "uncertainty_probe_is_untrained": True,
    }


def _reservation_record(
    runtime: _SandboxRuntime,
    preflight: dict[str, object],
) -> dict[str, object]:
    model = _bootstrap_model()
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RUNNING",
        "reservation_policy": "absent-only-exclusive-create-before-first-episode",
        "artifact_path": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
        "initial_model": _model_identity(model),
        "schema_preflight": preflight,
        "corpus_custody": _corpus_custody(),
        "execution_custody": _execution_custody(runtime),
    }


def _prepare() -> tuple[_SandboxRuntime, dict[str, object]]:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"V5 session artifact already exists: {OUTPUT_PATH}")
    _assert_frozen_dependencies()
    preflight = _schema_preflight()
    runtime = _sandbox_runtime()
    return runtime, _reservation_record(runtime, preflight)


def run_session(
    runtime: _SandboxRuntime,
    expected_initial_model: dict[str, object],
) -> dict[str, object]:
    treatment_model = _SessionModel(_bootstrap_model(), plasticity=True)
    control_model = _SessionModel(_bootstrap_model(), plasticity=False)
    loop_off_model = _SessionModel(
        _bootstrap_model(),
        plasticity=False,
        ablation=INVENTORY_LOOP_OFF,
        audit_no_update_requests=False,
    )
    treatment_initial = _model_identity(treatment_model)
    control_initial = _model_identity(control_model)
    loop_off_initial = _model_identity(loop_off_model)
    if not (
        treatment_initial == control_initial == loop_off_initial == expected_initial_model
    ):
        raise RuntimeError("V5 treatment, control, and loop-off growth diverged")

    tasks = continual_toy_swe_train_stream_v5()
    treatment_adapter = ToySweExecutionAdapter(tasks, runtime)
    control_adapter = ToySweExecutionAdapter(tasks, runtime)
    loop_off_adapter = ToySweExecutionAdapter(tasks, runtime)
    treatment = coding.PersistentCodingLearner(treatment_model, treatment_adapter)
    control = coding.PersistentCodingLearner(control_model, control_adapter)
    loop_off = coding.PersistentCodingLearner(loop_off_model, loop_off_adapter)

    retention_curve = [
        _retention_checkpoint("before-v5", treatment_model, control_model)
    ]
    task_records: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        task.authenticate()
        attempts = _run_task(treatment, treatment_model, treatment_adapter, task)
        control_attempts = _run_task(
            control,
            control_model,
            control_adapter,
            task,
        )
        # The causal arm is run only after both main arms have sealed this task.
        loop_off_attempt = _run_loop_off_first_attempt(
            loop_off,
            loop_off_model,
            loop_off_adapter,
            task,
        )
        task_records.append(
            {
                "task_index": task_index,
                "task_sha256": task.task_sha256,
                "public_observation_sha256": task.public.observation_sha256,
                "attempts": attempts,
                "control_attempts": control_attempts,
                "inventory_loop_off_first_attempt": loop_off_attempt,
            }
        )
        retention_curve.append(
            _retention_checkpoint(
                f"after-v5-task-{task_index + 1}",
                treatment_model,
                control_model,
            )
        )

    # Evaluator labels below open only after all three arms sealed all decisions.
    stage_scores, failure_distribution = _posthoc_score(tasks, task_records)
    capability_curve = _capability_curve(task_records)
    treatment_final = _model_identity(treatment_model)
    control_final = _model_identity(control_model)
    loop_off_final = _model_identity(loop_off_model)
    return {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "claim_boundary": (
            "fresh public-TRAIN integrated V5 continual coding evidence only; "
            "not protected evaluation, family-disjoint transfer, Simple-SWE M0, "
            "SWE-bench, external-model comparison, or broad intelligence evidence"
        ),
        "cohort_novel_at_artifact_spend": True,
        "corpus_custody": _corpus_custody(),
        "execution_custody": _execution_custody(runtime),
        "task_count": len(tasks),
        "max_attempts_for_every_task": MAX_ATTEMPTS,
        "private_authored_budgets_used_for_cognition": False,
        "previous_receipt_is_only_retry_feedback": True,
        "one_persistent_treatment_learner": True,
        "one_persistent_no_update_control": True,
        "full_replay_between_tasks": False,
        "target_task_teacher_sources_used": 0,
        "protected_dev_or_eval_inputs": 0,
        "only_sealed_learner_emitted_programs_enter_sandbox": True,
        "typed_signature_blocks_per_frame": 1,
        "retention_checkpoint_count": len(retention_curve),
        "retention_prior_cases_per_arm_per_checkpoint": PUBLIC_PRIOR_CASES,
        "retention_unknown_cue_probes_per_arm_per_checkpoint": 1,
        "treatment_initial_model": treatment_initial,
        "treatment_final_model": treatment_final,
        "control_initial_model": control_initial,
        "control_final_model": control_final,
        "control_model_digest_unchanged": (
            control_initial["training_digest"] == control_final["training_digest"]
        ),
        "inventory_loop_off": {
            "ablation": INVENTORY_LOOP_OFF,
            "protocol": (
                "independently-grown-no-update; one original-frame first attempt "
                "after both main arms; no retry; no retained feedback"
            ),
            "initial_model": loop_off_initial,
            "final_model": loop_off_final,
            "model_digest_unchanged": (
                loop_off_initial["training_digest"]
                == loop_off_final["training_digest"]
            ),
            "public_interface_exactness": _public_interface_exactness(task_records),
        },
        "treatment_learning": _learning_summary(task_records, "attempts"),
        "control_learning": _learning_summary(task_records, "control_attempts"),
        "inventory_loop_off_learning": _learning_summary(
            task_records,
            "inventory_loop_off_first_attempt",
        ),
        "treatment_attempt_outcomes": _attempt_distribution(task_records, "attempts"),
        "control_attempt_outcomes": _attempt_distribution(
            task_records,
            "control_attempts",
        ),
        "inventory_loop_off_attempt_outcomes": _attempt_distribution(
            task_records,
            "inventory_loop_off_first_attempt",
        ),
        "failure_distribution": failure_distribution,
        "overall_scores_and_advantage": _overall_scores(task_records),
        "retry_outcome_summary": {
            "treatment": _aggregate_retry_outcomes(task_records, "score"),
            "control": _aggregate_retry_outcomes(task_records, "control_score"),
        },
        "stage_scores": stage_scores,
        "capability_curve": capability_curve,
        "delayed_retention_first_attempt_scores": (
            _delayed_retention_first_attempt_scores(task_records)
        ),
        "physical_trace_summary": {
            "treatment": _physical_trace_summary(task_records, "attempts"),
            "control": _physical_trace_summary(task_records, "control_attempts"),
            "inventory_loop_off": _physical_trace_summary(
                task_records,
                "inventory_loop_off_first_attempt",
            ),
        },
        "retention_curve": retention_curve,
        "treatment_sandbox_executions": treatment_adapter.sandbox_executions,
        "control_sandbox_executions": control_adapter.sandbox_executions,
        "inventory_loop_off_sandbox_executions": loop_off_adapter.sandbox_executions,
        "treatment_policy_feedback_events": treatment_adapter.policy_feedback_events,
        "control_policy_feedback_events": control_adapter.policy_feedback_events,
        "inventory_loop_off_policy_feedback_events": (
            loop_off_adapter.policy_feedback_events
        ),
        "tasks": task_records,
    }


def _schema_preflight() -> dict[str, object]:
    """Authenticate only public V5 frames; never grow or invoke a learner."""

    tasks = continual_toy_swe_train_stream_v5()
    if len(tasks) != TASK_COUNT or MAX_ATTEMPTS != 3:
        raise RuntimeError("V5 task count or uniform max-attempt policy changed")
    frame_roots = []
    for task in tasks:
        public = task.public
        public.authenticate()
        if _PRIVATE_FIELDS & set(public.record()):
            raise RuntimeError("private evaluator metadata entered public projection")
        frame = coding.ObservationFrame.seal(
            _frame_blocks(
                starter_source=public.starter_source,
                current_source=public.starter_source,
                signature=public.signature.to_record(),
                public_prompt=public.prompt,
                task_commitment_sha256=public.task_commitment_sha256,
                relative_path=public.relative_path,
            ),
            (_WRITE_AFFORDANCE,),
        )
        signature_blocks = [
            block for block in frame.blocks if block.kind == "typed-signature"
        ]
        if len(signature_blocks) != 1:
            raise RuntimeError("V5 frame does not contain exactly one typed signature")
        signature = json.loads(signature_blocks[0].text)
        if (
            _json(signature) != signature_blocks[0].text
            or signature != public.signature.to_record()
        ):
            raise RuntimeError("V5 typed signature is not canonical or task-bound")
        if any(block.kind in _PRIVATE_FIELDS for block in frame.blocks):
            raise RuntimeError("private evaluator block entered learner cognition")
        frame_roots.append(frame.sha256)
    return {
        "schema": SCHEMA,
        "task_count": len(tasks),
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "train_stream_custody_sha256": TRAIN_STREAM_CUSTODY_SHA256,
        "frame_roots_sha256": _digest("v5-preflight-frame-roots", frame_roots),
        "typed_signature_blocks_per_frame": 1,
        "output_path": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
        "artifact_created": False,
        "model_bootstrap_calls": 0,
        "adapter_instantiations": 0,
        "private_labels_used_for_frame_construction": 0,
        "learner_invocations": 0,
    }


def _write_exclusive(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_terminal(path: Path, record: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-schema",
        action="store_true",
        help="authenticate only the frozen public schema and inert frame seals",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_schema:
        _assert_frozen_dependencies()
        print(json.dumps(_schema_preflight(), indent=2, sort_keys=True))
        return 0

    runtime, reservation = _prepare()
    _write_exclusive(OUTPUT_PATH, reservation)
    try:
        result = run_session(runtime, reservation["initial_model"])
        if (
            result["corpus_custody"] != reservation["corpus_custody"]
            or result["execution_custody"] != reservation["execution_custody"]
        ):
            raise RuntimeError("final V5 custody differs from exact reservation")
        result["reservation_custody_matches_final"] = True
    except BaseException as error:
        failed = dict(reservation)
        failed["status"] = "FAILED"
        failed["error_type"] = type(error).__name__
        failed["error"] = str(error)
        _write_terminal(OUTPUT_PATH, failed)
        raise
    completed_reservation = dict(reservation)
    completed_reservation["status"] = "COMPLETED"
    result["reservation"] = completed_reservation
    _write_terminal(OUTPUT_PATH, result)
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
