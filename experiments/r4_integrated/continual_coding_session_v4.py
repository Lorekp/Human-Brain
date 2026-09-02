"""Prepared V4 factorized continual coding session; importing does not spend it.

One persistent treatment learner and one independently grown frozen control face
the fresh public-TRAIN V4 stream.  Both begin from the same eight public coding
successes.  Only authenticated execution consequences can update treatment.
No private field enters learner observations.  Terminal policy and tests open
only after a sealed action to produce its environment consequence; stage,
group, budget, and feedback labels open only for post-hoc scoring.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from experiments.r4_integrated.continual_toy_swe_tasks_v4 import (
    TASK_COUNT,
    TRAIN_STREAM_CUSTODY_SHA256,
    TRAIN_STREAM_SHA256,
    ContinualToySweTrainTaskV4,
    continual_toy_swe_train_stream_v4,
    learner_visible_observation_v4,
    train_stream_custody_v4,
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

SCHEMA = "continual-coding-session-v4"
RESERVATION_SCHEMA = "continual-coding-session-reservation-v4"
WRITE_TOOL = "write_source"
MAX_ATTEMPTS = 3
PUBLIC_PRIOR_CASES = 8

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
WORKER_PATH = REPO_ROOT / "src/humanbrain/simple_swe/_worker.py"
OUTPUT_PATH = REPO_ROOT / "benchmarks/r4_integrated_continual_coding_session_v4.json"
FACTORIZED_GATE_PATH = REPO_ROOT / "benchmarks/r4_factorized_output_binding_gate_v1.json"

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
    "benchmarks/simple_swe_protocol_v1.json": (
        "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
    ),
    "experiments/r4_integrated/continual_toy_swe_tasks_v4.py": (
        "520ea0ad97f5f211295ccd48b7900f4eaa475bcb73bd8d7a5816cba5165e552f"
    ),
    "experiments/r4_integrated/public_coding_prior_v1.py": (
        "95b0bbae62cc1fc08dd94e0f25e32116c2bbe671c7d5ceb10b50a4617ff0ac5b"
    ),
    "src/humanbrain/cortex.py": (
        "9cf37fc36978c3389a86a2856401f6c528d1a95f491dab0eb1b8662edcd055db"
    ),
    "src/humanbrain/factorized_pointer.py": (
        "595b4ab5b8bb486e843f52fbd73e4958d5e3d6d0efad6cb690491757a94d1017"
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
        "private_environment_sha256",
        "private_test_sha256",
        "stage",
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


def _assert_frozen_dependencies() -> None:
    for relative_path, expected in _FROZEN_DEPENDENCIES.items():
        actual = _sha256_file(REPO_ROOT / relative_path)
        if actual != expected:
            raise RuntimeError(f"frozen V4 dependency changed: {relative_path}")


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
    """Execute only exact learner-emitted source after an action is sealed."""

    def __init__(
        self,
        tasks: tuple[ContinualToySweTrainTaskV4, ...],
        runtime: _SandboxRuntime,
    ) -> None:
        self._tasks = {task.task_sha256: task for task in tasks}
        self._runtime = runtime
        self._task_by_frame: dict[str, ContinualToySweTrainTaskV4] = {}
        self._source_by_task = {
            task.task_sha256: task.public.starter_source for task in tasks
        }
        self._audit_by_action: dict[str, dict[str, object]] = {}
        self.sandbox_executions = 0
        self.policy_feedback_events = 0

    def bind(
        self,
        frame: coding.ObservationFrame,
        task: ContinualToySweTrainTaskV4,
    ) -> None:
        task.authenticate()
        if task.task_sha256 not in self._tasks:
            raise ValueError("frame task is outside the V4 TRAIN stream")
        self._task_by_frame[frame.sha256] = task

    def source(self, task: ContinualToySweTrainTaskV4) -> str:
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
                observations=(coding.ObservationBlock.seal("tool-error", "unbound frame"),),
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

        # The terminal-action policy is evaluator-private until this exact action
        # exists.  A STOP task has no hidden oracle, so a program must fail rather
        # than receive a vacuous pass from an empty case set.
        if task.private.expected_action == "stop":
            self.policy_feedback_events += 1
            self._audit_by_action[action.sha256] = {
                "kind": "environment-feedback",
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


class _AuditedFactorizedModel:
    """Delegate cognition while retaining the signed plasticity receipt."""

    def __init__(self, model: FactorizedPointerCellModel) -> None:
        self._model = model
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
        self.last_update_request = {
            "polarity": "failure",
            "evidence_sha256": sha256(feedback_text.encode()).hexdigest(),
        }
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
        self.last_update_request = {
            "polarity": "success",
            "evidence_sha256": sha256(output.encode()).hexdigest(),
        }
        self.last_update = self._model.learn(
            prompt,
            output,
            inventory,
            cue_text=cue_text,
            strength=strength,
        )
        return self.last_update


class _NoUpdateFactorizedModel:
    """Independent physical inference with both plasticity paths disabled."""

    def __init__(self, model: FactorizedPointerCellModel) -> None:
        self._model = model
        self.last_update: None = None
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
    ) -> None:
        del prompt, inventory, cue_text
        self.last_update_request = {
            "polarity": "failure",
            "evidence_sha256": sha256(feedback_text.encode()).hexdigest(),
        }

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> None:
        del prompt, inventory, cue_text, strength
        self.last_update_request = {
            "polarity": "success",
            "evidence_sha256": sha256(output.encode()).hexdigest(),
        }


SessionModel = _AuditedFactorizedModel | _NoUpdateFactorizedModel

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
    task: ContinualToySweTrainTaskV4,
    previous: coding.ExecutionReceipt | None = None,
) -> coding.ObservationFrame:
    public = learner_visible_observation_v4(task)
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
        "continual-coding-bootstrap-corpus-v4",
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
        "materialized_output": trace.materialized_output,
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
    full = {
        "query_sha256": trace.query_sha256,
        "model_digest": trace.model_digest,
        "ablation": trace.ablation,
        "cue_words": trace.cue_words,
        "fragment_steps": full_steps,
        "binding_races": bindings,
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
        "emitted_fragment_cells": list(trace.emitted_fragment_cells),
        "emitted_token_ids": list(trace.emitted_token_ids),
        "selected_stop": trace.selected_stop,
        "terminal_stop": trace.terminal_stop,
        "output": trace.output,
        "trace_sha256": _digest("factorized-completion-trace-v1", full),
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


def _episode_status(
    result: coding.EpisodeResult,
    explicit_stop: bool,
) -> str:
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


def _episode_record(
    result: coding.EpisodeResult,
    model: SessionModel,
    adapter: ToySweExecutionAdapter,
) -> dict[str, object]:
    if not result.is_authentic():
        raise RuntimeError("persistent episode result failed authentication")
    neuronal_trace = model.last_trace
    if neuronal_trace is None:
        raise RuntimeError("factorized completion did not expose its trace")
    explicit_stop = (
        isinstance(result.decision, coding.StopDecision)
        and result.completion_trace.materialization_error is None
        and _explicit_stop_text(result.completion_trace.materialized_output)
    )
    execution = (
        adapter.audit(result.decision)
        if isinstance(result.decision, coding.ToolAction)
        else None
    )
    choice = {
        "raw_pointer_output": result.completion_trace.raw_pointer_output,
        "materialized_output": result.completion_trace.materialized_output,
        "materialization_error": result.completion_trace.materialization_error,
        "decision": _decision_record(result.decision),
    }
    return {
        "status": _episode_status(result, explicit_stop),
        "explicit_neuronal_stop": explicit_stop,
        "materialization_failure": (
            result.completion_trace.materialization_error is not None
        ),
        "factorized_trace": _factorized_trace_record(neuronal_trace),
        "completion_trace": _completion_trace_record(result.completion_trace),
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
        "choice_sha256": _digest("continual-coding-choice-v4", choice),
        "episode_sha256": result.sha256,
    }


def _run_task(
    learner: coding.PersistentCodingLearner,
    model: SessionModel,
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV4,
) -> list[dict[str, object]]:
    """Apply one public three-attempt policy without consulting authored budget."""

    frame = _frame(adapter, task)
    attempts: list[dict[str, object]] = []
    for attempt_index in range(MAX_ATTEMPTS):
        result = learner.episode(frame)
        record = _episode_record(result, model, adapter)
        record["attempt"] = attempt_index + 1
        attempts.append(record)
        if record["status"] == "passed" or record["explicit_neuronal_stop"]:
            break
        if result.receipt is not None:
            frame = _frame(adapter, task, result.receipt)
    return attempts


def _extract_tool_source(output: str) -> str | None:
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
        local: f"_bound_local_{index}"
        for index, local in enumerate(collector.locals)
    }
    normalized = _AlphaRename(renames).visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _prior_replay_case(
    model: SessionModel,
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


def _unknown_cue_probe(model: SessionModel) -> dict[str, object]:
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


def _retention_arm(model: SessionModel) -> dict[str, object]:
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
        "byte_exact": sum(int(case["byte_exact"]) for case in cases),
        "alpha_equivalent": sum(int(case["alpha_equivalent"]) for case in cases),
        "valid_write_source": sum(int(case["valid_write_source"]) for case in cases),
        "terminal_stop": sum(int(case["terminal_stop"]) for case in cases),
        "cases": cases,
        "unknown_cue": uncertainty,
    }


def _retention_checkpoint(
    label: str,
    treatment: SessionModel,
    control: SessionModel,
) -> dict[str, object]:
    return {
        "label": label,
        "treatment": _retention_arm(treatment),
        "control": _retention_arm(control),
    }


def _changed_retry(attempts: list[dict[str, object]]) -> bool:
    choices = [str(attempt["choice_sha256"]) for attempt in attempts]
    return any(left != right for left, right in pairwise(choices))


def _arm_score(
    expected_action: str,
    attempts: list[dict[str, object]],
) -> dict[str, object]:
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
        # These frozen tasks ask whether uncertainty is recognized before a
        # speculative action.  Failure-driven abstention remains informative,
        # but it is not an immediate-STOP success.
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
    return {
        "correct": correct,
        "winning_attempt": winning,
        "observed_terminal_attempt": observed_terminal,
        "recovery": winning is not None and winning > 1,
        "late_uncertainty_recovery": late_uncertainty is not None,
        "late_uncertainty_attempt": late_uncertainty,
        "retry_changed": _changed_retry(attempts),
        "failure": failure,
    }


def _posthoc_score(
    tasks: tuple[ContinualToySweTrainTaskV4, ...],
    task_records: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    stage_scores: dict[str, Counter[str]] = {}
    treatment_failures: Counter[str] = Counter()
    control_failures: Counter[str] = Counter()
    for task, record in zip(tasks, task_records, strict=True):
        private = task.private
        treatment = _arm_score(private.expected_action, record["attempts"])
        control = _arm_score(private.expected_action, record["control_attempts"])
        record.update(
            {
                "stage": private.stage,
                "capability_group": private.capability_group,
                "expected_action": private.expected_action,
                "authored_attempt_budget": private.attempt_budget,
                "feedback_pressure": private.feedback_pressure,
                "private_environment_sha256": private.environment_sha256,
                "score": treatment,
                "control_score": control,
            }
        )
        counts = stage_scores.setdefault(private.stage, Counter())
        counts["tasks"] += 1
        counts["treatment_correct"] += int(treatment["correct"])
        counts["control_correct"] += int(control["correct"])
        if treatment["failure"] is not None:
            treatment_failures[str(treatment["failure"])] += 1
        if control["failure"] is not None:
            control_failures[str(control["failure"])] += 1
    stage_record = {
        stage: {
            **dict(counts),
            "treatment_rate": counts["treatment_correct"] / counts["tasks"],
            "control_rate": counts["control_correct"] / counts["tasks"],
        }
        for stage, counts in sorted(stage_scores.items())
    }
    failures = {
        "treatment": dict(sorted(treatment_failures.items())),
        "control": dict(sorted(control_failures.items())),
    }
    return stage_record, failures


def _capability_curve(task_records: list[dict[str, object]]) -> list[dict[str, object]]:
    treatment_correct = 0
    control_correct = 0
    treatment_recoveries = 0
    control_recoveries = 0
    treatment_late_uncertainty = 0
    control_late_uncertainty = 0
    curve = []
    for index, record in enumerate(task_records):
        score = record["score"]
        control_score = record["control_score"]
        treatment_correct += int(score["correct"])
        control_correct += int(control_score["correct"])
        treatment_recoveries += int(score["recovery"])
        control_recoveries += int(control_score["recovery"])
        treatment_late_uncertainty += int(score["late_uncertainty_recovery"])
        control_late_uncertainty += int(
            control_score["late_uncertainty_recovery"]
        )
        seen = index + 1
        curve.append(
            {
                "task_index": index,
                "tasks_seen": seen,
                "stage": record["stage"],
                "capability_group": record["capability_group"],
                "treatment_correct": treatment_correct,
                "control_correct": control_correct,
                "treatment_rate": treatment_correct / seen,
                "control_rate": control_correct / seen,
                "treatment_recoveries": treatment_recoveries,
                "control_recoveries": control_recoveries,
                "treatment_late_uncertainty_recoveries": (
                    treatment_late_uncertainty
                ),
                "control_late_uncertainty_recoveries": control_late_uncertainty,
            }
        )
    return curve


def _attempt_distribution(
    task_records: list[dict[str, object]],
    key: str,
) -> dict[str, int]:
    counter = Counter(
        str(attempt["status"])
        for record in task_records
        for attempt in record[key]
    )
    return dict(sorted(counter.items()))


def _learning_summary(
    task_records: list[dict[str, object]],
    key: str,
) -> dict[str, object]:
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


def _model_identity(model: SessionModel | FactorizedPointerCellModel) -> dict[str, object]:
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
            REPO_ROOT / "experiments/r4_integrated/continual_toy_swe_tasks_v4.py"
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
        "train_stream_custody": train_stream_custody_v4(),
        "target_task_count": TASK_COUNT,
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "public_prior_cases": PUBLIC_PRIOR_CASES,
        "bootstrap_corpus_sha256": _bootstrap_corpus_sha256(),
        "model_config": dict(_MODEL_CONFIG),
        "source_corpus_sha256": SOURCE_CORPUS_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "target_tasks_present_at_freeze": False,
        "synthetic_stop_supervision_examples": 0,
        "uncertainty_probe_is_untrained": True,
    }


def _reservation_record(runtime: _SandboxRuntime) -> dict[str, object]:
    model = _bootstrap_model()
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RUNNING",
        "artifact_path": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
        "initial_model": _model_identity(model),
        "corpus_custody": _corpus_custody(),
        "execution_custody": _execution_custody(runtime),
    }


def _prepare() -> tuple[_SandboxRuntime, dict[str, object]]:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"V4 session artifact already exists: {OUTPUT_PATH}")
    _assert_frozen_dependencies()
    _schema_preflight()
    runtime = _sandbox_runtime()
    return runtime, _reservation_record(runtime)


def run_session(
    runtime: _SandboxRuntime,
    expected_initial_model: dict[str, object],
) -> dict[str, object]:
    treatment_model = _AuditedFactorizedModel(_bootstrap_model())
    control_model = _NoUpdateFactorizedModel(_bootstrap_model())
    treatment_initial = _model_identity(treatment_model)
    control_initial = _model_identity(control_model)
    if treatment_initial != control_initial or treatment_initial != expected_initial_model:
        raise RuntimeError("treatment and control did not start from identical growth")

    tasks = continual_toy_swe_train_stream_v4()
    treatment_adapter = ToySweExecutionAdapter(tasks, runtime)
    control_adapter = ToySweExecutionAdapter(tasks, runtime)
    treatment = coding.PersistentCodingLearner(treatment_model, treatment_adapter)
    control = coding.PersistentCodingLearner(control_model, control_adapter)

    retention_curve = [
        _retention_checkpoint("before-v4", treatment_model, control_model)
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
        task_records.append(
            {
                "task_index": task_index,
                "task_sha256": task.task_sha256,
                "public_observation_sha256": task.public.observation_sha256,
                "attempts": attempts,
                "control_attempts": control_attempts,
            }
        )
        retention_curve.append(
            _retention_checkpoint(
                f"after-v4-task-{task_index + 1}",
                treatment_model,
                control_model,
            )
        )

    # These evaluator-only labels open only after both arms sealed all V4 actions.
    stage_scores, failure_distribution = _posthoc_score(tasks, task_records)
    capability_curve = _capability_curve(task_records)
    treatment_final = _model_identity(treatment_model)
    control_final = _model_identity(control_model)
    return {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "claim_boundary": (
            "fresh developmental public-TRAIN integrated continual coding evidence "
            "only; not protected evaluation, family-disjoint transfer, Simple-SWE "
            "M0, or a completed mesoscopic forcing-gate claim"
        ),
        "corpus_custody": _corpus_custody(),
        "execution_custody": _execution_custody(runtime),
        "task_count": len(tasks),
        "max_attempts_for_every_task": MAX_ATTEMPTS,
        "private_authored_budgets_used_for_cognition": False,
        "one_persistent_treatment_learner": True,
        "full_replay_between_tasks": False,
        "target_task_teacher_sources_used": 0,
        "protected_dev_or_eval_inputs": 0,
        "only_learner_emitted_programs_enter_sandbox": True,
        "typed_signature_blocks_per_frame": 1,
        "treatment_initial_model": treatment_initial,
        "treatment_final_model": treatment_final,
        "control_initial_model": control_initial,
        "control_final_model": control_final,
        "control_model_digest_unchanged": (
            control_initial["training_digest"] == control_final["training_digest"]
        ),
        "treatment_learning": _learning_summary(task_records, "attempts"),
        "control_learning": _learning_summary(task_records, "control_attempts"),
        "treatment_attempt_outcomes": _attempt_distribution(task_records, "attempts"),
        "control_attempt_outcomes": _attempt_distribution(
            task_records,
            "control_attempts",
        ),
        "failure_distribution": failure_distribution,
        "stage_scores": stage_scores,
        "capability_curve": capability_curve,
        "retention_curve": retention_curve,
        "treatment_sandbox_executions": treatment_adapter.sandbox_executions,
        "control_sandbox_executions": control_adapter.sandbox_executions,
        "treatment_policy_feedback_events": treatment_adapter.policy_feedback_events,
        "control_policy_feedback_events": control_adapter.policy_feedback_events,
        "tasks": task_records,
    }


def _schema_preflight() -> dict[str, object]:
    """Authenticate only public V4 schema; never grow or invoke a learner."""

    tasks = continual_toy_swe_train_stream_v4()
    if len(tasks) != TASK_COUNT or MAX_ATTEMPTS != 3:
        raise RuntimeError("V4 task count or uniform attempt policy changed")
    frame_roots = []
    for task in tasks:
        task.authenticate()
        public = learner_visible_observation_v4(task)
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
            raise RuntimeError("V4 frame does not contain exactly one typed signature")
        signature = json.loads(signature_blocks[0].text)
        if (
            _json(signature) != signature_blocks[0].text
            or signature.get("entry_point") != public.signature.entry_point
        ):
            raise RuntimeError("V4 typed signature is not canonical or task-bound")
        if any(block.kind in _PRIVATE_FIELDS for block in frame.blocks):
            raise RuntimeError("private evaluator block entered learner cognition")
        frame_roots.append(frame.sha256)
    return {
        "schema": SCHEMA,
        "task_count": len(tasks),
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "train_stream_custody_sha256": TRAIN_STREAM_CUSTODY_SHA256,
        "frame_roots_sha256": _digest("v4-preflight-frame-roots", frame_roots),
        "output_path": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
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


def main() -> int:
    runtime, reservation = _prepare()
    _write_exclusive(OUTPUT_PATH, reservation)
    try:
        result = run_session(runtime, reservation["initial_model"])
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
