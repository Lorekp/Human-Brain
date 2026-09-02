"""Prepared V3 continual coding session with cue-routed episodic recall.

Treatment and no-update control start from independent routed models built only
from the frozen public prior. The fresh V3 target stream is reserved before its
first action and is never exposed to successful target sources or private
research labels during model decisions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.continual_toy_swe_tasks_v3 import (
    TASK_COUNT,
    TRAIN_STREAM_SHA256,
    ContinualToySweTrainTaskV3,
    continual_toy_swe_train_stream_v3,
)
from experiments.r4_integrated.public_coding_prior_v1 import (
    PUBLIC_CODING_PRIOR_V1_SHA256,
    SOURCE_CORPUS_SHA256,
    SOURCE_FREEZE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    public_coding_prior_v1,
)
from humanbrain import persistent_coding as coding
from humanbrain.episodic_pointer import (
    CueRoutedPointerCellModel,
    EpisodicRoutingTrace,
    RoutedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import (
    PointerTrainingExample,
    SymbolInventory,
    identifier_inventory,
)
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.sandbox import (
    ExecutionResult,
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)

SCHEMA = "continual-coding-session-v3"
RESERVATION_SCHEMA = "continual-coding-session-reservation-v3"
WRITE_TOOL = "write_source"
MAX_ATTEMPTS = 2
PUBLIC_PRIOR_ROUTES = 8
GENERIC_STOP_ROUTES = 1
ROUTE_COUNT = PUBLIC_PRIOR_ROUTES + GENERIC_STOP_ROUTES

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
WORKER_PATH = REPO_ROOT / "src/humanbrain/simple_swe/_worker.py"
OUTPUT_PATH = REPO_ROOT / "benchmarks/r4_integrated_continual_coding_session_v3.json"

_MODEL_CONFIG = {
    "max_order": 256,
    "max_phrases": 256,
    "min_phrase_frequency": 2,
}
_STOP_CUE = "insufficient evidence"
_GENERIC_STOP_STARTER = (
    "def pending_solution():\n"
    '    """No repository evidence is available."""\n'
    "    raise NotImplementedError\n"
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
    worker_sha256 = sha256(WORKER_PATH.read_bytes()).hexdigest()
    return _SandboxRuntime(
        protocol=protocol,
        executable=executable,
        executable_sha256=identity["executable_sha256"],
        worker_source=load_worker_snapshot(worker_sha256),
        worker_sha256=worker_sha256,
    )


class ToySweExecutionAdapter:
    """Execute the exact learner-authored source through frozen Simple-SWE."""

    def __init__(
        self,
        tasks: tuple[ContinualToySweTrainTaskV3, ...],
        runtime: _SandboxRuntime,
    ) -> None:
        self._tasks = {task.task_sha256: task for task in tasks}
        self._runtime = runtime
        self._task_by_frame: dict[str, ContinualToySweTrainTaskV3] = {}
        self._source_by_task = {
            task.task_sha256: task.public.starter_source for task in tasks
        }

    def bind(
        self,
        frame: coding.ObservationFrame,
        task: ContinualToySweTrainTaskV3,
    ) -> None:
        task.authenticate()
        if task.task_sha256 not in self._tasks:
            raise ValueError("frame task is outside this V3 TRAIN stream")
        self._task_by_frame[frame.sha256] = task

    def source(self, task: ContinualToySweTrainTaskV3) -> str:
        return self._source_by_task[task.task_sha256]

    def execute(self, action: coding.ToolAction) -> coding.ExecutionReceipt:
        task = self._task_by_frame.get(action.frame_sha256)
        if task is None:
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
        self._source_by_task[task.task_sha256] = source
        result = self._execute_source(task, source)
        return coding.ExecutionReceipt.seal(
            action,
            accepted=True,
            succeeded=result.passed,
            observations=(
                coding.ObservationBlock.seal("sandbox-outcome", result.outcome),
                coding.ObservationBlock.seal("sandbox-detail", result.detail),
            ),
        )

    def _execute_source(
        self,
        task: ContinualToySweTrainTaskV3,
        source: str,
    ) -> ExecutionResult:
        # The evaluator opens committed private cases only after an action exists.
        generated = task.private.generated_task
        return execute_completion(
            source,
            generated.public,
            generated.private,
            self._runtime.protocol.execution,
            python_executable=self._runtime.executable,
            python_executable_sha256=self._runtime.executable_sha256,
            worker_source=self._runtime.worker_source,
            worker_sha256=self._runtime.worker_sha256,
        )


class _NoUpdateModel:
    """Independent routed inference with learning and failure memory disabled."""

    def __init__(self, model: CueRoutedPointerCellModel) -> None:
        self._model = model

    @property
    def training_digest(self) -> str:
        return self._model.training_digest

    @property
    def last_routing_trace(self) -> EpisodicRoutingTrace | None:
        return self._model.last_routing_trace

    @property
    def route_count(self) -> int:
        return self._model.route_count

    @property
    def grown_branches(self) -> int:
        return self._model.grown_branches

    @property
    def failure_events(self) -> int:
        return self._model.failure_events

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        return self._model.raw_complete(
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
    ) -> None:
        del prompt, inventory, cue_text

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> None:
        del prompt, output, inventory, cue_text, strength


RoutedModel = CueRoutedPointerCellModel | _NoUpdateModel

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
    starter_source: str,
    current_source: str,
    public_prompt: str,
    previous: coding.ExecutionReceipt | None = None,
) -> tuple[coding.ObservationBlock, ...]:
    blocks = [
        coding.ObservationBlock.seal("starter-source", starter_source),
        coding.ObservationBlock.seal("current-source", current_source),
    ]
    if previous is not None:
        blocks.extend(previous.observations)
    blocks.append(coding.ObservationBlock.seal("public-task", public_prompt))
    return tuple(blocks)


def _frame(
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV3,
    previous: coding.ExecutionReceipt | None = None,
) -> coding.ObservationFrame:
    frame = coding.ObservationFrame.seal(
        _frame_blocks(
            task.public.starter_source,
            adapter.source(task),
            task.public.prompt,
            previous,
        ),
        (_WRITE_AFFORDANCE,),
    )
    adapter.bind(frame, task)
    return frame


def _training_frame(starter_source: str, public_prompt: str) -> coding.ObservationFrame:
    return coding.ObservationFrame.seal(
        _frame_blocks(starter_source, starter_source, public_prompt),
        (_WRITE_AFFORDANCE,),
    )


def _bootstrap_examples() -> tuple[RoutedPointerTrainingExample, ...]:
    prior = public_coding_prior_v1()
    if len(prior) != PUBLIC_PRIOR_ROUTES:
        raise ValueError("frozen public coding prior must contain exactly eight cases")

    stop_frame = _training_frame(_GENERIC_STOP_STARTER, _STOP_CUE)
    examples = [
        RoutedPointerTrainingExample(
            pointer=PointerTrainingExample(
                coding._prompt(stop_frame),
                "<stop>uncertain</stop>",
                identifier_inventory(_GENERIC_STOP_STARTER),
            ),
            cue_text=_STOP_CUE,
        )
    ]
    for case in prior:
        frame = _training_frame(case.starter_source, case.public_prompt)
        payload = {
            "arguments": {"source": case.successful_source},
            "name": WRITE_TOOL,
        }
        examples.append(
            RoutedPointerTrainingExample(
                pointer=PointerTrainingExample(
                    coding._prompt(frame),
                    f"<tool>{coding._json(payload)}</tool>",
                    identifier_inventory(case.starter_source),
                ),
                cue_text=case.public_prompt,
            )
        )
    if len(examples) != ROUTE_COUNT:
        raise ValueError("V3 bootstrap must contain exactly nine unique routes")
    return tuple(examples)


def _bootstrap_corpus_sha256() -> str:
    return _digest(
        "continual-coding-bootstrap-corpus-v3",
        {
            "model_config": _MODEL_CONFIG,
            "routes": [
                {
                    "cue_text": example.cue_text,
                    "inventory_sha256": example.pointer.inventory.sha256,
                    "inventory_symbols": example.pointer.inventory.symbols,
                    "output": example.pointer.output,
                    "prompt": example.pointer.prompt,
                }
                for example in _bootstrap_examples()
            ],
        },
    )


def _bootstrap_model() -> CueRoutedPointerCellModel:
    return CueRoutedPointerCellModel.grow(
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


def _routing_trace_record(trace: EpisodicRoutingTrace) -> dict[str, object]:
    return {
        "cue_sha256": trace.cue_sha256,
        "cue_count": trace.cue_count,
        "route_scores": list(trace.route_scores),
        "selected_route": trace.selected_route,
        "selected_stop": trace.selected_stop,
        "refractory_route": trace.refractory_route,
        "spike_time_ms": trace.spike_time_ms,
        "simulated_steps": trace.simulated_steps,
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


def _episode_record(
    result: coding.EpisodeResult,
    routing_trace: EpisodicRoutingTrace,
) -> dict[str, object]:
    receipt = result.receipt
    if isinstance(result.decision, coding.StopDecision):
        status = f"stop:{result.decision.reason}"
    elif receipt is None or not receipt.accepted:
        status = "tool-rejected"
    elif receipt.succeeded:
        status = "passed"
    else:
        outcome = next(
            (
                block.text
                for block in receipt.observations
                if block.kind == "sandbox-outcome"
            ),
            "failed",
        )
        status = f"sandbox:{outcome}"
    return {
        "status": status,
        "routing_trace": _routing_trace_record(routing_trace),
        "completion_trace": _completion_trace_record(result.completion_trace),
        "decision": _decision_record(result.decision),
        "receipt": _receipt_record(receipt),
        "prior_model_digest": result.prior_model_digest,
        "model_digest": result.model_digest,
        "learning_applied": result.learning_applied,
        "episode_sha256": result.sha256,
    }


def _choice_record(attempt: dict[str, object]) -> dict[str, object]:
    routing = attempt["routing_trace"]
    completion = attempt["completion_trace"]
    decision = attempt["decision"]
    if not isinstance(routing, dict) or not isinstance(completion, dict):
        raise TypeError("attempt trace records are malformed")
    if not isinstance(decision, dict):
        raise TypeError("attempt decision record is malformed")
    decision_choice = {
        key: decision[key]
        for key in ("kind", "reason", "tool_name", "arguments_json")
        if key in decision
    }
    return {
        "selected_route": routing["selected_route"],
        "selected_stop": routing["selected_stop"],
        "raw_pointer_output": completion["raw_pointer_output"],
        "materialized_output": completion["materialized_output"],
        "decision": decision_choice,
    }


def _second_attempt_changed(attempts: list[dict[str, object]]) -> bool:
    return len(attempts) == MAX_ATTEMPTS and _choice_record(
        attempts[0]
    ) != _choice_record(attempts[1])


def _run_task(
    learner: coding.PersistentCodingLearner,
    model: RoutedModel,
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV3,
) -> tuple[list[dict[str, object]], bool]:
    """Apply the same two-attempt, terminal-STOP policy to either arm."""

    frame = _frame(adapter, task)
    attempts: list[dict[str, object]] = []
    for attempt in range(MAX_ATTEMPTS):
        result = learner.episode(frame)
        routing_trace = model.last_routing_trace
        if routing_trace is None:
            raise RuntimeError("routed completion did not expose its trace")
        record = _episode_record(result, routing_trace)
        record["attempt"] = attempt + 1
        attempts.append(record)
        if record["status"] == "passed":
            return attempts, True
        if isinstance(result.decision, coding.StopDecision):
            break
        if result.receipt is not None:
            frame = _frame(adapter, task, result.receipt)
    return attempts, False


def _routing_summary(
    task_records: list[dict[str, object]],
    attempts_key: str,
    changed_key: str,
) -> dict[str, object]:
    attempts = [
        attempt
        for task_record in task_records
        for attempt in task_record[attempts_key]
    ]
    traces = [attempt["routing_trace"] for attempt in attempts]
    selected_routes = sorted({int(trace["selected_route"]) for trace in traces})
    return {
        "route_diversity": len(selected_routes),
        "selected_routes": selected_routes,
        "selected_stop_attempts": sum(bool(trace["selected_stop"]) for trace in traces),
        "refractory_attempts": sum(
            trace["refractory_route"] is not None for trace in traces
        ),
        "changed_second_attempts": sum(
            bool(task_record[changed_key]) for task_record in task_records
        ),
    }


def _execution_custody(runtime: _SandboxRuntime) -> dict[str, str]:
    source_paths = {
        "session_source_sha256": Path(__file__),
        "task_stream_source_sha256": (
            REPO_ROOT / "experiments/r4_integrated/continual_toy_swe_tasks_v3.py"
        ),
        "public_prior_source_sha256": (
            REPO_ROOT / "experiments/r4_integrated/public_coding_prior_v1.py"
        ),
        "persistent_coding_source_sha256": (
            REPO_ROOT / "src/humanbrain/persistent_coding.py"
        ),
        "episodic_pointer_source_sha256": (
            REPO_ROOT / "src/humanbrain/episodic_pointer.py"
        ),
        "pointer_binding_source_sha256": (
            REPO_ROOT / "src/humanbrain/pointer_binding.py"
        ),
        "language_source_sha256": REPO_ROOT / "src/humanbrain/language.py",
        "cortex_source_sha256": REPO_ROOT / "src/humanbrain/cortex.py",
    }
    custody = {
        name: sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    custody.update(
        {
            "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "worker_sha256": runtime.worker_sha256,
            "python_executable_sha256": runtime.executable_sha256,
        }
    )
    return custody


def _corpus_custody() -> dict[str, object]:
    return {
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "target_task_count": TASK_COUNT,
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "public_prior_routes": PUBLIC_PRIOR_ROUTES,
        "generic_stop_routes": GENERIC_STOP_ROUTES,
        "route_count": ROUTE_COUNT,
        "duplicate_route_repetitions": 0,
        "route_cue_source": "exact_public_prompt_only",
        "bootstrap_corpus_sha256": _bootstrap_corpus_sha256(),
        "model_config": dict(_MODEL_CONFIG),
        "source_corpus_sha256": SOURCE_CORPUS_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "target_tasks_present_at_freeze": False,
    }


def _reservation_record(runtime: _SandboxRuntime) -> dict[str, object]:
    model = _bootstrap_model()
    if model.route_count != ROUTE_COUNT:
        raise RuntimeError("routed prior did not build exactly nine routes")
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RUNNING",
        "initial_model_digest": model.training_digest,
        "corpus_custody": _corpus_custody(),
        "execution_custody": _execution_custody(runtime),
    }


def _prepare(output_path: Path) -> tuple[_SandboxRuntime, dict[str, object]]:
    if output_path.exists():
        raise FileExistsError(f"V3 session artifact already exists: {output_path}")
    runtime = _sandbox_runtime()
    return runtime, _reservation_record(runtime)


def run_session(runtime: _SandboxRuntime) -> dict[str, object]:
    model = _bootstrap_model()
    control_model = _NoUpdateModel(_bootstrap_model())
    initial_model_digest = model.training_digest
    control_initial_digest = control_model.training_digest
    if initial_model_digest != control_initial_digest:
        raise RuntimeError("treatment and no-update control did not share one prior")
    if model.route_count != ROUTE_COUNT or control_model.route_count != ROUTE_COUNT:
        raise RuntimeError("treatment and control route counts changed")

    tasks = continual_toy_swe_train_stream_v3()
    if len(tasks) != TASK_COUNT:
        raise ValueError("V3 continual task count changed")
    adapter = ToySweExecutionAdapter(tasks, runtime)
    learner = coding.PersistentCodingLearner(model, adapter)
    control_adapter = ToySweExecutionAdapter(tasks, runtime)
    control = coding.PersistentCodingLearner(control_model, control_adapter)

    failures: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    control_failures: Counter[str] = Counter()
    control_outcomes: Counter[str] = Counter()
    task_records: list[dict[str, object]] = []
    capability_curve: list[dict[str, object]] = []
    solved_online = 0
    control_solved_online = 0
    second_attempt_recoveries = 0
    control_second_attempt_recoveries = 0
    changed_second_attempts = 0
    control_changed_second_attempts = 0

    for task_index, task in enumerate(tasks):
        task.authenticate()
        attempts, passed = _run_task(learner, model, adapter, task)
        control_attempts, control_passed = _run_task(
            control,
            control_model,
            control_adapter,
            task,
        )
        for record in attempts:
            status = str(record["status"])
            outcomes[status] += 1
            if status != "passed":
                failures[status] += 1
        for record in control_attempts:
            status = str(record["status"])
            control_outcomes[status] += 1
            if status != "passed":
                control_failures[status] += 1

        changed = _second_attempt_changed(attempts)
        control_changed = _second_attempt_changed(control_attempts)
        changed_second_attempts += int(changed)
        control_changed_second_attempts += int(control_changed)
        solved_online += int(passed)
        control_solved_online += int(control_passed)
        second_attempt_recoveries += int(
            passed
            and len(attempts) == MAX_ATTEMPTS
            and attempts[-1]["status"] == "passed"
        )
        control_second_attempt_recoveries += int(
            control_passed
            and len(control_attempts) == MAX_ATTEMPTS
            and control_attempts[-1]["status"] == "passed"
        )
        task_records.append(
            {
                "task_index": task_index,
                "task_sha256": task.task_sha256,
                "public_observation_sha256": task.public.observation_sha256,
                "attempts": attempts,
                "solved_online": passed,
                "second_attempt_changed": changed,
                "control_attempts": control_attempts,
                "control_solved_online": control_passed,
                "control_second_attempt_changed": control_changed,
            }
        )
        capability_curve.append(
            {
                "task_index": task_index,
                "tasks_seen": task_index + 1,
                "solved_online": solved_online,
                "control_solved_online": control_solved_online,
                "changed_second_attempts": changed_second_attempts,
                "control_changed_second_attempts": control_changed_second_attempts,
                "cumulative_rate": solved_online / (task_index + 1),
                "control_cumulative_rate": control_solved_online
                / (task_index + 1),
            }
        )

    # Research-only labels open only after both arms have sealed every action.
    for task, task_record, curve_point in zip(
        tasks,
        task_records,
        capability_curve,
        strict=True,
    ):
        stage = task.private.stage
        capability_group = task.private.capability_group
        task_record["stage"] = stage
        task_record["capability_group"] = capability_group
        curve_point["stage"] = stage
        curve_point["capability_group"] = capability_group

    return {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "claim_boundary": (
            "prepared fresh public-TRAIN routed continual toy pressure only; "
            "no protected evaluation or mesoscopic forcing-gate claim"
        ),
        "corpus_custody": _corpus_custody(),
        "execution_custody": _execution_custody(runtime),
        "task_count": len(tasks),
        "max_attempts": MAX_ATTEMPTS,
        "one_persistent_treatment_model": True,
        "full_replay_between_tasks": False,
        "target_task_teacher_sources_used": 0,
        "exact_learner_action_source_executed": True,
        "protected_dev_or_eval_inputs": 0,
        "initial_model_digest": initial_model_digest,
        "final_model_digest": learner.model_digest,
        "grown_branches": model.grown_branches,
        "routing": {
            **_routing_summary(task_records, "attempts", "second_attempt_changed"),
            "failure_events": model.failure_events,
        },
        "learning_applied_events": sum(
            int(attempt["learning_applied"])
            for task in task_records
            for attempt in task["attempts"]
        ),
        "attempt_outcome_distribution": dict(sorted(outcomes.items())),
        "failure_distribution": dict(sorted(failures.items())),
        "solved_online": solved_online,
        "second_attempt_recoveries": second_attempt_recoveries,
        "changed_second_attempts": changed_second_attempts,
        "capability_curve": capability_curve,
        "frozen_no_update_control": {
            "initial_model_digest": control_initial_digest,
            "final_model_digest": control.model_digest,
            "model_digest_unchanged": control.model_digest == control_initial_digest,
            "routing": {
                **_routing_summary(
                    task_records,
                    "control_attempts",
                    "control_second_attempt_changed",
                ),
                "failure_events": control_model.failure_events,
            },
            "learning_applied_events": sum(
                int(attempt["learning_applied"])
                for task in task_records
                for attempt in task["control_attempts"]
            ),
            "attempt_outcome_distribution": dict(
                sorted(control_outcomes.items())
            ),
            "failure_distribution": dict(sorted(control_failures.items())),
            "solved_online": control_solved_online,
            "second_attempt_recoveries": control_second_attempt_recoveries,
            "changed_second_attempts": control_changed_second_attempts,
        },
        "tasks": task_records,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime, reservation = _prepare(args.output)
    _write_exclusive(args.output, reservation)
    try:
        result = run_session(runtime)
    except BaseException as error:
        failed = dict(reservation)
        failed["status"] = "FAILED"
        failed["error_type"] = type(error).__name__
        failed["error"] = str(error)
        _write_terminal(args.output, failed)
        raise
    completed_reservation = dict(reservation)
    completed_reservation["status"] = "COMPLETED"
    result["reservation"] = completed_reservation
    _write_terminal(args.output, result)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
