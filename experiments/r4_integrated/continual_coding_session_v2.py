"""Fresh V2 continual coding session over one persistent neuronal learner.

Both treatment and no-update control begin from independent copies of the same
target-disjoint public prior. Target tasks provide observations and real
Simple-SWE consequences, never successful sources or private teachers.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.continual_toy_swe_tasks_v2 import (
    TASK_COUNT,
    TRAIN_STREAM_SHA256,
    ContinualToySweTrainTaskV2,
    continual_toy_swe_train_stream_v2,
)
from experiments.r4_integrated.public_coding_prior_v1 import (
    PUBLIC_CODING_PRIOR_V1_SHA256,
    SOURCE_CORPUS_SHA256,
    SOURCE_FREEZE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    public_coding_prior_v1,
)
from humanbrain import persistent_coding as coding
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import (
    PointerBoundCellModel,
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

SCHEMA = "continual-coding-session-v2"
RESERVATION_SCHEMA = "continual-coding-session-reservation-v2"
WRITE_TOOL = "write_source"
MAX_ATTEMPTS = 2
PUBLIC_PRIOR_CASES = 8
TRAINING_REPETITIONS = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
WORKER_PATH = REPO_ROOT / "src/humanbrain/simple_swe/_worker.py"
OUTPUT_PATH = REPO_ROOT / "benchmarks/r4_integrated_continual_coding_session_v2.json"

_MODEL_CONFIG = {
    "max_order": 256,
    "max_phrases": 256,
    "min_phrase_frequency": 2,
}
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
    """Execute the exact learner-authored source in the real Simple-SWE seam."""

    def __init__(
        self,
        tasks: tuple[ContinualToySweTrainTaskV2, ...],
        runtime: _SandboxRuntime,
    ) -> None:
        self._tasks = {task.task_sha256: task for task in tasks}
        self._runtime = runtime
        self._task_by_frame: dict[str, ContinualToySweTrainTaskV2] = {}
        self._source_by_task = {
            task.task_sha256: task.public.starter_source for task in tasks
        }

    def bind(
        self,
        frame: coding.ObservationFrame,
        task: ContinualToySweTrainTaskV2,
    ) -> None:
        task.authenticate()
        if task.task_sha256 not in self._tasks:
            raise ValueError("frame task is outside this V2 TRAIN stream")
        self._task_by_frame[frame.sha256] = task

    def source(self, task: ContinualToySweTrainTaskV2) -> str:
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
                        "tool-error", "write_source requires one source string"
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
        task: ContinualToySweTrainTaskV2,
        source: str,
    ) -> ExecutionResult:
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
    """Independent pointer model with identical inference and no plasticity."""

    def __init__(self, model: PointerBoundCellModel) -> None:
        self._model = model

    @property
    def training_digest(self) -> str:
        return self._model.training_digest

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        return self._model.raw_complete(
            prompt,
            inventory,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        strength: float = 1.0,
    ) -> None:
        del prompt, output, inventory, strength


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
    # The public decision evidence remains nearest the completion boundary.
    blocks.append(coding.ObservationBlock.seal("public-task", public_prompt))
    return tuple(blocks)


def _frame(
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV2,
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


def _bootstrap_examples() -> tuple[PointerTrainingExample, ...]:
    prior = public_coding_prior_v1()
    if len(prior) != PUBLIC_PRIOR_CASES:
        raise ValueError("frozen public coding prior must contain exactly eight cases")

    stop_frame = _training_frame(_GENERIC_STOP_STARTER, "insufficient evidence")
    examples = [
        PointerTrainingExample(
            coding._prompt(stop_frame),
            "<stop>uncertain</stop>",
            identifier_inventory(_GENERIC_STOP_STARTER),
        )
    ]
    for case in prior:
        frame = _training_frame(case.starter_source, case.public_prompt)
        payload = {
            "arguments": {"source": case.successful_source},
            "name": WRITE_TOOL,
        }
        examples.append(
            PointerTrainingExample(
                coding._prompt(frame),
                f"<tool>{coding._json(payload)}</tool>",
                identifier_inventory(case.starter_source),
            )
        )
    return tuple(examples)


def _bootstrap_corpus_sha256() -> str:
    examples = _bootstrap_examples()
    return _digest(
        "continual-coding-bootstrap-corpus-v2",
        {
            "examples": [
                {
                    "inventory_sha256": example.inventory.sha256,
                    "inventory_symbols": example.inventory.symbols,
                    "output": example.output,
                    "prompt": example.prompt,
                }
                for example in examples
                for _ in range(TRAINING_REPETITIONS)
            ],
            "model_config": _MODEL_CONFIG,
        },
    )


def _bootstrap_model() -> PointerBoundCellModel:
    examples = _bootstrap_examples()
    repeated = tuple(
        example
        for example in examples
        for _ in range(TRAINING_REPETITIONS)
    )
    return PointerBoundCellModel.grow(
        repeated,
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


def _episode_record(result: coding.EpisodeResult) -> dict[str, object]:
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
        "completion_trace": _completion_trace_record(result.completion_trace),
        "decision": _decision_record(result.decision),
        "receipt": _receipt_record(receipt),
        "prior_model_digest": result.prior_model_digest,
        "model_digest": result.model_digest,
        "learning_applied": result.learning_applied,
        "episode_sha256": result.sha256,
    }


def _run_task(
    learner: coding.PersistentCodingLearner,
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV2,
) -> tuple[list[dict[str, object]], bool]:
    """Apply the same two-attempt, terminal-STOP policy to either arm."""

    frame = _frame(adapter, task)
    attempts: list[dict[str, object]] = []
    for attempt in range(MAX_ATTEMPTS):
        result = learner.episode(frame)
        record = _episode_record(result)
        record["attempt"] = attempt + 1
        attempts.append(record)
        if record["status"] == "passed":
            return attempts, True
        if isinstance(result.decision, coding.StopDecision):
            break
        if result.receipt is not None:
            frame = _frame(adapter, task, result.receipt)
    return attempts, False


def _execution_custody(runtime: _SandboxRuntime) -> dict[str, str]:
    source_paths = {
        "session_source_sha256": Path(__file__),
        "task_stream_source_sha256": (
            REPO_ROOT / "experiments/r4_integrated/continual_toy_swe_tasks_v2.py"
        ),
        "public_prior_source_sha256": (
            REPO_ROOT / "experiments/r4_integrated/public_coding_prior_v1.py"
        ),
        "persistent_coding_source_sha256": (
            REPO_ROOT / "src/humanbrain/persistent_coding.py"
        ),
        "pointer_binding_source_sha256": (
            REPO_ROOT / "src/humanbrain/pointer_binding.py"
        ),
        "language_source_sha256": REPO_ROOT / "src/humanbrain/language.py",
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
        "public_coding_prior_cases": PUBLIC_PRIOR_CASES,
        "generic_stop_examples": 1,
        "training_repetitions": TRAINING_REPETITIONS,
        "bootstrap_corpus_sha256": _bootstrap_corpus_sha256(),
        "model_config": dict(_MODEL_CONFIG),
        "source_corpus_sha256": SOURCE_CORPUS_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "target_tasks_present_at_freeze": False,
    }


def _reservation_record(runtime: _SandboxRuntime) -> dict[str, object]:
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RUNNING",
        "initial_model_digest": _bootstrap_model().training_digest,
        "corpus_custody": _corpus_custody(),
        "execution_custody": _execution_custody(runtime),
    }


def run_session(runtime: _SandboxRuntime) -> dict[str, object]:
    # Freeze independent treatment/control states before any target observation.
    model = _bootstrap_model()
    control_model = _NoUpdateModel(_bootstrap_model())
    initial_model_digest = model.training_digest
    control_initial_digest = control_model.training_digest
    if initial_model_digest != control_initial_digest:
        raise RuntimeError("treatment and no-update control did not share one prior")

    tasks = continual_toy_swe_train_stream_v2()
    if len(tasks) != TASK_COUNT:
        raise ValueError("V2 continual task count changed")
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

    for task_index, task in enumerate(tasks):
        task.authenticate()
        attempts, passed = _run_task(learner, adapter, task)
        control_attempts, control_passed = _run_task(
            control,
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
                "control_attempts": control_attempts,
                "control_solved_online": control_passed,
            }
        )
        capability_curve.append(
            {
                "task_index": task_index,
                "tasks_seen": task_index + 1,
                "solved_online": solved_online,
                "control_solved_online": control_solved_online,
                "cumulative_rate": solved_online / (task_index + 1),
                "control_cumulative_rate": control_solved_online
                / (task_index + 1),
            }
        )

    # Research labels open only after every treatment/control action is sealed.
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
            "fresh public-TRAIN continual toy coding pressure only; no protected "
            "evaluation and no mesoscopic integrated forcing-gate claim"
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
        "learning_applied_events": sum(
            int(attempt["learning_applied"])
            for task in task_records
            for attempt in task["attempts"]
        ),
        "attempt_outcome_distribution": dict(sorted(outcomes.items())),
        "failure_distribution": dict(sorted(failures.items())),
        "solved_online": solved_online,
        "second_attempt_recoveries": second_attempt_recoveries,
        "capability_curve": capability_curve,
        "frozen_no_update_control": {
            "initial_model_digest": control_initial_digest,
            "final_model_digest": control.model_digest,
            "model_digest_unchanged": control.model_digest == control_initial_digest,
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
    runtime = _sandbox_runtime()
    reservation = _reservation_record(runtime)
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
