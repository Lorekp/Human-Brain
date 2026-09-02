"""Fresh TRAIN-only continual coding session over one persistent cell learner.

The session is intentionally small.  It establishes the integrated
inspect -> act/STOP -> execute -> observe -> learn loop and measures a frozen
no-update control.  Both arms begin from the same target-disjoint public coding
prior; target-task teachers and repaired sources are never supplied.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.continual_toy_swe_tasks_v1 import (
    TRAIN_STREAM_SHA256,
    ContinualToySweTrainTaskV1,
    continual_toy_swe_train_stream_v1,
)
from experiments.r4_integrated.public_coding_prior_v1 import (
    PUBLIC_CODING_PRIOR_V1_SHA256,
    SOURCE_CORPUS_SHA256,
    SOURCE_FREEZE_COMMIT,
    SOURCE_MANIFEST_SHA256,
    public_coding_prior_v1,
)
from humanbrain import CellLanguageModel, ModelConfig
from humanbrain import persistent_coding as coding
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.sandbox import (
    ExecutionResult,
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)

SCHEMA = "continual-coding-session-v1"
WRITE_TOOL = "write_source"
MAX_ATTEMPTS = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
WORKER_PATH = REPO_ROOT / "src/humanbrain/simple_swe/_worker.py"
OUTPUT_PATH = REPO_ROOT / "benchmarks/r4_integrated_continual_coding_session_v1.json"


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
    """Real Simple-SWE execution behind the generic repository seam."""

    def __init__(
        self,
        tasks: tuple[ContinualToySweTrainTaskV1, ...],
        runtime: _SandboxRuntime,
    ) -> None:
        self._tasks = {task.task_sha256: task for task in tasks}
        self._runtime = runtime
        self._task_by_frame: dict[str, ContinualToySweTrainTaskV1] = {}
        self._source_by_task = {
            task.task_sha256: task.public.starter_source for task in tasks
        }

    def bind(
        self,
        frame: coding.ObservationFrame,
        task: ContinualToySweTrainTaskV1,
    ) -> None:
        task.authenticate()
        if task.task_sha256 not in self._tasks:
            raise ValueError("frame task is outside this TRAIN stream")
        self._task_by_frame[frame.sha256] = task

    def source(self, task: ContinualToySweTrainTaskV1) -> str:
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
        task: ContinualToySweTrainTaskV1,
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
    """Causal control: identical inference with plasticity disabled."""

    def __init__(self, model: CellLanguageModel) -> None:
        self._model = model

    @property
    def training_digest(self) -> str:
        return self._model.training_digest

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> str:
        return self._model.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )

    def learn(self, text: str, *, strength: float = 1.0) -> None:
        del text, strength


_WRITE_AFFORDANCE = coding.ToolAffordance.seal(
    WRITE_TOOL,
    {
        "additionalProperties": False,
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
        "type": "object",
    },
)


def _frame(
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV1,
    previous: coding.ExecutionReceipt | None = None,
) -> coding.ObservationFrame:
    blocks = [
        coding.ObservationBlock.seal(
            "current-source",
            adapter.source(task),
        )
    ]
    if previous is not None:
        blocks.extend(previous.observations)
    # Keep the task language nearest the decision marker; the neuronal model
    # has finite context and should spend it on evidence, not custody hashes.
    blocks.append(
        coding.ObservationBlock.seal(
            "public-task",
            task.public.prompt,
        )
    )
    frame = coding.ObservationFrame.seal(blocks, (_WRITE_AFFORDANCE,))
    adapter.bind(frame, task)
    return frame


def _bootstrap_model() -> CellLanguageModel:
    stop_frame = coding.ObservationFrame.seal(
        (coding.ObservationBlock.seal("public-task", "insufficient evidence"),),
        (_WRITE_AFFORDANCE,),
    )
    trajectories = [f"{coding._prompt(stop_frame)}\n<stop>uncertain</stop>"]
    for example in public_coding_prior_v1():
        frame = coding.ObservationFrame.seal(
            (
                coding.ObservationBlock.seal(
                    "current-source", example.starter_source
                ),
                coding.ObservationBlock.seal(
                    "public-task", example.public_prompt
                ),
            ),
            (_WRITE_AFFORDANCE,),
        )
        payload = {
            "arguments": {"source": example.successful_source},
            "name": WRITE_TOOL,
        }
        trajectories.append(
            f"{coding._prompt(frame)}\n<tool>{coding._json(payload)}</tool>"
        )
    corpus = tuple(
        repeated
        for trajectory in trajectories
        for repeated in (trajectory, trajectory)
    )
    return CellLanguageModel.grow(
        corpus,
        ModelConfig(max_order=256, max_phrases=64, min_phrase_frequency=2),
    )


def _episode_record(result: coding.EpisodeResult) -> dict[str, object]:
    receipt = result.receipt
    if isinstance(result.decision, coding.StopDecision):
        status = f"stop:{result.decision.reason}"
        tool_name = None
    else:
        tool_name = result.decision.tool_name
        if receipt is None or not receipt.accepted:
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
        "tool_name": tool_name,
        "frame_sha256": result.decision.frame_sha256,
        "decision_sha256": result.decision.sha256,
        "receipt_sha256": None if receipt is None else receipt.sha256,
        "learning_applied": result.learning_applied,
        "prior_model_digest": result.prior_model_digest,
        "model_digest": result.model_digest,
    }


def _run_task(
    learner: coding.PersistentCodingLearner,
    adapter: ToySweExecutionAdapter,
    task: ContinualToySweTrainTaskV1,
) -> tuple[list[dict[str, object]], bool]:
    """Apply the same public two-attempt/terminal-STOP policy to either arm."""

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
    return {
        "session_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "persistent_coding_source_sha256": sha256(
            (REPO_ROOT / "src/humanbrain/persistent_coding.py").read_bytes()
        ).hexdigest(),
        "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "worker_sha256": runtime.worker_sha256,
        "python_executable_sha256": runtime.executable_sha256,
    }


def _reservation_record(runtime: _SandboxRuntime) -> dict[str, object]:
    return {
        "schema": "continual-coding-session-reservation-v1",
        "status": "RUNNING",
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "initial_model_digest": _bootstrap_model().training_digest,
        "execution_custody": _execution_custody(runtime),
    }


def run_session(runtime: _SandboxRuntime) -> dict[str, object]:
    # Freeze both independent neural states from the disjoint public prior before
    # any target observation enters either model.
    model = _bootstrap_model()
    control_model = _NoUpdateModel(_bootstrap_model())
    initial_model_digest = model.training_digest
    control_initial_digest = control_model.training_digest
    if initial_model_digest != control_initial_digest:
        raise RuntimeError("learning and no-update arms did not share one prior")

    tasks = continual_toy_swe_train_stream_v1()
    adapter = ToySweExecutionAdapter(tasks, runtime)
    learner = coding.PersistentCodingLearner(model, adapter)
    control_adapter = ToySweExecutionAdapter(tasks, runtime)
    control = coding.PersistentCodingLearner(control_model, control_adapter)

    failures: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    control_failures: Counter[str] = Counter()
    control_outcomes: Counter[str] = Counter()
    task_records: list[dict[str, object]] = []
    solved_online = 0
    control_solved_online = 0
    second_attempt_recoveries = 0
    control_second_attempt_recoveries = 0
    capability_curve: list[dict[str, object]] = []

    for task_index, task in enumerate(tasks):
        task.authenticate()
        attempts, passed = _run_task(learner, adapter, task)
        control_attempts, control_passed = _run_task(
            control, control_adapter, task
        )
        for record in attempts:
            outcomes[record["status"]] += 1
            if record["status"] != "passed":
                failures[record["status"]] += 1
        for record in control_attempts:
            control_outcomes[record["status"]] += 1
            if record["status"] != "passed":
                control_failures[record["status"]] += 1

        solved_online += int(passed)
        control_solved_online += int(control_passed)
        second_attempt_recoveries += int(
            passed
            and len(attempts) == 2
            and attempts[1]["status"] == "passed"
        )
        control_second_attempt_recoveries += int(
            control_passed
            and len(control_attempts) == 2
            and control_attempts[1]["status"] == "passed"
        )

        task_records.append(
            {
                "task_index": task_index,
                "task_sha256": task.task_sha256,
                "attempts": attempts,
                "solved_online": passed,
                "control_attempts": control_attempts,
                "control_solved_online": control_passed,
            }
        )
        capability_curve.append(
            {
                "task_index": task_index,
                "solved_online": solved_online,
                "control_solved_online": control_solved_online,
                "tasks_seen": task_index + 1,
                "cumulative_rate": solved_online / (task_index + 1),
                "control_cumulative_rate": control_solved_online
                / (task_index + 1),
            }
        )

    # Add private research labels only after every action and consequence in
    # both arms is sealed. They cannot influence any session decision.
    for task, task_record, curve_point in zip(
        tasks, task_records, capability_curve, strict=True
    ):
        stage = task.private.stage
        task_record["stage"] = stage
        curve_point["stage"] = stage

    return {
        "schema": SCHEMA,
        "claim_boundary": (
            "fresh public-TRAIN toy pure-function preflight only; not the "
            "mesoscopic integrated forcing gate; no protected evidence"
        ),
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "public_coding_prior_cases": len(public_coding_prior_v1()),
        "public_coding_prior_source": {
            "corpus_sha256": SOURCE_CORPUS_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "freeze_commit": SOURCE_FREEZE_COMMIT,
            "target_tasks_present_at_freeze": False,
        },
        "tokenizer_fit_inputs": "public_coding_prior_only",
        "task_count": len(tasks),
        "max_attempts": MAX_ATTEMPTS,
        "one_persistent_model": True,
        "full_replay_between_tasks": False,
        "target_task_teacher_sources_used": 0,
        "exact_learner_action_source_executed": True,
        "protected_dev_or_eval_inputs": 0,
        "initial_model_digest": initial_model_digest,
        "final_model_digest": learner.model_digest,
        "grown_branches": model.grown_branches,
        "attempt_outcome_distribution": dict(sorted(outcomes.items())),
        "failure_distribution": dict(sorted(failures.items())),
        "solved_online": solved_online,
        "second_attempt_recoveries": second_attempt_recoveries,
        "capability_curve": capability_curve,
        "execution_custody": _execution_custody(runtime),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime = _sandbox_runtime()
    reservation = _reservation_record(runtime)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(reservation, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    try:
        result = run_session(runtime)
    except BaseException as error:
        failed = dict(reservation)
        failed["status"] = "FAILED"
        failed["error_type"] = type(error).__name__
        failed["error"] = str(error)
        args.output.write_text(
            json.dumps(failed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    completed_reservation = dict(reservation)
    completed_reservation["status"] = "COMPLETED"
    result["reservation"] = completed_reservation
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
