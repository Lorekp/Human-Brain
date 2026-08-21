from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from humanbrain.simple_swe.adapter import CompletionAudit, SynthesizedCompletion
from humanbrain.simple_swe.artifacts import (
    ArtifactEvent,
    ArtifactIntegrityError,
    ArtifactLedger,
)
from humanbrain.simple_swe.protocol import FAILURE_TAXONOMY, ProtocolSpec
from humanbrain.simple_swe.reveal import (
    EvaluationBundle,
    evaluation_reveal_commitment,
)
from humanbrain.simple_swe.sandbox import ExecutionResult, execute_completion
from humanbrain.simple_swe.tasks import GeneratedTask, PublicTask, canonical_sha256


class CompletionAdapter(Protocol):
    checkpoint_sha256: str

    def synthesize(self, task: PublicTask) -> SynthesizedCompletion: ...


CompletionExecutor = Callable[..., ExecutionResult]


@dataclass(frozen=True)
class CellEvaluationSummary:
    task_count: int
    passed: int
    pass_at_1: float
    outcome_counts: Mapping[str, int]
    tier_counts: Mapping[str, Mapping[str, int | float]]
    completion_seal_sha256: str
    summary_event_sha256: str


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_runtime(executable: Path) -> dict[str, object]:
    probe = (
        "import json,sys;"
        "print(json.dumps({'implementation':sys.implementation.name,"
        "'version':list(sys.version_info[:3])},sort_keys=True,separators=(',',':')))"
    )
    try:
        result = subprocess.run(
            [str(executable), "-I", "-S", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        value = json.loads(result.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        raise ArtifactIntegrityError("sandbox Python runtime probe failed") from error
    if not isinstance(value, dict):
        raise ArtifactIntegrityError("sandbox Python runtime probe is invalid")
    return value


def _runtime_identity(executable: Path, language: object) -> dict[str, str]:
    if not executable.is_file():
        raise ArtifactIntegrityError("sandbox Python executable is not a file")
    if not isinstance(language, str) or not language.startswith("python-"):
        raise ArtifactIntegrityError("protocol language runtime is invalid")
    try:
        expected = tuple(int(part) for part in language.removeprefix("python-").split("."))
    except ValueError as error:
        raise ArtifactIntegrityError("protocol Python version is invalid") from error
    if len(expected) != 2:
        raise ArtifactIntegrityError("protocol must freeze Python major/minor")
    probe = _probe_runtime(executable)
    implementation = probe.get("implementation")
    version = probe.get("version")
    if (
        not isinstance(implementation, str)
        or not isinstance(version, list)
        or len(version) != 3
        or any(isinstance(part, bool) or not isinstance(part, int) for part in version)
    ):
        raise ArtifactIntegrityError("sandbox Python runtime identity is invalid")
    if tuple(version[:2]) != expected:
        raise ArtifactIntegrityError(
            f"sandbox Python must match protocol {language}, got {version[0]}.{version[1]}"
        )
    return {
        "implementation": implementation,
        "python_version": ".".join(str(part) for part in version),
        "executable_sha256": _sha256_file(executable),
    }


def _validate_run_prefix(
    bundle: EvaluationBundle,
    adapter: CompletionAdapter,
    ledger: ArtifactLedger,
) -> tuple[ArtifactEvent, ArtifactEvent]:
    if ledger.protocol_sha256 != bundle.protocol_sha256:
        raise ArtifactIntegrityError("runner ledger protocol differs from eval bundle")
    events = ledger.verify()
    if len(events) < 2:
        raise ArtifactIntegrityError("evaluation reveal is incomplete")
    run_start, revealed = events[:2]
    if run_start.kind != "run-start":
        raise ArtifactIntegrityError("first artifact is not run-start")
    required_run_start = {
        "freeze_manifest_sha256": bundle.freeze_manifest_sha256,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "checkpoint_state_sha256": bundle.checkpoint_state_sha256,
        "source_head": bundle.source_head,
    }
    if any(run_start.body.get(key) != value for key, value in required_run_start.items()):
        raise ArtifactIntegrityError("run-start differs from eval bundle freeze")
    for field_name in ("completion_adapter_sha256", "compiler_sha256"):
        value = run_start.body.get(field_name)
        if not isinstance(value, str) or len(value) != 64:
            raise ArtifactIntegrityError(f"run-start lacks {field_name}")
    if adapter.checkpoint_sha256 != bundle.checkpoint_sha256:
        raise ArtifactIntegrityError("completion adapter checkpoint differs from eval bundle")
    if revealed.kind != "evaluation-revealed" or revealed.body != evaluation_reveal_commitment(
        bundle
    ):
        raise ArtifactIntegrityError("evaluation reveal commitments differ")
    return run_start, revealed


def _completion_body(
    index: int,
    task: GeneratedTask,
    completion: SynthesizedCompletion,
) -> dict[str, object]:
    return {
        "task_index": index,
        "task_id": task.public.task_id,
        "task_sha256": task.public.task_sha256,
        "test_sha256": task.private.test_sha256,
        "source": completion.source,
        "source_sha256": sha256(completion.source.encode("utf-8")).hexdigest(),
        "emitted_text": completion.emitted_text,
        "emitted_text_sha256": sha256(completion.emitted_text.encode("utf-8")).hexdigest(),
        "audit": asdict(completion.audit),
    }


def _validate_completion_body(
    body: dict[str, object],
    *,
    index: int,
    task: GeneratedTask,
    bundle: EvaluationBundle,
    protocol: ProtocolSpec,
    run_start: ArtifactEvent,
) -> None:
    expected_keys = {
        "task_index",
        "task_id",
        "task_sha256",
        "test_sha256",
        "source",
        "source_sha256",
        "emitted_text",
        "emitted_text_sha256",
        "audit",
    }
    if set(body) != expected_keys:
        raise ArtifactIntegrityError("cell completion artifact fields differ")
    commitments = {
        "task_index": index,
        "task_id": task.public.task_id,
        "task_sha256": task.public.task_sha256,
        "test_sha256": task.private.test_sha256,
    }
    if any(body.get(key) != value for key, value in commitments.items()):
        raise ArtifactIntegrityError("cell completion task commitment differs")
    source = body.get("source")
    emitted_text = body.get("emitted_text")
    audit = body.get("audit")
    if not isinstance(source, str) or not isinstance(emitted_text, str):
        raise ArtifactIntegrityError("cell completion source is not text")
    if not isinstance(audit, dict):
        raise ArtifactIntegrityError("cell completion audit is not an object")
    audit_fields = {field.name for field in fields(CompletionAudit)}
    if set(audit) != audit_fields:
        raise ArtifactIntegrityError("cell completion audit fields differ")
    source_sha256 = sha256(source.encode("utf-8")).hexdigest()
    emitted_text_sha256 = sha256(emitted_text.encode("utf-8")).hexdigest()
    if body["source_sha256"] != source_sha256 or audit["source_sha256"] != source_sha256:
        raise ArtifactIntegrityError("cell completion source hash differs")
    if (
        body["emitted_text_sha256"] != emitted_text_sha256
        or audit["emitted_text_sha256"] != emitted_text_sha256
    ):
        raise ArtifactIntegrityError("cell completion emission hash differs")
    token_ids = audit.get("emitted_token_ids")
    if (
        not isinstance(token_ids, (list, tuple))
        or len(token_ids) != protocol.max_new_tokens
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        )
        or audit.get("emitted_token_count") != protocol.max_new_tokens
        or audit.get("emitted_token_sha256") != canonical_sha256(list(token_ids))
    ):
        raise ArtifactIntegrityError("cell completion token audit differs")
    expected_audit = {
        "adapter_sha256": run_start.body["completion_adapter_sha256"],
        "compiler_sha256": run_start.body["compiler_sha256"],
        "compiler_algorithm_source": "emitted-tokens-only",
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "learned_checkpoint_state_before": bundle.checkpoint_state_sha256,
        "learned_checkpoint_state_after": bundle.checkpoint_state_sha256,
        "prompt_sha256": sha256(task.public.prompt.encode("utf-8")).hexdigest(),
        "direct_synthesis_no_tools": True,
    }
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise ArtifactIntegrityError("cell completion freeze audit differs")
    for field_name in ("adapter_id", "compiler_id", "compiler_mode"):
        if not isinstance(audit.get(field_name), str) or not audit[field_name]:
            raise ArtifactIntegrityError(f"cell completion lacks {field_name}")


def _completion_seal_body(
    bundle: EvaluationBundle,
    completion_events: list[ArtifactEvent],
    runtime_identity: dict[str, str],
) -> dict[str, object]:
    event_hashes = [event.file_sha256 for event in completion_events]
    return {
        "task_count": len(bundle.tasks),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "learned_checkpoint_state_sha256": bundle.checkpoint_state_sha256,
        "completion_event_sha256s": event_hashes,
        "completion_set_sha256": canonical_sha256(event_hashes),
        "sandbox_runtime": runtime_identity,
        "hidden_tests_executed_before_seal": False,
        "private_cases_not_passed_to_adapter": True,
    }


def _result_body(
    index: int,
    task: GeneratedTask,
    completion_event: ArtifactEvent,
    result: ExecutionResult,
) -> dict[str, object]:
    return {
        "task_index": index,
        "task_id": task.public.task_id,
        "task_sha256": task.public.task_sha256,
        "test_sha256": task.private.test_sha256,
        "completion_event_sha256": completion_event.file_sha256,
        "outcome": result.outcome,
        "passed": result.passed,
        "wall_seconds": result.wall_seconds,
        "detail": result.detail,
        "sandboxed": result.sandboxed,
    }


def _validate_result_body(
    body: dict[str, object],
    *,
    index: int,
    task: GeneratedTask,
    completion_event: ArtifactEvent,
) -> None:
    expected_keys = {
        "task_index",
        "task_id",
        "task_sha256",
        "test_sha256",
        "completion_event_sha256",
        "outcome",
        "passed",
        "wall_seconds",
        "detail",
        "sandboxed",
    }
    if set(body) != expected_keys:
        raise ArtifactIntegrityError("task result artifact fields differ")
    expected = {
        "task_index": index,
        "task_id": task.public.task_id,
        "task_sha256": task.public.task_sha256,
        "test_sha256": task.private.test_sha256,
        "completion_event_sha256": completion_event.file_sha256,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise ArtifactIntegrityError("task result commitment differs")
    outcome = body.get("outcome")
    wall_seconds = body.get("wall_seconds")
    if outcome not in FAILURE_TAXONOMY or body.get("passed") is not (outcome == "passed"):
        raise ArtifactIntegrityError("task result taxonomy differs")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds < 0.0
    ):
        raise ArtifactIntegrityError("task result wall time is invalid")
    if not isinstance(body.get("detail"), str) or not isinstance(body.get("sandboxed"), bool):
        raise ArtifactIntegrityError("task result diagnostics are invalid")


def _summary_body(
    tasks: tuple[GeneratedTask, ...],
    results: list[ArtifactEvent],
    seal: ArtifactEvent,
) -> dict[str, object]:
    outcome_counts = {outcome: 0 for outcome in FAILURE_TAXONOMY}
    tier_counts: dict[str, dict[str, int | float]] = {}
    passed = 0
    for task, event in zip(tasks, results, strict=True):
        outcome = str(event.body["outcome"])
        outcome_counts[outcome] += 1
        if outcome == "passed":
            passed += 1
        tier = str(task.public.tier)
        bucket = tier_counts.setdefault(tier, {"tasks": 0, "passed": 0})
        bucket["tasks"] = int(bucket["tasks"]) + 1
        if outcome == "passed":
            bucket["passed"] = int(bucket["passed"]) + 1
    for bucket in tier_counts.values():
        bucket["pass_at_1"] = int(bucket["passed"]) / int(bucket["tasks"])
    result_hashes = [event.file_sha256 for event in results]
    return {
        "sample_policy": "pass@1-one-cell-completion-per-task",
        "task_count": len(tasks),
        "passed": passed,
        "pass_at_1": passed / len(tasks),
        "outcome_counts": outcome_counts,
        "tier_counts": tier_counts,
        "completion_seal_sha256": seal.file_sha256,
        "result_event_sha256s": result_hashes,
        "result_set_sha256": canonical_sha256(result_hashes),
        "no_test_time_learning": True,
        "no_task_cherry_picking": True,
    }


def _summary_from_event(event: ArtifactEvent) -> CellEvaluationSummary:
    body = event.body
    return CellEvaluationSummary(
        task_count=int(body["task_count"]),
        passed=int(body["passed"]),
        pass_at_1=float(body["pass_at_1"]),
        outcome_counts=dict(body["outcome_counts"]),
        tier_counts={str(tier): dict(counts) for tier, counts in body["tier_counts"].items()},
        completion_seal_sha256=str(body["completion_seal_sha256"]),
        summary_event_sha256=event.file_sha256,
    )


def _validate_baseline_events(events: list[ArtifactEvent], *, cell_summary_sha256: str) -> None:
    names: set[str] = set()
    for event in events:
        body = event.body
        if event.kind != "baseline-control" or set(body) != {
            "system_name",
            "task_count",
            "pass_at_1",
            "report_sha256",
            "cell_summary_sha256",
            "reporting_control_only",
            "hidden_cell_inference_path",
        }:
            raise ArtifactIntegrityError("only reporting baselines may follow cell summary")
        if (
            body.get("cell_summary_sha256") != cell_summary_sha256
            or body.get("reporting_control_only") is not True
            or body.get("hidden_cell_inference_path") is not False
        ):
            raise ArtifactIntegrityError("baseline control changed the cell score boundary")
        name = body.get("system_name")
        if not isinstance(name, str) or not name or name in names:
            raise ArtifactIntegrityError("baseline control identity is invalid")
        task_count = body.get("task_count")
        score = body.get("pass_at_1")
        report_sha256 = body.get("report_sha256")
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count < 1
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            or not isinstance(report_sha256, str)
            or len(report_sha256) != 64
            or any(char not in "0123456789abcdef" for char in report_sha256)
        ):
            raise ArtifactIntegrityError("baseline control aggregate is invalid")
        names.add(name)


def run_cell_evaluation(
    *,
    bundle: EvaluationBundle,
    adapter: CompletionAdapter,
    ledger: ArtifactLedger,
    protocol: ProtocolSpec,
    executor: CompletionExecutor = execute_completion,
    python_executable: str | Path | None = None,
) -> CellEvaluationSummary:
    """Seal every direct cell completion before opening any hidden tests."""
    if bundle.protocol_sha256 != protocol.sha256:
        raise ArtifactIntegrityError("runner protocol differs from eval bundle")
    run_start, _revealed = _validate_run_prefix(bundle, adapter, ledger)
    executable = Path(python_executable or sys.executable).resolve()
    runtime_identity = _runtime_identity(executable, protocol.raw.get("language"))
    events = list(ledger.verify())
    position = 2
    completion_events: list[ArtifactEvent] = []
    while position < len(events) and events[position].kind == "cell-completion":
        index = len(completion_events)
        if index >= len(bundle.tasks):
            raise ArtifactIntegrityError("artifact ledger has extra cell completions")
        event = events[position]
        _validate_completion_body(
            event.body,
            index=index,
            task=bundle.tasks[index],
            bundle=bundle,
            protocol=protocol,
            run_start=run_start,
        )
        completion_events.append(event)
        position += 1
    if len(completion_events) < len(bundle.tasks):
        if position != len(events):
            raise ArtifactIntegrityError("artifact ledger interrupted completion order")
        for index in range(len(completion_events), len(bundle.tasks)):
            task = bundle.tasks[index]
            completion = adapter.synthesize(task.public)
            body = _completion_body(index, task, completion)
            _validate_completion_body(
                body,
                index=index,
                task=task,
                bundle=bundle,
                protocol=protocol,
                run_start=run_start,
            )
            completion_events.append(ledger.append("cell-completion", body))
        events = list(ledger.verify())
        position = 2 + len(completion_events)
    seal_body = _completion_seal_body(bundle, completion_events, runtime_identity)
    if position == len(events):
        seal = ledger.append("cell-completions-sealed", seal_body)
        events = list(ledger.verify())
    else:
        seal = events[position]
        if seal.kind != "cell-completions-sealed" or seal.body != seal_body:
            raise ArtifactIntegrityError("cell completion seal differs")
    position += 1
    result_events: list[ArtifactEvent] = []
    while position < len(events) and events[position].kind == "task-result":
        index = len(result_events)
        if index >= len(bundle.tasks):
            raise ArtifactIntegrityError("artifact ledger has extra task results")
        event = events[position]
        _validate_result_body(
            event.body,
            index=index,
            task=bundle.tasks[index],
            completion_event=completion_events[index],
        )
        result_events.append(event)
        position += 1
    if len(result_events) < len(bundle.tasks):
        if position != len(events):
            raise ArtifactIntegrityError("artifact ledger interrupted result order")
        for index in range(len(result_events), len(bundle.tasks)):
            task = bundle.tasks[index]
            source = str(completion_events[index].body["source"])
            result = executor(
                source,
                task.public,
                task.private,
                protocol.execution,
                python_executable=executable,
            )
            body = _result_body(index, task, completion_events[index], result)
            _validate_result_body(
                body,
                index=index,
                task=task,
                completion_event=completion_events[index],
            )
            result_events.append(ledger.append("task-result", body))
        events = list(ledger.verify())
        position = 3 + len(completion_events) + len(result_events)
    summary_body = _summary_body(bundle.tasks, result_events, seal)
    if position == len(events):
        summary_event = ledger.append("cell-summary", summary_body)
        events = list(ledger.verify())
    else:
        summary_event = events[position]
        if summary_event.kind != "cell-summary" or summary_event.body != summary_body:
            raise ArtifactIntegrityError("cell summary differs from task results")
    position += 1
    _validate_baseline_events(events[position:], cell_summary_sha256=summary_event.file_sha256)
    return _summary_from_event(summary_event)


def record_baseline_control(
    *,
    ledger: ArtifactLedger,
    system_name: str,
    task_count: int,
    pass_at_1: float,
    report_sha256: str,
) -> ArtifactEvent:
    """Append an aggregate reporting control after the immutable cell score."""
    events = list(ledger.verify())
    summary_events = [event for event in events if event.kind == "cell-summary"]
    if len(summary_events) != 1:
        raise ArtifactIntegrityError("baseline controls require one finalized cell summary")
    summary = summary_events[0]
    summary_index = events.index(summary)
    _validate_baseline_events(events[summary_index + 1 :], cell_summary_sha256=summary.file_sha256)
    if not isinstance(system_name, str) or not system_name.strip():
        raise ValueError("baseline system_name cannot be empty")
    if isinstance(task_count, bool) or task_count < 1:
        raise ValueError("baseline task_count must be positive")
    if task_count != summary.body.get("task_count"):
        raise ValueError("baseline task_count must match the cell cohort")
    if not math.isfinite(pass_at_1) or not 0.0 <= pass_at_1 <= 1.0:
        raise ValueError("baseline pass_at_1 must be in [0, 1]")
    if len(report_sha256) != 64 or any(char not in "0123456789abcdef" for char in report_sha256):
        raise ValueError("baseline report_sha256 must be lowercase SHA-256")
    body = {
        "system_name": system_name.strip(),
        "task_count": task_count,
        "pass_at_1": pass_at_1,
        "report_sha256": report_sha256,
        "cell_summary_sha256": summary.file_sha256,
        "reporting_control_only": True,
        "hidden_cell_inference_path": False,
    }
    for event in events[summary_index + 1 :]:
        if event.body.get("system_name") == body["system_name"]:
            if event.body != body:
                raise ArtifactIntegrityError("baseline control was already recorded differently")
            return event
    return ledger.append("baseline-control", body)


__all__ = [
    "CellEvaluationSummary",
    "record_baseline_control",
    "run_cell_evaluation",
]
