from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from humanbrain.simple_swe.adapter import (
    CellCompletionAdapter,
    CompletionAudit,
    CompletionRequest,
    SynthesizedCompletion,
)
from humanbrain.simple_swe.artifacts import (
    ArtifactEvent,
    ArtifactIntegrityError,
    ArtifactLedger,
)
from humanbrain.simple_swe.freeze import (
    FreezeIntegrityError,
    verify_freeze_manifest,
)
from humanbrain.simple_swe.protocol import (
    FAILURE_TAXONOMY,
    ProtocolSpec,
    load_protocol,
)
from humanbrain.simple_swe.reveal import (
    EvaluationBundle,
    canonical_evaluation_paths,
    evaluation_reveal_commitment,
    load_evaluation_bundle,
    run_start_commitment,
    validate_evaluation_bundle,
)
from humanbrain.simple_swe.sandbox import (
    ExecutionResult,
    execute_completion,
    load_worker_snapshot,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, canonical_sha256


class CompletionAdapter(Protocol):
    checkpoint_sha256: str

    def synthesize(self, request: CompletionRequest) -> SynthesizedCompletion: ...


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


def _validate_run_prefix(
    bundle: EvaluationBundle,
    adapter: CompletionAdapter,
    ledger: ArtifactLedger,
    expected_run_start: dict[str, object],
) -> tuple[ArtifactEvent, ArtifactEvent]:
    if ledger.protocol_sha256 != bundle.protocol_sha256:
        raise ArtifactIntegrityError("runner ledger protocol differs from eval bundle")
    events = ledger.verify()
    if len(events) < 4:
        raise ArtifactIntegrityError("evaluation reveal is incomplete")
    run_start, seed_reserved, anchor, revealed = events[:4]
    if run_start.kind != "run-start" or run_start.body != expected_run_start:
        raise ArtifactIntegrityError("run-start differs from frozen production mode")
    if (
        seed_reserved.kind != "evaluation-seed-reserved"
        or seed_reserved.body.get("eval_seed_sha256") != bundle.eval_seed_sha256
        or seed_reserved.body.get("seed_persisted_before_task_generation") is not True
        or seed_reserved.body.get("evaluation_tasks_generated") is not False
    ):
        raise ArtifactIntegrityError("eval seed reservation commitment differs")
    if (
        anchor.kind != "external-anchor-recorded"
        or not isinstance(anchor.body.get("anchor_payload_sha256"), str)
        or anchor.body.get("operator_supplied_receipt") is not True
        or anchor.body.get("harness_verified_external_publication") is not False
    ):
        raise ArtifactIntegrityError("external anchor commitment differs")
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
    run_start: ArtifactEvent,
) -> dict[str, object]:
    event_hashes = [event.file_sha256 for event in completion_events]
    return {
        "task_count": len(bundle.tasks),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "learned_checkpoint_state_sha256": bundle.checkpoint_state_sha256,
        "completion_event_sha256s": event_hashes,
        "completion_set_sha256": canonical_sha256(event_hashes),
        "sandbox_runtime": runtime_identity,
        "canonical_runner_id": run_start.body["canonical_runner_id"],
        "canonical_runner_sha256": run_start.body["canonical_runner_sha256"],
        "sandbox_worker_sha256": run_start.body["sandbox_worker_sha256"],
        "production_dependencies_injected": run_start.body[
            "production_dependencies_injected"
        ],
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


def _run_cell_evaluation_engine(
    *,
    bundle: EvaluationBundle,
    adapter: CompletionAdapter,
    ledger: ArtifactLedger,
    protocol: ProtocolSpec,
    executor: CompletionExecutor,
    python_executable: Path,
    runtime_identity: dict[str, str],
    expected_run_start: dict[str, object],
    prepare_results: Callable[[], bytes],
) -> CellEvaluationSummary:
    """Private dependency-injected engine shared by production and focused tests."""
    if bundle.protocol_sha256 != protocol.sha256:
        raise ArtifactIntegrityError("runner protocol differs from eval bundle")
    run_start, _revealed = _validate_run_prefix(
        bundle, adapter, ledger, expected_run_start
    )
    events = list(ledger.verify())
    position = 4
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
            completion = adapter.synthesize(
                CompletionRequest(
                    prompt=task.public.prompt,
                    entry_point=task.public.entry_point,
                    parameter_names=tuple(task.public.parameter_names),
                )
            )
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
        position = 4 + len(completion_events)
    seal_body = _completion_seal_body(
        bundle, completion_events, runtime_identity, run_start
    )
    if position == len(events):
        seal = ledger.append("cell-completions-sealed", seal_body)
        events = list(ledger.verify())
    else:
        seal = events[position]
        if seal.kind != "cell-completions-sealed" or seal.body != seal_body:
            raise ArtifactIntegrityError("cell completion seal differs")
    position += 1
    worker_source = prepare_results()
    if sha256(worker_source).hexdigest() != run_start.body["sandbox_worker_sha256"]:
        raise ArtifactIntegrityError("prepared sandbox worker differs from run-start")
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
                python_executable=python_executable,
                worker_source=worker_source,
                worker_sha256=str(run_start.body["sandbox_worker_sha256"]),
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
        position = 5 + len(completion_events) + len(result_events)
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


def _run_cell_evaluation_for_testing(
    *,
    bundle: EvaluationBundle,
    adapter: CompletionAdapter,
    ledger: ArtifactLedger,
    protocol: ProtocolSpec,
    executor: CompletionExecutor = execute_completion,
    python_executable: str | Path | None = None,
) -> CellEvaluationSummary:
    """Run fixture dependencies while permanently labeling their ledger noncanonical."""
    events = ledger.verify()
    if not events:
        raise ArtifactIntegrityError("test engine requires a fixture run-start")
    expected_run_start = dict(events[0].body)
    if (
        expected_run_start.get("canonical_runner_id") != "test-injected-engine-v1"
        or expected_run_start.get("production_dependencies_injected") is not True
    ):
        raise ArtifactIntegrityError("test engine cannot write a canonical ledger")
    executable = resolve_python_executable(python_executable)
    worker_sha256 = expected_run_start.get("sandbox_worker_sha256")
    if not isinstance(worker_sha256, str):
        raise ArtifactIntegrityError("fixture run-start lacks worker hash")
    runtime_identity = expected_run_start.get("sandbox_runtime")
    if not isinstance(runtime_identity, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in runtime_identity.items()
    ):
        raise ArtifactIntegrityError("fixture run-start lacks runtime identity")
    return _run_cell_evaluation_engine(
        bundle=bundle,
        adapter=adapter,
        ledger=ledger,
        protocol=protocol,
        executor=executor,
        python_executable=executable,
        runtime_identity=runtime_identity,
        expected_run_start=expected_run_start,
        prepare_results=lambda: load_worker_snapshot(worker_sha256),
    )


def run_cell_evaluation(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
) -> CellEvaluationSummary:
    """Run the sole canonical adapter, bundle, runtime, worker, and executor path."""
    protocol = load_protocol(protocol_path)
    manifest = verify_freeze_manifest(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        checkpoint_path=checkpoint_path,
        source_root=source_root,
    )
    manifest_path = Path(manifest_path).resolve()
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    paths = canonical_evaluation_paths(manifest_path, manifest)
    bundle = load_evaluation_bundle(paths.bundle_path)
    validate_evaluation_bundle(bundle, protocol, manifest, manifest_sha256)
    ledger = ArtifactLedger(paths.ledger_root, protocol_sha256=protocol.sha256)
    expected_run_start = run_start_commitment(manifest, manifest_sha256)
    adapter = CellCompletionAdapter(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        checkpoint_path=checkpoint_path,
        source_root=source_root,
    )

    def prepare_results() -> bytes:
        verified = verify_freeze_manifest(
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            checkpoint_path=checkpoint_path,
            source_root=source_root,
        )
        if verified != manifest:
            raise FreezeIntegrityError("freeze changed at completion seal boundary")
        return load_worker_snapshot(manifest.sandbox_worker_sha256)

    runtime_identity = {
        "implementation": manifest.sandbox_python_implementation,
        "python_version": manifest.sandbox_python_version,
        "executable_sha256": manifest.sandbox_python_executable_sha256,
    }
    return _run_cell_evaluation_engine(
        bundle=bundle,
        adapter=adapter,
        ledger=ledger,
        protocol=protocol,
        executor=execute_completion,
        python_executable=Path(manifest.sandbox_python_executable),
        runtime_identity=runtime_identity,
        expected_run_start=expected_run_start,
        prepare_results=prepare_results,
    )


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
