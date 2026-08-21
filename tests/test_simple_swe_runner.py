from __future__ import annotations

import inspect
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from humanbrain.simple_swe import runner, sandbox
from humanbrain.simple_swe.adapter import (
    CompletionAudit,
    CompletionRequest,
    SynthesizedCompletion,
)
from humanbrain.simple_swe.artifacts import ArtifactIntegrityError, ArtifactLedger
from humanbrain.simple_swe.freeze import FreezeIntegrityError
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.reveal import (
    EvaluationBundle,
    evaluation_reveal_commitment,
)
from humanbrain.simple_swe.runner import (
    _run_cell_evaluation_for_testing,
    record_baseline_control,
    run_cell_evaluation,
)
from humanbrain.simple_swe.sandbox import ExecutionResult
from humanbrain.simple_swe.tasks import (
    GeneratedTask,
    HiddenCase,
    PrivateTask,
    PublicTask,
    canonical_sha256,
)

PROTOCOL_PATH = Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
ADAPTER_SHA256 = "a" * 64
CHECKPOINT_SHA256 = "b" * 64
CHECKPOINT_STATE_SHA256 = "c" * 64
COMPILER_SHA256 = "d" * 64
FREEZE_SHA256 = "e" * 64
WORKER_SHA256 = sha256(
    Path(sandbox.__file__).with_name("_worker.py").read_bytes()
).hexdigest()


def _task(index: int, tier: int) -> GeneratedTask:
    task_id = f"fixture/eval/{index:03d}"
    entry_point = f"solve_{index:03d}"
    regressions = (HiddenCase(args=(0,), expected=index, distribution="fixed-regression"),)
    properties = (HiddenCase(args=(7,), expected=7 + index, distribution="edge-distribution"),)
    private_record = {
        "task_id": task_id,
        "entry_point": entry_point,
        "parameter_names": ["value"],
        "regression_cases": [asdict(case) for case in regressions],
        "property_cases": [asdict(case) for case in properties],
    }
    test_sha256 = canonical_sha256(private_record)
    public_record = {
        "task_id": task_id,
        "partition": "eval",
        "tier": tier,
        "family": f"fixture_t{tier}",
        "ordinal": index,
        "entry_point": entry_point,
        "parameter_names": ["value"],
        "prompt": f"def {entry_point}(value): Return value plus {index}.",
        "hidden_test_sha256": test_sha256,
    }
    task_sha256 = canonical_sha256(public_record)
    return GeneratedTask(
        public=PublicTask(**public_record, task_sha256=task_sha256),
        private=PrivateTask(
            task_id=task_id,
            public_task_sha256=task_sha256,
            regression_cases=regressions,
            property_cases=properties,
            test_sha256=test_sha256,
        ),
    )


def _bundle(tmp_path: Path, protocol: ProtocolSpec) -> EvaluationBundle:
    return EvaluationBundle(
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol.sha256,
        freeze_manifest_sha256=FREEZE_SHA256,
        checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_state_sha256=CHECKPOINT_STATE_SHA256,
        source_head="fixture-source-head",
        eval_seed_sha256="f" * 64,
        eval_seed=b"fixture-seed-is-private".ljust(32, b"\0"),
        tasks=(_task(0, 0), _task(1, 1), _task(2, 2)),
        bundle_sha256="1" * 64,
        path=tmp_path / "private-eval-bundle.json",
    )


def _ledger(tmp_path: Path, bundle: EvaluationBundle) -> ArtifactLedger:
    ledger = ArtifactLedger(tmp_path / "ledger", protocol_sha256=bundle.protocol_sha256)
    ledger.append(
        "run-start",
        {
            "freeze_manifest_sha256": bundle.freeze_manifest_sha256,
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "checkpoint_state_sha256": bundle.checkpoint_state_sha256,
            "source_head": bundle.source_head,
            "completion_adapter_sha256": ADAPTER_SHA256,
            "compiler_sha256": COMPILER_SHA256,
            "canonical_runner_id": "test-injected-engine-v1",
            "canonical_runner_sha256": "9" * 64,
            "sandbox_worker_sha256": WORKER_SHA256,
            "sandbox_runtime": {
                "implementation": "cpython",
                "python_version": "3.11.9",
                "executable_sha256": "8" * 64,
            },
            "production_dependencies_injected": True,
        },
    )
    ledger.append(
        "evaluation-seed-reserved",
        {
            "eval_seed_sha256": bundle.eval_seed_sha256,
            "seed_reservation_sha256": "7" * 64,
            "seed_persisted_before_task_generation": True,
            "evaluation_tasks_generated": False,
        },
    )
    ledger.append(
        "external-anchor-recorded",
        {
            "anchor_payload_sha256": "6" * 64,
            "public_uri": "https://example.invalid/fixture-anchor",
            "receipt_sha256": "5" * 64,
            "operator_supplied_receipt": True,
            "harness_verified_external_publication": False,
        },
    )
    ledger.append("evaluation-revealed", evaluation_reveal_commitment(bundle))
    return ledger


class FixtureCellAdapter:
    checkpoint_sha256 = CHECKPOINT_SHA256

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []

    def synthesize(self, request: CompletionRequest) -> SynthesizedCompletion:
        self.calls.append(request.entry_point)
        if request.entry_point == self.fail_on:
            raise RuntimeError("synthetic interruption")
        ordinal = int(request.entry_point.rsplit("_", 1)[1])
        source = f"def {request.entry_point}(value):\n    return value + {ordinal}\n"
        emitted_text = source
        token_ids = tuple(range(192))
        return SynthesizedCompletion(
            source=source,
            emitted_text=emitted_text,
            audit=CompletionAudit(
                adapter_id="humanbrain-cell-completion-adapter-v1",
                adapter_sha256=ADAPTER_SHA256,
                compiler_id="raw-python-source-v1",
                compiler_sha256=COMPILER_SHA256,
                compiler_mode="raw-python",
                compiler_algorithm_source="emitted-tokens-only",
                checkpoint_sha256=CHECKPOINT_SHA256,
                learned_checkpoint_state_before=CHECKPOINT_STATE_SHA256,
                learned_checkpoint_state_after=CHECKPOINT_STATE_SHA256,
                prompt_sha256=sha256(request.prompt.encode("utf-8")).hexdigest(),
                emitted_token_ids=token_ids,
                emitted_token_count=len(token_ids),
                emitted_token_sha256=canonical_sha256(list(token_ids)),
                emitted_text_sha256=sha256(emitted_text.encode("utf-8")).hexdigest(),
                source_sha256=sha256(source.encode("utf-8")).hexdigest(),
                direct_synthesis_no_tools=True,
            ),
        )


def test_all_completions_are_sealed_before_hidden_tests_execute(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    bundle = _bundle(tmp_path, protocol)
    ledger = _ledger(tmp_path, bundle)
    adapter = FixtureCellAdapter()
    executor_calls: list[str] = []
    outcomes = ("passed", "regression_failure", "syntax_error")

    def executor(
        source,
        public,
        private,
        limits,
        *,
        python_executable=None,
        worker_source=None,
        worker_sha256=None,
    ):
        del source, limits, python_executable, worker_source, worker_sha256
        events = ledger.verify()
        kinds = [event.kind for event in events]
        assert kinds.count("cell-completion") == len(bundle.tasks)
        assert "cell-completions-sealed" in kinds
        assert public.hidden_test_sha256 == private.test_sha256
        executor_calls.append(public.task_id)
        outcome = outcomes[len(executor_calls) - 1]
        return ExecutionResult(
            outcome=outcome,
            passed=outcome == "passed",
            wall_seconds=0.01,
            detail="fixture result",
            sandboxed=True,
        )

    summary = _run_cell_evaluation_for_testing(
        bundle=bundle,
        adapter=adapter,
        ledger=ledger,
        protocol=protocol,
        executor=executor,
    )

    assert adapter.calls == [task.public.entry_point for task in bundle.tasks]
    assert executor_calls == [task.public.task_id for task in bundle.tasks]
    assert summary.task_count == 3
    assert summary.passed == 1
    assert summary.pass_at_1 == pytest.approx(1 / 3)
    assert summary.outcome_counts["passed"] == 1
    assert summary.outcome_counts["regression_failure"] == 1
    assert summary.outcome_counts["syntax_error"] == 1
    assert summary.tier_counts == {
        "0": {"tasks": 1, "passed": 1, "pass_at_1": 1.0},
        "1": {"tasks": 1, "passed": 0, "pass_at_1": 0.0},
        "2": {"tasks": 1, "passed": 0, "pass_at_1": 0.0},
    }
    assert [event.kind for event in ledger.verify()] == [
        "run-start",
        "evaluation-seed-reserved",
        "external-anchor-recorded",
        "evaluation-revealed",
        "cell-completion",
        "cell-completion",
        "cell-completion",
        "cell-completions-sealed",
        "task-result",
        "task-result",
        "task-result",
        "cell-summary",
    ]
    seal = next(event for event in ledger.verify() if event.kind == "cell-completions-sealed")
    assert seal.body["hidden_tests_executed_before_seal"] is False
    assert seal.body["private_cases_not_passed_to_adapter"] is True
    assert seal.body["production_dependencies_injected"] is True


def test_interrupted_synthesis_resumes_without_resampling_a_task(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    bundle = _bundle(tmp_path, protocol)
    ledger = _ledger(tmp_path, bundle)
    interrupted = FixtureCellAdapter(fail_on=bundle.tasks[1].public.entry_point)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run_cell_evaluation_for_testing(
            bundle=bundle,
            adapter=interrupted,
            ledger=ledger,
            protocol=protocol,
        )
    first_completion_sha256 = ledger.verify()[4].file_sha256
    resumed = FixtureCellAdapter()

    summary = _run_cell_evaluation_for_testing(
        bundle=bundle,
        adapter=resumed,
        ledger=ledger,
        protocol=protocol,
        executor=lambda *_args, **_kwargs: ExecutionResult(
            "passed", True, 0.01, "fixture result", True
        ),
    )

    assert interrupted.calls == [
        bundle.tasks[0].public.entry_point,
        bundle.tasks[1].public.entry_point,
    ]
    assert resumed.calls == [
        bundle.tasks[1].public.entry_point,
        bundle.tasks[2].public.entry_point,
    ]
    assert ledger.verify()[4].file_sha256 == first_completion_sha256
    assert summary.passed == len(bundle.tasks)


def test_baseline_controls_cannot_precede_or_change_the_cell_summary(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    bundle = _bundle(tmp_path, protocol)
    ledger = _ledger(tmp_path, bundle)

    with pytest.raises(ArtifactIntegrityError, match="cell summary"):
        record_baseline_control(
            ledger=ledger,
            system_name="fixture-conventional-baseline",
            task_count=3,
            pass_at_1=1.0,
            report_sha256="2" * 64,
        )
    summary = _run_cell_evaluation_for_testing(
        bundle=bundle,
        adapter=FixtureCellAdapter(),
        ledger=ledger,
        protocol=protocol,
        executor=lambda *_args, **_kwargs: ExecutionResult(
            "passed", True, 0.01, "fixture result", True
        ),
    )
    cell_summary_event = ledger.verify()[-1]

    with pytest.raises(ValueError, match=r"match.*cohort"):
        record_baseline_control(
            ledger=ledger,
            system_name="partial-baseline",
            task_count=2,
            pass_at_1=1.0,
            report_sha256="3" * 64,
        )

    baseline_event = record_baseline_control(
        ledger=ledger,
        system_name="fixture-conventional-baseline",
        task_count=3,
        pass_at_1=1.0,
        report_sha256="2" * 64,
    )
    resumed_summary = _run_cell_evaluation_for_testing(
        bundle=bundle,
        adapter=FixtureCellAdapter(),
        ledger=ledger,
        protocol=protocol,
    )

    assert baseline_event.kind == "baseline-control"
    assert baseline_event.body["reporting_control_only"] is True
    assert ledger.verify()[-2].file_sha256 == cell_summary_event.file_sha256
    assert resumed_summary == summary


def test_canonical_runner_accepts_only_frozen_artifact_paths() -> None:
    fields = set(inspect.signature(run_cell_evaluation).parameters)

    assert fields == {
        "manifest_path",
        "protocol_path",
        "checkpoint_path",
        "source_root",
    }
    assert "bundle" not in fields
    assert "adapter" not in fields
    assert "executor" not in fields
    assert "python_executable" not in fields


def test_canonical_runner_rejects_an_in_memory_bundle_argument(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL_PATH)

    with pytest.raises(TypeError, match="unexpected keyword argument 'bundle'"):
        run_cell_evaluation(bundle=_bundle(tmp_path, protocol))  # type: ignore[call-arg]


def test_canonical_runner_loads_and_validates_the_bundle_file_before_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text("{}", encoding="utf-8")
    bundle_path = tmp_path / "runs" / "private" / "eval-bundle.json"
    loaded_bundle = object()
    calls: list[object] = []
    monkeypatch.setattr(
        runner, "load_protocol", lambda _path: SimpleNamespace(sha256="a" * 64)
    )
    monkeypatch.setattr(
        runner, "verify_freeze_manifest", lambda **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        runner,
        "canonical_evaluation_paths",
        lambda *_args: SimpleNamespace(
            bundle_path=bundle_path, ledger_root=tmp_path / "ledger"
        ),
    )

    def load_bundle(path):
        calls.append(Path(path))
        return loaded_bundle

    def reject_bundle(bundle, *_args):
        calls.append(bundle)
        raise FreezeIntegrityError("deterministic bundle validation failed")

    monkeypatch.setattr(runner, "load_evaluation_bundle", load_bundle)
    monkeypatch.setattr(runner, "validate_evaluation_bundle", reject_bundle)
    monkeypatch.setattr(
        runner,
        "CellCompletionAdapter",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("adapter constructed before bundle validation")
        ),
    )

    with pytest.raises(FreezeIntegrityError, match="deterministic bundle validation"):
        run_cell_evaluation(
            manifest_path=manifest_path,
            protocol_path=PROTOCOL_PATH,
            checkpoint_path=tmp_path / "checkpoint.npz",
            source_root=tmp_path / "source",
        )

    assert calls == [bundle_path, loaded_bundle]


def test_injected_test_engine_cannot_write_a_canonical_runner_prefix(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    bundle = _bundle(tmp_path, protocol)
    fixture_ledger = _ledger(tmp_path / "fixture", bundle)
    canonical_ledger = ArtifactLedger(
        tmp_path / "canonical-ledger", protocol_sha256=protocol.sha256
    )
    for index, event in enumerate(fixture_ledger.verify()):
        body = dict(event.body)
        if index == 0:
            body["canonical_runner_id"] = "humanbrain-simple-swe-sealed-runner-v1"
            body["production_dependencies_injected"] = False
        canonical_ledger.append(event.kind, body)

    with pytest.raises(ArtifactIntegrityError, match="cannot write a canonical"):
        _run_cell_evaluation_for_testing(
            bundle=bundle,
            adapter=FixtureCellAdapter(),
            ledger=canonical_ledger,
            protocol=protocol,
        )
