from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as outcomes
from experiments.r3_pilot import common_failure_conditioned_action_replan_v1 as replans
from experiments.r3_pilot import continual_rust_token_edit_failure_replan_cohort_v1 as cohort
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

PREREGISTRATION_COMMIT = "1" * 40
EXPECTED_SPEC_SHA256 = "19e88590e6158f69e7c995b0af018aecc2973a14721394e5d74e762e819ffbc9"


def _root(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _scores(decision: outcomes.ActionDecisionV1) -> dict[str, int]:
    return dict(
        zip(
            decision.activation.active_action_sha256s,
            decision.effective_scores,
            strict=True,
        )
    )


class _FakeSelectedExecutor:
    def __init__(self, *, reservation: Path, destination: Path) -> None:
        self.reservation = reservation
        self.destination = destination
        self.source_sha256s: list[str] = []

    def __call__(self, prepared, selection, batch) -> RustSelectedActionExecutionV2:
        assert self.reservation.is_file()
        assert not self.destination.exists()
        index = tuple(candidate.sha256 for candidate in cohort._CANDIDATES).index(
            selection.candidate_sha256
        )
        source = cohort._EXECUTION_SOURCES[index]
        source_sha256 = sha256(source.encode("ascii")).hexdigest()
        outputs = (
            cohort.ADAPTATION_SCHEMA_A_OUTPUTS,
            cohort.ADAPTATION_SCHEMA_B_OUTPUTS,
            cohort.EVALUATION_SCHEMA_A_OUTPUTS,
            cohort.EVALUATION_SCHEMA_B_OUTPUTS,
        )[index]
        prepared._consume_issued_selection(selection)
        with prepared._lock:
            prepared._source_release_count += 1
            prepared._execution_attempt_count += 1
            prepared._successful_execution_count += 1
        self.source_sha256s.append(source_sha256)
        return RustSelectedActionExecutionV2(
            selected_source=SimpleNamespace(
                authority_sha256=prepared.authority_sha256,
                selection_sha256=selection.sha256,
                action_handle=cohort._CANDIDATES[index].candidate_id,
                source=source,
                source_sha256=source_sha256,
            ),
            rustc_identity=SimpleNamespace(),
            preparation=SimpleNamespace(batch_sha256=_root(f"fake-batch:{batch!r}")),
            dependency_receipt=SimpleNamespace(),
            compile_receipt=SimpleNamespace(compile_invocations=1),
            run_receipt=SimpleNamespace(run_invocations=1),
            outputs=outputs,
            sha256=_root(f"fake-execution:{index}:{source_sha256}"),
        )


def _disable_nested_execution_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RustSelectedActionExecutionV2,
        "authenticate_structure",
        lambda self: None,
    )


def test_c4_fixtures_and_token_projections_are_exact_and_fresh() -> None:
    cohort.DEFAULT_PUBLIC_SPEC.authenticate()
    adaptation_frame, adaptation = cohort._project_action_sources(cohort.ADAPTATION_SOURCE)
    evaluation_frame, evaluation = cohort._project_action_sources(cohort.EVALUATION_SOURCE)

    assert cohort.DEFAULT_PUBLIC_SPEC.sha256 == EXPECTED_SPEC_SHA256
    assert cohort.DEFAULT_PUBLIC_SPEC.frozen_mechanism_commit == (
        "11e29b61aed6a5f6c381c7470b38bf5f2effe915"
    )
    assert len(cohort.DEFAULT_PUBLIC_SPEC.frozen_mechanism_file_sha256s) == 6
    assert cohort.ACTION_SCHEMA_INDICES == (2, 3)
    assert cohort.ACTION_SHA256S == (
        "610269c84c10d687d7efcb0f435c982bcf39a97d295b58fbe43a14e41efb04f5",
        "7cf44511bced68635e3c77a199542061b404b5861fe6a51c7c6ba88b01c2b014",
    )
    assert cohort.ACTION_BASE_SCORES == (6, 4)
    assert tuple(cohort._application_record(item) for item in adaptation) == (
        (2, 48, 49, 45),
        (3, 48, 49, 52),
    )
    assert tuple(cohort._application_record(item) for item in evaluation) == (
        (2, 51, 52, 48),
        (3, 51, 52, 55),
    )
    assert adaptation_frame.source_sha256 == cohort.SOURCE_SHA256S[0]
    assert tuple(item.edited_source.source_sha256 for item in adaptation) == (
        cohort.SOURCE_SHA256S[1],
        cohort.SOURCE_SHA256S[2],
    )
    assert evaluation_frame.source_sha256 == cohort.SOURCE_SHA256S[3]
    assert tuple(item.edited_source.source_sha256 for item in evaluation) == (
        cohort.SOURCE_SHA256S[4],
        cohort.SOURCE_SHA256S[5],
    )
    assert len(set(cohort.SOURCE_SHA256S)) == 6
    assert len(cohort.EXCLUDED_PRIOR_SOURCE_SHA256S) == 24
    assert not set(cohort.SOURCE_SHA256S).intersection(cohort.EXCLUDED_PRIOR_SOURCE_SHA256S)


def test_four_fake_executions_fail_replan_pass_and_reuse_without_evaluation_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_nested_execution_authentication(monkeypatch)
    destination = tmp_path / "c4.result.json"
    reservation = tmp_path / "c4.result.json.reservation.json"
    fake = _FakeSelectedExecutor(reservation=reservation, destination=destination)
    feedback_open_counts: list[int] = []

    def adaptation_feedback() -> tuple[tuple[int, ...], ...]:
        feedback_open_counts.append(len(fake.source_sha256s))
        return cohort.ADAPTATION_EXPECTED_OUTPUTS

    def evaluation_feedback() -> tuple[tuple[int, ...], ...]:
        feedback_open_counts.append(len(fake.source_sha256s))
        return cohort.EVALUATION_EXPECTED_OUTPUTS

    result = cohort._run_and_write_at_paths_for_test_v1(
        preregistration_commit=PREREGISTRATION_COMMIT,
        destination=destination,
        reservation_path=reservation,
        execute=fake,
        adaptation_feedback=adaptation_feedback,
        evaluation_feedback=evaluation_feedback,
    )
    result.authenticate()

    action_a, action_b = cohort.ACTION_SHA256S
    assert result.status == cohort.STATUS_PASS
    assert tuple(item.passed_cell for item in result.attempts) == (0, 1, 0, 1)
    assert tuple(fake.source_sha256s) == cohort.EXECUTION_SOURCE_SHA256S
    assert feedback_open_counts == [1, 4]
    assert result.failure_outcome.selected_action_sha256 == action_a
    assert _scores(result.failure_outcome.decision) == {action_a: 6, action_b: 4}
    assert result.refractory_off.status == replans.REFRACTORY_OFF
    assert result.refractory_off.decision is not None
    assert result.refractory_off.decision.status == outcomes.STOP_PHYSICAL_RACE
    assert _scores(result.refractory_off.decision) == {action_a: 4, action_b: 4}
    assert result.gate_off.status == replans.STOP_GATE_OFF
    assert result.gate_off.successor is result.parent_memory
    assert result.gate_off.retry_activation is None
    assert result.gate_off.decision is None
    assert result.replan.status == replans.REFRACTORY_RERACED
    assert result.replan.retry_activation == result.refractory_off.retry_activation
    assert result.replan.pulsed_activation is not None
    assert result.replan.pulsed_activation.active_action_sha256s == cohort.ACTION_SHA256S
    assert result.replan.decision is not None
    assert result.replan.decision.status == outcomes.READY
    assert result.replan.decision.selected_action_sha256 == action_b
    assert _scores(result.replan.decision) == {action_a: 2, action_b: 4}
    assert result.final_memory.inhibitory_ticks == (1, 0)
    assert result.final_memory.excitatory_ticks == (0, 1)
    assert result.evaluation_parent.selected_action_sha256 == action_a
    assert _scores(result.evaluation_parent) == {action_a: 6, action_b: 4}
    assert result.evaluation_final.selected_action_sha256 == action_b
    assert _scores(result.evaluation_final) == {action_a: 4, action_b: 6}
    assert result.work.outcome_advance_count == 2
    assert result.work.replan_count == 1
    assert result.work.evaluation_outcome_advance_count == 0
    assert len(result.authorities) == len(result.catalogues) == 2
    assert all(
        item == cohort._EXPECTED_PAIR_WORK_COUNTERS for item in result.authority_work_counters
    )

    document = json.loads(destination.read_text(encoding="ascii"))
    assert document["terminal_status"] == cohort.STATUS_PASS
    assert document["result_sha256"] == result.sha256
    assert len(document["full_authenticated_transcript"]["attempts"]) == 4
    assert (
        "compile_receipt" in document["full_authenticated_transcript"]["attempts"][0]["execution"]
    )
    assert reservation.is_file()


def test_preflight_precedes_reservation_and_post_reservation_abort_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "c4.result.json"
    reservation = tmp_path / "c4.result.json.reservation.json"
    preflight_calls = 0
    execution_calls = 0

    monkeypatch.setattr(
        cohort,
        "_production_evidence_paths_v1",
        lambda: (destination, reservation),
    )

    def unavailable_rustc():
        nonlocal preflight_calls
        preflight_calls += 1
        raise cohort.rust_executor.RustSelectedActionExecutionError(
            "concrete rustc executable could not be resolved"
        )

    monkeypatch.setattr(cohort.rust_executor, "_discover_rustc", unavailable_rustc)
    with pytest.raises(
        cohort.rust_executor.RustSelectedActionExecutionError,
        match="concrete rustc executable could not be resolved",
    ):
        cohort.run_and_write_cohort_v1(preregistration_commit=PREREGISTRATION_COMMIT)

    assert preflight_calls == 1
    assert not reservation.exists()
    assert not destination.exists()

    def fail_after_reservation(_prepared, _selection, _batch):
        nonlocal execution_calls
        execution_calls += 1
        assert reservation.is_file()
        assert not destination.exists()
        raise RuntimeError("injected C4 execution failure")

    with pytest.raises(RuntimeError, match="injected C4 execution failure"):
        cohort._run_and_write_at_paths_for_test_v1(
            preregistration_commit=PREREGISTRATION_COMMIT,
            destination=destination,
            reservation_path=reservation,
            execute=fail_after_reservation,
            adaptation_feedback=lambda: cohort.ADAPTATION_EXPECTED_OUTPUTS,
            evaluation_feedback=lambda: cohort.EVALUATION_EXPECTED_OUTPUTS,
        )

    assert execution_calls == 1
    assert reservation.is_file()
    document = json.loads(destination.read_text(encoding="ascii"))
    assert document["terminal_status"] == cohort.STATUS_EXECUTION_ABORT
    assert document["exception_type"] == "RuntimeError"
