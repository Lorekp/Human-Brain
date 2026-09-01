from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_continual_token_edit_consequence_policy_v1 as policy
from experiments.r3_pilot import continual_rust_token_edit_outcome_reuse_c3b_v1 as cohort
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

PREREGISTRATION_COMMIT = "1" * 40
EXPECTED_SPEC_SHA256 = "0905ff1099e38db828edf86135ba64c94b2e21ad63ea991db7c937ad0e6e9576"


def _root(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


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
        source = cohort._SOURCES[index]
        source_sha256 = sha256(source.encode("ascii")).hexdigest()
        outputs = (
            cohort.SEED_FAILED_OUTPUTS,
            cohort.SEED_EXPECTED_OUTPUTS,
            cohort.ADAPTATION_BASELINE_OUTPUTS,
            cohort.ADAPTATION_EXPECTED_OUTPUTS,
            cohort.EVALUATION_BASELINE_OUTPUTS,
            cohort.EVALUATION_EXPECTED_OUTPUTS,
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
            preparation=SimpleNamespace(
                batch_sha256=_root(f"fake-batch:{batch!r}"),
            ),
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


def test_c3_fixtures_are_fresh_and_exact() -> None:
    cohort.DEFAULT_PUBLIC_SPEC.authenticate()

    assert cohort.DEFAULT_PUBLIC_SPEC.sha256 == EXPECTED_SPEC_SHA256
    assert (
        tuple(sha256(source.encode("ascii")).hexdigest() for source in cohort._SOURCES)
        == cohort.SOURCE_SHA256S
    )
    assert len(set(cohort.SOURCE_SHA256S)) == 6
    assert not set(cohort.SOURCE_SHA256S).intersection(cohort.EXCLUDED_PRIOR_SOURCE_SHA256S)
    assert len(cohort.EXCLUDED_PRIOR_SOURCE_SHA256S) == 18
    assert cohort.SCHEMA_INDEX == 3
    assert cohort.SCHEMA_SHA256 == (
        "7cf44511bced68635e3c77a199542061b404b5861fe6a51c7c6ba88b01c2b014"
    )
    assert cohort.DEFAULT_PUBLIC_SPEC.frozen_mechanism_commit == (
        "d9a3018cebaadda28b524bd32c7d25c4c56970e6"
    )
    assert len(cohort.DEFAULT_PUBLIC_SPEC.frozen_mechanism_file_sha256s) == 6


def test_six_fake_executions_reuse_one_pass_without_evaluation_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_nested_execution_authentication(monkeypatch)
    destination = tmp_path / "c3.result.json"
    reservation = tmp_path / "c3.result.json.reservation.json"
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

    assert result.status == cohort.STATUS_PASS
    assert result.reservation.preregistration_commit == PREREGISTRATION_COMMIT
    assert tuple(item.passed_cell for item in result.attempts) == (0, 1, 0, 1, 0, 1)
    assert tuple(fake.source_sha256s) == cohort.SOURCE_SHA256S
    assert feedback_open_counts == [4, 6]
    assert result.base_checkpoint.support_counts[cohort.SCHEMA_INDEX] == 1
    assert sum(result.base_checkpoint.support_counts) == 1
    assert result.adaptation_normal.status == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert result.adaptation_exploration.status == policy.EXPLORATION_READY
    assert result.adaptation_exploration.selected_schema_index == cohort.SCHEMA_INDEX
    assert result.adaptation_exploration.candidate_view.applicable_schema_indices == (
        0,
        1,
        2,
        3,
    )
    adaptation_scores = dict(
        zip(
            result.adaptation_exploration.activation.active_action_sha256s,
            result.adaptation_exploration.activation.base_scores,
            strict=True,
        )
    )
    assert adaptation_scores[cohort.SCHEMA_SHA256] == 2
    assert result.evaluation_parent.status == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert result.evaluation_outcome_off.status == policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert result.evaluation_successor.status == policy.READY
    assert result.evaluation_successor.selected_schema_index == cohort.SCHEMA_INDEX
    assert result.evaluation_successor.candidate_view.applicable_schema_indices == (
        0,
        1,
        2,
        3,
    )
    assert len(result.successor_policy.outcome_memory.provenance) == 1
    assert result.work.record()["policy_advance_count"] == 1
    assert result.work.record()["evaluation_policy_advance_count"] == 0

    document = json.loads(destination.read_text(encoding="ascii"))
    assert document["terminal_status"] == cohort.STATUS_PASS
    assert document["result_sha256"] == result.sha256
    assert len(document["full_authenticated_transcript"]["attempts"]) == 6
    assert (
        "compile_receipt" in document["full_authenticated_transcript"]["attempts"][0]["execution"]
    )
    assert reservation.is_file()


def test_reservation_precedes_execution_and_abort_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_nested_execution_authentication(monkeypatch)
    destination = tmp_path / "c3.result.json"
    reservation = tmp_path / "c3.result.json.reservation.json"
    calls = 0

    def fail_after_reservation(_prepared, _selection, _batch):
        nonlocal calls
        calls += 1
        assert reservation.is_file()
        assert not destination.exists()
        raise RuntimeError("injected execution failure")

    with pytest.raises(RuntimeError, match="injected execution failure"):
        cohort._run_and_write_at_paths_for_test_v1(
            preregistration_commit=PREREGISTRATION_COMMIT,
            destination=destination,
            reservation_path=reservation,
            execute=fail_after_reservation,
            adaptation_feedback=lambda: cohort.ADAPTATION_EXPECTED_OUTPUTS,
            evaluation_feedback=lambda: cohort.EVALUATION_EXPECTED_OUTPUTS,
        )

    assert calls == 1
    assert reservation.is_file()
    document = json.loads(destination.read_text(encoding="ascii"))
    assert document["terminal_status"] == cohort.STATUS_EXECUTION_ABORT
    assert document["exception_type"] == "RuntimeError"


def test_production_compiler_preflight_precedes_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "c3b.result.json"
    reservation = tmp_path / "c3b.result.json.reservation.json"
    calls = 0

    monkeypatch.setattr(
        cohort,
        "_production_evidence_paths_v1",
        lambda: (destination, reservation),
    )

    def unavailable_rustc():
        nonlocal calls
        calls += 1
        raise cohort.rust_executor.RustSelectedActionExecutionError(
            "concrete rustc executable could not be resolved"
        )

    monkeypatch.setattr(cohort.rust_executor, "_discover_rustc", unavailable_rustc)
    with pytest.raises(
        cohort.rust_executor.RustSelectedActionExecutionError,
        match="concrete rustc executable could not be resolved",
    ):
        cohort.run_and_write_cohort_v1(
            preregistration_commit=PREREGISTRATION_COMMIT,
        )

    assert calls == 1
    assert not reservation.exists()
    assert not destination.exists()
