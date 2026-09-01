from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as adapter
from experiments.r3_pilot import (
    continual_rust_diagnostic_factor_recombination_cohort_v1 as cohort,
)
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

PREREGISTRATION_COMMIT = "1" * 40
EXPECTED_SPEC_SHA256 = "52b62afa4df5f1c0d1f9e35bbfcd3872afc6fc74939e3e3be91a3f985b4d6d2c"


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


class _FakeDiagnosticCompiler:
    def __init__(self) -> None:
        self.source_sha256s: list[str] = []

    def __call__(
        self,
        rustc_identity_sha256: str,
        source: str,
    ) -> adapter.RustcDiagnosticCompileAttemptV1:
        episode = cohort.ORIGINAL_SOURCES.index(source)
        code, _ = cohort.EXPECTED_DIAGNOSTICS[episode]
        token = "_wrong_value" if code == "E0308" else "_missing_value"
        byte_start = source.rindex(token)
        byte_end = byte_start + len(token)
        diagnostic = {
            "$message_type": "diagnostic",
            "message": "fix-bearing text excluded from cognition",
            "code": {"code": code, "explanation": None},
            "level": "error",
            "spans": [
                {
                    "file_name": cohort.SOURCE_FILE_NAME,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "is_primary": True,
                    "label": "secret label",
                    "suggested_replacement": "secret replacement",
                }
            ],
            "children": [],
            "rendered": "secret rendered diagnostic",
        }
        summary = {
            "$message_type": "diagnostic",
            "message": "aborting due to previous error",
            "code": None,
            "level": "error",
            "spans": [],
            "children": [],
            "rendered": "error: aborting due to previous error",
        }
        stderr = b"".join(
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in (diagnostic, summary)
        )
        source_sha256 = sha256(source.encode("ascii")).hexdigest()
        self.source_sha256s.append(source_sha256)
        return adapter.seal_rustc_diagnostic_compile_attempt_v1(
            rustc_identity_sha256=rustc_identity_sha256,
            source_file_name=cohort.SOURCE_FILE_NAME,
            source=source.encode("ascii"),
            arguments=cohort.DIAGNOSTIC_ARGUMENTS,
            returncode=1,
            stdout=b"",
            stderr=stderr,
            compile_invocations=1,
        )


class _FakeSelectedExecutor:
    def __init__(self, *, rustc_identity_sha256: str) -> None:
        self.rustc_identity_sha256 = rustc_identity_sha256
        self.source_sha256s: list[str] = []

    def __call__(self, prepared, selection, batch) -> RustSelectedActionExecutionV2:
        binding = prepared._binding_for_candidate(selection.candidate_sha256)
        source = binding.rust_source
        source_sha256 = binding.rust_source_sha256
        selected_sources = (
            cohort.EPISODE_CANDIDATE_SOURCES[0][0],
            cohort.EPISODE_CANDIDATE_SOURCES[1][3],
            cohort.EPISODE_CANDIDATE_SOURCES[2][1],
        )
        outputs_by_source = {
            sha256(value.encode("ascii")).hexdigest(): outputs
            for value, outputs in zip(
                selected_sources,
                cohort.EXPECTED_OUTPUTS,
                strict=True,
            )
        }
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
                action_handle=binding.candidate.candidate_id,
                source=source,
                source_sha256=source_sha256,
            ),
            rustc_identity=SimpleNamespace(sha256=self.rustc_identity_sha256),
            preparation=SimpleNamespace(batch_sha256=_root(f"batch:{batch!r}")),
            dependency_receipt=SimpleNamespace(),
            compile_receipt=SimpleNamespace(compile_invocations=1),
            run_receipt=SimpleNamespace(run_invocations=1),
            outputs=outputs_by_source.get(source_sha256, ((-999,),)),
            sha256=_root(f"execution:{len(self.source_sha256s)}:{source_sha256}"),
        )


def test_c5_public_spec_freezes_full_fresh_cartesian_bank() -> None:
    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate()

    assert spec.sha256 == EXPECTED_SPEC_SHA256
    assert len(spec.source_sha256s) == len(set(spec.source_sha256s)) == 15
    assert not set(spec.source_sha256s).intersection(spec.excluded_source_sha256s)
    assert len(spec.hypothesis_sha256s) == len(set(spec.hypothesis_sha256s)) == 4
    assert len(spec.candidate_sha256s) == len(set(spec.candidate_sha256s)) == 4
    assert set(dict(spec.hypothesis_candidate_rows)) == set(spec.hypothesis_sha256s)
    assert set(dict(spec.hypothesis_candidate_rows).values()) == set(spec.candidate_sha256s)
    assert spec.expected_diagnostics == (
        ("E0425", adapter.FIRST_HALF),
        ("E0308", adapter.SECOND_HALF),
        ("E0425", adapter.SECOND_HALF),
    )


def test_two_diagonal_lessons_select_and_execute_unseen_cross(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RustSelectedActionExecutionV2,
        "authenticate_structure",
        lambda self: None,
    )
    destination = tmp_path / "c5.result.json"
    reservation = tmp_path / "c5.result.json.reservation.json"
    rustc_identity_sha256 = _root("fake-rustc-identity")
    diagnostics = _FakeDiagnosticCompiler()
    executor = _FakeSelectedExecutor(rustc_identity_sha256=rustc_identity_sha256)
    feedback_open_counts: list[int] = []
    expected_by_role = dict(zip(cohort.ROLES, cohort.EXPECTED_OUTPUTS, strict=True))

    def output_feedback(role: str) -> tuple[tuple[int, ...], ...]:
        feedback_open_counts.append(len(executor.source_sha256s))
        return expected_by_role[role]

    result = cohort._run_and_write_at_paths_for_test_v1(
        preregistration_commit=PREREGISTRATION_COMMIT,
        rustc_identity_sha256=rustc_identity_sha256,
        destination=destination,
        reservation_path=reservation,
        diagnostic_compile=diagnostics,
        execute=executor,
        output_feedback=output_feedback,
    )
    result.authenticate()

    expected_selected_sources = (
        cohort.EPISODE_CANDIDATE_SOURCES[0][0],
        cohort.EPISODE_CANDIDATE_SOURCES[1][3],
        cohort.EPISODE_CANDIDATE_SOURCES[2][1],
    )
    assert result.status == cohort.STATUS_PASS
    assert tuple(diagnostics.source_sha256s) == cohort.SOURCE_SHA256S[0::5]
    assert tuple(executor.source_sha256s) == tuple(
        sha256(source.encode("ascii")).hexdigest() for source in expected_selected_sources
    )
    assert feedback_open_counts == [1, 2, 3]
    assert tuple(
        decision.selected_hypothesis_sha256 for decision in result.evaluation_decisions
    ) == (None, None, cohort.H01_SHA256)
    assert all(decision.selected_hypothesis_sha256 is None for decision in result.control_decisions)
    assert len(result.final_memory.provenance) == 2
    assert result.work.evaluation_memory_advance_count == 0
    assert all(
        counters == cohort._EXPECTED_INTERVENTION_COUNTERS
        for counters in result.authority_work_counters
    )

    document = json.loads(destination.read_text(encoding="ascii"))
    assert document["terminal_status"] == cohort.STATUS_PASS
    assert document["result_sha256"] == result.sha256
    assert len(document["full_authenticated_transcript"]["selected_attempts"]) == 3
    assert reservation.is_file()


def test_public_output_mismatch_is_retained_as_rooted_behavior_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RustSelectedActionExecutionV2,
        "authenticate_structure",
        lambda self: None,
    )
    destination = tmp_path / "c5-fail.result.json"
    reservation = tmp_path / "c5-fail.result.json.reservation.json"
    rustc_identity_sha256 = _root("behavior-fail-rustc")
    diagnostics = _FakeDiagnosticCompiler()
    executor = _FakeSelectedExecutor(rustc_identity_sha256=rustc_identity_sha256)
    expected_by_role = dict(zip(cohort.ROLES, cohort.EXPECTED_OUTPUTS, strict=True))

    def wrong_output_feedback(role: str) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(value + 1 for value in row) for row in expected_by_role[role])

    with pytest.raises(
        cohort.C5BehaviorFailure,
        match="selected candidate failed public output feedback",
    ):
        cohort._run_and_write_at_paths_for_test_v1(
            preregistration_commit=PREREGISTRATION_COMMIT,
            rustc_identity_sha256=rustc_identity_sha256,
            destination=destination,
            reservation_path=reservation,
            diagnostic_compile=diagnostics,
            execute=executor,
            output_feedback=wrong_output_feedback,
        )

    retained = json.loads(destination.read_text(encoding="ascii"))
    partial_transcript = retained["partial_transcript"]
    assert retained["terminal_status"] == cohort.STATUS_BEHAVIOR_FAIL
    assert retained["failure_type"] == "C5BehaviorFailure"
    assert partial_transcript["stage"] == cohort.ROLES[0]
    assert partial_transcript["selected_attempt"]["consequence"]["passed_cell"] == 0
    assert retained["partial_transcript_sha256"] == cohort._digest(
        cohort._FAILURE_NAMESPACE,
        partial_transcript,
    )
    assert len(diagnostics.source_sha256s) == len(executor.source_sha256s) == 1
    assert reservation.is_file()


def test_preflight_precedes_reservation_and_post_reservation_abort_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "production.result.json"
    reservation = tmp_path / "production.result.json.reservation.json"
    monkeypatch.setattr(
        cohort,
        "_production_evidence_paths_v1",
        lambda: (destination, reservation),
    )
    monkeypatch.setattr(
        cohort,
        "_preflight_rustc_v1",
        lambda: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )
    with pytest.raises(RuntimeError, match="preflight failed"):
        cohort.run_and_write_cohort_v1(preregistration_commit=PREREGISTRATION_COMMIT)
    assert not destination.exists()
    assert not reservation.exists()

    def fail_after_reservation(_identity: str, _source: str):
        assert reservation.is_file()
        assert not destination.exists()
        raise RuntimeError("diagnostic failed after reservation")

    with pytest.raises(RuntimeError, match="diagnostic failed after reservation"):
        cohort._run_and_write_at_paths_for_test_v1(
            preregistration_commit=PREREGISTRATION_COMMIT,
            rustc_identity_sha256=_root("post-reservation-rustc"),
            destination=destination,
            reservation_path=reservation,
            diagnostic_compile=fail_after_reservation,
            execute=lambda *_: (_ for _ in ()).throw(AssertionError("not reached")),
            output_feedback=lambda _: (_ for _ in ()).throw(AssertionError("not reached")),
        )
    retained = json.loads(destination.read_text(encoding="ascii"))
    assert retained["terminal_status"] == cohort.STATUS_EXECUTION_ABORT
    assert retained["error_type"] == "RuntimeError"
    assert reservation.is_file()
