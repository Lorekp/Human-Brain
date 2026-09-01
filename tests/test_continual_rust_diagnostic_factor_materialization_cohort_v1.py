from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as adapter
from experiments.r3_pilot import (
    continual_rust_diagnostic_factor_materialization_cohort_v1 as cohort,
)
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

PREREGISTRATION_COMMIT = "1" * 40
EXPECTED_SPEC_SHA256 = "b514ff417680b0a7238e705b9cfa110bf1b9c94c1b32f16a3ad084e64792d394"


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


_PRIMARY_TOKENS = (
    "_c6_train_a_missing",
    "_c6_train_b_wrong_value",
    "_c6_eval_c_missing",
)


class _FakeDiagnosticCompiler:
    def __init__(self) -> None:
        self.source_sha256s: list[str] = []

    def __call__(
        self,
        rustc_identity_sha256: str,
        source: str,
    ) -> adapter.RustcDiagnosticCompileAttemptV1:
        episode = cohort.ORIGINAL_SOURCES.index(source)
        code, _region = cohort.EXPECTED_DIAGNOSTICS[episode]
        token = _PRIMARY_TOKENS[episode]
        byte_start, byte_end = cohort.EXPECTED_PRIMARY_BYTE_RANGES[episode]
        assert source[byte_start:byte_end] == token
        diagnostic = {
            "$message_type": "diagnostic",
            "message": "repair-bearing diagnostic text is outside cognition",
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
        self.pre_execute_counts: list[tuple[int, int]] = []
        self.callback_counts: list[tuple[int, int]] = []
        self.source_sha256s: list[str] = []

    def __call__(self, prepared, decision, batch) -> RustSelectedActionExecutionV2:
        self.pre_execute_counts.append((prepared.materialization_count, prepared.release_count))
        episode = cohort.ORIGINAL_SOURCE_SHA256S.index(prepared.receipt.source_sha256)
        expected_hypotheses = (
            cohort.H00_SHA256,
            cohort.H11_SHA256,
            cohort.H01_SHA256,
        )
        expected_hypothesis = expected_hypotheses[episode]

        def execute_source(selected_source):
            self.callback_counts.append((prepared.materialization_count, prepared.release_count))
            self.source_sha256s.append(selected_source.source_sha256)
            outputs = (
                cohort.EXPECTED_OUTPUTS[episode]
                if decision.selected_hypothesis_sha256 == expected_hypothesis
                else ((-999,),)
            )
            return RustSelectedActionExecutionV2(
                selected_source=selected_source,
                rustc_identity=SimpleNamespace(sha256=self.rustc_identity_sha256),
                preparation=SimpleNamespace(batch_sha256=_root(f"batch:{batch!r}")),
                dependency_receipt=SimpleNamespace(),
                compile_receipt=SimpleNamespace(compile_invocations=1),
                run_receipt=SimpleNamespace(run_invocations=1),
                outputs=outputs,
                sha256=_root(
                    f"execution:{len(self.source_sha256s)}:{selected_source.source_sha256}"
                ),
            )

        return prepared.execute_selected(decision, execute_source)


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_feedback,
    stem: str,
):
    monkeypatch.setattr(
        RustSelectedActionExecutionV2,
        "authenticate_structure",
        lambda self: None,
    )
    destination = tmp_path / f"{stem}.result.json"
    reservation = tmp_path / f"{stem}.result.json.reservation.json"
    rustc_identity_sha256 = _root(f"{stem}:rustc")
    diagnostics = _FakeDiagnosticCompiler()
    executor = _FakeSelectedExecutor(rustc_identity_sha256=rustc_identity_sha256)
    result = cohort._run_and_write_at_paths_for_test_v1(
        preregistration_commit=PREREGISTRATION_COMMIT,
        rustc_identity_sha256=rustc_identity_sha256,
        destination=destination,
        reservation_path=reservation,
        diagnostic_compile=diagnostics,
        execute=executor,
        output_feedback=output_feedback(executor),
    )
    return result, destination, reservation, diagnostics, executor


def test_c6_spec_freezes_originals_bindings_and_51_prior_roots_only() -> None:
    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate()
    record = spec.record()

    assert spec.sha256 == EXPECTED_SPEC_SHA256
    assert len(spec.original_source_sha256s) == len(set(spec.original_source_sha256s)) == 3
    assert len(spec.excluded_source_sha256s) == len(set(spec.excluded_source_sha256s)) == 51
    assert not set(spec.original_source_sha256s).intersection(spec.excluded_source_sha256s)
    assert len(spec.hypothesis_sha256s) == len(set(spec.hypothesis_sha256s)) == 4
    assert spec.allow_exploration_by_episode == (True, True, False)
    assert spec.expected_primary_byte_ranges == ((190, 209), (383, 406), (331, 349))
    assert not any("candidate" in key or "materialized" in key for key in record)


def test_two_lessons_select_before_three_exact_materializations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback_open_counts: list[int] = []
    expected_by_role = dict(zip(cohort.ROLES, cohort.EXPECTED_OUTPUTS, strict=True))

    def feedback_factory(executor: _FakeSelectedExecutor):
        def output_feedback(role: str) -> tuple[tuple[int, ...], ...]:
            feedback_open_counts.append(len(executor.source_sha256s))
            return expected_by_role[role]

        return output_feedback

    result, destination, reservation, diagnostics, executor = _run(
        tmp_path,
        monkeypatch,
        output_feedback=feedback_factory,
        stem="c6-pass",
    )
    result.authenticate()

    materialized_roots = tuple(
        item.materialization.materialized_source_sha256 for item in result.selected_attempts
    )
    assert result.status == cohort.STATUS_PASS
    assert tuple(diagnostics.source_sha256s) == cohort.ORIGINAL_SOURCE_SHA256S
    assert tuple(executor.source_sha256s) == materialized_roots
    assert executor.pre_execute_counts == [(0, 0)] * 3
    assert executor.callback_counts == [(1, 1)] * 3
    assert feedback_open_counts == [1, 2, 3]
    assert len(set(materialized_roots)) == 3
    assert not set(materialized_roots).intersection(cohort.ORIGINAL_SOURCE_SHA256S)
    assert not set(materialized_roots).intersection(cohort.EXCLUDED_PRIOR_SOURCE_SHA256S)
    assert tuple(item.selected_hypothesis_sha256 for item in result.selected_attempts) == (
        cohort.H00_SHA256,
        cohort.H11_SHA256,
        cohort.H01_SHA256,
    )
    assert tuple(item.authority_receipt.allow_exploration for item in result.selected_attempts) == (
        True,
        True,
        False,
    )
    assert tuple(
        decision.selected_hypothesis_sha256 for decision in result.evaluation_decisions
    ) == (None, None, cohort.H01_SHA256)
    assert all(decision.selected_hypothesis_sha256 is None for decision in result.control_decisions)
    assert len(result.final_memory.provenance) == 2
    assert result.work.materialization_count == result.work.release_count == 3
    assert result.work.alternative_materialization_count == 0
    assert result.work.evaluation_memory_advance_count == 0

    document = json.loads(destination.read_text(encoding="ascii"))
    assert document["terminal_status"] == cohort.STATUS_PASS
    assert document["result_sha256"] == result.sha256
    assert len(document["full_authenticated_transcript"]["selected_attempts"]) == 3
    assert reservation.is_file()


def test_public_output_mismatch_retains_rooted_behavior_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RustSelectedActionExecutionV2,
        "authenticate_structure",
        lambda self: None,
    )
    destination = tmp_path / "c6-fail.result.json"
    reservation = tmp_path / "c6-fail.result.json.reservation.json"
    rustc_identity_sha256 = _root("c6-fail:rustc")
    diagnostics = _FakeDiagnosticCompiler()
    executor = _FakeSelectedExecutor(rustc_identity_sha256=rustc_identity_sha256)
    expected_by_role = dict(zip(cohort.ROLES, cohort.EXPECTED_OUTPUTS, strict=True))

    def wrong_output_feedback(role: str) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(value + 1 for value in row) for row in expected_by_role[role])

    with pytest.raises(
        cohort.C6BehaviorFailure,
        match="selected materialization failed public output feedback",
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
    partial = retained["partial_transcript"]
    assert retained["terminal_status"] == cohort.STATUS_BEHAVIOR_FAIL
    assert retained["failure_type"] == "C6BehaviorFailure"
    assert partial["stage"] == cohort.ROLES[0]
    assert partial["selected_attempt"]["attempt"]["pre_execute_counts"] == [0, 0]
    assert partial["selected_attempt"]["attempt"]["post_grant_counts"] == [1, 1]
    assert partial["selected_attempt"]["consequence"]["passed_cell"] == 0
    assert retained["partial_transcript_sha256"] == cohort._digest(
        cohort._FAILURE_NAMESPACE,
        partial,
    )
    assert len(diagnostics.source_sha256s) == len(executor.source_sha256s) == 1
    assert reservation.is_file()


def test_preflight_then_post_reservation_abort_and_one_shot_exclusivity(
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

    call = dict(
        preregistration_commit=PREREGISTRATION_COMMIT,
        rustc_identity_sha256=_root("post-reservation-rustc"),
        destination=destination,
        reservation_path=reservation,
        diagnostic_compile=fail_after_reservation,
        execute=lambda *_: (_ for _ in ()).throw(AssertionError("not reached")),
        output_feedback=lambda _: (_ for _ in ()).throw(AssertionError("not reached")),
    )
    with pytest.raises(RuntimeError, match="diagnostic failed after reservation"):
        cohort._run_and_write_at_paths_for_test_v1(**call)
    retained_bytes = destination.read_bytes()
    retained = json.loads(retained_bytes.decode("ascii"))
    assert retained["terminal_status"] == cohort.STATUS_EXECUTION_ABORT
    assert retained["error_type"] == "RuntimeError"
    assert reservation.is_file()

    with pytest.raises(FileExistsError):
        cohort._run_and_write_at_paths_for_test_v1(**call)
    assert destination.read_bytes() == retained_bytes
