from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from experiments.r3_pilot import common_unordered_cue_factor_text_edit_authority_v1 as authority
from experiments.r3_pilot import (
    continual_rust_diagnostic_unordered_cue_binding_cohort_v1 as cohort,
)
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

PREREGISTRATION_COMMIT = "1" * 40
EXPECTED_SPEC_SHA256 = "a6372e986cec76b04e9e9efd412cc6e3ae5c1ba862566801acf2f8c02d3f9fda"


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


_PRIMARY_TOKENS = (
    "c8_t00_gap",
    "c8_t11_text",
    "c8_t01_gap",
    "c8_t10_text",
    "c8_e01_gap",
    "c8_e10_text",
)


class _FakeDiagnosticCompiler:
    def __init__(self) -> None:
        self.source_sha256s: list[str] = []

    def __call__(
        self,
        rustc_identity_sha256: str,
        source: str,
    ) -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
        episode = cohort.ORIGINAL_SOURCES.index(source)
        code, _region = cohort.EXPECTED_DIAGNOSTICS[episode]
        byte_start, byte_end = cohort.EXPECTED_PRIMARY_BYTE_RANGES[episode]
        assert source[byte_start:byte_end] == _PRIMARY_TOKENS[episode]
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
        return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
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
    def __init__(self) -> None:
        self.pre_execute_counts: list[tuple[int, int]] = []

    def __call__(self, prepared, decision, batch) -> RustSelectedActionExecutionV2:
        self.pre_execute_counts.append((prepared.materialization_count, prepared.release_count))
        return prepared.execute_selected(decision, batch)


def _install_fake_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rustc_identity_sha256: str,
) -> list[str]:
    selected_roots: list[str] = []
    monkeypatch.setattr(RustSelectedActionExecutionV2, "authenticate_structure", lambda self: None)

    def prepare_execution(authority_sha256: str, batch):
        return SimpleNamespace(
            authority_sha256=authority_sha256,
            batch=batch,
            batch_sha256=_root(f"batch:{batch!r}"),
        )

    monkeypatch.setattr(authority.interventions, "_prepare_execution", prepare_execution)
    monkeypatch.setattr(
        authority.interventions,
        "_authenticate_execution_preparation",
        lambda *_args, **_kwargs: None,
    )

    def execute_prepared(selected_source, prepared, *, authenticate_current):
        authenticate_current()
        selected_roots.append(selected_source.source_sha256)
        episode = cohort.EXPECTED_MATERIALIZED_SOURCE_SHA256S.index(selected_source.source_sha256)
        return RustSelectedActionExecutionV2(
            selected_source=selected_source,
            rustc_identity=SimpleNamespace(sha256=rustc_identity_sha256),
            preparation=prepared,
            dependency_receipt=SimpleNamespace(),
            compile_receipt=SimpleNamespace(compile_invocations=1),
            run_receipt=SimpleNamespace(run_invocations=1),
            outputs=cohort.EXPECTED_OUTPUTS[episode],
            sha256=_root(f"execution:{episode}:{selected_source.source_sha256}"),
        )

    monkeypatch.setattr(authority.interventions, "_execute_prepared", execute_prepared)
    return selected_roots


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_feedback,
    stem: str,
):
    destination = tmp_path / f"{stem}.result.json"
    reservation = tmp_path / f"{stem}.result.json.reservation.json"
    rustc_identity_sha256 = _root(f"{stem}:rustc")
    selected_roots = _install_fake_execution(
        monkeypatch,
        rustc_identity_sha256=rustc_identity_sha256,
    )
    diagnostics = _FakeDiagnosticCompiler()
    executor = _FakeSelectedExecutor()
    result = cohort._run_and_write_at_paths_for_test_v1(
        preregistration_commit=PREREGISTRATION_COMMIT,
        rustc_identity_sha256=rustc_identity_sha256,
        destination=destination,
        reservation_path=reservation,
        diagnostic_compile=diagnostics,
        execute=executor,
        output_feedback=output_feedback(selected_roots),
    )
    return result, destination, reservation, diagnostics, executor, selected_roots


def test_c8_spec_freezes_full_grid_and_exact_63_root_exclusion() -> None:
    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate()

    assert spec.sha256 == EXPECTED_SPEC_SHA256
    assert spec.frozen_mechanism_commit == "2eae160c645b1e5fb0be996b7d1d4bac7c862508"
    assert len(spec.original_source_sha256s) == len(set(spec.original_source_sha256s)) == 6
    assert len(spec.excluded_source_sha256s) == len(set(spec.excluded_source_sha256s)) == 63
    assert not set(spec.original_source_sha256s).intersection(spec.excluded_source_sha256s)
    assert not set(cohort.EXPECTED_MATERIALIZED_SOURCE_SHA256S).intersection(
        {*spec.original_source_sha256s, *spec.excluded_source_sha256s}
    )
    assert spec.exploration_hypothesis_sha256s == (
        cohort.H00_SHA256,
        cohort.H11_SHA256,
        cohort.H01_SHA256,
        cohort.H10_SHA256,
    )
    assert spec.evaluation_hypothesis_sha256s == (cohort.H01_SHA256, cohort.H10_SHA256)
    assert spec.allow_exploration_by_episode == (True, True, True, True, False, False)
    assert spec.expected_primary_byte_ranges == (
        (200, 210),
        (357, 368),
        (333, 343),
        (222, 233),
        (334, 344),
        (221, 232),
    )


def test_full_grid_learns_bindings_and_executes_six_selected_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback_open_counts: list[int] = []
    expected_by_role = dict(zip(cohort.ROLES, cohort.EXPECTED_OUTPUTS, strict=True))

    def feedback_factory(selected_roots: list[str]):
        def output_feedback(role: str) -> object:
            feedback_open_counts.append(len(selected_roots))
            if role == cohort.EVALUATION_ORACLE_ROLE:
                return cohort.EXPECTED_OUTPUTS[cohort.TRAIN_COUNT :]
            return expected_by_role[role]

        return output_feedback

    result, destination, reservation, diagnostics, executor, selected_roots = _run(
        tmp_path,
        monkeypatch,
        output_feedback=feedback_factory,
        stem="c8-pass",
    )
    result.authenticate()

    assert result.status == cohort.STATUS_PASS
    assert tuple(diagnostics.source_sha256s) == cohort.ORIGINAL_SOURCE_SHA256S
    assert tuple(selected_roots) == cohort.EXPECTED_MATERIALIZED_SOURCE_SHA256S
    assert executor.pre_execute_counts == [(0, 0)] * 6
    assert feedback_open_counts == [1, 2, 3, 4, 6]
    assert tuple(item.selected_hypothesis_sha256 for item in result.selected_attempts) == (
        cohort.H00_SHA256,
        cohort.H11_SHA256,
        cohort.H01_SHA256,
        cohort.H10_SHA256,
        cohort.H01_SHA256,
        cohort.H10_SHA256,
    )
    assert all(item.selected_hypothesis_sha256 is None for item in result.parent_cross_decisions)
    assert all(
        item.status == unordered.STOP_PHYSICAL_RACE and item.selected_hypothesis_sha256 is None
        for item in result.diagonal_cross_decisions
    )
    assert tuple(
        item.selected_hypothesis_sha256 for item in result.three_lesson_cross_decisions
    ) == (cohort.H01_SHA256, None)
    assert tuple(item.selected_hypothesis_sha256 for item in result.evaluation_decisions) == (
        cohort.H01_SHA256,
        cohort.H10_SHA256,
    )
    assert all(item.selected_hypothesis_sha256 is None for item in result.control_decisions)
    assert len(result.final_memory.provenance) == 4
    assert len(result.final_memory.binding_evidence) == 16
    assert result.work.memory_advance_count == 4
    assert result.work.evaluation_memory_advance_count == 0
    assert result.work.evaluation_oracle_open_count == 1
    assert result.work.alternative_materialization_count == 0

    evidence = {
        (item.cue_sha256, item.action_axis, item.component_sha256): (
            item.excitatory_ticks,
            item.inhibitory_ticks,
        )
        for item in result.final_memory.binding_evidence
    }
    assert evidence == cohort._expected_binding_evidence(result.observations)
    assert all(inhibitory == 0 for _excitatory, inhibitory in evidence.values())

    document = json.loads(destination.read_text(encoding="ascii"))
    assert document["terminal_status"] == cohort.STATUS_PASS
    assert document["result_sha256"] == result.sha256
    assert len(document["full_authenticated_transcript"]["selected_attempts"]) == 6
    assert reservation.is_file()


def test_output_mismatch_retains_rooted_behavior_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "c8-fail.result.json"
    reservation = tmp_path / "c8-fail.result.json.reservation.json"
    rustc_identity_sha256 = _root("c8-fail:rustc")
    _install_fake_execution(monkeypatch, rustc_identity_sha256=rustc_identity_sha256)
    expected_by_role = dict(zip(cohort.ROLES, cohort.EXPECTED_OUTPUTS, strict=True))

    def wrong_feedback(role: str) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(value + 1 for value in row) for row in expected_by_role[role])

    with pytest.raises(cohort.C8BehaviorFailure, match="failed public output feedback"):
        cohort._run_and_write_at_paths_for_test_v1(
            preregistration_commit=PREREGISTRATION_COMMIT,
            rustc_identity_sha256=rustc_identity_sha256,
            destination=destination,
            reservation_path=reservation,
            diagnostic_compile=_FakeDiagnosticCompiler(),
            execute=_FakeSelectedExecutor(),
            output_feedback=wrong_feedback,
        )
    retained = json.loads(destination.read_text(encoding="ascii"))
    assert retained["terminal_status"] == cohort.STATUS_BEHAVIOR_FAIL
    assert retained["partial_transcript"]["stage"] == cohort.ROLES[0]
    assert retained["partial_transcript"]["selected_attempt"]["consequence"]["passed_cell"] == 0
    assert reservation.is_file()


def test_preflight_and_one_shot_exclusivity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "production.result.json"
    reservation = tmp_path / "production.result.json.reservation.json"
    monkeypatch.setattr(cohort, "_production_evidence_paths_v1", lambda: (destination, reservation))
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
    assert json.loads(retained_bytes.decode("ascii"))["terminal_status"] == (
        cohort.STATUS_EXECUTION_ABORT
    )
    with pytest.raises(FileExistsError):
        cohort._run_and_write_at_paths_for_test_v1(**call)
    assert destination.read_bytes() == retained_bytes
