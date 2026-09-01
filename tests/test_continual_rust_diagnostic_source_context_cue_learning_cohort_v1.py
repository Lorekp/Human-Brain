from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import (
    common_rustc_message_line_unordered_cue_text_edit_authority_v1 as authority,
)
from experiments.r3_pilot import (
    continual_rust_diagnostic_source_context_cue_learning_cohort_v1 as cohort,
)
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

PREREGISTRATION_COMMIT = "1" * 40
EXPECTED_SPEC_SHA256 = "6ebf21e45798066f2a0e6f57d12571cdf15b5b12cba859b7e4fd28c7febb2c97"


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


class _FakeDiagnosticCompiler:
    def __init__(self) -> None:
        self.source_sha256s: list[str] = []

    def __call__(
        self,
        rustc_identity_sha256: str,
        source: str,
    ) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes]:
        episode = cohort.ORIGINAL_SOURCES.index(source)
        code = cohort.EXPECTED_DIAGNOSTICS[episode]
        byte_start, byte_end = cohort.EXPECTED_PRIMARY_BYTE_RANGES[episode]
        assert source[byte_start:byte_end] == cohort.EXPECTED_PRIMARY_TEXTS[episode]
        diagnostic = {
            "$message_type": "diagnostic",
            "message": cohort.EXPECTED_PARENT_MESSAGES[episode],
            "code": {"code": code, "explanation": "excluded explanation"},
            "level": "error",
            "spans": [
                {
                    "file_name": cohort.SOURCE_FILE_NAME,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "is_primary": True,
                    "label": "excluded child-like label suggests the answer",
                    "suggested_replacement": "excluded replacement leaks the answer",
                }
            ],
            "children": [],
            "rendered": "excluded rendered compiler output",
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
        attempt = rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
            rustc_identity_sha256=rustc_identity_sha256,
            source_file_name=cohort.SOURCE_FILE_NAME,
            source=source.encode("ascii"),
            arguments=cohort.DIAGNOSTIC_ARGUMENTS,
            returncode=1,
            stdout=b"",
            stderr=stderr,
            compile_invocations=1,
        )
        return attempt, stderr


class _FakeSelectedExecutor:
    def __init__(self) -> None:
        self.pre_execute_work: list[tuple[int, int, int, int]] = []

    def __call__(self, prepared, decision, batch) -> RustSelectedActionExecutionV2:
        work = prepared.work_counters
        self.pre_execute_work.append(
            (
                work.materialization_count,
                work.release_count,
                work.execution_attempt_count,
                work.successful_execution_count,
            )
        )
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


def test_c10_spec_freezes_message_line_fixture_and_exact_87_root_exclusion() -> None:
    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate()

    assert spec.sha256 == EXPECTED_SPEC_SHA256
    assert spec.frozen_mechanism_commit == "4ba0cf5caedd48b02bf5afdcb1fe0962cb439270"
    assert len(spec.frozen_mechanism_file_sha256s) == 12
    assert tuple(map(len, cohort.ORIGINAL_SOURCES)) == cohort.ORIGINAL_SOURCE_SIZES
    assert all(not source.endswith("\n") for source in cohort.ORIGINAL_SOURCES)
    assert (
        tuple(sha256(source.encode("ascii")).hexdigest() for source in cohort.ORIGINAL_SOURCES)
        == cohort.ORIGINAL_SOURCE_SHA256S
    )
    assert len(spec.excluded_source_sha256s) == len(set(spec.excluded_source_sha256s)) == 87
    assert not set(spec.original_source_sha256s).intersection(spec.excluded_source_sha256s)
    assert not set(cohort.EXPECTED_MATERIALIZED_SOURCE_SHA256S).intersection(
        {*spec.original_source_sha256s, *spec.excluded_source_sha256s}
    )
    assert spec.expected_message_tokens == cohort.EXPECTED_MESSAGE_TOKENS
    assert spec.expected_source_line_tokens == cohort.EXPECTED_SOURCE_LINE_TOKENS
    assert spec.expected_merged_cue_counts == (12, 7, 12, 7, 12, 7)
    assert spec.expected_evaluation_supported_cue_intersections == (
        ("cannot", "find", "in", "map", "scope", "this", "value", "vescar"),
        ("map", "mismatched", "orlune", "types"),
    )
    merged_tokens = tuple(
        {*message_tokens, *source_line_tokens}
        for message_tokens, source_line_tokens in zip(
            spec.expected_message_tokens,
            spec.expected_source_line_tokens,
            strict=True,
        )
    )
    trained_tokens = set().union(*merged_tokens[: cohort.TRAIN_COUNT])
    evaluation_only_tokens = set().union(*merged_tokens[cohort.TRAIN_COUNT :]) - trained_tokens
    assert len(evaluation_only_tokens) == 7
    assert set(spec.evaluation_identifier_tokens).issubset(evaluation_only_tokens)
    assert spec.expected_evaluation_identifier_support_ticks == 0
    assert spec.expected_score_matrices == (
        ("E0425", 50, 46, 26, 22),
        ("E0308", 26, 22, 18, 14),
    )
    assert dict(spec.expected_work_counts)["evaluation_memory_advance_count"] == 0
    assert dict(spec.expected_work_counts)["alternative_materialization_count"] == 0


def test_full_grid_learns_message_line_bindings_and_executes_six_selected_sources(
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
        stem="c10-pass",
    )
    result.authenticate()

    assert result.status == cohort.STATUS_PASS
    assert tuple(diagnostics.source_sha256s) == cohort.ORIGINAL_SOURCE_SHA256S
    assert tuple(item.message_tokens for item in result.message_line_envelopes) == (
        cohort.EXPECTED_MESSAGE_TOKENS
    )
    assert tuple(item.source_line_tokens for item in result.message_line_envelopes) == (
        cohort.EXPECTED_SOURCE_LINE_TOKENS
    )
    assert (
        tuple(len(item.observation.cue_sha256s) for item in result.message_line_envelopes)
        == cohort.EXPECTED_MERGED_CUE_COUNTS
    )
    assert all(
        set(item.observation.cue_sha256s)
        == {*item.message_cue_sha256s, *item.source_line_cue_sha256s}
        for item in result.message_line_envelopes
    )
    assert tuple(selected_roots) == cohort.EXPECTED_MATERIALIZED_SOURCE_SHA256S
    assert executor.pre_execute_work == [(0, 0, 0, 0)] * 6
    assert feedback_open_counts == [1, 2, 3, 4, 6]
    assert (
        tuple(item.selected_hypothesis_sha256 for item in result.adaptation_normal_decisions)
        == cohort.EXPECTED_NORMAL_TRAIN_SELECTIONS
    )
    assert all(
        item.status.startswith("STOP_") and item.selected_hypothesis_sha256 is None
        for item in result.parent_evaluation_decisions
    )
    assert tuple(
        item.selected_hypothesis_sha256 for item in result.diagonal_evaluation_decisions
    ) == (cohort.H00_SHA256, cohort.H11_SHA256)
    assert tuple(
        item.selected_hypothesis_sha256 for item in result.three_lesson_evaluation_decisions
    ) == (cohort.H01_SHA256, None)
    assert tuple(item.selected_hypothesis_sha256 for item in result.evaluation_decisions) == (
        cohort.H01_SHA256,
        cohort.H10_SHA256,
    )
    assert all(item.selected_hypothesis_sha256 is None for item in result.control_decisions)
    assert (
        tuple(
            tuple(sorted(cohort._decision_scores(item).items()))
            for item in result.evaluation_decisions
        )
        == cohort.EXPECTED_FINAL_SCORE_MAPS
    )
    assert tuple(
        tuple(
            sorted(
                zip(
                    item.action_decision.activation.active_action_sha256s,
                    item.action_decision.activation.base_scores,
                    strict=True,
                )
            )
        )
        for item in result.evaluation_decisions
    ) == (
        tuple(
            sorted(
                (
                    (cohort.H00_SHA256, 32),
                    (cohort.H01_SHA256, 36),
                    (cohort.H10_SHA256, 8),
                    (cohort.H11_SHA256, 12),
                )
            )
        ),
        tuple(
            sorted(
                (
                    (cohort.H00_SHA256, 18),
                    (cohort.H01_SHA256, 14),
                    (cohort.H10_SHA256, 26),
                    (cohort.H11_SHA256, 22),
                )
            )
        ),
    )
    assert len(result.final_memory.provenance) == 4
    assert len(result.final_memory.binding_evidence) == 100
    assert result.evaluation_checkpoint_before_sha256 == (result.evaluation_checkpoint_after_sha256)
    assert result.work.memory_advance_count == 4
    assert result.work.evaluation_memory_advance_count == 0
    assert result.work.evaluation_execution_before_oracle_count == 2
    assert result.work.alternative_materialization_count == 0

    trained_cues = set().union(
        *(set(item.observation.cue_sha256s) for item in result.message_line_envelopes[:4])
    )
    evaluation_cues = set().union(
        *(set(item.observation.cue_sha256s) for item in result.message_line_envelopes[4:])
    )
    evaluation_only_cues = evaluation_cues - trained_cues
    assert len(evaluation_only_cues) == 7
    assert evaluation_only_cues.isdisjoint(trained_cues)
    assert all(
        row.cue_sha256 not in evaluation_only_cues for row in result.final_memory.binding_evidence
    )

    document = json.loads(destination.read_text(encoding="ascii"))
    record = document["authenticated_result_record"]
    assert document["terminal_status"] == cohort.STATUS_PASS
    assert document["result_sha256"] == result.sha256
    assert len(record["message_line_envelopes"]) == 6
    assert record["work"]["evaluation_oracle_open_count"] == 1
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
