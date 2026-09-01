from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_factorized_text_edit_materializer_v1 as materializer
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import (
    continual_rust_diagnostic_occurrence_localization_c11_v1 as predecessor,
)
from experiments.r3_pilot import (
    continual_rust_diagnostic_occurrence_localization_c11b_v1 as cohort,
)
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

PREREGISTRATION_COMMIT = "1" * 40
_TARGET_VALUES = ((61, 37), (43, 67), (71, 47), (53, 73), (79, 59), (89, 83))


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
            "code": {"code": code, "explanation": "excluded"},
            "level": "error",
            "spans": [
                {
                    "file_name": cohort.SOURCE_FILE_NAME,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "is_primary": True,
                    "label": "excluded parent label",
                    "suggested_replacement": "excluded replacement",
                }
            ],
            "children": [
                {
                    "message": "excluded child suggestion",
                    "code": None,
                    "level": "help",
                    "spans": [],
                    "children": [],
                    "rendered": "excluded rendered child",
                }
            ],
            "rendered": "excluded rendered parent",
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
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
            for row in (diagnostic, summary)
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


def _install_fake_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rustc_identity_sha256: str,
) -> list[tuple[int, int, str]]:
    selected: list[tuple[int, int, str]] = []
    interventions = cohort.authority.delegate_authority.interventions
    monkeypatch.setattr(RustSelectedActionExecutionV2, "authenticate_structure", lambda self: None)

    def prepare_execution(authority_sha256: str, batch):
        return SimpleNamespace(
            authority_sha256=authority_sha256,
            batch=batch,
            batch_sha256=_root(f"batch:{batch!r}"),
        )

    monkeypatch.setattr(interventions, "_prepare_execution", prepare_execution)
    monkeypatch.setattr(
        interventions,
        "_authenticate_execution_preparation",
        lambda *_args, **_kwargs: None,
    )

    def execute_prepared(selected_source, prepared, *, authenticate_current):
        authenticate_current()
        source = selected_source.source
        episode = next(
            index
            for index, primary in enumerate(cohort.EXPECTED_PRIMARY_TEXTS)
            if primary in source
        )
        roster = cohort.occurrences.project_preceding_let_declaration_occurrences_v1(
            cohort.ORIGINAL_SOURCES[episode],
            primary_byte_start=cohort.EXPECTED_PRIMARY_BYTE_RANGES[episode][0],
            primary_byte_end=cohort.EXPECTED_PRIMARY_BYTE_RANGES[episode][1],
        )
        context = cohort.CONTEXT_WORDS[episode]
        primary = cohort.EXPECTED_PRIMARY_TEXTS[episode]
        relation_index = next(
            index
            for index, item in enumerate(roster.occurrences)
            if (
                f"let {primary} = {item.identifier};" in source
                or f"{context}({item.identifier})" in source
            )
        )
        value = _TARGET_VALUES[episode][relation_index]
        outputs = tuple(tuple(cell + value for cell in row) for row in prepared.batch)
        selected.append((episode, relation_index, selected_source.source_sha256))
        return RustSelectedActionExecutionV2(
            selected_source=selected_source,
            rustc_identity=SimpleNamespace(sha256=rustc_identity_sha256),
            preparation=prepared,
            dependency_receipt=SimpleNamespace(),
            compile_receipt=SimpleNamespace(compile_invocations=1),
            run_receipt=SimpleNamespace(run_invocations=1),
            outputs=outputs,
            sha256=_root(f"execution:{episode}:{relation_index}:{selected_source.source_sha256}"),
        )

    monkeypatch.setattr(interventions, "_execute_prepared", execute_prepared)
    return selected


def test_c11b_spec_freezes_fresh_exact_two_occurrence_cohort() -> None:
    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate(require_frozen_commit=True)
    cohort._authenticate_malformed_roster_controls_v1()

    assert spec.frozen_mechanism_commit == "1ff07c2b5458cc5860cecbf02afc5ab8e62bff3c"
    assert len(spec.frozen_mechanism_file_sha256s) == 15
    assert len(spec.excluded_source_sha256s) == 143
    assert not set(spec.original_source_sha256s).intersection(spec.excluded_source_sha256s)
    assert spec.train_operator_indexes == (0, 1, 0, 1)
    assert spec.train_relation_order_indexes == ((0, 1), (1, 0), (0, 1), (1, 0))
    assert not any("target" in field.name for field in cohort.dataclasses.fields(spec))

    rosters = tuple(
        cohort.occurrences.project_preceding_let_declaration_occurrences_v1(
            source,
            primary_byte_start=byte_range[0],
            primary_byte_end=byte_range[1],
        )
        for source, byte_range in zip(
            cohort.ORIGINAL_SOURCES, cohort.EXPECTED_PRIMARY_BYTE_RANGES, strict=True
        )
    )
    assert all(
        tuple(item.relation for item in roster.occurrences) == cohort.RELATIONS
        for roster in rosters
    )
    assert sum(len(item.occurrences) for item in rosters) == 12

    predecessor_protocol = (
        (0, 0, 0),
        (0, 0, 1),
        (1, 1, 1),
        (1, 1, 0),
        (2, 0, 0),
        (2, 0, 1),
        (3, 1, 1),
        (3, 1, 0),
        (4, 0, 1),
        (5, 1, 0),
    )
    predecessor_materialized_roots: list[str] = []
    operation_bindings = dict(predecessor.OPERATOR_BINDINGS)
    for episode, operator_index, relation_index in predecessor_protocol:
        source = predecessor.ORIGINAL_SOURCES[episode]
        source_bytes = source.encode("ascii")
        byte_start, byte_end = predecessor.EXPECTED_PRIMARY_BYTE_RANGES[episode]
        roster = predecessor.occurrences.project_preceding_let_declaration_occurrences_v1(
            source,
            primary_byte_start=byte_start,
            primary_byte_end=byte_end,
        )
        edit = materializer._compose_edit(
            source=source_bytes,
            primary_span=predecessor.rustc_feedback._seal_span(
                file_name=predecessor.SOURCE_FILE_NAME,
                byte_start=byte_start,
                byte_end=byte_end,
            ),
            operation=operation_bindings[
                predecessor.OPERATOR_ACTION_SHA256S[operator_index]
            ],
            marker=roster.occurrences[relation_index].identifier,
        )
        edit_start, edit_end = edit.byte_range
        materialized = (
            source_bytes[:edit_start]
            + edit.new_text.encode("ascii")
            + source_bytes[edit_end:]
        )
        predecessor_materialized_roots.append(sha256(materialized).hexdigest())

    assert tuple(predecessor_materialized_roots) == (
        "043654a72e840728743e816026e3e2168472c58d99dace9f92486cdbb9dccf4c",
        "41d88c29b99af965637d6fec963fb177f79bfc8237fee36fc9aa734553617b54",
        "78b547d8925c6f9e5089fd98b15b68c1d770cc57550218b6844013348a515237",
        "07f4e3f8f6e1e932319146df507252815797acec809ba3a477b0f50604cfa3e6",
        "0e64010a82f448322b8c42ed6e73bee82391b0d069a5e6ff349935e4fe1ccf47",
        "54a4a279c62c5dfb2c5c1ffb5bd7944ec8ac56ff3fb4dbd2eb9223e0ef8452bd",
        "15475c869b7c5dbf90030264145f59d462fa66d5f49c3f17543d12270d55cb87",
        "cc9cbd482f93a7a2c761bb22b01c121ee9670f49cdaea0242654af997e6766ff",
        "511caf6c304c7f64101af4b868e49360067bfa7a6eed2fc20b5c68b37d3b68d2",
        "5bce3fd6d5e55d66da2bb7f55adb0fc6d9ef1c66d17c639ab88d2e5d587ed057",
    )
    assert set(predecessor_materialized_roots).issubset(spec.excluded_source_sha256s)
    assert {
        "099d23abb6fd2cf68894b9b0739b5a134b4be172368e1938b2577f69e6c8caf9",
        "1501ca341da74b246e1a59e006c56ed2a7251f39e7c78ae7c8ed36ed047d5e66",
        "205b3934fe85ed23676639dfcba907f0d81522c02bf9c95c4d32f7a4d71511b3",
        "77b0f6c81f38a4edc68f9ab5212bf281fa7c08576c6241937fb1c6d8836cfaa5",
        "7963786ba604c2da0ab21f8d9bbb032436c9abfff71b964b6a1c1e5da0613bd4",
        "991c87864d821cefcf10e2a9dd9a383540cb1597d36b5ccbd939058567d482c0",
        "180ff462710fde6978682cf9de58c2e412645d36e616959fbf66447b1532f7b2",
        "6255543848aa1646402de9eb7a9780a5dd67c67f95400a27a59118292c691076",
        "6745cd76302ab4b9f8f0d393783804604ea14c0824b3e5f3abffd854056b5937",
        "7bf81394175578ae1ccd2ac1aa8a0e63d008498b5ee3705b33913a38055c62a3",
        "97ad17997290a4a4ec42f3721299ee937de50a70c57e8b0738a6cba5da579c9c",
        "c32cf1d3398cf09cdbba91529789049b67f9113a96e1ac3ee6e2889169b4ef8d",
        "c933d69fc571aebf24eaa5cba0255e6b5767a3316a4884337c44725f2519c56d",
        "d67a08288db55c84a4e7df58c68c9a62819d118c4f80aa4ff196606c21166046",
        "ed19d4db7c290cd81eb57f39e3cb818ca2a5f51770e3b9b8dc71252f9cf496df",
        "f190adc6241af291b5f64256eb813bbdb75c9d2bf6f6d354b70b5375751dd13b",
    }.issubset(spec.excluded_source_sha256s)


def test_pair_gate_rejects_zero_or_two_passes_before_advance() -> None:
    checkpoint = _root("checkpoint")
    for cells in ((0, 0), (1, 1)):
        with pytest.raises(cohort.C11BBehaviorFailure) as raised:
            cohort._unique_pair_pass_index(
                cells,
                episode_index=0,
                checkpoint_sha256=checkpoint,
            )
        assert raised.value.partial_transcript["memory_advance_count"] == 0


def test_c11b_fake_end_to_end_pairs_both_targets_then_evaluates_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rustc_identity_sha256 = _root("c11b-fake-rustc")
    selected = _install_fake_execution(
        monkeypatch,
        rustc_identity_sha256=rustc_identity_sha256,
    )
    diagnostics = _FakeDiagnosticCompiler()
    destination = tmp_path / "c11b.result.json"
    reservation_path = tmp_path / "c11b.result.json.reservation.json"
    expected_by_role = dict(
        zip(cohort.TRAIN_FEEDBACK_ROLES, cohort.EXPECTED_OUTPUTS[:4], strict=True)
    )
    feedback_calls: list[str] = []

    def output_feedback(role: str):
        feedback_calls.append(role)
        if role == cohort.EVALUATION_ORACLE_ROLE:
            return cohort.EXPECTED_OUTPUTS[4:]
        return expected_by_role[role]

    result = cohort._run_and_write_at_paths_for_test_v1(
        preregistration_commit=PREREGISTRATION_COMMIT,
        rustc_identity_sha256=rustc_identity_sha256,
        destination=destination,
        reservation_path=reservation_path,
        diagnostic_compile=diagnostics,
        execute=lambda prepared, decision, batch: prepared.execute_selected(decision, batch),
        output_feedback=output_feedback,
    )

    assert result.status == cohort.STATUS_PASS
    assert diagnostics.source_sha256s == list(cohort.ORIGINAL_SOURCE_SHA256S)
    assert feedback_calls == [*cohort.TRAIN_FEEDBACK_ROLES, cohort.EVALUATION_ORACLE_ROLE]
    assert [item[:2] for item in selected[:8]] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 1),
        (3, 0),
    ]
    assert [item[:2] for item in selected[8:]] == [(4, 1), (5, 0)]
    assert tuple(
        tuple(item.consequence.passed_cell for item in pair)
        for pair in result.training_attempt_pairs
    ) == ((1, 0), (1, 0), (0, 1), (0, 1))
    assert result.passing_outcome_indexes == (0, 0, 1, 1)
    assert tuple(
        item.selected_hypothesis_sha256 for item in result.evaluation_decisions
    ) == cohort.EVALUATION_HYPOTHESIS_SHA256S
    assert all(item.selected_hypothesis_sha256 is None for item in result.control_decisions)
    assert result.work.record(include_sha256=False) == dict(cohort.EXPECTED_WORK_COUNTS)
    assert destination.exists() and reservation_path.exists()

    original_attempt = result.training_attempt_pairs[0][0]
    tampered_outputs = ((999, 1000, 1001),)
    tampered_consequence = replace(
        original_attempt.consequence,
        expected_outputs=tampered_outputs,
        passed_cell=int(original_attempt.consequence.observed_outputs == tampered_outputs),
        sha256="",
    )
    tampered_consequence = replace(
        tampered_consequence,
        sha256=cohort._digest(
            cohort._CONSEQUENCE_NAMESPACE,
            tampered_consequence.record(include_sha256=False),
        ),
    )
    tampered_attempt = replace(
        original_attempt,
        consequence=tampered_consequence,
        sha256="",
    )
    tampered_attempt = replace(
        tampered_attempt,
        sha256=cohort._digest(
            cohort._ATTEMPT_NAMESPACE,
            tampered_attempt.record(include_sha256=False),
        ),
    )
    first_pair = (tampered_attempt, result.training_attempt_pairs[0][1])
    tampered_result = replace(
        result,
        training_attempt_pairs=(first_pair, *result.training_attempt_pairs[1:]),
        sha256="",
    )
    tampered_result = replace(
        tampered_result,
        sha256=cohort._digest(
            cohort._RESULT_NAMESPACE,
            tampered_result.record(include_sha256=False),
        ),
    )
    with pytest.raises(
        cohort.ContinualRustDiagnosticOccurrenceLocalizationC11BError,
        match="TRAIN output",
    ):
        tampered_result.authenticate()
