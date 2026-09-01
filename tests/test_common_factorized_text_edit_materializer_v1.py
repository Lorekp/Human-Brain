from __future__ import annotations

import json
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot.common_factorized_text_edit_materializer_v1 import (
    DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR,
    REPLACE_PRIMARY_SPAN_WITH_ANCHOR,
    FactorizedTextEditMaterializerError,
    prepare_factorized_text_edit_materializer_v1,
)
from humanbrain import rust_selected_action_executor as rust_executor

SOURCE_FILE = "case.rs"
SOURCE = """pub fn solve(_samples: &[i64]) -> Vec<i64> {
    let _first_anchor = 7_i64;
    let _second_anchor = 9_i64;
    vec![_missing_value]
}
"""
SELECTION_CONTEXT = sha256(b"factorized-edit-selection-context").hexdigest()


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


OPERATOR_ROOTS = (_root("operator:declare"), _root("operator:replace"))
TARGET_ROOTS = (_root("target:first"), _root("target:second"))
OPERATOR_BINDINGS = (
    (OPERATOR_ROOTS[0], DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR),
    (OPERATOR_ROOTS[1], REPLACE_PRIMARY_SPAN_WITH_ANCHOR),
)
TARGET_BINDINGS = (
    (TARGET_ROOTS[0], "_first_anchor"),
    (TARGET_ROOTS[1], "_second_anchor"),
)
BANK = tuple(
    factors.seal_factorized_hypothesis_v1((operator_root, target_root))
    for operator_root in OPERATOR_ROOTS
    for target_root in TARGET_ROOTS
)


def _attempt(*, message: str = "cannot find value", suggestion: str = "secret fix"):
    source_bytes = SOURCE.encode("ascii")
    byte_start = source_bytes.index(b"_missing_value")
    byte_end = byte_start + len(b"_missing_value")
    diagnostic = {
        "$message_type": "diagnostic",
        "message": message,
        "code": {"code": "E0425", "explanation": "secret explanation"},
        "level": "error",
        "spans": [
            {
                "file_name": SOURCE_FILE,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "line_start": 4,
                "line_end": 4,
                "column_start": 10,
                "column_end": 24,
                "is_primary": True,
                "text": [{"text": "secret source excerpt", "highlight_start": 1}],
                "label": "human repair label",
                "suggested_replacement": suggestion,
                "suggestion_applicability": "MachineApplicable",
            }
        ],
        "children": [{"message": "try the answer", "level": "help", "spans": []}],
        "rendered": "rendered repair guidance",
    }
    stderr = json.dumps(diagnostic, sort_keys=True).encode("utf-8") + b"\n"
    return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("rustc-identity"),
        source_file_name=SOURCE_FILE,
        source=source_bytes,
        arguments=("--crate-type", "lib", "--error-format=json", SOURCE_FILE),
        returncode=1,
        stdout=b"",
        stderr=stderr,
        compile_invocations=1,
    )


def _decision(
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    *,
    operator_index: int,
    target_index: int,
    selection_context_sha256: str = SELECTION_CONTEXT,
) -> factors.FactorizedFeedbackDecisionV1:
    memory = factors.start_factorized_feedback_hypothesis_memory_v1(BANK)
    observation = rustc_feedback.project_rust_factorized_feedback_v1(
        attempt,
        selection_context_sha256=selection_context_sha256,
        active_hypothesis_sha256s=tuple(hypothesis.sha256 for hypothesis in reversed(BANK)),
    )
    selected = next(
        hypothesis
        for hypothesis in BANK
        if hypothesis.component_sha256s
        == (OPERATOR_ROOTS[operator_index], TARGET_ROOTS[target_index])
    )
    return memory.route(
        observation,
        exploration_hypothesis_sha256=selected.sha256,
    )


def _prepare(
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    *,
    expected_checkpoint_sha256: str,
    allow_exploration: bool = True,
):
    return prepare_factorized_text_edit_materializer_v1(
        diagnostic_attempt=attempt,
        source=SOURCE,
        selection_context_sha256=SELECTION_CONTEXT,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=tuple(reversed(OPERATOR_BINDINGS)),
        target_bindings=tuple(reversed(TARGET_BINDINGS)),
    )


def test_declares_primary_identifier_only_after_selected_executor_release() -> None:
    attempt = _attempt()
    decision = _decision(attempt, operator_index=0, target_index=0)
    authority = _prepare(attempt, expected_checkpoint_sha256=decision.checkpoint_sha256)

    assert authority.materialization_count == 0
    assert authority.release_count == 0
    assert authority.materialization is None
    assert authority.release is None

    selected_sources: list[rust_executor.RustSelectedSourceV2] = []

    def execute(selected_source: rust_executor.RustSelectedSourceV2) -> str:
        selected_sources.append(selected_source)
        return "executed"

    result = rust_executor._execute_v2_authority(authority, decision, execute)

    insertion = SOURCE.index("\n", SOURCE.index("_first_anchor")) + 1
    new_text = "    let _missing_value = _first_anchor;\n"
    expected_source = f"{SOURCE[:insertion]}{new_text}{SOURCE[insertion:]}"
    assert result == "executed"
    assert len(selected_sources) == 1
    assert selected_sources[0].source == expected_source
    assert authority.materialization_count == 1
    assert authority.release_count == 1
    assert authority.materialization is not None
    assert authority.release is not None
    assert authority.materialization.edit.byte_range == (insertion, insertion)
    assert authority.materialization.edit.new_text == new_text
    assert authority.release.materialization == authority.materialization
    assert selected_sources[0].release_sha256 == authority.release.sha256
    authority.materialization.authenticate()
    authority.release.authenticate()
    with pytest.raises(FactorizedTextEditMaterializerError, match="consumed"):
        authority.execute_selected(decision, execute)


def test_replaces_primary_span_and_rejects_mismatched_diagnostic_context() -> None:
    attempt = _attempt(message="ignored message", suggestion="ignored suggestion")
    valid_decision = _decision(attempt, operator_index=1, target_index=1)
    mismatched_decision = _decision(
        attempt,
        operator_index=1,
        target_index=1,
        selection_context_sha256=_root("different-selection-context"),
    )
    authority = _prepare(attempt, expected_checkpoint_sha256=valid_decision.checkpoint_sha256)

    with pytest.raises(FactorizedTextEditMaterializerError, match="diagnostic custody"):
        authority.execute_selected(mismatched_decision, lambda selected: selected)
    assert (authority.materialization_count, authority.release_count) == (0, 0)

    selected = authority.execute_selected(valid_decision, lambda selected_source: selected_source)
    primary_start = SOURCE.index("_missing_value")
    primary_end = primary_start + len("_missing_value")
    expected_source = f"{SOURCE[:primary_start]}_second_anchor{SOURCE[primary_end:]}"
    assert type(selected) is rust_executor.RustSelectedSourceV2
    assert selected.source == expected_source
    assert authority.materialization is not None
    assert authority.materialization.edit.record(include_sha256=False) == {
        "range": {"byte_start": primary_start, "byte_end": primary_end},
        "new_text": "_second_anchor",
    }
    assert (authority.materialization_count, authority.release_count) == (1, 1)


def test_binds_checkpoint_and_forbids_evaluation_exploration() -> None:
    attempt = _attempt()
    memory = factors.start_factorized_feedback_hypothesis_memory_v1(BANK)
    decision = _decision(attempt, operator_index=0, target_index=0)
    outcome = factors.seal_factorized_feedback_outcome_v1(
        decision=decision,
        proposal_sha256=_root("proposal:event"),
        execution_sha256=_root("execution:event"),
        consequence_sha256=_root("consequence:event"),
        feedback_sha256=_root("feedback:event"),
        passed_cell=1,
    )
    successor = factors.advance_factorized_feedback_hypothesis_memory_v1(memory, outcome)
    wrong_checkpoint = successor.route(
        decision.observation,
        exploration_hypothesis_sha256=decision.selected_hypothesis_sha256,
    )
    training_authority = _prepare(
        attempt,
        expected_checkpoint_sha256=decision.checkpoint_sha256,
    )

    with pytest.raises(FactorizedTextEditMaterializerError, match="checkpoint differs"):
        training_authority.execute_selected(wrong_checkpoint, lambda selected: selected)
    assert (training_authority.materialization_count, training_authority.release_count) == (0, 0)

    evaluation_authority = _prepare(
        attempt,
        expected_checkpoint_sha256=decision.checkpoint_sha256,
        allow_exploration=False,
    )
    with pytest.raises(FactorizedTextEditMaterializerError, match="forbids exploration"):
        evaluation_authority.execute_selected(decision, lambda selected: selected)
    assert (evaluation_authority.materialization_count, evaluation_authority.release_count) == (
        0,
        0,
    )
