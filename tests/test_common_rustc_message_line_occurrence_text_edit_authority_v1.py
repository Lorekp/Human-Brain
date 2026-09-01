from __future__ import annotations

import inspect
import json
from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_materializer_v1 as edits
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_preceding_let_declaration_occurrences_v1 as occurrences
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import (
    common_rustc_message_line_occurrence_text_edit_authority_v1 as authority,
)
from experiments.r3_pilot import (
    common_rustc_message_line_unordered_cue_text_edit_authority_v1 as delegate_authority,
)
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from humanbrain import rust_selected_action_executor as rust_executor

DECLARE_SOURCE = """pub fn solve(_samples: &[i64]) -> Vec<i64> {
    let far_anchor = 7_i64;
    let near_anchor = 9_i64;
    vec![missing_value]
}
"""
REPLACE_SOURCE = """pub fn solve(_samples: &[i64]) -> Vec<i64> {
    let far_anchor = 7_i64;
    let near_anchor = 9_i64;
    let wrong_value = "wrong";
    vec![wrong_value]
}
"""


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


ACTION_BINDINGS = catalogue.canonical_factorized_text_edit_operator_bindings_v1()
ACTION_ROOTS_BY_OPERATION = {operation: root for root, operation in ACTION_BINDINGS}
RELATION_ROOTS = tuple(
    occurrences.preceding_declaration_relation_component_sha256_v1(relation)
    for relation in occurrences.PRECEDING_DECLARATION_RELATIONS
)
BANK = tuple(
    factors.seal_factorized_hypothesis_v1((action_root, relation_root))
    for action_root, _operation in ACTION_BINDINGS
    for relation_root in RELATION_ROOTS
)


def _stderr(
    source: str,
    *,
    primary_identifier: str,
    source_file_name: str,
    code: str,
    message: str,
) -> bytes:
    source_bytes = source.encode("ascii")
    primary_bytes = primary_identifier.encode("ascii")
    primary_start = source_bytes.rindex(primary_bytes)
    diagnostic = {
        "$message_type": "diagnostic",
        "message": message,
        "code": {"code": code, "explanation": "withheld"},
        "level": "error",
        "spans": [
            {
                "file_name": source_file_name,
                "byte_start": primary_start,
                "byte_end": primary_start + len(primary_bytes),
                "is_primary": True,
                "label": "withheld",
                "suggested_replacement": "withheld",
            }
        ],
        "children": [{"message": "withheld note", "level": "note", "spans": []}],
        "rendered": "withheld rendered diagnostic",
    }
    return json.dumps(diagnostic, sort_keys=True).encode("utf-8") + b"\n"


def _attempt(
    source: str,
    *,
    source_file_name: str,
    stderr: bytes,
) -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
    return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("c11-authority-rustc"),
        source_file_name=source_file_name,
        source=source.encode("ascii"),
        arguments=("--crate-type", "lib", "--error-format=json", source_file_name),
        returncode=1,
        stdout=b"",
        stderr=stderr,
        compile_invocations=1,
    )


def _fixture(
    source: str = DECLARE_SOURCE,
    *,
    label: str = "declare",
) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes, str]:
    is_replace = "let wrong_value" in source
    primary_identifier = "wrong_value" if is_replace else "missing_value"
    source_file_name = f"c11_authority_{label}.rs"
    stderr = _stderr(
        source,
        primary_identifier=primary_identifier,
        source_file_name=source_file_name,
        code="E0308" if is_replace else "E0425",
        message="mismatched types" if is_replace else "cannot find value in this scope",
    )
    return (
        _attempt(source, source_file_name=source_file_name, stderr=stderr),
        stderr,
        primary_identifier,
    )


def _prepare(
    *,
    source: str = DECLARE_SOURCE,
    label: str = "declare",
    memory: unordered.FrozenUnorderedCueFactorMemoryV1 | None = None,
    expected_checkpoint_sha256: str | None = None,
    allow_exploration: bool = True,
):
    selected_memory = (
        unordered.start_unordered_cue_factor_memory_v1(BANK) if memory is None else memory
    )
    attempt, stderr, _primary_identifier = _fixture(source, label=label)
    return authority.prepare_rustc_message_line_occurrence_text_edit_authority_v1(
        diagnostic_attempt=attempt,
        stderr=stderr,
        source=source,
        selection_context_sha256=_root(f"c11-selection-context:{label}"),
        expected_checkpoint_sha256=(
            selected_memory.checkpoint_sha256
            if expected_checkpoint_sha256 is None
            else expected_checkpoint_sha256
        ),
        expected_factorized_checkpoint_sha256=(
            selected_memory.factorized_checkpoint_sha256
        ),
        allow_exploration=allow_exploration,
    )


def _decision(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    prepared: authority.PreparedRustcMessageLineOccurrenceTextEditAuthorityV1,
    *,
    operation: str,
    relation: str,
) -> unordered.UnorderedCueFactorDecisionV1:
    component_pair = (
        ACTION_ROOTS_BY_OPERATION[operation],
        occurrences.preceding_declaration_relation_component_sha256_v1(relation),
    )
    selected = next(
        hypothesis for hypothesis in BANK if hypothesis.component_sha256s == component_pair
    )
    return memory.route(
        prepared.projected_observation,
        exploration_hypothesis_sha256=selected.sha256,
    )


def _install_fake_execution(monkeypatch: pytest.MonkeyPatch) -> type:
    class FakeExecution:
        def __init__(self, selected_source: rust_executor.RustSelectedSourceV2) -> None:
            self.selected_source = selected_source

        def authenticate_structure(self) -> None:
            self.selected_source.authenticate_structure()

    monkeypatch.setattr(
        delegate_authority.interventions,
        "_prepare_execution",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        delegate_authority.interventions,
        "_authenticate_execution_preparation",
        lambda *_args, **_kwargs: None,
    )

    def execute(
        selected_source: rust_executor.RustSelectedSourceV2,
        _prepared: object,
        *,
        authenticate_current,
    ):
        authenticate_current()
        return FakeExecution(selected_source)

    monkeypatch.setattr(delegate_authority.interventions, "_execute_prepared", execute)
    monkeypatch.setattr(rust_executor, "RustSelectedActionExecutionV2", FakeExecution)
    return FakeExecution


def _rehash_wrapper_receipt(
    receipt: authority.RustcMessageLineOccurrenceTextEditAuthorityReceiptV1,
) -> authority.RustcMessageLineOccurrenceTextEditAuthorityReceiptV1:
    return replace(
        receipt,
        sha256=authority._digest(
            authority._AUTHORITY_NAMESPACE,
            receipt.record(include_sha256=False),
        ),
    )


def _rehash_delegate_receipt(
    receipt: delegate_authority.RustcMessageLineUnorderedCueTextEditAuthorityReceiptV1,
) -> delegate_authority.RustcMessageLineUnorderedCueTextEditAuthorityReceiptV1:
    return replace(
        receipt,
        sha256=delegate_authority._digest(
            delegate_authority._AUTHORITY_NAMESPACE,
            receipt.record(include_sha256=False),
        ),
    )


def test_factory_derives_both_relation_bindings_with_zero_work_and_no_target_inputs() -> None:
    parameters = inspect.signature(
        authority.prepare_rustc_message_line_occurrence_text_edit_authority_v1
    ).parameters
    assert {
        "target_bindings",
        "marker",
        "markers",
        "candidates",
        "occurrence_roster",
        "target_span",
        "correctness",
        "roles",
        "outputs",
    }.isdisjoint(parameters)

    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(memory=memory)
    prepared.receipt.authenticate()
    prepared.occurrence_roster.authenticate()
    identifiers_by_relation = {
        item.relation: item.identifier for item in prepared.occurrence_roster.occurrences
    }
    roots_to_identifiers = dict(prepared.receipt.relation_target_bindings)

    assert identifiers_by_relation == {
        occurrences.FARTHER_PRECEDING_DECLARATION: "far_anchor",
        occurrences.NEARER_PRECEDING_DECLARATION: "near_anchor",
    }
    assert roots_to_identifiers == {
        occurrences.preceding_declaration_relation_component_sha256_v1(
            occurrences.FARTHER_PRECEDING_DECLARATION
        ): "far_anchor",
        occurrences.preceding_declaration_relation_component_sha256_v1(
            occurrences.NEARER_PRECEDING_DECLARATION
        ): "near_anchor",
    }
    assert prepared.receipt.occurrence_roster_sha256 == prepared.occurrence_roster.sha256
    assert prepared.receipt.delegate_authority_sha256 == prepared.authority_sha256
    assert prepared.custody_sha256 == prepared.receipt.sha256
    assert prepared.materialization is None
    assert prepared.release is None
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    attempt, stderr, _identifier = _fixture(DECLARE_SOURCE, label="forbidden")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        authority.prepare_rustc_message_line_occurrence_text_edit_authority_v1(
            diagnostic_attempt=attempt,
            stderr=stderr,
            source=DECLARE_SOURCE,
            selection_context_sha256=_root("forbidden-context"),
            expected_checkpoint_sha256=memory.checkpoint_sha256,
            expected_factorized_checkpoint_sha256=memory.factorized_checkpoint_sha256,
            allow_exploration=True,
            target_bindings=((_root("caller-target"), "caller_marker"),),
        )


def test_stop_and_checkpoint_mismatch_leave_delegate_at_zero_work() -> None:
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    stopped = _prepare(memory=memory, label="stop")
    stop = memory.route(stopped.projected_observation)
    with pytest.raises(ValueError, match="selected"):
        stopped.execute_selected(stop, ((1,),))
    assert stopped.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    mismatched = _prepare(
        memory=memory,
        label="mismatch",
        expected_checkpoint_sha256=_root("wrong-checkpoint"),
    )
    selected = _decision(
        memory,
        mismatched,
        operation=edits.DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR,
        relation=occurrences.FARTHER_PRECEDING_DECLARATION,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        mismatched.execute_selected(selected, ((1,),))
    assert mismatched.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)


@pytest.mark.parametrize(
    ("source", "label", "operation", "relation", "expected_fragment"),
    (
        (
            DECLARE_SOURCE,
            "selected-declare",
            edits.DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR,
            occurrences.FARTHER_PRECEDING_DECLARATION,
            "    let missing_value = far_anchor;\n",
        ),
        (
            REPLACE_SOURCE,
            "selected-replace",
            edits.REPLACE_PRIMARY_SPAN_WITH_ANCHOR,
            occurrences.NEARER_PRECEDING_DECLARATION,
            "    vec![near_anchor]\n",
        ),
    ),
)
def test_valid_selected_relation_delegates_exactly_one_materialization_and_execution(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    label: str,
    operation: str,
    relation: str,
    expected_fragment: str,
) -> None:
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(source=source, label=label, memory=memory)
    selected = _decision(memory, prepared, operation=operation, relation=relation)
    fake_execution = _install_fake_execution(monkeypatch)

    result = prepared.execute_selected(selected, ((1,),))

    assert type(result) is fake_execution
    assert expected_fragment in result.selected_source.source
    assert result.selected_source.authority_sha256 == prepared.delegate_authority_sha256
    assert prepared.materialization is not None
    assert prepared.release is not None
    assert prepared.materialization_count == 1
    assert prepared.release_count == 1
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(1, 1, 1, 1)


def test_forged_roster_delegate_and_relation_mapping_each_fail_closed() -> None:
    roster_forged = _prepare(label="forge-roster")
    roster_forged._occurrence_roster = replace(
        roster_forged._occurrence_roster,
        sha256=_root("forged-roster"),
    )
    with pytest.raises(authority.RustcMessageLineOccurrenceTextEditAuthorityError):
        _ = roster_forged.occurrence_roster

    delegate_forged = _prepare(label="forge-delegate")
    changed_delegate_receipt = replace(
        delegate_forged._delegate._receipt,
        expected_checkpoint_sha256=_root("forged-delegate-checkpoint"),
    )
    delegate_forged._delegate._receipt = _rehash_delegate_receipt(
        changed_delegate_receipt
    )
    with pytest.raises(authority.RustcMessageLineOccurrenceTextEditAuthorityError):
        _ = delegate_forged.projected_observation

    mapping_forged = _prepare(label="forge-mapping")
    swapped_identifiers = tuple(
        (root, identifier)
        for (root, _old_identifier), (_other_root, identifier) in zip(
            mapping_forged.receipt.relation_target_bindings,
            reversed(mapping_forged.receipt.relation_target_bindings),
            strict=True,
        )
    )
    changed_mapping_receipt = replace(
        mapping_forged._receipt,
        relation_target_bindings=swapped_identifiers,
    )
    mapping_forged._receipt = _rehash_wrapper_receipt(changed_mapping_receipt)
    with pytest.raises(
        authority.RustcMessageLineOccurrenceTextEditAuthorityError,
        match="custody changed",
    ):
        _ = mapping_forged.receipt


def test_identifier_rename_and_padding_change_custody_but_preserve_relation_roots() -> None:
    padded_renamed = DECLARE_SOURCE.replace(
        "    let far_anchor = 7_i64;\n",
        "\n    let outer_anchor = 7_i64;\n",
    )
    original = _prepare(label="original")
    changed = _prepare(source=padded_renamed, label="renamed-padded")

    original_roots = tuple(
        item.relation_component_sha256 for item in original.occurrence_roster.occurrences
    )
    changed_roots = tuple(
        item.relation_component_sha256 for item in changed.occurrence_roster.occurrences
    )
    assert original_roots == changed_roots == RELATION_ROOTS
    assert original.occurrence_roster.sha256 != changed.occurrence_roster.sha256
    assert original.custody_sha256 != changed.custody_sha256
    assert original.delegate_authority_sha256 != changed.delegate_authority_sha256
    assert changed.occurrence_roster.occurrences[0].identifier == "outer_anchor"
    assert original.work_counters == changed.work_counters == (
        authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)
    )
