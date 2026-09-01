from __future__ import annotations

import json
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_rustc_unordered_cue_factor_adapter_v1 as rustc_unordered
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from experiments.r3_pilot import common_unordered_cue_factor_text_edit_authority_v1 as authority
from humanbrain import rust_selected_action_executor as rust_executor

SOURCE_FILE = "case.rs"
SOURCE = """pub fn solve(_samples: &[i64]) -> Vec<i64> {
    let _first_anchor = 7_i64;
    let _second_anchor = 9_i64;
    vec![_missing_value]
}
"""


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


ACTION_ROOTS = catalogue.canonical_factorized_text_edit_operator_action_roots_v1()
TARGET_ROOTS = (_root("unordered-target:first"), _root("unordered-target:second"))
TARGET_BINDINGS = (
    (TARGET_ROOTS[0], "_first_anchor"),
    (TARGET_ROOTS[1], "_second_anchor"),
)
BANK = tuple(
    factors.seal_factorized_hypothesis_v1((action_root, target_root))
    for action_root in ACTION_ROOTS
    for target_root in TARGET_ROOTS
)


def _attempt() -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
    source_bytes = SOURCE.encode("ascii")
    byte_start = source_bytes.index(b"_missing_value")
    diagnostic = {
        "$message_type": "diagnostic",
        "message": "cannot find value",
        "code": {"code": "E0425", "explanation": "withheld"},
        "level": "error",
        "spans": [
            {
                "file_name": SOURCE_FILE,
                "byte_start": byte_start,
                "byte_end": byte_start + len(b"_missing_value"),
                "line_start": 4,
                "line_end": 4,
                "column_start": 10,
                "column_end": 24,
                "is_primary": True,
                "text": [{"text": "withheld", "highlight_start": 1}],
                "label": "withheld",
                "suggested_replacement": "withheld",
                "suggestion_applicability": "MachineApplicable",
            }
        ],
        "children": [],
        "rendered": "withheld",
    }
    return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("unordered-rustc"),
        source_file_name=SOURCE_FILE,
        source=source_bytes,
        arguments=("--crate-type", "lib", "--error-format=json", SOURCE_FILE),
        returncode=1,
        stdout=b"",
        stderr=json.dumps(diagnostic, sort_keys=True).encode("utf-8") + b"\n",
        compile_invocations=1,
    )


def _selection_context(label: str = "main") -> str:
    return _root(f"selection-context:{label}")


def _projected_observation(
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    label: str,
) -> unordered.UnorderedCueFactorObservationV1:
    return rustc_unordered.project_rust_unordered_cue_factor_observation_v1(
        attempt,
        selection_context_sha256=_selection_context(label),
        active_hypothesis_sha256s=tuple(hypothesis.sha256 for hypothesis in BANK),
    )


def _decision(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    observation: unordered.UnorderedCueFactorObservationV1,
    *,
    action_index: int = 0,
    target_index: int = 0,
) -> unordered.UnorderedCueFactorDecisionV1:
    selected = next(
        hypothesis
        for hypothesis in BANK
        if hypothesis.component_sha256s == (ACTION_ROOTS[action_index], TARGET_ROOTS[target_index])
    )
    return memory.route(
        observation,
        exploration_hypothesis_sha256=selected.sha256,
    )


def _prepare(
    *,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    selection_context_sha256: str | None = None,
    allow_exploration: bool = True,
):
    return authority.prepare_unordered_cue_factor_text_edit_authority_v1(
        diagnostic_attempt=attempt,
        source=SOURCE,
        selection_context_sha256=(
            _selection_context() if selection_context_sha256 is None else selection_context_sha256
        ),
        expected_checkpoint_sha256=memory.checkpoint_sha256,
        expected_factorized_checkpoint_sha256=memory.factorized_checkpoint_sha256,
        allow_exploration=allow_exploration,
        target_bindings=tuple(reversed(TARGET_BINDINGS)),
    )


def _install_fake_execution(monkeypatch: pytest.MonkeyPatch) -> type:
    class FakeExecution:
        def __init__(self, selected_source: rust_executor.RustSelectedSourceV2) -> None:
            self.selected_source = selected_source

        def authenticate_structure(self) -> None:
            self.selected_source.authenticate_structure()

    monkeypatch.setattr(authority.interventions, "_prepare_execution", lambda *_args: object())
    monkeypatch.setattr(
        authority.interventions,
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

    monkeypatch.setattr(authority.interventions, "_execute_prepared", execute)
    monkeypatch.setattr(rust_executor, "RustSelectedActionExecutionV2", FakeExecution)
    return FakeExecution


def test_outer_decision_root_drives_one_post_selection_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _attempt()
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(attempt=attempt, memory=memory)
    decision = _decision(memory, prepared.projected_observation)
    forged_observation = _projected_observation(attempt, "forged")
    forged_decision = _decision(memory, forged_observation)

    with pytest.raises(authority.UnorderedCueFactorTextEditAuthorityError, match="observation"):
        prepared.execute_selected(forged_decision, ((1,),))
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)
    assert prepared.receipt.selection_context_sha256 == _selection_context()
    assert prepared.receipt.projected_observation_sha256 == prepared.projected_observation.sha256

    fake_execution = _install_fake_execution(monkeypatch)
    result = prepared.execute_selected(decision, ((1,),))

    primary_start = SOURCE.index("_missing_value")
    primary_end = primary_start + len("_missing_value")
    expected_source = f"{SOURCE[:primary_start]}_first_anchor{SOURCE[primary_end:]}"
    assert type(result) is fake_execution
    assert result.selected_source.source == expected_source
    assert result.selected_source.selection_sha256 == decision.sha256
    assert result.selected_source.authority_sha256 == prepared.authority_sha256
    assert prepared.materialization is not None
    assert prepared.release is not None
    assert prepared.materialization.decision_sha256 == decision.sha256
    assert (
        prepared.materialization.factorized_decision_sha256 == decision.factorized_decision.sha256
    )
    assert prepared.materialization.observation_sha256 == decision.observation.sha256
    assert (
        prepared.materialization.binding_observation_sha256 == decision.binding_observation.sha256
    )
    assert prepared.release.selection_sha256 == decision.sha256
    assert prepared.release.sha256 == result.selected_source.release_sha256
    prepared.materialization.authenticate()
    prepared.release.authenticate()
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(1, 1, 1, 1)

    with pytest.raises(authority.UnorderedCueFactorTextEditAuthorityError, match="consumed"):
        prepared.execute_selected(decision, ((1,),))


def test_exploration_policy_and_preparation_failure_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _attempt()
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    evaluation = _prepare(
        attempt=attempt,
        memory=memory,
        allow_exploration=False,
    )
    decision = _decision(
        memory,
        evaluation.projected_observation,
        action_index=1,
        target_index=1,
    )

    with pytest.raises(authority.UnorderedCueFactorTextEditAuthorityError, match="forbids"):
        evaluation.execute_selected(decision, ((1,),))
    assert evaluation.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    prepared = _prepare(attempt=attempt, memory=memory)

    def fail_preparation(*_args):
        raise RuntimeError("synthetic preparation failure")

    monkeypatch.setattr(authority.interventions, "_prepare_execution", fail_preparation)
    with pytest.raises(RuntimeError, match="synthetic preparation failure"):
        prepared.execute_selected(decision, ((1,),))
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)
    with pytest.raises(authority.UnorderedCueFactorTextEditAuthorityError, match="consumed"):
        prepared.execute_selected(decision, ((1,),))
