from __future__ import annotations

import inspect
import json
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_rustc_lexical_unordered_cue_adapter_v1 as lexical
from experiments.r3_pilot import (
    common_rustc_lexical_unordered_cue_text_edit_authority_v1 as authority,
)
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from humanbrain import rust_selected_action_executor as rust_executor

SOURCE_FILE = "c9_authority.rs"
SOURCE = """pub fn solve(_samples: &[i64]) -> Vec<i64> {
    let _first_anchor = 7_i64;
    let _second_anchor = 9_i64;
    vec![_missing_value]
}
"""


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


ACTION_ROOTS = catalogue.canonical_factorized_text_edit_operator_action_roots_v1()
TARGET_ROOTS = (_root("c9-target:first"), _root("c9-target:second"))
TARGET_BINDINGS = (
    (TARGET_ROOTS[0], "_first_anchor"),
    (TARGET_ROOTS[1], "_second_anchor"),
)
BANK = tuple(
    factors.seal_factorized_hypothesis_v1((action_root, target_root))
    for action_root in ACTION_ROOTS
    for target_root in TARGET_ROOTS
)


def _stderr(message: str = "cannot find value") -> bytes:
    source_bytes = SOURCE.encode("ascii")
    byte_start = source_bytes.index(b"_missing_value")
    diagnostic = {
        "$message_type": "diagnostic",
        "message": message,
        "code": {"code": "E0425", "explanation": "withheld"},
        "level": "error",
        "spans": [
            {
                "file_name": SOURCE_FILE,
                "byte_start": byte_start,
                "byte_end": byte_start + len(b"_missing_value"),
                "is_primary": True,
                "label": "withheld",
                "suggested_replacement": "withheld",
            }
        ],
        "children": [{"message": "withheld fix", "level": "help", "spans": []}],
        "rendered": "withheld rendered fix",
    }
    return json.dumps(diagnostic, sort_keys=True).encode("utf-8") + b"\n"


def _attempt(stderr: bytes) -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
    return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("c9-authority-rustc"),
        source_file_name=SOURCE_FILE,
        source=SOURCE.encode("ascii"),
        arguments=("--crate-type", "lib", "--error-format=json", SOURCE_FILE),
        returncode=1,
        stdout=b"",
        stderr=stderr,
        compile_invocations=1,
    )


def _selection_context(label: str = "main") -> str:
    return _root(f"c9-selection-context:{label}")


def _decision(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    observation: unordered.UnorderedCueFactorObservationV1,
    *,
    bank: tuple[factors.FactorizedHypothesisV1, ...] = BANK,
    action_index: int = 0,
    target_root: str = TARGET_ROOTS[0],
) -> unordered.UnorderedCueFactorDecisionV1:
    selected = next(
        hypothesis
        for hypothesis in bank
        if hypothesis.component_sha256s == (ACTION_ROOTS[action_index], target_root)
    )
    return memory.route(
        observation,
        exploration_hypothesis_sha256=selected.sha256,
    )


def _prepare(
    *,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    stderr: bytes,
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    selection_context_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    allow_exploration: bool = True,
):
    return authority.prepare_rustc_lexical_unordered_cue_text_edit_authority_v1(
        diagnostic_attempt=attempt,
        stderr=stderr,
        source=SOURCE,
        selection_context_sha256=(
            _selection_context() if selection_context_sha256 is None else selection_context_sha256
        ),
        expected_checkpoint_sha256=(
            memory.checkpoint_sha256
            if expected_checkpoint_sha256 is None
            else expected_checkpoint_sha256
        ),
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


def test_factory_derives_one_self_authenticating_envelope_with_zero_work() -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)

    prepared.lexical_envelope.authenticate()
    prepared.receipt.authenticate()
    assert prepared.projected_observation == prepared.lexical_envelope.observation
    assert prepared.receipt.lexical_envelope_sha256 == prepared.lexical_envelope.sha256
    assert prepared.receipt.raw_stderr_sha256 == sha256(stderr).hexdigest()
    assert prepared.receipt.projected_observation_sha256 == prepared.projected_observation.sha256
    assert prepared.materialization is None
    assert prepared.release is None
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)


def test_wrong_stderr_and_replaced_envelope_fail_custody() -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)

    with pytest.raises(authority.RustcLexicalUnorderedCueTextEditAuthorityError, match="envelope"):
        _prepare(
            attempt=attempt,
            stderr=_stderr("mismatched types"),
            memory=memory,
        )

    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)
    forged = lexical.project_rustc_lexical_unordered_cue_observation_v1(
        attempt,
        stderr=stderr,
        selection_context_sha256=_selection_context("forged"),
        active_hypothesis_sha256s=tuple(hypothesis.sha256 for hypothesis in BANK),
    )
    prepared._lexical_envelope = forged
    with pytest.raises(authority.RustcLexicalUnorderedCueTextEditAuthorityError, match="custody"):
        _ = prepared.projected_observation


def test_factory_has_no_caller_observation_parameter() -> None:
    signature = inspect.signature(
        authority.prepare_rustc_lexical_unordered_cue_text_edit_authority_v1
    )
    assert "observation" not in signature.parameters
    assert "projected_observation" not in signature.parameters

    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        authority.prepare_rustc_lexical_unordered_cue_text_edit_authority_v1(
            diagnostic_attempt=attempt,
            stderr=stderr,
            source=SOURCE,
            selection_context_sha256=_selection_context(),
            expected_checkpoint_sha256=memory.checkpoint_sha256,
            expected_factorized_checkpoint_sha256=memory.factorized_checkpoint_sha256,
            allow_exploration=True,
            target_bindings=TARGET_BINDINGS,
            projected_observation=object(),
        )


def test_checkpoint_and_bank_mismatch_stop_before_work() -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    wrong_checkpoint = _prepare(
        attempt=attempt,
        stderr=stderr,
        memory=memory,
        expected_checkpoint_sha256=_root("wrong-checkpoint"),
    )
    decision = _decision(memory, wrong_checkpoint.projected_observation)
    with pytest.raises(
        authority.RustcLexicalUnorderedCueTextEditAuthorityError,
        match="checkpoint",
    ):
        wrong_checkpoint.execute_selected(decision, ((1,),))
    assert wrong_checkpoint.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    bad_target = _root("c9-target:foreign")
    bad_bank = tuple(
        factors.seal_factorized_hypothesis_v1((action_root, target_root))
        for action_root in ACTION_ROOTS
        for target_root in (TARGET_ROOTS[0], bad_target)
    )
    bad_memory = unordered.start_unordered_cue_factor_memory_v1(bad_bank)
    bank_bound = _prepare(attempt=attempt, stderr=stderr, memory=bad_memory)
    bad_decision = _decision(
        bad_memory,
        bank_bound.projected_observation,
        bank=bad_bank,
        target_root=TARGET_ROOTS[0],
    )
    with pytest.raises(
        authority.RustcLexicalUnorderedCueTextEditAuthorityError,
        match="bank",
    ):
        bank_bound.execute_selected(bad_decision, ((1,),))
    assert bank_bound.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)


def test_only_selected_decision_materializes_executes_and_authority_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)
    stop = memory.route(prepared.projected_observation)
    with pytest.raises(
        authority.RustcLexicalUnorderedCueTextEditAuthorityError,
        match="selected",
    ):
        prepared.execute_selected(stop, ((1,),))
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    decision = _decision(memory, prepared.projected_observation)
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
    assert type(prepared.materialization) is authority.UnorderedCueFactorTextEditMaterializationV1
    assert type(prepared.release) is authority.UnorderedCueFactorTextEditReleaseV1
    assert prepared.materialization.decision_sha256 == decision.sha256
    assert prepared.materialization.observation_sha256 == prepared.projected_observation.sha256
    assert prepared.release.selection_sha256 == decision.sha256
    assert prepared.release.sha256 == result.selected_source.release_sha256
    prepared.materialization.authenticate()
    prepared.release.authenticate()
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(1, 1, 1, 1)

    with pytest.raises(
        authority.RustcLexicalUnorderedCueTextEditAuthorityError,
        match="consumed",
    ):
        prepared.execute_selected(decision, ((1,),))
