from __future__ import annotations

import inspect
import json
from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_ascii_word_unigram_cues_v1 as ascii_cues
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_rustc_message_line_unordered_cue_adapter_v1 as message_line
from experiments.r3_pilot import (
    common_rustc_message_line_unordered_cue_text_edit_authority_v1 as authority,
)
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from humanbrain import rust_selected_action_executor as rust_executor

SOURCE_FILE = "c10_authority.rs"
SOURCE = """pub fn solve(_samples: &[i64]) -> Vec<i64> {
    let _first_anchor = 7_i64;
    let _second_anchor = 9_i64;
    vec![_missing_value]
}
"""
SOURCE_BYTES = SOURCE.encode("ascii")
PRIMARY_TEXT = b"_missing_value"
PRIMARY_START = SOURCE_BYTES.index(PRIMARY_TEXT)
PRIMARY_END = PRIMARY_START + len(PRIMARY_TEXT)
ARGUMENTS = ("--crate-type", "lib", "--error-format=json", SOURCE_FILE)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


ACTION_ROOTS = catalogue.canonical_factorized_text_edit_operator_action_roots_v1()
TARGET_ROOTS = (_root("c10-target:first"), _root("c10-target:second"))
TARGET_BINDINGS = (
    (TARGET_ROOTS[0], "_first_anchor"),
    (TARGET_ROOTS[1], "_second_anchor"),
)
BANK = tuple(
    factors.seal_factorized_hypothesis_v1((action_root, target_root))
    for action_root in ACTION_ROOTS
    for target_root in TARGET_ROOTS
)


def _span(byte_start: int, byte_end: int) -> dict[str, object]:
    return {
        "file_name": SOURCE_FILE,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "is_primary": True,
        "label": "withheld",
        "suggested_replacement": "withheld",
    }


def _stderr(
    message: str = "cannot find value",
    *,
    primary_spans: tuple[tuple[int, int], ...] = ((PRIMARY_START, PRIMARY_END),),
) -> bytes:
    diagnostic = {
        "$message_type": "diagnostic",
        "message": message,
        "code": {"code": "E0425", "explanation": "withheld"},
        "level": "error",
        "spans": [_span(start, end) for start, end in primary_spans],
        "children": [{"message": "withheld fix", "level": "help", "spans": []}],
        "rendered": "withheld rendered fix",
    }
    return json.dumps(diagnostic, sort_keys=True).encode("utf-8") + b"\n"


def _attempt(stderr: bytes) -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
    return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("c10-authority-rustc"),
        source_file_name=SOURCE_FILE,
        source=SOURCE_BYTES,
        arguments=ARGUMENTS,
        returncode=1,
        stdout=b"",
        stderr=stderr,
        compile_invocations=1,
    )


def _selection_context(label: str = "main") -> str:
    return _root(f"c10-selection-context:{label}")


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
    source: str = SOURCE,
    selection_context_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    allow_exploration: bool = True,
):
    return authority.prepare_rustc_message_line_unordered_cue_text_edit_authority_v1(
        diagnostic_attempt=attempt,
        stderr=stderr,
        source=source,
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


def test_factory_owns_one_self_authenticating_envelope_with_zero_work() -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)

    prepared.message_line_envelope.authenticate()
    prepared.receipt.authenticate()
    primary_span = authority._sole_custodied_primary_span(prepared.message_line_envelope)
    sealed_span = prepared.message_line_envelope.source_line_envelope.span
    assert (primary_span.byte_start, primary_span.byte_end) == (
        sealed_span.byte_start,
        sealed_span.byte_end,
    )
    assert prepared.projected_observation == prepared.message_line_envelope.observation
    assert prepared.receipt.message_line_envelope_sha256 == prepared.message_line_envelope.sha256
    assert prepared.receipt.raw_stderr_sha256 == sha256(stderr).hexdigest()
    assert prepared.receipt.projected_observation_sha256 == prepared.projected_observation.sha256
    assert not hasattr(prepared, "_source_bytes")
    assert prepared.materialization is None
    assert prepared.release is None
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)


def test_bad_inputs_envelope_custody_and_caller_observation_fail_closed() -> None:
    parameters = inspect.signature(
        authority.prepare_rustc_message_line_unordered_cue_text_edit_authority_v1
    ).parameters
    assert "observation" not in parameters
    assert "projected_observation" not in parameters
    assert "active_hypothesis_sha256s" not in parameters

    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="envelope derivation failed",
    ):
        _prepare(
            attempt=attempt,
            stderr=_stderr("mismatched types"),
            memory=memory,
        )
    with pytest.raises(ValueError):
        _prepare(
            attempt=attempt,
            stderr=stderr,
            memory=memory,
            source=SOURCE.replace("solve", "salve"),
        )

    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)
    forged = message_line.project_rustc_message_line_unordered_cue_observation_v1(
        attempt,
        stderr=stderr,
        source=SOURCE,
        selection_context_sha256=_selection_context("forged"),
    )
    prepared._message_line_envelope = forged
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="custody changed",
    ):
        _ = prepared.projected_observation

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        authority.prepare_rustc_message_line_unordered_cue_text_edit_authority_v1(
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


def test_checkpoint_observation_and_bank_mismatch_stop_before_work() -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)

    wrong_checkpoint = _prepare(
        attempt=attempt,
        stderr=stderr,
        memory=memory,
        expected_checkpoint_sha256=_root("wrong-checkpoint"),
    )
    checkpoint_decision = _decision(memory, wrong_checkpoint.projected_observation)
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="checkpoint",
    ):
        wrong_checkpoint.execute_selected(checkpoint_decision, ((1,),))
    assert wrong_checkpoint.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    observation_bound = _prepare(attempt=attempt, stderr=stderr, memory=memory)
    other_context = _prepare(
        attempt=attempt,
        stderr=stderr,
        memory=memory,
        selection_context_sha256=_selection_context("other"),
    )
    observation_decision = _decision(memory, other_context.projected_observation)
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="observation custody",
    ):
        observation_bound.execute_selected(observation_decision, ((1,),))
    assert observation_bound.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    foreign_target = _root("c10-target:foreign")
    bad_bank = tuple(
        factors.seal_factorized_hypothesis_v1((action_root, target_root))
        for action_root in ACTION_ROOTS
        for target_root in (TARGET_ROOTS[0], foreign_target)
    )
    bad_memory = unordered.start_unordered_cue_factor_memory_v1(bad_bank)
    bank_bound = _prepare(attempt=attempt, stderr=stderr, memory=bad_memory)
    bank_decision = _decision(
        bad_memory,
        bank_bound.projected_observation,
        bank=bad_bank,
        target_root=TARGET_ROOTS[0],
    )
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="bank",
    ):
        bank_bound.execute_selected(bank_decision, ((1,),))
    assert bank_bound.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)


def test_exact_sole_primary_span_custody_is_required() -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)
    envelope = prepared.message_line_envelope

    alternate_span = ascii_cues.seal_ascii_byte_span_v1(
        SOURCE_BYTES,
        byte_start=PRIMARY_START + 1,
        byte_end=PRIMARY_END,
    )
    alternate_line = ascii_cues.project_ascii_word_unigram_cues_v1(
        SOURCE_BYTES,
        span=alternate_span,
    )
    forged = replace(envelope, source_line_envelope=alternate_line)
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="primary-span custody changed",
    ):
        authority._sole_custodied_primary_span(forged)

    other_start = SOURCE_BYTES.index(b"_first_anchor")
    multiple_stderr = _stderr(
        primary_spans=(
            (PRIMARY_START, PRIMARY_END),
            (other_start, other_start + len(b"_first_anchor")),
        )
    )
    with pytest.raises(ValueError, match=r"custody|envelope derivation failed"):
        _prepare(
            attempt=_attempt(multiple_stderr),
            stderr=multiple_stderr,
            memory=memory,
        )
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)


def test_stop_and_forbidden_exploration_never_materialize() -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)

    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)
    stop = memory.route(prepared.projected_observation)
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="selected",
    ):
        prepared.execute_selected(stop, ((1,),))
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)

    forbidden = _prepare(
        attempt=attempt,
        stderr=stderr,
        memory=memory,
        allow_exploration=False,
    )
    exploration = _decision(memory, forbidden.projected_observation)
    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="forbids exploration",
    ):
        forbidden.execute_selected(exploration, ((1,),))
    assert forbidden.work_counters == authority.UnorderedCueFactorTextEditWorkV1(0, 0, 0, 0)


def test_selected_decision_materializes_exact_span_executes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr = _stderr()
    attempt = _attempt(stderr)
    memory = unordered.start_unordered_cue_factor_memory_v1(BANK)
    prepared = _prepare(attempt=attempt, stderr=stderr, memory=memory)
    decision = _decision(memory, prepared.projected_observation)
    fake_execution = _install_fake_execution(monkeypatch)

    result = prepared.execute_selected(decision, ((1,),))
    expected_source = f"{SOURCE[:PRIMARY_START]}_first_anchor{SOURCE[PRIMARY_END:]}"
    primary_span = authority._sole_custodied_primary_span(prepared.message_line_envelope)

    assert type(result) is fake_execution
    assert result.selected_source.source == expected_source
    assert result.selected_source.selection_sha256 == decision.sha256
    assert result.selected_source.authority_sha256 == prepared.authority_sha256
    assert prepared.materialization is not None
    assert prepared.release is not None
    assert type(prepared.materialization) is authority.UnorderedCueFactorTextEditMaterializationV1
    assert type(prepared.release) is authority.UnorderedCueFactorTextEditReleaseV1
    assert prepared.materialization.primary_span_sha256 == primary_span.sha256
    assert prepared.materialization.edit.byte_range == (PRIMARY_START, PRIMARY_END)
    assert (
        prepared.materialization.original_source_sha256
        == prepared.message_line_envelope.source_line_envelope.span.source_sha256
    )
    assert prepared.materialization.decision_sha256 == decision.sha256
    assert prepared.materialization.observation_sha256 == prepared.projected_observation.sha256
    assert prepared.release.selection_sha256 == decision.sha256
    assert prepared.release.sha256 == result.selected_source.release_sha256
    prepared.materialization.authenticate()
    prepared.release.authenticate()
    assert prepared.work_counters == authority.UnorderedCueFactorTextEditWorkV1(1, 1, 1, 1)

    with pytest.raises(
        authority.RustcMessageLineUnorderedCueTextEditAuthorityError,
        match="consumed",
    ):
        prepared.execute_selected(decision, ((1,),))
