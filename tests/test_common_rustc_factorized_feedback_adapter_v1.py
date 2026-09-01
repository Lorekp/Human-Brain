from __future__ import annotations

import json
from hashlib import sha256

import pytest
from experiments.r3_pilot.common_factorized_feedback_hypothesis_memory_v1 import (
    seal_factorized_hypothesis_v1,
)
from experiments.r3_pilot.common_rustc_factorized_feedback_adapter_v1 import (
    COMPILE_FAILED,
    COMPILED,
    NO_CODE,
    RustcFactorizedFeedbackAdapterError,
    project_rust_factorized_feedback_v1,
    seal_rustc_diagnostic_compile_attempt_v1,
)

SOURCE_FILE = "case.rs"
SOURCE = b"x" * 100
ARGUMENTS = ("--crate-type", "lib", "--error-format=json", SOURCE_FILE)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _bank_roots() -> tuple[str, ...]:
    axis_zero = (_root("component:class:0"), _root("component:class:1"))
    axis_one = (_root("component:region:0"), _root("component:region:1"))
    return tuple(
        hypothesis.sha256
        for hypothesis in (
            seal_factorized_hypothesis_v1((axis_zero[0], axis_one[0])),
            seal_factorized_hypothesis_v1((axis_zero[0], axis_one[1])),
            seal_factorized_hypothesis_v1((axis_zero[1], axis_one[0])),
            seal_factorized_hypothesis_v1((axis_zero[1], axis_one[1])),
        )
    )


def _span(
    byte_start: int,
    byte_end: int,
    *,
    file_name: str = SOURCE_FILE,
    label: str = "diagnostic label",
    suggested_replacement: str | None = "host-proposed-fix",
) -> dict[str, object]:
    return {
        "file_name": file_name,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "line_start": 1,
        "line_end": 1,
        "column_start": byte_start + 1,
        "column_end": byte_end + 1,
        "is_primary": True,
        "text": [{"text": "secret source excerpt", "highlight_start": 1}],
        "label": label,
        "suggested_replacement": suggested_replacement,
        "suggestion_applicability": "MachineApplicable",
        "future_span_field": {"ignored": True},
    }


def _error(
    code: str | None,
    spans: list[dict[str, object]],
    *,
    message: str = "human diagnostic message",
    rendered: str = "rendered repair guidance",
) -> dict[str, object]:
    return {
        "$message_type": "diagnostic",
        "message": message,
        "code": None if code is None else {"code": code, "explanation": "secret"},
        "level": "error",
        "spans": spans,
        "children": [
            {
                "message": "try the exact answer",
                "level": "help",
                "spans": [],
            }
        ],
        "rendered": rendered,
        "future_diagnostic_field": ["ignored"],
    }


def _stderr(*records: dict[str, object]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True).encode("utf-8") + b"\n" for record in records
    )


def _attempt(
    stderr: bytes,
    *,
    returncode: int = 1,
    compile_invocations: int = 1,
    arguments: tuple[str, ...] = ARGUMENTS,
):
    return seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("rustc-identity"),
        source_file_name=SOURCE_FILE,
        source=SOURCE,
        arguments=arguments,
        returncode=returncode,
        stdout=b"",
        stderr=stderr,
        compile_invocations=compile_invocations,
    )


def test_projects_all_parent_error_codes_and_selected_source_regions() -> None:
    stderr = _stderr(
        _error(
            "E0425",
            [
                _span(8, 12),
                _span(0, 900, file_name="dependency.rs"),
            ],
        ),
        _error(None, [_span(74, 82)]),
        {
            "$message_type": "diagnostic",
            "message": "warning ignored",
            "code": {"code": "unused"},
            "level": "warning",
            "spans": [_span(40, 41)],
        },
        {"$message_type": "future-record", "payload": "ignored"},
    )
    attempt = _attempt(stderr)

    assert attempt.status == COMPILE_FAILED
    assert {diagnostic.code for diagnostic in attempt.error_diagnostics} == {
        "E0425",
        NO_CODE,
    }
    assert {
        (span.file_name, span.byte_start, span.byte_end)
        for diagnostic in attempt.error_diagnostics
        for span in diagnostic.primary_spans
    } == {(SOURCE_FILE, 8, 12), (SOURCE_FILE, 74, 82)}

    candidates = _bank_roots()
    observation = project_rust_factorized_feedback_v1(
        attempt,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=tuple(reversed(candidates)),
    )
    observation.authenticate()

    assert tuple(map(len, observation.axis_cue_sha256s)) == (2, 2)
    assert observation.active_hypothesis_sha256s == tuple(sorted(candidates))


def test_fix_bearing_text_changes_audit_root_but_not_cognitive_cues() -> None:
    first = _attempt(
        _stderr(
            _error(
                "E0308",
                [_span(60, 64, label="first label", suggested_replacement="first fix")],
                message="first message",
                rendered="first rendering",
            )
        )
    )
    second = _attempt(
        _stderr(
            _error(
                "E0308",
                [_span(60, 64, label="second label", suggested_replacement="second fix")],
                message="second message",
                rendered="second rendering",
            )
        )
    )
    candidates = _bank_roots()
    first_observation = project_rust_factorized_feedback_v1(
        first,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=candidates,
    )
    second_observation = project_rust_factorized_feedback_v1(
        second,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=candidates,
    )

    assert first.stderr_sha256 != second.stderr_sha256
    assert first.sha256 != second.sha256
    assert first_observation.selection_context_sha256 != (
        second_observation.selection_context_sha256
    )
    assert first_observation.axis_cue_sha256s == second_observation.axis_cue_sha256s

    changed_policy = _attempt(
        _stderr(_error("E0308", [_span(60, 64)])),
        arguments=(*ARGUMENTS, "--cfg", "adapter_policy_variant"),
    )
    changed_policy_observation = project_rust_factorized_feedback_v1(
        changed_policy,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=candidates,
    )
    assert changed_policy_observation.axis_cue_sha256s[0] != (first_observation.axis_cue_sha256s[0])
    assert changed_policy_observation.axis_cue_sha256s[1] != (first_observation.axis_cue_sha256s[1])


def test_empty_uncoded_abort_summary_does_not_create_a_class_cue() -> None:
    substantive = _error("E0308", [_span(60, 64)])
    without_summary = _attempt(_stderr(substantive))
    with_summary = _attempt(
        _stderr(
            substantive,
            _error(
                None,
                [],
                message="aborting due to previous error",
                rendered="error: aborting due to previous error",
            ),
        )
    )
    candidates = _bank_roots()
    without_observation = project_rust_factorized_feedback_v1(
        without_summary,
        selection_context_sha256=_root("summary-context"),
        active_hypothesis_sha256s=candidates,
    )
    with_observation = project_rust_factorized_feedback_v1(
        with_summary,
        selection_context_sha256=_root("summary-context"),
        active_hypothesis_sha256s=candidates,
    )

    assert without_summary.sha256 != with_summary.sha256
    assert without_observation.selection_context_sha256 != (
        with_observation.selection_context_sha256
    )
    assert without_observation.axis_cue_sha256s == with_observation.axis_cue_sha256s

    genuine_uncoded = _attempt(_stderr(_error(None, [_span(20, 24)])))
    assert {diagnostic.code for diagnostic in genuine_uncoded.error_diagnostics} == {NO_CODE}


def test_terminal_attempt_states_and_unlocated_projection_are_explicit() -> None:
    compiled = _attempt(b"", returncode=0)
    compiled.authenticate()
    assert compiled.status == COMPILED
    with pytest.raises(RustcFactorizedFeedbackAdapterError):
        project_rust_factorized_feedback_v1(
            compiled,
            selection_context_sha256=_root("compiled-context"),
            active_hypothesis_sha256s=_bank_roots(),
        )

    with pytest.raises(RustcFactorizedFeedbackAdapterError):
        _attempt(b"", returncode=1)
    with pytest.raises(RustcFactorizedFeedbackAdapterError):
        _attempt(_stderr(_error("E0425", [_span(90, 101)])))
    with pytest.raises(RustcFactorizedFeedbackAdapterError):
        _attempt(_stderr(_error("E0425", [])), compile_invocations=True)

    unlocated = _attempt(_stderr(_error("E0425", [])))
    observation = project_rust_factorized_feedback_v1(
        unlocated,
        selection_context_sha256=_root("unlocated-context"),
        active_hypothesis_sha256s=_bank_roots(),
    )
    assert tuple(map(len, observation.axis_cue_sha256s)) == (1, 1)
