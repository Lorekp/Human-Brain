from __future__ import annotations

import json
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as ordered
from experiments.r3_pilot.common_rustc_lexical_unordered_cue_adapter_v1 import (
    MAX_LEXICAL_FEATURES,
    RustcLexicalUnorderedCueAdapterError,
    project_rustc_lexical_unordered_cue_observation_v1,
)

SOURCE_FILE = "c9_lexical_fixture.rs"
SOURCE = b"x" * 100
ARGUMENTS = ("--crate-type", "lib", "--error-format=json", SOURCE_FILE)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _diagnostic(
    *,
    code: str,
    message: str,
    byte_start: int = 8,
    byte_end: int = 12,
    child_message: str = "child_fix_secret",
    rendered: str = "rendered_fix_secret",
    replacement: str = "replacement_fix_secret",
) -> dict[str, object]:
    return {
        "$message_type": "diagnostic",
        "message": message,
        "code": {"code": code, "explanation": "code_explanation_secret"},
        "level": "error",
        "spans": [
            {
                "file_name": SOURCE_FILE,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "is_primary": True,
                "label": "span_label_secret",
                "suggested_replacement": replacement,
            }
        ],
        "children": [
            {
                "message": child_message,
                "level": "help",
                "spans": [
                    {
                        "file_name": SOURCE_FILE,
                        "byte_start": byte_start,
                        "byte_end": byte_end,
                        "is_primary": False,
                        "suggested_replacement": replacement,
                    }
                ],
            }
        ],
        "rendered": rendered,
    }


def _stderr(*records: dict[str, object]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True).encode("utf-8") + b"\n" for record in records
    )


def _attempt(stderr: bytes) -> ordered.RustcDiagnosticCompileAttemptV1:
    return ordered.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("rustc-identity"),
        source_file_name=SOURCE_FILE,
        source=SOURCE,
        arguments=ARGUMENTS,
        returncode=1,
        stdout=b"",
        stderr=stderr,
    )


def _project(
    stderr: bytes,
    *,
    active: tuple[str, ...] = (_root("hypothesis-a"),),
):
    return project_rustc_lexical_unordered_cue_observation_v1(
        _attempt(stderr),
        stderr=stderr,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=active,
    )


def test_code_changes_with_identical_message_do_not_change_cognition() -> None:
    message = "cannot find value `private_source_identifier` in this scope"
    first = _project(_stderr(_diagnostic(code="E0425", message=message)))
    second = _project(_stderr(_diagnostic(code="E0308", message=message)))

    first.authenticate()
    second.authenticate()
    assert first.attempt.sha256 != second.attempt.sha256
    assert first.lexical_tokens == (
        "cannot",
        "find",
        "identifier",
        "in",
        "private",
        "scope",
        "source",
        "this",
        "value",
    )
    assert first.lexical_cue_sha256s == second.lexical_cue_sha256s
    assert first.region_cue_sha256s == second.region_cue_sha256s
    assert first.observation == second.observation


def test_parent_message_change_changes_lexical_cues() -> None:
    first = _project(_stderr(_diagnostic(code="E0425", message="cannot find value")))
    second = _project(_stderr(_diagnostic(code="E0425", message="mismatched types")))

    assert first.region_cue_sha256s == second.region_cue_sha256s
    assert first.lexical_cue_sha256s != second.lexical_cue_sha256s
    assert first.observation != second.observation


def test_child_rendered_and_replacement_payloads_are_provenance_only() -> None:
    common = {"code": "E0425", "message": "cannot find value"}
    first = _project(
        _stderr(
            _diagnostic(
                **common,
                child_message="first_child_fix",
                rendered="first_rendered_fix",
                replacement="first_replacement_fix",
            )
        )
    )
    second = _project(
        _stderr(
            _diagnostic(
                **common,
                child_message="second_child_fix",
                rendered="second_rendered_fix",
                replacement="second_replacement_fix",
            )
        )
    )

    assert first.attempt.stderr_sha256 != second.attempt.stderr_sha256
    assert first.sha256 != second.sha256
    assert first.lexical_tokens == ("cannot", "find", "value")
    assert first.lexical_cue_sha256s == second.lexical_cue_sha256s
    assert first.observation == second.observation


def test_raw_stderr_digest_mismatch_is_rejected() -> None:
    authenticated_stderr = _stderr(_diagnostic(code="E0425", message="cannot find value"))
    other_stderr = _stderr(_diagnostic(code="E0425", message="mismatched types"))

    with pytest.raises(
        RustcLexicalUnorderedCueAdapterError,
        match="does not match the authenticated attempt",
    ):
        project_rustc_lexical_unordered_cue_observation_v1(
            _attempt(authenticated_stderr),
            stderr=other_stderr,
            selection_context_sha256=_root("selection-context"),
            active_hypothesis_sha256s=(_root("hypothesis-a"),),
        )


def test_active_hypothesis_membership_is_absent_from_envelope_and_cognition() -> None:
    stderr = _stderr(_diagnostic(code="E0425", message="cannot find value"))
    first = _project(stderr, active=(_root("hypothesis-a"),))
    second = _project(
        stderr,
        active=(_root("hypothesis-c"), _root("hypothesis-b")),
    )

    assert first == second
    assert first.observation == second.observation
    assert "active_hypothesis_sha256s" not in first.record()
    assert not hasattr(first.observation, "active_hypothesis_sha256s")


def test_code_cue_is_absent_while_exact_ordered_region_cue_is_retained() -> None:
    stderr = _stderr(_diagnostic(code="E0425", message="cannot find value"))
    attempt = _attempt(stderr)
    active = (_root("hypothesis-a"),)
    ordered_observation = ordered.project_rust_factorized_feedback_v1(
        attempt,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=active,
    )
    envelope = project_rustc_lexical_unordered_cue_observation_v1(
        attempt,
        stderr=stderr,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=active,
    )

    assert envelope.region_cue_sha256s == ordered_observation.axis_cue_sha256s[1]
    assert set(ordered_observation.axis_cue_sha256s[0]).isdisjoint(envelope.observation.cue_sha256s)
    assert envelope.observation.cue_sha256s == tuple(
        sorted((*envelope.lexical_cue_sha256s, *envelope.region_cue_sha256s))
    )


def test_lexical_feature_budget_fails_closed_instead_of_truncating() -> None:
    message = " ".join(f"word{index}" for index in range(MAX_LEXICAL_FEATURES + 1))

    with pytest.raises(
        RustcLexicalUnorderedCueAdapterError,
        match="lexical feature budget exceeded",
    ):
        _project(_stderr(_diagnostic(code="E0425", message=message)))
