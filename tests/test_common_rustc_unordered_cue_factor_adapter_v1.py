from __future__ import annotations

import json
from hashlib import sha256

from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as ordered
from experiments.r3_pilot.common_rustc_unordered_cue_factor_adapter_v1 import (
    project_rust_unordered_cue_factor_observation_v1,
)

SOURCE_FILE = "c8_unordered_fixture.rs"
SOURCE = b"x" * 100
ARGUMENTS = ("--crate-type", "lib", "--error-format=json", SOURCE_FILE)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _diagnostic(code: str, byte_start: int, byte_end: int) -> dict[str, object]:
    return {
        "$message_type": "diagnostic",
        "message": "environment-only diagnostic text",
        "code": {"code": code, "explanation": "environment-only explanation"},
        "level": "error",
        "spans": [
            {
                "file_name": SOURCE_FILE,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "is_primary": True,
            }
        ],
    }


def _attempt() -> ordered.RustcDiagnosticCompileAttemptV1:
    stderr = b"".join(
        json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
        for record in (
            _diagnostic("E0425", 8, 12),
            _diagnostic("E0308", 74, 82),
        )
    )
    return ordered.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("rustc-identity"),
        source_file_name=SOURCE_FILE,
        source=SOURCE,
        arguments=ARGUMENTS,
        returncode=1,
        stdout=b"",
        stderr=stderr,
    )


def test_flattens_both_extracted_axes_into_one_canonical_opaque_set() -> None:
    attempt = _attempt()
    context = _root("selection-context")
    active = (_root("hypothesis-b"), _root("hypothesis-a"))
    extracted = ordered.project_rust_factorized_feedback_v1(
        attempt,
        selection_context_sha256=context,
        active_hypothesis_sha256s=active,
    )

    observation = project_rust_unordered_cue_factor_observation_v1(
        attempt,
        selection_context_sha256=context,
        active_hypothesis_sha256s=active,
    )
    observation.authenticate()

    assert observation.selection_context_sha256 == extracted.selection_context_sha256
    assert observation.cue_sha256s == tuple(
        sorted(cue for axis in extracted.axis_cue_sha256s for cue in axis)
    )
    assert len(observation.cue_sha256s) == 4
    assert not hasattr(observation, "axis_cue_sha256s")
    assert not hasattr(observation, "active_hypothesis_sha256s")


def test_active_hypothesis_membership_is_not_retained_in_cognition() -> None:
    attempt = _attempt()
    context = _root("selection-context")

    first = project_rust_unordered_cue_factor_observation_v1(
        attempt,
        selection_context_sha256=context,
        active_hypothesis_sha256s=(_root("hypothesis-a"),),
    )
    second = project_rust_unordered_cue_factor_observation_v1(
        attempt,
        selection_context_sha256=context,
        active_hypothesis_sha256s=(_root("hypothesis-c"), _root("hypothesis-b")),
    )

    assert first == second
    assert first.sha256 == second.sha256
