from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot.common_factorized_text_edit_materializer_v1 import (
    DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR,
    REPLACE_PRIMARY_SPAN_WITH_ANCHOR,
    PreparedFactorizedTextEditMaterializerV1,
)
from experiments.r3_pilot.common_factorized_text_edit_operator_catalogue_v1 import (
    DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR_ACTION_V1,
    REPLACE_PRIMARY_SPAN_WITH_ANCHOR_ACTION_V1,
    FactorizedTextEditOperatorCatalogueError,
    canonical_factorized_text_edit_operator_action_roots_v1,
    canonical_factorized_text_edit_operator_actions_v1,
    canonical_factorized_text_edit_operator_bindings_v1,
    prepare_factorized_text_edit_operator_catalogue_v1,
)

SOURCE_FILE = "catalogue_case.rs"
SOURCE = """pub fn solve() -> i64 {
    let _first_anchor = 7_i64;
    let _second_anchor = 9_i64;
    _missing_value
}
"""


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


TARGET_BINDINGS = (
    (_root("catalogue-target:first"), "_first_anchor"),
    (_root("catalogue-target:second"), "_second_anchor"),
)


def _attempt() -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
    source = SOURCE.encode("ascii")
    byte_start = source.index(b"_missing_value")
    diagnostic = {
        "$message_type": "diagnostic",
        "message": "cannot find value",
        "code": {"code": "E0425", "explanation": "unused"},
        "level": "error",
        "spans": [
            {
                "file_name": SOURCE_FILE,
                "byte_start": byte_start,
                "byte_end": byte_start + len(b"_missing_value"),
                "line_start": 4,
                "line_end": 4,
                "column_start": 5,
                "column_end": 19,
                "is_primary": True,
                "text": [],
                "label": "unused",
                "suggested_replacement": "unused",
                "suggestion_applicability": "MachineApplicable",
            }
        ],
        "children": [],
        "rendered": "unused",
    }
    return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("catalogue-rustc"),
        source_file_name=SOURCE_FILE,
        source=source,
        arguments=("--crate-type", "lib", "--error-format=json", SOURCE_FILE),
        returncode=1,
        stdout=b"",
        stderr=json.dumps(diagnostic, sort_keys=True).encode("utf-8") + b"\n",
        compile_invocations=1,
    )


def test_catalogue_actions_are_distinct_authenticated_roots() -> None:
    actions = canonical_factorized_text_edit_operator_actions_v1()
    roots = canonical_factorized_text_edit_operator_action_roots_v1()
    bindings = canonical_factorized_text_edit_operator_bindings_v1()

    assert set(actions) == {
        DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR_ACTION_V1,
        REPLACE_PRIMARY_SPAN_WITH_ANCHOR_ACTION_V1,
    }
    assert roots == tuple(action.sha256 for action in actions)
    assert bindings == tuple((action.sha256, action.operation) for action in actions)
    assert len(set(roots)) == 2
    assert {operation for _root_value, operation in bindings} == {
        DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR,
        REPLACE_PRIMARY_SPAN_WITH_ANCHOR,
    }
    for action in actions:
        action.authenticate()

    changed = replace(actions[0], operation=actions[1].operation)
    with pytest.raises(FactorizedTextEditOperatorCatalogueError, match="authentication"):
        changed.authenticate()


def test_factory_returns_exact_existing_authority_with_sealed_bindings() -> None:
    attempt = _attempt()
    checkpoint_sha256 = _root("catalogue-checkpoint")
    selection_context = _root("catalogue-selection-context")

    authority = prepare_factorized_text_edit_operator_catalogue_v1(
        diagnostic_attempt=attempt,
        source=SOURCE,
        selection_context_sha256=selection_context,
        expected_checkpoint_sha256=checkpoint_sha256,
        allow_exploration=True,
        target_bindings=tuple(reversed(TARGET_BINDINGS)),
    )

    assert type(authority) is PreparedFactorizedTextEditMaterializerV1
    assert authority.receipt.operator_bindings == (
        canonical_factorized_text_edit_operator_bindings_v1()
    )
    assert authority.receipt.target_bindings == tuple(sorted(TARGET_BINDINGS))
    assert authority.receipt.expected_checkpoint_sha256 == checkpoint_sha256
    assert authority.receipt.allow_exploration is True
    assert (authority.materialization_count, authority.release_count) == (0, 0)
