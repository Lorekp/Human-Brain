from __future__ import annotations

import inspect
import json
from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_ascii_word_unigram_cues_v1 as ascii_cues
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as ordered
from experiments.r3_pilot.common_rustc_message_line_unordered_cue_adapter_v1 import (
    MAX_TOTAL_CUES,
    RustcMessageLineUnorderedCueAdapterError,
    project_rustc_message_line_unordered_cue_observation_v1,
)

SOURCE_FILE = "c10_message_line_fixture.rs"
SOURCE = "fn demo() {\n    let shared_line = missing_value;\n}"
PRIMARY_TEXT = "missing_value"
PRIMARY_START = SOURCE.encode("ascii").index(PRIMARY_TEXT.encode("ascii"))
PRIMARY_END = PRIMARY_START + len(PRIMARY_TEXT)
ARGUMENTS = ("--crate-type", "lib", "--error-format=json", SOURCE_FILE)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _span(byte_start: int, byte_end: int) -> dict[str, object]:
    return {
        "file_name": SOURCE_FILE,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "is_primary": True,
        "label": "diagnostic_span_label_secret",
        "suggested_replacement": "diagnostic_replacement_secret",
    }


def _diagnostic(
    *,
    message: str,
    code: str = "E0425",
    primary_spans: tuple[tuple[int, int], ...] = ((PRIMARY_START, PRIMARY_END),),
) -> dict[str, object]:
    return {
        "$message_type": "diagnostic",
        "message": message,
        "code": {"code": code, "explanation": "code_explanation_secret"},
        "level": "error",
        "spans": [_span(start, end) for start, end in primary_spans],
        "children": [
            {
                "message": "child_suggestion_secret",
                "level": "help",
                "spans": [],
            }
        ],
        "rendered": "rendered_diagnostic_secret",
    }


def _stderr(*records: dict[str, object]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True).encode("utf-8") + b"\n" for record in records
    )


def _attempt(
    stderr: bytes,
    *,
    source: str = SOURCE,
) -> ordered.RustcDiagnosticCompileAttemptV1:
    return ordered.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=_root("rustc-identity"),
        source_file_name=SOURCE_FILE,
        source=source.encode("ascii"),
        arguments=ARGUMENTS,
        returncode=1,
        stdout=b"",
        stderr=stderr,
    )


def _project(
    *,
    message: str = "cannot find value `shared_line` in this scope",
    source: str = SOURCE,
):
    stderr = _stderr(_diagnostic(message=message))
    return project_rustc_message_line_unordered_cue_observation_v1(
        _attempt(stderr, source=source),
        stderr=stderr,
        source=source,
        selection_context_sha256=_root("selection-context"),
    )


def test_successful_reauthentication_and_shared_tokens_collapse() -> None:
    envelope = _project()

    envelope.authenticate()
    assert envelope.message_tokens == (
        "cannot",
        "find",
        "in",
        "line",
        "scope",
        "shared",
        "this",
        "value",
    )
    assert envelope.source_line_tokens == (
        "let",
        "line",
        "missing",
        "shared",
        "value",
    )
    assert envelope.source_line_envelope.line == b"    let shared_line = missing_value;"
    assert envelope.source_line_envelope.span.span_bytes == b"missing_value"

    distinct_tokens = set((*envelope.message_tokens, *envelope.source_line_tokens))
    assert {"line", "shared", "value"}.issubset(
        set(envelope.message_tokens).intersection(envelope.source_line_tokens)
    )
    assert len(envelope.observation.cue_sha256s) == len(distinct_tokens)
    assert len(envelope.observation.cue_sha256s) < (
        len(envelope.message_cue_sha256s) + len(envelope.source_line_cue_sha256s)
    )
    assert envelope.observation.cue_sha256s == tuple(
        sorted(ascii_cues.ascii_word_unigram_cue_sha256_v1(token) for token in distinct_tokens)
    )


def test_projection_has_no_active_hypothesis_or_provider_role_input() -> None:
    parameters = inspect.signature(
        project_rustc_message_line_unordered_cue_observation_v1
    ).parameters
    envelope = _project()

    assert tuple(parameters) == (
        "attempt",
        "stderr",
        "source",
        "selection_context_sha256",
    )
    assert "active_hypothesis_sha256s" not in parameters
    assert "active_hypothesis_sha256s" not in envelope.record()
    assert not hasattr(envelope.observation, "active_hypothesis_sha256s")


def test_mutated_envelope_source_span_and_stderr_fail_closed() -> None:
    envelope = _project()
    mutated_inventory = replace(envelope, message_tokens=("mutated",))
    mutated_span = replace(
        envelope.source_line_envelope.span,
        byte_start=envelope.source_line_envelope.span.byte_start + 1,
    )
    mutated_line = replace(envelope.source_line_envelope, span=mutated_span)
    mutated_span_envelope = replace(envelope, source_line_envelope=mutated_line)

    with pytest.raises(
        RustcMessageLineUnorderedCueAdapterError,
        match="envelope authentication failed",
    ):
        mutated_inventory.authenticate()
    with pytest.raises(
        RustcMessageLineUnorderedCueAdapterError,
        match="envelope authentication failed",
    ):
        mutated_span_envelope.authenticate()

    stderr = _stderr(_diagnostic(message="cannot find value"))
    attempt = _attempt(stderr)
    with pytest.raises(
        RustcMessageLineUnorderedCueAdapterError,
        match="source does not match",
    ):
        project_rustc_message_line_unordered_cue_observation_v1(
            attempt,
            stderr=stderr,
            source=SOURCE.replace("demo", "dome"),
            selection_context_sha256=_root("selection-context"),
        )

    other_stderr = _stderr(_diagnostic(message="mismatched types"))
    with pytest.raises(
        RustcMessageLineUnorderedCueAdapterError,
        match="attempt and stderr authentication failed",
    ):
        project_rustc_message_line_unordered_cue_observation_v1(
            attempt,
            stderr=other_stderr,
            source=SOURCE,
            selection_context_sha256=_root("selection-context"),
        )


@pytest.mark.parametrize(
    "records",
    (
        (
            _diagnostic(message="cannot find value"),
            _diagnostic(message="mismatched types", code="E0308"),
        ),
        (
            _diagnostic(
                message="cannot find value",
                primary_spans=(
                    (PRIMARY_START, PRIMARY_END),
                    (SOURCE.index("shared_line"), SOURCE.index("shared_line") + len("shared_line")),
                ),
            ),
        ),
    ),
)
def test_exactly_one_error_and_one_primary_span_are_required(
    records: tuple[dict[str, object], ...],
) -> None:
    stderr = _stderr(*records)

    with pytest.raises(
        RustcMessageLineUnorderedCueAdapterError,
        match=r"exactly one (retained error diagnostic|primary span)",
    ):
        project_rustc_message_line_unordered_cue_observation_v1(
            _attempt(stderr),
            stderr=stderr,
            source=SOURCE,
            selection_context_sha256=_root("selection-context"),
        )


def test_merged_cue_budget_rejects_instead_of_truncating() -> None:
    source = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    line_tokens = tuple(source.split())
    message_tokens = (
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
    )
    stderr = _stderr(
        _diagnostic(
            message=" ".join(message_tokens),
            primary_spans=((0, len(line_tokens[0])),),
        )
    )

    assert len(set((*line_tokens, *message_tokens))) > MAX_TOTAL_CUES
    with pytest.raises(
        RustcMessageLineUnorderedCueAdapterError,
        match="merged message-line cue budget exceeded",
    ):
        project_rustc_message_line_unordered_cue_observation_v1(
            _attempt(stderr, source=source),
            stderr=stderr,
            source=source,
            selection_context_sha256=_root("selection-context"),
        )


def test_cues_use_only_generic_token_roots_not_code_or_region_roots() -> None:
    envelope = _project()
    attempt = envelope.attempt
    active = (_root("hypothesis-a"),)
    ordered_observation = ordered.project_rust_factorized_feedback_v1(
        attempt,
        selection_context_sha256=_root("selection-context"),
        active_hypothesis_sha256s=active,
    )
    code_root = ordered._class_cue_sha256(attempt, "E0425")

    assert envelope.message_cue_sha256s == tuple(
        ascii_cues.ascii_word_unigram_cue_sha256_v1(token) for token in envelope.message_tokens
    )
    assert envelope.source_line_cue_sha256s == tuple(
        ascii_cues.ascii_word_unigram_cue_sha256_v1(token) for token in envelope.source_line_tokens
    )
    assert code_root not in envelope.observation.cue_sha256s
    assert set(ordered_observation.axis_cue_sha256s[1]).isdisjoint(envelope.observation.cue_sha256s)
    assert not hasattr(envelope, "code_cue_sha256s")
    assert not hasattr(envelope, "region_cue_sha256s")
