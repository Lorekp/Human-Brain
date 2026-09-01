"""Post-selection text-edit materialization for factorized Rust hypotheses.

The authority binds opaque factor-component roots to a fixed two-operation,
two-target host catalogue.  It keeps the failing Rust source under custody but
does not compose an edit or a successor source until an authenticated physical
hypothesis decision is passed to :meth:`execute_selected`.

Rustc messages, rendered diagnostics, labels, and suggestions are unavailable
to this module.  Only the authenticated error primary span is used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from typing import TypeVar

from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as outcomes
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "COMMON_FACTORIZED_TEXT_EDIT_MATERIALIZER_V1"
CLAIM_BOUNDARY = (
    "post-selection-composition-of-one-host-supplied-text-edit-operation-and-target-"
    "against-one-authenticated-rustc-primary-span-not-message-or-suggestion-use-"
    "operator-or-target-discovery-diagnosis-correctness-search-repository-debugging-"
    "swe-or-intelligence"
)

DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR = "DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR"
REPLACE_PRIMARY_SPAN_WITH_ANCHOR = "REPLACE_PRIMARY_SPAN_WITH_ANCHOR"

_OPERATIONS = frozenset((DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR, REPLACE_PRIMARY_SPAN_WITH_ANCHOR))
_SHA256_HEX = frozenset("0123456789abcdef")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ACTION_HANDLE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")

_AUTHORITY_NAMESPACE = b"common-factorized-text-edit-authority-v1"
_EDIT_NAMESPACE = b"common-factorized-text-edit-v1"
_MATERIALIZATION_NAMESPACE = b"common-factorized-text-edit-materialization-v1"
_RELEASE_NAMESPACE = b"common-factorized-text-edit-release-v1"

_CONSTRUCTION_TOKEN = object()
_T = TypeVar("_T")


class FactorizedTextEditMaterializerError(ValueError):
    """A binding, selection, diagnostic, materialization, or release changed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _canonical_operator_bindings(
    bindings: object,
) -> tuple[tuple[str, str], tuple[str, str]]:
    if type(bindings) is not tuple or len(bindings) != 2:
        raise TypeError("operator bindings must be one exact two-row tuple")
    if any(
        type(row) is not tuple
        or len(row) != 2
        or not _valid_sha256(row[0])
        or row[1] not in _OPERATIONS
        for row in bindings
    ):
        raise FactorizedTextEditMaterializerError("operator bindings are invalid")
    result = tuple(sorted(bindings))
    if (
        len({root for root, _operation in result}) != 2
        or {operation for _root, operation in result} != _OPERATIONS
    ):
        raise FactorizedTextEditMaterializerError(
            "operator bindings must bijectively bind both operations"
        )
    return result  # type: ignore[return-value]


def _canonical_target_bindings(
    bindings: object,
) -> tuple[tuple[str, str], tuple[str, str]]:
    if type(bindings) is not tuple or len(bindings) != 2:
        raise TypeError("target bindings must be one exact two-row tuple")
    if any(
        type(row) is not tuple
        or len(row) != 2
        or not _valid_sha256(row[0])
        or type(row[1]) is not str
        or _IDENTIFIER.fullmatch(row[1]) is None
        for row in bindings
    ):
        raise FactorizedTextEditMaterializerError("target bindings are invalid")
    result = tuple(sorted(bindings))
    if (
        len({root for root, _marker in result}) != 2
        or len({marker for _root, marker in result}) != 2
    ):
        raise FactorizedTextEditMaterializerError(
            "target bindings must bijectively bind two markers"
        )
    return result  # type: ignore[return-value]


def _source_bytes(
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    source: str,
) -> bytes:
    if type(attempt) is not rustc_feedback.RustcDiagnosticCompileAttemptV1:
        raise TypeError("one exact rustc diagnostic attempt is required")
    attempt.authenticate()
    if attempt.status != rustc_feedback.COMPILE_FAILED:
        raise FactorizedTextEditMaterializerError("a structured compile failure is required")
    try:
        encoded = source.encode("ascii", errors="strict")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise FactorizedTextEditMaterializerError("source custody requires ASCII Rust") from exc
    spans = tuple(
        span for diagnostic in attempt.error_diagnostics for span in diagnostic.primary_spans
    )
    if (
        not encoded
        or len(encoded) > rustc_feedback.MAX_SOURCE_BYTES
        or attempt.source_sha256 != sha256(encoded).hexdigest()
        or attempt.source_size_bytes != len(encoded)
        or len(spans) != 1
        or spans[0].file_name != attempt.source_file_name
        or spans[0].byte_start == spans[0].byte_end
    ):
        raise FactorizedTextEditMaterializerError(
            "rustc diagnostic and source custody do not match"
        )
    return encoded


def _marker_declaration_span(source: bytes, marker: str) -> tuple[int, int]:
    marker_bytes = marker.encode("ascii")
    pattern = re.compile(
        rb"(?m)^[\t ]*let[\t ]+(" + re.escape(marker_bytes) + rb")(?=[\t ]*(?::|=))"
    )
    matches = tuple(pattern.finditer(source))
    if len(matches) != 1:
        raise FactorizedTextEditMaterializerError(
            "each target marker must have exactly one local declaration"
        )
    return matches[0].span(1)


@dataclass(frozen=True, slots=True)
class FactorizedTextEditAuthorityReceiptV1:
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    diagnostic_attempt_sha256: str
    selection_context_sha256: str
    expected_checkpoint_sha256: str
    allow_exploration: bool
    operator_bindings: tuple[tuple[str, str], tuple[str, str]]
    target_bindings: tuple[tuple[str, str], tuple[str, str]]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_file_name": self.source_file_name,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "diagnostic_attempt_sha256": self.diagnostic_attempt_sha256,
            "selection_context_sha256": self.selection_context_sha256,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "allow_exploration": self.allow_exploration,
            "operator_bindings": self.operator_bindings,
            "target_bindings": self.target_bindings,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            type(self.source_file_name) is not str
            or not self.source_file_name
            or "/" in self.source_file_name
            or "\\" in self.source_file_name
            or not _valid_sha256(self.source_sha256)
            or type(self.source_size_bytes) is not int
            or not 1 <= self.source_size_bytes <= rustc_feedback.MAX_SOURCE_BYTES
            or not _valid_sha256(self.diagnostic_attempt_sha256)
            or not _valid_sha256(self.selection_context_sha256)
            or not _valid_sha256(self.expected_checkpoint_sha256)
            or type(self.allow_exploration) is not bool
            or _canonical_operator_bindings(self.operator_bindings) != self.operator_bindings
            or _canonical_target_bindings(self.target_bindings) != self.target_bindings
            or len(
                {
                    *(root for root, _operation in self.operator_bindings),
                    *(root for root, _marker in self.target_bindings),
                }
            )
            != 4
            or self.sha256 != _digest(_AUTHORITY_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedTextEditMaterializerError(
                "factorized text-edit authority authentication failed"
            )


@dataclass(frozen=True, slots=True)
class FactorizedTextEditV1:
    """One LSP-shaped byte range plus replacement text."""

    byte_range: tuple[int, int]
    new_text: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "range": {
                "byte_start": self.byte_range[0],
                "byte_end": self.byte_range[1],
            },
            "new_text": self.new_text,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        try:
            new_text_bytes = self.new_text.encode("ascii", errors="strict")
        except (AttributeError, UnicodeEncodeError) as exc:
            raise FactorizedTextEditMaterializerError(
                "text edit must contain exact ASCII replacement text"
            ) from exc
        if (
            type(self.byte_range) is not tuple
            or len(self.byte_range) != 2
            or any(type(value) is not int for value in self.byte_range)
            or not 0 <= self.byte_range[0] <= self.byte_range[1]
            or not new_text_bytes
            or self.sha256 != _digest(_EDIT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedTextEditMaterializerError("text-edit authentication failed")


@dataclass(frozen=True, slots=True)
class FactorizedTextEditMaterializationV1:
    authority_sha256: str
    decision_sha256: str
    diagnostic_attempt_sha256: str
    selected_hypothesis_sha256: str
    operator_component_sha256: str
    target_component_sha256: str
    primary_span_sha256: str
    original_source_sha256: str
    edit: FactorizedTextEditV1
    materialized_source_sha256: str
    materialized_source_size_bytes: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "authority_sha256": self.authority_sha256,
            "decision_sha256": self.decision_sha256,
            "diagnostic_attempt_sha256": self.diagnostic_attempt_sha256,
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "operator_component_sha256": self.operator_component_sha256,
            "target_component_sha256": self.target_component_sha256,
            "primary_span_sha256": self.primary_span_sha256,
            "original_source_sha256": self.original_source_sha256,
            "edit": self.edit.record(),
            "materialized_source_sha256": self.materialized_source_sha256,
            "materialized_source_size_bytes": self.materialized_source_size_bytes,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.edit) is not FactorizedTextEditV1:
            raise TypeError("materialization text edit must be exact")
        self.edit.authenticate()
        roots = (
            self.authority_sha256,
            self.decision_sha256,
            self.diagnostic_attempt_sha256,
            self.selected_hypothesis_sha256,
            self.operator_component_sha256,
            self.target_component_sha256,
            self.primary_span_sha256,
            self.original_source_sha256,
            self.materialized_source_sha256,
        )
        if (
            any(not _valid_sha256(value) for value in roots)
            or type(self.materialized_source_size_bytes) is not int
            or not 1 <= self.materialized_source_size_bytes <= rustc_feedback.MAX_SOURCE_BYTES
            or self.sha256 != _digest(_MATERIALIZATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedTextEditMaterializerError(
                "factorized text-edit materialization authentication failed"
            )


@dataclass(frozen=True, slots=True)
class FactorizedTextEditReleaseV1:
    authority_sha256: str
    selection_sha256: str
    action_handle: str
    source_sha256: str
    materialization: FactorizedTextEditMaterializationV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "authority_sha256": self.authority_sha256,
            "selection_sha256": self.selection_sha256,
            "action_handle": self.action_handle,
            "source_sha256": self.source_sha256,
            "materialization": self.materialization.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.materialization) is not FactorizedTextEditMaterializationV1:
            raise TypeError("release materialization must be exact")
        self.materialization.authenticate()
        if (
            not _valid_sha256(self.authority_sha256)
            or not _valid_sha256(self.selection_sha256)
            or type(self.action_handle) is not str
            or _ACTION_HANDLE.fullmatch(self.action_handle) is None
            or not _valid_sha256(self.source_sha256)
            or self.authority_sha256 != self.materialization.authority_sha256
            or self.selection_sha256 != self.materialization.decision_sha256
            or self.source_sha256 != self.materialization.materialized_source_sha256
            or self.sha256 != _digest(_RELEASE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedTextEditMaterializerError(
                "factorized text-edit release authentication failed"
            )


def _make_authority_receipt(
    *,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    selection_context_sha256: str,
    expected_checkpoint_sha256: str,
    allow_exploration: bool,
    operator_bindings: tuple[tuple[str, str], tuple[str, str]],
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> FactorizedTextEditAuthorityReceiptV1:
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_file_name": attempt.source_file_name,
        "source_sha256": attempt.source_sha256,
        "source_size_bytes": attempt.source_size_bytes,
        "diagnostic_attempt_sha256": attempt.sha256,
        "selection_context_sha256": selection_context_sha256,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "allow_exploration": allow_exploration,
        "operator_bindings": operator_bindings,
        "target_bindings": target_bindings,
    }
    result = FactorizedTextEditAuthorityReceiptV1(
        source_file_name=attempt.source_file_name,
        source_sha256=attempt.source_sha256,
        source_size_bytes=attempt.source_size_bytes,
        diagnostic_attempt_sha256=attempt.sha256,
        selection_context_sha256=selection_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=operator_bindings,
        target_bindings=target_bindings,
        sha256=_digest(_AUTHORITY_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _seal_edit(byte_range: tuple[int, int], new_text: str) -> FactorizedTextEditV1:
    payload = {
        "range": {"byte_start": byte_range[0], "byte_end": byte_range[1]},
        "new_text": new_text,
    }
    result = FactorizedTextEditV1(
        byte_range=byte_range,
        new_text=new_text,
        sha256=_digest(_EDIT_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _compose_edit(
    *,
    source: bytes,
    primary_span: rustc_feedback.RustcPrimarySpanV1,
    operation: str,
    marker: str,
) -> FactorizedTextEditV1:
    primary_bytes = source[primary_span.byte_start : primary_span.byte_end]
    try:
        primary_identifier = primary_bytes.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise FactorizedTextEditMaterializerError(
            "rustc primary span is not one ASCII identifier"
        ) from exc
    if _IDENTIFIER.fullmatch(primary_identifier) is None:
        raise FactorizedTextEditMaterializerError("rustc primary span is not one ASCII identifier")
    anchor_start, anchor_end = _marker_declaration_span(source, marker)
    if operation == REPLACE_PRIMARY_SPAN_WITH_ANCHOR:
        return _seal_edit((primary_span.byte_start, primary_span.byte_end), marker)
    if operation != DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR:
        raise FactorizedTextEditMaterializerError("selected operator binding changed")
    line_start = source.rfind(b"\n", 0, anchor_start) + 1
    line_end = source.find(b"\n", anchor_end)
    if line_end < 0:
        raise FactorizedTextEditMaterializerError(
            "declaration target marker must be on a terminated source line"
        )
    line_end += 1
    indent_end = line_start
    while indent_end < len(source) and source[indent_end] in (9, 32):
        indent_end += 1
    indentation = source[line_start:indent_end]
    newline = b"\r\n" if source[line_end - 2 : line_end] == b"\r\n" else b"\n"
    new_text = (
        indentation + b"let " + primary_bytes + b" = " + marker.encode("ascii") + b";" + newline
    ).decode("ascii")
    return _seal_edit((line_end, line_end), new_text)


def _seal_materialization(
    *,
    authority_sha256: str,
    decision: factors.FactorizedFeedbackDecisionV1,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    selected_hypothesis: factors.FactorizedHypothesisV1,
    primary_span: rustc_feedback.RustcPrimarySpanV1,
    original_source_sha256: str,
    edit: FactorizedTextEditV1,
    materialized_source: bytes,
) -> FactorizedTextEditMaterializationV1:
    payload = {
        "authority_sha256": authority_sha256,
        "decision_sha256": decision.sha256,
        "diagnostic_attempt_sha256": attempt.sha256,
        "selected_hypothesis_sha256": selected_hypothesis.sha256,
        "operator_component_sha256": selected_hypothesis.component_sha256s[0],
        "target_component_sha256": selected_hypothesis.component_sha256s[1],
        "primary_span_sha256": primary_span.sha256,
        "original_source_sha256": original_source_sha256,
        "edit": edit.record(),
        "materialized_source_sha256": sha256(materialized_source).hexdigest(),
        "materialized_source_size_bytes": len(materialized_source),
    }
    result = FactorizedTextEditMaterializationV1(
        authority_sha256=authority_sha256,
        decision_sha256=decision.sha256,
        diagnostic_attempt_sha256=attempt.sha256,
        selected_hypothesis_sha256=selected_hypothesis.sha256,
        operator_component_sha256=selected_hypothesis.component_sha256s[0],
        target_component_sha256=selected_hypothesis.component_sha256s[1],
        primary_span_sha256=primary_span.sha256,
        original_source_sha256=original_source_sha256,
        edit=edit,
        materialized_source_sha256=payload["materialized_source_sha256"],
        materialized_source_size_bytes=len(materialized_source),
        sha256=_digest(_MATERIALIZATION_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _seal_release(
    materialization: FactorizedTextEditMaterializationV1,
) -> FactorizedTextEditReleaseV1:
    action_handle = f"factorized_{materialization.selected_hypothesis_sha256}"
    payload = {
        "authority_sha256": materialization.authority_sha256,
        "selection_sha256": materialization.decision_sha256,
        "action_handle": action_handle,
        "source_sha256": materialization.materialized_source_sha256,
        "materialization": materialization.record(),
    }
    result = FactorizedTextEditReleaseV1(
        authority_sha256=materialization.authority_sha256,
        selection_sha256=materialization.decision_sha256,
        action_handle=action_handle,
        source_sha256=materialization.materialized_source_sha256,
        materialization=materialization,
        sha256=_digest(_RELEASE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class PreparedFactorizedTextEditMaterializerV1:
    """One-shot selected-source authority over exact diagnostic/source custody."""

    __slots__ = (
        "_allow_exploration",
        "_attempt",
        "_consumed",
        "_expected_checkpoint_sha256",
        "_lock",
        "_materialization",
        "_materialization_count",
        "_operator_bindings",
        "_receipt",
        "_release",
        "_release_count",
        "_selection_context_sha256",
        "_source",
        "_source_bytes",
        "_target_bindings",
    )

    def __init__(
        self,
        *,
        attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
        source: str,
        source_bytes: bytes,
        selection_context_sha256: str,
        expected_checkpoint_sha256: str,
        allow_exploration: bool,
        operator_bindings: tuple[tuple[str, str], tuple[str, str]],
        target_bindings: tuple[tuple[str, str], tuple[str, str]],
        receipt: FactorizedTextEditAuthorityReceiptV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("factorized text-edit materializers are factory-owned")
        self._attempt = attempt
        self._source = source
        self._source_bytes = source_bytes
        self._selection_context_sha256 = selection_context_sha256
        self._expected_checkpoint_sha256 = expected_checkpoint_sha256
        self._allow_exploration = allow_exploration
        self._operator_bindings = operator_bindings
        self._target_bindings = target_bindings
        self._receipt = receipt
        self._lock = Lock()
        self._consumed = False
        self._materialization_count = 0
        self._release_count = 0
        self._materialization: FactorizedTextEditMaterializationV1 | None = None
        self._release: FactorizedTextEditReleaseV1 | None = None
        self._authenticate_current()

    def _authenticate_current(self) -> None:
        if type(self._receipt) is not FactorizedTextEditAuthorityReceiptV1:
            raise TypeError("factorized text-edit authority receipt must be exact")
        source_bytes = _source_bytes(self._attempt, self._source)
        if source_bytes != self._source_bytes:
            raise FactorizedTextEditMaterializerError("source custody changed")
        for _root, marker in self._target_bindings:
            _marker_declaration_span(source_bytes, marker)
        expected = _make_authority_receipt(
            attempt=self._attempt,
            selection_context_sha256=self._selection_context_sha256,
            expected_checkpoint_sha256=self._expected_checkpoint_sha256,
            allow_exploration=self._allow_exploration,
            operator_bindings=self._operator_bindings,
            target_bindings=self._target_bindings,
        )
        if expected != self._receipt:
            raise FactorizedTextEditMaterializerError("authority custody changed")
        if (
            type(self._materialization_count) is not int
            or type(self._release_count) is not int
            or self._materialization_count not in (0, 1)
            or self._release_count not in (0, 1)
            or self._materialization_count != self._release_count
            or (self._materialization_count == 0)
            != (self._materialization is None and self._release is None)
        ):
            raise FactorizedTextEditMaterializerError("authority counters changed")
        if self._materialization is not None and self._release is not None:
            self._materialization.authenticate()
            self._release.authenticate()
            if self._release.materialization != self._materialization:
                raise FactorizedTextEditMaterializerError("authority release changed")

    @property
    def receipt(self) -> FactorizedTextEditAuthorityReceiptV1:
        self._authenticate_current()
        return self._receipt

    @property
    def authority_sha256(self) -> str:
        return self.receipt.sha256

    @property
    def materialization_count(self) -> int:
        self._authenticate_current()
        return self._materialization_count

    @property
    def release_count(self) -> int:
        self._authenticate_current()
        return self._release_count

    @property
    def materialization(self) -> FactorizedTextEditMaterializationV1 | None:
        self._authenticate_current()
        return self._materialization

    @property
    def release(self) -> FactorizedTextEditReleaseV1 | None:
        self._authenticate_current()
        return self._release

    def execute_selected(
        self,
        selection: factors.FactorizedFeedbackDecisionV1,
        callback: Callable[[rust_executor.RustSelectedSourceV2], _T],
    ) -> _T:
        """Materialize and release exactly the physically selected hypothesis."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        self._authenticate_current()
        if type(selection) is not factors.FactorizedFeedbackDecisionV1:
            raise TypeError("selection must be one exact factorized feedback decision")
        selection.authenticate()
        if (
            selection.status
            not in (
                outcomes.READY,
                outcomes.EXPLORATION_READY,
            )
            or selection.selected_hypothesis_sha256 is None
            or not selection.cell_gate
            or not selection.outcome_gate
            or selection.axis_gates != (True, True)
        ):
            raise FactorizedTextEditMaterializerError(
                "only an enabled selected factorized decision can materialize"
            )
        if selection.checkpoint_sha256 != self._expected_checkpoint_sha256:
            raise FactorizedTextEditMaterializerError(
                "factorized decision checkpoint differs from authority"
            )
        if not self._allow_exploration and (
            selection.status != outcomes.READY
            or selection.exploration_hypothesis_sha256 is not None
        ):
            raise FactorizedTextEditMaterializerError(
                "factorized text-edit authority forbids exploration"
            )
        expected_observation = rustc_feedback.project_rust_factorized_feedback_v1(
            self._attempt,
            selection_context_sha256=self._selection_context_sha256,
            active_hypothesis_sha256s=selection.observation.active_hypothesis_sha256s,
        )
        if selection.observation != expected_observation:
            raise FactorizedTextEditMaterializerError(
                "factorized decision does not match rustc diagnostic custody"
            )
        hypotheses = tuple(score.hypothesis for score in selection.hypothesis_scores)
        operator_roots = {root for root, _operation in self._operator_bindings}
        target_roots = {root for root, _marker in self._target_bindings}
        expected_component_pairs = {
            (operator_root, target_root)
            for operator_root in operator_roots
            for target_root in target_roots
        }
        if (
            len(hypotheses) != 4
            or {hypothesis.component_sha256s for hypothesis in hypotheses}
            != expected_component_pairs
        ):
            raise FactorizedTextEditMaterializerError(
                "factorized decision bank differs from authority bindings"
            )
        selected = next(
            (
                hypothesis
                for hypothesis in hypotheses
                if hypothesis.sha256 == selection.selected_hypothesis_sha256
            ),
            None,
        )
        if selected is None:
            raise FactorizedTextEditMaterializerError("selected hypothesis left the bound bank")
        operation = dict(self._operator_bindings)[selected.component_sha256s[0]]
        marker = dict(self._target_bindings)[selected.component_sha256s[1]]
        primary_span = next(
            span
            for diagnostic in self._attempt.error_diagnostics
            for span in diagnostic.primary_spans
        )

        with self._lock:
            if self._consumed:
                raise FactorizedTextEditMaterializerError(
                    "factorized text-edit authority was consumed"
                )
            self._consumed = True

        edit = _compose_edit(
            source=self._source_bytes,
            primary_span=primary_span,
            operation=operation,
            marker=marker,
        )
        start, end = edit.byte_range
        materialized_source = (
            self._source_bytes[:start] + edit.new_text.encode("ascii") + self._source_bytes[end:]
        )
        if not materialized_source or len(materialized_source) > rustc_feedback.MAX_SOURCE_BYTES:
            raise FactorizedTextEditMaterializerError(
                "materialized Rust source exceeds the bounded source policy"
            )
        materialization = _seal_materialization(
            authority_sha256=self.authority_sha256,
            decision=selection,
            attempt=self._attempt,
            selected_hypothesis=selected,
            primary_span=primary_span,
            original_source_sha256=self._receipt.source_sha256,
            edit=edit,
            materialized_source=materialized_source,
        )
        release = _seal_release(materialization)
        selected_source = rust_executor.make_rust_selected_source_v2(
            authority_sha256=self.authority_sha256,
            selection_sha256=selection.sha256,
            release_sha256=release.sha256,
            action_handle=release.action_handle,
            source=materialized_source.decode("ascii"),
        )
        selected_source.authenticate_structure()
        if (
            selected_source.authority_sha256 != self.authority_sha256
            or selected_source.selection_sha256 != selection.sha256
            or selected_source.release_sha256 != release.sha256
            or selected_source.action_handle != release.action_handle
            or selected_source.source_sha256 != release.source_sha256
        ):
            raise FactorizedTextEditMaterializerError(
                "selected-source grant differs from factorized release"
            )
        with self._lock:
            self._materialization = materialization
            self._release = release
            self._materialization_count = 1
            self._release_count = 1
        return callback(selected_source)


def prepare_factorized_text_edit_materializer_v1(
    *,
    diagnostic_attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    source: str,
    selection_context_sha256: str,
    expected_checkpoint_sha256: str,
    allow_exploration: bool,
    operator_bindings: tuple[tuple[str, str], tuple[str, str]],
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> PreparedFactorizedTextEditMaterializerV1:
    """Freeze exact diagnostic/source custody and a two-by-two edit catalogue."""

    if not _valid_sha256(selection_context_sha256):
        raise FactorizedTextEditMaterializerError("selection context root is invalid")
    if not _valid_sha256(expected_checkpoint_sha256):
        raise FactorizedTextEditMaterializerError("expected checkpoint root is invalid")
    if type(allow_exploration) is not bool:
        raise TypeError("allow_exploration must be one exact boolean")
    checked_operators = _canonical_operator_bindings(operator_bindings)
    checked_targets = _canonical_target_bindings(target_bindings)
    if (
        len(
            {
                *(root for root, _operation in checked_operators),
                *(root for root, _marker in checked_targets),
            }
        )
        != 4
    ):
        raise FactorizedTextEditMaterializerError(
            "operator and target component roots must be distinct"
        )
    source_bytes = _source_bytes(diagnostic_attempt, source)
    for _root, marker in checked_targets:
        _marker_declaration_span(source_bytes, marker)
    receipt = _make_authority_receipt(
        attempt=diagnostic_attempt,
        selection_context_sha256=selection_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=checked_operators,
        target_bindings=checked_targets,
    )
    return PreparedFactorizedTextEditMaterializerV1(
        attempt=diagnostic_attempt,
        source=source,
        source_bytes=source_bytes,
        selection_context_sha256=selection_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=checked_operators,
        target_bindings=checked_targets,
        receipt=receipt,
        construction_token=_CONSTRUCTION_TOKEN,
    )
