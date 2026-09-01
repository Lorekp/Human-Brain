"""Occurrence-derived wrapper around the frozen message-line edit authority.

The factory derives a complete two-occurrence roster from the authenticated
Rust source and sole primary span.  It binds the roster's generic farther and
nearer relation roots to the corresponding source identifiers, then gives
those bindings to the established selected-only message-line authority.

This wrapper adds custody, not another decision or execution path.  The outer
unordered-cue memory still selects one operation x relation hypothesis, and
the frozen delegate remains the only component allowed to materialize, release,
or execute an edit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import common_factorized_text_edit_materializer_v1 as edits
from experiments.r3_pilot import common_preceding_let_declaration_occurrences_v1 as occurrences
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_rustc_message_line_unordered_cue_adapter_v1 as message_line
from experiments.r3_pilot import (
    common_rustc_message_line_unordered_cue_text_edit_authority_v1 as delegate_authority,
)
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "humanbrain.rustc-message-line-occurrence-text-edit-authority.v1"
CLAIM_BOUNDARY = (
    "source-derived-complete-two-preceding-let-declaration-occurrence-roster-"
    "bound-by-generic-farther-nearer-relation-to-one-frozen-selected-only-"
    "message-line-text-edit-authority-not-learned-tokenization-parsing-"
    "relation-or-operation-discovery-semantics-diagnosis-synthesis-search-"
    "debugging-transfer-swe-or-intelligence"
)

UnorderedCueFactorTextEditMaterializationV1 = (
    delegate_authority.UnorderedCueFactorTextEditMaterializationV1
)
UnorderedCueFactorTextEditReleaseV1 = delegate_authority.UnorderedCueFactorTextEditReleaseV1
UnorderedCueFactorTextEditWorkV1 = delegate_authority.UnorderedCueFactorTextEditWorkV1

_SHA256_HEX = frozenset("0123456789abcdef")
_AUTHORITY_NAMESPACE = b"rustc-message-line-occurrence-text-edit-authority-v1"
_CONSTRUCTION_TOKEN = object()


class RustcMessageLineOccurrenceTextEditAuthorityError(ValueError):
    """The occurrence roster, delegate, source, or selected decision changed."""


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


def _derive_relation_target_bindings(
    roster: occurrences.PrecedingLetDeclarationRosterV1,
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Bind each generic relation root to its exact source identifier."""

    if type(roster) is not occurrences.PrecedingLetDeclarationRosterV1:
        raise TypeError("occurrence roster must be exact")
    roster.authenticate()
    if tuple(item.relation for item in roster.occurrences) != (
        occurrences.FARTHER_PRECEDING_DECLARATION,
        occurrences.NEARER_PRECEDING_DECLARATION,
    ):
        raise RustcMessageLineOccurrenceTextEditAuthorityError(
            "occurrence roster relation order changed"
        )
    rows: list[tuple[str, str]] = []
    for item in roster.occurrences:
        expected_root = occurrences.preceding_declaration_relation_component_sha256_v1(
            item.relation
        )
        try:
            identifier = item.identifier_bytes.decode("ascii", errors="strict")
        except (AttributeError, UnicodeDecodeError) as exc:
            raise RustcMessageLineOccurrenceTextEditAuthorityError(
                "occurrence identifier is not exact ASCII"
            ) from exc
        if item.relation_component_sha256 != expected_root:
            raise RustcMessageLineOccurrenceTextEditAuthorityError(
                "occurrence relation component root changed"
            )
        rows.append((expected_root, identifier))
    try:
        return edits._canonical_target_bindings(tuple(rows))
    except (TypeError, ValueError) as exc:
        raise RustcMessageLineOccurrenceTextEditAuthorityError(
            "occurrence relation target bindings are invalid"
        ) from exc


@dataclass(frozen=True, slots=True)
class RustcMessageLineOccurrenceTextEditAuthorityReceiptV1:
    source_sha256: str
    diagnostic_attempt_sha256: str
    message_line_envelope_sha256: str
    occurrence_roster_sha256: str
    primary_span_sha256: str
    relation_target_bindings: tuple[tuple[str, str], tuple[str, str]]
    delegate_authority_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_sha256": self.source_sha256,
            "diagnostic_attempt_sha256": self.diagnostic_attempt_sha256,
            "message_line_envelope_sha256": self.message_line_envelope_sha256,
            "occurrence_roster_sha256": self.occurrence_roster_sha256,
            "primary_span_sha256": self.primary_span_sha256,
            "relation_target_bindings": self.relation_target_bindings,
            "delegate_authority_sha256": self.delegate_authority_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        try:
            canonical_bindings = edits._canonical_target_bindings(
                self.relation_target_bindings
            )
            expected_relation_roots = {
                occurrences.preceding_declaration_relation_component_sha256_v1(
                    relation
                )
                for relation in occurrences.PRECEDING_DECLARATION_RELATIONS
            }
        except (TypeError, ValueError) as exc:
            raise RustcMessageLineOccurrenceTextEditAuthorityError(
                "occurrence text-edit receipt bindings are invalid"
            ) from exc
        if (
            any(
                not _valid_sha256(value)
                for value in (
                    self.source_sha256,
                    self.diagnostic_attempt_sha256,
                    self.message_line_envelope_sha256,
                    self.occurrence_roster_sha256,
                    self.primary_span_sha256,
                    self.delegate_authority_sha256,
                )
            )
            or canonical_bindings != self.relation_target_bindings
            or {root for root, _identifier in self.relation_target_bindings}
            != expected_relation_roots
            or self.sha256 != _digest(_AUTHORITY_NAMESPACE, self.record(include_sha256=False))
        ):
            raise RustcMessageLineOccurrenceTextEditAuthorityError(
                "occurrence text-edit authority receipt authentication failed"
            )


def _make_receipt(
    *,
    envelope: message_line.RustcMessageLineUnorderedCueEnvelopeV1,
    roster: occurrences.PrecedingLetDeclarationRosterV1,
    relation_target_bindings: tuple[tuple[str, str], tuple[str, str]],
    delegate: delegate_authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1,
) -> RustcMessageLineOccurrenceTextEditAuthorityReceiptV1:
    primary_span = delegate_authority._sole_custodied_primary_span(envelope)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_sha256": roster.source_sha256,
        "diagnostic_attempt_sha256": envelope.attempt.sha256,
        "message_line_envelope_sha256": envelope.sha256,
        "occurrence_roster_sha256": roster.sha256,
        "primary_span_sha256": primary_span.sha256,
        "relation_target_bindings": relation_target_bindings,
        "delegate_authority_sha256": delegate.authority_sha256,
    }
    result = RustcMessageLineOccurrenceTextEditAuthorityReceiptV1(
        source_sha256=roster.source_sha256,
        diagnostic_attempt_sha256=envelope.attempt.sha256,
        message_line_envelope_sha256=envelope.sha256,
        occurrence_roster_sha256=roster.sha256,
        primary_span_sha256=primary_span.sha256,
        relation_target_bindings=relation_target_bindings,
        delegate_authority_sha256=delegate.authority_sha256,
        sha256=_digest(_AUTHORITY_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class PreparedRustcMessageLineOccurrenceTextEditAuthorityV1:
    """Factory-owned occurrence custody around one frozen C10 authority."""

    def __init__(
        self,
        *,
        occurrence_roster: occurrences.PrecedingLetDeclarationRosterV1,
        delegate: delegate_authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1,
        receipt: RustcMessageLineOccurrenceTextEditAuthorityReceiptV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("occurrence text-edit authorities are factory-owned")
        self._occurrence_roster = occurrence_roster
        self._delegate = delegate
        self._receipt = receipt
        self._authenticate_current()

    def _authenticate_current(self) -> None:
        try:
            self._receipt.authenticate()
            envelope = self._delegate.message_line_envelope
            primary_span = delegate_authority._sole_custodied_primary_span(envelope)
            reproduced_roster = occurrences.project_preceding_let_declaration_occurrences_v1(
                envelope.source,
                primary_byte_start=primary_span.byte_start,
                primary_byte_end=primary_span.byte_end,
            )
            expected_bindings = _derive_relation_target_bindings(reproduced_roster)
            delegate_receipt = self._delegate.receipt
        except (TypeError, ValueError) as exc:
            raise RustcMessageLineOccurrenceTextEditAuthorityError(
                "occurrence text-edit authority could not rederive custody"
            ) from exc
        if (
            reproduced_roster != self._occurrence_roster
            or self._receipt.source_sha256 != reproduced_roster.source_sha256
            or self._receipt.diagnostic_attempt_sha256 != envelope.attempt.sha256
            or self._receipt.message_line_envelope_sha256 != envelope.sha256
            or self._receipt.occurrence_roster_sha256 != reproduced_roster.sha256
            or self._receipt.primary_span_sha256 != primary_span.sha256
            or self._receipt.relation_target_bindings != expected_bindings
            or delegate_receipt.target_bindings != expected_bindings
            or self._receipt.delegate_authority_sha256 != self._delegate.authority_sha256
        ):
            raise RustcMessageLineOccurrenceTextEditAuthorityError(
                "occurrence roster to delegate custody changed"
            )

    @property
    def message_line_envelope(self) -> message_line.RustcMessageLineUnorderedCueEnvelopeV1:
        self._authenticate_current()
        return self._delegate.message_line_envelope

    @property
    def projected_observation(self) -> unordered.UnorderedCueFactorObservationV1:
        self._authenticate_current()
        return self._delegate.projected_observation

    @property
    def occurrence_roster(self) -> occurrences.PrecedingLetDeclarationRosterV1:
        self._authenticate_current()
        return self._occurrence_roster

    @property
    def receipt(self) -> RustcMessageLineOccurrenceTextEditAuthorityReceiptV1:
        self._authenticate_current()
        return self._receipt

    @property
    def custody_sha256(self) -> str:
        self._authenticate_current()
        return self._receipt.sha256

    @property
    def authority_sha256(self) -> str:
        """Return the frozen delegate root retained by selected-source execution."""

        self._authenticate_current()
        return self._delegate.authority_sha256

    @property
    def delegate_authority_sha256(self) -> str:
        self._authenticate_current()
        return self._delegate.authority_sha256

    @property
    def materialization(self) -> UnorderedCueFactorTextEditMaterializationV1 | None:
        self._authenticate_current()
        return self._delegate.materialization

    @property
    def release(self) -> UnorderedCueFactorTextEditReleaseV1 | None:
        self._authenticate_current()
        return self._delegate.release

    @property
    def materialization_count(self) -> int:
        self._authenticate_current()
        return self._delegate.materialization_count

    @property
    def release_count(self) -> int:
        self._authenticate_current()
        return self._delegate.release_count

    @property
    def work_counters(self) -> UnorderedCueFactorTextEditWorkV1:
        self._authenticate_current()
        return self._delegate.work_counters

    def execute_selected(
        self,
        selection: unordered.UnorderedCueFactorDecisionV1,
        batch: tuple[tuple[int, ...], ...],
    ) -> rust_executor.RustSelectedActionExecutionV2:
        """Delegate one selected operation x occurrence relation unchanged."""

        self._authenticate_current()
        return self._delegate.execute_selected(selection, batch)


def prepare_rustc_message_line_occurrence_text_edit_authority_v1(
    *,
    diagnostic_attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    stderr: bytes,
    source: str,
    selection_context_sha256: str,
    expected_checkpoint_sha256: str,
    expected_factorized_checkpoint_sha256: str,
    allow_exploration: bool,
) -> PreparedRustcMessageLineOccurrenceTextEditAuthorityV1:
    """Derive a complete occurrence roster and freeze its C10 delegate."""

    try:
        envelope = message_line.project_rustc_message_line_unordered_cue_observation_v1(
            diagnostic_attempt,
            stderr=stderr,
            source=source,
            selection_context_sha256=selection_context_sha256,
        )
        primary_span = delegate_authority._sole_custodied_primary_span(envelope)
        roster = occurrences.project_preceding_let_declaration_occurrences_v1(
            source,
            primary_byte_start=primary_span.byte_start,
            primary_byte_end=primary_span.byte_end,
        )
        relation_target_bindings = _derive_relation_target_bindings(roster)
        delegate = (
            delegate_authority.prepare_rustc_message_line_unordered_cue_text_edit_authority_v1(
                diagnostic_attempt=diagnostic_attempt,
                stderr=stderr,
                source=source,
                selection_context_sha256=selection_context_sha256,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_factorized_checkpoint_sha256=expected_factorized_checkpoint_sha256,
                allow_exploration=allow_exploration,
                target_bindings=relation_target_bindings,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RustcMessageLineOccurrenceTextEditAuthorityError(
            "occurrence text-edit authority derivation failed"
        ) from exc
    if delegate.message_line_envelope != envelope:
        raise RustcMessageLineOccurrenceTextEditAuthorityError(
            "delegate message-line envelope differs from occurrence envelope"
        )
    receipt = _make_receipt(
        envelope=envelope,
        roster=roster,
        relation_target_bindings=relation_target_bindings,
        delegate=delegate,
    )
    return PreparedRustcMessageLineOccurrenceTextEditAuthorityV1(
        occurrence_roster=roster,
        delegate=delegate,
        receipt=receipt,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA",
    "PreparedRustcMessageLineOccurrenceTextEditAuthorityV1",
    "RustcMessageLineOccurrenceTextEditAuthorityError",
    "RustcMessageLineOccurrenceTextEditAuthorityReceiptV1",
    "UnorderedCueFactorTextEditMaterializationV1",
    "UnorderedCueFactorTextEditReleaseV1",
    "UnorderedCueFactorTextEditWorkV1",
    "prepare_rustc_message_line_occurrence_text_edit_authority_v1",
]
