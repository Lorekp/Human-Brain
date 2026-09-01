"""Decision-bound Rust text-edit authority for lexical unordered cues.

The factory owns diagnostic projection: it accepts the exact rustc attempt,
raw stderr, and selected source, then derives one authenticated lexical
envelope.  No caller-provided cognitive observation is accepted.  Only an
outer unordered-cue decision matching that envelope, the two checkpoint roots,
and the canonical two-operation by two-target bank may materialize one edit.

Materialization, release, and V2 execution reuse the established C8 records and
helpers unchanged.  This remains a host-owned edit catalogue and actuator, not
cue discovery, localization, diagnosis, synthesis, or debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock

from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_materializer_v1 as edits
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_rust_action_intervention_v1 as interventions
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_rustc_lexical_unordered_cue_adapter_v1 as lexical
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from experiments.r3_pilot import common_unordered_cue_factor_text_edit_authority_v1 as c8
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "humanbrain.rustc-lexical-unordered-cue-text-edit-authority.v1"
CLAIM_BOUNDARY = (
    "post-selection-materialization-of-one-authenticated-rustc-parent-message-"
    "lexical-plus-relative-region-unordered-cue-decision-using-one-host-owned-"
    "two-operation-two-target-rust-text-edit-catalogue-not-cue-or-action-"
    "discovery-localization-diagnosis-synthesis-search-debugging-transfer-swe-"
    "or-intelligence"
)

UnorderedCueFactorTextEditMaterializationV1 = c8.UnorderedCueFactorTextEditMaterializationV1
UnorderedCueFactorTextEditReleaseV1 = c8.UnorderedCueFactorTextEditReleaseV1
UnorderedCueFactorTextEditWorkV1 = c8.UnorderedCueFactorTextEditWorkV1

_SHA256_HEX = frozenset("0123456789abcdef")
_AUTHORITY_NAMESPACE = b"rustc-lexical-unordered-cue-text-edit-authority-v1"
_CONSTRUCTION_TOKEN = object()


class RustcLexicalUnorderedCueTextEditAuthorityError(ValueError):
    """The lexical envelope, decision, source, bank, or release changed."""


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


@dataclass(frozen=True, slots=True)
class RustcLexicalUnorderedCueTextEditAuthorityReceiptV1:
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    diagnostic_attempt_sha256: str
    raw_stderr_sha256: str
    raw_stderr_size_bytes: int
    selection_context_sha256: str
    lexical_envelope_sha256: str
    projected_observation_sha256: str
    expected_checkpoint_sha256: str
    expected_factorized_checkpoint_sha256: str
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
            "raw_stderr_sha256": self.raw_stderr_sha256,
            "raw_stderr_size_bytes": self.raw_stderr_size_bytes,
            "selection_context_sha256": self.selection_context_sha256,
            "lexical_envelope_sha256": self.lexical_envelope_sha256,
            "projected_observation_sha256": self.projected_observation_sha256,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "expected_factorized_checkpoint_sha256": (self.expected_factorized_checkpoint_sha256),
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
            or any(
                not _valid_sha256(value)
                for value in (
                    self.diagnostic_attempt_sha256,
                    self.raw_stderr_sha256,
                    self.selection_context_sha256,
                    self.lexical_envelope_sha256,
                    self.projected_observation_sha256,
                    self.expected_checkpoint_sha256,
                    self.expected_factorized_checkpoint_sha256,
                )
            )
            or type(self.raw_stderr_size_bytes) is not int
            or not 1 <= self.raw_stderr_size_bytes <= rustc_feedback.MAX_STDERR_BYTES
            or type(self.allow_exploration) is not bool
            or edits._canonical_operator_bindings(self.operator_bindings) != self.operator_bindings
            or edits._canonical_target_bindings(self.target_bindings) != self.target_bindings
            or len(
                {
                    *(root for root, _operation in self.operator_bindings),
                    *(root for root, _marker in self.target_bindings),
                }
            )
            != 4
            or self.sha256 != _digest(_AUTHORITY_NAMESPACE, self.record(include_sha256=False))
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical unordered-cue text-edit authority receipt authentication failed"
            )


def _make_receipt(
    *,
    envelope: lexical.RustcLexicalUnorderedCueEnvelopeV1,
    expected_checkpoint_sha256: str,
    expected_factorized_checkpoint_sha256: str,
    allow_exploration: bool,
    operator_bindings: tuple[tuple[str, str], tuple[str, str]],
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> RustcLexicalUnorderedCueTextEditAuthorityReceiptV1:
    attempt = envelope.attempt
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_file_name": attempt.source_file_name,
        "source_sha256": attempt.source_sha256,
        "source_size_bytes": attempt.source_size_bytes,
        "diagnostic_attempt_sha256": attempt.sha256,
        "raw_stderr_sha256": sha256(envelope.stderr).hexdigest(),
        "raw_stderr_size_bytes": len(envelope.stderr),
        "selection_context_sha256": envelope.selection_context_sha256,
        "lexical_envelope_sha256": envelope.sha256,
        "projected_observation_sha256": envelope.observation.sha256,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "expected_factorized_checkpoint_sha256": expected_factorized_checkpoint_sha256,
        "allow_exploration": allow_exploration,
        "operator_bindings": operator_bindings,
        "target_bindings": target_bindings,
    }
    result = RustcLexicalUnorderedCueTextEditAuthorityReceiptV1(
        source_file_name=attempt.source_file_name,
        source_sha256=attempt.source_sha256,
        source_size_bytes=attempt.source_size_bytes,
        diagnostic_attempt_sha256=attempt.sha256,
        raw_stderr_sha256=payload["raw_stderr_sha256"],
        raw_stderr_size_bytes=len(envelope.stderr),
        selection_context_sha256=envelope.selection_context_sha256,
        lexical_envelope_sha256=envelope.sha256,
        projected_observation_sha256=envelope.observation.sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_factorized_checkpoint_sha256=expected_factorized_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=operator_bindings,
        target_bindings=target_bindings,
        sha256=_digest(_AUTHORITY_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class PreparedRustcLexicalUnorderedCueTextEditAuthorityV1:
    """One-shot authority from one derived lexical envelope to V2 execution."""

    def __init__(
        self,
        *,
        lexical_envelope: lexical.RustcLexicalUnorderedCueEnvelopeV1,
        source_bytes: bytes,
        operator_bindings: tuple[tuple[str, str], tuple[str, str]],
        target_bindings: tuple[tuple[str, str], tuple[str, str]],
        receipt: RustcLexicalUnorderedCueTextEditAuthorityReceiptV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("lexical unordered-cue text-edit authorities are factory-owned")
        self._lexical_envelope = lexical_envelope
        self._source_bytes = source_bytes
        self._operator_bindings = operator_bindings
        self._target_bindings = target_bindings
        self._receipt = receipt
        self._lock = Lock()
        self._consumed = False
        self._materialization: UnorderedCueFactorTextEditMaterializationV1 | None = None
        self._release: UnorderedCueFactorTextEditReleaseV1 | None = None
        self._materialization_count = 0
        self._release_count = 0
        self._execution_attempt_count = 0
        self._successful_execution_count = 0
        self._authenticate_current()

    def _authenticate_current(self) -> None:
        try:
            self._lexical_envelope.authenticate()
        except (TypeError, ValueError) as exc:
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical envelope authentication failed inside authority"
            ) from exc
        self._receipt.authenticate()
        attempt = self._lexical_envelope.attempt
        checked_source = edits._source_bytes(
            attempt,
            self._source_bytes.decode("ascii", errors="strict"),
        )
        try:
            reproduced_envelope = lexical.project_rustc_lexical_unordered_cue_observation_v1(
                attempt,
                stderr=self._lexical_envelope.stderr,
                selection_context_sha256=self._receipt.selection_context_sha256,
                active_hypothesis_sha256s=c8._active_hypothesis_sha256s(
                    self._operator_bindings,
                    self._target_bindings,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical envelope could not be rederived inside authority"
            ) from exc
        if (
            checked_source != self._source_bytes
            or reproduced_envelope != self._lexical_envelope
            or edits._canonical_operator_bindings(self._operator_bindings)
            != self._operator_bindings
            or edits._canonical_target_bindings(self._target_bindings) != self._target_bindings
            or self._receipt.source_file_name != attempt.source_file_name
            or self._receipt.source_sha256 != attempt.source_sha256
            or self._receipt.source_size_bytes != attempt.source_size_bytes
            or self._receipt.diagnostic_attempt_sha256 != attempt.sha256
            or self._receipt.raw_stderr_sha256 != sha256(self._lexical_envelope.stderr).hexdigest()
            or self._receipt.raw_stderr_size_bytes != len(self._lexical_envelope.stderr)
            or self._receipt.selection_context_sha256
            != self._lexical_envelope.selection_context_sha256
            or self._receipt.lexical_envelope_sha256 != self._lexical_envelope.sha256
            or self._receipt.projected_observation_sha256
            != self._lexical_envelope.observation.sha256
            or self._receipt.operator_bindings != self._operator_bindings
            or self._receipt.target_bindings != self._target_bindings
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical unordered-cue text-edit authority custody changed"
            )

    @property
    def lexical_envelope(self) -> lexical.RustcLexicalUnorderedCueEnvelopeV1:
        self._authenticate_current()
        return self._lexical_envelope

    @property
    def projected_observation(self) -> unordered.UnorderedCueFactorObservationV1:
        self._authenticate_current()
        return self._lexical_envelope.observation

    @property
    def receipt(self) -> RustcLexicalUnorderedCueTextEditAuthorityReceiptV1:
        self._authenticate_current()
        return self._receipt

    @property
    def authority_sha256(self) -> str:
        self._receipt.authenticate()
        return self._receipt.sha256

    @property
    def materialization(self) -> UnorderedCueFactorTextEditMaterializationV1 | None:
        return self._materialization

    @property
    def release(self) -> UnorderedCueFactorTextEditReleaseV1 | None:
        return self._release

    @property
    def materialization_count(self) -> int:
        return self._materialization_count

    @property
    def release_count(self) -> int:
        return self._release_count

    @property
    def work_counters(self) -> UnorderedCueFactorTextEditWorkV1:
        result = UnorderedCueFactorTextEditWorkV1(
            materialization_count=self._materialization_count,
            release_count=self._release_count,
            execution_attempt_count=self._execution_attempt_count,
            successful_execution_count=self._successful_execution_count,
        )
        result.authenticate()
        return result

    def execute_selected(
        self,
        selection: unordered.UnorderedCueFactorDecisionV1,
        batch: tuple[tuple[int, ...], ...],
    ) -> rust_executor.RustSelectedActionExecutionV2:
        """Materialize and execute exactly the outer lexical decision winner."""

        self._authenticate_current()
        if type(selection) is not unordered.UnorderedCueFactorDecisionV1:
            raise TypeError("selection must be one exact unordered-cue factor decision")
        selection.authenticate()
        inner = selection.factorized_decision
        if type(inner) is not factors.FactorizedFeedbackDecisionV1:
            raise TypeError("lexical selection must retain one exact factor decision")
        inner.authenticate()
        if (
            selection.status not in (unordered.READY, unordered.EXPLORATION_READY)
            or selection.selected_hypothesis_sha256 is None
            or not selection.cell_gate
            or not selection.outcome_gate
            or selection.axis_gates != (True, True)
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "only one enabled selected lexical decision can materialize"
            )
        if (
            selection.checkpoint_sha256 != self._receipt.expected_checkpoint_sha256
            or inner.checkpoint_sha256 != self._receipt.expected_factorized_checkpoint_sha256
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical decision checkpoint differs from authority"
            )
        if selection.observation != self._lexical_envelope.observation:
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical decision does not match envelope observation custody"
            )
        if not self._receipt.allow_exploration and (
            selection.status != unordered.READY
            or selection.exploration_hypothesis_sha256 is not None
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical unordered-cue text-edit authority forbids exploration"
            )

        hypotheses = tuple(score.hypothesis for score in selection.hypothesis_scores)
        operator_roots = {root for root, _operation in self._operator_bindings}
        target_roots = {root for root, _marker in self._target_bindings}
        expected_pairs = {
            (operator_root, target_root)
            for operator_root in operator_roots
            for target_root in target_roots
        }
        expected_hypotheses = set(
            c8._active_hypothesis_sha256s(
                self._operator_bindings,
                self._target_bindings,
            )
        )
        if (
            len(hypotheses) != 4
            or {hypothesis.component_sha256s for hypothesis in hypotheses} != expected_pairs
            or {hypothesis.sha256 for hypothesis in hypotheses} != expected_hypotheses
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical decision bank differs from authority bindings"
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
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical selected hypothesis left the bound bank"
            )

        with self._lock:
            if self._consumed:
                raise RustcLexicalUnorderedCueTextEditAuthorityError(
                    "lexical unordered-cue text-edit authority was consumed"
                )
            self._consumed = True

        prepared_execution = interventions._prepare_execution(self.authority_sha256, batch)
        operation = dict(self._operator_bindings)[selected.component_sha256s[0]]
        marker = dict(self._target_bindings)[selected.component_sha256s[1]]
        attempt = self._lexical_envelope.attempt
        primary_span = next(
            span for diagnostic in attempt.error_diagnostics for span in diagnostic.primary_spans
        )
        edit = edits._compose_edit(
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
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "materialized Rust source exceeds the bounded source policy"
            )
        materialization = c8._seal_materialization(
            authority_sha256=self.authority_sha256,
            decision=selection,
            attempt=attempt,
            selected_hypothesis=selected,
            primary_span=primary_span,
            original_source_sha256=self._receipt.source_sha256,
            edit=edit,
            materialized_source=materialized_source,
        )
        release = c8._seal_release(materialization)
        with self._lock:
            self._materialization = materialization
            self._release = release
            self._materialization_count = 1
            self._release_count = 1

        selected_source = rust_executor.make_rust_selected_source_v2(
            authority_sha256=self.authority_sha256,
            selection_sha256=selection.sha256,
            release_sha256=release.sha256,
            action_handle=release.action_handle,
            source=materialized_source.decode("ascii"),
        )
        if type(selected_source) is not rust_executor.RustSelectedSourceV2:
            raise TypeError("selected-source factory returned one inexact grant")
        selected_source.authenticate_structure()
        if (
            selected_source.authority_sha256 != self.authority_sha256
            or selected_source.selection_sha256 != selection.sha256
            or selected_source.release_sha256 != release.sha256
            or selected_source.action_handle != release.action_handle
            or selected_source.source_sha256 != release.source_sha256
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "selected-source grant differs from lexical release"
            )
        with self._lock:
            self._execution_attempt_count = 1

        def authenticate_current() -> None:
            self._authenticate_current()
            interventions._authenticate_execution_preparation(
                prepared_execution,
                authority_sha256=self.authority_sha256,
            )

        result = interventions._execute_prepared(
            selected_source,
            prepared_execution,
            authenticate_current=authenticate_current,
        )
        if type(result) is not rust_executor.RustSelectedActionExecutionV2:
            raise TypeError("lexical unordered-cue execution result must be exact V2")
        result.authenticate_structure()
        if (
            result.selected_source != selected_source
            or result.selected_source.authority_sha256 != self.authority_sha256
            or result.selected_source.selection_sha256 != selection.sha256
        ):
            raise RustcLexicalUnorderedCueTextEditAuthorityError(
                "lexical execution differs from selected-source custody"
            )
        with self._lock:
            self._successful_execution_count = 1
        return result


def prepare_rustc_lexical_unordered_cue_text_edit_authority_v1(
    *,
    diagnostic_attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    stderr: bytes,
    source: str,
    selection_context_sha256: str,
    expected_checkpoint_sha256: str,
    expected_factorized_checkpoint_sha256: str,
    allow_exploration: bool,
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> PreparedRustcLexicalUnorderedCueTextEditAuthorityV1:
    """Derive and freeze one lexical envelope, lineage, and edit catalogue."""

    if not _valid_sha256(selection_context_sha256):
        raise RustcLexicalUnorderedCueTextEditAuthorityError("selection context root is invalid")
    if not _valid_sha256(expected_checkpoint_sha256):
        raise RustcLexicalUnorderedCueTextEditAuthorityError(
            "expected unordered-cue checkpoint root is invalid"
        )
    if not _valid_sha256(expected_factorized_checkpoint_sha256):
        raise RustcLexicalUnorderedCueTextEditAuthorityError(
            "expected factorized checkpoint root is invalid"
        )
    if type(allow_exploration) is not bool:
        raise TypeError("allow_exploration must be one exact boolean")
    operator_bindings = edits._canonical_operator_bindings(
        catalogue.canonical_factorized_text_edit_operator_bindings_v1()
    )
    checked_targets = edits._canonical_target_bindings(target_bindings)
    if (
        len(
            {
                *(root for root, _operation in operator_bindings),
                *(root for root, _marker in checked_targets),
            }
        )
        != 4
    ):
        raise RustcLexicalUnorderedCueTextEditAuthorityError(
            "operator and target component roots must be distinct"
        )
    source_bytes = edits._source_bytes(diagnostic_attempt, source)
    for _root, marker in checked_targets:
        edits._marker_declaration_span(source_bytes, marker)
    if not any(diagnostic.primary_spans for diagnostic in diagnostic_attempt.error_diagnostics):
        raise RustcLexicalUnorderedCueTextEditAuthorityError(
            "text-edit authority requires one selected-source primary span"
        )
    try:
        envelope = lexical.project_rustc_lexical_unordered_cue_observation_v1(
            diagnostic_attempt,
            stderr=stderr,
            selection_context_sha256=selection_context_sha256,
            active_hypothesis_sha256s=c8._active_hypothesis_sha256s(
                operator_bindings,
                checked_targets,
            ),
        )
    except lexical.RustcLexicalUnorderedCueAdapterError as exc:
        raise RustcLexicalUnorderedCueTextEditAuthorityError(
            "lexical envelope derivation failed"
        ) from exc
    receipt = _make_receipt(
        envelope=envelope,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_factorized_checkpoint_sha256=expected_factorized_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=operator_bindings,
        target_bindings=checked_targets,
    )
    return PreparedRustcLexicalUnorderedCueTextEditAuthorityV1(
        lexical_envelope=envelope,
        source_bytes=source_bytes,
        operator_bindings=operator_bindings,
        target_bindings=checked_targets,
        receipt=receipt,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA",
    "PreparedRustcLexicalUnorderedCueTextEditAuthorityV1",
    "RustcLexicalUnorderedCueTextEditAuthorityError",
    "RustcLexicalUnorderedCueTextEditAuthorityReceiptV1",
    "UnorderedCueFactorTextEditMaterializationV1",
    "UnorderedCueFactorTextEditReleaseV1",
    "UnorderedCueFactorTextEditWorkV1",
    "prepare_rustc_lexical_unordered_cue_text_edit_authority_v1",
]
