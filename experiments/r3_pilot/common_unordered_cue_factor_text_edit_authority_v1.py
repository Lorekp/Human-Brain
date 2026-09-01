"""Decision-bound Rust text-edit authority for unordered-cue factor memory.

The authority keeps one failing Rust source under custody and accepts only one
authenticated :class:`UnorderedCueFactorDecisionV1`.  The outer decision stays
authoritative: its exact root is carried into the selected-source grant, while
the retained inner factor decision proves which concrete action/target pair won
the physical race.

This module deliberately does not route the decision through the older ordered
Rust factor projection.  It reuses that lane's bounded edit primitives and the
existing V2 compile/run implementation without changing either frozen module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock

from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_materializer_v1 as edits
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_rust_action_intervention_v1 as interventions
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_rustc_unordered_cue_factor_adapter_v1 as rustc_unordered
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "humanbrain.unordered-cue-factor-text-edit-authority.v1"
CLAIM_BOUNDARY = (
    "post-selection-materialization-of-one-authenticated-unordered-cue-factor-"
    "decision-using-one-host-owned-two-operation-two-target-rust-text-edit-"
    "catalogue-not-cue-or-action-discovery-localization-diagnosis-synthesis-"
    "search-debugging-transfer-swe-or-intelligence"
)

_SHA256_HEX = frozenset("0123456789abcdef")
_ACTION_HANDLE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")
_AUTHORITY_NAMESPACE = b"unordered-cue-factor-text-edit-authority-v1"
_MATERIALIZATION_NAMESPACE = b"unordered-cue-factor-text-edit-materialization-v1"
_RELEASE_NAMESPACE = b"unordered-cue-factor-text-edit-release-v1"
_CONSTRUCTION_TOKEN = object()


class UnorderedCueFactorTextEditAuthorityError(ValueError):
    """The decision, binding, source, materialization, or release changed."""


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


def _active_hypothesis_sha256s(
    operator_bindings: tuple[tuple[str, str], tuple[str, str]],
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> tuple[str, ...]:
    return tuple(
        factors.seal_factorized_hypothesis_v1((operator_root, target_root)).sha256
        for operator_root, _operation in operator_bindings
        for target_root, _marker in target_bindings
    )


@dataclass(frozen=True, slots=True)
class UnorderedCueFactorTextEditAuthorityReceiptV1:
    source_file_name: str
    source_sha256: str
    source_size_bytes: int
    diagnostic_attempt_sha256: str
    selection_context_sha256: str
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
            "selection_context_sha256": self.selection_context_sha256,
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
                    self.selection_context_sha256,
                    self.projected_observation_sha256,
                    self.expected_checkpoint_sha256,
                    self.expected_factorized_checkpoint_sha256,
                )
            )
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
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue text-edit authority receipt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class UnorderedCueFactorTextEditMaterializationV1:
    authority_sha256: str
    decision_sha256: str
    factorized_decision_sha256: str
    diagnostic_attempt_sha256: str
    observation_sha256: str
    binding_observation_sha256: str
    selected_hypothesis_sha256: str
    operator_action_sha256: str
    target_component_sha256: str
    primary_span_sha256: str
    original_source_sha256: str
    edit: edits.FactorizedTextEditV1
    materialized_source_sha256: str
    materialized_source_size_bytes: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "authority_sha256": self.authority_sha256,
            "decision_sha256": self.decision_sha256,
            "factorized_decision_sha256": self.factorized_decision_sha256,
            "diagnostic_attempt_sha256": self.diagnostic_attempt_sha256,
            "observation_sha256": self.observation_sha256,
            "binding_observation_sha256": self.binding_observation_sha256,
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "operator_action_sha256": self.operator_action_sha256,
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
        if type(self.edit) is not edits.FactorizedTextEditV1:
            raise TypeError("unordered-cue materialization edit must be exact")
        self.edit.authenticate()
        if (
            any(
                not _valid_sha256(value)
                for value in (
                    self.authority_sha256,
                    self.decision_sha256,
                    self.factorized_decision_sha256,
                    self.diagnostic_attempt_sha256,
                    self.observation_sha256,
                    self.binding_observation_sha256,
                    self.selected_hypothesis_sha256,
                    self.operator_action_sha256,
                    self.target_component_sha256,
                    self.primary_span_sha256,
                    self.original_source_sha256,
                    self.materialized_source_sha256,
                )
            )
            or type(self.materialized_source_size_bytes) is not int
            or not 1 <= self.materialized_source_size_bytes <= rustc_feedback.MAX_SOURCE_BYTES
            or self.sha256 != _digest(_MATERIALIZATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue text-edit materialization authentication failed"
            )


@dataclass(frozen=True, slots=True)
class UnorderedCueFactorTextEditReleaseV1:
    authority_sha256: str
    selection_sha256: str
    factorized_decision_sha256: str
    action_handle: str
    source_sha256: str
    materialization: UnorderedCueFactorTextEditMaterializationV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "authority_sha256": self.authority_sha256,
            "selection_sha256": self.selection_sha256,
            "factorized_decision_sha256": self.factorized_decision_sha256,
            "action_handle": self.action_handle,
            "source_sha256": self.source_sha256,
            "materialization": self.materialization.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.materialization) is not UnorderedCueFactorTextEditMaterializationV1:
            raise TypeError("unordered-cue release materialization must be exact")
        self.materialization.authenticate()
        if (
            not _valid_sha256(self.authority_sha256)
            or not _valid_sha256(self.selection_sha256)
            or not _valid_sha256(self.factorized_decision_sha256)
            or type(self.action_handle) is not str
            or _ACTION_HANDLE.fullmatch(self.action_handle) is None
            or not _valid_sha256(self.source_sha256)
            or self.authority_sha256 != self.materialization.authority_sha256
            or self.selection_sha256 != self.materialization.decision_sha256
            or self.factorized_decision_sha256 != self.materialization.factorized_decision_sha256
            or self.source_sha256 != self.materialization.materialized_source_sha256
            or self.sha256 != _digest(_RELEASE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue text-edit release authentication failed"
            )


@dataclass(frozen=True, slots=True)
class UnorderedCueFactorTextEditWorkV1:
    materialization_count: int
    release_count: int
    execution_attempt_count: int
    successful_execution_count: int

    def authenticate(self) -> None:
        values = (
            self.materialization_count,
            self.release_count,
            self.execution_attempt_count,
            self.successful_execution_count,
        )
        if (
            any(type(value) is not int or value not in (0, 1) for value in values)
            or self.release_count > self.materialization_count
            or self.execution_attempt_count > self.release_count
            or self.successful_execution_count > self.execution_attempt_count
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue text-edit work counters changed"
            )


def _make_receipt(
    *,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    selection_context_sha256: str,
    projected_observation: unordered.UnorderedCueFactorObservationV1,
    expected_checkpoint_sha256: str,
    expected_factorized_checkpoint_sha256: str,
    allow_exploration: bool,
    operator_bindings: tuple[tuple[str, str], tuple[str, str]],
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> UnorderedCueFactorTextEditAuthorityReceiptV1:
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_file_name": attempt.source_file_name,
        "source_sha256": attempt.source_sha256,
        "source_size_bytes": attempt.source_size_bytes,
        "diagnostic_attempt_sha256": attempt.sha256,
        "selection_context_sha256": selection_context_sha256,
        "projected_observation_sha256": projected_observation.sha256,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "expected_factorized_checkpoint_sha256": expected_factorized_checkpoint_sha256,
        "allow_exploration": allow_exploration,
        "operator_bindings": operator_bindings,
        "target_bindings": target_bindings,
    }
    result = UnorderedCueFactorTextEditAuthorityReceiptV1(
        source_file_name=attempt.source_file_name,
        source_sha256=attempt.source_sha256,
        source_size_bytes=attempt.source_size_bytes,
        diagnostic_attempt_sha256=attempt.sha256,
        selection_context_sha256=selection_context_sha256,
        projected_observation_sha256=projected_observation.sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_factorized_checkpoint_sha256=expected_factorized_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=operator_bindings,
        target_bindings=target_bindings,
        sha256=_digest(_AUTHORITY_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _seal_materialization(
    *,
    authority_sha256: str,
    decision: unordered.UnorderedCueFactorDecisionV1,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    selected_hypothesis: factors.FactorizedHypothesisV1,
    primary_span: rustc_feedback.RustcPrimarySpanV1,
    original_source_sha256: str,
    edit: edits.FactorizedTextEditV1,
    materialized_source: bytes,
) -> UnorderedCueFactorTextEditMaterializationV1:
    payload = {
        "authority_sha256": authority_sha256,
        "decision_sha256": decision.sha256,
        "factorized_decision_sha256": decision.factorized_decision.sha256,
        "diagnostic_attempt_sha256": attempt.sha256,
        "observation_sha256": decision.observation.sha256,
        "binding_observation_sha256": decision.binding_observation.sha256,
        "selected_hypothesis_sha256": selected_hypothesis.sha256,
        "operator_action_sha256": selected_hypothesis.component_sha256s[0],
        "target_component_sha256": selected_hypothesis.component_sha256s[1],
        "primary_span_sha256": primary_span.sha256,
        "original_source_sha256": original_source_sha256,
        "edit": edit.record(),
        "materialized_source_sha256": sha256(materialized_source).hexdigest(),
        "materialized_source_size_bytes": len(materialized_source),
    }
    result = UnorderedCueFactorTextEditMaterializationV1(
        authority_sha256=authority_sha256,
        decision_sha256=decision.sha256,
        factorized_decision_sha256=decision.factorized_decision.sha256,
        diagnostic_attempt_sha256=attempt.sha256,
        observation_sha256=decision.observation.sha256,
        binding_observation_sha256=decision.binding_observation.sha256,
        selected_hypothesis_sha256=selected_hypothesis.sha256,
        operator_action_sha256=selected_hypothesis.component_sha256s[0],
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
    materialization: UnorderedCueFactorTextEditMaterializationV1,
) -> UnorderedCueFactorTextEditReleaseV1:
    action_handle = f"unordered_factor_{materialization.selected_hypothesis_sha256}"
    payload = {
        "authority_sha256": materialization.authority_sha256,
        "selection_sha256": materialization.decision_sha256,
        "factorized_decision_sha256": materialization.factorized_decision_sha256,
        "action_handle": action_handle,
        "source_sha256": materialization.materialized_source_sha256,
        "materialization": materialization.record(),
    }
    result = UnorderedCueFactorTextEditReleaseV1(
        authority_sha256=materialization.authority_sha256,
        selection_sha256=materialization.decision_sha256,
        factorized_decision_sha256=materialization.factorized_decision_sha256,
        action_handle=action_handle,
        source_sha256=materialization.materialized_source_sha256,
        materialization=materialization,
        sha256=_digest(_RELEASE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class PreparedUnorderedCueFactorTextEditAuthorityV1:
    """One-shot authority from one outer C8 decision to one V2 execution."""

    def __init__(
        self,
        *,
        attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
        source_bytes: bytes,
        projected_observation: unordered.UnorderedCueFactorObservationV1,
        operator_bindings: tuple[tuple[str, str], tuple[str, str]],
        target_bindings: tuple[tuple[str, str], tuple[str, str]],
        receipt: UnorderedCueFactorTextEditAuthorityReceiptV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("unordered-cue text-edit authorities are factory-owned")
        self._attempt = attempt
        self._source_bytes = source_bytes
        self._projected_observation = projected_observation
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
        self._attempt.authenticate()
        self._projected_observation.authenticate()
        self._receipt.authenticate()
        checked_source = edits._source_bytes(
            self._attempt,
            self._source_bytes.decode("ascii", errors="strict"),
        )
        reproduced_observation = rustc_unordered.project_rust_unordered_cue_factor_observation_v1(
            self._attempt,
            selection_context_sha256=self._receipt.selection_context_sha256,
            active_hypothesis_sha256s=_active_hypothesis_sha256s(
                self._operator_bindings,
                self._target_bindings,
            ),
        )
        if (
            checked_source != self._source_bytes
            or reproduced_observation != self._projected_observation
            or edits._canonical_operator_bindings(self._operator_bindings)
            != self._operator_bindings
            or edits._canonical_target_bindings(self._target_bindings) != self._target_bindings
            or self._receipt.source_file_name != self._attempt.source_file_name
            or self._receipt.source_sha256 != self._attempt.source_sha256
            or self._receipt.source_size_bytes != self._attempt.source_size_bytes
            or self._receipt.diagnostic_attempt_sha256 != self._attempt.sha256
            or self._receipt.projected_observation_sha256 != self._projected_observation.sha256
            or self._receipt.operator_bindings != self._operator_bindings
            or self._receipt.target_bindings != self._target_bindings
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue text-edit authority custody changed"
            )

    @property
    def receipt(self) -> UnorderedCueFactorTextEditAuthorityReceiptV1:
        self._authenticate_current()
        return self._receipt

    @property
    def projected_observation(self) -> unordered.UnorderedCueFactorObservationV1:
        self._authenticate_current()
        return self._projected_observation

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
        """Materialize and execute exactly the outer decision's physical winner."""

        self._authenticate_current()
        if type(selection) is not unordered.UnorderedCueFactorDecisionV1:
            raise TypeError("selection must be one exact unordered-cue factor decision")
        selection.authenticate()
        inner = selection.factorized_decision
        if type(inner) is not factors.FactorizedFeedbackDecisionV1:
            raise TypeError("unordered-cue selection must retain one exact factor decision")
        inner.authenticate()
        if (
            selection.status not in (unordered.READY, unordered.EXPLORATION_READY)
            or selection.selected_hypothesis_sha256 is None
            or not selection.cell_gate
            or not selection.outcome_gate
            or selection.axis_gates != (True, True)
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "only one enabled selected unordered-cue decision can materialize"
            )
        if (
            selection.checkpoint_sha256 != self._receipt.expected_checkpoint_sha256
            or inner.checkpoint_sha256 != self._receipt.expected_factorized_checkpoint_sha256
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue decision checkpoint differs from authority"
            )
        if selection.observation != self._projected_observation:
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue decision does not match diagnostic observation custody"
            )
        if not self._receipt.allow_exploration and (
            selection.status != unordered.READY
            or selection.exploration_hypothesis_sha256 is not None
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue text-edit authority forbids exploration"
            )

        hypotheses = tuple(score.hypothesis for score in selection.hypothesis_scores)
        operator_roots = {root for root, _operation in self._operator_bindings}
        target_roots = {root for root, _marker in self._target_bindings}
        expected_pairs = {
            (operator_root, target_root)
            for operator_root in operator_roots
            for target_root in target_roots
        }
        if (
            len(hypotheses) != 4
            or {hypothesis.component_sha256s for hypothesis in hypotheses} != expected_pairs
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue decision bank differs from authority bindings"
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
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue selected hypothesis left the bound bank"
            )

        with self._lock:
            if self._consumed:
                raise UnorderedCueFactorTextEditAuthorityError(
                    "unordered-cue text-edit authority was consumed"
                )
            self._consumed = True

        prepared_execution = interventions._prepare_execution(self.authority_sha256, batch)
        operation = dict(self._operator_bindings)[selected.component_sha256s[0]]
        marker = dict(self._target_bindings)[selected.component_sha256s[1]]
        primary_span = next(
            span
            for diagnostic in self._attempt.error_diagnostics
            for span in diagnostic.primary_spans
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
            raise UnorderedCueFactorTextEditAuthorityError(
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
            raise UnorderedCueFactorTextEditAuthorityError(
                "selected-source grant differs from unordered-cue release"
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
            raise TypeError("unordered-cue execution result must be exact V2")
        result.authenticate_structure()
        if (
            result.selected_source != selected_source
            or result.selected_source.authority_sha256 != self.authority_sha256
            or result.selected_source.selection_sha256 != selection.sha256
        ):
            raise UnorderedCueFactorTextEditAuthorityError(
                "unordered-cue execution differs from selected-source custody"
            )
        with self._lock:
            self._successful_execution_count = 1
        return result


def prepare_unordered_cue_factor_text_edit_authority_v1(
    *,
    diagnostic_attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    source: str,
    selection_context_sha256: str,
    expected_checkpoint_sha256: str,
    expected_factorized_checkpoint_sha256: str,
    allow_exploration: bool,
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> PreparedUnorderedCueFactorTextEditAuthorityV1:
    """Project and freeze one diagnostic, decision lineage, and edit catalogue."""

    if not _valid_sha256(selection_context_sha256):
        raise UnorderedCueFactorTextEditAuthorityError("selection context root is invalid")
    if not _valid_sha256(expected_checkpoint_sha256):
        raise UnorderedCueFactorTextEditAuthorityError(
            "expected unordered-cue checkpoint root is invalid"
        )
    if not _valid_sha256(expected_factorized_checkpoint_sha256):
        raise UnorderedCueFactorTextEditAuthorityError(
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
        raise UnorderedCueFactorTextEditAuthorityError(
            "operator and target component roots must be distinct"
        )
    source_bytes = edits._source_bytes(diagnostic_attempt, source)
    for _root, marker in checked_targets:
        edits._marker_declaration_span(source_bytes, marker)
    projected_observation = rustc_unordered.project_rust_unordered_cue_factor_observation_v1(
        diagnostic_attempt,
        selection_context_sha256=selection_context_sha256,
        active_hypothesis_sha256s=_active_hypothesis_sha256s(
            operator_bindings,
            checked_targets,
        ),
    )
    receipt = _make_receipt(
        attempt=diagnostic_attempt,
        selection_context_sha256=selection_context_sha256,
        projected_observation=projected_observation,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_factorized_checkpoint_sha256=expected_factorized_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=operator_bindings,
        target_bindings=checked_targets,
    )
    return PreparedUnorderedCueFactorTextEditAuthorityV1(
        attempt=diagnostic_attempt,
        source_bytes=source_bytes,
        projected_observation=projected_observation,
        operator_bindings=operator_bindings,
        target_bindings=checked_targets,
        receipt=receipt,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA",
    "PreparedUnorderedCueFactorTextEditAuthorityV1",
    "UnorderedCueFactorTextEditAuthorityError",
    "UnorderedCueFactorTextEditAuthorityReceiptV1",
    "UnorderedCueFactorTextEditMaterializationV1",
    "UnorderedCueFactorTextEditReleaseV1",
    "UnorderedCueFactorTextEditWorkV1",
    "prepare_unordered_cue_factor_text_edit_authority_v1",
]
