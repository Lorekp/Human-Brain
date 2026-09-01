"""Authenticated concrete operator catalogue for factorized text edits.

The catalogue owns the two action identities understood by the existing
factorized text-edit materializer.  Callers may choose target-marker bindings,
but cannot substitute a different operator root-to-operation mapping through
this factory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import common_factorized_text_edit_materializer_v1 as materializer
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback

SCHEMA = "COMMON_FACTORIZED_TEXT_EDIT_OPERATOR_CATALOGUE_V1"
CLAIM_BOUNDARY = (
    "authenticated-host-owned-two-action-text-edit-operator-catalogue-not-"
    "operation-discovery-diagnosis-repair-correctness-search-synthesis-swe-or-"
    "intelligence"
)

_ACTION_NAMESPACE = b"common-factorized-text-edit-operator-action-v1"
_OPERATIONS = frozenset(
    (
        materializer.DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR,
        materializer.REPLACE_PRIMARY_SPAN_WITH_ANCHOR,
    )
)


class FactorizedTextEditOperatorCatalogueError(ValueError):
    """A sealed catalogue action identity failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _action_sha256(operation: str) -> str:
    return sha256(
        _ACTION_NAMESPACE + b"\0" + _canonical_json({"operation": operation, "schema": SCHEMA})
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FactorizedTextEditOperatorActionV1:
    """One rooted concrete edit operation owned by this catalogue."""

    operation: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "operation": self.operation,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            type(self.operation) is not str
            or self.operation not in _OPERATIONS
            or type(self.sha256) is not str
            or self.sha256 != _action_sha256(self.operation)
        ):
            raise FactorizedTextEditOperatorCatalogueError(
                "factorized text-edit operator action authentication failed"
            )


def _seal_action(operation: str) -> FactorizedTextEditOperatorActionV1:
    action = FactorizedTextEditOperatorActionV1(
        operation=operation,
        sha256=_action_sha256(operation),
    )
    action.authenticate()
    return action


DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR_ACTION_V1 = _seal_action(
    materializer.DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR
)
REPLACE_PRIMARY_SPAN_WITH_ANCHOR_ACTION_V1 = _seal_action(
    materializer.REPLACE_PRIMARY_SPAN_WITH_ANCHOR
)

_CANONICAL_ACTIONS = tuple(
    sorted(
        (
            DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR_ACTION_V1,
            REPLACE_PRIMARY_SPAN_WITH_ANCHOR_ACTION_V1,
        ),
        key=lambda action: action.sha256,
    )
)


def canonical_factorized_text_edit_operator_actions_v1() -> tuple[
    FactorizedTextEditOperatorActionV1, FactorizedTextEditOperatorActionV1
]:
    """Return the authenticated catalogue in canonical root order."""

    for action in _CANONICAL_ACTIONS:
        if type(action) is not FactorizedTextEditOperatorActionV1:
            raise TypeError("factorized text-edit operator actions must be exact")
        action.authenticate()
    return _CANONICAL_ACTIONS


def canonical_factorized_text_edit_operator_action_roots_v1() -> tuple[str, str]:
    """Return the two canonical action roots for neuronal action memories."""

    actions = canonical_factorized_text_edit_operator_actions_v1()
    return (actions[0].sha256, actions[1].sha256)


def canonical_factorized_text_edit_operator_bindings_v1() -> tuple[
    tuple[str, str], tuple[str, str]
]:
    """Return the sealed action-root to concrete-operation bindings."""

    actions = canonical_factorized_text_edit_operator_actions_v1()
    return (
        (actions[0].sha256, actions[0].operation),
        (actions[1].sha256, actions[1].operation),
    )


def prepare_factorized_text_edit_operator_catalogue_v1(
    *,
    diagnostic_attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    source: str,
    selection_context_sha256: str,
    expected_checkpoint_sha256: str,
    allow_exploration: bool,
    target_bindings: tuple[tuple[str, str], tuple[str, str]],
) -> materializer.PreparedFactorizedTextEditMaterializerV1:
    """Prepare the existing authority with this catalogue's sealed operators."""

    return materializer.prepare_factorized_text_edit_materializer_v1(
        diagnostic_attempt=diagnostic_attempt,
        source=source,
        selection_context_sha256=selection_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        allow_exploration=allow_exploration,
        operator_bindings=canonical_factorized_text_edit_operator_bindings_v1(),
        target_bindings=target_bindings,
    )
