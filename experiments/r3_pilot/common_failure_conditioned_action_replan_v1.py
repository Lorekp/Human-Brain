"""One-decision refractory replanning after an exact failed action outcome.

The durable inhibitory tick remains owned by the existing continual outcome
memory.  This module adds one context-bound two-point pulse to the failed
action's next base activation, then asks the immutable successor memory to run
the full action race once.  The pulse is not persisted beyond that decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256

from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as outcomes

SCHEMA = "COMMON_FAILURE_CONDITIONED_ACTION_REPLAN_V1"
CLAIM_BOUNDARY = (
    "one-context-bound-two-point-refractory-pulse-and-one-physical-rerace-after-an-"
    "exact-caller-attested-failure-in-a-fixed-opaque-action-catalogue-not-durable-"
    "refractory-state-action-discovery-diagnosis-source-synthesis-or-intelligence"
)

REPLANNED = "REPLANNED"
REFRACTORY_OFF = "REFRACTORY_OFF"
STOP_GATE_OFF = "STOP_GATE_OFF"
REFRACTORY_SCORE_PULSE = 2

_PULSE_CONTEXT_NAMESPACE = b"common-failure-conditioned-action-replan-pulse-context-v1"
_TRACE_NAMESPACE = b"common-failure-conditioned-action-replan-trace-v1"


class FailureConditionedActionReplanError(ValueError):
    """A failure-conditioned replan request or trace failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class FailureConditionedActionReplanV1:
    """Authenticated outcome advance plus at most one refractory decision."""

    status: str
    parent: outcomes.FrozenContinualActionOutcomeMemoryV1
    failed_outcome: outcomes.SelectedActionOutcomeV1
    successor: outcomes.FrozenContinualActionOutcomeMemoryV1
    original_activation: outcomes.ActionActivationV1
    pulsed_activation: outcomes.ActionActivationV1 | None
    decision: outcomes.ActionDecisionV1 | None
    plasticity_gate: bool
    consequence_gate: bool
    refractory_gate: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "parent_checkpoint_sha256": self.parent.checkpoint_sha256,
            "failed_outcome_sha256": self.failed_outcome.sha256,
            "failed_action_sha256": self.failed_outcome.selected_action_sha256,
            "successor_checkpoint_sha256": self.successor.checkpoint_sha256,
            "original_activation_sha256": self.original_activation.sha256,
            "pulsed_activation_sha256": (
                None if self.pulsed_activation is None else self.pulsed_activation.sha256
            ),
            "decision_sha256": None if self.decision is None else self.decision.sha256,
            "decision_count": int(self.decision is not None),
            "refractory_score_pulse": (
                REFRACTORY_SCORE_PULSE if self.pulsed_activation is not None else 0
            ),
            "plasticity_gate": self.plasticity_gate,
            "consequence_gate": self.consequence_gate,
            "refractory_gate": self.refractory_gate,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.parent) is not outcomes.FrozenContinualActionOutcomeMemoryV1:
            raise TypeError("replan parent must be exact")
        if type(self.failed_outcome) is not outcomes.SelectedActionOutcomeV1:
            raise TypeError("replan failed outcome must be exact")
        if type(self.successor) is not outcomes.FrozenContinualActionOutcomeMemoryV1:
            raise TypeError("replan successor must be exact")
        if type(self.original_activation) is not outcomes.ActionActivationV1:
            raise TypeError("replan original activation must be exact")
        if self.pulsed_activation is not None and (
            type(self.pulsed_activation) is not outcomes.ActionActivationV1
        ):
            raise TypeError("replan pulsed activation must be exact")
        if self.decision is not None and type(self.decision) is not outcomes.ActionDecisionV1:
            raise TypeError("replan decision must be exact")
        if any(
            type(gate) is not bool
            for gate in (
                self.plasticity_gate,
                self.consequence_gate,
                self.refractory_gate,
            )
        ):
            raise TypeError("replan gates must be exact booleans")

        self.parent.authenticate()
        self.failed_outcome.authenticate()
        self.successor.authenticate()
        self.original_activation.authenticate()
        if self.pulsed_activation is not None:
            self.pulsed_activation.authenticate()
        if self.decision is not None:
            self.decision.authenticate()

        if self.status == STOP_GATE_OFF:
            shape_is_valid = (
                self.successor is self.parent
                and self.pulsed_activation is None
                and self.decision is None
            )
        elif self.status == REFRACTORY_OFF:
            shape_is_valid = (
                self.successor is not self.parent
                and self.pulsed_activation is None
                and self.decision is not None
            )
        elif self.status == REPLANNED:
            shape_is_valid = (
                self.successor is not self.parent
                and self.pulsed_activation is not None
                and self.decision is not None
            )
        else:
            shape_is_valid = False
        if not shape_is_valid:
            raise FailureConditionedActionReplanError("replan trace shape changed")

        expected = _derive_replan(
            self.parent,
            self.failed_outcome,
            self.original_activation,
            plasticity_gate=self.plasticity_gate,
            consequence_gate=self.consequence_gate,
            refractory_gate=self.refractory_gate,
        )
        if self.record() != expected.record():
            raise FailureConditionedActionReplanError("replan trace does not rederive")


def _validate_request(
    parent: outcomes.FrozenContinualActionOutcomeMemoryV1,
    failed_outcome: outcomes.SelectedActionOutcomeV1,
    original_activation: outcomes.ActionActivationV1,
    *,
    plasticity_gate: bool,
    consequence_gate: bool,
    refractory_gate: bool,
) -> None:
    if type(parent) is not outcomes.FrozenContinualActionOutcomeMemoryV1:
        raise TypeError("replan requires one exact parent")
    if type(failed_outcome) is not outcomes.SelectedActionOutcomeV1:
        raise TypeError("replan requires one exact selected action outcome")
    if type(original_activation) is not outcomes.ActionActivationV1:
        raise TypeError("replan requires one exact next activation")
    if any(type(gate) is not bool for gate in (plasticity_gate, consequence_gate, refractory_gate)):
        raise TypeError("replan gates must be exact booleans")

    parent.authenticate()
    failed_outcome.authenticate()
    original_activation.authenticate()
    if failed_outcome.passed_cell != 0:
        raise FailureConditionedActionReplanError("replanning requires an exact failed outcome")

    failed_activation = failed_outcome.decision.activation
    if (
        len(failed_activation.active_action_sha256s) < 2
        or original_activation.active_action_sha256s != failed_activation.active_action_sha256s
        or original_activation.base_scores != failed_activation.base_scores
    ):
        raise FailureConditionedActionReplanError(
            "next activation must preserve the competing action race"
        )
    spent_contexts = {row.selection_context_sha256 for row in parent.provenance} | {
        failed_activation.selection_context_sha256
    }
    if original_activation.selection_context_sha256 in spent_contexts:
        raise FailureConditionedActionReplanError("next selection context is not fresh")


def _seal_pulsed_activation(
    *,
    parent: outcomes.FrozenContinualActionOutcomeMemoryV1,
    failed_outcome: outcomes.SelectedActionOutcomeV1,
    successor: outcomes.FrozenContinualActionOutcomeMemoryV1,
    original_activation: outcomes.ActionActivationV1,
) -> outcomes.ActionActivationV1:
    failed_action = failed_outcome.selected_action_sha256
    failed_position = original_activation.active_action_sha256s.index(failed_action)
    original_score = original_activation.base_scores[failed_position]
    if original_score < REFRACTORY_SCORE_PULSE:
        raise FailureConditionedActionReplanError(
            "failed action cannot receive one exact refractory score pulse"
        )

    pulsed_scores = list(original_activation.base_scores)
    pulsed_scores[failed_position] -= REFRACTORY_SCORE_PULSE
    context_payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "successor_checkpoint_sha256": successor.checkpoint_sha256,
        "failed_outcome_sha256": failed_outcome.sha256,
        "failed_action_sha256": failed_action,
        "original_activation_sha256": original_activation.sha256,
        "original_selection_context_sha256": original_activation.selection_context_sha256,
        "refractory_score_pulse": REFRACTORY_SCORE_PULSE,
        "decision_count": 1,
    }
    return outcomes.seal_action_activation_v1(
        selection_context_sha256=_digest(_PULSE_CONTEXT_NAMESPACE, context_payload),
        active_action_sha256s=original_activation.active_action_sha256s,
        base_scores=tuple(pulsed_scores),
    )


def _seal_trace(
    *,
    status: str,
    parent: outcomes.FrozenContinualActionOutcomeMemoryV1,
    failed_outcome: outcomes.SelectedActionOutcomeV1,
    successor: outcomes.FrozenContinualActionOutcomeMemoryV1,
    original_activation: outcomes.ActionActivationV1,
    pulsed_activation: outcomes.ActionActivationV1 | None,
    decision: outcomes.ActionDecisionV1 | None,
    plasticity_gate: bool,
    consequence_gate: bool,
    refractory_gate: bool,
) -> FailureConditionedActionReplanV1:
    partial = FailureConditionedActionReplanV1(
        status=status,
        parent=parent,
        failed_outcome=failed_outcome,
        successor=successor,
        original_activation=original_activation,
        pulsed_activation=pulsed_activation,
        decision=decision,
        plasticity_gate=plasticity_gate,
        consequence_gate=consequence_gate,
        refractory_gate=refractory_gate,
        sha256="",
    )
    return replace(
        partial,
        sha256=_digest(_TRACE_NAMESPACE, partial.record(include_sha256=False)),
    )


def _derive_replan(
    parent: outcomes.FrozenContinualActionOutcomeMemoryV1,
    failed_outcome: outcomes.SelectedActionOutcomeV1,
    original_activation: outcomes.ActionActivationV1,
    *,
    plasticity_gate: bool,
    consequence_gate: bool,
    refractory_gate: bool,
) -> FailureConditionedActionReplanV1:
    _validate_request(
        parent,
        failed_outcome,
        original_activation,
        plasticity_gate=plasticity_gate,
        consequence_gate=consequence_gate,
        refractory_gate=refractory_gate,
    )
    successor = outcomes.advance_continual_action_outcome_memory_v1(
        parent,
        failed_outcome,
        plasticity_gate=plasticity_gate,
        consequence_gate=consequence_gate,
    )
    if successor is parent:
        return _seal_trace(
            status=STOP_GATE_OFF,
            parent=parent,
            failed_outcome=failed_outcome,
            successor=parent,
            original_activation=original_activation,
            pulsed_activation=None,
            decision=None,
            plasticity_gate=plasticity_gate,
            consequence_gate=consequence_gate,
            refractory_gate=refractory_gate,
        )

    if refractory_gate:
        pulsed_activation = _seal_pulsed_activation(
            parent=parent,
            failed_outcome=failed_outcome,
            successor=successor,
            original_activation=original_activation,
        )
        decision_activation = pulsed_activation
        status = REPLANNED
    else:
        pulsed_activation = None
        decision_activation = original_activation
        status = REFRACTORY_OFF
    decision = successor.decide(decision_activation)
    return _seal_trace(
        status=status,
        parent=parent,
        failed_outcome=failed_outcome,
        successor=successor,
        original_activation=original_activation,
        pulsed_activation=pulsed_activation,
        decision=decision,
        plasticity_gate=plasticity_gate,
        consequence_gate=consequence_gate,
        refractory_gate=refractory_gate,
    )


def advance_and_replan_after_failure_v1(
    parent: outcomes.FrozenContinualActionOutcomeMemoryV1,
    failed_outcome: outcomes.SelectedActionOutcomeV1,
    next_activation: outcomes.ActionActivationV1,
    *,
    plasticity_gate: bool = True,
    consequence_gate: bool = True,
    refractory_gate: bool = True,
) -> FailureConditionedActionReplanV1:
    """Advance one exact FAIL, then run at most one full-catalogue rerace."""

    result = _derive_replan(
        parent,
        failed_outcome,
        next_activation,
        plasticity_gate=plasticity_gate,
        consequence_gate=consequence_gate,
        refractory_gate=refractory_gate,
    )
    result.authenticate()
    return result


__all__ = [
    "CLAIM_BOUNDARY",
    "REFRACTORY_OFF",
    "REFRACTORY_SCORE_PULSE",
    "REPLANNED",
    "SCHEMA",
    "STOP_GATE_OFF",
    "FailureConditionedActionReplanError",
    "FailureConditionedActionReplanV1",
    "advance_and_replan_after_failure_v1",
]
