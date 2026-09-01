from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as outcomes
from experiments.r3_pilot import common_failure_conditioned_action_replan_v1 as replans


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _activation(
    context_label: str,
    scored_actions: tuple[tuple[str, int], ...],
) -> outcomes.ActionActivationV1:
    return outcomes.seal_action_activation_v1(
        selection_context_sha256=_root(context_label),
        active_action_sha256s=tuple(action for action, _ in scored_actions),
        base_scores=tuple(score for _, score in scored_actions),
    )


def _scores(decision: outcomes.ActionDecisionV1) -> dict[str, int]:
    return dict(
        zip(
            decision.activation.active_action_sha256s,
            decision.effective_scores,
            strict=True,
        )
    )


def test_failed_winner_gets_one_authenticated_refractory_rerace() -> None:
    action_a = _root("failure-replan-action-a")
    action_b = _root("failure-replan-action-b")
    parent = outcomes.start_continual_action_outcome_memory_v1((action_a, action_b))
    parent_root = parent.checkpoint_sha256
    parent_arrays = parent.checkpoint_array_sha256s

    initial_activation = _activation(
        "failure-replan-initial-context",
        ((action_a, 6), (action_b, 4)),
    )
    initial_decision = parent.decide(initial_activation)
    assert initial_decision.status == outcomes.READY
    assert initial_decision.selected_action_sha256 == action_a
    assert _scores(initial_decision) == {action_a: 6, action_b: 4}

    failed_outcome = outcomes.seal_selected_action_outcome_v1(
        decision=initial_decision,
        proposed_source_sha256=_root("failure-replan-proposed-source"),
        execution_sha256=_root("failure-replan-execution"),
        consequence_sha256=_root("failure-replan-consequence"),
        auxiliary_event_sha256s=(),
        passed_cell=0,
    )
    refractory_off = replans.advance_and_replan_after_failure_v1(
        parent,
        failed_outcome,
        refractory_gate=False,
    )
    refractory_off.authenticate()
    assert refractory_off.status == replans.REFRACTORY_OFF
    assert refractory_off.retry_activation is not None
    assert refractory_off.retry_activation.active_action_sha256s == (
        initial_activation.active_action_sha256s
    )
    assert refractory_off.retry_activation.base_scores == initial_activation.base_scores
    assert (
        refractory_off.retry_activation.selection_context_sha256
        != initial_activation.selection_context_sha256
    )
    assert refractory_off.pulsed_activation is None
    assert refractory_off.decision is not None
    assert refractory_off.decision.activation is refractory_off.retry_activation
    assert refractory_off.decision.status == outcomes.STOP_PHYSICAL_RACE
    assert refractory_off.decision.selected_action_sha256 is None
    assert _scores(refractory_off.decision) == {action_a: 4, action_b: 4}

    result = replans.advance_and_replan_after_failure_v1(
        parent,
        failed_outcome,
    )
    result.authenticate()
    repeated = replans.advance_and_replan_after_failure_v1(
        parent,
        failed_outcome,
    )
    repeated.authenticate()
    assert result.sha256 == repeated.sha256
    assert result.record() == repeated.record()
    assert result.retry_activation == repeated.retry_activation
    assert result.pulsed_activation == repeated.pulsed_activation
    assert result.decision == repeated.decision

    assert result.status == replans.REFRACTORY_RERACED
    assert result.failed_outcome is failed_outcome
    assert result.retry_activation == refractory_off.retry_activation
    assert result.successor.checkpoint_sha256 == refractory_off.successor.checkpoint_sha256
    assert result.pulsed_activation is not None
    assert result.retry_activation is not None
    assert result.pulsed_activation.active_action_sha256s == (
        result.retry_activation.active_action_sha256s
    )
    assert dict(
        zip(
            result.pulsed_activation.active_action_sha256s,
            result.pulsed_activation.base_scores,
            strict=True,
        )
    ) == {action_a: 4, action_b: 4}
    assert result.decision is not None
    assert result.decision.activation is result.pulsed_activation
    assert result.decision.status == outcomes.READY
    assert result.decision.selected_action_sha256 == action_b
    assert _scores(result.decision) == {action_a: 2, action_b: 4}

    gate_off = replans.advance_and_replan_after_failure_v1(
        parent,
        failed_outcome,
        plasticity_gate=False,
        consequence_gate=False,
    )
    gate_off.authenticate()
    assert gate_off.status == replans.STOP_GATE_OFF
    assert gate_off.successor is parent
    assert gate_off.retry_activation is None
    assert gate_off.pulsed_activation is None
    assert gate_off.decision is None
    assert parent.checkpoint_sha256 == parent_root
    assert parent.checkpoint_array_sha256s == parent_arrays

    with pytest.raises(replans.FailureConditionedActionReplanError):
        replace(result, decision=initial_decision).authenticate()
    with pytest.raises(replans.FailureConditionedActionReplanError):
        replace(result, sha256=_root("failure-replan-tampered-trace")).authenticate()
