from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as memory


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _activation(
    context: str,
    scores: dict[str, int],
) -> memory.ActionActivationV1:
    return memory.seal_action_activation_v1(
        selection_context_sha256=_root(context),
        active_action_sha256s=tuple(scores),
        base_scores=tuple(scores.values()),
    )


def _outcome(
    checkpoint: memory.FrozenContinualActionOutcomeMemoryV1,
    activation: memory.ActionActivationV1,
    *,
    label: str,
    passed_cell: int,
    exploration_gate: bool = False,
    auxiliary_event_sha256s: tuple[str, ...] = (),
) -> memory.SelectedActionOutcomeV1:
    decision = checkpoint.decide(
        activation,
        exploration_gate=exploration_gate,
    )
    assert decision.status in (memory.READY, memory.EXPLORATION_READY)
    return memory.seal_selected_action_outcome_v1(
        decision=decision,
        proposed_source_sha256=_root(f"{label}:source"),
        execution_sha256=_root(f"{label}:execution"),
        consequence_sha256=_root(f"{label}:consequence"),
        auxiliary_event_sha256s=auxiliary_event_sha256s,
        passed_cell=passed_cell,
    )


def test_explored_pass_promotes_normal_decision_and_gates_are_causal() -> None:
    action = _root("action")
    checkpoint = memory.start_continual_action_outcome_memory_v1((action,))
    activation = _activation("attempt", {action: 2})

    before = checkpoint.decide(activation)
    explored = checkpoint.decide(activation, exploration_gate=True)
    assert before.status == memory.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert before.tonic_score == 2
    assert explored.status == memory.EXPLORATION_READY
    assert explored.tonic_score == 0
    assert explored.selected_action_sha256 == action

    outcome = memory.seal_selected_action_outcome_v1(
        decision=explored,
        proposed_source_sha256=_root("attempt:source"),
        execution_sha256=_root("attempt:execution"),
        consequence_sha256=_root("attempt:consequence"),
        auxiliary_event_sha256s=(),
        passed_cell=1,
    )
    parent_root = checkpoint.checkpoint_sha256
    parent_arrays = checkpoint.checkpoint_array_sha256s
    successor = memory.advance_continual_action_outcome_memory_v1(checkpoint, outcome)

    assert checkpoint.checkpoint_sha256 == parent_root
    assert checkpoint.checkpoint_array_sha256s == parent_arrays
    assert checkpoint.excitatory_ticks == (0,)
    assert successor.excitatory_ticks == (1,)
    assert successor.inhibitory_ticks == (0,)
    promoted = successor.decide(_activation("fresh", {action: 2}))
    assert promoted.status == memory.READY
    assert promoted.effective_scores == (4,)
    assert promoted.selected_action_sha256 == action
    assert (
        successor.decide(_activation("outcome-off", {action: 2}), outcome_gate=False).status
        == memory.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    )
    assert (
        successor.decide(_activation("cell-off", {action: 6}), cell_gate=False).status
        == memory.STOP_CELL_OFF
    )
    assert successor.decide(_activation("empty", {})).status == memory.STOP_NO_CANDIDATE


def test_fail_inhibits_base_support_and_pass_fail_cancel_to_base() -> None:
    action = _root("revisable")
    initial = memory.start_continual_action_outcome_memory_v1((action,))
    failed = memory.advance_continual_action_outcome_memory_v1(
        initial,
        _outcome(
            initial,
            _activation("failure", {action: 6}),
            label="failure",
            passed_cell=0,
        ),
    )
    inhibited = failed.decide(_activation("after-failure", {action: 6}))
    assert failed.excitatory_ticks == (0,)
    assert failed.inhibitory_ticks == (1,)
    assert inhibited.status == memory.READY
    assert inhibited.effective_scores == (4,)

    cancelled = memory.advance_continual_action_outcome_memory_v1(
        failed,
        _outcome(
            failed,
            _activation("recovery", {action: 6}),
            label="recovery",
            passed_cell=1,
        ),
    )
    restored = cancelled.decide(_activation("after-recovery", {action: 6}))
    assert cancelled.excitatory_ticks == (1,)
    assert cancelled.inhibitory_ticks == (1,)
    assert restored.effective_scores == (6,)


def test_two_rival_outcome_geometry_revises_only_with_paired_evidence() -> None:
    winner = _root("W")
    rival = _root("G")
    initial = memory.start_continual_action_outcome_memory_v1((winner, rival))
    pair = {winner: 6, rival: 4}
    assert initial.decide(_activation("pair-initial", pair)).selected_action_sha256 == winner

    winner_failed = memory.advance_continual_action_outcome_memory_v1(
        initial,
        _outcome(
            initial,
            _activation("winner-failure", pair),
            label="winner-failure",
            passed_cell=0,
        ),
    )
    failed_pair = winner_failed.decide(_activation("pair-failed", pair))
    assert failed_pair.effective_scores == tuple(
        4 for _ in failed_pair.activation.active_action_sha256s
    )
    assert failed_pair.status == memory.STOP_PHYSICAL_RACE

    rival_success = memory.advance_continual_action_outcome_memory_v1(
        initial,
        _outcome(
            initial,
            _activation("rival-alone", {rival: 4}),
            label="rival-success-only",
            passed_cell=1,
        ),
    )
    success_pair = rival_success.decide(_activation("pair-success", pair))
    score_by_action = dict(
        zip(
            success_pair.activation.active_action_sha256s,
            success_pair.effective_scores,
            strict=True,
        )
    )
    assert score_by_action == {winner: 6, rival: 6}
    assert success_pair.status == memory.STOP_PHYSICAL_RACE

    paired = memory.advance_continual_action_outcome_memory_v1(
        winner_failed,
        _outcome(
            winner_failed,
            _activation("rival-after-failure", {rival: 4}),
            label="paired-rival-success",
            passed_cell=1,
        ),
    )
    revised = paired.decide(_activation("pair-revised", pair))
    revised_scores = dict(
        zip(
            revised.activation.active_action_sha256s,
            revised.effective_scores,
            strict=True,
        )
    )
    assert revised_scores == {winner: 4, rival: 6}
    assert revised.status == memory.READY
    assert revised.selected_action_sha256 == rival


def test_gate_off_advances_return_exact_parent() -> None:
    action = _root("gate-action")
    checkpoint = memory.start_continual_action_outcome_memory_v1((action,))
    outcome = _outcome(
        checkpoint,
        _activation("gate-attempt", {action: 4}),
        label="gate-attempt",
        passed_cell=1,
    )

    assert (
        memory.advance_continual_action_outcome_memory_v1(
            checkpoint, outcome, plasticity_gate=False
        )
        is checkpoint
    )
    assert (
        memory.advance_continual_action_outcome_memory_v1(
            checkpoint, outcome, consequence_gate=False
        )
        is checkpoint
    )


def test_decision_execution_and_consequence_replay_are_rejected() -> None:
    action = _root("replay-action")
    initial = memory.start_continual_action_outcome_memory_v1((action,))
    first_outcome = _outcome(
        initial,
        _activation("replay-first", {action: 6}),
        label="replay-first",
        passed_cell=1,
    )
    successor = memory.advance_continual_action_outcome_memory_v1(initial, first_outcome)

    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        memory.advance_continual_action_outcome_memory_v1(successor, first_outcome)

    repeated_activation_decision = successor.decide(first_outcome.decision.activation)
    repeated_activation = memory.seal_selected_action_outcome_v1(
        decision=repeated_activation_decision,
        proposed_source_sha256=_root("repeated-activation:source"),
        execution_sha256=_root("repeated-activation:execution"),
        consequence_sha256=_root("repeated-activation:consequence"),
        auxiliary_event_sha256s=(),
        passed_cell=1,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="activation replay"):
        memory.advance_continual_action_outcome_memory_v1(successor, repeated_activation)

    proposed_decision = successor.decide(_activation("replay-proposed", {action: 6}))
    repeated_proposed = memory.seal_selected_action_outcome_v1(
        decision=proposed_decision,
        proposed_source_sha256=first_outcome.proposed_source_sha256,
        execution_sha256=_root("repeated-proposed:execution"),
        consequence_sha256=_root("repeated-proposed:consequence"),
        auxiliary_event_sha256s=(),
        passed_cell=1,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="proposed source replay"):
        memory.advance_continual_action_outcome_memory_v1(successor, repeated_proposed)

    cross_role_decision = successor.decide(_activation("replay-cross-role", {action: 6}))
    cross_role = memory.seal_selected_action_outcome_v1(
        decision=cross_role_decision,
        proposed_source_sha256=first_outcome.execution_sha256,
        execution_sha256=_root("cross-role:execution"),
        consequence_sha256=_root("cross-role:consequence"),
        auxiliary_event_sha256s=(),
        passed_cell=1,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="cross-role"):
        memory.advance_continual_action_outcome_memory_v1(successor, cross_role)

    provenance_alias_decision = successor.decide(_activation("replay-provenance-root", {action: 6}))
    provenance_alias = memory.seal_selected_action_outcome_v1(
        decision=provenance_alias_decision,
        proposed_source_sha256=successor.provenance[0].sha256,
        execution_sha256=_root("provenance-alias:execution"),
        consequence_sha256=_root("provenance-alias:consequence"),
        auxiliary_event_sha256s=(),
        passed_cell=1,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="cross-role"):
        memory.advance_continual_action_outcome_memory_v1(successor, provenance_alias)

    second_decision = successor.decide(_activation("replay-second", {action: 6}))
    same_execution = memory.seal_selected_action_outcome_v1(
        decision=second_decision,
        proposed_source_sha256=_root("replay-second:source"),
        execution_sha256=first_outcome.execution_sha256,
        consequence_sha256=_root("replay-second:consequence"),
        auxiliary_event_sha256s=(),
        passed_cell=1,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="execution replay"):
        memory.advance_continual_action_outcome_memory_v1(successor, same_execution)

    same_consequence = memory.seal_selected_action_outcome_v1(
        decision=second_decision,
        proposed_source_sha256=_root("replay-third:source"),
        execution_sha256=_root("replay-third:execution"),
        consequence_sha256=first_outcome.consequence_sha256,
        auxiliary_event_sha256s=(),
        passed_cell=0,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="consequence replay"):
        memory.advance_continual_action_outcome_memory_v1(successor, same_consequence)


def test_auxiliary_event_roots_are_bounded_retained_and_replay_checked() -> None:
    action = _root("auxiliary-action")
    initial = memory.start_continual_action_outcome_memory_v1((action,))
    auxiliary = tuple(sorted((_root("auxiliary:batch"), _root("auxiliary:manifest"))))
    first = _outcome(
        initial,
        _activation("auxiliary-first", {action: 6}),
        label="auxiliary-first",
        passed_cell=1,
        auxiliary_event_sha256s=auxiliary,
    )
    successor = memory.advance_continual_action_outcome_memory_v1(initial, first)

    assert first.auxiliary_event_sha256s == auxiliary
    assert successor.provenance[0].auxiliary_event_sha256s == auxiliary

    replay = _outcome(
        successor,
        _activation("auxiliary-second", {action: 6}),
        label="auxiliary-second",
        passed_cell=1,
        auxiliary_event_sha256s=(first.proposed_source_sha256,),
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="cross-role"):
        memory.advance_continual_action_outcome_memory_v1(successor, replay)

    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="canonical"):
        _outcome(
            initial,
            _activation("auxiliary-uncanonical", {action: 6}),
            label="auxiliary-uncanonical",
            passed_cell=1,
            auxiliary_event_sha256s=tuple(reversed(auxiliary)),
        )

    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        _outcome(
            initial,
            _activation("auxiliary-over-cap", {action: 6}),
            label="auxiliary-over-cap",
            passed_cell=1,
            auxiliary_event_sha256s=tuple(
                sorted(
                    _root(f"auxiliary-over-cap:{index}")
                    for index in range(memory.MAX_AUXILIARY_EVENT_ROOTS + 1)
                )
            ),
        )


def test_spliced_and_tampered_outcomes_are_rejected() -> None:
    action = _root("splice-action")
    other = _root("other-action")
    target = memory.start_continual_action_outcome_memory_v1((action,))
    foreign = memory.start_continual_action_outcome_memory_v1((other,))
    foreign_outcome = _outcome(
        foreign,
        _activation("foreign", {other: 6}),
        label="foreign",
        passed_cell=1,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        memory.advance_continual_action_outcome_memory_v1(target, foreign_outcome)

    authentic = _outcome(
        target,
        _activation("tamper", {action: 6}),
        label="tamper",
        passed_cell=1,
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        replace(authentic, passed_cell=0).authenticate()
    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        replace(authentic, selected_action_sha256=other).authenticate()
    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        replace(authentic.decision.activation, base_scores=(1,)).authenticate()

    inconsistent = replace(
        authentic.decision,
        excitatory_ticks=(1,),
        sha256="",
    )
    inconsistent = replace(
        inconsistent,
        sha256=memory._digest(
            memory._DECISION_NAMESPACE,
            inconsistent.record(include_sha256=False),
        ),
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        inconsistent.authenticate()

    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        memory.seal_selected_action_outcome_v1(
            decision=authentic.decision,
            proposed_source_sha256=authentic.selected_action_sha256,
            execution_sha256=_root("alias:execution"),
            consequence_sha256=_root("alias:consequence"),
            auxiliary_event_sha256s=(),
            passed_cell=1,
        )

    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        memory.seal_selected_action_outcome_v1(
            decision=authentic.decision,
            proposed_source_sha256=target.checkpoint_sha256,
            execution_sha256=_root("parent-alias:execution"),
            consequence_sha256=_root("parent-alias:consequence"),
            auxiliary_event_sha256s=(),
            passed_cell=1,
        )

    action_context = _activation("action-context", {action: 6})
    action_context = replace(
        action_context,
        selection_context_sha256=action,
        sha256="",
    )
    action_context = replace(
        action_context,
        sha256=memory._digest(
            memory._ACTIVATION_NAMESPACE,
            action_context.record(include_sha256=False),
        ),
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="selection context"):
        target.decide(action_context)


def test_checkpoint_replays_ticks_and_freezes_arrays() -> None:
    action = _root("array-action")
    checkpoint = memory.start_continual_action_outcome_memory_v1((action,))
    advanced = memory.advance_continual_action_outcome_memory_v1(
        checkpoint,
        _outcome(
            checkpoint,
            _activation("array", {action: 6}),
            label="array",
            passed_cell=1,
        ),
    )
    raw = object.__getattribute__(advanced, "_excitatory")
    assert raw.flags.writeable is False
    with pytest.raises(ValueError):
        raw[0] = np.uint8(9)

    object.__setattr__(advanced, "_excitatory", np.asarray((9,), dtype=np.uint8))
    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        advanced.authenticate()


def test_unrelated_action_ticks_remain_stable() -> None:
    learned = _root("learned")
    unrelated = _root("unrelated")
    initial = memory.start_continual_action_outcome_memory_v1((learned, unrelated))
    successor = memory.advance_continual_action_outcome_memory_v1(
        initial,
        _outcome(
            initial,
            _activation("learned-only", {learned: 6}),
            label="learned-only",
            passed_cell=1,
        ),
    )
    by_action = dict(zip(successor.action_sha256s, successor.excitatory_ticks, strict=True))
    assert by_action == {learned: 1, unrelated: 0}


def test_checkpoint_authenticates_the_ordered_prefix_root_chain() -> None:
    action = _root("chain-action")
    initial = memory.start_continual_action_outcome_memory_v1((action,))
    first = memory.advance_continual_action_outcome_memory_v1(
        initial,
        _outcome(
            initial,
            _activation("chain-first", {action: 6}),
            label="chain-first",
            passed_cell=1,
        ),
    )
    second = memory.advance_continual_action_outcome_memory_v1(
        first,
        _outcome(
            first,
            _activation("chain-second", {action: 6}),
            label="chain-second",
            passed_cell=0,
        ),
    )

    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="parent chain"):
        memory._freeze_memory(
            action_sha256s=second.action_sha256s,
            provenance=tuple(reversed(second.provenance)),
            excitatory=np.asarray(second.excitatory_ticks, dtype=np.uint8),
            inhibitory=np.asarray(second.inhibitory_ticks, dtype=np.uint8),
        )

    forged_row = replace(
        second.provenance[1],
        parent_checkpoint_sha256=_root("forged-parent"),
        sha256="",
    )
    forged_row = replace(
        forged_row,
        sha256=memory._digest(
            memory._PROVENANCE_NAMESPACE,
            forged_row.record(include_sha256=False),
        ),
    )
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="parent chain"):
        memory._freeze_memory(
            action_sha256s=second.action_sha256s,
            provenance=(second.provenance[0], forged_row),
            excitatory=np.asarray(second.excitatory_ticks, dtype=np.uint8),
            inhibitory=np.asarray(second.inhibitory_ticks, dtype=np.uint8),
        )


def test_catalogue_is_fixed_canonical_and_bounded() -> None:
    roots = tuple(_root(f"catalogue-{index}") for index in range(2))
    checkpoint = memory.start_continual_action_outcome_memory_v1(tuple(reversed(roots)))
    same_catalogue = memory.start_continual_action_outcome_memory_v1(tuple(sorted(roots)))
    assert checkpoint.action_sha256s == tuple(sorted(roots))
    assert checkpoint.catalogue_sha256 == same_catalogue.catalogue_sha256
    with pytest.raises(memory.ContinualActionOutcomeMemoryError):
        checkpoint.decide(_activation("unknown", {_root("unknown"): 4}))
    with pytest.raises(memory.ContinualActionOutcomeMemoryError, match="capacity"):
        memory.start_continual_action_outcome_memory_v1(
            tuple(_root(f"too-many-{index}") for index in range(memory.MAX_ACTIONS + 1))
        )


def test_physical_race_encodes_catalogue_indices_across_all_output_digits() -> None:
    roots = tuple(sorted(_root(f"wide-{index}") for index in range(6)))
    target = roots[5]
    checkpoint = memory.start_continual_action_outcome_memory_v1(roots)
    decision = checkpoint.decide(_activation("wide-index", {target: 6}))

    assert decision.status == memory.READY
    assert decision.selected_action_sha256 == target
    assert decision.selected_action_index == 5
    assert decision.physical_output_record["token"] == 5


def test_high_logical_scores_remain_physically_ordered_without_clipping() -> None:
    winner = _root("high-winner")
    runner_up = _root("high-runner-up")
    checkpoint = memory.start_continual_action_outcome_memory_v1((winner, runner_up))
    decision = checkpoint.decide(
        _activation(
            "high-scores",
            {
                winner: memory.MAX_OUTPUT_SCORE,
                runner_up: memory.MAX_OUTPUT_SCORE - 1,
            },
        ),
        exploration_gate=True,
    )

    assert memory.MAX_OUTPUT_SCORE == 36
    assert decision.status == memory.EXPLORATION_READY
    assert decision.selected_action_sha256 == winner
