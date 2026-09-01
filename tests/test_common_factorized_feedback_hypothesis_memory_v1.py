from __future__ import annotations

from hashlib import sha256

import pytest
from experiments.r3_pilot.common_continual_action_outcome_memory_v1 import (
    EXPLORATION_READY,
    READY,
    STOP_CELL_OFF,
    STOP_INSUFFICIENT_OUTCOME_EVIDENCE,
    TONIC_SCORE,
)
from experiments.r3_pilot.common_factorized_feedback_hypothesis_memory_v1 import (
    FactorizedFeedbackHypothesisMemoryError,
    FactorizedFeedbackOutcomeV1,
    FactorizedHypothesisV1,
    FrozenFactorizedFeedbackHypothesisMemoryV1,
    advance_factorized_feedback_hypothesis_memory_v1,
    seal_factorized_feedback_observation_v1,
    seal_factorized_feedback_outcome_v1,
    seal_factorized_hypothesis_v1,
    start_factorized_feedback_hypothesis_memory_v1,
)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _bank() -> dict[tuple[int, int], FactorizedHypothesisV1]:
    axis_zero = (_root("axis-zero-component-0"), _root("axis-zero-component-1"))
    axis_one = (_root("axis-one-component-0"), _root("axis-one-component-1"))
    return {
        (left, right): seal_factorized_hypothesis_v1((axis_zero[left], axis_one[right]))
        for left in range(2)
        for right in range(2)
    }


def _start() -> tuple[
    dict[tuple[int, int], FactorizedHypothesisV1],
    FrozenFactorizedFeedbackHypothesisMemoryV1,
]:
    bank = _bank()
    memory = start_factorized_feedback_hypothesis_memory_v1(tuple(reversed(tuple(bank.values()))))
    return bank, memory


def _observation(
    memory: FrozenFactorizedFeedbackHypothesisMemoryV1,
    *,
    tag: str,
    axis_zero_cues: tuple[str, ...],
    axis_one_cues: tuple[str, ...],
    reverse_candidates: bool = True,
):
    candidates = memory.hypothesis_sha256s
    return seal_factorized_feedback_observation_v1(
        selection_context_sha256=_root(f"selection:{tag}"),
        axis_cue_sha256s=(axis_zero_cues, axis_one_cues),
        active_hypothesis_sha256s=(
            tuple(reversed(candidates)) if reverse_candidates else candidates
        ),
    )


def _outcome(
    decision,
    *,
    tag: str,
    passed_cell: int,
    execution_sha256: str | None = None,
    feedback_sha256: str | None = None,
) -> FactorizedFeedbackOutcomeV1:
    return seal_factorized_feedback_outcome_v1(
        decision=decision,
        proposal_sha256=_root(f"proposal:{tag}"),
        execution_sha256=execution_sha256 or _root(f"execution:{tag}"),
        consequence_sha256=_root(f"consequence:{tag}"),
        feedback_sha256=feedback_sha256 or _root(f"feedback:{tag}"),
        passed_cell=passed_cell,
    )


def _train_diagonal():
    bank, parent = _start()
    cue_a0 = _root("cue:axis-zero:0")
    cue_a1 = _root("cue:axis-zero:1")
    cue_b0 = _root("cue:axis-one:0")
    cue_b1 = _root("cue:axis-one:1")

    observation_00 = _observation(
        parent,
        tag="diagonal-00",
        axis_zero_cues=(cue_a0,),
        axis_one_cues=(cue_b0,),
    )
    decision_00 = parent.route(
        observation_00,
        exploration_hypothesis_sha256=bank[(0, 0)].sha256,
    )
    first = advance_factorized_feedback_hypothesis_memory_v1(
        parent,
        _outcome(decision_00, tag="diagonal-00", passed_cell=1),
    )

    observation_11 = _observation(
        first,
        tag="diagonal-11",
        axis_zero_cues=(cue_a1,),
        axis_one_cues=(cue_b1,),
    )
    decision_11 = first.route(
        observation_11,
        exploration_hypothesis_sha256=bank[(1, 1)].sha256,
    )
    second = advance_factorized_feedback_hypothesis_memory_v1(
        first,
        _outcome(decision_11, tag="diagonal-11", passed_cell=1),
    )
    return bank, parent, first, second, (cue_a0, cue_a1, cue_b0, cue_b1)


def test_full_cartesian_bank_authenticates_and_prelearning_routes_stop() -> None:
    bank, memory = _start()
    memory.authenticate()
    assert len(memory.hypotheses) == 4
    assert set(memory.hypothesis_sha256s) == {hypothesis.sha256 for hypothesis in bank.values()}
    assert memory.axis_cue_sha256s == ((), ())

    cue_a = (_root("prelearn:a:1"), _root("prelearn:a:0"))
    cue_b = (_root("prelearn:b:1"), _root("prelearn:b:0"))
    observation = _observation(
        memory,
        tag="prelearn",
        axis_zero_cues=cue_a,
        axis_one_cues=cue_b,
    )
    canonical = _observation(
        memory,
        tag="prelearn",
        axis_zero_cues=tuple(reversed(cue_a)),
        axis_one_cues=tuple(reversed(cue_b)),
        reverse_candidates=False,
    )
    assert observation == canonical

    decision = memory.route(observation)
    assert decision.status == STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert decision.selected_hypothesis_sha256 is None
    assert len(decision.action_decision.activation.active_action_sha256s) == 4
    assert memory.route(observation, outcome_gate=False).status == (
        STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    )
    assert memory.route(observation, cell_gate=False).status == STOP_CELL_OFF
    assert memory.route(observation, axis_gates=(False, True)).selected_hypothesis_sha256 is None

    incomplete = seal_factorized_feedback_observation_v1(
        selection_context_sha256=_root("selection:incomplete"),
        axis_cue_sha256s=((_root("incomplete:a"),), (_root("incomplete:b"),)),
        active_hypothesis_sha256s=memory.hypothesis_sha256s[:-1],
    )
    with pytest.raises(FactorizedFeedbackHypothesisMemoryError):
        memory.route(incomplete)


def test_diagonal_feedback_recombines_into_unseen_cross_hypothesis() -> None:
    bank, parent, first, second, cues = _train_diagonal()
    cue_a0, _, cue_b0, cue_b1 = cues

    assert parent.axis_cue_sha256s == ((), ())
    assert len(first.provenance) == 1
    assert len(second.provenance) == 2
    for outcome in second.provenance:
        assert outcome.decision.status == EXPLORATION_READY
        assert len(outcome.decision.action_decision.activation.active_action_sha256s) == 4
        assert outcome.decision.action_decision.activation.base_scores.count(TONIC_SCORE) == 1

    cross = _observation(
        second,
        tag="unseen-cross-01",
        axis_zero_cues=(cue_a0,),
        axis_one_cues=(cue_b1,),
    )
    decision = second.route(cross)
    score_by_root = {
        score.hypothesis.sha256: score.total_score for score in decision.hypothesis_scores
    }
    assert decision.status == READY
    assert decision.selected_hypothesis_sha256 == bank[(0, 1)].sha256
    assert score_by_root == {
        bank[(0, 0)].sha256: 2,
        bank[(0, 1)].sha256: 4,
        bank[(1, 0)].sha256: 0,
        bank[(1, 1)].sha256: 2,
    }

    reverse_input = _observation(
        second,
        tag="unseen-cross-01",
        axis_zero_cues=(cue_a0,),
        axis_one_cues=(cue_b1,),
        reverse_candidates=False,
    )
    assert reverse_input == cross
    assert second.route(reverse_input) == decision
    assert second.route(cross, axis_gates=(True, False)).selected_hypothesis_sha256 is None
    assert second.route(cross, axis_gates=(False, True)).selected_hypothesis_sha256 is None

    negative_a = _root("cue:axis-zero:negative")
    negative_b = _root("cue:axis-one:negative")
    negative_observation = _observation(
        second,
        tag="per-cue-negative",
        axis_zero_cues=(negative_a,),
        axis_one_cues=(negative_b,),
    )
    negative_decision = second.route(
        negative_observation,
        exploration_hypothesis_sha256=bank[(0, 0)].sha256,
    )
    mixed = advance_factorized_feedback_hypothesis_memory_v1(
        second,
        _outcome(negative_decision, tag="per-cue-negative", passed_cell=0),
    )
    mixed_probe = _observation(
        mixed,
        tag="per-cue-rectification",
        axis_zero_cues=(cue_a0, negative_a),
        axis_one_cues=(cue_b0, negative_b),
    )
    mixed_decision = mixed.route(mixed_probe)
    assert mixed_decision.selected_hypothesis_sha256 == bank[(0, 0)].sha256
    assert (
        next(
            score.total_score
            for score in mixed_decision.hypothesis_scores
            if score.hypothesis.sha256 == bank[(0, 0)].sha256
        )
        == 4
    )


def test_selected_failure_inhibits_both_factors_and_replay_is_rejected() -> None:
    bank, _, _, parent, cues = _train_diagonal()
    cue_a0, _, _, cue_b1 = cues
    cross = _observation(
        parent,
        tag="cross-failure",
        axis_zero_cues=(cue_a0,),
        axis_one_cues=(cue_b1,),
    )
    failure_decision = parent.route(cross)
    assert failure_decision.selected_hypothesis_sha256 == bank[(0, 1)].sha256
    failure = _outcome(failure_decision, tag="cross-failure", passed_cell=0)

    assert (
        advance_factorized_feedback_hypothesis_memory_v1(
            parent,
            failure,
            plasticity_gate=False,
        )
        is parent
    )
    assert (
        advance_factorized_feedback_hypothesis_memory_v1(
            parent,
            failure,
            consequence_gate=False,
        )
        is parent
    )
    parent_root = parent.checkpoint_sha256
    successor = advance_factorized_feedback_hypothesis_memory_v1(parent, failure)
    assert parent.checkpoint_sha256 == parent_root

    fresh_cross = _observation(
        successor,
        tag="cross-after-failure",
        axis_zero_cues=(cue_a0,),
        axis_one_cues=(cue_b1,),
    )
    stopped = successor.route(fresh_cross)
    assert stopped.status == STOP_INSUFFICIENT_OUTCOME_EVIDENCE
    assert stopped.selected_hypothesis_sha256 is None
    assert all(score.total_score == 0 for score in stopped.hypothesis_scores)

    with pytest.raises(FactorizedFeedbackHypothesisMemoryError):
        advance_factorized_feedback_hypothesis_memory_v1(successor, failure)

    execution_replay_observation = _observation(
        successor,
        tag="execution-replay",
        axis_zero_cues=(_root("execution-replay:a"),),
        axis_one_cues=(_root("execution-replay:b"),),
    )
    execution_replay_decision = successor.route(
        execution_replay_observation,
        exploration_hypothesis_sha256=bank[(0, 0)].sha256,
    )
    execution_replay = _outcome(
        execution_replay_decision,
        tag="execution-replay",
        passed_cell=1,
        execution_sha256=failure.execution_sha256,
    )
    with pytest.raises(FactorizedFeedbackHypothesisMemoryError):
        advance_factorized_feedback_hypothesis_memory_v1(successor, execution_replay)

    feedback_replay_observation = _observation(
        successor,
        tag="feedback-replay",
        axis_zero_cues=(_root("feedback-replay:a"),),
        axis_one_cues=(_root("feedback-replay:b"),),
    )
    feedback_replay_decision = successor.route(
        feedback_replay_observation,
        exploration_hypothesis_sha256=bank[(1, 1)].sha256,
    )
    feedback_replay = _outcome(
        feedback_replay_decision,
        tag="feedback-replay",
        passed_cell=1,
        feedback_sha256=failure.feedback_sha256,
    )
    with pytest.raises(FactorizedFeedbackHypothesisMemoryError):
        advance_factorized_feedback_hypothesis_memory_v1(successor, feedback_replay)

    cue_as_execution_observation = _observation(
        successor,
        tag="cue-as-execution",
        axis_zero_cues=(_root("cue-as-execution:a"),),
        axis_one_cues=(_root("cue-as-execution:b"),),
    )
    cue_as_execution_decision = successor.route(
        cue_as_execution_observation,
        exploration_hypothesis_sha256=bank[(0, 0)].sha256,
    )
    cue_as_execution = _outcome(
        cue_as_execution_decision,
        tag="cue-as-execution",
        passed_cell=1,
        execution_sha256=cue_a0,
    )
    with pytest.raises(FactorizedFeedbackHypothesisMemoryError):
        advance_factorized_feedback_hypothesis_memory_v1(successor, cue_as_execution)

    execution_as_cue_observation = _observation(
        successor,
        tag="execution-as-cue",
        axis_zero_cues=(failure.execution_sha256,),
        axis_one_cues=(_root("execution-as-cue:b"),),
    )
    execution_as_cue_decision = successor.route(
        execution_as_cue_observation,
        exploration_hypothesis_sha256=bank[(1, 1)].sha256,
    )
    execution_as_cue = _outcome(
        execution_as_cue_decision,
        tag="execution-as-cue",
        passed_cell=1,
    )
    with pytest.raises(FactorizedFeedbackHypothesisMemoryError):
        advance_factorized_feedback_hypothesis_memory_v1(successor, execution_as_cue)
