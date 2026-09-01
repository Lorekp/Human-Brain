from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as memory


def _root(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _bank() -> tuple[
    tuple[factors.FactorizedHypothesisV1, ...],
    dict[tuple[int, int], factors.FactorizedHypothesisV1],
    tuple[tuple[str, str], tuple[str, str]],
]:
    components = (
        (_root("action-a0"), _root("action-a1")),
        (_root("action-b0"), _root("action-b1")),
    )
    by_key = {
        (axis_zero, axis_one): factors.seal_factorized_hypothesis_v1(
            (components[0][axis_zero], components[1][axis_one])
        )
        for axis_zero in range(2)
        for axis_one in range(2)
    }
    return (
        tuple(by_key[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1))),
        by_key,
        components,
    )


def _observation(
    label: str,
    *cues: str,
) -> memory.UnorderedCueFactorObservationV1:
    return memory.seal_unordered_cue_factor_observation_v1(
        selection_context_sha256=_root(f"context-{label}"),
        cue_sha256s=tuple(cues),
    )


def _outcome(
    decision: memory.UnorderedCueFactorDecisionV1,
    label: str,
    *,
    passed_cell: int,
) -> memory.UnorderedCueFactorOutcomeV1:
    return memory.seal_unordered_cue_factor_outcome_v1(
        decision=decision,
        proposal_sha256=_root(f"proposal-{label}"),
        execution_sha256=_root(f"execution-{label}"),
        consequence_sha256=_root(f"consequence-{label}"),
        feedback_sha256=_root(f"feedback-{label}"),
        passed_cell=passed_cell,
    )


def _scores(decision: memory.UnorderedCueFactorDecisionV1) -> dict[str, int]:
    return {row.hypothesis.sha256: row.total_score for row in decision.hypothesis_scores}


def test_crossed_grid_learns_axis_bindings_from_unordered_cues() -> None:
    bank, hypotheses, components = _bank()
    parent = memory.start_unordered_cue_factor_memory_v1(bank)
    cues = {
        (0, 0): (_root("cue-a0"), _root("cue-b0")),
        (1, 1): (_root("cue-a1"), _root("cue-b1")),
        (0, 1): (_root("cue-a0"), _root("cue-b1")),
        (1, 0): (_root("cue-a1"), _root("cue-b0")),
    }
    current = parent
    checkpoints = [parent]
    for ordinal, key in enumerate(((0, 0), (1, 1), (0, 1), (1, 0)), start=1):
        observation = _observation(f"train-{ordinal}", *cues[key])
        decision = current.route(
            observation,
            exploration_hypothesis_sha256=hypotheses[key].sha256,
        )
        assert decision.status == memory.EXPLORATION_READY
        assert decision.selected_hypothesis_sha256 == hypotheses[key].sha256
        assert set(decision.binding_observation.axis_cue_sha256s[0]) == {
            memory.cue_axis_binding_sha256_v1(cue, 0) for cue in observation.cue_sha256s
        }
        assert set(decision.binding_observation.axis_cue_sha256s[1]) == {
            memory.cue_axis_binding_sha256_v1(cue, 1) for cue in observation.cue_sha256s
        }
        learned = _outcome(decision, f"train-{ordinal}", passed_cell=1)
        assert (
            memory.advance_unordered_cue_factor_memory_v1(
                current,
                learned,
                plasticity_gate=False,
            )
            is current
        )
        assert (
            memory.advance_unordered_cue_factor_memory_v1(
                current,
                learned,
                consequence_gate=False,
            )
            is current
        )
        current = memory.advance_unordered_cue_factor_memory_v1(current, learned)
        checkpoints.append(current)

        if ordinal == 2:
            diagonal_cross = current.route(
                _observation("diagonal-cross", _root("cue-b1"), _root("cue-a0"))
            )
            assert diagonal_cross.selected_hypothesis_sha256 is None
            assert diagonal_cross.status == memory.STOP_PHYSICAL_RACE

    evaluation = _observation("evaluation", _root("cue-b1"), _root("cue-a0"))
    reordered = _observation("evaluation", _root("cue-a0"), _root("cue-b1"))
    assert evaluation == reordered
    forged = replace(
        evaluation,
        cue_sha256s=tuple(reversed(evaluation.cue_sha256s)),
        sha256="",
    )
    forged = replace(
        forged,
        sha256=memory._digest(
            memory._OBSERVATION_NAMESPACE,
            forged.record(include_sha256=False),
        ),
    )
    with pytest.raises(memory.UnorderedCueFactorMemoryError):
        forged.authenticate()
    decision = current.route(evaluation)
    assert decision.status == memory.READY
    assert decision.selected_hypothesis_sha256 == hypotheses[(0, 1)].sha256
    assert _scores(decision) == {
        hypotheses[(0, 0)].sha256: 8,
        hypotheses[(0, 1)].sha256: 12,
        hypotheses[(1, 0)].sha256: 4,
        hypotheses[(1, 1)].sha256: 8,
    }
    assert current.route(evaluation, outcome_gate=False).selected_hypothesis_sha256 is None
    assert current.route(evaluation, cell_gate=False).status == memory.STOP_CELL_OFF
    assert current.route(evaluation, axis_gates=(True, False)).selected_hypothesis_sha256 is None
    assert current.route(evaluation, axis_gates=(False, True)).selected_hypothesis_sha256 is None

    evidence = {
        (
            row.cue_sha256,
            row.action_axis,
            row.component_sha256,
        ): (row.excitatory_ticks, row.inhibitory_ticks)
        for row in current.binding_evidence
    }
    assert evidence[(_root("cue-a0"), 0, components[0][0])] == (2, 0)
    assert evidence[(_root("cue-a0"), 0, components[0][1])] == (0, 0)
    assert evidence[(_root("cue-a0"), 1, components[1][0])] == (1, 0)
    assert evidence[(_root("cue-a0"), 1, components[1][1])] == (1, 0)
    assert evidence[(_root("cue-b1"), 0, components[0][0])] == (1, 0)
    assert evidence[(_root("cue-b1"), 0, components[0][1])] == (1, 0)
    assert evidence[(_root("cue-b1"), 1, components[1][0])] == (0, 0)
    assert evidence[(_root("cue-b1"), 1, components[1][1])] == (2, 0)
    assert parent.cue_sha256s == ()
    assert parent.provenance == ()
    assert len(current.binding_evidence) == 16
    assert len(current.provenance) == 4
    assert len({value.checkpoint_sha256 for value in checkpoints}) == 5


def test_failed_selected_composite_writes_inhibition_and_replay_is_rejected() -> None:
    bank, hypotheses, components = _bank()
    parent = memory.start_unordered_cue_factor_memory_v1(bank)
    observation = _observation("failed", _root("cue-a0"), _root("cue-b0"))
    decision = parent.route(
        observation,
        exploration_hypothesis_sha256=hypotheses[(1, 1)].sha256,
    )
    failed = _outcome(decision, "failed", passed_cell=0)
    successor = memory.advance_unordered_cue_factor_memory_v1(parent, failed)

    evidence = {
        (row.cue_sha256, row.action_axis, row.component_sha256): row
        for row in successor.binding_evidence
    }
    for cue in observation.cue_sha256s:
        assert evidence[(cue, 0, components[0][1])].inhibitory_ticks == 1
        assert evidence[(cue, 1, components[1][1])].inhibitory_ticks == 1
        assert evidence[(cue, 0, components[0][0])].inhibitory_ticks == 0
        assert evidence[(cue, 1, components[1][0])].inhibitory_ticks == 0
    assert all(row.excitatory_ticks == 0 for row in successor.binding_evidence)
    assert successor.route(observation).selected_hypothesis_sha256 is None
    assert parent.binding_evidence == ()
    assert parent.provenance == ()

    with pytest.raises(memory.UnorderedCueFactorMemoryError):
        memory.advance_unordered_cue_factor_memory_v1(successor, failed)
