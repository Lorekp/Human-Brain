from __future__ import annotations

import inspect
from collections.abc import Sequence

from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as bank
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute


def _checkpoint() -> gated.FrozenPredicateGateV1:
    lessons = (
        gated.observe_predicate_gate_lesson(
            accumulator_before=11,
            item=-10,
            threshold=3,
            observed_after=1,
        ),
        gated.observe_predicate_gate_lesson(
            accumulator_before=-5,
            item=7,
            threshold=-1,
            observed_after=-5,
        ),
        gated.observe_predicate_gate_lesson(
            accumulator_before=6,
            item=2,
            threshold=8,
            observed_after=8,
        ),
    )
    return gated.fit_causal_predicate_gate(
        lessons,
        feature_cell_ids=(911, 307),
        action_cell_ids=(701, 113),
    )


def _encoded(values: Sequence[int]) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in values
    )


def _execute(source: str, cases: tuple[tuple[list[int]], ...]) -> tuple[int, ...]:
    namespace: dict[str, object] = {}
    exec(source, {}, namespace)
    aggregate = namespace["aggregate"]
    return tuple(aggregate(list(values)) for (values,) in cases)  # type: ignore[operator]


def test_recurrent_fragment_relations_compose_two_fresh_selected_edits() -> None:
    signature = absolute.list_int_to_int_signature()
    publication = bank.CausalTaskObservationCandidatePlanBankV1.build(
        _checkpoint(),
        signature,
        (([64],), ([-2],), ([4],), ([5],)),
    )
    library = fragments.fit_reusable_program_fragment_library_v1(
        publication,
        (
            _encoded((64, 2, 4, 5)),
            _encoded((0, -2, 4, 0)),
        ),
    )

    relation_cases = (
        ([-8, 7, -1],),
        ([6, 5, 4, -3],),
        ([-2, 3, 9],),
        ([12, -11, 0, 5],),
        ([-9, 2, 6],),
        ([5, -4, 3],),
        ([8, 7],),
        ([-1, 0, 12],),
    )
    lesson_frontier = library.build(signature, relation_cases)
    abs_at_predicate = lesson_frontier.select(
        _encoded((-1, 1, 1, 0, 2, -1, 0, -1))
    )
    abs_as_selected_action = lesson_frontier.select(
        _encoded((9, 7, 5, 11, 11, 7, 0, 1))
    )
    assert abs_at_predicate.status == fragments.SELECTED
    assert abs_as_selected_action.status == fragments.STOP_MULTIPLE_EXACT
    assert abs_at_predicate.witness is not None
    assert abs_as_selected_action.witness is None
    assert len(abs_as_selected_action.exact_winner_keys) > 1

    lessons = (
        recurrent.FragmentRelationLesson(lesson_frontier, abs_at_predicate),
        recurrent.FragmentRelationLesson(lesson_frontier, abs_as_selected_action),
    )
    policy = recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        lessons,
    )
    state = policy.state
    assert state.status == recurrent.READY
    assert state.relation_count == 2
    assert len(set(state.relation_receipt_sha256s)) == 2
    assert len(set(item.precondition_sha256 for item in policy.relation_receipts)) == 2
    assert all("path" not in field for field in recurrent.FragmentRelationReceipt._fields)
    assert state.work.solved_relation_races == 2
    assert state.work.structural_explanation_matches == 2
    assert state.work.structural_explanation_attempts > 2
    assert state.work.intervention_races == 2
    assert state.work.intervention_candidate_programs > 2
    assert state.work.intervention_probe_executions > 0
    assert state.work.unselected_task_output_executions == 0
    assert state.work.learned_relation_cells == 2
    assert state.work.feature_cells == 5
    assert state.work.action_cells == 5
    assert state.work.relation_excitatory_ticks == 128
    assert state.work.relation_inhibitory_ticks == 1920
    assert state.work.policy_excitatory_ticks == 5
    assert state.work.policy_inhibitory_ticks == 20
    assert state.work.persistent_cell_state_bytes == 4_150

    build_signature = inspect.signature(policy.build)
    assert "desired" not in build_signature.parameters
    assert (
        inspect.getsource(recurrent.RecurrentFragmentRelationFrontierV2.select).count(
            "semantic_race.select_exact_semantic_vector("
        )
        == 1
    )

    fresh_cases = (
        ([-10, -4, 3, 8],),
        ([0, 5, -5, 2],),
        ([9, -1, -2, 4],),
        ([6, -7, 11],),
        ([-3, 3, 12, -4],),
    )
    frontier = policy.build(signature, fresh_cases)
    assert frontier.status == recurrent.READY
    assert frontier.state.case_count == len(fresh_cases)
    assert frontier.state.terminal_program_count == 2
    assert frontier.state.desired_reads_during_build == 0
    assert frontier.state.source_materializations_during_build == 0
    assert len(frontier.state.applied_relation_receipt_sha256s) == 2
    assert len(set(frontier.state.applied_relation_receipt_sha256s)) == 2
    assert frontier.state.work.root_rollouts <= recurrent.MAX_ROOTS
    assert frontier.state.work.affordance_races == 2
    assert (
        frontier.state.work.affordance_races
        <= recurrent.MAX_AFFORDANCE_RACES_PER_ROOT
    )
    assert frontier.state.work.policy_races == 5
    assert frontier.state.work.policy_races <= recurrent.MAX_POLICY_RACES_PER_ROOT
    assert frontier.state.work.effect_races == 2
    assert frontier.state.work.effect_races <= recurrent.MAX_EFFECT_RACES_PER_ROOT
    assert frontier.state.work.selected_actuations == 2
    assert frontier.state.work.committed_edits == 2
    assert frontier.state.work.committed_edits <= recurrent.MAX_COMMITTED_EDITS
    assert frontier.state.work.revisions == 0
    assert frontier.state.work.unselected_splice_executions == 0
    assert frontier.state.work.terminal_case_executions == 3 * len(fresh_cases)
    assert frontier.source_materialization_count == 0

    no_match = frontier.select(_encoded((1, 2, 3, 4, 5)))
    assert no_match.status == recurrent.STOP_NO_EXACT
    assert no_match.exact_proof_count == 0
    assert no_match.witness is None
    assert frontier.source_materialization_count == 0

    ambiguous_frontier = policy.build(signature, (([],),))
    ambiguous = ambiguous_frontier.select(_encoded((0,)))
    assert ambiguous.status == recurrent.STOP_MULTIPLE_EXACT
    assert len(ambiguous.exact_winner_keys) == 2
    assert ambiguous.exact_proof_count == 0
    assert ambiguous.witness is None
    assert ambiguous_frontier.source_materialization_count == 0

    selected = frontier.select(_encoded((7, 2, 7, 0, 10)))
    assert selected.status == recurrent.SELECTED
    assert selected.exact_winner_keys == (selected.winner_key,)
    assert selected.exact_proof_count == 1
    assert selected.materialized_witnesses == 1
    assert selected.witness is not None
    assert len(selected.witness.applied_relation_receipt_sha256s) == 2
    assert "abs(item_0) < 5" in selected.witness.source
    assert selected.witness.source.count("abs(item_0)") == 2
    assert _execute(selected.witness.source, fresh_cases) == (7, 2, 7, 0, 10)
    assert frontier.source_materialization_count == 1

    relation_off = policy.build(signature, fresh_cases, relation_cell_gate=False)
    assert relation_off.status == recurrent.STOP_RELATION
    assert relation_off.state.work.selected_actuations == 0
    policy_off = policy.build(signature, fresh_cases, policy_cell_gate=False)
    assert policy_off.status == recurrent.STOP_POLICY
    assert policy_off.state.work.selected_actuations == 0
    replay_off = policy.build(signature, fresh_cases, replay_cell_gate=False)
    assert replay_off.status == recurrent.STOP_REPLAY
    assert replay_off.state.work.selected_actuations == 0
    actuation_off = policy.build(signature, fresh_cases, actuation_gate=False)
    assert actuation_off.status == recurrent.STOP_ACTUATION
    assert actuation_off.state.work.affordance_races == 1
    assert actuation_off.state.work.selected_actuations == 0
    effect_off = policy.build(signature, fresh_cases, effect_cell_gate=False)
    assert effect_off.status == recurrent.STOP_EFFECT
    assert effect_off.state.work.selected_actuations == 1
    assert effect_off.state.work.committed_edits == 0
    assert effect_off.state.work.revisions == 1
    assert effect_off.state.work.effect_races == 1
    stopped = effect_off.select(_encoded((7, 2, 7, 0, 10)))
    assert stopped.status == recurrent.STOP_EFFECT
    assert stopped.race is None
    assert stopped.exact_proof_count == 0
    assert stopped.witness is None

    reversed_policy = recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        tuple(reversed(lessons)),
    )
    assert reversed_policy.state.checkpoint_sha256 != policy.state.checkpoint_sha256
    reversed_frontier = reversed_policy.build(signature, fresh_cases)
    assert reversed_frontier.status == recurrent.READY
    assert reversed_frontier.state.work.selected_actuations == 2
    reversed_selected = reversed_frontier.select(_encoded((7, 2, 7, 0, 10)))
    assert reversed_selected.status == recurrent.SELECTED
    assert reversed_selected.witness is not None
    assert reversed_selected.witness.program.sha256 == selected.witness.program.sha256
    assert _execute(reversed_selected.witness.source, fresh_cases) == (7, 2, 7, 0, 10)
