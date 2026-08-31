from __future__ import annotations

import inspect
from collections.abc import Sequence

from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
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


def test_solved_plan_fragments_compose_one_fresh_program_after_cell_selection() -> None:
    signature = absolute.list_int_to_int_signature()
    training_cases = (([64],), ([-2],), ([4],), ([5],))
    publication = bank.CausalTaskObservationCandidatePlanBankV1.build(
        _checkpoint(),
        signature,
        training_cases,
    )

    library = fragments.fit_reusable_program_fragment_library_v1(
        publication,
        (
            _encoded((64, 2, 4, 5)),
            _encoded((0, -2, 4, 0)),
        ),
    )
    state = library.state
    assert state.status == fragments.READY
    assert state.training_program_count == 2
    assert len(set(state.training_program_sha256s)) == 2
    assert state.fragment_cell_count == state.work.unique_fragment_cells
    assert state.fragment_occurrence_count == state.work.fragment_occurrences
    assert state.fragment_occurrence_count > state.fragment_cell_count
    assert state.recurring_fragment_cell_count > 0
    assert state.work.consolidated_occurrences > 0
    assert state.work.solved_plan_races == 2
    assert state.work.fragment_admission_races == state.fragment_occurrence_count - 1
    assert state.work.persistent_cell_state_bytes > 0
    assert state.work.host_fragment_payload_bytes > 0
    assert publication.materialized_witness_count == 2

    build_signature = inspect.signature(library.build)
    assert "desired" not in build_signature.parameters
    assert (
        inspect.getsource(fragments.ReusableProgramFragmentFrontierV1.select).count(
            "semantic_race.select_exact_semantic_vector("
        )
        == 1
    )

    fresh_cases = (
        ([-8, 7, -1],),
        ([6, 5, 4, -3],),
        ([-2, 3, 9],),
        ([12, -11, 0, 5],),
    )
    frontier = library.build(signature, fresh_cases)
    assert frontier.status == fragments.READY
    assert frontier.state.case_count == len(fresh_cases)
    assert frontier.state.recipe_count > 1
    assert frontier.state.desired_reads_during_build == 0
    assert frontier.state.source_materializations_during_build == 0
    assert frontier.state.work.prototype_recall_races == state.fragment_cell_count
    assert frontier.state.work.type_compatible_relations > 0
    assert frontier.state.work.structurally_valid_relations > 0
    assert frontier.state.work.unique_recipe_count == frontier.state.recipe_count
    assert frontier.source_materialization_count == 0
    assert all(
        recipe.program_sha256 not in state.training_program_sha256s
        for recipe in frontier._recipes
    )

    unknown = frontier.select(_encoded((61, 62, 63, 64)))
    assert unknown.status == fragments.STOP_NO_EXACT
    assert unknown.winner_key is None
    assert unknown.exact_proof_count == 0
    assert unknown.witness is None
    assert unknown.materialized_witnesses == 0
    assert frontier.source_materialization_count == 0

    equivalent_recipes = frontier.select(_encoded((9, 7, 5, 11)))
    assert equivalent_recipes.status == fragments.STOP_MULTIPLE_EXACT
    assert equivalent_recipes.winner_key is None
    assert len(equivalent_recipes.exact_winner_keys) > 1
    assert equivalent_recipes.exact_proof_count == 0
    assert equivalent_recipes.witness is None
    assert equivalent_recipes.materialized_witnesses == 0
    assert frontier.source_materialization_count == 0

    selected = frontier.select(_encoded((-1, 1, 1, 0)))
    assert selected.status == fragments.SELECTED
    assert selected.winner_key is not None
    assert selected.exact_winner_keys == (selected.winner_key,)
    assert selected.exact_proof_count == 1
    assert selected.materialized_witnesses == 1
    assert selected.witness is not None
    assert selected.witness.program.sha256 not in state.training_program_sha256s
    assert "abs(item_0) < 5" in selected.witness.source
    assert _execute(selected.witness.source, fresh_cases) == (-1, 1, 1, 0)
    assert frontier.source_materialization_count == 1

    ambiguous_frontier = library.build(signature, (([],),))
    ambiguous = ambiguous_frontier.select(_encoded((0,)))
    assert ambiguous.status == fragments.STOP_MULTIPLE_EXACT
    assert ambiguous.winner_key is None
    assert len(ambiguous.exact_winner_keys) > 1
    assert ambiguous.exact_proof_count == 0
    assert ambiguous.witness is None
    assert ambiguous.materialized_witnesses == 0
    assert ambiguous_frontier.source_materialization_count == 0

    cell_off = library.build(
        signature,
        fresh_cases,
        prototype_cell_gate=False,
    )
    assert cell_off.status == fragments.STOP_LIBRARY_DISABLED
    assert cell_off.state.recipe_count == 0
    stopped = cell_off.select(_encoded((-1, 1, 1, 0)))
    assert stopped.status == fragments.STOP_LIBRARY_DISABLED
    assert stopped.exact_proof_count == 0
    assert stopped.witness is None
    assert stopped.race is None
