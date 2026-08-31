from __future__ import annotations

import inspect
from collections.abc import Sequence

import pytest
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as bank
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
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


def test_candidate_bank_builds_before_one_exact_race_and_materializes_selected_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint()
    calls: list[int] = []
    original_race = semantic_race.select_exact_semantic_vector

    def counted_race(
        candidate_keys: Sequence[str],
        candidate_outputs: Sequence[Sequence[output_cells.TypedOutputEncoding]],
        desired_outputs: Sequence[output_cells.TypedOutputEncoding],
    ) -> semantic_race.TypedSemanticVectorRaceAudit:
        calls.append(len(candidate_keys))
        return original_race(candidate_keys, candidate_outputs, desired_outputs)

    monkeypatch.setattr(
        bank.semantic_race,
        "select_exact_semantic_vector",
        counted_race,
    )

    build_signature = inspect.signature(
        bank.CausalTaskObservationCandidatePlanBankV1.build
    )
    assert "desired" not in build_signature.parameters
    assert (
        inspect.getsource(bank.CausalTaskObservationCandidatePlanBankV1.select).count(
            "semantic_race.select_exact_semantic_vector("
        )
        == 1
    )

    cases = (([64],), ([-2],), ([4],), ([5],))
    publication = bank.CausalTaskObservationCandidatePlanBankV1.build(
        checkpoint,
        absolute.list_int_to_int_signature(),
        cases,
    )
    assert calls == []
    assert publication.state.status == bank.READY
    assert publication.state.candidate_count == 131 == bank.CANDIDATE_COUNT
    assert publication.state.case_count == len(cases)
    assert publication.state.candidate_keys == (
        bank.RAW_SUM_KEY,
        bank.ABSOLUTE_MAP_SUM_KEY,
        *(bank.predicate_candidate_key(value) for value in range(-64, 65)),
    )
    assert len(publication.state.plan_slice_receipt_sha256s) == 131
    assert publication.state.desired_reads_during_build == 0
    assert publication.state.proof_materializations_during_build == 0
    assert publication.state.semantic_races_during_build == 0
    assert publication.materialized_witness_count == 0
    assert all(
        candidate.publication.work.concrete_proof_witnesses == 0
        and candidate.publication.work.materialized_witnesses == 0
        for candidate in publication._candidates
    )
    assert all(
        candidate.kind == bank._RAW
        or candidate.publication._materialization is None
        for candidate in publication._candidates
    )

    ambiguous = publication.select(_encoded((0, -2, 0, 0)))
    assert calls == [131]
    assert ambiguous.status == bank.STOP_MULTIPLE_EXACT
    assert ambiguous.winner_key is None
    assert ambiguous.exact_proof_count == 0
    assert ambiguous.witness is None
    assert ambiguous.materialized_witnesses == 0
    assert bank.predicate_candidate_key(0) in ambiguous.exact_winner_keys
    assert bank.predicate_candidate_key(1) in ambiguous.exact_winner_keys
    assert publication.materialized_witness_count == 0

    no_match = publication.select(_encoded((7, 8, 9, 10)))
    assert calls == [131, 131]
    assert no_match.status == bank.STOP_NO_EXACT
    assert no_match.winner_key is None
    assert no_match.exact_winner_keys == ()
    assert no_match.exact_proof_count == 0
    assert no_match.witness is None
    assert no_match.materialized_witnesses == 0
    assert publication.materialized_witness_count == 0

    absolute_selected = publication.select(_encoded((64, 2, 4, 5)))
    assert calls == [131, 131, 131]
    assert absolute_selected.status == bank.SELECTED
    assert absolute_selected.winner_key == bank.ABSOLUTE_MAP_SUM_KEY
    assert absolute_selected.exact_winner_keys == (bank.ABSOLUTE_MAP_SUM_KEY,)
    assert absolute_selected.exact_proof_count == 1
    assert absolute_selected.materialized_witnesses == 1
    assert absolute_selected.witness is not None
    assert "abs(item_0)" in absolute_selected.witness.source
    assert publication.materialized_witness_count == 1

    predicate_selected = publication.select(_encoded((0, -2, 4, 0)))
    assert calls == [131, 131, 131, 131]
    assert predicate_selected.status == bank.SELECTED
    assert predicate_selected.winner_key == bank.predicate_candidate_key(5)
    assert predicate_selected.exact_winner_keys == (
        bank.predicate_candidate_key(5),
    )
    assert predicate_selected.exact_proof_count == 1
    assert predicate_selected.materialized_witnesses == 1
    assert predicate_selected.witness is not None
    assert "item_0 < 5" in predicate_selected.witness.source
    assert publication.materialized_witness_count == 2

    raw_selected = publication.select(_encoded((64, -2, 4, 5)))
    assert calls == [131, 131, 131, 131, 131]
    assert raw_selected.status == bank.SELECTED
    assert raw_selected.winner_key == bank.RAW_SUM_KEY
    assert raw_selected.exact_winner_keys == (bank.RAW_SUM_KEY,)
    assert raw_selected.exact_proof_count == 1
    assert raw_selected.materialized_witnesses == 1
    assert raw_selected.witness is not None
    assert "return sum(numbers)" in raw_selected.witness.source
    assert publication.materialized_witness_count == 3
