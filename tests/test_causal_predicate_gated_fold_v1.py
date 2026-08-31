from __future__ import annotations

import inspect

import numpy as np
import pytest

import experiments.r3_pilot.causal_predicate_gated_fold_v1 as gated


def _lessons() -> tuple[gated.PredicateGateLesson, ...]:
    return (
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


def _load_aggregate(source: str) -> object:
    namespace: dict[str, object] = {}
    exec(source, namespace)
    return namespace["aggregate"]


def test_causal_predicate_gate_transfers_into_one_fresh_fold_witness() -> None:
    lessons = _lessons()
    checkpoint = gated.fit_causal_predicate_gate(
        lessons,
        feature_cell_ids=(911, 307),
        action_cell_ids=(701, 113),
    )
    relabeled = gated.fit_causal_predicate_gate(
        lessons,
        feature_cell_ids=(23, 997),
        action_cell_ids=(17, 991),
    )

    assert checkpoint.persistent_state_bytes == gated.NEW_PERSISTENT_STATE_BYTES == 8
    assert checkpoint.excitatory.dtype == np.uint8
    assert checkpoint.inhibitory.dtype == np.uint8
    assert checkpoint.excitatory.shape == gated.STATE_SHAPE == (2, 2)
    assert not checkpoint.excitatory.flags.writeable
    assert not checkpoint.inhibitory.flags.writeable
    assert checkpoint.feature_trace[0] == checkpoint.feature_trace[2]
    assert checkpoint.feature_trace[0] != checkpoint.feature_trace[1]
    assert checkpoint.action_trace[0] == checkpoint.action_trace[2]
    assert checkpoint.action_trace[0] != checkpoint.action_trace[1]
    assert not np.array_equal(checkpoint.excitatory, relabeled.excitatory)

    build_parameters = inspect.signature(
        gated.PackedPhysicalPredicateGatedFoldV1.build
    ).parameters
    assert not {
        "desired",
        "outputs",
        "operator",
        "action",
        "family",
        "schema",
        "structural_certificate",
    } & set(build_parameters)

    fresh_cases = (
        ([],),
        ([-6, 4, 9],),
        ([9, 4, -6],),
        ([5, 1, -2, 10],),
        ([3, -7],),
    )
    publication = gated.PackedPhysicalPredicateGatedFoldV1.build(
        checkpoint,
        gated.list_int_to_int_signature(),
        fresh_cases,
        5,
    )
    relabeled_publication = gated.PackedPhysicalPredicateGatedFoldV1.build(
        relabeled,
        gated.list_int_to_int_signature(),
        tuple(reversed(fresh_cases)),
        5,
    )
    assert publication.status == gated.READY
    assert relabeled_publication.status == gated.READY
    assert publication.state.authorized_slice_count == 1
    assert publication.state.desired_reads_during_build == 0
    assert publication.state.action_label_inputs_during_build == 0
    assert publication.state.reused_typed_bool_cell_count == 2
    assert publication.state.new_action_cell_count == 2
    assert publication.state.new_persistent_state_bytes == 8
    assert publication.work.concrete_proof_witnesses == 0
    assert publication.work.materialized_witnesses == 0

    result = publication.materialize()
    relabeled_result = relabeled_publication.materialize()
    assert result.status == gated.READY
    assert result.exact_proof_count == 1
    assert result.desired_reads == 0
    witness = gated.require_predicate_gated_fold_witness(publication)
    relabeled_witness = gated.require_predicate_gated_fold_witness(
        relabeled_publication
    )
    assert witness is result.witness
    assert witness.source == relabeled_witness.source
    assert witness.source == (
        "def aggregate(numbers):\n"
        "    return sum([item_0 if item_0 < 5 else 0 for item_0 in numbers])\n"
    )
    aggregate = _load_aggregate(witness.source)
    assert callable(aggregate)
    assert [aggregate(row[0]) for row in fresh_cases] == [0, -2, -2, -1, -4]

    new_threshold_cases = (
        ([-9, -5, 0],),
        ([-5, -9, 0],),
        ([-4, -8, 6],),
        ([],),
    )
    new_threshold_publication = gated.PackedPhysicalPredicateGatedFoldV1.build(
        checkpoint,
        gated.list_int_to_int_signature(),
        new_threshold_cases,
        -4,
    )
    new_threshold_witness = gated.require_predicate_gated_fold_witness(
        new_threshold_publication
    )
    new_threshold_aggregate = _load_aggregate(new_threshold_witness.source)
    assert [new_threshold_aggregate(row[0]) for row in new_threshold_cases] == [
        -14,
        -14,
        -8,
        0,
    ]

    with pytest.raises(gated.PredicateGateFitStop) as consequence_stop:
        gated.fit_causal_predicate_gate(lessons, consequence_gate=False)
    assert consequence_stop.value.status == gated.STOP_CONSEQUENCE

    with pytest.raises(gated.PredicateGatedFoldBuildStop) as feature_stop:
        gated.PackedPhysicalPredicateGatedFoldV1.build(
            checkpoint,
            gated.list_int_to_int_signature(),
            fresh_cases,
            5,
            predicate_cell_gate=False,
        )
    assert feature_stop.value.status == gated.STOP_FEATURE

    with pytest.raises(gated.PredicateGatedFoldBuildStop) as action_stop:
        gated.PackedPhysicalPredicateGatedFoldV1.build(
            checkpoint,
            gated.list_int_to_int_signature(),
            fresh_cases,
            5,
            action_cell_gate=False,
        )
    assert action_stop.value.status == gated.STOP_ACTION_RACE
