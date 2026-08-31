from __future__ import annotations

import inspect

import experiments.r3_pilot.typed_semantic_packed_absolute_map_fold_v1 as absolute

CASES = (
    ([],),
    ([-2, 1],),
    ([1, 2],),
    ([-3, 3],),
    ([-1, 0, 1],),
)


def test_physical_absolute_map_fold_publication_is_desired_free_and_causal() -> None:
    build_parameters = inspect.signature(
        absolute.PackedPhysicalAbsoluteMapFoldV1.build
    ).parameters
    assert not {
        "desired",
        "probe_outputs",
        "operator",
        "target_action",
        "family",
        "schema",
        "structural_certificate",
    } & set(build_parameters)

    publication = absolute.PackedPhysicalAbsoluteMapFoldV1.build(
        absolute.list_int_to_int_signature(),
        CASES,
    )
    assert publication.status == absolute.READY
    assert publication.state.authorized_slice_count == 1
    assert publication.state.desired_reads_during_build == 0
    assert publication.state.operator_name_inputs_during_build == 0
    assert publication.state.new_cell_count == 0
    assert publication.state.new_persistent_state_bytes == 0
    assert publication.work.concrete_proof_witnesses == 0
    assert publication.work.materialized_witnesses == 0
    assert not hasattr(publication, "select")

    result = publication.materialize()
    assert result.status == absolute.READY
    assert result.exact_proof_count == 1
    assert result.desired_reads == 0
    witness = absolute.require_absolute_map_fold_witness(publication)
    assert witness is result.witness
    assert witness.source == (
        "def aggregate(numbers):\n"
        "    return sum([abs(item_0) for item_0 in numbers])\n"
    )
    assert result.relation_completed_units == 650
    assert result.physical_predictions == 30
    assert result.prediction_cache_hits == 12
    assert result.physical_races == 69

    namespace: dict[str, object] = {}
    exec(witness.source, namespace)
    aggregate = namespace["aggregate"]
    assert callable(aggregate)
    assert aggregate([-4, 1, 2]) == 7
    assert aggregate([0, -3, 2, -1]) == 6
    assert aggregate([3, -2, -1, 0]) == 6

    stopped = absolute.PackedPhysicalAbsoluteMapFoldV1.build(
        absolute.list_int_to_int_signature(),
        CASES,
        absolute_cell_gate=False,
    )
    assert stopped.status == absolute.STOP_PLAN_SLICE
    assert stopped.state.authorized_slice_count == 0
    assert stopped.plan_slice_receipt is None
    stopped_result = stopped.materialize()
    assert stopped_result.status == absolute.STOP_PLAN_SLICE
    assert stopped_result.exact_proof_count == 0
    assert stopped_result.witness is None
    assert stopped.work.concrete_proof_witnesses == 0
    assert stopped.work.materialized_witnesses == 0
