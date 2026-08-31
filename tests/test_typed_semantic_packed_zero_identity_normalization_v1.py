from __future__ import annotations

import experiments.r3_pilot.causal_predicate_gated_fold_v1 as gated
import experiments.r3_pilot.typed_semantic_packed_frontier_v4 as scalar


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


def _load_aggregate(source: str) -> object:
    namespace: dict[str, object] = {}
    exec(source, namespace)
    return namespace["aggregate"]


def test_zero_identity_reuses_child_and_keeps_predicate_fold_topology() -> None:
    arguments = (-11, 4, 13)
    frontier = scalar.PackedScalarFrontier.build(
        case_arguments=arguments,
        literals=(0,),
        max_ast_cost=3,
    )
    assert frontier.status == scalar.FRONTIER_READY
    frozen = frontier.state.frozen_workset
    assert frozen is not None
    identity_partitions = tuple(
        partition
        for partition in frozen.partitions
        if partition.normalization_id == "add-right-zero-v2"
    )
    assert len(identity_partitions) == 1
    identity = identity_partitions[0]
    normalization = identity.normalization
    assert identity.production == "add"
    assert identity.root_id is None
    assert identity.groups == ()
    assert normalization is not None
    target = next(
        partition
        for partition in frozen.partitions
        if partition.family_id == normalization.target_family_id
    )
    assert normalization.recovered_root_id == normalization.target_root_id
    assert normalization.target_root_id == target.root_id
    assert normalization.source_root_id != normalization.target_root_id
    assert all(enode.family_id != identity.family_id for enode in frontier.enodes)

    selection = frontier.select(arguments)
    assert selection.status == scalar.SELECTED
    assert selection.exact_proof_count == 1
    assert tuple(witness.source for witness in selection.witnesses) == ("x",)

    checkpoint = _checkpoint()
    signature = gated.list_int_to_int_signature()
    zero_cases = (
        ([-3, 2, 0],),
        ([4, -1],),
        ([],),
    )
    zero_publication = gated.PackedPhysicalPredicateGatedFoldV1.build(
        checkpoint,
        signature,
        zero_cases,
        0,
    )
    assert zero_publication.status == gated.READY
    assert zero_publication.state.authorized_slice_count == 1
    assert zero_publication.work.concrete_proof_witnesses == 0
    assert zero_publication.work.materialized_witnesses == 0
    zero_result = zero_publication.materialize()
    assert zero_result.status == gated.READY
    assert zero_result.exact_proof_count == 1
    assert zero_result.witness is not None
    assert zero_result.witness.source == (
        "def aggregate(numbers):\n"
        "    return sum([item_0 if item_0 < 0 else 0 for item_0 in numbers])\n"
    )
    zero_aggregate = _load_aggregate(zero_result.witness.source)
    assert [zero_aggregate(row[0]) for row in zero_cases] == [-3, -1, 0]

    nonzero_cases = (
        ([-6, 4, 9],),
        ([5, 1, -2, 10],),
        ([],),
    )
    nonzero_publication = gated.PackedPhysicalPredicateGatedFoldV1.build(
        checkpoint,
        signature,
        nonzero_cases,
        5,
    )
    assert nonzero_publication.status == gated.READY
    nonzero_result = nonzero_publication.materialize()
    assert nonzero_result.status == gated.READY
    assert nonzero_result.exact_proof_count == 1
    assert nonzero_result.witness is not None
    assert nonzero_result.witness.source == (
        "def aggregate(numbers):\n"
        "    return sum([item_0 if item_0 < 5 else 0 for item_0 in numbers])\n"
    )
