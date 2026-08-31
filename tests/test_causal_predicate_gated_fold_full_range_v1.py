from __future__ import annotations

import inspect

import experiments.r3_pilot.causal_predicate_gated_fold_v1 as gated
import pytest


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


def test_predicate_gated_fold_publishes_only_its_full_range_topology() -> None:
    checkpoint = _checkpoint()
    cases = (
        ([64],),
        ([-64],),
        ([64, -64],),
        ([-5, 5],),
        ([4],),
        ([5],),
        ([],),
    )
    signature = gated.list_int_to_int_signature()

    build_source = inspect.getsource(gated._PredicateGateListIntBuilder.build)
    assert "threshold" not in build_source
    assert "_predicate_count" not in build_source
    assert "less-than-literal-item" not in build_source
    assert "less-than-added" not in build_source
    assert '"maximum"' not in build_source
    assert '"minimum"' not in build_source

    publications: dict[int, gated.PackedPhysicalPredicateGatedFoldV1] = {}
    for threshold in (-64, 0, 64, 5):
        publication = gated.PackedPhysicalPredicateGatedFoldV1.build(
            checkpoint,
            signature,
            cases,
            threshold,
        )
        publications[threshold] = publication
        assert publication.status == gated.READY
        assert publication.state.authorized_slice_count == 1
        assert publication.state.desired_reads_during_build == 0
        assert publication.state.action_label_inputs_during_build == 0
        assert publication.plan_slice_receipt is not None
        publication._authenticate()
        assert publication.work.concrete_proof_witnesses == 0
        assert publication.work.materialized_witnesses == 0
        assert publication._materialization is None

    for threshold, publication in publications.items():
        result = publication.materialize()
        assert result.status == gated.READY
        assert result.exact_proof_count == 1
        assert result.witness is not None
        assert result.witness.source == (
            "def aggregate(numbers):\n"
            f"    return sum([item_0 if item_0 < {threshold} else 0 "
            "for item_0 in numbers])\n"
        )
        aggregate = _load_aggregate(result.witness.source)
        assert [aggregate(row[0]) for row in cases] == [
            sum(item for item in row[0] if item < threshold) for row in cases
        ]

    with pytest.raises(gated.PredicateGatedFoldBuildStop) as feature_stop:
        gated.PackedPhysicalPredicateGatedFoldV1.build(
            checkpoint,
            signature,
            cases,
            -64,
            predicate_cell_gate=False,
        )
    assert feature_stop.value.status == gated.STOP_FEATURE

    with pytest.raises(gated.PredicateGatedFoldBuildStop) as action_stop:
        gated.PackedPhysicalPredicateGatedFoldV1.build(
            checkpoint,
            signature,
            cases,
            64,
            action_cell_gate=False,
        )
    assert action_stop.value.status == gated.STOP_ACTION_RACE
