from __future__ import annotations

import pytest
from experiments.r3_pilot import public_train_fit_occurrence_factor_learner_v1 as learner


def test_contextual_fit_learner_reuses_carriers_for_complete_physical_structure() -> None:
    summary = learner.BUILD_SUMMARY
    assert summary.status == learner.READY
    assert summary.fit_exact_count == summary.fit_lesson_count == 372
    assert (
        summary.source_lesson_count,
        summary.edge_lesson_count,
        summary.factor_lesson_count,
        summary.relation_lesson_count,
        summary.domain_lesson_count,
        summary.copy_lesson_count,
        summary.signature_lesson_count,
    ) == (77, 68, 164, 9, 18, 18, 18)
    assert summary.local_source_cue_count == 26
    assert summary.contextual_source_query_count == 77
    assert summary.query_target_conflict_count == 0
    assert summary.trace_collision_count == 0
    assert summary.teacher_off_ready_count == 0
    assert summary.cell_off_ready_count == 0
    assert summary.geometry_derangement_ready_count == 0
    assert summary.nonlesson_query_count == 1_675
    assert summary.nonlesson_ready_count == 0
    assert summary.nonlesson_ambiguous_count == 0
    assert summary.input_order_invariant

    views = tuple(
        learner.require_public_train_fit_occurrence_factors_v1(
            learner.infer_public_train_fit_occurrence_factors_v1(bundle)
        )
        for bundle in learner.FIT_SOURCE_BUNDLES
    )
    assert len(views) == 9
    assert (
        sum(len(view.source_relays) for view in views),
        sum(len(view.edges) for view in views),
        sum(len(view.factor_slots) for view in views),
        sum(len(view.relations) for view in views),
        sum(len(view.domains) for view in views),
        sum(len(view.literal_copy_groups) for view in views),
        sum(len(view.signature_relays) for view in views),
    ) == (77, 68, 164, 9, 18, 18, 18)
    for view in views:
        carriers = set(view.occurrence_carriers)
        assert {row.carrier_cells for row in view.source_relays} == carriers
        assert all(
            row.parent_carrier_cells in carriers
            and row.child_carrier_cells in carriers
            for row in view.edges
        )
        assert all(row.carrier_cells in carriers for row in view.factor_slots)
        assert all(row.carrier_cells in carriers for row in view.relations)
        assert all(row.carrier_cells in carriers for row in view.domains)
        assert all(
            carrier in carriers
            for group in view.literal_copy_groups
            for carrier, _slot in group.members
        )
        assert all(
            row.root_carrier_cells == view.root_carrier_cells
            for row in view.signature_relays
        )

    first_bundle = learner.FIT_SOURCE_BUNDLES[0]
    reversed_view = learner.require_public_train_fit_occurrence_factors_v1(
        learner.infer_public_train_fit_occurrence_factors_v1(reversed(first_bundle))
    )
    assert reversed_view.sha256 == views[0].sha256

    duplicate = learner.infer_public_train_fit_occurrence_factors_v1(
        (*first_bundle, first_bundle[0])
    )
    assert duplicate.status == learner.STOP_AMBIGUOUS
    with pytest.raises(RuntimeError, match="duplicated"):
        learner.require_public_train_fit_occurrence_factors_v1(duplicate)

    assert learner.HELDOUT_INPUT_COUNT == 0
    assert learner.HELDOUT_TARGET_READ_COUNT == 0
    assert learner.DEV_INPUT_COUNT == 0
    assert learner.M0_INPUT_COUNT == 0
    assert learner.C1_INPUT_COUNT == 0
    assert learner.PATH2_INPUT_COUNT == 0
    assert learner.G0_INPUT_COUNT == 0
    assert learner.PROTECTED_INPUT_COUNT == 0
    assert learner.V1_PARTIAL_TARGET_READ_COUNT == 0
    assert learner.V4_CHECKPOINT_READ_COUNT == 0
    assert learner.V5_CHECKPOINT_READ_COUNT == 0
    assert learner.CANONICAL_PANEL_SPEND_COUNT == 0
