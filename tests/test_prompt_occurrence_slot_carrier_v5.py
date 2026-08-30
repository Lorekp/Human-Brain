from __future__ import annotations

import pytest
from experiments.r3_pilot import prompt_occurrence_slot_carrier_v5 as carrier_v5


def test_fresh_self_posed_occurrence_slot_carrier_and_fail_closed_join() -> None:
    summary = carrier_v5.STATIC_CONSUMER.build_summary
    assert summary.status == carrier_v5.READY
    assert summary.fresh_zero_state_nonzero_count == 0
    assert summary.fit_exact_count == summary.fit_lesson_count == 18
    assert summary.teacher_off_nonzero_count == 0
    assert summary.teacher_off_ready_count == 0
    assert summary.learned_state_bytes == carrier_v5.STATIC_CONSUMER.learned_state_bytes
    assert (
        carrier_v5.PUBLIC_TRAIN_INPUT_COUNT,
        carrier_v5.CANONICAL_INPUT_COUNT,
        carrier_v5.DESIRED_INPUT_COUNT,
        carrier_v5.PROGRAM_INPUT_COUNT,
        carrier_v5.PROPERTY_INPUT_COUNT,
        carrier_v5.TASK_INPUT_COUNT,
        carrier_v5.FAMILY_INPUT_COUNT,
        carrier_v5.DEV_INPUT_COUNT,
        carrier_v5.HELDOUT_INPUT_COUNT,
        carrier_v5.PROTECTED_INPUT_COUNT,
    ) == (0,) * 10

    treatment_certificate = carrier_v5.infer_occurrence_slot_carriers(
        reversed(carrier_v5.SELF_POSED_TREATMENT_EVENTS)
    )
    counterfactual_certificate = carrier_v5.infer_occurrence_slot_carriers(
        carrier_v5.SELF_POSED_COUNTERFACTUAL_EVENTS
    )
    assert treatment_certificate.status == carrier_v5.READY
    assert counterfactual_certificate.status == carrier_v5.READY
    assert not hasattr(treatment_certificate, "view")

    treatment = carrier_v5.require_occurrence_slot_carrier(treatment_certificate)
    counterfactual = carrier_v5.require_occurrence_slot_carrier(
        counterfactual_certificate
    )
    assert treatment.checkpoint_sha256 == counterfactual.checkpoint_sha256
    assert treatment.protocol_sha256 == counterfactual.protocol_sha256
    assert len(treatment.source_bindings) == len(counterfactual.source_bindings) == 4
    assert len(treatment.edges) == len(counterfactual.edges) == 3

    for view in (treatment, counterfactual):
        carriers = {binding.carrier_cells for binding in view.source_bindings}
        assert view.root_carrier_cells in carriers
        assert {
            endpoint
            for edge in view.edges
            for endpoint in (edge.parent_carrier_cells, edge.child_carrier_cells)
        } == carriers
        assert sorted(edge.child_position for edge in view.edges) == [0, 0, 1]
        assert sorted(edge.parent_factor_slot_occupancy for edge in view.edges) == [
            (0,),
            (0, 1, 2),
            (1,),
        ]
        assert sum(binding.raw_observation is not None for binding in view.source_bindings) == 3
        assert (
            sum(
                binding.typed_literal_event is not None
                for binding in view.source_bindings
            )
            == 1
        )
        assert sorted(
            len(binding.event_receipt_sha256s) for binding in view.source_bindings
        ) == [1, 1, 2, 2]

    treatment_carriers = {
        binding.carrier_cells for binding in treatment.source_bindings
    }
    counterfactual_carriers = {
        binding.carrier_cells for binding in counterfactual.source_bindings
    }
    assert treatment_carriers.isdisjoint(counterfactual_carriers)

    ambiguous = carrier_v5.infer_occurrence_slot_carriers(
        (
            carrier_v5.SELF_POSED_TREATMENT_EVENTS[0],
            carrier_v5.SELF_POSED_TREATMENT_EVENTS[0],
        )
    )
    assert ambiguous.status == carrier_v5.STOP_AMBIGUOUS
    assert ambiguous.stop_reason == "a physical occurrence event was duplicated"
    with pytest.raises(RuntimeError, match="physical occurrence event was duplicated"):
        carrier_v5.require_occurrence_slot_carrier(ambiguous)
