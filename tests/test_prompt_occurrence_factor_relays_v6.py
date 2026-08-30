from __future__ import annotations

import pytest
from experiments.r3_pilot import prompt_occurrence_factor_relays_v6 as relays


def test_fresh_v6_relays_complete_retained_eight_occurrence_shape() -> None:
    summary = relays.STATIC_CONSUMER.build_summary
    assert summary.status == relays.READY
    assert summary.fresh_zero_state_nonzero_count == 0
    assert summary.fit_exact_count == summary.fit_lesson_count == 80
    assert summary.teacher_off_ready_count == 0
    assert summary.relation_operator_count == len(relays.RELATION_OPERATORS) == 6
    assert summary.rhs_kind_count == 2
    assert summary.treatment_event_count == 38
    assert relays.PUBLIC_TRAIN_INPUT_COUNT == 0
    assert relays.CANONICAL_INPUT_COUNT == 0
    assert relays.DEV_INPUT_COUNT == 0
    assert relays.HELDOUT_INPUT_COUNT == 0
    assert relays.PROTECTED_INPUT_COUNT == 0

    certificate = relays.infer_occurrence_factor_relays_v6(
        reversed(relays.SELF_POSED_TREATMENT_EVENTS)
    )
    assert certificate.status == relays.READY
    view = relays.require_occurrence_factor_relays_v6(certificate)
    assert len(view.occurrence_carriers) == 8
    assert len(view.source_relays) == 8
    assert len(view.edges) == 7
    assert len(view.factor_slots) == 16
    assert len(view.relations) == 1
    assert len(view.domains) == 2
    assert len(view.literal_copy_groups) == 2
    assert len(view.signature_relays) == 2
    assert len(view.event_receipt_sha256s) == 38
    assert view.checkpoint_sha256 == summary.checkpoint_sha256
    assert view.protocol_sha256 == summary.protocol_sha256

    occurrence = {carrier: index for index, carrier in enumerate(view.occurrence_carriers)}
    edges = tuple(
        sorted(
            (
                occurrence[row.parent_carrier_cells],
                occurrence[row.child_carrier_cells],
                row.child_position,
                row.parent_slot_occupancy,
            )
            for row in view.edges
        )
    )
    assert edges == (
        (0, 1, 0, (0, 1, 2)),
        (0, 7, 1, (3,)),
        (1, 2, 0, (0, 1, 2)),
        (2, 3, 0, (0,)),
        (2, 4, 1, (1, 2)),
        (4, 5, 0, (1,)),
        (4, 6, 1, (0,)),
    )
    factors = {
        carrier: tuple(
            (row.factor_kind, row.type_assembly.value_type)
            for row in sorted(
                (item for item in view.factor_slots if item.carrier_cells == carrier),
                key=lambda item: item.local_slot,
            )
        )
        for carrier in view.occurrence_carriers
    }
    assert factors[view.occurrence_carriers[0]] == (
        ("literal", "Int"),
        ("literal", "Int"),
        ("role", "Int"),
        ("role", "List[Int]"),
    )
    assert factors[view.occurrence_carriers[4]] == (
        ("literal", "Int"),
        ("role", "Int"),
    )
    assert view.relations[0].operator == "ne"
    assert view.relations[0].left_slot == 0
    assert view.relations[0].right_slot is None
    assert view.relations[0].right_value_event is not None
    assert all(row.domain_kind_cell == relays.DOMAIN_LITERAL for row in view.domains)
    assert all(row.class_cells for row in view.domains)
    assert {row.signature_role for row in view.signature_relays} == {
        "parameter",
        "return",
    }
    assert {
        (row.signature_role, row.type_assembly.value_type) for row in view.signature_relays
    } == {("parameter", "List[Int]"), ("return", "Int")}
    assert all(len(row.members) == 1 for row in view.literal_copy_groups)

    assert relays.infer_occurrence_factor_relays_v6(()).status == relays.STOP_ZERO
    duplicate = (*relays.SELF_POSED_TREATMENT_EVENTS, relays.SELF_POSED_TREATMENT_EVENTS[0])
    assert relays.infer_occurrence_factor_relays_v6(duplicate).status == relays.STOP_AMBIGUOUS
    assert (
        relays.infer_occurrence_factor_relays_v6(relays.SELF_POSED_TREATMENT_EVENTS[:-1]).status
        == relays.STOP_UNSUPPORTED
    )
    with pytest.raises(TypeError, match="issued exact object"):
        relays.require_occurrence_factor_relays_v6(
            object()  # type: ignore[arg-type]
        )
