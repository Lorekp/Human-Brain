from __future__ import annotations

import inspect

from experiments.r3_pilot import prompt_occurrence_factor_relays_v6 as relays
from experiments.r3_pilot import (
    typed_semantic_packed_occurrence_factor_authorized_plan_slice_v1 as integrated,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([],),
    ([0, 1, 2, 3],),
    ([1, 2, 10],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
LITERALS = (
    -15,
    -14,
    -12,
    -10,
    -8,
    -6,
    -4,
    -2,
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
)
DESIRED = (0, 2, 2, 1, 3)


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "opaque",
        (ProgramParameter("items", ProgramType.List(integer)),),
        integer,
    )


class _Explodes:
    def __iter__(self):
        raise AssertionError("stopped build/selection iterated sealed evidence")


def test_v6_opaque_certificate_privately_authorizes_retained_plan_slice() -> None:
    build_parameters = inspect.signature(
        integrated.PackedOccurrenceFactorAuthorizedPlanSliceV1.build
    ).parameters
    assert tuple(build_parameters) == (
        "signature",
        "cases",
        "literals",
        "max_cost",
        "source_certificate",
    )
    assert "desired" not in build_parameters

    source = relays.infer_occurrence_factor_relays_v6(reversed(relays.SELF_POSED_TREATMENT_EVENTS))
    assert source.status == relays.READY
    frontier = integrated.PackedOccurrenceFactorAuthorizedPlanSliceV1.build(
        _signature(),
        CASES,
        LITERALS,
        9,
        source,
    )
    state = frontier.state
    receipt = frontier.plan_slice_receipt
    assert frontier.status == integrated.READY
    assert state.structural_freeze_attempts == 1
    assert state.plan_build_attempts == 1
    assert state.desired_reads_before_select == 0
    assert state.proof_witnesses_before_select == 0
    assert state.source_certificate_sha256 == source.sha256
    assert state.source_checkpoint_sha256 == source.checkpoint_sha256
    assert state.source_protocol_sha256 == relays.PROTOCOL.protocol_sha256
    assert state.physical_work_receipt_sha256 is not None
    assert state.converter_receipt is not None
    assert state.converter_receipt.work.occurrence_carrier_reads == 8
    assert state.converter_receipt.work.edge_relay_reads == 7
    assert state.converter_receipt.work.factor_slot_relay_reads == 16
    assert state.converter_receipt.work.relation_relay_reads == 1
    assert state.converter_receipt.work.domain_relay_reads == 2
    assert state.converter_receipt.work.copy_group_relay_reads == 2
    assert state.converter_receipt.work.signature_relay_reads == 2
    assert state.converter_receipt.work.structural_freeze_attempts == 1
    assert receipt is not None
    assert frontier.work is not None
    assert frontier.work.relation.completed_units == 25_652
    assert frontier.work.physical_predictions == 1_821
    assert frontier.work.physical_races == 4_381
    assert frontier.work.concrete_proof_witnesses == 0
    assert not hasattr(frontier, "structural_certificate")

    selected = frontier.select(DESIRED)
    assert selected.status == integrated.SELECTED
    assert selected.selected_eclass_id == receipt.authenticated_eclass_id
    assert selected.exact_proof_count == len(selected.witnesses) == 1
    assert selected.selected_proof_forest_sha256 is not None
    assert selected.work is not None
    assert selected.work.relation.completed_units == 26_073
    assert selected.work.physical_races == 4_382

    forged = integrated.PackedOccurrenceFactorAuthorizedPlanSliceV1.build(
        _signature(),
        _Explodes(),  # type: ignore[arg-type]
        _Explodes(),  # type: ignore[arg-type]
        9,
        object(),  # type: ignore[arg-type]
    )
    assert forged.status == integrated.STOP_SOURCE_CERTIFICATE
    assert forged.state.structural_freeze_attempts == 0
    assert forged.state.plan_build_attempts == 0
    assert forged.select(_Explodes()).status == integrated.STOP_SOURCE_CERTIFICATE  # type: ignore[arg-type]

    unsupported_source = relays.infer_occurrence_factor_relays_v6(
        relays.SELF_POSED_TREATMENT_EVENTS[:-1]
    )
    unsupported = integrated.PackedOccurrenceFactorAuthorizedPlanSliceV1.build(
        _signature(),
        _Explodes(),  # type: ignore[arg-type]
        _Explodes(),  # type: ignore[arg-type]
        9,
        unsupported_source,
    )
    assert unsupported.status == integrated.STOP_SOURCE_CERTIFICATE
    assert unsupported.state.plan_build_attempts == 0

    wrong_signature = TypedSignature(
        "opaque",
        (ProgramParameter("value", ProgramType.Int()),),
        ProgramType.Int(),
    )
    mismatched = integrated.PackedOccurrenceFactorAuthorizedPlanSliceV1.build(
        wrong_signature,
        _Explodes(),  # type: ignore[arg-type]
        _Explodes(),  # type: ignore[arg-type]
        9,
        source,
    )
    assert mismatched.status == integrated.STOP_SIGNATURE_MISMATCH
    assert mismatched.state.structural_freeze_attempts == 0
    assert mismatched.state.plan_build_attempts == 0
