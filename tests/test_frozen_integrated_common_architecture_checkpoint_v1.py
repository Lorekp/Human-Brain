from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import frozen_integrated_common_architecture_checkpoint_v1 as integrated


def test_integrated_checkpoint_physically_emits_complete_self_posed_action_frames() -> None:
    assert integrated.ZERO_EXECUTION_RECORD == {
        "generator": 0,
        "cohort_generation": 0,
        "adjudicative_seed": 0,
        "evaluator": 0,
        "bootstrap": 0,
        "outcome": 0,
        "dev": 0,
        "m0": 0,
        "protected": 0,
    }
    assert integrated.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA == 0
    assert integrated.CURRICULUM_RECORD["association_lesson_count"] == 100
    assert integrated.CURRICULUM_RECORD["transition_lesson_count"] == 40
    assert integrated.CURRICULUM_RECORD["maximum_frame_bytes"] == 13
    assert integrated.CURRICULUM_RECORD["regulatory_composite_frame_present"] is True
    assert integrated.CELL_NAMESPACE_RECORD["private-population-overlap_count"] == 0
    assert integrated.INTEGRATED_CELL_COUNT == 569_715

    checkpoint = integrated.build_frozen_integrated_common_architecture()
    assert checkpoint.checkpoint_sha256 == (integrated.EXPECTED_INTEGRATED_CHECKPOINT_SHA256)
    assert checkpoint.component_checkpoint_sha256s == (
        ("associative", integrated.EXPECTED_ASSOCIATIVE_CHECKPOINT_SHA256),
        (
            "factorized-transition",
            integrated.EXPECTED_TRANSITION_CHECKPOINT_SHA256,
        ),
    )
    with pytest.raises(FrozenInstanceError):
        checkpoint._checkpoint_sha256 = "0" * 64

    emissions = tuple(
        checkpoint.emit((fixture.intent_event,), expected_domain=fixture.domain)
        for fixture in integrated.SELF_POSED_FIXTURES
    )
    assert all(row.status == "EMITTED" for row in emissions)
    assert tuple(row.output_bytes for row in emissions) == tuple(
        fixture.physical_frame for fixture in integrated.SELF_POSED_FIXTURES
    )
    assert all(
        row.association_length_recall.output.token == len(row.output_bytes)
        and row.transition_length_recall.output.token == len(row.output_bytes)
        and row.physical_output_decision_count == len(row.output_bytes) + 2
        and len(row.byte_recalls) == len(row.output_bytes)
        and all(recall.output.decision == "TOKEN" for recall in row.byte_recalls)
        for row in emissions
    )

    clamp_zero = emissions[1]
    clamp_one = emissions[2]
    assert len(clamp_zero.output_bytes) == len(clamp_one.output_bytes) == 13
    assert clamp_zero.decoded_action is not None
    assert clamp_zero.decoded_action.regulatory is not None
    assert clamp_zero.decoded_action.regulatory.entity_token == 0x1020_3040
    assert clamp_zero.decoded_action.regulatory.value == 0
    assert clamp_one.decoded_action is not None
    assert clamp_one.decoded_action.regulatory is not None
    assert clamp_one.decoded_action.regulatory.entity_token == 0xA1B2_C3D4
    assert clamp_one.decoded_action.regulatory.value == 1
    assert tuple(
        row.decoded_action.spatial_action_id
        for row in emissions[3:]
        if row.decoded_action is not None
    ) == tuple(range(7))

    unknown = checkpoint.emit((b"unseen-self-posed-intent",), expected_domain="spatial")
    assert unknown.status == "STOP"
    assert unknown.stop_reason == "STOP_G0_PHYSICAL_FRAME_LENGTH_AUTHORITY"
    wrong_domain = checkpoint.emit(
        (integrated.SELF_POSED_FIXTURES[3].intent_event,),
        expected_domain="regulatory",
    )
    assert wrong_domain.status == "STOP"
    assert wrong_domain.stop_reason == "STOP_G0_PHYSICAL_ACTION_DOMAIN_MISMATCH"

    with pytest.raises(
        integrated.IntegratedCommonArchitectureStop,
        match="STOP_G0_PHYSICAL_ACTION_SEQUENCE_INCOMPLETE",
    ):
        integrated.authenticate_physical_sequence(
            (emissions[3].byte_recalls[0].output,), expected_domain="spatial"
        )

    assert sha256(Path(integrated.__file__).read_bytes()).hexdigest() == (integrated.SOURCE_SHA256)
    assert len(integrated.CONFIG_SHA256) == len(integrated.CURRICULUM_SHA256) == 64
