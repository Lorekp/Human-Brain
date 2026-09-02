from __future__ import annotations

from dataclasses import fields
from inspect import signature

import pytest
from experiments.r4_integrated import causal_source_relative_action_tape_v1 as gate
from humanbrain.source_relative_action_tape import (
    ActionTapePendingError,
    MotorStep,
    OccurrencePreservingActionTape,
    SourceRelativeMotorDemonstration,
    TapeOperation,
    TapeValence,
)


def _minimal_demonstration() -> SourceRelativeMotorDemonstration:
    cue = bytes.fromhex("d811")
    return SourceRelativeMotorDemonstration(
        context_event=bytes.fromhex("c801"),
        cue_inventory=(cue,),
        source_tokens=(bytes.fromhex("e800"), cue, bytes.fromhex("e822")),
        motor_steps=(
            MotorStep(TapeOperation.REBIND, 0),
            MotorStep(TapeOperation.COPY),
            MotorStep(TapeOperation.EMIT),
            MotorStep(TapeOperation.END),
        ),
    )


def test_opaque_causal_gate_replays_composes_recovers_and_ablates() -> None:
    result = gate.run_gate()

    assert result["status"] == gate.STATUS_PASS
    assert all(result["checks"].values())
    assert result["trajectory"]["composition"]["operation_kinds"] == (
        "rebind",
        "copy",
        "emit",
        "advance",
        "copy",
        "emit",
        "end",
    )
    assert result["trajectory"]["context_collision"]["decision"] == "STOP"
    assert (
        result["updates"]["initial_training_sha256"]
        != result["updates"]["post_positive_training_sha256"]
    )
    assert (
        result["updates"]["post_positive_training_sha256"]
        != result["updates"]["post_negative_training_sha256"]
    )


def test_motor_training_surface_has_no_answer_or_source_position_field() -> None:
    assert tuple(field.name for field in fields(SourceRelativeMotorDemonstration)) == (
        "context_event",
        "cue_inventory",
        "source_tokens",
        "motor_steps",
    )
    assert tuple(signature(OccurrencePreservingActionTape.respond).parameters) == (
        "self",
        "observation_events",
        "cue_inventory",
        "source_tokens",
        "ablation",
    )


def test_exact_pending_protocol_and_neutral_inference_are_read_only() -> None:
    demonstration = _minimal_demonstration()
    model = OccurrencePreservingActionTape.grow((demonstration, demonstration))
    initial_digest = model.checkpoint_sha256

    decision = model.respond(
        (demonstration.context_event,),
        demonstration.cue_inventory,
        demonstration.source_tokens,
    )
    assert decision.emissions == demonstration.cue_inventory
    with pytest.raises(ActionTapePendingError, match="resolve the pending"):
        model.respond(
            (demonstration.context_event,),
            demonstration.cue_inventory,
            demonstration.source_tokens,
        )

    update = model.resolve(TapeValence.NEUTRAL)
    assert update.changed_transitions == ()
    assert update.prior_training_sha256 == initial_digest
    assert update.training_sha256 == initial_digest
    assert model.checkpoint_sha256 == initial_digest
    with pytest.raises(ActionTapePendingError, match="no pending"):
        model.resolve(TapeValence.NEUTRAL)


def test_training_rejects_an_ambiguous_rebind_without_a_supplied_span() -> None:
    cue = bytes.fromhex("d922")
    with pytest.raises(ValueError, match="one unambiguous occurrence"):
        SourceRelativeMotorDemonstration(
            context_event=bytes.fromhex("c902"),
            cue_inventory=(cue,),
            source_tokens=(cue, bytes.fromhex("e933"), cue),
            motor_steps=(
                MotorStep(TapeOperation.REBIND, 0),
                MotorStep(TapeOperation.COPY),
                MotorStep(TapeOperation.EMIT),
                MotorStep(TapeOperation.END),
            ),
        )
