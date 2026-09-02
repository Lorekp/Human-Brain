from __future__ import annotations

from inspect import signature

import pytest
from humanbrain.raw_source_motor_controller import (
    RawSourceMotorAblation,
    RawSourceMotorController,
    RawSourceMotorPendingError,
    SuccessfulSourceMotorExperience,
)
from humanbrain.source_relative_action_tape import TapeValence

_CONTEXT = bytes.fromhex("c611")
_DISTRACTOR_CONTEXT = bytes.fromhex("c6ff")
_RIGHT = bytes.fromhex("f601")
_LINK = bytes.fromhex("f602")
_SEPARATOR = bytes.fromhex("f603")
_LEFT = bytes.fromhex("f604")


def _experience(seed: int) -> SuccessfulSourceMotorExperience:
    left_value = bytes((seed, 0xA1))
    right_value = bytes((seed, 0xB1))
    before = (
        bytes((seed, 0x01)),
        _RIGHT,
        _LINK,
        right_value,
        _SEPARATOR,
        bytes((seed, 0x02)),
        _LEFT,
        _LINK,
        left_value,
        _SEPARATOR,
        bytes((seed, 0x03)),
    )
    return SuccessfulSourceMotorExperience(
        context_event=_CONTEXT,
        before_tokens=before,
        after_tokens=(left_value, _SEPARATOR, right_value, _SEPARATOR),
        provenance_id=f"opaque-success-{seed}",
    )


def _controller() -> RawSourceMotorController:
    return RawSourceMotorController.grow((_experience(0xD1), _experience(0xD2)))


def _fresh_source() -> tuple[tuple[bytes, ...], bytes, bytes]:
    left_value = bytes.fromhex("e1a1")
    right_value = bytes.fromhex("e1b1")
    source = (
        bytes.fromhex("e101"),
        _RIGHT,
        _LINK,
        right_value,
        _SEPARATOR,
        bytes.fromhex("e1d1"),
        bytes.fromhex("e1d2"),
        bytes.fromhex("e1d3"),
        _LEFT,
        _LINK,
        left_value,
        _SEPARATOR,
        bytes.fromhex("e103"),
    )
    return source, left_value, right_value


def test_raw_source_transfer_renames_distracts_and_reorders_bound_inventory() -> None:
    controller = _controller()
    source, left_value, right_value = _fresh_source()
    initial_checkpoint = controller.checkpoint_sha256

    assert source.index(right_value) < source.index(left_value)
    decision = controller.respond((_DISTRACTOR_CONTEXT, _CONTEXT), source)

    assert decision.decision == "EMIT"
    assert decision.emissions == (
        left_value,
        _SEPARATOR,
        right_value,
        _SEPARATOR,
    )
    assert decision.binder_decision.decision == "BIND"
    assert decision.binder_decision.cue_inventory == (left_value, right_value)
    assert decision.tape_decision is not None
    assert decision.tape_decision.decision == "EMIT"
    assert decision.receipt is not None
    assert (
        decision.receipt.training_custody_sha256
        == controller.training_summary.training_custody_sha256
    )
    assert decision.receipt.binder_decision_sha256 == decision.binder_decision.sha256
    assert decision.receipt.tape_decision_sha256 == decision.tape_decision.sha256
    assert decision.physical_cell_steps == (
        decision.binder_decision.physical_cell_steps + decision.tape_decision.physical_cell_steps
    )
    with pytest.raises(RawSourceMotorPendingError, match="resolve the pending"):
        controller.respond((_CONTEXT,), source)

    update = controller.resolve(TapeValence.POSITIVE)
    assert update.receipt_sha256 == decision.receipt.sha256
    assert update.prior_training_sha256 == initial_checkpoint
    assert update.training_sha256 == controller.checkpoint_sha256
    assert update.training_sha256 != initial_checkpoint
    assert update.tape_update.changed_transitions
    with pytest.raises(RawSourceMotorPendingError, match="no pending"):
        controller.resolve(TapeValence.NEUTRAL)

    reinforced_checkpoint = controller.checkpoint_sha256
    controller.respond((_CONTEXT,), source)
    neutral = controller.resolve(TapeValence.NEUTRAL)
    assert neutral.tape_update.changed_transitions == ()
    assert neutral.prior_training_sha256 == reinforced_checkpoint
    assert neutral.training_sha256 == reinforced_checkpoint


def test_binder_off_and_tape_off_propagate_physical_stop_without_pending() -> None:
    controller = _controller()
    source, _, _ = _fresh_source()
    initial_checkpoint = controller.checkpoint_sha256

    binder_off = controller.respond(
        (_CONTEXT,),
        source,
        ablation=RawSourceMotorAblation.BINDER_OFF,
    )
    tape_off = controller.respond(
        (_CONTEXT,),
        source,
        ablation=RawSourceMotorAblation.TAPE_OFF,
    )

    assert binder_off.decision == "STOP"
    assert binder_off.emissions == ()
    assert binder_off.stop_stage == "binder"
    assert binder_off.stop_reason == "binder:context_schema:tonic_stop"
    assert binder_off.binder_decision.races[-1].stopped
    assert binder_off.tape_decision is None
    assert binder_off.receipt is None
    assert tape_off.decision == "STOP"
    assert tape_off.emissions == ()
    assert tape_off.stop_stage == "tape"
    assert tape_off.stop_reason == "tape:tape_focus:tonic_stop"
    assert tape_off.binder_decision.decision == "BIND"
    assert tape_off.tape_decision is not None
    assert tape_off.tape_decision.races[-1].stopped
    assert tape_off.receipt is None
    assert controller.checkpoint_sha256 == initial_checkpoint
    with pytest.raises(RawSourceMotorPendingError, match="no pending"):
        controller.resolve(TapeValence.NEGATIVE)


def test_public_inference_surface_has_no_training_or_routing_scaffold() -> None:
    assert tuple(signature(RawSourceMotorController.respond).parameters) == (
        "self",
        "context_events",
        "source_tokens",
        "ablation",
    )
