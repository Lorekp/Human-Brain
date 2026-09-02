from __future__ import annotations

from dataclasses import fields
from inspect import signature

from humanbrain.occurrence_neighborhood_scorer import OccurrenceNeighborhoodScorer
from humanbrain.raw_tape_role_binder import (
    NeuronalRawTapeRoleBinder,
    RawTapeRoleDemonstration,
    RoleBinderAblation,
)
from humanbrain.source_relative_action_tape import (
    MotorStep,
    OccurrencePreservingActionTape,
    SourceRelativeMotorDemonstration,
    TapeOperation,
    TapeValence,
)

_CONTEXT = b"extract-ordered-pair"


def _program() -> tuple[MotorStep, ...]:
    return (
        MotorStep(TapeOperation.REBIND, 0),
        MotorStep(TapeOperation.COPY),
        MotorStep(TapeOperation.EMIT),
        MotorStep(TapeOperation.REBIND, 1),
        MotorStep(TapeOperation.COPY),
        MotorStep(TapeOperation.EMIT),
        MotorStep(TapeOperation.END),
    )


def _training_sources() -> tuple[tuple[tuple[bytes, ...], tuple[bytes, ...]], ...]:
    return (
        (
            (
                b"header-a",
                b"left",
                b"=",
                b"ember",
                b";",
                b"gap-a",
                b"right",
                b"=",
                b"cairn",
                b";",
                b"tail-a",
            ),
            (b"ember", b"cairn"),
        ),
        (
            (
                b"header-b",
                b"left",
                b"=",
                b"moss",
                b";",
                b"gap-b",
                b"right",
                b"=",
                b"flint",
                b";",
                b"tail-b",
            ),
            (b"moss", b"flint"),
        ),
    )


def _models() -> tuple[NeuronalRawTapeRoleBinder, OccurrencePreservingActionTape]:
    binder_demos = tuple(
        RawTapeRoleDemonstration(
            context_event=_CONTEXT,
            source_tokens=source,
            demonstrated_roles=roles,
        )
        for source, roles in _training_sources()
    )
    motor_demos = tuple(
        SourceRelativeMotorDemonstration(
            context_event=_CONTEXT,
            cue_inventory=roles,
            source_tokens=source,
            motor_steps=_program(),
        )
        for source, roles in _training_sources()
    )
    return (
        NeuronalRawTapeRoleBinder.grow(binder_demos),
        OccurrencePreservingActionTape.grow(motor_demos),
    )


def _fresh_source() -> tuple[bytes, ...]:
    return (
        b"inventory-shift",
        b"=",
        b"decoy",
        b":",
        b"right",
        b"=",
        b"zephyr",
        b";",
        b"inserted-a",
        b"inserted-b",
        b"left",
        b"=",
        b"aurora",
        b";",
        b"tail-c",
    )


def test_fresh_names_distractors_and_inventory_shift_feed_the_motor_tape() -> None:
    binder, motor = _models()
    source = _fresh_source()
    initial_checkpoint = binder.checkpoint_sha256

    first_occurrence_inventory = tuple(dict.fromkeys(source))
    assert first_occurrence_inventory.index(b"zephyr") < first_occurrence_inventory.index(b"aurora")

    binding = binder.respond((b"context-distractor", _CONTEXT), source)

    assert binding.decision == "BIND"
    assert binding.cue_inventory == (b"aurora", b"zephyr")
    assert binding.stop_reason is None
    assert binding.physical_cell_steps > 0
    assert binder.checkpoint_sha256 == initial_checkpoint

    action = motor.respond(
        (b"context-distractor", _CONTEXT),
        binding.cue_inventory,
        source,
    )
    assert action.decision == "EMIT"
    assert action.emissions == (b"aurora", b"zephyr")
    motor.resolve(TapeValence.NEUTRAL)


def test_indexed_scores_match_the_scalar_fresh_and_ambiguous_cases() -> None:
    ambiguous_source = (
        b"noise",
        b"left",
        b"=",
        b"alpha",
        b";",
        b"middle",
        b"left",
        b"=",
        b"beta",
        b";",
        b"right",
        b"=",
        b"gamma",
        b";",
    )
    role_features = (
        frozenset({(-2, b"left"), (-1, b"="), (1, b";")}),
        frozenset({(-2, b"right"), (-1, b"="), (1, b";")}),
    )
    for source in (_fresh_source(), ambiguous_source):
        scorer = OccurrenceNeighborhoodScorer(source, neighborhood_radius=2)
        for features in role_features:
            reference = tuple(
                len(
                    features
                    & frozenset(
                        (offset, source[neighbor])
                        for offset in (-2, -1, 1, 2)
                        if 0 <= (neighbor := center + offset) < len(source)
                    )
                )
                for center in range(len(source))
            )
            assert scorer.score(features) == reference


def test_equal_raw_tape_evidence_physically_stops_without_partial_cues() -> None:
    binder, _ = _models()
    ambiguous_source = (
        b"noise",
        b"left",
        b"=",
        b"alpha",
        b";",
        b"middle",
        b"left",
        b"=",
        b"beta",
        b";",
        b"right",
        b"=",
        b"gamma",
        b";",
    )

    decision = binder.respond((_CONTEXT,), ambiguous_source)

    assert decision.decision == "STOP"
    assert decision.cue_inventory == ()
    assert decision.stop_reason == "role_0:collision"
    assert decision.races[-1].population == "role_0_neighborhood"
    assert decision.races[-1].simultaneous_spike_candidates == 2
    assert decision.races[-1].stop_reason == "collision"
    assert decision.races[-1].physical_cell_steps > ((len(ambiguous_source) + 1) * 600)


def test_binder_off_is_a_physical_tonic_stop() -> None:
    binder, _ = _models()
    initial_checkpoint = binder.checkpoint_sha256

    decision = binder.respond(
        (_CONTEXT,),
        _fresh_source(),
        ablation=RoleBinderAblation.BINDER_OFF,
    )

    assert decision.decision == "STOP"
    assert decision.cue_inventory == ()
    assert decision.stop_reason == "context_schema:tonic_stop"
    assert len(decision.races) == 1
    assert decision.races[0].population == "context_schema"
    assert decision.races[0].stop_reason == "tonic_stop"
    assert decision.physical_cell_steps > 0
    assert binder.checkpoint_sha256 == initial_checkpoint


def test_public_inference_has_no_curated_cue_or_source_position_input() -> None:
    assert tuple(field.name for field in fields(RawTapeRoleDemonstration)) == (
        "context_event",
        "source_tokens",
        "demonstrated_roles",
    )
    assert tuple(signature(NeuronalRawTapeRoleBinder.respond).parameters) == (
        "self",
        "context_events",
        "source_tokens",
        "ablation",
    )
