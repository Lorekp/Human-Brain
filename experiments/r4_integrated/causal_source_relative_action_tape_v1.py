"""In-memory opaque causal check for the source-relative action tape.

The fixture contains only two-byte opaque events and observed generic motor
activity.  Importing this module is inert; :func:`run_gate` does not write an
artifact or touch any protected or integrated coding cohort.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

from humanbrain.source_relative_action_tape import (
    ActionTapeAblation,
    ActionTapeConfig,
    ActionTapeDecision,
    MotorStep,
    OccurrencePreservingActionTape,
    SourceRelativeMotorDemonstration,
    TapeOperation,
    TapeValence,
)

SCHEMA = "r4-causal-source-relative-action-tape-v1"
STATUS_PASS = "PASS_R4_CAUSAL_SOURCE_RELATIVE_ACTION_TAPE_V1"
CLAIM_BOUNDARY = (
    "fresh opaque-byte microfixture evidence for learned generic motor replay, "
    "fresh occurrence-relative source copying and novel source-token composition, "
    "chain-local consequence recovery, unrelated behavioral retention, and "
    "physical relational-tape/assembly controls only; demonstrations include "
    "observed generic motor trajectories and inference receives an external "
    "opaque cue inventory as micro-gate scaffolding, so this is not autonomous "
    "focus/program induction, integrated learning, language, coding, repository, "
    "SWE, or broad intelligence evidence"
)

_A = bytes.fromhex("8110")
_B = bytes.fromhex("8120")
_REVISION = bytes.fromhex("8130")
_UNRELATED = bytes.fromhex("8140")
_DECOY = bytes.fromhex("8fff")

_TWO_FROM_FIRST = (
    MotorStep(TapeOperation.REBIND, 0),
    MotorStep(TapeOperation.COPY),
    MotorStep(TapeOperation.EMIT),
    MotorStep(TapeOperation.ADVANCE),
    MotorStep(TapeOperation.COPY),
    MotorStep(TapeOperation.EMIT),
    MotorStep(TapeOperation.END),
)
_ONE_FROM_SECOND = (
    MotorStep(TapeOperation.REBIND, 1),
    MotorStep(TapeOperation.COPY),
    MotorStep(TapeOperation.EMIT),
    MotorStep(TapeOperation.END),
)
_ONE_FROM_FIRST = (
    MotorStep(TapeOperation.REBIND, 0),
    MotorStep(TapeOperation.COPY),
    MotorStep(TapeOperation.EMIT),
    MotorStep(TapeOperation.END),
)

_CONFIG = ActionTapeConfig(
    max_demonstrations=11,
    max_observation_events=3,
    max_roles=2,
    max_tape_tokens=7,
    max_motor_steps=7,
    max_event_bytes=2,
    max_cells=37,
)


def _event(first: int, second: int) -> bytes:
    return bytes((first, second))


def _demonstration(
    context: bytes,
    seed: int,
    motor_steps: tuple[MotorStep, ...],
) -> SourceRelativeMotorDemonstration:
    cues = (_event(seed, 0x11), _event(seed, 0x22))
    source = (
        _event(seed, 0xA0),
        cues[0],
        _event(seed, 0xB0),
        cues[1],
        _event(seed, 0xC0),
    )
    return SourceRelativeMotorDemonstration(context, cues, source, motor_steps)


def _demonstrations() -> tuple[SourceRelativeMotorDemonstration, ...]:
    return (
        _demonstration(_A, 0x91, _TWO_FROM_FIRST),
        _demonstration(_A, 0x92, _TWO_FROM_FIRST),
        _demonstration(_B, 0x93, _ONE_FROM_SECOND),
        _demonstration(_B, 0x94, _ONE_FROM_SECOND),
        _demonstration(_REVISION, 0x95, _TWO_FROM_FIRST),
        _demonstration(_REVISION, 0x96, _TWO_FROM_FIRST),
        _demonstration(_REVISION, 0x97, _TWO_FROM_FIRST),
        _demonstration(_REVISION, 0x98, _ONE_FROM_SECOND),
        _demonstration(_REVISION, 0x99, _ONE_FROM_SECOND),
        _demonstration(_UNRELATED, 0x9A, _ONE_FROM_FIRST),
        _demonstration(_UNRELATED, 0x9B, _ONE_FROM_FIRST),
    )


def _fresh_tape() -> tuple[tuple[bytes, bytes], tuple[bytes, ...], bytes]:
    cues = (bytes.fromhex("d101"), bytes.fromhex("d102"))
    repeated = bytes.fromhex("e1aa")
    tape = (
        repeated,
        cues[1],
        bytes.fromhex("e1bb"),
        repeated,
        cues[0],
        repeated,
        bytes.fromhex("e1cc"),
    )
    return cues, tape, repeated


def _all_physical(result: ActionTapeDecision) -> bool:
    return (
        bool(result.races)
        and all(race.physical_cell_steps > 0 for race in result.races)
        and sum(race.physical_cell_steps for race in result.races) == result.physical_cell_steps
    )


def _physical_stop(result: ActionTapeDecision, population: str) -> bool:
    race = result.races[-1]
    return bool(
        result.decision == "STOP"
        and result.emissions == ()
        and result.eligibility is None
        and race.population == population
        and race.stopped
        and race.winner_cell == race.candidate_cells[-1]
        and _all_physical(result)
    )


def _record(result: ActionTapeDecision) -> dict[str, object]:
    return {
        "decision": result.decision,
        "emissions_sha256": sha256(b"".join(result.emissions)).hexdigest(),
        "emission_count": len(result.emissions),
        "eligibility_sha256": (None if result.eligibility is None else result.eligibility.sha256),
        "focus_occurrence_cells": result.focus_occurrence_cells,
        "operation_count": len(result.operations),
        "operation_kinds": tuple(operation.value for operation in result.operations),
        "physical_cell_steps": result.physical_cell_steps,
        "race_count": len(result.races),
        "sha256": result.sha256,
        "stop_reason": result.stop_reason,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_gate() -> dict[str, object]:
    """Run one fresh, public, non-code causal mechanism check in memory."""

    demonstrations = _demonstrations()
    training_tokens = {
        token for row in demonstrations for token in (*row.cue_inventory, *row.source_tokens)
    }
    model = OccurrencePreservingActionTape.grow(demonstrations, _CONFIG)
    initial_digest = model.checkpoint_sha256

    replay_row = demonstrations[0]
    context_collision = model.respond(
        (_A, _B),
        replay_row.cue_inventory,
        replay_row.source_tokens,
    )
    replay = model.respond(
        (_DECOY, replay_row.context_event),
        replay_row.cue_inventory,
        replay_row.source_tokens,
    )
    replay_update = model.resolve(TapeValence.NEUTRAL)

    fresh_cues, fresh_source, repeated = _fresh_tape()
    _require(set((*fresh_cues, *fresh_source)).isdisjoint(training_tokens), "fresh tape leaked")
    composition = model.respond((_DECOY, _A), fresh_cues, fresh_source)
    positive = model.resolve(TapeValence.POSITIVE)
    post_positive_digest = model.checkpoint_sha256

    unrelated_before = model.respond((_UNRELATED,), fresh_cues, fresh_source)
    model.resolve(TapeValence.NEUTRAL)

    first_revision = model.respond((_REVISION,), fresh_cues, fresh_source)
    negative = model.resolve(TapeValence.NEGATIVE)
    post_negative_digest = model.checkpoint_sha256
    revised = model.respond((_REVISION,), fresh_cues, fresh_source)
    revised_positive = model.resolve(TapeValence.POSITIVE)

    unrelated_after = model.respond((_DECOY, _UNRELATED), fresh_cues, fresh_source)
    model.resolve(TapeValence.NEUTRAL)

    relational_off = model.respond(
        (_A,),
        fresh_cues,
        fresh_source,
        ablation=ActionTapeAblation.RELATIONAL_TAPE_OFF,
    )
    assembly_off = model.respond(
        (_A,),
        fresh_cues,
        fresh_source,
        ablation=ActionTapeAblation.ASSEMBLY_OFF,
    )
    duplicate_focus = model.respond(
        (_UNRELATED,),
        fresh_cues,
        (fresh_cues[0], repeated, fresh_cues[0]),
    )

    expected_replay = (replay_row.cue_inventory[0], replay_row.source_tokens[2])
    expected_composition = (fresh_cues[0], repeated)
    checks = {
        "learned_replay": replay.emissions == expected_replay,
        "neutral_is_read_only": (
            replay_update.changed_transitions == ()
            and replay_update.prior_training_sha256 == initial_digest
            and replay_update.training_sha256 == initial_digest
        ),
        "fresh_source_relative_transfer": (
            composition.emissions == expected_composition
            and composition.focus_occurrence_cells == (33,)
            and composition.focus_occurrence_cells[0] != replay.focus_occurrence_cells[0]
            and repeated not in fresh_cues
            and repeated not in training_tokens
            and set(fresh_cues).isdisjoint(training_tokens)
            and set(composition.emissions) - set(fresh_cues) == {repeated}
        ),
        "all_generic_operations_selected": set(composition.operations) == set(TapeOperation),
        "positive_reinforces_selected_chain": (
            bool(positive.changed_transitions)
            and all(edge.plastic for edge in positive.changed_transitions)
            and positive.prior_training_sha256 == initial_digest
            and positive.training_sha256 == post_positive_digest
            and post_positive_digest != initial_digest
        ),
        "negative_anchor_recovery": (
            first_revision.emissions == (fresh_cues[0], repeated)
            and revised.emissions == (fresh_cues[1],)
            and first_revision.focus_occurrence_cells != revised.focus_occurrence_cells
            and bool(negative.changed_transitions)
            and first_revision.eligibility is not None
            and {edge.sha256 for edge in negative.changed_transitions}.issubset(
                {edge.sha256 for edge in first_revision.eligibility.transitions}
            )
            and negative.prior_training_sha256 == post_positive_digest
            and negative.training_sha256 == post_negative_digest
            and post_negative_digest != post_positive_digest
            and bool(revised_positive.changed_transitions)
        ),
        "unrelated_behavior_retained": (
            unrelated_before.emissions == unrelated_after.emissions == (fresh_cues[0],)
            and unrelated_before.focus_occurrence_cells == unrelated_after.focus_occurrence_cells
        ),
        "relational_tape_off_physical_stop": (
            relational_off.stop_reason == "tape_focus:tonic_stop"
            and relational_off.races[-1].candidate_scores == (0, 1)
            and _physical_stop(relational_off, "tape_focus")
        ),
        "assembly_off_physical_stop": (
            assembly_off.stop_reason == "program_root:tonic_stop"
            and assembly_off.races[-1].candidate_scores[-1] == 1
            and all(score == 0 for score in assembly_off.races[-1].candidate_scores[:-1])
            and _physical_stop(assembly_off, "program_root")
        ),
        "duplicate_occurrences_do_not_use_host_first_match": (
            duplicate_focus.stop_reason == "tape_focus:collision"
            and duplicate_focus.races[-1].simultaneous_spike_candidates == 2
            and duplicate_focus.races[-1].stop_relay_spike_step is not None
            and _physical_stop(duplicate_focus, "tape_focus")
        ),
        "complete_physical_treatment": _all_physical(composition),
        "competing_active_contexts_use_one_neuronal_race": (
            context_collision.stop_reason == "context_program:collision"
            and context_collision.races[-1].simultaneous_spike_candidates == 2
            and _physical_stop(context_collision, "context_program")
        ),
    }
    _require(all(checks.values()), "source-relative action-tape causal check failed")

    return {
        "schema": SCHEMA,
        "status": STATUS_PASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture": {
            "demonstration_count": len(demonstrations),
            "fixture_sha256": sha256(
                b"".join(
                    event
                    for row in demonstrations
                    for event in (
                        row.context_event,
                        *row.cue_inventory,
                        *row.source_tokens,
                    )
                )
            ).hexdigest(),
            "fresh_cue_count": len(fresh_cues),
            "fresh_source_occurrences": len(fresh_source),
        },
        "training_summary": asdict(model.training_summary),
        "checks": checks,
        "trajectory": {
            "replay": _record(replay),
            "composition": _record(composition),
            "first_revision": _record(first_revision),
            "revised": _record(revised),
            "unrelated_before": _record(unrelated_before),
            "unrelated_after": _record(unrelated_after),
            "relational_tape_off": _record(relational_off),
            "assembly_off": _record(assembly_off),
            "duplicate_focus": _record(duplicate_focus),
            "context_collision": _record(context_collision),
        },
        "updates": {
            "positive_changed_edges": len(positive.changed_transitions),
            "negative_changed_edges": len(negative.changed_transitions),
            "revised_positive_changed_edges": len(revised_positive.changed_transitions),
            "initial_training_sha256": initial_digest,
            "post_positive_training_sha256": post_positive_digest,
            "post_negative_training_sha256": post_negative_digest,
            "final_training_sha256": model.checkpoint_sha256,
        },
    }


__all__ = ["CLAIM_BOUNDARY", "SCHEMA", "STATUS_PASS", "run_gate"]
