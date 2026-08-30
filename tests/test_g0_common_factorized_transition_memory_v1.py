from __future__ import annotations

from dataclasses import dataclass

from experiments.g0 import common_event_associative_memory_v1 as whole_trace_memory
from experiments.g0.common_factorized_transition_memory_v1 import (
    BRANCH_SELECTOR_POPULATION_COUNT,
    BRANCH_ZERO_SCORE_SPIKE_STEP,
    CHECKPOINT_ARRAY_BYTES,
    CommonTransitionLesson,
    fit_common_factorized_transitions,
)
from experiments.g0.frozen_regulatory_adapter_v1 import (
    ACTION_CLAMP,
    ActionEvent,
    ObservationEvent,
    serialize_event,
)
from experiments.g0.spatial_candidate_adapter_v1 import SpatialStep, encode_transcript


@dataclass(frozen=True, slots=True)
class _TransitionFixture:
    context: tuple[bytes, ...]
    intervention: bytes
    consequence: bytes


def _regulatory_fixture(
    cue_token: int,
    consequence: int,
    background: int,
) -> _TransitionFixture:
    rows = [(cue_token, 1)]
    rows.extend(
        (0x0100_0000 + background * 0x100 + index, (background + index) % 2) for index in range(7)
    )
    context = serialize_event(ObservationEvent(tuple(rows), time=10 + background))
    intervention = serialize_event(ActionEvent(ACTION_CLAMP, cue_token, 1))
    return _TransitionFixture((context,), intervention, bytes((consequence,)))


def _spatial_fixture(
    cue: tuple[int, int, int],
    action: int,
    consequence: int,
    background: int,
) -> _TransitionFixture:
    def step(*, phase: int, action_id: int) -> bytes:
        image = bytearray(
            (index * 7 + background * 17 + phase * 29) % 128 for index in range(7 * 7 * 3)
        )
        image[:3] = bytes(cue)
        return encode_transcript(
            (
                SpatialStep(
                    image=bytes(image),
                    direction=(background + phase) % 4,
                    action_id=action_id,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                ),
            )
        )

    return _TransitionFixture(
        (step(phase=0, action_id=0),),
        step(phase=1, action_id=action),
        bytes((consequence,)),
    )


_REGISTRY = (
    ("reg-a", lambda background: _regulatory_fixture(0x71625344, 0x00, background)),
    ("reg-b", lambda background: _regulatory_fixture(0x62534435, 0xFF, background)),
    ("reg-a-renamed", lambda background: _regulatory_fixture(0xA4B5C6D7, 0x00, background)),
    ("reg-b-renamed", lambda background: _regulatory_fixture(0xB5C6D7E8, 0xFF, background)),
    ("spatial-a", lambda background: _spatial_fixture((193, 197, 199), 2, 0x00, background)),
    ("spatial-b", lambda background: _spatial_fixture((211, 223, 227), 5, 0xFF, background)),
    (
        "spatial-a-renamed",
        lambda background: _spatial_fixture((229, 233, 239), 3, 0x00, background),
    ),
    (
        "spatial-b-renamed",
        lambda background: _spatial_fixture((241, 251, 253), 6, 0xFF, background),
    ),
)


def _training_lessons() -> tuple[CommonTransitionLesson, ...]:
    return tuple(
        CommonTransitionLesson(
            fixture.context,
            fixture.intervention,
            fixture.consequence,
        )
        for _name, build in _REGISTRY
        for background in (0, 1)
        for fixture in (build(background),)
    )


def test_factorized_transition_cues_transfer_across_backgrounds_and_renames() -> None:
    lessons = _training_lessons()
    memory = fit_common_factorized_transitions(lessons)
    checkpoint = memory.checkpoint_sha256
    holdouts = tuple((name, build(3)) for name, build in _REGISTRY)
    replays = tuple(
        memory.recall(fixture.context, fixture.intervention) for _name, fixture in holdouts
    )
    assert len(replays) == 8
    assert all(replay.status == "RECALLED" for replay in replays)
    assert tuple(replay.output.token for replay in replays) == tuple(
        fixture.consequence[0] for _name, fixture in holdouts
    )
    assert all(
        replay.work.branch_selector_race_decisions == BRANCH_SELECTOR_POPULATION_COUNT
        for replay in replays
    )
    assert all(replay.work.branch_selector_cell_steps > 0 for replay in replays)
    assert memory.checkpoint_sha256 == checkpoint

    whole_trace = whole_trace_memory.fit_common_event_associations(
        tuple(
            whole_trace_memory.CommonAssociationLesson(
                (*lesson.context_events, lesson.intervention_event),
                lesson.consequence_byte,
            )
            for lesson in lessons
        )
    )
    whole_replays = tuple(
        whole_trace.recall((*fixture.context, fixture.intervention)) for _name, fixture in holdouts
    )
    assert all(replay.status == "STOP_UNKNOWN" for replay in whole_replays)
    assert memory.training_summary.checkpoint_array_bytes == CHECKPOINT_ARRAY_BYTES
    assert whole_trace.training_summary.checkpoint_array_bytes == (
        whole_trace_memory.CHECKPOINT_ARRAY_BYTES
    )
    assert CHECKPOINT_ARRAY_BYTES > whole_trace_memory.CHECKPOINT_ARRAY_BYTES

    first = holdouts[0][1]
    reset = memory.recall(first.context, first.intervention, memory_reset=True)
    assert reset.status == "STOP_UNKNOWN"
    assert reset.context_branch_zero_stop_count == len(reset.context_support_scores)
    assert reset.intervention_branch_zero_stop_count < len(reset.intervention_support_scores)
    mismatched = memory.recall(holdouts[0][1].context, holdouts[1][1].intervention)
    assert mismatched.status == "STOP_UNKNOWN"
    assert any(score > 12 for score in mismatched.context_support_scores)
    assert any(score > 12 for score in mismatched.intervention_support_scores)


def test_factorized_transition_controls_stop_without_domain_repairs() -> None:
    build = _REGISTRY[0][1]
    teacher_off = fit_common_factorized_transitions(
        tuple(
            CommonTransitionLesson(
                fixture.context,
                fixture.intervention,
                fixture.consequence,
                teacher_gate=False,
            )
            for fixture in (build(0), build(1))
        )
    )
    assert teacher_off.training_summary.work.teacher_cell_spikes == 0
    assert teacher_off.training_summary.work.committed_excitatory_ticks == 0
    fixture = build(3)
    assert teacher_off.recall(fixture.context, fixture.intervention).status == ("STOP_UNKNOWN")

    cell_off = fit_common_factorized_transitions(
        tuple(
            CommonTransitionLesson(
                fixture.context,
                fixture.intervention,
                fixture.consequence,
                cell_gate=False,
            )
            for fixture in (build(0), build(1))
        )
    )
    assert cell_off.training_summary.work.teacher_cell_spikes == 8
    assert cell_off.training_summary.work.committed_excitatory_ticks == 0
    cell_off_recall = cell_off.recall(fixture.context, fixture.intervention)
    assert cell_off_recall.status == "STOP_UNKNOWN"
    assert cell_off_recall.context_branch_zero_stop_count == len(
        cell_off_recall.context_support_scores
    )
    assert cell_off_recall.intervention_branch_zero_stop_count == len(
        cell_off_recall.intervention_support_scores
    )
    assert all(
        step == BRANCH_ZERO_SCORE_SPIKE_STEP
        for step in (
            *cell_off_recall.context_branch_spike_steps,
            *cell_off_recall.intervention_branch_spike_steps,
        )
    )
    assert all(
        count > 1
        for count in (
            *cell_off_recall.context_branch_simultaneous_spike_candidates,
            *cell_off_recall.intervention_branch_simultaneous_spike_candidates,
        )
    )

    base = build(0)
    second = build(1)
    ambiguous = fit_common_factorized_transitions(
        (
            CommonTransitionLesson(base.context, base.intervention, b"\x00"),
            CommonTransitionLesson(second.context, second.intervention, b"\x00"),
            CommonTransitionLesson(base.context, base.intervention, b"\xff"),
        )
    )
    collision = ambiguous.recall(base.context, base.intervention)
    assert collision.status == "STOP_AMBIGUOUS_COLLISION"
    assert collision.output.stop_reason == "COLLISION"
