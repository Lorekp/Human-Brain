from __future__ import annotations

import dataclasses
from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as projection
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as relation_v2
from experiments.r3_pilot import common_one_byte_relation_eventwise_v1 as eventwise

_CURRICULUM_VALUES = (0x1E, 0x63, 0xB4, 0xC9, 0x27, 0x78, 0x8D, 0xD2)
_RECOMBINED_VALUE = 0x1B

_input_cells = projection.CommonOneByteFillerCells()
_output_cells = output_projection.CommonEventCells()


def _digits(value: int) -> tuple[int, ...]:
    return tuple(
        (value // (projection.FILLER_RADIX ** (projection.FILLER_LANES - lane - 1)))
        % projection.FILLER_RADIX
        for lane in range(projection.FILLER_LANES)
    )


def _value(digits: tuple[int, ...]) -> int:
    value = 0
    for digit in digits:
        value = value * projection.FILLER_RADIX + digit
    return value


def _shift_fillers(value: int) -> int:
    return _value(
        tuple((digit + 1) % projection.FILLER_RADIX for digit in _digits(value))
    )


def _teacher_cells(value: int) -> tuple[int, ...]:
    decision = _output_cells.race_output(
        tuple(
            tuple(24 if candidate == digit else 0 for candidate in range(4))
            for digit in _digits(value)
        )
    )
    assert decision.decision == "TOKEN"
    return decision.active_cells


def _opaque(label: bytes) -> tuple[int, ...]:
    return _output_cells.project(label).trace_cells


def _context_lessons(
    relation_cells: tuple[int, ...],
    context_cells: tuple[int, ...],
    source_cells: tuple[int, ...],
) -> tuple[relation_v2.RelationGatedOneByteLesson, ...]:
    return tuple(
        relation_v2.RelationGatedOneByteLesson(
            input_frame=_input_cells.project_one_byte(bytes((value,))),
            relation_cells=relation_cells,
            context_cells=context_cells,
            source_cells=source_cells,
            teacher_output_cells=_teacher_cells(value),
        )
        for value in _CURRICULUM_VALUES
    )


def _feed(
    start: eventwise.FrozenEventwiseOneByteRelationV1,
    lessons: tuple[relation_v2.RelationGatedOneByteLesson, ...],
    **gates: bool,
) -> tuple[
    eventwise.FrozenEventwiseOneByteRelationV1,
    tuple[eventwise.EventwiseRelationTransition, ...],
]:
    state = start
    transitions: list[eventwise.EventwiseRelationTransition] = []
    for lesson in lessons:
        transition = eventwise.advance_eventwise_common_one_byte_relation_v1(
            state,
            lesson,
            **gates,
        )
        transitions.append(transition)
        state = transition.successor
    return state, tuple(transitions)


def test_eventwise_neuronal_completeness_consolidates_without_context_replay() -> None:
    relation = _opaque(b"eventwise/relation")
    unrelated_relation = _opaque(b"eventwise/unrelated-relation")
    context_a = _opaque(b"eventwise/context-a")
    context_b = _opaque(b"eventwise/context-b")
    context_c = _opaque(b"eventwise/context-c")
    unrelated_context_a = _opaque(b"eventwise/unrelated-context-a")
    unrelated_context_b = _opaque(b"eventwise/unrelated-context-b")
    source_a = _opaque(b"eventwise/source-a")
    source_b = _opaque(b"eventwise/source-b")
    source_c = _opaque(b"eventwise/source-c")
    unrelated_source_a = _opaque(b"eventwise/unrelated-source-a")
    unrelated_source_b = _opaque(b"eventwise/unrelated-source-b")

    first_context = _context_lessons(relation, context_a, source_a)
    new_events = _context_lessons(relation, context_b, source_b)
    switch_events = _context_lessons(relation, context_c, source_c)
    unrelated_a = _context_lessons(
        unrelated_relation,
        unrelated_context_a,
        unrelated_source_a,
    )
    unrelated_b = _context_lessons(
        unrelated_relation,
        unrelated_context_b,
        unrelated_source_b,
    )
    parent = relation_v2.fit_common_one_byte_copy_relation_v2(
        first_context + unrelated_a + unrelated_b
    )
    assert parent.relation_support_count(relation) == 1
    assert parent.relation_support_count(unrelated_relation) == 2

    parent_root = parent.checkpoint_sha256
    parent_arrays = tuple(
        array.tobytes(order="C") for _, array in parent._checkpoint_arrays()
    )
    parent_provenance = parent.training_summary.provenance
    unrelated_root = parent.relation_state_sha256(unrelated_relation)
    recombined = _input_cells.project_one_byte(bytes((_RECOMBINED_VALUE,)))
    assert parent.recall(recombined, relation).status == "ABSTAIN_LOW_SUPPORT"

    start = eventwise.start_eventwise_common_one_byte_relation_v1(parent)
    paused, first_four = _feed(start, new_events[:4])
    assert all(step.status == "PENDING_EVIDENCE" for step in first_four)
    assert all(not step.consolidated for step in first_four)
    assert paused.pending_event_count == 4
    assert paused.parent.checkpoint_sha256 == parent_root
    assert paused.parent.training_summary.provenance == parent_provenance

    resumed, last_four = _feed(paused, new_events[4:])
    assert all(step.status == "PENDING_EVIDENCE" for step in last_four[:-1])
    final_transition = last_four[-1]
    assert final_transition.status == "CONSOLIDATED"
    assert final_transition.consolidated
    assert final_transition.complete_route_count == 16
    assert final_transition.event_observation_ticks == 4
    assert final_transition.event_accumulator_update_ticks == 4
    assert final_transition.event_physical_cell_steps == 28_200
    assert final_transition.consolidation_relation_excitatory_ticks == 12
    assert final_transition.consolidation_route_excitatory_ticks == 16
    assert final_transition.consolidation_route_inhibitory_ticks == 0
    assert final_transition.consolidation_provenance_writes == 1
    assert final_transition.consolidation_work_ticks == 29
    assert resumed.event_count == 8
    assert resumed.pending_event_count == 0
    assert resumed.accumulator_update_ticks == 32
    assert resumed.physical_cell_steps == 8 * 28_200
    assert resumed.consolidation_count == 1
    assert not resumed.active
    assert not resumed.poison_latched
    assert set(resumed.route_ready_winners) == {-1}
    assert set(value for row in resumed.evidence_counts for value in row) == {0}

    uninterrupted, _ = _feed(start, new_events)
    shuffled_order = (3, 0, 6, 1, 7, 4, 2, 5)
    shuffled, _ = _feed(
        start,
        tuple(new_events[index] for index in shuffled_order),
    )
    assert resumed.checkpoint_sha256 == uninterrupted.checkpoint_sha256
    assert resumed.parent.checkpoint_sha256 == uninterrupted.parent.checkpoint_sha256
    assert shuffled.checkpoint_sha256 == uninterrupted.checkpoint_sha256
    assert shuffled.parent.checkpoint_sha256 == uninterrupted.parent.checkpoint_sha256

    ready = resumed.parent.recall(recombined, relation)
    assert ready.status == "READY"
    assert ready.output.active_cells == _teacher_cells(_RECOMBINED_VALUE)
    assert ready.relation_support_count == 2
    assert resumed.parent.training_summary.provenance[:-1] == parent_provenance
    assert resumed.parent.relation_state_sha256(unrelated_relation) == unrelated_root
    assert parent.checkpoint_sha256 == parent_root
    assert tuple(
        array.tobytes(order="C") for _, array in parent._checkpoint_arrays()
    ) == parent_arrays
    assert parent.training_summary.provenance == parent_provenance

    repeated, repeated_steps = _feed(start, (new_events[0],) * 8)
    assert repeated.parent.checkpoint_sha256 == parent_root
    assert repeated.pending_event_count == 8
    assert sum(winner >= 0 for winner in repeated.route_ready_winners) == 4
    assert not any(step.consolidated for step in repeated_steps)

    accumulator_off, accumulator_off_steps = _feed(
        start,
        new_events,
        accumulator_gate=False,
    )
    assert accumulator_off.checkpoint_sha256 == start.checkpoint_sha256
    assert all(step.status == "STOP_ACCUMULATOR_OFF" for step in accumulator_off_steps)
    assert all(step.event_physical_cell_steps == 0 for step in accumulator_off_steps)

    complete_off, complete_off_steps = _feed(
        start,
        new_events,
        complete_cell_gate=False,
    )
    assert complete_off_steps[-1].status == "STOP_COMPLETE_CELL_OFF"
    assert complete_off.parent.checkpoint_sha256 == parent_root
    assert complete_off.pending_event_count == 8
    assert sum(winner >= 0 for winner in complete_off.route_ready_winners) == 16
    assert complete_off.physical_cell_steps == 8 * 28_200

    teacher_off_events = tuple(
        dataclasses.replace(lesson, teacher_gate=False) for lesson in new_events
    )
    teacher_off, teacher_off_steps = _feed(start, teacher_off_events)
    assert teacher_off.checkpoint_sha256 == start.checkpoint_sha256
    assert all(step.status == "STOP_TEACHER_OFF" for step in teacher_off_steps)
    cell_off_events = tuple(
        dataclasses.replace(lesson, cell_gate=False) for lesson in new_events
    )
    cell_off, cell_off_steps = _feed(start, cell_off_events)
    assert cell_off.checkpoint_sha256 == start.checkpoint_sha256
    assert all(step.status == "STOP_CELL_OFF" for step in cell_off_steps)

    partial, _ = _feed(start, new_events[:4])
    switched = eventwise.advance_eventwise_common_one_byte_relation_v1(
        partial,
        switch_events[0],
    )
    assert switched.status == "STOP_INCOMPLETE_CONTEXT_SWITCH"
    assert switched.successor.checkpoint_sha256 == partial.checkpoint_sha256
    assert switched.successor.parent.checkpoint_sha256 == parent_root

    contradictory = list(new_events)
    contradictory[0] = dataclasses.replace(
        contradictory[0],
        teacher_output_cells=_teacher_cells(_shift_fillers(_CURRICULUM_VALUES[0])),
    )
    poisoned, poison_steps = _feed(start, tuple(contradictory))
    assert poison_steps[-1].status == "STOP_CONFLICT_POISON"
    assert poisoned.poison_latched
    assert poisoned.parent.checkpoint_sha256 == parent_root
    assert poisoned.parent.training_summary.provenance == parent_provenance
    assert not any(step.consolidated for step in poison_steps)

    assert eventwise.NEW_CELL_COUNT == 100
    assert eventwise.TOTAL_CELL_COUNT == relation_v2.TOTAL_CELL_COUNT + 100 == 65_742
    assert eventwise.EVIDENCE_STATE_BYTES == 64
    assert eventwise.READY_WINNER_STATE_BYTES == 16
    assert eventwise.LATCH_STATE_BYTES == 2
    assert eventwise.PERSISTENT_STATE_BYTES == 226
    assert eventwise.ROUTE_RACE_CELL_STEPS_PER_EVENT == 16_800
    assert eventwise.COMPLETE_RACE_CELL_STEPS_PER_EVENT == 11_400
    assert eventwise.PHYSICAL_CELL_STEPS_PER_ACCEPTED_EVENT == 28_200
    assert eventwise.PARENT_LESSON_REPLAY_COUNT == 0
    assert eventwise.CONFIG_RECORD["host_groupby_finalize_or_completeness_flag"] is False
    assert (
        eventwise.PRIOR_MEMORY_MUTATION_COUNT,
        eventwise.PRIOR_SCHEMA_MUTATION_COUNT,
        eventwise.CORPUS_INPUT_COUNT,
        eventwise.PATH2_INPUT_COUNT,
        eventwise.DEV_INPUT_COUNT,
        eventwise.M0_INPUT_COUNT,
        eventwise.PROTECTED_INPUT_COUNT,
        eventwise.G0_INPUT_COUNT,
        eventwise.CODING_INPUT_COUNT,
    ) == (0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert sha256(Path(eventwise.__file__).read_bytes()).hexdigest() == (
        eventwise.SOURCE_SHA256
    )
    assert len(eventwise.CONFIG_SHA256) == len(resumed.checkpoint_sha256) == 64
    assert len(final_transition.sha256) == 64
