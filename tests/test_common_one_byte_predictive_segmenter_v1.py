from __future__ import annotations

import dataclasses
from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as projection
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as relation_v2
from experiments.r3_pilot import common_one_byte_predictive_segmenter_v1 as segmenter
from experiments.r3_pilot import common_one_byte_relation_eventwise_v1 as eventwise

_CURRICULUM_VALUES = (0x1E, 0x63, 0xB4, 0xC9, 0x27, 0x78, 0x8D, 0xD2)

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


def _parent_lessons(
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


def _event(
    value: int,
    relation_cells: tuple[int, ...],
    interval_cell_id: int,
    *,
    shifted: bool = False,
) -> segmenter.PhysicalTimedRelationEvent:
    return segmenter.PhysicalTimedRelationEvent(
        input_frame=_input_cells.project_one_byte(bytes((value,))),
        relation_cells=relation_cells,
        teacher_output_cells=_teacher_cells(
            _shift_fillers(value) if shifted else value
        ),
        interval_cell_id=interval_cell_id,
    )


def _segment_events(
    relation_cells: tuple[int, ...],
    opening_interval: int,
    within_interval: int,
    *,
    shifted: bool = False,
) -> tuple[segmenter.PhysicalTimedRelationEvent, ...]:
    return tuple(
        _event(
            value,
            relation_cells,
            opening_interval if ordinal == 0 else within_interval,
            shifted=shifted,
        )
        for ordinal, value in enumerate(_CURRICULUM_VALUES)
    )


def _feed(
    start: segmenter.FrozenPredictiveSegmenterV1,
    events: tuple[segmenter.PhysicalTimedRelationEvent, ...],
    **gates: bool,
) -> tuple[
    segmenter.FrozenPredictiveSegmenterV1,
    tuple[segmenter.PredictiveSegmentationTransition, ...],
]:
    state = start
    transitions: list[segmenter.PredictiveSegmentationTransition] = []
    for event in events:
        transition = segmenter.advance_predictive_segmenter_v1(
            state,
            event,
            **gates,
        )
        transitions.append(transition)
        state = transition.successor
    return state, tuple(transitions)


def test_predictive_interval_segmenter_learns_boundaries_from_surprise() -> None:
    relation = _opaque(b"predictive-segmenter/relation")
    unrelated_relation = _opaque(b"predictive-segmenter/unrelated-relation")
    unrelated_a = _parent_lessons(
        unrelated_relation,
        _opaque(b"predictive-segmenter/unrelated-context-a"),
        _opaque(b"predictive-segmenter/unrelated-source-a"),
    )
    unrelated_b = _parent_lessons(
        unrelated_relation,
        _opaque(b"predictive-segmenter/unrelated-context-b"),
        _opaque(b"predictive-segmenter/unrelated-source-b"),
    )
    parent = relation_v2.fit_common_one_byte_copy_relation_v2(
        unrelated_a + unrelated_b
    )
    parent_root = parent.checkpoint_sha256
    parent_arrays = tuple(
        array.tobytes(order="C") for _, array in parent._checkpoint_arrays()
    )
    parent_provenance = parent.training_summary.provenance
    unrelated_root = parent.relation_state_sha256(unrelated_relation)
    assert unrelated_root is not None

    initial = segmenter.start_predictive_segmenter_v1(
        eventwise.start_eventwise_common_one_byte_relation_v1(parent)
    )
    interval_a, interval_b = segmenter.INTERVAL_CELL_IDS
    calibration_events = _segment_events(relation, interval_a, interval_a)
    shifted_events = _segment_events(
        relation,
        interval_b,
        interval_a,
        shifted=True,
    )
    identity_events = _segment_events(relation, interval_b, interval_a)

    teacher_off = segmenter.advance_predictive_segmenter_v1(
        initial,
        dataclasses.replace(calibration_events[0], teacher_gate=False),
    )
    cell_off = segmenter.advance_predictive_segmenter_v1(
        initial,
        dataclasses.replace(calibration_events[0], cell_gate=False),
    )
    assert teacher_off.status == "STOP_TEACHER_OFF"
    assert cell_off.status == "STOP_CELL_OFF"
    assert teacher_off.successor.checkpoint_sha256 == initial.checkpoint_sha256
    assert cell_off.successor.checkpoint_sha256 == initial.checkpoint_sha256

    calibrated, calibration_steps = _feed(initial, calibration_events)
    assert calibration_steps[0].status == "STREAM_START_SEGMENT_PENDING"
    assert calibration_steps[0].decision_authority == "STREAM_START"
    assert calibration_steps[-1].status == "SEGMENT_COMPLETE"
    assert calibration_steps[-1].segment_completed
    assert all(
        step.physical_cell_steps == segmenter.ROUTED_EVENT_CELL_STEPS
        for step in calibration_steps
    )
    assert calibrated.segment_count == 1
    assert calibrated.complete_hold
    assert not calibrated.calibration
    assert len(calibrated.internal_provenance_cell_ids) == 1
    assert calibrated.cue_excitatory == ((4, 0), (0, 0))
    assert calibrated.cue_inhibitory == ((0, 4), (0, 0))
    assert calibrated.v2_parent.relation_support_count(relation) == 1

    comparator_off = segmenter.advance_predictive_segmenter_v1(
        calibrated,
        shifted_events[0],
        comparator_gate=False,
    )
    assert comparator_off.status == "ABSTAIN_SEGMENT_AMBIGUITY"
    assert comparator_off.decision == "TONIC_STOP"
    assert comparator_off.successor.checkpoint_sha256 == calibrated.checkpoint_sha256

    plasticity_off, _ = _feed(
        calibrated,
        shifted_events,
        plasticity_gate=False,
    )
    plasticity_off, _ = _feed(
        plasticity_off,
        identity_events,
        plasticity_gate=False,
    )
    assert plasticity_off.cue_excitatory == calibrated.cue_excitatory
    assert plasticity_off.cue_inhibitory == calibrated.cue_inhibitory
    unlearned_cue = segmenter.advance_predictive_segmenter_v1(
        plasticity_off,
        identity_events[0],
        plasticity_gate=False,
    )
    assert unlearned_cue.status == "ABSTAIN_SEGMENT_AMBIGUITY"
    assert unlearned_cue.decision == "COLLISION"

    shifted, shifted_steps = _feed(calibrated, shifted_events)
    assert shifted_steps[0].decision == "NEW"
    assert shifted_steps[0].decision_authority == "CONSEQUENCE_SURPRISE"
    assert shifted_steps[0].mismatch_count == projection.FILLER_LANES
    assert shifted_steps[-1].segment_completed
    assert shifted.cue_excitatory == ((4, 0), (0, 1))
    assert shifted.cue_inhibitory == ((0, 4), (1, 0))

    ambiguous = segmenter.advance_predictive_segmenter_v1(
        shifted,
        shifted_events[0],
    )
    assert ambiguous.status == "ABSTAIN_SEGMENT_AMBIGUITY"
    assert ambiguous.decision == "COLLISION"
    assert ambiguous.decision_raw_scores == (12, 12)
    assert ambiguous.successor.checkpoint_sha256 == shifted.checkpoint_sha256

    trained, second_surprise_steps = _feed(shifted, identity_events)
    assert second_surprise_steps[0].decision_authority == "CONSEQUENCE_SURPRISE"
    assert second_surprise_steps[0].mismatch_count == projection.FILLER_LANES
    assert second_surprise_steps[-1].segment_completed
    assert trained.segment_count == 3
    assert trained.cue_excitatory == ((4, 0), (0, 2))
    assert trained.cue_inhibitory == ((0, 4), (2, 0))

    cue_off = segmenter.advance_predictive_segmenter_v1(
        trained,
        identity_events[0],
        cue_gate=False,
    )
    assert cue_off.status == "ABSTAIN_SEGMENT_AMBIGUITY"
    assert cue_off.decision == "COLLISION"
    assert cue_off.match_count == projection.FILLER_LANES
    assert cue_off.successor.checkpoint_sha256 == trained.checkpoint_sha256

    physical_fields = tuple(
        field.name for field in dataclasses.fields(segmenter.PhysicalTimedRelationEvent)
    )
    assert physical_fields == (
        "input_frame",
        "relation_cells",
        "teacher_output_cells",
        "interval_cell_id",
        "teacher_gate",
        "cell_gate",
    )
    former_labels_a = (b"context/a", b"source/a", b"latent/boundary")
    former_labels_b = (b"context/z", b"source/z", b"latent/continue")
    assert former_labels_a != former_labels_b
    label_blind_a = segmenter.advance_predictive_segmenter_v1(
        trained,
        dataclasses.replace(identity_events[1], interval_cell_id=interval_a),
    )
    label_blind_b = segmenter.advance_predictive_segmenter_v1(
        trained,
        dataclasses.replace(identity_events[1], interval_cell_id=interval_a),
    )
    assert label_blind_a.status == "SAME_SEGMENT_CONTINUE"
    assert label_blind_a.successor.checkpoint_sha256 == (
        label_blind_b.successor.checkpoint_sha256
    )
    assert segmenter.CONFIG_RECORD["latent_boundary_identifiability"] == (
        "identical-physical-event-consequence-and-interval-streams-have-"
        "identical-state-transitions"
    )

    shuffled_events = (
        dataclasses.replace(identity_events[0], interval_cell_id=interval_a),
        dataclasses.replace(identity_events[1], interval_cell_id=interval_b),
        *identity_events[2:],
    )
    shuffled, shuffled_steps = _feed(trained, shuffled_events)
    assert shuffled_steps[0].status == "SAME_SEGMENT_CONTINUE"
    assert shuffled_steps[1].status == "NEW_SEGMENT_PENDING"
    assert shuffled.inner.active
    assert shuffled.segment_count == 4
    assert len(shuffled.internal_provenance_cell_ids) == 3
    assert not shuffled.complete_hold

    learned_boundary = segmenter.advance_predictive_segmenter_v1(
        trained,
        identity_events[0],
    )
    assert learned_boundary.status == "NEW_SEGMENT_PENDING"
    assert learned_boundary.decision == "NEW"
    assert learned_boundary.decision_authority == "LEARNED_CUE"
    assert learned_boundary.match_count == projection.FILLER_LANES
    assert learned_boundary.mismatch_count == 0
    assert learned_boundary.provenance_cell_id not in (
        trained.internal_provenance_cell_ids
    )
    assert learned_boundary.successor.reserved_provenance_cell_id == (
        learned_boundary.provenance_cell_id
    )

    premature = segmenter.advance_predictive_segmenter_v1(
        learned_boundary.successor,
        dataclasses.replace(identity_events[1], interval_cell_id=interval_b),
    )
    assert premature.status == "STOP_PREMATURE_NEW"
    assert premature.successor.checkpoint_sha256 == (
        learned_boundary.successor.checkpoint_sha256
    )

    completed, remaining_steps = _feed(
        learned_boundary.successor,
        identity_events[1:],
    )
    assert remaining_steps[-1].status == "SEGMENT_COMPLETE"
    assert completed.segment_count == segmenter.PROVENANCE_CAPACITY
    assert len(completed.internal_provenance_cell_ids) == 4
    assert completed.internal_provenance_cell_ids[-1] == (
        learned_boundary.provenance_cell_id
    )
    assert completed.internal_provenance_cell_ids[:3] == (
        trained.internal_provenance_cell_ids
    )
    capacity = segmenter.advance_predictive_segmenter_v1(
        completed,
        identity_events[0],
    )
    assert capacity.status == "STOP_PROVENANCE_CAPACITY"
    assert capacity.successor.checkpoint_sha256 == completed.checkpoint_sha256

    swapped_initial = segmenter.start_predictive_segmenter_v1(
        eventwise.start_eventwise_common_one_byte_relation_v1(parent)
    )
    swapped_calibration = _segment_events(relation, interval_b, interval_b)
    swapped_shift = _segment_events(
        relation,
        interval_a,
        interval_b,
        shifted=True,
    )
    swapped_identity = _segment_events(relation, interval_a, interval_b)
    swapped, _ = _feed(swapped_initial, swapped_calibration)
    swapped, _ = _feed(swapped, swapped_shift)
    swapped, _ = _feed(swapped, swapped_identity)
    assert swapped.cue_excitatory == ((0, 2), (4, 0))
    assert swapped.cue_inhibitory == ((2, 0), (0, 4))
    refitted_boundary = segmenter.advance_predictive_segmenter_v1(
        swapped,
        swapped_identity[0],
    )
    assert refitted_boundary.status == "NEW_SEGMENT_PENDING"
    assert refitted_boundary.decision_authority == "LEARNED_CUE"
    assert refitted_boundary.mismatch_count == 0

    assert completed.v2_parent.training_summary.provenance[:2] == parent_provenance
    assert completed.v2_parent.relation_state_sha256(unrelated_relation) == (
        unrelated_root
    )
    assert parent.checkpoint_sha256 == parent_root
    assert tuple(
        array.tobytes(order="C") for _, array in parent._checkpoint_arrays()
    ) == parent_arrays
    assert parent.training_summary.provenance == parent_provenance

    assert segmenter.NEW_CELL_COUNT == 15
    assert segmenter.TOTAL_CELL_COUNT == eventwise.TOTAL_CELL_COUNT + 15 == 65_757
    assert segmenter.COMPARATOR_SYNAPSE_CHANNEL_COUNT == 64
    assert segmenter.CUE_SYNAPSE_CHANNEL_COUNT == 4
    assert segmenter.CUE_EH_STATE_BYTES == 8
    assert segmenter.ADDED_PERSISTENT_STATE_BYTES == 50
    assert segmenter.PERSISTENT_STATE_BYTES == 276
    assert segmenter.SEGMENTATION_CELL_STEPS_PER_EVENT == 9_000
    assert segmenter.ROUTED_EVENT_CELL_STEPS == 37_200
    assert (
        segmenter.EXTERNAL_CONTEXT_SOURCE_INPUT_COUNT,
        segmenter.BOUNDARY_BIT_INPUT_COUNT,
        segmenter.RAW_ELAPSED_TIME_INPUT_COUNT,
        segmenter.PARENT_LESSON_REPLAY_COUNT,
        segmenter.CORPUS_INPUT_COUNT,
        segmenter.PATH2_INPUT_COUNT,
        segmenter.DEV_INPUT_COUNT,
        segmenter.M0_INPUT_COUNT,
        segmenter.PROTECTED_INPUT_COUNT,
        segmenter.G0_INPUT_COUNT,
        segmenter.CODING_INPUT_COUNT,
    ) == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert sha256(Path(segmenter.__file__).read_bytes()).hexdigest() == (
        segmenter.SOURCE_SHA256
    )
    assert len(segmenter.CONFIG_SHA256) == len(completed.checkpoint_sha256) == 64
    assert len(remaining_steps[-1].sha256) == 64
