from __future__ import annotations

import dataclasses
from collections import Counter
from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as projection
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as copy_v2

_CURRICULUM_VALUES = (0x1E, 0x63, 0xB4, 0xC9, 0x27, 0x78, 0x8D, 0xD2)
_HELDOUT_VALUE = 0x1B

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
    return _value(tuple((digit + 1) % projection.FILLER_RADIX for digit in _digits(value)))


def _teacher_cells(value: int) -> tuple[int, ...]:
    rows = tuple(
        tuple(24 if candidate == digit else 0 for candidate in range(4))
        for digit in _digits(value)
    )
    decision = _output_cells.race_output(rows)
    assert decision.decision == "TOKEN"
    return decision.active_cells


def _opaque(label: bytes) -> tuple[int, ...]:
    return _output_cells.project(label).trace_cells


def _context_lessons(
    relation_cells: tuple[int, ...],
    context_cells: tuple[int, ...],
    source_cells: tuple[int, ...],
    *,
    shifted: bool = False,
    teacher_gate: bool = True,
    cell_gate: bool = True,
) -> tuple[copy_v2.RelationGatedOneByteLesson, ...]:
    return tuple(
        copy_v2.RelationGatedOneByteLesson(
            input_frame=_input_cells.project_one_byte(bytes((value,))),
            relation_cells=relation_cells,
            context_cells=context_cells,
            source_cells=source_cells,
            teacher_output_cells=_teacher_cells(
                _shift_fillers(value) if shifted else value
            ),
            teacher_gate=teacher_gate,
            cell_gate=cell_gate,
        )
        for value in _CURRICULUM_VALUES
    )


def test_v2_learns_and_revises_physical_copy_routes_across_contexts() -> None:
    relation = _opaque(b"relation/shared-transform")
    unrelated_relation = _opaque(b"relation/unrelated-transform")
    unknown_relation = _opaque(b"relation/relabel-control")
    context_a = _opaque(b"context/a")
    context_b = _opaque(b"context/b")
    context_c = _opaque(b"context/c")
    context_d = _opaque(b"context/d")
    source_a = _opaque(b"source/a")
    source_b = _opaque(b"source/b")
    source_c = _opaque(b"source/c")
    source_d = _opaque(b"source/d")
    dependent_source = _opaque(b"source/dependent")
    unrelated_context_a = _opaque(b"context/unrelated-a")
    unrelated_context_b = _opaque(b"context/unrelated-b")
    unrelated_source_a = _opaque(b"source/unrelated-a")
    unrelated_source_b = _opaque(b"source/unrelated-b")

    copy_a = _context_lessons(relation, context_a, source_a)
    copy_b = _context_lessons(relation, context_b, source_b)
    shifted_c = _context_lessons(
        relation,
        context_c,
        source_c,
        shifted=True,
    )
    shifted_d = _context_lessons(
        relation,
        context_d,
        source_d,
        shifted=True,
    )
    dependent = _context_lessons(relation, context_a, dependent_source)
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
    teacher_off_lessons = _context_lessons(
        relation,
        context_a,
        source_a,
        teacher_gate=False,
    )
    cell_off_lessons = _context_lessons(
        relation,
        context_a,
        source_a,
        cell_gate=False,
    )

    cover_frames = tuple(
        _input_cells.project_one_byte(bytes((value,)))
        for value in _CURRICULUM_VALUES
    )
    activations = Counter(
        cell for frame in cover_frames for cell in frame.filler_cells
    )
    assert len(activations) == projection.FILLER_CELL_COUNT == 16
    assert set(activations.values()) == {2}
    heldout = _input_cells.project_one_byte(bytes((_HELDOUT_VALUE,)))
    assert heldout.filler_cells not in {frame.filler_cells for frame in cover_frames}
    assert heldout.opaque_frame.sha256 not in {
        frame.opaque_frame.sha256 for frame in cover_frames
    }

    one_context = copy_v2.fit_common_one_byte_copy_relation_v2(copy_a)
    ready = copy_v2.fit_common_one_byte_copy_relation_v2(
        copy_a + copy_b + dependent + unrelated_a + unrelated_b
    )
    parent_checkpoint_before_extension = ready.checkpoint_sha256
    parent_arrays_before_extension = tuple(
        array.tobytes(order="C") for _, array in ready._checkpoint_arrays()
    )
    parent_provenance = ready.training_summary.provenance
    conflicted = copy_v2.extend_common_one_byte_copy_relation_v2(
        ready,
        shifted_c,
    )
    revised = copy_v2.extend_common_one_byte_copy_relation_v2(
        conflicted,
        shifted_d,
    )
    no_op_extension = copy_v2.extend_common_one_byte_copy_relation_v2(
        ready,
        copy_a + dependent + teacher_off_lessons + cell_off_lessons,
    )
    teacher_off = copy_v2.fit_common_one_byte_copy_relation_v2(
        teacher_off_lessons
    )
    cell_off = copy_v2.fit_common_one_byte_copy_relation_v2(cell_off_lessons)

    one_recall = one_context.recall(heldout, relation)
    assert one_recall.status == "ABSTAIN_LOW_SUPPORT"
    assert one_recall.output.decision == "STOP"
    assert one_context.relation_support_count(relation) == 1

    checkpoint_before_controls = ready.checkpoint_sha256
    copy_recall = ready.recall(heldout, relation)
    assert copy_recall.status == "READY"
    assert copy_recall.output.active_cells == _teacher_cells(_HELDOUT_VALUE)
    assert copy_recall.input_filler_cells == heldout.filler_cells
    assert copy_recall.relation_support_count == 2
    assert copy_recall.provenance_context_count == 2

    conflict_recall = conflicted.recall(heldout, relation)
    assert conflict_recall.status == "ABSTAIN_CONFLICT"
    assert conflict_recall.output.decision == "STOP"
    revised_recall = revised.recall(heldout, relation)
    assert revised_recall.status == "READY"
    assert revised_recall.output.active_cells == _teacher_cells(
        _shift_fillers(_HELDOUT_VALUE)
    )

    first_extension = conflicted.extension_summary
    second_extension = revised.extension_summary
    no_op_summary = no_op_extension.extension_summary
    assert first_extension is not None
    assert second_extension is not None
    assert no_op_summary is not None
    assert first_extension.parent_checkpoint_sha256 == ready.checkpoint_sha256
    assert first_extension.new_lesson_count == 8
    assert first_extension.accepted_new_context_count == 1
    assert first_extension.physical_observation_ticks == 32
    assert first_extension.committed_local_update_ticks == 32
    assert first_extension.relation_excitatory_update_ticks == 12
    assert first_extension.route_excitatory_update_ticks == 16
    assert first_extension.route_inhibitory_update_ticks == 16
    assert first_extension.committed_update_ticks == 76
    assert first_extension.parent_provenance_count == 4
    assert first_extension.retained_parent_provenance_count == 4
    assert first_extension.total_provenance_count == 5
    assert second_extension.parent_checkpoint_sha256 == conflicted.checkpoint_sha256
    assert second_extension.accepted_new_context_count == 1
    assert second_extension.committed_update_ticks == 76
    assert second_extension.total_provenance_count == 6
    assert conflicted.training_summary.provenance[:4] == parent_provenance
    assert revised.training_summary.provenance[:4] == parent_provenance

    assert no_op_summary.new_lesson_count == 32
    assert no_op_summary.active_context_group_count == 2
    assert no_op_summary.accepted_new_context_count == 0
    assert no_op_summary.duplicate_context_count == 1
    assert no_op_summary.dependent_context_count == 1
    assert no_op_summary.teacher_off_lesson_count == 8
    assert no_op_summary.cell_off_lesson_count == 8
    assert no_op_summary.physical_observation_ticks == 64
    assert no_op_summary.committed_update_ticks == 0
    assert no_op_summary.provenance_added == ()
    assert no_op_extension.training_summary.provenance == parent_provenance

    assert teacher_off.recall(heldout, relation).status == "ABSTAIN_UNKNOWN_RELATION"
    assert cell_off.recall(heldout, relation).status == "ABSTAIN_UNKNOWN_RELATION"
    assert ready.recall(heldout, relation, cell_gate=False).status == "STOP_CELL_OFF"
    assert (
        ready.recall(heldout, relation, relation_gate=False).status
        == "STOP_RELATION_OFF"
    )
    assert ready.recall(heldout, unknown_relation).status == "ABSTAIN_UNKNOWN_RELATION"

    permuted = _input_cells.project_one_byte(bytes((projection.LANE_PERMUTATION_CONTROL_VALUE,)))
    forged_lane_order = dataclasses.replace(
        heldout,
        filler_cells=permuted.filler_cells,
    )
    assert (
        ready.recall(forged_lane_order, relation).status
        == "STOP_INPUT_AUTHENTICATION"
    )

    unrelated_root = ready.relation_state_sha256(unrelated_relation)
    assert unrelated_root is not None
    assert conflicted.relation_state_sha256(unrelated_relation) == unrelated_root
    assert revised.relation_state_sha256(unrelated_relation) == unrelated_root
    assert no_op_extension.relation_state_sha256(unrelated_relation) == unrelated_root
    unrelated_recall = ready.recall(heldout, unrelated_relation)
    assert unrelated_recall.status == "READY"
    assert unrelated_recall.output.active_cells == _teacher_cells(_HELDOUT_VALUE)
    assert ready.relation_state_sha256(unrelated_relation) == unrelated_root
    assert ready.checkpoint_sha256 == checkpoint_before_controls
    assert ready.checkpoint_sha256 == parent_checkpoint_before_extension
    assert tuple(
        array.tobytes(order="C") for _, array in ready._checkpoint_arrays()
    ) == parent_arrays_before_extension
    assert ready.extension_summary is None

    summary = ready.training_summary
    assert summary.independent_context_count == 4
    assert summary.dependent_context_count == 1
    assert summary.relation_count == 2
    assert len(summary.provenance) == 4
    assert summary.maximum_excitatory_tick == 2
    assert summary.maximum_inhibitory_tick == 0
    assert summary.synaptic_state_bytes == 1_120
    assert summary.checkpoint_array_bytes == 8_744
    assert conflicted.training_summary.route_inhibitory_update_ticks == 16
    assert revised.training_summary.route_inhibitory_update_ticks == 32
    assert teacher_off.training_summary.local_physical_update_ticks == 0
    assert cell_off.training_summary.local_physical_update_ticks == 0

    assert copy_v2.NEW_CELL_COUNT == 73
    assert copy_v2.TOTAL_CELL_COUNT == projection.TOTAL_CELL_COUNT + 73 == 65_642
    assert copy_v2.ROUTE_SYNAPSE_CHANNEL_COUNT == 512
    assert copy_v2.FUSED_ROUTE_SIMULATED_CELL_COUNT == 0
    assert copy_v2.ROUTE_EH_STATE_BYTES == 1_024
    assert copy_v2.REUSED_OUTPUT_TONIC_CELL_ID == output_projection.STOP_CELL_ID
    assert (
        copy_v2.PRIOR_MEMORY_MUTATION_COUNT,
        copy_v2.PRIOR_SCHEMA_MUTATION_COUNT,
        copy_v2.CORPUS_INPUT_COUNT,
        copy_v2.PATH2_INPUT_COUNT,
        copy_v2.DEV_INPUT_COUNT,
        copy_v2.M0_INPUT_COUNT,
        copy_v2.PROTECTED_INPUT_COUNT,
        copy_v2.G0_INPUT_COUNT,
        copy_v2.CODING_INPUT_COUNT,
    ) == (0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert copy_v2.CONFIG_RECORD["projection_substrate_root_sha256"] == (
        projection.SUBSTRATE_ROOT_SHA256
    )
    assert sha256(Path(copy_v2.__file__).read_bytes()).hexdigest() == (
        copy_v2.SOURCE_SHA256
    )
    assert len(copy_v2.CONFIG_SHA256) == len(ready.checkpoint_sha256) == 64
