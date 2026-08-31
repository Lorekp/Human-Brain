from __future__ import annotations

import dataclasses
from collections import Counter
from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_episodic_transform_consolidation_v2 as memory
from experiments.r3_pilot import common_event_associative_memory_v2 as exact_memory
from experiments.r3_pilot import common_one_byte_copy_relation_v2 as transforms

_COVER_VALUES = (0x00, 0x4A, 0x94, 0xDE, 0x25, 0x6B, 0xBD, 0xF3)
_HELDOUT_VALUE = 0xE4

_input_cells = filler_projection.CommonOneByteFillerCells()
_output_cells = output_projection.CommonEventCells()


def _digits(value: int) -> tuple[int, ...]:
    return tuple(
        (
            value
            // (
                filler_projection.FILLER_RADIX
                ** (filler_projection.FILLER_LANES - lane - 1)
            )
        )
        % filler_projection.FILLER_RADIX
        for lane in range(filler_projection.FILLER_LANES)
    )


def _value(digits: tuple[int, ...]) -> int:
    result = 0
    for digit in digits:
        result = result * filler_projection.FILLER_RADIX + digit
    return result


def _shift(value: int) -> int:
    return _value(
        tuple(
            (digit + 1) % filler_projection.FILLER_RADIX
            for digit in _digits(value)
        )
    )


def _teacher_cells(value: int) -> tuple[int, ...]:
    scores = tuple(
        tuple(24 if candidate == digit else 0 for candidate in range(4))
        for digit in _digits(value)
    )
    output = _output_cells.race_output(scores)
    assert output.decision == "TOKEN"
    return output.active_cells


def _opaque(event: bytes) -> tuple[int, ...]:
    return _output_cells.project(event).trace_cells


def _context_views(
    *,
    relation_event: bytes,
    context_event: bytes,
    source_event: bytes,
    shifted: bool = False,
    teacher_gate: bool = True,
    cell_gate: bool = True,
) -> tuple[
    tuple[associative.CommonAssociationLesson, ...],
    tuple[exact_memory.PhysicalCommonAssociationLessonV2, ...],
    tuple[transforms.RelationGatedOneByteLesson, ...],
]:
    initial_exact: list[associative.CommonAssociationLesson] = []
    new_exact: list[exact_memory.PhysicalCommonAssociationLessonV2] = []
    semantic: list[transforms.RelationGatedOneByteLesson] = []
    relation_cells = _opaque(relation_event)
    context_cells = _opaque(context_event)
    source_cells = _opaque(source_event)
    for value in _COVER_VALUES:
        consequence = _shift(value) if shifted else value
        observation_events = (bytes((value,)), relation_event, source_event)
        output_cells = _teacher_cells(consequence)
        initial_exact.append(
            associative.CommonAssociationLesson(
                observation_events=observation_events,
                consequence_byte=bytes((consequence,)),
                teacher_gate=teacher_gate,
                cell_gate=cell_gate,
            )
        )
        new_exact.append(
            exact_memory.PhysicalCommonAssociationLessonV2(
                observation_events=observation_events,
                teacher_output_cells=output_cells,
                teacher_gate=teacher_gate,
                cell_gate=cell_gate,
            )
        )
        semantic.append(
            transforms.RelationGatedOneByteLesson(
                input_frame=_input_cells.project_one_byte(bytes((value,))),
                relation_cells=relation_cells,
                context_cells=context_cells,
                source_cells=source_cells,
                teacher_output_cells=output_cells,
                teacher_gate=teacher_gate,
                cell_gate=cell_gate,
            )
        )
    return tuple(initial_exact), tuple(new_exact), tuple(semantic)


def _assert_context_recalled(
    state: memory.FrozenCommonEpisodicTransformMemoryV2,
    *,
    relation_event: bytes,
    source_event: bytes,
    shifted: bool,
) -> None:
    for value in _COVER_VALUES:
        recall = state.recall_episode(
            (bytes((value,)), relation_event, source_event)
        )
        assert recall.status == "RECALLED"
        assert recall.output.token == (_shift(value) if shifted else value)


def test_replay_free_exact_delta_and_transform_consolidation_revise_together() -> None:
    relation_event = b"opaque/relation/main"
    unrelated_relation_event = b"opaque/relation/unrelated"
    relation_cells = _opaque(relation_event)
    unrelated_relation_cells = _opaque(unrelated_relation_event)

    legacy_a, delta_a, semantic_a = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/a",
        source_event=b"opaque/source/a",
    )
    legacy_b, _delta_b, semantic_b = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/b",
        source_event=b"opaque/source/b",
    )
    _legacy_c, delta_c, semantic_c = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/c",
        source_event=b"opaque/source/c",
        shifted=True,
    )
    _legacy_d, delta_d, semantic_d = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/d",
        source_event=b"opaque/source/d",
        shifted=True,
    )
    unrelated_legacy_a, _unrelated_delta_a, unrelated_semantic_a = _context_views(
        relation_event=unrelated_relation_event,
        context_event=b"opaque/context/unrelated-a",
        source_event=b"opaque/source/unrelated-a",
    )
    unrelated_legacy_b, _unrelated_delta_b, unrelated_semantic_b = _context_views(
        relation_event=unrelated_relation_event,
        context_event=b"opaque/context/unrelated-b",
        source_event=b"opaque/source/unrelated-b",
    )

    cover_frames = tuple(
        _input_cells.project_one_byte(bytes((value,))) for value in _COVER_VALUES
    )
    activations = Counter(
        cell for frame in cover_frames for cell in frame.filler_cells
    )
    assert len(activations) == filler_projection.FILLER_CELL_COUNT == 16
    assert set(activations.values()) == {2}
    heldout = _input_cells.project_one_byte(bytes((_HELDOUT_VALUE,)))
    assert heldout.filler_cells not in {frame.filler_cells for frame in cover_frames}

    initial_exact = (
        legacy_a + legacy_b + unrelated_legacy_a + unrelated_legacy_b
    )
    initial_semantic = (
        semantic_a + semantic_b + unrelated_semantic_a + unrelated_semantic_b
    )
    ready = memory.fit_common_episodic_transform_consolidation_v2(
        initial_exact,
        initial_semantic,
    )
    ready_root = ready.checkpoint_sha256
    ready_exact_base_root = ready.episodic_base_checkpoint_sha256
    ready_exact_root = ready.episodic_checkpoint_sha256
    ready_semantic_root = ready.semantic_checkpoint_sha256
    unrelated_root = ready.relation_state_sha256(unrelated_relation_cells)
    assert unrelated_root is not None
    assert ready.recall_semantic(heldout, relation_cells).status == "READY"
    assert ready.recall_semantic(heldout, relation_cells).output.token == (
        _HELDOUT_VALUE
    )
    assert ready.training_summary.initial_exact_v1_lesson_count == 32
    assert ready.training_summary.exact_delta_episode_count == 0
    assert ready.training_summary.exact_delta_checkpoint_array_bytes == 0
    assert ready.training_summary.semantic_synaptic_state_bytes == 1_120
    assert ready.training_summary.semantic_checkpoint_array_bytes == 8_744

    conflicted = memory.advance_common_episodic_transform_consolidation_v2(
        ready,
        delta_c,
        semantic_c,
    )
    conflicted_root = conflicted.checkpoint_sha256
    conflicted_exact_root = conflicted.episodic_checkpoint_sha256
    conflicted_semantic_root = conflicted.semantic_checkpoint_sha256
    conflict = conflicted.recall_semantic(heldout, relation_cells)
    assert conflict.status == "ABSTAIN_CONFLICT"
    assert conflict.output.decision == "STOP"
    for source, shifted in (
        (b"opaque/source/a", False),
        (b"opaque/source/b", False),
        (b"opaque/source/c", True),
    ):
        _assert_context_recalled(
            conflicted,
            relation_event=relation_event,
            source_event=source,
            shifted=shifted,
        )
    conflict_summary = conflicted.training_summary
    assert conflict_summary.parent_composite_checkpoint_sha256 == ready_root
    assert conflict_summary.exact_new_lesson_count == 8
    assert conflict_summary.exact_accepted_delta_row_count == 8
    assert conflict_summary.exact_rejected_lesson_count == 0
    assert conflict_summary.exact_parent_lesson_replay_count == 0
    assert conflict_summary.exact_delta_episode_count == 8
    assert conflict_summary.exact_delta_array_bytes_per_row == 96
    assert conflict_summary.exact_delta_array_bytes_added == 8 * 96
    assert conflict_summary.exact_delta_checkpoint_array_bytes == 8 * 96
    assert conflict_summary.semantic_new_lesson_count == 8
    assert conflict_summary.semantic_parent_lesson_replay_count == 0
    assert conflict_summary.semantic_accepted_new_context_count == 1
    assert conflict_summary.semantic_committed_update_ticks == 76
    assert conflict_summary.semantic_synaptic_state_bytes == 1_120
    assert conflict_summary.semantic_checkpoint_array_bytes == 8_744

    revised = memory.advance_common_episodic_transform_consolidation_v2(
        conflicted,
        delta_d,
        semantic_d,
    )
    shifted = revised.recall_semantic(heldout, relation_cells)
    assert shifted.status == "READY"
    assert shifted.output.token == _shift(_HELDOUT_VALUE)
    for source, is_shifted in (
        (b"opaque/source/a", False),
        (b"opaque/source/b", False),
        (b"opaque/source/c", True),
        (b"opaque/source/d", True),
    ):
        _assert_context_recalled(
            revised,
            relation_event=relation_event,
            source_event=source,
            shifted=is_shifted,
        )
    revised_summary = revised.training_summary
    assert revised_summary.parent_composite_checkpoint_sha256 == conflicted_root
    assert revised_summary.exact_accepted_delta_row_count == 8
    assert revised_summary.exact_delta_array_bytes_added == 8 * 96
    assert revised_summary.exact_delta_episode_count == 16
    assert revised_summary.exact_delta_checkpoint_array_bytes == 16 * 96
    assert revised_summary.semantic_accepted_new_context_count == 1
    assert revised_summary.semantic_committed_update_ticks == 76
    assert revised_summary.independent_context_count == 6
    assert revised_summary.relation_count == 2
    assert revised_summary.provenance_count == 6

    # Both parent composites and the unrelated relation remain exact and immutable.
    assert ready.checkpoint_sha256 == ready_root
    assert ready.episodic_base_checkpoint_sha256 == ready_exact_base_root
    assert ready.episodic_checkpoint_sha256 == ready_exact_root
    assert ready.semantic_checkpoint_sha256 == ready_semantic_root
    assert conflicted.checkpoint_sha256 == conflicted_root
    assert conflicted.episodic_checkpoint_sha256 == conflicted_exact_root
    assert conflicted.semantic_checkpoint_sha256 == conflicted_semantic_root
    assert conflicted.relation_state_sha256(unrelated_relation_cells) == unrelated_root
    assert revised.relation_state_sha256(unrelated_relation_cells) == unrelated_root
    assert revised.recall_semantic(
        heldout,
        unrelated_relation_cells,
    ).output.token == _HELDOUT_VALUE
    assert revised_summary.episodic_checkpoint_sha256 == (
        revised.episodic_checkpoint_sha256
    )
    assert revised_summary.semantic_checkpoint_sha256 == (
        revised.semantic_checkpoint_sha256
    )

    deranged = revised.recall_episode(
        (
            bytes((_COVER_VALUES[2],)),
            relation_event,
            b"opaque/source/deranged",
        )
    )
    assert deranged.status == "STOP_UNKNOWN"

    # Exact teacher/cell gates reject new rows without changing either child.
    _, teacher_off_delta, _ = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/teacher-off",
        source_event=b"opaque/source/teacher-off",
        teacher_gate=False,
    )
    _, cell_off_delta, _ = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/cell-off",
        source_event=b"opaque/source/cell-off",
        cell_gate=False,
    )
    teacher_off = memory.advance_common_episodic_transform_consolidation_v2(
        revised,
        teacher_off_delta,
        (),
    )
    cell_off = memory.advance_common_episodic_transform_consolidation_v2(
        revised,
        cell_off_delta,
        (),
    )
    for stopped in (teacher_off, cell_off):
        summary = stopped.training_summary
        assert summary.exact_new_lesson_count == 8
        assert summary.exact_accepted_delta_row_count == 0
        assert summary.exact_rejected_lesson_count == 8
        assert summary.exact_delta_array_bytes_added == 0
        assert summary.semantic_new_lesson_count == 0
        assert summary.semantic_accepted_new_context_count == 0
        assert summary.semantic_committed_update_ticks == 0
        assert stopped.episodic_checkpoint_sha256 == (
            revised.episodic_checkpoint_sha256
        )
        assert stopped.semantic_checkpoint_sha256 == (
            revised.semantic_checkpoint_sha256
        )
    assert teacher_off.recall_episode(
        (bytes((_COVER_VALUES[0],)), relation_event, b"opaque/source/teacher-off")
    ).status == "STOP_UNKNOWN"
    assert cell_off.recall_episode(
        (bytes((_COVER_VALUES[0],)), relation_event, b"opaque/source/cell-off")
    ).status == "STOP_UNKNOWN"

    # Duplicate and source-dependent semantic contexts add no semantic tick.
    _, _, dependent_semantic = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/a",
        source_event=b"opaque/source/dependent",
    )
    semantic_no_op = memory.advance_common_episodic_transform_consolidation_v2(
        ready,
        (),
        semantic_a + dependent_semantic,
    )
    no_op_extension = semantic_no_op.semantic_extension_summary
    assert no_op_extension is not None
    assert no_op_extension.accepted_new_context_count == 0
    assert no_op_extension.duplicate_context_count == 1
    assert no_op_extension.dependent_context_count == 1
    assert no_op_extension.committed_update_ticks == 0
    assert semantic_no_op.training_summary.exact_accepted_delta_row_count == 0
    assert semantic_no_op.episodic_checkpoint_sha256 == ready_exact_root
    assert semantic_no_op.relation_state_sha256(relation_cells) == (
        ready.relation_state_sha256(relation_cells)
    )

    # Semantic gates do not affect the exact child or its source-conditioned recall.
    exact_before_gates = revised.recall_episode(delta_c[0].observation_events)
    assert revised.recall_semantic(
        heldout,
        relation_cells,
        cell_gate=False,
    ).status == "STOP_CELL_OFF"
    assert revised.recall_semantic(
        heldout,
        relation_cells,
        relation_gate=False,
    ).status == "STOP_RELATION_OFF"
    exact_after_gates = revised.recall_episode(delta_c[0].observation_events)
    assert exact_before_gates.sha256 == exact_after_gates.sha256
    assert exact_after_gates.status == "RECALLED"

    permuted = _input_cells.project_one_byte(
        bytes((filler_projection.LANE_PERMUTATION_CONTROL_VALUE,))
    )
    forged_lane_order = dataclasses.replace(
        heldout,
        filler_cells=permuted.filler_cells,
    )
    assert revised.recall_semantic(
        forged_lane_order,
        relation_cells,
    ).status == "STOP_INPUT_AUTHENTICATION"

    assert memory.EXACT_DELTA_ARRAY_BYTES_PER_ROW == 96
    assert revised_summary.inherited_exact_v1_checkpoint_array_bytes == 1_572_864
    assert revised_summary.structural_route_choice_count == 16
    assert revised_summary.whole_byte_mapping_count == 256
    assert memory.CONFIG_RECORD["retained_v2_parent_count"] == 0
    assert memory.CONFIG_RECORD["composite_parent_object_count"] == 0
    assert memory.CONFIG_RECORD["exact_old_lesson_replay_count"] == 0
    assert memory.CONFIG_RECORD["semantic_old_lesson_replay_count"] == 0
    assert memory.CONFIG_RECORD["host_output_selection_count"] == 0
    assert memory.CONFIG_RECORD["semantic_decode_count"] == 0
    assert memory.CONFIG_RECORD["combined_compression_claim"] is False
    assert (
        memory.GENERATOR_EXECUTION_COUNT,
        memory.CORPUS_INPUT_COUNT,
        memory.PATH2_INPUT_COUNT,
        memory.DEV_INPUT_COUNT,
        memory.M0_INPUT_COUNT,
        memory.PROTECTED_INPUT_COUNT,
        memory.G0_INPUT_COUNT,
        memory.CODING_INPUT_COUNT,
    ) == (0, 0, 0, 0, 0, 0, 0, 0)
    assert "_parent" not in memory.FrozenCommonEpisodicTransformMemoryV2.__slots__
    source_path = Path(memory.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    assert "cumulative_episodic_lessons" not in source_text
    assert "total_checkpoint_array_bytes" not in source_text
    assert "compression_ratio" not in source_text
    assert sha256(source_path.read_bytes()).hexdigest() == memory.SOURCE_SHA256
    assert len(memory.CONFIG_SHA256) == len(revised.checkpoint_sha256) == 64

    # A is intentionally used only in the initial base; it is never replayed.
    assert len(delta_a) == 8
