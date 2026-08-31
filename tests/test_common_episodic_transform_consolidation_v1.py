from __future__ import annotations

import dataclasses
from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as output_projection
from experiments.g0 import common_event_cells_v2 as filler_projection
from experiments.r3_pilot import common_episodic_transform_consolidation_v1 as memory
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
    tuple[transforms.RelationGatedOneByteLesson, ...],
]:
    episodic: list[associative.CommonAssociationLesson] = []
    semantic: list[transforms.RelationGatedOneByteLesson] = []
    relation_cells = _opaque(relation_event)
    context_cells = _opaque(context_event)
    source_cells = _opaque(source_event)
    for value in _COVER_VALUES:
        consequence = _shift(value) if shifted else value
        episodic.append(
            associative.CommonAssociationLesson(
                observation_events=(
                    bytes((value,)),
                    relation_event,
                    source_event,
                ),
                consequence_byte=bytes((consequence,)),
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
                teacher_output_cells=_teacher_cells(consequence),
                teacher_gate=teacher_gate,
                cell_gate=cell_gate,
            )
        )
    return tuple(episodic), tuple(semantic)


def test_exact_episodes_and_relation_transform_consolidate_and_revise() -> None:
    relation_event = b"opaque/relation/main"
    unrelated_relation_event = b"opaque/relation/unrelated"
    relation_cells = _opaque(relation_event)
    unrelated_relation_cells = _opaque(unrelated_relation_event)

    episodes_a, semantic_a = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/a",
        source_event=b"opaque/source/a",
    )
    episodes_b, semantic_b = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/b",
        source_event=b"opaque/source/b",
    )
    episodes_c, semantic_c = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/c",
        source_event=b"opaque/source/c",
        shifted=True,
    )
    episodes_d, semantic_d = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/d",
        source_event=b"opaque/source/d",
        shifted=True,
    )
    unrelated_episodes_a, unrelated_semantic_a = _context_views(
        relation_event=unrelated_relation_event,
        context_event=b"opaque/context/unrelated-a",
        source_event=b"opaque/source/unrelated-a",
    )
    unrelated_episodes_b, unrelated_semantic_b = _context_views(
        relation_event=unrelated_relation_event,
        context_event=b"opaque/context/unrelated-b",
        source_event=b"opaque/source/unrelated-b",
    )

    heldout = _input_cells.project_one_byte(bytes((_HELDOUT_VALUE,)))
    cover_frames = tuple(
        _input_cells.project_one_byte(bytes((value,))) for value in _COVER_VALUES
    )
    assert heldout.filler_cells not in {frame.filler_cells for frame in cover_frames}
    assert {
        filler
        for frame in cover_frames
        for filler in frame.filler_cells
    } == {
        filler_projection.FILLER_CELL_OFFSET + index
        for index in range(filler_projection.FILLER_CELL_COUNT)
    }

    one_context = memory.fit_common_episodic_transform_consolidation_v1(
        episodes_a,
        semantic_a,
    )
    assert one_context.recall_episode(
        episodes_a[1].observation_events
    ).output.token == episodes_a[1].consequence_byte[0]
    assert one_context.recall_semantic(heldout, relation_cells).status == (
        "ABSTAIN_LOW_SUPPORT"
    )
    assert one_context.recall_episode(
        (bytes((_HELDOUT_VALUE,)), relation_event, b"opaque/source/a")
    ).status == "STOP_UNKNOWN"

    ready_episodes = (
        episodes_a
        + episodes_b
        + unrelated_episodes_a
        + unrelated_episodes_b
    )
    ready_semantic = (
        semantic_a
        + semantic_b
        + unrelated_semantic_a
        + unrelated_semantic_b
    )
    ready = memory.fit_common_episodic_transform_consolidation_v1(
        ready_episodes,
        ready_semantic,
    )
    ready_checkpoint = ready.checkpoint_sha256
    identity = ready.recall_semantic(heldout, relation_cells)
    assert identity.status == "READY"
    assert identity.output.token == _HELDOUT_VALUE
    assert ready.recall_episode(
        (bytes((_HELDOUT_VALUE,)), relation_event, b"opaque/source/a")
    ).status == "STOP_UNKNOWN"

    unrelated_root = ready.relation_state_sha256(unrelated_relation_cells)
    assert unrelated_root is not None
    conflicted_episodes = ready_episodes + episodes_c
    conflicted = memory.extend_common_episodic_transform_consolidation_v1(
        ready,
        conflicted_episodes,
        semantic_c,
    )
    conflict = conflicted.recall_semantic(heldout, relation_cells)
    assert conflict.status == "ABSTAIN_CONFLICT"
    assert conflict.output.decision == "STOP"
    assert conflicted.relation_state_sha256(unrelated_relation_cells) == unrelated_root

    final_episodes = conflicted_episodes + episodes_d
    revised = memory.extend_common_episodic_transform_consolidation_v1(
        conflicted,
        final_episodes,
        semantic_d,
    )
    shifted = revised.recall_semantic(heldout, relation_cells)
    assert shifted.status == "READY"
    assert shifted.output.token == _shift(_HELDOUT_VALUE)
    assert revised.relation_state_sha256(unrelated_relation_cells) == unrelated_root
    assert ready.checkpoint_sha256 == ready_checkpoint
    assert revised.recall_semantic(heldout, unrelated_relation_cells).output.token == (
        _HELDOUT_VALUE
    )

    # Contradictory semantic contexts retain every source-conditioned episode.
    for lesson in episodes_a + episodes_b + episodes_c + episodes_d:
        recall = revised.recall_episode(lesson.observation_events)
        assert recall.status == "RECALLED"
        assert recall.output.token == lesson.consequence_byte[0]
    deranged = revised.recall_episode(
        (
            bytes((_COVER_VALUES[2],)),
            relation_event,
            b"opaque/source/deranged",
        )
    )
    assert deranged.status == "STOP_UNKNOWN"
    assert revised.recall_semantic(heldout, relation_cells).output.token == (
        _shift(_HELDOUT_VALUE)
    )

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
    assert revised.recall_semantic(
        heldout,
        _opaque(b"opaque/relation/unknown"),
    ).status == "ABSTAIN_UNKNOWN_RELATION"

    # Duplicate and physically dependent contexts cannot vote twice.
    dependent_episodes, dependent_semantic = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/a",
        source_event=b"opaque/source/dependent",
    )
    no_op = memory.extend_common_episodic_transform_consolidation_v1(
        ready,
        ready_episodes + dependent_episodes,
        semantic_a + dependent_semantic,
    )
    no_op_summary = no_op.semantic_extension_summary
    assert no_op_summary is not None
    assert no_op_summary.accepted_new_context_count == 0
    assert no_op_summary.duplicate_context_count == 1
    assert no_op_summary.dependent_context_count == 1
    assert no_op_summary.committed_update_ticks == 0
    assert no_op.recall_semantic(heldout, relation_cells).output.token == _HELDOUT_VALUE

    # Teacher-off and cell-off lessons commit no learned tick in either view.
    teacher_off_episodes, teacher_off_semantic = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/teacher-off",
        source_event=b"opaque/source/teacher-off",
        teacher_gate=False,
    )
    cell_off_episodes, cell_off_semantic = _context_views(
        relation_event=relation_event,
        context_event=b"opaque/context/cell-off",
        source_event=b"opaque/source/cell-off",
        cell_gate=False,
    )
    teacher_off = memory.fit_common_episodic_transform_consolidation_v1(
        teacher_off_episodes,
        teacher_off_semantic,
    )
    cell_off = memory.fit_common_episodic_transform_consolidation_v1(
        cell_off_episodes,
        cell_off_semantic,
    )
    assert teacher_off.episodic_training_summary.work.committed_excitatory_ticks == 0
    assert teacher_off.episodic_training_summary.work.committed_inhibitory_ticks == 0
    assert teacher_off.semantic_training_summary.local_physical_update_ticks == 0
    assert cell_off.episodic_training_summary.work.committed_excitatory_ticks == 0
    assert cell_off.episodic_training_summary.work.committed_inhibitory_ticks == 0
    assert cell_off.semantic_training_summary.local_physical_update_ticks == 0
    assert teacher_off.recall_semantic(heldout, relation_cells).status == (
        "ABSTAIN_UNKNOWN_RELATION"
    )
    assert cell_off.recall_semantic(heldout, relation_cells).status == (
        "ABSTAIN_UNKNOWN_RELATION"
    )

    # Opaque relabeling preserves the two-context semantic law.
    relabeled_relation_event = b"opaque/relation/relabelled"
    relabeled_relation_cells = _opaque(relabeled_relation_event)
    relabeled_episodes_a, relabeled_semantic_a = _context_views(
        relation_event=relabeled_relation_event,
        context_event=b"opaque/context/relabelled-a",
        source_event=b"opaque/source/relabelled-a",
    )
    relabeled_episodes_b, relabeled_semantic_b = _context_views(
        relation_event=relabeled_relation_event,
        context_event=b"opaque/context/relabelled-b",
        source_event=b"opaque/source/relabelled-b",
    )
    relabeled = memory.fit_common_episodic_transform_consolidation_v1(
        relabeled_episodes_a + relabeled_episodes_b,
        relabeled_semantic_a + relabeled_semantic_b,
    )
    assert relabeled.recall_semantic(heldout, relabeled_relation_cells).output.token == (
        _HELDOUT_VALUE
    )
    assert relabeled.recall_semantic(heldout, relation_cells).status == (
        "ABSTAIN_UNKNOWN_RELATION"
    )

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

    summary = revised.training_summary
    assert summary.independent_context_count == 6
    assert summary.relation_count == 2
    assert summary.provenance_count == 6
    assert summary.structural_route_choice_count == 16
    assert summary.whole_byte_mapping_count == 256
    assert summary.semantic_synaptic_state_bytes == 1_120
    assert summary.semantic_checkpoint_array_bytes == 8_744
    assert summary.episodic_checkpoint_array_bytes == 1_572_864
    assert summary.semantic_parent_lesson_replay_count == 0
    assert summary.cumulative_episodic_refit_lesson_count == len(final_episodes)
    assert revised.semantic_training_summary.route_inhibitory_update_ticks == 32

    assert memory.CONFIG_RECORD["host_output_selection_count"] == 0
    assert memory.CONFIG_RECORD["raw_byte_or_digit_decode_after_projection_count"] == 0
    assert memory.CONFIG_RECORD["schema_per_context_growth"] is False
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
    assert len(memory.CONFIG_SHA256) == len(revised.checkpoint_sha256) == 64
    assert sha256(Path(memory.__file__).read_bytes()).hexdigest() == (
        memory.SOURCE_SHA256
    )
