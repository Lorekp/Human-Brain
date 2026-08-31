from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_associative_memory_v1 as associative
from experiments.g0 import common_event_cells_v1 as projection
from experiments.r3_pilot import common_event_associative_memory_v2 as memory


def _cue(source: bytes) -> tuple[bytes, ...]:
    return (b"opaque-subject", b"opaque-relation", source)


def _teacher_output_cells(value: int) -> tuple[int, ...]:
    digits = [0] * projection.OUTPUT_DIGIT_LANES
    remainder = value
    for lane in range(projection.OUTPUT_DIGIT_LANES - 1, -1, -1):
        digits[lane] = remainder % projection.OUTPUT_DIGIT_RADIX
        remainder //= projection.OUTPUT_DIGIT_RADIX
    rows = [[0] * projection.OUTPUT_DIGIT_RADIX for _lane in digits]
    for lane, digit in enumerate(digits):
        rows[lane][digit] = 2 * projection.STOP_SCORE
    decision = projection.CommonEventCells().race_output(
        tuple(tuple(row) for row in rows)
    )
    assert decision.token == value
    return decision.active_cells


def _physical_lesson(
    source: bytes,
    value: int,
    *,
    teacher_gate: bool = True,
    cell_gate: bool = True,
) -> memory.PhysicalCommonAssociationLessonV2:
    return memory.PhysicalCommonAssociationLessonV2(
        observation_events=_cue(source),
        teacher_output_cells=_teacher_output_cells(value),
        teacher_gate=teacher_gate,
        cell_gate=cell_gate,
    )


def _assert_recalled(
    state: memory.FrozenCommonEventAssociativeMemoryV2,
    source: bytes,
    value: int,
) -> None:
    recall = state.recall(_cue(source))
    assert recall.status == "RECALLED"
    assert recall.output.token == value


def test_replay_free_exact_delta_retains_base_conflicts_and_stays_flat() -> None:
    source_a = b"source-A"
    source_b = b"source-B"
    source_c = b"source-C"
    source_d = b"source-D"
    values = {source_a: 0x24, source_b: 0x91, source_c: 0x5A, source_d: 0xE3}
    base = associative.fit_common_event_associations(
        tuple(
            associative.CommonAssociationLesson(
                observation_events=_cue(source),
                consequence_byte=bytes((values[source],)),
            )
            for source in (source_a, source_b)
        )
    )
    base_root = base.checkpoint_sha256
    upgraded = memory.upgrade_common_event_associative_memory_v2(base)
    upgraded_root = upgraded.checkpoint_sha256

    assert upgraded.learned_state_shapes == ((0, 12), (0, 4), (0, 32))
    assert upgraded.provenance == ()
    assert upgraded.training_summary.legacy_base_count == 1
    assert upgraded.training_summary.delta_matrix_count == 1
    assert upgraded.training_summary.retained_v2_parent_count == 0
    assert upgraded.training_summary.parent_lesson_replay_count == 0
    assert (
        upgraded.training_summary.inherited_v1_checkpoint_array_bytes
        == associative.CHECKPOINT_ARRAY_BYTES
        == 1_572_864
    )
    _assert_recalled(upgraded, source_a, values[source_a])
    _assert_recalled(upgraded, source_b, values[source_b])
    assert upgraded.recall(_cue(b"unknown-source")).status == "STOP_UNKNOWN"

    transition_c = memory.advance_common_event_association_v2(
        upgraded,
        _physical_lesson(source_c, values[source_c]),
    )
    state_c = transition_c.successor
    state_c_root = state_c.checkpoint_sha256
    provenance_c = state_c.provenance
    assert transition_c.status == "APPENDED"
    assert transition_c.accepted
    assert transition_c.cue_addresses_added == 12
    assert transition_c.output_addresses_added == 4
    assert transition_c.provenance_bytes_added == 32
    assert transition_c.match_cells_added == 1
    assert transition_c.parent_lesson_replay_count == 0
    assert state_c.learned_state_shapes == ((1, 12), (1, 4), (1, 32))
    assert state_c.training_summary.delta_checkpoint_array_bytes == 96
    assert state_c.training_summary.cue_array_bytes == 48
    assert state_c.training_summary.output_array_bytes == 16
    assert state_c.training_summary.provenance_array_bytes == 32
    assert state_c.training_summary.canonical_metadata_bytes > 0
    assert len(provenance_c) == 1
    assert provenance_c[0].match_cell_id == memory.MATCH_CELL_OFFSET
    assert len(bytes.fromhex(provenance_c[0].sha256)) == 32

    transition_d = memory.advance_common_event_association_v2(
        state_c,
        _physical_lesson(source_d, values[source_d]),
    )
    state_d = transition_d.successor
    for source, value in values.items():
        _assert_recalled(state_d, source, value)
    assert state_d.learned_state_shapes == ((2, 12), (2, 4), (2, 32))
    assert state_d.training_summary.delta_checkpoint_array_bytes == 2 * 96
    assert state_d.training_summary.delta_match_cell_count == 2
    assert state_d.training_summary.total_v2_cell_count == memory.MATCH_CELL_OFFSET + 2
    assert state_d.training_summary.parent_checkpoint_sha256 == state_c_root
    assert state_d.recall(_cue(b"unknown-source")).status == "STOP_UNKNOWN"
    assert (
        state_d.recall((source_d, b"opaque-relation", b"opaque-subject")).status
        == "STOP_UNKNOWN"
    )

    assert base.checkpoint_sha256 == base_root
    assert upgraded.checkpoint_sha256 == upgraded_root
    assert state_c.checkpoint_sha256 == state_c_root
    assert state_c.provenance == provenance_c

    teacher_off = memory.advance_common_event_association_v2(
        state_d,
        _physical_lesson(b"source-off-teacher", 0x44, teacher_gate=False),
    )
    cell_off = memory.advance_common_event_association_v2(
        state_d,
        _physical_lesson(b"source-off-cell", 0x44, cell_gate=False),
    )
    for stopped, expected_status in (
        (teacher_off, "STOP_TEACHER_OFF"),
        (cell_off, "STOP_CELL_OFF"),
    ):
        assert stopped.status == expected_status
        assert not stopped.accepted
        assert stopped.successor is state_d
        assert stopped.successor_checkpoint_sha256 == state_d.checkpoint_sha256
        assert stopped.delta_episode_count_before == stopped.delta_episode_count_after
        assert stopped.cue_addresses_added == 0
        assert stopped.output_addresses_added == 0
        assert stopped.provenance_bytes_added == 0
        assert stopped.match_cells_added == 0
    gated_recall = state_d.recall(_cue(source_d), cell_gate=False)
    assert gated_recall.status == "STOP_CELL_OFF"
    assert gated_recall.output.token is None
    assert gated_recall.work.projection_work == 0
    assert gated_recall.work.base_recall_work == 0
    assert gated_recall.work.match_synaptic_equality_reads == 0

    repeated = memory.advance_common_event_association_v2(
        state_c,
        _physical_lesson(source_c, values[source_c]),
    ).successor
    repeated_recall = repeated.recall(_cue(source_c))
    assert repeated_recall.status == "RECALLED"
    assert repeated_recall.output.token == values[source_c]
    assert repeated_recall.delta_match_spikes == (True, True)

    delta_conflict = memory.advance_common_event_association_v2(
        state_c,
        _physical_lesson(source_c, values[source_c] ^ 0x03),
    ).successor
    assert (
        delta_conflict.recall(_cue(source_c)).status
        == "STOP_AMBIGUOUS_COLLISION"
    )
    legacy_conflict = memory.advance_common_event_association_v2(
        upgraded,
        _physical_lesson(source_a, values[source_a] ^ 0x03),
    ).successor
    legacy_conflict_recall = legacy_conflict.recall(_cue(source_a))
    assert legacy_conflict_recall.status == "STOP_AMBIGUOUS_COLLISION"
    assert legacy_conflict_recall.base_active_output_cells
    assert legacy_conflict_recall.delta_match_spikes == (True,)

    bulk = upgraded
    for ordinal in range(100):
        source = b"bulk-source-" + ordinal.to_bytes(2, "big")
        bulk = memory.advance_common_event_association_v2(
            bulk,
            _physical_lesson(source, ordinal),
        ).successor
    bulk_summary = bulk.training_summary
    assert bulk.learned_state_shapes == ((100, 12), (100, 4), (100, 32))
    assert bulk_summary.delta_episode_count == 100
    assert bulk_summary.delta_match_cell_count == 100
    assert bulk_summary.delta_checkpoint_array_bytes == 100 * 96
    assert bulk_summary.cue_array_bytes == 100 * 48
    assert bulk_summary.output_array_bytes == 100 * 16
    assert bulk_summary.provenance_array_bytes == 100 * 32
    assert bulk_summary.inherited_v1_checkpoint_array_bytes == 1_572_864
    assert bulk_summary.canonical_metadata_bytes > 0
    assert bulk_summary.legacy_base_count == 1
    assert bulk_summary.delta_matrix_count == 1
    assert bulk_summary.retained_v2_parent_count == 0
    assert bulk_summary.parent_lesson_replay_count == 0
    assert len(bulk.provenance) == 100
    assert base.checkpoint_sha256 == base_root
    assert upgraded.checkpoint_sha256 == upgraded_root
    assert "_parent" not in memory.FrozenCommonEventAssociativeMemoryV2.__slots__
    assert memory.FrozenCommonEventAssociativeMemoryV2.__slots__.count("_base") == 1

    assert memory.CUE_ARRAY_BYTES_PER_EPISODE == 48
    assert memory.OUTPUT_ARRAY_BYTES_PER_EPISODE == 16
    assert memory.PROVENANCE_ARRAY_BYTES_PER_EPISODE == 32
    assert memory.DELTA_ARRAY_BYTES_PER_EPISODE == 96
    assert memory.MATCH_SCORE == 12
    assert memory.MATCH_TONIC_SCORE == 11
    assert memory.CONFIG_RECORD["host_shard_iteration_or_ranking_count"] == 0
    assert memory.CONFIG_RECORD["raw_byte_or_digit_decode_after_projection_count"] == 0
    assert memory.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA == 0
    assert memory.CONFIG_RECORD[
        "generator_corpus_path2_dev_m0_protected_g0_eval_coding_inputs"
    ] == (0, 0, 0, 0, 0, 0, 0, 0)

    source_path = Path(memory.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    assert sha256(source_path.read_bytes()).hexdigest() == memory.SOURCE_SHA256
    for forbidden in (
        "._excitatory",
        "._inhibitory",
        "base._metadata",
        "base._training_summary",
        "base_v1._metadata",
        "base_v1._training_summary",
        "._ordered_trace",
        "._teacher_spike_cells",
        "._CONSTRUCTION_TOKEN",
        "base_recall.status",
        "base_recall.output.token",
        "winner_cell",
    ):
        assert forbidden not in source_text
    assert "base_recall.output.active_cells" in source_text
