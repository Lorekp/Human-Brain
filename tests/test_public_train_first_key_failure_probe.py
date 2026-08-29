from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from experiments.r3_pilot.public_train_first_key_failure_probe import (
    AllocationOrigin,
    localize_lifecycle,
    parse_checkpoint_rows,
)


def _checkpoint(
    owners: list[int], sources: list[list[int]], ticks: list[list[int]]
) -> bytes:
    segment_count = len(owners)
    assert len(sources) == len(ticks) == segment_count
    assert all(len(row) == 10 for row in sources)
    assert all(len(row) == 10 for row in ticks)
    body = b"".join(
        (
            struct.pack(f"<{segment_count}I", *owners),
            b"".join(struct.pack("<10I", *row) for row in sources),
            b"".join(struct.pack("<10H", *row) for row in ticks),
        )
    )
    header = bytearray(256)
    header[:8] = b"R3NTCP1\x00"
    header[8:10] = (1).to_bytes(2, "little")
    header[10:12] = (256).to_bytes(2, "little")
    header[16:24] = (256 + len(body)).to_bytes(8, "little")
    header[24:32] = len(body).to_bytes(8, "little")
    header[128:160] = hashlib.sha256(body).digest()
    header[192:224] = struct.pack(
        "<IIIQQBBH", segment_count, 0, 176, 3, 3, 0, 0, 0
    )
    return bytes(header) + body


def _origin(first: int, context: tuple[int, ...]) -> AllocationOrigin:
    return AllocationOrigin(
        first_segment_id=first,
        segment_count=1,
        global_step_index=first + 1,
        book_ordinal=0,
        book_zero_based_token_index=first + 1,
        target_token_id=101,
        source_winners=context,
        source_context=context,
    )


def _localize(
    checkpoint: bytes,
    origins: tuple[AllocationOrigin, ...],
    *,
    matching: int,
    active: int,
) -> dict[str, object]:
    return localize_lifecycle(
        checkpoint=checkpoint,
        recognition_segment_limit=len(origins),
        origins=origins,
        current_source_context=tuple(range(11)),
        current_token_columns=tuple(range(11)),
        column_count=22,
        base_cells_per_column=8,
        novel_race_cap=8,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        theta_match=16,
        theta_active=18,
        observed_matching_segments=matching,
        observed_active_segments=active,
        current_step_punished_segments=0,
    )


def test_checkpoint_parser_exposes_canonical_segment_rows() -> None:
    checkpoint = _checkpoint(
        [3, 11],
        [list(range(10)), list(range(1, 11))],
        [[255] * 10, [249] * 10],
    )
    parsed = parse_checkpoint_rows(checkpoint)
    assert parsed.segment_count == 2
    assert parsed.owners.tolist() == [3, 11]
    assert parsed.sources.tolist() == [list(range(10)), list(range(1, 11))]
    assert parsed.permanence_ticks.tolist() == [[255] * 10, [249] * 10]


def test_foreign_matching_rows_localize_winner_key_divergence() -> None:
    current = tuple(range(11))
    foreign = tuple(range(10)) + (99,)
    checkpoint = _checkpoint([3], [list(range(10))], [[255] * 10])
    result = _localize(checkpoint, (_origin(0, foreign),), matching=1, active=1)
    assert result["classification"] == "WINNER_KEY_DIVERGENCE"
    assert result["foreign_matching_segments"] == 1
    assert result["exact_source_deactivated_segments"] == 0
    assert result["checkpoint_rows_equal_step_counters"] is True
    assert result["current_source_context"] == list(current)


def test_exact_source_below_connected_ticks_localizes_punishment_deactivation() -> None:
    current = tuple(range(11))
    checkpoint = _checkpoint([3], [list(range(10))], [[249] * 10])
    result = _localize(checkpoint, (_origin(0, current),), matching=1, active=0)
    assert result["classification"] == "PUNISHMENT_INDUCED_DEACTIVATION"
    assert result["foreign_matching_segments"] == 0
    assert result["exact_source_deactivated_segments"] == 1
    assert result["exact_source_rows_below_connected_after_noninitial_ticks"] == 1
    groups = result["segment_groups"]
    assert isinstance(groups, list) and groups[0]["permanence_ticks"] == [249] * 10


def test_failing_step_appends_are_excluded_from_recognition_ledger() -> None:
    current = tuple(range(11))
    checkpoint = _checkpoint(
        [3, 11],
        [list(range(10)), list(range(10))],
        [[255] * 10, [255] * 10],
    )
    result = localize_lifecycle(
        checkpoint=checkpoint,
        recognition_segment_limit=1,
        origins=(_origin(0, current),),
        current_source_context=current,
        current_token_columns=tuple(range(11)),
        column_count=22,
        base_cells_per_column=8,
        novel_race_cap=8,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        theta_match=16,
        theta_active=18,
        observed_matching_segments=1,
        observed_active_segments=1,
        current_step_punished_segments=0,
    )
    assert result["calculated_post_step_matching_preexisting_rows"] == 1
    assert result["checkpoint_segment_count_after_step"] == 2


def test_banked_train_only_artifact_records_foreign_matching_not_deactivation() -> None:
    artifact = Path(
        "experiments/r3_pilot/public_train_first_key_failure_probe.json"
    )
    raw = artifact.read_bytes()
    payload = json.loads(raw)
    result = payload["result"]
    lifecycle = result["lifecycle"]

    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == (
        "1197fcd76b9257d653f91288f6313eaf75649239269d2511dd50aba2b21e9dbe"
    )
    assert payload["status"] == "BANKED_FIRST_RECURRENT_KEY_MISMATCH"
    assert payload["train_input"]["held_out_books_materialized"] == 0
    assert result["global_zero_based_step_index"] == 7_933
    assert result["key_was_known"] is False
    assert result["outcome_mismatch"] is False
    assert result["observed"]["matching_segments"] == 11
    assert result["observed"]["active_segments"] == 0
    assert result["observed"]["allocated_segments"] == 11
    assert result["observed"]["punished_segments"] == 0
    assert lifecycle["classification"] == "WINNER_KEY_DIVERGENCE"
    assert lifecycle["foreign_matching_segments"] == 11
    assert lifecycle["exact_source_deactivated_segments"] == 0
    assert lifecycle["checkpoint_rows_equal_step_counters"] is True
    assert all(
        group["source_relation"] == "DIVERGED_SOURCE"
        and group["origin_global_step_index"] == 249
        and group["origin_current_source_overlap_cells"] == 8
        and group["matching"] is True
        and group["active"] is False
        and group["owners_in_current_token_columns"] is False
        for group in lifecycle["segment_groups"]
    )
