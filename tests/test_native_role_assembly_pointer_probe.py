from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from experiments.r3_pilot.native_role_assembly_pointer_probe import (
    PASS_OUTCOME,
    run_development_probe,
)


@pytest.fixture(scope="module")
def report() -> Mapping[str, object]:
    return run_development_probe()


def _owners(report: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = report[key]
    assert isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    assert all(type(owner) is int for owner in value)
    return tuple(value)


def test_native_cue_rows_recruit_distinct_ordered_k11_assemblies(
    report: Mapping[str, object],
) -> None:
    assert report["outcome"] == PASS_OUTCOME
    assert report["assembly_k"] == 11
    assert (report["theta_match"], report["theta_active"]) == (16, 18)
    assert report["threshold_offset_mv"] == 0.0
    operation = _owners(report, "operation_owners")
    role = _owners(report, "role_owners")
    pointer = _owners(report, "pointer_owners")
    assert tuple(map(len, (operation, role, pointer))) == (11, 11, 11)
    assert all(len(set(owners)) == 11 for owners in (operation, role, pointer))
    assert len({operation, role, pointer}) == 3
    assert _owners(report, "pointer_cue_replay_owners") == pointer
    assert report["cue_repeat_segment_count_before"] == 22
    assert report["cue_repeat_segment_count_after"] == 22


def test_partial_pointer_completes_and_settles_to_exact_ordered_owners(
    report: Mapping[str, object],
) -> None:
    pointer = _owners(report, "pointer_owners")
    partial = _owners(report, "partial_pointer_cue")
    assert report["partial_overlap"] == 10
    assert sum(owner in pointer for owner in partial) == 10
    assert _owners(report, "completed_owners") == pointer
    assert _owners(report, "settled_owners") == pointer
    assert _owners(report, "repeated_partial_owners") == pointer


def test_all_other_ten_recurrence_is_necessary_and_recall_has_zero_growth(
    report: Mapping[str, object],
) -> None:
    pointer = _owners(report, "pointer_owners")
    ablated = _owners(report, "ablated_completed_owners")
    assert report["recurrent_rows"] == 11
    assert report["recurrent_live_pairs_per_row"] == 10
    assert report["recurrent_self_edges"] == 0
    assert report["recurrence_changes_outcome"] is True
    assert ablated != pointer
    assert report["segment_count_before_recall"] == 33
    assert report["segment_count_after_recall"] == 33
    assert report["ablated_segment_count_before_recall"] == 22
    assert report["ablated_segment_count_after_recall"] == 22


def test_claim_stays_below_typed_generation_and_spends_no_target(
    report: Mapping[str, object],
) -> None:
    boundary = report["claim_boundary"]
    assert isinstance(boundary, str)
    assert "no TypedWorkspace injection" in boundary
    assert report["target_attempt_count"] == 0
