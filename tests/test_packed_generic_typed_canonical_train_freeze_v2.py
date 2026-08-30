from __future__ import annotations

from pathlib import Path

import pytest
from experiments.r3_pilot import packed_generic_typed_canonical_train_freeze_v2 as panel
from experiments.r3_pilot import (
    typed_semantic_packed_generic_train_breadth_v3 as breadth,
)


def test_v1_roster_and_banked_dependencies_freeze_before_any_v3_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0

    def reject_outcome(*_args: object, **_kwargs: object) -> None:
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("dependency/row freeze attempted a packed outcome")

    monkeypatch.setattr(
        breadth.PackedGenericTypedTrainBreadthV3,
        "build",
        reject_outcome,
    )
    dependencies = panel._authenticate_dependencies()
    rows = panel.load_frozen_train_rows()

    assert build_calls == 0
    assert dependencies.sha256 == panel.EXPECTED_DEPENDENCY_SHA256
    assert dependencies.v1_commit_full.startswith(dependencies.v1_commit)
    assert dependencies.v3_commit_full.startswith(dependencies.v3_commit)
    assert dependencies.work_caps == (30_000, 30_000, 80_000)
    assert tuple(row.sha256 for row in rows) == panel.EXPECTED_V1_ROW_SHA256S
    assert panel.v1._roster_sha256(rows) == panel.EXPECTED_ROSTER_SHA256


def test_spent_panel_record_is_complete_without_replaying_the_panel() -> None:
    frozen_columns = (
        panel.EXPECTED_RESULT_SHA256S,
        panel.EXPECTED_STATUSES,
        panel.EXPECTED_FRONTIER_SHA256S,
        panel.EXPECTED_ROOT_COUNTS,
        panel.EXPECTED_BUILD_WORK_SHA256S,
        panel.EXPECTED_BUILD_RELATION_UNITS,
        panel.EXPECTED_BUILD_PREDICTIONS,
        panel.EXPECTED_BUILD_CACHE_HITS,
        panel.EXPECTED_BUILD_RACES,
        panel.EXPECTED_SELECTION_SHA256S,
        panel.EXPECTED_SELECTION_RELATION_UNITS,
        panel.EXPECTED_SELECTION_RACES,
        panel.EXPECTED_SELECTION_PROOFS,
        panel.EXPECTED_SELECTED_SOURCES,
    )

    assert all(len(column) == len(panel.EXPECTED_V1_ROW_SHA256S) for column in frozen_columns)
    assert panel.EXPECTED_PANEL_SHA256 == (
        "25144b5d34f21f2be2dcbb826847e3491b83863a31f0ea5837332c33d111c1dc"
    )
    assert panel.EXPECTED_STATUSES == (
        breadth.SELECTED,
        breadth.STOP_AMBIGUOUS,
        breadth.SELECTED,
        breadth.STOP_AMBIGUOUS,
        breadth.STOP_WORK_BOUND,
        breadth.SELECTED,
    )
    assert panel.EXPECTED_BUILD_RELATION_UNITS == (
        19_428,
        28_973,
        919,
        25_762,
        27_477,
        919,
    )
    assert panel.EXPECTED_SELECTION_RELATION_UNITS == (
        19_453,
        29_009,
        1_236,
        29_758,
        30_000,
        1_283,
    )
    assert panel.EXPECTED_SELECTION_RACES == (955, 1_357, 242, 4_572, 5_024, 242)
    assert panel.EXPECTED_SELECTION_PROOFS == (1, 2, 1, 10, 0, 1)
    assert sum(status == breadth.SELECTED for status in panel.EXPECTED_STATUSES) == 3
    assert sum(status == breadth.STOP_AMBIGUOUS for status in panel.EXPECTED_STATUSES) == 2
    assert panel.EXPECTED_SELECTION_RELATION_UNITS[4] == breadth.MAX_RECIPE_RELATION_WORK
    assert all(
        len(digest) == 64 and digest == digest.lower()
        for column in (
            panel.EXPECTED_RESULT_SHA256S,
            panel.EXPECTED_FRONTIER_SHA256S,
            panel.EXPECTED_BUILD_WORK_SHA256S,
            panel.EXPECTED_SELECTION_SHA256S,
        )
        for digest in column
    )


def test_v2_freeze_has_one_v3_build_select_path_and_no_forbidden_evidence() -> None:
    source = Path(panel.__file__).read_text(encoding="utf-8")

    assert source.count("breadth_v3.PackedGenericTypedTrainBreadthV3.build(") == 1
    assert source.count("frontier.select(") == 1
    assert "PackedGenericTypedTrainBreadthV2.build(" not in source
    assert "public_typed_semantic_synthesizer" not in source
    assert ".property_cases" not in source
    assert "generate_visible_partition" not in source
    assert "prompt_selection" not in source
    assert breadth.MAX_RECIPE_RELATION_WORK == 30_000
    assert breadth.MAX_PHYSICAL_PREDICTIONS == 30_000
    assert breadth.MAX_PHYSICAL_RACES == 80_000
