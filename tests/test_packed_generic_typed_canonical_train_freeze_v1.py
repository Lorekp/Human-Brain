from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import packed_generic_typed_canonical_train_freeze_v1 as panel
from experiments.r3_pilot import (
    typed_semantic_packed_generic_train_breadth_v2 as breadth,
)


def test_six_visible_train_rows_freeze_before_any_packed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0

    def reject_outcome(*_args: object, **_kwargs: object) -> None:
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("row freeze attempted a packed outcome")

    monkeypatch.setattr(
        breadth.PackedGenericTypedTrainBreadthV2,
        "build",
        reject_outcome,
    )
    rows = panel.load_canonical_train_rows()

    assert build_calls == 0
    assert tuple(row.family for row in rows) == panel.PUBLIC_FAMILIES
    assert tuple(row.task_id for row in rows) == panel.EXPECTED_TASK_IDS
    assert tuple(row.visible_problem_sha256 for row in rows) == (
        panel.EXPECTED_VISIBLE_PROBLEM_SHA256S
    )
    assert panel._roster_sha256(rows) == panel.EXPECTED_ROSTER_SHA256
    assert all(len(row.cases) == len(row.desired) == 4 for row in rows)
    assert all(
        row.sha256 == panel._digest("canonical-row", panel._row_payload(row)) for row in rows
    )


def test_fixed_canonical_panel_records_exact_terminal_stop_without_retuning() -> None:
    result = panel.run_canonical_train_freeze()

    assert result.verdict == panel.STOP_VERDICT
    assert result.roster_sha256 == panel.EXPECTED_ROSTER_SHA256
    assert result.selected_rows == result.stopped_rows == 3
    assert result.sha256 == "2e5f4ff70dac02c095ced103af03b432cc7d57b6f9a710a18b64d1cda3c6ffaa"
    assert tuple(row.status for row in result.results) == (
        breadth.SELECTED,
        breadth.STOP_AMBIGUOUS,
        breadth.SELECTED,
        panel.STOP_UNSUPPORTED_SIGNATURE,
        panel.STOP_UNSUPPORTED_SIGNATURE,
        breadth.SELECTED,
    )
    assert tuple(row.build_relation_units for row in result.results) == (
        19_428,
        28_973,
        919,
        0,
        0,
        919,
    )
    assert tuple(row.build_predictions for row in result.results) == (
        278,
        379,
        108,
        0,
        0,
        108,
    )
    assert tuple(row.build_cache_hits for row in result.results) == (
        266,
        537,
        34,
        0,
        0,
        34,
    )
    assert tuple(row.build_races for row in result.results) == (
        954,
        1_356,
        241,
        0,
        0,
        241,
    )
    assert tuple(row.selection_relation_units for row in result.results) == (
        19_453,
        29_009,
        1_236,
        0,
        0,
        1_283,
    )
    assert tuple(row.selection_races for row in result.results) == (
        955,
        1_357,
        242,
        0,
        0,
        242,
    )
    assert tuple(row.selection_proofs for row in result.results) == (1, 2, 1, 0, 0, 1)
    assert all(row.build_proofs == 0 for row in result.results)
    assert all(row.build_relation_units <= 30_000 for row in result.results)
    assert result.results[0].selected_source == (
        "def solve_86f0ac8ef86cc2ae(n):\n    return min(max(n, -2), 2)\n"
    )
    assert result.results[1].selected_source is None
    assert result.results[2].selected_source == (
        "def solve_e9123d138c4c0a05(text):\n"
        "    return ''.join((char_0 for char_0 in text if char_0 in "
        "'abcdefghijklmnopqrstuvwxyz'))\n"
    )
    assert result.results[5].selected_source == (
        "def solve_02dc6afe28e67f48(text):\n"
        "    return ''.join((char_0 for char_0 in text if char_0 in "
        "'abcdefghijklmnopqrstuvwxyz'))[::-1]\n"
    )
    unsupported = result.results[3:5]
    assert all(
        row.error
        == "packed generic breadth requires Int -> Int, List[Int] -> List[Int], or String -> String"
        and row.frontier_sha256 is None
        and row.root_eclass_ids == ()
        and row.selection_sha256 is None
        for row in unsupported
    )


def test_panel_has_no_old_synthesizer_or_non_train_evidence_path() -> None:
    source = Path(panel.__file__).read_text(encoding="utf-8")
    request_parameters = tuple(inspect.signature(panel._signature_request).parameters)
    rows = panel.load_canonical_train_rows()

    assert request_parameters == ("signature",)
    assert panel._signature_request(rows[0].signature) == panel._signature_request(
        rows[1].signature
    )
    assert panel._signature_request(rows[2].signature) == panel._signature_request(
        rows[5].signature
    )
    assert "public_typed_semantic_synthesizer" not in source
    assert ".property_cases" not in source
    assert 'generate_visible_partition(protocol, "dev")' not in source
    assert 'generate_visible_partition(protocol, "eval")' not in source
    assert breadth.MAX_RECIPE_RELATION_WORK == 30_000
    assert breadth.MAX_PHYSICAL_PREDICTIONS == 30_000
    assert breadth.MAX_PHYSICAL_RACES == 80_000
    assert sha256(panel.BREADTH_SOURCE_PATH.read_bytes()).hexdigest() == (
        panel.EXPECTED_BREADTH_SOURCE_SHA256
    )
    assert sha256(panel.BREADTH_TEST_PATH.read_bytes()).hexdigest() == (
        panel.EXPECTED_BREADTH_TEST_SHA256
    )
