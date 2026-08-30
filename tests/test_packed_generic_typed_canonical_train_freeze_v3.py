from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from experiments.r3_pilot import packed_generic_typed_canonical_train_freeze_v3 as panel
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4


def test_dependencies_and_roster_authenticate_without_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dependency preflight attempted a packed outcome")

    monkeypatch.setattr(
        breadth_v4.PackedGenericTypedTrainBreadthV4,
        "build",
        classmethod(forbidden),
    )
    receipt, rows, source_root = panel._authenticate_dependencies()

    assert receipt.sha256 == panel.EXPECTED_DEPENDENCY_ROOT
    assert source_root == panel.EXPECTED_SOURCE_ROOT
    assert receipt.roster_sha256 == panel.EXPECTED_ROSTER_SHA256
    assert receipt.row_sha256s == panel.EXPECTED_ROW_SHA256S
    assert receipt.work_caps == (30_000, 30_000, 80_000)
    assert len(rows) == 6
    assert tuple(row.sha256 for row in rows) == panel.EXPECTED_ROW_SHA256S
    assert all(len(row.cases) == len(row.desired) == 4 for row in rows)


def test_first_spend_receipts_freeze_stop_and_route_isolation() -> None:
    outcomes = panel.FROZEN_ROW_OUTCOMES

    assert panel.EXPECTED_PANEL_SHA256 == (
        "aaf3a03f6cb0bcd2142c266fb55ec5de0912c1e476a6c279cabfd33f9bd8436b"
    )
    assert len(outcomes) == 6
    assert tuple(row.row_sha256 for row in outcomes) == panel.EXPECTED_ROW_SHA256S
    assert tuple(row.result_sha256 for row in outcomes) == (panel.EXPECTED_ROW_RESULT_SHA256S)
    assert tuple(row.status for row in outcomes) == (
        breadth_v4.SELECTED,
        breadth_v4.STOP_AMBIGUOUS,
        breadth_v4.SELECTED,
        breadth_v4.STOP_AMBIGUOUS,
        breadth_v4.STOP_WORK_BOUND,
        breadth_v4.SELECTED,
    )
    assert tuple((row.build_relation_units, row.selection_relation_units) for row in outcomes) == (
        (19_428, 19_453),
        (28_973, 29_009),
        (919, 1_236),
        (25_762, 28_529),
        (27_477, 30_000),
        (919, 1_283),
    )
    assert tuple(row.root_count for row in outcomes) == (60, 211, 7, 122, 171, 7)
    assert tuple(row.selection_races for row in outcomes) == (
        955,
        1_357,
        242,
        4_572,
        5_024,
        242,
    )
    assert tuple(row.selection_proofs for row in outcomes) == (1, 2, 1, 10, 0, 1)

    assert all(row.build_proofs == 0 for row in outcomes)
    assert all(row.build_relation_units <= 30_000 for row in outcomes)
    assert all(row.selection_relation_units <= 30_000 for row in outcomes)
    assert all(row.checkpoint_sha256 == outcomes[0].checkpoint_sha256 for row in outcomes)
    assert all(outcomes[index].selection_factorization_events == () for index in (0, 1, 2, 5))
    for index in (3, 4):
        event_counts = dict(outcomes[index].selection_factorization_events)
        assert event_counts["workset:selected-relation-receipt-registry-open"] == 1
        assert event_counts["workset:selected-factor-position"] > 0

    for index in (0, 2, 5):
        selected = outcomes[index]
        assert selected.selection_proofs == 1
        assert selected.selected_eclass_id is not None
        assert selected.selected_proof_forest_sha256 is not None
        assert selected.race_sha256 is not None
    for index in (1, 3):
        ambiguous = outcomes[index]
        assert ambiguous.selection_proofs > 1
        assert ambiguous.selected_eclass_id is not None
        assert ambiguous.selected_proof_forest_sha256 is not None
        assert ambiguous.race_sha256 is not None
    work_bound = outcomes[4]
    assert work_bound.selection_relation_units == 30_000
    assert work_bound.selection_proofs == 0
    assert work_bound.selected_eclass_id is None
    assert work_bound.selected_proof_forest_sha256 is None
    assert work_bound.race_sha256 is None


def test_engine_seam_has_no_outcome_or_protected_route() -> None:
    source_path = Path(panel.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert (
        "desired"
        not in inspect.signature(breadth_v4.PackedGenericTypedTrainBreadthV4.build).parameters
    )
    assert "packed_generic_typed_canonical_train_freeze_v2" not in source
    assert not any(
        token in module
        for module in imported_modules
        for token in (
            "prompt",
            "property",
            "protected",
            "public_typed_semantic_synthesizer",
            "experiments.g0",
        )
    )
    assert panel.BREADTH_V4_BANKED_COMMIT == ("490fc3580f2a1d9b1cc70cad5a3233e4c9863f8f")
    assert panel.EXPECTED_DEPENDENCY_ROOT
    assert panel.EXPECTED_SOURCE_ROOT
    assert panel.EXPECTED_ROW_RESULT_SHA256S
