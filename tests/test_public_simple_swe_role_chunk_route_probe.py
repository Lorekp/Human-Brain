from __future__ import annotations

import inspect
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "experiments" / "r3_pilot"
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_role_chunk_route_probe as probe


def _prepared():
    protocol, donors, probes = probe._select_tasks()
    encoder, concept_streams, base_streams, task_records = (
        probe._prepare_representations(donors, probes)
    )
    return (
        protocol,
        donors,
        probes,
        encoder,
        concept_streams,
        base_streams,
        task_records,
    )


def test_visible_train_cohort_and_normalized_representations_are_frozen() -> None:
    _protocol, donors, targets, encoder, concept, base, records = _prepared()

    assert tuple((task.public.family, task.public.ordinal) for task in donors) == tuple(
        (family, ordinal)
        for ordinal in probe.DONOR_ORDINALS
        for family in probe.FAMILY_ORDER
    )
    assert tuple((task.public.family, task.public.ordinal) for task in targets) == tuple(
        (family, probe.PROBE_ORDINAL) for family in probe.FAMILY_ORDER
    )
    assert all(task.public.partition == "train" for task in (*donors, *targets))
    assert len(records) == 12
    assert encoder.tokenizer.concepts

    for family in probe.FAMILY_ORDER:
        keys = tuple((family, ordinal) for ordinal in (*probe.DONOR_ORDINALS, probe.PROBE_ORDINAL))
        assert len({concept[key] for key in keys}) == 1
        assert len({base[key] for key in keys}) == 1
    assert len({concept[(family, probe.PROBE_ORDINAL)] for family in probe.FAMILY_ORDER}) == 4
    assert len({base[(family, probe.PROBE_ORDINAL)] for family in probe.FAMILY_ORDER}) == 4
    assert all(
        any(probe.CONCEPT_ID_BASE <= token < probe.ROUTE_TOKEN_BASE for token in stream)
        for stream in concept.values()
    )


def test_conflict_controls_and_common_native_definition_are_exact() -> None:
    _protocol, donors, targets, _encoder, concept, base, _records = _prepared()
    conflicts = [
        probe._conflict_route(task.public.family, task.public.ordinal)
        for task in donors
    ]
    assert Counter(conflicts) == Counter({route: 2 for route in probe.ROUTE_TOKEN_IDS})
    assert all(
        conflict != probe.ROUTE_BY_FAMILY[task.public.family]
        for conflict, task in zip(conflicts, donors, strict=True)
    )

    definition, capacity = probe._definition(donors, targets, concept, base)
    assert capacity == {
        "concept_noninitial_arrivals": 486,
        "base_noninitial_arrivals": 1_022,
        "segment_capacity": 11_242,
        "max_pyramidal_cells": 44_010,
    }
    assert definition.config.column_count == 4_096
    assert definition.config.theta_match == 16
    assert definition.config.theta_active == 18
    assert definition.config.segment_capacity == 11_242
    assert set(probe.ROUTE_TOKEN_IDS) <= {token for token, _columns in definition.token_codes}


@pytest.mark.parametrize(
    ("correct", "events", "selected"),
    [
        ({"concept_true": 3, "concept_conflict": 1, "base_true": 2, "base_conflict": 0}, {"concept_true": 200, "base_true": 400}, "concept"),
        ({"concept_true": 4, "concept_conflict": 1, "base_true": 4, "base_conflict": 1}, {"concept_true": 400, "base_true": 400}, "base"),
        ({"concept_true": 4, "concept_conflict": 1, "base_true": 4, "base_conflict": 1}, {"concept_true": 200, "base_true": 400}, "concept"),
        ({"concept_true": 2, "concept_conflict": 0, "base_true": 2, "base_conflict": 0}, {"concept_true": 200, "base_true": 400}, None),
    ],
)
def test_gate_uses_matched_controls_then_accuracy_events_and_base_tie_break(
    correct: dict[str, int], events: dict[str, int], selected: str | None
) -> None:
    event_totals = {
        "concept_true": events["concept_true"],
        "concept_conflict": events["concept_true"],
        "base_true": events["base_true"],
        "base_conflict": events["base_true"],
    }
    assert probe._gate(correct, event_totals)["selected_representation"] == selected


def test_marker_is_exclusive_and_durable_for_all_four_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _protocol, _donors, targets = probe._select_tasks()
    marker_paths = {
        family: tmp_path / f"{family}.spent.json" for family in probe.FAMILY_ORDER
    }
    monkeypatch.setattr(probe, "SPEND_MARKER_PATHS", marker_paths)
    monkeypatch.setattr(probe, "OUTPUT_PATH", tmp_path / "cohort.json")

    record = probe._create_spend_marker(
        targets[0].public,
        attempt_index=1,
        checkpoint_sha256_by_arm={arm: "ab" * 32 for arm in probe.ARM_ORDER},
    )
    assert record["exclusive_create"] is True
    assert record["flushed_and_fsynced_before_four_generate_calls"] is True
    assert marker_paths[probe.FAMILY_ORDER[0]].exists()
    with pytest.raises(RuntimeError, match="refusing spent"):
        probe._create_spend_marker(
            targets[0].public,
            attempt_index=1,
            checkpoint_sha256_by_arm={arm: "ab" * 32 for arm in probe.ARM_ORDER},
        )


def test_marker_precedes_all_atomic_calls_and_expected_route_is_scored_afterward() -> None:
    source = inspect.getsource(probe.main)
    marker = source.index("marker = _create_spend_marker(")
    generation = source.index("raw_by_arm[arm] = _generate_once(restored[arm])")
    expected = source.index("expected_route = ROUTE_BY_FAMILY")
    assert marker < generation < expected
    assert probe.ARM_ORDER == (
        "concept_true",
        "concept_conflict",
        "base_true",
        "base_conflict",
    )
    assert source.count("_generate_once(") == 1
