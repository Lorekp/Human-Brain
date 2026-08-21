from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks import large_scale_capability as gate


def _synthetic_run(seed: int, cells: int, teacher: float, free: float) -> dict:
    return {
        "key": f"seed={seed}:cells={cells}",
        "stage": "uniform_sqrt_data",
        "stage_index": 1,
        "data_track": "sqrt_scaled",
        "topology": "uniform_no_self",
        "seed": seed,
        "cells": cells,
        "capability": {
            "teacher_forced": {"accuracy": teacher},
            "free_running": {"accuracy": free},
        },
        "dynamics": {
            "responsive_state_cell_fraction": 0.10,
            "mean_firing_hz_per_cell": 12.0,
        },
        "learning": {
            "readout_source_cell_fraction": 0.08,
        },
    }


def test_frozen_protocol_has_all_stages_and_passes_freshness_preflight() -> None:
    protocol, protocol_hash = gate._load_protocol(gate.DEFAULT_PROTOCOL)
    specifications = gate._specifications(protocol)

    assert len(protocol_hash) == 64
    assert len(specifications) == 148
    stage_counts = {
        stage: sum(specification["stage"] == stage for specification in specifications)
        for stage in {specification["stage"] for specification in specifications}
    }
    assert stage_counts == {
        "uniform_fixed_data": 20,
        "uniform_sqrt_data": 20,
        "fixed_data_groups": 36,
        "fixed_data_geometry": 72,
    }
    assert gate._plan(protocol, protocol_hash)["mode"] == "plan_only_no_validation_executed"


def test_freshness_preflight_rejects_a_prior_path_mention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    benchmarks = tmp_path / "benchmarks"
    artifacts.mkdir()
    benchmarks.mkdir()
    protocol = {
        "version": "freshness-test",
        "data": {
            "validation": {"files": [{"path": "fresh/validation.py"}]},
        },
    }
    protocol_path = benchmarks / "new_protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    (artifacts / "older.json").write_text(
        json.dumps({"path": "fresh/validation.py"}), encoding="utf-8"
    )
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="pre-registration history"):
        gate._assert_fresh_validation_history(protocol, protocol_path)


def test_five_seed_primary_rule_uses_effect_and_direction_not_bootstrap() -> None:
    protocol, _ = gate._load_protocol(gate.DEFAULT_PROTOCOL)
    runs = []
    for seed in (17, 29, 43, 59, 71):
        runs.append(_synthetic_run(seed, 6_144, teacher=0.10, free=0.05))
        runs.append(_synthetic_run(seed, 131_072, teacher=0.12, free=0.06))

    result = gate._scale_criterion(runs, protocol, "sqrt_scaled")

    assert result is not None
    assert result["passes"] is True
    assert result["teacher_forced_effect"]["positive_directions"] == 5
    assert result["teacher_forced_effect"]["exact_one_sided_sign_p"] == 0.03125
    assert (
        result["teacher_forced_effect"][
            "exact_one_sided_sign_flip_permutation_p"
        ]
        == 0.03125
    )
    assert "bootstrap" not in json.dumps(result).lower()


def test_primary_rule_remains_incomplete_without_all_five_pairs() -> None:
    protocol, _ = gate._load_protocol(gate.DEFAULT_PROTOCOL)
    runs = []
    for seed in (17, 29, 43, 59):
        runs.append(_synthetic_run(seed, 6_144, teacher=0.10, free=0.05))
        runs.append(_synthetic_run(seed, 131_072, teacher=0.12, free=0.06))

    assert gate._scale_criterion(runs, protocol, "sqrt_scaled") is None


def test_incremental_report_is_atomic_and_resumes_in_fixed_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, protocol_hash = gate._load_protocol(gate.DEFAULT_PROTOCOL)
    output = tmp_path / "report.json"

    def fake_run_one(_protocol: dict, specification: dict) -> dict:
        return {
            **specification,
            "sentinel": True,
            "capability": {
                "teacher_forced": {"accuracy": 0.1},
                "free_running": {"accuracy": 0.05},
            },
            "dynamics": {"responsive_state_cell_fraction": 0.1},
            "learning": {"applied_local_updates": 1},
        }

    monkeypatch.setattr(gate, "_run_one", fake_run_one)
    monkeypatch.setattr(
        gate,
        "_summary",
        lambda runs, _protocol: {"completed_runs": len(runs)},
    )

    first = gate.run(protocol, protocol_hash, output, limit_runs=1)
    second = gate.run(protocol, protocol_hash, output, limit_runs=1)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert len(first["runs"]) == 1
    assert len(second["runs"]) == 2
    assert [run["key"] for run in persisted["runs"]] == [
        specification["key"] for specification in gate._specifications(protocol)[:2]
    ]
    assert not output.with_suffix(".json.tmp").exists()
