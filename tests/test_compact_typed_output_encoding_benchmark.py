from __future__ import annotations

import ast
import copy
import json
import os
import sys
from pathlib import Path

import pytest

from benchmarks import compact_typed_output_encoding as benchmark


def _live_phase() -> str:
    from experiments.r3_pilot import public_typed_output_cells as output

    encoded = output.TypedOutputCellBank.encode("A")
    return "compact" if hasattr(encoded, "factor_winners_le") else "dense"


def _comparison_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    runtime = {
        fixture: {"median_seconds": 0.001, "result_sha256": f"result/{fixture}"}
        for fixture in benchmark.RUNTIME_FIXTURE_IDS
    }
    baseline_memory = {
        "strings_2000_max26": {
            "medians": {"tracemalloc_peak_delta_bytes": 51 * benchmark.MIB}
        },
        "detectors_1000_max26": {
            "medians": {"tracemalloc_peak_delta_bytes": 26 * benchmark.MIB}
        },
        "string_semantic_frontier_500x4": {
            "medians": {"tracemalloc_peak_delta_bytes": 100 * benchmark.MIB}
        },
    }
    compact_memory = {
        "strings_2000_max26": {
            "medians": {"tracemalloc_peak_delta_bytes": benchmark.MIB}
        },
        "detectors_1000_max26": {
            "medians": {"tracemalloc_peak_delta_bytes": benchmark.MIB // 2}
        },
        "string_semantic_frontier_500x4": {
            "medians": {"tracemalloc_peak_delta_bytes": 20 * benchmark.MIB}
        },
    }
    work_bound = {
        fixture: {"decision_sha256": f"decision/{fixture}", "wall_seconds": 1.0}
        for fixture in benchmark.WORK_BOUND_FIXTURE_IDS
    }
    common = {
        "schema": benchmark.SCHEMA,
        "protocol_sha256": "protocol",
        "environment": {"stable_sha256": "environment"},
        "provenance": {"harness_sha256": "harness"},
        "golden": {"scientific_sha256": "golden"},
        "production_audit_materialization_calls": 0,
    }
    baseline = {
        **common,
        "measurements": {
            "memory": baseline_memory,
            "runtime": copy.deepcopy(runtime),
            "work_bound": copy.deepcopy(work_bound),
        },
    }
    compact = {
        **common,
        "golden": {
            "scientific_sha256": "golden",
            "structural_storage": {
                "encoding_fields": ["factor_winners_le", "physical_digest"],
                "encoding_direct_numpy_array_fields": [],
                "factor_winners_bytes": 72,
                "compact_stored_payload_bytes": 104,
                "retired_encoding_attributes_present": [],
                "bank_finish_present": False,
                "detector_direct_numpy_array_fields": [],
                "detector_weights_present": False,
                "detector_stored_field_count": 2,
                "detector_target_field_count": 1,
                "detector_digest_field_count": 1,
                "detector_other_field_count": 0,
            },
        },
        "measurements": {
            "memory": compact_memory,
            "runtime": copy.deepcopy(runtime),
            "work_bound": copy.deepcopy(work_bound),
        },
    }
    return baseline, compact


def test_protocol_freezes_sampling_and_all_fixture_identities() -> None:
    assert benchmark.PROTOCOL["sampling"] == {
        "fast_runtime": {"warmups": 2, "measurements": 7},
        "synthesis_runtime": {"warmups": 2, "measurements": 3},
        "memory": {"warmups": 0, "fresh_process_samples": 7},
        "work_bound": {"warmups": 0, "measurements": 1},
    }
    assert len(benchmark.FAST_RUNTIME_FIXTURE_IDS) == 2
    assert len(benchmark.SYNTHESIS_FIXTURE_IDS) == 7
    assert benchmark.WORK_BOUND_FIXTURE_IDS == ("sentinel_a", "sentinel_b")
    sentinel_b = benchmark.PROTOCOL["work_bound_fixtures"][1]
    assert sentinel_b["expected_completed"] == 30_000
    assert sentinel_b["expected_attempted"] == 30_001
    assert benchmark._catalog_off_checkpoint() == (
        "ed7dff6036ab522840bca1c0a1b6fd7c2e0de0768be813bb33b1920b5f291af3"
    )


def test_source_manifest_covers_direct_humanbrain_execution_dependencies() -> None:
    assert {
        "src/humanbrain/cells.py",
        "src/humanbrain/resting_glif_race.py",
        "src/humanbrain/resting_glif_tier_beam.py",
        "src/humanbrain/typed_program.py",
        "src/humanbrain/typed_program_types.py",
        "src/humanbrain/typed_semantic_synthesis.py",
    }.issubset(benchmark.RELEVANT_SOURCE_PATHS)
    assert benchmark._is_binary_digest(b"\x00" * 32)
    assert not benchmark._is_binary_digest("00" * 32)
    assert not benchmark._is_binary_digest(bytearray(32))


def test_max_string_fixture_is_unique_printable_and_deterministic() -> None:
    rows = tuple(benchmark._max_string(index) for index in range(2_000))
    assert len(set(rows)) == 2_000
    assert all(len(row) == 26 for row in rows)
    assert all(0x20 <= ord(character) <= 0x7E for row in rows for character in row)
    assert rows == tuple(benchmark._max_string(index) for index in range(2_000))
    assert len({benchmark._max_string(0, lane) for lane in range(4)}) == 4


def test_top_level_is_stdlib_only_and_scope_guard_rejects_gated_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "numpy" not in imported
    assert "humanbrain" not in imported
    assert "experiments" not in imported
    assert "simple_swe_train_bridge_v1" not in source

    forbidden = benchmark.FORBIDDEN_MODULE_PREFIXES[0] + ".fixture"
    monkeypatch.setitem(sys.modules, forbidden, object())
    with pytest.raises(RuntimeError, match="forbidden benchmark modules"):
        benchmark._scope_guard()


def test_hidden_fast_worker_records_phase_boundary_and_exact_samples() -> None:
    process = benchmark._windows_process_snapshot()
    affinity = benchmark._parse_affinity_mask(process["affinity_mask"])
    phase = _live_phase()
    result = benchmark._run_worker(
        "runtime", "detector_grid_500x4", affinity, phase
    )

    assert result["warmups"] == 2
    assert result["measurements"] == 7
    assert len(result["raw_nanoseconds"]) == 7
    assert result["result"]["lane_spike_counts"] == [4, 4, 4, 4]
    assert result["production_audit_materialization_calls"] == 0
    assert result["worker"]["phase"] == phase
    assert result["worker"]["timed_boundary"] == result["timed_boundary"]
    assert len(result["worker"]["fixture_sha256"]) == 64
    assert len(result["worker"]["harness_sha256"]) == 64
    if os.name == "nt":
        assert result["worker_conditions"]["stable"]["process"]["priority_class"] == (
            "NORMAL_PRIORITY_CLASS"
        )


def test_atomic_artifact_publication_is_exclusive_and_hash_checked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    payload = {"schema": benchmark.SCHEMA, "phase": "unit-test"}
    benchmark._write_artifact_exclusive(path, payload)
    loaded, raw_sha256 = benchmark._load_artifact(path)

    assert loaded["schema"] == benchmark.SCHEMA
    assert len(raw_sha256) == 64
    with pytest.raises(FileExistsError):
        benchmark._write_artifact_exclusive(path, payload)

    loaded["phase"] = "tampered"
    path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash"):
        benchmark._load_artifact(path)

    stopped_path = tmp_path / "compact-stop.json"
    stopped = benchmark._finalize_capture(
        stopped_path,
        {
            "schema": benchmark.SCHEMA,
            "phase": "compact",
            "comparison": {"passed": False, "verdict": "STOP"},
        },
    )
    assert not stopped_path.exists()
    assert stopped["publication"]["status"] == "NOT_PUBLISHED_COMPARISON_STOP"
    assert len(stopped["stop_payload_sha256"]) == 64

    passed_path = tmp_path / "compact-pass.json"
    passed = benchmark._finalize_capture(
        passed_path,
        {
            "schema": benchmark.SCHEMA,
            "phase": "compact",
            "comparison": {"passed": True, "verdict": "PASS"},
        },
    )
    assert passed_path.exists()
    assert passed["publication"]["status"] == "PUBLISHED_COMPACT_PASS"


def test_repo_host_mutex_requires_and_records_operator_attestation() -> None:
    with pytest.raises(ValueError, match="attest"):
        with benchmark._capture_lock(False):
            pass

    with benchmark._capture_lock(True) as attestation:
        assert benchmark.HOST_LOCK_PATH.exists()
        assert attestation["operator_attested_no_other_project_gate_or_benchmark"]
        assert attestation["mutex_path"].endswith(
            ".compact_typed_output_benchmark.host.lock"
        )
    assert not benchmark.HOST_LOCK_PATH.exists()


def test_comparison_gates_require_scientific_parity_and_handle_zero_memory() -> None:
    baseline, compact = _comparison_artifacts()
    passing = benchmark._compare_artifacts(baseline, compact, "a" * 64)
    assert passing["passed"]
    assert passing["verdict"] == "PASS_COMPACT_TYPED_OUTPUT_ENCODING"

    drifted = copy.deepcopy(compact)
    drifted["golden"]["scientific_sha256"] = "drift"
    failed = benchmark._compare_artifacts(baseline, drifted, "a" * 64)
    assert not failed["passed"]
    assert not failed["gates"]["golden_parity"]["passed"]

    zero = copy.deepcopy(compact)
    zero["measurements"]["memory"]["strings_2000_max26"]["medians"][
        "tracemalloc_peak_delta_bytes"
    ] = 0
    invalid = benchmark._compare_artifacts(baseline, zero, "a" * 64)
    assert not invalid["gates"]["memory_strings_ratio_recorded"]["passed"]

    legacy = copy.deepcopy(compact)
    structure = legacy["golden"]["structural_storage"]
    structure["encoding_fields"] = ["cells", "active_factor_mask"]
    structure["retired_encoding_attributes_present"] = ["cells"]
    structure["detector_weights_present"] = True
    rejected = benchmark._compare_artifacts(baseline, legacy, "a" * 64)
    assert not rejected["gates"]["encoding_exact_stored_fields"]["passed"]
    assert not rejected["gates"]["encoding_retired_attributes_absent"]["passed"]
    assert not rejected["gates"]["detector_weights_absent"]["passed"]
