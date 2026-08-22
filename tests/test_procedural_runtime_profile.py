from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks import procedural_runtime_profile as profile


def _protocol() -> dict[str, object]:
    return json.loads(Path(profile.PROTOCOL_PATH).read_bytes())


def test_cli_defaults_to_a_nonwriting_nonexecuting_plan(
    capsys,
    monkeypatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("plan mode crossed an execution seam")

    monkeypatch.setattr(profile, "_spawn_worker", unexpected)
    monkeypatch.setattr(profile, "_create_evidence_root", unexpected)
    monkeypatch.setattr(profile, "_run_worker_row", unexpected)
    monkeypatch.setattr(profile, "_loaded_binary_identity", unexpected)
    monkeypatch.setattr(profile, "_prepare_native_row", unexpected)

    assert profile.main([]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["will_execute"] is False
    assert plan["protocol_sha256"] == profile.protocol_sha256()
    assert plan["source_bundle_sha256"] == profile.source_bundle()["bundle_sha256"]
    assert plan["worker_rows"] == 32
    assert plan["material_profile_executed"] is False


def test_plan_cli_is_self_bootstrapping_from_an_unrelated_working_directory(
    tmp_path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(profile.__file__).resolve())],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(completed.stdout)
    assert plan["will_execute"] is False
    assert plan["material_profile_executed"] is False
    assert plan["worker_rows"] == 32


def test_plan_runner_disables_python_bytecode_writes() -> None:
    assert profile.sys.dont_write_bytecode is True


def test_driven_workload_is_mechanically_identical_and_collision_free_at_both_n() -> None:
    proof = profile.validate_workload_derivations(_protocol())

    assert proof["external_events_identical_across_n"] is True
    assert proof["external_events_sha256"] == (
        "1307653946f13a353f821023dfbe8b76f237913c580bac1465dd75cc486c334d"
    )
    assert proof["configured_cells"] == {
        "1000000": {
            "ordered_rows_sha256": (
                "61bad86f26cb36f6e4c3d291b9475ff9ef0cd71e4b08a66a63c53e247f8f6873"
            ),
            "target_union_sha256": (
                "ab80f09192b785f0f2062720124be638459716fbef587763b5f4ebeecddcbebe"
            ),
            "target_union_size": 64,
            "source_and_target_union_size": 80,
        },
        "100000000000": {
            "ordered_rows_sha256": (
                "ac8e1bf43541476c1647d942132569e03c1d62fa5f09f63d791101a9deed8e87"
            ),
            "target_union_sha256": (
                "5b2fa611999461482ab0c5d899960a2ca079651cea2360d7cca00f34268df7b9"
            ),
            "target_union_size": 64,
            "source_and_target_union_size": 80,
        },
    }


def test_source_bound_production_constructor_has_no_configured_n_allocation_or_scan() -> None:
    proof = profile.validate_sparse_source_invariants()

    assert proof["passed"] is True
    assert proof["configured_ids_scanned_is_empirical"] is False
    assert proof["required_capacity_allocations"] == [
        "reserved_vec(config.resident_capacity)?",
        "reserved_vec(config.event_capacity)?",
        "reserved_vec(config.spike_capacity)?",
        "RuntimeScratch::try_new(&config)?",
    ]
    assert proof["forbidden_production_patterns"] == [
        "reserved_vec(config.configured_cells",
        "0..config.configured_cells",
        "Vec::with_capacity(config.configured_cells",
    ]
    assert proof["configured_population_uses"] == [
        "event.target_gid >= self.config.configured_cells",
        "event.source_gid >= self.config.configured_cells",
        "bounded_index(word, config.configured_cells - 1)",
    ]


def test_protocol_freezes_exact_runtime_costs_rows_and_nonclaims() -> None:
    protocol = _protocol()

    profile.validate_protocol(protocol)
    assert hashlib.sha256(Path(profile.PROTOCOL_PATH).read_bytes()).hexdigest() == (
        "a9639593cb38dfdced12dc0cb0618e712cdec20f7f373d1aec2a7ead497a9533"
    )

    assert len(protocol["execution"]["worker_order"]) == 32
    assert protocol["runtime"] == {
        "topology_protocol": "random123-philox2x64-10-v2",
        "topology": "uniform_no_self",
        "seed": 71,
        "resident_capacity": 96,
        "event_capacity": 96,
        "spike_capacity": 32,
        "fanout": 4,
        "interval_distinct_capacity": 96,
        "recurrent_weight_pa": 325.0,
        "recurrent_delay_steps": 1,
        "cost_model": (
            "O(T*R + event sorting/merging/delivery + S*F row generation) work "
            "and O(R_capacity + Q_capacity + S_capacity + F + D_capacity) "
            "storage, with configured N retained only as uint64 address metadata."
        ),
    }
    assert all(
        protocol["claims"][key] is None
        for key in (
            "speed",
            "throughput",
            "capability",
            "resident_100_billion_cells",
            "updated_100_billion_cells",
        )
    )
    assert protocol["status"]["executed"] is False

    drifted = deepcopy(protocol)
    drifted["execution"]["worker_order"][0], drifted["execution"]["worker_order"][1] = (
        drifted["execution"]["worker_order"][1],
        drifted["execution"]["worker_order"][0],
    )
    with pytest.raises(ValueError, match="worker order"):
        profile.validate_protocol(drifted)

    weakened = deepcopy(protocol)
    weakened["claims"]["hypothesis"] = "A pass proves a 100B-cell simulation."
    weakened["evidence_scope"] = "General 100B-cell capability evidence"
    with pytest.raises(ValueError, match="claim boundary"):
        profile.validate_protocol(weakened)


def test_protocol_rejects_a_self_consistent_driven_source_retune() -> None:
    from humanbrain.procedural_topology import ProceduralTopology, TopologyManifest

    retuned = deepcopy(_protocol())
    runtime = retuned["runtime"]
    driven = retuned["workloads"]["driven"]
    driven["source_gids"][0] = 12
    events = [
        {
            "due_step": 0,
            "target_gid": source,
            "source_gid": source,
            "edge_slot": 0,
            "current_pa": driven["current_pa"],
        }
        for source in driven["source_gids"]
    ]
    driven["external_events_sha256"] = profile._canonical_sha256(events)
    derivations: dict[str, object] = {}
    for configured_cells in retuned["execution"]["configured_cells"]:
        topology = ProceduralTopology(
            TopologyManifest(
                cell_count=configured_cells,
                fanout=runtime["fanout"],
                topology=runtime["topology"],
                seed=runtime["seed"],
            )
        )
        rows = [
            {
                "source_gid": source,
                "targets": [
                    edge.target_gid for edge in topology.base_edges(source)
                ],
            }
            for source in driven["source_gids"]
        ]
        targets = {target for row in rows for target in row["targets"]}
        derivations[str(configured_cells)] = {
            "ordered_rows_sha256": profile._canonical_sha256(
                {
                    "configured_cells": configured_cells,
                    "fanout": runtime["fanout"],
                    "seed": runtime["seed"],
                    "rows": rows,
                }
            ),
            "target_union_sha256": profile._canonical_sha256(sorted(targets)),
            "target_union_size": len(targets),
            "source_and_target_union_size": len(
                set(driven["source_gids"]) | targets
            ),
        }
    driven["topology_derivations"] = derivations

    with pytest.raises(ValueError, match="driven workload contract changed"):
        profile.validate_protocol(retuned)


def test_protocol_binds_every_result_affecting_source_byte() -> None:
    protocol = _protocol()
    actual = profile.source_bundle()

    assert protocol["source_binding"]["files"] == actual["files"]
    assert protocol["source_binding"]["bundle_sha256"] == actual["bundle_sha256"]


def test_execute_rejects_wrong_protocol_sha_before_execution_or_artifact_mutation(
    monkeypatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("authorization guard crossed an execution seam")

    monkeypatch.setattr(profile, "_spawn_worker", unexpected)
    monkeypatch.setattr(profile, "_create_evidence_root", unexpected)
    monkeypatch.setattr(profile, "_run_worker_row", unexpected)
    monkeypatch.setattr(profile, "_loaded_binary_identity", unexpected)
    monkeypatch.setattr(profile, "_prepare_native_row", unexpected)

    with pytest.raises(profile.ProfileExecutionRefused, match="exact raw protocol"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                "0" * 64,
                "--source-bundle-sha256",
                str(profile.source_bundle()["bundle_sha256"]),
            ]
        )


def test_worker_mode_requires_execute_before_any_worker_guard_or_runtime(
    monkeypatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("worker --execute guard crossed an execution seam")

    monkeypatch.setattr(profile, "_protocol_snapshot", unexpected)
    monkeypatch.setattr(profile, "_production_worker_dependencies", unexpected)
    monkeypatch.setattr(profile, "_run_worker_row", unexpected)

    with pytest.raises(profile.ProfileExecutionRefused, match="requires --execute"):
        profile.main(["--worker-row", "0"])


def test_worker_mode_requires_a_parent_issued_capability_before_native_runtime(
    monkeypatch,
) -> None:
    protocol = _protocol()
    protocol_sha = "c" * 64
    source_sha = "d" * 64

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("parent authorization guard crossed the native seam")

    monkeypatch.setattr(
        profile,
        "_protocol_snapshot",
        lambda: (b"{}", protocol, protocol_sha),
    )
    monkeypatch.setattr(
        profile,
        "source_bundle",
        lambda: {"bundle_sha256": source_sha},
    )
    monkeypatch.setattr(profile, "_repository_head", lambda: "clean-head")
    monkeypatch.setattr(profile, "_production_worker_dependencies", unexpected)
    monkeypatch.setattr(profile, "_run_worker_row", unexpected)

    with pytest.raises(profile.ProfileExecutionRefused, match="parent authorization"):
        profile.main(
            [
                "--execute",
                "--worker-row",
                "0",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
                "--expected-head",
                "clean-head",
                "--expected-binary-size-bytes",
                "456",
                "--expected-binary-sha256",
                "a" * 64,
                "--expected-native-source-digest",
                "b" * 64,
            ]
        )


def test_execute_rejects_wrong_source_sha_before_execution_or_artifact_mutation(
    monkeypatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("source authorization crossed an execution seam")

    monkeypatch.setattr(profile, "_spawn_worker", unexpected)
    monkeypatch.setattr(profile, "_create_evidence_root", unexpected)
    monkeypatch.setattr(profile, "_run_worker_row", unexpected)
    monkeypatch.setattr(profile, "_loaded_binary_identity", unexpected)
    monkeypatch.setattr(profile, "_prepare_native_row", unexpected)

    with pytest.raises(profile.ProfileExecutionRefused, match="exact source bundle"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                profile.protocol_sha256(),
                "--source-bundle-sha256",
                "0" * 64,
            ]
        )


def test_loaded_binary_identity_uses_the_neutral_portable_native_record(
    monkeypatch,
) -> None:
    from humanbrain import native as native_module

    record = {
        "binary_sha256": "a" * 64,
        "binary_size": 456,
        "source_digest": "b" * 64,
    }
    identity = SimpleNamespace(to_record=lambda: record)
    monkeypatch.setattr(
        native_module,
        "loaded_library_identity",
        lambda: identity,
        raising=False,
    )

    assert profile._loaded_binary_identity() == record


def test_production_worker_spawn_creates_one_fixed_subprocess_per_row(
    monkeypatch,
) -> None:
    protocol = _protocol()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="{}")

    monkeypatch.setattr(profile.subprocess, "run", run)
    identity = {
        "binary_sha256": "a" * 64,
        "binary_size": 456,
        "source_digest": "b" * 64,
    }
    for row in protocol["execution"]["worker_order"][:2]:
        assert profile._spawn_worker(
            row,
            "c" * 64,
            "d" * 64,
            "clean-head",
            identity,
            "f" * 64,
        ) == {}

    assert len(calls) == 2
    for ordinal, (command, kwargs) in enumerate(calls):
        assert command[0] == profile.sys.executable
        assert command[1] == str(Path(profile.__file__).resolve())
        assert "--execute" in command
        assert command[command.index("--worker-row") + 1] == str(ordinal)
        assert kwargs["cwd"] == profile.ROOT
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["HIP_VISIBLE_DEVICES"] == ""
        assert kwargs["env"][profile._WORKER_CAPABILITY_ENV] == "f" * 64
        assert "--expected-parent-pid" in command


def _quiet_accounting(configured_cells: int, native_payload_bytes: int) -> dict[str, object]:
    return {
        "claim": {"label": "procedural-runtime-correctness", "text": "bounded"},
        "interval": {
            "name": "native-procedural-advance",
            "start_step": 0,
            "step_count": 65_536,
            "biological_step_seconds": 0.00005,
            "wall_seconds": 0.0,
        },
        "cells": {
            "configured": configured_cells,
            "addressable": configured_cells,
            "resident_peak": 0,
            "resident_distinct": 0,
            "updated_distinct": 0,
            "spiking_distinct": 0,
        },
        "work": {
            "cell_step_evaluations": 0,
            "spike_events": 0,
            "analytic_catchups": 0,
            "analytic_step_equivalents": 0,
        },
        "edges": {
            "configured": configured_cells * 4,
            "resident": 0,
            "mutable": 0,
            "generated": 0,
        },
        "events": {
            "delivered": 0,
            "queue_peak": 0,
            "sent_bytes_by_shard_pair": [],
        },
        "memory": {
            "host_bytes": native_payload_bytes,
            "device_bytes": 0,
            "bytes_by_shard": [
                {
                    "shard": "cpu-native-procedural",
                    "host_bytes": native_payload_bytes,
                    "device_bytes": 0,
                }
            ],
        },
    }


def _accounting_for_row(row: dict[str, object], native_payload_bytes: int) -> dict[str, object]:
    configured_cells = int(row["configured_cells"])
    case = str(row["case"])
    protocol = _protocol()
    expected = protocol["workloads"][case]["expected"]
    step_count = protocol["workloads"][case]["step_count"]
    return {
        "claim": {"label": "procedural-runtime-correctness", "text": "bounded"},
        "interval": {
            "name": "native-procedural-advance",
            "start_step": 0,
            "step_count": step_count,
            "biological_step_seconds": 0.00005,
            "wall_seconds": 0.0,
        },
        "cells": {
            "configured": configured_cells,
            "addressable": configured_cells,
            "resident_peak": expected["resident_peak"],
            "resident_distinct": expected["resident_distinct"],
            "updated_distinct": expected["updated_distinct"],
            "spiking_distinct": expected["spiking_distinct"],
        },
        "work": {
            "cell_step_evaluations": expected["cell_step_evaluations"],
            "spike_events": expected["spike_events"],
            "analytic_catchups": 0,
            "analytic_step_equivalents": 0,
        },
        "edges": {
            "configured": configured_cells * 4,
            "resident": 0,
            "mutable": 0,
            "generated": expected["generated_edges"],
        },
        "events": {
            "delivered": expected["delivered_events"],
            "queue_peak": expected["queue_peak"],
            "sent_bytes_by_shard_pair": [],
        },
        "memory": {
            "host_bytes": native_payload_bytes,
            "device_bytes": 0,
            "bytes_by_shard": [
                {
                    "shard": "cpu-native-procedural",
                    "host_bytes": native_payload_bytes,
                    "device_bytes": 0,
                }
            ],
        },
    }


def _successful_worker_result(
    row: dict[str, object],
    *,
    protocol_sha: str,
    source_sha: str,
    head: str,
    binary_identity: dict[str, object],
) -> dict[str, object]:
    accounting = _accounting_for_row(row, 12_345)
    return {
        "row": row,
        "protocol_sha256": protocol_sha,
        "source_bundle_sha256": source_sha,
        "git_head": head,
        "backend": "cpu-native-rust",
        "environment": {
            "backend": "cpu-native-rust",
            "python_implementation": "CPython",
            "python_version": "3.14.0",
            "machine": "AMD64",
            "processor": "test-cpu",
            "platform": "test-platform",
            "logical_cpu_count": 8,
            "numpy_version": "2.3.0",
            "cuda_visible_devices": "",
            "hip_visible_devices": "",
            "rayon_num_threads": None,
        },
        "loaded_binary_identity": binary_identity,
        "timing": {"native_call_ns": 100, "e2e_ns": 200},
        "scale_accounting": accounting,
        "native_payload_bytes": 12_345,
        "retained_nonzero_cells": _protocol()["workloads"][str(row["case"])][
            "expected"
        ]["retained_nonzero_cells"],
        "process_memory": {
            "before_runtime": {
                "working_set_bytes": 10,
                "peak_working_set_bytes": 12,
                "private_memory_bytes": 8,
            },
            "after_construction": {
                "working_set_bytes": 11,
                "peak_working_set_bytes": 12,
                "private_memory_bytes": 9,
            },
            "after_advance": {
                "working_set_bytes": 12,
                "peak_working_set_bytes": 13,
                "private_memory_bytes": 10,
            },
            "private_delta_bytes": 2,
        },
        "capacity_overflowed": False,
    }


def test_worker_times_only_advance_and_reports_raw_accounting_and_memory(
    capsys,
    monkeypatch,
) -> None:
    calls: list[str] = []
    memories = iter(
        (
            {"working_set_mib": 10.0, "peak_working_set_mib": 12.0, "private_memory_mib": 8.0},
            {"working_set_mib": 11.0, "peak_working_set_mib": 12.0, "private_memory_mib": 9.0},
            {"working_set_mib": 12.0, "peak_working_set_mib": 13.0, "private_memory_mib": 10.0},
        )
    )
    ticks = iter((1_000, 1_060))
    accounting = _quiet_accounting(1_000_000, 12_345)

    class Prepared:
        def advance(self) -> object:
            calls.append("advance")
            return SimpleNamespace(
                accounting=SimpleNamespace(to_dict=lambda: accounting),
                native_call_ns=37,
                retained_nonzero_cells=0,
                capacity_overflowed=False,
            )

        def close(self) -> None:
            calls.append("close")

    def memory() -> dict[str, float]:
        calls.append("memory")
        return next(memories)

    dependencies = profile._WorkerDependencies(
        binary_identity=lambda: {
            "binary_size": 456,
            "binary_sha256": "a" * 64,
            "source_digest": "b" * 64,
        },
        environment_identity=lambda: {
            "backend": "cpu-native-rust",
            "python_implementation": "CPython",
            "python_version": "3.14.0",
            "machine": "AMD64",
            "processor": "test-cpu",
            "platform": "test-platform",
            "logical_cpu_count": 8,
            "numpy_version": "2.3.0",
            "cuda_visible_devices": "",
            "hip_visible_devices": "",
            "rayon_num_threads": None,
        },
        memory_observation=memory,
        prepare=lambda _protocol, _row: (calls.append("prepare") or Prepared()),
        clock_ns=lambda: (calls.append("clock") or next(ticks)),
    )
    monkeypatch.setattr(profile, "_production_worker_dependencies", lambda: dependencies)
    monkeypatch.setattr(profile, "_repository_head", lambda: "clean-head")
    monkeypatch.setattr(
        profile, "_validate_parent_worker_authorization", lambda **_kwargs: None
    )

    assert (
        profile.main(
            [
                "--execute",
                "--worker-row",
                "0",
                "--protocol-sha256",
                profile.protocol_sha256(),
                "--source-bundle-sha256",
                str(profile.source_bundle()["bundle_sha256"]),
                "--expected-head",
                "clean-head",
                "--expected-binary-size-bytes",
                "456",
                "--expected-binary-sha256",
                "a" * 64,
                "--expected-native-source-digest",
                "b" * 64,
            ]
        )
        == 0
    )

    row = json.loads(capsys.readouterr().out)
    assert row["row"] == {
        "ordinal": 0,
        "repetition": 0,
        "case": "quiet",
        "configured_cells": 1_000_000,
    }
    assert row["timing"] == {"native_call_ns": 37, "e2e_ns": 60}
    assert row["scale_accounting"] == accounting
    assert row["native_payload_bytes"] == 12_345
    assert row["process_memory"]["private_delta_bytes"] == 2 * 1024 * 1024
    assert row["loaded_binary_identity"] == {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    assert calls == [
        "memory",
        "prepare",
        "memory",
        "clock",
        "advance",
        "clock",
        "memory",
        "close",
    ]


def test_worker_rejects_loaded_binary_mismatch_before_memory_or_runtime(
    monkeypatch,
) -> None:
    def unexpected(*_args: object) -> object:
        raise AssertionError("binary identity guard crossed the measurement seam")

    dependencies = profile._WorkerDependencies(
        binary_identity=lambda: {
            "binary_size": 456,
            "binary_sha256": "c" * 64,
            "source_digest": "b" * 64,
        },
        environment_identity=unexpected,
        memory_observation=unexpected,
        prepare=unexpected,
        clock_ns=unexpected,
    )
    monkeypatch.setattr(profile, "_production_worker_dependencies", lambda: dependencies)
    monkeypatch.setattr(profile, "_repository_head", lambda: "clean-head")
    monkeypatch.setattr(
        profile, "_validate_parent_worker_authorization", lambda **_kwargs: None
    )

    with pytest.raises(profile.ProfileExecutionRefused, match="binary identity differs"):
        profile.main(
            [
                "--execute",
                "--worker-row",
                "0",
                "--protocol-sha256",
                profile.protocol_sha256(),
                "--source-bundle-sha256",
                str(profile.source_bundle()["bundle_sha256"]),
                "--expected-head",
                "clean-head",
                "--expected-binary-size-bytes",
                "456",
                "--expected-binary-sha256",
                "a" * 64,
                "--expected-native-source-digest",
                "b" * 64,
            ]
        )


def test_worker_rejects_loaded_binary_drift_after_advance_before_reporting(
    monkeypatch,
) -> None:
    original = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    drifted = {**original, "binary_sha256": "c" * 64}
    identity_calls = 0
    accounting = _quiet_accounting(1_000_000, 12_345)

    class Prepared:
        def advance(self) -> object:
            return SimpleNamespace(
                accounting=SimpleNamespace(to_dict=lambda: accounting),
                native_call_ns=37,
                retained_nonzero_cells=0,
                capacity_overflowed=False,
            )

        def close(self) -> None:
            return None

    def binary_identity() -> dict[str, object]:
        nonlocal identity_calls
        identity_calls += 1
        return original if identity_calls == 1 else drifted

    memories = iter(
        (
            {"working_set_mib": 10.0, "peak_working_set_mib": 12.0, "private_memory_mib": 8.0},
            {"working_set_mib": 11.0, "peak_working_set_mib": 12.0, "private_memory_mib": 9.0},
            {"working_set_mib": 12.0, "peak_working_set_mib": 13.0, "private_memory_mib": 10.0},
        )
    )
    ticks = iter((1_000, 1_060))
    dependencies = profile._WorkerDependencies(
        binary_identity=binary_identity,
        environment_identity=lambda: {"backend": "cpu-native-rust"},
        memory_observation=lambda: next(memories),
        prepare=lambda _protocol, _row: Prepared(),
        clock_ns=lambda: next(ticks),
    )
    monkeypatch.setattr(profile, "_production_worker_dependencies", lambda: dependencies)
    monkeypatch.setattr(profile, "_repository_head", lambda: "clean-head")
    monkeypatch.setattr(
        profile, "_validate_parent_worker_authorization", lambda **_kwargs: None
    )

    with pytest.raises(profile.ProfileExecutionRefused, match="changed during worker"):
        profile.main(
            [
                "--execute",
                "--worker-row",
                "0",
                "--protocol-sha256",
                profile.protocol_sha256(),
                "--source-bundle-sha256",
                str(profile.source_bundle()["bundle_sha256"]),
                "--expected-head",
                "clean-head",
                "--expected-binary-size-bytes",
                "456",
                "--expected-binary-sha256",
                "a" * 64,
                "--expected-native-source-digest",
                "b" * 64,
            ]
        )


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda record: record["scale_accounting"]["interval"].__setitem__(
            "start_step", 1
        ),
        lambda record: record["scale_accounting"]["events"].__setitem__(
            "sent_bytes_by_shard_pair", [{"source": "cpu", "target": "gpu", "bytes": 1}]
        ),
        lambda record: record["scale_accounting"]["memory"]["bytes_by_shard"][0].__setitem__(
            "host_bytes", 12_344
        ),
        lambda record: record["process_memory"].__setitem__(
            "private_delta_bytes", 999
        ),
    ],
)
def test_parent_rejects_incomplete_or_inconsistent_accounting_observations(
    corrupt,
) -> None:
    protocol = _protocol()
    row = protocol["execution"]["worker_order"][0]
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    record = _successful_worker_result(
        row,
        protocol_sha="d" * 64,
        source_sha="e" * 64,
        head="clean-head",
        binary_identity=binary_identity,
    )
    corrupt(record)

    with pytest.raises(ValueError, match="ScaleAccounting|memory observation"):
        profile._validate_worker_record(
            record=record,
            row=row,
            protocol=protocol,
            protocol_sha="d" * 64,
            source_sha="e" * 64,
            head="clean-head",
            binary_identity=binary_identity,
        )


def test_parent_rejects_a_gpu_visible_or_incomplete_worker_environment() -> None:
    protocol = _protocol()
    row = protocol["execution"]["worker_order"][0]
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    record = _successful_worker_result(
        row,
        protocol_sha="d" * 64,
        source_sha="e" * 64,
        head="clean-head",
        binary_identity=binary_identity,
    )
    record["environment"] = {
        "backend": "cpu-native-rust",
        "cuda_visible_devices": "0",
    }

    with pytest.raises(ValueError, match="worker environment"):
        profile._validate_worker_record(
            record=record,
            row=row,
            protocol=protocol,
            protocol_sha="d" * 64,
            source_sha="e" * 64,
            head="clean-head",
            binary_identity=binary_identity,
        )


def test_parent_rejects_an_unregistered_worker_record_field() -> None:
    protocol = _protocol()
    row = protocol["execution"]["worker_order"][0]
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    record = _successful_worker_result(
        row,
        protocol_sha="d" * 64,
        source_sha="e" * 64,
        head="clean-head",
        binary_identity=binary_identity,
    )
    record["unexpected_counter"] = 0

    with pytest.raises(ValueError, match="worker record"):
        profile._validate_worker_record(
            record=record,
            row=row,
            protocol=protocol,
            protocol_sha="d" * 64,
            source_sha="e" * 64,
            head="clean-head",
            binary_identity=binary_identity,
        )


def test_execute_uses_32_fresh_workers_in_frozen_order_and_writes_incrementally(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol = _protocol()
    protocol_sha = profile.protocol_sha256()
    source_sha = str(profile.source_bundle()["bundle_sha256"])
    head = "clean-profile-head"
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    spawned: list[dict[str, object]] = []
    capabilities_seen: list[str] = []

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*"))
            if path.is_file()
        )

    def spawn(
        row: dict[str, object],
        passed_protocol_sha: str,
        passed_source_sha: str,
        passed_head: str,
        passed_binary_identity: dict[str, object],
        worker_capability: str,
    ) -> dict[str, object]:
        profile._validate_parent_worker_authorization(
            row=row,
            protocol=protocol,
            protocol_sha=passed_protocol_sha,
            source_sha=passed_source_sha,
            git_head=passed_head,
            binary_identity=passed_binary_identity,
            expected_parent_pid=os.getppid(),
            capability=worker_capability,
        )
        spawned.append(row)
        capabilities_seen.append(worker_capability)
        return _successful_worker_result(
            row,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            head=head,
            binary_identity=binary_identity,
        )

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=lambda: binary_identity,
        spawn_worker=spawn,
        publish_record=profile._publish_record,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(profile, "_production_parent_dependencies", lambda: dependencies)

    assert (
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert spawned == protocol["execution"]["worker_order"]
    assert len(spawned) == 32
    assert len(set(capabilities_seen)) == 32
    assert summary["profile_pass"] is True
    assert summary["protocol_sha256"] == protocol_sha
    assert summary["source_bundle_sha256"] == source_sha
    assert summary["git_head"] == head
    assert summary["loaded_binary_identity"] == binary_identity
    assert summary["environment_identity"]["backend"] == "cpu-native-rust"
    evidence = sorted((evidence_root / "evidence").glob("*.json"))
    assert len(evidence) == 34
    assert evidence[0].name == "000-run-start.json"
    assert evidence[-1].name == "033-summary.json"
    run_start_bytes = evidence[0].read_bytes()
    assert all(capability.encode("ascii") not in run_start_bytes for capability in capabilities_seen)


def test_worker_failure_writes_one_terminal_record_and_permanently_consumes_root(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol_sha = profile.protocol_sha256()
    source_sha = str(profile.source_bundle()["bundle_sha256"])
    head = "clean-profile-head"
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*.json"))
        )

    def fail_worker(*_args: object) -> dict[str, object]:
        raise RuntimeError("fixed worker failed")

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=lambda: binary_identity,
        spawn_worker=fail_worker,
        publish_record=profile._publish_record,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(profile, "_production_parent_dependencies", lambda: dependencies)

    with pytest.raises(RuntimeError, match="fixed worker failed"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
            ]
        )

    evidence = sorted((evidence_root / "evidence").glob("*.json"))
    assert [path.name for path in evidence] == [
        "000-run-start.json",
        "001-terminal-failure.json",
    ]
    failure = json.loads(evidence[-1].read_bytes())["payload"]
    assert failure["event"] == "terminal-failure"
    assert failure["failed_worker_row"] == 0
    assert failure["protocol_sha256"] == protocol_sha
    assert failure["source_bundle_sha256"] == source_sha
    assert failure["git_head"] == head
    assert failure["loaded_binary_identity"] == binary_identity
    assert failure["retry_authorized"] is False

    with pytest.raises(FileExistsError, match="already exists"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
            ]
        )


def test_terminal_failure_publication_is_followed_by_an_integrity_recheck(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol = _protocol()
    protocol_sha = "c" * 64
    source_sha = "d" * 64
    head = "clean-profile-head"
    original = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    drifted = {**original, "binary_sha256": "e" * 64}
    failure_published = False

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*.json"))
        )

    def binary_identity() -> dict[str, object]:
        return drifted if failure_published else original

    def fail_worker(*_args: object) -> dict[str, object]:
        raise RuntimeError("fixed worker failed")

    def publish(path: Path, label: str, payload: dict[str, object]) -> str:
        nonlocal failure_published
        digest = profile._publish_record(path, label, payload)
        if label == "terminal-failure":
            failure_published = True
        return digest

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=binary_identity,
        spawn_worker=fail_worker,
        publish_record=publish,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(
        profile, "source_bundle", lambda: {"bundle_sha256": source_sha}
    )

    with pytest.raises(profile.ProfileExecutionRefused, match="binary identity changed"):
        profile._execute_profile(
            protocol=protocol,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            dependencies=dependencies,
        )

    assert (
        evidence_root / "evidence" / "001-terminal-failure.json"
    ).is_file()


def test_dirty_path_outside_evidence_root_fails_before_binary_or_root_creation(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol_sha = profile.protocol_sha256()
    source_sha = str(profile.source_bundle()["bundle_sha256"])

    def unexpected(*_args: object) -> object:
        raise AssertionError("dirty-tree guard crossed a mutation or binary seam")

    dependencies = profile._ParentDependencies(
        repository_head=lambda: "clean-profile-head",
        repository_status=lambda: ((" M", profile.ROOT / "unexpected.py"),),
        binary_identity=unexpected,
        spawn_worker=unexpected,
        publish_record=unexpected,
        create_evidence_root=unexpected,
    )
    monkeypatch.setattr(profile, "_production_parent_dependencies", lambda: dependencies)

    with pytest.raises(profile.ProfileExecutionRefused, match="completely clean HEAD"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
            ]
        )

    assert not evidence_root.exists()


def test_unexpected_file_inside_evidence_root_fails_before_first_worker_or_record(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol_sha = profile.protocol_sha256()
    source_sha = str(profile.source_bundle()["bundle_sha256"])
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }

    def status() -> tuple[tuple[str, Path], ...]:
        if evidence_root.exists():
            return (("??", (evidence_root / "intruder.json").resolve()),)
        return ()

    def unexpected(*_args: object) -> object:
        raise AssertionError("unexpected-file guard crossed worker/publication")

    dependencies = profile._ParentDependencies(
        repository_head=lambda: "clean-profile-head",
        repository_status=status,
        binary_identity=lambda: binary_identity,
        spawn_worker=unexpected,
        publish_record=unexpected,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(profile, "_production_parent_dependencies", lambda: dependencies)

    with pytest.raises(profile.ProfileExecutionRefused, match="unexpected evidence"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
            ]
        )

    assert evidence_root.exists()
    assert not tuple((evidence_root / "evidence").iterdir())


def test_loaded_binary_drift_after_worker_fails_before_worker_record_publication(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol_sha = profile.protocol_sha256()
    source_sha = str(profile.source_bundle()["bundle_sha256"])
    head = "clean-profile-head"
    original = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    drifted = {**original, "binary_sha256": "c" * 64}
    identity_calls = 0

    def binary_identity() -> dict[str, object]:
        nonlocal identity_calls
        identity_calls += 1
        return original if identity_calls <= 3 else drifted

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*.json"))
        )

    def spawn(row: dict[str, object], *_args: object) -> dict[str, object]:
        return _successful_worker_result(
            row,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            head=head,
            binary_identity=original,
        )

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=binary_identity,
        spawn_worker=spawn,
        publish_record=profile._publish_record,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(profile, "_production_parent_dependencies", lambda: dependencies)

    with pytest.raises(profile.ProfileExecutionRefused, match="binary identity changed"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
            ]
        )

    assert [
        path.name for path in sorted((evidence_root / "evidence").glob("*.json"))
    ] == ["000-run-start.json"]


def test_source_drift_after_worker_fails_before_worker_record_publication(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol = _protocol()
    protocol_sha = "c" * 64
    source_sha = "d" * 64
    head = "clean-profile-head"
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    source_calls = 0

    def bundle() -> dict[str, object]:
        nonlocal source_calls
        source_calls += 1
        return {
            "bundle_sha256": source_sha if source_calls <= 3 else "e" * 64
        }

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*.json"))
        )

    def spawn(row: dict[str, object], *_args: object) -> dict[str, object]:
        return _successful_worker_result(
            row,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            head=head,
            binary_identity=binary_identity,
        )

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=lambda: binary_identity,
        spawn_worker=spawn,
        publish_record=profile._publish_record,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(profile, "source_bundle", bundle)

    with pytest.raises(profile.ProfileExecutionRefused, match="source bundle changed"):
        profile._execute_profile(
            protocol=protocol,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            dependencies=dependencies,
        )

    assert [
        path.name for path in sorted((evidence_root / "evidence").glob("*.json"))
    ] == ["000-run-start.json"]


def test_in_place_evidence_tampering_is_detected_even_when_git_status_is_unchanged(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol = _protocol()
    protocol_sha = "c" * 64
    source_sha = "d" * 64
    head = "clean-profile-head"
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    spawn_count = 0

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*.json"))
        )

    def spawn(row: dict[str, object], *_args: object) -> dict[str, object]:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count != 1:
            raise AssertionError("tampered run-start reached another worker")
        return _successful_worker_result(
            row,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            head=head,
            binary_identity=binary_identity,
        )

    def publish(path: Path, label: str, payload: dict[str, object]) -> str:
        digest = profile._publish_record(path, label, payload)
        if label == "worker":
            run_start = evidence_root / "evidence" / "000-run-start.json"
            run_start.write_bytes(run_start.read_bytes() + b"\n")
        return digest

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=lambda: binary_identity,
        spawn_worker=spawn,
        publish_record=publish,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(
        profile, "source_bundle", lambda: {"bundle_sha256": source_sha}
    )

    with pytest.raises(profile.ProfileExecutionRefused, match="evidence bytes changed"):
        profile._execute_profile(
            protocol=protocol,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            dependencies=dependencies,
        )

    assert spawn_count == 1


def test_final_summary_publication_is_followed_by_an_integrity_recheck(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol = _protocol()
    protocol_sha = "c" * 64
    source_sha = "d" * 64
    head = "clean-profile-head"
    original = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }
    drifted = {**original, "binary_sha256": "e" * 64}
    summary_published = False

    def bundle() -> dict[str, object]:
        return {"bundle_sha256": source_sha}

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*.json"))
        )

    def binary_identity() -> dict[str, object]:
        return drifted if summary_published else original

    def spawn(row: dict[str, object], *_args: object) -> dict[str, object]:
        return _successful_worker_result(
            row,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            head=head,
            binary_identity=original,
        )

    def publish(path: Path, label: str, payload: dict[str, object]) -> str:
        nonlocal summary_published
        digest = profile._publish_record(path, label, payload)
        if label == "summary":
            summary_published = True
        return digest

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=binary_identity,
        spawn_worker=spawn,
        publish_record=publish,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(profile, "source_bundle", bundle)

    with pytest.raises(profile.ProfileExecutionRefused, match="binary identity changed"):
        profile._execute_profile(
            protocol=protocol,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            dependencies=dependencies,
        )

    assert (evidence_root / "evidence" / "033-summary.json").is_file()


def test_cross_n_timing_ratio_is_falsifiable_and_raw_summary_is_still_published(
    tmp_path,
    monkeypatch,
) -> None:
    evidence_root = tmp_path / "procedural_runtime_profile_v1"
    monkeypatch.setattr(profile, "EVIDENCE_ROOT", evidence_root)
    protocol_sha = profile.protocol_sha256()
    source_sha = str(profile.source_bundle()["bundle_sha256"])
    head = "clean-profile-head"
    binary_identity = {
        "binary_size": 456,
        "binary_sha256": "a" * 64,
        "source_digest": "b" * 64,
    }

    def status() -> tuple[tuple[str, Path], ...]:
        if not evidence_root.exists():
            return ()
        return tuple(
            ("??", path.resolve())
            for path in sorted(evidence_root.rglob("*.json"))
        )

    def spawn(row: dict[str, object], *_args: object) -> dict[str, object]:
        record = _successful_worker_result(
            row,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
            head=head,
            binary_identity=binary_identity,
        )
        if row["configured_cells"] == 100_000_000_000:
            record["timing"] = {"native_call_ns": 401, "e2e_ns": 802}
        return record

    dependencies = profile._ParentDependencies(
        repository_head=lambda: head,
        repository_status=status,
        binary_identity=lambda: binary_identity,
        spawn_worker=spawn,
        publish_record=profile._publish_record,
        create_evidence_root=profile._create_evidence_root,
    )
    monkeypatch.setattr(profile, "_production_parent_dependencies", lambda: dependencies)

    with pytest.raises(RuntimeError, match="profile gates failed"):
        profile.main(
            [
                "--execute",
                "--protocol-sha256",
                protocol_sha,
                "--source-bundle-sha256",
                source_sha,
            ]
        )

    envelope = json.loads(
        (evidence_root / "evidence" / "033-summary.json").read_bytes()
    )
    summary = envelope["payload"]
    assert summary["profile_pass"] is False
    assert summary["cases"]["quiet"]["native_call_cross_n_ratio"] == 4.01
    assert summary["cases"]["quiet"]["gates"]["native_call_ratio"] is False
