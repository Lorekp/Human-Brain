"""Preregistered CPU profile for the native procedural working-set runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Callable, Final, Sequence

sys.dont_write_bytecode = True

ROOT: Final = Path(__file__).resolve().parents[1]
for _repository_import_path in (ROOT, ROOT / "src"):
    if str(_repository_import_path) not in sys.path:
        sys.path.insert(0, str(_repository_import_path))
PROTOCOL_PATH: Final = ROOT / "benchmarks/procedural_runtime_profile_protocol_v1.json"
EVIDENCE_ROOT: Final = ROOT / "artifacts/procedural_runtime_profile_v1"
_PROTOCOL_MAX_BYTES: Final = 256 * 1024
_SOURCE_MAX_BYTES: Final = 8 * 1024 * 1024
_ARTIFACT_MAX_BYTES: Final = 2 * 1024 * 1024
_SOURCE_BUNDLE_NAMESPACE: Final = "humanbrain-procedural-runtime-profile-source-v1"
_WORKER_CAPABILITY_ENV: Final = (
    "HUMANBRAIN_PROCEDURAL_RUNTIME_PROFILE_WORKER_CAPABILITY"
)
_SOURCE_PATHS: Final = (
    "benchmarks/native_scaling.py",
    "benchmarks/procedural_runtime_profile.py",
    "data/allen/523000114_neuron_config.json",
    "pyproject.toml",
    "src/humanbrain/_native_src/Cargo.lock",
    "src/humanbrain/_native_src/Cargo.toml",
    "src/humanbrain/_native_src/src/lib.rs",
    "src/humanbrain/_native_src/src/procedural.rs",
    "src/humanbrain/__init__.py",
    "src/humanbrain/cells.py",
    "src/humanbrain/human_cell_identity.py",
    "src/humanbrain/native.py",
    "src/humanbrain/procedural_runtime.py",
    "src/humanbrain/procedural_topology.py",
    "src/humanbrain/readout.py",
    "src/humanbrain/recurrent.py",
    "src/humanbrain/recurrent_state.py",
    "src/humanbrain/scale_accounting.py",
    "src/humanbrain/simple_swe/__init__.py",
    "src/humanbrain/simple_swe/artifacts.py",
    "src/humanbrain/simple_swe/protocol.py",
    "src/humanbrain/simple_swe/tasks.py",
    "src/humanbrain/topology.py",
    "src/humanbrain/typed_program_types.py",
)
_ORDER_BLOCKS: Final = (
    (("quiet", 1_000_000), ("driven", 1_000_000), ("driven", 100_000_000_000), ("quiet", 100_000_000_000)),
    (("driven", 1_000_000), ("quiet", 100_000_000_000), ("quiet", 1_000_000), ("driven", 100_000_000_000)),
    (("quiet", 100_000_000_000), ("driven", 100_000_000_000), ("driven", 1_000_000), ("quiet", 1_000_000)),
    (("driven", 100_000_000_000), ("quiet", 1_000_000), ("quiet", 100_000_000_000), ("driven", 1_000_000)),
    (("quiet", 1_000_000), ("quiet", 100_000_000_000), ("driven", 100_000_000_000), ("driven", 1_000_000)),
    (("driven", 1_000_000), ("driven", 100_000_000_000), ("quiet", 1_000_000), ("quiet", 100_000_000_000)),
    (("quiet", 100_000_000_000), ("quiet", 1_000_000), ("driven", 1_000_000), ("driven", 100_000_000_000)),
    (("driven", 100_000_000_000), ("driven", 1_000_000), ("quiet", 100_000_000_000), ("quiet", 1_000_000)),
)
_EXPECTED_RUNTIME: Final = {
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
        "and O(R_capacity + Q_capacity + S_capacity + F + D_capacity) storage, "
        "with configured N retained only as uint64 address metadata."
    ),
}
_EXPECTED_COUNTERS: Final = {
    "quiet": {
        "resident_peak": 0,
        "resident_distinct": 0,
        "updated_distinct": 0,
        "spiking_distinct": 0,
        "cell_step_evaluations": 0,
        "spike_events": 0,
        "generated_edges": 0,
        "delivered_events": 0,
        "queue_peak": 0,
        "retained_nonzero_cells": 0,
    },
    "driven": {
        "resident_peak": 80,
        "resident_distinct": 80,
        "updated_distinct": 80,
        "spiking_distinct": 16,
        "cell_step_evaluations": 20_416,
        "spike_events": 16,
        "generated_edges": 64,
        "delivered_events": 80,
        "queue_peak": 64,
        "retained_nonzero_cells": 80,
    },
}
_EXPECTED_DRIVEN_SOURCE_GIDS: Final = (
    11,
    101,
    1009,
    10007,
    50021,
    70117,
    90001,
    123457,
    234569,
    345679,
    456791,
    567899,
    678913,
    789923,
    890987,
    999983,
)
_EXPECTED_DRIVEN_DERIVATIONS: Final = {
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
_EXPECTED_ACCEPTANCE: Final = {
    "exact_expected_accounting_every_row": True,
    "exact_cross_n_workload_counter_equality": True,
    "exact_cross_n_native_payload_equality": True,
    "zero_capacity_overflow": True,
    "zero_device_bytes": True,
    "native_call_within_e2e": True,
    "maximum_cross_n_median_native_call_ratio": 3.0,
    "maximum_cross_n_median_e2e_ratio": 3.0,
    "maximum_cross_n_median_private_delta_difference_bytes": 16_777_216,
}
_EXPECTED_PURPOSE: Final = (
    "A future bounded CPU engineering profile of whether native source-driven "
    "work and storage stay dependent on R/Q/S/F/D rather than configured "
    "address-space N. This protocol is not evidence that 100 billion cells are "
    "resident, updated, or capable."
)
_EXPECTED_EVIDENCE_SCOPE: Final = (
    "CPU-only bounded native procedural working-set engineering observations at "
    "configured N=10^6 and N=10^11; no GPU, throughput, capability, SWE, or "
    "100-billion-resident/updated-cell evidence"
)
_EXPECTED_CLAIMS: Final = {
    "hypothesis": (
        "With identical R/Q/S/F/D and externally injected workload, workload "
        "counters, native payload, and host timings should not materially depend "
        "on configured address-space N."
    ),
    "speed": None,
    "throughput": None,
    "capability": None,
    "resident_100_billion_cells": None,
    "updated_100_billion_cells": None,
}


class ProfileExecutionRefused(RuntimeError):
    """The fixed engineering profile was not authorized exactly."""


@dataclass(frozen=True)
class _WorkerDependencies:
    binary_identity: Callable[[], dict[str, object]]
    environment_identity: Callable[[], dict[str, object]]
    memory_observation: Callable[[], dict[str, float]]
    prepare: Callable[[dict[str, object], dict[str, object]], object]
    clock_ns: Callable[[], int]


@dataclass(frozen=True)
class _ParentDependencies:
    repository_head: Callable[[], str]
    repository_status: Callable[[], tuple[tuple[str, Path], ...]]
    binary_identity: Callable[[], dict[str, object]]
    spawn_worker: Callable[..., dict[str, object]]
    publish_record: Callable[[Path, str, dict[str, object]], str]
    create_evidence_root: Callable[[Path], None]


def _require_exact_keys(value: object, expected: set[str], owner: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be an object")
    if set(value) != expected:
        raise ValueError(f"{owner} must contain exactly {sorted(expected)}")
    return value


def _expected_worker_order() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repetition, block in enumerate(_ORDER_BLOCKS):
        for case, configured_cells in block:
            rows.append(
                {
                    "ordinal": len(rows),
                    "repetition": repetition,
                    "case": case,
                    "configured_cells": configured_cells,
                }
            )
    return rows


def _read_bounded(path: Path, maximum_bytes: int) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"bounded input is too large: {path}")
    return payload


def _protocol_snapshot() -> tuple[bytes, dict[str, object], str]:
    payload = _read_bounded(PROTOCOL_PATH, _PROTOCOL_MAX_BYTES)
    protocol = json.loads(payload)
    if not isinstance(protocol, dict):
        raise TypeError("profile protocol must be an object")
    if protocol.get("name") != "procedural-runtime-profile-v1":
        raise ValueError("unexpected profile protocol name")
    worker_order = protocol.get("execution", {}).get("worker_order")  # type: ignore[union-attr]
    if not isinstance(worker_order, list) or len(worker_order) != 32:
        raise ValueError("profile protocol must fix exactly 32 worker rows")
    validate_protocol(protocol)
    return payload, protocol, hashlib.sha256(payload).hexdigest()


def protocol_sha256() -> str:
    return _protocol_snapshot()[2]


def _framed_source_sha(entries: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    namespace = _SOURCE_BUNDLE_NAMESPACE.encode("utf-8")
    digest.update(len(namespace).to_bytes(8, "big"))
    digest.update(namespace)
    for relative_path, payload in entries:
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _worker_capability_sha256(capability: str) -> str:
    if (
        len(capability) != 64
        or capability.lower() != capability
        or any(character not in "0123456789abcdef" for character in capability)
    ):
        raise ValueError("worker capability must be 32 random bytes encoded as hex")
    return hashlib.sha256(capability.encode("ascii")).hexdigest()


def _new_worker_capabilities(count: int) -> tuple[str, ...]:
    capabilities = tuple(secrets.token_hex(32) for _ in range(count))
    if len(set(capabilities)) != count:
        raise RuntimeError("worker capability collision")
    return capabilities


def _validate_parent_worker_authorization(
    *,
    row: dict[str, object],
    protocol: dict[str, object],
    protocol_sha: str,
    source_sha: str,
    git_head: str,
    binary_identity: dict[str, object],
    expected_parent_pid: int | None,
    capability: str | None,
) -> None:
    refusal = "worker parent authorization is absent or invalid"
    if expected_parent_pid is None or expected_parent_pid != os.getppid():
        raise ProfileExecutionRefused(refusal)
    if capability is None:
        raise ProfileExecutionRefused(refusal)
    try:
        capability_sha = _worker_capability_sha256(capability)
        run_start_path = EVIDENCE_ROOT / "evidence" / "000-run-start.json"
        envelope = json.loads(_read_bounded(run_start_path, _ARTIFACT_MAX_BYTES))
        _require_exact_keys(
            envelope,
            {"schema", "kind", "payload_sha256", "payload"},
            "run-start envelope",
        )
        if (
            envelope["schema"] != "humanbrain.canonical-json-artifact.v1"
            or envelope["kind"] != "procedural-runtime-profile-record"
        ):
            raise ValueError("run-start envelope identity differs")
        payload = envelope["payload"]
        _require_exact_keys(
            payload,
            {
                "event",
                "protocol_sha256",
                "source_bundle_sha256",
                "source_binding",
                "git_head",
                "loaded_binary_identity",
                "worker_order",
                "worker_capability_sha256_by_ordinal",
                "workload_derivation_proof",
                "sparse_source_invariant_proof",
                "material_profile_started",
                "completed_worker_rows",
            },
            "run-start payload",
        )
        if envelope["payload_sha256"] != _canonical_sha256(payload):
            raise ValueError("run-start payload digest differs")
        if (
            payload["event"] != "run-start"
            or payload["protocol_sha256"] != protocol_sha
            or payload["source_bundle_sha256"] != source_sha
            or payload["git_head"] != git_head
            or payload["loaded_binary_identity"] != binary_identity
            or payload["worker_order"]
            != protocol["execution"]["worker_order"]  # type: ignore[index]
            or payload["material_profile_started"] is not True
            or payload["completed_worker_rows"] != 0
        ):
            raise ValueError("run-start is not bound to this worker")
        authorizations = payload["worker_capability_sha256_by_ordinal"]
        rows = protocol["execution"]["worker_order"]  # type: ignore[index]
        if not isinstance(authorizations, list) or len(authorizations) != len(rows):
            raise ValueError("worker authorization count differs")
        for ordinal, (authorization, expected_row) in enumerate(
            zip(authorizations, rows, strict=True)
        ):
            _require_exact_keys(
                authorization,
                {"ordinal", "sha256"},
                "worker authorization",
            )
            if (
                authorization["ordinal"] != ordinal
                or expected_row["ordinal"] != ordinal
                or not isinstance(authorization["sha256"], str)
                or len(authorization["sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in authorization["sha256"]
                )
            ):
                raise ValueError("worker authorization order differs")
        row_ordinal = int(row["ordinal"])
        if authorizations[row_ordinal]["sha256"] != capability_sha:
            raise ValueError("worker capability digest differs")
        evidence_directory = EVIDENCE_ROOT / "evidence"
        expected_names = {"000-run-start.json"}
        expected_names.update(
            f"{position + 1:03d}-worker-{int(previous['ordinal']):02d}.json"
            for position, previous in enumerate(rows[:row_ordinal])
        )
        actual_names = {path.name for path in evidence_directory.iterdir()}
        if actual_names != expected_names:
            raise ValueError("worker evidence prefix is not the exact frozen order")
    except ProfileExecutionRefused:
        raise
    except Exception as error:
        raise ProfileExecutionRefused(refusal) from error


def validate_sparse_source_invariants() -> dict[str, object]:
    """Statically bind production storage to capacities, never configured N."""

    source_path = ROOT / "src/humanbrain/_native_src/src/procedural.rs"
    source_bytes = _read_bounded(source_path, _SOURCE_MAX_BYTES)
    source = source_bytes.decode("utf-8")
    start = source.index("struct ProceduralRuntime {")
    end = source.index("struct DenseReference {", start)
    production = source[start:end]
    required = [
        "reserved_vec(config.resident_capacity)?",
        "reserved_vec(config.event_capacity)?",
        "reserved_vec(config.spike_capacity)?",
        "RuntimeScratch::try_new(&config)?",
    ]
    forbidden = [
        "reserved_vec(config.configured_cells",
        "0..config.configured_cells",
        "Vec::with_capacity(config.configured_cells",
    ]
    configured_population_uses = [
        "event.target_gid >= self.config.configured_cells",
        "event.source_gid >= self.config.configured_cells",
        "bounded_index(word, config.configured_cells - 1)",
    ]
    passed = (
        all(pattern in production for pattern in required)
        and not any(pattern in production for pattern in forbidden)
        and all(pattern in production for pattern in configured_population_uses)
        and production.count("config.configured_cells")
        == len(configured_population_uses)
    )
    if not passed:
        raise ValueError("production procedural runtime sparse source invariant changed")
    native_root = ROOT / "src/humanbrain/_native_src"
    cargo_inputs = [native_root / "Cargo.toml", native_root / "Cargo.lock"]
    build_script = native_root / "build.rs"
    if build_script.is_file():
        cargo_inputs.append(build_script)
    cargo_inputs.extend(
        path for path in (native_root / "src").rglob("*.rs") if path.is_file()
    )
    cargo_paths = sorted(path.relative_to(ROOT).as_posix() for path in cargo_inputs)
    if not all(path in _SOURCE_PATHS for path in cargo_paths):
        raise ValueError("profile source bundle omits a native Cargo/Rust build input")
    return {
        "passed": True,
        "configured_ids_scanned_is_empirical": False,
        "required_capacity_allocations": required,
        "forbidden_production_patterns": forbidden,
        "configured_population_uses": configured_population_uses,
        "native_cargo_inputs": cargo_paths,
        "procedural_rs_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def validate_workload_derivations(protocol: dict[str, object]) -> dict[str, object]:
    """Prove the fixed external events and counter64 target cardinalities."""

    from humanbrain.procedural_topology import ProceduralTopology, TopologyManifest

    execution = protocol["execution"]
    runtime = protocol["runtime"]
    workloads = protocol["workloads"]
    if not isinstance(execution, dict) or not isinstance(runtime, dict):
        raise TypeError("execution and runtime protocol records must be objects")
    if not isinstance(workloads, dict) or not isinstance(workloads.get("driven"), dict):
        raise TypeError("driven workload protocol record must be an object")
    driven = workloads["driven"]
    configured_counts = tuple(int(value) for value in execution["configured_cells"])
    sources = tuple(int(value) for value in driven["source_gids"])
    if len(sources) != len(set(sources)) or not all(
        0 <= source < min(configured_counts) for source in sources
    ):
        raise ValueError("driven source GIDs must be unique and valid at both N")
    events = [
        {
            "due_step": 0,
            "target_gid": source,
            "source_gid": source,
            "edge_slot": 0,
            "current_pa": float(driven["current_pa"]),
        }
        for source in sources
    ]
    events_sha256 = _canonical_sha256(events)
    if events_sha256 != driven["external_events_sha256"]:
        raise ValueError("driven external-event derivation hash changed")

    derived_by_count: dict[str, object] = {}
    declared = driven["topology_derivations"]
    if not isinstance(declared, dict):
        raise TypeError("topology_derivations must be an object")
    for configured_cells in configured_counts:
        topology = ProceduralTopology(
            TopologyManifest(
                cell_count=configured_cells,
                fanout=int(runtime["fanout"]),
                topology=str(runtime["topology"]),
                seed=int(runtime["seed"]),
            )
        )
        rows = [
            {
                "source_gid": source,
                "targets": [
                    edge.target_gid for edge in topology.base_edges(source)
                ],
            }
            for source in sources
        ]
        targets = {target for row in rows for target in row["targets"]}
        record = {
            "ordered_rows_sha256": _canonical_sha256(
                {
                    "configured_cells": configured_cells,
                    "fanout": int(runtime["fanout"]),
                    "seed": int(runtime["seed"]),
                    "rows": rows,
                }
            ),
            "target_union_sha256": _canonical_sha256(sorted(targets)),
            "target_union_size": len(targets),
            "source_and_target_union_size": len(set(sources) | targets),
        }
        if (
            len(targets) != len(sources) * int(runtime["fanout"])
            or len(set(sources) | targets) != len(sources) + len(targets)
        ):
            raise ValueError(
                f"counter64 workload targets are not collision-free for N={configured_cells}"
            )
        if record != declared.get(str(configured_cells)):
            raise ValueError(
                f"counter64 workload derivation changed for N={configured_cells}"
            )
        derived_by_count[str(configured_cells)] = record
    return {
        "external_events_identical_across_n": True,
        "external_events_sha256": events_sha256,
        "configured_cells": derived_by_count,
    }


def validate_protocol(protocol: dict[str, object]) -> None:
    """Reject every execution-affecting drift from the preregistered profile."""

    root = _require_exact_keys(
        protocol,
        {
            "name",
            "protocol_version",
            "baseline_commit",
            "purpose",
            "execution",
            "runtime",
            "workloads",
            "acceptance",
            "reporting",
            "implementation_invariants",
            "source_binding",
            "publication",
            "evidence_scope",
            "claims",
            "forbidden_runs",
            "status",
        },
        "profile protocol",
    )
    if (
        root["name"] != "procedural-runtime-profile-v1"
        or root["protocol_version"] != 1
        or root["baseline_commit"]
        != "a33d2ffd52e3365fb9e4f93c052453cd55ec8189"
    ):
        raise ValueError("profile protocol identity changed")
    if root["purpose"] != _EXPECTED_PURPOSE:
        raise ValueError("profile purpose and claim boundary changed")

    execution = _require_exact_keys(
        root["execution"],
        {
            "backend",
            "configured_cells",
            "repetitions_per_condition",
            "fresh_subprocess_per_row",
            "worker_order",
            "timed_regions",
        },
        "execution",
    )
    if (
        execution["backend"] != "cpu-native-rust"
        or execution["configured_cells"] != [1_000_000, 100_000_000_000]
        or execution["repetitions_per_condition"] != 8
        or execution["fresh_subprocess_per_row"] is not True
    ):
        raise ValueError("execution population, backend, or repetition contract changed")
    if execution["worker_order"] != _expected_worker_order():
        raise ValueError("exact balanced 32-row worker order changed")
    timed = _require_exact_keys(
        execution["timed_regions"], {"native_call_ns", "e2e_ns"}, "timed_regions"
    )
    if not all(isinstance(value, str) and value for value in timed.values()):
        raise ValueError("timed-region ownership descriptions must be non-empty")
    if (
        "single C ABI" not in str(timed["native_call_ns"])
        or "outside the interval" not in str(timed["e2e_ns"])
    ):
        raise ValueError("timing ownership contract changed")

    if root["runtime"] != _EXPECTED_RUNTIME:
        raise ValueError("fixed R/Q/S/F/D runtime cost contract changed")
    workloads = _require_exact_keys(root["workloads"], {"quiet", "driven"}, "workloads")
    quiet = _require_exact_keys(
        workloads["quiet"],
        {"step_count", "external_events_sha256", "event_count", "expected"},
        "quiet workload",
    )
    if (
        quiet["step_count"] != 65_536
        or quiet["event_count"] != 0
        or quiet["external_events_sha256"]
        != "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e936ba84165dc6d1be0bbf3"
        or quiet["expected"] != _EXPECTED_COUNTERS["quiet"]
    ):
        raise ValueError("quiet workload contract changed")
    driven = _require_exact_keys(
        workloads["driven"],
        {
            "step_count",
            "current_pa",
            "source_gids",
            "external_events_sha256",
            "event_count",
            "topology_derivations",
            "expected",
        },
        "driven workload",
    )
    if (
        driven["step_count"] != 256
        or driven["current_pa"] != 150_000.0
        or driven["event_count"] != 16
        or driven["source_gids"] != list(_EXPECTED_DRIVEN_SOURCE_GIDS)
        or driven["external_events_sha256"]
        != "1307653946f13a353f821023dfbe8b76f237913c580bac1465dd75cc486c334d"
        or driven["topology_derivations"] != _EXPECTED_DRIVEN_DERIVATIONS
        or driven["expected"] != _EXPECTED_COUNTERS["driven"]
    ):
        raise ValueError("driven workload contract changed")
    validate_workload_derivations(protocol)

    if root["acceptance"] != _EXPECTED_ACCEPTANCE:
        raise ValueError("acceptance gates changed")
    reporting = _require_exact_keys(
        root["reporting"],
        {
            "raw_rows",
            "scale_accounting",
            "native_payload_bytes",
            "process_memory_observations",
            "timing_observations",
            "environment_and_build_identity_outside_timed_regions",
            "incremental",
            "nonoverwriting",
            "resume",
            "rerun",
            "retune",
        },
        "reporting",
    )
    if (
        reporting["process_memory_observations"]
        != ["before_runtime", "after_construction", "after_advance"]
        or reporting["timing_observations"] != ["native_call_ns", "e2e_ns"]
        or any(
            reporting[key] is not expected
            for key, expected in {
                "raw_rows": True,
                "scale_accounting": True,
                "native_payload_bytes": True,
                "environment_and_build_identity_outside_timed_regions": True,
                "incremental": True,
                "nonoverwriting": True,
                "resume": False,
                "rerun": False,
                "retune": False,
            }.items()
        )
    ):
        raise ValueError("reporting and one-shot publication contract changed")
    invariants = _require_exact_keys(
        root["implementation_invariants"],
        {
            "configured_population_scan_or_allocation",
            "dense_reference_in_timed_worker",
            "gpu_calls",
        },
        "implementation_invariants",
    )
    if (
        invariants["dense_reference_in_timed_worker"] is not False
        or invariants["gpu_calls"] is not False
        or "not treated as an empirical measurement"
        not in str(invariants["configured_population_scan_or_allocation"])
    ):
        raise ValueError("source-bound implementation invariants changed")
    validate_sparse_source_invariants()

    binding = _require_exact_keys(
        root["source_binding"],
        {"framing", "paths", "files", "bundle_sha256"},
        "source_binding",
    )
    if (
        binding["framing"] != _SOURCE_BUNDLE_NAMESPACE
        or binding["paths"] != list(_SOURCE_PATHS)
        or not isinstance(binding["files"], list)
        or not isinstance(binding["bundle_sha256"], str)
    ):
        raise ValueError("source-binding contract changed")
    source_files = binding["files"]
    assert isinstance(source_files, list)
    if len(source_files) != len(_SOURCE_PATHS):
        raise ValueError("source-binding manifest must cover every fixed source path")
    for expected_path, record in zip(_SOURCE_PATHS, source_files, strict=True):
        source = _require_exact_keys(
            record, {"path", "size_bytes", "sha256"}, "source-binding file"
        )
        if (
            source["path"] != expected_path
            or type(source["size_bytes"]) is not int
            or int(source["size_bytes"]) < 0
            or not isinstance(source["sha256"], str)
            or len(source["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in source["sha256"])
        ):
            raise ValueError("source-binding file record is invalid")
    bundle_sha = binding["bundle_sha256"]
    if len(bundle_sha) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_sha
    ):
        raise ValueError("source-binding bundle digest must be lowercase SHA-256")
    publication = _require_exact_keys(
        root["publication"],
        {
            "evidence_root",
            "evidence_directory",
            "artifact_kind",
            "policy",
        },
        "publication",
    )
    if (
        publication["evidence_root"] != "artifacts/procedural_runtime_profile_v1"
        or publication["evidence_directory"]
        != "artifacts/procedural_runtime_profile_v1/evidence"
        or publication["artifact_kind"] != "procedural-runtime-profile-record"
        or "forbids resume" not in str(publication["policy"])
    ):
        raise ValueError("fixed nonoverwriting publication contract changed")
    claims = _require_exact_keys(
        root["claims"],
        {
            "hypothesis",
            "speed",
            "throughput",
            "capability",
            "resident_100_billion_cells",
            "updated_100_billion_cells",
        },
        "claims",
    )
    if claims != _EXPECTED_CLAIMS or root["evidence_scope"] != _EXPECTED_EVIDENCE_SCOPE:
        raise ValueError("profile hypothesis, evidence scope, or claim boundary changed")
    if root["forbidden_runs"] != [
        "GPU",
        "capability",
        "DEV/SWE/canonical/private evaluation",
        "Fly Brain",
    ]:
        raise ValueError("forbidden-run list changed")
    status = _require_exact_keys(
        root["status"],
        {"executed", "material_profile_artifact", "execution_requires_separate_explicit_hold_release"},
        "status",
    )
    if status != {
        "executed": False,
        "material_profile_artifact": None,
        "execution_requires_separate_explicit_hold_release": True,
    }:
        raise ValueError("unexecuted protocol status changed")


def _compute_source_bundle(paths: Sequence[str]) -> dict[str, object]:
    entries: list[tuple[str, bytes]] = []
    files: list[dict[str, object]] = []
    for value in paths:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError("source paths must be canonical repository-relative paths")
        payload = _read_bounded(ROOT / value, _SOURCE_MAX_BYTES)
        entries.append((value, payload))
        files.append(
            {
                "path": value,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "framing": _SOURCE_BUNDLE_NAMESPACE,
        "files": files,
        "bundle_sha256": _framed_source_sha(entries),
    }


def _validate_source_binding(
    protocol: dict[str, object], actual: dict[str, object]
) -> None:
    binding = protocol.get("source_binding")
    if not isinstance(binding, dict):
        raise TypeError("protocol source_binding must be an object")
    expected = {
        "framing": binding.get("framing"),
        "files": binding.get("files"),
        "bundle_sha256": binding.get("bundle_sha256"),
    }
    if expected != actual:
        raise ValueError("profile source binding differs from current source bytes")


def source_bundle() -> dict[str, object]:
    _, protocol, _ = _protocol_snapshot()
    actual = _compute_source_bundle(_SOURCE_PATHS)
    _validate_source_binding(protocol, actual)
    return actual


def _repository_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _loaded_binary_identity() -> dict[str, object]:
    from humanbrain.native import loaded_library_identity

    return dict(loaded_library_identity().to_record())


def _environment_identity() -> dict[str, object]:
    import numpy as np

    return {
        "backend": "cpu-native-rust",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }


class _NativePreparedRow:
    def __init__(self, runtime: object, step_count: int, events: tuple[object, ...]) -> None:
        self._runtime = runtime
        self._step_count = step_count
        self._events = events

    def advance(self) -> object:
        return self._runtime.advance(self._step_count, self._events)  # type: ignore[attr-defined]

    def close(self) -> None:
        self._runtime.close()  # type: ignore[attr-defined]


def _prepare_native_row(
    protocol: dict[str, object],
    row: dict[str, object],
) -> _NativePreparedRow:
    from humanbrain.procedural_runtime import (
        NativeProceduralRuntime,
        ProceduralInputEvent,
        ProceduralRuntimeConfig,
    )
    from humanbrain.procedural_topology import TopologyManifest

    runtime_record = protocol["runtime"]
    workloads = protocol["workloads"]
    assert isinstance(runtime_record, dict) and isinstance(workloads, dict)
    configured_cells = int(row["configured_cells"])
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=configured_cells,
            fanout=int(runtime_record["fanout"]),
            topology=str(runtime_record["topology"]),
            seed=int(runtime_record["seed"]),
        ),
        resident_capacity=int(runtime_record["resident_capacity"]),
        event_capacity=int(runtime_record["event_capacity"]),
        spike_capacity=int(runtime_record["spike_capacity"]),
        interval_distinct_capacity=int(runtime_record["interval_distinct_capacity"]),
        recurrent_weight_pa=float(runtime_record["recurrent_weight_pa"]),
        recurrent_delay_steps=int(runtime_record["recurrent_delay_steps"]),
    )
    workload = workloads[str(row["case"])]
    assert isinstance(workload, dict)
    events: tuple[object, ...] = ()
    if row["case"] == "driven":
        events = tuple(
            ProceduralInputEvent(
                due_step=0,
                target_gid=int(source),
                source_gid=int(source),
                edge_slot=0,
                current_pa=float(workload["current_pa"]),
            )
            for source in workload["source_gids"]
        )
    runtime = NativeProceduralRuntime(config)
    return _NativePreparedRow(runtime, int(workload["step_count"]), events)


def _memory_observation() -> dict[str, float]:
    from benchmarks.native_scaling import _process_memory

    return _process_memory()


def _production_worker_dependencies() -> _WorkerDependencies:
    return _WorkerDependencies(
        binary_identity=_loaded_binary_identity,
        environment_identity=_environment_identity,
        memory_observation=_memory_observation,
        prepare=_prepare_native_row,
        clock_ns=time.perf_counter_ns,
    )


def _portable_binary_identity(value: object) -> dict[str, object]:
    identity = _require_exact_keys(
        value,
        {"binary_size", "binary_sha256", "source_digest"},
        "loaded binary identity",
    )
    if (
        type(identity["binary_size"]) is not int
        or int(identity["binary_size"]) <= 0
        or not isinstance(identity["binary_sha256"], str)
        or len(str(identity["binary_sha256"])) != 64
        or any(character not in "0123456789abcdef" for character in str(identity["binary_sha256"]))
        or not isinstance(identity["source_digest"], str)
        or len(str(identity["source_digest"])) != 64
        or any(character not in "0123456789abcdef" for character in str(identity["source_digest"]))
    ):
        raise ValueError("loaded binary identity is invalid")
    return dict(identity)


def _memory_bytes(value: object) -> dict[str, int]:
    memory = _require_exact_keys(
        value,
        {"working_set_mib", "peak_working_set_mib", "private_memory_mib"},
        "process memory observation",
    )
    result: dict[str, int] = {}
    for name, raw in memory.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0
        ):
            raise ValueError("process memory observations must be finite non-negative")
        result[name.replace("_mib", "_bytes")] = int(
            round(float(raw) * 1024.0 * 1024.0)
        )
    return result


def _validate_environment_identity(value: object) -> None:
    environment = _require_exact_keys(
        value,
        {
            "backend",
            "python_implementation",
            "python_version",
            "machine",
            "processor",
            "platform",
            "logical_cpu_count",
            "numpy_version",
            "cuda_visible_devices",
            "hip_visible_devices",
            "rayon_num_threads",
        },
        "worker environment",
    )
    required_strings = (
        "python_implementation",
        "python_version",
        "machine",
        "platform",
        "numpy_version",
    )
    if (
        environment["backend"] != "cpu-native-rust"
        or environment["cuda_visible_devices"] != ""
        or environment["hip_visible_devices"] != ""
        or any(
            not isinstance(environment[name], str) or not environment[name]
            for name in required_strings
        )
        or not isinstance(environment["processor"], str)
        or type(environment["logical_cpu_count"]) is not int
        or int(environment["logical_cpu_count"]) <= 0
        or not (
            environment["rayon_num_threads"] is None
            or isinstance(environment["rayon_num_threads"], str)
        )
    ):
        raise ValueError("worker environment identity is invalid or GPU-visible")


def _validate_measured_row(
    *,
    protocol: dict[str, object],
    row: dict[str, object],
    result: object,
    accounting: dict[str, object],
    e2e_ns: int,
) -> None:
    expected = _EXPECTED_COUNTERS[str(row["case"])]
    configured_cells = int(row["configured_cells"])
    _require_exact_keys(
        accounting,
        {"claim", "interval", "cells", "work", "edges", "events", "memory"},
        "ScaleAccounting",
    )
    claim = _require_exact_keys(accounting["claim"], {"label", "text"}, "claim")
    interval = _require_exact_keys(
        accounting["interval"],
        {
            "name",
            "start_step",
            "step_count",
            "biological_step_seconds",
            "wall_seconds",
        },
        "interval",
    )
    cells = _require_exact_keys(
        accounting["cells"],
        {
            "configured",
            "addressable",
            "resident_peak",
            "resident_distinct",
            "updated_distinct",
            "spiking_distinct",
        },
        "cells",
    )
    work = _require_exact_keys(
        accounting["work"],
        {
            "cell_step_evaluations",
            "spike_events",
            "analytic_catchups",
            "analytic_step_equivalents",
        },
        "work",
    )
    edges = _require_exact_keys(
        accounting["edges"],
        {"configured", "resident", "mutable", "generated"},
        "edges",
    )
    events = _require_exact_keys(
        accounting["events"],
        {"delivered", "queue_peak", "sent_bytes_by_shard_pair"},
        "events",
    )
    memory = _require_exact_keys(
        accounting["memory"],
        {"host_bytes", "device_bytes", "bytes_by_shard"},
        "memory",
    )
    actual = {
        "resident_peak": cells.get("resident_peak"),
        "resident_distinct": cells.get("resident_distinct"),
        "updated_distinct": cells.get("updated_distinct"),
        "spiking_distinct": cells.get("spiking_distinct"),
        "cell_step_evaluations": work.get("cell_step_evaluations"),
        "spike_events": work.get("spike_events"),
        "generated_edges": edges.get("generated"),
        "delivered_events": events.get("delivered"),
        "queue_peak": events.get("queue_peak"),
        "retained_nonzero_cells": getattr(result, "retained_nonzero_cells", None),
    }
    workloads = protocol["workloads"]
    runtime = protocol["runtime"]
    assert isinstance(workloads, dict) and isinstance(runtime, dict)
    workload = workloads[str(row["case"])]
    assert isinstance(workload, dict)
    native_call_ns = getattr(result, "native_call_ns", None)
    counter_values = (
        *actual.values(),
        cells["configured"],
        cells["addressable"],
        work["analytic_catchups"],
        work["analytic_step_equivalents"],
        edges["configured"],
        edges["resident"],
        edges["mutable"],
        memory["host_bytes"],
        memory["device_bytes"],
    )
    if any(type(value) is not int or value < 0 for value in counter_values):
        raise ValueError("ScaleAccounting counters must be non-negative integers")
    if actual != expected:
        raise ValueError("measured workload counters differ from preregistered literals")
    if (
        claim.get("label") != "procedural-runtime-correctness"
        or not isinstance(claim.get("text"), str)
        or not claim["text"]
        or interval.get("name") != "native-procedural-advance"
        or interval.get("start_step") != 0
        or cells.get("configured") != configured_cells
        or cells.get("addressable") != configured_cells
        or edges.get("configured") != configured_cells * int(runtime["fanout"])
        or edges.get("resident") != 0
        or edges.get("mutable") != 0
        or interval.get("step_count") != workload["step_count"]
        or interval.get("biological_step_seconds") != 0.00005
        or interval.get("wall_seconds") != 0.0
        or work.get("analytic_catchups") != 0
        or work.get("analytic_step_equivalents") != 0
        or events.get("sent_bytes_by_shard_pair") != []
        or memory.get("device_bytes") != 0
        or getattr(result, "capacity_overflowed", None) is not False
    ):
        raise ValueError("measured ScaleAccounting invariant failed")
    bytes_by_shard = memory.get("bytes_by_shard")
    if not isinstance(bytes_by_shard, list) or len(bytes_by_shard) != 1:
        raise ValueError("measured ScaleAccounting memory shards differ")
    shard = _require_exact_keys(
        bytes_by_shard[0], {"shard", "host_bytes", "device_bytes"}, "memory shard"
    )
    if shard != {
        "shard": "cpu-native-procedural",
        "host_bytes": memory["host_bytes"],
        "device_bytes": 0,
    }:
        raise ValueError("measured ScaleAccounting memory shard is inconsistent")
    if (
        type(native_call_ns) is not int
        or native_call_ns < 0
        or type(e2e_ns) is not int
        or e2e_ns < native_call_ns
    ):
        raise ValueError("native-call timing must be an integer inside e2e timing")


def _run_worker_row(
    *,
    protocol: dict[str, object],
    row: dict[str, object],
    protocol_sha: str,
    source_sha: str,
    git_head: str,
    binary_identity: dict[str, object],
    dependencies: _WorkerDependencies,
) -> dict[str, object]:
    before_runtime = _memory_bytes(dependencies.memory_observation())
    prepared = dependencies.prepare(protocol, row)
    after_construction = _memory_bytes(dependencies.memory_observation())
    e2e_start = dependencies.clock_ns()
    try:
        result = prepared.advance()  # type: ignore[attr-defined]
        e2e_ns = dependencies.clock_ns() - e2e_start
        after_advance = _memory_bytes(dependencies.memory_observation())
    finally:
        prepared.close()  # type: ignore[attr-defined]
    if _portable_binary_identity(dependencies.binary_identity()) != binary_identity:
        raise ProfileExecutionRefused("loaded native binary changed during worker")
    accounting = result.accounting.to_dict()  # type: ignore[attr-defined]
    _validate_measured_row(
        protocol=protocol,
        row=row,
        result=result,
        accounting=accounting,
        e2e_ns=e2e_ns,
    )
    environment = dependencies.environment_identity()
    _validate_environment_identity(environment)
    return {
        "row": row,
        "protocol_sha256": protocol_sha,
        "source_bundle_sha256": source_sha,
        "git_head": git_head,
        "backend": "cpu-native-rust",
        "environment": environment,
        "loaded_binary_identity": binary_identity,
        "timing": {
            "native_call_ns": result.native_call_ns,  # type: ignore[attr-defined]
            "e2e_ns": e2e_ns,
        },
        "scale_accounting": accounting,
        "native_payload_bytes": accounting["memory"]["host_bytes"],  # type: ignore[index]
        "retained_nonzero_cells": result.retained_nonzero_cells,  # type: ignore[attr-defined]
        "process_memory": {
            "before_runtime": before_runtime,
            "after_construction": after_construction,
            "after_advance": after_advance,
            "private_delta_bytes": (
                after_advance["private_memory_bytes"]
                - before_runtime["private_memory_bytes"]
            ),
        },
        "capacity_overflowed": False,
    }


def _validate_process_memory_observation(value: object) -> None:
    memory = _require_exact_keys(
        value,
        {
            "before_runtime",
            "after_construction",
            "after_advance",
            "private_delta_bytes",
        },
        "process memory observation",
    )
    samples: dict[str, dict[str, object]] = {}
    for name in ("before_runtime", "after_construction", "after_advance"):
        sample = _require_exact_keys(
            memory[name],
            {"working_set_bytes", "peak_working_set_bytes", "private_memory_bytes"},
            f"process memory observation {name}",
        )
        if any(type(raw) is not int or raw < 0 for raw in sample.values()):
            raise ValueError("process memory observations must be non-negative integers")
        if sample["peak_working_set_bytes"] < sample["working_set_bytes"]:
            raise ValueError("process memory peak cannot be below the working set")
        samples[name] = sample
    private_delta = memory["private_delta_bytes"]
    expected_delta = (
        samples["after_advance"]["private_memory_bytes"]
        - samples["before_runtime"]["private_memory_bytes"]
    )
    if type(private_delta) is not int or private_delta != expected_delta:
        raise ValueError("process memory observation private delta is inconsistent")


def _repository_status() -> tuple[tuple[str, Path], ...]:
    payload = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
    )
    entries: list[tuple[str, Path]] = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        if len(raw) < 4 or raw[2:3] != b" ":
            entries.append(("invalid", ROOT / raw.decode("utf-8", "replace")))
            continue
        status = raw[:2].decode("ascii", "replace")
        relative_path = raw[3:].decode("utf-8")
        entries.append((status, (ROOT / relative_path).resolve()))
    return tuple(entries)


def _spawn_worker(
    row: dict[str, object],
    protocol_sha: str,
    source_sha: str,
    head: str,
    binary_identity: dict[str, object],
    worker_capability: str,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["HIP_VISIBLE_DEVICES"] = ""
    environment[_WORKER_CAPABILITY_ENV] = worker_capability
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--execute",
            "--worker-row",
            str(row["ordinal"]),
            "--protocol-sha256",
            protocol_sha,
            "--source-bundle-sha256",
            source_sha,
            "--expected-head",
            head,
            "--expected-binary-size-bytes",
            str(binary_identity["binary_size"]),
            "--expected-binary-sha256",
            str(binary_identity["binary_sha256"]),
            "--expected-native-source-digest",
            str(binary_identity["source_digest"]),
            "--expected-parent-pid",
            str(os.getpid()),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    if len(completed.stdout.encode("utf-8")) > 2 * 1024 * 1024:
        raise RuntimeError("worker result exceeded the bounded result size")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise TypeError("worker result must be an object")
    return result


def _create_evidence_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / "evidence").mkdir()


def _publish_record(path: Path, label: str, payload: dict[str, object]) -> str:
    from humanbrain.simple_swe.artifacts import atomic_create_canonical_json

    publication = atomic_create_canonical_json(
        path,
        kind="procedural-runtime-profile-record",
        payload={"event": label, **payload},
    )
    return publication.file_sha256


def _production_parent_dependencies() -> _ParentDependencies:
    return _ParentDependencies(
        repository_head=_repository_head,
        repository_status=_repository_status,
        binary_identity=_loaded_binary_identity,
        spawn_worker=_spawn_worker,
        publish_record=_publish_record,
        create_evidence_root=_create_evidence_root,
    )


def _verify_parent_snapshot(
    *,
    dependencies: _ParentDependencies,
    expected_head: str,
    expected_source_sha: str,
    expected_binary: dict[str, object],
    evidence_file_sha256: dict[Path, str],
) -> None:
    if dependencies.repository_head() != expected_head:
        raise ProfileExecutionRefused("repository HEAD changed during the profile")
    if source_bundle()["bundle_sha256"] != expected_source_sha:
        raise ProfileExecutionRefused("profile source bundle changed during the profile")
    actual_binary = _portable_binary_identity(dependencies.binary_identity())
    if actual_binary != expected_binary:
        raise ProfileExecutionRefused("loaded native binary identity changed")
    expected_status = {("??", path.resolve()) for path in evidence_file_sha256}
    actual_status = set(dependencies.repository_status())
    if actual_status != expected_status:
        raise ProfileExecutionRefused(
            "repository status contains dirty or unexpected evidence paths"
        )
    actual_files = {
        path.resolve() for path in EVIDENCE_ROOT.rglob("*") if path.is_file()
    }
    actual_directories = {
        path.resolve() for path in EVIDENCE_ROOT.rglob("*") if path.is_dir()
    }
    if actual_files != set(evidence_file_sha256) or actual_directories != {
        (EVIDENCE_ROOT / "evidence").resolve()
    }:
        raise ProfileExecutionRefused(
            "evidence root contains paths outside the exact runner inventory"
        )
    for path, expected_sha256 in evidence_file_sha256.items():
        actual_sha256 = hashlib.sha256(
            _read_bounded(path, _ARTIFACT_MAX_BYTES)
        ).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ProfileExecutionRefused("published evidence bytes changed")


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _validate_worker_record(
    *,
    record: dict[str, object],
    row: dict[str, object],
    protocol: dict[str, object],
    protocol_sha: str,
    source_sha: str,
    head: str,
    binary_identity: dict[str, object],
) -> None:
    _require_exact_keys(
        record,
        {
            "row",
            "protocol_sha256",
            "source_bundle_sha256",
            "git_head",
            "backend",
            "environment",
            "loaded_binary_identity",
            "timing",
            "scale_accounting",
            "native_payload_bytes",
            "retained_nonzero_cells",
            "process_memory",
            "capacity_overflowed",
        },
        "worker record",
    )
    if (
        record.get("row") != row
        or record.get("protocol_sha256") != protocol_sha
        or record.get("source_bundle_sha256") != source_sha
        or record.get("git_head") != head
        or record.get("backend") != "cpu-native-rust"
        or record.get("loaded_binary_identity") != binary_identity
        or record.get("capacity_overflowed") is not False
    ):
        raise ValueError("worker attestation differs from the parent profile")
    if _contains_key(record, "configured_ids_scanned"):
        raise ValueError("configured_ids_scanned is not an empirical profile counter")
    _validate_environment_identity(record.get("environment"))
    timing = record.get("timing")
    accounting = record.get("scale_accounting")
    if not isinstance(timing, dict) or not isinstance(accounting, dict):
        raise TypeError("worker timing and ScaleAccounting must be objects")
    _require_exact_keys(timing, {"native_call_ns", "e2e_ns"}, "worker timing")
    result = SimpleNamespace(
        native_call_ns=timing.get("native_call_ns"),
        retained_nonzero_cells=record.get("retained_nonzero_cells"),
        capacity_overflowed=record.get("capacity_overflowed"),
    )
    _validate_measured_row(
        protocol=protocol,
        row=row,
        result=result,
        accounting=accounting,
        e2e_ns=timing.get("e2e_ns"),  # type: ignore[arg-type]
    )
    memory = accounting["memory"]
    if (
        record.get("native_payload_bytes") != memory["host_bytes"]  # type: ignore[index]
        or not isinstance(record.get("process_memory"), dict)
    ):
        raise ValueError("worker memory observation is inconsistent")
    _validate_process_memory_observation(record["process_memory"])


def _cross_n_ratio(values: Sequence[int | float]) -> tuple[float | None, bool]:
    low = min(values)
    high = max(values)
    if low == 0:
        return (0.0, True) if high == 0 else (None, False)
    return float(high / low), True


def _workload_signature(record: dict[str, object]) -> tuple[object, ...]:
    accounting = record["scale_accounting"]
    cells = accounting["cells"]  # type: ignore[index]
    work = accounting["work"]  # type: ignore[index]
    edges = accounting["edges"]  # type: ignore[index]
    events = accounting["events"]  # type: ignore[index]
    return (
        cells["resident_peak"],  # type: ignore[index]
        cells["resident_distinct"],  # type: ignore[index]
        cells["updated_distinct"],  # type: ignore[index]
        cells["spiking_distinct"],  # type: ignore[index]
        work["cell_step_evaluations"],  # type: ignore[index]
        work["spike_events"],  # type: ignore[index]
        edges["generated"],  # type: ignore[index]
        events["delivered"],  # type: ignore[index]
        events["queue_peak"],  # type: ignore[index]
        record["retained_nonzero_cells"],
    )


def _profile_summary(
    records: list[dict[str, object]],
    protocol: dict[str, object],
    binary_identity: dict[str, object],
    protocol_sha: str,
    source_sha: str,
    git_head: str,
) -> dict[str, object]:
    acceptance = protocol["acceptance"]
    assert isinstance(acceptance, dict)
    case_summaries: dict[str, object] = {}
    all_case_gates: list[bool] = []
    for case in ("quiet", "driven"):
        by_count = {
            configured_cells: [
                record
                for record in records
                if record["row"]["case"] == case  # type: ignore[index]
                and record["row"]["configured_cells"] == configured_cells  # type: ignore[index]
            ]
            for configured_cells in (1_000_000, 100_000_000_000)
        }
        if any(len(rows) != 8 for rows in by_count.values()):
            raise ValueError("summary requires eight rows per case and configured N")
        observations: dict[str, dict[str, float | int]] = {}
        for configured_cells, rows in by_count.items():
            observations[str(configured_cells)] = {
                "native_call_median_ns": median(
                    int(row["timing"]["native_call_ns"]) for row in rows  # type: ignore[index]
                ),
                "e2e_median_ns": median(
                    int(row["timing"]["e2e_ns"]) for row in rows  # type: ignore[index]
                ),
                "private_delta_median_bytes": median(
                    int(row["process_memory"]["private_delta_bytes"])  # type: ignore[index]
                    for row in rows
                ),
                "native_payload_bytes": int(rows[0]["native_payload_bytes"]),
            }
        native_values = [
            observations[str(value)]["native_call_median_ns"]
            for value in (1_000_000, 100_000_000_000)
        ]
        e2e_values = [
            observations[str(value)]["e2e_median_ns"]
            for value in (1_000_000, 100_000_000_000)
        ]
        native_ratio, native_defined = _cross_n_ratio(native_values)
        e2e_ratio, e2e_defined = _cross_n_ratio(e2e_values)
        counter_equal = len({_workload_signature(row) for row in records if row["row"]["case"] == case}) == 1  # type: ignore[index]
        payload_equal = len(
            {int(row["native_payload_bytes"]) for row in records if row["row"]["case"] == case}  # type: ignore[index]
        ) == 1
        private_difference = abs(
            float(observations["1000000"]["private_delta_median_bytes"])
            - float(observations["100000000000"]["private_delta_median_bytes"])
        )
        gates = {
            "workload_counters_equal": counter_equal,
            "native_payload_equal": payload_equal,
            "native_call_ratio": native_defined
            and native_ratio is not None
            and native_ratio
            <= float(acceptance["maximum_cross_n_median_native_call_ratio"]),
            "e2e_ratio": e2e_defined
            and e2e_ratio is not None
            and e2e_ratio <= float(acceptance["maximum_cross_n_median_e2e_ratio"]),
            "private_delta_difference": private_difference
            <= int(acceptance["maximum_cross_n_median_private_delta_difference_bytes"]),
        }
        all_case_gates.extend(gates.values())
        case_summaries[case] = {
            "observations": observations,
            "native_call_cross_n_ratio": native_ratio,
            "e2e_cross_n_ratio": e2e_ratio,
            "private_delta_cross_n_difference_bytes": private_difference,
            "gates": gates,
        }
    identity_gate = all(
        record["loaded_binary_identity"] == binary_identity for record in records
    )
    environment_gate = len(
        {
            _canonical_sha256(record["environment"])
            for record in records
        }
    ) == 1
    all_case_gates.extend((identity_gate, environment_gate))
    return {
        "material_profile_executed": True,
        "worker_rows": len(records),
        "protocol_sha256": protocol_sha,
        "source_bundle_sha256": source_sha,
        "git_head": git_head,
        "loaded_binary_identity": binary_identity,
        "environment_identity": records[0]["environment"],
        "cases": case_summaries,
        "gates": {
            "all_worker_binary_identities_equal": identity_gate,
            "all_worker_environment_identities_equal": environment_gate,
            "all_case_gates_passed": all(all_case_gates[:-2]),
        },
        "profile_pass": all(all_case_gates),
        "claims": protocol["claims"],
        "evidence_scope": protocol["evidence_scope"],
    }


def _execute_profile(
    *,
    protocol: dict[str, object],
    protocol_sha: str,
    source_sha: str,
    dependencies: _ParentDependencies,
) -> dict[str, object]:
    if EVIDENCE_ROOT.exists():
        raise FileExistsError(f"profile evidence root already exists: {EVIDENCE_ROOT}")
    starting_head = dependencies.repository_head()
    if dependencies.repository_status():
        raise ProfileExecutionRefused("profile execution requires a completely clean HEAD")
    if source_bundle()["bundle_sha256"] != source_sha:
        raise ProfileExecutionRefused("profile source changed before execution")
    binary_identity = _portable_binary_identity(dependencies.binary_identity())
    if dependencies.repository_status() or dependencies.repository_head() != starting_head:
        raise ProfileExecutionRefused("native preflight changed the clean repository")
    rows = protocol["execution"]["worker_order"]  # type: ignore[index]
    worker_capabilities = _new_worker_capabilities(len(rows))
    worker_capability_bindings = [
        {
            "ordinal": int(row["ordinal"]),
            "sha256": _worker_capability_sha256(capability),
        }
        for row, capability in zip(rows, worker_capabilities, strict=True)
    ]

    dependencies.create_evidence_root(EVIDENCE_ROOT)
    evidence_directory = EVIDENCE_ROOT / "evidence"
    evidence_file_sha256: dict[Path, str] = {}
    _verify_parent_snapshot(
        dependencies=dependencies,
        expected_head=starting_head,
        expected_source_sha=source_sha,
        expected_binary=binary_identity,
        evidence_file_sha256=evidence_file_sha256,
    )
    run_start = evidence_directory / "000-run-start.json"
    run_start_sha256 = dependencies.publish_record(
        run_start,
        "run-start",
        {
            "protocol_sha256": protocol_sha,
            "source_bundle_sha256": source_sha,
            "source_binding": protocol["source_binding"],
            "git_head": starting_head,
            "loaded_binary_identity": binary_identity,
            "worker_order": rows,
            "worker_capability_sha256_by_ordinal": worker_capability_bindings,
            "workload_derivation_proof": validate_workload_derivations(protocol),
            "sparse_source_invariant_proof": validate_sparse_source_invariants(),
            "material_profile_started": True,
            "completed_worker_rows": 0,
        },
    )
    evidence_file_sha256[run_start.resolve()] = run_start_sha256

    records: list[dict[str, object]] = []
    for ordinal, (row, worker_capability) in enumerate(
        zip(rows, worker_capabilities, strict=True), start=1
    ):
        _verify_parent_snapshot(
            dependencies=dependencies,
            expected_head=starting_head,
            expected_source_sha=source_sha,
            expected_binary=binary_identity,
            evidence_file_sha256=evidence_file_sha256,
        )
        try:
            worker_record = dependencies.spawn_worker(
                row,
                protocol_sha,
                source_sha,
                starting_head,
                binary_identity,
                worker_capability,
            )
            _verify_parent_snapshot(
                dependencies=dependencies,
                expected_head=starting_head,
                expected_source_sha=source_sha,
                expected_binary=binary_identity,
                evidence_file_sha256=evidence_file_sha256,
            )
            _validate_worker_record(
                record=worker_record,
                row=row,
                protocol=protocol,
                protocol_sha=protocol_sha,
                source_sha=source_sha,
                head=starting_head,
                binary_identity=binary_identity,
            )
        except Exception as error:
            _verify_parent_snapshot(
                dependencies=dependencies,
                expected_head=starting_head,
                expected_source_sha=source_sha,
                expected_binary=binary_identity,
                evidence_file_sha256=evidence_file_sha256,
            )
            failure_path = evidence_directory / f"{ordinal:03d}-terminal-failure.json"
            failure_sha256 = dependencies.publish_record(
                failure_path,
                "terminal-failure",
                {
                    "failed_worker_row": int(row["ordinal"]),
                    "completed_worker_rows": len(records),
                    "protocol_sha256": protocol_sha,
                    "source_bundle_sha256": source_sha,
                    "git_head": starting_head,
                    "loaded_binary_identity": binary_identity,
                    "error": f"{type(error).__name__}: {error}",
                    "retry_authorized": False,
                    "resume_authorized": False,
                },
            )
            evidence_file_sha256[failure_path.resolve()] = failure_sha256
            _verify_parent_snapshot(
                dependencies=dependencies,
                expected_head=starting_head,
                expected_source_sha=source_sha,
                expected_binary=binary_identity,
                evidence_file_sha256=evidence_file_sha256,
            )
            raise
        path = evidence_directory / f"{ordinal:03d}-worker-{int(row['ordinal']):02d}.json"
        worker_sha256 = dependencies.publish_record(path, "worker", worker_record)
        evidence_file_sha256[path.resolve()] = worker_sha256
        records.append(worker_record)

    summary = _profile_summary(
        records,
        protocol,
        binary_identity,
        protocol_sha,
        source_sha,
        starting_head,
    )
    _verify_parent_snapshot(
        dependencies=dependencies,
        expected_head=starting_head,
        expected_source_sha=source_sha,
        expected_binary=binary_identity,
        evidence_file_sha256=evidence_file_sha256,
    )
    summary_path = evidence_directory / "033-summary.json"
    summary_sha256 = dependencies.publish_record(summary_path, "summary", summary)
    evidence_file_sha256[summary_path.resolve()] = summary_sha256
    _verify_parent_snapshot(
        dependencies=dependencies,
        expected_head=starting_head,
        expected_source_sha=source_sha,
        expected_binary=binary_identity,
        evidence_file_sha256=evidence_file_sha256,
    )
    return summary


def _plan() -> dict[str, object]:
    _, protocol, digest = _protocol_snapshot()
    bundle = source_bundle()
    return {
        "will_execute": False,
        "material_profile_executed": False,
        "protocol_sha256": digest,
        "source_bundle_sha256": bundle["bundle_sha256"],
        "worker_rows": len(protocol["execution"]["worker_order"]),  # type: ignore[index]
        "evidence_root": str(EVIDENCE_ROOT),
        "evidence_root_exists": EVIDENCE_ROOT.exists(),
        "required_flags": [
            "--execute",
            "--protocol-sha256",
            digest,
            "--source-bundle-sha256",
            str(bundle["bundle_sha256"]),
        ],
        "evidence_scope": protocol["evidence_scope"],
        "claims": protocol["claims"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-row", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--source-bundle-sha256")
    parser.add_argument("--expected-head", help=argparse.SUPPRESS)
    parser.add_argument("--expected-binary-size-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--expected-binary-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-native-source-digest", help=argparse.SUPPRESS)
    parser.add_argument("--expected-parent-pid", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.worker_row is not None and not arguments.execute:
        raise ProfileExecutionRefused("worker execution requires --execute")
    if arguments.worker_row is not None:
        _, protocol, expected_protocol_sha = _protocol_snapshot()
        if arguments.protocol_sha256 != expected_protocol_sha:
            raise ProfileExecutionRefused(
                "worker requires the exact raw protocol SHA-256"
            )
        bundle = source_bundle()
        expected_source_sha = str(bundle["bundle_sha256"])
        if arguments.source_bundle_sha256 != expected_source_sha:
            raise ProfileExecutionRefused(
                "worker requires the exact source bundle SHA-256"
            )
        actual_head = _repository_head()
        if arguments.expected_head != actual_head:
            raise ProfileExecutionRefused("worker repository HEAD differs from parent")
        rows = protocol["execution"]["worker_order"]  # type: ignore[index]
        if not 0 <= arguments.worker_row < len(rows):
            raise ProfileExecutionRefused("worker row ordinal is outside the protocol")
        expected_binary = _portable_binary_identity(
            {
                "binary_size": arguments.expected_binary_size_bytes,
                "binary_sha256": arguments.expected_binary_sha256,
                "source_digest": arguments.expected_native_source_digest,
            }
        )
        worker_capability = os.environ.pop(_WORKER_CAPABILITY_ENV, None)
        _validate_parent_worker_authorization(
            row=rows[arguments.worker_row],
            protocol=protocol,
            protocol_sha=expected_protocol_sha,
            source_sha=expected_source_sha,
            git_head=actual_head,
            binary_identity=expected_binary,
            expected_parent_pid=arguments.expected_parent_pid,
            capability=worker_capability,
        )
        dependencies = _production_worker_dependencies()
        actual_binary = _portable_binary_identity(dependencies.binary_identity())
        if actual_binary != expected_binary:
            raise ProfileExecutionRefused(
                "loaded native binary identity differs from the parent preflight"
            )
        measured = _run_worker_row(
            protocol=protocol,
            row=rows[arguments.worker_row],
            protocol_sha=expected_protocol_sha,
            source_sha=expected_source_sha,
            git_head=actual_head,
            binary_identity=actual_binary,
            dependencies=dependencies,
        )
        print(json.dumps(measured, sort_keys=True))
        return 0
    if arguments.execute:
        _, protocol, expected_protocol_sha = _protocol_snapshot()
        if arguments.protocol_sha256 != expected_protocol_sha:
            raise ProfileExecutionRefused(
                "--execute requires the exact raw protocol SHA-256"
            )
        expected_source_sha = str(source_bundle()["bundle_sha256"])
        if arguments.source_bundle_sha256 != expected_source_sha:
            raise ProfileExecutionRefused(
                "--execute requires the exact source bundle SHA-256"
            )
        summary = _execute_profile(
            protocol=protocol,
            protocol_sha=expected_protocol_sha,
            source_sha=expected_source_sha,
            dependencies=_production_parent_dependencies(),
        )
        print(json.dumps(summary, sort_keys=True))
        if not summary["profile_pass"]:
            raise RuntimeError("procedural runtime engineering profile gates failed")
        return 0
    print(json.dumps(_plan(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
