"""Address/query-only engineering evidence for the counter64 topology oracle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import tracemalloc
from dataclasses import asdict
from math import isfinite, log10
from pathlib import Path
from time import perf_counter, perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import humanbrain.procedural_topology as procedural_topology_module
from humanbrain.procedural_topology import (
    ProceduralTopology,
    TopologyManifest,
    topology_implementation_contract,
    verify_topology_implementation,
)
from humanbrain.scale_accounting import (
    CellCounts,
    EdgeCounts,
    EventCounts,
    MeasurementInterval,
    MemoryCounts,
    ScaleAccounting,
    ScaleClaim,
    ShardMemory,
    WorkCounts,
)
from humanbrain.simple_swe.artifacts import (
    CANONICAL_JSON_ARTIFACT_SCHEMA,
    atomic_create_canonical_json,
)

BENCHMARK_PROTOCOL_VERSION = 2
PROTOCOL_PATH = ROOT / "benchmarks/procedural_address_space_protocol_v2.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/procedural_address_space_v2.json"
ARTIFACT_KIND = "procedural-address-space-evidence"
PUBLICATION_POLICY = (
    "Same-directory fsynced temporary file followed by non-overwriting atomic "
    "hard-link publication; the canonical envelope persists its payload SHA-256."
)
_PROTOCOL_KEYS = frozenset(
    {
        "name",
        "benchmark_protocol_version",
        "purpose",
        "implementation_contract",
        "cell_counts",
        "fanout",
        "seed",
        "valid_gids_by_cell_count",
        "invalid_gids_by_cell_count",
        "acceptance",
        "publication",
        "evidence_scope",
        "hypotheses",
        "notes",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _require_exact_int(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"protocol {field} must be an integer >= {minimum}")
    return value


def validate_protocol(protocol: dict[str, object]) -> None:
    """Validate the entire v2 runner schema and every implementation duplicate."""

    if set(protocol) != _PROTOCOL_KEYS:
        raise ValueError("benchmark protocol fields differ from v2")
    if protocol["name"] != "procedural-address-space-v2":
        raise ValueError("benchmark protocol name differs from v2")
    if protocol["benchmark_protocol_version"] != BENCHMARK_PROTOCOL_VERSION:
        raise ValueError("benchmark protocol version differs from v2")
    if protocol["implementation_contract"] != topology_implementation_contract():
        raise ValueError("benchmark implementation contract differs from runtime")
    if protocol["publication"] != {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_schema": CANONICAL_JSON_ARTIFACT_SCHEMA,
        "policy": PUBLICATION_POLICY,
    }:
        raise ValueError("benchmark publication contract differs from runtime")

    cell_counts_value = protocol["cell_counts"]
    if not isinstance(cell_counts_value, list) or not cell_counts_value:
        raise ValueError("protocol cell_counts must be a non-empty array")
    cell_counts = tuple(
        _require_exact_int(value, "cell_counts[]", minimum=2)
        for value in cell_counts_value
    )
    if len(set(cell_counts)) != len(cell_counts):
        raise ValueError("protocol cell_counts must be unique")
    fanout = _require_exact_int(protocol["fanout"], "fanout", minimum=1)
    seed = _require_exact_int(protocol["seed"], "seed", minimum=0)

    valid_by_count = protocol["valid_gids_by_cell_count"]
    invalid_by_count = protocol["invalid_gids_by_cell_count"]
    if not isinstance(valid_by_count, dict) or not isinstance(invalid_by_count, dict):
        raise ValueError("protocol gid maps must be objects")
    expected_count_keys = {str(cell_count) for cell_count in cell_counts}
    if set(valid_by_count) != expected_count_keys or set(invalid_by_count) != expected_count_keys:
        raise ValueError("protocol gid maps must exactly cover cell_counts")

    acceptance = protocol["acceptance"]
    if not isinstance(acceptance, dict) or set(acceptance) != {
        "maximum_fixed_query_time_or_memory_ratio",
        "required_valid_queries_per_topology",
        "required_invalid_queries_per_topology",
        "required_returned_edges_per_topology",
    }:
        raise ValueError("protocol acceptance contract is malformed")
    maximum_ratio = acceptance["maximum_fixed_query_time_or_memory_ratio"]
    if (
        type(maximum_ratio) not in {int, float}
        or not isfinite(float(maximum_ratio))
        or float(maximum_ratio) <= 0.0
    ):
        raise ValueError("protocol maximum ratio must be positive")
    required_valid = _require_exact_int(
        acceptance["required_valid_queries_per_topology"],
        "required_valid_queries_per_topology",
        minimum=1,
    )
    required_invalid = _require_exact_int(
        acceptance["required_invalid_queries_per_topology"],
        "required_invalid_queries_per_topology",
        minimum=1,
    )
    required_edges = _require_exact_int(
        acceptance["required_returned_edges_per_topology"],
        "required_returned_edges_per_topology",
        minimum=1,
    )
    if required_edges != required_valid * fanout:
        raise ValueError("protocol returned-edge count differs from query work")

    topology_names = topology_implementation_contract()["topology_names"]
    for cell_count in cell_counts:
        for topology in topology_names:
            TopologyManifest(cell_count, fanout, topology, seed)
        valid = valid_by_count[str(cell_count)]
        invalid = invalid_by_count[str(cell_count)]
        if not isinstance(valid, list) or len(valid) != required_valid:
            raise ValueError("protocol valid GID work differs from acceptance")
        if not isinstance(invalid, list) or len(invalid) != required_invalid:
            raise ValueError("protocol invalid GID work differs from acceptance")
        if any(type(gid) is not int or not 0 <= gid < cell_count for gid in valid):
            raise ValueError("protocol valid GID is outside its address range")
        if any(type(gid) is not int or 0 <= gid < cell_count for gid in invalid):
            raise ValueError("protocol invalid GID falls inside its address range")

    for field in ("purpose", "evidence_scope"):
        if not isinstance(protocol[field], str) or not protocol[field].strip():
            raise ValueError(f"protocol {field} must be non-empty text")
    for field in ("hypotheses", "notes"):
        value = protocol[field]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
        ):
            raise ValueError(f"protocol {field} must be non-empty text entries")


def topology_implementation_preflight() -> dict[str, object]:
    """Run and record implementation self-checks before any benchmark worker."""

    check = verify_topology_implementation()
    source_path = Path(procedural_topology_module.__file__).resolve()
    canonical_source = source_path.read_bytes().replace(b"\r\n", b"\n")
    return {
        "philox_known_answers": list(check.philox_known_answers),
        "small_materialized_reference": check.materialized_reference,
        "implementation_source": {
            "path": source_path.relative_to(ROOT).as_posix(),
            "canonical_lf_bytes": len(canonical_source),
            "canonical_lf_sha256": hashlib.sha256(canonical_source).hexdigest(),
        },
    }


def _load_protocol() -> tuple[bytes, dict[str, object], str]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    if not isinstance(protocol, dict):
        raise TypeError("benchmark protocol must be a JSON object")
    validate_protocol(protocol)
    return protocol_bytes, protocol, protocol_sha256


def _require_protocol_confirmation(
    confirmed_protocol_sha: str | None,
    protocol_sha256: str,
) -> None:
    if confirmed_protocol_sha != protocol_sha256:
        raise ValueError(
            "protocol SHA-256 confirmation must exactly equal the raw protocol "
            f"SHA-256 {protocol_sha256}"
        )


def _execution_plan(artifact_path: Path) -> dict[str, object]:
    _, protocol, protocol_sha256 = _load_protocol()
    return {
        "will_execute": False,
        "protocol_sha256": protocol_sha256,
        "artifact_path": str(artifact_path),
        "artifact_already_exists": artifact_path.exists(),
        "required_flags": [
            "--execute",
            "--confirm-protocol-sha",
            protocol_sha256,
        ],
        "hypotheses": protocol["hypotheses"],
        "evidence_scope": (
            "CPU scalar address/query-only engineering evidence; no resident, "
            "updated, spiking, token-throughput, GPU, or capability claim"
        ),
    }


def _query_topology(
    *,
    cell_count: int,
    fanout: int,
    topology: str,
    seed: int,
    valid_gids: tuple[int, ...],
    invalid_gids: tuple[int, ...],
) -> dict[str, object]:
    """Run one fixed scalar query workload; never materialize a population."""

    gc.collect()
    tracemalloc.start()
    interval_started = perf_counter()
    construction_started = perf_counter_ns()
    oracle = ProceduralTopology(
        TopologyManifest(
            cell_count=cell_count,
            fanout=fanout,
            topology=topology,
            seed=seed,
        )
    )
    construction_ns = perf_counter_ns() - construction_started

    query_ns: list[int] = []
    query_records: list[dict[str, object]] = []
    invariant_rows: list[bool] = []
    for gid in valid_gids:
        query_started = perf_counter_ns()
        descriptor = oracle.describe_cell(gid)
        edges = oracle.base_edges(gid)
        query_ns.append(perf_counter_ns() - query_started)
        targets = tuple(edge.target_gid for edge in edges)
        invariant_rows.append(
            len(edges) == fanout
            and tuple(edge.edge_slot for edge in edges) == tuple(range(fanout))
            and len(set(targets)) == fanout
            and gid not in targets
            and all(0 <= target < cell_count for target in targets)
        )
        query_records.append(
            {
                "descriptor": asdict(descriptor),
                "targets": targets,
            }
        )

    invalid_cell_queries_rejected = 0
    invalid_edge_queries_rejected = 0
    for gid in invalid_gids:
        try:
            oracle.describe_cell(gid)
        except (TypeError, ValueError):
            invalid_cell_queries_rejected += 1
        try:
            oracle.base_edges(gid)
        except (TypeError, ValueError):
            invalid_edge_queries_rejected += 1

    wall_seconds = perf_counter() - interval_started
    _, traced_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    generated_edges = len(valid_gids) * fanout
    accounting = ScaleAccounting(
        claim=ScaleClaim(
            label="procedural address queries",
            text=(
                "Configured identities were validated and immutable base edges were "
                "examined; this is address/query-only evidence with no resident or "
                "updated cell simulation."
            ),
        ),
        interval=MeasurementInterval(
            name="procedural topology address queries",
            start_step=0,
            step_count=0,
            biological_step_seconds=None,
            wall_seconds=wall_seconds,
        ),
        cells=CellCounts(
            configured=cell_count,
            addressable=cell_count,
            resident_peak=0,
            resident_distinct=0,
            updated_distinct=0,
            spiking_distinct=0,
        ),
        work=WorkCounts(
            cell_step_evaluations=0,
            spike_events=0,
            analytic_catchups=0,
            analytic_step_equivalents=0,
        ),
        edges=EdgeCounts(
            configured=cell_count * fanout,
            resident=0,
            mutable=0,
            generated=generated_edges,
        ),
        events=EventCounts(
            delivered=0,
            queue_peak=0,
            sent_bytes_by_shard_pair=(),
        ),
        memory=MemoryCounts(
            bytes_by_shard=(
                ShardMemory(
                    shard="process-0",
                    host_bytes=traced_peak_bytes,
                    device_bytes=0,
                ),
            )
        ),
    )
    invalid_pass = (
        invalid_cell_queries_rejected == len(invalid_gids)
        and invalid_edge_queries_rejected == len(invalid_gids)
    )
    return {
        "topology": topology,
        "cell_count": cell_count,
        "fanout": fanout,
        "valid_cell_queries": len(valid_gids),
        "valid_edge_row_queries": len(valid_gids),
        "invalid_cell_queries_rejected": invalid_cell_queries_rejected,
        "invalid_edge_queries_rejected": invalid_edge_queries_rejected,
        "sampled_below_2_32": any(gid < 1 << 32 for gid in valid_gids),
        "sampled_at_or_above_2_32": any(gid >= 1 << 32 for gid in valid_gids),
        "construction_ns": construction_ns,
        "query_median_ns": statistics.median(query_ns),
        "query_max_ns": max(query_ns),
        "tracemalloc_peak_bytes": traced_peak_bytes,
        "memory_metric": (
            "peak Python allocation bytes traced during manifest construction and "
            "the fixed scalar query workload; not process RSS or resident cell state"
        ),
        "query_sha256": hashlib.sha256(_canonical_bytes(query_records)).hexdigest(),
        "all_invariants_pass": all(invariant_rows) and invalid_pass,
        "scale_accounting": accounting.to_dict(),
    }


def _worker(cell_count: int, protocol: dict[str, object]) -> dict[str, object]:
    valid_by_count = protocol["valid_gids_by_cell_count"]
    invalid_by_count = protocol["invalid_gids_by_cell_count"]
    if not isinstance(valid_by_count, dict) or not isinstance(invalid_by_count, dict):
        raise TypeError("protocol gid maps must be objects")
    key = str(cell_count)
    valid_gids = tuple(int(value) for value in valid_by_count[key])
    invalid_gids = tuple(int(value) for value in invalid_by_count[key])
    implementation_contract = protocol["implementation_contract"]
    if not isinstance(implementation_contract, dict):
        raise TypeError("protocol implementation contract must be an object")
    return {
        "cell_count": cell_count,
        "results": [
            _query_topology(
                cell_count=cell_count,
                fanout=int(protocol["fanout"]),
                topology=str(topology),
                seed=int(protocol["seed"]),
                valid_gids=valid_gids,
                invalid_gids=invalid_gids,
            )
            for topology in implementation_contract["topology_names"]
        ],
    }


def _log_log_slope(cell_counts: list[float], values: list[float]) -> float:
    x = [log10(value) for value in cell_counts]
    y = [log10(value) for value in values]
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    numerator = sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y))
    denominator = sum((value - x_mean) ** 2 for value in x)
    return numerator / denominator


def _scaling_summary(
    workers: list[dict[str, object]],
    topologies: list[str],
    maximum_ratio: float,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for topology in topologies:
        rows = [
            next(row for row in worker["results"] if row["topology"] == topology)
            for worker in workers
        ]
        cell_counts = [float(row["cell_count"]) for row in rows]
        query_times = [float(row["query_median_ns"]) for row in rows]
        peaks = [float(row["tracemalloc_peak_bytes"]) for row in rows]
        query_ratio = max(query_times) / min(query_times)
        memory_ratio = max(peaks) / min(peaks)
        summary[topology] = {
            "fixed_query_time_max_min_ratio": query_ratio,
            "tracemalloc_peak_max_min_ratio": memory_ratio,
            "fixed_query_time_log_log_slope": _log_log_slope(
                cell_counts, query_times
            ),
            "tracemalloc_peak_log_log_slope": _log_log_slope(cell_counts, peaks),
            "maximum_preregistered_ratio": maximum_ratio,
            "passes_no_material_scale_slope": (
                query_ratio <= maximum_ratio and memory_ratio <= maximum_ratio
            ),
        }
    return summary


def run(
    artifact_path: Path,
    *,
    confirmed_protocol_sha: str,
) -> dict[str, object]:
    _, protocol, protocol_sha256 = _load_protocol()
    _require_protocol_confirmation(confirmed_protocol_sha, protocol_sha256)
    if artifact_path.exists():
        raise FileExistsError(
            f"engineering runner refuses to overwrite artifact: {artifact_path}"
        )
    preflight = topology_implementation_preflight()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if dirty:
        raise RuntimeError("engineering run requires a frozen clean worktree")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    workers: list[dict[str, object]] = []
    for cell_count in protocol["cell_counts"]:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(cell_count),
                "--confirm-protocol-sha",
                protocol_sha256,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        workers.append(json.loads(completed.stdout))
    implementation_contract = protocol["implementation_contract"]
    if not isinstance(implementation_contract, dict):
        raise TypeError("protocol implementation contract must be an object")
    topologies = [str(value) for value in implementation_contract["topology_names"]]
    scaling = _scaling_summary(
        workers,
        topologies,
        float(protocol["acceptance"]["maximum_fixed_query_time_or_memory_ratio"]),
    )
    all_rows = [row for worker in workers for row in worker["results"]]
    artifact_payload: dict[str, object] = {
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "git_commit": git_commit,
        "preflight": preflight,
        "evidence_scope": protocol["evidence_scope"],
        "workers": workers,
        "scaling": scaling,
        "gate_2_pass": (
            all(bool(row["all_invariants_pass"]) for row in all_rows)
            and all(
                bool(item["passes_no_material_scale_slope"])
                for item in scaling.values()
            )
        ),
    }
    atomic_create_canonical_json(
        artifact_path,
        kind=ARTIFACT_KIND,
        payload=artifact_payload,
    )
    persisted = json.loads(artifact_path.read_bytes())
    if not isinstance(persisted, dict):
        raise AssertionError("published artifact envelope must be an object")
    return persisted


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--confirm-protocol-sha")
    args = parser.parse_args(argv)
    if args.worker is not None:
        _, protocol, protocol_sha256 = _load_protocol()
        _require_protocol_confirmation(args.confirm_protocol_sha, protocol_sha256)
        print(json.dumps(_worker(args.worker, protocol), sort_keys=True))
        return
    if not args.execute:
        print(json.dumps(_execution_plan(args.artifact), indent=2, sort_keys=True))
        return
    if args.confirm_protocol_sha is None:
        parser.error("--execute requires --confirm-protocol-sha")
    print(
        json.dumps(
            run(
                args.artifact,
                confirmed_protocol_sha=args.confirm_protocol_sha,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
