"""Construction scaling for direct structured human-cell topology samplers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from math import ceil, gcd
from pathlib import Path
from time import perf_counter

import numpy as np
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
from humanbrain.topology import build_recurrent_targets
from native_scaling import _process_memory

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks/structured_topology_protocol_v3.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/structured_topology_construction_v3.json"


def _structure_invariants(
    targets: np.ndarray,
    topology: str,
) -> dict[str, float]:
    cells, fanout = targets.shape
    sources = np.arange(cells, dtype=np.int64)[:, None]
    if topology in {"single_scale_groups", "nested_groups"}:
        micro_size = max(fanout + 1, round(cells**0.5))
        metrics = {
            "same_micro_edge_fraction": float(
                np.mean(targets // micro_size == sources // micro_size)
            )
        }
        if topology == "nested_groups":
            metrics["same_meso_edge_fraction"] = float(
                np.mean(targets // (micro_size * 4) == sources // (micro_size * 4))
            )
        return metrics
    if topology == "mixed_membership":
        group_count = max(4, round(cells**0.5))
        while gcd(5, group_count) != 1:
            group_count += 1
        source_primary = sources % group_count
        source_secondary = (sources * 5 + 1) % group_count
        target_primary = targets % group_count
        target_secondary = (targets * 5 + 1) % group_count
        shared = (
            (target_primary == source_primary)
            | (target_primary == source_secondary)
            | (target_secondary == source_primary)
            | (target_secondary == source_secondary)
        )
        return {"shared_membership_edge_fraction": float(np.mean(shared))}

    local_edges = fanout - max(1, round(0.15 * fanout))
    local_targets = targets[:, :local_edges].astype(np.int64)
    long_targets = targets[:, local_edges:].astype(np.int64)
    if topology == "ring_1d":
        local_separation = np.abs(local_targets - sources)
        local_separation = np.minimum(local_separation, cells - local_separation)
        long_separation = np.abs(long_targets - sources)
        long_separation = np.minimum(long_separation, cells - long_separation)
        return {
            "local_offset_edge_fraction": float(
                np.mean(local_separation <= round(cells**0.25))
            ),
            "long_range_edge_fraction": float(np.mean(long_separation >= cells // 4)),
        }

    if topology == "layered_sheet":
        surface_cells = ceil(cells / 6)
        side = ceil(surface_cells**0.5)
        source_surface = sources // 6
        local_surface = local_targets // 6
        long_surface = long_targets // 6
        local_dx = np.abs(local_surface % side - source_surface % side)
        local_dy = np.abs(local_surface // side - source_surface // side)
        local_depth = np.abs(local_targets % 6 - sources % 6)
        long_dx = long_surface % side - source_surface % side
        long_dy = long_surface // side - source_surface // side
        threshold = max(2, side // 4)
        return {
            "local_offset_edge_fraction": float(
                np.mean((local_dx <= 2) & (local_dy <= 2) & (local_depth <= 1))
            ),
            "long_range_edge_fraction": float(
                np.mean(long_dx * long_dx + long_dy * long_dy >= threshold * threshold)
            ),
        }

    dimensions = 2 if topology == "sheet_2d" else 3
    side = ceil(cells ** (1.0 / dimensions))
    local_squared = np.zeros_like(local_targets)
    long_squared = np.zeros_like(long_targets)
    local_within_offset = np.ones_like(local_targets, dtype=np.bool_)
    for dimension in range(dimensions):
        divisor = side**dimension
        source_coordinate = (sources // divisor) % side
        local_difference = np.abs((local_targets // divisor) % side - source_coordinate)
        long_difference = (long_targets // divisor) % side - source_coordinate
        local_within_offset &= local_difference <= 2
        local_squared += local_difference * local_difference
        long_squared += long_difference * long_difference
    threshold = max(2, side // 4)
    return {
        "local_offset_edge_fraction": float(np.mean(local_within_offset)),
        "long_range_edge_fraction": float(np.mean(long_squared >= threshold * threshold)),
    }


def _worker(
    cells: int,
    topology: str,
    protocol: dict[str, object],
) -> dict[str, object]:
    fanout = int(protocol["fanout"])
    before = _process_memory()
    started = perf_counter()
    targets = build_recurrent_targets(
        cells,
        fanout,
        np.random.default_rng(int(protocol["seed"])),
        topology,
    )
    seconds = perf_counter() - started
    after = _process_memory()
    sources = np.arange(cells, dtype=np.int32)[:, None]
    edge_records = cells * fanout
    scale_accounting = ScaleAccounting(
        claim=ScaleClaim(
            label="materialized topology construction",
            text=(
                "Configured edges were generated into resident host target records; "
                "no cell dynamics ran."
            ),
        ),
        interval=MeasurementInterval(
            name="topology construction",
            start_step=0,
            step_count=0,
            biological_step_seconds=None,
            wall_seconds=seconds,
        ),
        cells=CellCounts(
            configured=cells,
            addressable=0,
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
            configured=edge_records,
            resident=edge_records,
            mutable=0,
            generated=edge_records,
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
                    host_bytes=int(targets.nbytes),
                    device_bytes=0,
                ),
            )
        ),
    )
    return {
        "cells": cells,
        "topology": topology,
        "fanout": fanout,
        "seconds": seconds,
        "cells_per_second": cells / seconds,
        "output_mib": targets.nbytes / (1024.0 * 1024.0),
        "private_memory_increment_mib": (
            after["private_memory_mib"] - before["private_memory_mib"]
        ),
        "target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
        "degree_matched": targets.shape == (cells, fanout),
        "indices_in_range": bool(np.all((targets >= 0) & (targets < cells))),
        "self_edges": int(np.count_nonzero(targets == sources)),
        "rows_with_duplicate_targets": int(
            np.count_nonzero(np.any(np.diff(np.sort(targets, axis=1), axis=1) == 0, axis=1))
        ),
        "structure_invariants": _structure_invariants(targets, topology),
        "scale_accounting": scale_accounting.to_dict(),
    }


def _exponent(rows: list[dict[str, object]], topology: str, metric: str) -> float:
    selected = [row for row in rows if row["topology"] == topology]
    cells = np.asarray([float(row["cells"]) for row in selected])
    values = np.asarray([float(row[metric]) for row in selected])
    return float(np.polyfit(np.log(cells), np.log(values), 1)[0])


def run(artifact_path: Path) -> dict[str, object]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    results: list[dict[str, object]] = []
    for topology in protocol["topologies"]:
        for cells in protocol["cell_counts"]:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(cells),
                    "--topology",
                    str(topology),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            results.append(json.loads(completed.stdout))
    artifact: dict[str, object] = {
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "results": results,
        "scaling": {
            topology: {
                "construction_time_exponent": _exponent(results, topology, "seconds"),
                "output_memory_exponent": _exponent(results, topology, "output_mib"),
            }
            for topology in protocol["topologies"]
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--topology")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.worker is not None:
        if args.topology is None:
            parser.error("--worker requires --topology")
        print(json.dumps(_worker(args.worker, args.topology, protocol), sort_keys=True))
    else:
        print(json.dumps(run(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
