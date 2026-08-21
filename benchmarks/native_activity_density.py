"""Exact frontier speed as the fraction of nonquiescent cells increases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from humanbrain.native import loaded_build_info
from humanbrain.recurrent import RecurrentCellCircuit
from native_scaling import _config, _persistent_array_mib, _process_memory
from native_scaling_v2 import _state_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks/native_activity_density_protocol_v1.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/native_activity_density_v1.json"


def _seed_density(
    circuit: RecurrentCellCircuit,
    density: float,
    protocol: dict[str, object],
) -> int:
    circuit.reset_sequence()
    cells = circuit.config.reservoir_cells
    count = max(1, round(cells * density))
    indices = (np.arange(count, dtype=np.int64) * cells // count).astype(np.intp)
    circuit._synaptic_current_pa[indices] = float(protocol["seeded_synaptic_current_pa"])
    circuit._eligibility[indices] = float(protocol["seeded_eligibility"])
    return count


def _time_mode(
    circuit: RecurrentCellCircuit,
    density: float,
    protocol: dict[str, object],
    method_name: str,
) -> tuple[float, tuple[int, int, int]]:
    samples: list[float] = []
    profiles: list[tuple[int, int, int]] = []
    method = getattr(circuit._native_kernel, method_name)
    for _ in range(int(protocol["timing_repeats"])):
        _seed_density(circuit, density, protocol)
        started = perf_counter()
        method(int(protocol["token_id"]))
        samples.append(perf_counter() - started)
        profiles.append(circuit._native_kernel.last_frontier_profile_ns)
    return median(samples), tuple(int(median(values)) for values in zip(*profiles, strict=True))


def _exact_pair(
    circuit: RecurrentCellCircuit,
    density: float,
    protocol: dict[str, object],
) -> dict[str, object]:
    seeded_cells = _seed_density(circuit, density, protocol)
    frontier_spikes = circuit._native_kernel.observe(int(protocol["token_id"]))
    frontier_sha = _state_sha256(circuit)
    frontier_updated = circuit._native_kernel.last_active_cells

    _seed_density(circuit, density, protocol)
    dense_spikes = circuit._native_kernel._observe_parallel(int(protocol["token_id"]))
    dense_sha = _state_sha256(circuit)
    return {
        "requested_density": density,
        "seeded_nonzero_cells": seeded_cells,
        "frontier_updated_cells": frontier_updated,
        "frontier_updated_fraction": frontier_updated / circuit.config.reservoir_cells,
        "spike_counts_equal": frontier_spikes == dense_spikes,
        "persistent_state_bit_exact": frontier_sha == dense_sha,
        "frontier_state_sha256": frontier_sha,
        "dense_state_sha256": dense_sha,
    }


def _worker(cell_count: int, protocol: dict[str, object]) -> dict[str, object]:
    loaded_build_info()
    before = _process_memory()
    started = perf_counter()
    circuit = RecurrentCellCircuit(
        int(protocol["token_cells"]),
        _config(cell_count, protocol),
    )
    construction_seconds = perf_counter() - started
    density_results: list[dict[str, object]] = []
    for density_value in protocol["densities"]:
        density = float(density_value)
        exact = _exact_pair(circuit, density, protocol)
        frontier_seconds, profile_ns = _time_mode(
            circuit,
            density,
            protocol,
            "observe",
        )
        dense_seconds, _ = _time_mode(
            circuit,
            density,
            protocol,
            "_observe_parallel",
        )
        density_results.append(
            {
                **exact,
                "frontier_seconds_per_token": frontier_seconds,
                "frontier_tokens_per_second": 1.0 / frontier_seconds,
                "dense_seconds_per_token": dense_seconds,
                "dense_tokens_per_second": 1.0 / dense_seconds,
                "frontier_over_dense_speedup": dense_seconds / frontier_seconds,
                "full_state_scan_microseconds": profile_ns[0] / 1_000.0,
                "frontier_build_microseconds": profile_ns[1] / 1_000.0,
                "frontier_steps_microseconds": profile_ns[2] / 1_000.0,
            }
        )
    after = _process_memory()
    return {
        "configured_active_cells": cell_count,
        "token_steps": circuit._token_steps,
        "construction_seconds": construction_seconds,
        "persistent_array_mib": _persistent_array_mib(circuit),
        "private_memory_increment_mib": (
            after["private_memory_mib"] - before["private_memory_mib"]
        ),
        "rayon_threads": int(os.environ.get("RAYON_NUM_THREADS", "0")),
        "densities": density_results,
    }


def run(artifact_path: Path) -> dict[str, object]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(protocol["dense_worker_threads"])
    results: list[dict[str, object]] = []
    for cells in protocol["cell_counts"]:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(cells)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        results.append(json.loads(completed.stdout))
    artifact: dict[str, object] = {
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "results": results,
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
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.worker is not None:
        print(json.dumps(_worker(args.worker, protocol), sort_keys=True))
    else:
        print(json.dumps(run(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
