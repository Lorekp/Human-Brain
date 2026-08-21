"""Reproducible exact CPU scaling from 100k to one million human cells."""

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

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks/native_scaling_protocol_v2.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/native_scaling_v2.json"


def _tokens(count: int, token_cells: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, token_cells, count, dtype=np.int64)


def _time_frontier(
    circuit: RecurrentCellCircuit,
    tokens: np.ndarray,
    repeats: int,
) -> tuple[float, list[int], list[tuple[int, int, int]]]:
    samples: list[float] = []
    active_counts: list[int] = []
    profile_ns: list[tuple[int, int, int]] = []
    for repeat in range(repeats):
        circuit.reset_sequence()
        started = perf_counter()
        for token in tokens:
            circuit.observe(int(token))
            if repeat == 0:
                active_counts.append(circuit._native_kernel.last_active_cells)
                profile_ns.append(circuit._native_kernel.last_frontier_profile_ns)
        samples.append(perf_counter() - started)
    return median(samples), active_counts, profile_ns


def _time_dense(
    circuit: RecurrentCellCircuit,
    tokens: np.ndarray,
    repeats: int,
    method_name: str,
) -> float:
    samples: list[float] = []
    method = getattr(circuit._native_kernel, method_name)
    for _ in range(repeats):
        circuit.reset_sequence()
        started = perf_counter()
        for token in tokens:
            method(int(token))
        samples.append(perf_counter() - started)
    return median(samples)


def _worker(cell_count: int, protocol: dict[str, object]) -> dict[str, object]:
    loaded_build_info()
    token_cells = int(protocol["token_cells"])
    repeats = int(protocol["timing_repeats"])
    frontier_count = int(protocol["frontier_token_counts"][str(cell_count)])
    dense_count = int(protocol["dense_token_counts"][str(cell_count)])
    before = _process_memory()
    started = perf_counter()
    circuit = RecurrentCellCircuit(token_cells, _config(cell_count, protocol))
    construction_seconds = perf_counter() - started
    after_construction = _process_memory()

    warmup = _tokens(8, token_cells, int(protocol["seed"]) + cell_count + 1)
    circuit.reset_sequence()
    for token in warmup:
        circuit.observe(int(token))
    frontier_tokens = _tokens(
        frontier_count,
        token_cells,
        int(protocol["seed"]) + cell_count,
    )
    frontier_seconds, active_counts, profile_ns = _time_frontier(
        circuit,
        frontier_tokens,
        repeats,
    )
    dense_tokens = frontier_tokens[:dense_count]
    serial_seconds = _time_dense(circuit, dense_tokens, repeats, "_observe_serial")
    parallel_seconds = _time_dense(circuit, dense_tokens, repeats, "_observe_parallel")
    after_run = _process_memory()
    token_steps = circuit._token_steps

    def timing(
        seconds: float,
        tokens: int,
        *,
        dense: bool,
    ) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "tokens": tokens,
            "seconds": seconds,
            "tokens_per_second": tokens / seconds,
        }
        metric = (
            "actual_cell_steps_per_second"
            if dense
            else "configured_cell_step_equivalents_per_second"
        )
        result[metric] = cell_count * token_steps * tokens / seconds
        return result

    return {
        "configured_active_cells": cell_count,
        "token_steps": token_steps,
        "sparse_recurrent_edges": cell_count * int(protocol["recurrent_fanout"]),
        "construction_seconds": construction_seconds,
        "persistent_array_mib": _persistent_array_mib(circuit),
        "frontier": {
            **timing(frontier_seconds, frontier_count, dense=False),
            "first_token_updated_cells": active_counts[0],
            "last_token_updated_cells": active_counts[-1],
            "maximum_updated_cells": max(active_counts),
            "median_full_state_scan_microseconds": median(
                profile[0] for profile in profile_ns
            )
            / 1_000.0,
            "median_frontier_build_microseconds": median(
                profile[1] for profile in profile_ns
            )
            / 1_000.0,
            "median_frontier_steps_microseconds": median(
                profile[2] for profile in profile_ns
            )
            / 1_000.0,
        },
        "dense_serial": timing(serial_seconds, dense_count, dense=True),
        "dense_parallel": timing(parallel_seconds, dense_count, dense=True),
        "dense_parallel_speedup": serial_seconds / parallel_seconds,
        "rayon_threads": int(os.environ.get("RAYON_NUM_THREADS", "0")),
        "baseline_memory": before,
        "after_construction_memory": after_construction,
        "after_run_memory": after_run,
        "private_memory_increment_mib": (
            after_run["private_memory_mib"] - before["private_memory_mib"]
        ),
        "normal_matrix_allocated": circuit._normal_matrix is not None,
    }


def _state_arrays(circuit: RecurrentCellCircuit) -> tuple[np.ndarray, ...]:
    state = circuit._reservoir.state
    return (
        circuit._synaptic_current_pa,
        circuit._eligibility,
        state.voltage_relative_mv,
        state.after_spike_currents_pa,
        state.threshold_spike_mv,
        state.threshold_voltage_mv,
        state.refractory_steps,
    )


def _state_sha256(circuit: RecurrentCellCircuit) -> str:
    digest = hashlib.sha256()
    for array in _state_arrays(circuit):
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _equivalence(protocol: dict[str, object]) -> dict[str, object]:
    cells = int(protocol["equivalence_cells"])
    token_count = int(protocol["equivalence_tokens"])
    token_cells = int(protocol["token_cells"])
    optimized = RecurrentCellCircuit(token_cells, _config(cells, protocol))
    reference = RecurrentCellCircuit(token_cells, _config(cells, protocol))
    topology_equal = np.array_equal(
        optimized._recurrent_targets,
        reference._recurrent_targets,
    ) and np.array_equal(optimized._afferent_targets, reference._afferent_targets)
    tokens = _tokens(token_count, token_cells, int(protocol["seed"]) + 901)
    optimized_spikes: list[int] = []
    reference_spikes: list[int] = []
    optimized_started = perf_counter()
    for token in tokens:
        optimized_spikes.append(optimized._native_kernel.observe(int(token)))
    optimized_seconds = perf_counter() - optimized_started
    reference_started = perf_counter()
    for token in tokens:
        reference_spikes.append(reference._native_kernel._observe_reference_v1(int(token)))
    reference_seconds = perf_counter() - reference_started
    arrays_equal = all(
        np.array_equal(actual, expected)
        for actual, expected in zip(
            _state_arrays(optimized),
            _state_arrays(reference),
            strict=True,
        )
    )
    return {
        "cells": cells,
        "tokens": token_count,
        "topology_arrays_equal": topology_equal,
        "spike_counts_equal": optimized_spikes == reference_spikes,
        "persistent_state_bit_exact": arrays_equal,
        "optimized_state_sha256": _state_sha256(optimized),
        "reference_state_sha256": _state_sha256(reference),
        "optimized_tokens_per_second": token_count / optimized_seconds,
        "reference_v1_tokens_per_second": token_count / reference_seconds,
        "speedup": reference_seconds / optimized_seconds,
        "maximum_updated_cells": optimized._native_kernel.last_active_cells,
    }


def _exponent(results: list[dict[str, object]], path: tuple[str, ...]) -> float:
    cells = np.asarray([float(result["configured_active_cells"]) for result in results])
    values: list[float] = []
    for result in results:
        value: object = result
        for key in path:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    return float(np.polyfit(np.log(cells), np.log(np.asarray(values)), 1)[0])


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
    equivalence_process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--equivalence"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    artifact: dict[str, object] = {
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "results": results,
        "equivalence": json.loads(equivalence_process.stdout),
        "scaling": {
            "frontier_seconds_per_token_exponent": (
                _exponent(results, ("frontier", "seconds"))
                - _exponent(results, ("frontier", "tokens"))
            ),
            "dense_parallel_seconds_per_token_exponent": (
                _exponent(results, ("dense_parallel", "seconds"))
                - _exponent(results, ("dense_parallel", "tokens"))
            ),
            "persistent_memory_exponent": _exponent(results, ("persistent_array_mib",)),
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
    parser.add_argument("--equivalence", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.worker is not None:
        print(json.dumps(_worker(args.worker, protocol), sort_keys=True))
    elif args.equivalence:
        print(json.dumps(_equivalence(protocol), sort_keys=True))
    else:
        print(json.dumps(run(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
