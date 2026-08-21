from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from humanbrain.cuda import build_cuda, cuda_device_name
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "cuda_scaling_protocol_v1.json"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "cuda_scaling_v1.json"


def _config(cells: int, protocol: dict[str, object]) -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
        reservoir_cells=cells,
        active_reservoir_cells=cells,
        input_fanout=int(protocol["input_fanout"]),
        recurrent_fanout=int(protocol["recurrent_fanout"]),
        token_interval_ms=float(protocol["token_interval_ms"]),
        input_pulse_ms=1.5,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        inhibitory_fraction=0.2,
        random_seed=int(protocol["seed"]),
        recurrent_topology=str(protocol["topology"]),
        readout_learning_rule="three_factor",
    )


def _all_arrays(circuit: RecurrentCellCircuit) -> list[np.ndarray]:
    arrays = [value for value in vars(circuit).values() if isinstance(value, np.ndarray)]
    arrays.extend(
        value for value in vars(circuit._reservoir).values() if isinstance(value, np.ndarray)
    )
    arrays.extend(vars(circuit._reservoir.state).values())
    return arrays


def _persistent_array_mib(circuit: RecurrentCellCircuit) -> float:
    unique = {id(array): array for array in _all_arrays(circuit)}
    return sum(array.nbytes for array in unique.values()) / (1024.0 * 1024.0)


def _gpu_snapshot() -> dict[str, object] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    values = [value.strip() for value in completed.stdout.strip().splitlines()[0].split(",")]
    if len(values) != 6:
        return {"raw": completed.stdout.strip()}
    return {
        "name": values[0],
        "driver_version": values[1],
        "memory_used_mib": float(values[2]),
        "memory_total_mib": float(values[3]),
        "utilization_percent": float(values[4]),
        "power_watts": float(values[5]),
    }


def _state_arrays(circuit: RecurrentCellCircuit) -> list[np.ndarray]:
    state = circuit._reservoir.state
    return [
        circuit._synaptic_current_pa,
        circuit._eligibility,
        state.voltage_relative_mv,
        state.after_spike_currents_pa,
        state.threshold_spike_mv,
        state.threshold_voltage_mv,
        state.refractory_steps,
    ]


def _state_digest(circuit: RecurrentCellCircuit) -> str:
    digest = hashlib.sha256()
    for array in _state_arrays(circuit):
        digest.update(array.tobytes())
    return digest.hexdigest()


def _equivalence(protocol: dict[str, object]) -> dict[str, object]:
    cells = int(protocol["equivalence_cells"])
    tokens = np.random.default_rng(int(protocol["seed"]) + 1).integers(
        0,
        int(protocol["token_cells"]),
        int(protocol["equivalence_tokens"]),
    )
    previous = os.environ.get("HUMANBRAIN_DEVICE")
    try:
        os.environ["HUMANBRAIN_DEVICE"] = "cpu"
        cpu = RecurrentCellCircuit(int(protocol["token_cells"]), _config(cells, protocol))
        os.environ["HUMANBRAIN_DEVICE"] = "cuda"
        cuda = RecurrentCellCircuit(int(protocol["token_cells"]), _config(cells, protocol))
        exact = True
        spike_totals_equal = True
        for token in tokens:
            cpu_step = cpu.observe(int(token))
            cuda_step = cuda.observe(int(token))
            spike_totals_equal &= cpu_step.reservoir_spikes == cuda_step.reservoir_spikes
            exact &= all(
                np.array_equal(cuda_array, cpu_array)
                for cuda_array, cpu_array in zip(
                    _state_arrays(cuda), _state_arrays(cpu), strict=True
                )
            )
        digest = _state_digest(cuda)
        cuda.reset_sequence()
        cpu.reset_sequence()
        for token in (7, 7):
            spike_totals_equal &= (
                cuda.observe(token).reservoir_spikes == cpu.observe(token).reservoir_spikes
            )
        reset_exact = all(
            np.array_equal(cuda_array, cpu_array)
            for cuda_array, cpu_array in zip(_state_arrays(cuda), _state_arrays(cpu), strict=True)
        )
        cuda._native_kernel.close()
        cuda._native_kernel.close()
        return {
            "cells": cells,
            "tokens": int(tokens.size),
            "all_state_arrays_bit_exact": bool(exact),
            "spike_totals_exact": bool(spike_totals_equal),
            "reset_and_repeated_token_exact": bool(reset_exact),
            "final_state_sha256": digest,
        }
    finally:
        if previous is None:
            os.environ.pop("HUMANBRAIN_DEVICE", None)
        else:
            os.environ["HUMANBRAIN_DEVICE"] = previous


def _build_timing() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="humanbrain-cuda-build-") as directory:
        started = time.perf_counter()
        build = build_cuda(directory, force=True)
        cold_seconds = time.perf_counter() - started
        started = time.perf_counter()
        cached = build_cuda(directory)
        cached_seconds = time.perf_counter() - started
    return {
        "source_digest": build.source_digest,
        "architecture": build.architecture,
        "cold_compile_seconds": cold_seconds,
        "cached_build_lookup_seconds": cached_seconds,
        "cache_identity_equal": build == cached,
    }


def _run_size(cells: int, tokens: int, protocol: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    circuit = RecurrentCellCircuit(int(protocol["token_cells"]), _config(cells, protocol))
    construction_seconds = time.perf_counter() - started
    kernel = circuit._native_kernel
    if kernel.backend != "cuda":
        raise RuntimeError(f"expected CUDA backend, got {kernel.backend}")
    stream = np.random.default_rng(int(protocol["seed"]) + cells).integers(
        0, circuit.token_cells, tokens
    )
    started = time.perf_counter()
    circuit.observe(int(stream[0]))
    first_token_seconds = time.perf_counter() - started
    first_kernel_ms = kernel.last_kernel_milliseconds
    for token in stream[: int(protocol["warmup_tokens"])]:
        circuit.observe(int(token))

    elapsed_runs: list[float] = []
    kernel_runs: list[float] = []
    spike_runs: list[int] = []
    for _ in range(int(protocol["timing_repeats"])):
        circuit.reset_sequence()
        kernel_ms = 0.0
        spikes = 0
        started = time.perf_counter()
        for token in stream:
            step = circuit.observe(int(token))
            spikes += step.reservoir_spikes
            kernel_ms += kernel.last_kernel_milliseconds
        elapsed_runs.append(time.perf_counter() - started)
        kernel_runs.append(kernel_ms / 1_000.0)
        spike_runs.append(spikes)
    elapsed = float(np.median(elapsed_runs))
    kernel_seconds = float(np.median(kernel_runs))
    token_steps = circuit._token_steps
    result = {
        "cells": cells,
        "active_cells": circuit.active_reservoir_cells,
        "tokens": tokens,
        "token_steps": token_steps,
        "biological_token_interval_ms": token_steps * circuit.cell_parameters.dt_ms,
        "sparse_recurrent_edges": cells * int(protocol["recurrent_fanout"]),
        "touched_cells_per_step": cells,
        "recurrent_edges_examined_per_step": cells * int(protocol["recurrent_fanout"]),
        "construction_seconds": construction_seconds,
        "first_token_seconds": first_token_seconds,
        "first_kernel_milliseconds": first_kernel_ms,
        "timing_seconds": elapsed,
        "kernel_seconds": kernel_seconds,
        "end_to_end_tokens_per_second": tokens / elapsed,
        "kernel_tokens_per_second": tokens / kernel_seconds,
        "cell_steps_per_second": cells * token_steps * tokens / elapsed,
        "reservoir_spikes": int(np.median(spike_runs)),
        "mean_spikes_per_token": float(np.median(spike_runs)) / tokens,
        "spikes_per_cell_step": float(np.median(spike_runs)) / (cells * token_steps * tokens),
        "host_persistent_array_mib": _persistent_array_mib(circuit),
        "device_persistent_array_mib": kernel.device_bytes / (1024.0 * 1024.0),
        "host_to_device_bytes_per_token": kernel.host_to_device_bytes_per_token,
        "device_to_host_bytes_per_token": kernel.device_to_host_bytes_per_token,
        "normal_matrix_allocated": circuit._normal_matrix is not None,
        "backend": kernel.backend,
        "device": kernel.device_name,
        "state_sha256": _state_digest(circuit),
    }
    kernel.close()
    return result


def run(
    artifact_path: Path,
    *,
    cell_counts: list[int] | None = None,
    token_override: int | None = None,
) -> dict[str, object]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    cells = cell_counts or [int(value) for value in protocol["cell_counts"]]
    previous = os.environ.get("HUMANBRAIN_DEVICE")
    os.environ["HUMANBRAIN_DEVICE"] = "cuda"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = {
            "protocol": protocol,
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "executed_cell_counts": cells,
            "omitted_protocol_cell_counts": [
                int(value) for value in protocol["cell_counts"] if int(value) not in cells
            ],
            "device": cuda_device_name(),
            "gpu_before": _gpu_snapshot(),
            "build": _build_timing(),
            "equivalence": _equivalence(protocol),
            "results": [],
            "gpu_after": None,
            "complete": False,
        }
        for cell_count in cells:
            report["results"].append(
                _run_size(
                    cell_count,
                    token_override or int(protocol["token_counts"].get(str(cell_count), 16)),
                    protocol,
                )
            )
            artifact_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        report["gpu_after"] = _gpu_snapshot()
        report["complete"] = True
    finally:
        if previous is None:
            os.environ.pop("HUMANBRAIN_DEVICE", None)
        else:
            os.environ["HUMANBRAIN_DEVICE"] = previous
    report["gate"] = {
        "minimum_cells": 100_000,
        "minimum_end_to_end_tokens_per_second": 100.0,
        "passing_sizes": [
            result["cells"]
            for result in report["results"]
            if result["cells"] > 100_000 and result["end_to_end_tokens_per_second"] > 100.0
        ],
    }
    artifact_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--cells", help="comma-separated cell counts")
    parser.add_argument("--tokens", type=int)
    args = parser.parse_args()
    cell_counts = [int(value) for value in args.cells.split(",")] if args.cells else None
    print(
        json.dumps(
            run(args.artifact, cell_counts=cell_counts, token_override=args.tokens), indent=2
        )
    )


if __name__ == "__main__":
    main()
