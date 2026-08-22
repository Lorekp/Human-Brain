"""Reproducible native sparse-kernel speed and memory scaling benchmark."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter, time_ns

import numpy as np
from humanbrain.native import build_native, loaded_build_info
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks/native_scaling_protocol_v1.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/native_scaling_v1.json"


def _config(cell_count: int, protocol: dict[str, object]) -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
        reservoir_cells=cell_count,
        input_fanout=int(protocol["input_fanout"]),
        recurrent_fanout=int(protocol["recurrent_fanout"]),
        token_interval_ms=float(protocol["token_interval_ms"]),
        input_pulse_ms=1.5,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        recurrent_topology=str(protocol["topology"]),
        random_seed=int(protocol["seed"]),
    )


def _all_arrays(circuit: RecurrentCellCircuit) -> list[np.ndarray]:
    population = circuit._reservoir
    values: list[object] = []
    values.extend(vars(circuit).values())
    values.extend(vars(population).values())
    values.extend(vars(population.state).values())
    values.extend(vars(population._native_kernel).values())
    arrays = [value for value in values if isinstance(value, np.ndarray)]
    return list({id(array): array for array in arrays}.values())


def _persistent_array_mib(circuit: RecurrentCellCircuit) -> float:
    return sum(array.nbytes for array in _all_arrays(circuit)) / (1024.0 * 1024.0)


def _process_memory() -> dict[str, float]:
    if os.name != "nt":
        import resource

        peak_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {
            "working_set_mib": peak_kib / 1024.0,
            "peak_working_set_mib": peak_kib / 1024.0,
            "private_memory_mib": peak_kib / 1024.0,
        }

    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    process = get_current_process()
    ok = get_process_memory_info(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    scale = 1024.0 * 1024.0
    return {
        "working_set_mib": counters.WorkingSetSize / scale,
        "peak_working_set_mib": counters.PeakWorkingSetSize / scale,
        "private_memory_mib": counters.PrivateUsage / scale,
    }


def _token_stream(token_cells: int, token_count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, token_cells, token_count, dtype=np.int64)


def _time_tokens(circuit: RecurrentCellCircuit, tokens: np.ndarray, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        circuit.reset_sequence()
        started = perf_counter()
        for token in tokens:
            circuit.observe(int(token))
        samples.append(perf_counter() - started)
    return median(samples)


def _python_step(circuit: RecurrentCellCircuit, current: np.ndarray) -> np.ndarray:
    """Removed Python/NumPy production cell step, used only as timing oracle."""
    population = circuit._reservoir
    p = population.parameters
    constants = population.constants
    state = population.state
    active = state.refractory_steps == 0
    np.subtract(state.refractory_steps, 1, out=state.refractory_steps, where=~active)
    if not active.any():
        return active
    previous_voltage = state.voltage_relative_mv
    previous_after = state.after_spike_currents_pa
    previous_after.sum(axis=1, out=population._work_after_sum)
    np.add(current, population._work_after_sum, out=population._work_total_current)
    total_current = population._work_total_current
    np.multiply(p.leak_ns, previous_voltage, out=population._work_next_voltage)
    np.subtract(total_current, population._work_next_voltage, out=population._work_next_voltage)
    np.multiply(
        constants.voltage_step_scale,
        population._work_next_voltage,
        out=population._work_next_voltage,
    )
    np.add(previous_voltage, population._work_next_voltage, out=population._work_next_voltage)
    next_voltage = population._work_next_voltage
    beta = total_current * constants.threshold_beta_scale
    next_voltage_threshold = (
        constants.threshold_exp_decay * state.threshold_voltage_mv
        + constants.threshold_voltage_factor * previous_voltage
        + constants.threshold_beta_factor * beta
    )
    np.multiply(
        state.threshold_spike_mv,
        constants.spike_threshold_decay,
        out=population._work_next_spike_threshold,
    )
    next_spike_threshold = population._work_next_spike_threshold
    np.multiply(previous_after, constants.after_spike_decay, out=population._work_next_after)
    next_after = population._work_next_after
    np.copyto(state.voltage_relative_mv, next_voltage, where=active)
    np.copyto(state.threshold_voltage_mv, next_voltage_threshold, where=active)
    np.copyto(state.threshold_spike_mv, next_spike_threshold, where=active)
    np.copyto(state.after_spike_currents_pa, next_after, where=active[:, None])
    crossed = active & (
        next_voltage
        > (p.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold)
    )
    if not crossed.any():
        return crossed
    spike_rows = np.flatnonzero(crossed)
    state.voltage_relative_mv[spike_rows] = (
        p.voltage_reset_scale * next_voltage[spike_rows] + p.voltage_reset_offset_mv
    )
    state.after_spike_currents_pa[spike_rows] = (
        constants.after_spike_amplitudes_pa
        + next_after[spike_rows] * constants.after_spike_cut_decay
    )
    state.threshold_spike_mv[spike_rows] = (
        next_spike_threshold[spike_rows] * constants.spike_cut_threshold_decay
        + p.threshold_spike_increment_mv
    )
    state.refractory_steps[spike_rows] = p.spike_cut_steps
    return crossed


def _python_observe(circuit: RecurrentCellCircuit, token_id: int) -> None:
    afferents = circuit._afferent_targets[token_id]
    step_current_pa = np.empty_like(circuit._synaptic_current_pa)
    for step in range(circuit._token_steps):
        np.copyto(step_current_pa, circuit._synaptic_current_pa)
        if step < circuit._pulse_steps:
            step_current_pa[afferents] += circuit.config.input_current_pa
        spikes = _python_step(circuit, step_current_pa)
        circuit._synaptic_current_pa *= circuit._recurrent_decay
        circuit._eligibility *= circuit._eligibility_decay
        if spikes.any():
            presynaptic = np.flatnonzero(spikes)
            circuit._eligibility[presynaptic] += 1.0
            targets = circuit._recurrent_targets[presynaptic].reshape(-1)
            weights = circuit._recurrent_weights_pa[presynaptic].reshape(-1)
            np.add.at(circuit._synaptic_current_pa, targets, weights)


def _reference_timing(protocol: dict[str, object]) -> dict[str, float | int]:
    cells = int(protocol["reference_cell_count"])
    token_count = int(protocol["reference_token_count"])
    repeats = int(protocol["reference_repeats"])
    token_cells = int(protocol["token_cells"])
    circuit = RecurrentCellCircuit(token_cells, _config(cells, protocol))
    tokens = _token_stream(token_cells, token_count, int(protocol["seed"]) + 1)
    samples: list[float] = []
    for _ in range(repeats):
        circuit.reset_sequence()
        started = perf_counter()
        for token in tokens:
            _python_observe(circuit, int(token))
        samples.append(perf_counter() - started)
    seconds = median(samples)
    return {
        "cells": cells,
        "tokens": token_count,
        "seconds": seconds,
        "tokens_per_second": token_count / seconds,
        "cell_steps_per_second": cells * circuit._token_steps * token_count / seconds,
    }


def _worker(cell_count: int, token_count: int, protocol: dict[str, object]) -> dict[str, object]:
    token_cells = int(protocol["token_cells"])
    repeats = int(protocol["timing_repeats"])
    before = _process_memory()
    started = perf_counter()
    circuit = RecurrentCellCircuit(token_cells, _config(cell_count, protocol))
    construction_seconds = perf_counter() - started
    after_construction = _process_memory()
    tokens = _token_stream(token_cells, token_count, int(protocol["seed"]) + cell_count)

    circuit.reset_sequence()
    first_started = perf_counter()
    circuit.observe(int(tokens[0]))
    first_token_seconds = perf_counter() - first_started
    warmup_count = min(8, token_count)
    circuit.reset_sequence()
    for token in tokens[:warmup_count]:
        circuit.observe(int(token))
    seconds = _time_tokens(circuit, tokens, repeats)
    after_run = _process_memory()
    token_steps = circuit._token_steps
    memory_increment = {
        key: after_run[key] - before[key]
        for key in ("working_set_mib", "peak_working_set_mib", "private_memory_mib")
    }
    return {
        "cells": cell_count,
        "tokens": token_count,
        "token_steps": token_steps,
        "sparse_recurrent_edges": cell_count * int(protocol["recurrent_fanout"]),
        "seconds": seconds,
        "tokens_per_second": token_count / seconds,
        "cell_steps_per_second": cell_count * token_steps * token_count / seconds,
        "construction_seconds": construction_seconds,
        "first_token_seconds": first_token_seconds,
        "persistent_array_mib": _persistent_array_mib(circuit),
        "baseline_memory": before,
        "after_construction_memory": after_construction,
        "after_run_memory": after_run,
        "inference_memory_increment_mib": memory_increment,
        "normal_matrix_allocated": circuit._normal_matrix is not None,
    }


def _cold_build_timing() -> dict[str, object]:
    # Windows keeps a loaded DLL locked until process exit, so the cold-build
    # target intentionally lives under the ignored runtime directory instead
    # of a TemporaryDirectory that cannot be removed after the load timing.
    directory = ROOT / ".runtime/native-build-benchmark" / str(time_ns())
    started = perf_counter()
    build = build_native(directory)
    cold_seconds = perf_counter() - started
    load_started = perf_counter()
    ctypes.CDLL(str(build.library_path))
    load_seconds = perf_counter() - load_started
    cached_started = perf_counter()
    loaded_build_info()
    cached_seconds = perf_counter() - cached_started
    return {
        "cold_compile_seconds": cold_seconds,
        "library_load_seconds": load_seconds,
        "cached_build_lookup_seconds": cached_seconds,
        "source_digest": loaded_build_info().source_digest,
    }


def _scaling_exponent(results: list[dict[str, object]], metric: str) -> float:
    cells = np.asarray([float(result["cells"]) for result in results])
    values = np.asarray([float(result[metric]) for result in results])
    return float(np.polyfit(np.log(cells), np.log(values), 1)[0])


def run(artifact_path: Path) -> dict[str, object]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    build_timing = _cold_build_timing()
    workers: list[dict[str, object]] = []
    for cells in protocol["cell_counts"]:
        token_count = int(protocol["token_counts"][str(cells)])
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(cells),
            "--tokens",
            str(token_count),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        workers.append(json.loads(completed.stdout))

    reference = _reference_timing(protocol)
    native_reference_size = next(
        result for result in workers if int(result["cells"]) == int(reference["cells"])
    )
    artifact: dict[str, object] = {
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "native_build": build_timing,
        "python_reference": reference,
        "native_speedup_at_reference_size": (
            float(native_reference_size["tokens_per_second"])
            / float(reference["tokens_per_second"])
        ),
        "results": workers,
        "scaling": {
            "seconds_per_token_exponent": _scaling_exponent(workers, "seconds")
            - _scaling_exponent(workers, "tokens"),
            "persistent_memory_exponent": _scaling_exponent(workers, "persistent_array_mib"),
            "cell_step_throughput_exponent": _scaling_exponent(
                workers,
                "cell_steps_per_second",
            ),
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
    parser.add_argument("--tokens", type=int)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.worker is not None:
        if args.tokens is None:
            parser.error("--worker requires --tokens")
        print(json.dumps(_worker(args.worker, args.tokens, protocol), sort_keys=True))
        return
    print(json.dumps(run(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
