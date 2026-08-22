"""Scale-optimization GPU benchmark for the CUDA cell-circuit path.

Mirrors the registered sparse_cuda_engineering workload (observe+race
inference, observe+strengthen training) across network sizes so baseline and
optimized branches are compared like-for-like. This harness writes its JSON
results next to itself (never into artifacts/) and performs no protocol
registration or --execute gating: it is an engineering instrument, not a
sealed evidence run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig

ROOT = Path(__file__).resolve().parents[1]

# Mirrors benchmarks/sparse_cuda_engineering_protocol_v1.json circuit fields,
# with reservoir_cells varied per size point.
PROTOCOL_MIRROR = {
    "token_cells": 256,
    "input_fanout": 32,
    "recurrent_fanout": 6,
    "token_interval_ms": 5.0,
    "input_pulse_ms": 1.5,
    "input_current_pa": 4000.0,
    "recurrent_current_pa": 325.0,
    "eligibility_tau_ms": 40.0,
    "recurrent_topology": "uniform_no_self",
    "readout_learning_rule": "sparse_three_factor",
    "readout_fan_in": 64,
    "readout_structural_interval": 1,
    "readout_structural_grace": 32,
    "readout_turnover_probability": 1.0,
    "readout_homeostatic_rate": 0.05,
    "readout_activity_decay": 0.99,
    "seed": 2317,
}

BOOTSTRAP_EVENTS = 64
WARMUP_TOKENS = 4
TIMED_TOKENS = 64


def _config(cells: int) -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
        reservoir_cells=cells,
        active_reservoir_cells=cells,
        input_fanout=PROTOCOL_MIRROR["input_fanout"],
        recurrent_fanout=PROTOCOL_MIRROR["recurrent_fanout"],
        token_interval_ms=PROTOCOL_MIRROR["token_interval_ms"],
        input_pulse_ms=PROTOCOL_MIRROR["input_pulse_ms"],
        input_current_pa=PROTOCOL_MIRROR["input_current_pa"],
        recurrent_current_pa=PROTOCOL_MIRROR["recurrent_current_pa"],
        eligibility_tau_ms=PROTOCOL_MIRROR["eligibility_tau_ms"],
        recurrent_topology=PROTOCOL_MIRROR["recurrent_topology"],
        readout_learning_rule=PROTOCOL_MIRROR["readout_learning_rule"],
        readout_fan_in=PROTOCOL_MIRROR["readout_fan_in"],
        readout_structural_interval=PROTOCOL_MIRROR["readout_structural_interval"],
        readout_structural_grace=PROTOCOL_MIRROR["readout_structural_grace"],
        readout_turnover_probability=PROTOCOL_MIRROR["readout_turnover_probability"],
        readout_homeostatic_rate=PROTOCOL_MIRROR["readout_homeostatic_rate"],
        readout_activity_decay=PROTOCOL_MIRROR["readout_activity_decay"],
        random_seed=PROTOCOL_MIRROR["seed"],
    )


def _gpu_used_mib() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return float(completed.stdout.strip().splitlines()[0])


def _phase_inference(circuit: RecurrentCellCircuit, stream: np.ndarray) -> dict[str, object]:
    circuit.reset_sequence()
    observe_ns = 0
    race_ns = 0
    kernel_ms = 0.0
    spikes = 0
    started_total = time.perf_counter_ns()
    for token in stream:
        started = time.perf_counter_ns()
        step = circuit.observe(int(token))
        observe_ns += time.perf_counter_ns() - started
        kernel_ms += float(circuit._native_kernel.last_kernel_milliseconds)
        spikes += step.reservoir_spikes
        started = time.perf_counter_ns()
        circuit.race()
        race_ns += time.perf_counter_ns() - started
    total_ns = time.perf_counter_ns() - started_total
    return {
        "tokens": int(stream.size),
        "wall_seconds": total_ns / 1e9,
        "end_to_end_tokens_per_second": stream.size * 1e9 / total_ns,
        "observe_seconds": observe_ns / 1e9,
        "race_seconds": race_ns / 1e9,
        "kernel_seconds": kernel_ms / 1e3,
        "kernel_tokens_per_second": stream.size * 1e3 / kernel_ms,
        "reservoir_spike_events": int(spikes),
        "mean_spikes_per_token": spikes / float(stream.size),
    }


def _phase_training(circuit: RecurrentCellCircuit, stream: np.ndarray) -> dict[str, object]:
    circuit.reset_sequence()
    observe_ns = 0
    strengthen_ns = 0
    kernel_ms = 0.0
    spikes = 0
    updates_before = circuit.local_update_count
    recruitments_before = circuit.structural_recruitment_count
    deletions_before = circuit.structural_deletion_count
    started_total = time.perf_counter_ns()
    for index, token in enumerate(stream):
        started = time.perf_counter_ns()
        step = circuit.observe(int(token))
        observe_ns += time.perf_counter_ns() - started
        kernel_ms += float(circuit._native_kernel.last_kernel_milliseconds)
        spikes += step.reservoir_spikes
        started = time.perf_counter_ns()
        circuit.strengthen((int(token) + 1) % circuit.token_cells)
        strengthen_ns += time.perf_counter_ns() - started
    total_ns = time.perf_counter_ns() - started_total
    del index
    return {
        "tokens": int(stream.size),
        "wall_seconds": total_ns / 1e9,
        "end_to_end_tokens_per_second": stream.size * 1e9 / total_ns,
        "observe_seconds": observe_ns / 1e9,
        "strengthen_seconds": strengthen_ns / 1e9,
        "kernel_seconds": kernel_ms / 1e3,
        "kernel_tokens_per_second": stream.size * 1e3 / kernel_ms,
        "reservoir_spike_events": int(spikes),
        "mean_spikes_per_token": spikes / float(stream.size),
        "applied_teaching_updates": circuit.local_update_count - updates_before,
        "structural_recruitments": (
            circuit.structural_recruitment_count - recruitments_before
        ),
        "structural_deletions": circuit.structural_deletion_count - deletions_before,
    }


def _run_size(cells: int, seed_offset: int) -> dict[str, object]:
    token_cells = PROTOCOL_MIRROR["token_cells"]
    seed = PROTOCOL_MIRROR["seed"] + seed_offset
    bootstrap_stream = np.random.default_rng(seed + 101).integers(0, token_cells, BOOTSTRAP_EVENTS)
    warmup_stream = np.random.default_rng(seed + 202).integers(0, token_cells, WARMUP_TOKENS)
    inference_stream = np.random.default_rng(seed + 303).integers(0, token_cells, TIMED_TOKENS)
    training_stream = np.random.default_rng(seed + 404).integers(0, token_cells, TIMED_TOKENS)

    gpu_before = _gpu_used_mib()
    started = time.perf_counter_ns()
    circuit = RecurrentCellCircuit(token_cells, _config(cells))
    construction_seconds = (time.perf_counter_ns() - started) / 1e9
    kernel = circuit._native_kernel
    try:
        if kernel.backend != "cuda":
            raise RuntimeError(f"expected CUDA backend, got {kernel.backend}")
        bootstrap_start = time.perf_counter_ns()
        bootstrap_spikes = 0
        for token in bootstrap_stream:
            bootstrap_spikes += circuit.observe(int(token)).reservoir_spikes
            circuit.strengthen(PROTOCOL_MIRROR["token_cells"] // 2)
        bootstrap_seconds = (time.perf_counter_ns() - bootstrap_start) / 1e9
        circuit.reset_sequence()
        for token in warmup_stream:
            circuit.observe(int(token))
            circuit.race()
        inference = _phase_inference(circuit, inference_stream)
        training = _phase_training(circuit, training_stream)
        responsive_cells = int(np.count_nonzero(circuit._eligibility))
        return {
            "cells": cells,
            "active_cells": circuit.active_reservoir_cells,
            "construction_seconds": construction_seconds,
            "bootstrap_seconds": bootstrap_seconds,
            "bootstrap_spike_events": int(bootstrap_spikes),
            "inference": inference,
            "training": training,
            "integration_steps_per_token": circuit._token_steps,
            "responsive_eligibility_cells": responsive_cells,
            "host_persistent_array_mib": sum(
                array.nbytes
                for owner in (circuit,)
                for array in vars(owner).values()
                if isinstance(array, np.ndarray)
            )
            / (1024.0 * 1024.0),
            "device_persistent_array_mib": kernel.device_bytes / (1024.0 * 1024.0),
            "device_to_host_bytes_per_token": kernel.device_to_host_bytes_per_token,
            "host_to_device_bytes_per_token": kernel.host_to_device_bytes_per_token,
            "gpu_memory_used_mib_before": gpu_before,
            "gpu_memory_used_mib_after": _gpu_used_mib(),
            "backend": kernel.backend,
            "device": kernel.device_name,
        }
    finally:
        kernel.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="128,1000,10000,100000")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    cells_list = [int(value) for value in args.cells.split(",")]

    import os

    previous = os.environ.get("HUMANBRAIN_DEVICE")
    os.environ["HUMANBRAIN_DEVICE"] = "cuda"
    try:
        results = []
        for offset, cells in enumerate(cells_list):
            print(f"running cells={cells} ...", flush=True)
            result = _run_size(cells, offset)
            results.append(result)
            print(
                json.dumps(
                    {
                        "cells": cells,
                        "construction_s": result["construction_seconds"],
                        "inference_tps": result["inference"]["end_to_end_tokens_per_second"],
                        "training_tps": result["training"]["end_to_end_tokens_per_second"],
                        "inference_kernel_tps": result["inference"]["kernel_tokens_per_second"],
                        "spikes_per_token": result["inference"]["mean_spikes_per_token"],
                    }
                ),
                flush=True,
            )
    finally:
        if previous is None:
            os.environ.pop("HUMANBRAIN_DEVICE", None)
        else:
            os.environ["HUMANBRAIN_DEVICE"] = previous

    output = args.output or (ROOT / "benchmarks" / f"scale_opt_gpu_{args.label}.json")
    output.write_text(
        json.dumps(
            {
                "label": args.label,
                "protocol_mirror": PROTOCOL_MIRROR,
                "timing": {
                    "bootstrap_events": BOOTSTRAP_EVENTS,
                    "warmup_tokens": WARMUP_TOKENS,
                    "timed_tokens": TIMED_TOKENS,
                },
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
