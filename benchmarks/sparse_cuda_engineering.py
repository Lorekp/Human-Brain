from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "sparse_cuda_engineering_protocol_v1.json"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "sparse_cuda_engineering_v1.json"


def _config(protocol: dict[str, Any]) -> RecurrentCircuitConfig:
    circuit = protocol["circuit"]
    return RecurrentCircuitConfig(
        reservoir_cells=int(circuit["reservoir_cells"]),
        active_reservoir_cells=int(circuit["active_reservoir_cells"]),
        input_fanout=int(circuit["input_fanout"]),
        recurrent_fanout=int(circuit["recurrent_fanout"]),
        token_interval_ms=float(circuit["token_interval_ms"]),
        input_pulse_ms=float(circuit["input_pulse_ms"]),
        input_current_pa=float(circuit["input_current_pa"]),
        recurrent_current_pa=float(circuit["recurrent_current_pa"]),
        eligibility_tau_ms=float(circuit["eligibility_tau_ms"]),
        recurrent_topology=str(circuit["recurrent_topology"]),
        readout_learning_rule=str(circuit["readout_learning_rule"]),
        readout_fan_in=int(circuit["readout_fan_in"]),
        readout_structural_interval=int(circuit["readout_structural_interval"]),
        readout_structural_grace=int(circuit["readout_structural_grace"]),
        readout_turnover_probability=float(circuit["readout_turnover_probability"]),
        readout_homeostatic_rate=float(circuit["readout_homeostatic_rate"]),
        readout_activity_decay=float(circuit["readout_activity_decay"]),
        random_seed=int(circuit["seed"]),
    )


def _all_host_arrays(circuit: RecurrentCellCircuit) -> list[np.ndarray]:
    owners = [circuit, circuit._reservoir, circuit._reservoir.state]
    if circuit._sparse_readout is not None:
        owners.append(circuit._sparse_readout)
    return [
        value
        for owner in owners
        for value in vars(owner).values()
        if isinstance(value, np.ndarray)
    ]


def _host_persistent_bytes(circuit: RecurrentCellCircuit) -> int:
    unique = {id(array): array for array in _all_host_arrays(circuit)}
    return sum(array.nbytes for array in unique.values())


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
    values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
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


def _git_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": revision, "worktree_clean": not bool(status.strip())}


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _bootstrap(
    circuit: RecurrentCellCircuit,
    protocol: dict[str, Any],
    stream: np.ndarray,
) -> tuple[float, int]:
    timing = protocol["timing"]
    target = int(timing["bootstrap_target"])
    started = time.perf_counter_ns()
    spikes = 0
    for token in stream:
        spikes += circuit.observe(int(token)).reservoir_spikes
        circuit.strengthen(target)
    return (time.perf_counter_ns() - started) / 1_000_000_000.0, spikes


def _warm(circuit: RecurrentCellCircuit, stream: np.ndarray) -> None:
    circuit.reset_sequence()
    for token in stream:
        circuit.observe(int(token))
        circuit.race()


def _time_inference(
    circuit: RecurrentCellCircuit,
    stream: np.ndarray,
) -> dict[str, object]:
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
        "wall_seconds": total_ns / 1_000_000_000.0,
        "tokens_per_second": stream.size * 1_000_000_000.0 / total_ns,
        "observe_wall_seconds": observe_ns / 1_000_000_000.0,
        "race_wall_seconds": race_ns / 1_000_000_000.0,
        "kernel_seconds": kernel_ms / 1_000.0,
        "kernel_tokens_per_second": stream.size * 1_000.0 / kernel_ms,
        "reservoir_spike_events": spikes,
    }


def _time_training(
    circuit: RecurrentCellCircuit,
    stream: np.ndarray,
) -> dict[str, object]:
    circuit.reset_sequence()
    observe_ns = 0
    strengthen_ns = 0
    kernel_ms = 0.0
    spikes = 0
    updates_before = circuit.local_update_count
    recruitments_before = circuit.structural_recruitment_count
    deletions_before = circuit.structural_deletion_count
    started_total = time.perf_counter_ns()
    for token in stream:
        started = time.perf_counter_ns()
        step = circuit.observe(int(token))
        observe_ns += time.perf_counter_ns() - started
        kernel_ms += float(circuit._native_kernel.last_kernel_milliseconds)
        spikes += step.reservoir_spikes
        started = time.perf_counter_ns()
        circuit.strengthen((int(token) + 1) % circuit.token_cells)
        strengthen_ns += time.perf_counter_ns() - started
    total_ns = time.perf_counter_ns() - started_total
    return {
        "tokens": int(stream.size),
        "wall_seconds": total_ns / 1_000_000_000.0,
        "tokens_per_second": stream.size * 1_000_000_000.0 / total_ns,
        "observe_wall_seconds": observe_ns / 1_000_000_000.0,
        "strengthen_wall_seconds": strengthen_ns / 1_000_000_000.0,
        "kernel_seconds": kernel_ms / 1_000.0,
        "kernel_tokens_per_second": stream.size * 1_000.0 / kernel_ms,
        "reservoir_spike_events": spikes,
        "applied_teaching_updates": circuit.local_update_count - updates_before,
        "structural_recruitments": (
            circuit.structural_recruitment_count - recruitments_before
        ),
        "structural_deletions": circuit.structural_deletion_count - deletions_before,
    }


def _time_sparse_current(
    circuit: RecurrentCellCircuit,
    calls: int,
) -> dict[str, object]:
    started = time.perf_counter_ns()
    digest = hashlib.sha256()
    for _ in range(calls):
        digest.update(circuit.output_current_pa().tobytes())
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "calls": calls,
        "wall_seconds": elapsed_ns / 1_000_000_000.0,
        "microseconds_per_call": elapsed_ns / calls / 1_000.0,
        "output_sha256": digest.hexdigest(),
    }


def _run_repeat(protocol: dict[str, Any], repeat: int) -> dict[str, object]:
    circuit_spec = protocol["circuit"]
    timing = protocol["timing"]
    seed = int(circuit_spec["seed"])
    token_cells = int(circuit_spec["token_cells"])
    bootstrap_stream = np.random.default_rng(seed + 101).integers(
        0, token_cells, int(timing["bootstrap_training_events"])
    )
    warmup_stream = np.random.default_rng(seed + 202).integers(
        0, token_cells, int(timing["warmup_tokens"])
    )
    inference_stream = np.random.default_rng(seed + 303).integers(
        0, token_cells, int(timing["inference_tokens"])
    )
    training_stream = np.random.default_rng(seed + 404).integers(
        0, token_cells, int(timing["training_tokens"])
    )

    started = time.perf_counter_ns()
    circuit = RecurrentCellCircuit(token_cells, _config(protocol))
    construction_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
    kernel = circuit._native_kernel
    try:
        if kernel.backend != "cuda":
            raise RuntimeError(f"expected CUDA backend, got {kernel.backend}")
        if circuit.active_reservoir_cells != int(circuit_spec["reservoir_cells"]):
            raise RuntimeError("200k timing cannot use dormant reservoir padding")
        bootstrap_seconds, bootstrap_spikes = _bootstrap(
            circuit, protocol, bootstrap_stream
        )
        _warm(circuit, warmup_stream)
        inference = _time_inference(circuit, inference_stream)
        sparse_current = _time_sparse_current(
            circuit, int(timing["readout_microprofile_calls"])
        )
        training = _time_training(circuit, training_stream)
        readout = circuit._sparse_readout
        assert readout is not None
        cells = circuit.config.reservoir_cells
        # The measured phase starts from reset and eligibility has a positive
        # finite decay, so nonzero traces are the exact spiking-cell union for
        # this registered short run (not a general lifetime spike counter).
        spiking_cells = int(np.count_nonzero(circuit._eligibility))
        dense_bytes = circuit.token_cells * cells * np.dtype(np.float64).itemsize
        host_bytes = _host_persistent_bytes(circuit)
        return {
            "repeat": repeat,
            "construction_seconds": construction_seconds,
            "bootstrap_seconds": bootstrap_seconds,
            "bootstrap_spike_events": bootstrap_spikes,
            "inference": inference,
            "training": training,
            "sparse_current_microprofile": sparse_current,
            "memory": {
                "host_persistent_array_bytes": host_bytes,
                "host_persistent_array_mib": host_bytes / (1024.0 * 1024.0),
                "device_persistent_array_bytes": kernel.device_bytes,
                "device_persistent_array_mib": kernel.device_bytes
                / (1024.0 * 1024.0),
                "sparse_readout_persistent_bytes": readout.storage_bytes,
                "dense_control_readout_bytes": dense_bytes,
                "dense_to_sparse_readout_byte_ratio": dense_bytes
                / readout.storage_bytes,
            },
            "transfer": {
                "host_to_device_bytes_per_token": (
                    kernel.host_to_device_bytes_per_token
                ),
                "device_to_host_bytes_per_token": (
                    kernel.device_to_host_bytes_per_token
                ),
            },
            "cell_audit": {
                "configured_reservoir_cells": cells,
                "active_reservoir_cells": circuit.active_reservoir_cells,
                "updated_reservoir_cells_per_integration_step": cells,
                "integration_steps_per_token": circuit._token_steps,
                "responsive_eligibility_cells": spiking_cells,
                "unique_spiking_reservoir_cells": spiking_cells,
                "reservoir_activity_density": circuit.reservoir_activity_density,
                "learned_synapses": circuit.learned_synapse_count,
                "occupied_output_synapses_per_cell": (
                    circuit.occupied_output_synapses_per_cell
                ),
            },
            "backend": kernel.backend,
            "device": kernel.device_name,
        }
    finally:
        kernel.close()


def _median(repeats: list[dict[str, object]], *path: str) -> float:
    values: list[float] = []
    for repeat in repeats:
        value: object = repeat
        for key in path:
            assert isinstance(value, dict)
            value = value[key]
        values.append(float(value))
    return float(np.median(values))


def _summary(protocol: dict[str, Any], repeats: list[dict[str, object]]) -> dict[str, object]:
    inference_tps = _median(repeats, "inference", "tokens_per_second")
    training_tps = _median(repeats, "training", "tokens_per_second")
    ratio = _median(repeats, "memory", "dense_to_sparse_readout_byte_ratio")
    gates = protocol["gates"]
    return {
        "inference_end_to_end_tokens_per_second": inference_tps,
        "training_end_to_end_tokens_per_second": training_tps,
        "inference_kernel_tokens_per_second": _median(
            repeats, "inference", "kernel_tokens_per_second"
        ),
        "training_kernel_tokens_per_second": _median(
            repeats, "training", "kernel_tokens_per_second"
        ),
        "construction_seconds": _median(repeats, "construction_seconds"),
        "bootstrap_seconds": _median(repeats, "bootstrap_seconds"),
        "sparse_current_microseconds_per_call": _median(
            repeats, "sparse_current_microprofile", "microseconds_per_call"
        ),
        "host_persistent_array_mib": _median(
            repeats, "memory", "host_persistent_array_mib"
        ),
        "device_persistent_array_mib": _median(
            repeats, "memory", "device_persistent_array_mib"
        ),
        "device_to_host_bytes_per_token": _median(
            repeats, "transfer", "device_to_host_bytes_per_token"
        ),
        "host_to_device_bytes_per_token": _median(
            repeats, "transfer", "host_to_device_bytes_per_token"
        ),
        "dense_to_sparse_readout_byte_ratio": ratio,
        "gates": {
            "inference_over_100_tps": inference_tps
            >= float(gates["minimum_inference_tokens_per_second"]),
            "training_over_100_tps": training_tps
            >= float(gates["minimum_training_tokens_per_second"]),
            "readout_memory_over_100x": ratio
            >= float(gates["minimum_dense_to_sparse_readout_byte_ratio"]),
            "all_pass": (
                inference_tps >= float(gates["minimum_inference_tokens_per_second"])
                and training_tps >= float(gates["minimum_training_tokens_per_second"])
                and ratio >= float(gates["minimum_dense_to_sparse_readout_byte_ratio"])
            ),
        },
    }


def run(
    artifact_path: Path,
    *,
    execute: bool,
    expected_protocol_sha256: str | None = None,
    label: str = "baseline",
) -> dict[str, object]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol: dict[str, Any] = json.loads(protocol_bytes)
    if not execute:
        return {
            "status": "plan-only",
            "protocol_sha256": protocol_sha256,
            "cells": int(protocol["circuit"]["reservoir_cells"]),
            "repeats": int(protocol["timing"]["independent_repeats"]),
        }
    if expected_protocol_sha256 != protocol_sha256:
        raise ValueError(
            "protocol SHA-256 mismatch; inspect plan mode and pass the registered hash"
        )
    if artifact_path.exists():
        raise FileExistsError(f"timing artifact already exists: {artifact_path}")

    previous_device = os.environ.get("HUMANBRAIN_DEVICE")
    os.environ["HUMANBRAIN_DEVICE"] = "cuda"
    report: dict[str, object] = {
        "status": "running",
        "label": label,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "implementation": _git_state(),
        "gpu_before": _gpu_snapshot(),
        "gpu_after": None,
        "repeats": [],
        "summary": None,
    }
    _write_report(artifact_path, report)
    try:
        repeats = report["repeats"]
        assert isinstance(repeats, list)
        for repeat in range(int(protocol["timing"]["independent_repeats"])):
            repeats.append(_run_repeat(protocol, repeat))
            _write_report(artifact_path, report)
        report["summary"] = _summary(protocol, repeats)
        report["gpu_after"] = _gpu_snapshot()
        report["status"] = "complete"
    finally:
        if previous_device is None:
            os.environ.pop("HUMANBRAIN_DEVICE", None)
        else:
            os.environ["HUMANBRAIN_DEVICE"] = previous_device
        _write_report(artifact_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.artifact,
                execute=args.execute,
                expected_protocol_sha256=args.protocol_sha256,
                label=args.label,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
