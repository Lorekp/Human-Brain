"""Bit-exactness oracle for the CUDA recurrent path (engineering instrument).

Runs one deterministic token schedule (mirrors scale_opt_gpu_bench phases:
bootstrap observe+strengthen, inference observe+race, training
observe+strengthen) at a given cell count, then SHA-256s every mutable device
state array byte-for-byte together with the per-token spike counts and
structural counters. Run it before and after a kernel change; the digests must
match exactly. Writes JSON next to itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

os.environ["HUMANBRAIN_DEVICE"] = "cuda"

from benchmarks.scale_opt_gpu_bench import PROTOCOL_MIRROR, _config  # noqa: E402
from humanbrain.recurrent import RecurrentCellCircuit  # noqa: E402

BOOTSTRAP_TOKENS = 48
INFERENCE_TOKENS = 32
TRAINING_TOKENS = 24


def state_arrays(circuit: RecurrentCellCircuit) -> dict[str, np.ndarray]:
    circuit._native_kernel.synchronize_to_host()
    state = circuit._reservoir.state
    return {
        "synaptic_current": circuit._synaptic_current_pa,
        "eligibility": circuit._eligibility,
        "voltage": state.voltage_relative_mv,
        "after": state.after_spike_currents_pa,
        "threshold_spike": state.threshold_spike_mv,
        "threshold_voltage": state.threshold_voltage_mv,
        "refractory": state.refractory_steps,
    }


def canonical_digest(circuit: RecurrentCellCircuit, spikes: list[int]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state_arrays(circuit)):
        hasher.update(name.encode())
        hasher.update(np.ascontiguousarray(state_arrays(circuit)[name]).tobytes())
    hasher.update(np.asarray(spikes, dtype=np.int64).tobytes())
    hasher.update(str(circuit.local_update_count).encode())
    hasher.update(str(circuit.structural_recruitment_count).encode())
    hasher.update(str(circuit.structural_deletion_count).encode())
    return hasher.hexdigest()


def run(cells: int) -> dict[str, object]:
    token_cells = PROTOCOL_MIRROR["token_cells"]
    seed = PROTOCOL_MIRROR["seed"]
    rng = np.random.default_rng(seed + 909)
    stream = rng.integers(0, token_cells, BOOTSTRAP_TOKENS + INFERENCE_TOKENS + TRAINING_TOKENS)

    circuit = RecurrentCellCircuit(token_cells, _config(cells))
    kernel = circuit._native_kernel
    try:
        assert kernel.backend == "cuda", kernel.backend
        spikes: list[int] = []
        index = 0
        for _ in range(BOOTSTRAP_TOKENS):
            spikes.append(circuit.observe(int(stream[index])).reservoir_spikes)
            circuit.strengthen(PROTOCOL_MIRROR["token_cells"] // 2)
            index += 1
        bootstrap_digest = canonical_digest(circuit, spikes)

        circuit.reset_sequence()
        for _ in range(INFERENCE_TOKENS):
            spikes.append(circuit.observe(int(stream[index])).reservoir_spikes)
            circuit.race()
            index += 1
        inference_digest = canonical_digest(circuit, spikes)

        circuit.reset_sequence()
        for _ in range(TRAINING_TOKENS):
            spikes.append(circuit.observe(int(stream[index])).reservoir_spikes)
            circuit.strengthen((int(stream[index]) + 1) % token_cells)
            index += 1
        training_digest = canonical_digest(circuit, spikes)

        return {
            "cells": cells,
            "bootstrap_digest": bootstrap_digest,
            "inference_digest": inference_digest,
            "training_digest": training_digest,
            "total_spike_events": int(sum(spikes)),
            "local_updates": circuit.local_update_count,
            "structural_recruitments": circuit.structural_recruitment_count,
            "structural_deletions": circuit.structural_deletion_count,
        }
    finally:
        kernel.close()


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    cells_list = [int(value) for value in (sys.argv[2:] or ["128", "1000000"])]
    results = []
    for cells in cells_list:
        print(f"parity cells={cells} ...", flush=True)
        outcome = run(cells)
        results.append(outcome)
        print(json.dumps(outcome), flush=True)
    output = Path(__file__).with_name(f"scale_opt_gpu_parity_{label}.json")
    output.write_text(
        json.dumps({"label": label, "results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
