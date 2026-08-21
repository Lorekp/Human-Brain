"""Non-capability memory/throughput smoke for very large local learners."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


def _persistent_array_mebibytes(circuit: RecurrentCellCircuit) -> float:
    arrays: list[np.ndarray] = []
    arrays.extend(value for value in vars(circuit).values() if isinstance(value, np.ndarray))
    arrays.extend(
        value for value in vars(circuit._reservoir).values() if isinstance(value, np.ndarray)
    )
    arrays.extend(
        value for value in vars(circuit._reservoir.state).values() if isinstance(value, np.ndarray)
    )
    unique = {id(array): array for array in arrays}
    return sum(array.nbytes for array in unique.values()) / (1024 * 1024)


def _run_one(cells: int, training_examples: int) -> dict[str, object]:
    tokens = ((np.arange(training_examples + 1, dtype=np.int64) * 37 + 11) % 256).tolist()
    config = RecurrentCircuitConfig(
        reservoir_cells=cells,
        input_fanout=32,
        recurrent_fanout=6,
        token_interval_ms=2.0,
        input_pulse_ms=1.2,
        input_current_pa=8_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        recurrent_topology="uniform_no_self",
        readout_learning_rule="three_factor",
        local_learning_rate_pa=16.0,
    )
    gc.collect()
    tracemalloc.start()
    construction_started = perf_counter()
    circuit = RecurrentCellCircuit(token_cells=256, config=config)
    construction_seconds = perf_counter() - construction_started
    growth_started = perf_counter()
    circuit.grow_sequence(tokens)
    growth_seconds = perf_counter() - growth_started
    circuit.freeze_learning()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if circuit._normal_matrix is not None:
        raise RuntimeError("local scale smoke allocated a ridge normal matrix")
    if circuit._target_moments is not None or circuit._feature_batch is not None:
        raise RuntimeError("local scale smoke allocated ridge statistics")

    updates = circuit.local_update_count
    synapse_weight_events = updates * 2 * (cells + 1)
    avoided_normal_bytes = (cells + 1) ** 2 * np.dtype(np.float64).itemsize
    return {
        "reservoir_cells": cells,
        "output_cells": circuit.token_cells,
        "training_examples": training_examples,
        "competitive_updates": updates,
        "learned_readout_synapses": circuit.learned_synapse_count,
        "normal_matrix_allocated": False,
        "ridge_normal_matrix_mib_avoided": avoided_normal_bytes / (1024 * 1024),
        "wall_seconds": {
            "construction": construction_seconds,
            "growth": growth_seconds,
            "total": construction_seconds + growth_seconds,
        },
        "throughput": {
            "examples_per_second": training_examples / growth_seconds,
            "competitive_updates_per_second": updates / growth_seconds,
            "synapse_weight_events": synapse_weight_events,
            "synapse_weight_events_per_second": synapse_weight_events / growth_seconds,
        },
        "memory_mib": {
            "peak_traced": peak / (1024 * 1024),
            "persistent_arrays": _persistent_array_mebibytes(circuit),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="24576,98304")
    parser.add_argument("--training-examples", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if not sizes or any(size < 32 for size in sizes):
        raise ValueError("sizes must contain cell counts of at least 32")
    if args.training_examples < 1:
        raise ValueError("training examples must be positive")

    report = {
        "protocol": {
            "name": "local-readout-large-scale-smoke-v1",
            "purpose": (
                "scale/memory smoke only; fixed synthetic byte-token sequence; "
                "no train/calibration/validation/test data and no capability claim"
            ),
            "sizes": sizes,
            "training_examples": args.training_examples,
            "output_cells": 256,
            "learning_rule": "three_factor",
            "learning_rate_pa": 16.0,
            "peak_memory": "Python and NumPy allocations traced by tracemalloc",
        },
        "measurements": [_run_one(cells, args.training_examples) for cells in sizes],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
