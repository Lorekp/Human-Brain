"""Reproducible speed and memory benchmarks for human-cell simulation."""

from __future__ import annotations

import argparse
import gc
import json
import tracemalloc
from collections.abc import Callable, Sequence
from statistics import median
from time import perf_counter

import numpy as np
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.cortex import DendriticCortex
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


def _median_seconds(function: Callable[[], None], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append(perf_counter() - started)
    return median(samples)


def _peak_mebibytes(function: Callable[[], None]) -> float:
    gc.collect()
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


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


def _circuit_config(cell_count: int) -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
        reservoir_cells=cell_count,
        input_fanout=max(4, round(cell_count / 12)),
        recurrent_fanout=6,
        token_interval_ms=5.0,
        input_pulse_ms=1.5,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        recurrent_topology="uniform_no_self",
    )


def _cell_steps(
    cell_count: int,
    step_count: int,
    repeats: int,
) -> dict[str, float | int]:
    current = np.full(cell_count, 850.0, dtype=np.float64)

    def run() -> None:
        population = AdaptiveGlifPopulation(cell_count)
        for _ in range(step_count):
            population.step(current)

    elapsed = _median_seconds(run, repeats)
    cell_steps_per_s = cell_count * step_count / elapsed
    return {
        "cells": cell_count,
        "steps": step_count,
        "elapsed_s": elapsed,
        "cell_steps_per_s": cell_steps_per_s,
        "million_cell_steps_per_s": cell_steps_per_s / 1e6,
        "peak_traced_mib": _peak_mebibytes(run),
    }


def _recurrent_observations(
    cell_count: int,
    token_count: int,
    repeats: int,
) -> dict[str, float | int]:
    config = _circuit_config(cell_count)
    tokens = np.random.default_rng(81).integers(0, 256, token_count)

    def run() -> None:
        circuit = RecurrentCellCircuit(token_cells=256, config=config)
        for token in tokens:
            circuit.observe(int(token))

    probe = RecurrentCellCircuit(token_cells=256, config=config)
    persistent_mib = _persistent_array_mebibytes(probe)
    elapsed = _median_seconds(run, repeats)
    simulated_steps = token_count * round(config.token_interval_ms / 0.05)
    cell_steps_per_s = cell_count * simulated_steps / elapsed
    return {
        "cells": cell_count,
        "tokens": token_count,
        "simulated_steps": simulated_steps,
        "elapsed_s": elapsed,
        "tokens_per_s": token_count / elapsed,
        "cell_steps_per_s": cell_steps_per_s,
        "million_cell_steps_per_s": cell_steps_per_s / 1e6,
        "persistent_array_mib": persistent_mib,
        "peak_traced_mib": _peak_mebibytes(run),
    }


def _recurrent_training(
    cell_count: int,
    token_count: int,
    repeats: int,
) -> dict[str, float | int]:
    config = _circuit_config(cell_count)
    tokens = np.random.default_rng(105).integers(0, 256, token_count + 1)

    def run() -> None:
        circuit = RecurrentCellCircuit(token_cells=256, config=config)
        circuit.grow_sequence(tokens)

    elapsed = _median_seconds(run, repeats)
    return {
        "cells": cell_count,
        "training_examples": token_count,
        "elapsed_s": elapsed,
        "training_examples_per_s": token_count / elapsed,
        "peak_traced_mib": _peak_mebibytes(run),
    }


def _output_races(
    cell_count: int,
    race_count: int,
    repeats: int,
) -> dict[str, float | int]:
    circuit = RecurrentCellCircuit(token_cells=256, config=_circuit_config(cell_count))
    circuit.observe(0)
    rng = np.random.default_rng(94)
    circuit._readout_synapses_pa[:] = rng.normal(0.0, 10.0, circuit._readout_synapses_pa.shape)
    circuit._pacemaker_synapses_pa[:] = rng.uniform(200.0, 700.0, 256)

    def run() -> None:
        for _ in range(race_count):
            circuit.race()

    elapsed = _median_seconds(run, repeats)
    return {
        "output_cells": 256,
        "races": race_count,
        "elapsed_s": elapsed,
        "races_per_s": race_count / elapsed,
        "peak_traced_mib": _peak_mebibytes(run),
    }


def _dendritic_races(race_count: int, repeats: int) -> dict[str, float | int]:
    cortex = DendriticCortex(cell_count=256, max_order=8)
    training = np.random.default_rng(63).integers(0, 256, 4_096)
    cortex.grow_sequence(training)
    context = training[-8:]

    def run() -> None:
        for _ in range(race_count):
            cortex.race(context)

    elapsed = _median_seconds(run, repeats)
    return {
        "output_cells": 256,
        "races": race_count,
        "elapsed_s": elapsed,
        "races_per_s": race_count / elapsed,
        "peak_traced_mib": _peak_mebibytes(run),
    }


def _metric_exponent(
    sizes: Sequence[int],
    measurements: Sequence[dict[str, float | int]],
    metric: str,
) -> float:
    values = np.asarray([item[metric] for item in measurements], dtype=np.float64)
    return float(np.polyfit(np.log(sizes), np.log(values), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="96,384,768,1536")
    parser.add_argument("--cell-steps", type=int, default=2_000)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--training-tokens", type=int, default=128)
    parser.add_argument("--races", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if len(sizes) < 2 or any(size < 8 for size in sizes):
        raise ValueError("sizes must contain at least two cell counts of eight or more")

    cell_step = [_cell_steps(size, args.cell_steps, args.repeats) for size in sizes]
    recurrent_observe = [_recurrent_observations(size, args.tokens, args.repeats) for size in sizes]
    recurrent_training = [
        _recurrent_training(size, args.training_tokens, args.repeats) for size in sizes
    ]
    result = {
        "protocol": {
            "sizes": sizes,
            "cell_steps": args.cell_steps,
            "tokens": args.tokens,
            "training_tokens": args.training_tokens,
            "races": args.races,
            "repeats": args.repeats,
            "timing_statistic": "median",
            "peak_memory": "Python and NumPy allocations traced by tracemalloc",
        },
        "cell_step": cell_step,
        "recurrent_observe": recurrent_observe,
        "recurrent_training": recurrent_training,
        "empirical_scaling_exponents": {
            "cell_step_elapsed_vs_cells": _metric_exponent(
                sizes, cell_step, "elapsed_s"
            ),
            "cell_step_peak_memory_vs_cells": _metric_exponent(
                sizes, cell_step, "peak_traced_mib"
            ),
            "recurrent_observe_elapsed_vs_cells": _metric_exponent(
                sizes, recurrent_observe, "elapsed_s"
            ),
            "recurrent_observe_persistent_memory_vs_cells": _metric_exponent(
                sizes, recurrent_observe, "persistent_array_mib"
            ),
            "recurrent_training_elapsed_vs_cells": _metric_exponent(
                sizes, recurrent_training, "elapsed_s"
            ),
            "recurrent_training_peak_memory_vs_cells": _metric_exponent(
                sizes, recurrent_training, "peak_traced_mib"
            ),
        },
        "output_race": _output_races(max(sizes), args.races, args.repeats),
        "dendritic_race": _dendritic_races(args.races, args.repeats),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
