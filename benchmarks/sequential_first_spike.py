"""Machine-local scalar/batch timing for sequential first-spike races."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from collections.abc import Callable
from statistics import median
from time import perf_counter

import numpy as np
from humanbrain.cortex import CellRace, DendriticCortex
from humanbrain.sequential_first_spike import DendriticFirstSpikeBatch


def _inputs(row_count: int, cell_count: int) -> tuple[np.ndarray, np.ndarray]:
    stop_cell = cell_count - 1
    rows = np.arange(row_count, dtype=np.int64)
    cell_ids = np.column_stack((rows % stop_cell, np.full(row_count, stop_cell)))
    currents = np.full((row_count, 2), 1_035.0, dtype=np.float64)
    currents[:, 0] = 1_800.0
    return cell_ids, currents


def _median_call_seconds(
    function: Callable[[], tuple[CellRace, ...]],
    *,
    repeats: int,
    iterations: int,
) -> float:
    function()
    samples: list[float] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = perf_counter()
            for _ in range(iterations):
                function()
            samples.append((perf_counter() - started) / iterations)
    finally:
        if gc_enabled:
            gc.enable()
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,4,16,64,256,1024,4096")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-races", type=int, default=4096)
    args = parser.parse_args()
    row_counts = tuple(int(value) for value in args.rows.split(","))
    if not row_counts or any(value < 1 for value in row_counts):
        raise ValueError("rows must contain positive integers")
    if args.repeats < 1 or args.target_races < 1:
        raise ValueError("repeats and target-races must be positive")

    cell_count = 8192
    cortex = DendriticCortex(cell_count=cell_count, seed=29)
    batch = DendriticFirstSpikeBatch(cortex)
    measurements: list[dict[str, float | int | bool]] = []
    for row_count in row_counts:
        cell_ids, currents = _inputs(row_count, cell_count)

        def scalar(
            ids: np.ndarray = cell_ids,
            amplitudes: np.ndarray = currents,
            rows: int = row_count,
        ) -> tuple[CellRace, ...]:
            return tuple(
                cortex.race_currents_first_spike(ids[row], amplitudes[row]) for row in range(rows)
            )

        def vectorized(
            ids: np.ndarray = cell_ids,
            amplitudes: np.ndarray = currents,
        ) -> tuple[CellRace, ...]:
            return batch.race_many(ids, amplitudes)

        scalar_result = scalar()
        batch_result = vectorized()
        if batch_result != scalar_result:
            raise RuntimeError(f"batch result diverged at {row_count} rows")
        iterations = max(1, args.target_races // row_count)
        scalar_seconds = _median_call_seconds(
            scalar,
            repeats=args.repeats,
            iterations=iterations,
        )
        batch_seconds = _median_call_seconds(
            vectorized,
            repeats=args.repeats,
            iterations=iterations,
        )
        measurements.append(
            {
                "rows": row_count,
                "iterations_per_sample": iterations,
                "scalar_seconds": scalar_seconds,
                "batch_seconds": batch_seconds,
                "scalar_races_per_second": row_count / scalar_seconds,
                "batch_races_per_second": row_count / batch_seconds,
                "speedup": scalar_seconds / batch_seconds,
                "exact_cell_race_parity": True,
            }
        )

    crossover = next(
        (int(item["rows"]) for item in measurements if item["speedup"] > 1.0),
        None,
    )
    print(
        json.dumps(
            {
                "schema": "sequential-first-spike-performance-v1",
                "claim_boundary": (
                    "machine-local fitted-GLIF3 kernel timing; no capability evidence"
                ),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "protocol": {
                    "candidates_per_row": 2,
                    "candidate_current_pa": 1_800.0,
                    "cell_count": cell_count,
                    "repeats": args.repeats,
                    "seed": 29,
                    "target_races_per_sample": args.target_races,
                    "timing_statistic": "median",
                    "tonic_stop_current_pa": 1_035.0,
                },
                "crossover_rows": crossover,
                "measurements": measurements,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
