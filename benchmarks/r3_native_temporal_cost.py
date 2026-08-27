"""Measure indexed fitted-GLIF3 temporal cost at increasing live-segment counts.

This is a synthetic, pre-outcome engineering benchmark.  It deliberately
bypasses today's non-runnable calibration authority, but it does not bypass the
native circuit: construction, learning, recognition, plateau current, basket
inhibition, and emitted winners all run through ``NativeTemporalCircuit``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalCircuitStep,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)

ACTIVE_COLUMNS = 11
PAIRS_PER_SEGMENT = 10
BASE_CELLS_PER_COLUMN = 2
MAX_SEGMENTS_PER_CELL = 255
TARGET_TOKEN = 1
FIRST_CONTEXT_TOKEN = 1_000
DEFAULT_SEGMENT_TARGETS = (1_023, 8_008, 32_010)
DEFAULT_OUTPUT = Path(__file__).with_name("r3_native_temporal_cost.json")


@contextlib.contextmanager
def _synthetic_preoutcome_authority() -> Iterator[None]:
    """Bypass only the frozen runner's intentional pre-outcome hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x51" * 32, b"\xa7" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _exact_context_count(segment_target: int) -> int:
    if segment_target <= 0:
        raise ValueError("segment targets must be positive")
    context_count, remainder = divmod(segment_target, ACTIVE_COLUMNS)
    if remainder:
        raise ValueError("segment targets must be multiples of eleven")
    return context_count


def _definition(segment_target: int) -> tuple[NativeTemporalCircuitDefinition, int]:
    context_count = _exact_context_count(segment_target)
    context_columns = context_count * ACTIVE_COLUMNS
    column_count = context_columns + ACTIVE_COLUMNS
    base_pyramidal_cells = column_count * BASE_CELLS_PER_COLUMN

    owner_cells_per_target_column = math.ceil(
        context_count / MAX_SEGMENTS_PER_CELL
    )
    reserve_cells_per_target_column = max(
        0,
        2 * math.ceil((owner_cells_per_target_column - BASE_CELLS_PER_COLUMN) / 2),
    )
    max_pyramidal_cells = (
        base_pyramidal_cells
        + reserve_cells_per_target_column * ACTIVE_COLUMNS
    )
    config = NativeTemporalCircuitConfig(
        column_count=column_count,
        max_pyramidal_cells=max_pyramidal_cells,
        segment_capacity=segment_target,
        construction_seed=17,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        permanence_increment_ticks=50,
        negative_decrement_ticks=2,
        base_cells_per_column=BASE_CELLS_PER_COLUMN,
        novel_race_cap=BASE_CELLS_PER_COLUMN,
        prediction_quorum=ACTIVE_COLUMNS,
        max_segments_per_cell=MAX_SEGMENTS_PER_CELL,
        theta_match=18,
        theta_active=20,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )
    token_codes: dict[int, tuple[int, ...]] = {}
    for context_index in range(context_count):
        first_column = context_index * ACTIVE_COLUMNS
        token_codes[FIRST_CONTEXT_TOKEN + context_index] = tuple(
            range(first_column, first_column + ACTIVE_COLUMNS)
        )
    token_codes[TARGET_TOKEN] = tuple(range(context_columns, column_count))
    physical_cell_count = 2 * max_pyramidal_cells + 3 * column_count
    return (
        NativeTemporalCircuitDefinition(
            config,
            token_codes,
            (0.0,) * physical_cell_count,
        ),
        reserve_cells_per_target_column,
    )


def _transition(
    circuit: NativeTemporalCircuit,
    context_token: int,
    mode: TemporalStepMode,
) -> TemporalCircuitStep:
    steps = circuit.advance_block(
        [context_token, TARGET_TOKEN],
        mode=mode,
        reset_before=True,
    )
    if len(steps) != 2:
        raise RuntimeError("native transition did not emit exactly two token steps")
    return steps[1]


def _populate(
    circuit: NativeTemporalCircuit,
    context_count: int,
) -> int:
    started_ns = time.perf_counter_ns()
    for context_index in range(context_count):
        target = _transition(
            circuit,
            FIRST_CONTEXT_TOKEN + context_index,
            TemporalStepMode.LEARN,
        )
        if (
            target.allocated_segments != ACTIVE_COLUMNS
            or target.predicted_columns != 0
            or target.punished_segments != 0
        ):
            raise RuntimeError("synthetic population transition violated novelty")
        if context_index and context_index % 500 == 0:
            print(
                f"  populated {context_index}/{context_count} contexts",
                flush=True,
            )
    return time.perf_counter_ns() - started_ns


def _timing_summary(samples_ns: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(samples_ns)
    median_ns = statistics.median(ordered)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": len(ordered),
        "min_ns": ordered[0],
        "median_ns": median_ns,
        "mean_ns": statistics.fmean(ordered),
        "p95_ns": ordered[p95_index],
        "max_ns": ordered[-1],
        "transitions_per_second_at_median": 1_000_000_000.0 / median_ns,
        "token_steps_per_second_at_median": 2_000_000_000.0 / median_ns,
    }


def _measure_transitions(
    circuit: NativeTemporalCircuit,
    *,
    context_token: int,
    mode: TemporalStepMode,
    warmup: int,
    samples: int,
) -> tuple[dict[str, float | int], TemporalCircuitStep]:
    last: TemporalCircuitStep | None = None
    for _ in range(warmup):
        last = _transition(circuit, context_token, mode)

    durations: list[int] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            started_ns = time.perf_counter_ns()
            last = _transition(circuit, context_token, mode)
            durations.append(time.perf_counter_ns() - started_ns)
    finally:
        if gc_enabled:
            gc.enable()
    if last is None:
        raise RuntimeError("timing loop emitted no transition")
    return _timing_summary(durations), last


def _run_scale(
    segment_target: int,
    *,
    warmup: int,
    read_samples: int,
    learn_samples: int,
) -> dict[str, object]:
    print(f"building {segment_target:,}-segment definition", flush=True)
    definition_started_ns = time.perf_counter_ns()
    definition, reserve_cells_per_target_column = _definition(segment_target)
    definition_ns = time.perf_counter_ns() - definition_started_ns
    context_count = _exact_context_count(segment_target)
    probe_context_token = FIRST_CONTEXT_TOKEN

    native_started_ns = time.perf_counter_ns()
    with NativeTemporalCircuit.fresh(definition) as circuit:
        native_construction_ns = time.perf_counter_ns() - native_started_ns
        population_ns = _populate(circuit, context_count)
        populated_identity = circuit.checkpoint_identity()
        if populated_identity.segment_count != segment_target:
            raise RuntimeError("native segment count differs from requested scale")

        probe = _transition(
            circuit,
            probe_context_token,
            TemporalStepMode.READ_ONLY,
        )
        if probe.predicted_columns != ACTIVE_COLUMNS:
            raise RuntimeError("known context did not activate all target columns")

        read_timing, read_last = _measure_transitions(
            circuit,
            context_token=probe_context_token,
            mode=TemporalStepMode.READ_ONLY,
            warmup=warmup,
            samples=read_samples,
        )
        learn_timing, learn_last = _measure_transitions(
            circuit,
            context_token=probe_context_token,
            mode=TemporalStepMode.LEARN,
            warmup=warmup,
            samples=learn_samples,
        )
        final_identity = circuit.checkpoint_identity()

    if (
        read_last.predicted_columns != ACTIVE_COLUMNS
        or learn_last.predicted_columns != ACTIVE_COLUMNS
        or learn_last.allocated_segments != 0
        or learn_last.punished_segments != 0
        or final_identity.segment_count != segment_target
    ):
        raise RuntimeError("measured transition changed the intended candidate condition")

    target_column_occupancy = (
        BASE_CELLS_PER_COLUMN + reserve_cells_per_target_column
    )
    return {
        "requested_live_segments": segment_target,
        "actual_live_segments": final_identity.segment_count,
        "synthetic_contexts": context_count,
        "definition_build_ns": definition_ns,
        "native_construction_ns": native_construction_ns,
        "population_ns": population_ns,
        "population_segments_per_second": (
            segment_target * 1_000_000_000.0 / population_ns
        ),
        "configured": {
            "column_count": definition.config.column_count,
            "max_pyramidal_cells": definition.config.max_pyramidal_cells,
            "physical_cell_count": len(definition.threshold_offsets_mv),
            "committed_pyramidal_cells": final_identity.next_reserve_cell,
            "target_column_occupancy": target_column_occupancy,
            "cohort_count": final_identity.cohort_count,
        },
        "measured_condition": {
            "recognition_counts_basis": (
                "exact from the fixture's disjoint ten-pair segments and the native "
                "inverted-index contract; the circuit-step ABI directly exposes only "
                "predictions and physical spikes"
            ),
            "transition_token_steps": 2,
            "recognition_candidate_segments": ACTIVE_COLUMNS,
            "recognition_active_segments": ACTIVE_COLUMNS,
            "recognition_postings_visited": (
                ACTIVE_COLUMNS * PAIRS_PER_SEGMENT
            ),
            "predicted_columns": read_last.predicted_columns,
            "winner_spikes": len(read_last.winner_spikes),
            "basket_spikes": len(read_last.basket_spikes),
            "all_winner_to_basket_ticks": all(
                basket.tick == winner.tick + 1
                for winner, basket in zip(
                    read_last.winner_spikes,
                    read_last.basket_spikes,
                    strict=True,
                )
            ),
            "all_basket_to_gaba_ticks": all(
                onset == basket.tick + 1
                for basket, onset in zip(
                    read_last.basket_spikes,
                    read_last.inhibition_onset_ticks,
                    strict=True,
                )
            ),
        },
        "read_only_recognized_transition": read_timing,
        "learn_confirmed_transition": learn_timing,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--segment-targets",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEGMENT_TARGETS),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--read-samples", type=int, default=200)
    parser.add_argument("--learn-samples", type=int, default=100)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.read_samples <= 0 or args.learn_samples <= 0:
        parser.error("warmup must be nonnegative and sample counts must be positive")
    for target in args.segment_targets:
        _exact_context_count(target)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with _synthetic_preoutcome_authority():
        rows = [
            _run_scale(
                target,
                warmup=args.warmup,
                read_samples=args.read_samples,
                learn_samples=args.learn_samples,
            )
            for target in args.segment_targets
        ]
        native_identity = loaded_library_identity()

    payload = {
        "schema_version": 1,
        "benchmark": "r3-native-temporal-indexed-physical-cost",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "synthetic TRAIN-safe engineering timing only; no held-out, C1, "
            "capability, or cohort evidence"
        ),
        "authority_mode": (
            "synthetic pre-outcome authority bypass; official calibration remains disabled"
        ),
        "runtime_path": (
            "NativeTemporalCircuit.advance_block -> indexed TemporalStore recognition -> "
            "fitted GLIF3 plateau/basket first-spike race"
        ),
        "method": {
            "segment_targets": list(args.segment_targets),
            "warmup_transitions_per_mode": args.warmup,
            "read_samples_per_scale": args.read_samples,
            "learn_samples_per_scale": args.learn_samples,
            "timer": "time.perf_counter_ns",
            "gc_during_samples": "disabled",
            "threshold_offsets": "synthetic all-zero finite f64",
            "contexts": (
                "disjoint eleven-column codes; one learned segment per target column "
                "and context"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_size": native_identity.binary_size,
            "native_binary_sha256": native_identity.binary_sha256,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
