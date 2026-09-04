"""TRAIN-only native cost smoke over a canonical first-book prefix.

This is pre-outcome engineering evidence.  It uses the production R3-F0
codebook, fitted seed-17 thresholds, and physical native learning path, but a
bounded segment capacity and a synthetic authority bypass keep it outside C1.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from collections.abc import Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot.calibration_v3_definitions import build_r3_f0_definition
from experiments.r3_pilot.r3_f0_train_input import load_verified_train_input
from humanbrain.native import _enable_temporal_profile_build, loaded_library_identity
import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    NativeTemporalRuntimeStatus,
    TemporalCircuitStep,
    TemporalStepMode,
)


DEFAULT_PREFIX_TOKENS = 4_096
ACTIVE_COLUMNS = 11
ARM = (16, 18)
CONSTRUCTION_SEED = 17
TOTAL_C1_STEPS = 33_135_753
CAP_SECONDS = 15 * 60 * 60
HEADROOM_SECONDS = int(CAP_SECONDS * 0.9)
OUTPUT_PATH = Path(__file__).with_name("r3_native_train_prefix_cost.json")
COUNTER_FIELDS = (
    "candidate_segments",
    "matching_segments",
    "active_segments",
    "postings_visited",
)


@contextlib.contextmanager
def _synthetic_preoutcome_authority() -> Iterator[None]:
    """Bypass only the frozen protocol's intentional runnable hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x51" * 32, b"\xa7" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
    }


def _completed_distribution(
    values: Sequence[int],
) -> dict[str, float | int | None]:
    if values:
        return _distribution(values)
    return {
        "count": 0,
        "sum": 0,
        "min": None,
        "median": None,
        "mean": None,
        "p95": None,
        "p99": None,
        "max": None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=DEFAULT_PREFIX_TOKENS,
        help=f"canonical first-book token count (default: {DEFAULT_PREFIX_TOKENS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"artifact path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()
    if args.prefix_tokens < 2:
        parser.error("--prefix-tokens must be at least 2")
    return args


def _load_prefix(prefix_tokens: int) -> tuple[object, object, np.ndarray, int]:
    started_ns = time.perf_counter_ns()
    verified = load_verified_train_input()
    input_load_ns = time.perf_counter_ns() - started_ns
    books = tuple(verified.iter_books())
    if len(books) != 49 or books[0].token_count < prefix_tokens:
        raise RuntimeError("canonical TRAIN first book cannot supply the requested prefix")
    prefix = np.frombuffer(books[0].token_bytes_le_i32, dtype="<i4")[:prefix_tokens]
    if prefix.size != prefix_tokens:
        raise RuntimeError("canonical TRAIN prefix length drifted")
    return verified, books[0], prefix, input_load_ns


def _bounded_definition(prefix_tokens: int) -> tuple[
    dict[str, object],
    NativeTemporalCircuitDefinition,
    NativeTemporalCircuitDefinition,
    int,
]:
    started_ns = time.perf_counter_ns()
    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=False)
    production = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    bounded = NativeTemporalCircuitDefinition(
        replace(
            production.config,
            segment_capacity=(prefix_tokens - 1) * ACTIVE_COLUMNS,
        ),
        production.token_codes,
        production.threshold_offsets_mv,
    )
    return protocol, production, bounded, time.perf_counter_ns() - started_ns


def _counters(step: TemporalCircuitStep) -> tuple[int, int, int, int]:
    missing = [name for name in COUNTER_FIELDS if not hasattr(step, name)]
    if missing:
        raise RuntimeError(
            "native recognition counter instrumentation is unavailable: "
            + ", ".join(missing)
        )
    return tuple(int(getattr(step, name)) for name in COUNTER_FIELDS)  # type: ignore[return-value]


def _identity_payload(identity: object) -> dict[str, object]:
    return {
        "body_sha256": identity.body_sha256,
        "checkpoint_bytes": identity.checkpoint_bytes,
        "circuit_identity_sha256": identity.circuit_identity_sha256,
        "cohort_count": identity.cohort_count,
        "event_count": identity.event_count,
        "learning_step": identity.learning_step,
        "logical_state_sha256": identity.logical_state_sha256,
        "next_reserve_cell": identity.next_reserve_cell,
        "poison_code": identity.poison_code,
        "protocol_identity_sha256": identity.protocol_identity_sha256,
        "segment_count": identity.segment_count,
        "semantic_index_sha256": identity.semantic_index_sha256,
        "source_bundle_identity_sha256": identity.source_bundle_identity_sha256,
    }


def _run_prefix(
    circuit: NativeTemporalCircuit,
    prefix: np.ndarray,
) -> tuple[
    dict[str, list[int]],
    dict[str, int | bool],
    dict[str, int],
    dict[str, int | str] | None,
]:
    series = {
        "latency_ns": [],
        "candidate_segments": [],
        "matching_segments": [],
        "active_segments": [],
        "postings_visited": [],
        "live_segments": [],
        "predicted_columns": [],
        "allocated_segments": [],
        "punished_segments": [],
    }
    live_segments = 0
    predicted_steps = 0
    winner_spikes = 0
    basket_spikes = 0
    causal_winner_to_basket = True
    causal_basket_to_gaba = True
    scientific_stop = None

    gc_enabled = gc.isenabled()
    gc.disable()
    replay_started_ns = time.perf_counter_ns()
    try:
        for index, raw_token in enumerate(prefix):
            started_ns = time.perf_counter_ns()
            try:
                emitted = circuit.advance_block(
                    [int(raw_token)],
                    mode=TemporalStepMode.LEARN,
                    reset_before=index == 0,
                )
            except NativeTemporalRuntimeError as exc:
                if exc.status is not NativeTemporalRuntimeStatus.SCIENTIFIC_STOP:
                    raise
                if exc.completed_prefix != 0 or exc.completed_steps:
                    raise RuntimeError(
                        "single-token scientific STOP reported committed work"
                    ) from exc
                scientific_stop = {
                    "zero_based_step_index": index,
                    "token_id": int(raw_token),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "native_status": exc.status.name,
                    "native_status_code": int(exc.status),
                    "poison_code": exc.poison_code,
                    "stage_code": exc.stage_code,
                    "completed_step_count": len(series["latency_ns"]),
                    "live_segment_count": live_segments,
                    "failed_call_latency_ns": time.perf_counter_ns() - started_ns,
                }
                break
            latency_ns = time.perf_counter_ns() - started_ns
            if len(emitted) != 1:
                raise RuntimeError("native prefix step did not emit one result")
            step = emitted[0]
            candidate, matching, active, postings = _counters(step)
            live_segments += step.allocated_segments
            predicted_steps += int(step.predicted)
            winner_spikes += len(step.winner_spikes)
            basket_spikes += len(step.basket_spikes)
            causal_winner_to_basket &= all(
                basket.tick == winner.tick + 1
                for winner, basket in zip(
                    step.winner_spikes,
                    step.basket_spikes,
                    strict=True,
                )
            )
            causal_basket_to_gaba &= all(
                onset == basket.tick + 1
                for basket, onset in zip(
                    step.basket_spikes,
                    step.inhibition_onset_ticks,
                    strict=True,
                )
            )
            series["latency_ns"].append(latency_ns)
            series["candidate_segments"].append(candidate)
            series["matching_segments"].append(matching)
            series["active_segments"].append(active)
            series["postings_visited"].append(postings)
            series["live_segments"].append(live_segments)
            series["predicted_columns"].append(step.predicted_columns)
            series["allocated_segments"].append(step.allocated_segments)
            series["punished_segments"].append(step.punished_segments)
    finally:
        replay_wall_ns = time.perf_counter_ns() - replay_started_ns
        if gc_enabled:
            gc.enable()

    if scientific_stop is not None:
        scientific_stop["elapsed_replay_wall_ns"] = replay_wall_ns
    phase_totals = circuit.engineering_profile_summary().to_record()
    if phase_totals["completed_steps"] != len(series["latency_ns"]):
        raise RuntimeError("native phase profile completed-step count drifted")
    return (
        series,
        {
            "replay_wall_ns": replay_wall_ns,
            "predicted_steps": predicted_steps,
            "winner_spikes": winner_spikes,
            "basket_spikes": basket_spikes,
            "all_winner_to_basket_ticks": causal_winner_to_basket,
            "all_basket_to_gaba_ticks": causal_basket_to_gaba,
        },
        phase_totals,
        scientific_stop,
    )


def main() -> int:
    args = _parse_args()
    _enable_temporal_profile_build()
    prefix_tokens = args.prefix_tokens
    output_path = args.output
    verified, first_book, prefix, input_load_ns = _load_prefix(prefix_tokens)
    protocol, production, bounded, definition_build_ns = _bounded_definition(prefix_tokens)
    production_token_ids = {token for token, _columns in production.token_codes}
    if any(int(token) not in production_token_ids for token in prefix):
        raise RuntimeError("TRAIN prefix contains a token outside the production codebook")

    prefix_bytes = np.asarray(prefix, dtype="<i4").tobytes()
    threshold_bytes = np.asarray(
        bounded.threshold_offsets_mv,
        dtype="<f8",
    ).tobytes()
    production_identity = production.circuit_identity_sha256()
    bounded_identity = bounded.circuit_identity_sha256()

    with _synthetic_preoutcome_authority():
        construction_started_ns = time.perf_counter_ns()
        with NativeTemporalCircuit.fresh(bounded) as circuit:
            native_construction_ns = time.perf_counter_ns() - construction_started_ns
            series, physical, phase_totals, scientific_stop = _run_prefix(circuit, prefix)
            final_identity = (
                None if scientific_stop is not None else circuit.checkpoint_identity()
            )
        native_identity = loaded_library_identity()

    if scientific_stop is not None:
        payload = {
            "schema_version": 2,
            "benchmark": "r3-native-train-prefix-cost",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "outcome": "SCIENTIFIC_STOP",
            "claim_boundary": (
                f"first {prefix_tokens:,} canonical TRAIN tokens only; synthetic pre-outcome authority, "
                "bounded capacity, no held-out/C1/capability/cohort evidence"
            ),
            "train_input": {
                "candidate_id": verified.candidate_id,
                "train_input_authority_sha256": verified.train_input_authority_sha256,
                "train_stream_sha256": verified.train_stream_sha256,
                "tokenizer_sha256": verified.tokenizer_sha256,
                "prefix_book": first_book.short_name,
                "prefix_token_count": prefix_tokens,
                "prefix_token_sha256_le_i32": hashlib.sha256(prefix_bytes).hexdigest(),
                "held_out_books_materialized": 0,
                "input_load_ns": input_load_ns,
            },
            "definition": {
                "arm": list(ARM),
                "construction_seed": CONSTRUCTION_SEED,
                "protocol_sha256": protocol_authority.protocol_sha256(),
                "protocol_runnable": protocol["source_binding"]["runnable"],
                "production_definition_identity_sha256": production_identity,
                "bounded_definition_identity_sha256": bounded_identity,
                "bounded_segment_capacity": bounded.config.segment_capacity,
                "definition_build_ns": definition_build_ns,
                "native_construction_ns": native_construction_ns,
            },
            "stop": scientific_stop,
            "completed_measurement": {
                "mode": "LEARN",
                "timer": "time.perf_counter_ns",
                "gc_during_replay": "disabled",
                "native_phase_timer": "std::time::Instant",
                "temporal_phase_totals": phase_totals,
                "distributions": {
                    name: _completed_distribution(values)
                    for name, values in series.items()
                },
                "predicted_steps": physical["predicted_steps"],
                "winner_spikes": physical["winner_spikes"],
                "basket_spikes": physical["basket_spikes"],
                "all_winner_to_basket_ticks": physical[
                    "all_winner_to_basket_ticks"
                ],
                "all_basket_to_gaba_ticks": physical["all_basket_to_gaba_ticks"],
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "native_source_digest": native_identity.source_digest,
                "native_binary_size": native_identity.binary_size,
                "native_binary_sha256": native_identity.binary_sha256,
                "native_cargo_features": ["temporal-profile"],
            },
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote partial {output_path}", flush=True)
        return 2

    assert final_identity is not None
    if final_identity.segment_count != series["live_segments"][-1]:
        raise RuntimeError("tracked and native live-segment counts differ")
    if not physical["all_winner_to_basket_ticks"] or not physical[
        "all_basket_to_gaba_ticks"
    ]:
        raise RuntimeError("physical winner/basket/GABA causality failed")

    latency = _distribution(series["latency_ns"])
    call_rate = prefix_tokens * 1_000_000_000.0 / int(latency["sum"])
    loop_rate = prefix_tokens * 1_000_000_000.0 / int(physical["replay_wall_ns"])
    required_rate = TOTAL_C1_STEPS / CAP_SECONDS
    headroom_rate = TOTAL_C1_STEPS / HEADROOM_SECONDS
    cap_projection = {
        "schedule_steps": TOTAL_C1_STEPS,
        "cap_seconds": CAP_SECONDS,
        "required_steps_per_second": required_rate,
        "ten_percent_headroom_steps_per_second": headroom_rate,
        "abi_call_steps_per_second": call_rate,
        "instrumented_loop_steps_per_second": loop_rate,
        "abi_call_multiple_of_required": call_rate / required_rate,
        "instrumented_loop_multiple_of_required": loop_rate / required_rate,
        "abi_call_projected_hours": TOTAL_C1_STEPS / call_rate / 3_600,
        "instrumented_loop_projected_hours": TOTAL_C1_STEPS / loop_rate / 3_600,
        "claim_boundary": (
            "straight-line prefix projection only; excludes full-run growth, "
            "construction, TRAIN input loading, checkpoints, I/O, and arm topology"
        ),
    }

    payload = {
        "schema_version": 2,
        "benchmark": "r3-native-train-prefix-cost",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            f"first {prefix_tokens:,} canonical TRAIN tokens only; synthetic pre-outcome authority, "
            "bounded capacity, no held-out/C1/capability/cohort evidence"
        ),
        "train_input": {
            "candidate_id": verified.candidate_id,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "full_train_token_count": verified.token_count,
            "prefix_book": first_book.short_name,
            "prefix_token_count": prefix_tokens,
            "prefix_token_sha256_le_i32": hashlib.sha256(prefix_bytes).hexdigest(),
            "prefix_unique_token_ids": int(np.unique(prefix).size),
            "train_books_materialized": 49,
            "held_out_books_materialized": 0,
            "input_load_ns": input_load_ns,
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "protocol_sha256": protocol_authority.protocol_sha256(),
            "protocol_runnable": protocol["source_binding"]["runnable"],
            "production_definition_identity_sha256": production_identity,
            "bounded_definition_identity_sha256": bounded_identity,
            "production_segment_capacity": production.config.segment_capacity,
            "bounded_segment_capacity": bounded.config.segment_capacity,
            "bounded_capacity_rule": "(prefix_tokens - one book seed) * 11",
            "column_count": bounded.config.column_count,
            "max_pyramidal_cells": bounded.config.max_pyramidal_cells,
            "physical_cell_count": len(bounded.threshold_offsets_mv),
            "token_code_count": len(bounded.token_codes),
            "threshold_offsets_sha256_le_f64": hashlib.sha256(
                threshold_bytes
            ).hexdigest(),
            "definition_build_ns": definition_build_ns,
            "native_construction_ns": native_construction_ns,
        },
        "measurement": {
            "mode": "LEARN",
            "timer": "time.perf_counter_ns",
            "gc_during_replay": "disabled",
            "native_phase_timer": "std::time::Instant",
            "temporal_phase_totals": phase_totals,
            "book_resets": 1,
            "transition_steps": prefix_tokens - 1,
            "latency_ns": latency,
            "recognition": {
                "candidate_segments": _distribution(series["candidate_segments"]),
                "matching_segments": _distribution(series["matching_segments"]),
                "active_segments": _distribution(series["active_segments"]),
                "postings_visited": _distribution(series["postings_visited"]),
            },
            "live_segments_final": final_identity.segment_count,
            "allocated_segments_total": sum(series["allocated_segments"]),
            "punished_segments_total": sum(series["punished_segments"]),
            "predicted_steps": physical["predicted_steps"],
            "winner_spikes": physical["winner_spikes"],
            "basket_spikes": physical["basket_spikes"],
            "all_winner_to_basket_ticks": physical["all_winner_to_basket_ticks"],
            "all_basket_to_gaba_ticks": physical["all_basket_to_gaba_ticks"],
            "replay_wall_ns": physical["replay_wall_ns"],
        },
        "final_identity": _identity_payload(final_identity),
        "cap_projection": cap_projection,
        "step_series": series,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_size": native_identity.binary_size,
            "native_binary_sha256": native_identity.binary_sha256,
            "native_cargo_features": ["temporal-profile"],
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
