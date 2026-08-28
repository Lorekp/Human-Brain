"""Fail-fast native physical cost slice over the first two TRAIN books.

This is pre-outcome engineering evidence.  It authenticates only TRAIN data,
uses the production codebook, defaults to fitted seed-17 cells at arm (16, 18),
resets transient context atomically at each book boundary, and never enters C1.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator, Sequence
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
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot.calibration_v3_definitions import build_r3_f0_definition
from experiments.r3_pilot.r3_f0_train_input import load_verified_train_input
from humanbrain.native import loaded_library_identity
import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    NativeTemporalRuntimeStatus,
    TemporalCircuitStep,
    TemporalStepMode,
)


ACTIVE_COLUMNS = 11
ARM = (16, 18)
BOOK_COUNT = 2
CONSTRUCTION_SEED = 17
OUTPUT_PATH = Path(__file__).with_name("r3_native_train_multibook_cost.json")
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


def _optional_distribution(
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
    parser.add_argument("--theta-match", type=int, default=ARM[0])
    parser.add_argument("--theta-active", type=int, default=ARM[1])
    parser.add_argument("--construction-seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        help=f"artifact path (default arm only: {OUTPUT_PATH})",
    )
    args = parser.parse_args()
    if not 1 <= args.theta_match <= args.theta_active <= 20:
        parser.error("thresholds must satisfy 1 <= theta-match <= theta-active <= 20")
    if not 0 <= args.construction_seed <= 0xFFFF_FFFF:
        parser.error("construction-seed must fit uint32")
    non_default = (
        (args.theta_match, args.theta_active) != ARM
        or args.construction_seed != CONSTRUCTION_SEED
    )
    if non_default and args.output is None:
        parser.error("non-default threshold or seed arms require an explicit --output")
    args.output = args.output or OUTPUT_PATH
    if non_default and args.output.resolve() == OUTPUT_PATH.resolve():
        parser.error("non-default arms cannot overwrite the committed default artifact")
    return args


def _load_books() -> tuple[object, tuple[tuple[object, np.ndarray], ...], int]:
    started_ns = time.perf_counter_ns()
    verified = load_verified_train_input()
    input_load_ns = time.perf_counter_ns() - started_ns
    encoded = tuple(verified.iter_books())
    if len(encoded) != 49:
        raise RuntimeError("canonical verified TRAIN book count drifted")
    selected: list[tuple[object, np.ndarray]] = []
    for ordinal, book in enumerate(encoded[:BOOK_COUNT]):
        if book.ordinal != ordinal or book.token_count < 2:
            raise RuntimeError("canonical first-two-book authority drifted")
        tokens = np.frombuffer(book.token_bytes_le_i32, dtype="<i4")
        if tokens.size != book.token_count:
            raise RuntimeError(f"{book.short_name} token materialization drifted")
        selected.append((book, tokens))
    return verified, tuple(selected), input_load_ns


def _bounded_definition(
    transition_count: int,
    arm: tuple[int, int],
    construction_seed: int,
) -> tuple[
    dict[str, Any],
    NativeTemporalCircuitDefinition,
    NativeTemporalCircuitDefinition,
    int,
]:
    started_ns = time.perf_counter_ns()
    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=False)
    if protocol["source_binding"]["runnable"] is not False:
        raise RuntimeError("pre-outcome benchmark requires the C1 runnable hold")
    production = build_r3_f0_definition(protocol, arm, construction_seed)
    bounded = NativeTemporalCircuitDefinition(
        replace(
            production.config,
            segment_capacity=transition_count * ACTIVE_COLUMNS,
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


def _new_series() -> dict[str, list[int]]:
    return {
        "latency_ns": [],
        "candidate_segments": [],
        "matching_segments": [],
        "active_segments": [],
        "postings_visited": [],
        "allocated_segments": [],
        "punished_segments": [],
    }


def _extend_series(
    target: dict[str, list[int]], source: dict[str, list[int]]
) -> None:
    for name in target:
        target[name].extend(source[name])


def _summary(
    *,
    series: dict[str, list[int]],
    replay_wall_ns: int,
    live_segments_start: int,
    live_segments_end: int,
    predicted_steps: int,
    winner_spikes: int,
    basket_spikes: int,
    all_winner_to_basket_ticks: bool,
    all_basket_to_gaba_ticks: bool,
) -> dict[str, Any]:
    completed = len(series["latency_ns"])
    latency_sum = sum(series["latency_ns"])
    return {
        "completed_token_steps": completed,
        "latency_ns": _optional_distribution(series["latency_ns"]),
        "recognition": {
            name: _optional_distribution(series[name])
            for name in COUNTER_FIELDS
        },
        "segments": {
            "live_start": live_segments_start,
            "live_end": live_segments_end,
            "live_delta": live_segments_end - live_segments_start,
            "allocated_total": sum(series["allocated_segments"]),
            "punished_total": sum(series["punished_segments"]),
        },
        "mechanism": {
            "predicted_steps": predicted_steps,
            "winner_spikes": winner_spikes,
            "basket_spikes": basket_spikes,
            "all_winner_to_basket_ticks": all_winner_to_basket_ticks,
            "all_basket_to_gaba_ticks": all_basket_to_gaba_ticks,
        },
        "rate": {
            "abi_call_steps_per_second": (
                completed * 1_000_000_000.0 / latency_sum
                if latency_sum
                else None
            ),
            "instrumented_loop_steps_per_second": (
                completed * 1_000_000_000.0 / replay_wall_ns
                if replay_wall_ns
                else None
            ),
            "replay_wall_ns": replay_wall_ns,
        },
    }


def _run_books(
    circuit: NativeTemporalCircuit,
    selected: tuple[tuple[object, np.ndarray], ...],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    aggregate_series = _new_series()
    aggregate_predicted = 0
    aggregate_winner_spikes = 0
    aggregate_basket_spikes = 0
    aggregate_winner_to_basket = True
    aggregate_basket_to_gaba = True
    book_summaries: list[dict[str, Any]] = []
    live_segments = 0
    scientific_stop: dict[str, Any] | None = None
    global_step = 0

    gc_enabled = gc.isenabled()
    gc.disable()
    aggregate_started_ns = time.perf_counter_ns()
    try:
        for book, tokens in selected:
            series = _new_series()
            live_start = live_segments
            predicted_steps = 0
            winner_spikes = 0
            basket_spikes = 0
            winner_to_basket = True
            basket_to_gaba = True
            reset_evidence: dict[str, Any] | None = None
            book_started_ns = time.perf_counter_ns()

            for token_index, raw_token in enumerate(tokens):
                token_id = int(raw_token)
                started_ns = time.perf_counter_ns()
                try:
                    emitted = circuit.advance_block(
                        [token_id],
                        mode=TemporalStepMode.LEARN,
                        reset_before=token_index == 0,
                    )
                except NativeTemporalRuntimeError as exc:
                    if exc.status is not NativeTemporalRuntimeStatus.SCIENTIFIC_STOP:
                        raise
                    if exc.completed_prefix != 0 or exc.completed_steps:
                        raise RuntimeError(
                            "single-token scientific STOP reported committed work"
                        ) from exc
                    scientific_stop = {
                        "kind": "NATIVE_SCIENTIFIC_STOP",
                        "book_ordinal": book.ordinal,
                        "book_short_name": book.short_name,
                        "book_zero_based_token_index": token_index,
                        "global_zero_based_step_index": global_step,
                        "token_id": token_id,
                        "native_status": exc.status.name,
                        "native_status_code": int(exc.status),
                        "poison_code": exc.poison_code,
                        "stage_code": exc.stage_code,
                        "error_message": str(exc),
                        "failed_call_latency_ns": time.perf_counter_ns()
                        - started_ns,
                        "live_segments_before_failed_step": live_segments,
                    }
                    break

                latency_ns = time.perf_counter_ns() - started_ns
                if len(emitted) != 1:
                    raise RuntimeError("native token step did not emit one result")
                step = emitted[0]
                candidate, matching, active, postings = _counters(step)
                series["latency_ns"].append(latency_ns)
                series["candidate_segments"].append(candidate)
                series["matching_segments"].append(matching)
                series["active_segments"].append(active)
                series["postings_visited"].append(postings)
                series["allocated_segments"].append(step.allocated_segments)
                series["punished_segments"].append(step.punished_segments)
                live_before = live_segments
                live_segments += step.allocated_segments
                predicted_steps += int(step.predicted)
                winner_spikes += len(step.winner_spikes)
                basket_spikes += len(step.basket_spikes)
                winner_to_basket &= all(
                    basket.tick == winner.tick + 1
                    for winner, basket in zip(
                        step.winner_spikes, step.basket_spikes, strict=True
                    )
                )
                basket_to_gaba &= all(
                    onset == basket.tick + 1
                    for basket, onset in zip(
                        step.basket_spikes,
                        step.inhibition_onset_ticks,
                        strict=True,
                    )
                )

                if token_index == 0:
                    reset_evidence = {
                        "reset_before_requested": True,
                        "event_count_after_step": step.event_count,
                        "candidate_segments": candidate,
                        "matching_segments": matching,
                        "active_segments": active,
                        "postings_visited": postings,
                        "allocated_segments": step.allocated_segments,
                        "punished_segments": step.punished_segments,
                        "empty_transient_state_observed": (
                            step.event_count == 1
                            and candidate == 0
                            and matching == 0
                            and active == 0
                            and postings == 0
                            and step.allocated_segments == 0
                            and step.punished_segments == 0
                        ),
                    }
                    if not reset_evidence["empty_transient_state_observed"]:
                        raise RuntimeError(
                            f"{book.short_name} boundary reset did not expose empty transient state"
                        )
                elif active == 0 and matching > 0 and step.allocated_segments < ACTIVE_COLUMNS:
                    scientific_stop = {
                        "kind": "CONTEXT_COLLISION_STOP",
                        "reason": (
                            "matching-but-not-active context failed to allocate the full "
                            "11-cell temporal key"
                        ),
                        "book_ordinal": book.ordinal,
                        "book_short_name": book.short_name,
                        "book_zero_based_token_index": token_index,
                        "global_zero_based_step_index": global_step,
                        "token_id": token_id,
                        "event_count": step.event_count,
                        "candidate_segments": candidate,
                        "matching_segments": matching,
                        "active_segments": active,
                        "postings_visited": postings,
                        "allocated_segments": step.allocated_segments,
                        "punished_segments": step.punished_segments,
                        "predicted": step.predicted,
                        "predicted_columns": step.predicted_columns,
                        "live_segments_before_step": live_before,
                        "live_segments_after_step": live_segments,
                    }
                    global_step += 1
                    break

                global_step += 1
                if (token_index + 1) % 16_384 == 0:
                    print(
                        f"{book.short_name}: {token_index + 1:,}/{book.token_count:,} "
                        f"tokens, {live_segments:,} live segments",
                        flush=True,
                    )

            book_wall_ns = time.perf_counter_ns() - book_started_ns
            _extend_series(aggregate_series, series)
            aggregate_predicted += predicted_steps
            aggregate_winner_spikes += winner_spikes
            aggregate_basket_spikes += basket_spikes
            aggregate_winner_to_basket &= winner_to_basket
            aggregate_basket_to_gaba &= basket_to_gaba
            completed = len(series["latency_ns"])
            complete = scientific_stop is None and completed == book.token_count
            checkpoint_segments = None
            if complete:
                checkpoint_segments = circuit.checkpoint_identity().segment_count
                if checkpoint_segments != live_segments:
                    raise RuntimeError("tracked and native live-segment counts differ")
            book_summaries.append(
                {
                    "ordinal": book.ordinal,
                    "corpus_index": book.corpus_index,
                    "short_name": book.short_name,
                    "token_count": book.token_count,
                    "token_sha256_le_i32": book.token_sha256_le_i32,
                    "transition_steps": book.token_count - 1,
                    "completed": complete,
                    "reset_evidence": reset_evidence,
                    "native_checkpoint_segment_count": checkpoint_segments,
                    **_summary(
                        series=series,
                        replay_wall_ns=book_wall_ns,
                        live_segments_start=live_start,
                        live_segments_end=live_segments,
                        predicted_steps=predicted_steps,
                        winner_spikes=winner_spikes,
                        basket_spikes=basket_spikes,
                        all_winner_to_basket_ticks=winner_to_basket,
                        all_basket_to_gaba_ticks=basket_to_gaba,
                    ),
                }
            )
            if scientific_stop is not None:
                break
            print(
                f"{book.short_name}: complete, {live_segments:,} live segments",
                flush=True,
            )
    finally:
        aggregate_wall_ns = time.perf_counter_ns() - aggregate_started_ns
        if gc_enabled:
            gc.enable()

    aggregate = _summary(
        series=aggregate_series,
        replay_wall_ns=aggregate_wall_ns,
        live_segments_start=0,
        live_segments_end=live_segments,
        predicted_steps=aggregate_predicted,
        winner_spikes=aggregate_winner_spikes,
        basket_spikes=aggregate_basket_spikes,
        all_winner_to_basket_ticks=aggregate_winner_to_basket,
        all_basket_to_gaba_ticks=aggregate_basket_to_gaba,
    )
    aggregate["book_summaries"] = book_summaries
    aggregate["book_resets_requested"] = len(book_summaries)
    aggregate["completed_books"] = sum(
        int(book["completed"]) for book in book_summaries
    )
    return aggregate, scientific_stop


def main() -> int:
    args = _parse_args()
    arm = (args.theta_match, args.theta_active)
    verified, selected, input_load_ns = _load_books()
    transition_count = sum(book.token_count - 1 for book, _tokens in selected)
    total_tokens = sum(book.token_count for book, _tokens in selected)
    protocol, production, bounded, definition_build_ns = _bounded_definition(
        transition_count,
        arm,
        args.construction_seed,
    )
    production_token_ids = {token for token, _columns in production.token_codes}
    if any(
        int(token) not in production_token_ids
        for _book, tokens in selected
        for token in tokens
    ):
        raise RuntimeError("selected TRAIN books contain an out-of-codebook token")

    selected_bytes = b"".join(
        np.asarray(tokens, dtype="<i4").tobytes() for _book, tokens in selected
    )
    threshold_bytes = np.asarray(
        bounded.threshold_offsets_mv, dtype="<f8"
    ).tobytes()
    production_identity = production.circuit_identity_sha256()
    bounded_identity = bounded.circuit_identity_sha256()

    with _synthetic_preoutcome_authority():
        construction_started_ns = time.perf_counter_ns()
        with NativeTemporalCircuit.fresh(bounded) as circuit:
            native_construction_ns = time.perf_counter_ns() - construction_started_ns
            measurement, scientific_stop = _run_books(circuit, selected)
            final_identity = (
                circuit.checkpoint_identity() if scientific_stop is None else None
            )
        native_identity = loaded_library_identity()

    if scientific_stop is None:
        assert final_identity is not None
        if final_identity.segment_count != measurement["segments"]["live_end"]:
            raise RuntimeError("final native and tracked live-segment counts differ")
        if not measurement["mechanism"]["all_winner_to_basket_ticks"] or not measurement[
            "mechanism"
        ]["all_basket_to_gaba_ticks"]:
            raise RuntimeError("physical winner/basket/GABA causality failed")

    payload = {
        "schema_version": 1,
        "benchmark": "r3-native-train-multibook-cost",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": "PASS" if scientific_stop is None else "SCIENTIFIC_STOP",
        "claim_boundary": (
            "one physical LEARN replay of the first two canonical TRAIN books only; "
            f"arm ({arm[0]},{arm[1]}), seed {args.construction_seed}, production "
            "codebook, exact atomic transient reset "
            "at both book boundaries, synthetic pre-outcome authority, and a "
            "transition-bounded segment store; no held-out/C1/capability/cohort, "
            "later-book, checkpoint-I/O, or multi-arm evidence"
        ),
        "train_input": {
            "candidate_id": verified.candidate_id,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "full_train_token_count": verified.token_count,
            "verified_train_books_materialized": 49,
            "replayed_book_count": BOOK_COUNT,
            "replayed_token_count": total_tokens,
            "replayed_transition_count": transition_count,
            "replayed_token_sha256_le_i32": hashlib.sha256(
                selected_bytes
            ).hexdigest(),
            "replayed_unique_token_ids": int(
                np.unique(
                    np.concatenate([tokens for _book, tokens in selected])
                ).size
            ),
            "books": [
                {
                    "ordinal": book.ordinal,
                    "corpus_index": book.corpus_index,
                    "short_name": book.short_name,
                    "token_count": book.token_count,
                    "token_sha256_le_i32": book.token_sha256_le_i32,
                }
                for book, _tokens in selected
            ],
            "held_out_books_materialized": 0,
            "input_load_ns": input_load_ns,
        },
        "definition": {
            "arm": list(arm),
            "construction_seed": args.construction_seed,
            "protocol_sha256": protocol_authority.protocol_sha256(),
            "protocol_runnable": protocol["source_binding"]["runnable"],
            "production_definition_identity_sha256": production_identity,
            "bounded_definition_identity_sha256": bounded_identity,
            "production_segment_capacity": production.config.segment_capacity,
            "bounded_segment_capacity": bounded.config.segment_capacity,
            "bounded_capacity_rule": (
                "sum((book_token_count - one transient seed) * 11)"
            ),
            "worst_case_transition_count": transition_count,
            "max_segments_per_transition": ACTIVE_COLUMNS,
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
            "single_token_abi_calls": True,
            **measurement,
        },
        "scientific_stop": scientific_stop,
        "final_checkpoint_identity": (
            None
            if final_identity is None
            else {
                "segment_count": final_identity.segment_count,
                "event_count": final_identity.event_count,
                "learning_step": final_identity.learning_step,
                "logical_state_sha256": final_identity.logical_state_sha256,
            }
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_size": native_identity.binary_size,
            "native_binary_sha256": native_identity.binary_sha256,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0 if scientific_stop is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
