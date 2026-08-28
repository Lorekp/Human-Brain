"""Bounded TRAIN-only localizer for the first R3-F0 owner-reuse event.

This is fresh development evidence, not a calibration-v3 retry.  It executes
exactly Genesis calls 0..19,841 unless functional owner/allocation identity
diverges first.  Recognition counters are telemetry: a foreign theta-match
that remains below active and does not change allocation is not a failure.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import struct
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
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    TemporalStepMode,
)


ACTIVE_COLUMNS = 11
ARM = (16, 18)
CONSTRUCTION_SEED = 17
BOUNDARY_CALLS = 19_842
OUTPUT_PATH = Path(__file__).with_name("r3_owner_reuse_first_divergence.json")
_TOKEN = struct.Struct("<I")
_WINNERS = struct.Struct("<11I")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"new development artifact path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def _winner_tuple(step: object) -> tuple[int, ...]:
    winners = tuple(int(spike.cell_id) for spike in getattr(step, "winner_spikes"))
    if len(winners) != ACTIVE_COLUMNS:
        raise ValueError(f"winner width is {len(winners)}, expected {ACTIVE_COLUMNS}")
    return winners


def _step_counters(step: object) -> dict[str, int | bool]:
    return {
        "event_count": int(getattr(step, "event_count")),
        "candidate_segments": int(getattr(step, "candidate_segments")),
        "matching_segments": int(getattr(step, "matching_segments")),
        "active_segments": int(getattr(step, "active_segments")),
        "predicted_columns": int(getattr(step, "predicted_columns")),
        "predicted": bool(getattr(step, "predicted")),
        "allocated_segments": int(getattr(step, "allocated_segments")),
        "punished_segments": int(getattr(step, "punished_segments")),
        "postings_visited": int(getattr(step, "postings_visited")),
    }


class OwnerReuseLedger:
    """Minimal semantic ledger for exact-source transition ownership."""

    def __init__(self) -> None:
        self.owners_by_key: dict[bytes, bytes] = {}
        self.owner_tuples: set[bytes] = set()
        self.seed_tuples: set[bytes] = set()
        self.fanout_by_source: dict[bytes, int] = {}
        self.previous_winners: bytes | None = None
        self.calls_issued = 0
        self.accepted_calls = 0
        self.resets = 0
        self.transitions = 0
        self.repeats = 0
        self.allocated_segments = 0
        self.punished_segments = 0
        self.maximum_candidate_segments = 0
        self.maximum_matching_segments = 0
        self.maximum_active_segments = 0
        self.maximum_postings_visited = 0
        self.first_matching_diagnostic: dict[str, object] | None = None
        self.winner_stream = hashlib.sha256()
        self.key_owner_stream = hashlib.sha256()

    def _detail(
        self,
        kind: str,
        *,
        call_index: int,
        book_ordinal: int,
        book_name: str,
        token_index: int,
        token_id: int,
        counters: dict[str, int | bool],
        expected: object,
        observed: object,
        source_winners: bytes | None = None,
    ) -> dict[str, object]:
        return {
            "kind": kind,
            "call_index": call_index,
            "book_ordinal": book_ordinal,
            "book_name": book_name,
            "token_index": token_index,
            "token_id": token_id,
            "expected": expected,
            "observed": observed,
            "source_winner_cells": (
                None if source_winners is None else list(_WINNERS.unpack(source_winners))
            ),
            "source_fanout": (
                None
                if source_winners is None
                else self.fanout_by_source.get(source_winners, 0)
            ),
            "native_counters": counters,
        }

    def observe(
        self,
        step: object,
        *,
        call_index: int,
        book_ordinal: int,
        book_name: str,
        token_index: int,
        token_id: int,
        reset_before: bool,
    ) -> dict[str, object] | None:
        """Return the first semantic divergence, otherwise accept one call."""

        counters = _step_counters(step)
        try:
            winner_ids = _winner_tuple(step)
        except ValueError as exc:
            return self._detail(
                "WINNER_WIDTH",
                call_index=call_index,
                book_ordinal=book_ordinal,
                book_name=book_name,
                token_index=token_index,
                token_id=token_id,
                counters=counters,
                expected=ACTIVE_COLUMNS,
                observed=str(exc),
            )
        winners = _WINNERS.pack(*winner_ids)
        self.calls_issued += 1
        self.winner_stream.update(winners)
        self.allocated_segments += int(counters["allocated_segments"])
        self.punished_segments += int(counters["punished_segments"])
        self.maximum_candidate_segments = max(
            self.maximum_candidate_segments, int(counters["candidate_segments"])
        )
        self.maximum_matching_segments = max(
            self.maximum_matching_segments, int(counters["matching_segments"])
        )
        self.maximum_active_segments = max(
            self.maximum_active_segments, int(counters["active_segments"])
        )
        self.maximum_postings_visited = max(
            self.maximum_postings_visited, int(counters["postings_visited"])
        )

        if reset_before:
            reset_observed = [
                counters["event_count"],
                counters["candidate_segments"],
                counters["matching_segments"],
                counters["active_segments"],
                counters["allocated_segments"],
                counters["punished_segments"],
                counters["predicted_columns"],
            ]
            if reset_observed != [1, 0, 0, 0, 0, 0, 0]:
                return self._detail(
                    "BOOK_RESET_SEED",
                    call_index=call_index,
                    book_ordinal=book_ordinal,
                    book_name=book_name,
                    token_index=token_index,
                    token_id=token_id,
                    counters=counters,
                    expected=[1, 0, 0, 0, 0, 0, 0],
                    observed=reset_observed,
                )
            if winners not in self.owner_tuples:
                self.seed_tuples.add(winners)
            self.previous_winners = winners
            self.resets += 1
            self.accepted_calls += 1
            return None

        if self.previous_winners is None:
            return self._detail(
                "MISSING_SOURCE_CONTEXT",
                call_index=call_index,
                book_ordinal=book_ordinal,
                book_name=book_name,
                token_index=token_index,
                token_id=token_id,
                counters=counters,
                expected="one accepted reset seed",
                observed=None,
            )

        source = self.previous_winners
        source_fanout = self.fanout_by_source.get(source, 0)
        expected_matching = ACTIVE_COLUMNS * source_fanout
        if (
            int(counters["matching_segments"]) != expected_matching
            and self.first_matching_diagnostic is None
        ):
            self.first_matching_diagnostic = self._detail(
                "FOREIGN_OR_MISSING_SOURCE_MATCH_DIAGNOSTIC",
                call_index=call_index,
                book_ordinal=book_ordinal,
                book_name=book_name,
                token_index=token_index,
                token_id=token_id,
                counters=counters,
                expected=expected_matching,
                observed=counters["matching_segments"],
                source_winners=source,
            )

        key = _TOKEN.pack(token_id) + source
        known_owner = self.owners_by_key.get(key)
        if known_owner is None:
            if int(counters["allocated_segments"]) != ACTIVE_COLUMNS:
                return self._detail(
                    "NOVEL_TRANSITION_DID_NOT_ALLOCATE",
                    call_index=call_index,
                    book_ordinal=book_ordinal,
                    book_name=book_name,
                    token_index=token_index,
                    token_id=token_id,
                    counters=counters,
                    expected=ACTIVE_COLUMNS,
                    observed=counters["allocated_segments"],
                    source_winners=source,
                )
            if winners in self.owner_tuples or winners in self.seed_tuples:
                return self._detail(
                    "NONINJECTIVE_OWNER_TUPLE",
                    call_index=call_index,
                    book_ordinal=book_ordinal,
                    book_name=book_name,
                    token_index=token_index,
                    token_id=token_id,
                    counters=counters,
                    expected="globally new winner tuple",
                    observed=list(winner_ids),
                    source_winners=source,
                )
            self.owners_by_key[key] = winners
            self.owner_tuples.add(winners)
            self.fanout_by_source[source] = source_fanout + 1
            self.key_owner_stream.update(key)
            self.key_owner_stream.update(winners)
        else:
            repeat_observed = {
                "allocated_segments": counters["allocated_segments"],
                "owners_equal": winners == known_owner,
            }
            if repeat_observed != {"allocated_segments": 0, "owners_equal": True}:
                return self._detail(
                    "REPEAT_DID_NOT_REUSE_STORED_OWNER",
                    call_index=call_index,
                    book_ordinal=book_ordinal,
                    book_name=book_name,
                    token_index=token_index,
                    token_id=token_id,
                    counters=counters,
                    expected={"allocated_segments": 0, "owners_equal": True},
                    observed={
                        **repeat_observed,
                        "expected_winner_cells": list(_WINNERS.unpack(known_owner)),
                        "observed_winner_cells": list(winner_ids),
                    },
                    source_winners=source,
                )
            self.repeats += 1

        self.previous_winners = winners
        self.transitions += 1
        self.accepted_calls += 1
        return None

    def measurement(self, wall_ns: int) -> dict[str, int | float | str]:
        return {
            "calls_issued": self.calls_issued,
            "accepted_calls": self.accepted_calls,
            "resets": self.resets,
            "transitions": self.transitions,
            "distinct_keys": len(self.owners_by_key),
            "repeats": self.repeats,
            "allocated_segments": self.allocated_segments,
            "punished_segments": self.punished_segments,
            "maximum_candidate_segments": self.maximum_candidate_segments,
            "maximum_matching_segments": self.maximum_matching_segments,
            "maximum_active_segments": self.maximum_active_segments,
            "maximum_postings_visited": self.maximum_postings_visited,
            "winner_stream_sha256": self.winner_stream.hexdigest(),
            "key_owner_stream_sha256": self.key_owner_stream.hexdigest(),
            "wall_ns": wall_ns,
            "calls_per_second": (
                0.0 if wall_ns == 0 else self.calls_issued * 1_000_000_000.0 / wall_ns
            ),
        }


def _bounded_definition() -> tuple[
    dict[str, Any],
    NativeTemporalCircuitDefinition,
    NativeTemporalCircuitDefinition,
]:
    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=True)
    production = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    bounded = NativeTemporalCircuitDefinition(
        replace(
            production.config,
            segment_capacity=(BOUNDARY_CALLS - 1) * ACTIVE_COLUMNS,
        ),
        production.token_codes,
        production.threshold_offsets_mv,
    )
    return protocol, production, bounded


def _run(
    circuit: NativeTemporalCircuit,
    tokens: np.ndarray,
    *,
    book_ordinal: int,
    book_name: str,
) -> tuple[OwnerReuseLedger, dict[str, object] | None, int]:
    ledger = OwnerReuseLedger()
    divergence: dict[str, object] | None = None
    started_ns = time.perf_counter_ns()
    for call_index, raw_token in enumerate(tokens):
        token_id = int(raw_token)
        try:
            emitted = circuit.advance_block(
                [token_id],
                mode=TemporalStepMode.LEARN,
                reset_before=call_index == 0,
            )
        except NativeTemporalRuntimeError as exc:
            divergence = {
                "kind": "NATIVE_STOP",
                "call_index": call_index,
                "book_ordinal": book_ordinal,
                "book_name": book_name,
                "token_index": call_index,
                "token_id": token_id,
                "status": exc.status.name,
                "poison_code": exc.poison_code,
                "message": str(exc),
            }
            break
        if len(emitted) != 1:
            divergence = {
                "kind": "ABI_STEP_WIDTH",
                "call_index": call_index,
                "book_ordinal": book_ordinal,
                "book_name": book_name,
                "token_index": call_index,
                "token_id": token_id,
                "expected": 1,
                "observed": len(emitted),
            }
            break
        divergence = ledger.observe(
            emitted[0],
            call_index=call_index,
            book_ordinal=book_ordinal,
            book_name=book_name,
            token_index=call_index,
            token_id=token_id,
            reset_before=call_index == 0,
        )
        if divergence is not None:
            break
    return ledger, divergence, time.perf_counter_ns() - started_ns


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite development artifact: {args.output}")

    input_started_ns = time.perf_counter_ns()
    verified = load_verified_train_input()
    first_book = next(verified.iter_books())
    if first_book.token_count < BOUNDARY_CALLS:
        raise RuntimeError("Genesis is shorter than the fixed owner-reuse boundary")
    prefix_bytes = first_book.token_bytes_le_i32[: BOUNDARY_CALLS * 4]
    tokens = np.frombuffer(prefix_bytes, dtype="<i4")
    if tokens.size != BOUNDARY_CALLS:
        raise RuntimeError("Genesis prefix length drifted")
    input_load_ns = time.perf_counter_ns() - input_started_ns

    protocol, production, bounded = _bounded_definition()
    with NativeTemporalCircuit.fresh(bounded) as circuit:
        ledger, divergence, wall_ns = _run(
            circuit,
            tokens,
            book_ordinal=first_book.ordinal,
            book_name=first_book.short_name,
        )

    if divergence is None and ledger.calls_issued != BOUNDARY_CALLS:
        divergence = {
            "kind": "FIXED_BOUNDARY_NOT_REACHED",
            "expected": BOUNDARY_CALLS,
            "observed": ledger.calls_issued,
        }
    if divergence is None and ledger.repeats == 0:
        divergence = {
            "kind": "OWNER_REUSE_NOT_EXERCISED_AT_FIXED_BOUNDARY",
            "expected": "at least one repeated temporal key",
            "observed": 0,
        }

    native_identity = loaded_library_identity()
    payload = {
        "artifact_id": "r3-owner-reuse-first-divergence-development-v1",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "PASS_OWNER_REUSE_THROUGH_FIXED_BOUNDARY"
            if divergence is None
            else "STOP_FIRST_FUNCTIONAL_DIVERGENCE"
        ),
        "claim_boundary": (
            "one fresh bounded TRAIN-only development replay of Genesis calls "
            "0..19,841 at theta-match 16/theta-active 18 and construction seed 17 "
            "using current native source; this is not a calibration-v3 outcome, "
            "retry, arm selection, or protocol reinterpretation and accesses no "
            "held-out book, C1 stream, confirmation continuation, or checkpoint; "
            "foreign/missing matching telemetry is recorded but is non-gating"
        ),
        "semantic_gate": {
            "candidate_segments": "diagnostic only; one-contact candidates are legitimate",
            "matching_segments": (
                "diagnostic only; first deviation from 11 times exact-source "
                "stored-key fanout is recorded without stopping"
            ),
            "repeat_owner_reuse": (
                "requires zero allocation and the stored winner tuple; prediction, "
                "active, candidate, and punishment counters are diagnostic"
            ),
        },
        "fixed_boundary": {
            "book": "Genesis",
            "first_call_index": 0,
            "last_call_index": BOUNDARY_CALLS - 1,
            "maximum_calls": BOUNDARY_CALLS,
            "automatic_expansion": False,
        },
        "train_input": {
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "genesis_ordinal": first_book.ordinal,
            "genesis_token_sha256_le_i32": first_book.token_sha256_le_i32,
            "prefix_token_count": int(tokens.size),
            "prefix_sha256_le_i32": hashlib.sha256(prefix_bytes).hexdigest(),
            "held_out_books_materialized": 0,
            "input_load_ns": input_load_ns,
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "protocol_sha256": protocol_authority.protocol_sha256(),
            "protocol_source_runnable": protocol["source_binding"]["runnable"],
            "production_circuit_identity_sha256": production.circuit_identity_sha256(),
            "bounded_circuit_identity_sha256": bounded.circuit_identity_sha256(),
            "production_segment_capacity": production.config.segment_capacity,
            "bounded_segment_capacity": bounded.config.segment_capacity,
        },
        "measurement": ledger.measurement(wall_ns),
        "first_matching_diagnostic": ledger.first_matching_diagnostic,
        "first_divergence": divergence,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as artifact:
        json.dump(payload, artifact, indent=2, sort_keys=True, allow_nan=False)
        artifact.write("\n")
    print(json.dumps({"artifact": str(args.output), "status": payload["status"]}))
    return 0 if divergence is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
