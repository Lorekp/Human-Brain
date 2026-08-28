"""Run one fail-fast TRAIN-order R3-F0 development pass.

This runner is deliberately narrower than the frozen result protocol.  It
authenticates the canonical TRAIN stream, executes one fresh physical LEARN
pass at arm (16, 18), seed 17, and checks only the section-5 identity and
recurrence invariants.  It never loads held-out data, enters C1, publishes a
checkpoint, or runs confirmation continuations.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
import gc
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
import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalRuntimeError,
    TemporalStepMode,
)


ACTIVE_COLUMNS = 11
ARM = (16, 18)
CONSTRUCTION_SEED = 17
EXPECTED_BOOKS = 49
EXPECTED_CALLS = 2_117_412
EXPECTED_TRANSITIONS = 2_117_363
EXPECTED_DISTINCT_KEYS = 2_116_946
EXPECTED_SEGMENTS = 23_286_406
EXPECTED_COHORTS = 1_286
EXPECTED_PYRAMIDAL_CELLS = 114_400
EXPECTED_FINAL_EVENT_COUNT = 37_538
OUTPUT_PATH = Path(__file__).with_name("r3_f0_train_structure_development.json")
_WINNERS = struct.Struct("<11I")
_DEVELOPMENT_PROTOCOL_IDENTITY = "51" * 32
_DEVELOPMENT_SOURCE_IDENTITY = "a7" * 32


class DevelopmentStop(RuntimeError):
    """First violated R3-F0 development invariant."""

    def __init__(self, kind: str, **detail: object) -> None:
        super().__init__(kind)
        self.detail = {"kind": kind, **detail}


@contextlib.contextmanager
def _synthetic_preoutcome_authority() -> Iterator[None]:
    """Bypass only the frozen protocol's intentional runnable hold."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(
        bytes.fromhex(_DEVELOPMENT_PROTOCOL_IDENTITY),
        bytes.fromhex(_DEVELOPMENT_SOURCE_IDENTITY),
    )
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"development artifact path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def _winner_bytes(step: object) -> bytes:
    spikes = tuple(getattr(step, "winner_spikes"))
    if len(spikes) != ACTIVE_COLUMNS:
        raise DevelopmentStop("WINNER_WIDTH", observed=len(spikes))
    return _WINNERS.pack(*(int(spike.cell_id) for spike in spikes))


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


def _run_pass(
    circuit: NativeTemporalCircuit,
    books: tuple[tuple[object, np.ndarray], ...],
) -> tuple[dict[str, Any], dict[str, object] | None]:
    owners_by_key: dict[bytes, bytes] = {}
    seen_owner_tuples: set[bytes] = set()
    fanout_by_source: dict[bytes, int] = {}
    winner_stream = hashlib.sha256()
    key_map = hashlib.sha256()
    calls = 0
    resets = 0
    transitions = 0
    repeats = 0
    allocated_segments = 0
    punished_segments = 0
    maximum_candidates = 0
    maximum_postings = 0
    stop: dict[str, object] | None = None

    gc_was_enabled = gc.isenabled()
    gc.disable()
    started_ns = time.perf_counter_ns()
    try:
        for book, tokens in books:
            previous_winners: bytes | None = None
            for token_index, raw_token in enumerate(tokens):
                token_id = int(raw_token)
                is_reset = token_index == 0
                if is_reset:
                    resets += 1
                key = (
                    None
                    if previous_winners is None
                    else struct.pack("<I", token_id) + previous_winners
                )
                known_owners = None if key is None else owners_by_key.get(key)
                expected_matching = (
                    0
                    if previous_winners is None
                    else ACTIVE_COLUMNS * fanout_by_source.get(previous_winners, 0)
                )
                try:
                    emitted = circuit.advance_block(
                        [token_id],
                        mode=TemporalStepMode.LEARN,
                        reset_before=is_reset,
                    )
                except NativeTemporalRuntimeError as exc:
                    raise DevelopmentStop(
                        "NATIVE_STOP",
                        book_ordinal=book.ordinal,
                        book_name=book.short_name,
                        token_index=token_index,
                        token_id=token_id,
                        status=exc.status.name,
                        poison_code=exc.poison_code,
                        message=str(exc),
                    ) from exc
                if len(emitted) != 1:
                    raise DevelopmentStop("ABI_STEP_WIDTH", observed=len(emitted))
                step = emitted[0]
                calls += 1
                winners = _winner_bytes(step)
                winner_stream.update(winners)
                allocated_segments += int(step.allocated_segments)
                punished_segments += int(step.punished_segments)
                maximum_candidates = max(maximum_candidates, int(step.candidate_segments))
                maximum_postings = max(maximum_postings, int(step.postings_visited))

                if is_reset:
                    if not (
                        step.event_count == 1
                        and step.candidate_segments == 0
                        and step.matching_segments == 0
                        and step.active_segments == 0
                        and step.allocated_segments == 0
                        and step.punished_segments == 0
                    ):
                        raise DevelopmentStop(
                            "BOOK_RESET",
                            book_ordinal=book.ordinal,
                            observed_event_count=step.event_count,
                            matching=step.matching_segments,
                            active=step.active_segments,
                            allocated=step.allocated_segments,
                        )
                    previous_winners = winners
                    continue

                transitions += 1
                assert key is not None and previous_winners is not None
                if int(step.matching_segments) != expected_matching:
                    raise DevelopmentStop(
                        "FOREIGN_OR_MISSING_SOURCE_MATCH",
                        book_ordinal=book.ordinal,
                        token_index=token_index,
                        token_id=token_id,
                        expected_matching=expected_matching,
                        observed_matching=int(step.matching_segments),
                    )

                if known_owners is None:
                    if step.allocated_segments != ACTIVE_COLUMNS or step.predicted_columns != 0:
                        raise DevelopmentStop(
                            "NOVEL_TRANSITION_DID_NOT_ALLOCATE",
                            book_ordinal=book.ordinal,
                            token_index=token_index,
                            token_id=token_id,
                            allocated=int(step.allocated_segments),
                            predicted_columns=int(step.predicted_columns),
                        )
                    if winners in seen_owner_tuples:
                        raise DevelopmentStop(
                            "NONINJECTIVE_WINNER_TUPLE",
                            book_ordinal=book.ordinal,
                            token_index=token_index,
                            token_id=token_id,
                        )
                    owners_by_key[key] = winners
                    seen_owner_tuples.add(winners)
                    fanout_by_source[previous_winners] = (
                        fanout_by_source.get(previous_winners, 0) + 1
                    )
                    key_map.update(key)
                    key_map.update(winners)
                else:
                    repeats += 1
                    if not (
                        step.allocated_segments == 0
                        and step.predicted_columns == ACTIVE_COLUMNS
                        and winners == known_owners
                    ):
                        raise DevelopmentStop(
                            "REPEAT_DID_NOT_REUSE_STORED_OWNERS",
                            book_ordinal=book.ordinal,
                            token_index=token_index,
                            token_id=token_id,
                            allocated=int(step.allocated_segments),
                            predicted_columns=int(step.predicted_columns),
                            owners_equal=winners == known_owners,
                        )
                previous_winners = winners

                if calls % 16_384 == 0:
                    print(
                        f"{calls:,}/{EXPECTED_CALLS:,} calls; "
                        f"{len(owners_by_key):,} distinct keys",
                        flush=True,
                    )
    except DevelopmentStop as exc:
        stop = exc.detail
    finally:
        wall_ns = time.perf_counter_ns() - started_ns
        if gc_was_enabled:
            gc.enable()

    observed = {
        "calls": calls,
        "resets": resets,
        "transitions": transitions,
        "distinct_keys": len(owners_by_key),
        "repeats": repeats,
        "allocated_segments": allocated_segments,
    }
    expected = {
        "calls": EXPECTED_CALLS,
        "resets": EXPECTED_BOOKS,
        "transitions": EXPECTED_TRANSITIONS,
        "distinct_keys": EXPECTED_DISTINCT_KEYS,
        "repeats": EXPECTED_TRANSITIONS - EXPECTED_DISTINCT_KEYS,
        "allocated_segments": EXPECTED_SEGMENTS,
    }
    if stop is None and observed != expected:
        stop = DevelopmentStop(
            "FINAL_PASS_COUNTS", expected=expected, observed=observed
        ).detail
    measurement = {
        **observed,
        "punished_segments": punished_segments,
        "maximum_candidate_segments": maximum_candidates,
        "maximum_postings_visited": maximum_postings,
        "winner_stream_sha256": winner_stream.hexdigest(),
        "key_owner_map_sha256": key_map.hexdigest(),
        "wall_ns": wall_ns,
        "calls_per_second": calls * 1_000_000_000.0 / wall_ns,
    }
    return measurement, stop


def main() -> int:
    args = _parse_args()
    loaded_started = time.perf_counter_ns()
    verified = load_verified_train_input()
    encoded_books = tuple(verified.iter_books())
    books = tuple(
        (book, np.frombuffer(book.token_bytes_le_i32, dtype="<i4"))
        for book in encoded_books
    )
    load_ns = time.perf_counter_ns() - loaded_started
    if len(books) != EXPECTED_BOOKS or sum(len(tokens) for _, tokens in books) != EXPECTED_CALLS:
        raise RuntimeError("canonical TRAIN authority counts drifted")

    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=False)
    if protocol["source_binding"]["runnable"] is not False:
        raise RuntimeError("development runner requires the pre-outcome runnable hold")
    definition = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    if definition.config.segment_capacity != EXPECTED_SEGMENTS:
        raise RuntimeError("R3-F0 segment capacity drifted")

    measurement: dict[str, Any] | None = None
    stop: dict[str, object] | None = None
    final_identity: object | None = None
    with _synthetic_preoutcome_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            measurement, stop = _run_pass(circuit, books)
            if stop is None:
                final_identity = circuit.checkpoint_identity()
                expected_identity_counts = (
                    EXPECTED_SEGMENTS,
                    EXPECTED_COHORTS,
                    EXPECTED_PYRAMIDAL_CELLS,
                    EXPECTED_FINAL_EVENT_COUNT,
                    EXPECTED_CALLS,
                    0,
                )
                observed_identity_counts = (
                    final_identity.segment_count,
                    final_identity.cohort_count,
                    final_identity.next_reserve_cell,
                    final_identity.event_count,
                    final_identity.learning_step,
                    final_identity.poison_code,
                )
                if observed_identity_counts != expected_identity_counts:
                    stop = DevelopmentStop(
                        "FINAL_CHECKPOINT_COUNTS",
                        expected=expected_identity_counts,
                        observed=observed_identity_counts,
                    ).detail
                expected_identity_hashes = {
                    "circuit_identity_sha256": definition.circuit_identity_sha256(),
                    "protocol_identity_sha256": _DEVELOPMENT_PROTOCOL_IDENTITY,
                    "source_bundle_identity_sha256": _DEVELOPMENT_SOURCE_IDENTITY,
                }
                observed_identity_hashes = {
                    "circuit_identity_sha256": final_identity.circuit_identity_sha256,
                    "protocol_identity_sha256": final_identity.protocol_identity_sha256,
                    "source_bundle_identity_sha256": (
                        final_identity.source_bundle_identity_sha256
                    ),
                }
                if stop is None and observed_identity_hashes != expected_identity_hashes:
                    stop = DevelopmentStop(
                        "FINAL_CHECKPOINT_IDENTITIES",
                        expected=expected_identity_hashes,
                        observed=observed_identity_hashes,
                    ).detail

    native_identity = loaded_library_identity()
    payload = {
        "artifact_id": "r3-f0-train-structure-development-v1",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_R3_F0_TRAIN_STRUCTURE_DEVELOPMENT" if stop is None else "STOP",
        "claim_boundary": (
            "one fresh TRAIN-only development pass at arm (16,18), seed17; "
            "synthetic pre-outcome authority; no held-out, C1, threshold selection, "
            "source refreeze, checkpoint publication, or confirmation evidence"
        ),
        "development_falsifier": {
            "strict_fanout_matching_equality": (
                "development-only: before each transition, matching_segments must "
                "equal 11 times the number of previously stored target keys for the "
                "exact W_prev; excess is a foreign-source match and deficit is an "
                "eroded/missing exact-source match"
            )
        },
        "train_input": {
            "book_count": len(books),
            "token_count": verified.token_count,
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "input_load_ns": load_ns,
            "held_out_books_materialized": 0,
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "protocol_sha256": protocol_authority.protocol_sha256(),
            "segment_capacity": definition.config.segment_capacity,
        },
        "measurement": measurement,
        "stop": stop,
        "final_checkpoint_identity": (
            None if final_identity is None else _identity_payload(final_identity)
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0 if stop is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
