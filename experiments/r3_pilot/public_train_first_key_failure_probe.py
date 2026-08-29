"""Public TRAIN-only first recurrent-key failure localization.

This development probe deliberately ignores the indexed recognizer's broad
candidate count.  It stops at the first transition where the native matching
or active rows disagree with the exact prior-winner-key ledger, then decodes
one post-step native checkpoint.  Rows appended by the failing step are
excluded, so the resulting segment ledger describes the recognition state
that entered that step.

The probe is diagnostic only: it does not select thresholds, retry calibration,
repair the runtime, materialize held-out data, or enter C1.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
from typing import Any, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    TemporalCircuitStep,
    TemporalStepMode,
)


ACTIVE_COLUMNS = 11
PAIRS_PER_SEGMENT = 10
ARM = (16, 18)
CONSTRUCTION_SEED = 17
DEFAULT_MAX_STEPS = 25_000
OUTPUT_PATH = Path(__file__).with_name(
    "public_train_first_key_failure_probe.json"
)
CHECKPOINT_MAGIC = b"R3NTCP1\x00"
CHECKPOINT_HEADER_BYTES = 256
NONE = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class AllocationOrigin:
    """One contiguous native append and the prior key that created it."""

    first_segment_id: int
    segment_count: int
    global_step_index: int
    book_ordinal: int
    book_zero_based_token_index: int
    target_token_id: int
    source_winners: tuple[int, ...]
    source_context: tuple[int, ...]

    @property
    def segment_stop(self) -> int:
        return self.first_segment_id + self.segment_count


@dataclass(frozen=True, slots=True)
class ParsedCheckpoint:
    """The canonical checkpoint-v1 sections needed by this probe."""

    body_sha256: str
    segment_count: int
    cohort_columns: np.ndarray
    owners: np.ndarray
    sources: np.ndarray
    permanence_ticks: np.ndarray


def _sha256_u32(values: Sequence[int]) -> str:
    payload = struct.pack(f"<{len(values)}I", *values)
    return hashlib.sha256(payload).hexdigest()


def parse_checkpoint_rows(checkpoint: bytes) -> ParsedCheckpoint:
    """Decode only the frozen checkpoint-v1 segment structure, read-only."""

    if type(checkpoint) is not bytes:
        raise TypeError("checkpoint must be immutable bytes")
    if len(checkpoint) < CHECKPOINT_HEADER_BYTES:
        raise ValueError("checkpoint is shorter than its frozen header")
    header = checkpoint[:CHECKPOINT_HEADER_BYTES]
    if header[:8] != CHECKPOINT_MAGIC:
        raise ValueError("checkpoint magic mismatch")
    if int.from_bytes(header[8:10], "little") != 1:
        raise ValueError("unsupported checkpoint version")
    if int.from_bytes(header[10:12], "little") != CHECKPOINT_HEADER_BYTES:
        raise ValueError("checkpoint header width mismatch")
    total_bytes = int.from_bytes(header[16:24], "little")
    body_bytes = int.from_bytes(header[24:32], "little")
    if total_bytes != len(checkpoint) or body_bytes != len(checkpoint) - 256:
        raise ValueError("checkpoint framing mismatch")
    body_digest = hashlib.sha256(checkpoint[CHECKPOINT_HEADER_BYTES:]).digest()
    if body_digest != header[128:160]:
        raise ValueError("checkpoint body digest mismatch")

    (
        segment_count,
        cohort_count,
        _next_reserve_cell,
        _event_count,
        _learning_step,
        prior_context_length,
        winner_context_length,
        _poison_code,
    ) = struct.unpack("<IIIQQBBH", header[192:224])
    expected_body = (
        cohort_count * 4
        + prior_context_length * 4
        + winner_context_length * 4
        + segment_count * (4 + PAIRS_PER_SEGMENT * 4 + PAIRS_PER_SEGMENT * 2)
    )
    if body_bytes != expected_body:
        raise ValueError("checkpoint scalar/body layout mismatch")

    offset = CHECKPOINT_HEADER_BYTES
    cohort_columns = np.frombuffer(
        checkpoint, dtype="<u4", count=cohort_count, offset=offset
    )
    offset += cohort_count * 4
    offset += prior_context_length * 4
    offset += winner_context_length * 4
    owners = np.frombuffer(
        checkpoint, dtype="<u4", count=segment_count, offset=offset
    )
    offset += segment_count * 4
    sources = np.frombuffer(
        checkpoint,
        dtype="<u4",
        count=segment_count * PAIRS_PER_SEGMENT,
        offset=offset,
    ).reshape(segment_count, PAIRS_PER_SEGMENT)
    offset += segment_count * PAIRS_PER_SEGMENT * 4
    permanence_ticks = np.frombuffer(
        checkpoint,
        dtype="<u2",
        count=segment_count * PAIRS_PER_SEGMENT,
        offset=offset,
    ).reshape(segment_count, PAIRS_PER_SEGMENT)
    if offset + segment_count * PAIRS_PER_SEGMENT * 2 != len(checkpoint):
        raise ValueError("checkpoint section cursor mismatch")
    return ParsedCheckpoint(
        body_sha256=body_digest.hex(),
        segment_count=segment_count,
        cohort_columns=cohort_columns,
        owners=owners,
        sources=sources,
        permanence_ticks=permanence_ticks,
    )


def _owner_column(
    owner: int,
    *,
    cohort_columns: np.ndarray,
    column_count: int,
    base_cells_per_column: int,
    novel_race_cap: int,
) -> int:
    base_total = column_count * base_cells_per_column
    if owner < base_total:
        return owner // base_cells_per_column
    cohort_index = (owner - base_total) // novel_race_cap
    if not 0 <= cohort_index < len(cohort_columns):
        raise ValueError("segment owner is outside checkpoint column membership")
    return int(cohort_columns[cohort_index])


def _origin_for_segment(
    segment_id: int, origins: Sequence[AllocationOrigin]
) -> AllocationOrigin | None:
    # Append batches are ordered and non-overlapping.  The first failure is
    # early enough that a reverse scan is both simpler and negligible.
    for origin in reversed(origins):
        if origin.first_segment_id <= segment_id < origin.segment_stop:
            return origin
        if segment_id >= origin.segment_stop:
            break
    return None


def localize_lifecycle(
    *,
    checkpoint: bytes,
    recognition_segment_limit: int,
    origins: Sequence[AllocationOrigin],
    current_source_context: Sequence[int],
    current_token_columns: Sequence[int],
    column_count: int,
    base_cells_per_column: int,
    novel_race_cap: int,
    connected_permanence_ticks: int,
    initial_permanence_ticks: int,
    theta_match: int,
    theta_active: int,
    observed_matching_segments: int,
    observed_active_segments: int,
    current_step_punished_segments: int,
) -> dict[str, Any]:
    """Classify the first mismatch from native rows and append provenance."""

    parsed = parse_checkpoint_rows(checkpoint)
    if not 0 <= recognition_segment_limit <= parsed.segment_count:
        raise ValueError("recognition segment limit is outside the checkpoint")
    context = tuple(int(value) for value in current_source_context)
    if len(context) != ACTIVE_COLUMNS or len(set(context)) != ACTIVE_COLUMNS:
        raise ValueError("current source context must contain 11 unique cells")
    columns = tuple(int(value) for value in current_token_columns)
    if len(columns) != ACTIVE_COLUMNS or len(set(columns)) != ACTIVE_COLUMNS:
        raise ValueError("current token columns must contain 11 unique columns")

    sources = parsed.sources[:recognition_segment_limit]
    ticks = parsed.permanence_ticks[:recognition_segment_limit]
    live = sources != NONE
    overlap = live & np.isin(sources, np.asarray(context, dtype=np.uint32))
    potential_contacts = overlap.sum(axis=1, dtype=np.uint8) * 2
    connected_contacts = (
        (overlap & (ticks >= connected_permanence_ticks)).sum(
            axis=1, dtype=np.uint8
        )
        * 2
    )
    matching_ids = np.flatnonzero(potential_contacts >= theta_match)
    active_ids = np.flatnonzero(connected_contacts >= theta_active)

    current_context_set = frozenset(context)
    current_column_set = frozenset(columns)
    exact_origin_ids: set[int] = set()
    for origin in origins:
        if frozenset(origin.source_context) == current_context_set:
            exact_origin_ids.update(
                range(
                    origin.first_segment_id,
                    min(origin.segment_stop, recognition_segment_limit),
                )
            )
    relevant_ids = sorted(set(map(int, matching_ids)) | exact_origin_ids)

    grouped: dict[tuple[object, ...], dict[str, Any]] = {}
    foreign_matching = 0
    exact_deactivated = 0
    punishment_marked_exact = 0
    missing_origins = 0
    for segment_id in relevant_ids:
        origin = _origin_for_segment(segment_id, origins)
        if origin is None:
            missing_origins += 1
            relation = "UNKNOWN_ORIGIN"
            origin_hash = None
            origin_overlap = None
            origin_step = None
        else:
            origin_set = frozenset(origin.source_context)
            relation = (
                "EXACT_CURRENT_SOURCE"
                if origin_set == current_context_set
                else "DIVERGED_SOURCE"
            )
            origin_hash = _sha256_u32(origin.source_context)
            origin_overlap = len(origin_set & current_context_set)
            origin_step = origin.global_step_index

        potential = int(potential_contacts[segment_id])
        connected = int(connected_contacts[segment_id])
        is_matching = potential >= theta_match
        is_active = connected >= theta_active
        row_sources = tuple(
            int(value) for value in sources[segment_id] if int(value) != NONE
        )
        row_ticks = tuple(int(value) for value in ticks[segment_id][: len(row_sources)])
        owner = int(parsed.owners[segment_id])
        owner_column = _owner_column(
            owner,
            cohort_columns=parsed.cohort_columns,
            column_count=column_count,
            base_cells_per_column=base_cells_per_column,
            novel_race_cap=novel_race_cap,
        )
        below_connected = sum(
            tick < connected_permanence_ticks
            for source, tick in zip(row_sources, row_ticks, strict=True)
            if source in current_context_set
        )
        if relation == "DIVERGED_SOURCE" and is_matching:
            foreign_matching += 1
        if relation == "EXACT_CURRENT_SOURCE" and is_matching and not is_active:
            exact_deactivated += 1
            if below_connected and initial_permanence_ticks >= connected_permanence_ticks:
                punishment_marked_exact += 1

        key = (
            relation,
            origin_hash,
            origin_step,
            origin_overlap,
            row_sources,
            row_ticks,
            potential,
            connected,
            is_matching,
            is_active,
            owner_column in current_column_set,
        )
        group = grouped.setdefault(
            key,
            {
                "source_relation": relation,
                "origin_source_context_sha256": origin_hash,
                "origin_global_step_index": origin_step,
                "origin_current_source_overlap_cells": origin_overlap,
                "row_sources": list(row_sources),
                "permanence_ticks": list(row_ticks),
                "potential_contacts": potential,
                "connected_contacts": connected,
                "matching": is_matching,
                "active": is_active,
                "below_connected_overlap_pairs": below_connected,
                "owners_in_current_token_columns": owner_column in current_column_set,
                "segment_ids": [],
                "owner_cells": [],
                "owner_columns": [],
            },
        )
        group["segment_ids"].append(segment_id)
        group["owner_cells"].append(owner)
        group["owner_columns"].append(owner_column)

    calculated_matching = int(len(matching_ids))
    calculated_active = int(len(active_ids))
    counters_equal = (
        calculated_matching == observed_matching_segments
        and calculated_active == observed_active_segments
    )
    if not counters_equal and current_step_punished_segments:
        classification = "POST_STEP_MUTATION_AMBIGUITY"
    elif foreign_matching and punishment_marked_exact:
        classification = "MIXED_WINNER_DIVERGENCE_AND_PUNISHMENT_DEACTIVATION"
    elif foreign_matching:
        classification = "WINNER_KEY_DIVERGENCE"
    elif punishment_marked_exact:
        classification = "PUNISHMENT_INDUCED_DEACTIVATION"
    elif counters_equal:
        classification = "SCORER_KEY_LEDGER_DIVERGENCE"
    else:
        classification = "UNRESOLVED_POST_STEP_ROW_DRIFT"

    return {
        "classification": classification,
        "checkpoint_body_sha256": parsed.body_sha256,
        "checkpoint_segment_count_after_step": parsed.segment_count,
        "recognition_segment_limit_before_step": recognition_segment_limit,
        "current_source_context": list(context),
        "current_source_context_sha256": _sha256_u32(context),
        "current_token_columns": list(columns),
        "observed_matching_segments": observed_matching_segments,
        "observed_active_segments": observed_active_segments,
        "calculated_post_step_matching_preexisting_rows": calculated_matching,
        "calculated_post_step_active_preexisting_rows": calculated_active,
        "checkpoint_rows_equal_step_counters": counters_equal,
        "current_step_punished_segments": current_step_punished_segments,
        "foreign_matching_segments": foreign_matching,
        "exact_source_deactivated_segments": exact_deactivated,
        "exact_source_rows_below_connected_after_noninitial_ticks": (
            punishment_marked_exact
        ),
        "segments_without_allocation_origin": missing_origins,
        "segment_groups": list(grouped.values()),
    }


@contextlib.contextmanager
def _synthetic_preoutcome_authority() -> Iterator[None]:
    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(b"\x51" * 32, b"\xa7" * 32)
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _winner_cells(step: TemporalCircuitStep) -> tuple[int, ...]:
    winners = tuple(spike.cell_id for spike in step.winner_spikes)
    if len(winners) != ACTIVE_COLUMNS or len(set(winners)) != ACTIVE_COLUMNS:
        raise RuntimeError("native step did not expose 11 unique winners")
    return winners


def _run_probe(
    definition: NativeTemporalCircuitDefinition,
    tokens: np.ndarray,
    *,
    book: object,
) -> tuple[dict[str, Any], object]:
    token_columns = dict(definition.token_codes)
    key_owners: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}
    fanout_by_source: dict[tuple[int, ...], int] = {}
    origins: list[AllocationOrigin] = []
    previous_winners: tuple[int, ...] | None = None
    previous_context: tuple[int, ...] | None = None
    live_segments = 0

    with _synthetic_preoutcome_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for token_index, raw_token in enumerate(tokens):
                token_id = int(raw_token)
                recognition_limit = live_segments
                step = circuit.advance_block(
                    [token_id],
                    mode=TemporalStepMode.LEARN,
                    reset_before=token_index == 0,
                )[0]
                winners = _winner_cells(step)
                if token_index == 0:
                    previous_winners = winners
                    previous_context = tuple(step.context[:ACTIVE_COLUMNS])
                    continue

                assert previous_winners is not None and previous_context is not None
                expected_recognition = (
                    ACTIVE_COLUMNS * fanout_by_source.get(previous_winners, 0)
                )
                recognition_mismatch = (
                    int(step.matching_segments), int(step.active_segments)
                ) != (expected_recognition, expected_recognition)
                key = (previous_winners, token_id)
                known_owners = key_owners.get(key)
                if known_owners is None:
                    outcome_mismatch = (
                        int(step.allocated_segments),
                        int(step.predicted_columns),
                        int(step.predicted),
                    ) != (ACTIVE_COLUMNS, 0, 0)
                    outcome_expectation = [ACTIVE_COLUMNS, 0, 0]
                else:
                    outcome_mismatch = (
                        int(step.allocated_segments),
                        int(step.predicted_columns),
                        int(step.predicted),
                        int(winners == known_owners),
                    ) != (0, ACTIVE_COLUMNS, 1, 1)
                    outcome_expectation = [0, ACTIVE_COLUMNS, 1, 1]

                if recognition_mismatch or outcome_mismatch:
                    checkpoint = circuit.checkpoint()
                    lifecycle = localize_lifecycle(
                        checkpoint=checkpoint,
                        recognition_segment_limit=recognition_limit,
                        origins=origins,
                        current_source_context=previous_context,
                        current_token_columns=token_columns[token_id],
                        column_count=definition.config.column_count,
                        base_cells_per_column=definition.config.base_cells_per_column,
                        novel_race_cap=definition.config.novel_race_cap,
                        connected_permanence_ticks=(
                            definition.config.connected_permanence_ticks
                        ),
                        initial_permanence_ticks=(
                            definition.config.initial_permanence_ticks
                        ),
                        theta_match=definition.config.theta_match,
                        theta_active=definition.config.theta_active,
                        observed_matching_segments=int(step.matching_segments),
                        observed_active_segments=int(step.active_segments),
                        current_step_punished_segments=int(step.punished_segments),
                    )
                    return (
                        {
                            "kind": "FIRST_RECURRENT_KEY_MISMATCH",
                            "book_ordinal": int(book.ordinal),
                            "book_short_name": str(book.short_name),
                            "book_zero_based_token_index": token_index,
                            "global_zero_based_step_index": token_index,
                            "token_id": token_id,
                            "previous_winners": list(previous_winners),
                            "previous_winner_key_sha256": _sha256_u32(
                                previous_winners
                            ),
                            "expected_source_fanout": fanout_by_source.get(
                                previous_winners, 0
                            ),
                            "expected_matching_segments": expected_recognition,
                            "expected_active_segments": expected_recognition,
                            "observed": {
                                "candidate_segments": int(step.candidate_segments),
                                "matching_segments": int(step.matching_segments),
                                "active_segments": int(step.active_segments),
                                "allocated_segments": int(step.allocated_segments),
                                "predicted_columns": int(step.predicted_columns),
                                "predicted": bool(step.predicted),
                                "punished_segments": int(step.punished_segments),
                                "postings_visited": int(step.postings_visited),
                            },
                            "recognition_mismatch": recognition_mismatch,
                            "key_was_known": known_owners is not None,
                            "outcome_mismatch": outcome_mismatch,
                            "outcome_expectation": outcome_expectation,
                            "lifecycle": lifecycle,
                        },
                        circuit.checkpoint_identity(),
                    )

                if step.allocated_segments:
                    origins.append(
                        AllocationOrigin(
                            first_segment_id=live_segments,
                            segment_count=int(step.allocated_segments),
                            global_step_index=token_index,
                            book_ordinal=int(book.ordinal),
                            book_zero_based_token_index=token_index,
                            target_token_id=token_id,
                            source_winners=previous_winners,
                            source_context=previous_context,
                        )
                    )
                live_segments += int(step.allocated_segments)
                if known_owners is None:
                    key_owners[key] = winners
                    fanout_by_source[previous_winners] = (
                        fanout_by_source.get(previous_winners, 0) + 1
                    )
                previous_winners = winners
                previous_context = tuple(step.context[:ACTIVE_COLUMNS])

            return (
                {
                    "kind": "NO_MISMATCH_WITHIN_PUBLIC_PREFIX",
                    "completed_steps": len(tokens),
                    "live_segments": live_segments,
                },
                circuit.checkpoint_identity(),
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if not 2 <= args.max_steps <= DEFAULT_MAX_STEPS:
        parser.error(f"max-steps must be in [2, {DEFAULT_MAX_STEPS}]")
    if args.output.exists():
        parser.error("output already exists; this first-result probe refuses overwrite")
    return args


def main() -> int:
    args = _parse_args()
    verified = load_verified_train_input()
    books = tuple(verified.iter_books())
    if len(books) != 49:
        raise RuntimeError("canonical TRAIN book count drifted")
    book = books[0]
    all_tokens = np.frombuffer(book.token_bytes_le_i32, dtype="<i4")
    if len(all_tokens) < args.max_steps:
        raise RuntimeError("first public TRAIN book is shorter than the probe bound")
    tokens = all_tokens[: args.max_steps]

    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=False)
    production = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    definition = NativeTemporalCircuitDefinition(
        replace(
            production.config,
            segment_capacity=(args.max_steps - 1) * ACTIVE_COLUMNS,
        ),
        production.token_codes,
        production.threshold_offsets_mv,
    )
    stop, identity = _run_probe(definition, tokens, book=book)
    native_identity = loaded_library_identity()
    payload = {
        "artifact_id": "public-train-first-recurrent-key-failure-v1",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "BANKED_FIRST_RECURRENT_KEY_MISMATCH"
            if stop["kind"] == "FIRST_RECURRENT_KEY_MISMATCH"
            else "NO_MISMATCH_WITHIN_PUBLIC_PREFIX"
        ),
        "claim_boundary": (
            "one public TRAIN-only Genesis prefix at frozen arm (16,18), seed 17; "
            "first exact recurrent-key ledger mismatch only; one post-step native "
            "checkpoint; no threshold selection/retry, repair, held-out, confirmation, "
            "C1, capability, or transfer evidence"
        ),
        "probe_bound_steps": args.max_steps,
        "train_input": {
            "book_ordinal": int(book.ordinal),
            "book_short_name": str(book.short_name),
            "book_token_sha256_le_i32": str(book.token_sha256_le_i32),
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "held_out_books_materialized": 0,
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "bounded_segment_capacity": definition.config.segment_capacity,
            "production_circuit_identity_sha256": (
                production.circuit_identity_sha256()
            ),
            "bounded_circuit_identity_sha256": definition.circuit_identity_sha256(),
            "protocol_sha256": protocol_authority.protocol_sha256(),
        },
        "result": stop,
        "checkpoint_identity_after_first_result": {
            "segment_count": identity.segment_count,
            "cohort_count": identity.cohort_count,
            "next_reserve_cell": identity.next_reserve_cell,
            "event_count": identity.event_count,
            "learning_step": identity.learning_step,
            "logical_state_sha256": identity.logical_state_sha256,
        },
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
    artifact_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}")
    print(f"artifact_sha256={artifact_sha256}")
    print(
        "classification="
        + str(stop.get("lifecycle", {}).get("classification", stop["kind"]))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
