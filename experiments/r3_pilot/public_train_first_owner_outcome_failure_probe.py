"""One-shot public TRAIN replay to localize the first owner-outcome failure.

The dynamic ledger checks only observable owner behavior.  Global temporal
candidate/matching/active counters remain diagnostics and are constrained only
by their native monotone ordering.  The replay uses the fixed arm (16,18), seed
17, all canonical TRAIN books in order, and stops immediately after the first
novel/repeated owner-outcome violation.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot import calibration_v3_train_scoring as train_scoring
from experiments.r3_pilot import public_train_first_key_failure_probe as lifecycle_probe
from experiments.r3_pilot.calibration_v3_definitions import build_r3_f0_definition
from experiments.r3_pilot.r3_f0_train_input import load_verified_train_input
from humanbrain.native import loaded_library_identity
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
CONSTRUCTION_SEED = 17
EXPECTED = train_scoring.CALIBRATION_V3_R3_F0_EXPECTATIONS
EXPECTED_WINNER_ROWS = EXPECTED.expected_distinct_keys + EXPECTED.expected_book_resets
SPEND_MARKER_PATH = Path(__file__).with_suffix(".spend.json")
OUTPUT_PATH = Path(__file__).with_suffix(".json")
MAX_CHECKPOINT_COPY_BYTES = 256 * 1024 * 1024
DIAGNOSTIC_COUNTER_CONTRACT = (
    "0 <= active <= matching <= candidate <= postings; no exact global-count test"
)
OWNER_FAILURE_KINDS = frozenset(
    {"NOVEL_KEY_OWNER_OUTCOME", "REPEATED_KEY_OWNER_OUTCOME"}
)
CLAIM_BOUNDARY = (
    "one spent public TRAIN-only owner-outcome replay at arm (16,18), seed 17, "
    "over the canonical 49-book stream in frozen order, stopping immediately on "
    "the first actual owner-outcome/native/diagnostic-contract failure or otherwise "
    "finishing TRAIN; global recognition counts are diagnostics only; no "
    "calibration-v3 reinterpretation/retry, threshold change, confirmation, held-out, "
    "C1, capability, or transfer evidence"
)


def _canonical_lf_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class OwnerOutcomeLedger:
    """Compact exact ``(previous winners, token) -> owner winners`` ledger."""

    __slots__ = (
        "calls",
        "key_book_ordinals",
        "key_count",
        "key_global_steps",
        "key_index",
        "key_owner_tags",
        "key_source_tags",
        "key_token_indices",
        "key_tokens",
        "local_event",
        "previous_source_tag",
        "previous_winners",
        "repeats",
        "resets",
        "winner_count",
        "winner_index",
        "winner_rows",
    )

    def __init__(self, *, expected_keys: int, expected_winner_rows: int) -> None:
        if expected_keys < 1 or expected_winner_rows < expected_keys:
            raise ValueError("ledger capacities are inconsistent")
        self.winner_rows = np.empty(
            (expected_winner_rows, ACTIVE_COLUMNS), dtype=np.uint32
        )
        self.winner_index = train_scoring._OwnerRowIndex(
            self.winner_rows, expected_winner_rows
        )
        self.key_index = train_scoring._KeyIndex(expected_keys)
        self.key_source_tags = np.empty(expected_keys, dtype=np.uint32)
        self.key_owner_tags = np.empty(expected_keys, dtype=np.uint32)
        self.key_tokens = np.empty(expected_keys, dtype=np.int32)
        self.key_global_steps = np.empty(expected_keys, dtype=np.uint32)
        self.key_book_ordinals = np.empty(expected_keys, dtype=np.uint8)
        self.key_token_indices = np.empty(expected_keys, dtype=np.uint32)
        self.winner_count = 0
        self.key_count = 0
        self.repeats = 0
        self.resets = 0
        self.calls = 0
        self.local_event = 0
        self.previous_source_tag: int | None = None
        self.previous_winners: tuple[int, ...] | None = None

    @staticmethod
    def _winners(step: object) -> tuple[int, ...]:
        try:
            winners = tuple(int(value.cell_id) for value in step.winner_spikes)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("winner_spikes") from error
        if (
            len(winners) != ACTIVE_COLUMNS
            or len(set(winners)) != ACTIVE_COLUMNS
            or any(not 0 <= value <= 0xFFFF_FFFF for value in winners)
        ):
            raise ValueError("winner_spikes")
        return winners

    @staticmethod
    def _counter(step: object, label: str) -> int:
        value = getattr(step, label)
        if type(value) is not int or value < 0:
            raise ValueError(label)
        return value

    def _intern_winners(self, winners: tuple[int, ...]) -> int:
        existing = self.winner_index.lookup(winners)
        if existing is not None:
            return existing
        if self.winner_count >= len(self.winner_rows):
            raise RuntimeError("winner tuple ledger capacity exhausted")
        row_index = self.winner_count
        self.winner_rows[row_index] = winners
        self.winner_index.insert(winners, row_index)
        self.winner_count += 1
        return row_index

    def lookup_winner(self, winners: Sequence[int]) -> int | None:
        if len(winners) != ACTIVE_COLUMNS:
            return None
        return self.winner_index.lookup(winners)

    def winner_tuple(self, tag: int) -> tuple[int, ...]:
        if not 0 <= tag < self.winner_count:
            raise IndexError("winner tag outside committed ledger")
        return tuple(int(value) for value in self.winner_rows[tag])

    def key_record(self, record: int) -> dict[str, Any]:
        if not 0 <= record < self.key_count:
            raise IndexError("key record outside committed ledger")
        source_tag = int(self.key_source_tags[record])
        owner_tag = int(self.key_owner_tags[record])
        return {
            "key_record_index": record,
            "source_tag": source_tag,
            "source_winners": list(self.winner_tuple(source_tag)),
            "token_id": int(self.key_tokens[record]),
            "owner_tag": owner_tag,
            "owner_cells": list(self.winner_tuple(owner_tag)),
            "origin_global_step_index": int(self.key_global_steps[record]),
            "origin_book_ordinal": int(self.key_book_ordinals[record]),
            "origin_book_zero_based_token_index": int(
                self.key_token_indices[record]
            ),
            "first_segment_id": record * ACTIVE_COLUMNS,
            "segment_ids": list(
                range(record * ACTIVE_COLUMNS, (record + 1) * ACTIVE_COLUMNS)
            ),
        }

    @property
    def storage_bytes(self) -> int:
        arrays = (
            self.winner_rows,
            self.key_source_tags,
            self.key_owner_tags,
            self.key_tokens,
            self.key_global_steps,
            self.key_book_ordinals,
            self.key_token_indices,
            self.winner_index._hashes,
            self.winner_index._row_plus_one,
            self.key_index._keys,
            self.key_index._owner_plus_one,
        )
        return sum(int(array.nbytes) for array in arrays)

    @staticmethod
    def _failure(
        kind: str,
        *,
        book_ordinal: int,
        book_token_index: int,
        global_step_index: int,
        token_id: int | None,
        expected: Sequence[object] | str,
        observed: Sequence[object] | str,
        diagnostic: dict[str, int] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        failure: dict[str, Any] = {
            "kind": kind,
            "book_ordinal": book_ordinal,
            "book_zero_based_token_index": book_token_index,
            "global_zero_based_step_index": global_step_index,
            "token_id": token_id,
            "expected": list(expected) if not isinstance(expected, str) else expected,
            "observed": list(observed) if not isinstance(observed, str) else observed,
        }
        if diagnostic is not None:
            failure["diagnostic_counters"] = diagnostic
        failure.update(extra)
        return failure

    def observe(
        self,
        *,
        reset_before: bool,
        step: object,
        book_ordinal: int,
        book_token_index: int,
        global_step_index: int,
    ) -> dict[str, Any]:
        """Observe one committed native step and mutate only after a passing check."""

        try:
            token_id = self._counter(step, "token_id")
            event_count = self._counter(step, "event_count")
            candidate = self._counter(step, "candidate_segments")
            matching = self._counter(step, "matching_segments")
            active = self._counter(step, "active_segments")
            postings = self._counter(step, "postings_visited")
            allocated = self._counter(step, "allocated_segments")
            predicted_columns = self._counter(step, "predicted_columns")
            punished = self._counter(step, "punished_segments")
            predicted = step.predicted
            if type(predicted) is not bool:
                raise ValueError("predicted")
            winners = self._winners(step)
        except (AttributeError, ValueError) as error:
            return {
                "violation": self._failure(
                    "STEP_SHAPE_CONTRACT",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=None,
                    expected="non-negative native counters, boolean prediction, 11 unique winners",
                    observed=str(error),
                ),
                "key_was_known": None,
            }
        diagnostic = {
            "candidate_segments": candidate,
            "matching_segments": matching,
            "active_segments": active,
            "postings_visited": postings,
            "allocated_segments": allocated,
            "predicted_columns": predicted_columns,
            "predicted": int(predicted),
            "punished_segments": punished,
        }
        if (
            active > matching
            or matching > candidate
            or candidate > postings
            or allocated > ACTIVE_COLUMNS
            or predicted_columns > ACTIVE_COLUMNS
        ):
            return {
                "violation": self._failure(
                    "DIAGNOSTIC_COUNTER_CONTRACT",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=token_id,
                    expected=DIAGNOSTIC_COUNTER_CONTRACT,
                    observed=[active, matching, candidate, postings],
                    diagnostic=diagnostic,
                ),
                "key_was_known": None,
            }

        if reset_before:
            reset_observed = (
                event_count,
                allocated,
                predicted_columns,
                int(predicted),
            )
            if book_token_index != 0 or reset_observed != (1, 0, 0, 0):
                return {
                    "violation": self._failure(
                        "BOOK_RESET_OWNER_OUTCOME",
                        book_ordinal=book_ordinal,
                        book_token_index=book_token_index,
                        global_step_index=global_step_index,
                        token_id=token_id,
                        expected=[1, 0, 0, 0],
                        observed=reset_observed,
                        diagnostic=diagnostic,
                    ),
                    "key_was_known": None,
                }
            current_tag = self._intern_winners(winners)
            self.previous_source_tag = current_tag
            self.previous_winners = winners
            self.local_event = 1
            self.resets += 1
            self.calls += 1
            return {
                "violation": None,
                "key_was_known": None,
                "source_winners": None,
                "observed_winners": list(winners),
                "diagnostic_counters": diagnostic,
            }

        if self.previous_source_tag is None or self.previous_winners is None:
            return {
                "violation": self._failure(
                    "MISSING_BOOK_RESET",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=token_id,
                    expected="reset context seed",
                    observed="transition without prior winners",
                    diagnostic=diagnostic,
                ),
                "key_was_known": None,
            }
        if event_count != self.local_event + 1 or book_token_index != self.local_event:
            return {
                "violation": self._failure(
                    "LOCAL_EVENT_COUNT",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=token_id,
                    expected=[self.local_event + 1, self.local_event],
                    observed=[event_count, book_token_index],
                    diagnostic=diagnostic,
                ),
                "key_was_known": None,
            }

        source_tag = self.previous_source_tag
        source_winners = self.previous_winners
        key_record = self.key_index.lookup(source_tag, token_id)
        if key_record is None:
            owner_observed = (allocated, predicted_columns, int(predicted))
            if owner_observed != (ACTIVE_COLUMNS, 0, 0):
                return {
                    "violation": self._failure(
                        "NOVEL_KEY_OWNER_OUTCOME",
                        book_ordinal=book_ordinal,
                        book_token_index=book_token_index,
                        global_step_index=global_step_index,
                        token_id=token_id,
                        expected=[ACTIVE_COLUMNS, 0, 0],
                        observed=owner_observed,
                        diagnostic=diagnostic,
                        source_winners=list(source_winners),
                        observed_winners=list(winners),
                        winner_identity_diverged=False,
                    ),
                    "key_was_known": False,
                    "key_record_index": None,
                    "source_winners": list(source_winners),
                    "observed_winners": list(winners),
                    "diagnostic_counters": diagnostic,
                }
            if self.key_count >= len(self.key_tokens):
                raise RuntimeError("owner key ledger capacity exhausted")
            owner_tag = self._intern_winners(winners)
            record = self.key_count
            self.key_source_tags[record] = source_tag
            self.key_owner_tags[record] = owner_tag
            self.key_tokens[record] = token_id
            self.key_global_steps[record] = global_step_index
            self.key_book_ordinals[record] = book_ordinal
            self.key_token_indices[record] = book_token_index
            self.key_index.insert(source_tag, token_id, record)
            self.key_count += 1
            current_tag = owner_tag
            key_was_known = False
        else:
            owner_tag = int(self.key_owner_tags[key_record])
            expected_owners = self.winner_tuple(owner_tag)
            owner_observed = (
                allocated,
                predicted_columns,
                int(predicted),
                int(winners == expected_owners),
            )
            if owner_observed != (0, ACTIVE_COLUMNS, 1, 1):
                return {
                    "violation": self._failure(
                        "REPEATED_KEY_OWNER_OUTCOME",
                        book_ordinal=book_ordinal,
                        book_token_index=book_token_index,
                        global_step_index=global_step_index,
                        token_id=token_id,
                        expected=[0, ACTIVE_COLUMNS, 1, 1],
                        observed=owner_observed,
                        diagnostic=diagnostic,
                        source_winners=list(source_winners),
                        expected_owner_cells=list(expected_owners),
                        observed_winners=list(winners),
                        expected_key_record_index=key_record,
                        winner_identity_diverged=winners != expected_owners,
                    ),
                    "key_was_known": True,
                    "key_record_index": key_record,
                    "source_winners": list(source_winners),
                    "expected_owner_cells": list(expected_owners),
                    "observed_winners": list(winners),
                    "diagnostic_counters": diagnostic,
                }
            current_tag = owner_tag
            key_was_known = True
            self.repeats += 1

        self.previous_source_tag = current_tag
        self.previous_winners = winners
        self.local_event += 1
        self.calls += 1
        return {
            "violation": None,
            "key_was_known": key_was_known,
            "key_record_index": key_record,
            "source_winners": list(source_winners),
            "observed_winners": list(winners),
            "diagnostic_counters": diagnostic,
        }


def create_spend_marker(
    *,
    marker_path: Path = SPEND_MARKER_PATH,
    output_path: Path = OUTPUT_PATH,
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Exclusively create and fsync the durable marker before circuit creation."""

    if marker_path.exists() or output_path.exists():
        raise RuntimeError("refusing spent replay; spend marker or result artifact exists")
    marker_payload = {
        "schema_version": 1,
        "experiment": "public-train-first-owner-outcome-failure-v1",
        "state": "OWNER_OUTCOME_REPLAY_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arm": list(ARM),
        "construction_seed": CONSTRUCTION_SEED,
        "claim_boundary": CLAIM_BOUNDARY,
        "canonical_artifact_path": _display_path(output_path),
        "probe_source_sha256": _sha256_file(Path(__file__)),
        "authority": authority,
    }
    serialized = json.dumps(
        marker_payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        with marker_path.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing spent replay; spend marker exists") from error
    if output_path.exists():
        raise RuntimeError(
            "result artifact appeared while creating spend marker; marker retained"
        )
    return {
        "path": _display_path(marker_path),
        "sha256": _sha256_file(marker_path),
        "sha256_canonical_lf": _canonical_lf_sha256(marker_path.read_bytes()),
        "state": marker_payload["state"],
        "exclusive_create": True,
        "flushed_and_fsynced_before_run": True,
        "probe_source_sha256": marker_payload["probe_source_sha256"],
    }


def _sha256_u32(values: Sequence[int]) -> str:
    return lifecycle_probe._sha256_u32(values)


def _ledger_origin_relation(
    ledger: OwnerOutcomeLedger,
    segment_id: int,
    current_source: tuple[int, ...],
) -> tuple[int, dict[str, Any], str]:
    record_index = segment_id // ACTIVE_COLUMNS
    record = ledger.key_record(record_index)
    relation = (
        "EXACT_CURRENT_SOURCE"
        if tuple(record["source_winners"]) == current_source
        else "DIVERGED_SOURCE"
    )
    return record_index, record, relation


def localize_owner_failure(
    *,
    checkpoint: bytes,
    recognition_segment_limit: int,
    ledger: OwnerOutcomeLedger,
    outcome: dict[str, Any],
    step: TemporalCircuitStep,
    definition: NativeTemporalCircuitDefinition,
    current_token_columns: tuple[int, ...],
) -> dict[str, Any]:
    """Decode the minimum post-step segment lifecycle for one owner violation."""

    parsed = lifecycle_probe.parse_checkpoint_rows(checkpoint)
    if recognition_segment_limit != ledger.key_count * ACTIVE_COLUMNS:
        raise RuntimeError("segment/owner-ledger prefix identity drifted")
    if not 0 <= recognition_segment_limit <= parsed.segment_count:
        raise RuntimeError("recognition segment limit is outside checkpoint")
    current_source = tuple(int(value) for value in outcome["source_winners"])
    observed_winners = tuple(int(value) for value in outcome["observed_winners"])
    if len(current_source) != ACTIVE_COLUMNS or len(observed_winners) != ACTIVE_COLUMNS:
        raise RuntimeError("owner failure lost its 11-cell source/owner tuples")

    sources = parsed.sources[:recognition_segment_limit]
    ticks = parsed.permanence_ticks[:recognition_segment_limit]
    live = sources != lifecycle_probe.NONE
    overlap = live & np.isin(sources, np.asarray(current_source, dtype=np.uint32))
    potential_contacts = overlap.sum(axis=1, dtype=np.uint8) * 2
    connected_contacts = (
        (
            overlap
            & (ticks >= definition.config.connected_permanence_ticks)
        ).sum(axis=1, dtype=np.uint8)
        * 2
    )
    matching_ids = np.flatnonzero(
        potential_contacts >= definition.config.theta_match
    ).astype(np.int64, copy=False)
    active_ids = np.flatnonzero(
        connected_contacts >= definition.config.theta_active
    ).astype(np.int64, copy=False)

    expected_record = outcome.get("key_record_index")
    expected_ids: set[int] = set()
    if expected_record is not None:
        expected_ids.update(
            range(
                int(expected_record) * ACTIVE_COLUMNS,
                (int(expected_record) + 1) * ACTIVE_COLUMNS,
            )
        )
    relevant_ids = sorted(
        set(int(value) for value in matching_ids)
        | set(int(value) for value in active_ids)
        | expected_ids
    )
    matching_set = set(int(value) for value in matching_ids)
    active_set = set(int(value) for value in active_ids)
    current_column_set = frozenset(current_token_columns)

    grouped_ids: dict[int, list[int]] = {}
    for segment_id in relevant_ids:
        record_index = segment_id // ACTIVE_COLUMNS
        grouped_ids.setdefault(record_index, []).append(segment_id)

    segment_groups: list[dict[str, Any]] = []
    foreign_matching = 0
    foreign_active_post = 0
    foreign_matching_with_reduced_ticks = 0
    exact_owner_matching = 0
    exact_owner_active_post = 0
    exact_owner_with_reduced_ticks = 0
    matching_owner_record: int | None = None
    current_punishment_deactivation_candidates = 0
    for record_index in sorted(grouped_ids):
        segment_ids = grouped_ids[record_index]
        record = ledger.key_record(record_index)
        origin_source = tuple(int(value) for value in record["source_winners"])
        origin_owners = tuple(int(value) for value in record["owner_cells"])
        relation = (
            "EXACT_CURRENT_SOURCE"
            if origin_source == current_source
            else "DIVERGED_SOURCE"
        )
        owner_cells: list[int] = []
        owner_columns: list[int] = []
        row_sources: list[tuple[int, ...]] = []
        row_ticks: list[tuple[int, ...]] = []
        potentials: list[int] = []
        connected: list[int] = []
        reversed_connected: list[int] = []
        reduced_overlap_pairs: list[int] = []
        for segment_id in segment_ids:
            row = tuple(
                int(value)
                for value in sources[segment_id]
                if int(value) != lifecycle_probe.NONE
            )
            permanence = tuple(
                int(value) for value in ticks[segment_id][: len(row)]
            )
            owner = int(parsed.owners[segment_id])
            owner_column = lifecycle_probe._owner_column(
                owner,
                cohort_columns=parsed.cohort_columns,
                column_count=definition.config.column_count,
                base_cells_per_column=definition.config.base_cells_per_column,
                novel_race_cap=definition.config.novel_race_cap,
            )
            potential = int(potential_contacts[segment_id])
            connected_now = int(connected_contacts[segment_id])
            connected_if_reversed = 2 * sum(
                source in current_source
                and min(0xFFFF, tick + definition.config.negative_decrement_ticks)
                >= definition.config.connected_permanence_ticks
                for source, tick in zip(row, permanence, strict=True)
            )
            reduced_pairs = sum(
                source in current_source
                and tick < definition.config.initial_permanence_ticks
                for source, tick in zip(row, permanence, strict=True)
            )
            owner_cells.append(owner)
            owner_columns.append(owner_column)
            row_sources.append(row)
            row_ticks.append(permanence)
            potentials.append(potential)
            connected.append(connected_now)
            reversed_connected.append(connected_if_reversed)
            reduced_overlap_pairs.append(reduced_pairs)

            is_matching = segment_id in matching_set
            is_active_post = segment_id in active_set
            if relation == "DIVERGED_SOURCE" and is_matching:
                foreign_matching += 1
                if reduced_pairs:
                    foreign_matching_with_reduced_ticks += 1
            if relation == "DIVERGED_SOURCE" and is_active_post:
                foreign_active_post += 1
            if expected_record is not None and record_index == int(expected_record):
                exact_owner_matching += int(is_matching)
                exact_owner_active_post += int(is_active_post)
                exact_owner_with_reduced_ticks += int(bool(reduced_pairs))
            if (
                is_matching
                and not is_active_post
                and connected_if_reversed >= definition.config.theta_active
                and connected_now < definition.config.theta_active
            ):
                current_punishment_deactivation_candidates += 1

        full_record = len(segment_ids) == ACTIVE_COLUMNS
        owner_tuple_matches_observed = (
            full_record and tuple(owner_cells) == observed_winners
        )
        if owner_tuple_matches_observed and all(
            segment_id in matching_set for segment_id in segment_ids
        ):
            matching_owner_record = record_index
        segment_groups.append(
            {
                "key_record_index": record_index,
                "key_relation": (
                    "EXPECTED_REPEATED_KEY"
                    if expected_record is not None
                    and record_index == int(expected_record)
                    else relation
                ),
                "origin_global_step_index": record["origin_global_step_index"],
                "origin_book_ordinal": record["origin_book_ordinal"],
                "origin_book_zero_based_token_index": record[
                    "origin_book_zero_based_token_index"
                ],
                "origin_token_id": record["token_id"],
                "origin_source_winners": list(origin_source),
                "origin_source_sha256": _sha256_u32(origin_source),
                "origin_current_source_overlap_cells": len(
                    frozenset(origin_source) & frozenset(current_source)
                ),
                "origin_owner_cells": list(origin_owners),
                "segment_ids": segment_ids,
                "owner_cells": owner_cells,
                "owner_columns": owner_columns,
                "owners_in_current_token_columns": all(
                    column in current_column_set for column in owner_columns
                ),
                "owner_tuple_matches_observed_winners": owner_tuple_matches_observed,
                "row_sources_uniform": len(set(row_sources)) == 1,
                "row_sources": (
                    list(row_sources[0])
                    if len(set(row_sources)) == 1
                    else [list(row) for row in row_sources]
                ),
                "permanence_ticks_uniform": len(set(row_ticks)) == 1,
                "permanence_ticks": (
                    list(row_ticks[0])
                    if len(set(row_ticks)) == 1
                    else [list(row) for row in row_ticks]
                ),
                "potential_contacts": (
                    potentials[0]
                    if len(set(potentials)) == 1
                    else potentials
                ),
                "connected_contacts_post_step": (
                    connected[0]
                    if len(set(connected)) == 1
                    else connected
                ),
                "connected_contacts_if_one_punishment_reversed": (
                    reversed_connected[0]
                    if len(set(reversed_connected)) == 1
                    else reversed_connected
                ),
                "reduced_overlap_pairs": (
                    reduced_overlap_pairs[0]
                    if len(set(reduced_overlap_pairs)) == 1
                    else reduced_overlap_pairs
                ),
                "matching_segments": sum(
                    segment_id in matching_set for segment_id in segment_ids
                ),
                "active_segments_post_step": sum(
                    segment_id in active_set for segment_id in segment_ids
                ),
            }
        )

    observed_owner_tag = ledger.lookup_winner(observed_winners)
    prior_owner_records: list[dict[str, Any]] = []
    prior_owner_record_count = 0
    if observed_owner_tag is not None and ledger.key_count:
        record_ids = np.flatnonzero(
            ledger.key_owner_tags[: ledger.key_count] == observed_owner_tag
        )
        prior_owner_record_count = int(len(record_ids))
        for raw_record in record_ids[:8]:
            prior_owner_records.append(ledger.key_record(int(raw_record)))

    violation = outcome["violation"]
    winner_identity_diverged = bool(violation.get("winner_identity_diverged", False))
    owner_identity_reused_for_novel = (
        violation["kind"] == "NOVEL_KEY_OWNER_OUTCOME"
        and observed_owner_tag is not None
    )
    matched_owner_group = next(
        (
            group
            for group in segment_groups
            if group["key_record_index"] == matching_owner_record
        ),
        None,
    )
    if (
        violation["kind"] == "NOVEL_KEY_OWNER_OUTCOME"
        and matched_owner_group is not None
        and matched_owner_group["key_relation"] == "DIVERGED_SOURCE"
        and matched_owner_group["active_segments_post_step"] == 0
        and matched_owner_group["reduced_overlap_pairs"] != 0
    ):
        classification = "NOVEL_KEY_CAPTURED_BY_FOREIGN_DEACTIVATED_MATCHING_OWNERS"
    elif (
        violation["kind"] == "NOVEL_KEY_OWNER_OUTCOME"
        and matched_owner_group is not None
        and matched_owner_group["key_relation"] == "DIVERGED_SOURCE"
        and matched_owner_group["active_segments_post_step"]
    ):
        classification = "NOVEL_KEY_CAPTURED_BY_FOREIGN_ACTIVE_OWNERS"
    elif expected_record is not None and exact_owner_with_reduced_ticks:
        classification = "REPEATED_KEY_EXACT_OWNER_DEACTIVATION"
    elif foreign_active_post:
        classification = "FOREIGN_ACTIVE_OWNER_CAPTURE"
    elif winner_identity_diverged:
        classification = "REPEATED_KEY_WINNER_IDENTITY_DIVERGENCE"
    elif owner_identity_reused_for_novel:
        classification = "NOVEL_KEY_REUSED_EXISTING_WINNER_IDENTITY"
    else:
        classification = "OWNER_OUTCOME_FAILURE_UNRESOLVED"

    post_matching = int(len(matching_ids))
    post_active = int(len(active_ids))
    appended_ids = list(range(recognition_segment_limit, parsed.segment_count))
    appended_owners = [int(parsed.owners[index]) for index in appended_ids]
    appended_columns = [
        lifecycle_probe._owner_column(
            owner,
            cohort_columns=parsed.cohort_columns,
            column_count=definition.config.column_count,
            base_cells_per_column=definition.config.base_cells_per_column,
            novel_race_cap=definition.config.novel_race_cap,
        )
        for owner in appended_owners
    ]
    return {
        "classification": classification,
        "checkpoint_body_sha256": parsed.body_sha256,
        "checkpoint_segment_count_after_step": parsed.segment_count,
        "recognition_segment_limit_before_step": recognition_segment_limit,
        "current_source_winners": list(current_source),
        "current_source_sha256": _sha256_u32(current_source),
        "current_token_columns": list(current_token_columns),
        "observed_winners": list(observed_winners),
        "observed_winner_tag_before_step": observed_owner_tag,
        "observed_owner_prior_key_record_count": prior_owner_record_count,
        "observed_owner_prior_key_records_first_eight": prior_owner_records,
        "owner_identity_reused_for_novel_key": owner_identity_reused_for_novel,
        "winner_identity_diverged": winner_identity_diverged,
        "matching_owner_key_record_index": matching_owner_record,
        "native_matching_segments_pre_learning_commit": int(step.matching_segments),
        "native_active_segments_pre_learning_commit": int(step.active_segments),
        "post_checkpoint_matching_segments": post_matching,
        "post_checkpoint_active_segments": post_active,
        "counter_checkpoint_temporal_relation": (
            "native counters are pre-learning-commit; checkpoint rows are "
            "post-reinforcement/punishment/append; equality is diagnostic only"
        ),
        "post_checkpoint_equals_precommit_matching_diagnostic": (
            post_matching == int(step.matching_segments)
        ),
        "post_checkpoint_equals_precommit_active_diagnostic": (
            post_active == int(step.active_segments)
        ),
        "punishment_occurred": int(step.punished_segments) > 0,
        "punished_segments": int(step.punished_segments),
        "post_step_mutation_ambiguity": True,
        "current_punishment_deactivation_candidates": (
            current_punishment_deactivation_candidates
        ),
        "exact_owner_segments": {
            "applicable": expected_record is not None,
            "key_record_index": expected_record,
            "matching": exact_owner_matching,
            "active_post_checkpoint": exact_owner_active_post,
            "with_reduced_overlap_ticks": exact_owner_with_reduced_ticks,
            "inactive_or_deactivated": (
                expected_record is not None
                and exact_owner_active_post < ACTIVE_COLUMNS
            ),
        },
        "foreign_segments": {
            "matching": foreign_matching,
            "active_post_checkpoint": foreign_active_post,
            "active_existed_post_checkpoint": foreign_active_post > 0,
            "matching_with_reduced_overlap_ticks": (
                foreign_matching_with_reduced_ticks
            ),
        },
        "failing_step_appended_segments": {
            "provenance": "not in the successful-key ledger",
            "segment_ids": appended_ids,
            "owner_cells": appended_owners,
            "owner_columns": appended_columns,
        },
        "segment_groups": segment_groups,
    }


def _new_metrics() -> dict[str, Any]:
    return {
        "executed_token_steps": 0,
        "executed_transition_steps": 0,
        "allocated_segments": 0,
        "punished_segments": 0,
        "predicted_steps": 0,
        "partial_prediction_steps": 0,
        "native_step_call_ns_total": 0,
        "native_step_call_ns_max": 0,
    }


def _record_step_metrics(
    metrics: dict[str, Any], step: TemporalCircuitStep, native_call_ns: int
) -> None:
    metrics["executed_token_steps"] += 1
    metrics["allocated_segments"] += int(step.allocated_segments)
    metrics["punished_segments"] += int(step.punished_segments)
    metrics["predicted_steps"] += int(step.predicted)
    metrics["partial_prediction_steps"] += int(
        0 < int(step.predicted_columns) < ACTIVE_COLUMNS
    )
    metrics["native_step_call_ns_total"] += native_call_ns
    metrics["native_step_call_ns_max"] = max(
        metrics["native_step_call_ns_max"], native_call_ns
    )


def _finalize_metrics(
    metrics: dict[str, Any],
    *,
    replay_wall_ns: int,
    checkpoint_export_ns: int,
    ledger: OwnerOutcomeLedger,
    live_segments: int,
) -> dict[str, Any]:
    executed = int(metrics["executed_token_steps"])
    return {
        "executed_token_steps": executed,
        "executed_transition_steps": int(metrics["executed_transition_steps"]),
        "replay_wall_ns": replay_wall_ns,
        "checkpoint_export_ns": checkpoint_export_ns,
        "end_to_end_steps_per_second": (
            0.0 if replay_wall_ns == 0 else executed * 1e9 / replay_wall_ns
        ),
        "gc_during_replay": "disabled",
        "single_token_native_calls": True,
        "native_step_call_ns_total": int(metrics["native_step_call_ns_total"]),
        "native_step_call_ns_mean": (
            0.0
            if executed == 0
            else int(metrics["native_step_call_ns_total"]) / executed
        ),
        "native_step_call_ns_max": int(metrics["native_step_call_ns_max"]),
        "allocated_segments": int(metrics["allocated_segments"]),
        "punished_segments": int(metrics["punished_segments"]),
        "predicted_steps": int(metrics["predicted_steps"]),
        "partial_prediction_steps": int(metrics["partial_prediction_steps"]),
        "tracked_live_segments": live_segments,
        "ledger_key_count": ledger.key_count,
        "ledger_repeat_count": ledger.repeats,
        "ledger_winner_tuple_count": ledger.winner_count,
        "ledger_book_resets": ledger.resets,
        "ledger_preallocated_storage_bytes": ledger.storage_bytes,
    }


def run_stream(
    circuit: NativeTemporalCircuit,
    books: tuple[tuple[object, np.ndarray], ...],
    *,
    ledger: OwnerOutcomeLedger,
    definition: NativeTemporalCircuitDefinition,
) -> tuple[dict[str, Any], dict[str, Any], object | None]:
    """Run the spent TRAIN stream and return immediately after the first failure."""

    token_columns = dict(definition.token_codes)
    metrics = _new_metrics()
    live_segments = 0
    global_step = 0
    replay_started_ns = time.perf_counter_ns()
    checkpoint_export_ns = 0
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for book, tokens in books:
            for token_index, raw_token in enumerate(tokens):
                token_id = int(raw_token)
                live_before = live_segments
                native_started_ns = time.perf_counter_ns()
                try:
                    emitted = circuit.advance_block(
                        [token_id],
                        mode=TemporalStepMode.LEARN,
                        reset_before=token_index == 0,
                    )
                except NativeTemporalRuntimeError as error:
                    native_call_ns = time.perf_counter_ns() - native_started_ns
                    replay_wall_ns = time.perf_counter_ns() - replay_started_ns
                    result = {
                        "kind": "FIRST_NATIVE_FAILURE",
                        "book_ordinal": int(book.ordinal),
                        "book_short_name": str(book.short_name),
                        "book_zero_based_token_index": token_index,
                        "global_zero_based_step_index": global_step,
                        "token_id": token_id,
                        "native_status": error.status.name,
                        "native_status_code": int(error.status),
                        "native_scientific_stop": (
                            error.status is NativeTemporalRuntimeStatus.SCIENTIFIC_STOP
                        ),
                        "error_message": str(error),
                        "failed_native_call_ns": native_call_ns,
                        "live_segments_before_failed_step": live_segments,
                        "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
                        "lifecycle": {
                            "classification": "NATIVE_FAILURE_BEFORE_OWNER_OUTCOME",
                            "checkpoint_available": False,
                        },
                    }
                    return (
                        result,
                        _finalize_metrics(
                            metrics,
                            replay_wall_ns=replay_wall_ns,
                            checkpoint_export_ns=0,
                            ledger=ledger,
                            live_segments=live_segments,
                        ),
                        None,
                    )
                native_call_ns = time.perf_counter_ns() - native_started_ns
                if len(emitted) != 1:
                    raise RuntimeError("single-token native call did not emit one step")
                step = emitted[0]
                _record_step_metrics(metrics, step, native_call_ns)
                live_segments += int(step.allocated_segments)
                if token_index != 0:
                    metrics["executed_transition_steps"] += 1
                outcome = ledger.observe(
                    reset_before=token_index == 0,
                    step=step,
                    book_ordinal=int(book.ordinal),
                    book_token_index=token_index,
                    global_step_index=global_step,
                )
                violation = outcome["violation"]
                if violation is not None:
                    replay_wall_ns = time.perf_counter_ns() - replay_started_ns
                    if violation["kind"] not in OWNER_FAILURE_KINDS:
                        result = {
                            "kind": "FIRST_REPLAY_CONTRACT_FAILURE",
                            "book_ordinal": int(book.ordinal),
                            "book_short_name": str(book.short_name),
                            "book_zero_based_token_index": token_index,
                            "global_zero_based_step_index": global_step,
                            "token_id": token_id,
                            "violation": violation,
                            "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
                            "lifecycle": {
                                "classification": "NON_OWNER_CONTRACT_FAILURE",
                                "checkpoint_attempted": False,
                            },
                        }
                        return (
                            result,
                            _finalize_metrics(
                                metrics,
                                replay_wall_ns=replay_wall_ns,
                                checkpoint_export_ns=0,
                                ledger=ledger,
                                live_segments=live_segments,
                            ),
                            None,
                        )
                    checkpoint_started_ns = time.perf_counter_ns()
                    identity = circuit.checkpoint_identity()
                    if identity.checkpoint_bytes > MAX_CHECKPOINT_COPY_BYTES:
                        lifecycle = {
                            "classification": "OWNER_FAILURE_CHECKPOINT_COPY_BUDGET_STOP",
                            "checkpoint_attempted": False,
                            "checkpoint_bytes": identity.checkpoint_bytes,
                            "maximum_checkpoint_copy_bytes": MAX_CHECKPOINT_COPY_BYTES,
                            "recognition_segment_limit_before_step": live_before,
                            "failing_step_appended_segment_count": (
                                identity.segment_count - live_before
                            ),
                        }
                    else:
                        checkpoint = circuit.checkpoint()
                        lifecycle = localize_owner_failure(
                            checkpoint=checkpoint,
                            recognition_segment_limit=live_before,
                            ledger=ledger,
                            outcome=outcome,
                            step=step,
                            definition=definition,
                            current_token_columns=tuple(token_columns[token_id]),
                        )
                    checkpoint_export_ns = time.perf_counter_ns() - checkpoint_started_ns
                    result = {
                        "kind": "FIRST_OWNER_OUTCOME_FAILURE",
                        "book_ordinal": int(book.ordinal),
                        "book_short_name": str(book.short_name),
                        "book_zero_based_token_index": token_index,
                        "global_zero_based_step_index": global_step,
                        "token_id": token_id,
                        "key_was_known": outcome["key_was_known"],
                        "violation": violation,
                        "observed": outcome["diagnostic_counters"],
                        "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
                        "lifecycle": lifecycle,
                        "checkpoint_identity_after_failure": {
                            "segment_count": identity.segment_count,
                            "cohort_count": identity.cohort_count,
                            "next_reserve_cell": identity.next_reserve_cell,
                            "event_count": identity.event_count,
                            "learning_step": identity.learning_step,
                            "logical_state_sha256": identity.logical_state_sha256,
                        },
                    }
                    return (
                        result,
                        _finalize_metrics(
                            metrics,
                            replay_wall_ns=replay_wall_ns,
                            checkpoint_export_ns=checkpoint_export_ns,
                            ledger=ledger,
                            live_segments=live_segments,
                        ),
                        identity,
                    )
                if live_segments != ledger.key_count * ACTIVE_COLUMNS:
                    raise RuntimeError("tracked segments diverged from passing owner ledger")
                global_step += 1
                if (token_index + 1) % 16_384 == 0:
                    print(
                        f"{book.short_name}: {token_index + 1:,}/{book.token_count:,} "
                        f"tokens, {live_segments:,} segments, {ledger.key_count:,} keys",
                        flush=True,
                    )
            print(
                f"{book.short_name}: complete, {live_segments:,} segments, "
                f"{ledger.key_count:,} keys",
                flush=True,
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    replay_wall_ns = time.perf_counter_ns() - replay_started_ns
    checkpoint_started_ns = time.perf_counter_ns()
    identity = circuit.checkpoint_identity()
    checkpoint_export_ns = time.perf_counter_ns() - checkpoint_started_ns
    result = {
        "kind": "TRAIN_OWNER_OUTCOMES_COMPLETE",
        "completed_books": len(books),
        "completed_token_steps": metrics["executed_token_steps"],
        "completed_transition_steps": metrics["executed_transition_steps"],
        "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
        "lifecycle": {
            "classification": "NO_OWNER_OUTCOME_FAILURE_IN_CANONICAL_TRAIN"
        },
        "final_checkpoint_identity": {
            "segment_count": identity.segment_count,
            "cohort_count": identity.cohort_count,
            "next_reserve_cell": identity.next_reserve_cell,
            "event_count": identity.event_count,
            "learning_step": identity.learning_step,
            "logical_state_sha256": identity.logical_state_sha256,
        },
    }
    return (
        result,
        _finalize_metrics(
            metrics,
            replay_wall_ns=replay_wall_ns,
            checkpoint_export_ns=checkpoint_export_ns,
            ledger=ledger,
            live_segments=live_segments,
        ),
        identity,
    )


def validate_artifact(payload: dict[str, Any]) -> None:
    """Validate an already-banked result without touching the native runtime."""

    _require(
        payload.get("artifact_id")
        == "public-train-first-owner-outcome-failure-v1",
        "artifact identity mismatch",
    )
    _require(payload.get("schema_version") == 1, "artifact schema mismatch")
    _require(payload.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary drift")
    _require(
        payload.get("definition", {}).get("arm") == list(ARM)
        and payload.get("definition", {}).get("construction_seed")
        == CONSTRUCTION_SEED,
        "artifact definition mismatch",
    )
    _require(
        payload.get("spend_marker", {}).get("state")
        == "OWNER_OUTCOME_REPLAY_SPENT"
        and payload.get("spend_marker", {}).get("exclusive_create") is True
        and payload.get("spend_marker", {}).get(
            "flushed_and_fsynced_before_run"
        )
        is True,
        "artifact spend marker contract mismatch",
    )
    _require(
        payload.get("train_input", {}).get("held_out_books_materialized") == 0,
        "artifact crossed into held-out material",
    )
    result = payload.get("result", {})
    _require(
        result.get("diagnostic_counter_contract") == DIAGNOSTIC_COUNTER_CONTRACT,
        "artifact global diagnostic contract drift",
    )
    status = payload.get("status")
    if status == "BANKED_FIRST_OWNER_OUTCOME_FAILURE":
        _require(
            result.get("kind") == "FIRST_OWNER_OUTCOME_FAILURE"
            and result.get("violation", {}).get("kind")
            in {"NOVEL_KEY_OWNER_OUTCOME", "REPEATED_KEY_OWNER_OUTCOME"}
            and bool(result.get("lifecycle", {}).get("classification")),
            "owner failure artifact is incomplete",
        )
    elif status == "PASS_ALL_CANONICAL_TRAIN_OWNER_OUTCOMES":
        _require(
            result.get("kind") == "TRAIN_OWNER_OUTCOMES_COMPLETE",
            "TRAIN-complete artifact result mismatch",
        )
    elif status == "BANKED_REPLAY_INFRASTRUCTURE_FAILURE":
        _require(bool(result.get("kind")), "infrastructure failure has no result")
    else:
        raise RuntimeError("artifact status mismatch")


def read_banked_artifact(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    validate_artifact(payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    if SPEND_MARKER_PATH.exists() or OUTPUT_PATH.exists():
        parser.error("refusing spent replay; spend marker or result artifact exists")
    return args


def _write_result(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing to overwrite owner-outcome artifact") from error


def main() -> int:
    _parse_args()
    preparation_started_ns = time.perf_counter_ns()
    verified = load_verified_train_input()
    encoded_books = tuple(verified.iter_books())
    if len(encoded_books) != EXPECTED.expected_book_resets:
        raise RuntimeError("canonical TRAIN book count drifted")
    books: list[tuple[object, np.ndarray]] = []
    for ordinal, book in enumerate(encoded_books):
        if int(book.ordinal) != ordinal or int(book.token_count) < 2:
            raise RuntimeError("canonical TRAIN order drifted")
        tokens = np.frombuffer(book.token_bytes_le_i32, dtype="<i4")
        if len(tokens) != int(book.token_count):
            raise RuntimeError("canonical TRAIN token materialization drifted")
        books.append((book, tokens))
    total_tokens = sum(len(tokens) for _book, tokens in books)
    transition_count = sum(len(tokens) - 1 for _book, tokens in books)
    if (
        total_tokens != EXPECTED.expected_tokens
        or transition_count != EXPECTED.expected_transitions
        or verified.train_stream_sha256 != EXPECTED.train_stream_sha256
    ):
        raise RuntimeError("canonical TRAIN authority/count contract drifted")

    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=False)
    production = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    bounded = NativeTemporalCircuitDefinition(
        replace(
            production.config,
            segment_capacity=transition_count * ACTIVE_COLUMNS,
        ),
        production.token_codes,
        production.threshold_offsets_mv,
    )
    codebook_tokens = {token for token, _columns in bounded.token_codes}
    for book, tokens in books:
        if any(int(token) not in codebook_tokens for token in np.unique(tokens)):
            raise RuntimeError(f"{book.short_name} contains an out-of-codebook token")
    ledger = OwnerOutcomeLedger(
        expected_keys=EXPECTED.expected_distinct_keys,
        expected_winner_rows=EXPECTED_WINNER_ROWS,
    )
    native_identity_before = loaded_library_identity()
    preparation_ns = time.perf_counter_ns() - preparation_started_ns
    authority = {
        "train_input_authority_sha256": verified.train_input_authority_sha256,
        "train_stream_sha256": verified.train_stream_sha256,
        "tokenizer_sha256": verified.tokenizer_sha256,
        "protocol_sha256": protocol_authority.protocol_sha256(),
        "production_definition_identity_sha256": (
            production.circuit_identity_sha256()
        ),
        "bounded_definition_identity_sha256": bounded.circuit_identity_sha256(),
        "native_source_digest": native_identity_before.source_digest,
        "native_binary_sha256": native_identity_before.binary_sha256,
        "expected_train_books": len(books),
        "expected_train_tokens": total_tokens,
        "expected_train_transitions": transition_count,
    }
    spend_marker = create_spend_marker(authority=authority)

    construction_ns = 0
    result: dict[str, Any]
    timing_load: dict[str, Any]
    final_identity: object | None
    try:
        with lifecycle_probe._synthetic_preoutcome_authority():
            construction_started_ns = time.perf_counter_ns()
            with NativeTemporalCircuit.fresh(bounded) as circuit:
                construction_ns = time.perf_counter_ns() - construction_started_ns
                result, timing_load, final_identity = run_stream(
                    circuit,
                    tuple(books),
                    ledger=ledger,
                    definition=bounded,
                )
        if result["kind"] == "FIRST_OWNER_OUTCOME_FAILURE":
            status = "BANKED_FIRST_OWNER_OUTCOME_FAILURE"
        elif result["kind"] == "TRAIN_OWNER_OUTCOMES_COMPLETE":
            status = "PASS_ALL_CANONICAL_TRAIN_OWNER_OUTCOMES"
        else:
            status = "BANKED_REPLAY_INFRASTRUCTURE_FAILURE"
    except Exception as error:
        status = "BANKED_REPLAY_INFRASTRUCTURE_FAILURE"
        final_identity = None
        result = {
            "kind": "UNEXPECTED_REPLAY_EXCEPTION",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
            "lifecycle": {"classification": "UNEXPECTED_REPLAY_EXCEPTION"},
        }
        timing_load = {
            "executed_token_steps": ledger.calls,
            "executed_transition_steps": ledger.key_count + ledger.repeats,
            "replay_wall_ns": 0,
            "checkpoint_export_ns": 0,
            "ledger_key_count": ledger.key_count,
            "ledger_repeat_count": ledger.repeats,
            "ledger_winner_tuple_count": ledger.winner_count,
            "ledger_book_resets": ledger.resets,
            "ledger_preallocated_storage_bytes": ledger.storage_bytes,
            "tracked_live_segments": ledger.key_count * ACTIVE_COLUMNS,
        }

    native_identity_after = loaded_library_identity()
    payload = {
        "artifact_id": "public-train-first-owner-outcome-failure-v1",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "spend_marker": spend_marker,
        "train_input": {
            "candidate_id": verified.candidate_id,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "full_train_book_count": len(books),
            "full_train_token_count": total_tokens,
            "full_train_transition_count": transition_count,
            "held_out_books_materialized": 0,
            "books": [
                {
                    "ordinal": int(book.ordinal),
                    "corpus_index": int(book.corpus_index),
                    "short_name": str(book.short_name),
                    "token_count": int(book.token_count),
                    "token_sha256_le_i32": str(book.token_sha256_le_i32),
                }
                for book, _tokens in books
            ],
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "protocol_sha256": protocol_authority.protocol_sha256(),
            "protocol_runnable": protocol["source_binding"]["runnable"],
            "production_definition_identity_sha256": (
                production.circuit_identity_sha256()
            ),
            "bounded_definition_identity_sha256": bounded.circuit_identity_sha256(),
            "production_segment_capacity": production.config.segment_capacity,
            "bounded_segment_capacity": bounded.config.segment_capacity,
            "bounded_capacity_rule": "all canonical TRAIN transitions * 11",
        },
        "result": result,
        "timing_load": {
            "pre_run_preparation_ns": preparation_ns,
            "native_construction_ns": construction_ns,
            **timing_load,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest_before": native_identity_before.source_digest,
            "native_source_digest_after": native_identity_after.source_digest,
            "native_binary_sha256_before": native_identity_before.binary_sha256,
            "native_binary_sha256_after": native_identity_after.binary_sha256,
        },
    }
    validate_artifact(payload)
    _write_result(payload)
    print(f"wrote {OUTPUT_PATH}", flush=True)
    print(f"status={status}", flush=True)
    print(f"artifact_sha256={_sha256_file(OUTPUT_PATH)}", flush=True)
    return 0 if status != "BANKED_REPLAY_INFRASTRUCTURE_FAILURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
