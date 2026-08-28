"""Pure streaming structural scoring for the calibration-v3 R3-F0 TRAIN pass.

The scorer consumes already-emitted native-like steps.  It never opens the
TRAIN corpus, runs a circuit, or publishes state.  Its large state is held in
fixed-width arrays: an exact ``(source winner tuple, token) -> owner tuple``
index, the novel owner rows needed by the overlap gate, compact source fanout,
and byte-sized owner loads.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from experiments.r3_pilot.calibration_v3_overlap_scoring import (
    first_same_position_overlap_violation,
)

_UINT32_MAX: Final = (1 << 32) - 1
_UINT64_MASK: Final = (1 << 64) - 1
_SEED_SOURCE_TAG: Final = 1 << 31
_WINNER_WIDTH: Final = 11
_WINNERS: Final = struct.Struct("<11I")
_TOKEN: Final = struct.Struct("<i")
_COHORT: Final = struct.Struct("<III")
_ALLOWED_ARMS: Final = ((10, 13), (16, 18), (18, 20), (20, 20))

# Python 3.11 is the frozen calibration runtime, so PEP 695's ``type`` syntax
# is deliberately unavailable here.
MetricValue: TypeAlias = int | str | list[int] | list[str]  # noqa: UP040
ScoreResult: TypeAlias = dict[str, MetricValue]  # noqa: UP040


@dataclass(frozen=True, slots=True)
class R3F0TrainExpectations:
    """Literal structural contract, replaceable only for bounded synthetic tests."""

    column_count: int = 4_096
    max_pyramidal_cells: int = 131_072
    base_cells_per_column: int = 8
    cohort_width: int = 8
    max_segments_per_cell: int = 255
    construction_seed: int = 17
    expected_book_resets: int = 49
    expected_tokens: int = 2_117_412
    expected_transitions: int = 2_117_363
    expected_distinct_keys: int = 2_116_946
    expected_segments: int = 23_286_406
    expected_recruited_columns: int = 1_286
    expected_next_reserve_cell: int = 114_400
    expected_max_depth: int = 1_128
    expected_final_event_count: int = 37_538
    train_stream_sha256: str = (
        "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
    )
    token_codebook_sha256: str = (
        "388177db5f1d8cba9f0606d5e065148498418b10d27d7abf6117d09b44238a9c"
    )

    @property
    def expected_repeats(self) -> int:
        return self.expected_transitions - self.expected_distinct_keys

    @property
    def expected_total_cohorts(self) -> int:
        base = self.column_count * self.base_cells_per_column
        return (self.expected_next_reserve_cell - base) // self.cohort_width


CALIBRATION_V3_R3_F0_EXPECTATIONS: Final = R3F0TrainExpectations()


def _canonical(value: object) -> MetricValue:
    if isinstance(value, (tuple, list, np.ndarray)):
        return [
            int(item) if isinstance(item, (int, np.integer, bool)) else str(item)
            for item in value
        ]
    if isinstance(value, (int, np.integer, bool)):
        return int(value)
    return str(value)


def _stop(
    kind: str,
    *,
    call_index: int,
    book_ordinal: int,
    token_index: int,
    expected: object,
    observed: object,
    **evidence: object,
) -> ScoreResult:
    result: ScoreResult = {
        "status": "STOP",
        "stop_kind": kind,
        "call_index": call_index,
        "book_ordinal": book_ordinal,
        "token_index": token_index,
        "expected": _canonical(expected),
        "observed": _canonical(observed),
    }
    for label in sorted(evidence):
        result[label] = _canonical(evidence[label])
    return result


def _table_capacity(entry_count: int) -> int:
    required = max(8, (entry_count * 10 + 6) // 7)
    return 1 << (required - 1).bit_length()


def _mix_u64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _winner_hash(row: Sequence[int]) -> int:
    value = 0xCBF29CE484222325
    for position, raw in enumerate(row):
        value ^= _mix_u64(int(raw) + (position << 32))
        value = (value * 0x100000001B3) & _UINT64_MASK
    return value


class _OwnerRowIndex:
    """Open-addressed exact row index; hashes only choose candidate slots."""

    __slots__ = ("_hashes", "_mask", "_row_plus_one", "_rows")

    def __init__(self, rows: np.ndarray, expected_rows: int) -> None:
        capacity = _table_capacity(expected_rows)
        self._hashes = np.zeros(capacity, dtype=np.uint64)
        self._row_plus_one = np.zeros(capacity, dtype=np.uint32)
        self._mask = capacity - 1
        self._rows = rows

    def lookup(self, row: Sequence[int]) -> int | None:
        row_hash = _winner_hash(row)
        slot = _mix_u64(row_hash) & self._mask
        while True:
            stored = int(self._row_plus_one[slot])
            if stored == 0:
                return None
            row_index = stored - 1
            if int(self._hashes[slot]) == row_hash and all(
                int(self._rows[row_index, position]) == int(row[position])
                for position in range(_WINNER_WIDTH)
            ):
                return row_index
            slot = (slot + 1) & self._mask

    def insert(self, row: Sequence[int], row_index: int) -> None:
        row_hash = _winner_hash(row)
        slot = _mix_u64(row_hash) & self._mask
        while int(self._row_plus_one[slot]) != 0:
            slot = (slot + 1) & self._mask
        self._hashes[slot] = row_hash
        self._row_plus_one[slot] = row_index + 1


class _KeyIndex:
    """Exact packed-u32-pair map from source/token to novel owner row."""

    __slots__ = ("_keys", "_mask", "_owner_plus_one")

    def __init__(self, expected_entries: int) -> None:
        capacity = _table_capacity(expected_entries)
        self._keys = np.zeros(capacity, dtype=np.uint64)
        self._owner_plus_one = np.zeros(capacity, dtype=np.uint32)
        self._mask = capacity - 1

    @staticmethod
    def _key(source_tag: int, token_id: int) -> int:
        return (source_tag << 32) | token_id

    def lookup(self, source_tag: int, token_id: int) -> int | None:
        key = self._key(source_tag, token_id)
        slot = _mix_u64(key) & self._mask
        while True:
            stored = int(self._owner_plus_one[slot])
            if stored == 0:
                return None
            if int(self._keys[slot]) == key:
                return stored - 1
            slot = (slot + 1) & self._mask

    def insert(self, source_tag: int, token_id: int, owner_index: int) -> None:
        key = self._key(source_tag, token_id)
        slot = _mix_u64(key) & self._mask
        while int(self._owner_plus_one[slot]) != 0:
            slot = (slot + 1) & self._mask
        self._keys[slot] = key
        self._owner_plus_one[slot] = owner_index + 1


def _policy_name(value: object) -> str:
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


def _sha256_text(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize_codebook(
    token_columns: Mapping[int, Sequence[int]] | Sequence[tuple[int, Sequence[int]]],
    expectations: R3F0TrainExpectations,
) -> tuple[dict[int, tuple[int, ...]] | None, str | None, ScoreResult | None]:
    raw_items = token_columns.items() if isinstance(token_columns, Mapping) else token_columns
    try:
        items = list(raw_items)
    except (TypeError, ValueError) as exc:
        return None, None, _stop(
            "TOKEN_CODEBOOK",
            call_index=-1,
            book_ordinal=-1,
            token_index=-1,
            expected="iterable token/columns pairs",
            observed=type(exc).__name__,
        )
    normalized: dict[int, tuple[int, ...]] = {}
    try:
        for raw_token, raw_columns in items:
            if type(raw_token) is not int or not 0 <= raw_token <= 0x7FFFFFFF:
                raise ValueError("token ID outside signed little-endian i32")
            columns = tuple(sorted(raw_columns))
            if (
                len(columns) != _WINNER_WIDTH
                or any(type(column) is not int for column in columns)
                or len(set(columns)) != _WINNER_WIDTH
                or not all(0 <= column < expectations.column_count for column in columns)
            ):
                raise ValueError("token columns are not eleven unique in-sheet integers")
            if raw_token in normalized:
                raise ValueError("duplicate token ID")
            normalized[raw_token] = columns
    except (TypeError, ValueError) as exc:
        return None, None, _stop(
            "TOKEN_CODEBOOK",
            call_index=-1,
            book_ordinal=-1,
            token_index=-1,
            expected="canonical token -> eleven sorted columns",
            observed=str(exc),
        )
    preimage = [[token, list(normalized[token])] for token in sorted(normalized)]
    digest = _sha256_text(preimage)
    if digest != expectations.token_codebook_sha256:
        return None, None, _stop(
            "TOKEN_CODEBOOK_IDENTITY",
            call_index=-1,
            book_ordinal=-1,
            token_index=-1,
            expected=expectations.token_codebook_sha256,
            observed=digest,
        )
    return normalized, digest, None


def _config_stop(config: object, expectations: R3F0TrainExpectations) -> ScoreResult | None:
    fields = (
        ("column_count", expectations.column_count),
        ("max_pyramidal_cells", expectations.max_pyramidal_cells),
        ("segment_capacity", expectations.expected_segments),
        ("construction_seed", expectations.construction_seed),
        ("connected_permanence_ticks", 250),
        ("initial_permanence_ticks", 255),
        ("permanence_increment_ticks", 50),
        ("negative_decrement_ticks", 2),
        ("base_cells_per_column", expectations.base_cells_per_column),
        ("novel_race_cap", expectations.cohort_width),
        ("prediction_quorum", _WINNER_WIDTH),
        ("max_segments_per_cell", expectations.max_segments_per_cell),
    )
    for label, expected in fields:
        observed = getattr(config, label, None)
        if type(observed) is not int or observed != expected:
            return _stop(
                "CONFIG_CONTRACT",
                call_index=-1,
                book_ordinal=-1,
                token_index=-1,
                expected=[label, expected],
                observed=[label, observed],
            )
    arm = (getattr(config, "theta_match", None), getattr(config, "theta_active", None))
    if arm not in _ALLOWED_ARMS:
        return _stop(
            "CONFIG_THRESHOLD_ARM",
            call_index=-1,
            book_ordinal=-1,
            token_index=-1,
            expected=[f"{left}/{right}" for left, right in _ALLOWED_ARMS],
            observed=f"{arm[0]}/{arm[1]}",
        )
    for label, expected in (
        ("context_policy", "BOUND_WINNERS"),
        ("plasticity_policy", "PLASTIC"),
    ):
        observed = _policy_name(getattr(config, label, None))
        if observed != expected:
            return _stop(
                "CONFIG_CONTRACT",
                call_index=-1,
                book_ordinal=-1,
                token_index=-1,
                expected=[label, expected],
                observed=[label, observed],
            )
    base = expectations.column_count * expectations.base_cells_per_column
    if (
        expectations.expected_distinct_keys >= _SEED_SOURCE_TAG
        or expectations.expected_segments != _WINNER_WIDTH * expectations.expected_distinct_keys
        or expectations.expected_tokens
        != expectations.expected_book_resets + expectations.expected_transitions
        or expectations.expected_next_reserve_cell < base
        or (expectations.expected_next_reserve_cell - base) % expectations.cohort_width
        or expectations.expected_next_reserve_cell > expectations.max_pyramidal_cells
    ):
        return _stop(
            "EXPECTATION_CONTRACT",
            call_index=-1,
            book_ordinal=-1,
            token_index=-1,
            expected="internally consistent fixed-width TRAIN expectations",
            observed="inconsistent expectations",
        )
    return None


def _step_int(step: object, label: str) -> int:
    value = getattr(step, label)
    if type(value) is not int or value < 0:
        raise ValueError(label)
    return value


def _cells(values: object) -> tuple[int, ...]:
    return tuple(
        int(getattr(value, "cell_id", value))
        for value in values  # type: ignore[union-attr]
    )


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_source(identity: object) -> tuple[object, str | None]:
    """Normalize a flat immutable identity snapshot without relaxing fields."""

    if isinstance(identity, Mapping):
        return identity, None
    if not isinstance(identity, tuple):
        return identity, None
    normalized: dict[str, object] = {}
    for index, entry in enumerate(identity):
        if not isinstance(entry, tuple) or len(entry) != 2:
            return identity, f"identity pair {index} is not an exact 2-tuple"
        label, value = entry
        if type(label) is not str:
            return identity, f"identity pair {index} label is not str"
        if label in normalized:
            return identity, f"identity field {label!r} is duplicated"
        normalized[label] = value
    return normalized, None


def _identity_value(identity: object, label: str) -> object:
    if isinstance(identity, Mapping):
        if label not in identity:
            raise AttributeError(label)
        return identity[label]
    return getattr(identity, label)


def _identity_int(identity: object, label: str) -> int:
    value = _identity_value(identity, label)
    if type(value) is not int or value < 0:
        raise ValueError(label)
    return value


class _TrainScorer:
    __slots__ = (
        "allocated_segments",
        "book_lengths",
        "calls",
        "candidate_maximum",
        "codebook",
        "cohort_count",
        "cohort_stream",
        "column_available",
        "column_depth",
        "context_stream",
        "expectations",
        "fanout_maximum",
        "key_index",
        "key_owner_stream",
        "local_event",
        "next_reserve_cell",
        "owner_fanout",
        "owner_index",
        "owner_loads",
        "owner_rows",
        "owner_tokens",
        "previous_source_tag",
        "previous_winners",
        "punished_segments",
        "recruited",
        "repeats",
        "reserve_columns",
        "resets",
        "seed_fanout",
        "seed_rows",
        "token_stream",
        "transitions",
        "winner_stream",
    )

    def __init__(
        self,
        expectations: R3F0TrainExpectations,
        codebook: dict[int, tuple[int, ...]],
    ) -> None:
        self.expectations = expectations
        self.codebook = codebook
        expected = expectations.expected_distinct_keys
        self.owner_rows = np.empty((expected, _WINNER_WIDTH), dtype=np.uint32)
        self.owner_tokens = np.empty(expected, dtype=np.uint32)
        self.owner_fanout = np.zeros(expected, dtype=np.uint32)
        self.owner_index = _OwnerRowIndex(self.owner_rows, expected)
        self.key_index = _KeyIndex(expected)
        self.owner_loads = np.zeros(expectations.max_pyramidal_cells, dtype=np.uint8)
        base = expectations.column_count * expectations.base_cells_per_column
        self.reserve_columns = np.full(
            expectations.max_pyramidal_cells - base,
            -1,
            dtype=np.int32,
        )
        self.column_depth = np.full(
            expectations.column_count,
            expectations.base_cells_per_column,
            dtype=np.uint32,
        )
        self.column_available = np.full(
            expectations.column_count,
            expectations.base_cells_per_column,
            dtype=np.uint32,
        )
        self.recruited = np.zeros(expectations.column_count, dtype=np.bool_)
        self.seed_rows: dict[bytes, int] = {}
        self.seed_fanout: list[int] = []
        self.previous_source_tag: int | None = None
        self.previous_winners: tuple[int, ...] | None = None
        self.calls = 0
        self.resets = 0
        self.transitions = 0
        self.repeats = 0
        self.allocated_segments = 0
        self.punished_segments = 0
        self.candidate_maximum = 0
        self.fanout_maximum = 0
        self.local_event = 0
        self.book_lengths: list[int] = []
        self.next_reserve_cell = base
        self.cohort_count = 0
        self.token_stream = hashlib.sha256()
        self.winner_stream = hashlib.sha256()
        self.context_stream = hashlib.sha256()
        self.key_owner_stream = hashlib.sha256()
        self.cohort_stream = hashlib.sha256()

    @property
    def distinct_keys(self) -> int:
        return self.transitions - self.repeats

    def _position(self) -> tuple[int, int, int]:
        return self.calls, self.resets - 1, self.local_event

    def _source_fanout(self, source_tag: int) -> int:
        if source_tag >= _SEED_SOURCE_TAG:
            return self.seed_fanout[source_tag - _SEED_SOURCE_TAG]
        return int(self.owner_fanout[source_tag])

    def _increment_source_fanout(self, source_tag: int) -> None:
        if source_tag >= _SEED_SOURCE_TAG:
            index = source_tag - _SEED_SOURCE_TAG
            self.seed_fanout[index] += 1
            observed = self.seed_fanout[index]
        else:
            self.owner_fanout[source_tag] += 1
            observed = int(self.owner_fanout[source_tag])
        self.fanout_maximum = max(self.fanout_maximum, observed)

    def _cell_column(self, cell: int) -> int | None:
        exp = self.expectations
        base = exp.column_count * exp.base_cells_per_column
        if not 0 <= cell < self.next_reserve_cell:
            return None
        if cell < base:
            return cell // exp.base_cells_per_column
        column = int(self.reserve_columns[cell - base])
        return None if column < 0 else column

    def _validate_physical_step(
        self,
        step: object,
        token_columns: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | ScoreResult:
        call_index, book, token_index = self._position()
        try:
            winners = _cells(step.winner_spikes)
            relays = _cells(step.winner_relay_spikes)
            winner_context = tuple(int(value) for value in step.winner_context)
            context = tuple(int(value) for value in step.context)
        except (AttributeError, TypeError, ValueError) as exc:
            return _stop(
                "STEP_PHYSICAL_SHAPE",
                call_index=call_index,
                book_ordinal=book,
                token_index=token_index,
                expected="winner/relay/context fields",
                observed=type(exc).__name__,
            )
        expected_relays = tuple(self.expectations.max_pyramidal_cells + cell for cell in winners)
        expected_context = winners + expected_relays
        if (
            len(winners) != _WINNER_WIDTH
            or len(set(winners)) != _WINNER_WIDTH
            or not all(0 <= cell < self.expectations.max_pyramidal_cells for cell in winners)
            or relays != expected_relays
            or winner_context != expected_context
            or context != expected_context
        ):
            return _stop(
                "WINNER_RELAY_CONTEXT_IDENTITY",
                call_index=call_index,
                book_ordinal=book,
                token_index=token_index,
                expected=expected_context,
                observed=context,
                relays=relays,
                winner_context=winner_context,
                winners=winners,
            )
        return winners, relays, context

    def _validate_existing_columns(
        self,
        winners: tuple[int, ...],
        token_columns: tuple[int, ...],
    ) -> ScoreResult | None:
        observed = tuple(self._cell_column(cell) for cell in winners)
        if observed != token_columns:
            call_index, book, token_index = self._position()
            return _stop(
                "WINNER_TOKEN_COLUMNS",
                call_index=call_index,
                book_ordinal=book,
                token_index=token_index,
                expected=token_columns,
                observed=[-1 if column is None else column for column in observed],
                winners=winners,
            )
        return None

    def _recruit_for_novel(
        self,
        winners: tuple[int, ...],
        token_columns: tuple[int, ...],
    ) -> ScoreResult | None:
        exp = self.expectations
        base = exp.column_count * exp.base_cells_per_column
        exhausted = [
            (position, column)
            for position, column in enumerate(token_columns)
            if int(self.column_available[column]) == 0
        ]
        prospective_next = self.next_reserve_cell + len(exhausted) * exp.cohort_width
        if prospective_next > exp.max_pyramidal_cells:
            call_index, book, token_index = self._position()
            return _stop(
                "RESERVE_CAPACITY",
                call_index=call_index,
                book_ordinal=book,
                token_index=token_index,
                expected=exp.max_pyramidal_cells,
                observed=prospective_next,
            )
        for cohort_offset, (position, column) in enumerate(exhausted):
            start = self.next_reserve_cell + cohort_offset * exp.cohort_width
            stop = start + exp.cohort_width
            if not start <= winners[position] < stop:
                call_index, book, token_index = self._position()
                return _stop(
                    "COHORT_CONTIGUITY",
                    call_index=call_index,
                    book_ordinal=book,
                    token_index=token_index,
                    expected=[start, stop - 1, column],
                    observed=[winners[position], token_columns[position]],
                )
            self.reserve_columns[start - base : stop - base] = column
            self.column_depth[column] += exp.cohort_width
            self.column_available[column] += exp.cohort_width
            self.recruited[column] = True
            self.cohort_stream.update(_COHORT.pack(start, column, exp.cohort_width))
            self.cohort_count += 1
        self.next_reserve_cell = prospective_next
        return self._validate_existing_columns(winners, token_columns)

    def observe(self, reset_before: bool, step: object) -> ScoreResult | None:
        call_index = self.calls
        book_ordinal = self.resets - 1
        token_index = self.local_event
        if type(reset_before) is not bool:
            return _stop(
                "RESET_FLAG",
                call_index=call_index,
                book_ordinal=book_ordinal,
                token_index=token_index,
                expected="bool",
                observed=type(reset_before).__name__,
            )
        try:
            token_id = _step_int(step, "token_id")
            event_count = _step_int(step, "event_count")
            candidate = _step_int(step, "candidate_segments")
            matching = _step_int(step, "matching_segments")
            active = _step_int(step, "active_segments")
            allocated = _step_int(step, "allocated_segments")
            predicted_columns = _step_int(step, "predicted_columns")
            punished = _step_int(step, "punished_segments")
            predicted = step.predicted
            if type(predicted) is not bool:
                raise ValueError("predicted")
        except (AttributeError, ValueError) as exc:
            return _stop(
                "STEP_COUNTER_SHAPE",
                call_index=call_index,
                book_ordinal=book_ordinal,
                token_index=token_index,
                expected="exact non-negative integer counters and bool predicted",
                observed=str(exc),
            )
        columns = self.codebook.get(token_id)
        if columns is None:
            return _stop(
                "UNKNOWN_TOKEN",
                call_index=call_index,
                book_ordinal=book_ordinal,
                token_index=token_index,
                expected="token present in frozen codebook",
                observed=token_id,
            )
        if reset_before:
            book_ordinal = self.resets
            token_index = 0
            self.local_event = 0
        elif self.previous_source_tag is None:
            return _stop(
                "MISSING_BOOK_RESET",
                call_index=call_index,
                book_ordinal=book_ordinal,
                token_index=token_index,
                expected="reset before first step",
                observed="transition",
            )
        physical = self._validate_physical_step(step, columns)
        if isinstance(physical, dict):
            return physical
        winners, _relays, context = physical

        if reset_before:
            reset_observed = (
                event_count,
                candidate,
                matching,
                active,
                allocated,
                predicted_columns,
                int(predicted),
            )
            if reset_observed != (1, 0, 0, 0, 0, 0, 0):
                return _stop(
                    "BOOK_RESET_SEED",
                    call_index=call_index,
                    book_ordinal=book_ordinal,
                    token_index=0,
                    expected=[1, 0, 0, 0, 0, 0, 0],
                    observed=reset_observed,
                )
            column_stop = self._validate_existing_columns(winners, columns)
            if column_stop is not None:
                return column_stop
            packed = _WINNERS.pack(*winners)
            owner = self.owner_index.lookup(winners)
            if owner is not None:
                source_tag = owner
            else:
                seed_index = self.seed_rows.get(packed)
                if seed_index is None:
                    seed_index = len(self.seed_rows)
                    if _SEED_SOURCE_TAG + seed_index > _UINT32_MAX:
                        return _stop(
                            "SEED_SOURCE_CAPACITY",
                            call_index=call_index,
                            book_ordinal=book_ordinal,
                            token_index=0,
                            expected=_UINT32_MAX - _SEED_SOURCE_TAG,
                            observed=seed_index,
                        )
                    self.seed_rows[packed] = seed_index
                    self.seed_fanout.append(0)
                source_tag = _SEED_SOURCE_TAG + seed_index
            self.previous_source_tag = source_tag
            self.previous_winners = winners
            self.resets += 1
            self.book_lengths.append(1)
            self.local_event = 1
        else:
            expected_event = self.local_event + 1
            if event_count != expected_event:
                return _stop(
                    "LOCAL_EVENT_COUNT",
                    call_index=call_index,
                    book_ordinal=book_ordinal,
                    token_index=token_index,
                    expected=expected_event,
                    observed=event_count,
                )
            assert self.previous_source_tag is not None
            assert self.previous_winners is not None
            fanout = self._source_fanout(self.previous_source_tag)
            expected_recognition = _WINNER_WIDTH * fanout
            recognition = (candidate, matching, active)
            if recognition != (expected_recognition,) * 3:
                return _stop(
                    "SOURCE_FANOUT_COUNTS",
                    call_index=call_index,
                    book_ordinal=book_ordinal,
                    token_index=token_index,
                    expected=[expected_recognition] * 3,
                    observed=recognition,
                    source_fanout=fanout,
                )
            known_owner = self.key_index.lookup(self.previous_source_tag, token_id)
            if known_owner is None:
                if (allocated, predicted_columns, int(predicted)) != (_WINNER_WIDTH, 0, 0):
                    return _stop(
                        "NOVEL_KEY_ALLOCATION",
                        call_index=call_index,
                        book_ordinal=book_ordinal,
                        token_index=token_index,
                        expected=[_WINNER_WIDTH, 0, 0],
                        observed=[allocated, predicted_columns, int(predicted)],
                    )
                if self.distinct_keys >= self.expectations.expected_distinct_keys:
                    return _stop(
                        "DISTINCT_KEY_CAPACITY",
                        call_index=call_index,
                        book_ordinal=book_ordinal,
                        token_index=token_index,
                        expected=self.expectations.expected_distinct_keys,
                        observed=self.distinct_keys + 1,
                    )
                packed = _WINNERS.pack(*winners)
                if self.owner_index.lookup(winners) is not None or packed in self.seed_rows:
                    return _stop(
                        "NONINJECTIVE_OWNER_TUPLE",
                        call_index=call_index,
                        book_ordinal=book_ordinal,
                        token_index=token_index,
                        expected="globally new winner tuple",
                        observed=winners,
                    )
                cohort_stop = self._recruit_for_novel(winners, columns)
                if cohort_stop is not None:
                    return cohort_stop
                for column, owner_cell in zip(columns, winners, strict=True):
                    load = int(self.owner_loads[owner_cell])
                    if load >= self.expectations.max_segments_per_cell:
                        return _stop(
                            "OWNER_LOAD_CAP",
                            call_index=call_index,
                            book_ordinal=book_ordinal,
                            token_index=token_index,
                            expected=self.expectations.max_segments_per_cell - 1,
                            observed=load,
                            owner_cell=owner_cell,
                        )
                    load += 1
                    self.owner_loads[owner_cell] = load
                    if load == self.expectations.max_segments_per_cell:
                        self.column_available[column] -= 1
                owner_index = self.distinct_keys
                self.owner_rows[owner_index] = winners
                self.owner_tokens[owner_index] = token_id
                self.owner_index.insert(winners, owner_index)
                self.key_index.insert(self.previous_source_tag, token_id, owner_index)
                self._increment_source_fanout(self.previous_source_tag)
                self.key_owner_stream.update(_TOKEN.pack(token_id))
                self.key_owner_stream.update(_WINNERS.pack(*self.previous_winners))
                self.key_owner_stream.update(_WINNERS.pack(*winners))
                self.allocated_segments += allocated
                source_tag = owner_index
            else:
                expected_winners = tuple(int(value) for value in self.owner_rows[known_owner])
                repeat_observed = (
                    allocated,
                    predicted_columns,
                    int(predicted),
                    int(winners == expected_winners),
                )
                if repeat_observed != (0, _WINNER_WIDTH, 1, 1):
                    return _stop(
                        "REPEAT_OWNER_REPLAY",
                        call_index=call_index,
                        book_ordinal=book_ordinal,
                        token_index=token_index,
                        expected=[0, _WINNER_WIDTH, 1, 1],
                        observed=repeat_observed,
                        expected_winners=expected_winners,
                        observed_winners=winners,
                    )
                column_stop = self._validate_existing_columns(winners, columns)
                if column_stop is not None:
                    return column_stop
                self.repeats += 1
                source_tag = known_owner
            self.previous_source_tag = source_tag
            self.previous_winners = winners
            self.transitions += 1
            self.local_event = event_count
            self.book_lengths[-1] += 1

        self.calls += 1
        self.punished_segments += punished
        self.candidate_maximum = max(self.candidate_maximum, candidate)
        self.token_stream.update(_TOKEN.pack(token_id))
        self.winner_stream.update(_WINNERS.pack(*winners))
        self.context_stream.update(struct.pack("<22I", *context))
        return None

    def finalize(self, identity: object, config: object, codebook_sha256: str) -> ScoreResult:
        exp = self.expectations
        counts_expected = (
            exp.expected_tokens,
            exp.expected_book_resets,
            exp.expected_transitions,
            exp.expected_distinct_keys,
            exp.expected_repeats,
            exp.expected_segments,
        )
        counts_observed = (
            self.calls,
            self.resets,
            self.transitions,
            self.distinct_keys,
            self.repeats,
            self.allocated_segments,
        )
        if counts_observed != counts_expected:
            return _stop(
                "FINAL_TRAIN_COUNTS",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected=counts_expected,
                observed=counts_observed,
            )
        token_digest = self.token_stream.hexdigest()
        if token_digest != exp.train_stream_sha256:
            return _stop(
                "TRAIN_STREAM_IDENTITY",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected=exp.train_stream_sha256,
                observed=token_digest,
            )
        recruited_columns = int(np.count_nonzero(self.recruited))
        maximum_depth = int(self.column_depth.max(initial=0))
        structure_expected = (
            exp.expected_total_cohorts,
            exp.expected_recruited_columns,
            exp.expected_next_reserve_cell,
            exp.expected_max_depth,
        )
        structure_observed = (
            self.cohort_count,
            recruited_columns,
            self.next_reserve_cell,
            maximum_depth,
        )
        if structure_observed != structure_expected:
            return _stop(
                "FINAL_COHORT_STRUCTURE",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected=structure_expected,
                observed=structure_observed,
            )
        identity, identity_error = _identity_source(identity)
        if identity_error is not None:
            return _stop(
                "FINAL_IDENTITY_SHAPE",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected="native identity object, Mapping, or unique tuple pairs",
                observed=identity_error,
            )
        try:
            identity_counts = (
                _identity_int(identity, "segment_count"),
                _identity_int(identity, "cohort_count"),
                _identity_int(identity, "next_reserve_cell"),
                _identity_int(identity, "event_count"),
                _identity_int(identity, "learning_step"),
                _identity_int(identity, "prior_context_length"),
                _identity_int(identity, "winner_context_length"),
                _identity_int(identity, "poison_code"),
            )
        except (AttributeError, ValueError) as exc:
            return _stop(
                "FINAL_IDENTITY_SHAPE",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected="canonical checkpoint identity counters",
                observed=str(exc),
            )
        expected_identity_counts = (
            exp.expected_segments,
            exp.expected_total_cohorts,
            exp.expected_next_reserve_cell,
            exp.expected_final_event_count,
            exp.expected_tokens,
            2 * _WINNER_WIDTH,
            2 * _WINNER_WIDTH,
            0,
        )
        if identity_counts != expected_identity_counts:
            return _stop(
                "FINAL_IDENTITY_COUNTS",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected=expected_identity_counts,
                observed=identity_counts,
            )
        identity_hash_fields = (
            "body_sha256",
            "circuit_identity_sha256",
            "logical_state_sha256",
            "protocol_identity_sha256",
            "semantic_index_sha256",
            "source_bundle_identity_sha256",
        )
        bad_hashes = [
            label
            for label in identity_hash_fields
            if not _is_lower_sha256(
                identity.get(label)
                if isinstance(identity, Mapping)
                else getattr(identity, label, None)
            )
        ]
        if bad_hashes:
            return _stop(
                "FINAL_IDENTITY_HASHES",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected="six lowercase SHA-256 identities",
                observed=bad_hashes,
            )
        required_overlap = int(config.theta_match) // 2
        overlap = first_same_position_overlap_violation(
            self.owner_tokens[: self.distinct_keys],
            self.owner_rows[: self.distinct_keys],
            required_overlap,
        )
        if overlap is not None:
            return _stop(
                "SAME_TOKEN_WINNER_OVERLAP",
                call_index=self.calls,
                book_ordinal=self.resets - 1,
                token_index=self.local_event,
                expected=required_overlap - 1,
                observed=overlap.overlap_count,
                indexed_position_pair=overlap.indexed_position_pair,
                left_index=overlap.left_index,
                left_winners=overlap.left_winners,
                matching_positions=overlap.matching_positions,
                right_index=overlap.right_index,
                right_winners=overlap.right_winners,
                token_id=overlap.token_id,
            )
        owner_loads = self.owner_loads[: self.next_reserve_cell]
        depth_bytes = self.column_depth.astype("<u4", copy=False).tobytes()
        result: ScoreResult = {
            "status": "PASS_R3_F0_TRAIN_STRUCTURE",
            "calls": self.calls,
            "book_resets": self.resets,
            "book_token_lengths": list(self.book_lengths),
            "transition_events": self.transitions,
            "distinct_temporal_keys": self.distinct_keys,
            "repeat_transitions": self.repeats,
            "allocated_segments": self.allocated_segments,
            "punished_segments": self.punished_segments,
            "maximum_candidate_segments": self.candidate_maximum,
            "maximum_source_fanout": self.fanout_maximum,
            "maximum_owner_load": int(owner_loads.max(initial=0)),
            "total_cohorts": self.cohort_count,
            "recruited_columns": recruited_columns,
            "committed_pyramidal_cells": self.next_reserve_cell,
            "maximum_column_depth": maximum_depth,
            "overlap_novel_rows": self.distinct_keys,
            "overlap_rejection_bound": required_overlap,
            "threshold_arm": [int(config.theta_match), int(config.theta_active)],
            "token_codebook_sha256": codebook_sha256,
            "train_stream_sha256": token_digest,
            "winner_stream_sha256": self.winner_stream.hexdigest(),
            "context_stream_sha256": self.context_stream.hexdigest(),
            "key_owner_map_sha256": self.key_owner_stream.hexdigest(),
            "owner_loads_sha256": hashlib.sha256(owner_loads.tobytes()).hexdigest(),
            "column_depths_sha256": hashlib.sha256(depth_bytes).hexdigest(),
            "cohort_log_sha256": self.cohort_stream.hexdigest(),
        }
        for label in identity_hash_fields:
            result[f"final_{label}"] = str(_identity_value(identity, label))
        return result


class R3F0TrainScorer:
    """Public incremental scorer for one authoritative R3-F0 TRAIN execution."""

    __slots__ = (
        "_codebook_sha256",
        "_config",
        "_final_result",
        "_state",
        "_stop_result",
    )

    def __init__(
        self,
        *,
        config: object,
        token_columns: Mapping[int, Sequence[int]]
        | Sequence[tuple[int, Sequence[int]]],
        expectations: R3F0TrainExpectations = CALIBRATION_V3_R3_F0_EXPECTATIONS,
    ) -> None:
        self._config = config
        self._state: _TrainScorer | None = None
        self._codebook_sha256: str | None = None
        self._stop_result: ScoreResult | None = _config_stop(config, expectations)
        self._final_result: ScoreResult | None = None
        if self._stop_result is not None:
            return
        codebook, digest, stop = _normalize_codebook(token_columns, expectations)
        if stop is not None:
            self._stop_result = stop
            return
        assert codebook is not None and digest is not None
        self._codebook_sha256 = digest
        self._state = _TrainScorer(expectations, codebook)

    @property
    def stopped(self) -> bool:
        """Whether the first deterministic scientific STOP has been latched."""

        return self._stop_result is not None

    def _latch(self, stop: ScoreResult) -> ScoreResult:
        if self._stop_result is None:
            self._stop_result = stop
        return self._stop_result

    def observe(self, reset_before: bool, step: object) -> ScoreResult | None:
        """Consume one physical record and return the first STOP, if any."""

        if self._final_result is not None:
            raise RuntimeError("R3F0TrainScorer is already finalized")
        if self._stop_result is not None:
            return self._stop_result
        assert self._state is not None
        stop = self._state.observe(reset_before, step)
        return None if stop is None else self._latch(stop)

    def observe_record(self, record: tuple[bool, object]) -> ScoreResult | None:
        """Consume one ``(reset_before, step)`` record."""

        if self._final_result is not None:
            raise RuntimeError("R3F0TrainScorer is already finalized")
        if self._stop_result is not None:
            return self._stop_result
        assert self._state is not None
        try:
            reset_before, step = record
        except (TypeError, ValueError) as exc:
            return self._latch(
                _stop(
                    "RECORD_SHAPE",
                    call_index=self._state.calls,
                    book_ordinal=self._state.resets - 1,
                    token_index=self._state.local_event,
                    expected="(reset_before, step)",
                    observed=type(exc).__name__,
                )
            )
        return self.observe(reset_before, step)

    def finalize(self, identity: object) -> ScoreResult:
        """Seal scoring against the authoritative terminal checkpoint identity."""

        if self._final_result is not None:
            return self._final_result
        if self._stop_result is not None:
            self._final_result = self._stop_result
            return self._final_result
        assert self._state is not None and self._codebook_sha256 is not None
        result = self._state.finalize(
            identity,
            self._config,
            self._codebook_sha256,
        )
        if result["status"] == "STOP":
            self._latch(result)
        self._final_result = result
        return result


def score_r3_f0_train_structure(
    records: Iterable[tuple[bool, object]],
    *,
    identity: object,
    config: object,
    token_columns: Mapping[int, Sequence[int]] | Sequence[tuple[int, Sequence[int]]],
    expectations: R3F0TrainExpectations = CALIBRATION_V3_R3_F0_EXPECTATIONS,
) -> ScoreResult:
    """Score one ordered TRAIN replay or return its first scientific STOP.

    ``records`` contains ``(reset_before, step)`` pairs in physical call order.
    Only novel temporal keys enter the same-token overlap arrays; exact repeat
    replay therefore cannot manufacture a collision with its stored owner row.
    """

    scorer = R3F0TrainScorer(
        config=config,
        token_columns=token_columns,
        expectations=expectations,
    )
    for record in records:
        if scorer.observe_record(record) is not None:
            break
    return scorer.finalize(identity)
