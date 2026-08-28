"""Exact bounded-memory same-position winner-overlap scoring for R3-F0."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np
from numpy.typing import NDArray

WINNER_WIDTH: Final = 11
REQUIRED_OVERLAPS: Final = frozenset({5, 8, 9, 10})
POSITION_GROUPS: Final = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10))
INDEXED_POSITION_PAIRS: Final = tuple(
    pair for group in POSITION_GROUPS for pair in combinations(group, 2)
)
_CANDIDATE_BLOCK_ROWS: Final = 256
_UINT32_MAX: Final = np.iinfo(np.uint32).max


@dataclass(frozen=True, slots=True)
class SamePositionOverlapEvidence:
    """The lexicographically first violating input-row pair."""

    left_index: int
    right_index: int
    token_id: int
    required_overlap: int
    overlap_count: int
    matching_positions: tuple[int, ...]
    indexed_position_pair: tuple[int, int]
    left_winners: tuple[int, ...]
    right_winners: tuple[int, ...]


def _normalized_token_ids(
    token_ids: NDArray[np.integer] | list[int] | tuple[int, ...],
    row_count: int,
) -> NDArray[np.uint32]:
    values = np.asarray(token_ids)
    if values.ndim != 1 or values.shape[0] != row_count:
        raise ValueError("token_ids must have exact shape [N]")
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError("token_ids must contain exact integers")
    if values.size and (
        (values.dtype.kind == "i" and int(values.min()) < 0)
        or int(values.max()) > _UINT32_MAX
    ):
        raise ValueError("token_ids must fit uint32")
    return values.astype(np.uint32, copy=False)


def _first_violation_in_candidate_group(
    group_indices: NDArray[np.int64],
    winner_rows: NDArray[np.uint32],
    required_overlap: int,
) -> tuple[int, int, int] | None:
    """Verify one exact-key group without materializing all row pairs."""

    group_size = int(group_indices.size)
    for left_start in range(0, group_size - 1, _CANDIDATE_BLOCK_ROWS):
        left_stop = min(left_start + _CANDIDATE_BLOCK_ROWS, group_size)
        left_indices = group_indices[left_start:left_stop]
        left_rows = winner_rows[left_indices]
        best_in_left_block: tuple[int, int, int] | None = None

        for right_start in range(left_start, group_size, _CANDIDATE_BLOCK_ROWS):
            right_stop = min(right_start + _CANDIDATE_BLOCK_ROWS, group_size)
            right_indices = group_indices[right_start:right_stop]
            right_rows = winner_rows[right_indices]
            overlap_counts = np.count_nonzero(
                left_rows[:, np.newaxis, :] == right_rows[np.newaxis, :, :],
                axis=2,
            )
            violations = overlap_counts >= required_overlap
            violations &= left_indices[:, np.newaxis] < right_indices[np.newaxis, :]
            flat_hits = np.flatnonzero(violations)
            if not flat_hits.size:
                continue
            left_offset, right_offset = divmod(
                int(flat_hits[0]),
                int(right_indices.size),
            )
            candidate = (
                int(left_indices[left_offset]),
                int(right_indices[right_offset]),
                int(overlap_counts[left_offset, right_offset]),
            )
            if (
                best_in_left_block is None
                or candidate[:2] < best_in_left_block[:2]
            ):
                best_in_left_block = candidate

        if best_in_left_block is not None:
            return best_in_left_block
    return None


def first_same_position_overlap_violation(
    token_ids: NDArray[np.integer] | list[int] | tuple[int, ...],
    winner_rows: NDArray[np.uint32],
    required_overlap: int,
) -> SamePositionOverlapEvidence | None:
    """Return the first same-token row pair matching in at least ``r`` positions.

    "First" is the lexicographically smallest ``(left_index, right_index)``.
    Four fixed position groups of sizes 3, 3, 3, and 2 contribute their ten
    within-group pairs. Any five matching positions contain one such pair by
    pigeonhole, so these exact keys cover every supported threshold without
    materializing ``C(11, r)`` subsets. Candidate groups are sorted on the
    collision-free key ``(token, winner[i], winner[j])`` and verified in fixed
    blocks against all eleven positions.
    """

    if type(required_overlap) is not int or required_overlap not in REQUIRED_OVERLAPS:
        raise ValueError("required_overlap must be one of 5, 8, 9, or 10")
    rows = np.asarray(winner_rows)
    if rows.dtype != np.uint32:
        raise TypeError("winner_rows must have dtype uint32")
    if rows.ndim != 2 or rows.shape[1] != WINNER_WIDTH:
        raise ValueError("winner_rows must have exact shape [N, 11]")
    tokens = _normalized_token_ids(token_ids, int(rows.shape[0]))
    if rows.shape[0] < 2:
        return None

    row_indices = np.arange(rows.shape[0], dtype=np.int64)
    best: tuple[int, int, int] | None = None
    for left_position, right_position in INDEXED_POSITION_PAIRS:
        order = np.lexsort(
            (
                row_indices,
                rows[:, right_position],
                rows[:, left_position],
                tokens,
            )
        )
        sorted_tokens = tokens[order]
        sorted_left = rows[order, left_position]
        sorted_right = rows[order, right_position]
        same_key = (
            (sorted_tokens[1:] == sorted_tokens[:-1])
            & (sorted_left[1:] == sorted_left[:-1])
            & (sorted_right[1:] == sorted_right[:-1])
        )
        group_starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(~same_key) + 1)
        )
        group_stops = np.concatenate(
            (group_starts[1:], np.array([order.size], dtype=np.int64))
        )
        candidate_groups = group_stops - group_starts >= 2
        for start, stop in zip(
            group_starts[candidate_groups],
            group_stops[candidate_groups],
            strict=True,
        ):
            candidate = _first_violation_in_candidate_group(
                order[start:stop],
                rows,
                required_overlap,
            )
            if candidate is not None and (best is None or candidate[:2] < best[:2]):
                best = candidate

    if best is None:
        return None
    left_index, right_index, overlap_count = best
    matching_positions = tuple(
        int(position)
        for position in np.flatnonzero(rows[left_index] == rows[right_index])
    )
    indexed_position_pair = next(
        pair
        for pair in INDEXED_POSITION_PAIRS
        if pair[0] in matching_positions and pair[1] in matching_positions
    )
    return SamePositionOverlapEvidence(
        left_index=left_index,
        right_index=right_index,
        token_id=int(tokens[left_index]),
        required_overlap=required_overlap,
        overlap_count=overlap_count,
        matching_positions=matching_positions,
        indexed_position_pair=indexed_position_pair,
        left_winners=tuple(int(value) for value in rows[left_index]),
        right_winners=tuple(int(value) for value in rows[right_index]),
    )
