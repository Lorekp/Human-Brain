from itertools import combinations

import numpy as np
import pytest
from experiments.r3_pilot.calibration_v3_overlap_scoring import (
    INDEXED_POSITION_PAIRS,
    POSITION_GROUPS,
    first_same_position_overlap_violation,
)


def _brute_first(
    token_ids: np.ndarray,
    winner_rows: np.ndarray,
    required_overlap: int,
) -> tuple[int, int, tuple[int, ...]] | None:
    for left_index in range(len(token_ids) - 1):
        for right_index in range(left_index + 1, len(token_ids)):
            if token_ids[left_index] != token_ids[right_index]:
                continue
            matching = tuple(
                int(position)
                for position in np.flatnonzero(
                    winner_rows[left_index] == winner_rows[right_index]
                )
            )
            if len(matching) >= required_overlap:
                return left_index, right_index, matching
    return None


def test_four_group_pair_index_covers_every_five_position_subset() -> None:
    assert POSITION_GROUPS == ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10))
    assert len(INDEXED_POSITION_PAIRS) == 10
    assert len(set(INDEXED_POSITION_PAIRS)) == 10

    indexed = set(INDEXED_POSITION_PAIRS)
    for subset in combinations(range(11), 5):
        assert any(pair in indexed for pair in combinations(subset, 2))


def test_scoring_input_contract_is_exact() -> None:
    rows = np.zeros((2, 11), dtype=np.uint32)
    with pytest.raises(TypeError, match="dtype uint32"):
        first_same_position_overlap_violation([0, 0], rows.astype(np.uint64), 5)
    with pytest.raises(ValueError, match=r"shape \[N, 11\]"):
        first_same_position_overlap_violation([0, 0], rows[:, :10], 5)
    with pytest.raises(ValueError, match="one of 5, 8, 9, or 10"):
        first_same_position_overlap_violation([0, 0], rows, 6)


def test_exact_index_matches_randomized_brute_force() -> None:
    random = np.random.default_rng(20260828)
    for _trial in range(80):
        row_count = int(random.integers(2, 42))
        token_ids = random.integers(0, 5, size=row_count, dtype=np.uint32)
        winner_rows = random.integers(
            0,
            128,
            size=(row_count, 11),
            dtype=np.uint32,
        )
        for _injection in range(4):
            left_index, right_index = random.choice(row_count, size=2, replace=False)
            token_ids[right_index] = token_ids[left_index]
            positions = random.choice(
                11,
                size=int(random.integers(0, 12)),
                replace=False,
            )
            winner_rows[right_index, positions] = winner_rows[left_index, positions]

        for required_overlap in (5, 8, 9, 10):
            expected = _brute_first(token_ids, winner_rows, required_overlap)
            observed = first_same_position_overlap_violation(
                token_ids,
                winner_rows,
                required_overlap,
            )
            if expected is None:
                assert observed is None
                continue
            assert observed is not None
            left_index, right_index, matching_positions = expected
            assert (observed.left_index, observed.right_index) == (
                left_index,
                right_index,
            )
            assert observed.token_id == int(token_ids[left_index])
            assert observed.overlap_count == len(matching_positions)
            assert observed.matching_positions == matching_positions
            assert set(observed.indexed_position_pair).issubset(matching_positions)


def test_large_adversarial_candidate_group_has_bounded_exact_no_violation() -> None:
    row_count = 4_096
    token_ids = np.zeros(row_count, dtype=np.uint32)
    winner_rows = np.empty((row_count, 11), dtype=np.uint32)
    winner_rows[:, 0] = 7
    winner_rows[:, 1] = 11
    row_offsets = np.arange(row_count, dtype=np.uint32) * 16 + 100
    for position in range(2, 11):
        winner_rows[:, position] = row_offsets + position

    assert first_same_position_overlap_violation(token_ids, winner_rows, 5) is None
