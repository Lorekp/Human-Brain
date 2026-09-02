from __future__ import annotations

import numpy as np
import pytest
from humanbrain.cortex import DendriticCortex
from humanbrain.sequential_first_spike import DendriticFirstSpikeBatch


def _race_rows(row_count: int) -> tuple[np.ndarray, np.ndarray, int]:
    cell_count = 512
    stop_cell = cell_count - 1
    rows = np.arange(row_count, dtype=np.int64)
    cell_ids = np.column_stack((rows % stop_cell, np.full(row_count, stop_cell)))
    currents = np.full((row_count, 2), 800.0, dtype=np.float64)
    currents[rows % 2 == 0, 0] = 1_800.0
    currents[rows % 2 == 1, 1] = 1_800.0
    return cell_ids, currents, stop_cell


def test_batch_matches_scalar_cell_races_and_ordered_stop_choices() -> None:
    cortex = DendriticCortex(cell_count=512, seed=29)
    cell_ids, currents, stop_cell = _race_rows(257)

    scalar = tuple(
        cortex.race_currents_first_spike(cell_ids[row], currents[row])
        for row in range(cell_ids.shape[0])
    )
    batched = DendriticFirstSpikeBatch(cortex).race_many(cell_ids, currents)

    assert batched == scalar
    assert any(race.winner == stop_cell for race in batched)
    assert any(race.winner != stop_cell for race in batched)


def test_batch_matches_scalar_multi_candidate_races() -> None:
    cortex = DendriticCortex(cell_count=512, seed=41)
    rows = np.arange(129, dtype=np.int64)
    offsets = np.arange(5, dtype=np.int64)
    cell_ids = (rows[:, None] * 7 + offsets[None, :]) % 512
    currents = np.random.default_rng(73).uniform(650.0, 2_100.0, cell_ids.shape)

    scalar = tuple(
        cortex.race_currents_first_spike(cell_ids[row], currents[row])
        for row in range(cell_ids.shape[0])
    )
    batched = DendriticFirstSpikeBatch(cortex).race_many(cell_ids, currents)

    assert batched == scalar


def test_batch_preserves_the_scalar_no_spike_failure() -> None:
    cortex = DendriticCortex(cell_count=8, seed=3)
    batch = DendriticFirstSpikeBatch(cortex)
    cell_ids = np.array([[0, 7]], dtype=np.int64)
    currents = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(RuntimeError, match="active human cells did not spike"):
        cortex.race_currents_first_spike(cell_ids[0], currents[0])
    with pytest.raises(RuntimeError, match="active human cells did not spike"):
        batch.race_many(cell_ids, currents)
