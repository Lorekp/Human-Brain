from __future__ import annotations

import numpy as np
import pytest

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import (
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
    RestingGlifRaceResult,
)


RACE_STEPS = 600
CANDIDATE_COUNT = 65


def _stepwise_race(currents_pa: np.ndarray) -> RestingGlifRaceResult:
    population = AdaptiveGlifPopulation(CANDIDATE_COUNT)
    for step in range(RACE_STEPS):
        fired = np.flatnonzero(population.step(currents_pa))
        if fired.size:
            return RestingGlifRaceResult(
                winner_cell=int(fired[0]),
                spike_step=step,
                simultaneous_spike_candidates=int(fired.size),
            )
    raise AssertionError("stepwise reference did not spike")


def _stepwise_accumulated_spike_mask(
    currents_pa: np.ndarray, *, step_count: int
) -> np.ndarray:
    population = AdaptiveGlifPopulation(len(currents_pa))
    seen = np.zeros(population.cell_count, dtype=np.bool_)
    for _ in range(step_count):
        seen |= population.step(currents_pa)
    return seen


def test_compiled_six_row_batch_matches_fresh_stepwise_glif5_races() -> None:
    winner_cells = (0, 1, 7, 31, 42, 64)
    currents = np.full(
        (len(winner_cells), CANDIDATE_COUNT),
        600.0,
        dtype=np.float64,
    )
    currents[np.arange(len(winner_cells)), winner_cells] = 1_800.0
    before = currents.copy()
    expected = tuple(_stepwise_race(row) for row in currents)

    compiled = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    actual = compiled.race_many(currents)

    assert actual == expected == tuple(
        RestingGlifRaceResult(winner, 72, 1) for winner in winner_cells
    )
    np.testing.assert_array_equal(currents, before)


def test_compiled_batch_exposes_co_spikes_and_unique_mode_rejects_them() -> None:
    currents = np.full((6, CANDIDATE_COUNT), 600.0, dtype=np.float64)
    currents[:, 3] = 1_800.0
    currents[2, 11] = 1_800.0

    compiled = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    reported = compiled.race_many(currents, require_unique=False)

    assert reported[2] == _stepwise_race(currents[2])
    assert reported[2] == RestingGlifRaceResult(
        winner_cell=3,
        spike_step=72,
        simultaneous_spike_candidates=2,
    )
    with pytest.raises(
        RestingGlifRaceFailure,
        match=r"race row 2 produced 2 simultaneous first spikes",
    ):
        compiled.race_many(currents)


def test_compiled_accumulated_mask_matches_stepwise_glif5_for_finite_currents() -> None:
    step_count = 30
    random_currents = np.random.default_rng(0).uniform(-5_000.0, 5_000.0, 257)
    currents = np.ascontiguousarray(
        np.concatenate(
            (
                np.asarray(
                    (-5_000.0, -1.0, 0.0, 1.0, 2_500.0, 4_375.0, 5_000.0)
                ),
                random_currents,
            )
        )
    )
    expected = _stepwise_accumulated_spike_mask(
        currents, step_count=step_count
    )

    compiled = CompiledRestingGlifRace.compile(step_count=step_count)
    actual = compiled.accumulated_spike_mask(currents)

    np.testing.assert_array_equal(actual, expected)
    assert tuple(actual[:7]) == (False, False, False, False, False, True, True)
    assert actual.dtype == np.dtype(np.bool_)
    assert actual.flags.c_contiguous
    assert not actual.flags.writeable


def test_compiled_accumulated_mask_rejects_invalid_current_vectors() -> None:
    compiled = CompiledRestingGlifRace.compile(step_count=30)

    for invalid in (
        np.asarray(1.0),
        np.empty(0),
        np.zeros((1, 1)),
        np.asarray((np.nan,)),
        np.asarray((np.inf,)),
    ):
        with pytest.raises(ValueError, match="currents_pa"):
            compiled.accumulated_spike_mask(invalid)
