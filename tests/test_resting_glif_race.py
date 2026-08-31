from __future__ import annotations

import re

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


def _dense_compiled_race(
    compiled: CompiledRestingGlifRace,
    currents_pa: np.ndarray,
    *,
    require_unique: bool = True,
) -> tuple[RestingGlifRaceResult, ...]:
    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    winning_current = np.max(currents, axis=1)
    crossing = compiled._unit_voltage[:, None] * winning_current[None, :] > (
        compiled.parameters.threshold_inf_relative_mv
        + compiled._unit_voltage_threshold[:, None] * winning_current[None, :]
    )
    step_count = int(compiled._unit_voltage.size)
    crossed = np.any(crossing, axis=0)
    first_steps = np.argmax(crossing, axis=0).astype(np.intp, copy=False)
    first_steps = np.where(crossed, first_steps, step_count)

    safe_steps = np.minimum(first_steps, step_count - 1)
    fired = compiled._unit_voltage[safe_steps, None] * currents > (
        compiled.parameters.threshold_inf_relative_mv
        + compiled._unit_voltage_threshold[safe_steps, None] * currents
    )
    simultaneous = np.count_nonzero(fired, axis=1)
    candidate_ids = np.arange(currents.shape[1], dtype=np.intp)[None, :]
    winners = np.min(
        np.where(fired, candidate_ids, currents.shape[1]),
        axis=1,
    )

    results: list[RestingGlifRaceResult] = []
    for row in range(currents.shape[0]):
        if first_steps[row] == step_count:
            raise RestingGlifRaceFailure(
                f"race row {row} produced no spike within its fixed bound"
            )
        count = int(simultaneous[row])
        if require_unique and count != 1:
            raise RestingGlifRaceFailure(
                f"race row {row} produced {count} simultaneous first spikes"
            )
        results.append(
            RestingGlifRaceResult(
                winner_cell=int(winners[row]),
                spike_step=int(first_steps[row]),
                simultaneous_spike_candidates=count,
            )
        )
    return tuple(results)


def _threshold_crossover_rows(
    compiled: CompiledRestingGlifRace, row_count: int
) -> np.ndarray:
    steps = np.asarray((0, 30, 72, 100), dtype=np.intp)
    gain = compiled._unit_voltage[steps] - compiled._unit_voltage_threshold[steps]
    boundary = compiled.parameters.threshold_inf_relative_mv / gain
    crossing_currents = np.column_stack(
        (
            np.nextafter(boundary, -np.inf),
            boundary,
            np.nextafter(boundary, np.inf),
        )
    ).reshape(-1)

    currents = np.zeros((row_count, CANDIDATE_COUNT), dtype=np.float64)
    row_ids = np.arange(row_count)
    winner_ids = (row_ids * 17) % CANDIDATE_COUNT
    currents[row_ids, winner_ids] = crossing_currents[
        row_ids % crossing_currents.size
    ]
    return currents


def _assert_dense_parity(
    compiled: CompiledRestingGlifRace,
    currents: np.ndarray,
    *,
    require_unique: bool = True,
) -> None:
    try:
        expected = _dense_compiled_race(
            compiled,
            currents,
            require_unique=require_unique,
        )
    except RestingGlifRaceFailure as expected_error:
        with pytest.raises(
            RestingGlifRaceFailure,
            match=rf"^{re.escape(str(expected_error))}$",
        ):
            compiled.race_many(currents, require_unique=require_unique)
    else:
        assert compiled.race_many(currents, require_unique=require_unique) == expected


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


@pytest.mark.parametrize("row_count", (64, 256, 4096))
def test_binary_first_crossing_matches_dense_threshold_crossover_rows(
    row_count: int,
) -> None:
    compiled = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    currents = _threshold_crossover_rows(compiled, row_count)
    before = currents.copy()

    _assert_dense_parity(compiled, currents)

    np.testing.assert_array_equal(currents, before)


def test_binary_first_crossing_matches_dense_co_spike_and_no_spike_failures() -> (
    None
):
    compiled = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    currents = _threshold_crossover_rows(compiled, 64)

    co_spike = currents.copy()
    co_spike_row = 7
    winner = int(np.argmax(co_spike[co_spike_row]))
    co_spike[co_spike_row, (winner + 1) % CANDIDATE_COUNT] = co_spike[
        co_spike_row, winner
    ]
    _assert_dense_parity(compiled, co_spike, require_unique=False)
    _assert_dense_parity(compiled, co_spike)

    no_spike = currents.copy()
    no_spike[5] = 0.0
    _assert_dense_parity(compiled, no_spike)


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
