from __future__ import annotations

import inspect

import numpy as np

from humanbrain import resting_glif_tier_beam as tier_module
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_tier_beam import (
    CompiledRestingGlifTierBeam,
    FixedTierSelfInhibition,
    RETAINED,
    STOP_LEADING_TIER,
)


RACE_STEPS = 600


def _selector(*, beam_width: int = 4) -> CompiledRestingGlifTierBeam:
    return CompiledRestingGlifTierBeam.compile(
        candidate_count=8,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_400.0,
        inhibition_current_pa=3_000.0,
        beam_width=beam_width,
    )


def _stepwise_first_tier(currents: np.ndarray) -> tuple[int, tuple[int, ...]]:
    population = AdaptiveGlifPopulation(len(currents))
    for step in range(RACE_STEPS):
        fired = np.flatnonzero(population.step(currents))
        if fired.size:
            return step, tuple(int(cell) for cell in fired)
    raise AssertionError("stepwise fixture did not spike")


def test_first_compiled_tier_matches_stepwise_resting_glif_ensemble() -> None:
    currents = np.asarray(
        (3_400.0, 600.0, 3_400.0, 1_800.0, 2_600.0, 3_400.0, 2_200.0, 1_000.0),
        dtype=np.float64,
    )
    expected_step, expected_cells = _stepwise_first_tier(currents)

    result = _selector(beam_width=3).select(currents)

    assert result.status == RETAINED
    assert result.rounds[0].race.spike_step == expected_step
    assert result.rounds[0].tier_cells == expected_cells == (0, 2, 5)
    assert result.rounds[0].race.simultaneous_spike_candidates == 3
    assert result.retained_candidate_cells == expected_cells


def test_tier_crossing_b_cutoff_expands_every_simultaneous_spiker() -> None:
    currents = np.asarray(
        (3_400.0, 3_000.0, 600.0, 1_400.0, 3_000.0, 1_800.0, 3_000.0, 1_000.0),
        dtype=np.float64,
    )

    result = _selector(beam_width=2).select(currents)

    assert result.status == RETAINED
    assert tuple(round_.tier_cells for round_ in result.rounds) == ((0,), (1, 4, 6))
    assert result.retained_candidate_cells == (0, 1, 4, 6)
    assert len(result.retained_candidate_cells) > 2


def test_every_tier_member_remains_physically_self_inhibited() -> None:
    currents = np.asarray(
        (3_400.0, 3_000.0, 600.0, 1_400.0, 3_000.0, 1_800.0, 3_000.0, 1_000.0),
        dtype=np.float64,
    )
    result = _selector(beam_width=2).select(currents)

    first_tier = result.rounds[0].tier_cells
    second_before = result.rounds[1].currents_before_pa
    assert all(second_before[cell] == currents[cell] - 3_000.0 for cell in first_tier)
    for cell in result.retained_cells:
        assert result.final_currents_pa[cell] == currents[cell] - 3_000.0
    assert all(
        len(round_.currents_before_pa) == len(currents)
        and len(round_.currents_after_pa) == len(currents)
        for round_ in result.rounds
    )


def test_stop_cell_participates_in_and_closes_the_leading_tier() -> None:
    currents = np.asarray(
        (600.0, 1_000.0, 3_400.0, 1_400.0, 1_800.0, 2_200.0, 2_600.0, 3_400.0),
        dtype=np.float64,
    )

    result = _selector().select(currents, stop_cell=7)

    assert result.status == STOP_LEADING_TIER
    assert len(result.rounds) == 1
    assert result.rounds[0].tier_cells == (2, 7)
    assert result.rounds[0].leading_stop
    assert result.retained_cells == (2, 7)
    assert result.retained_candidate_cells == ()


def test_batched_tier_beam_is_exactly_the_scalar_ledger() -> None:
    currents = np.asarray(
        (
            (
                3_400.0,
                3_000.0,
                600.0,
                1_400.0,
                3_000.0,
                1_800.0,
                3_000.0,
                1_000.0,
            ),
            (
                600.0,
                1_000.0,
                3_400.0,
                1_400.0,
                1_800.0,
                2_200.0,
                2_600.0,
                3_400.0,
            ),
            (
                3_400.0,
                600.0,
                3_400.0,
                1_800.0,
                2_600.0,
                3_400.0,
                2_200.0,
                1_000.0,
            ),
        ),
        dtype=np.float64,
    )
    selector = _selector(beam_width=2)

    expected = tuple(selector.select(row, stop_cell=7) for row in currents)
    observed = selector.select_many(currents, stop_cell=7)

    assert observed == expected
    assert observed[0].rounds[1].tier_cells == (1, 4, 6)
    assert observed[1].status == STOP_LEADING_TIER
    assert observed[1].rounds[0].tier_cells == (2, 7)


def test_batched_tier_beam_handles_current_24_by_25_match_shape() -> None:
    selector = CompiledRestingGlifTierBeam.compile(
        candidate_count=25,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_000.0,
        inhibition_current_pa=7_000.0,
        beam_width=1,
    )
    currents = np.full((24, 25), 600.0, dtype=np.float64)
    currents[:, 24] = 2_360.0
    currents[np.arange(24), np.arange(24)] = 2_520.0

    observed = selector.select_many(currents, stop_cell=24)

    assert len(observed) == 24
    assert tuple(result.retained_candidate_cells for result in observed) == tuple(
        (row,) for row in range(24)
    )
    assert all(result.status == RETAINED for result in observed)
    assert all(
        result.inhibition_projection_before_sha256
        == result.inhibition_projection_after_sha256
        == selector.inhibition_projection_sha256
        for result in observed
    )


def test_sparse_projection_hash_is_immutable_and_selector_has_no_host_top_k() -> None:
    projection = FixedTierSelfInhibition.materialize(
        candidate_count=400,
        inhibition_current_pa=2_500.0,
    )
    first = CompiledRestingGlifTierBeam.compile(
        candidate_count=400,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_000.0,
        inhibition_current_pa=2_500.0,
    )
    second = CompiledRestingGlifTierBeam.compile(
        candidate_count=400,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_000.0,
        inhibition_current_pa=2_500.0,
    )
    currents = np.full(400, 600.0, dtype=np.float64)
    currents[[17, 101]] = 3_000.0
    currents[[233, 399]] = 2_200.0

    result = first.select(currents)

    assert projection.weights_pa.shape == (400,)
    assert not projection.weights_pa.flags.writeable
    assert first.inhibition_projection_sha256 == second.inhibition_projection_sha256
    assert result.inhibition_projection_before_sha256 == first.inhibition_projection_sha256
    assert result.inhibition_projection_after_sha256 == first.inhibition_projection_sha256
    assert result.retained_candidate_cells == (17, 101, 233, 399)
    assert all(len(round_.currents_before_pa) == 400 for round_ in result.rounds)
    source = inspect.getsource(tier_module.CompiledRestingGlifTierBeam.select)
    for forbidden in (
        "sort(",
        "sorted(",
        "argsort",
        "argpartition",
        "argmax",
        "delete(",
    ):
        assert forbidden not in source
