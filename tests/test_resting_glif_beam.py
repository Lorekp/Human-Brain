from __future__ import annotations

import inspect

import numpy as np
import pytest

from humanbrain import resting_glif_beam as beam_module
from humanbrain.resting_glif_beam import (
    CompiledRestingGlifBeam,
    FixedWinnerSelfInhibition,
)
from humanbrain.resting_glif_race import (
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
)


RACE_STEPS = 600


def _eight_candidate_selector() -> CompiledRestingGlifBeam:
    return CompiledRestingGlifBeam.compile(
        candidate_count=8,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_400.0,
        inhibition_current_pa=3_000.0,
    )


def test_b4_matches_repeated_resting_glif_races_on_distinct_currents() -> None:
    currents = np.asarray(
        (600.0, 2_200.0, 1_000.0, 3_400.0, 1_400.0, 3_000.0, 1_800.0, 2_600.0),
        dtype=np.float64,
    )
    before = currents.copy()
    selector = _eight_candidate_selector()

    reference = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)
    reference_currents = currents.copy()
    expected_races = []
    for _ in range(4):
        race = reference.race_many(reference_currents[None, :])[0]
        expected_races.append(race)
        reference_currents[race.winner_cell] -= 3_000.0

    result = selector.select(currents)

    assert result.winner_cells == (3, 5, 7, 1)
    assert tuple(round_.race for round_ in result.rounds) == tuple(expected_races)
    assert all(
        round_.race.simultaneous_spike_candidates == 1
        and round_.race.spike_step >= 0
        for round_ in result.rounds
    )
    np.testing.assert_array_equal(currents, before)


def test_prior_winners_remain_physically_self_inhibited() -> None:
    currents = np.asarray(
        (600.0, 2_200.0, 1_000.0, 3_400.0, 1_400.0, 3_000.0, 1_800.0, 2_600.0),
        dtype=np.float64,
    )
    result = _eight_candidate_selector().select(currents)

    first = result.winner_cells[0]
    assert first == 3
    assert all(
        round_.currents_before_pa[first] == currents[first] - 3_000.0
        for round_ in result.rounds[1:]
    )
    for winner in result.winner_cells:
        assert result.final_currents_pa[winner] == currents[winner] - 3_000.0
    assert all(
        len(round_.currents_before_pa) == len(currents)
        and len(round_.currents_after_pa) == len(currents)
        for round_ in result.rounds
    )


def test_strict_physical_tie_is_rejected() -> None:
    currents = np.asarray(
        (600.0, 3_400.0, 1_000.0, 3_400.0, 1_400.0, 3_000.0, 1_800.0, 2_600.0),
        dtype=np.float64,
    )

    with pytest.raises(
        RestingGlifRaceFailure,
        match=r"race row 0 produced 2 simultaneous first spikes",
    ):
        _eight_candidate_selector().select(currents)


def test_projection_hash_is_deterministic_and_selection_is_immutable() -> None:
    projection = FixedWinnerSelfInhibition.materialize(
        candidate_count=8,
        inhibition_current_pa=3_000.0,
    )
    first = _eight_candidate_selector()
    second = _eight_candidate_selector()
    narrow = CompiledRestingGlifBeam.compile(
        candidate_count=8,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_400.0,
        inhibition_current_pa=3_000.0,
        beam_width=2,
    )
    changed = CompiledRestingGlifBeam.compile(
        candidate_count=8,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_400.0,
        inhibition_current_pa=3_100.0,
    )
    currents = np.asarray(
        (600.0, 2_200.0, 1_000.0, 3_400.0, 1_400.0, 3_000.0, 1_800.0, 2_600.0),
        dtype=np.float64,
    )
    checkpoint = first.inhibition_projection_sha256

    result_a = first.select(currents)
    result_b = first.select(currents)

    assert projection.weights_pa.shape == (8,)
    assert not projection.weights_pa.flags.writeable
    assert checkpoint == second.inhibition_projection_sha256
    assert checkpoint != changed.inhibition_projection_sha256
    assert narrow.select(currents).winner_cells == (3, 5)
    assert result_a == result_b
    assert result_a.inhibition_projection_before_sha256 == checkpoint
    assert result_a.inhibition_projection_after_sha256 == checkpoint
    assert first.live_inhibition_projection_sha256 == checkpoint
    with pytest.raises(
        ValueError,
        match="must exceed the declared current span",
    ):
        CompiledRestingGlifBeam.compile(
            candidate_count=8,
            step_count=RACE_STEPS,
            min_current_pa=600.0,
            max_current_pa=3_400.0,
            inhibition_current_pa=2_800.0,
        )
    with pytest.raises(ValueError, match="left the compiled current envelope"):
        first.select(np.asarray((599.0, *currents[1:]), dtype=np.float64))


def test_four_hundred_candidate_first_ply_smoke_keeps_full_population() -> None:
    candidate_count = 400
    currents = np.full(candidate_count, 600.0, dtype=np.float64)
    currents[17] = 1_800.0
    currents[101] = 2_200.0
    currents[233] = 2_600.0
    currents[399] = 3_000.0
    selector = CompiledRestingGlifBeam.compile(
        candidate_count=candidate_count,
        step_count=RACE_STEPS,
        min_current_pa=600.0,
        max_current_pa=3_000.0,
        inhibition_current_pa=2_500.0,
    )

    result = selector.select(currents)

    assert result.winner_cells == (399, 233, 101, 17)
    assert len(result.rounds) == 4
    assert all(
        len(round_.currents_before_pa) == candidate_count
        and len(round_.currents_after_pa) == candidate_count
        and round_.race.simultaneous_spike_candidates == 1
        for round_ in result.rounds
    )
    source = inspect.getsource(beam_module.CompiledRestingGlifBeam.select)
    for forbidden in (
        "sort(",
        "sorted(",
        "argsort",
        "argpartition",
        "argmax",
        "delete(",
    ):
        assert forbidden not in source
