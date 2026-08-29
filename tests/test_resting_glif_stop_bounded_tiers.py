from __future__ import annotations

import inspect

import numpy as np

from humanbrain import resting_glif_stop_bounded_tiers as stop_module
from humanbrain.resting_glif_stop_bounded_tiers import (
    CompiledStopBoundedRestingGlifTiers,
    STOP_FRONTIER_REACHED,
)


RACE_STEPS = 600


def _selector() -> CompiledStopBoundedRestingGlifTiers:
    return CompiledStopBoundedRestingGlifTiers.compile(
        candidate_count=8,
        step_count=RACE_STEPS,
        min_current_pa=-10_000.0,
        max_current_pa=3_400.0,
        inhibition_current_pa=14_000.0,
    )


def test_stop_is_the_physical_frontier_and_lower_cells_are_not_awaited() -> None:
    currents = np.asarray(
        (3_400.0, 3_400.0, 2_600.0, 1_800.0, 1_000.0, -10_000.0, 600.0, 1_500.0),
        dtype=np.float64,
    )

    result = _selector().enumerate(currents, stop_cell=7)

    assert result.status == STOP_FRONTIER_REACHED
    assert result.rounds[-1].stop_frontier
    assert 7 in result.rounds[-1].tier_cells
    assert all(not round_.stop_frontier for round_ in result.rounds[:-1])
    assert {0, 1, 2}.issubset(result.admissible_candidate_cells)
    assert {4, 5, 6}.isdisjoint(result.admissible_candidate_cells)
    assert {4, 5, 6}.issubset(result.excluded_candidate_cells)


def test_candidate_tied_with_stop_is_retained_in_complete_frontier_tier() -> None:
    currents = np.asarray(
        (3_400.0, 2_600.0, 600.0, 1_500.0, 1_000.0, -10_000.0, 600.0, 1_500.0),
        dtype=np.float64,
    )

    result = _selector().enumerate(currents, stop_cell=7)

    assert result.rounds[-1].tier_cells == (3, 7)
    assert result.rounds[-1].race.simultaneous_spike_candidates == 2
    assert 3 in result.admissible_candidate_cells
    assert 3 not in result.excluded_candidate_cells


def test_leading_stop_ends_before_negative_nonspiking_candidates() -> None:
    currents = np.full(8, -10_000.0, dtype=np.float64)
    currents[7] = 650.0

    result = _selector().enumerate(currents, stop_cell=7)

    assert len(result.rounds) == 1
    assert result.rounds[0].tier_cells == (7,)
    assert result.rounds[0].stop_frontier
    assert result.admissible_candidate_cells == ()
    assert result.excluded_candidate_cells == tuple(range(7))


def test_complete_tier_inhibition_and_projection_hash_are_immutable() -> None:
    currents = np.asarray(
        (3_400.0, 2_600.0, 3_400.0, 1_800.0, 1_000.0, -10_000.0, 600.0, 1_500.0),
        dtype=np.float64,
    )
    selector = _selector()
    checkpoint = selector.inhibition_projection_sha256

    result = selector.enumerate(currents, stop_cell=7)

    assert result.rounds[0].tier_cells == (0, 2)
    assert all(
        result.final_currents_pa[cell] == currents[cell] - 14_000.0
        for cell in result.admissible_candidate_cells
    )
    assert selector.projection_storage_shape == (8,)
    assert result.inhibition_projection_before_sha256 == checkpoint
    assert result.inhibition_projection_after_sha256 == checkpoint
    assert selector.live_inhibition_projection_sha256 == checkpoint


def test_stop_bounded_enumeration_has_no_host_top_k_or_candidate_deletion() -> None:
    source = inspect.getsource(
        stop_module.CompiledStopBoundedRestingGlifTiers.enumerate
    )
    for forbidden in (
        "sort(",
        "sorted(",
        "argsort",
        "argpartition",
        "argmax",
        "delete(",
    ):
        assert forbidden not in source
