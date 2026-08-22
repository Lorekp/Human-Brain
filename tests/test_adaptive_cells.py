from pathlib import Path

import numpy as np

from humanbrain.cells import (
    AdaptiveGlifPopulation,
    AllenGlif5Parameters,
    simulate_glif5_current,
)
from humanbrain.human_cell_identity import HUMAN_GLIF5_IDENTITY


ROOT = Path(__file__).resolve().parents[1]
ALLEN_GLIF5_CONFIG = ROOT / "data" / "allen" / "523000114_neuron_config.json"


def test_adaptive_cell_parameters_match_pinned_allen_human_artifact() -> None:
    loaded = AllenGlif5Parameters.from_allen_config(ALLEN_GLIF5_CONFIG)
    assert loaded == AllenGlif5Parameters()
    assert loaded.model_id == HUMAN_GLIF5_IDENTITY.model_id
    assert loaded.specimen_id == HUMAN_GLIF5_IDENTITY.specimen_id
    assert np.isclose(loaded.membrane_tau_ms, 78.59019688076135)


def test_adaptive_human_cell_rests_and_raises_its_threshold_after_spike() -> None:
    cell = AllenGlif5Parameters()
    rest = simulate_glif5_current(cell, 0.0, duration_ms=25.0)
    assert np.nanmax(np.abs(rest.voltage_mv - cell.rest_mv)) < 1e-10

    population = AdaptiveGlifPopulation(1, cell)
    initial_threshold = float(population.threshold_relative_mv[0])
    while not population.step(500.0)[0]:
        pass
    assert population.threshold_relative_mv[0] > initial_threshold
    assert population.state.after_spike_currents_pa[0].sum() < 0.0
    assert population.state.refractory_steps[0] == cell.spike_cut_steps


def test_adaptive_population_keeps_cell_state_separate() -> None:
    population = AdaptiveGlifPopulation(3)
    observed = np.zeros(3, dtype=bool)
    for _ in range(400):
        observed |= population.step([500.0, 0.0, 0.0])
    assert observed.tolist() == [True, False, False]
    assert population.state.threshold_spike_mv[0] > 0.0
    assert population.state.threshold_spike_mv[1] == 0.0
