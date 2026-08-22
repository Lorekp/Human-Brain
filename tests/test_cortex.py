from pathlib import Path

import numpy as np

from humanbrain.cells import AllenGlif3Parameters, simulate_glif3_current
from humanbrain.cortex import DendriticCortex, HumanCellParameters
from humanbrain.human_cell_identity import HUMAN_GLIF3_IDENTITY


ROOT = Path(__file__).resolve().parents[1]
ALLEN_CONFIG = ROOT / "data" / "allen" / "503612028_neuron_config.json"


def test_language_cell_parameters_match_pinned_allen_human_artifact() -> None:
    params = HumanCellParameters()
    loaded = AllenGlif3Parameters.from_allen_config(ALLEN_CONFIG)
    assert params.cell == loaded
    assert loaded.model_id == HUMAN_GLIF3_IDENTITY.model_id
    assert loaded.specimen_id == HUMAN_GLIF3_IDENTITY.specimen_id
    assert np.isclose(params.cell.rest_mv, -69.04794057210288)
    assert np.isclose(params.cell.membrane_tau_ms, 19.424743742156274)
    assert params.tau_nmda_ms == 35.0


def test_pinned_allen_human_cell_rests_and_spikes_under_current() -> None:
    cell = AllenGlif3Parameters.from_allen_config(ALLEN_CONFIG)
    rest = simulate_glif3_current(cell, 0.0, duration_ms=50.0)
    driven = simulate_glif3_current(cell, 350.0, duration_ms=100.0)
    assert np.nanmax(np.abs(rest.voltage_mv - cell.rest_mv)) < 1e-10
    assert len(driven.spike_times_ms) >= 1


def test_grown_context_selects_target_by_first_cell_spike() -> None:
    cortex = DendriticCortex(cell_count=16, max_order=4)
    for _ in range(4):
        cortex.grow_sequence([1, 2, 3, 4])
    race = cortex.race([9, 1, 2, 3])
    assert race.winner == 4
    assert 0.0 < race.spike_time_ms <= cortex.parameters.race_ms
    assert race.active_cells >= 1


def test_longer_dendritic_context_overrides_ambiguous_single_token() -> None:
    cortex = DendriticCortex(cell_count=16, max_order=4)
    cortex.grow_sequence([5, 2, 8])
    cortex.grow_sequence([6, 2, 9])
    cortex.grow_sequence([5, 2, 8])
    assert cortex.race([5, 2]).winner == 8
    assert cortex.race([6, 2]).winner == 9


def test_longest_dendritic_branch_suppresses_frequent_shorter_branch() -> None:
    cortex = DendriticCortex(cell_count=16, max_order=4)
    for _ in range(20):
        cortex.grow_sequence([1, 2, 3])
    cortex.grow_sequence([9, 1, 2, 4])
    assert cortex.race([9, 1, 2]).winner == 4


def test_dendritic_interpolation_keeps_discounted_lower_order_candidates() -> None:
    cortex = DendriticCortex(cell_count=16, max_order=4)
    for _ in range(4):
        cortex.grow_sequence([1, 2, 3])
    cortex.grow_sequence([9, 1, 2, 4])
    active, drives = cortex._dendritic_drive([9, 1, 2])
    assert set(active) >= {3, 4}
    assert drives[list(active).index(4)] > drives[list(active).index(3)]


def test_compiled_membrane_response_matches_stepwise_euler_race() -> None:
    cortex = DendriticCortex(cell_count=16, max_order=4)
    cortex.grow_sequence([5, 2, 8])
    cortex.grow_sequence([6, 2, 9])
    active, drives = cortex._dendritic_drive([5, 2])
    cell = cortex.parameters.cell
    threshold = cell.threshold_relative_mv + cortex._threshold_offset[active]
    expected = None
    for phase_index, phase_drive in enumerate((drives, drives + 6.0)):
        voltage = np.zeros(len(active), dtype=np.float64)
        current = phase_drive * cortex.parameters.branch_current_pa
        for step in range(cortex._race_steps):
            voltage += cortex._voltage_step_scale * (
                current - cell.leak_ns * voltage
            )
            spiking = voltage >= threshold
            if np.any(spiking):
                indices = np.flatnonzero(spiking)
                local = int(
                    indices[np.argmax(voltage[indices] - threshold[indices])]
                )
                expected = (
                    int(active[local]),
                    (phase_index * cortex._race_steps + step + 1) * cell.dt_ms,
                )
                break
            current *= cortex._nmda_decay
        if expected is not None:
            break
    race = cortex.race([5, 2])
    assert expected is not None
    assert (race.winner, race.spike_time_ms) == expected
