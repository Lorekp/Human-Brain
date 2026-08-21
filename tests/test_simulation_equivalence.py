from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytest
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.cortex import CellRace, DendriticCortex
from humanbrain.development import DevelopmentalGrowthConfig, DevelopmentalGrowthController
from humanbrain.recurrent import (
    RecurrentCellCircuit,
    RecurrentCircuitConfig,
    RecurrentRace,
)


def _reference_step(
    population: AdaptiveGlifPopulation,
    injected_current_pa: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Original array-indexed GLIF5 step retained as an equivalence oracle."""
    current = np.asarray(injected_current_pa, dtype=np.float64)
    if current.ndim == 0:
        current = np.full(population.cell_count, float(current), dtype=np.float64)
    if current.shape != (population.cell_count,):
        raise ValueError("injected current must be scalar or one value per cell")

    p = population.parameters
    state = population.state
    spiking = np.zeros(population.cell_count, dtype=bool)
    active = state.refractory_steps == 0
    state.refractory_steps[~active] -= 1
    if not np.any(active):
        return spiking

    rows = np.flatnonzero(active)
    previous_voltage = state.voltage_relative_mv[rows].copy()
    previous_after = state.after_spike_currents_pa[rows].copy()
    total_current = current[rows] + previous_after.sum(axis=1)
    next_voltage = previous_voltage + population._voltage_step_scale * (
        total_current - p.leak_ns * previous_voltage
    )
    beta = total_current * population._threshold_beta_scale
    previous_voltage_threshold = state.threshold_voltage_mv[rows]
    next_voltage_threshold = (
        population._threshold_phi * (previous_voltage - beta) * population._threshold_exp_leak
        + population._threshold_exp_decay
        * (
            previous_voltage_threshold
            - population._threshold_phi * (previous_voltage - beta)
            - population._threshold_voltage_ratio * beta
        )
        + population._threshold_voltage_ratio * beta
    )
    next_spike_threshold = state.threshold_spike_mv[rows] * population._spike_threshold_decay
    next_after = previous_after * population._after_spike_decay
    state.voltage_relative_mv[rows] = next_voltage
    state.threshold_voltage_mv[rows] = next_voltage_threshold
    state.threshold_spike_mv[rows] = next_spike_threshold
    state.after_spike_currents_pa[rows] = next_after

    crossed = next_voltage > (
        p.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold
    )
    if not np.any(crossed):
        return spiking
    spike_rows = rows[crossed]
    spiking[spike_rows] = True
    state.voltage_relative_mv[spike_rows] = (
        p.voltage_reset_scale * next_voltage[crossed] + p.voltage_reset_offset_mv
    )
    state.after_spike_currents_pa[spike_rows] = (
        population._after_spike_amplitudes_pa
        + next_after[crossed] * population._after_spike_cut_decay
    )
    state.threshold_spike_mv[spike_rows] = (
        next_spike_threshold[crossed] * population._spike_cut_threshold_decay
        + p.threshold_spike_increment_mv
    )
    state.refractory_steps[spike_rows] = p.spike_cut_steps
    return spiking


def _assert_same_state(
    actual: AdaptiveGlifPopulation,
    expected: AdaptiveGlifPopulation,
) -> None:
    np.testing.assert_allclose(
        actual.state.voltage_relative_mv,
        expected.state.voltage_relative_mv,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        actual.state.after_spike_currents_pa,
        expected.state.after_spike_currents_pa,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        actual.state.threshold_spike_mv,
        expected.state.threshold_spike_mv,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        actual.state.threshold_voltage_mv,
        expected.state.threshold_voltage_mv,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        actual.state.refractory_steps,
        expected.state.refractory_steps,
    )


def _vectorized_reference_step(
    population: AdaptiveGlifPopulation,
    current: np.ndarray,
) -> np.ndarray:
    """Last Python/NumPy production step, kept only as a native oracle."""
    p = population.parameters
    state = population.state
    active = state.refractory_steps == 0
    np.subtract(state.refractory_steps, 1, out=state.refractory_steps, where=~active)
    if not active.any():
        return active

    previous_voltage = state.voltage_relative_mv
    previous_after = state.after_spike_currents_pa
    previous_after.sum(axis=1, out=population._work_after_sum)
    np.add(current, population._work_after_sum, out=population._work_total_current)
    total_current = population._work_total_current

    np.multiply(p.leak_ns, previous_voltage, out=population._work_next_voltage)
    np.subtract(total_current, population._work_next_voltage, out=population._work_next_voltage)
    np.multiply(
        population._voltage_step_scale,
        population._work_next_voltage,
        out=population._work_next_voltage,
    )
    np.add(previous_voltage, population._work_next_voltage, out=population._work_next_voltage)
    next_voltage = population._work_next_voltage

    beta = total_current * population._threshold_beta_scale
    next_voltage_threshold = (
        population._threshold_exp_decay * state.threshold_voltage_mv
        + population._threshold_voltage_factor * previous_voltage
        + population._threshold_beta_factor * beta
    )
    np.multiply(
        state.threshold_spike_mv,
        population._spike_threshold_decay,
        out=population._work_next_spike_threshold,
    )
    next_spike_threshold = population._work_next_spike_threshold
    np.multiply(previous_after, population._after_spike_decay, out=population._work_next_after)
    next_after = population._work_next_after

    np.copyto(state.voltage_relative_mv, next_voltage, where=active)
    np.copyto(state.threshold_voltage_mv, next_voltage_threshold, where=active)
    np.copyto(state.threshold_spike_mv, next_spike_threshold, where=active)
    np.copyto(state.after_spike_currents_pa, next_after, where=active[:, None])
    crossed = active & (
        next_voltage
        > (p.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold)
    )
    if not crossed.any():
        return crossed

    spike_rows = np.flatnonzero(crossed)
    state.voltage_relative_mv[spike_rows] = (
        p.voltage_reset_scale * next_voltage[spike_rows] + p.voltage_reset_offset_mv
    )
    state.after_spike_currents_pa[spike_rows] = (
        population._after_spike_amplitudes_pa
        + next_after[spike_rows] * population._after_spike_cut_decay
    )
    state.threshold_spike_mv[spike_rows] = (
        next_spike_threshold[spike_rows] * population._spike_cut_threshold_decay
        + p.threshold_spike_increment_mv
    )
    state.refractory_steps[spike_rows] = p.spike_cut_steps
    return crossed


def _python_reference_observe(circuit: RecurrentCellCircuit, token_id: int) -> int:
    """Last Python sparse token loop, retained only for equivalence testing."""
    afferents = circuit._afferent_targets[token_id]
    step_current_pa = np.empty_like(circuit._synaptic_current_pa)
    spike_total = 0
    for step in range(circuit._token_steps):
        np.copyto(step_current_pa, circuit._synaptic_current_pa)
        if step < circuit._pulse_steps:
            step_current_pa[afferents] += circuit.config.input_current_pa
        spikes = _vectorized_reference_step(circuit._reservoir, step_current_pa)
        circuit._synaptic_current_pa *= circuit._recurrent_decay
        circuit._eligibility *= circuit._eligibility_decay
        if spikes.any():
            presynaptic = np.flatnonzero(spikes)
            spike_total += int(presynaptic.size)
            circuit._eligibility[presynaptic] += 1.0
            targets = circuit._recurrent_targets[presynaptic].reshape(-1)
            weights = circuit._recurrent_weights_pa[presynaptic].reshape(-1)
            np.add.at(circuit._synaptic_current_pa, targets, weights)
    circuit._observed_tokens += 1
    return spike_total


def _circuit_state_arrays(circuit: RecurrentCellCircuit) -> tuple[np.ndarray, ...]:
    state = circuit._reservoir.state
    return (
        circuit._synaptic_current_pa,
        circuit._eligibility,
        state.voltage_relative_mv,
        state.after_spike_currents_pa,
        state.threshold_spike_mv,
        state.threshold_voltage_mv,
        state.refractory_steps,
    )


def test_native_recurrent_token_loop_is_bit_exact_to_python_reference() -> None:
    config = RecurrentCircuitConfig(
        reservoir_cells=96,
        active_reservoir_cells=72,
        input_fanout=8,
        recurrent_fanout=6,
        token_interval_ms=5.0,
        input_pulse_ms=1.5,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        recurrent_topology="nested_groups",
        random_seed=29,
    )
    actual = RecurrentCellCircuit(token_cells=37, config=config)
    expected = RecurrentCellCircuit(token_cells=37, config=config)
    tokens = np.random.default_rng(9031).integers(0, actual.token_cells, 64)

    for token in tokens:
        step = actual.observe(int(token))
        expected_spikes = _python_reference_observe(expected, int(token))
        assert step.reservoir_spikes == expected_spikes
        np.testing.assert_array_equal(actual._synaptic_current_pa, expected._synaptic_current_pa)
        np.testing.assert_array_equal(actual._eligibility, expected._eligibility)
        np.testing.assert_array_equal(
            actual._reservoir.state.voltage_relative_mv,
            expected._reservoir.state.voltage_relative_mv,
        )
        np.testing.assert_array_equal(
            actual._reservoir.state.after_spike_currents_pa,
            expected._reservoir.state.after_spike_currents_pa,
        )
        np.testing.assert_array_equal(
            actual._reservoir.state.threshold_spike_mv,
            expected._reservoir.state.threshold_spike_mv,
        )
        np.testing.assert_array_equal(
            actual._reservoir.state.threshold_voltage_mv,
            expected._reservoir.state.threshold_voltage_mv,
        )
        np.testing.assert_array_equal(
            actual._reservoir.state.refractory_steps,
            expected._reservoir.state.refractory_steps,
        )


def test_fused_parallel_kernel_is_bit_exact_to_native_v1_for_64_tokens() -> None:
    config = RecurrentCircuitConfig(
        reservoir_cells=8_192,
        input_fanout=32,
        recurrent_fanout=6,
        token_interval_ms=5.0,
        input_pulse_ms=1.5,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        recurrent_topology="uniform_no_self",
        random_seed=31,
    )
    actual = RecurrentCellCircuit(token_cells=67, config=config)
    expected = RecurrentCellCircuit(token_cells=67, config=config)
    tokens = np.random.default_rng(8_031).integers(0, actual.token_cells, 64)

    actual_spikes = [actual._native_kernel._observe_parallel(int(token)) for token in tokens]
    expected_spikes = [
        expected._native_kernel._observe_reference_v1(int(token)) for token in tokens
    ]

    assert actual_spikes == expected_spikes
    np.testing.assert_array_equal(actual._synaptic_current_pa, expected._synaptic_current_pa)
    np.testing.assert_array_equal(actual._eligibility, expected._eligibility)
    np.testing.assert_array_equal(
        actual._reservoir.state.voltage_relative_mv,
        expected._reservoir.state.voltage_relative_mv,
    )
    np.testing.assert_array_equal(
        actual._reservoir.state.after_spike_currents_pa,
        expected._reservoir.state.after_spike_currents_pa,
    )
    np.testing.assert_array_equal(
        actual._reservoir.state.threshold_spike_mv,
        expected._reservoir.state.threshold_spike_mv,
    )
    np.testing.assert_array_equal(
        actual._reservoir.state.threshold_voltage_mv,
        expected._reservoir.state.threshold_voltage_mv,
    )
    np.testing.assert_array_equal(
        actual._reservoir.state.refractory_steps,
        expected._reservoir.state.refractory_steps,
    )


def test_quiescent_frontier_is_exact_across_activity_densities() -> None:
    config = RecurrentCircuitConfig(
        reservoir_cells=4_096,
        input_fanout=32,
        recurrent_fanout=6,
        token_interval_ms=5.0,
        input_pulse_ms=1.5,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        recurrent_topology="uniform_no_self",
        random_seed=47,
    )
    circuit = RecurrentCellCircuit(token_cells=67, config=config)

    for density in (0.01, 0.2, 1.0):
        seeded = max(1, round(config.reservoir_cells * density))
        indices = np.arange(seeded, dtype=np.intp) * config.reservoir_cells // seeded
        circuit.reset_sequence()
        circuit._synaptic_current_pa[indices] = 25.0
        circuit._eligibility[indices] = 0.5
        frontier_spikes = circuit._native_kernel.observe(13)
        frontier_state = tuple(array.copy() for array in _circuit_state_arrays(circuit))
        assert circuit._native_kernel.last_active_cells >= seeded
        scan_ns, build_ns, step_ns = circuit._native_kernel.last_frontier_profile_ns
        assert scan_ns > 0
        assert build_ns >= 0
        assert step_ns > 0

        circuit.reset_sequence()
        circuit._synaptic_current_pa[indices] = 25.0
        circuit._eligibility[indices] = 0.5
        dense_spikes = circuit._native_kernel._observe_parallel(13)

        assert frontier_spikes == dense_spikes
        for actual, expected in zip(
            _circuit_state_arrays(circuit),
            frontier_state,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


def test_native_transition_remains_exact_after_developmental_mutations_and_reset() -> None:
    config = RecurrentCircuitConfig(
        reservoir_cells=48,
        active_reservoir_cells=16,
        input_fanout=8,
        recurrent_fanout=8,
        token_interval_ms=2.0,
        input_pulse_ms=1.2,
        input_current_pa=8_000.0,
        recurrent_current_pa=325.0,
        eligibility_tau_ms=40.0,
        recurrent_topology="uniform_no_self",
        readout_learning_rule="three_factor",
        random_seed=29,
    )
    development = DevelopmentalGrowthConfig(
        stage_active_cells=(16, 32, 48),
        initial_recurrent_fanout=4,
        final_recurrent_fanout=4,
        structural_seed=121,
        homeostatic_scaling=True,
    )
    actual = RecurrentCellCircuit(token_cells=37, config=config)
    expected = RecurrentCellCircuit(token_cells=37, config=config)
    actual_growth = DevelopmentalGrowthController(actual, development)
    expected_growth = DevelopmentalGrowthController(expected, development)
    actual_growth.recruit_to(32)
    expected_growth.recruit_to(32)
    actual_growth.structural_score[:] = np.arange(8, dtype=np.float64)[None, :]
    expected_growth.structural_score[:] = actual_growth.structural_score
    actual_growth.structural_exposure[actual._recurrent_weights_pa != 0.0] = 1.0
    expected_growth.structural_exposure[expected._recurrent_weights_pa != 0.0] = 1.0
    activity = np.linspace(0.25, 2.0, 32)
    actual_growth.cell_activity[actual_growth.active_mask] = activity
    expected_growth.cell_activity[expected_growth.active_mask] = activity
    actual_growth._stage_examples = expected_growth._stage_examples = 4
    assert actual_growth.apply_activity_homeostasis() == (
        expected_growth.apply_activity_homeostasis()
    )
    actual_growth.prune_to_final_fanout()
    expected_growth.prune_to_final_fanout()
    np.testing.assert_array_equal(
        actual._recurrent_weights_pa,
        expected._recurrent_weights_pa,
    )
    actual.reset_sequence()
    expected.reset_sequence()

    tokens = np.random.default_rng(805).integers(0, actual.token_cells, 32)
    for token in tokens:
        step = actual.observe(int(token))
        expected_spikes = _python_reference_observe(expected, int(token))
        assert step.reservoir_spikes == expected_spikes
        for actual_array, expected_array in zip(
            _circuit_state_arrays(actual),
            _circuit_state_arrays(expected),
            strict=True,
        ):
            np.testing.assert_array_equal(actual_array, expected_array)


def test_vectorized_glif_step_matches_original_during_spikes_and_refractory() -> None:
    actual = AdaptiveGlifPopulation(31)
    expected = AdaptiveGlifPopulation(31)
    rng = np.random.default_rng(702)

    for step in range(2_000):
        current = rng.uniform(0.0, 8_000.0, 31)
        if step % 7 == 0:
            current[::3] = 0.0
        actual_spikes = actual.step(current)
        expected_spikes = _reference_step(expected, current)
        np.testing.assert_array_equal(actual_spikes, expected_spikes)
        _assert_same_state(actual, expected)


def _reference_race(circuit: RecurrentCellCircuit) -> RecurrentRace:
    current = circuit.output_current_pa()
    output = AdaptiveGlifPopulation(circuit.token_cells, circuit.cell_parameters)
    p = circuit.cell_parameters
    steps = max(1, round(circuit.config.output_race_ms / p.dt_ms))
    peak = p.rest_mv
    for step in range(steps):
        fired = output.step(current)
        peak = max(
            peak,
            p.rest_mv + float(np.max(output.state.voltage_relative_mv)),
        )
        if fired.any():
            return RecurrentRace(
                winner=int(np.flatnonzero(fired)[0]),
                spike_time_ms=(step + 1) * p.dt_ms,
                peak_voltage_mv=peak,
                active_cells=circuit.token_cells,
                simulated_steps=step + 1,
            )
    raise RuntimeError("output human cells did not spike during the race")


def test_compiled_output_race_matches_stepwise_glif5_cells() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=67,
        config=RecurrentCircuitConfig(
            reservoir_cells=96,
            input_fanout=8,
            recurrent_fanout=6,
            recurrent_topology="uniform_no_self",
        ),
    )
    circuit.observe(9)
    rng = np.random.default_rng(18)
    circuit._readout_synapses_pa[:] = rng.normal(0.0, 120.0, circuit._readout_synapses_pa.shape)
    circuit._pacemaker_synapses_pa[:] = rng.uniform(100.0, 1_000.0, 67)

    expected = _reference_race(circuit)
    actual = circuit.race()

    assert actual.winner == expected.winner
    assert actual.spike_time_ms == expected.spike_time_ms
    assert actual.simulated_steps == expected.simulated_steps
    assert actual.active_cells == expected.active_cells
    assert actual.peak_voltage_mv == pytest.approx(expected.peak_voltage_mv, abs=1e-12)


def _reference_dendritic_race(
    cortex: DendriticCortex,
    context: Sequence[int],
) -> CellRace:
    active, association_drive = cortex._dendritic_drive(context)
    p = cortex.parameters
    cell = p.cell
    threshold = cell.threshold_relative_mv + cortex._threshold_offset[active]
    peak = cell.rest_mv
    phases = (association_drive, association_drive + 6.0)
    for phase_index, phase_drive in enumerate(phases):
        i_nmda = phase_drive * p.branch_current_pa
        voltage = cortex._unit_nmda_response[:, None] * i_nmda[None, :]
        crossings = voltage >= threshold[None, :]
        crossing_steps = np.flatnonzero(np.any(crossings, axis=1))
        if crossing_steps.size == 0:
            peak = max(peak, cell.rest_mv + float(np.max(voltage)))
            continue
        step = int(crossing_steps[0])
        peak = max(peak, cell.rest_mv + float(np.max(voltage[: step + 1])))
        indices = np.flatnonzero(crossings[step])
        local = int(indices[np.argmax(voltage[step, indices] - threshold[indices])])
        elapsed_steps = phase_index * cortex._race_steps + step + 1
        return CellRace(
            winner=int(active[local]),
            spike_time_ms=elapsed_steps * cell.dt_ms,
            peak_voltage_mv=peak,
            active_cells=len(active),
            simulated_steps=elapsed_steps,
        )
    raise RuntimeError("active human cells did not spike under arousal")


def test_searchsorted_dendritic_race_matches_response_matrix() -> None:
    cortex = DendriticCortex(cell_count=256, max_order=8, seed=13)
    training = np.random.default_rng(44).integers(0, 256, 2_048)
    cortex.grow_sequence(training)

    for end in range(64, 1_024, 31):
        context = training[end - 8 : end]
        expected = _reference_dendritic_race(cortex, context)
        actual = cortex.race(context)
        assert actual.winner == expected.winner
        assert actual.spike_time_ms == expected.spike_time_ms
        assert actual.simulated_steps == expected.simulated_steps
        assert actual.active_cells == expected.active_cells
        assert actual.peak_voltage_mv == pytest.approx(
            expected.peak_voltage_mv,
            abs=1e-12,
        )


def test_batched_readout_statistics_match_sequential_outer_products() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=17,
        config=RecurrentCircuitConfig(
            reservoir_cells=32,
            input_fanout=4,
            recurrent_fanout=4,
            recurrent_topology="uniform_no_self",
        ),
    )
    feature_count = circuit.config.reservoir_cells + 1
    expected_normal = np.zeros((feature_count, feature_count), dtype=np.float64)
    expected_targets = np.zeros((circuit.token_cells, feature_count), dtype=np.float64)
    tokens = np.random.default_rng(903).integers(0, circuit.token_cells, 301)

    circuit.reset_sequence()
    for token, target in pairwise(tokens):
        circuit.observe(int(token))
        feature = np.empty(feature_count, dtype=np.float64)
        feature[:-1] = circuit.eligibility
        feature[-1] = 1.0
        expected_normal += np.outer(feature, feature)
        expected_targets[int(target)] += feature
        circuit.strengthen(int(target))
    circuit._flush_growth_statistics()

    np.testing.assert_allclose(
        circuit._normal_matrix,
        expected_normal,
        rtol=2e-13,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        circuit._target_moments,
        expected_targets,
        rtol=0.0,
        atol=0.0,
    )

    regularized = expected_normal.copy()
    diagonal = np.diag_indices_from(regularized)
    regularized[diagonal] += circuit.config.readout_ridge
    regularized[-1, -1] -= circuit.config.readout_ridge * 0.99
    expected_coefficients = np.linalg.solve(regularized, expected_targets.T).T
    circuit._consolidate_readout()
    gain = circuit.config.output_synapse_gain_pa
    np.testing.assert_allclose(
        circuit._readout_synapses_pa,
        expected_coefficients[:, :-1] * gain,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        circuit._pacemaker_synapses_pa,
        expected_coefficients[:, -1] * gain,
        rtol=2e-10,
        atol=2e-10,
    )


def test_inference_only_circuit_does_not_allocate_quadratic_growth_state() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=17,
        config=RecurrentCircuitConfig(
            reservoir_cells=768,
            input_fanout=64,
            recurrent_fanout=6,
            recurrent_topology="uniform_no_self",
        ),
    )

    assert circuit._normal_matrix is None
    assert circuit._target_moments is None
    assert circuit._feature_batch is None

    circuit.observe(3)

    assert circuit._normal_matrix is None
    assert circuit._target_moments is None
    assert circuit._feature_batch is None
