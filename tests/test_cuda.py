from __future__ import annotations

import numpy as np
import pytest
from humanbrain.cuda import cuda_available, should_use_cuda
from humanbrain.development import DevelopmentalGrowthConfig, DevelopmentalGrowthController
from humanbrain.native import NativeBuildError
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


def _circuit() -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=37,
        config=RecurrentCircuitConfig(
            reservoir_cells=384,
            active_reservoir_cells=320,
            input_fanout=12,
            recurrent_fanout=6,
            token_interval_ms=5.0,
            input_pulse_ms=1.5,
            input_current_pa=4_000.0,
            recurrent_current_pa=325.0,
            eligibility_tau_ms=40.0,
            recurrent_topology="nested_groups",
            random_seed=29,
        ),
    )


def _assert_same_state(actual: RecurrentCellCircuit, expected: RecurrentCellCircuit) -> None:
    np.testing.assert_array_equal(actual._synaptic_current_pa, expected._synaptic_current_pa)
    np.testing.assert_array_equal(actual._eligibility, expected._eligibility)
    actual_state = actual._reservoir.state
    expected_state = expected._reservoir.state
    np.testing.assert_array_equal(
        actual_state.voltage_relative_mv, expected_state.voltage_relative_mv
    )
    np.testing.assert_array_equal(
        actual_state.after_spike_currents_pa, expected_state.after_spike_currents_pa
    )
    np.testing.assert_array_equal(
        actual_state.threshold_spike_mv, expected_state.threshold_spike_mv
    )
    np.testing.assert_array_equal(
        actual_state.threshold_voltage_mv, expected_state.threshold_voltage_mv
    )
    np.testing.assert_array_equal(actual_state.refractory_steps, expected_state.refractory_steps)


def _developmental_circuit() -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=19,
        config=RecurrentCircuitConfig(
            reservoir_cells=48,
            active_reservoir_cells=12,
            input_fanout=4,
            recurrent_fanout=8,
            token_interval_ms=2.0,
            input_pulse_ms=1.2,
            input_current_pa=8_000.0,
            recurrent_current_pa=325.0,
            eligibility_tau_ms=40.0,
            recurrent_topology="uniform_no_self",
            readout_learning_rule="three_factor",
            random_seed=43,
        ),
    )


def _assert_same_structure(
    actual: RecurrentCellCircuit,
    expected: RecurrentCellCircuit,
) -> None:
    np.testing.assert_array_equal(actual._afferent_targets, expected._afferent_targets)
    np.testing.assert_array_equal(actual._recurrent_targets, expected._recurrent_targets)
    np.testing.assert_array_equal(actual._recurrent_weights_pa, expected._recurrent_weights_pa)


def test_device_policy_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    assert not should_use_cuda(1_000_000)
    monkeypatch.setattr("humanbrain.cuda.cuda_available", lambda: True)
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "auto")
    assert not should_use_cuda(99_999)
    assert should_use_cuda(100_000)
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    assert should_use_cuda(8)


def test_forced_cuda_fails_instead_of_silently_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    monkeypatch.setattr("humanbrain.cuda.cuda_available", lambda: False)
    with pytest.raises(NativeBuildError, match="no CUDA backend"):
        should_use_cuda(100_000)


def test_cuda_token_loop_is_bit_exact_to_cpu_and_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not cuda_available():
        pytest.skip("CUDA cooperative kernel is unavailable")
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    expected = _circuit()
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    actual = _circuit()
    assert actual.simulation_backend == "cuda"
    assert "NVIDIA" in actual.simulation_device
    assert expected.simulation_backend == "cpu"

    tokens = np.random.default_rng(9031).integers(0, actual.token_cells, 64)
    for token in tokens:
        actual_step = actual.observe(int(token))
        expected_step = expected.observe(int(token))
        assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
        _assert_same_state(actual, expected)

    actual.reset_sequence()
    expected.reset_sequence()
    _assert_same_state(actual, expected)
    for token in (7, 7):
        actual_step = actual.observe(token)
        expected_step = expected.observe(token)
        assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
        _assert_same_state(actual, expected)

    actual._native_kernel.close()
    actual._native_kernel.close()


def test_cuda_developmental_structure_homeostasis_pruning_and_reset_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not cuda_available():
        pytest.skip("CUDA cooperative kernel is unavailable")
    development = DevelopmentalGrowthConfig(
        stage_active_cells=(12, 24, 48),
        initial_recurrent_fanout=4,
        final_recurrent_fanout=4,
        structural_seed=121,
        homeostatic_scaling=True,
    )
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    expected = _developmental_circuit()
    expected_growth = DevelopmentalGrowthController(expected, development)
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    actual = _developmental_circuit()
    try:
        actual_growth = DevelopmentalGrowthController(actual, development)
        _assert_same_structure(actual, expected)

        for token in (3, 7, 11):
            actual_step = actual.observe(token)
            expected_step = expected.observe(token)
            assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
            _assert_same_state(actual, expected)
            actual_growth.record_plasticity()
            expected_growth.record_plasticity()

        assert actual_growth.recruit_to(24) == expected_growth.recruit_to(24) == 12
        _assert_same_structure(actual, expected)
        for token in (2, 13):
            actual_step = actual.observe(token)
            expected_step = expected.observe(token)
            assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
            _assert_same_state(actual, expected)

        activity = np.linspace(0.25, 2.0, 24)
        actual_growth.cell_activity[actual_growth.active_mask] = activity
        expected_growth.cell_activity[expected_growth.active_mask] = activity
        actual_growth._stage_examples = expected_growth._stage_examples = 4
        assert actual_growth.apply_activity_homeostasis() == (
            expected_growth.apply_activity_homeostasis()
        )
        _assert_same_structure(actual, expected)
        actual_step = actual.observe(5)
        expected_step = expected.observe(5)
        assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
        _assert_same_state(actual, expected)

        scores = np.arange(8, dtype=np.float64)[None, :]
        actual_growth.structural_score[:] = scores
        expected_growth.structural_score[:] = scores
        actual_growth.structural_exposure[actual._recurrent_weights_pa != 0.0] = 1.0
        expected_growth.structural_exposure[expected._recurrent_weights_pa != 0.0] = 1.0
        assert actual_growth.prune_to_final_fanout() == (
            expected_growth.prune_to_final_fanout()
        )
        _assert_same_structure(actual, expected)

        actual.reset_sequence()
        expected.reset_sequence()
        _assert_same_state(actual, expected)
        for token in (17, 17):
            actual_step = actual.observe(token)
            expected_step = expected.observe(token)
            assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
            _assert_same_state(actual, expected)
    finally:
        actual._native_kernel.close()
