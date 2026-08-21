from __future__ import annotations

import numpy as np
import pytest
from humanbrain.cuda import cuda_available, should_use_cuda
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
