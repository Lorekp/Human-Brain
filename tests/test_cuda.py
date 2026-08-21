from __future__ import annotations

import numpy as np
import pytest
from humanbrain.cuda import cuda_available, should_use_cuda
from humanbrain.development import DevelopmentalGrowthConfig, DevelopmentalGrowthController
from humanbrain.native import NativeBuildError
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel
from humanbrain.tokenizer import BytePhraseTokenizer


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


def _assert_same_sparse_state(
    actual: RecurrentCellCircuit,
    expected: RecurrentCellCircuit,
) -> None:
    actual_readout = actual._sparse_readout
    expected_readout = expected._sparse_readout
    assert actual_readout is not None
    assert expected_readout is not None
    for name, expected_array in expected_readout.checkpoint_arrays().items():
        np.testing.assert_array_equal(
            actual_readout.checkpoint_arrays()[name], expected_array
        )
    assert actual_readout.checkpoint_counters() == (
        expected_readout.checkpoint_counters()
    )
    np.testing.assert_array_equal(
        actual._pacemaker_synapses_pa, expected._pacemaker_synapses_pa
    )
    assert actual.local_update_count == expected.local_update_count
    assert actual.structural_recruitment_count == expected.structural_recruitment_count
    assert actual.structural_deletion_count == expected.structural_deletion_count
    assert getattr(actual, "_last_output_winner", None) == getattr(
        expected, "_last_output_winner", None
    )
    assert getattr(actual, "_last_teaching_current_pa", None) == getattr(
        expected, "_last_teaching_current_pa", None
    )
    assert getattr(actual, "_last_teaching_spike_time_ms", None) == getattr(
        expected, "_last_teaching_spike_time_ms", None
    )


def _sparse_circuit() -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=256,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=4,
            recurrent_fanout=2,
            token_interval_ms=2.0,
            input_pulse_ms=1.0,
            input_current_pa=8_000.0,
            recurrent_current_pa=325.0,
            eligibility_tau_ms=40.0,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=2,
            readout_structural_interval=1,
            readout_structural_grace=2,
            readout_turnover_probability=1.0,
            random_seed=97,
        ),
    )


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


def test_cuda_sparse_readout_lifecycle_and_checkpoint_continuation_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    if not cuda_available():
        pytest.skip("CUDA cooperative kernel is unavailable")

    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    expected = _sparse_circuit()
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    actual = _sparse_circuit()
    loaded_actual = None
    try:
        assert expected.simulation_backend == "cpu"
        assert actual.simulation_backend == "cuda"
        assert "NVIDIA" in actual.simulation_device

        for token in (97, 98):
            actual_step = actual.observe(token)
            expected_step = expected.observe(token)
            assert actual_step == expected_step
            _assert_same_state(actual, expected)
            np.testing.assert_array_equal(
                actual.output_current_pa(), expected.output_current_pa()
            )
            assert actual.race() == expected.race()

        target = (expected.race().winner + 1) % expected.token_cells
        actual.strengthen(target)
        expected.strengthen(target)
        _assert_same_sparse_state(actual, expected)
        np.testing.assert_array_equal(actual.output_current_pa(), expected.output_current_pa())
        assert actual.race() == expected.race()

        actual.reset_sequence()
        expected.reset_sequence()
        _assert_same_state(actual, expected)
        actual_step = actual.observe(99)
        expected_step = expected.observe(99)
        assert actual_step == expected_step
        _assert_same_state(actual, expected)

        actual.freeze_learning()
        expected.freeze_learning()
        tokenizer = BytePhraseTokenizer()
        expected_model = RecurrentCellLanguageModel(
            tokenizer, expected, training_digest="0" * 64
        )
        actual_model = RecurrentCellLanguageModel(
            tokenizer, actual, training_digest="0" * 64
        )
        expected_path = tmp_path / "cpu-sparse.npz"
        actual_path = tmp_path / "cuda-sparse.npz"
        expected_model.save(expected_path)
        actual_model.save(actual_path)

        monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
        loaded_expected = RecurrentCellLanguageModel.load(expected_path)
        monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
        loaded_actual = RecurrentCellLanguageModel.load(actual_path)
        assert loaded_expected._circuit.simulation_backend == "cpu"
        assert loaded_actual._circuit.simulation_backend == "cuda"

        assert loaded_actual.predict_next("ab") == loaded_expected.predict_next("ab")
        _assert_same_state(loaded_actual._circuit, loaded_expected._circuit)
        _assert_same_sparse_state(loaded_actual._circuit, loaded_expected._circuit)

        expected_readout = loaded_expected._circuit._sparse_readout
        actual_readout = loaded_actual._circuit._sparse_readout
        assert expected_readout is not None
        assert actual_readout is not None
        expected_readout.teach(
            target=10,
            winner=11,
            learning_rate_pa=64.0,
            maximum_weight_pa=1_600.0,
            trace_decay=np.exp(-2.0 / 40.0),
            target_post_gate=1.0,
            winner_post_gate=1.0,
            activity_decay=0.99,
        )
        actual_readout.teach(
            target=10,
            winner=11,
            learning_rate_pa=64.0,
            maximum_weight_pa=1_600.0,
            trace_decay=np.exp(-2.0 / 40.0),
            target_post_gate=1.0,
            winner_post_gate=1.0,
            activity_decay=0.99,
        )
        expected_event = expected_readout.advance_target(10)
        actual_event = actual_readout.advance_target(10)
        assert actual_event == expected_event
        turnover = dict(
            output=10,
            structural_event=expected_event,
            random_seed=97,
            minimum_age=2,
            turnover_probability=1.0,
            learning_rate_pa=64.0,
            maximum_weight_pa=1_600.0,
            homeostatic_rate=0.05,
        )
        assert actual_readout.turnover(**turnover) == expected_readout.turnover(**turnover)
        _assert_same_sparse_state(loaded_actual._circuit, loaded_expected._circuit)
        np.testing.assert_array_equal(
            loaded_actual._circuit.output_current_pa(),
            loaded_expected._circuit.output_current_pa(),
        )
        assert loaded_actual._circuit.race() == loaded_expected._circuit.race()

        loaded_actual._circuit.reset_sequence()
        loaded_expected._circuit.reset_sequence()
        _assert_same_state(loaded_actual._circuit, loaded_expected._circuit)
    finally:
        actual._native_kernel.close()
        if loaded_actual is not None:
            loaded_actual._circuit._native_kernel.close()


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
