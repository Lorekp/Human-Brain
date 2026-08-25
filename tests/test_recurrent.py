import numpy as np
import pytest

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


def small_circuit() -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=8,
        config=RecurrentCircuitConfig(
            reservoir_cells=64,
            input_fanout=12,
            recurrent_fanout=4,
        ),
    )


def test_observation_changes_only_cell_and_synapse_state() -> None:
    circuit = small_circuit()
    before = circuit.eligibility
    step = circuit.observe(3)
    assert step.reservoir_spikes > 0
    assert not np.array_equal(circuit.eligibility, before)
    assert circuit.observed_tokens == 1
    assert circuit.learned_synapse_count == 0


def test_local_plasticity_learns_repeated_next_tokens() -> None:
    circuit = small_circuit()
    circuit.grow_sequence([1, 2, 3, 1, 2, 3], epochs=12)
    assert circuit.learned_synapse_count > 0
    assert circuit.predict_next([1]).winner == 2
    assert circuit.predict_next([1, 2]).winner == 3


def test_output_is_first_adaptive_cell_spike() -> None:
    circuit = small_circuit()
    circuit.grow_sequence([1, 4, 1, 4], epochs=8)
    circuit.reset_sequence()
    circuit.observe(1)
    race = circuit.race()
    assert race.winner == 4
    assert 0.0 < race.spike_time_ms <= circuit.config.output_race_ms
    assert race.simulated_steps > 0
    assert race.active_cells == circuit.token_cells


def test_growth_result_does_not_depend_on_sequence_order() -> None:
    sequences = (
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6],
        [7, 1, 2, 7, 1, 2],
    )
    forward = small_circuit()
    reverse = small_circuit()
    for sequence in sequences:
        forward.grow_sequence(sequence, epochs=3)
    for sequence in reversed(sequences):
        reverse.grow_sequence(sequence, epochs=3)
    forward_predictions = [forward.predict_next([token]).winner for token in (1, 2, 4, 5)]
    reverse_predictions = [reverse.predict_next([token]).winner for token in (1, 2, 4, 5)]
    assert forward_predictions == reverse_predictions


def test_three_factor_update_uses_trace_competition_and_target_modulation() -> None:
    learning_rate = 10.0
    circuit = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=1,
            recurrent_fanout=1,
            readout_learning_rule="three_factor",
            local_learning_rate_pa=learning_rate,
        ),
    )
    # R0: seed the (K, R) trace bank with the incumbent [3, 4] pattern split
    # across traces (row 0 cell 0 and row 3 cell 5) so the flat source slots
    # 0 and 3·R+5 carry exactly the same squared norm as the single-trace
    # original (1 + 3² + 4² = 26) while proving multi-row slot addressing.
    circuit._eligibility_traces[:] = 0.0
    circuit._eligibility_traces[0, 0] = 3.0
    circuit._eligibility_traces[3, 5] = 4.0
    circuit._output_heterogeneity_pa.fill(0.0)
    circuit._observed_tokens = 1

    circuit.strengthen(2)

    flat_features = circuit._feature_traces.copy()
    step = learning_rate / np.sqrt(1.0 + float(flat_features @ flat_features))
    expected = step * flat_features
    np.testing.assert_allclose(circuit._readout_synapses_pa[2], expected)
    np.testing.assert_allclose(circuit._readout_synapses_pa[0], -expected)
    assert not circuit._readout_synapses_pa[1].any()
    assert not circuit._readout_synapses_pa[3].any()
    np.testing.assert_allclose(
        circuit._pacemaker_synapses_pa,
        (-step, 0.0, step, 0.0),
    )
    assert circuit.local_update_count == 1

    circuit.strengthen(2)
    np.testing.assert_allclose(circuit._readout_synapses_pa[2], 2.0 * expected)
    np.testing.assert_allclose(circuit._readout_synapses_pa[1], -expected)
    assert circuit.local_update_count == 2

    before = circuit._readout_synapses_pa.copy()
    circuit.strengthen(2)
    np.testing.assert_array_equal(circuit._readout_synapses_pa, before)
    assert circuit.local_update_count == 2


def test_three_factor_growth_never_allocates_a_reservoir_normal_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservoir_cells = 384
    circuit = RecurrentCellCircuit(
        token_cells=17,
        config=RecurrentCircuitConfig(
            reservoir_cells=reservoir_cells,
            input_fanout=32,
            recurrent_fanout=6,
            recurrent_topology="uniform_no_self",
            readout_learning_rule="three_factor",
        ),
    )
    original_zeros = np.zeros

    def reject_normal_matrix(shape, *args, **kwargs):
        assert shape != (reservoir_cells + 1, reservoir_cells + 1)
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", reject_normal_matrix)
    circuit.grow_sequence([1, 2, 3, 1, 2, 3], epochs=2)

    assert circuit._normal_matrix is None
    assert circuit._target_moments is None
    assert circuit._feature_batch is None
    assert circuit.learned_synapse_count > 0


def test_three_factor_local_rule_learns_repeated_next_tokens() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=8,
        config=RecurrentCircuitConfig(
            reservoir_cells=64,
            input_fanout=12,
            recurrent_fanout=4,
            readout_learning_rule="three_factor",
        ),
    )
    circuit.grow_sequence([1, 2, 3, 1, 2, 3], epochs=12)

    assert circuit.predict_next([1]).winner == 2
    assert circuit.predict_next([1, 2]).winner == 3
