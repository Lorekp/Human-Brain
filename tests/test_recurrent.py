import numpy as np

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
