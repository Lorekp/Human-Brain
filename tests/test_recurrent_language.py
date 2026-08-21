import numpy as np
import pytest

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig


def test_recurrent_language_model_learns_and_completes() -> None:
    model = RecurrentCellLanguageModel.grow(
        ["abcabcabcabc"],
        RecurrentModelConfig(
            growth_epochs=8,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=64,
                input_fanout=12,
                recurrent_fanout=4,
            ),
        ),
    )
    assert model.vocabulary_size == 256
    assert model.learned_synapses > 0
    assert model.predict_next("a").text == "b"
    assert model.complete("ab", max_new_tokens=1) == "c"


def test_recurrent_language_evaluation_runs_through_cell_races() -> None:
    model = RecurrentCellLanguageModel.grow(
        ["xyzxyzxyzxyz"],
        RecurrentModelConfig(
            growth_epochs=8,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=64,
                input_fanout=12,
                recurrent_fanout=4,
            ),
        ),
    )
    result = model.evaluate("xyzxyz", max_predictions=5)
    assert result.predictions == 5
    assert result.accuracy >= 0.6
    assert result.tokens_per_second > 0.0


def test_dormant_reservoir_cells_never_enter_the_eligibility_trace() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(
            reservoir_cells=24,
            active_reservoir_cells=8,
            input_fanout=4,
            recurrent_fanout=2,
            token_interval_ms=1.0,
            input_pulse_ms=0.5,
            input_current_pa=8_000.0,
        ),
    )

    circuit.observe(0)

    assert circuit.active_reservoir_cells == 8
    assert not circuit.eligibility[8:].any()


def test_dormant_padding_preserves_the_active_subcircuit() -> None:
    shared = {
        "input_fanout": 4,
        "recurrent_fanout": 2,
        "token_interval_ms": 1.0,
        "input_pulse_ms": 0.5,
        "input_current_pa": 8_000.0,
        "random_seed": 31,
    }
    compact = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(reservoir_cells=8, **shared),
    )
    padded = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(
            reservoir_cells=24, active_reservoir_cells=8, **shared
        ),
    )

    for token_id in (0, 1, 2):
        compact.observe(token_id)
        padded.observe(token_id)

    assert (compact.eligibility == padded.eligibility[:8]).all()


def test_silent_observation_still_teaches_the_output_bias() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=1,
            recurrent_fanout=1,
            token_interval_ms=0.05,
            input_pulse_ms=0.05,
            input_current_pa=1.0,
        ),
    )

    step = circuit.observe(0)
    circuit.strengthen(1)

    assert step.reservoir_spikes == 0
    assert circuit._growth_examples == 1
    assert circuit._target_moments[1, -1] == 1.0


def test_recurrent_checkpoint_round_trip(tmp_path) -> None:
    model = RecurrentCellLanguageModel.grow(
        ["cell synapse cell synapse"],
        RecurrentModelConfig(
            growth_epochs=4,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=64,
                input_fanout=12,
                recurrent_fanout=4,
            ),
        ),
    )
    path = tmp_path / "recurrent-cell.npz"
    digest = model.save(path)
    loaded = RecurrentCellLanguageModel.load(path)
    assert len(digest) == 64
    assert loaded.training_digest == model.training_digest
    assert loaded.learned_synapses == model.learned_synapses
    assert loaded.predict_next("cell ") == model.predict_next("cell ")


def test_three_factor_checkpoint_contains_synapses_but_no_ridge_state(tmp_path) -> None:
    model = RecurrentCellLanguageModel.grow(
        ["abcabcabcabc"],
        RecurrentModelConfig(
            growth_epochs=8,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=64,
                input_fanout=12,
                recurrent_fanout=4,
                readout_learning_rule="three_factor",
            ),
        ),
    )
    assert model._circuit.learning_frozen
    with pytest.raises(RuntimeError, match="frozen"):
        model._circuit.strengthen(1)

    path = tmp_path / "local-recurrent-cell.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as state:
        assert "readout_synapses_pa" in state.files
        assert "pacemaker_synapses_pa" in state.files
        assert "normal_matrix" not in state.files
        assert "target_moments" not in state.files

    loaded = RecurrentCellLanguageModel.load(path)
    assert loaded._circuit.learning_frozen
    assert loaded._circuit._normal_matrix is None
    assert loaded._circuit._target_moments is None
    assert loaded.predict_next("a") == model.predict_next("a")
