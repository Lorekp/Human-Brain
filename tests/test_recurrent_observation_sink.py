"""R0-c Thread-1: observation_sink logging seam on RecurrentCellCircuit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


def _circuit(feature_family: str = "spike_counts") -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=16,
        config=RecurrentCircuitConfig(
            reservoir_cells=32,
            input_fanout=4,
            recurrent_fanout=4,
            random_seed=7,
            readout_learning_rule="ridge",
            feature_family=feature_family,
        ),
    )


def _ids() -> list[int]:
    rng = np.random.default_rng(3)
    return [int(t) for t in rng.integers(0, 16, size=24)]


def test_sink_defaults_to_none_and_validates() -> None:
    circuit = _circuit()
    assert circuit.observation_sink is None
    with pytest.raises(ValueError):
        circuit.observation_sink = "not callable"  # type: ignore[assignment]
    circuit.observation_sink = None  # reset stays legal


def test_sink_fires_per_token_in_order_after_boundary() -> None:
    circuit = _circuit()
    seen: list[tuple[int, tuple]] = []
    post_boundary: list[np.ndarray] = []

    def sink(circ: RecurrentCellCircuit, step) -> None:
        seen.append(
            (step.token_id, tuple(np.asarray(circ.feature_state_vector()).shape))
        )
        post_boundary.append(np.asarray(circ.feature_state_vector()).copy())

    circuit.observation_sink = sink
    ids = _ids()
    for token_id in ids:
        circuit.observe(token_id)
        # Sink already fired for THIS token inside observe(): the captured
        # row must equal the post-observe vector the caller sees now.
        assert np.array_equal(post_boundary[-1], circuit.feature_state_vector())
    assert [token_id for token_id, _ in seen] == ids


def test_sink_is_readonly_observer_bit_identity() -> None:
    ids = _ids()
    bare = _circuit()
    observed = _circuit()
    traces: list[np.ndarray] = []
    voltages: list[np.ndarray] = []

    def sink(circ: RecurrentCellCircuit, step) -> None:
        traces.append(np.asarray(circ._feature_traces).copy())
        voltages.append(np.asarray(circ._reservoir.state.voltage_relative_mv).copy())

    observed.observation_sink = sink
    for index, token_id in enumerate(ids):
        bare.observe(token_id)
        observed.observe(token_id)
        assert np.array_equal(traces[-1], np.asarray(bare._feature_traces))
        assert np.array_equal(
            voltages[-1], np.asarray(bare._reservoir.state.voltage_relative_mv)
        )
        assert np.array_equal(
            np.asarray(observed.feature_state_vector()),
            np.asarray(bare.feature_state_vector()),
        )
    assert observed.observed_tokens == bare.observed_tokens


def test_sink_on_traces_family_unchanged_path() -> None:
    circuit = _circuit(feature_family="traces")
    calls: list[int] = []
    circuit.observation_sink = lambda circ, step: calls.append(step.token_id)
    for token_id in _ids():
        circuit.observe(token_id)
    assert len(calls) == 24
