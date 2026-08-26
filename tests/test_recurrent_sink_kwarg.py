"""Red-first regression: RecurrentCellCircuit must honor the observation_sink kwarg.

Bug: __init__ accepted observation_sink but a late unconditional
``self._observation_sink = None`` (~line 504) silently dropped any value the
caller passed. Constructing with observation_sink=X must leave
circuit._observation_sink is X after init completes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


def _sink(circuit: RecurrentCellCircuit, step) -> None:  # pragma: no cover - identity only
    return None


def _circuit(**kwargs) -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=16,
        config=RecurrentCircuitConfig(
            reservoir_cells=32,
            input_fanout=4,
            recurrent_fanout=4,
            random_seed=7,
            readout_learning_rule="ridge",
        ),
        **kwargs,
    )


def test_observation_sink_kwarg_survives_init() -> None:
    sink = _sink
    circuit = _circuit(observation_sink=sink)
    assert circuit._observation_sink is sink
    assert circuit.observation_sink is sink


def test_observation_sink_default_stays_none() -> None:
    circuit = _circuit()
    assert circuit._observation_sink is None
    assert circuit.observation_sink is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
