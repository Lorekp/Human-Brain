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

import humanbrain.recurrent as recurrent_module
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


def test_observation_sink_requests_native_spike_count_mirroring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror_flags: list[bool] = []
    original = recurrent_module.recurrent_kernel

    def recording_kernel(*args, **kwargs):
        mirror_flags.append(kwargs["mirror_token_spike_counts"])
        return original(*args, **kwargs)

    monkeypatch.setattr(recurrent_module, "recurrent_kernel", recording_kernel)
    _circuit(observation_sink=_sink)
    assert mirror_flags == [True]


def test_restore_accepts_sink_before_native_rebind() -> None:
    circuit = _circuit()
    circuit.observe(0)
    circuit.strengthen(1)
    circuit.freeze_learning()
    state = circuit.capture_cell_synapse_state()
    restored = RecurrentCellCircuit.restore_cell_synapse_state(
        state,
        observation_sink=_sink,
    )
    assert restored.observation_sink is _sink


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
