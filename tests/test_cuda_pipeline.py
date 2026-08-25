"""Double-buffered CUDA overlap pipeline: exactness and lifecycle guards."""

from __future__ import annotations

import os

import numpy as np
import pytest
from humanbrain.cuda import CudaRecurrentKernel, async_pipeline_enabled, cuda_available
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig

pytestmark = pytest.mark.skipif(
    not cuda_available(), reason="CUDA cooperative kernel is unavailable"
)


def _config(seed: int) -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
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
        readout_learning_rule="sparse_three_factor",
        random_seed=seed,
    )


@pytest.fixture(autouse=True)
def force_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")


def _ids(count: int, vocab: int) -> list[int]:
    rng = np.random.default_rng(5)
    return [int(value) for value in rng.integers(0, vocab, size=count)]


def _replay_winners(circuit: RecurrentCellCircuit, ids: list[int]) -> list[int]:
    circuit.reset_sequence()
    circuit.observe(ids[0])
    winners = []
    for token_id in ids[1:]:
        winners.append(int(circuit.race().winner))
        circuit.observe(token_id)
    return winners


def test_async_grow_matches_sync_bit_exact(monkeypatch) -> None:
    ids = _ids(48, 37)

    monkeypatch.setenv("HUMANBRAIN_CUDA_ASYNC", "0")
    sync_circuit = RecurrentCellCircuit(token_cells=37, config=_config(29))
    assert isinstance(sync_circuit._native_kernel, CudaRecurrentKernel)
    try:
        sync_circuit.grow_sequence(ids, epochs=2)
        sync_circuit.freeze_learning()
        sync_sha = sync_circuit.cell_synapse_state_sha256
        sync_eligibility = sync_circuit.eligibility.copy()
        sync_winners = _replay_winners(sync_circuit, ids)
    finally:
        sync_circuit._native_kernel.close()

    monkeypatch.setenv("HUMANBRAIN_CUDA_ASYNC", "1")
    async_circuit = RecurrentCellCircuit(token_cells=37, config=_config(29))
    assert isinstance(async_circuit._native_kernel, CudaRecurrentKernel)
    assert async_circuit._native_kernel.async_pipeline_active
    try:
        async_circuit.grow_sequence(ids, epochs=2)
        async_circuit.freeze_learning()
        async_sha = async_circuit.cell_synapse_state_sha256
        async_eligibility = async_circuit.eligibility.copy()
        async_winners = _replay_winners(async_circuit, ids)
    finally:
        async_circuit._native_kernel.close()

    assert async_sha == sync_sha
    assert np.array_equal(async_eligibility, sync_eligibility)
    assert async_winners == sync_winners


def test_sync_observe_unchanged_when_pipeline_active() -> None:
    """observe() stays synchronous and exact while the pipeline is armed."""
    token_ids = _ids(12, 37)
    results = []
    for enabled in ("0", "1"):
        os.environ["HUMANBRAIN_CUDA_ASYNC"] = enabled
        circuit = RecurrentCellCircuit(token_cells=37, config=_config(31))
        try:
            circuit.reset_sequence()
            spikes = [
                circuit.observe(token_id).reservoir_spikes for token_id in token_ids
            ]
            results.append((spikes, circuit.eligibility.copy()))
        finally:
            circuit._native_kernel.close()
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])


def test_pipeline_stall_and_order_guards() -> None:
    os.environ["HUMANBRAIN_CUDA_ASYNC"] = "1"
    circuit = RecurrentCellCircuit(token_cells=37, config=_config(37))
    kernel = circuit._native_kernel
    assert isinstance(kernel, CudaRecurrentKernel)
    try:
        depth = kernel._PIPELINE_DEPTH
        handles = [kernel.observe_async(token % 37) for token in range(depth)]
        with pytest.raises(RuntimeError, match="stall"):
            kernel.observe_async(3)
        with pytest.raises(RuntimeError, match="order"):
            kernel.collect(handles[1])
        assert kernel.collect(handles[0]) >= 0
        for handle in handles[1:]:
            kernel.collect(handle)
        assert kernel.pending_tokens == 0
    finally:
        kernel.close()


def test_drain_on_synchronize_matches_sync_replay() -> None:
    os.environ["HUMANBRAIN_CUDA_ASYNC"] = "1"
    circuit = RecurrentCellCircuit(token_cells=37, config=_config(41))
    kernel = circuit._native_kernel
    assert isinstance(kernel, CudaRecurrentKernel)
    ids = _ids(6, 37)
    try:
        assert kernel._PIPELINE_DEPTH >= 3
        handles = [kernel.observe_async(token_id) for token_id in ids[:3]]
        assert kernel.pending_tokens == len(handles)
        # Do NOT collect: any synchronize entry point must drain first.
        kernel.synchronize_to_host()
        assert kernel.pending_tokens == 0
        submitted = ids[:3]
        reference = RecurrentCellCircuit(token_cells=37, config=_config(41))
        try:
            reference.reset_sequence()
            for token_id in submitted:
                reference.observe(token_id)
            np.testing.assert_array_equal(
                circuit.eligibility, reference.eligibility
            )
        finally:
            reference._native_kernel.close()
    finally:
        kernel.close()


def test_disabled_pipeline_falls_back_to_sync_loop(monkeypatch) -> None:
    monkeypatch.setenv("HUMANBRAIN_CUDA_ASYNC", "0")
    assert not async_pipeline_enabled()
    circuit = RecurrentCellCircuit(token_cells=37, config=_config(43))
    kernel = circuit._native_kernel
    try:
        assert not kernel.async_pipeline_active
        with pytest.raises(RuntimeError, match="disabled"):
            kernel.observe_async(1)
        ids = _ids(20, 37)
        circuit.grow_sequence(ids, epochs=1)
        circuit.freeze_learning()
        assert circuit.cell_synapse_state_sha256
    finally:
        kernel.close()


def test_grow_sequence_rejects_out_of_range_upfront() -> None:
    os.environ["HUMANBRAIN_CUDA_ASYNC"] = "1"
    circuit = RecurrentCellCircuit(token_cells=37, config=_config(47))
    try:
        with pytest.raises(ValueError, match="outside output-cell population"):
            circuit.grow_sequence([1, 2, 999], epochs=1)
        assert circuit.observed_tokens == 0
    finally:
        circuit._native_kernel.close()
