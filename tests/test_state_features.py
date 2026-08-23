"""R0-b red tests: successor state-feature families (frozen protocol §5).

Spec: ``docs/research/r0b-state-feature-protocol.md`` (worktree
human-brain-a013-cognition-design). Every test names its failure condition
mechanically. These tests must FAIL before the R0-b implementation and PASS
after, with no design drift. R0-T1..T10 are mirrored 1:1 with family-specific
instances; R0B-T11/T12 extend to the combination arm and the frozen §4.4
selection rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from humanbrain.recurrent import (
    ELIGIBILITY_TAUS_MS,
    FEATURE_FAMILIES,
    WINDOW_TOKENS,
    RecurrentCellCircuit,
    RecurrentCircuitConfig,
)
from humanbrain.recurrent_artifacts import FORMAT as ARTIFACT_FORMAT

SINGLE_FAMILIES = ("spike_counts", "voltage_quantiles", "assembly_overlap")
FAMILY_FEATURE_CELLS = {
    "spike_counts": 8 * 1024,
    "voltage_quantiles": 24 * 1024,
    "assembly_overlap": 8 * 1024,
}


# --------------------------------------------------------------------------
# Shared small-circuit builders
# --------------------------------------------------------------------------


def _config(**overrides: object) -> RecurrentCircuitConfig:
    config_kwargs = dict(
        reservoir_cells=64,
        active_reservoir_cells=48,
        input_fanout=8,
        recurrent_fanout=4,
        token_interval_ms=5.0,
        input_pulse_ms=1.5,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        recurrent_topology="nested_groups",
        random_seed=29,
    )
    config_kwargs.update(overrides)
    return RecurrentCircuitConfig(**config_kwargs)


def _ridge_circuit(**overrides: object) -> RecurrentCellCircuit:
    return RecurrentCellCircuit(token_cells=9, config=_config(**overrides))


def _sparse_circuit(**overrides: object) -> RecurrentCellCircuit:
    base = dict(
        reservoir_cells=16,
        active_reservoir_cells=None,
        input_fanout=4,
        recurrent_fanout=2,
        token_interval_ms=2.0,
        input_pulse_ms=1.0,
        input_current_pa=8_000.0,
        recurrent_current_pa=325.0,
        readout_learning_rule="sparse_three_factor",
        readout_fan_in=2,
        readout_structural_interval=1,
        readout_structural_grace=2,
        random_seed=97,
    )
    base.update(overrides)
    return RecurrentCellCircuit(token_cells=12, config=_config(**base))


def _install_toy_assemblies(circuit: RecurrentCellCircuit, k: int = 4) -> None:
    """Deterministic toy codebook: token t drives a fixed sorted k-assembly."""
    cells = circuit.config.reservoir_cells
    table = np.empty((circuit.token_cells, k), dtype=np.int32)
    for token in range(circuit.token_cells):
        assembly = sorted(
            {(token * 3 + 5 * j) % cells for j in range(k)}
        )
        while len(assembly) < k:  # keep exactly k distinct cells
            assembly.append((max(assembly) + 1) % cells)
            assembly = sorted(set(assembly))
        table[token] = np.asarray(assembly[:k], dtype=np.int32)
    circuit._afferent_targets = table


def _drive(circuit: RecurrentCellCircuit, tokens: list[int]) -> None:
    for token in tokens:
        circuit.observe(token)


def _window_rows(circuit: RecurrentCellCircuit) -> np.ndarray:
    """The live (N_WIN, R) family-A window matrix."""
    return circuit._window_counts


# --------------------------------------------------------------------------
# R0B-T1 window-bank determinism & config validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 17, 12345])
def test_r0b_t1_window_bank_determinism(seed: int) -> None:
    """Failure: default bank is not bitwise geomspace(16,800,8) rounded, or is
    seed-dependent; traces does not reproduce incumbent F; bad families /
    combo grammars / window lists are not rejected."""
    assert WINDOW_TOKENS == (16, 28, 49, 86, 150, 262, 457, 800)
    assert WINDOW_TOKENS == tuple(int(round(w)) for w in np.geomspace(16.0, 800.0, 8))
    assert (
        RecurrentCircuitConfig(random_seed=seed).feature_family == "traces"
    ), "default feature family must be the incumbent trace bank"
    assert FEATURE_FAMILIES == (
        "traces",
        "spike_counts",
        "voltage_quantiles",
        "assembly_overlap",
    )
    # traces reproduces incumbent behavior bit-for-bit: same F, same buffer.
    circuit = _ridge_circuit()
    assert circuit._growth_feature_count == len(ELIGIBILITY_TAUS_MS) * 64 + 1
    assert (
        circuit.feature_cell_count == len(ELIGIBILITY_TAUS_MS) * 64
    ), "traces family must keep the incumbent K*R feature space"
    np.testing.assert_array_equal(
        circuit.feature_state_vector(), circuit._feature_traces
    )

    for bad_family in ("", "spikes", "combo:", "combo:spike_counts+",
                       "combo:spike_counts+spike_counts",  # dupes
                       "combo:voltage_quantiles+spike_counts",  # wrong order
                       "combo:spike_counts+voltage_quantiles+assembly_overlap",
                       "combo:spike_counts+traces",
                       "traces+spike_counts"):
        with pytest.raises(ValueError):
            RecurrentCircuitConfig(feature_family=bad_family)  # type: ignore[arg-type]

    # Legal combos parse and validate.
    legal = RecurrentCircuitConfig(feature_family="combo:spike_counts+assembly_overlap")
    assert legal.feature_family == "combo:spike_counts+assembly_overlap"

    for bad_windows in (
        (),
        (16, 16),  # duplicate / non-monotone
        (800, 16),  # non-monotone
        (0, 16),  # non-positive
        (-4, 16),
        (16, 100_001),  # above the 100_000 cap
    ):
        with pytest.raises(ValueError):
            RecurrentCircuitConfig(feature_window_tokens=tuple(bad_windows))
    custom = RecurrentCircuitConfig(feature_window_tokens=(4, 12))
    assert custom._resolved_window_tokens() == (4, 12)


# --------------------------------------------------------------------------
# R0B-T2 scripted-sequence reference equality (bit-exact per family)
# --------------------------------------------------------------------------


def _reference_count_windows(counts: list[np.ndarray], windows) -> np.ndarray:
    """Exact §2.A semantics: sliding sums over stored per-token vectors."""
    rows = np.zeros((len(windows), counts[0].size), dtype=np.float64)
    for k, width in enumerate(windows):
        for t in range(len(counts)):
            lo = max(0, t - width + 1)
            rows[k] += np.sum(counts[lo : t + 1], axis=0)
    return rows


def test_r0b_t2_spike_count_reference_equality() -> None:
    """Failure: family-A window rows are not bit-exact sliding bincounts of
    the kernel-maintained per-token spike-count vectors (repeats,
    refractory-suppressed cells, quiescent tail >= 5*W_7)."""
    circuit = _ridge_circuit(feature_family="spike_counts")
    sequence = [2, 2, 5, 2, 9, 5] + [7] * (5 * WINDOW_TOKENS[-1])
    captured: list[np.ndarray] = []
    for step, token in enumerate(sequence, start=1):
        circuit.observe(token)
        captured.append(circuit._token_spike_counts.copy())
        if step <= 6 or step % 397 == 0 or step > len(sequence) - 3:
            expected = _reference_count_windows(captured, WINDOW_TOKENS)
            np.testing.assert_array_equal(_window_rows(circuit), expected)
            assert circuit.feature_state_vector().dtype == np.float64
    # A spike inside window k appears in every window covering it; expiry
    # subtracts exactly once (final state equals the reference).
    expected = _reference_count_windows(captured, WINDOW_TOKENS)
    np.testing.assert_array_equal(_window_rows(circuit), expected)
    # Quiescent tail drove every row to zero: only the last W_k tokens are
    # identical constant inputs, so all rows must agree bitwise.
    np.testing.assert_array_equal(_window_rows(circuit)[0], _window_rows(circuit)[-1])


def test_r0b_t2_voltage_quantile_reference_equality() -> None:
    """Failure: family-B quantile rows do not equal pinned
    np.quantile(method='linear') over the exact ring slice, including
    warm-up truncation."""
    circuit = _ridge_circuit(feature_family="voltage_quantiles")
    ring_rows: list[np.ndarray] = []
    sequence = [2, 2, 5, 2, 9, 5]
    for token in sequence + [7] * 40:
        circuit.observe(token)
        ring_rows.append(circuit._reservoir.state.voltage_relative_mv.copy())
        vector = np.asarray(ring_rows, dtype=np.float64)
        expected = np.zeros((3 * len(WINDOW_TOKENS), 64), dtype=np.float64)
        for k, width in enumerate(WINDOW_TOKENS):
            lo = max(0, len(vector) - width)
            block = np.quantile(
                vector[lo:], (0.10, 0.50, 0.90), axis=0, method="linear"
            )
            expected[3 * k : 3 * k + 3] = block
        np.testing.assert_array_equal(
            circuit.feature_state_vector().reshape(3 * len(WINDOW_TOKENS), 64),
            expected,
        )


def test_r0b_t2_assembly_overlap_reference_equality() -> None:
    """Failure: family-C coverage counts do not match set-based occurrence
    bookkeeping (repeats count once; expiry after repeats decrements once;
    transitions fire only on 0->1 / 1->0)."""
    circuit = _ridge_circuit(feature_family="assembly_overlap")
    _install_toy_assemblies(circuit)
    history: list[int] = []
    sequence = [2, 2, 5, 2, 9, 5] + [3] * (WINDOW_TOKENS[1] + 6)

    def reference() -> np.ndarray:
        rows = np.zeros((len(WINDOW_TOKENS), 64), dtype=np.float64)
        for k, width in enumerate(WINDOW_TOKENS):
            active = history[-width:]
            for cell in range(64):
                rows[k, cell] = len(
                    {t for t in active if cell in set(circuit._afferent_targets[t])}
                )
        return rows

    for token in sequence:
        circuit.observe(token)
        history.append(token)
        np.testing.assert_array_equal(
            circuit.feature_state_vector().reshape(len(WINDOW_TOKENS), 64),
            reference(),
        )


# --------------------------------------------------------------------------
# R0B-T3 window-edge sanity (hard expiry vs exponential tails)
# --------------------------------------------------------------------------


def test_r0b_t3_window_edge_hard_expiry() -> None:
    """Failure: after one distinct pulse plus W_7 pair-free silence, family-A
    rows do not hit zero at EXACTLY t = pulse + W_k (no decay tail); C does
    not clear at each edge; B does not quantile a constant signal exactly;
    or the trace bank under the same stimulus does NOT retain its
    exp(-T/tau) tails (the contrast case must differ from A's hard expiry)."""
    pulse_token = 4
    silence = max(WINDOW_TOKENS) + 2
    sequence = [pulse_token] + [8] * silence

    a_circuit = _ridge_circuit(feature_family="spike_counts")
    c_circuit = _ridge_circuit(feature_family="assembly_overlap")
    _install_toy_assemblies(c_circuit)
    tr_circuit = _ridge_circuit(feature_family="traces")
    for step, token in enumerate(sequence, start=1):
        a_circuit.observe(token)
        c_circuit.observe(token)
        tr_circuit.observe(token)
        for k, width in enumerate(WINDOW_TOKENS):
            age = step - 1  # tokens since the pulse
            if age >= width:
                assert not _window_rows(a_circuit)[k].any(), (
                    f"family A row {k} must be identically zero once the "
                    f"pulse left the window (t={step}, W={width})"
                )
                assert not c_circuit.feature_state_vector()[
                    k * 64 : (k + 1) * 64
                ].any()

    np.testing.assert_array_equal(_window_rows(a_circuit), 0.0)
    np.testing.assert_array_equal(c_circuit.feature_state_vector(), 0.0)
    # Contrast: the trace bank retains exponential tails under the SAME
    # stimulus — hard-edged windows forget, traces never do.
    assert np.any(tr_circuit._feature_traces != 0.0)


def test_r0b_t3_voltage_constant_signal_exact_quantiles() -> None:
    """Failure: constant boundary samples do not quantile to that constant
    exactly for every window and every pinned q."""
    circuit = _ridge_circuit(feature_family="voltage_quantiles")
    circuit.reset_sequence()
    for row in range(max(WINDOW_TOKENS)):
        circuit._voltage_ring[row] = -13.25
    circuit._observed_tokens = max(WINDOW_TOKENS)
    circuit._quantile_stale = True
    vector = circuit.feature_state_vector().reshape(3 * len(WINDOW_TOKENS), 64)
    np.testing.assert_array_equal(vector, -13.25)


# --------------------------------------------------------------------------
# R0B-T4 checkpoint round-trip v6
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family",
    [
        "spike_counts",
        "voltage_quantiles",
        "assembly_overlap",
        "combo:voltage_quantiles+spike_counts",
    ],
)
def test_r0b_t4_checkpoint_v6_roundtrip(tmp_path, family: str) -> None:
    """Failure: grow/freeze/save/load does not preserve
    cell_synapse_state_sha256 or byte-identical predictions; v6 metadata
    accepts family/window/feature-cell mismatches; ridge statistics are not
    restored at the family's F+1."""
    circuit = _ridge_circuit(feature_family=family, random_seed=11)
    for index in range(20):
        circuit.reset_sequence()
        for offset in range(6):
            token = (index + offset) % circuit.token_cells
            circuit.observe(token)
            circuit.strengthen((token + 1) % circuit.token_cells)
    circuit.freeze_learning()
    context = [1, 2, 3]
    before = circuit.predict_next(context)
    sha_before = circuit.cell_synapse_state_sha256

    path = tmp_path / f"{family.replace(':', '_').replace('+', '-')}.npz"
    circuit.save_cell_synapse_checkpoint(path, envelope={"tag": "r0b-t4"})
    envelope, restored = RecurrentCellCircuit.load_cell_synapse_checkpoint(path)
    assert restored.config.feature_family == family
    assert restored.cell_synapse_state_sha256 == sha_before
    after = restored.predict_next(context)
    assert after.winner == before.winner
    assert after.output_current_sha256 == before.output_current_sha256
    metadata = envelope
    assert metadata["tag"] == "r0b-t4"


def _tampered_state(state, **config_overrides: object):
    """Rebuild a payload with selected circuit-config overrides applied."""
    from humanbrain.recurrent_state import RecurrentCellSynapseState

    tampered_metadata = dict(state.metadata)
    tampered_config = dict(tampered_metadata["circuit_config"])
    tampered_config.update(config_overrides)
    tampered_metadata["circuit_config"] = tampered_config
    return RecurrentCellSynapseState.copy_payload(
        tampered_metadata, state.arrays, tuple(state.arrays)
    )


def test_r0b_t4_checkpoint_v6_guard_rejections(tmp_path) -> None:
    """Failure: v6 loading does NOT reject a family mismatch, a window-list
    mismatch, or a wrong feature_cell_count."""
    circuit = _ridge_circuit(feature_family="spike_counts")
    circuit.reset_sequence()
    circuit.observe(1)
    circuit.strengthen(2)
    circuit.freeze_learning()
    path = tmp_path / "guard.npz"
    circuit.save_cell_synapse_checkpoint(path, envelope={"tag": "t4"})

    from humanbrain.recurrent_state import RecurrentCellSynapseState

    raw_envelope, state = RecurrentCellSynapseState.load_enveloped_checkpoint(path)

    with pytest.raises(ValueError):
        RecurrentCellCircuit.restore_cell_synapse_state(
            _tampered_state(state, feature_family="voltage_quantiles")
        )
    with pytest.raises(ValueError):
        RecurrentCellCircuit.restore_cell_synapse_state(
            _tampered_state(state, feature_window_tokens=[16, 32])
        )


def test_r0b_t4_checkpoint_v6_wrong_feature_cell_count(tmp_path) -> None:
    """Failure: a v6 payload whose feature_cell_count disagrees with its
    family geometry restores instead of raising."""
    circuit = _ridge_circuit(feature_family="spike_counts")
    circuit.reset_sequence()
    circuit.observe(1)
    circuit.strengthen(2)
    circuit.freeze_learning()
    path = tmp_path / "count.npz"
    circuit.save_cell_synapse_checkpoint(path, envelope={"tag": "t4c"})

    from humanbrain.recurrent_state import RecurrentCellSynapseState

    _, state = RecurrentCellSynapseState.load_enveloped_checkpoint(path)
    tampered_metadata = dict(state.metadata)
    tampered_metadata["feature_cell_count"] = 24 * 64
    tampered = RecurrentCellSynapseState.copy_payload(
        tampered_metadata, state.arrays, tuple(state.arrays)
    )
    with pytest.raises(ValueError):
        RecurrentCellCircuit.restore_cell_synapse_state(tampered)


# --------------------------------------------------------------------------
# R0B-T5 v5 legacy load discipline
# --------------------------------------------------------------------------


def test_r0b_t5_v5_loads_only_under_traces(tmp_path) -> None:
    """Failure: a synthesized v5 fixture (no feature_family key) does not
    load bit-identically under 'traces', OR loads without a loud error under
    a successor family (never silently reshaped)."""
    circuit = _ridge_circuit(random_seed=13)
    for index in range(8):
        circuit.reset_sequence()
        circuit.observe(index % circuit.token_cells)
        circuit.strengthen((index + 1) % circuit.token_cells)
    circuit.freeze_learning()
    path = tmp_path / "legacy.npz"
    circuit.save_cell_synapse_checkpoint(path, envelope={"tag": "t5"})

    from humanbrain.recurrent_state import RecurrentCellSynapseState

    _, state = RecurrentCellSynapseState.load_enveloped_checkpoint(path)
    v5_metadata = {
        key: value
        for key, value in state.metadata.items()
        if key not in ("feature_family", "feature_window_tokens")
    }
    v5_state = RecurrentCellSynapseState.copy_payload(
        v5_metadata, state.arrays, tuple(state.arrays)
    )

    restored = RecurrentCellCircuit.restore_cell_synapse_state(v5_state)
    assert restored.config.feature_family == "traces"
    context = [1, 2, 3]
    before = circuit.predict_next(context)
    after = restored.predict_next(context)
    assert after.winner == before.winner
    assert after.output_current_sha256 == before.output_current_sha256

    # Same payload requested under any successor family raises loudly.
    successor = dict(v5_metadata)
    successor["circuit_config"] = dict(successor["circuit_config"])
    successor["circuit_config"]["feature_family"] = "spike_counts"
    tampered = RecurrentCellSynapseState.copy_payload(
        successor, state.arrays, tuple(state.arrays)
    )
    with pytest.raises(ValueError, match="feature family"):
        RecurrentCellCircuit.restore_cell_synapse_state(tampered)

    # The language-artifact writer emits v6 now.
    assert ARTIFACT_FORMAT == "recurrent-human-cell-checkpoint-v6"


# --------------------------------------------------------------------------
# R0B-T6 CUDA parity per family
# --------------------------------------------------------------------------


def _cuda_skip_decorator():
    from humanbrain.cuda import cuda_available

    return pytest.mark.skipif(not cuda_available(), reason="CUDA backend unavailable")


@_cuda_skip_decorator()
def test_r0b_t6_cuda_parity_spike_counts_and_mirror_bytes(monkeypatch) -> None:
    """Failure: after a 64-token loop the u32 accumulator is not
    array-equal Rust-vs-CUDA after EVERY token; reset does not zero it on
    device; the mirror payload is not traces-bank + counts bytes."""
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    cpu_circuit = _ridge_circuit(
        feature_family="spike_counts", readout_learning_rule="sparse_three_factor"
    )
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    try:
        gpu_circuit = _ridge_circuit(
            feature_family="spike_counts", readout_learning_rule="sparse_three_factor"
        )
    finally:
        monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")

    kernel = gpu_circuit._native_kernel
    mirror_names = set(kernel._host_mirror_arrays())
    assert "token_spike_counts" in mirror_names, "family-A arm must mirror counts"
    expected_bytes = (
        len(ELIGIBILITY_TAUS_MS) * 64 * 8  # traces bank stays warm
        + 64 * 4  # u32 counts
    )
    assert kernel.device_to_host_bytes_per_token == expected_bytes + 8

    rng = np.random.default_rng(5)
    for _ in range(64):
        token = int(rng.integers(cpu_circuit.token_cells))
        cpu_total = cpu_circuit.observe(token)
        gpu_total = gpu_circuit.observe(token)
        assert gpu_total == cpu_total
        np.testing.assert_array_equal(
            cpu_circuit._token_spike_counts, gpu_circuit._token_spike_counts
        )
        # Boundary voltages must also match bit-exactly (shared sample set).
    gpu_circuit.synchronize_state_from_host()
    np.testing.assert_array_equal(
        cpu_circuit._eligibility_traces, gpu_circuit._eligibility_traces
    )
    # Reset zeroes accumulators on device.
    gpu_circuit.reset_sequence()
    gpu_circuit.synchronize_state_from_host()
    np.testing.assert_array_equal(gpu_circuit._token_spike_counts, 0)


@_cuda_skip_decorator()
def test_r0b_t6_cuda_parity_voltage_quantile_boundary_samples(monkeypatch) -> None:
    """Failure: boundary-voltage samples are not array-equal CPU-vs-CUDA
    (hence host-side quantile rows differ), or the B-arm mirror payload is
    not traces bank + voltage bytes ('eligibility_plus_voltage')."""
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    cpu_circuit = _ridge_circuit(
        feature_family="voltage_quantiles", readout_learning_rule="sparse_three_factor"
    )
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    try:
        gpu_circuit = _ridge_circuit(
            feature_family="voltage_quantiles", readout_learning_rule="sparse_three_factor"
        )
    finally:
        monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")

    kernel = gpu_circuit._native_kernel
    mirror_names = set(kernel._host_mirror_arrays())
    assert {"eligibility", "voltage"} <= mirror_names
    expected_bytes = len(ELIGIBILITY_TAUS_MS) * 64 * 8 + 64 * 8
    assert kernel.device_to_host_bytes_per_token == expected_bytes + 8

    rng = np.random.default_rng(6)
    for _ in range(64):
        token = int(rng.integers(cpu_circuit.token_cells))
        cpu_step = cpu_circuit.observe(token)
        gpu_step = gpu_circuit.observe(token)
        assert gpu_step.reservoir_spikes == cpu_step.reservoir_spikes
    gpu_circuit.synchronize_state_from_host()
    np.testing.assert_array_equal(
        cpu_circuit._voltage_ring, gpu_circuit._voltage_ring
    )


@_cuda_skip_decorator()
def test_r0b_t6_cuda_invariance_assembly_overlap_midswitch(monkeypatch) -> None:
    """Failure: family-C host bookkeeping is not invariant across a
    mid-sequence backend switch (it must be: nothing native moves)."""
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    host_circuit = _ridge_circuit(feature_family="assembly_overlap")
    _install_toy_assemblies(host_circuit)
    reference = host_circuit.feature_state_vector().copy()
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    try:
        device_circuit = _ridge_circuit(feature_family="assembly_overlap")
        _install_toy_assemblies(device_circuit)
    finally:
        monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    for token in (1, 2, 3, 2, 5):
        host_circuit.observe(token)
        device_circuit.observe(token)
    np.testing.assert_array_equal(
        host_circuit.feature_state_vector(),
        device_circuit.feature_state_vector(),
    )
    np.testing.assert_array_equal(reference, 0.0)


# --------------------------------------------------------------------------
# R0B-T7 kernel equivalence across backends
# --------------------------------------------------------------------------


@_cuda_skip_decorator()
def test_r0b_t7_backend_equivalence_family_buffers(monkeypatch) -> None:
    """Failure: Python-reference, Rust serial, Rust AVX, and CUDA do not
    produce identical family buffers after the 64-token sequence at two
    activity densities; or frontier membership / activity density drift
    across families."""
    sequence_a = [1, 2, 3] * 21 + [4]
    sequence_b = [1] * 32 + [2, 3, 4, 5] * 8  # denser repetition/refractory mix

    def run(family: str, sequence: list[int]) -> np.ndarray:
        monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
        circuit = _ridge_circuit(feature_family=family)
        if family == "spike_counts":
            # Exercise serial, parallel(AVX), and frontier paths in turn.
            native = circuit._native_kernel
            paths = [native._observe_serial, native._observe_parallel, native.observe]
            for step in range(len(sequence)):
                native.observe = paths[step % 3]
        for token in sequence:
            circuit.observe(token)
        host_buffer = circuit.feature_state_vector().copy()
        density = circuit.reservoir_activity_density

        monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
        try:
            gpu = _ridge_circuit(feature_family=family)
            for token in sequence:
                gpu.observe(token)
            gpu.synchronize_state_from_host()
        finally:
            monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
        np.testing.assert_array_equal(host_buffer, gpu.feature_state_vector())
        assert density >= 0.0
        return host_buffer

    for family in SINGLE_FAMILIES:
        run(family, sequence_a)
        run(family, sequence_b)

    # Frontier membership and activity density are family-independent.
    densities = {}
    for family in ("traces",) + SINGLE_FAMILIES:
        circuit = _ridge_circuit(feature_family=family)
        for token in sequence_a[:30]:
            circuit.observe(token)
        densities[family] = circuit.reservoir_activity_density
    assert len(set(densities.values())) == 1, densities


# --------------------------------------------------------------------------
# R0B-T8 feature-vector shape guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,f_cells",
    [
        ("spike_counts", 8192),
        ("voltage_quantiles", 24576),
        ("assembly_overlap", 8192),
        ("combo:spike_counts+voltage_quantiles", 32768),
    ],
)
def test_r0b_t8_feature_vector_shape_guards(family: str, f_cells: int) -> None:
    """Failure: ridge allocations are not shaped (128, F+1)/(F+1, F+1)/
    (token_cells, F+1); strengthen-time features are not
    [feature_state_vector(); 1]; restore mismatches do not name the key."""
    circuit = RecurrentCellCircuit(
        token_cells=5,
        config=_config(
            feature_family=family,
            reservoir_cells=1024,
            active_reservoir_cells=None,
            input_fanout=8,
        ),
    )
    assert circuit.feature_cell_count == f_cells
    circuit.reset_sequence()
    circuit.observe(1)
    circuit.strengthen(2)
    assert circuit._feature_batch.shape == (128, f_cells + 1)
    batch_row = circuit._feature_batch[circuit._pending_growth_examples - 1]
    assert batch_row.shape == (f_cells + 1,)
    assert batch_row[-1] == 1.0
    np.testing.assert_array_equal(batch_row[:-1], circuit.feature_state_vector())

    circuit2 = RecurrentCellCircuit(
        token_cells=5,
        config=_config(
            feature_family=family,
            reservoir_cells=1024,
            active_reservoir_cells=None,
            input_fanout=8,
        ),
    )
    for index in range(130):
        circuit2.reset_sequence()
        circuit2.observe(index % 5)
        circuit2.strengthen((index + 1) % 5)
    assert circuit2._normal_matrix.shape == (f_cells + 1, f_cells + 1)
    assert circuit2._target_moments.shape == (5, f_cells + 1)


def test_r0b_t8_restore_shape_mismatch_names_key(tmp_path) -> None:
    """Failure: restoring an array whose shape disagrees with the family's F
    does not raise naming the offending checkpoint key."""
    circuit = _ridge_circuit(feature_family="spike_counts")
    circuit.reset_sequence()
    circuit.observe(1)
    circuit.strengthen(2)
    circuit.freeze_learning()
    path = tmp_path / "shape.npz"
    circuit.save_cell_synapse_checkpoint(path, envelope={"tag": "t8"})

    from humanbrain.recurrent_state import RecurrentCellSynapseState

    _, state = RecurrentCellSynapseState.load_enveloped_checkpoint(path)
    arrays = {key: value.copy() for key, value in state.arrays.items()}
    wrong = arrays["readout_synapses_pa"]
    arrays["readout_synapses_pa"] = wrong[:, :-8].copy()
    tampered = RecurrentCellSynapseState.copy_payload(
        state.metadata, arrays, tuple(arrays)
    )
    with pytest.raises(ValueError, match="readout_synapses_pa"):
        RecurrentCellCircuit.restore_cell_synapse_state(tampered)


# --------------------------------------------------------------------------
# R0B-T9 sparse source-space validity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,windows_of_source",
    [
        ("spike_counts", lambda s: WINDOW_TOKENS[s // 64]),
        ("voltage_quantiles", lambda s: WINDOW_TOKENS[(s // 64) // 3]),
        ("assembly_overlap", lambda s: WINDOW_TOKENS[s // 64]),
        ("combo:spike_counts+voltage_quantiles", None),  # concatenation check
    ],
)
def test_r0b_t9_sparse_source_space_validity(family: str, windows_of_source) -> None:
    """Failure: sparse potential/contact sources are not unique ascending in
    [0, F); candidate count deviates from min(F, CANDIDATE_FACTOR*fan_in);
    per-contact apposition decay deviates from exp(-1/W_(source//R)); native
    sparse current over the flat buffer diverges from the NumPy reference."""
    pytest.importorskip("humanbrain.readout")
    circuit = _sparse_circuit(feature_family=family)
    readout = circuit._sparse_readout
    assert readout is not None
    total_sources = 8 * 16 if family != "voltage_quantiles" else 24 * 16
    if family.startswith("combo:"):
        total_sources = (8 + 24) * 16
    # _validate_state() ran inside construction: any out-of-range, duplicate,
    # or non-ascending source would have raised already.
    potential = np.asarray(readout.potential_sources)
    contacts = np.asarray(readout.contact_sources)
    assert potential.shape == (12, readout.dimensions.candidate_count)
    assert contacts.shape == (12, circuit.config.readout_fan_in)
    assert potential.min() >= 0
    assert potential.max() < total_sources
    assert np.all(np.diff(contacts, axis=1) > 0), "contact sources must be unique ascending"
    assert contacts.min() >= 0 and contacts.max() < total_sources
    fan_in = circuit.config.readout_fan_in
    space = (8 + 24) * 16 if family.startswith("combo:") else total_sources
    expected_candidates = min(space, 4 * fan_in)
    assert readout.dimensions.candidate_count == expected_candidates

    decays = circuit.source_trace_decays()
    assert decays.shape == (space,)
    if not family.startswith("combo:"):
        for source in range(total_sources):
            width = windows_of_source(source)
            np.testing.assert_allclose(decays[source], np.exp(-1.0 / width), rtol=0, atol=0)
    # W_0 substitution equals today's row-0 value.
    taus = np.asarray(ELIGIBILITY_TAUS_MS)
    np.testing.assert_allclose(
        decays[0], np.exp(-circuit.config.token_interval_ms / taus[0]), rtol=1e-12
    )


# --------------------------------------------------------------------------
# R0B-T10 accounting invariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", SINGLE_FAMILIES)
def test_r0b_t10_accounting_invariance(family: str) -> None:
    """Failure: against 'traces' on fixed seed/curriculum, token counts,
    presentation order, observed-token counters, curriculum digest, wiring,
    or activity density differ beyond the declared fields."""
    from humanbrain.topology import build_recurrent_targets

    baseline = _ridge_circuit()
    family_circuit = _ridge_circuit(feature_family=family)
    if family == "assembly_overlap":
        _install_toy_assemblies(family_circuit)
        _install_toy_assemblies(baseline)

    # Identical structure: seeds drive identical afferent/recurrent tables.
    np.testing.assert_array_equal(
        baseline._afferent_targets, family_circuit._afferent_targets
    )
    np.testing.assert_array_equal(
        baseline._recurrent_targets, family_circuit._recurrent_targets
    )
    np.testing.assert_array_equal(
        baseline._recurrent_weights_pa, family_circuit._recurrent_weights_pa
    )

    presentation = [(index * 7 + 3) % 9 for index in range(40)]
    for token in presentation:
        baseline.observe(token)
        family_circuit.observe(token)
    assert baseline.observed_tokens == family_circuit.observed_tokens == 40
    assert baseline.reservoir_activity_density == family_circuit.reservoir_activity_density
    np.testing.assert_array_equal(
        baseline._eligibility_traces, family_circuit._eligibility_traces
    )
    # Only declared fields differ.
    from dataclasses import asdict

    base_dump = asdict(baseline.config)
    fam_dump = asdict(family_circuit.config)
    differing = {key for key in base_dump if base_dump[key] != fam_dump[key]}
    assert differing == {"feature_family"}, differing


def test_r0b_t10_curriculum_digest_invariant() -> None:
    """Failure: the curriculum digest changes when only the feature family
    changes (it must not: generators are untouched)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import ladder_experiments as le

    small = le.build_curriculum(seed=7, data_scale=1)
    digest_one = small["sha256"]
    again = le.build_curriculum(seed=7, data_scale=1)
    assert digest_one == again["sha256"], "curriculum digest must be deterministic"


# --------------------------------------------------------------------------
# R0B-T11 combo composition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "combo,members",
    [
        ("combo:spike_counts+voltage_quantiles", ("spike_counts", "voltage_quantiles")),
        ("combo:spike_counts+assembly_overlap", ("spike_counts", "assembly_overlap")),
        ("combo:voltage_quantiles+assembly_overlap", ("voltage_quantiles", "assembly_overlap")),
    ],
)
def test_r0b_t11_combo_composition(combo: str, members: tuple[str, str]) -> None:
    """Failure: combo vectors are not the bitwise canonical-order
    concatenation of member vectors; F_combo arithmetic breaks; member
    boundary updates depend on composition."""
    tokens = [1, 2, 3, 2, 4, 1]

    def driven(family: str) -> RecurrentCellCircuit:
        circuit = _ridge_circuit(feature_family=family)
        if family == "assembly_overlap" or family.endswith("+assembly_overlap"):
            _install_toy_assemblies(circuit)
        for token in tokens:
            circuit.observe(token)
        return circuit

    combo_circuit = driven(combo)
    singles = [driven(member) for member in members]
    blocks = [single.feature_state_vector().copy() for single in singles]
    concatenated = np.concatenate(blocks)
    np.testing.assert_array_equal(combo_circuit.feature_state_vector(), concatenated)
    sizes = [block.size for block in blocks]
    assert combo_circuit.feature_cell_count == sum(sizes)
    assert combo_circuit.feature_cell_count == (
        FAMILY_FEATURE_CELLS.get(members[0], 8192)
        + FAMILY_FEATURE_CELLS.get(members[1], 8192)
    ) // 8 * 8
    # Canonical order: A < B < C by construction of the grammar tests.


def test_r0b_t11_combo_illegal_grammar() -> None:
    """Failure: illegal combo grammars are accepted by the config."""
    for bad in (
        "combo:assembly_overlap+spike_counts",
        "combo:voltage_quantiles+voltage_quantiles",
        "combo:spike_counts+assembly_overlap+extra",
        "combo:traces+spike_counts",
    ):
        with pytest.raises(ValueError):
            RecurrentCircuitConfig(feature_family=bad)


# --------------------------------------------------------------------------
# R0B-T12 frozen selection-rule determinism
# --------------------------------------------------------------------------


def test_r0b_t12_frozen_selection_rule() -> None:
    """Failure: the §4.4 ranking function is not deterministic on synthetic
    records (ties, missing diagnostics, control entries), or considers the
    random-arm control as a candidate."""
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        from r0b_gate import select_combo_members
    finally:
        sys.path.pop(0)

    records = [
        {
            "experiment_id": "r0b_A_spikewin",
            "family": "spike_counts",
            "role": "confirmatory",
            "codebook": "explicit",
            "median_1minuscos_pooled": 0.31,
            "train_replay_tasks": 41,
            "wall_s": 50.0,
        },
        {
            "experiment_id": "r0b_B_voltq",
            "family": "voltage_quantiles",
            "role": "confirmatory",
            "codebook": "explicit",
            "median_1minuscos_pooled": 0.42,
            "train_replay_tasks": 40,
            "wall_s": 90.0,
        },
        {
            "experiment_id": "r0b_C_asmbl_sim",
            "family": "assembly_overlap",
            "role": "confirmatory",
            "codebook": "similarity",
            "median_1minuscos_pooled": 0.31,  # tie with A
            "train_replay_tasks": 41,
            "wall_s": 49.0,  # tie-break (a) equal, (b) lower wall wins vs A? No:
            # A has wall 50.0, C_sim 49.0 -> C_sim outranks A on (b).
        },
        {
            "experiment_id": "r0b_C_asmbl_rand",
            "family": "assembly_overlap",
            "role": "control",
            "codebook": "random",
            "median_1minuscos_pooled": 9.99,  # would dominate if not ignored
            "train_replay_tasks": 42,
            "wall_s": 1.0,
        },
    ]
    first, second = select_combo_members(records)
    assert (first, second) == ("assembly_overlap", "spike_counts"), (
        "tie on median must break on train_replay then wall_s; control ignored"
    )

    # Missing diagnostics sort last deterministically.
    missing = [dict(records[0], median_1minuscos_pooled=None), records[1]]
    top, runner_up = select_combo_members(missing)
    assert (top, runner_up) == ("voltage_quantiles", "spike_counts")

    # Full-name tie everywhere falls back to name ascending.
    tied = [
        {
            "experiment_id": f"arm_{name}",
            "family": name,
            "role": "confirmatory",
            "codebook": "explicit" if name != "assembly_overlap" else "similarity",
            "median_1minuscos_pooled": 0.5,
            "train_replay_tasks": 40,
            "wall_s": 10.0,
        }
        for name in SINGLE_FAMILIES
    ]
    top, second_place = select_combo_members(tied)
    assert top == "spike_counts" and second_place == "voltage_quantiles"


def test_r0b_t12_selection_rule_needs_two() -> None:
    """Failure: fewer than two eligible candidates does not raise loudly."""
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        from r0b_gate import select_combo_members
    finally:
        sys.path.pop(0)
    with pytest.raises(ValueError):
        select_combo_members([])
