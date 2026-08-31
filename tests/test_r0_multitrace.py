"""R0 red tests: multi-timescale eligibility traces (frozen protocol §4).

Every test names its failure condition mechanically. The spec is
``docs/research/r0-multitrace-protocol.md`` @ 771bd93; these tests must FAIL
before the R0 implementation and PASS after, with no design drift.
"""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pytest

from humanbrain.recurrent import (
    ELIGIBILITY_TAUS_MS,
    RecurrentCellCircuit,
    RecurrentCircuitConfig,
)
from humanbrain.recurrent_artifacts import FORMAT as ARTIFACT_FORMAT_V5


# --------------------------------------------------------------------------
# Shared small-circuit builders
# --------------------------------------------------------------------------


def _ridge_circuit(**overrides: object) -> RecurrentCellCircuit:
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
    return RecurrentCellCircuit(
        token_cells=9,
        config=RecurrentCircuitConfig(**config_kwargs),
    )


def _sparse_circuit(**overrides: object) -> RecurrentCellCircuit:
    config_kwargs = dict(
        reservoir_cells=16,
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
    config_kwargs.update(overrides)
    return RecurrentCellCircuit(
        token_cells=12,
        config=RecurrentCircuitConfig(**config_kwargs),
    )


def _trace_matrix(circuit: RecurrentCellCircuit) -> np.ndarray:
    """The live (K, R) trace bank; row k is trace k."""
    return circuit._eligibility_traces.reshape(
        circuit.trace_count, circuit.config.reservoir_cells
    )


# --------------------------------------------------------------------------
# R0-T1 tau-bank determinism
# --------------------------------------------------------------------------


def test_r0_t1_tau_bank_determinism() -> None:
    """Failure: default bank is not exactly geomspace(80,4000,8) or the shim
    does not build the 1-tuple; invalid lists are not rejected."""
    assert ELIGIBILITY_TAUS_MS == tuple(np.geomspace(80.0, 4000.0, 8))
    assert len(ELIGIBILITY_TAUS_MS) == 8
    assert ELIGIBILITY_TAUS_MS[0] == 80.0 and ELIGIBILITY_TAUS_MS[-1] == 4000.0
    for seed in (0, 17, 12345):
        assert (
            RecurrentCircuitConfig(random_seed=seed).eligibility_taus_ms
            == ELIGIBILITY_TAUS_MS
        ), "default tau bank must be independent of random_seed"
    shim = RecurrentCircuitConfig(eligibility_tau_ms=80.0)
    assert shim.eligibility_taus_ms == (80.0,)
    shim_40 = RecurrentCircuitConfig(eligibility_tau_ms=40.0)
    assert shim_40.eligibility_taus_ms == (40.0,)
    # JSON round trips turn tuples into lists; constructors must accept them.
    from_list = RecurrentCircuitConfig(eligibility_taus_ms=list(ELIGIBILITY_TAUS_MS))
    assert from_list.eligibility_taus_ms == ELIGIBILITY_TAUS_MS
    for bad in (
        (),
        (80.0, 80.0),  # duplicate
        (4000.0, 80.0),  # non-monotone
        (-1.0, 2.0, 3.0),  # non-positive
        (80.0, float("nan")),  # non-finite
        (80.0, float("inf")),
    ):
        with pytest.raises(ValueError):
            RecurrentCircuitConfig(eligibility_taus_ms=tuple(bad))


# --------------------------------------------------------------------------
# R0-T2 per-trace decay vs hand-computed reference (bit-exact op order)
# --------------------------------------------------------------------------


def _reference_glif_step(
    circuit: RecurrentCellCircuit, step_current: np.ndarray
) -> np.ndarray:
    """Incumbent NumPy GLIF oracle, retained as the native equivalence baseline."""
    population = circuit._reservoir
    p = population.parameters
    state = population.state
    constants = population.constants
    active = state.refractory_steps == 0
    np.subtract(state.refractory_steps, 1, out=state.refractory_steps, where=~active)
    crossed = np.zeros(state.refractory_steps.shape, dtype=bool)
    rows = np.flatnonzero(active)
    previous_voltage = state.voltage_relative_mv[rows]
    previous_after = state.after_spike_currents_pa[rows]
    total_current = step_current[rows] + previous_after.sum(axis=1)
    next_voltage = previous_voltage + constants.voltage_step_scale * (
        total_current - p.leak_ns * previous_voltage
    )
    beta = total_current * constants.threshold_beta_scale
    next_voltage_threshold = (
        constants.threshold_exp_decay * state.threshold_voltage_mv[rows]
        + constants.threshold_voltage_factor * previous_voltage
        + constants.threshold_beta_factor * beta
    )
    next_spike_threshold = state.threshold_spike_mv[rows] * constants.spike_threshold_decay
    next_after = previous_after * constants.after_spike_decay
    state.voltage_relative_mv[rows] = next_voltage
    state.threshold_voltage_mv[rows] = next_voltage_threshold
    state.threshold_spike_mv[rows] = next_spike_threshold
    state.after_spike_currents_pa[rows] = next_after
    crossed_rows = next_voltage > (
        p.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold
    )
    if not crossed_rows.any():
        return crossed
    spike_rows = rows[crossed_rows]
    crossed[spike_rows] = True
    state.voltage_relative_mv[spike_rows] = (
        p.voltage_reset_scale * next_voltage[crossed_rows]
        + p.voltage_reset_offset_mv
    )
    state.after_spike_currents_pa[spike_rows] = (
        constants.after_spike_amplitudes_pa
        + next_after[crossed_rows] * constants.after_spike_cut_decay
    )
    state.threshold_spike_mv[spike_rows] = (
        next_spike_threshold[crossed_rows] * constants.spike_cut_threshold_decay
        + p.threshold_spike_increment_mv
    )
    state.refractory_steps[spike_rows] = p.spike_cut_steps
    return crossed


def _multirow_python_reference_observe(
    circuit: RecurrentCellCircuit, token_id: int
) -> int:
    """NumPy oracle applying EXACTLY spike-determine -> decay-all -> increment.

    Per 0.05 ms step, every row k decays as ``e_k <- e_k * lambda_k`` then
    every active spiking cell gains 1.0 in EVERY row, once per spike event.
    Advances the circuit's live state in place, like the incumbent oracle.
    """
    afferents = circuit._afferent_targets[token_id]
    step_current_pa = np.empty_like(circuit._synaptic_current_pa)
    spike_total = 0
    for step in range(circuit._token_steps):
        np.copyto(step_current_pa, circuit._synaptic_current_pa)
        if step < circuit._pulse_steps:
            step_current_pa[afferents] += circuit.config.input_current_pa
        spikes = _reference_glif_step(circuit, step_current_pa)
        circuit._synaptic_current_pa *= circuit._recurrent_decay
        # Operate on the (K, R) view of the contiguous flat bank; in-place ops
        # write straight through the view.
        traces = circuit._eligibility_traces.reshape(
            circuit.trace_count, circuit.config.reservoir_cells
        )
        traces *= circuit._eligibility_decays[:, None]
        if spikes.any():
            presynaptic = np.flatnonzero(spikes)
            spike_total += int(presynaptic.size)
            traces += spikes.astype(np.float64)[None, :]  # every row gains once
            targets = circuit._recurrent_targets[presynaptic].reshape(-1)
            weights = circuit._recurrent_weights_pa[presynaptic].reshape(-1)
            np.add.at(circuit._synaptic_current_pa, targets, weights)
    circuit._observed_tokens += 1
    return spike_total


def test_r0_t2_per_trace_decay_bit_exact_vs_hand_computed() -> None:
    """Failure: any row diverges bitwise from e<-e*lambda_k; +=spike ordering;
    a spike missing from some row; refractory steps contributing."""
    config = dict(
        reservoir_cells=48,
        input_fanout=6,
        recurrent_fanout=4,
        token_interval_ms=5.0,
        input_pulse_ms=1.5,
        # Afferent input alone can never spike a cell: all spikes in the
        # scripted train come from the manually seeded synaptic currents, so
        # the tail below is quiescent by construction.
        input_current_pa=1.0,
        recurrent_current_pa=650.0,
        recurrent_topology="nested_groups",
        random_seed=11,
    )
    actual = _ridge_circuit(**config)
    expected = _ridge_circuit(**config)
    for circuit in (actual, expected):
        circuit.reset_sequence()
        circuit._synaptic_current_pa[[3, 9, 17]] = 25_000.0
    tokens = [3, 3, 3, 7, 7, 1, 3, 5, 5, 2]  # immediate repeats force refractory
    for token in tokens:
        step = actual.observe(token)
        expected_spikes = _multirow_python_reference_observe(expected, token)
        assert step.reservoir_spikes == expected_spikes
        np.testing.assert_array_equal(  # bit-exact, NOT allclose
            _trace_matrix(actual), _trace_matrix(expected)
        )
    # A spike appearing in row k must appear in ALL rows: seeded spikers are
    # nonzero in every row while never-spiked cells are zero in every row.
    matrix = _trace_matrix(actual)
    ever_spiked = np.flatnonzero(np.count_nonzero(matrix, axis=0) > 0)
    assert ever_spiked.size > 0, "scripted train produced no spikes"
    assert np.all(matrix[:, ever_spiked] != 0.0)
    quiet = np.setdiff1d(np.arange(matrix.shape[1]), ever_spiked)
    assert np.all(matrix[:, quiet] == 0.0)

    # Quiescent tail >= 5*tau_7 (20 s bio time = 4000 tokens at T = 5 ms).
    # First DRAIN through the exact oracle: residual recurrent activity can
    # still spike for a few tokens even though afferent input alone (1 pA)
    # never can. Once 25 consecutive tokens are spike-free, every remaining
    # sub-step is pure per-row decay (currents only shrink from there), which
    # the reference computes as iterated multiplication.
    tail_token = 4
    quiet_run = 0
    while quiet_run < 25:
        native_spikes = actual.observe(tail_token).reservoir_spikes
        reference_spikes = _multirow_python_reference_observe(expected, tail_token)
        assert native_spikes == reference_spikes
        np.testing.assert_array_equal(
            _trace_matrix(actual), _trace_matrix(expected)
        )
        quiet_run = quiet_run + 1 if native_spikes == 0 else 0
    tail_tokens = 4000
    for index in range(tail_tokens):
        assert actual.observe(tail_token).reservoir_spikes == 0, (
            f"tail step {index} spiked; the tail is not quiescent"
        )
    lambdas = actual._eligibility_decays
    head = _trace_matrix(expected).copy()
    for _ in range(tail_tokens * actual._token_steps):
        head *= lambdas[:, None]
    np.testing.assert_array_equal(_trace_matrix(actual), head)


# --------------------------------------------------------------------------
# R0-T3 token-window sanity
# --------------------------------------------------------------------------


def test_r0_t3_token_window_ratios_match_exp_minus_T_over_tau() -> None:
    """Failure: per-token amplitude ratios deviate from exp(-T/tau_k) beyond
    1e-12 relative (T = 5 ms)."""
    config = dict(
        reservoir_cells=256,
        input_fanout=4,
        recurrent_fanout=2,
        recurrent_current_pa=1.0,  # negligible spread: no re-spikes of the probe
        input_current_pa=4_000.0,
        random_seed=5,
    )
    circuit = _ridge_circuit(**config)
    # Cells start at rest: repeat the priming token until its afferents have
    # spiked (incumbent dynamics show the first one or two tokens at rest can
    # stay silent), then switch to tokens that never pulse the probe again.
    circuit.observe(0)
    for _ in range(6):
        circuit.observe(0)
        matrix = _trace_matrix(circuit)
        if np.count_nonzero(matrix[0]):
            break
    assert np.count_nonzero(matrix[0]), "priming token produced no spikes"
    later_afferents = np.concatenate(
        [circuit._afferent_targets[t] for t in range(1, 9)]
    )
    candidates = [
        int(cell)
        for cell in np.argsort(-matrix[0])
        if cell not in set(later_afferents.tolist())
    ]
    assert candidates, "no probe cell survived the later afferent sets"
    probe = None
    for candidate in candidates:
        if np.all(matrix[:, candidate] > 0.0):
            probe = candidate
            break
    assert probe is not None, "probe did not spike into every row"
    expected_ratios = np.exp(
        -circuit.config.token_interval_ms / np.asarray(ELIGIBILITY_TAUS_MS)
    )
    values = matrix[:, probe].copy()
    for token in range(1, 9):
        circuit.observe(token)
        current = _trace_matrix(circuit)[:, probe]
        ratios = current / values
        np.testing.assert_allclose(
            ratios,
            expected_ratios,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"token-window ratio mismatch at observe {token}",
        )
        values = current.copy()


# --------------------------------------------------------------------------
# R0-T4 checkpoint round-trip v5
# --------------------------------------------------------------------------


def _grow_and_freeze_ridge(tmp_path, name: str):
    circuit = _ridge_circuit()
    rng = np.random.default_rng(3)
    sequences = [rng.integers(0, 9, 24) for _ in range(4)]
    for sequence in sequences:
        circuit.grow_sequence([int(token) for token in sequence], epochs=1)
    circuit.freeze_learning()
    context = [int(token) for token in sequences[0][:10]]
    envelope = {"model": "r0-red-test", "tokenizer": "identity"}
    path = tmp_path / f"{name}.npz"
    circuit.save_cell_synapse_checkpoint(path, envelope=envelope)
    return circuit, path, context


def test_r0_t4_checkpoint_roundtrip_v5(tmp_path) -> None:
    """Failure: v5 round-trip changes the state hash or predictions; tampered
    taus / feature_cell_count load; ridge stats restore at a wrong F."""
    circuit, path, context = _grow_and_freeze_ridge(tmp_path, "v5")
    digest_before = circuit.cell_synapse_state_sha256
    prediction_before = circuit.predict_next(context)

    envelope, loaded = RecurrentCellCircuit.load_cell_synapse_checkpoint(path)
    assert loaded.cell_synapse_state_sha256 == digest_before
    prediction_after = loaded.predict_next(context)
    assert prediction_after.winner == prediction_before.winner
    np.testing.assert_array_equal(
        loaded.output_current_pa(), circuit.output_current_pa()
    )

    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
    cell_metadata = metadata["cell_synapse_state"]
    assert metadata["envelope"] == {
        "model": "r0-red-test",
        "tokenizer": "identity",
    }
    feature_cell_count = len(ELIGIBILITY_TAUS_MS) * 64
    assert cell_metadata["eligibility_taus_ms"] == list(ELIGIBILITY_TAUS_MS)
    assert cell_metadata["feature_cell_count"] == feature_cell_count
    assert loaded._normal_matrix.shape == (feature_cell_count + 1,) * 2
    assert loaded._target_moments.shape == (9, feature_cell_count + 1)
    assert loaded._growth_feature_count == feature_cell_count + 1

    def rewrite_metadata(mutate) -> bytes:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        parsed = json.loads(str(arrays["metadata"]))
        mutate(parsed["cell_synapse_state"])
        arrays["metadata"] = np.asarray(json.dumps(parsed, sort_keys=True))
        buffer = tmp_path / "tampered.npz"
        with buffer.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return buffer.read_bytes()

    def break_taus(state: dict) -> None:
        state["eligibility_taus_ms"] = list(ELIGIBILITY_TAUS_MS[:-1]) + [5000.0]

    def break_feature_count(state: dict) -> None:
        state["feature_cell_count"] = feature_cell_count + 1

    for mutator in (break_taus, break_feature_count):
        tampered_path = tmp_path / "tampered.npz"
        tampered_path.write_bytes(rewrite_metadata(mutator))
        with pytest.raises(ValueError):
            RecurrentCellCircuit.load_cell_synapse_checkpoint(tampered_path)


def test_r0_t4b_v5_format_string_is_written() -> None:
    # Superseded by R0-b protocol §1.5: writers always emit v6; v5/v3 remain
    # readable legacy formats.
    assert ARTIFACT_FORMAT_V5 == "recurrent-human-cell-checkpoint-v6"


# --------------------------------------------------------------------------
# R0-T5 v3 legacy load
# --------------------------------------------------------------------------


def test_r0_t5_v3_legacy_single_tau_loads_bit_identically(tmp_path) -> None:
    """Failure: a v3-format single-tau fixture does not restore K=1 state
    bit-identically, or loads silently under multi-tau expectations."""
    from humanbrain.recurrent_artifacts import load_recurrent_checkpoint
    from humanbrain.tokenizer import BytePhraseTokenizer

    # A real fitted tokenizer keeps vocab/output-cell alignment intact.
    corpus = ["abcd", "abcd", "cdab"]
    tokenizer = BytePhraseTokenizer.fit(corpus, None)
    token_cells = tokenizer.vocab_size
    circuit = RecurrentCellCircuit(
        token_cells=token_cells,
        config=RecurrentCircuitConfig(
            reservoir_cells=64,
            input_fanout=8,
            recurrent_fanout=4,
            eligibility_tau_ms=80.0,
            random_seed=4,
        ),
    )
    rng = np.random.default_rng(4)
    sequences = [rng.integers(0, token_cells, 20) for _ in range(3)]
    for sequence in sequences:
        circuit.grow_sequence([int(token) for token in sequence], epochs=1)
    circuit.freeze_learning()
    context = [int(token) for token in sequences[0][:8]]

    from humanbrain.recurrent_language import RecurrentCellLanguageModel

    model = RecurrentCellLanguageModel(
        tokenizer, circuit, training_digest="0" * 64
    )
    v5_path = tmp_path / "born-v5.npz"
    model.save(v5_path)

    # Rewrite the saved artifact into its exact v3-era equivalent: same arrays,
    # legacy format string, legacy circuit_config (scalar tau, no tau list),
    # and no multi-trace guard keys.
    with np.load(v5_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    metadata = json.loads(str(arrays["metadata"]))
    # Writers emit v6 since R0-b §1.5.
    assert metadata["format"] == "recurrent-human-cell-checkpoint-v6"
    cell_metadata = dict(metadata)
    cell_metadata["format"] = "recurrent-human-cell-checkpoint-v3"
    cell_metadata.pop("eligibility_taus_ms", None)
    cell_metadata.pop("feature_cell_count", None)
    legacy_config = dict(cell_metadata["circuit_config"])
    legacy_config.pop("eligibility_taus_ms", None)
    assert legacy_config["eligibility_tau_ms"] == 80.0
    cell_metadata["circuit_config"] = legacy_config
    arrays["metadata"] = np.asarray(json.dumps(cell_metadata, sort_keys=True))
    v3_path = tmp_path / "legacy-v3.npz"
    with v3_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    loaded_model = load_recurrent_checkpoint(v3_path)
    loaded = loaded_model._circuit
    assert isinstance(loaded, RecurrentCellCircuit)
    assert loaded.trace_count == 1
    assert loaded.config.eligibility_taus_ms == (80.0,)
    # Eligibility traces are transient state and stay unpersisted (v3 and v5
    # alike), so byte-identical restore means: identical PERSISTED arrays and,
    # from an equally trace-free state, identical output currents and races.
    circuit.reset_sequence()
    loaded.reset_sequence()
    np.testing.assert_array_equal(
        loaded.output_current_pa(), circuit.output_current_pa()
    )
    race_before = circuit.predict_next(context)
    race_after = loaded.predict_next(context)
    assert race_after.winner == race_before.winner

    # The same payload must refuse to satisfy a multi-trace configuration:
    # mixed-era guards raise loudly instead of reshaping.
    tampered = dict(cell_metadata)
    tampered["eligibility_taus_ms"] = list(ELIGIBILITY_TAUS_MS)
    tampered["feature_cell_count"] = 8 * 64
    mixed_arrays = {
        "metadata": np.asarray(json.dumps(tampered, sort_keys=True)),
        **{k: v for k, v in arrays.items() if k != "metadata"},
    }
    mixed_path = tmp_path / "mixed-era.npz"
    with mixed_path.open("wb") as stream:
        np.savez_compressed(stream, **mixed_arrays)
    with np.load(mixed_path, allow_pickle=False) as payload:
        with pytest.raises(ValueError, match="eligibility_taus_ms"):
            RecurrentCellCircuit.decode_cell_synapse_state(
                json.loads(str(payload["metadata"])), payload
            )


# --------------------------------------------------------------------------
# R0-T6 CUDA parity (skipped when no cooperative CUDA device is present)
# --------------------------------------------------------------------------


def _cuda_dense_circuit(**overrides: object):
    kwargs = dict(
        reservoir_cells=384,
        active_reservoir_cells=320,
        input_fanout=12,
        recurrent_fanout=6,
        eligibility_tau_ms=None,  # placeholder replaced below
    )
    kwargs.pop("eligibility_tau_ms")
    kwargs.update(overrides)
    return _ridge_circuit(**kwargs)


def test_r0_t6_cuda_parity_every_row_and_reset(monkeypatch) -> None:
    """Failure: CUDA diverges from Rust on ANY trace row over 64 tokens; reset
    leaves stale device rows; mirror byte accounting ignores K."""
    from humanbrain.cuda import cuda_available

    if not cuda_available():
        pytest.skip("CUDA cooperative kernel is unavailable")

    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    expected = _cuda_dense_circuit()
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    actual = _cuda_dense_circuit()
    try:
        assert actual.simulation_backend == "cuda"
        assert expected.simulation_backend == "cpu"
        state_arrays = (
            actual._synaptic_current_pa,
            actual._eligibility_traces,
            actual._reservoir.state.voltage_relative_mv,
            actual._reservoir.state.after_spike_currents_pa,
            actual._reservoir.state.threshold_spike_mv,
            actual._reservoir.state.threshold_voltage_mv,
            actual._reservoir.state.refractory_steps,
        )
        assert actual._native_kernel.device_to_host_bytes_per_token == (
            sum(array.nbytes for array in state_arrays)
            + np.dtype(np.uint64).itemsize
        )
        tokens = np.random.default_rng(9031).integers(0, actual.token_cells, 64)
        for token in tokens:
            actual_step = actual.observe(int(token))
            expected_step = expected.observe(int(token))
            assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
            np.testing.assert_array_equal(
                actual._eligibility_traces, expected._eligibility_traces
            )
        # Reset zeroes ALL rows on device (download proves it).
        actual.reset_sequence()
        expected.reset_sequence()
        np.testing.assert_array_equal(
            actual._eligibility_traces, expected._eligibility_traces
        )
        assert not actual._eligibility_traces.any()
        # Repeated-token persistence per row after reset.
        for token in (7, 7, 7):
            actual_step = actual.observe(token)
            expected_step = expected.observe(token)
            assert actual_step.reservoir_spikes == expected_step.reservoir_spikes
            np.testing.assert_array_equal(
                actual._eligibility_traces, expected._eligibility_traces
            )
    finally:
        actual._native_kernel.close()


def test_r0_t6b_cuda_eligibility_mirror_bytes_scale_with_k(monkeypatch) -> None:
    """Failure: the sparse-readout eligibility-mode mirror is not K*8 B/cell."""
    from humanbrain.cuda import cuda_available

    if not cuda_available():
        pytest.skip("CUDA cooperative kernel is unavailable")

    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    expected = _sparse_circuit()
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    actual = _sparse_circuit()
    try:
        assert actual.simulation_backend == "cuda"
        kernel = actual._native_kernel
        k = actual.trace_count
        cells = actual.config.reservoir_cells
        assert kernel.device_to_host_bytes_per_token == 8 * k * cells + 8
        actual.observe(11)
        expected.observe(11)
        np.testing.assert_array_equal(
            actual._eligibility_traces, expected._eligibility_traces
        )
        assert actual.race().winner == expected.race().winner
    finally:
        actual._native_kernel.close()


# --------------------------------------------------------------------------
# R0-T7 kernel equivalence across backends
# --------------------------------------------------------------------------


def test_r0_t7_backend_equivalence_two_densities_and_frontier_any_row(
    monkeypatch,
) -> None:
    """Failure: serial/AVX/python-reference/CUDA disagree on any row, or the
    frontier drops a cell that is quiet in row 0 but active in row k."""
    from humanbrain.cuda import cuda_available

    config = dict(
        reservoir_cells=512,
        input_fanout=32,
        recurrent_fanout=6,
        input_current_pa=4_000.0,
        recurrent_current_pa=325.0,
        recurrent_topology="uniform_no_self",
        random_seed=47,
    )
    reference = _ridge_circuit(**config)
    serial = _ridge_circuit(**config)
    avx = _ridge_circuit(**config)

    for density in (0.01, 0.2):
        seeded = max(1, round(config["reservoir_cells"] * density))
        indices = (
            np.arange(seeded, dtype=np.intp) * config["reservoir_cells"] // seeded
        )
        states = []
        for circuit in (reference, serial, avx):
            circuit.reset_sequence()
            circuit._synaptic_current_pa[indices] = 25.0
            # Split the eligibility seed across rows so frontier activity can
            # only be right if it considers ANY nonzero row.
            matrix = _trace_matrix(circuit)
            matrix.fill(0.0)
            half = max(1, seeded // 2)
            matrix[0, indices[:half]] = 0.5
            matrix[min(3, circuit.trace_count - 1), indices[half:]] = 0.5
            states.append(
                tuple(
                    array.copy()
                    for array in (
                        circuit._synaptic_current_pa,
                        circuit._eligibility_traces,
                    )
                )
            )

        reference.reset_sequence()
        serial.reset_sequence()
        avx.reset_sequence()
        for circuit, state in zip(
            (reference, serial, avx), states, strict=True
        ):
            circuit._synaptic_current_pa[:] = state[0]
            circuit._eligibility_traces[:] = state[1]

        tokens = np.random.default_rng(8031).integers(0, reference.token_cells, 64)
        for token in tokens:
            python_spikes = _multirow_python_reference_observe(reference, int(token))
            serial_spikes = serial._native_kernel._observe_serial(int(token))
            avx_spikes = avx._native_kernel._observe_parallel(int(token))
            assert serial_spikes == python_spikes == avx_spikes
            np.testing.assert_array_equal(
                _trace_matrix(serial), _trace_matrix(reference)
            )
            np.testing.assert_array_equal(_trace_matrix(avx), _trace_matrix(serial))

    if not cuda_available():
        pytest.skip("CUDA cooperative kernel is unavailable")
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    cpu_circuit = _ridge_circuit(**config)
    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    gpu_circuit = _ridge_circuit(**config)
    try:
        cpu_circuit.reset_sequence()
        gpu_circuit.reset_sequence()
        tokens = np.random.default_rng(9031).integers(0, cpu_circuit.token_cells, 64)
        for token in tokens:
            cpu_spikes = cpu_circuit.observe(int(token)).reservoir_spikes
            gpu_spikes = gpu_circuit.observe(int(token)).reservoir_spikes
            assert cpu_spikes == gpu_spikes
            np.testing.assert_array_equal(
                _trace_matrix(cpu_circuit), _trace_matrix(gpu_circuit)
            )
    finally:
        gpu_circuit._native_kernel.close()


# --------------------------------------------------------------------------
# R0-T8 feature-vector shape guards
# --------------------------------------------------------------------------


def test_r0_t8_feature_shapes_and_strengthen_content(tmp_path) -> None:
    """Failure: ridge storage shapes ignore K; strengthen-time features are not
    the flat buffer with trailing bias; restore mismatch is unnamed."""
    circuit = _ridge_circuit()
    k = circuit.trace_count
    cells = circuit.config.reservoir_cells
    features = k * cells + 1
    circuit.observe(3)
    snapshot = circuit._feature_traces.copy()
    circuit.strengthen(4)
    assert circuit._growth_feature_count == features
    assert circuit._feature_batch.shape == (128, features)
    row = circuit._feature_batch[circuit._pending_growth_examples - 1]
    assert row[-1] == 1.0
    np.testing.assert_array_equal(row[:-1], snapshot)
    circuit.grow_sequence([0, 1, 2, 3, 4, 5], epochs=30)
    assert circuit._normal_matrix.shape == (features, features)
    assert circuit._target_moments.shape == (9, features)
    circuit.freeze_learning()
    assert circuit._readout_synapses_pa.shape == (9, k * cells)
    assert circuit._pacemaker_synapses_pa.shape == (9,)

    # Restore-time shape mismatch raises with the offending array named.
    _, path, _ = _grow_and_freeze_ridge(tmp_path, "shapes")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    arrays["normal_matrix"] = arrays["normal_matrix"][: features - 1]
    broken = tmp_path / "broken-shape.npz"
    with broken.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError, match="normal_matrix"):
        RecurrentCellCircuit.load_cell_synapse_checkpoint(broken)


# --------------------------------------------------------------------------
# R0-T9 sparse source-space validity
# --------------------------------------------------------------------------


def test_r0_t9_sparse_source_space_extends_to_feature_traces() -> None:
    """Failure: sources stay confined to row 0, candidate count ignores K, the
    apposition decay is not exp(-T/tau_(source//R)), or native current
    disagrees with the NumPy flat-buffer reference."""
    circuit = _sparse_circuit(reservoir_cells=64)
    k = circuit.trace_count
    r = circuit.config.reservoir_cells
    readout = circuit._sparse_readout
    assert readout is not None
    d = readout.dimensions
    assert d.reservoir_cells == k * r
    assert d.candidate_count == min(k * r, readout.CANDIDATE_FACTOR * d.fan_in)
    assert np.all(readout.potential_sources >= 0)
    assert np.all(readout.potential_sources < k * r)
    assert np.all(
        readout.potential_sources[:, 1:] > readout.potential_sources[:, :-1]
    )

    # Apposition decay per slot equals exp(-T / tau_(source // R)) with the
    # reservoir trace contribution nulled (zero traces): reset wipes traces,
    # then one observe satisfies the race precondition without adding spikes.
    circuit.reset_sequence()
    circuit.observe(5)
    readout.apposition_eligibility[0, 0] = 1.0
    readout.apposition_eligibility[1, 1] = 2.0
    apposition_before = readout.apposition_eligibility.copy()
    winner = circuit.race().winner
    target = (winner + 1) % circuit.token_cells
    readout.teach(
        target=target,
        winner=(target + 1) % circuit.token_cells,
        learning_rate_pa=1e-12,
        maximum_weight_pa=1_600.0,
        target_post_gate=0.0,
        winner_post_gate=0.0,
        activity_decay=circuit.config.readout_activity_decay,
    )
    per_source = np.exp(
        -circuit.config.token_interval_ms
        / np.asarray(ELIGIBILITY_TAUS_MS)[
            readout.potential_sources // r
        ]
    )
    np.testing.assert_allclose(
        readout.apposition_eligibility,
        apposition_before * per_source,
        rtol=0.0,
        atol=1e-15,
    )

    # Native current over the flat source space matches NumPy.
    circuit._eligibility_traces[:] = np.linspace(
        0.0, 1.0, k * r, dtype=np.float64
    ).reshape(circuit._eligibility_traces.shape)
    weights = np.random.default_rng(8).uniform(
        0.0, 100.0, readout.contact_weights_pa.shape
    )
    readout.contact_weights_pa[:] = weights
    base = 450.0
    native_current = readout.current(base)
    gathered = circuit._feature_traces[readout.contact_sources]
    reference_current = (
        base
        + circuit._pacemaker_synapses_pa
        + circuit._output_heterogeneity_pa
        + (weights * gathered).sum(axis=1)
    )
    np.testing.assert_allclose(native_current, reference_current, rtol=1e-12)


# --------------------------------------------------------------------------
# R0-T10 accounting invariance
# --------------------------------------------------------------------------


def test_r0_t10_accounting_invariance_and_trace0_identity() -> None:
    """Failure: multi-tau changes token accounting, presentation counters, or
    trace-0 statistics; cross-trace interaction leaks into row 0."""

    def run(config: RecurrentCircuitConfig, seed: int) -> tuple[dict, np.ndarray]:
        circuit = RecurrentCellCircuit(token_cells=9, config=config)
        rng = np.random.default_rng(seed)
        sequences = [
            tuple(int(token) for token in rng.integers(0, 9, 16)) for _ in range(3)
        ]
        presentation_log: list[tuple[int, int]] = []
        strengthens = 0
        for epoch in range(2):
            for sequence_index, sequence in enumerate(sequences):
                circuit.reset_sequence()
                for position in range(len(sequence) - 1):
                    circuit.observe(sequence[position])
                    circuit.strengthen(sequence[position + 1])
                    strengthens += 1
                presentation_log.append((epoch, sequence_index))
        account = {
            "observed_tokens": circuit.observed_tokens,
            "growth_examples": circuit._growth_examples,
            "local_updates": circuit._local_updates,
            "activity_density": circuit.reservoir_activity_density,
            "presentation_events": len(presentation_log),
            "strengthen_events": strengthens,
        }
        return account, _trace_matrix(circuit).copy()

    single_account, single_matrix = run(
        RecurrentCircuitConfig(
            reservoir_cells=64, random_seed=13, eligibility_tau_ms=80.0
        ),
        21,
    )
    multi_account, multi_matrix = run(
        RecurrentCircuitConfig(reservoir_cells=64, random_seed=13), 21
    )
    assert multi_account == single_account, "token accounting drifted between eras"
    # Trace 0 IS today's trace: identical inputs evolve row 0 identically.
    np.testing.assert_array_equal(multi_matrix[0], single_matrix[0])
    # reservoir_activity_density remains a trace-0 statistic.
    density_reference = float(np.count_nonzero(multi_matrix[0]) / 64)
    assert multi_account["activity_density"] == density_reference
