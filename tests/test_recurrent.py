import numpy as np
import pytest
from humanbrain.native import NativeThreeFactorLearner
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


@pytest.mark.parametrize("satisfied", [False, True])
def test_projected_three_factor_empty_basis_matches_original_exactly(satisfied) -> None:
    trace = np.array([3.0, -4.0, 0.5])
    heterogeneity = np.array([0.1, -0.2, 0.3])
    weights = np.zeros((3, 3))
    if satisfied:
        weights[1] = trace
    bias = np.array([0.3, -0.1, 0.2])
    projected_weights, projected_bias = weights.copy(), bias.copy()
    original = NativeThreeFactorLearner(
        eligibility=trace,
        output_heterogeneity_pa=heterogeneity,
        readout_synapses_pa=weights,
        pacemaker_synapses_pa=bias,
    )
    projected = NativeThreeFactorLearner(
        eligibility=trace,
        output_heterogeneity_pa=heterogeneity,
        readout_synapses_pa=projected_weights,
        pacemaker_synapses_pa=projected_bias,
    )
    scratch = np.full(3, np.nan)
    assert projected.update_projected(1, 1.0, 1.0, np.empty((0, 3)), scratch, 1e-13) == (
        original.update(1, 1.0, 1.0)
    )
    np.testing.assert_array_equal(scratch, trace)
    np.testing.assert_array_equal(projected_weights, weights)
    np.testing.assert_array_equal(projected_bias, bias)


@pytest.mark.parametrize("satisfied", [False, True])
def test_projected_three_factor_uses_original_gate_and_normalizer(satisfied) -> None:
    trace = np.array([3.0, 4.0])
    basis = np.array([[1.0, 0.0]])
    # The original and projected gates disagree in both directions.
    weights = np.array([[1.0, 0.0] if satisfied else [-1.0, 1.0], [0.0, 0.0]])
    before = weights.copy()
    bias = np.zeros(2)
    scratch = np.full(2, np.nan)
    learner = NativeThreeFactorLearner(
        eligibility=trace,
        output_heterogeneity_pa=np.zeros(2),
        readout_synapses_pa=weights,
        pacemaker_synapses_pa=bias,
    )
    result = learner.update_projected(0, 1.0, 2.0, basis, scratch, 1e-13)
    np.testing.assert_array_equal(scratch, [0.0, 4.0])
    np.testing.assert_array_equal(weights[:, 0], before[:, 0])
    if satisfied:
        assert result is None
        np.testing.assert_array_equal(weights, before)
        np.testing.assert_array_equal(bias, [0.0, 0.0])
    else:
        assert result == 1
        step = 1.0 / np.sqrt(26.0)
        np.testing.assert_allclose(weights - before, [[0.0, 4 * step], [0.0, -4 * step]])
        np.testing.assert_array_equal(bias, [step, -step])


@pytest.mark.parametrize("small_residual", [0.0, 1e-15])
def test_projected_three_factor_exhaustion_preserves_synapses_and_bias(small_residual) -> None:
    direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    basis = direction.reshape(1, 3)
    trace = direction * 3.0 + np.array([0.0, 0.0, small_residual])
    weights, bias = np.zeros((2, 3)), np.zeros(2)
    scratch = np.full(3, np.nan)
    learner = NativeThreeFactorLearner(
        eligibility=trace,
        output_heterogeneity_pa=np.zeros(2),
        readout_synapses_pa=weights,
        pacemaker_synapses_pa=bias,
    )
    assert learner.update_projected(0, 1.0, 1.0, basis, scratch, 1e-13) is None
    np.testing.assert_array_equal(scratch, np.zeros(3))
    assert not weights.any()
    assert not bias.any()
    # The old path still learns this trace: the opt-in no-op did not replace it.
    assert learner.update(0, 1.0, 1.0) == 1
    assert weights.any() and bias.any()


def test_projected_three_factor_rejects_scratch_alias_before_mutation() -> None:
    trace = np.array([3.0, 4.0])
    weights, bias = np.zeros((2, 2)), np.zeros(2)
    learner = NativeThreeFactorLearner(
        eligibility=trace,
        output_heterogeneity_pa=np.zeros(2),
        readout_synapses_pa=weights,
        pacemaker_synapses_pa=bias,
    )
    with pytest.raises(ValueError, match="must not alias"):
        learner.update_projected(0, 1.0, 1.0, np.empty((0, 2)), trace, 1e-13)
    np.testing.assert_array_equal(trace, [3.0, 4.0])
    assert not weights.any() and not bias.any()


def test_arow_native_matches_reference_across_correlated_contrasts() -> None:
    trace = np.zeros(4)
    weights, bias = np.zeros((2, 4)), np.array([7.0, -3.0])
    covariance, scratch = np.eye(4), np.zeros(4)
    expected_weights, expected_covariance = weights.copy(), covariance.copy()
    learner = NativeThreeFactorLearner(
        eligibility=trace, output_heterogeneity_pa=np.array([9.0, -5.0]),
        readout_synapses_pa=weights, pacemaker_synapses_pa=bias,
    )
    for delta in ([1., 2., 0., -1.], [2., -1., 1., 0.], [-1., -2., 1., 1.]) * 4:
        trace[:] = delta
        margin = float((expected_weights[0] - expected_weights[1]) @ trace)
        p = expected_covariance @ trace
        beta = 1.0 / (trace @ p + 1.0)
        alpha = (1.0 - margin) * beta
        if margin < 1.0:
            expected_weights += np.array([0.5 * alpha * p, -0.5 * alpha * p])
            expected_covariance -= beta * np.outer(p, p)
        result = learner.update_arow(covariance, scratch)
        assert result["gap_before"] == pytest.approx(margin, abs=1e-13)
        assert result["native_update_applied"] == (margin < 1.0)
        if margin < 1.0:
            np.testing.assert_allclose(scratch, p, atol=1e-13)
            assert result["alpha"] == pytest.approx(alpha, abs=1e-13)
            assert result["directional_variance"] == pytest.approx(trace @ p, abs=1e-13)
        np.testing.assert_allclose(weights, expected_weights, atol=1e-13)
        np.testing.assert_allclose(covariance, expected_covariance, atol=1e-13)
        np.testing.assert_array_equal(covariance, covariance.T)
        np.testing.assert_array_equal(bias, [7.0, -3.0])
    np.linalg.cholesky(covariance)
    assert not np.array_equal(covariance, np.eye(4))


@pytest.mark.parametrize("trace, go", [([1., 0.], [1., 0.]), ([0., 0.], [0., 0.])])
def test_arow_native_gate_and_zero_contrast_leave_covariance_unchanged(trace, go) -> None:
    weights = np.array([go, [0., 0.]])
    covariance, scratch, bias = np.eye(2), np.ones(2), np.zeros(2)
    before = weights.copy()
    learner = NativeThreeFactorLearner(
        eligibility=np.array(trace), output_heterogeneity_pa=np.zeros(2),
        readout_synapses_pa=weights, pacemaker_synapses_pa=bias,
    )
    result = learner.update_arow(covariance, scratch)
    assert not result["native_update_applied"]
    assert result["directional_variance"] is result["beta"] is None
    assert result["alpha"] == 0
    np.testing.assert_array_equal(weights, before)
    np.testing.assert_array_equal(covariance, np.eye(2))
    np.testing.assert_array_equal(scratch, 0)
    np.testing.assert_array_equal(bias, 0)


@pytest.mark.parametrize("bad", [
    [[-1., 0.], [0., 1.]], [[0., 0.], [0., 0.]],
    [[1., 0.2], [0., 1.]], [[float("nan"), 0.], [0., 1.]],
])
def test_arow_native_invalid_covariance_rejects_before_mutation(bad) -> None:
    covariance = np.array(bad)
    before = covariance.copy()
    weights, bias = np.zeros((2, 2)), np.zeros(2)
    learner = NativeThreeFactorLearner(
        eligibility=np.array([1., 0.]), output_heterogeneity_pa=np.zeros(2),
        readout_synapses_pa=weights, pacemaker_synapses_pa=bias,
    )
    with pytest.raises(RuntimeError, match="invalid covariance"):
        learner.update_arow(covariance, np.zeros(2))
    np.testing.assert_array_equal(covariance, before)
    np.testing.assert_array_equal(weights, 0)
    np.testing.assert_array_equal(bias, 0)


def test_arow_native_rejects_aliased_workspace() -> None:
    weights, bias, trace = np.zeros((2, 2)), np.zeros(2), np.array([1., 2.])
    learner = NativeThreeFactorLearner(
        eligibility=trace, output_heterogeneity_pa=np.zeros(2),
        readout_synapses_pa=weights, pacemaker_synapses_pa=bias,
    )
    with pytest.raises(ValueError, match="must not alias"):
        learner.update_arow(np.eye(2), trace)
    with pytest.raises(ValueError, match="must not alias"):
        learner.update_arow(weights, np.zeros(2))
    np.testing.assert_array_equal(weights, 0)


def test_arow_native_rejects_readonly_or_input_aliased_weights() -> None:
    for alias in ("eligibility", "bias", "heterogeneity", "readonly"):
        weights = np.zeros((2, 2))
        trace, bias, heterogeneity = np.ones(2), np.zeros(2), np.zeros(2)
        if alias == "eligibility":
            trace = weights[0]
        elif alias == "bias":
            bias = weights[0]
        elif alias == "heterogeneity":
            heterogeneity = weights[0]
        else:
            weights.setflags(write=False)
        learner = NativeThreeFactorLearner(
            eligibility=trace, output_heterogeneity_pa=heterogeneity,
            readout_synapses_pa=weights, pacemaker_synapses_pa=bias,
        )
        with pytest.raises(ValueError, match=r"writable|must not alias"):
            learner.update_arow(np.eye(2), np.zeros(2))
        np.testing.assert_array_equal(weights, 0)


@pytest.mark.parametrize("bad", [
    [[1., 0.2], [0., 1.]], [[float("nan"), 0.], [0., 1.]],
])
def test_arow_native_noop_still_rejects_invalid_covariance(bad) -> None:
    weights, bias = np.array([[2., 0.], [0., 0.]]), np.zeros(2)
    learner = NativeThreeFactorLearner(
        eligibility=np.array([1., 0.]), output_heterogeneity_pa=np.zeros(2),
        readout_synapses_pa=weights, pacemaker_synapses_pa=bias,
    )
    with pytest.raises(RuntimeError, match="invalid covariance"):
        learner.update_arow(np.array(bad), np.zeros(2))
    np.testing.assert_array_equal(weights, [[2., 0.], [0., 0.]])


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
