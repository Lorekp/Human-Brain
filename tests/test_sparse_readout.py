from __future__ import annotations

import math

import numpy as np
import pytest

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.readout import SparseCellReadout, SparseReadoutDimensions


def _uniform_decays(circuit, value: float):
    """Uniform per-source apposition decay over the circuit's feature space."""
    size = circuit.trace_count * circuit.config.reservoir_cells
    return np.full(size, float(value))


def _scalar_sparse_current(circuit: RecurrentCellCircuit) -> np.ndarray:
    readout = circuit._sparse_readout
    assert readout is not None
    result = np.empty(circuit.token_cells, dtype=np.float64)
    for output in range(circuit.token_cells):
        current = float(circuit.config.output_base_current_pa)
        current += float(circuit._pacemaker_synapses_pa[output])
        current += float(circuit._output_heterogeneity_pa[output])
        for slot in range(readout.dimensions.fan_in):
            source = int(readout.contact_sources[output, slot])
            current += float(readout.contact_weights_pa[output, slot]) * float(
                circuit._feature_traces[source]
            )
        result[output] = current
    return result


def _counter_uniform(seed: int, event: int, output: int, stream: int) -> float:
    mask = (1 << 64) - 1
    value = (
        (seed & mask)
        ^ ((event * 0x9E3779B97F4A7C15) & mask)
        ^ ((output * 0xBF58476D1CE4E5B9) & mask)
        ^ ((stream * 0x94D049BB133111EB) & mask)
    )
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return float(value >> 11) * (1.0 / float(1 << 53))


def _scalar_turnover(
    *,
    output_count: int,
    output: int,
    potential: np.ndarray,
    traces: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray,
    ages: np.ndarray,
    activity: float,
    structural_event: int,
    random_seed: int,
    minimum_age: int,
    turnover_probability: float,
    learning_rate_pa: float,
    maximum_weight_pa: float,
    homeostatic_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    expected_sources = sources.copy()
    expected_weights = weights.copy()
    expected_ages = ages.copy()
    scale = math.exp(homeostatic_rate * (1.0 / output_count - activity))
    for slot in range(expected_weights.size):
        if expected_weights[slot] > 0.0:
            expected_weights[slot] = min(
                maximum_weight_pa, expected_weights[slot] * scale
            )
    vacancy = next(
        (slot for slot, weight in enumerate(expected_weights) if weight == 0.0),
        None,
    )
    deleted = False
    if vacancy is None:
        for slot in range(expected_weights.size):
            if expected_ages[slot] < minimum_age:
                continue
            weakness = 1.0 - min(
                1.0, max(0.0, expected_weights[slot] / maximum_weight_pa)
            )
            hazard = turnover_probability * weakness
            if _counter_uniform(random_seed, structural_event, output, slot) < hazard:
                vacancy = slot
                expected_weights[slot] = 0.0
                expected_ages[slot] = 0
                deleted = True
                break
    if vacancy is None:
        return expected_sources, expected_weights, expected_ages, False, False

    available = [
        candidate
        for candidate, source in enumerate(potential)
        if source == expected_sources[vacancy] or source not in expected_sources
    ]
    positive_total = sum(max(0.0, float(traces[index])) for index in available)
    use_uniform = positive_total == 0.0
    total = float(len(available)) if use_uniform else positive_total
    draw = _counter_uniform(
        random_seed ^ 0xA0761D6478BD642F,
        structural_event,
        output,
        expected_weights.size,
    ) * total
    selected = None
    last_selectable = None
    for candidate in available:
        score = 1.0 if use_uniform else max(0.0, float(traces[candidate]))
        if score > 0.0:
            last_selectable = candidate
        if draw < score:
            selected = candidate
            break
        draw -= score
    assert last_selectable is not None
    selected = last_selectable if selected is None else selected
    new_source = int(potential[selected])
    new_weight = min(
        maximum_weight_pa,
        learning_rate_pa * max(1.0, float(traces[selected])),
    )
    kept = [slot for slot in range(expected_sources.size) if slot != vacancy]
    rows = [
        (
            int(expected_sources[slot]),
            float(expected_weights[slot]),
            int(expected_ages[slot]),
        )
        for slot in kept
    ]
    rows.append((new_source, new_weight, 0))
    rows.sort(key=lambda row: row[0])
    expected_sources[:] = [row[0] for row in rows]
    expected_weights[:] = [row[1] for row in rows]
    expected_ages[:] = [row[2] for row in rows]
    return expected_sources, expected_weights, expected_ages, True, deleted


def _literal_sparse_readout(reservoir_cells: int, fan_in: int) -> SparseCellReadout:
    output_cells = 2
    candidate_count = min(reservoir_cells, 4 * fan_in)
    potential = np.tile(
        np.arange(candidate_count, dtype=np.int32), (output_cells, 1)
    )
    contacts = np.tile(np.arange(fan_in, dtype=np.int32), (output_cells, 1))
    return SparseCellReadout(
        dimensions=SparseReadoutDimensions(
            output_cells=output_cells,
            reservoir_cells=reservoir_cells,
            fan_in=fan_in,
            candidate_count=candidate_count,
        ),
        reservoir_eligibility=np.zeros(reservoir_cells, dtype=np.float64),
        output_heterogeneity_pa=np.zeros(output_cells, dtype=np.float64),
        pacemaker_synapses_pa=np.zeros(output_cells, dtype=np.float64),
        potential_sources=potential,
        apposition_eligibility=np.zeros(
            (output_cells, candidate_count), dtype=np.float64
        ),
        contact_sources=contacts,
        contact_weights_pa=np.zeros((output_cells, fan_in), dtype=np.float64),
        contact_ages=np.zeros((output_cells, fan_in), dtype=np.uint32),
        output_activity_ema=np.zeros(output_cells, dtype=np.float64),
        target_presentations=np.zeros(output_cells, dtype=np.uint64),
    )


def test_sparse_readout_allocates_only_bounded_local_contact_state() -> None:
    output_cells = 7
    reservoir_cells = 32
    fan_in = 4
    circuit = RecurrentCellCircuit(
        token_cells=output_cells,
        config=RecurrentCircuitConfig(
            reservoir_cells=reservoir_cells,
            input_fanout=4,
            recurrent_fanout=2,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=fan_in,
        ),
    )

    readout = circuit._sparse_readout
    assert readout is not None
    candidate_count = 4 * fan_in
    assert readout.potential_sources.shape == (output_cells, candidate_count)
    assert readout.apposition_eligibility.shape == (
        output_cells,
        candidate_count,
    )
    assert readout.contact_sources.shape == (output_cells, fan_in)
    assert readout.contact_weights_pa.shape == (output_cells, fan_in)
    assert readout.contact_ages.shape == (output_cells, fan_in)
    assert circuit._readout_synapses_pa is None
    assert circuit.learned_synapse_count == 0

    for output in range(output_cells):
        potential = readout.potential_sources[output]
        contacts = readout.contact_sources[output]
        assert np.all(potential[:-1] < potential[1:])
        assert np.all(contacts[:-1] < contacts[1:])
        assert set(contacts).issubset(set(potential))
    assert not readout.apposition_eligibility.any()
    assert not readout.contact_weights_pa.any()
    assert not readout.contact_ages.any()


def test_sparse_output_current_sums_only_explicit_contacts_in_source_order() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            eligibility_tau_ms=80.0,
            recurrent_fanout=1,
            output_base_current_pa=400.0,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=2,
        ),
    )
    circuit._feature_traces[:] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    circuit._output_heterogeneity_pa[:] = (0.125, -0.25, 0.5)
    circuit._pacemaker_synapses_pa[:] = (5.0, 7.0, 11.0)
    readout = circuit._sparse_readout
    assert readout is not None
    readout.contact_sources[:] = ((1, 6), (0, 7), (2, 5))
    readout.contact_weights_pa[:] = ((2.0, 3.0), (5.0, 7.0), (11.0, 13.0))

    expected = np.empty(3, dtype=np.float64)
    for output in range(3):
        current = (
            circuit.config.output_base_current_pa
            + circuit._pacemaker_synapses_pa[output]
            + circuit._output_heterogeneity_pa[output]
        )
        for slot in range(2):
            source = readout.contact_sources[output, slot]
            current += (
                readout.contact_weights_pa[output, slot]
                * circuit._feature_traces[source]
            )
        expected[output] = current

    np.testing.assert_array_equal(circuit.output_current_pa(), expected)


def test_sparse_learning_uses_pre_post_events_and_preserves_dale_signs() -> None:
    learning_rate = 10.0
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            eligibility_tau_ms=80.0,
            recurrent_fanout=1,
            output_base_current_pa=450.0,
            output_synapse_gain_pa=1_600.0,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=8,
            local_learning_rate_pa=learning_rate,
        ),
    )
    circuit._feature_traces[:] = (3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    circuit._output_heterogeneity_pa.fill(0.0)
    readout = circuit._sparse_readout
    assert readout is not None
    readout.contact_weights_pa[0, :2] = (30.0, 40.0)
    readout.contact_weights_pa[2, :2] = (1.0, 2.0)
    circuit._observed_tokens = 1

    circuit.strengthen(2)

    expected_trace = circuit._feature_traces.copy()
    np.testing.assert_array_equal(readout.apposition_eligibility[2], expected_trace)
    np.testing.assert_array_equal(readout.apposition_eligibility[0], expected_trace)
    assert not readout.apposition_eligibility[1].any()
    np.testing.assert_array_equal(
        readout.contact_weights_pa[2],
        (31.0, 42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert not readout.contact_weights_pa[0].any()
    assert not readout.contact_weights_pa[1].any()
    assert np.all(readout.contact_weights_pa >= 0.0)
    np.testing.assert_array_equal(
        circuit._pacemaker_synapses_pa,
        (0.0, 0.0, learning_rate),
    )
    assert circuit.local_update_count == 1
    assert circuit._last_output_winner == 0
    assert circuit._last_teaching_current_pa > circuit.output_current_pa()[2]
    assert circuit._last_teaching_spike_time_ms > 0.0


def test_first_spike_arbitration_reports_discarded_simultaneous_candidates() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=5,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            recurrent_fanout=1,
            output_base_current_pa=450.0,
            eligibility_tau_ms=80.0,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=8,
        ),
    )
    circuit._output_heterogeneity_pa.fill(0.0)
    circuit._observed_tokens = 1

    result = circuit.race()

    assert result.winner == 0
    assert result.simultaneous_spike_candidates == 5
    assert result.discarded_simultaneous_candidates == 4


def test_structural_turnover_is_slow_seeded_local_and_capacity_bounded() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=2,
            recurrent_fanout=1,
            output_base_current_pa=450.0,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=2,
            local_learning_rate_pa=10.0,
            readout_structural_interval=2,
            readout_structural_grace=4,
            readout_turnover_probability=1.0,
        ),
    )
    target = 2
    readout = circuit._sparse_readout
    assert readout is not None
    potential = readout.potential_sources[target]
    initial_sources = readout.contact_sources[target].copy()
    candidate = next(source for source in potential if source not in set(initial_sources))
    circuit._feature_traces.fill(0.0)
    circuit._feature_traces[candidate] = 100.0
    circuit._output_heterogeneity_pa.fill(0.0)
    circuit._observed_tokens = 1

    circuit.strengthen(target)

    np.testing.assert_array_equal(readout.contact_sources[target], initial_sources)
    assert circuit.learned_synapse_count == 0
    assert circuit.structural_recruitment_count == 0
    assert circuit.structural_deletion_count == 0

    circuit.strengthen(target)

    final_sources = readout.contact_sources[target]
    assert circuit.structural_recruitment_count == 1
    assert circuit.structural_deletion_count == 0
    assert len(set(final_sources)) == circuit.config.readout_fan_in
    assert set(final_sources).issubset(set(potential))
    assert np.count_nonzero(final_sources != initial_sources) <= 2
    assert 0 < circuit.learned_synapse_count <= circuit.token_cells * 2
    assert np.all(readout.contact_weights_pa >= 0.0)


def test_sparse_ffi_rejects_nonfinite_or_out_of_range_plasticity_scalars() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            recurrent_fanout=1,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=2,
        ),
    )
    readout = circuit._sparse_readout
    assert readout is not None

    with pytest.raises(ValueError, match="base current"):
        readout.current(float("nan"))
    with pytest.raises(ValueError, match="trace decay"):
        readout.teach(
            target=1,
            winner=0,
            learning_rate_pa=10.0,
            maximum_weight_pa=1_600.0,
            trace_decays=np.full(readout.dimensions.reservoir_cells, float("nan")),
            target_post_gate=1.0,
            winner_post_gate=1.0,
            activity_decay=0.99,
        )
    with pytest.raises(ValueError, match="turnover probability"):
        readout.turnover(
            output=1,
            structural_event=1,
            random_seed=17,
            minimum_age=2,
            turnover_probability=-0.1,
            learning_rate_pa=10.0,
            maximum_weight_pa=1_600.0,
            homeostatic_rate=0.05,
        )


def test_turnover_respects_contact_grace_and_scales_homeostasis_exactly() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=2,
            recurrent_fanout=1,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=2,
        ),
    )
    readout = circuit._sparse_readout
    assert readout is not None
    output = 1
    readout.contact_weights_pa[output] = (100.0, 200.0)
    readout.contact_ages[output] = (2, 2)
    readout.output_activity_ema[output] = 0.75
    expected = readout.contact_weights_pa[output].copy()
    expected *= np.exp(0.2 * (0.25 - 0.75))

    recruited, deleted = readout.turnover(
        output=output,
        structural_event=1,
        random_seed=17,
        minimum_age=3,
        turnover_probability=1.0,
        learning_rate_pa=10.0,
        maximum_weight_pa=1_600.0,
        homeostatic_rate=0.2,
    )

    assert not recruited
    assert not deleted
    np.testing.assert_array_equal(readout.contact_weights_pa[output], expected)

    readout.contact_ages[output] = (3, 3)
    original_sources = readout.contact_sources[output].copy()
    recruited, deleted = readout.turnover(
        output=output,
        structural_event=2,
        random_seed=17,
        minimum_age=3,
        turnover_probability=1.0,
        learning_rate_pa=10.0,
        maximum_weight_pa=1_600.0,
        homeostatic_rate=0.0,
    )

    assert recruited and deleted
    assert np.count_nonzero(readout.contact_sources[output] != original_sources) <= 2
    assert np.count_nonzero(readout.contact_ages[output] == 0) == 1
    assert readout.structural_recruitments == 1
    assert readout.structural_deletions == 1


def test_native_turnover_matches_scalar_seeded_structural_cases() -> None:
    cases = (
        {
            "name": "weighted_tie",
            "reservoir_cells": 4,
            "weights": (0.0, 100.0),
            "ages": (0, 3),
            "traces": (5.0, 0.0, 5.0, 0.0),
            "event": 5,
            "seed": 17,
            "probability": 0.0,
            "maximum": 1_600.0,
            "flags": (True, False),
        },
        {
            "name": "weighted_tie_second_seed",
            "reservoir_cells": 4,
            "weights": (0.0, 100.0),
            "ages": (0, 3),
            "traces": (5.0, 0.0, 5.0, 0.0),
            "event": 5,
            "seed": 43,
            "probability": 0.0,
            "maximum": 1_600.0,
            "flags": (True, False),
        },
        {
            "name": "full_delete_replace",
            "reservoir_cells": 4,
            "weights": (1.0, 100.0),
            "ages": (3, 3),
            "traces": (0.0, 0.0, 10.0, 0.0),
            "event": 2,
            "seed": 17,
            "probability": 1.0,
            "maximum": 1.0e12,
            "flags": (True, True),
        },
        {
            "name": "vacancy_recruitment",
            "reservoir_cells": 4,
            "weights": (0.0, 100.0),
            "ages": (0, 3),
            "traces": (0.0, 0.0, 9.0, 0.0),
            "event": 7,
            "seed": 29,
            "probability": 0.0,
            "maximum": 1_600.0,
            "flags": (True, False),
        },
        {
            "name": "candidate_pool_equals_capacity_no_change",
            "reservoir_cells": 2,
            "weights": (100.0, 200.0),
            "ages": (3, 3),
            "traces": (5.0, 5.0),
            "event": 11,
            "seed": 43,
            "probability": 0.0,
            "maximum": 1_600.0,
            "flags": (False, False),
        },
        {
            "name": "candidate_pool_equals_capacity_reactivation",
            "reservoir_cells": 2,
            "weights": (0.0, 200.0),
            "ages": (0, 3),
            "traces": (9.0, 5.0),
            "event": 13,
            "seed": 59,
            "probability": 0.0,
            "maximum": 1_600.0,
            "flags": (True, False),
        },
    )
    selected_by_tie_seed: list[int] = []
    for case in cases:
        readout = _literal_sparse_readout(int(case["reservoir_cells"]), fan_in=2)
        readout.contact_weights_pa[0] = case["weights"]
        readout.contact_ages[0] = case["ages"]
        readout.apposition_eligibility[0] = case["traces"]
        original_sources = readout.contact_sources[0].copy()
        expected = _scalar_turnover(
            output_count=2,
            output=0,
            potential=readout.potential_sources[0],
            traces=readout.apposition_eligibility[0],
            sources=readout.contact_sources[0],
            weights=readout.contact_weights_pa[0],
            ages=readout.contact_ages[0],
            activity=0.0,
            structural_event=int(case["event"]),
            random_seed=int(case["seed"]),
            minimum_age=3,
            turnover_probability=float(case["probability"]),
            learning_rate_pa=10.0,
            maximum_weight_pa=float(case["maximum"]),
            homeostatic_rate=0.0,
        )

        result = readout.turnover(
            output=0,
            structural_event=int(case["event"]),
            random_seed=int(case["seed"]),
            minimum_age=3,
            turnover_probability=float(case["probability"]),
            learning_rate_pa=10.0,
            maximum_weight_pa=float(case["maximum"]),
            homeostatic_rate=0.0,
        )

        np.testing.assert_array_equal(readout.contact_sources[0], expected[0])
        np.testing.assert_array_equal(readout.contact_weights_pa[0], expected[1])
        np.testing.assert_array_equal(readout.contact_ages[0], expected[2])
        assert result == expected[3:] == case["flags"], case["name"]
        if str(case["name"]).startswith("weighted_tie"):
            selected_by_tie_seed.append(
                next(source for source in readout.contact_sources[0] if source not in (1,))
            )
        if case["name"] == "candidate_pool_equals_capacity_reactivation":
            np.testing.assert_array_equal(readout.contact_sources[0], original_sources)
    assert set(selected_by_tie_seed) == {0, 2}


def test_sparse_three_factor_temporal_and_output_locality_ablations() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            eligibility_tau_ms=80.0,
            recurrent_fanout=1,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=8,
        ),
    )
    readout = circuit._sparse_readout
    assert readout is not None
    readout.contact_weights_pa[:] = 100.0
    weights_before = readout.contact_weights_pa.copy()

    circuit._feature_traces[:] = np.arange(1.0, 9.0)
    readout.teach(
        target=1,
        winner=0,
        learning_rate_pa=10.0,
        maximum_weight_pa=1_600.0,
        trace_decays=_uniform_decays(circuit, 1.0),
        target_post_gate=0.0,
        winner_post_gate=0.0,
        activity_decay=0.99,
    )
    assert not readout.apposition_eligibility.any()
    np.testing.assert_array_equal(readout.contact_weights_pa, weights_before)

    circuit._feature_traces.fill(0.0)
    readout.teach(
        target=1,
        winner=0,
        learning_rate_pa=10.0,
        maximum_weight_pa=1_600.0,
        trace_decays=_uniform_decays(circuit, 1.0),
        target_post_gate=1.0,
        winner_post_gate=1.0,
        activity_decay=0.99,
    )
    assert not readout.apposition_eligibility.any()
    np.testing.assert_array_equal(readout.contact_weights_pa, weights_before)

    readout.apposition_eligibility[0] = 4.0
    readout.apposition_eligibility[1] = 5.0
    unrelated_trace = np.arange(8.0)
    readout.apposition_eligibility[2] = unrelated_trace
    unrelated_weights = readout.contact_weights_pa[2].copy()
    readout.teach(
        target=1,
        winner=0,
        learning_rate_pa=10.0,
        maximum_weight_pa=1_600.0,
        trace_decays=_uniform_decays(circuit, 0.0),
        target_post_gate=1.0,
        winner_post_gate=1.0,
        activity_decay=0.99,
    )
    assert not readout.apposition_eligibility.any()
    np.testing.assert_array_equal(readout.contact_weights_pa, weights_before)

    readout.apposition_eligibility[2] = unrelated_trace
    circuit._feature_traces[:] = np.arange(1.0, 9.0)
    readout.teach(
        target=1,
        winner=0,
        learning_rate_pa=1.0,
        maximum_weight_pa=1_600.0,
        trace_decays=_uniform_decays(circuit, 1.0),
        target_post_gate=1.0,
        winner_post_gate=1.0,
        activity_decay=0.99,
    )
    np.testing.assert_array_equal(readout.apposition_eligibility[2], unrelated_trace)
    np.testing.assert_array_equal(readout.contact_weights_pa[2], unrelated_weights)


def test_full_fan_in_matches_independent_scalar_current_and_update_oracle() -> None:
    learning_rate = 7.0
    trace_decay = np.exp(-5.0 / 80.0)
    activity_decay = 0.9
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            eligibility_tau_ms=80.0,
            recurrent_fanout=1,
            output_base_current_pa=450.0,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=8,
            local_learning_rate_pa=learning_rate,
            readout_activity_decay=activity_decay,
            readout_structural_interval=100,
        ),
    )
    readout = circuit._sparse_readout
    assert readout is not None
    exact_sources = np.arange(8, dtype=np.int32)
    for output in range(3):
        np.testing.assert_array_equal(readout.potential_sources[output], exact_sources)
        np.testing.assert_array_equal(readout.contact_sources[output], exact_sources)

    circuit._feature_traces[:] = (1.0, 0.5, 2.0, 0.25, 0.0, 0.75, 1.5, 0.125)
    circuit._output_heterogeneity_pa.fill(0.0)
    circuit._pacemaker_synapses_pa[:] = (120.0, 20.0, 0.0)
    readout.contact_weights_pa[:] = np.asarray(
        (
            (40.0, 30.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0),
            (5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.125),
            (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0),
        ),
        dtype=np.float64,
    )
    readout.apposition_eligibility[:] = np.arange(24.0).reshape(3, 8) / 16.0
    readout.contact_ages[:] = np.arange(24, dtype=np.uint32).reshape(3, 8)
    readout.output_activity_ema[:] = (0.2, 0.3, 0.4)
    np.testing.assert_array_equal(circuit.output_current_pa(), _scalar_sparse_current(circuit))
    circuit._observed_tokens = 1
    assert circuit.race().winner == 0

    target = 2
    winner = 0
    expected_traces = readout.apposition_eligibility.copy()
    expected_weights = readout.contact_weights_pa.copy()
    expected_ages = readout.contact_ages.copy()
    expected_activity = readout.output_activity_ema.copy()
    expected_pacemaker = circuit._pacemaker_synapses_pa.copy()
    for output in range(3):
        for candidate in range(8):
            expected_traces[output, candidate] *= trace_decay
    for candidate in range(8):
        expected_traces[target, candidate] += circuit._feature_traces[candidate]
        expected_traces[winner, candidate] += circuit._feature_traces[candidate]
    for output, modulation in ((target, 1.0), (winner, -1.0)):
        for slot in range(8):
            expected_weights[output, slot] = min(
                1_600.0,
                max(
                    0.0,
                    expected_weights[output, slot]
                    + learning_rate * modulation * expected_traces[output, slot],
                ),
            )
    expected_ages += np.uint32(1)
    expected_activity *= activity_decay
    expected_activity[target] += 1.0 - activity_decay
    expected_activity[winner] += 1.0 - activity_decay
    expected_pacemaker[target] += learning_rate
    expected_pacemaker[winner] -= learning_rate

    circuit.strengthen(target)

    np.testing.assert_array_equal(readout.apposition_eligibility, expected_traces)
    np.testing.assert_array_equal(readout.contact_weights_pa, expected_weights)
    np.testing.assert_array_equal(readout.contact_ages, expected_ages)
    np.testing.assert_array_equal(readout.output_activity_ema, expected_activity)
    np.testing.assert_array_equal(circuit._pacemaker_synapses_pa, expected_pacemaker)
    assert circuit.local_update_count == 1
    assert circuit.structural_recruitment_count == 0
    assert circuit.structural_deletion_count == 0
    np.testing.assert_array_equal(circuit.output_current_pa(), _scalar_sparse_current(circuit))

    circuit.reset_sequence()
    circuit.observe(1)
    continued_current = _scalar_sparse_current(circuit)
    np.testing.assert_array_equal(circuit.output_current_pa(), continued_current)
    assert circuit.race().winner == int(np.argmax(continued_current))


def test_one_wrong_example_cannot_bypass_structural_recruitment_bound() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=2,
            eligibility_tau_ms=80.0,
            recurrent_fanout=1,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=4,
            readout_structural_interval=1,
        ),
    )
    circuit._feature_traces[:] = np.arange(1.0, 17.0)
    circuit._output_heterogeneity_pa.fill(0.0)
    circuit._observed_tokens = 1
    before = circuit.learned_synapse_count

    circuit.strengthen(2)

    assert circuit.learned_synapse_count - before <= (
        SparseCellReadout.MAX_RECRUITMENTS_PER_EVENT
    )
    assert circuit.structural_recruitment_count <= (
        SparseCellReadout.MAX_RECRUITMENTS_PER_EVENT
    )
    assert circuit.local_update_count == 1
    assert circuit.occupied_output_synapses_per_cell == (
        circuit.learned_synapse_count / circuit.token_cells
    )
    assert circuit.reservoir_activity_density == 1.0


def test_correct_presentation_runs_scheduled_structural_maintenance() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            eligibility_tau_ms=80.0,
            recurrent_fanout=1,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=2,
            readout_structural_interval=1,
        ),
    )
    circuit._feature_traces[:] = np.arange(1.0, 9.0)
    circuit._output_heterogeneity_pa.fill(0.0)
    circuit._observed_tokens = 1

    circuit.strengthen(0)

    assert circuit._last_output_winner == 0
    assert circuit.local_update_count == 0
    assert circuit.structural_recruitment_count == 1
    assert circuit.learned_synapse_count == 1


def test_sparse_freeze_and_sequence_reset_preserve_only_long_term_synapse_state() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=3,
        config=RecurrentCircuitConfig(
            reservoir_cells=8,
            input_fanout=2,
            eligibility_tau_ms=80.0,
            recurrent_fanout=1,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=2,
            readout_structural_interval=1,
        ),
    )
    circuit._feature_traces[:] = np.arange(1.0, 9.0)
    circuit._output_heterogeneity_pa.fill(0.0)
    circuit._observed_tokens = 1
    circuit.strengthen(2)
    readout = circuit._sparse_readout
    assert readout is not None
    saved_arrays = {
        key: value.copy() for key, value in readout.checkpoint_arrays().items()
    }
    saved_counters = readout.checkpoint_counters()

    circuit.reset_sequence()

    assert circuit.observed_tokens == 0
    assert not circuit.eligibility.any()
    for key, value in saved_arrays.items():
        np.testing.assert_array_equal(readout.checkpoint_arrays()[key], value)
    assert readout.checkpoint_counters() == saved_counters

    circuit.freeze_learning()
    circuit.observe(0)
    with pytest.raises(RuntimeError, match="frozen"):
        circuit.strengthen(1)


def test_over_100k_sparse_readout_has_reachable_100x_memory_gate() -> None:
    output_cells = 256
    reservoir_cells = 200_000
    fan_in = 64
    eligibility = np.zeros(reservoir_cells, dtype=np.float64)
    readout = SparseCellReadout.create(
        output_cells=output_cells,
        reservoir_cells=reservoir_cells,
        fan_in=fan_in,
        rng=np.random.default_rng(17),
        reservoir_eligibility=eligibility,
        output_heterogeneity_pa=np.zeros(output_cells, dtype=np.float64),
        pacemaker_synapses_pa=np.zeros(output_cells, dtype=np.float64),
    )

    dense_classifier_bytes = output_cells * reservoir_cells * np.dtype(np.float64).itemsize
    assert readout.dimensions.candidate_count == 4 * fan_in
    assert dense_classifier_bytes / readout.storage_bytes >= 100.0
    assert readout.potential_sources.max() < reservoir_cells
    assert readout.active_synapse_count == 0
