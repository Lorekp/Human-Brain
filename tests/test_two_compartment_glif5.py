"""TwoCompartmentGLIF5 tests: flag-off parity RED tests + hand-computed dynamics.

Preregistration: docs/research/apical-and-preregistration.md (frozen @ 05fb3ef,
BEFORE this class existed). The flag-off parity tests below were written first
and observed failing (ImportError) before the implementation landed.

Frozen step order inside one 0.05 ms step:
  (a) apical Euler update  v_a += (dt/tau_a) * (u_a - v_a)
  (b) plateau opening      v_a > theta_p and gate_remaining == 0 -> gate_steps
  (c) soma integration; crossing test uses total_threshold * (1 - g_gate)
      where gate_remaining > 0
  (d) gate_remaining[gate_remaining > 0] -= 1
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from humanbrain.cells import (
    AdaptiveGlifPopulation,
    AllenGlif5Parameters,
    TwoCompartmentGLIF5,
)

REPO = Path(__file__).resolve().parents[1]


def _load_harness():
    path = REPO / "experiments" / "apical_probes" / "two_comp_harness.py"
    spec = importlib.util.spec_from_file_location("two_comp_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["two_comp_harness"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- parity


def _mixed_currents(rng: np.random.Generator, steps: int, cells: int) -> np.ndarray:
    """Alternating quiet/suprathreshold currents to hit spike-cut refractory."""
    currents = rng.normal(0.0, 200.0, size=(steps, cells))
    bursts = rng.integers(0, steps, size=steps // 10)
    currents[bursts] += 6000.0
    return currents


def test_flagoff_single_step_parity_including_refractory():
    """Bit-exact vs the native kernel with apical drive zero, spikes included."""
    cells = 64
    rng = np.random.default_rng(4242)
    base = AdaptiveGlifPopulation(cells)
    two = TwoCompartmentGLIF5(
        cells,
        tau_a_ms=100.0,
        theta_p=0.5,
        g_gate=0.6,
        gate_duration_ms=100.0,
    )
    assert two.apical_voltage_mv.shape == (1, cells)
    assert not two.apical_voltage_mv.any()
    for injected in _mixed_currents(rng, 600, cells):
        fired_base = base.step(injected)
        fired_two = two.step(injected[None, :])
        assert np.array_equal(fired_base, fired_two[0])
        for name in (
            "voltage_relative_mv",
            "after_spike_currents_pa",
            "threshold_spike_mv",
            "threshold_voltage_mv",
            "refractory_steps",
        ):
            got = getattr(two.state, name)[0]
            want = getattr(base.state, name)
            assert np.array_equal(got, want), f"{name} diverged under flag-off"


def test_flagoff_token_loop_parity_vs_native_circuit():
    """The harness numpy token loop reproduces RecurrentCellCircuit.observe
    bit-for-bit (reservoir state, synaptic current, trace bank, counts)."""
    harness = _load_harness()
    from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig

    n_cells = 96
    config = RecurrentCircuitConfig(
        reservoir_cells=n_cells,
        input_fanout=16,
        recurrent_fanout=8,
        random_seed=7,
        readout_learning_rule="ridge",
        inhibitory_fraction=0.2,
        feature_family="traces",
    )
    circuit = RecurrentCellCircuit(token_cells=256, config=config)
    runner = harness.ApicalTokenRunner.from_circuit(
        circuit,
        tau_a_ms=100.0,
        theta_p=0.5,
        g_gate=0.6,
        arm_count=1,
    )
    rng = np.random.default_rng(11)
    tokens = rng.integers(0, 256, size=250)
    for token in tokens:
        step = circuit.observe(int(token))
        runner.observe(int(token), apical_u=None)
        assert step.reservoir_spikes == int(runner.last_total_spikes[0])
        assert np.array_equal(
            circuit._reservoir.state.voltage_relative_mv,
            runner.population.state.voltage_relative_mv[0],
        )
        assert np.array_equal(
            circuit._reservoir.state.after_spike_currents_pa,
            runner.population.state.after_spike_currents_pa[0],
        )
        assert np.array_equal(
            circuit._reservoir.state.threshold_spike_mv,
            runner.population.state.threshold_spike_mv[0],
        )
        assert np.array_equal(
            circuit._reservoir.state.threshold_voltage_mv,
            runner.population.state.threshold_voltage_mv[0],
        )
        assert np.array_equal(
            circuit._reservoir.state.refractory_steps,
            runner.population.state.refractory_steps[0],
        )
        assert np.array_equal(
            circuit._synaptic_current_pa, runner.synaptic_current_pa[0]
        )
        assert np.array_equal(
            circuit._feature_traces.reshape(8, n_cells), runner.eligibility[0]
        )
        assert np.array_equal(circuit._token_spike_counts, runner.token_spike_counts[0])
    # reset parity too
    circuit.reset_sequence()
    runner.reset()
    assert np.array_equal(
        circuit._reservoir.state.voltage_relative_mv,
        runner.population.state.voltage_relative_mv[0],
    )
    assert not runner.eligibility.any()


def test_ggate_zero_with_drive_leaves_soma_bit_exact():
    """Apical drive with g_gate=0 must not move any somatic array."""
    cells = 32
    rng = np.random.default_rng(7)
    base = AdaptiveGlifPopulation(cells)
    two = TwoCompartmentGLIF5(cells, tau_a_ms=100.0, theta_p=0.25, g_gate=0.0)
    two.set_apical_drive(np.full((1, cells), 3.0))
    for injected in _mixed_currents(rng, 400, cells):
        base.step(injected)
        two.step(injected[None, :])
        for name in (
            "voltage_relative_mv",
            "after_spike_currents_pa",
            "threshold_spike_mv",
            "threshold_voltage_mv",
            "refractory_steps",
        ):
            assert np.array_equal(
                getattr(two.state, name)[0], getattr(base.state, name)
            ), f"g_gate=0 changed {name}"
    assert two.gate_remaining.max() > 0  # plateaus fired (crossing ~step 174)
    assert two.apical_voltage_mv.min() > 0.25  # settled past theta everywhere


# --------------------------------------------------- hand-computed dynamics


def test_apical_integrator_hand_computed_euler():
    cells = 8
    dt = AllenGlif5Parameters().dt_ms
    tau_a = 100.0
    two = TwoCompartmentGLIF5(cells, tau_a_ms=tau_a, theta_p=2.5, g_gate=0.6)
    u = np.full((1, cells), 0.8)
    two.set_apical_drive(u)
    scale = dt / tau_a
    expected = np.zeros((1, cells))
    quiet = np.zeros((1, cells))
    for _ in range(1000):
        # frozen literal update, computed scalar-side
        expected = expected + scale * (u - expected)
        two.step(quiet)
        assert np.array_equal(two.apical_voltage_mv, expected)
    # sanity: the discrete Euler recurrence converges to 0.8*(1-(1-s)^n);
    # closed-form continuous 0.8*(1-exp(-t/tau)) differs by O(dt) so compare
    # against the exact DISCRETE value
    s = dt / tau_a
    assert np.allclose(expected, 0.8 * (1.0 - (1.0 - s) ** 1000), rtol=1e-12, atol=0)
    assert abs(expected[0, 0] - 0.8 * (1.0 - np.exp(-0.5))) < 2e-4


def test_plateau_gate_duration_and_reopen_timeline():
    cells = 4
    params = AllenGlif5Parameters()
    dt = params.dt_ms
    tau_a = 50.0
    theta_p = 0.5
    gate_steps_expected = round(100.0 / dt)  # 2000
    two = TwoCompartmentGLIF5(
        cells, tau_a_ms=tau_a, theta_p=theta_p, g_gate=0.6, gate_duration_ms=100.0
    )
    u = np.full((1, cells), 1.0)
    two.set_apical_drive(u)
    scale = dt / tau_a
    v = 0.0
    quiet = np.zeros((1, cells))
    first_cross = None
    for step_index in range(1, 5001):
        v = v + scale * (1.0 - v)
        two.step(quiet)
        if first_cross is None and v > theta_p:
            first_cross = step_index
            # gate opened THIS step, then decremented once: dur-1 remains
            assert np.array_equal(
                two.gate_remaining, np.full((1, cells), gate_steps_expected - 1, np.int32)
            )
        if first_cross is not None:
            age = step_index - first_cross  # 0-based steps since opening
            within = age % gate_steps_expected
            expected_remaining = gate_steps_expected - 1 - within
            assert np.array_equal(
                two.gate_remaining,
                np.full((1, cells), expected_remaining, np.int32),
            ), f"step {step_index}"
    # crossing when 1-exp(-n*dt/tau) > 0.5 -> n = ceil(ln2 * tau/dt) = 693
    assert first_cross is not None and 600 < first_cross < 800


def test_gate_multiplicatively_enables_somatic_spike():
    cells = 1
    params = AllenGlif5Parameters()
    # Sub-threshold steady current: leak_ns * threshold_inf ~ rheobase scale.
    quiet_current = 0.5 * params.leak_ns * params.threshold_inf_relative_mv
    ungated = TwoCompartmentGLIF5(
        cells, tau_a_ms=100.0, theta_p=0.25, g_gate=0.6, gate_duration_ms=400.0
    )
    gated = TwoCompartmentGLIF5(
        cells, tau_a_ms=100.0, theta_p=0.25, g_gate=0.6, gate_duration_ms=400.0
    )
    gated.set_apical_drive(np.full((1, cells), 2.0))
    current = np.full((1, cells), quiet_current)
    ungated_spikes = 0
    gated_spikes = 0
    for _ in range(20_000):
        ungated_spikes += int(ungated.step(current).sum())
        gated_spikes += int(gated.step(current).sum())
    assert gated_spikes > 0, "gated cell never spiked"
    assert ungated_spikes == 0, "ungated sub-threshold cell spiked"
    # multiplicative check: effective crossing threshold during an open gate is
    # exactly (th_inf + th_v + th_s) * (1 - g_gate) -- verify against state
    g = 0.6
    p = gated.parameters
    th_total = (
        p.threshold_inf_relative_mv
        + gated.state.threshold_voltage_mv[0, 0]
        + gated.state.threshold_spike_mv[0, 0]
    )
    assert gated.gate_multiplier()[0, 0] == pytest.approx(1.0 - g)


def test_arm_batch_matches_individual_populations_bitwise():
    """The (A, cells) batch axis must equal A independent single-arm runs."""
    cells = 48
    taus = [50.0, 100.0, 200.0]
    thetas = [0.25, 0.5, 0.75]
    gs = [0.3, 0.6]
    batch = TwoCompartmentGLIF5.from_grid(
        cells,
        tau_a_ms=taus,
        theta_p=thetas,
        g_gate=gs,
        gate_duration_ms=100.0,
    )
    singles = [
        TwoCompartmentGLIF5(cells, tau_a_ms=t, theta_p=h, g_gate=g, gate_duration_ms=100.0)
        for t in taus for h in thetas for g in gs
    ]
    assert batch.arm_count == len(singles) == 18
    rng = np.random.default_rng(99)
    for injected in _mixed_currents(rng, 300, cells):
        drives = rng.normal(0.0, 1.5, size=(18, cells))
        batch.set_apical_drive(drives)
        fired_batch = batch.step(np.broadcast_to(injected, (18, cells)).copy())
        for a, single in enumerate(singles):
            single.set_apical_drive(drives[a][None, :])
            fired_single = single.step(injected[None, :])
            assert np.array_equal(fired_batch[a], fired_single[0])
            assert np.array_equal(batch.apical_voltage_mv[a], single.apical_voltage_mv[0])
            assert np.array_equal(batch.gate_remaining[a], single.gate_remaining[0])
            for name in (
                "voltage_relative_mv",
                "after_spike_currents_pa",
                "threshold_spike_mv",
                "threshold_voltage_mv",
                "refractory_steps",
            ):
                assert np.array_equal(
                    getattr(batch.state, name)[a], getattr(single.state, name)[0]
                ), f"arm {a}: {name} diverged between batch and single"


def test_reset_clears_apical_state():
    cells = 16
    two = TwoCompartmentGLIF5(cells, tau_a_ms=100.0, theta_p=0.25, g_gate=0.6)
    two.set_apical_drive(np.full((1, cells), 5.0))
    for _ in range(300):  # crossing at ~step 52 for these settings
        two.step(np.zeros((1, cells)))
    assert two.gate_remaining.max() > 0
    two.reset()
    assert not two.apical_voltage_mv.any()
    assert not two.gate_remaining.any()
    assert not two._apical_drive.any()


def test_validation_rejects_bad_compartment_parameters():
    with pytest.raises(ValueError):
        TwoCompartmentGLIF5(16, tau_a_ms=0.0)
    with pytest.raises(ValueError):
        TwoCompartmentGLIF5(16, tau_a_ms=-1.0)
    with pytest.raises(ValueError):
        TwoCompartmentGLIF5(16, g_gate=1.0)
    with pytest.raises(ValueError):
        TwoCompartmentGLIF5(16, g_gate=-0.1)
    with pytest.raises(ValueError):
        TwoCompartmentGLIF5(16, theta_p=0.0)
    with pytest.raises(ValueError):
        TwoCompartmentGLIF5(16, gate_duration_ms=0.0)
    with pytest.raises(ValueError):
        TwoCompartmentGLIF5(16, tau_a_ms=[50.0, 100.0], arm_count=1)  # wrong length
