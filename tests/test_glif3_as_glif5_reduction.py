"""Red + green tests: GLIF3-as-GLIF5 exact expressibility and the V1 mixture.

Contract for docs/research/glif3-as-glif5-expressibility.md (preregistered
verdict: REDUCIBLE-WITH-HARMLESS-OVERHEAD).

Tests 1-6 are ACTIVE guards on the verified reduction: bit-exact degenerate
LIF equivalence, ulp-budgeted pinned-specimen equivalence under the
delta-shifted amplitude recipe, inert-threshold bit-zero invariant, the
necessity of the shift, and the one-step wrapper timestamp offset.

Tests 7-10 are RED until V1's MixedSpecimenPopulation lands; they skip with a
"RED until ..." reason and auto-enforce on landing. Names there are proposals;
implementers may rename with a one-line patch, keeping the assertions.
"""
from __future__ import annotations

import numpy as np
import pytest

from humanbrain.cells import (
    AllenGlif3Parameters,
    AllenGlif5Parameters,
    AdaptiveGlifPopulation,
    FittedGlif5Constants,
    simulate_glif3_current,
    simulate_glif5_current,
)
from humanbrain.human_cell_identity import HUMAN_GLIF3_IDENTITY, HUMAN_GLIF5_IDENTITY

P3 = AllenGlif3Parameters()

VOLTAGE_ULP_BUDGET = 16
ASC_ULP_BUDGET = 8


def _reduced_glif5(
    shift_amplitudes: bool,
    zero_asc: bool = False,
    collide_b_with_leak: bool = False,
) -> AllenGlif5Parameters:
    """The reduction recipe from glif3-as-glif5-expressibility.md §2.1."""
    taus = np.asarray(P3.after_spike_taus_ms, dtype=np.float64)
    decay = np.exp(-P3.dt_ms / taus)
    if zero_asc:
        amplitudes: tuple[float, float] = (0.0, 0.0)
    elif shift_amplitudes:
        amplitudes = tuple(float(a) for a in np.asarray(P3.after_spike_amplitudes_pa) * decay)
    else:
        amplitudes = P3.after_spike_amplitudes_pa
    leak_per_s = (1_000.0 / P3.input_resistance_mohm) / P3.capacitance_pf * 1_000.0
    b_voltage = leak_per_s if collide_b_with_leak else 19.773561643976418
    return AllenGlif5Parameters(
        rest_mv=P3.rest_mv,
        dt_ms=P3.dt_ms,
        capacitance_pf=P3.capacitance_pf,
        input_resistance_mohm=P3.input_resistance_mohm,
        threshold_inf_relative_mv=P3.threshold_relative_mv,
        spike_cut_steps=P3.spike_cut_steps,
        after_spike_amplitudes_pa=amplitudes,
        after_spike_taus_ms=P3.after_spike_taus_ms,
        voltage_reset_scale=0.0,
        voltage_reset_offset_mv=0.0,
        threshold_spike_increment_mv=0.0,
        threshold_voltage_coupling_per_s=0.0,
        threshold_voltage_decay_per_s=b_voltage,
    )


def _ulp_distance(x: float, y: float, cap: int = 65) -> int:
    if x == y:
        return 0
    lo, hi = sorted((x, y))
    scale = float(np.spacing(max(abs(lo), abs(hi))))
    if scale == 0.0 or (hi - lo) / scale > cap:
        return cap + 1
    cursor, count = lo, 0
    while cursor != hi and count <= cap:
        cursor = float(np.nextafter(cursor, np.inf))
        count += 1
    return min(count, cap + 1)


def _run_pair(stimulus: np.ndarray, params: AllenGlif5Parameters, zero_asc_reference: bool):
    """Replay cells.py's GLIF3 loop vs the native kernel; return per-step state.

    Comparison objects follow doc §2.4/§4: committed voltage after each step,
    and GLIF3's post-decay ASC working value (which feeds its voltage update)
    against the kernel's raw stored ASC entering the same step.
    """
    reference_amps = (0.0, 0.0) if zero_asc_reference else P3.after_spike_amplitudes_pa
    p3 = AllenGlif3Parameters(after_spike_amplitudes_pa=reference_amps)
    relative_v = 0.0
    after = np.zeros(len(p3.after_spike_taus_ms), dtype=np.float64)
    amps = np.asarray(p3.after_spike_amplitudes_pa, dtype=np.float64)
    taus = np.asarray(p3.after_spike_taus_ms, dtype=np.float64)
    asc_decay = np.exp(-p3.dt_ms / taus)
    cut_decay = np.exp(-p3.spike_cut_ms / taus)
    refractory = 0

    pop = AdaptiveGlifPopulation(1, params)
    spikes3: list[int] = []
    spikes5: list[int] = []
    voltage_ulps: list[int] = []
    asc_ulps: list[int] = []

    for step_index, injected in enumerate(stimulus):
        injected_pa = float(injected)

        # --- GLIF3 side: cells.py:443-461 -------------------------------
        working: tuple[float, float] | None = None
        if refractory > 0:
            refractory -= 1
        else:
            after *= asc_decay
            working = (float(after[0]), float(after[1]))
            relative_v += (
                p3.dt_ms / p3.capacitance_pf
                * (injected_pa + float(after.sum()) - p3.leak_ns * relative_v)
            )
            if relative_v > p3.threshold_relative_mv:
                spikes3.append(step_index)
                relative_v = 0.0
                after = amps + after * cut_decay
                refractory = p3.spike_cut_steps

        # --- kernel side: snapshot entering ASC, then advance -------------
        kernel_integrating = int(pop.state.refractory_steps[0]) == 0
        asc_kernel = (
            (
                float(pop.state.after_spike_currents_pa[0][0]),
                float(pop.state.after_spike_currents_pa[0][1]),
            )
            if kernel_integrating
            else None
        )
        fired = bool(pop.step(injected_pa)[0])
        if fired:
            spikes5.append(step_index)

        voltage_ulps.append(_ulp_distance(relative_v, float(pop.state.voltage_relative_mv[0])))
        if working is not None and asc_kernel is not None:
            asc_ulps.extend(
                _ulp_distance(a3, a5) for a3, a5 in zip(working, asc_kernel)
            )
    return spikes3, spikes5, voltage_ulps, asc_ulps


# --- active guards -----------------------------------------------------------


def test_reduced_parameter_set_passes_validation_with_degenerate_constants() -> None:
    params = _reduced_glif5(shift_amplitudes=True)
    assert params.voltage_reset_scale == 0.0
    assert params.voltage_reset_offset_mv == 0.0
    assert params.threshold_spike_increment_mv == 0.0
    assert params.threshold_voltage_coupling_per_s == 0.0
    constants = FittedGlif5Constants.from_parameters(params)
    # phi may be a signed zero but must be finite; factors collapse to zero
    assert np.isfinite(constants.threshold_phi)
    assert constants.threshold_phi == 0.0
    assert constants.threshold_voltage_factor == 0.0
    assert constants.threshold_beta_factor == 0.0


def test_inert_threshold_state_stays_bit_zero_through_spikes() -> None:
    population = AdaptiveGlifPopulation(4, _reduced_glif5(shift_amplitudes=True))
    current = np.array([600.0, 450.0, -80.0, 0.0])
    spiked = np.zeros(4, dtype=bool)
    for _ in range(1500):
        spiked |= population.step(current)
    assert spiked.tolist() == [True, True, False, False]
    assert (population.state.threshold_voltage_mv.view(np.uint64) == 0).all()
    assert (population.state.threshold_spike_mv.view(np.uint64) == 0).all()


def test_degenerate_lif_reduction_is_bit_exact_end_to_end() -> None:
    rng = np.random.default_rng(20260823)
    stimulus = rng.uniform(-40.0, 900.0, size=2500).round(6)
    spikes3, spikes5, voltage_ulps, asc_ulps = _run_pair(
        stimulus, _reduced_glif5(shift_amplitudes=True, zero_asc=True), zero_asc_reference=True
    )
    assert spikes3 == spikes5
    assert max(voltage_ulps) == 0
    assert max(asc_ulps) == 0


def test_shifted_amplitude_reduction_matches_pinned_glif3_within_ulp_budgets() -> None:
    rng = np.random.default_rng(20260823)
    stimulus = rng.uniform(-40.0, 900.0, size=2500).round(6)
    spikes3, spikes5, voltage_ulps, asc_ulps = _run_pair(
        stimulus, _reduced_glif5(shift_amplitudes=True), zero_asc_reference=False
    )
    assert len(spikes3) >= 3  # regime must exercise repeated spiking + refractory
    assert spikes3 == spikes5
    assert max(voltage_ulps) <= VOLTAGE_ULP_BUDGET
    assert max(asc_ulps) <= ASC_ULP_BUDGET


def test_unshifted_naive_amplitudes_diverge_macroscopically() -> None:
    rng = np.random.default_rng(20260823)
    stimulus = rng.uniform(-40.0, 900.0, size=2500).round(6)
    _, _, voltage_ulps, _ = _run_pair(
        stimulus, _reduced_glif5(shift_amplitudes=False), zero_asc_reference=False
    )
    # The naive clone must NOT pass the ulp budget: guard against "simplifying"
    # the recipe by dropping the delta-shift (measured ~5.6 mV divergence).
    assert max(voltage_ulps) > VOLTAGE_ULP_BUDGET


def test_current_clamp_wrappers_disagree_by_exactly_one_step() -> None:
    lif3 = AllenGlif3Parameters(after_spike_amplitudes_pa=(0.0, 0.0))
    trace3 = simulate_glif3_current(lif3, 600.0, duration_ms=40.0)
    trace5 = simulate_glif5_current(
        _reduced_glif5(shift_amplitudes=True, zero_asc=True), 600.0, duration_ms=40.0
    )
    assert trace3.spike_times_ms, "reference LIF must spike under 600 pA"
    offsets = [
        (t3 - t5) / P3.dt_ms
        for t3, t5 in zip(trace3.spike_times_ms, trace5.spike_times_ms)
    ]
    assert len(offsets) == len(trace3.spike_times_ms) == len(trace5.spike_times_ms)
    # labels differ by exactly one STEP INDEX; fp timestamp arithmetic makes
    # the ratio 0.999.../1.001... at large indices, so compare with tolerance
    assert all(abs(offset - 1.0) < 1e-6 for offset in offsets)


# --- red until V1's mixture constructor lands --------------------------------


def _mixture_api_present() -> bool:
    try:
        from humanbrain import cells as _cells  # noqa: PLC0415
    except ImportError:  # pragma: no cover - import always succeeds in-repo
        return False
    return hasattr(_cells, "MixedSpecimenPopulation")


_RED_REASON = (
    "RED until MixedSpecimenPopulation lands (V1); contract at "
    "docs/research/glif3-as-glif5-expressibility.md §6"
)


@pytest.mark.skipif(not _mixture_api_present(), reason=_RED_REASON)
def test_mixed_specimen_population_exists_and_takes_deterministic_assignment() -> None:
    from humanbrain.cells import MixedSpecimenPopulation

    identities = [HUMAN_GLIF3_IDENTITY, HUMAN_GLIF5_IDENTITY]
    assignment = [identities[0]] * 4 + [identities[1]] * 4
    pop = MixedSpecimenPopulation(cell_count=8, specimen_assignment=assignment, seed=11)
    assert pop.cell_count == 8
    rebuilt = MixedSpecimenPopulation(cell_count=8, specimen_assignment=assignment, seed=11)
    assert rebuilt.specimen_labels() == pop.specimen_labels()


@pytest.mark.skipif(not _mixture_api_present(), reason=_RED_REASON)
def test_mixture_verifies_both_pinned_config_hashes_per_specimen() -> None:
    from humanbrain.cells import MixedSpecimenPopulation

    assignment = [HUMAN_GLIF3_IDENTITY] * 2 + [HUMAN_GLIF5_IDENTITY] * 2
    pop = MixedSpecimenPopulation(cell_count=4, specimen_assignment=assignment, seed=3)
    manifest = pop.composition_manifest()
    hashes = {record.model_id: record.config_sha256 for record in manifest.specimen_records()}
    assert hashes[HUMAN_GLIF3_IDENTITY.model_id] == HUMAN_GLIF3_IDENTITY.config_sha256
    assert hashes[HUMAN_GLIF5_IDENTITY.model_id] == HUMAN_GLIF5_IDENTITY.config_sha256
    # raw config bytes re-hashed against the pinned digests, not trusted blindly
    assert pop.verify_provenance() is True


@pytest.mark.skipif(not _mixture_api_present(), reason=_RED_REASON)
def test_mixture_rejects_glif3_half_attributed_to_the_glif5_identity() -> None:
    """A reduced GLIF3 subpopulation must not silently claim fit 523000114.

    AllenGlif5Parameters.__post_init__ forces HUMAN_GLIF5_IDENTITY onto every
    instance today; the mixture layer must carry an explicit derived identity
    for specimen 488392806's dynamics instead (doc §5).
    """
    from humanbrain.cells import MixedSpecimenPopulation

    reduced = _reduced_glif5(shift_amplitudes=True)
    assert reduced.model_id != HUMAN_GLIF3_IDENTITY.model_id  # dataclass pins GLIF5 id
    with pytest.raises(ValueError):
        # provenance-correct construction requires a derived identity; passing
        # bare reduced GLIF5-stamped parameters for GLIF3 cells is rejected
        MixedSpecimenPopulation(
            cell_count=2,
            specimen_assignment=[HUMAN_GLIF3_IDENTITY, HUMAN_GLIF5_IDENTITY],
            seed=5,
            subpopulation_parameters={HUMAN_GLIF3_IDENTITY.model_id: reduced},
        )
    # the phi NaN hazard is guarded at construction time (doc §2.3)
    colliding = _reduced_glif5(shift_amplitudes=True, collide_b_with_leak=True)
    with pytest.raises(ValueError):
        MixedSpecimenPopulation(
            cell_count=2,
            specimen_assignment=[HUMAN_GLIF3_IDENTITY, HUMAN_GLIF5_IDENTITY],
            seed=5,
            subpopulation_parameters={HUMAN_GLIF3_IDENTITY.model_id: colliding},
        )


@pytest.mark.skipif(not _mixture_api_present(), reason=_RED_REASON)
def test_mixture_checkpoint_round_trip_preserves_specimen_labels() -> None:
    from humanbrain.cells import MixedSpecimenPopulation

    assignment = [HUMAN_GLIF3_IDENTITY] * 3 + [HUMAN_GLIF5_IDENTITY] * 5
    pop = MixedSpecimenPopulation(cell_count=8, specimen_assignment=assignment, seed=7)
    for _ in range(50):
        pop.step(np.full(8, 420.0))
    checkpoint = pop.to_checkpoint()
    restored = MixedSpecimenPopulation.from_checkpoint(checkpoint)
    assert restored.specimen_labels() == pop.specimen_labels()
    for _ in range(50):
        restored.step(np.full(8, 420.0))
    for _ in range(50):
        pop.step(np.full(8, 420.0))
    np.testing.assert_array_equal(
        restored.state.voltage_relative_mv, pop.state.voltage_relative_mv
    )
