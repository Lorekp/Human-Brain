"""Scratch verification harness for glif3-as-glif5-expressibility.md (v2).

Compares the COMMITTED STATE ENTERING each step, bit-for-bit:
  A-C. pinned GLIF3 (simulate_glif3_current loop replay) vs AdaptiveGlifPopulation
       native kernel under reduced parameters with delta-shifted ASC amplitudes.
  D.   zeroed-ASC degenerate LIF (predicted bit-exact everywhere),
  E.   unshifted amplitudes (predicted macroscopic divergence -> red control).
"""

from __future__ import annotations

import struct
import sys

import numpy as np

sys.path.insert(0, "src")

from humanbrain.cells import (
    AllenGlif3Parameters,
    FittedGlif5Constants,
    AllenGlif5Parameters,
    AdaptiveGlifPopulation,
)

P3 = AllenGlif3Parameters()


def reduced_glif5(shift_amplitudes: bool, zero_asc: bool = False) -> AllenGlif5Parameters:
    taus = np.asarray(P3.after_spike_taus_ms, dtype=np.float64)
    decay = np.exp(-P3.dt_ms / taus)
    amps = np.asarray(P3.after_spike_amplitudes_pa, dtype=np.float64)
    if zero_asc:
        return AllenGlif5Parameters(
            rest_mv=P3.rest_mv,
            dt_ms=P3.dt_ms,
            capacitance_pf=P3.capacitance_pf,
            input_resistance_mohm=P3.input_resistance_mohm,
            threshold_inf_relative_mv=P3.threshold_relative_mv,
            spike_cut_steps=P3.spike_cut_steps,
            after_spike_amplitudes_pa=(0.0, 0.0),
            after_spike_taus_ms=P3.after_spike_taus_ms,
            voltage_reset_scale=0.0,
            voltage_reset_offset_mv=0.0,
            threshold_spike_increment_mv=0.0,
            threshold_voltage_coupling_per_s=0.0,
        )
    shifted = tuple(float(a) for a in (amps * decay))
    naive = P3.after_spike_amplitudes_pa
    return AllenGlif5Parameters(
        rest_mv=P3.rest_mv,
        dt_ms=P3.dt_ms,
        capacitance_pf=P3.capacitance_pf,
        input_resistance_mohm=P3.input_resistance_mohm,
        threshold_inf_relative_mv=P3.threshold_relative_mv,
        spike_cut_steps=P3.spike_cut_steps,
        after_spike_amplitudes_pa=shifted if shift_amplitudes else naive,
        after_spike_taus_ms=P3.after_spike_taus_ms,
        voltage_reset_scale=0.0,
        voltage_reset_offset_mv=0.0,
        threshold_spike_increment_mv=0.0,
        threshold_voltage_coupling_per_s=0.0,
    )


def bits(x) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def run_glif3(stimulus: np.ndarray, params: AllenGlif3Parameters):
    """Committed state entering every step: (v_bits, decayed_asc_or_None, r).

    decayed_asc is recorded only at integrating steps: the post-decay array
    that actually feeds the voltage update (mapping-2 observable).
    """
    p = params
    relative_v = 0.0
    after = np.zeros(len(p.after_spike_taus_ms), dtype=np.float64)
    amps = np.asarray(p.after_spike_amplitudes_pa, dtype=np.float64)
    taus = np.asarray(p.after_spike_taus_ms, dtype=np.float64)
    asc_decay = np.exp(-p.dt_ms / taus)
    cut_decay = np.exp(-p.spike_cut_ms / taus)
    refractory = 0
    states = []
    spikes = []
    for injected in stimulus:
        if refractory > 0:
            states.append((bits(relative_v), None, refractory))
            refractory -= 1
            continue
        after *= asc_decay
        states.append((bits(relative_v), tuple(bits(x) for x in after), refractory))
        relative_v += (
            p.dt_ms / p.capacitance_pf
            * (float(injected) + float(after.sum()) - p.leak_ns * relative_v)
        )
        if relative_v > p.threshold_relative_mv:
            spikes.append(len(states) - 1)
            relative_v = 0.0
            after = amps + after * cut_decay
            refractory = p.spike_cut_steps
    return states, spikes


def run_reduced(params: AllenGlif5Parameters, stimulus: np.ndarray):
    pop = AdaptiveGlifPopulation(1, params)
    states = []
    spikes = []
    for injected in stimulus:
        integrating = int(pop.state.refractory_steps[0]) == 0
        # mapping-2 observable: the kernel's raw stored ASC feeds its voltage
        # update pre-decay, exactly like GLIF3's post-decay working value.
        asc = (
            tuple(bits(float(x)) for x in pop.state.after_spike_currents_pa[0])
            if integrating
            else None
        )
        states.append(
            (bits(pop.state.voltage_relative_mv[0]), asc, int(pop.state.refractory_steps[0]))
        )
        fired = bool(pop.step(float(injected))[0])
        if fired:
            spikes.append(len(states) - 1)
    # inert-threshold audit
    u_bits = {bits(x) for x in pop.state.threshold_voltage_mv}
    s_bits = {bits(x) for x in pop.state.threshold_spike_mv}
    return states, spikes, u_bits, s_bits


def ulp_distance(b1: int, b2: int) -> int:
    """Exact ulp distance for close values; capped estimate across sign flips."""
    x1 = struct.unpack("<d", struct.pack("<Q", b1))[0]
    x2 = struct.unpack("<d", struct.pack("<Q", b2))[0]
    if x1 == x2:
        return 0
    lo, hi = sorted((x1, x2))
    scale = np.spacing(max(abs(lo), abs(hi)))
    if scale == 0.0 or (hi - lo) / scale > 64.0:  # far apart: report cap only
        return 65
    count = 0
    cursor = lo
    while cursor != hi and count <= 64:
        cursor = float(np.nextafter(cursor, np.inf))
        count += 1
    return min(count, 65)


def compare(label: str, stimulus: np.ndarray, shift: bool, zero_asc: bool) -> None:
    if zero_asc:  # degenerate control: zero the ASCs on BOTH sides
        glif3_params = AllenGlif3Parameters(after_spike_amplitudes_pa=(0.0, 0.0))
    else:
        glif3_params = P3
    states3, spikes3 = run_glif3(stimulus, glif3_params)
    params = reduced_glif5(shift_amplitudes=shift, zero_asc=zero_asc)
    states5, spikes5, u_bits, s_bits = run_reduced(params, stimulus)
    assert len(states3) == len(states5)
    print(f"== {label}")
    print(f"   spikes glif3={len(spikes3)} reduced={len(spikes5)} identical={spikes3 == spikes5}")
    print(f"   inert thresholds: threshold_voltage bits={u_bits} threshold_spike bits={s_bits}")
    first = None
    worst_v = 0
    worst_a = 0
    n_mismatch = 0
    max_abs_dv = 0.0
    max_abs_da = 0.0
    r_mismatch = None
    for t, ((v3, a3, r3), (v5, a5, r5)) in enumerate(zip(states3, states5)):
        if r3 != r5:
            if r_mismatch is None:
                r_mismatch = t
            continue
        if a3 is not None and a5 is not None:
            for x3, x5 in zip(a3, a5):
                if x3 != x5:
                    xa = abs(
                        struct.unpack("<d", struct.pack("<Q", x3))[0]
                        - struct.unpack("<d", struct.pack("<Q", x5))[0]
                    )
                    max_abs_da = max(max_abs_da, xa)
                    worst_a = max(worst_a, ulp_distance(x3, x5))
        if v3 != v5:
            n_mismatch += 1
            dv = abs(
                struct.unpack("<d", struct.pack("<Q", v3))[0]
                - struct.unpack("<d", struct.pack("<Q", v5))[0]
            )
            max_abs_dv = max(max_abs_dv, dv)
            worst_v = max(worst_v, ulp_distance(v3, v5))
            if first is None:
                first = t
    print(f"   refractory mismatches: {'none' if r_mismatch is None else f'from step {r_mismatch}'}")
    print(f"   voltage: first mismatch step={first} mismatches={n_mismatch}/{len(states3)} "
          f"worst={worst_v} ulp, max|dv|={max_abs_dv:.3e} mV")
    print(f"   asc(post-decay view): worst<={worst_a} ulp, max|da|={max_abs_da:.3e} pA")


def main() -> None:
    params = reduced_glif5(shift_amplitudes=True)
    c = FittedGlif5Constants.from_parameters(params)
    leak_per_s = params.leak_ns / params.capacitance_pf * 1000.0
    print(f"leak_per_s={leak_per_s!r} phi={c.threshold_phi!r} "
          f"vfactor={c.threshold_voltage_factor!r} bfactor={c.threshold_beta_factor!r}")

    rng = np.random.default_rng(20260823)
    stim_const = np.full(4000, 600.0)
    stim_noisy = rng.uniform(-40.0, 900.0, size=6000).round(6)
    stim_piece = np.concatenate([
        np.full(800, 450.0), np.full(200, 0.0), np.full(1200, -80.0), np.full(1800, 750.0)
    ])
    stim_zero = rng.uniform(-60.0, 120.0, size=3000).round(6)  # subthreshold regime

    for name, stim in (("A-const600", stim_const), ("B-noisy", stim_noisy), ("C-piecewise", stim_piece)):
        compare(name, stim, shift=True, zero_asc=False)
    compare("D-zeroasc-noisy", stim_noisy, shift=True, zero_asc=True)
    compare("E-subthreshold-noisy", stim_zero, shift=True, zero_asc=False)
    compare("F-red-unshifted-naive", stim_const, shift=False, zero_asc=False)


if __name__ == "__main__":
    main()
