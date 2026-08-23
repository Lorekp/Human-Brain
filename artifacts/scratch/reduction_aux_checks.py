"""Aux checks v2:
1. Sum-order identity: one-step voltage equality from a hand-set nonzero ASC
   state (kernel integrates raw stored; GLIF3 integrates post-decay; under the
   delta-shift the two sums must agree BITWISE if summation order matches).
2. V1 throughput: AdaptiveGlifPopulation(N).step wall-time, GLIF5 params vs
   reduced GLIF3 params (identical instruction stream => no dispatch cost).
"""

from __future__ import annotations

import struct
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from humanbrain.cells import (
    AllenGlif3Parameters,
    AllenGlif5Parameters,
    AdaptiveGlifPopulation,
    FittedGlif5Constants,
)

P3 = AllenGlif3Parameters()
K = P3.dt_ms / P3.capacitance_pf


def bits(x) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


# --- 1. one-step sum-order check --------------------------------------------
taus = np.asarray(P3.after_spike_taus_ms, dtype=np.float64)
delta = np.exp(-P3.dt_ms / taus)
amps_shifted = tuple(float(a) for a in np.asarray(P3.after_spike_amplitudes_pa) * delta)

reduced = AllenGlif5Parameters(
    rest_mv=P3.rest_mv,
    dt_ms=P3.dt_ms,
    capacitance_pf=P3.capacitance_pf,
    input_resistance_mohm=P3.input_resistance_mohm,
    threshold_inf_relative_mv=P3.threshold_relative_mv,
    spike_cut_steps=P3.spike_cut_steps,
    after_spike_amplitudes_pa=amps_shifted,
    after_spike_taus_ms=P3.after_spike_taus_ms,
    voltage_reset_scale=0.0,
    voltage_reset_offset_mv=0.0,
    threshold_spike_increment_mv=0.0,
    threshold_voltage_coupling_per_s=0.0,
)

rng = np.random.default_rng(7)
mismatches = 0
worst_ulp = 0
trials = 2000
for _ in range(trials):
    a_stored = rng.uniform(-800.0, 800.0, size=2).round(3)
    v0 = float(rng.uniform(-10.0, 24.0))
    i0 = float(rng.uniform(-100.0, 900.0))

    # GLIF3 reference: decay then integrate (cells.py:448-453)
    d = np.asarray(a_stored) * delta          # fl(delta * a) elementwise
    s_np = float(d.sum())                     # numpy 2-term sum
    v3 = v0 + K * (i0 + s_np - P3.leak_ns * v0)

    # Kernel path: integrate raw stored (lib.rs:76-84); feed it delta*a_stored
    pop = AdaptiveGlifPopulation(1, reduced)
    pop.state.voltage_relative_mv[0] = v0
    pop.state.threshold_voltage_mv[0] = 0.0
    pop.state.threshold_spike_mv[0] = 0.0
    pop.state.after_spike_currents_pa[0] = d  # mapping-2 state
    pop.step(i0)
    v5 = float(pop.state.voltage_relative_mv[0])
    # exclude crossings: only compare subthreshold outcomes
    if v3 > P3.threshold_relative_mv or v5 > P3.threshold_relative_mv:
        continue
    if bits(v3) != bits(v5):
        mismatches += 1
        lo, hi = sorted((v3, v5))
        cursor, count = lo, 0
        while cursor != hi and count < 100:
            cursor = float(np.nextafter(cursor, np.inf))
            count += 1
        worst_ulp = max(worst_ulp, count)
print(f"sum-order check: {trials} trials, mismatches={mismatches}, worst ulp={worst_ulp}")

# --- 2. throughput ----------------------------------------------------------
N = 65_536
STEPS = 300
full = AllenGlif5Parameters()


def bench(params: AllenGlif5Parameters, label: str) -> float:
    pop = AdaptiveGlifPopulation(N, params)
    current = np.full(N, 420.0)
    pop.step(current)  # warm
    t0 = time.perf_counter()
    for _ in range(STEPS):
        pop.step(current)
    dt = (time.perf_counter() - t0) / STEPS
    print(f"{label}: {dt * 1e3:.3f} ms/step  ({N / dt / 1e6:.2f} Mcell-steps/s)")
    return dt


t_full = bench(full, f"N={N} full GLIF5   ")
t_red = bench(reduced, f"N={N} reduced G3 ")
print(f"dispatch overhead ratio: {t_red / t_full:.4f} (identical streams expected => ~1)")

c = FittedGlif5Constants.from_parameters(reduced)
print(f"reduced constants: phi={c.threshold_phi!r} vfactor={c.threshold_voltage_factor!r} "
      f"bfactor={c.threshold_beta_factor!r}")
