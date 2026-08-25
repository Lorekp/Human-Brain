# R3 learning-rule spec — phase-2 plasticity for the episodic-store mainline

**Date:** 2026-08-25
**Status:** preregistration freeze — phase-2 learning rule, frozen BEFORE implementation
**Extends:** [r3-segment-growth-preregistration.md](r3-segment-growth-preregistration.md)
(structural pilot: segment growth mechanics, gates G-S0..S3)
**Learning-rule sources:** [deep-dive-nest-eprop-scale.md](deep-dive-nest-eprop-scale.md)
(e-prop skeleton), burst-fraction telemetry (`burst_metrics.py` @ ee3b7c6),
race outcomes via observation sink.

This document freezes WHAT updates synapses in path-2 phase 2, after the
structural pilot (phase 1) grows segments. Phase 1 answers "do grown segments
recognize transitions"; this spec governs "what plasticity makes recognition
improve on held-out books."

## 1. Eligibility trace per segment contact

e-prop's per-synapse eligibility is `e_ji^t = ψ_j^t · z̄_i^{t−1}` — presynaptic
trace times postsynaptic excitability. Adapted to dendritic segments:

- **Presynaptic side (z̄):** running spike-trace of the presynaptic winner
  cell, decay τ_z = 20 ms token-frame convention (z̄ ← z̄·exp(−Δt/τ_z) + z).
- **Postsynaptic side (ψ):** segment-level excitability = sigmoid of the
  segment's current matching score minus θ_m (potential threshold), computed
  only when the owning cell is active. For GLIF5 adaptation-bearing cells the
  ALIF recursion ε_a accumulates: ε_a ← ε_a·(1−1/τ_a) + β·s_t with τ_a = 50 ms,
  entering ψ multiplicatively (adaptation-bearing cells hold credit longer —
  the DD-NEST finding that adaptation buys temporal credit).
- **Contact granularity:** one e per (segment, contact) pair; contacts beyond
  the 20-contact cap are ineligible by construction (frozen in r3-prereg).

## 2. Local third factor L(t) — novelty/consolidation signal

L(t) is computed ONLY from quantities available at the cell/segment level:

```
L(t) = w_b · B(t) + w_r · R(t) + w_n · N(t)

B(t) = burst indicator of the postsynaptic cell at t      (from BurstFractionTracker)
R(t) = race-outcome signal: +1 if owning cell won the output race,
       −1 if it was runner-up, 0 otherwise                 (from observation sink)
N(t) = novelty: 1 if this (context-token) conjunction has no existing
       committed segment on this cell yet, else 0          (local set membership)
```

Weights (w_b, w_r, w_n) are frozen constants at prereg time: (0.4, 0.4, 0.2).

**Locality proof:** B(t) requires only the cell's own predicted-state history;
R(t) arrives as a scalar broadcast (reward-e-prop shape, DD-NEST §4 — a diffuse
scalar is biologically the neuromodulator case and computationally one float);
N(t) reads only the cell's own segment table. No cross-population reads, no
supervised targets, no gradient machinery. Contract tripwires apply.

## 3. Update rule mapped onto the sparse runtime

```
ΔW_contact = η · L(t) · e_contact(t)        for touched contacts only
permanence += ΔW                            clipped to [0, 1]
connected ⇔ permanence ≥ 0.50               (r3-prereg constants)
```

- **Event-driven flush:** touched contacts accumulate ΔW in a staged buffer;
  flush at sequence boundaries via the Seam-B indexed staged membership
  (delivery-phase semantics per cpu-seams-spec.md). Learning memory bounded
  by touched cells — never a population-wide scan.
- **Reads** go through Seam A `read_cells` (36.9 µs @ K=16, 1e6 residents).
- **η schedule:** η₀ = 0.01 constant for the phase-2 probe; no annealing until
  stability data exists (avoid tuning freedom before measurement).

## 4. Stability analysis

- **Homeostatic row bound:** each segment's total connected weight is bounded:
  if Σ permanence > s_max = 16 (of cap 20), new growth pauses for that segment
  (growth-cap interaction already in r3-prereg).
- **Trace stability:** τ_z, τ_a fixed; e is bounded ∈ [0, ψ_max] by
  construction (sigmoid ψ, binary z traces).
- **Update magnitude:** |ΔW| ≤ η · |L|max · e_max = 0.01 · 1.0 · 1.0 = 0.01
  per contact per event — three orders below the permanence clip, so no
  single event can flip connectivity. Structural change requires repetition,
  which is the intended behavior.
- Relation to γ < 2/(d+2): that bound governs the apical-LMS lane (P-A); this
  spec's discrete-event update is self-bounding via the clip and needs no
  spectral condition.

## 5. Phase-2 probe gates

Inherits r3-prereg G-S0..S3 structure, adding learning-specific gates:

| Gate | Requirement |
|---|---|
| G-L1 | TRAIN transition recognition improves vs phase-1 endpoint (+0.05 Wilson-clear) |
| G-L2 | Held-out recognition improves above burst baseline (Δ ≥ 0.05, Wilson 95% LB > 0) |
| G-L3 | Attribution: growth-off + L-on arm shows no held-out movement (learning, not drift, does the work) |

Cohort ladder identical: C1 @128k gates C2 @1M directional replication.
Burst-fraction telemetry rides every replay log (already instrumented).

## 6. Falsifier

The recurring program failure signature, pre-declared: **TRAIN-book
transition recognition improves while held-out books sit at burst baseline**
(U-STATE P3: 38.3% TRAIN → 0% VAL) ⇒ the learning memorizes without
transferable structure; phase-2 dies and path 2 falls back to the probe-
defined path-3 boundary. Secondary falsifier: G-L3 fails (held-out moves in
the growth-off arm) ⇒ observed movement is drift or measurement artifact,
not learning — redesign required before any further spend.

If instead held-out recognition moves under G-L1..G-L3 jointly, R3 proceeds:
path 2 has its first measured learning mechanics behind frozen gates.

## Verdict semantics

Four mechanical verdicts, mirroring r3-prereg: PASS_ALL, NEGATIVE_MEMORIZE,
NEGATIVE_DRIFT, MIXED_SCALE_BLOCK (@128k passes, @1M replication fails).
