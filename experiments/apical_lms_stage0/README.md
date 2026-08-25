# B' APICAL-LMS Stage-0 harness — READY TO FIRE (corpus leg license-gated)

Preregistration: `docs/research/path-decision-synthesis.md` @ e16d22c,
section **Probe P-A — B′ = APICAL-LMS Stage‑0** (execution-order step 2).
Benchmark-shape constants transcribed from
`docs/research/deep-dive-dendricl-ctp.md` §1.1–1.3. All gate values were
frozen in `freeze.py` BEFORE implementation (RED-first discipline: the flag-off
parity tests in `tests/test_apical_lms.py` were written and observed failing
with `ModuleNotFoundError` before any of this code existed).

## What this is

Garg-2022-style synthetic in-context regression driven through the existing
`TwoCompartmentGLIF5` substrate with a hand-implemented LOCAL apical update

```
u_A ← α·u_A + γ·e_t·z_t          z_t = W_A x_t   (orthonormal W_A ⇒ exact lift)
e_t = y_t − ⟨u_A, z_t⟩           (next-token residual, teacher-forced reveal)
```

riding `observe()`:

1. drive = u_A published via `set_apical_drive`; token presented under native
   observe() semantics (leading pulse, per-token spike counts);
2. apical voltage v_A logged after each token window;
3. label revealed; residual formed; update applied **only on supervised
   positions** (gated OFF at decision points);
4. `u_A` lives OUTSIDE the population and is never touched by somatic spikes.

A classical reference recursion `w_ref ← α·w_ref + γ·e_ref·x_t` runs alongside
on the same revealed stream (its own residuals). With orthonormal `W_A`,
`u_A(t) = W_A w_ref(t)` identically — so the ONLY difference between the two
trajectories is the fitted membrane transfer function. That is exactly the
Stage-0 question: **can a fitted GLIF5 apical membrane host the recursion?**

## Frozen constants (`freeze.py`, do not edit in place)

| constant | value | source |
| --- | --- | --- |
| α | 0.99 | "α ≈ 0.99-class" (spec); DendriCL learned 0.994–0.997 |
| d_apical / cell_count | 384 / 384 | width cliff ≤ 384 (DD-DCL §1.3); one unit per apical dim (§1.1) |
| γ | 0.2 × γ⋆ = 0.2/(386) ≈ 5.1813e−4 | learned γ ≈ 0.2×γ⋆ inside γ < 2/(d_eff+2) (§1.2); bound at d=384: 5.1813e−3 (10× headroom) |
| R² gate | ≥ 0.85 | benchmark band 0.85–0.93 (DD-DCL §1.3) |
| task shape | d=10 (primary), k=2d supervised steps, 8 query steps, σ=1 | Garg/DendriCL shape |
| substrate | tau_a=100 ms, theta_p=0.5, g_gate=0.6, 5 ms tokens, 1.5 ms pulses @5000 pA | existing class defaults + stack cadence |

## Gate / falsifier (pre-set)

Ridge linear probe: features = logged v_A trajectory (context positions),
targets = reference online-LMS states, train/test split BY EPISODE.
**PASS ⇔ mean held-out R² ≥ 0.85.** Below threshold → a fitted GLIF5 apical
membrane cannot host the recursion → B′ dies at Stage 0 (stop-on-failure).

## Status

- [x] Unit suite green (`tests/test_apical_lms.py`): flag-off bit-parity vs
      stock harness (direct loop AND full episode), determinism, update-rule
      bitwise identity vs hand-computed traces, hosted-lift identity,
      decision-point gating, soma-independence of u_A, stability-bound
      rejection, generator contract, scorer sanity, corpus-leg guard.
- [x] Fixture smoke leg run (`smoke_fixture_verdict.json`) — READINESS ONLY,
      gates nothing. Measured mean held-out R^2 vs the reference trajectory:
      **0.797 @ 24 train / 8 test episodes; 0.847 @ 96 / 32** — rising with
      probe training rows (probe starvation at fixture scale), against the
      0.85 gate and DendriCL's abstract-compartment 0.873 @ d=10. Whether it
      clears the gate with cohort-scale training rows is precisely what the
      licensed corpus leg decides.
- [x] **Corpus cohort leg FIRED (2026-08-25) — VERDICT: FAIL (terminal).**
      Fired under `--license-token retro-probe:PASS` (commit d999ed3) at
      640 train / 160 test episodes with the AMENDED dual-layer gate
      (14cc57b): layer (a) probe R² = 0.837981 < 0.85 → FAIL (stop-on-failure,
      exit code 1); layer (b) hosting relationship HELD at all four swept
      points (decoded ≥ abstract-ref task R²). No width cliff at cohort
      scale. Full readout: `corpus_leg/FINDINGS.md`,
      `corpus_leg/corpus_leg_verdict.json`. **B′ dies at Stage 0** per the
      frozen preregistration semantics; cohort is spent — no reruns.

The corpus guard is mechanical: `--cohort` without a license token exits
immediately (`enforce_cohort_guard`).

## Files

| file | role |
| --- | --- |
| `freeze.py` | frozen constants + stability-bound validation |
| `garg_task.py` | Garg episode generator + orthonormal W_A builder |
| `lms.py` | canonical local rule (bitwise-testable, no hidden state) |
| `runner.py` | `Stage0Runner(ApicalTokenRunner)` — LMS rides observe() |
| `scoring.py` | ridge probe, per-component R², verdict/gate |
| `run_stage0.py` | CLI: fixture leg (default) / license-gated corpus leg |
