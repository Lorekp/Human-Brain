# Benchmark validation: Stage-0 harness vs DendriCL's published Garg-task numbers

**Date:** 2026-08-25
**Status:** COMPLETE — non-decisive calibration legs (fixture-class; corpus cohort leg NOT fired,
license untouched). Harness commit under test: 23bb93b.
**Runner:** `run_benchmark_validation.py` (this directory); per-leg JSONs in `results/`;
tables regenerated verbatim by `analyze_results.py`.
**CPU-only:** single-threaded NumPy replay through `TwoCompartmentGLIF5` (fitted membrane);
no training anywhere in the loop. Total wall ~95 min for 31 legs.

## Verdict in five lines

1. **The fitted GLIF5 apical membrane hosts the recursion essentially losslessly** — decoded-through-membrane
   task R² ≥ abstract-recursion task R² at EVERY operating point (ridge shrinkage through the
   membrane actually helps, +0.03…+0.10). Physiology is not where the distance to DendriCL lives.
2. **We reproduce the qualitative published curve but not its level**: best-arm decoded task R²
   0.44–0.57 vs their 0.649–0.820 (Δ −0.21…−0.34); probe fidelity 0.81–0.89 vs their 0.87–0.97.
   The residual gap is constants/meta-tuning (they BPTT-train α, γ, W_A; we hand-set), not membrane.
3. **A transcription hazard was caught and quantified**: the freeze derives γ from d_apical
   (γ=5.18e−4), but DendriCL's γ⋆=1/(d+2) is w.r.t. the TASK dimension. Under the frozen reading
   the recursion barely runs within k=2d steps (task R² 0.018–0.065) while probe R² stays
   0.83–0.86 — **high probe R² alone cannot distinguish "membrane hosts the recursion" from
   "recursion barely ran."** Always report the task layer alongside the probe.
4. **No width cliff exists in this substrate out to d_apical=384 — mechanistically.** At fixed
   in-bound γ, every metric is bit-flat across widths 10→384 while spikes/token scale 7.6→266.6:
   `apical_voltage_mv` integrates only `_apical_drive` (cells.py step part (a)), so v_A is an exact
   linear low-pass of u_A and linearly invertible at ANY width. DendriCL's cliff was a BPTT/γ-stability
   phenomenon; hand-set in-bound constants exclude it by construction. The joint-freeze arm's decline
   (decoded 0.53→0.03 across widths) is pure γ-shrinkage artifact, not physiology.
5. Determinism/harness health: exact-lift identity ≤ 1.2e−14 on all 31 legs; two independent
   config-coincidence cross-checks match to 4 decimals; unit suite 14/14 green at run time.

## 1. What was asked

Run the Stage-0 harness at DendriCL's PUBLISHED operating points
(`docs/research/deep-dive-dendricl-ctp.md` §1: d=10 primary / d=20 secondary / super-d
d∈{25,30,40,50}, k=2d supervised steps, σ=1, w_std=1) and compare our measured R² trajectory
against their reported curve — do we reproduce their abstract-compartment numbers with our
fitted-GLIF5 membrane? Then sweep d_eff ∈ {10, 32, 128, 384} to map the width cliff empirically.

## 2. What DendriCL published vs what we can measure (comparability contract)

Their published quantities (verbatim from deep-dive §1.3):

| published quantity | value |
| --- | --- |
| Garg-2022 ICL task R² | 0.807 @ d=10; 0.820 @ d=20 |
| Super-d seed-stability (50k BPTT steps) | 0.809 (d25), 0.807 (d30), 0.787 (d40), 0.649 (d50), σ ≤ 0.036 |
| Apical linear probe R² (u_A → reference online-LMS trajectory) | 0.873 @ d10, 0.931 @ d20, 0.97 @ d30 |
| Continuous ablation (no LIF) | 0.833 |
| Pure LIF baseline | 0.801 @ d10 → 0.086 @ d20 |
| Active Dendrites (Iyer 2022) | 0.668 @ d10 → 0.061 @ d20 |
| Width cliff | d_apical ≤ 384 trains; ≥ 512 diverges at d=20 |

Structural differences bounding the comparison (stated up front):

1. **No BPTT here.** They meta-train (α, γ, W_A, readout) end-to-end for 10k–50k steps; the
   Stage-0 contract is hand-set constants inside the stability region + frozen orthonormal-lift
   W_A, zero parameter updates. We measure what the *fitted membrane* does to a recursion it is
   *given*, not what meta-training could reach.
2. **Three R² layers**, because "R²" means two different things in their table:
   - `probe R²` = held-out linear-probe fidelity of logged apical voltages v_A vs the reference
     online-LMS trajectory (split BY EPISODE) — the Stage-0 gate metric, comparable to their
     probe column;
   - `ref task R²` = task-level R² of the abstract reference recursion itself (predictions
     ⟨w_ref(t), x_t⟩ at query positions vs noiseless w·x_t, Garg normalized-error convention;
     labels are unrevealed at queries);
   - `decoded task R²` = same task metric with ŵ decoded FROM THE MEMBRANE through the fitted
     ridge probe (train episodes' context rows → test episodes' query-position voltages). The
     honest end-to-end fitted-GLIF5 number.
3. **Exact-lift identity (verified per leg):** with orthonormal W_A, u_A(t)=W_A·w_ref(t)
   identically; max deviation ≤ 1.2e−14 across all legs (`exact_lift_max_abs_err` in each JSON).
   Ref-vs-hosted differences therefore CANNOT be constants — only the v_A transfer path carries
   physiology.
4. **Query count:** §1 gives no explicit per-episode query count; the freeze pins query_len=8.
   State is frozen across the query tail, so this affects estimation precision only. Recorded as
   the one shape constant not extractable verbatim from §1.
5. Their eval reads out through a trained somatic path; our decoded layer reads the apical state
   directly (Stage-0 asks about the STATE, not the soma readout).

## 3. Gamma-arm control discovered by the validation (important)

DendriCL's Prop-1 bound γ < 2/d and optimum γ⋆ = 1/(d+2) are stated w.r.t. the TASK dimension d
of x_t (their configs vary d=5…50 with d_apical fixed at 384; learned γ ≈ 0.2·γ⋆). Our frozen
`freeze.py` derives γ from **d_apical** (=384): γ_frozen = 0.2/386 ≈ 5.18e−4. With the
orthonormal lift the update lives entirely on a d_task-dimensional subspace, so the task-dim
reading is the mechanistically matched one. Both readings were run:

- **frozen** — α=0.99, γ = 0.2·γ⋆(d_apical=384) [preregistered configuration];
- **gstar_a99** — α=0.99, γ = 0.2·γ⋆(d_task) [their transcription];
- **gstar_a996** — α=0.996, γ = 0.2·γ⋆(d_task) [adds their learned-α class 0.994–0.997].

`freeze.py` was NOT edited (any change would be a new preregistration); arms are locally-declared
variants of the frozen dataclass inside the runner script, each validated inside the task-dim
stability bound (recorded per JSON alongside the d_apical bound). Consequence: the frozen γ is
~32× smaller than the gstar γ at d_task=10 and leaves the recursion essentially unmoved within
k=2d steps, while probe fidelity stays high — see finding 3 above.

## 4. T1 — Validation at DendriCL's published operating points

160 train / 40 held-out episodes per point; base seed 20260825. Verbatim values:

| d_task | DendriCL published task R² | DendriCL published probe R² | arm | our probe R² (gate metric) | our abstract-ref task R² | our decoded-through-membrane task R² | Δ decoded vs published task |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.807 | 0.873 | frozen | 0.8262 | 0.0183 | 0.0269 | -0.7801 |
| 10 | 0.807 | 0.873 | gstar_a99 | 0.8109 | 0.4247 | 0.5309 | -0.2761 |
| 10 | 0.807 | 0.873 | gstar_a996 | 0.8217 | 0.4422 | 0.5451 | -0.2619 |
| 20 | 0.820 | 0.931 | frozen | 0.8533 | 0.0367 | 0.0456 | -0.7744 |
| 20 | 0.820 | 0.931 | gstar_a99 | 0.8446 | 0.4582 | 0.5362 | -0.2838 |
| 20 | 0.820 | 0.931 | gstar_a996 | 0.8610 | 0.4986 | 0.5743 | -0.2457 |
| 25 | 0.809 | -- | frozen | 0.8559 | 0.0395 | 0.0459 | -0.7631 |
| 25 | 0.809 | -- | gstar_a99 | 0.8470 | 0.4040 | 0.4540 | -0.3550 |
| 25 | 0.809 | -- | gstar_a996 | 0.8667 | 0.4463 | 0.4954 | -0.3136 |
| 30 | 0.807 | 0.97 | frozen | 0.8530 | 0.0436 | 0.0494 | -0.7576 |
| 30 | 0.807 | 0.97 | gstar_a99 | 0.8449 | 0.3820 | 0.4206 | -0.3864 |
| 30 | 0.807 | 0.97 | gstar_a996 | 0.8685 | 0.4301 | 0.4663 | -0.3407 |
| 40 | 0.787 | -- | frozen | 0.8649 | 0.0562 | 0.0629 | -0.7241 |
| 40 | 0.787 | -- | gstar_a99 | 0.8578 | 0.3841 | 0.4191 | -0.3679 |
| 40 | 0.787 | -- | gstar_a996 | 0.8859 | 0.4521 | 0.4849 | -0.3021 |
| 50 | 0.649 | -- | frozen | 0.8648 | 0.0584 | 0.0648 | -0.5842 |
| 50 | 0.649 | -- | gstar_a99 | 0.8593 | 0.3430 | 0.3717 | -0.2773 |
| 50 | 0.649 | -- | gstar_a996 | 0.8921 | 0.4184 | 0.4438 | -0.2052 |

Reading: their task curve declines gently with d (0.807→0.649); ours is roughly FLAT
(0.44–0.57 best arm) — we do not reproduce their super-d degradation shape, and our worst
absolute gap is at mid-d (−0.34 at d=30) while it narrows at d=50 (−0.21). Against the
non-learning baselines they report (Pure LIF 0.086 @d20; Active Dendrites 0.061 @d20) our
0.49–0.57 @d20 sits far above chance: the substrate+local-rule combination genuinely performs
in-context regression, just below their meta-trained level.

## 5. T2 — Seed stability

Extra episode-draw seeds under gstar_a996; W_A/wiring seeds stay frozen. NOTE the variance
sources differ: their σ≤0.036 varies 50k-step BPTT TRAINING runs; ours varies evaluation
episode draws only.

| leg | seeds | mean probe R² | σ probe R² | mean decoded task R² | σ decoded task R² | DendriCL seed σ |
| --- | --- | --- | --- | --- | --- | --- |
| seeds_d10 (incl. base-seed anchor) | 4 | 0.8324 | 0.0075 | 0.5200 | 0.0243 | ≤ 0.036 |
| seeds_d30 (incl. base-seed anchor) | 3 | 0.8708 | 0.0023 | 0.5102 | 0.0394 | ≤ 0.036 |

Eval-noise at d30 (σ 0.0394 over n=3) is marginally wider than their reported ceiling; probe
σ is 5–15× tighter than their band. No instability signature anywhere near their failure modes
(grokking/trimodality); curves are smooth and monotone in every run — their "graceful absence
of failure" taxonomy applies to our legs too.

## 6. T3 — Width sweep (empirical cliff map)

d_eff ∈ {10, 32, 128, 384}, cell_count = d_apical ("one unit per apical dimension", freeze.py),
d_task=10 anchor. Two γ arms: **joint** = γ derived from d_apical (preregistered joint
declaration); **gstar_a99** = γ pinned at 0.2·γ⋆(10) isolating pure width response. Fanouts
capped at min(frozen, cell_count) mechanically. Verbatim:

| d_apical = cell_count | γ | bound 2/(d_ap+2) | arm | mean spikes/token | probe R² | abstract-ref task R² | decoded task R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 1.667e-02 | 1.667e-01 | joint | 7.6 | 0.8109 | 0.4247 | 0.5309 |
| 10 | 1.667e-02 | 1.667e-01 | gstar_a99 | 7.6 | 0.8109 | 0.4247 | 0.5309 |
| 32 | 5.882e-03 | 5.882e-02 | joint | 24.7 | 0.8213 | 0.1880 | 0.2602 |
| 32 | 1.667e-02 | 5.882e-02 | gstar_a99 | 24.7 | 0.8109 | 0.4247 | 0.5309 |
| 128 | 1.538e-03 | 1.538e-02 | joint | 93.3 | 0.8252 | 0.0540 | 0.0780 |
| 128 | 1.667e-02 | 1.538e-02 | gstar_a99 | 93.3 | 0.8109 | 0.4247 | 0.5309 |
| 384 | 5.181e-04 | 5.181e-03 | frozen | 266.6 | 0.8262 | 0.0183 | 0.0269 |
| 384 | 5.181e-04 | 5.181e-03 | joint (=frozen config) | 266.6 | 0.8262 | 0.0183 | 0.0269 |
| 384 | 1.667e-02 | 5.181e-03 | gstar_a99 | 266.6 | 0.8109 | 0.4247 | 0.5309 |

(At width 10 the two readings coincide: d_apical = d_task. width384_joint IS
val_d10_primary_frozen — identical config, identical numbers, a free determinism cross-check.)

Findings:

- **gstar arm is bit-flat**: probe/ref/decoded identical to 4 decimals from 10 to 384 units,
  while spikes/token scale 35×. Mechanism (verified in `src/humanbrain/cells.py::step`, part
  (a)): `apical_voltage_mv += (drive − voltage)·scale` depends ONLY on `_apical_drive`; no
  somatic/spike/gate term ever writes into it. v_A is therefore an exact linear low-pass of the
  u_A sequence — linearly invertible at any width, wiring-independent, spike-independent. In
  THIS substrate, apical transfer fidelity cannot degrade with width.
- **joint arm declines only through γ-shrinkage**: decoded 0.53→0.26→0.08→0.03 tracks
  γ = 0.2/(w+2) exactly; probe stays ~0.81–0.83 (still decodable, just tracking a slower
  target). Any "cliff" read off this arm is a step-size policy artifact, not membrane physics.
- **DendriCL's cliff has no counterpart here by construction**: theirs emerges when BPTT pushes
  γ past the LMS stability region at large width (≥512 diverges at d=20). Hand-set γ declared
  jointly inside the in-subspace bound cannot exit it. The preregistered ≤384 ceiling remains a
  prudent guard, but the empirical map shows the constraint binds the CONSTANTS, never the
  membrane.

## 7. Consolidated findings & license-decision implications

1. **Stage-0's physiological question is answered YES with margin.** The thing Stage-0 existed
   to falsify — "a fitted GLIF5 apical membrane cannot host the recursion" — did not happen:
   trajectory decode is near-lossless (probe 0.81–0.89; decoded ≥ abstract-ref everywhere).
   The measured physiology tax on DendriCL's idealized linear recurrence is ~ZERO.
2. **The remaining distance to their published curve is meta-tuning, not substrate**: their
   BPTT tunes (α, γ, W_A, readout) end-to-end; our hand-set pair reaches ~60–70% of their task
   R² at matched operating points. If B′ later needs their LEVEL, that argues for a tuned
   readout/meta-step — orthogonal to, and compatible with, the local-learning contract for the
   online update itself.
3. **Methodological warning adopted into practice**: probe R² must never be reported alone.
   The frozen-γ arm scored 0.83–0.86 probe while the hosted recursion accomplished ~nothing
   (task R² 0.02–0.07). A future reader of the licensed cohort leg should gate on BOTH layers.
4. **γ-transcription hazard for any future preregistration revision**: d_eff in "γ < 2/(d_eff+2)"
   must be READ AS d_task (DendriCL Prop-1 semantics), not d_apical. As frozen, Stage-0's
   effective learning rate makes the recursion ~static on the k=2d budget. This document
   quantifies both readings; changing the freeze itself is a new preregistration, out of scope
   here.
5. These legs were NON-DECISIVE by design (fixture-class; no `--cohort`; license untouched).
   The 0.85-gate verdict still belongs to the licensed corpus leg. For calibration: the
   fixture-scale smoke (96/32) reached 0.847 probe and this work reached 0.81–0.89 at larger
   scales — i.e., the preregistered 0.85 gate sits INSIDE our measured probe band, so the
   cohort leg's outcome will hinge on episode count and split, exactly as the README predicted.

## 8. Process notes (bugs found during this work)

- First width-joint pass silently derived γ from d_task (arm-routing bug in the orchestrator):
  exposed because width32/128/384_joint returned bit-identical numbers to the gstar arm —
  the determinism cross-checks paid for themselves. Fixed (`make_constants` mode dispatch),
  affected legs deleted and re-run; everything reported here is post-fix output.
- No changes to any file outside `experiments/apical_lms_stage0/benchmark_validation/`.

## 9. Provenance & reproduction

- Published comparators transcribed verbatim from `docs/research/deep-dive-dendricl-ctp.md`
  §1.3 (single source of truth for their numbers here).
- Per-leg raw results: `benchmark_validation/results/*.json` — each records the α/γ/bounds
  actually used, fanouts, probe per-dim R², both task R² layers, query-position count,
  exact-lift error, spikes/token, wall time.
- Regenerate tables: `python experiments/apical_lms_stage0/benchmark_validation/analyze_results.py all`.
- Re-run everything (resumable, skips existing): `python experiments/apical_lms_stage0/benchmark_validation/run_benchmark_validation.py [--force]`.
