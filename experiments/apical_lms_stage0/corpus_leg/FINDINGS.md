# B' Stage-0 CORPUS LEG — decisive dual-layer execution

**Date:** 2026-08-25
**Status:** TERMINAL **FAIL** — cohort spent in one shot, stop-on-failure honored.
**B' dies at Stage 0** per the frozen preregistration semantics
(`docs/research/path-decision-synthesis.md`, Probe P-A).
**License:** `retro-probe:PASS` (commit d999ed3). Harness 23bb93b; dual-layer
gate amendment 14cc57b. Branch `codex/path2-pilot`; leg fired at HEAD=d999ed3
(clean tree for all Stage-0 files).

## Verdict in five lines

1. **Layer (a) — the frozen Stage-0 gate — FAILED:** mean held-out probe R² =
   **0.837981 < 0.85** (median 0.840668) at the preregistered configuration
   (d_task=10, α=0.99, d_apical=cell_count=384, γ=5.1813e−4 = 0.2·γ⋆(384),
   ridge λ=1e−3, seed 20260825). The licensed CLI invocation exited rc=1
   (stop-on-failure exercised at cohort scale for the first time).
2. **Layer (b) — the hosting relationship — HELD at every matched point**
   (decoded-through-membrane task R² ≥ abstract-recursion task R², margins
   +0.0068 … +0.0929): the fitted GLIF5 apical transfer remains essentially
   lossless. The failure is therefore NOT membrane destruction of content; it
   is the single-frame linear-probe fidelity of v_A saturating below the
   published band's 0.85 floor (~0.83–0.84 asymptote) at every scale tried.
   The gate was frozen on this metric, so the verdict stands: **FAIL**.
3. **Determinism certified:** the mandated CLI leg and the independent
   three-layer recompute agree to **exactly 0.0** on probe R²; exact-lift
   identity u_A = W_A·w_ref holds to **2.78e−16** over 800 episodes.
4. **No width cliff at cohort scale under frozen constants** — confirmed:
   probe R² flat within 0.0126 across d_eff ∈ {10, 32, 128, 384}
   (0.8253 / 0.8341 / 0.8373 / 0.8380) while spikes/token scale 7.6 → 266.6;
   decoded/ref task R² decline tracks γ(d_eff) = 0.2/(d_eff+2) exactly
   (pure γ-shrinkage artifact, zero physiology). T3's mechanism finding
   replicates at 4× episode count.
5. **Probe starvation is ruled out as the explanation of the miss:** probe R²
   trajectory vs training rows — 0.797 @24/8 (480 rows) → 0.847 @96/32
   (1920 rows, fixture smoke, non-decisive) → 0.8262 @160/40 (3200 rows,
   calibration, non-decisive) → **0.8380 @640/160 (12800 rows, DECISIVE)**.
   The metric plateaus at ~0.83–0.84; the 0.85 band floor sits above the
   plateau, so no episode budget inside this protocol reaches the gate.

## Documented invocation (fixed before firing)

```
python experiments/apical_lms_stage0/run_stage0.py --cohort \
    --license-token retro-probe:PASS \
    --train-episodes 640 --test-episodes 160
```

Sizing rationale (delegated "at licensing time"; frozen before the one-shot):
(i) at/above the sanctioned 160/40 minimum; (ii) same 80/20 split convention
as the validated benchmark protocol; (iii) large enough to remove the README's
documented probe-starvation confounder so the verdict reflects the membrane,
not the probe budget. No frozen constant, shape, gate, or seed policy changed.
The size was NOT revisited after the outcome (one shot means one shot).

## Dual-layer gate readout (verbatim)

| layer | metric | value | threshold | result |
| --- | --- | --- | --- | --- |
| (a) probe | mean held-out probe R² (v_A → w_ref, split BY episode) | 0.837981 | ≥ 0.85 | **FAIL** |
| (b) hosting | d_task=10, d_apical=10 : decoded vs ref task R² | 0.485122 vs 0.392271 | decoded ≥ ref | HOLD (+0.092851) |
| (b) hosting | d_task=10, d_apical=32 | 0.231450 vs 0.172390 | decoded ≥ ref | HOLD (+0.059060) |
| (b) hosting | d_task=10, d_apical=128 | 0.068636 vs 0.049477 | decoded ≥ ref | HOLD (+0.019159) |
| (b) hosting | d_task=10, d_apical=384 (= primary config) | 0.023648 vs 0.016869 | decoded ≥ ref | HOLD (+0.006779) |

Gate conclusion = layer(a) AND layer(b) = FALSE AND TRUE = **FALSE**.

Width-sweep diagnostics (non-gating), all at 640/160, joint γ reading:

| d_apical = cell_count | γ | bound 2/(d+2) | spikes/token | probe R² | ref task R² | decoded task R² |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 1.667e−02 | 1.667e−01 | 7.6 | 0.8253 | 0.3923 | 0.4851 |
| 32 | 5.882e−03 | 5.882e−02 | 24.7 | 0.8341 | 0.1724 | 0.2314 |
| 128 | 1.538e−03 | 1.538e−02 | 93.3 | 0.8373 | 0.0495 | 0.0686 |
| 384 | 5.181e−04 | 5.181e−03 | 266.6 | 0.8380 | 0.0169 | 0.0236 |

## Interpretation (evidence notes; the verdict is not relitigated)

- What died: the *preregistered certification* that a fitted GLIF5 apical
  membrane hosts an online-LMS recursion at DendriCL-published-band probe
  fidelity. Measured best-case fidelity is 0.838 (cohort scale), −0.012 below
  the frozen floor, and the four-point budget curve shows a plateau, not a
  starved climb — the shortfall is structural within this protocol
  (single-frame features v_A(t); the membrane low-passes u_A, leaving a lag
  residual a memoryless linear map cannot remove).
- What did NOT die: membrane transparency itself — layer (b) held with
  positive margin at every operating point, replicating the calibration wave's
  finding at cohort scale. The γ-transcription hazard (frozen γ ≈ static
  recursion at high ref-probe fidelity) again illustrates why the two layers
  must be read together: here they DISAGREE, and the frozen gate resolves the
  disagreement against B'.
- Any future attempt to reach the band (multi-lag probe features, tuned
  constants, learned W_A/readout) is a NEW preregistration with a new cohort —
  explicitly out of scope here; this leg's cohort is spent. Recorded for the
  synthesis step-5 fork ruling: P-B race-bias and the path-2/path-3 probes are
  unaffected by this verdict.

## Files

| file | role |
| --- | --- |
| `run_corpus_leg.py` | license-guarded orchestrator: fires the documented CLI, dual-layer recompute, width sweep, verdict assembly |
| `corpus_leg_verdict.json` | terminal verdict artifact (both layers verbatim, cross-checks, diagnostics, provenance) |
| `results/cohort_primary_licensed_invocation.json` | layer-(a) artifact of record from the licensed CLI (rc=1) |
| `results/cohort_primary_dual_layer.json` | primary point, three-layer payload + determinism cross-check |
| `results/corpus_width{10,32,128}_d10_frozenjoint.json` | sweep legs |
| `results/*.log` | captured stdout/stderr of the licensed invocation |

Reproduce: `python experiments/apical_lms_stage0/corpus_leg/run_corpus_leg.py --license-token retro-probe:PASS`
(will refuse to overwrite existing artifacts implicitly — results are deterministic; delete deliberately to re-run).
Total wall 889 s, CPU-only, single-threaded NumPy replay through TwoCompartmentGLIF5.
