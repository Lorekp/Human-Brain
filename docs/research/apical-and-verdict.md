# APICAL-AND week-1 verdict — NEGATIVE-FALSIFIED at G-A1 (2026-08-25)

Status: executed one-shot under `docs/research/apical-and-preregistration.md`
(frozen @ 05fb3ef, committed BEFORE implementation). Cohort spent exactly once
at N=1024. Flag-off parity gate PASSED before any flag-on run. All numbers
below are verbatim from the committed artifacts.

## Verbatim terminal numbers (18 arms x 60 VAL pairs, N=1024)

### G-A1 conjunction index — FAIL in all 18 arms (gate: median CI >= 1.5)

| Arm            | median CI | mean CI | frac pairs CI>1 | pairs scored |
|----------------|-----------|---------|-----------------|--------------|
| A50_0.75_0.6   | **1.0290** (best) | 1.0305 | 1.000 | 60 |
| A100_0.75_0.6  | 1.0284    | 1.0295 | 1.000 | 60 |
| A200_0.75_0.6  | 1.0261    | 1.0282 | 1.000 | 60 |
| A200_0.5_0.6   | 1.0253    | 1.0264 | 1.000 | 60 |
| A100_0.5_0.6   | 1.0247    | 1.0271 | 1.000 | 60 |
| A50_0.5_0.6    | 1.0247    | 1.0274 | 1.000 | 60 |
| A200_0.25_0.6  | 1.0184    | 1.0212 | 1.000 | 60 |
| A50_0.25_0.6   | 1.0180    | 1.0208 | 1.000 | 60 |
| A100_0.25_0.6  | 1.0178    | 1.0207 | 1.000 | 60 |
| A200_0.75_0.3  | 0.9992    | 0.9993 | 0.417 | 60 |
| A100_0.75_0.3  | 0.9988    | 0.9994 | 0.417 | 60 |
| A50_0.75_0.3   | 0.9988    | 0.9994 | 0.400 | 60 |
| A200_0.5_0.3   | 0.9968    | 0.9977 | 0.267 | 60 |
| A50_0.5_0.3    | 0.9968    | 0.9978 | 0.317 | 60 |
| A100_0.5_0.3   | 0.9967    | 0.9977 | 0.283 | 60 |
| A200_0.25_0.3  | 0.9943    | 0.9948 | 0.133 | 60 |
| A50_0.25_0.3   | 0.9932    | 0.9946 | 0.133 | 60 |
| A100_0.25_0.3  | 0.9932    | 0.9942 | 0.117 | 60 |

Zero pairs excluded (no both-silent cases). Window means for the median arm
A100_0.25_0.6: CT = 13650.4, T = 13376.2, C = 0.00 somatic spikes per 150 ms
window; context-phase spikes per 300 ms rep = 0.0 (the open gate alone never
fires the soma; noise sigma 50 pA stays sub-threshold).

Reading: the mechanism works mechanically (C = 0 vs CT >> 0 proves responses
require tokens; CI > 1 in every pair at g_gate = 0.6 shows the gate adds a
real, consistent gain) but the gain is ~2-3%, not the required >= 50%. The
burst response is RECRUITMENT-limited, not threshold-limited: recurrent
cascade currents (~650 pA edges) dominate over afferent threshold crossings,
so multiplicatively lowering the soma threshold barely changes how many cells
the cascade recruits.

### G-A2 separation at decision points (R0-G1) — FAIL in all 18 arms

Pooled median 1 - cosine over VAL trace-bank vectors (570 within-family pairs
per arm; strata who_said / address_lookup / verse_continuation, 20 each):

- range across arms: **0.000976 (A200_0.75_0.3) to 0.002182 (A200_0.25_0.6)**;
- gate demanded > 0.10; missed by ~50x. Decision-point states are
  near-collinear even WITH apical context drive active.

### G-A3 decode z >= 2 vs unigram — FAIL in all 18 arms

Two-sigma threshold (frozen): 0.061851478566709396 (= unigram 29/639 +
2 sigma, sigma = 0.008234). Comparators: unigram 0.04538341158059468,
trigram 0.162754303599374, majority-prior-within-answer-vocab
0.0406885758998435.

- Best decoder per arm (logistic or MLP over {raw voltage+spikes, trace bank},
  TRAIN-answer-vocab head): range **0.01564945226917058 (A100_0.75_0.3) to
  0.0297339593114241 (A100_0.25_0.3)**.
- Program-wide best single config: A100_0.25_0.3 / raw_voltage_spikes / MLP =
  **0.0297339593114241** (19/639) - below the majority-prior control
  (26/639) and below unigram.
- Shuffled-label MLP controls span 0.009389671361502348 - 0.03129890453834116:
  learned decoders sit inside the shuffle band. Conditioning on the state is
  worth less than ignoring it.

## Mechanical verdict

Per preregistration §5:

- Per-arm PASS requires G-A1 AND G-A2 AND G-A3 -> **0/18 arms pass**;
- PROGRAM VERDICT: **NEGATIVE-FALSIFIED**;
- Failure localizes at G-A1 itself: the pre-declared falsifier clause
  ("AND-index pass + decode below majority prior") did NOT fire because no arm
  achieved conjunction pass - the coincidence amplifier fails as a cell-level
  mechanism at these dynamics, upstream of any readout question;
- Per stop-on-failure, no further legs are owed. The G-A2/G-A3 legs were run
  anyway as localization diagnostics and fail independently by wide margins.

## What this falsifies and what survives

FALSIFIED: multiplicative plateau-gated threshold lowering on the fitted human
GLIF5 soma does not produce super-linear context-token coincidence gain in the
regime where the token response is dominated by recurrent recruitment. The
apical route AS SPECIFIED (tau_a integrator + plateau + multiplicative gate)
is dead for QA reading on this evidence base, joining R0/R0-b/R0-c/race-pivot.

SURVIVES: the two-compartment machinery itself (`TwoCompartmentGLIF5`,
bit-exact flag-off parity vs native kernel proven at N=96 AND N=1024 on real
corpus text - 192 tokens / 9337 and 105053 native spikes), the item-parallel
CPU harness with snapshot/restore branch trials, and the observation that C=0
/ CT>>0 demonstrates genuine apical modulation of stimulus responses. If
Direction A's U-STATE union vector (their week-1 gates passed) ever needs a
nonlinear per-cell encoder, the substrate exists - but the "AND creates
identity" hypothesis is dead as tested.

## Disclosures and deviations (all deliberate, none silent)

1. Burst delivery operationalization: preregistration froze "burst window
   50 ms" while token delivery at the ladder cadence (5 ms/token) needs up to
   150 ms for 30 tokens. Resolution: bursts capped at MAX_BURST_TOKENS=30 so
   delivery fits inside the frozen 150 ms RESPONSE window; each token pulses
   its afferents for the full fitted 1.5 ms (30 substeps), identical to native
   observe(). An initial fixture run with one-substep pulses produced zero
   spikes everywhere (charge 50x too small) and was fixed BEFORE the cohort -
   fixture numbers gated nothing (prereg §6.1).
2. The driver ran the QA replay leg after G-A1 had terminally failed (stop-on-
   failure says no further legs are OWED; running them as disclosed
   localization diagnostics wastes nothing and was completed).
3. First pooled decode attempt wedged when externally-killed workers lost
   queued tasks (classic Pool behavior); replaced with isolated per-arm
   subprocesses + JSON checkpoints + bounded retry rounds. All 18 arms
   completed under the new runner; no numbers changed hands silently.
4. U-STATE coordination: their D5 FINDINGS.md did not exist at execution
   start (manifest records `ustate_findings_present=false`), so the frozen
   FALLBACK context summary applied. Their capacity-precondition commit
   (a139945) landed mid-run; per one-shot discipline the cohort was not
   restarted on the union drive.
5. Environment note: tests/test_large_scale_capability freshness preflight
   fails on cells.py because a SPENT protocol's source pin predates commits
   b6abaf1-era edits (expected 19934 bytes vs 21020 at HEAD before this work);
   pre-existing drift, not caused by this task; pin not touched per
   spent-protocol discipline.

## Files

- `src/humanbrain/cells.py` - TwoCompartmentGLIF5 (arm-batched, CPU numpy,
  bit-exact flag-off); `tests/test_two_compartment_glif5.py` (9 tests, RED
  first).
- `experiments/apical_probes/two_comp_harness.py` - ApicalTokenRunner (numpy
  token loop, bit-exact vs native serial kernel) + ContextSummarizer.
- `experiments/apical_probes/run_apical_probes.py` - parity gate + G-A1
  battery + decision-point replay driver (item-parallel subprocesses).
- `experiments/apical_probes/probes.py` + `probes_decode_worker.py` -
  G-A2/G-A3 evaluation with per-arm checkpoints; `decode_cache/*.json`.
- Artifacts committed: `parity_check_{fixture,cohort}.json`,
  `conjunction_results_{fixture,cohort}.json`,
  `logged_states_apical_{fixture,cohort}_manifest.json`, `probes_results.json`.
  Local-only: `logged_states_apical_cohort.npz` (~array hashes recorded in its
  manifest).

## Bottom line for the roadmap

APICAL-AND (Direction B) terminates NEGATIVE-FALSIFIED at the first gate: the
plateau-gate AND produces ~1.03x somatic amplification against a 1.5x
requirement, states remain collapsed (G-A2 ~0.0015 vs 0.10 required), and
decoders score below the no-state controls (G-A3 best 2.97% vs 4.07% majority
prior). With Direction A (U-STATE) passing its separation gates the same week,
the program's live route is explicit input-side structure (A), not dendritic
coincidence generation (B). Recommended disposition per custom-architecture-
directions.md ranking: elevate A's integration leg; archive B's negative with
the localized mechanism finding (recruitment-limited regime defeats threshold
multiplication).
