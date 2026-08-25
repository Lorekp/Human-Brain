# T-ROUTE E2 verdict — RECOVERY_FAILS_SPARSITY_INSUFFICIENT

**Verdict: `RECOVERY_FAILS_SPARSITY_INSUFFICIENT`** — mechanical per Change
Note 3 (`experiments/t_route/e2/CHANGE_NOTE_3.md`, committed pre-spend at
`6ea2295`) C5: **no frozen config passes G-T1** on the who_said gating stratum.
The homeostat was indeed the annihilator of ABSOLUTE gold overlap — dropping it
lifts median overlap from arm A's exactly-0.0 to 74–81 bits (~7x randomized-W,
arm-B-scale) — but it does **not** recover ROUTING: the shuffled-question
control through the same weights achieves statistically indistinguishable
overlap (base/shuf ratio ≈ 1.00–1.06), so the `base_med >= 2*shuf_med`
conjunct fails on every config and both splits. The raw unnormalized W is not
a router; it is a static common-byte prior.

One-shot execution: 2026-08-25, branch `codex/a013-cognition-design`,
`run_e2.py` ran once (3.8 s wall) after the C4 precondition PASSED: all six E1
routing sets (`armA`, `armB`, `shuffled_question`, `randomized_W`,
`wrong_passage`, `shuffled_W`) were re-derived deterministically from the
frozen recipe and matched the hash-verified E1 log BIT-EXACT — E2 consumed the
same pairing data as E1's spend (278,754 events / 2,794 unique pairs). Gates
were evaluated by `score_e2.py` from the logged sets only. No reruns, no
threshold edits, no seed changes, no added k values.

## Gate results (verbatim)

G-T1 conjuncts per config, who_said stratum (eligible = GOLD_UNMATCHED-excluded,
matched flags from the verified E1 log); address_lookup / verse_continuation
have zero eligible items in both splits by corpus construction (reported as
`null`, consistent with E1):

| config | split | n | base_med | shuf_med | rand_med | base>0 | base>=2x_shuf | base>=1.5x_rand | G-T1 |
|---|---|---|---|---|---|---|---|---|---|
| arm_k8 | TRAIN-self | 60 | 80.0 | 80.0 | 11.0 | true | **false** | true | FAIL |
| arm_k8 | VAL | 20 | 74.0 | 71.0 | 10.5 | true | **false** | true | FAIL |
| arm_k16 | TRAIN-self | 60 | 81.0 | 81.0 | 11.0 | true | **false** | true | FAIL |
| arm_k16 | VAL | 20 | 74.5 | 70.5 | 10.5 | true | **false** | true | FAIL |
| arm_k32 | TRAIN-self | 60 | 80.0 | 80.0 | 11.0 | true | **false** | true | FAIL |
| arm_k32 | VAL | 20 | 74.0 | 70.5 | 10.5 | true | **false** | true | FAIL |

Enrichment vs randomized-W: 7.05x–7.36x across configs/splits — the arm-B-class
"~7x random" signature. Enrichment vs shuffled-question: 1.000–1.057 — no
question-specificity.

## What recovered and what did not

1. **E1's localization is CONFIRMED**: arm A's IDF-gain homeostat, not the
   Hebbian accumulation, destroyed gold-region drive. The identical trained
   accumulator with the transform replaced by identity (k=32, guarded inline to
   route bit-identically to the raw accumulator) jumps 0.0 → 80.0/74.0.
2. **Routing specificity does NOT come back.** The winner set barely depends on
   which question drives it (diagnostics below), because W's largest synapses
   sit on common-byte co-occurrences that EVERY natural-language question
   activates ("Who said", spaces, etc.). High gold overlap follows from gold
   regions being made of those same common bytes — a static prior, not
   retrieval.
3. **Sparsity neither helps nor hurts routing**: medians are flat across the
   sweep (train 80.0 → 81.0 → 80.0 for k=8→16→32; val 74.0 → 74.5 → 74.0).
   Keeping only the top-8 of 32 synapses per row loses nothing and gains
   nothing. Per the frozen C5 contrast: sparsity is a memory-footprint lever
   ONLY here, not a routing lever.

## Diagnostics (non-gated, computed read-only from the logged sets)

| quantity | train | val |
|---|---|---|
| distinct winner bits ever selected, arm_k8/k16/k32 | 178 / 178 / 179 | 170 / 168 / 168 |
| distinct winner bits, randomized_W control | 959 | 741 |
| median Jaccard(winners_i, winners_shuffled-q_i), k=8/16/32 | .809 / .803 / .809 | .778 / .784 / .790 |
| median Jaccard(winners_k32, winners_armB_i) | 0.369 | 0.362 |

The raw-W arms collapse onto a fixed ~170-bit pool (vs 741–959 for random-W)
that is >78% shared between DIFFERENT questions — the query-independent mirror
of arm A's rare-byte pool. It overlaps arm B's winner sets only partially
(~0.37): two different static priors with similar gold statistics.

## Versus arm B (frozen descriptive contrast)

No config beats arm B on VAL (best 74.5 vs 80.0); on TRAIN-self k16 edges it
81.0 vs 79.5 — within noise of a static prior. Headroom context: median gold
region sizes are 238 (train) / 212 (val) bits against K=128 winners, so all
these arms sit at roughly 55–63% of the achievable ceiling. Since arm B's
correlation signature is already measured twice as insufficient for extraction,
and raw-W carries the same class of signal with less support structure, nothing
here recommends raw W over arm B or vice versa: both encode byte-frequency
co-occurrence, neither encodes question-conditioned location.

## Mechanical outcome and campaign implication

Per-config G-T1: {k8: FAIL, k16: FAIL, k32: FAIL} → any-config disjunction
fails → `RECOVERY_FAILS_SPARSITY_INSUFFICIENT`. The E1 wound generalizes: it is
not "normalization destroys routing signal" but **no monotone transform of this
accumulator can produce question-specific winners**, because the accumulated
mass itself is dominated by frequency-driven co-occurrence shared across all
queries. Arm A dies from destroying that mass (IDF); raw W dies from it being
query-unspecific. Within this representation+rule class the next discriminator
must make the third factor (or eligibility) quote-specific rather than
frequency-driven — the escalation ladder (synthetic episodes, then global-scalar
RPE) remains as noted at D5, now with the additional constraint that any
candidate must demonstrably break the ~0.8 cross-question winner Jaccard.

## Pre-spend trail

- Change Note 3 committed BEFORE any E2 encoding (`6ea2295`).
- Harness + scorer committed before spend (`5595bee`) with `toy_check.py` ALL
  GREEN pre-spend: top-k unit rules (ascending-column tie-break, identity,
  clamp), bit-identical routing at k=fan_in, full scorer verdicts on synthetic
  logs vs independently recomputed expectations.

## Artifacts

- `experiments/t_route/e2/run_e2.py` — one-shot harness (integrity → rebuild →
  C4 bit-exact reproduction check → top-k arms → overwrite-refusing log).
- `experiments/t_route/e2/score_e2.py` — gates from logged sets only.
- `experiments/t_route/e2/e2_manifest.json` — array-level sha256 manifest
  (binary `e2_logged_winner_sets.npz` stays local, untracked).
- `experiments/t_route/e2/e2_results.json` — machine-readable gates + verdict.
