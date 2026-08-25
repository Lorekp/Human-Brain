# T-ROUTE E3 verdict — QUOTE_SPAN_GATE_FAILS

**Verdict: `QUOTE_SPAN_GATE_FAILS`** — mechanical per Change Note 4
(`experiments/t_route/e3/CHANGE_NOTE_4.md`, committed pre-spend at `933bb9c`):
the single frozen config `arm_q16` fails G-T1' on BOTH splits
(`base_med >= 2*shuf_med` fails at enrichment 1.000 / 1.007), so neither
verdict branch that requires G-T1' is reachable. The quote-span-gated third
factor does NOT turn the corpus-prior accumulator into a routing table.

Per the frozen C5 semantics and the program-state section of
`docs/research/t-route-e1e2-verdict.md`, this is the **program-grade
conclusion**: the last untested variant inside local-learning constraints has
now been spent and failed. Within this representation+rule class (byte-identity
SDRs, corpus-global token-pair Hebbian accumulation, kWinner readout),
Bible-QA needs either non-local optimization or an architecture whose
connectivity is trained by construction (the R3 episodic-store hybrid).

One-shot execution: 2026-08-25, branch `codex/a013-cognition-design`. Harness +
scorer + toys committed pre-spend (`f7ea8e3`) with `toy_check.py` ALL GREEN
(24 checks). `run_e3.py` ran once (6.0 s wall) after ALL C4 preconditions
PASSED: integrity pins verified; ungated counter equals the E1/E2 aggregate
exactly (278,754 events / 2,794 unique pairs); all six E1 routing sets AND all
seven E2 logged sets re-derived BIT-EXACT from the hash-verified logs.
Transparency note: the first invocation STOPPED in step 2b BEFORE any logging
(an implementation-time assertion demanded equal pair-event masses between the
quote-gated and random-span counters — mathematically wrong: equal span lengths
do not imply equal anchor-window coverage of masked positions). Nothing was
logged or spent by that invocation; the bogus assert was replaced by a
documented non-invariance note and the pipeline then completed once. No gates,
seeds, spans, or thresholds were touched.

## What was actually run

The gate kept plasticity ONLY for pairs whose postsynaptic passage position lies
inside an in-passage occurrence of the episode's verbatim quote: exactly the 60
TRAIN who_said episodes trained the gated arms (140 quote-less episodes
contributed zero events; VAL never touches W). Gated event mass: 41,595 pairs
(14.9% of ungated) across 1,364 unique pairs; random-span control 37,972 across
1,393. Post-transform: E2's top-k row sparsifier at the frozen E2-best k=16.
Routing readout unchanged (full-question population union → K=128 winners).

## Gate results (verbatim)

G-T1' conjuncts on who_said (eligible = GOLD_UNMATCHED-excluded, matched flags
from the verified E1 log); address_lookup / verse_continuation have zero
eligible items in both splits by corpus construction (`null`, consistent with
E1/E2):

| split | n | base_med | shuf_med | rand_med | rs16_med | shufq_rs16_med | base>0 | base>=2x_shuf | base>=1.5x_rand | spec_q | spec_rs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-self | 60 | 79.0 | 79.0 | 11.0 | 82.5 | 84.0 | true | **false** | true | 1.000 | 0.982 |
| VAL | 20 | 72.5 | 72.0 | 10.5 | 75.5 | 75.0 | true | **false** | true | 1.007 | 1.007 |

Descriptive baselines recomputed from the hash-verified logs on the same
eligible items — E2 `arm_k16`: 81.0 / 74.5; E1 `armB`: 79.5 / 80.0. Enrichment
vs randomized-W remains the arm-B-class ~7x signature (7.18x train / 6.90x val)
while enrichment vs shuffled-question is 1.000–1.007 — numerically identical to
E2's failure signature (1.000–1.057) despite removing 85% of the training mass
and confining it to quoted regions.

## The random-span control is the decisive reading

Random window placement (same inclusion set, same per-episode span lengths,
`default_rng(17)` positions) reproduces the gated arm's behavior EXACTLY:
spec_rs = 0.982 / 1.007 vs spec_q = 1.000 / 1.007 — and random spans even carry
slightly HIGHER absolute gold drive (82.5 / 75.5 vs 79.0 / 72.5). Placement of
the eligibility window carries no signal whatsoever: whether the surviving
synapses are those touching verbatim-quote positions or arbitrary ~37-byte
windows, the winner statistics are the same static prior. This kills the
hypothesis at its mechanism, not merely at a threshold: making the third factor
quote-specific in PLACEMENT does not make the accumulator quote-conditioned in
CONTENT.

## Why it cannot work here (mechanism, consistent with E1/E2)

Winners are passage-token-IDENTITY bits, and gold regions are also
token-identity bit sets; position never exists as an addressable variable in
this substrate. The pairing rule aggregates over positions before anything else,
so gating which positions contribute only rescales WHICH token-pair counts
survive — it cannot create synapses that answer "where". Diagnostics confirm the
E2 collapse verbatim: distinct winner bits ever selected = 176 / 171
(train / val) vs 959 / 741 for randomized-W; median cross-question winner
Jaccard = 0.842 / 0.842 — WORSE than E2's 0.78–0.81, i.e., more concentrated,
not less. Notably, an item's winner set equals its shuffled-question twin's set
only ~0.5% of the time — winners DO move per question, but only inside the same
fixed ~170-bit pool, so gold overlap (a property of which bytes are in the pool)
is invariant to which question drove the selection. Quote-specific eligibility
without position-addressable synapses redistributes mass within the prior; it
cannot build a routing table.

## Program state

Fifteen preregistered executions complete. The trained-mapping family is now
falsified in ALL THREE tested third-factor forms: broadcast/margin (E1 homeostat
arm), normalization-free raw mass (E2), and quote-specific eligibility (E3).
Per `t-route-e1e2-verdict.md`, the escalation ladder inside local learning is
EXHAUSTED as specified: Bible-QA requires non-local optimization or
trained-by-construction connectivity (R3 episodic-store hybrid). Any future
local-rule candidate must first demonstrate position-addressable synapses (e.g.,
per-neuron eligibility tags bound at encode time) — token-identity substrates
have been shown incapable of expressing the required function three times.

## Artifacts

- `experiments/t_route/e3/run_e3.py` — one-shot harness (integrity → rebuild →
  census/gated counters → C4 bit-exact reproduction of E1+E2 → gated arms →
  overwrite-refusing log).
- `experiments/t_route/e3/score_e3.py` — gates from logged sets only.
- `experiments/t_route/e3/toy_check.py` — 24 pre-spend unit checks, ALL GREEN.
- `experiments/t_route/e3/e3_manifest.json` — array-level sha256 manifest
  (binary `e3_logged_winner_sets.npz` stays local, untracked).
- `experiments/t_route/e3/e3_results.json` — machine-readable medians, conjuncts,
  specificity ratios, diagnostics, verdict.
