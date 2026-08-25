# T-ROUTE E1 verdict — NEGATIVE_CORE_WOUND (D5)

**Verdict: `NEGATIVE_CORE_WOUND`** — mechanical per
`docs/research/trained-inter-space-preregistration.md` §7: **G-T1 FAILED on
both splits** (gating stratum `who_said`, arm A). Per §6 stop-on-failure the
trained-inter-space hypothesis takes its core wound: locally-trained
projections carry **no more cross-space structure than raw fixed-random
overlap** — strictly less (0 vs 11 bits). T-ROUTE and the correlational
direction die together at the shared root. The G-T2 pipeline leg was **NOT
spent** (§6.2). Escalation options remaining within the class, noted per §7:
synthetic-episode scaling first, then a global-scalar RPE third factor.

One-shot execution: 2026-08-25, branch `codex/a013-cognition-design`, run once
(`run_e1.py` -> `logged_winner_sets.npz` + `e1_manifest.json`; `score_e1.py`
evaluated gates **from the logged sets only**, then applied §6 stop rules).
No reruns, no threshold edits, no seed changes.

## Integrity (step 0 — all pins verified before spend)

| pin | value | status |
|---|---|---|
| freeze candidate raw sha256 | `361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c` | OK |
| TRAIN manifest raw / canonical | `d82fe615…0a0e8` / `d256c27d…9a7b` | OK |
| VAL manifest raw / canonical | `4f9be9b7…37dec` / `2771de48…55bec` | OK |
| tokenizer vocab == 256 | 256 | OK |
| counts 200 / 60 | 200 / 60 | OK |

Pairing events total **278,754** over **2,794** unique token pairs (one-shot,
TRAIN only, O_MAX=8 anchors x H_WIN=12 window). Families: train who_said 60 /
address_lookup 70 / verse_continuation 70; val 20/20/20.

## GOLD_UNMATCHED census (positional, never judged)

Answer token sequence has a verbatim contiguous match in its own passage
stream for **who_said only**: train matched 60/200 (unmatched 140 = address 70
+ verse 70); val matched 20/60 (unmatched 40 = address 20 + verse 20).
Excluded from every G-T1/G-T3 denominator per §4.

## Gate results (verbatim)

### G-T1 routing enrichment — **FAIL** (decisive)

Median overlap |winners ∩ gold_bits|, who_said stratum, eligible items:

| split | n | base_med (arm A) | shuf_med | rand_med | base>0 | base>=2x_shuf | base>=1.5x_rand |
|---|---|---|---|---|---|---|---|
| TRAIN-self | 60 | **0.0** | 0.0 | 11.0 | false | false | false |
| VAL | 20 | **0.0** | 0.0 | 10.5 | false | false | false |

Gate requires BOTH splits to pass all three conjuncts — failed on both.
Reported strata address_lookup / verse_continuation have zero eligible items
in both splits (empty denominators; reported as `null`, not scored).

Diagnostic contrast (same logged sets, same eligibility): **arm B**
(dense unnormalized correlation) median overlap **79.5 (train) / 80.0 (val)**
— ~7x randomized-W — while arm A sits at exactly 0. The correlational
soft-binding signature retains common-token mass that arm A's homeostat
destroys (below).

### G-T3 anti-leak

- **G-T3b shuffled-W PASS**: rand_med 11.0; shufw_med 12.0 inside band
  [7.333…, 16.5] (= rand/1.5 .. rand*1.5), who_said pooled splits. Topology
  and value multiset are not the problem — the *trained values under IDF gain*
  are.
- **G-T3a wrong-passage UNDEFINED (denominator empty)**: 0 eligible pairs.
  By corpus construction sigma(i) = (i + len(split)//2) maps every who_said
  item onto address_lookup/verse_continuation items, whose answers never match
  their passages (see census) -> every pair is positionally excluded per §4.
  Recorded as `pass=false, n_eligible=0` in e1_results.json. This cannot flip
  the D5 outcome: §7 keys off G-T1 first, and G-T1 failed outright.

### G-T2 full pipeline — **NOT SPENT** (§6.2 stop-on-failure before T1 pass)

## Localization (why arm A routes at exactly zero)

Post-hoc inspection of the logged artifact + recomputation of the frozen
quantities (no protocol change):

1. Byte-level world: vocab is bytes; gold answer regions are made almost
   entirely of high-frequency bytes (space 42, s, d, e, a, o, l, i, h in the
   first 20 train gold regions).
2. Homeostatic IDF gain `ln((N_pass+1)/(df[b]+1))` puts common-byte rows near
   the floor (space/comma/e/t: df 125–130 -> idf ~0.43–0.47) and rare-byte
   rows (incl. non-ASCII UTF-8 punctuation) near the ceiling (idf up to 5.30)
   — a ~12x relative drive advantage for rare-byte rows.
3. Arm A winners therefore collapse onto a fixed pool of rare-byte SDR bits:
   top winner bits appear in ALL 200 train routings; only 554 distinct bits
   are ever selected across 200x128 selections; these rows are disjoint from
   gold regions by construction -> overlap exactly 0 on VAL, max 8 on TRAIN.
4. Controls through identical machinery behave normally (randomized-W ~ chance
   10.5–11.0; shuffled-W 10–13), so plumbing/orientation is sound — this is a
   real property of sparse trained W + IDF row gain, not an artifact.
5. Arm B's strong diagnostic enrichment comes precisely from what it does NOT
   do: no normalization of any kind keeps common-byte mass routable. Its
   correlation content, however, is exactly the signature already measured
   twice as insufficient for extraction (strategy ranking #2).

## Comparators (UNCHANGED, verbatim)

- unigram span-acc 0.04538341158059468 (29/639 = 4.54%)
- majority-prior-within-answer-vocab 0.0406885758998435 (26/639 = 4.07%)
- random-ranking-seed11 0.001564945226917058 (1/639)

## Pre-spend amendment trail

- Change Note 1 (committed pre-E1 at 1e6ed89): vacuous scale-then-normalize
  order fixed to `W'[b,:] = idf(b) * W[b,:] / ||W[b,:]||_2`.
- Change Note 2 (bbe4dac, PRE-SPEND — no episode encoded when written):
  `shuffled_values_copy` subtracted `_delta` after permuting, breaking the §5
  value-multiset contract for trained accumulators; fixed directly
  (`out._values = permuted values`), red-first regression test added
  (`test_shuffle_values_preserves_multiset_on_trained_accumulator`). Control
  machinery only; no arms/gates/seeds/thresholds/readout changes.

## Artifacts

- `experiments/t_route/run_e1.py` — D4 logging harness (integrity -> frozen
  tokenizer/projector -> one-shot aggregated dW via equivalence-tested
  `apply_dense_delta` -> homeostat -> arm B -> controls -> observation-sink
  winner sets).
- `experiments/t_route/score_e1.py` — gates from logged sets only; §6 gate
  order and stop rules; conditional G-T2 leg fed by logged hop-1 sets +
  UNCHANGED `probes.py SpanScorer`.
- `experiments/t_route/e1_manifest.json` — array-level sha256 manifest
  (binary `logged_winner_sets.npz` stays local, untracked).
- `experiments/t_route/e1_results.json` — full machine-readable results +
  verdict.
