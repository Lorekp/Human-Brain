# Probe P-B — predictive-state depolarizing bias into the first-spike race

**Date:** 2026-08-25 · Branch `codex/path2-pilot` @ e16d22c base · Preregistration
frozen before implementation: [preregistration.json](preregistration.json).
Runner: `run_probe.py --fixture` then `--cohort` (stop-on-failure between legs;
cohort spent exactly once). Verdict: **see §Verdict below** (filled from
[cohort_results.json](cohort_results.json)).

## What was tested

The TM-native mechanism from `docs/research/path-decision-synthesis.md`
§Probe P-B, verbatim: *small sustained somatic bias, proportional to
matched-segment count, on cells whose distal branch matched the recent winner
history, consumed by the existing basket-cell first-spike race*, with the
§2.3 constants (θ ≈ 13 connected of ~20 grown; negative-evidence decay ≈ 4% ×
increment). Never tested in this program before this probe.

## Lane-native instantiation (adopt principles/constants, not the simulator)

Substrate: `humanbrain.cortex.DendriticCortex` — Allen GLIF3 fitted parameters,
compiled unit-response first-spike race under basket inhibition, branches keyed
by phase-ordered presynaptic token-cell tuples.

| HTM concept | Lane implementation |
| --- | --- |
| segment | one grown branch (winner-history key → postsynaptic token cell) |
| recent winner history | actual recent race winners (teacher-forced context at eval; ground-truth replay during growth) |
| grown contacts (~20 cap) | contact budget cap = 20 per segment |
| connected contacts | `min(cap, floor(strength))`; each co-occurrence potentiates one contact (increment 1.0) |
| segment match | connected ≥ θ = 13 AND key equals a tail of the current history |
| somatic bias | CURRENT addition of 30 pA × matched-segment count (linear pre-first-spike ⇒ interchangeable with threshold offset up to scaling); consumed by the unchanged incumbent race code path |
| predictedSegmentDecrement ± | during TRAIN replay only: every θ-connected converged-but-lost segment ∓0.04 × increment per event (floor 0); `decrement_minus` (strengthen losers) is diagnostic only |

Bias lands only on cells already in the raced population: any θ-connected
matching branch already feeds the Witten-Bell drive, so the bias sharpens the
existing basket-cell competition rather than recruiting cells the incumbent
race would never run.

## Arms (frozen)

`off` (parity anchor) · `proportional` (PRIMARY) · `shuffled` control (same
per-race magnitude multiset, random distinct active assignment,
Generator(PCG64(11)) advanced once per biased race) · `decrement_plus`
(proportional + negative evidence) · `decrement_minus` (diagnostic).

Scoring: race-resolved only — the race winner IS the prediction; NO offline
heads anywhere. Teacher-forced span protocol mirrors
`tools/qa_baselines.answer_span_eval` verbatim (same wordization, same
per-position match counting). TRAIN-self vs VAL split reported at every stage
(the 38.3%-to-0% early warning). Frozen comparators: unigram 4.538341%,
majority-prior 4.068858%, trigram 16.275430% (informational).

## N semantics (recorded deviation note)

Prior legs sized a latent reservoir (race-pivot gate, APICAL-AND). This lane's
raced population IS the token-cell sheet, so N = pyramidal cell budget:
word-type cells (TRAIN-stream frequency rank, ties lexicographic) + 1 reserved
UNK cell. Fixture N=96, cohort N=1024. Restriction can only LOWER our ceiling
(VAL targets without a dedicated cell auto-miss), so gates passed under
restriction would be conservative. Coverage recorded per leg.

## Fixture leg N=96 (infrastructure only — PASS)

- Flag-off bit-parity: race records AND grown branch-state hash byte-identical
  between incumbent constructor path and flag-off probe path (400-context
  probe). Gate order honored: parity was red-tested FIRST
  (`tests/test_race_bias.py`, written before `src/humanbrain/race_bias.py`
  existed and confirmed failing on import).
- Double-run determinism: PASS.
- All five arms executed with finite telemetry; bias visibly accelerates first
  spikes (median winning spike 0.50 ms biased vs 11.85 ms flag-off).
- Full record: [fixture_results.json](fixture_results.json).

## Cohort leg N=1024 (the single decisive pass)

Coverage: 11,740 train word types → 1023 word cells + UNK; VAL answer-position
coverage 561/639 = 87.79%; prompt-token coverage 79.63%.

**Status:** the cohort completed (single decisive pass, never re-run). First
three finishers, logged verbatim mid-run:

| arm | VAL span_acc | TRAIN-self span_acc | median winning spike |
|---|---|---|---|
| off | 0.131455 (84/639) | 0.652862 (1403/2149) | 18.20 ms |
| proportional | **0.145540 (93/639)** | 0.388553 (835/2149) | 1.10 ms |
| shuffled | 0.035994 (23/639) | 0.069800 (150/2149) | 1.15 ms |

Mid-run reading (superseded by §Verdict below): `proportional` is above the shuffled
control by +10.95 points (unpaired 2-proportion z ≈ 6.8 vs 23/639) and above
both frozen comparators (unigram 4.54%, majority-prior 4.07%) by >3x; VAL
movement is real (14.55% > flag-off 13.15%), so this is NOT the
38.3%-to-0% memorization signature — bias REDUCED TRAIN-self (65.3% → 38.9%)
while raising VAL. The decrement± replay arms finished after this was written;
their numbers and the mechanical verdict (best primary arm vs shuffled via
paired McNemar on position flags, plus comparator gates) are recorded in
§Verdict below and in [cohort_results.json](cohort_results.json).

## Verdict — PASS

Filled verbatim from [cohort_results.json](cohort_results.json) (leg
`cohort_n1024`, wall clock 45,840.4 s, prereg sha256
`1fa29c014268bd1814b204ed26d3eb10e6e235891fea7f6ff10bdb51f7781abd`). Mechanical
mapping as coded in `run_probe.py`: best primary arm = argmax VAL span_acc over
the primary set {`proportional`, `decrement_plus`} → **`proportional`** (the
exact VAL tie with diagnostic-only `decrement_minus`, 0.145540, does not enter
the primary set); verdict rule
`pass = above_shuffled AND above_unigram AND above_majority_prior` → **true ⇒
COHORT VERDICT: PASS**.

### All five arms (race-resolved span accuracy)

| arm | VAL span_acc | TRAIN-self span_acc | median winning spike (VAL) |
|---|---|---|---|
| off (parity anchor) | 0.131455 (84/639) | 0.652862 (1403/2149) | 18.20 ms |
| **proportional (PRIMARY)** | **0.145540 (93/639)** | 0.388553 (835/2149) | 1.10 ms |
| shuffled (control) | 0.035994 (23/639) | 0.069800 (150/2149) | 1.15 ms |
| decrement_plus (primary candidate) | 0.139280 (89/639) | 0.292694 (629/2149) | 1.10 ms |
| decrement_minus (diagnostic) | 0.145540 (93/639) | 0.387622 (833/2149) | 1.10 ms |

### Paired McNemar, best arm vs shuffled

Per-position correctness flags, identical eval order (disclosure 3): arm-only
correct **b = 86**, control-only correct **c = 10**, discordant pairs
**n = 96**, stored statistic **z = (b − c)/√n = 7.756718** → two-sided
normal-approximation p ≈ **8.7×10⁻¹⁵** (exact binomial over the discordant
pairs: ≈ 3.2×10⁻¹⁶). The advantage over matched-magnitude random assignment is
decisive, and shuffled shares the speed profile (median winning spike 1.15 ms
vs 1.10 ms), so the effect is assignment-specific, not acceleration per se.

### Comparator gates (all required; all true)

| gate | best-arm VAL | comparator | margin | result |
|---|---|---|---|---|
| `above_shuffled_control` | 14.5540% | 3.5994% | +10.95 pts | true |
| `above_unigram_4.538` | 14.5540% | unigram 4.538341% | 3.21× | true |
| `above_majority_prior_4.069` | 14.5540% | majority-prior 4.068858% | 3.58× | true |

Trigram 16.275430% was informational/ungated.

### Caveats (honest reading)

- **Memorization component is real:** best-arm TRAIN-self 38.855% ≫ VAL
  14.554% (`train_self_to_val_ratio` = 0.374569). But the direction argues the
  bias is not mere memorization: vs flag-off it LOWERED TRAIN-self
  (65.286% → 38.855%) while RAISING VAL (13.146% → 14.554%) — the opposite of
  the 38.3%-to-0% failure signature.
- **Both decrement arms sit at/near proportional** (they run WITH the bias
  active, disclosure 4): `decrement_minus` ties it exactly on VAL (0.145540)
  and `decrement_plus` is 0.63 points under (0.139280). Negative-evidence
  replay at ∓0.04×increment changed little — the proportional somatic bias
  alone carries the effect.

## Files

- `src/humanbrain/race_bias.py` — PredictiveBiasConfig, matching/bias/decrement machinery
- `src/humanbrain/cortex.py` — minimal opt-in hook (predictive_bias kwarg, matching_segments, adjust_branch_strength; default-off byte-parity pinned by tests)
- `tests/test_race_bias.py` — 17 tests, flag-off bit-parity FIRST (red before implementation)
- `experiments/race_bias_probes/` — preregistration.json, run_probe.py, fixture/cohort results, FINDINGS.md

## Disclosures (all deliberate, none silent)

1. N reinterpreted as token-cell sheet budget (see above) — recorded pre-run in preregistration.json.
2. Somatic bias implemented as current addition, not threshold offset (equivalent pre-first-spike; documented in prereg).
3. Shuffled-control RNG consumed per biased race call during evaluation; the paired McNemar comparison uses a second identical-order eval pass, so both compared arms carry consistent pass-2 assignments; headline accuracies use the first pass.
4. Decrement-replay races run WITH proportional bias active (deployment-consistent: the system lives with its bias on).
5. `_grow_one_with_history` mirrors `grow_sequence`'s per-token body (sequence-relative API cannot take mid-stream continuations); parity tests pin both paths.
