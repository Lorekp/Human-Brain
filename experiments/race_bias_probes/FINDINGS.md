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

**Status at commit time:** the cohort is IN FLIGHT (single decisive pass,
never re-run). Completed arms (logged verbatim from the runner):

| arm | VAL span_acc | TRAIN-self span_acc | median winning spike |
|---|---|---|---|
| off | 0.131455 (84/639) | 0.652862 (1403/2149) | 18.20 ms |
| proportional | **0.145540 (93/639)** | 0.388553 (835/2149) | 1.10 ms |
| shuffled | 0.035994 (23/639) | 0.069800 (150/2149) | 1.15 ms |

Interim reading (NOT yet the verdict): `proportional` is above the shuffled
control by +10.95 points (unpaired 2-proportion z ≈ 6.8 vs 23/639) and above
both frozen comparators (unigram 4.54%, majority-prior 4.07%) by >3x; VAL
movement is real (14.55% > flag-off 13.15%), so this is NOT the
38.3%-to-0% memorization signature — bias REDUCED TRAIN-self (65.3% → 38.9%)
while raising VAL. The decrement± replay arms are still running; the
mechanical verdict (best primary arm vs shuffled via paired McNemar on
position flags, plus comparator gates) lands in
[cohort_results.json](cohort_results.json) + a follow-up commit when the
cohort exits.

<!-- COHORT_RESULTS -->

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
