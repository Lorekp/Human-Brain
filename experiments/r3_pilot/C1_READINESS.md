# C1 readiness — R3 segment-growth cohort

**Status: HARD NO-GO — current runner is diagnostic-only; C1 cohort unspent.**

The independent prelaunch audit found that the merged runner cannot produce a
protocol-valid C1 verdict. Do not launch `run_c1_cohort.py` as a cohort.
Authoritative analysis and repair gates:
[r3-c1-prelaunch-audit.md](../../docs/research/r3-c1-prelaunch-audit.md).

## Why launch is blocked

1. **Wrong scored substrate.** The frozen protocol requires GLIF3 distal
   segment matches to create plateau current and a fitted-cell/basket
   first-spike race to emit recognition. The runner instead scores integer
   threshold results from a Python `SegmentGrowthTracker`.
2. **Logical pools are not simulated cells.** Only the 131,072-cell GLIF5 lane
   is simulated. The advertised extra 49,152 pool IDs have no membrane state
   and do not spike.
3. **Epoch 0 grows state.** A1/A3 will predict during the sweep that must stay
   at zero segments.
4. **The production spike preflight is bypassed** because it uses the arm-global
   counter after epoch 0.
5. **Held-out evaluation learns online.** The final sweep allocates,
   reinforces, and punishes while HELD-OUT is being scored.
6. **Four scientific choices were changed without a formal amendment:** winner
   source, A3 sampling, the permanence update, and cross-book/pass state.
7. **Full checkpoint/resume equality and contract tripwires are not proven.**
8. **The 15-hour arithmetic and live projection are wrong.**
9. **CN-1's dense activity is not a viable coincidence code.** At its measured
    ~56.9% occupancy, random 20-contact segments cross θm with probability
    ~0.80 and θa with probability ~0.31.

## What is valid

- Corpus hash and 66/49/17 book split.
- Frozen tokenizer reconstruction and pool-feasibility arithmetic.
- Tokenizer §5 overlap calibration: deterministic artifact and 17 related
  tests pass; concept-stack 20-contact accidental tails are
  P(X≥10)=4.162e-10 and P(X≥13)=1.786e-15.
- P-B verdict versus its frozen shuffled/unigram/majority controls.
- Generic segment-growth unit mechanics and deterministic reruns.
- CN-1 rheobase/recruitment measurements (not its dense C1 drive choice).
- The generic observation-sink/CUDA mirror, restore-time sink attachment, and
  live dynamics capture fixes committed on master as `f721fa9`.
- Native mechanism fixtures now materialize the temporal pool cells and execute
  segment-driven fitted-cell/basket/GABA races, including deterministic
  multiple-owner arbitration. The nonofficial synthetic-authority q=255
  development diagnostic rejects seed-17 `(10,13)` at production exposure
  1,597 (`matching=11`, `active=0`, `allocated=0`). The next candidate,
  `(16,18)`, passes production-overlap and target-disjoint at development seeds
  3, 7, 17, and 29: all 2,041 learned-source recurrences select their stored
  targets and all 128 disjoint novel probes per seed remain at zero matches and
  predictions. Exact identities and counts are in
  [q255_physical_development_smoke.json](q255_physical_development_smoke.json).
  This is not an official threshold selection, confirmation, or R3-F0 result.

The first canonical TRAIN-order precursor then falsifies `(16,18)` for corpus
use much earlier than the synthetic screen could expose. A production-codebook,
seed-17 LEARN replay stops inside Genesis at zero-based token 19,841 with 11
matching segments, zero active segments, and zero replacement allocation. The
19,842-step trustworthy prefix ran at 2,530.74 steps/s. This is a localized
context-collision STOP, not a two-book scaling result. Both higher threshold
arms then fail q=255 production-overlap at every development seed: `(18,20)`
retains only 18 physical contacts and `(20,20)` does not match after one of 11
source winner/relay pairs changes on replay. Threshold-only repair is exhausted;
the next repair must stabilize context-conditioned winners under network growth.
Exact source identities and the TRAIN STOP record are in
[r3_native_train_multibook_cost.json](../../benchmarks/r3_native_train_multibook_cost.json).

A separate public synthetic TRAIN smoke now exposes the native physical output
race through Python. After learning `0 -> 1`, cue `0` arrives alone and the
fitted owner/output/basket races emit token `1` from 11 physical inputs with no
candidate token, score, or Python argmax. This closes the output transport seam,
not generation, generalization, coding, or C1 capability.

The physical TRAIN-prefix cost ladder now completes the entire 122,253-token
Genesis prefix with 1,047,305 live segments. At that endpoint the instrumented
loop measures 1,063.72 steps/s, a straight-line 8.65 h projection for the
corrected three-arm schedule (1.73x the required rate), down from 3,291.70
steps/s at 4,096 tokens. This is still synthetic, pre-outcome prefix evidence:
it excludes later-book growth, construction, checkpoint/I/O, and final arm
topology. The exact ladder is in
[r3_native_train_prefix_scaling.json](../../benchmarks/r3_native_train_prefix_scaling.json),
so it narrows but does not close `C1-COST-DENSE-4096` or authorize C1.

The lean full-TRAIN development/cost runner subsequently completed all 49
TRAIN books (2,117,412 token steps) at `(16,18)`, seed 17, without a native
scientific STOP or held-out materialization. It ended at 23,286,478 live
segments with exact native/tracked accounting and physical winner/basket/GABA
causality. This is not an R3-F0 structural pass: the frozen distinct-key model
expects 23,286,406 segments, and the 72-segment excess decomposes into a
seven-repeat net shortfall from full 11-column reuse, with five partial
one-column predictions. The aggregate artifact cannot distinguish
punishment-induced deactivation from winner-key divergence. Concurrent native
work after about book 12 also contaminates its recorded 291.54 steps/s, so the
timing is not cap evidence. Exact evidence is in
[r3_native_train_full_two_tier_theta16_18_seed17.json](../../benchmarks/r3_native_train_full_two_tier_theta16_18_seed17.json).

The nested Gate-3 source closure is stale after the native repairs. A dry-run
refresh changes its bundle SHA-256 from `7830b2fe...` to `244538ac...`; the
draft check still passes against its historical committed blobs and the
runnable check still fails closed. Refreeze is deliberately deferred until the
official q=255/R3-F0 execution closure is complete, so this status update does
not create a source-current execution authority.

The sparse recognition prototype is control-only evidence. It benchmarks
`|W|=54`; frozen g=50 produces roughly 64k winners on the initial slice and
74.6k over the 4,096-step confirmation.

## Correct schedule

```text
epoch-0 TRAIN
+ 3 exposure TRAIN
+ final TRAIN
+ final HELD-OUT
= 5 * 2,117,412 + 458,191
= 11,045,251 steps/arm
= 33,135,753 steps for three serialized arms
```

The prior 8,927,839-step/arm projection omitted one full TRAIN sweep. No
measured current end-to-end mode yet fits the 15-hour cap; the full-TRAIN run
above was timing-contaminated and remains narrower than an end-to-end cap
measurement.

## Required sequence before GO

- [ ] **R3-F0 neuronal fixture:** physical pool cells, segment-triggered plateau
      drive, fitted-cell first-spike race, growth-off structural zero, and
      repeated/novel/punishment/bit-exact fixtures. Native mechanism slices are
      green; q=255 development's first synthetic passing candidate `(16,18)`
      fails the original canonical TRAIN-order precursor at Genesis token
      19,841. The repaired two-tier cost runner completes all 49 TRAIN books but
      exceeds the exact R3-F0 segment count by 72, while both higher arms fail
      q=255. Threshold-only repair is exhausted; stabilize replayed winner
      identities before confirmation or official R3-F0.
- [ ] **Sparse context assembly gate:** freeze and compare k-capped actual
      recurrent spikes; require repeated-context stability, unrelated-context
      separation, and a bounded coincidence false-positive rate. Supersede the
      dense CN-1 C1 drive/preflight choice.
- [ ] **R3-F1 scientific controls:** disposable epoch-0 shadows, exposure-local
      preflight, frozen read-only evaluation, frozen boundary reset policy,
      mass-matched A3, and uninterrupted/resumed deep-state equality.
- [x] **Tokenizer §5 artifact:** constants-only hypergeometric concept-vs-byte
      calibration committed and source-current
      (`token_overlap_calibration.json`).
- [ ] **C1-COST-DENSE-4096:** actual production activity masks, dense segment
      matching, physical plateau/race cost, checkpoint/I/O cost, and conservative
      projection below 13.5 h. The physical full-Genesis prefix is green at a
      narrow 8.65 h projection, but later growth, required costs, and final
      topology remain unmeasured.
- [ ] Refresh the nested Gate-3 and outer R3 source bindings once the official
      q=255/R3-F0 execution closure is final; keep `runnable=false` until then.
- [ ] **CN-2:** formally refreeze substrate, winner source, A3, permanence rule,
      boundaries, evaluation, runtime changes, schedule, and cap behavior.
- [ ] Corrected dry run and short smoke pass.
- [ ] Explicit user GO.

## Permitted commands

Focused unit tests and short pre-outcome fixtures are permitted. The old dry
run may be used only to reproduce and localize the documented failures; its
`ok` field is not readiness evidence.

The following long command remains prohibited:

```powershell
python experiments/r3_pilot/run_c1_cohort.py --output-dir experiments/r3_pilot/c1_run
```

No C2 work opens until a valid C1 pass.
