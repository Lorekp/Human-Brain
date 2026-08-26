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
measured current end-to-end mode fits the 15-hour cap.

## Required sequence before GO

- [ ] **R3-F0 neuronal fixture:** physical pool cells, segment-triggered plateau
      drive, fitted-cell first-spike race, growth-off structural zero, and
      repeated/novel/punishment/bit-exact fixtures.
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
      projection below 13.5 h.
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
