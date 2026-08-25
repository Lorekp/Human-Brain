# Cell-Brain Program Map (the high-level picture)

Maintained by the orchestrator. Updated as waves complete. Everything here is
evidence-linked; design docs live on `codex/a013-cognition-design` unless noted.

## Goal ladder (bottom = done, top = end state)

```
R8  Beat Gemma-4-12B on chosen SWE bench          [comparison design in flight]
R7  Real SWE bench (contamination-audited)         [bench choice in flight]
R6  Leetcode-style self-posed tasks (simple_swe)   [machinery exists, unopened]
R5  Word prediction beats n-gram + tiny-LM         [cell-LM v1 protocol in flight]
R4  Scale ladder: capacity + transfer vs N         [protocol drafted, arms {PARAM}]
R3  Programs from cells, no memorization           [SEALED NEGATIVE at N=128]
R2  Speed: faster sim -> bigger brain              [GPU done; CPU in flight]
R1  Correct, auditable one-shot machinery          [DONE: correction 5 + audits]
```

## Sealed evidence base (immutable)

- a012 one-shot (N=128, ridge): 16/16 integrity gates passed; retention fixed
  by prompt-last framing (0.00007 -> 0.939); capacity 40/42 TRAIN replay;
  transfer 0/18 calibration; margins strong where committed (median 726 pA).
- Prior speed gates: 200k cells @144 tok/s CUDA (registered), 843 tok/s @98k
  native frontier at low activity density; ridge readout O(N^2)/O(N^3) —
  retired for large arms.
- GPU optimizer (codex/scale-opt-gpu, ca5c193): 497 tps @100k, 1M cells
  constructed+run @35 tps; full-touch roofline ~1M on this card.

## Root-cause thesis (drives everything)

Explicit token identities + monolithic reservoir + ridge readout = tasks
collide in feature space (state separation ~0.001) -> memorization without
transfer. Fix order: DISTRIBUTED CODES first (similarity becomes physical
overlap), then error-gated local learning + replay (dormant machinery exists),
then regions (E = distribution mechanism, P = processor), then scale arms.

## Live workstreams (this wave)

| Lane | Branch | Deliverable | Status |
|---|---|---|---|
| CPU/native optimizer | codex/scale-opt-cpu | throughput curve + optimizations, bit-exact | running (~5h) |
| Representation study | cognition-design | codebook recommendation + decisive experiment | running (~4.8h) |
| Regions design (re-dispatch) | cognition-design | E/P architecture + cost model to 1e9 | running |
| Cognitive-properties verify | cognition-design | verify/complete roadmap doc | running |
| Teaching matrix verify | cognition-design | verify/complete matrix doc | running |
| Cell-LM v1 protocol | cognition-design | first word-prediction protocol + dual baselines | just dispatched |
| Gemma comparison design | cognition-design | bench choice, harness, fairness, milestone ladder | just dispatched |

## Queue after wave lands

1. Cross-review all design docs (agents review each other's; parent-authored
   ones already flagged for verification — 2 of 2 covered).
2. Freeze a013 arm sizes from CPU optimizer's measured curve.
3. Implementation order (test-first): codebooks -> error-gated local rule +
   replay -> workspace broadcast -> cell-LM v1 at small N -> scale arms ->
   simple_swe opening -> bench harness -> first Gemma comparison.
4. Merge discipline: design docs merge to master only after cross-review;
   code merges only with exactness gates + red tests green.

## Standing risks

- Portal key instability (401/400 killed 4 agents; 2 recovered via
  re-dispatch, 2 via parent-authoring + verify agents).
- Single-GPU host: 1B-scale arms are sparse-residency native runtime only.
- Contamination: any bench task the cell system or Gemma has seen is void —
  comparison design must audit this before R7.
