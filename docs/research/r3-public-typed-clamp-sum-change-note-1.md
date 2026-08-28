# Change Note 1 — clamp-plus-Sum capacity accounting

**Date:** 2026-08-28  
**Status:** post-run accounting correction; the one-shot capability verdict is unchanged

This note corrects only the explanatory capacity accounting in
`public_simple_swe_native_typed_clamp_sum_probe.json` (commit `3678054`). The
artifact is not rewritten and the model is not rerun.

The source correctly gave the circuit enough capacity, but its accounting
formula treated each eligible typed teacher call as two plan-store arrivals
*in addition to* one ordinary typed-state arrival. Native
`teach_typed_token` instead performs two physical learning steps total for an
eligible cue — transfer then exact — and advances the compact typed-hole state
in memory once. A non-eligible teacher call performs one physical learning
step. The formula also counted five plan-component arrivals after the new
any-bound cue was added; a two-bound session now commits six: parameter,
result, neutral, any-bound, lower, and upper.

For this frozen run the corrected physical learning-step count is:

- visible TRAIN prompt: 284;
- retained plan commits: 6;
- 20 eligible teacher calls at two steps each: 40;
- 4 non-eligible teacher calls at one step each: 4;
- total: **334**, matching the artifact's observed
  `learning_steps_after_train` exactly.

The corresponding allocation ceiling is `(334 - 1) × 11 = 3,663` segments.
The frozen source configured 3,872 segments, leaving 209 segments of harmless
headroom; the run actually ended with 3,515 live segments. Therefore no
capacity stop, truncation, retry, or outcome-dependent adjustment occurred.

Affected explanatory fields are
`retained_plan_component_arrivals` (5 should be 6) and
`total_learning_arrivals` (353 should be 334). The recorded
`exact_and_backoff_plan_store_arrivals=40` remains the correct number of
physical transfer-plus-exact stores, but those 40 must not be added again to
all 24 logical teacher calls.

The evidence boundary is unchanged: the single teacher-free `/001` attempt
produced the exact typed stream and compiled source, ordinary EOS, exact bound
copy traces, fitted-cell races, and a 4-regression + 32-property sandbox pass.
It remains same-family visible TRAIN follow-on evidence, not family-disjoint
transfer, M0, evaluation, or C1 evidence.
