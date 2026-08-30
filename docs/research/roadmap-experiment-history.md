# Roadmap experiment history and retired directions

Status: historical companion to [`roadmap-reading-to-coding.md`](roadmap-reading-to-coding.md).

This document owns the chronology, failure localization, and retired-mechanism
history that used to make the main roadmap grow with every experiment wave.
The roadmap should link here rather than reproduce this diary.

For the current evidence table and exact claim boundaries, see
[`../benchmark-status.md`](../benchmark-status.md). For the implementation-level
program map, see [`../program-map.md`](../program-map.md).

## 1. Reservoir / readout era — closed

The original reservoir-computing family was systematically tested rather than
quietly abandoned.

- Multi-timescale eligibility traces and follow-on state-feature families failed
  to create useful context separation at the required capacity.
- Race/broadcast improved output diversity without producing correct QA.
- U-STATE/SDR storage demonstrated that separation and storage could succeed
  while generalization to unseen question-context pairings remained zero.
- APICAL-AND and other fixed-connectivity representation changes did not close
  the missing mapping between question space and content space.

The durable conclusion is that **capacity and separation are not sufficient**.
A useful learner needs learned relational/long-range binding between different
content spaces. Retuning presentation schedules, codebook geometry,
replay-prioritization, EI fraction, or old ridge/readout constants is not
licensed.

Key references:

- [`r0_multitrace/`](../../experiments/r0_multitrace/)
- [`r0b_state_features/`](../../experiments/r0b_state_features/)
- [`race-pivot-gate-verdict.md`](race-pivot-gate-verdict.md)
- [`custom-architecture-directions.md`](custom-architecture-directions.md)

## 2. Path-2 reading / persistent identity

The reading lane moved from an invalid monolithic identity criterion toward a
factorized persistent-successor criterion.

Important historical boundaries:

- calibration-v3 exposed candidate/matching versus active-path distinctions;
- a long public-TRAIN replay authenticated roughly 976k exact prefix steps before
  a later allocation criterion failed;
- the failure was reclassified when a repeated source sharing one output column
  with a novel successor showed that native identity is factorized by output
  column rather than `11 * novel_keys`;
- a target-independent fixture now pins stable complete tuples, stable exact-
  source owners per output column, and terminal storage equal to actual
  allocation sum.

The completed dense TRAIN build reached all 49 books / 2,117,412 tokens, but it
is not itself a capability verdict. The currently licensed adjudication is the
fresh source-bound factorized replay from empty native and scorer state:

- [`run_p2_factorized_successor_train49_replay_v1.py`](../../experiments/r3_pilot/run_p2_factorized_successor_train49_replay_v1.py)

Protected held-out/C1 material remains unopened.

## 3. Coding: whole-sequence continuation → action learning

The early coding bridge established useful plumbing but not general synthesis.

- PAC probes showed execute → observe → route → build loops on narrow TRAIN
  templates.
- The frozen fresh four-task coding cohort produced valid source but collapsed
  onto two learned whole programs: `STOP_FRESH_CODING_CHALLENGE`.
- Static prompt-to-production decoders could replay teachers after representation
  repair but still failed unseen composition.
- Blind edit search and the first persistent sensorimotor actor were closed after
  honest STOPs.
- Reward-modulated action learning localized magnitude aliasing; exact signed-
  magnitude representation then produced 10/10 atomic actions.

These experiments retired whole-program retrieval and one-step greedy editing as
credible general mechanisms.

Representative artifacts:

- [`fresh_coding_challenge.json`](../../experiments/r3_pilot/fresh_coding_challenge.json)
- [`direct_production_cell_composition_probe.json`](../../experiments/r3_pilot/direct_production_cell_composition_probe.json)
- [`direct_production_cell_role_gated_probe.json`](../../experiments/r3_pilot/direct_production_cell_role_gated_probe.json)

## 4. Learned successor planning → typed AST planning

The next architecture learned primitive transition consequences and used them to
plan.

- Visible scalar H=2 successor planning passed 24/24 mixed-order tasks.
- The fresh k=3 pair transferred the correct order but exposed unsupported
  hypothetical states, motivating an explicit physical `UNKNOWN` state.
- Typed AST site planning passed visible one-edit tasks across multiple depths.
- Genuine H=2 typed-AST planning recovered noncommutative order and real
  child-to-root replanning.
- Several beam/anytime variants localized physical tie and STOP-frontier issues;
  the final STOP-bounded anytime variant preserved the exact visible result with
  less work.

Representative artifacts:

- [`public_typed_horizon2_successor_visible_probe.json`](../../experiments/r3_pilot/public_typed_horizon2_successor_visible_probe.json)
- [`public_typed_horizon2_successor_fresh_k3_probe.json`](../../experiments/r3_pilot/public_typed_horizon2_successor_fresh_k3_probe.json)
- [`public_typed_ast_horizon2_int_visible_probe.json`](../../experiments/r3_pilot/public_typed_ast_horizon2_int_visible_probe.json)

## 5. Factorized typed edits → broad bridge STOP

Factorizing site/schema/operator/operand/literal improved controlled transfer.
H=1 factor transfer passed renamed bindings, changed prompt surface, signed
literals, and unseen incumbents. Created-child H=2 eventually passed after both
structural and compositional-prompt repairs.

However, the broad public Simple-SWE factorized bridge stopped 2/7 attempted
because prompt evidence could not uniquely resolve leading structural tiers in
multiple families. That result retired further H=2 prompt-pattern patching.

Reference:

- [`public_factorized_simple_swe_train_bridge_v1.json`](../../experiments/r3_pilot/public_factorized_simple_swe_train_bridge_v1.json)

The replacement direction became bottom-up typed semantic synthesis.

## 6. Bottom-up typed semantic synthesis

A frozen bottom-up synthesizer using only typed signatures and four visible I/O
rows passed the first broad 30-task TRAIN gate:

- 10/10 scalar offset;
- 10/10 `List.map(add)`;
- 10/10 `List.map(abs)`;
- all 960 property cases excluded from model selection passed in the sandbox.

This established a strong **programming-by-examples** mechanism, not prompt-only
coding or domain-general learning.

Reference:

- [`public_typed_semantic_simple_swe_train_bridge_v1.json`](../../experiments/r3_pilot/public_typed_semantic_simple_swe_train_bridge_v1.json)

The next six-family TRAIN gate stopped after 13/60 attempts and 6 passes. It
localized three distinct issues:

1. complete-tier declaration overflow on scalar clamp;
2. retained bound-context diversity overflow on map-count;
3. four-row observational aliasing (`sum(x)` versus the intended clamp-sum).

The spent cohort is terminal and must not be retried or tuned.

References:

- [`public-typed-semantic-six-family-train-gate-v1-verdict.md`](public-typed-semantic-six-family-train-gate-v1-verdict.md)
- [`public_typed_semantic_six_family_train_gate_v1.json`](../../experiments/r3_pilot/public_typed_semantic_six_family_train_gate_v1.json)

## 7. Compact outputs and packed semantic families

The dense 3,284-cell stored output representation was replaced by a compact
winner-plus-digest representation while preserving physical bank identities and
race results. All 44 frozen representation/runtime gates passed.

Reference:

- [`compact-typed-output-encoding.md`](compact-typed-output-encoding.md)

The semantic frontier then moved to packed operator families, typed slots, and
physical semantic equivalence classes so concrete programs are materialized only
after a physical semantic winner exists.

Milestones:

- packed scalar V3 closes cost-5 clamp selection under the unchanged 30k work
  budget;
- packed full-typed slice composes `List[Int] -> Int` sum and both less-than to
  count orientations;
- cost-9 / larger-grammar scaling remains an active engineering boundary.

Current source:

- [`typed_semantic_packed_frontier_v3.py`](../../experiments/r3_pilot/typed_semantic_packed_frontier_v3.py)
- [`typed_semantic_packed_frontier_full_typed_v1.py`](../../experiments/r3_pilot/typed_semantic_packed_frontier_full_typed_v1.py)

## 8. Prompt-factor learning lineage

The original canonical TRAIN prompt manifest stopped before learning because 90
prompt appearances collapsed to only 42 distinct role-normalized physical
inputs and several factors lacked sufficient independent support.

Reference:

- [`public-train-prompt-factor-manifest-v1-stop.md`](public-train-prompt-factor-manifest-v1-stop.md)

Self-posed development then proceeded through several explicit generations:

- V2 showed small add/subtract and nine-factor atomic learning could work;
- same-union multi-factor composition failed;
- V3 local-span atomic generalization stopped at 40/54;
- V4 exclusive WTA improved to 44/54 but exposed shared-feature credit errors;
- V5 teacher-gated branch-local credit passed 54/54 atomic replay, 18/18
  two-factor compositions, and 21/21 compositor controls with causal teacher-
  off/cell-off/permuted-label negatives.

Current source:

- [`teacher_gated_factor_workspace_v5.py`](../../experiments/r3_pilot/teacher_gated_factor_workspace_v5.py)
- [`teacher_gated_factor_workspace_v5_protocol.py`](../../experiments/r3_pilot/teacher_gated_factor_workspace_v5_protocol.py)

## 9. Joint prompt-factor + semantic selection

The first frozen self-posed joint matrix now connects learned V5 factor relays
to generic packed-family admission, then uses visible-I/O physical races to
resolve exact semantics before selected-proof materialization.

It passed 3/3 treatment rows and 4/4 controls. This proves the **joint seam** only;
it does not establish canonical TRAIN prompt learning, family-disjoint transfer,
or domain generality.

Reference:

- [`joint_prompt_factor_packed_full_typed_v1.py`](../../experiments/r3_pilot/joint_prompt_factor_packed_full_typed_v1.py)

## 10. Retired directions

Do not reopen these without a new falsifiable rationale and a newly named gate:

- ridge/readout retuning on the old feature space;
- static whole-prompt decoders;
- whole-program lookup/continuation as the composition mechanism;
- blind edit search and the failed persistent babbling scheduler;
- greedy single-step actor as a planner;
- task-specific H=2 prompt-projection patches;
- raising spent work bounds to rescue failed canonical tasks;
- restoring task/family/ordinal IDs to manufacture prompt uniqueness;
- leaking hidden property/DEV outcomes back into selection or architecture
  design.

## 11. Why this history is separate

The main roadmap should answer:

1. What are we trying to build?
2. What design principles constrain it?
3. Where are we now?
4. What gates must pass next?
5. What claims remain sealed?

This document answers a different question: **how did we get here, and which
roads have already been falsified?** Keeping those roles separate prevents the
roadmap from becoming an ever-growing lab notebook.
