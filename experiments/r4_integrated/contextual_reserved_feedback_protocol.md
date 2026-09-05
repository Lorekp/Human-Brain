# Reserve feedback opportunities for later executed errors

Status: preregistered public TRAIN comparison, 2026-09-05. One fixed run with
new transfer instances; no protected evaluation or oracle-weight deployment.

## Failure and hypothesis

The [capped-update comparison](contextual_capped_update_result.md) rejected smaller
steps: existing fixed updates retained 2/16 skills and capped updates 1/16.
Nine fixed-arm failures satisfied every admitted contrast. Seven of their final
wrong leaders were actually executed after the 32-pair budget had been spent,
and received teacher repairs but no learning update. The
[TRAIN feasibility diagnostic](contextual_representation_feasibility_result.md)
verified that the unchanged features admit the required finite training ranking;
its host witness is excluded from this experiment.

Test whether allocating the same maximum number of feedback offers across the
whole teaching session improves retained execution. Preserve opportunities to
learn from later actual errors instead of exhausting all new-pair admissions
early. This changes on-policy coverage, replay timing and ledger size; it is not
an identical-ledger comparison or simply an increase in total training offers.

Adoption requires more retained affine-neuronal successes than the concurrent
baseline without fewer supported fresh successes. Targets remain at least 6/8
fresh and 12/16 retained edits, including both properties for three fresh
entities. Unsupported errors and STOPs remain separate. These generated cases
alone cannot pass the integrated roadmap gate or establish generalization.

## Fixed architecture, actions and inputs

Both BASELINE and RESERVED use BINDING gain 16, fixed comparison updates at rate
1, initial zero evidence weights, the existing learned occurrence motor and
affine currents with interpolated GLIF. After primary STOP, allow one unique raw
candidate exploration; use external cosine only if no unique raw candidate
exists. Correct explorations do not create negative feedback. Preserve the
[actual FAIL/teacher-PASS contract](contextual_exploration_protocol.md).

Reuse the same sixteen teaching requests, four documents and counterbalanced
48-encounter schedule. Every encounter records the learner's first attempt and
one existing teacher execution. Replays repeat certified comparisons without new
teacher executions. Do not add epochs, solver labels, candidate filters, rate or
threshold changes. Each updater has a same-policy frozen mate receiving its exact
documents and feedback; between updaters, actual on-policy negatives may differ.

`contextual_reserved_feedback_cases.json` links the openly reused teaching data
and fixes six new transfer documents before model outputs: eight supported
requests in the existing simple/neighbor/repeated-value/paraphrase strata and two
unsupported requests. No transfer or exam result influences feedback allocation.
The source inventory and all choices are committed before the single run.

## Exactly one allocation change

BASELINE keeps the previous policy: at most 32 new pairs, followed by at most
four distinct older cyclic replays per admitted correction, with the final
correction's rehearsal completed. Its maximum realized offer count is 150
(32 new plus 118 replay), although the historical replay guard allows 128.
Keep all 48 encounters and teacher executions after admission stops.

RESERVED permits at most one new pair per encounter, hence at most 48, within
**150 total pair-feedback offers**. Count an offer even if the native unit-margin
gate produces no update. A qualifying new pair comes only from the actual edited
FAIL and the same encounter's teacher PASS.

Use one-based encounter `t` from 1 to 48. After admitting that encounter's new
pair, if any, compute `remaining=48-t` and `available=150-used-remaining`. Rehearse
up to `min(4, distinct eligible old pairs, available)` comparisons, using the
existing cyclic order. Exclude the newly admitted pair from that encounter's
replay. Rehearsal may run after a correct encounter, releasing its unused new-pair
reservation. Do not rank replay choices by labels, scores, margins or failures.

Maintain `offers_used <= 150-(48-t)` after every encounter, leaving at least one
offer for each possible later error. Never spend negative availability or exceed
150. With all offers spent, replay count is `150-new_pairs`; if all 48 encounters
need correction, at most 102 replay offers remain. If insufficient certified
pairs or the per-encounter cap prevent spending the budget, report underspend
without invented feedback, extra epochs or terminal padding.

## Measurements, QA and interpretation

Report actual new/replay offers, native updates, remaining reservations, replay
triggers, ledger bytes and matched frozen exposure. Distinguish first attempts,
fallbacks and teacher outcomes. Identify late errors that are admitted, remain
wrong without admission, or become correct under the changed trajectory.

For each final wrong leader occurrence, distinguish whether it was ever actually
executed as an edited FAIL (with encounter steps), whether it was admitted as a
negative (with exact pair IDs), and the final teacher-minus-that-leader margins
for those admitted pairs. Join saved execution rows and the certified ledger;
these read-only diagnostics must never influence training or replay selection.

At unchanged final weights, execute all sixteen retention and ten transfer
requests through the same predeclared primary/reference/ablation policies as the
capped comparison. Keep all candidates visible. Report primary affine outcomes,
raw-selection agreement, both-properties counts, unsupported actions, final
global ranks and certified-pair fit. Shared-motor controls isolate evidence
selection, not the entire neuronal system; retain the external encoder/store
disclosure. Do not choose an ablation from the results.

Record same-source thresholded-selection transitions around new correction and
rehearsal, including replay-only encounters. Do not invent a new-correction
transition where no new pair was offered. Separate correct-to-STOP and
STOP-to-correct subsets from other selection changes. These repeated probes are
diagnostics, not independent executed tasks.

Use the original public KS2 passage/questions/rubric unchanged, ingesting once
per learner for all exam policies. Preserve pending human marks. Assert source
absence at action time, decoy integrity, fixed motor, read-only evaluation and
zero frozen mutation. Report total and phase runtime, parent memory limits,
retained raw/derived bytes, external ledger and encoder costs.

Before the frozen run, use inert synthetic schedules to verify reservation
arithmetic under failures, successes, early/late corrections, distinct replay
selection and budget underspend. Review actual custody paths and preserved
baseline behavior. Do not repeat unaffected core tests without a concrete concern.
Preserve any interrupted experiment and harness corrections separately.

Budget one allocation policy and one two-arm comparison, with at most two hours
for implementation/diagnosis starting 02:47 UTC. After the result, return the
coverage-versus-interference decision to the active integration plan. No tuning
on spent transfer instances, further oracle work or protected evaluation.
