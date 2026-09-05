# Compact versus varied teaching in the integrated learner

Public TRAIN development contract, 2026-09-05. Freeze the reviewed source,
both teaching fixtures and the common evaluation fixture before one comparison.
This tests teaching presentation with the existing RESERVED learner.

## Observed failure, hypothesis and decision

The [three-stream robustness result](contextual_robustness_result.md) shows weak
fresh execution despite preserved training contrasts and faithful edit execution.
RESERVED solves 5/8, 3/8 and 0/8 fresh requests; PROTECTED solves 1/8, 2/8 and 1/8.
Neither policy abstains on unsupported requests. The fixed encoder already has
contextual features. Teaching, however, repeatedly presents compact facts and
a narrow question set before evaluation introduces clauses, distractions and
varied questions.

Hypothesis: teaching the same facts through varied source and question forms
improves downstream evidence selection within the current learning architecture.
Compare COMPACT with VARIED. Both use RESERVED feedback allocation; protection
is disabled. This intervention changes a declared presentation bundle, including
source context, candidate inventory, query features and on-policy negatives.
It does not separately identify the effects of syntax or question diversity.

Adopt varied teaching for subsequent development only if VARIED has **strictly
more supported fresh executed successes**, with no decrease in any of:

- the four familiar-form fresh requests;
- the four broader-form fresh requests;
- the sixteen common old-fact retention requests;
- the two correct unsupported STOPs.

Report paired gains/losses and both-properties entity counts, not only totals.
Absolute targets remain 6/8 fresh, 12/16 retained and both properties for three
fresh entities. Appropriate unsupported behavior remains required for the full
integrated milestone. A relative adoption pass is neither that milestone nor
independent reading generalization. On failure, keep the compact baseline and
close the test; do not soften the rule or select a favorable ablation.

Budget: at most two hours of preparation/diagnosis starting **04:57 UTC**, then
one sequential COMPACT/VARIED comparison with at most thirty minutes total
comparison wall time, including child waits. Do not run model previews, extend
teaching epochs, sweep rates or tune to the fresh outputs. Preserve any failed
execution and diagnose infrastructure faults before a separately recorded
recovery decision. Close this experiment with evidence and a next capability
decision even when it fails.

## Teaching and common evaluation inputs

`contextual_curriculum_compact_train.json` preserves the original sixteen TRAIN
case records and 48-encounter schedule. `contextual_curriculum_varied_train.json`
uses the same fact/entity/property inventory, case identities, workspace schema,
teacher values, source identities and encounter order. Only source prose and
request text differ. Each case has one fixed query form repeated three times.
No new teacher facts or additional encounters enter VARIED.

Each arm ingests exactly one rendition of each of four source documents, once.
Teacher values must retain unique source occurrences compatible with the existing
teacher-span custody. Do not repeatedly ingest paraphrases, silently select a
teacher occurrence or filter candidates to resolve ambiguity. Static checks may
validate fixture meaning, span reachability and schema without querying the model.

`contextual_curriculum_cases.json` links COMPACT as its `training_source`. This
supplies the same sixteen original retention requests to both arms; it does not
authorize ingesting compact documents into VARIED. Common queries drive final
retention execution, ranks and the before/after-fresh source-competition probes.
Online teaching diagnostics continue to use each arm's actual queries.

The common transfer fixture contains six new source documents and ten requests:
four supported familiar-form requests for two entities, four broader-form requests
for two other entities, and two unsupported requests. Each supported entity has
transport and audit facts; include a multiword value in each supported stratum.
Every gold value must be an unambiguous contiguous one-to-three-word candidate
span. Fresh substitutions are disjoint from prior contextual comparison cohorts.
Inputs are fixed together, shared across arms, and receive no feedback.

These are authored development exercises. They provide executed transfer pressure
but do not replace official public-exam comprehension or independent adjudication.
Common old-fact probes may use a different question form from VARIED teaching;
label this explicitly when distinguishing retention from query-form transfer.

## Unchanged learner and authentic feedback

Use BINDING candidate interactions at gain 16, fixed comparison rate one,
zero initial evidence weights, the existing frozen MiniLM encoder and learned
literal-edit motor, and affine currents with interpolated GLIF. Both arms use
`comparison_protection_policy="none"`. No cell, representation, feature,
candidate-generation, motor, learning-rate or threshold change is permitted.
The TRAIN-only host feasibility witness and weights remain excluded.

Follow the [RESERVED allocation contract](contextual_reserved_feedback_protocol.md):
48 teacher executions, at most one new actual edited-FAIL/teacher-PASS pair per
encounter, and at most 150 total pair offers including native gate no-ops.
After encounter t, reserve 48-t potential corrections and replay up to four
distinct eligible older pairs within the remaining budget, in existing cyclic
order. Exclude the newly admitted pair from that encounter's replay. Report
underspend without padding. Primary STOP permits the existing unique raw-score
candidate exploration, then external cosine only if no unique raw candidate
exists. Correct exploration is not negative feedback.

Each arm has a same-seed, same-policy frozen mate receiving that arm's exact
documents and certified feedback. Frozen learning remains disabled. The updating
arms may collect different failures, offers and ledger sizes; equal limits do
not establish equal exposure or compute. Teacher repairs are scaffolding, not
learner success or autonomous recovery.

Retain all eleven predeclared evaluation policies, including affine, raw-score,
frozen and external references. All share the learned motor and isolate evidence
selection, not the whole neuronal system. The frozen encoder, raw documents,
candidate vectors and feedback ledger remain disclosed external components.

## Measurements, custody and QA

Record first-attempt teaching execution, exploration and teacher outcomes,
new/replay offers, native updates, and same-parent authenticated feedback.
Immediately after teaching, save read-only scores for the actual taught queries
separately from common old-fact probes. These raw-selection diagnostics are not
extra executed successes. They distinguish acquisition of the taught forms from
performance on the common query forms.

Save common old-query scores/ranks before and after fresh ingestion at unchanged
weights. Keep actual TRAIN pair margins tied to their original queries and
eligibilities. Coverage of a common probe requires exact query identity for both
admitted pairs and executed failures; matching only a case name is insufficient.
Zero exact-query coverage does not mean the underlying fact was never taught.

Execute all existing 572 transfer/retention policy actions. Verify live source
absence, selected/materialized edits, decoy integrity and read-only evaluation.
Inspect fresh errors by stratum and span/property/source competition, and old
failures against actual TRAIN coverage without treating paraphrases as identical
certified contrasts. Raw-score/affine agreement tests the final neuronal readout.
Internal source retention is not raw-source erasure or semantic compression.

Keep one unchanged official KS2 Q6a/b helper call per arm, with one ingestion
per learner for all exam policies and no feedback. These repeated items are
regression observations, not fresh exam evidence. Preserve official-rubric human
marks as pending. Additional official development items require their own
manifest under the [reading plan](../../docs/research/reading-comprehension-evaluation.md).

Report phase and total wall time, parent-memory scope, source/candidate growth,
array/byte payload, ledger and encoder costs without adding overlapping measures.
Record source/native/encoder identities and save trained arrays. Because COMPACT
keeps the original teaching trajectory, verify its trained arrays match the
previous RESERVED checkpoint; a mismatch requires diagnosis, not interpretation
as a new scientific result. Do not expect VARIED weights or stores to match.

Before freezing, verify actual-source fixture routing and validation, strict
report serialization, early collision rejection, failure preservation, common
probe propagation and unchanged default behavior with inert synthetic checks.
The shared runner may gain a narrow optional teaching-fixture seam and corrected
query-specific diagnostic attribution; the learner and native source stay fixed.
Do not rerun unaffected native suites. Independently audit saved results without
model replay. Return to the [active integrated plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
when the comparison closes; another projection or rate experiment is not an
automatic successor.
