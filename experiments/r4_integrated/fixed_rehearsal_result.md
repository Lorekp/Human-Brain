# Fixed rehearsal preserves taught answers and returns them to Python

2026-09-05. Frozen source: `2eb0fb19bacbf1e6b08b265840bba2b9e70f7d32`.
One run passes the [fixed qualification](../../docs/research/fixed-rehearsal-plan.md).
Allocating the same new teaching budget across A and B lets cells retain A and
acquire B. The conventional control also passes, and fits sooner. This qualifies
the supervised allocation strategy on these taught items, not a general learner.

## Findings

The same 16 exposed SQuAD 2.0 TRAIN questions retain their complete paragraphs,
original annotations and saved MiniLM representations. Every prediction receives
the question's correct paragraph inventory: this is an oracle source condition.
Each block contains six supported questions and two STOP cases.

Authenticated after-A parents include weights, Adam moments and steps, and offer
and update counters. Each parent already has 2,048 A offers. Four updating
continuations receive 2,048 **new** offers each: 8,192 total. B-only repeats B;
rehearsal alternates B and A, allocating 1,024 offers to each block. The external
store supplies fixed labeled lessons. There is no error-selected replay, extra
teaching, optimizer change or repeated encoder inference.

Final counts separate exact supported answers from correct STOPs. The same final
prediction objects go directly to action, without another reader inference.

| Policy | A (6 + 2) | B (6 + 2) | Executed (12 + 4) |
| --- | ---: | ---: | ---: |
| Cells, B-only | 2 + 0 | 6 + 2 | 8 + 2 |
| Cells, rehearsal | 6 + 2 | 6 + 2 | 12 + 4 |
| Conventional, rehearsal | 6 + 2 | 6 + 2 | 12 + 4 |
| Frozen producer, trained heads, rehearsal | 2 + 2 | 0 + 1 | 2 + 3 |
| Cells, no new learning | 6 + 2 | 0 + 0 | 6 + 2 |
| Conventional, no new learning | 6 + 2 | 0 + 0 | 6 + 2 |
| External cosine | 0 + 0 | 0 + 0 | 0 + 0 |

Rehearsal cells preserve all six A decisions lost by B-only and gain all six
supported B answers over their matched trained, learning-disabled parent. A stays
perfect at every sampled probe. B is one supported/two STOPs at 128 new offers,
three/two at 256, and six/two at 384, remaining perfect thereafter. Conventional
rehearsal fits both blocks at the first 128-offer probe and retains them.
Probe spacing is 128 offers; these are sampled first-fit times. Rehearsal cells
show **retention, not an observed loss-and-recovery episode**.

The frozen-producer control starts from its own weaker after-A parent (A2+2),
whereas full cells start at A6+2. Its failure limits attribution across the
combined learning history; it does not isolate whether producer updates during
this continuation are necessary. No full after-A producer was newly frozen for
that ablation. No advantage over the conventional learner is demonstrated.

## Execution, validation and costs

The B-only branch reproduces all 2,048 original B update receipts, 256 original
B probe rows and its complete final checkpoint **exactly**. Its final hash is
`12235c7bb99de2900fd95f0e6865810612b1a6d1fcf6feda3737998d2661616e`.
This validates deterministic continuation from the authenticated parent. Saved
audits check all 8,192 new offer targets/masks/counters, 1,904 prediction records,
300 initial/final checkpoint arrays, and unchanged learning-disabled state.
They do not independently recompute gradients or the full learned trajectory.

All 112 policy actions and two shared motor demonstrations execute in real Python
workspaces. All 342 retained files match receipts, with no collateral changes.
The learned motor uses an explicit ordinal configuration instruction and the
selected value, with generic Python literal escaping. It does not receive the
reading question or target answer. No teacher correction occurs during actions.

The new executor explicitly uses isolated Python with UTF-8 and LF output,
retains raw stdout/stderr bytes and hashes, and decodes strictly. All action
roundtrips pass the strict audit, including the previously problematic external
en-dash selection; that selection remains a wrong answer. Neutral preparation
reproduced the old isolated-child encoding defect and Windows newline translation
before qualifying the repair. The old failed audit remains unchanged. Existing
trimmed-output scoring does not establish leading/trailing-whitespace fidelity.

The run takes **62.601 seconds** including primary artifact writing; the outer
process takes **63.424 seconds**, excluding its own final receipt write. Peak
parent working set is **333.262 MiB**, within 20 minutes and 2 GiB, with no retry.
Shared word vectors occupy 6,021,120 bytes. Counted learner/optimizer, input and
motor objects occupy **21,629,049 bytes**; the separately counted source/label
graph occupies 657,635 bytes. The complete retained experiment graph is
138,533,214 bytes, including parent reports and checkpoint/probe copies. These
overlapping scopes are not additive and are not interchangeable with RSS.

The [preparation receipt](../../benchmarks/fixed_rehearsal_preparation_v1.json)
preserves restoration, inert runner/failure-path QA, neutral transport checks
and native setup. The [report](../../benchmarks/fixed_rehearsal_v1.json), SHA256
`a219ece11778ae7b9ded22e1103fb33556baf50f8aebbf66e4deef4c3312779d`,
has matching [completion](../../benchmarks/fixed_rehearsal_v1.completion.json)
and [launch](../../benchmarks/fixed_rehearsal_v1.launch.json) receipts.
The [audit bundle](../../benchmarks/fixed_rehearsal_v1.audit.json) embeds the
three passed saved-evidence audits and their checkers; audits perform no new
model or child execution. The local checkpoint archive is 9,031,554 bytes,
SHA256 `404edb1d9866331cfbdac747fd979626b262f59350895d09584b32302fa3b458`,
at the report's retained `.runtime` path.

## Next decision

Return to the persistent action workload with fresh original public questions
and a cumulative, unlabeled source pool. Remove the evaluator's correct-paragraph
routing from actual attempts; retain an explicitly oracle paragraph diagnostic
to distinguish source competition from within-paragraph errors. Freeze selection,
feedback/rehearsal allocation, matched controls, later untaught tasks and resource
limits before one session. Do not extend fitting on these 16 questions.

This is the next integration step because new-question usefulness and source
competition remain untested for the qualified parents. Raw paragraphs, vectors
and emitted span bytes remain stored, so deleting source files would not create
source-absent recall. That still needs an explicit learned read/write/output
contract and payload accounting. Official exam structured responses also remain
pending; the scalar span interface cannot silently substitute for them.
No architecture is adopted, protected gate opened, independent generalization
claimed or autonomous recovery demonstrated. Public-data pretraining overlap
remains unknown; the original attribution and exposure notices still apply.
