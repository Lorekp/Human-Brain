# Executed question revisitation and bounded replay

Public development, declared before outputs on this cohort. The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns the two-hour implementation/diagnosis budget. Prior protocols and artifacts
remain unchanged; the baseline is commit `71055fe`.

## Failure, hypothesis and fixed components

The preceding pairwise learner selected one old answer for every fresh question.
Its actual corrections admitted a query-independent answer ordering, and ten of
eleven initially satisfied pair margins were later forgotten. Hypothesis:
revisiting old questions can supply conditional distinctions, while rehearsal of
executed corrections can preserve them long enough to improve fresh behavior.

Keep the pinned MiniLM encoder, generic 1–3-word candidate spans, 48-word windows,
1537 eligibility features, pairwise native plasticity at learning rate 1 and
margin 1, GLIF race, and two-demonstration occurrence motor. Shared pair bias is
discarded. No semantic candidate filtering or new model components. A small
storage-root lineage supports safe rebinding of historical occurrences after
append-only ingestion; count its retained bytes. The learner continues to retain
raw sources and vectors in disclosed external storage.

## Fixed workload and learning arms

[`contextual_revisit_cases.json`](contextual_revisit_cases.json) declares sixteen
requests over eight teaching documents, two properties per entity. Its exact
48-entry schedule presents each request three times:

1. Entities in forward order, each property's query adjacent; the first property
   alternates across entities.
2. Entities in reverse order with the same within-entity query order. The last
   entity is revisited immediately across the pass boundary.
3. Entity indices 2, 3, 4, 5, 6, 7, 0, 1, with reversed within-entity query order.

Ingest each teaching document once, when first encountered. Remove its live
input before querying. Each encounter starts with an autonomous neuronal edit
and actual isolated Python execution. If evidence selection STOPs, permit one
explicit external-cosine exploration and execute it; never credit exploration as
the neuronal first attempt. Then execute the teacher correction separately.

An executed wrong edit followed by teacher PASS admits one immutable pair of
original occurrence decisions and the corresponding execution receipts. Correct
attempts, absent executable negatives or ambiguous teacher occurrences admit no
pair. Retain at most 32 pairs. Finish the 32nd correction's scheduled rehearsal,
then stop all later updates; remaining attempts and teacher exposures are recorded.

- **REVISIT:** one native pairwise update per newly admitted correction.
- **REPLAY:** the same on-policy correction procedure plus a cyclic sweep of up
  to four distinct older pairs after each new correction, excluding the new pair.
  Maintain one persistent cursor through the insertion-ordered ledger. At most
  128 rehearsal updates; no extra epochs or post-evaluation learning.
- Each has a same-seed **FROZEN** arm, identical source ingestion and exact
  original/rehearsed feedback exposure. Record its own first attempts separately.
  Their on-policy negatives may differ; do not claim identical supervision
  between updating methods.

Rehearsal binds the original query and exact retained candidate occurrence to
the current parent, verifies original custody/eligibility, and applies its
certified consequence. It does not re-execute a workspace or select a candidate
by answer text. The external ledger influences learning only, never inference.
Report its serialized and numeric payload bytes. Repeated rewards are disclosed
experience reuse, not new independent actions, teacher facts or autonomous repair.

## Evaluation and diagnosis

After each new correction and its optional replay, measure read-only conventional
selection on previously encountered teaching requests. These diagnostic answers
stay evaluator-only and never choose updates or the replay schedule.

After teaching, ingest the six fresh documents once, then evaluate the fixed
eight supported requests and two unsupported probes without updates or retries.
Replay the sixteen old requests for retention. Run updating neurons, matched
frozen neurons, conventional learned-score selection and external encoder cosine
through the same learned motor. Preserve multiple lexical matches, repeated
values and distractors; execute selected wrong/malformed edits and check decoys.
Keep original public KS2 Q6a/b untuned, using its existing official-rubric policy
including pending human scoring of noncanonical answers.

Report first-attempt executed success by pass, fresh accuracy, old-task retention,
both-property entity scores, unsupported STOPs, teacher/exploration exposure,
new/replayed update counts, candidate-occurrence score/spike disagreement, ledger
bytes, total storage, time, cell work and process memory. Target at least 6/8 fresh
and 12/16 retained edits, with both properties for three fresh entities and a
useful updating advantage. Unsupported behavior remains a separate requirement
of the broader integration gate. Self-authored fixtures are development only.

The receipt-only diagnostic must distinguish unique executed contrasts from
replay repetitions. Check opposite preferences/cycles on identical document/span
occurrences, with different questions, and compare full-feature versus pure
span-only hard-margin feasibility using a conventional linear-program reference.
Independently verify solver margins; allow at most 30 seconds per fit, and report
timeouts/numerical failures as inconclusive. Failed least squares is not proof
of infeasibility. Measure final margins against first and last replay gaps.
No new negatives, global-candidate or transfer-answer fitting in this diagnostic.

Request order alone does not prove conditional teaching. If the ledger lacks
such evidence, report admission unmet. If conditional contrasts cannot fit,
reconsider representation; if fitting succeeds but learning is lost, address
retention; if retained success lacks fresh transfer, address memorization. Do not
extend this cohort's budget or retune cells after observing transfer outputs.

## Launch and evidence

Fix source/case/protocol hashes before the run, record actual loaded native
identity, and refuse existing output paths. Write receipts with explicit LF
newlines so version control preserves their cross-referenced byte hashes.

```powershell
$env:HUMANBRAIN_NATIVE_CACHE = Join-Path (Get-Location) '.native-cache'
$env:HUMANBRAIN_DEVICE = 'cpu'
./.venv/Scripts/python.exe -m experiments.r4_integrated.contextual_revisit_python_session --output benchmarks/contextual_revisit_python_development_v1.json
```
