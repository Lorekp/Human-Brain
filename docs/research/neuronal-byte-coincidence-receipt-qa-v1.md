# Prospective coincidence receipt QA

Status: root-reviewed and selected for one ordinary launch, 2026-09-06. Append
the two nodes unchanged, commit and obtain a clean doctor. Run these first, then
the separately reviewed single worker-exchange node, fail-fast under one shared
60-second/2,048-MiB watchdog. Use fresh basetemp/log/JUnit stem
`.pytest-tmp/neuronal-byte-coincidence-integration-qa-v1`. No selected node has run.

Use two appended nodes in the existing `tests/test_byte_sequence_decoder.py`.
The source fragment is `.pytest-tmp/coincidence-receipt-qa-v1-fragment.py`.
Reuse its existing qualified `_competitive_receipt_qa_reader` and save helper;
do not call an old test node. Those helpers build lists and SimpleNamespace
objects only. The new fixture authors all 50 additional work values using scalar
arithmetic and never calls either production expected-work formula to construct
expected data. Both validators must independently accept or reject it.

## Fixed cases

1. `test_coincidence_receipt_maximum_complete_row`: one synthetic complete
   prediction/feedback pair at N128, all8128 comparator currents/activities,
   17 free decisions and 17 teacher decisions with all128 teacher branches.
   Query length31, maximal516/129/2 head populations in both phases, complete
   mixture257 and posterior vectors, old qualified full metadata/text/work
   reserve. Compact using actual v3 entrypoints, serialize once, parse once,
   validate both readers with writer and independent helper. Save exact encoded
   SHA, line bytes, preparse and parsed tree bytes. Both must fit the existing
   2MiB guards. This is a geometry fixture, not a neuronal feature evaluation.
2. `test_coincidence_receipt_frontend_work_and_query_bindings`: N2/two decisions,
   both frontends crossed with competitive-free/full-teacher phases. Four
   positive raw fixtures are compacted and validated by both implementations.
   Sixteen fixed negative fixtures cover cross-frontend substitution in both
   directions, cross-phase substitution in both directions, free/teacher trusted
   query length mismatch, missing threshold/shift-copy counters, retired dense
   products, incorrect current storage/threshold count, an undeclared key, bool
   as a work integer, teacher local reduction, graded extra gate key and invalid
   enabled teacher flag. Each must raise ValueError in both validators.

No new competition theorem or codec arithmetic experiment is selected. Existing
qualified scalar trace/teacher checks run as part of validating the new rows.
Do not strip fields, convert the candidate into a graded row or use a synthetic
singleton tape to bypass policy checks. The adjacency-only state fixture does
not certify bool checkpoint dtype: strict bool-versus-float tensor custody is
the separate integrated auditor's obligation. Source/type component QA already
owns actual tape construction; this test does not reconstruct a tape.

## Exposure and call ledger

Exactly two nodes, six synthetic raw reader objects and42 synthetic step records:
34 in the maximum pair and8 in the four small readers. The sixteen negative rows
are deep copies, not new neural calls. Zero models, tensor construction, feature
evaluations, checkpoint loads, roster generation, neural teachers, child
processes, actions or benchmark-output reads.

Direct production calls from the two test bodies total51:

| Entry point | Maximum | Dispatch positive/negative | Total |
| --- | ---: | ---: | ---: |
| compact_prediction | 1 | 0 | 1 |
| compact_feedback | 1 | 0 | 1 |
| compact_reader | 0 | 4 | 4 |
| bounded_record | 1 | 0 | 1 |
| validate_reader | 2 | 4+16 | 22 |
| independent check_reader | 2 | 4+16 | 22 |

These are direct calls, not a claim that wrappers omit nested checks. Including
their internal v3 calls gives six compact_reader, one compact_prediction, one
compact_feedback, one validate_prediction and29 writer validate_reader calls;
the independent22 and bounded_record1 remain unchanged. Frozen helpers perform
their normal additional scalar checks. No runtime monkeypatch instruments them.

Raw inputs, compact outputs, mutations and the maximal JSONL are exclusively
saved before final assertions where available; passed receipts are written only
after the corresponding checks. Preserve any failed attempt. Guard failure
does not authorize a larger bound or a smaller fixture. Keep2MiB line/tree and
24MiB common receipt allowance prospective and unchanged. No cost admission or
memory/speed conclusion follows from these synthetic serializer checks.

Fragment lint may ignore F821 only because its two named helper dependencies
already exist in the destination test module; lint the appended source normally
before execution. Root alone owns append and runtime selection.
