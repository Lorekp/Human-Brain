# Shared temporal query map: fixed five-arm return

Status: prospective implementation, no cost or acquisition admitted. The
[selected review](../../docs/research/local-shared-temporal-query-review.md)
owns this one structural comparison. It replaces dense A with one physical
32-input/32-output temporal circuit B, reused across all 32 sensory-code channels.
The normalized query population Z remains 32-by-32 with real-arithmetic energy
at most 32. Forward projection is Q=B.T@Z; all occupied tape keys participate in
the same soft start competition. No chosen token, lag, role or address is supplied.

Keep raw-byte codes, query normalization, 128-slot literal tape, shared codec,
causal prefix filtering, rate 0.01, seed 733 and the ordered acquisition roster.
B uses ordinary private Xavier initialization at seed+4 with no extra scale.
Its actual initializer and geometry differ from dense A. Corresponding code
channels are structurally shared; this prior is not evidence that learning
cross-channel rotations caused earlier failures.

The held-teacher signal is H=sum_s((p_s-b_final,s)*K_s). Each physical B synapse
accumulates its 32 local products Z[i,d]*H[j,d] in float64; validate the sum,
cast once to float32, and publish at rate 0.01 with the held codec proposals.
All proposals must be finite before either group changes. There is no upstream
query learning or reverse history credit. The conditional descent argument is
not a machine-arithmetic or combined-learning guarantee.

## Arms, parity and success

Five ordered arms are learned, address_frozen, slow_frozen, writes_disabled,
and dense_reference. The four candidates share real initial arrays. Address
frozen leaves the codec plastic; slow frozen preserves both groups while fast
sensory storage remains active; writes disabled retains allocation/adjacency
but zero sensory/key associations. Computed products and enabled publications
are counted separately from numerical parameter changes.

`local_lag_query_reference.py` preserves the exact main class at
`8ef70d6f512caffc0cbc9eac2f19b772298447b2`. It starts fresh, not from learned
weights. Require exact authenticated initial/final state, all 1,024 scientific
lesson rows and twenty outputs versus the prior learned arm, excluding arm/time
and explicitly new work only. Fixed input codes and initial codec arrays match
across all five; B[32,32] is not falsely matched to A[1024,1024]. No production
compatibility branch, dummy buffers or old module-global substitution is used.

Each arm receives 1,024 lessons: 5,120 genuine first attempts and 15,360 teacher
byte/EOS offers overall. Each first attempt precedes its teacher. There are
twelve retained memories per arm, twenty final queries/actions per arm and two
separate motor demonstrations. Two-byte generation includes its ordinary EOS
decision. Delay adds query-only distractions; interference adds only the
declared unrelated records. Source-free requests contain only id, memory_id,
query_hex and delay_reads. Source routing is not filesystem isolation.

Success requires learned 8/8 taught, 4/4 novel, 4/4 delayed and 4/4 interference;
strict novel superiority to the three matched candidate controls; no stratum
regression versus dense_reference; and strictly more total successes than the
dense reference's authenticated 17/20. Do not require impossible novel
superiority over its existing 4/4. Combinations are development-exposed, not
fresh adjudication. Failure closes this fixed fit without automatic extension.
The report's `learned_retrieval_advantage` means novel superiority over those
three matched controls only; it does not mean novel superiority over dense A.

## Fresh cost, streaming and work

Cost uses four fixed maximum-length lessons per arm: twenty lessons, sixty
teacher offers, five cap fixtures, 100 actions and two demonstrations. Each cap
ingests 127 periods plus EOS, forces non-EOS through ordinary copied/restored
codec parameters and verifies exact original-weight restoration. No cap teacher
or fitted cost state is retained. Cap ingestion remains measured other cost.

For each arm project `2*1024*(max_model_seconds+max_journal_seconds+cap_seconds)`.
Other time is twice outer elapsed minus summed model and journal times, then
add thirty seconds. Preserve the 900-second aggregate run-plus-saved-audit
ceiling and 2,048-MiB process-tree watchdog. Root owns admission and every actual
cost/full/native-action launch, with absolute paths and exact `-X utf8` prefix.

Every scientific lesson is preserved as one ASCII-escaped UTF-8 JSONL line with
LF. Timed journal work includes recursive bounds, serialization, write, flush,
rolling hash and scalar summaries. Report and completion bind row order/count,
bytes and SHA; no full lesson list remains live. Both auditors stream the same
whole historical 6,144-row journal and compare its selected learned rows to the
new dense reference. Current and historical readers each retain at most two
overlapping row trees. No scientific field is dropped.

Fix 262,144 bytes per encoded line or recursive CPython row tree, four live row
trees, four line buffers and allocation factor two: E=3 MiB. The run estimate
is max(cost parent peak+worker peak, cost outer tree peak)+E+128 MiB. The audit
estimate is finished cost-audit tree peak+E+2F+32 MiB, with authenticated full-only
historical checkpoint payload F=105,236,476 bytes. Final admission uses their
maximum under 2,048 MiB. A pending cost checker cannot claim admission before
its actual completed audit monitor is bound. Full run and audit rederive this
final projection. These reserves are engineering estimates, not RSS bounds.

Report raw bytes/EOS, query lag shifts, normalizations, 32 circuit applications
and 32,768 projection-synapse evaluations per candidate query. Each candidate
teacher computes 32,768 local products and 31,744 channel reductions, including
frozen modes, before its one address group opportunity. The dense reference's
million-element operations stay separately counted. Preserve actual tape/key,
codec, prefix and cursor work, including both acquisition query traversals,
final ingestions, delay distractions and cap work. Worker per-output time
includes its preceding distractions; package time also includes restore and
serialization. Host action execution remains supplied scaffolding.

## Ordinary QA

Run doctor and only reviewed authored nodes in the existing decoder test file.
Cover independent multi-channel scalar forward/update arithmetic, held codec
publication, channel-permutation equivariance, finite/cast boundaries, actual
work, four controls, snapshots, exact archived dense-source trajectory, cap
restoration, pure journal/projection algebra and a new source-free child.
Archive-bind historical dense math fixtures instead of keeping compatibility
code. No TRAIN/cost/endpoint model workload is part of QA. Record every test
attempt and small-fixture exposure before source freeze.

Doctor passed at `.pytest-tmp/dev/doctor-remxq_x_/receipt.json` on design
revision `8ad48d5dd2e28aa3ec7c9eb611781cd1e0317128`. Seventeen distinct
reviewed cases pass with no failure, skip or repeated node:

- `.pytest-tmp/shared-temporal-e2e-v1.xml`: one smallest authored end-to-end
  ingestion/free/teacher/snapshot case, 1.73 seconds.
- `.pytest-tmp/shared-temporal-focused-v1.xml`: fifteen scalar, sharing,
  control, finite, archive-parity, cap, factory/criterion and pure journal
  cases, 3.42 seconds.
- `.pytest-tmp/shared-temporal-worker-v1.xml`: one new five-arm source-free
  child fixture, 5.15 seconds. Ten query requests use untrained saved models
  and empty-document memories; no native action fixture runs.

These fixtures construct 32 learner instances, including five child restores,
and perform thirty authored ingestions: 741 raw document bytes and 38 EOS
symbols. There are 28 ordinary free predictions, seventeen accepted teacher
calls (51 byte/EOS offers), four valid query-only traversals and one rejected
overflowing query-norm traversal. All fifty traversals process 306 raw query
bytes plus fifty EOS symbols; two symbols belong to the rejected query. The
five 127-period cap fixtures are disposable QA, not cost admission. Pure
journal QA writes twenty synthetic rows and rejects two oversized rows without
a model call. There are zero TRAIN lessons, endpoint probes, motor
demonstrations or executed actions. Historical dense fixtures now import the
explicit archive; the new source and three-lesson trajectory test establishes
exact parity without a production compatibility path.
