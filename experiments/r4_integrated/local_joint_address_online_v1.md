# Joint-address online v1: prospective executed-learning recipe

Status: selected direction; no execution admitted. Proposed runner:
`experiments/r4_integrated/local_joint_address_online_v1.py`. Freeze the new
source, reviewed QA, independent checkers and fresh cost admission before one
run. No learner, action, teacher or test was executed to prepare this recipe.
The [active plan](../../docs/research/persistent-memory-byte-development.md)
and current no-backpropagation constraint govern implementation.

## One forward topology, one learner

Construct generic joint features BEFORE the sole learned address output map:
`X_s = Z @ K_s.T`, then `score_s = sum(B * X_s)` for every occupied start s.
Z is the current normalized query lag matrix and K_s is its stored cue matrix.
The B synapses receive their actual forward feature activity and the declared
address output error; local publication sums `X_s * e_s`. Do not implement an
algebraically equivalent hidden `B.T @ Z` projection followed by manual
K-transposed error transport. No autograd/backpropagation, reverse retained
history, external per-synapse solver, distilled trainer or separate executor.

Use the same query/features/B, marginal free recall, tape and codec for acting
and receiving supervised correction. The teacher follows the same forward
address/codec route with the explicitly corrected prefix, not a WTA policy or
hidden winning-path target. The selected local rule must be verified in actual
source equations before QA. A method name or absence of backward() is insufficient.

The selected implementation materializes the actual float32 feature bank
`X[N,32,32]` once BEFORE B scoring in both free and teacher queries. Retain the
current teacher bank through its corrected prefix and local B publication, then
discard it. There is no streaming replay or recomputation during learning.
At N=128 the bank occupies 524,288 bytes; count it and any float64 cast, weighted
product and accumulator temporaries. Do not journal the bank. The same current
presynaptic activity drives scoring and the local update.

## Four fresh arms and fixed exposure

Order: `learned`, `address_frozen`, `slow_frozen`, `writes_disabled`. Fresh seed
733, rate 0.01, 1,024 lessons per arm. Keep the existing ordered 256-by-4 TRAIN
schedule, eight taught checks, four new-value checks, four delayed checks and
four interference checks. The raw inputs/targets and schedule hash are fixed
before launch; no new roster, exposure sweep or teacher schedule is introduced.

The address-frozen arm keeps B fixed while the codec learns; slow-frozen keeps
both learned components fixed with sensory storage active; writes-disabled keeps
sensory/key storage zero with the same declared allocation/adjacency behavior.
Controls undergo the same actual action and correction schedule. All branches
still perform their declared neural computations even if no weights change.

There is no trained historical reference arm and no old backpropagating learner
construction/forward/teaching in QA or evaluation. The earlier 8/4/4/2 result
is static historical context only. Its checkpoint is not a warm start, and exact
historical trajectory parity is not a requirement of this changed topology.

| Workload | Lessons / first predictions | Teacher byte/EOS offers | Acquisition execution receipts | Final execution receipts | Motor demos | Cost caps |
|---|---:|---:|---:|---:|---:|---:|
| Fresh cost | 16 | 48 | 16 | 80 | 2 | 4 |
| Full | 4,096 | 12,288 | 4,096 | 80 | 2 | 0 |

Cost selects lesson indices `[0,1,2,3]` in every arm from the fixed schedule;
all have the same declared document/query/teacher lengths. All final cases are
executed in cost as well as full. Cost weights/state are discarded, and full
starts afresh. Four cost-only cap fixtures each ingest 127 periods plus EOS in
a disposable 128-slot state and force a non-EOS ordinary codec winner for three
real decisions. Restore exact parameters and discard the clone. The caps add
508 raw bytes, four EOS symbols and twelve decisions; no teacher or action.

Motor demos occur once per workload, before its first acquisition action, and
the motor then stays frozen through acquisition and final evaluation. Thus cost
has 98 total execution receipts including demos, and full has 4,178. An execution
receipt is not automatically an emitted model action or a successful task.

## Same-instance executed acquisition

Keep one learner instance per arm across lessons. For every lesson:

1. Ingest its raw documents through the actual learner and record model/state
   identity and before hashes. Predict once before exposing correction or grade.
2. Execute exactly that output using the existing target-free UTF-8 action bridge
   and frozen motor. Bind output bytes/status, actual materialized files,
   subprocess result and action receipt hash to this model/state version.
3. Grade the real task externally. Record separately model byte/EOS correctness,
   emitted action, lossless execution, actual task success and pending fallback.
   Existing generated-value self-consistency or returncode alone is not task
   success. Rejected/empty/limited output may execute the unchanged pending
   fallback; it does not count as a successful model action.
4. After normal success or failure completes, call the new reviewed local
   `teach(query, target, same_state, .01, ...)` on the SAME learner. Supply exactly
   two teacher bytes plus EOS, recording actual offers/enabled/changed updates.
   Bind its before state/version to the executed attempt and its after hashes.
5. Continue that instance. A retry would require another counted prediction and
   real action; no retry is included here. Watchdog/infrastructure failure stops
   the run and preserves partial exposure rather than fabricating task feedback.

Corrections are explicit supervised signals delivered after executed first
attempts. Since the execution grade is not a numeric input to this selected
update, this is not a reward-driven learning claim. The host grades and supplies
teacher bytes; it does not compute B/codec updates. No action is replaced by the
teacher's answer, and no external learner transfers learned weights afterward.

Final evaluation freezes plasticity and executes all twenty first outputs per
arm through the same bridge. New-value assignments precede any target teaching;
delay reuses the same retained states with only declared query-only reads;
interference adds only the existing unrelated documents. Keep twelve final
memories per arm. Final source-free reload is the same learner class and actual
trained weights, not a differently trained executing system.

## Honest work and streamed evidence

Record actual raw document/query bytes and EOS, tape steps, key/sensory
opportunities, both free and teacher query traversals, delay reads, output/teacher
codec slots and all update offers versus changes. For each queried occupied slot,
forward X construction requires 32,768 products and 31,744 channel reductions;
B scoring consumes 1,024 feature-synapse products and 1,023 reductions per slot.
For an ordinary N=50 query this is 1,638,400 feature products, 1,587,200 feature
reductions, 51,200 B-synapse products and 51,150 score reductions. One teacher
then adds 51,200 local products and 50,176 local reductions. At the N=128 cap,
feature products are 4,194,304 and B-synapse products are 131,072.
The teacher uses its existing current bank for N*1,024 local products and
(N-1)*1,024 local reductions; no second feature construction is counted or run.
Count fixed feature work in controls too.
Do not inherit the old cheap projected-query counter or omit its transient work.

Each complete learning record binds the original prediction, actual action and
grade, same-instance teacher receipt, before/after hashes and all work. Stream
learning and action JSONL in exact arm/lesson order, with headers and completion
binding source/config/input hashes, row count, byte length and SHA. Stream the
80 final receipts as well. Do not retain an all-lessons list or repeat large
source manifests in each row. No historical training journal is replayed.

Preserve all three-position teacher observations required by the saved checker;
do not save X arrays or unused diagnostic projections in every row. The actual
feature bank and its cast/accumulator temporaries have declared lifetimes. Receipt
row/tree limits fail closed rather than authorize trimming scientific evidence.
The new checker audits saved scalar/step consistency, ordered before/after
hash custody and actual actions. Endpoint tensors verify declared frozen/control
invariants. It does not claim exact numerical per-lesson B-update replay without
the unsaved X/intermediate weights. Exact local arithmetic is established by
source review and ordinary independent QA, without old-reference execution or
learner reexecution in the saved audit.

## Complete time and memory admission

For each of four sampled lessons per arm a, measure three disjoint components:

- M: sum of neural/model phases before and after execution. Begin before input
  validation/ingestion/snapshots/hash, include first prediction, then resume after
  the action for supervised teaching and post-state checks. Exclude A and J.
- A: begin before reading/materializing the actual prediction for the bridge;
  include execution, external grading and action-receipt conversion/guards/
  serialization/flush/hash. End before teaching starts.
- J: complete learning-row conversion, tree/line guards, serialization/write/
  flush/hash and scalar bookkeeping after teaching. It excludes action-row I/O
  already timed in A. Its own completed timing stays in a scalar summary.

Let m_a, a_a and j_a be separate sampled maxima; g_a is that arm's complete
128-slot cap prediction time. Let S be the sum of actual sampled M/A/J durations,
and O=cost_outer-S, with finite nonnegative residual within declared clock
tolerance. Imports, cap construction/restoration, full final 80-action package,
checkpoint export/loading, two demos and other fixed work remain in O.

```
T_run = 2 * (1024 * sum_a(m_a + a_a + j_a + g_a) + O) + 30
```

The cap is an additional pessimistic generation allowance; it does not subtract
sampled generation. Separate action/journal maxima scale real per-lesson Python
and I/O work. Nested package timers are not added again. The fixed final 80-case
package is measured in cost and full, so it is not multiplied by 1,024.

Selected saved-audit component: stream and time each sampled complete learning/
action row pair from BEFORE either row is read/parsed through all validation and
scalar summaries, with nonoverlapping per-pair timers. Fixed eighty-case final
artifact checks stay outside these timers and inside the residual.
Let v_a be its per-arm maximum, V its actual sampled sum, and A_outer the completed
cost-audit watchdog time. The final admission adds:

```
T_audit = 2 * (1024 * sum_a(v_a) + (A_outer - V))
T_final = T_run + T_audit
```

This scales growing row validation while charging fixed final receipts/checkpoint
validation in the residual. The checker returns pending admission until its
finished monitor is bound. This timer/formula is selected prospectively and must pass independent source
review before cost; do not substitute an unmeasured allowance afterward.

Each line/tree has a 262,144-byte guard at writer and reader. Bound at most two
current row trees/two line copies per process, factor two on trees. Common
E=3 MiB covers concurrent parent/worker; audit also receives E conservatively.
Implement each audited learning/action pair in a bounded function scope that
returns scalar summaries only. Release both parsed rows and lines before reading
the next pair; generator/caller references must not retain a preceding pair.
Writers likewise release completed rows before constructing the next lesson.
Keep 128 MiB run and 32 MiB audit reserves. Cost/full and both audits must load
the same complete four-model/48-tape endpoint payload, so F=0 full-only tensor
increment. Unchanged numeric model/state shapes would total 33,801,664 bytes;
final main accounting must confirm or explicitly revise this before freeze.

```
P_run = max(cost_parent_peak + cost_worker_peak, cost_owned_tree_peak) + 3 + 128 MiB
P_audit = completed_cost_audit_tree_peak + 3 + 32 MiB
P_final = max(P_run, P_audit)
```

The full activity bank is at most 524,288 float32 bytes; its full float64 cast
and a separately materialized float64 local weighted product each require up to
1,048,576 bytes. The 32-by-32 float64 reduction is 8,192 bytes; its float32 cast
and proposal each require 4,096 bytes. Count additional actual query/codec
arrays and validation temporaries in main accounting, without assuming fusion.
The feature/cast/product/accumulator, cap clone, state-hash copies and runtime
objects remain in measured peaks; no aliasing or zero-work assumption. The
estimates are engineering margins, not allocator proofs. Require T_final<=900
seconds and P_final<=2,048 MiB; actual aggregate run-plus-saved-audit time and
process-tree watchdogs enforce the same ceilings. A failed cost stops, without
fewer lessons/cases, a different rule or relaxed ceiling.

Root launches only after source freeze, absolute paths, exact UTF-8 parent/worker
commands, native preflight, checker/admission hashes and actual PID ancestry are
bound. The full run and audit rederive admission from the completed cost and
cost-audit monitors. Keep failed attempts and their actual exposure evidence.

## QA and the scoped preservation gate

Doctor precedes any later QA. Use reviewed authored cases in existing test files:
one real same-instance predict/action/grade/teach path first, independent scalar
forward-X/B-local-publication math, all four controls, teacher-input timing,
pending fallback versus task correctness, ordinary finite/state checks, bounded
cap, fresh source-free transport and pure timer/streaming/audit algebra. No old
backpropagating reference execution, long acquisition or endpoint probe is QA.
Count every authored prediction, teacher position and action separately.

The reviewed implementation QA passed 20 distinct authored cases, with no failed
or skipped cases and no broad historical-suite execution:

| Receipt in `.pytest-tmp/` | Cases | Pytest seconds |
| --- | ---: | ---: |
| `joint-online-e2e-v1.xml` | 1 | 4.360 |
| `joint-online-math-v1.xml` | 7 | 1.987 |
| `joint-online-remaining-v1.xml` | 11 | 6.86 |
| `joint-online-atomicity-v1.xml` | 1 | 1.90 |

The initial end-to-end case executed a generated period through the actual motor
before three teacher offers on the same learner and memory. The two motor
demonstrations and this one acquisition action are the only real action receipts
in QA. Nine free predictions made 20 output decisions: two in that first case,
12 across four disposable cap clones and six in the four-mode fresh worker.
The empty writes-disabled worker correctly generates two zero bytes and reaches
the limit; the other three empty-document modes emit EOS. This expectation was
corrected during source review before the fixture ran.

Across these cases, there were 21 learner constructions and four disposable
clones, 17 ingestions comprising 534 raw document bytes plus 17 EOS symbols,
and 18 successful query traversals comprising 114 raw query bytes plus 18 EOS
symbols. Eight corrective teaching calls offered 24 byte/EOS targets; one
deliberate late proposal-validation rejection completed its three offers but
published neither parameter group. Three invalid-autograd queries and one
invalid learning-flag call were rejected before activity. Pure grading,
streaming and projection cases add no neuronal or action exposure; their 16
synthetic row pairs are accounting fixtures. The independent auditors passed
72 authored standard-library helper cases without models or real artifact audits.

The selected checkout doctor passed before QA at
`.pytest-tmp/dev/doctor-u7ybrb5l/receipt.json`; the separately executed atomicity
case used `.pytest-tmp/dev/doctor-g2dekjc5/receipt.json`. Native action/worker
fixtures used the prepared cache recorded in
`.pytest-tmp/joint-address-native-preparation.json`. The first passing test
reported a pytest cache-access warning; its receipt is preserved. Later tests
disabled that optional cache. After the first end-to-end test, the only harness
edit added its actually imported `local_lag_query_v1.py` to the source manifest;
the final fresh-worker test verified the resulting source binding. Ruff and
whitespace checks passed. These QA exposures are separate from prospective
cost and acquisition budgets and do not constitute an admitted experiment.

Report `compliant_executed_learning_preserved` only if actual source/local-rule
review passes, all acquisition attempts execute before same-instance teaching,
learned achieves 8/8 taught, 4/4 novel, 4/4 delayed and at least 2/4 interference,
and novel score strictly exceeds each matched control. Report all outputs and
failure/grade categories, with prior 8/4/4/2 only as static context.

The original absolute twenty-check gate remains distinct: 18/20 does not pass
it. Even 20/20 would be completion of this narrow exposed workload, not general
language, hidden transfer or software-engineering qualification. The finite
literal tape, last-31-byte-plus-EOS query horizon and supplied action motor stay
explicit. Close the fixed run without automatic additional fitting or rule search.
