# Direct lag-query addressing: fixed six-arm return

Status: prospective implementation. No cost or acquisition is admitted.
The selected [interface review](../../docs/research/local-lag-query-review.md)
owns this single structural comparison and its conditional descent derivation.

Replace recurrent query compression with a newest-first 32-by-32 raw-byte lag
population. Every query byte and one EOS advances the generic lag circuit. The
last 32 symbols remain; longer questions are explicitly truncated sensory
context. At query end form `z = sqrt(32) * lag / max(norm(lag), 1e-8)` and use one
learned 1,024-by-1,024 projection. Codes, literal tape, stored key normalization,
shared codec, forward branch cursors and causal teacher-prefix filtering remain
unchanged. There is no identifier parser, chosen lag or answer-specific feature.

The local address contribution is `z * h`, where `h = K.T * (p - b_final)`.
Compute the contribution in float64, validate, cast once to float32, and publish
at rate 0.01 with the held-sequence codec proposals. The fixed query population
has no learned query parameters, eligibility, upstream feedback or query update
group. Codec credit remains the existing current-prefix approximation. The
real-arithmetic energy bound controls the held-codec same-query address step;
it is not a floating-point or combined-learning convergence guarantee.

## Fixed comparison and reference custody

The six ordered arms are `learned`, `address_frozen`, `slow_frozen`,
`writes_disabled`, `sequence_reference`, and `byte_reference`. The four candidates
share their actual new tensors and initializer. Address-frozen freezes A while
the codec learns. Slow-frozen freezes both but still stores sensory associations.
Writes-disabled stores zero sensory/key values while retaining counted allocation
and adjacency. No dummy old query tensors or compatibility mode is introduced.

`local_sequence_memory_reference.py` statically preserves the exact sequence
learner at `0b3d7d34f451fbbd315f472aa07732c0cd3eb0e0`.
`local_memory_event_reference.py` preserves the strongest a1ab byte baseline.
Both start fresh, with their original seed 733 initial arrays; neither starts
from a learned endpoint. Their saved initial/final states, 1,024 scientific
lesson receipts and twenty outputs must exactly reproduce prior authenticated
runs, excluding only arm/time and explicitly new accounting. The larger candidate
A is a genuine geometry change, not a tensor-matching fiction. Fixed codes and
codec identity arrays match where genuinely common.

Preserve seed 733, private code seed+2 and address seed+4, rate 0.01, the ordered
1,024 lessons per arm, twelve retained memories, two-byte generation cap, and
the exposed 8/4/4/4 action cohort. Full acquisition has 6,144 first attempts,
18,432 teacher byte/EOS offers, 120 final actions and two motor demonstrations.
All first attempts precede their teachers. Source-free workers receive only
model/state arrays and four-field query requests, never source documents,
targets or selected starts. Delays add query/address traversals without codec
generation or teaching; interference appends only the declared unrelated records.

## Cost and accounting contract

Fresh cost uses four fixed maximum-length lessons per arm: 24 lessons, 72 teacher
offers, six complete-cap predictions, 120 actions and two demos. Each cap ingests
one document containing 127 periods plus EOS, fills all 128 candidate/sequence
slots, forces non-EOS via ordinary temporarily changed output parameters, and
verifies exact slow-weight restoration. No cap teacher or fitted cost state is
retained. Cap ingestion belongs to other measured cost; full-cap prediction is
also projected once per future lesson.

Retain the 900-second aggregate execution ceiling including saved audit and
2,048-MiB process-tree ceiling. Stream every complete scientific lesson, in
index/arm order, to `lessons.jsonl`; no scientific field is removed. Each line is
ASCII-escaped UTF-8 JSON with one LF. The report retains only per-arm scalar
counts, work totals, model and journal time sums/maxima. The row's original
`seconds` remains model/receipt time. A separate journal timer includes recursive
size checks, serialization, write, flush, rolling SHA and scalar accumulation.
The journal is closed before its hash, byte count and row count are bound in both
report and completion. At most the original row and its outer arm/index mapping
are live; each is released before the next model lesson. Readers stream rows too.

For each arm, projected learning is `2 * 1024 * (max_model_seconds +
max_journal_seconds + cap_seconds)`, summed over six arms. Other cost is twice
`outer_elapsed - sum_model_seconds - sum_journal_seconds`, followed by a
30-second allowance. Separate maxima deliberately cover model and logging peaks
even when they occur on different lessons. Journal work scales with the full
1,024 lessons; it is not hidden in the four-sample other-cost allowance.

Both writer and independent reader reject a line above 262,144 encoded bytes or
a row above 262,144 recursive CPython bytes, counting keys and values without
alias deduplication. Their list capacities may differ, so their measured tree
sizes need not be identical. With two live rows and allocation factor two, the
streaming allowance E is 1,572,864 bytes (1.5 MiB). Fixed engineering reserve is
32 MiB for each process. The run estimate is
`max(cost_parent_peak + cost_worker_peak, cost_outer_tree_peak) + E + 32 MiB`.
Cost and full auditors parse the same authenticated historical reports. The
additional full-only checkpoint payload F is 232,400,120 byte-reference bytes
plus 80,431,116 sequence-reference bytes, totaling 312,831,236 bytes. The audit
estimate is `finished_cost_audit_tree_peak + E + 2F + 32 MiB`. Final admission
uses the maximum of these estimates and requires it to fit 2,048 MiB. These are
conservative engineering estimates, not mathematical RSS bounds; the watchdog
remains authoritative. The cost checker's preliminary projection explicitly has
`audit_peak_pending=true` and `admissible=false`. Root authenticates the completed
cost-audit monitor before finalizing admission; the full runner and independent
checker rederive that same projection. No sampling from a full fit, relaxed
criterion or receipt deletion may rescue a failed estimate.
Root owns native preparation, exact `-X utf8`/absolute-path launch, audits and
conditional acquisition. No acquisition begins in this implementation lane.

Report raw bytes/EOS separately from lag transitions, whole-query normalization,
1,048,576 projection-synapse evaluations per candidate query, occupied-start
scores, codec potentials/probabilities, prefix updates and cursor operations.
Each candidate teacher also forms 1,048,576 local address-synapse contributions,
including address-frozen and slow-frozen cases; these are opportunities computed
before publication, not claims that every synapse changed.
Reference recurrent steps/eligibility remain separate actual work. Count codec
and address group offers, enabled publications and numerical changes separately;
the sequence reference alone retains its old query-learning group. Do not store
an unused 1,024-vector query feedback or full A changes per lesson. Hashes and
required compact belief/mixture receipts remain; all copying, validation, external
logging, serialization and audit work stays inside the declared resource scope.
Worker per-output time includes its preceding delay-only query traversals in
this sequence-family harness; package time additionally includes imports, state
restore, validation and serialization. The host motor applies and executes the
declared edit schema after free output; teachers never execute a successful
learner action on its behalf.

Qualification requires learned 8/8 taught, 4/4 novel, 4/4 delayed and 4/4
interference, both novel assignment swaps correct, and strict novel superiority
over every matched control and both references. Also describe different-query
answers and target-odds shifts from existing saved priors, without extra calls or
new thresholds. Literal sensory storage, generic identity copying and the supplied
host occurrence motor do not establish semantic compression or general learned
action selection. On failure close the fixed comparison; no automatic fit follows.

## Ordinary QA before freeze

Run doctor, then reviewed existing decoder-file cases for normalization, lag
order/truncation, independent scalar local A publication, finite boundaries,
held codec timing, controls, snapshots and source-free transport. Preserve old
sequence fixtures on the explicit archive, and retain exact two-reference source
and trajectory parity. Use small authored strings only; no TRAIN/cost/endpoint
model run. Record all attempts, failures, recoveries and subset exposures before
source freeze. The memory-projection helper receives authored scalar QA rather
than a trial acquisition.

Doctor passed at `.pytest-tmp/dev/doctor-ihqa6gxi/receipt.json`. Nineteen distinct
reviewed cases pass across these retained receipts:

- `.pytest-tmp/lag-local-v1.xml`: nine local math/control cases, 2.43 seconds.
- `.pytest-tmp/lag-integration-v1.xml`: ten cases, 3.10 seconds, including the
  repeated scalar oracle after a counter-only addition; exact archived source
  and three-lesson trajectory parity; six authored lesson/cap cases; factory and
  pure journal/projection checks.
- `.pytest-tmp/lag-worker-v1.xml`: one new six-arm fresh-process transport case,
  5.57 seconds. No native action fixture or acquisition workload is executed.

No test failure occurred in this lane. The 20 case executions construct 40 tiny
or standard untrained learner instances, including six child restores, and make
34 authored ingestions: 868 raw document bytes plus 40 EOS symbols. They make
38 ordinary free predictions, 18 accepted teacher calls (54 byte/EOS offers),
six valid query-only traversals and one rejected overflowing-norm traversal.
Across all 63 traversals, 402 raw query bytes plus 63 EOS symbols are processed;
two of those symbols belong to the rejected query. All six cost-cap fixtures
are ordinary authored 127-period tests, independent of cost admission. Pure
journal QA writes 24 synthetic rows and rejects two oversized rows without any
model call. These exposures are QA only; there are zero TRAIN lessons, endpoint
probes, motor demonstrations or executed actions. Old sequence numerical
fixtures now explicitly import the frozen archive; no production compatibility
mode or duplicate historical child workload is retained.
