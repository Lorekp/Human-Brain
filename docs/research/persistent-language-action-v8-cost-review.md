# V8 cost review

Status: corrected runner, independent audit and banking pass, 2026-09-07.
Runtime `8e8df91d7e7c6889dff15f97d9fec379ebff71bb` completes
192 attempts, 72 source transactions, 6 correction transactions and 132 actual
children. Its watchdog takes 2206.5499429 seconds at 866.890625 MiB observed
process-tree peak. These are execution/resource results, not capability scores.
The [initial failed launch](../../experiments/r4_integrated/persistent_language_action_v8_initialization_failure_result.md)
remains separate development cost and lost exposure; it is not folded into this
corrected-run denominator.

This review reads only the immutable
[report](../../.pytest-tmp/persistent-language-action-v8-corrected-run/report.json),
[completion](../../.pytest-tmp/persistent-language-action-v8-corrected-run/completion.json)
and [checkpoint scalar telemetry](../../.pytest-tmp/persistent-language-action-v8-corrected-run/checkpoints.jsonl).
Their SHA256 values are respectively
`7900847a205301d5ff28ebc10348dbcb02e07b290722d2023c89ce99d6f11f31`,
`93ddeac384e13f56585bc49478e51857ced05adb69fd55e69f5e356722c6f59f` and
`9f128a4e43064133816b317c9a3d592941668a9a15fad271b4916a5fa58f3c9c`.
No tensor payload is read, hashed or decoded here, and no task/teacher trace is
replayed. These full-run records supersede the earlier provisional HEAD23 sample.
Final aggregation additionally uses the
[saved audit](../../.pytest-tmp/persistent-language-action-v8-corrected-review.json)
(`037f65e62c9170205ad063d40dd0e8f9a0069d0fb9e03f7499ff1df9233194bf`),
[result bank](../../benchmarks/persistent_language_action_v8.result.json)
(`87f0c8e39cbbaaed0558d1affb1ca87f52177f4cb6aee0650b74cfcbf06f392a`)
and [outer measurement](../../benchmarks/persistent_language_action_v8.measurement.json)
(`41096da1c22c469f2dae5062715f698bf61eca5f72ff9674b5d8b4ee63aa2a50`).

The corrected run, first-pass audit (104.7858249 seconds) and banker
(16.4619962 seconds) total2327.7977640 seconds. The measurement's explicitly
listed preparation, including failed transport launch/audit/bank and serializer
QA, adds34.5975260: **2362.3952900 seconds**, maximum observed peak866.890625 MiB.
This is the listed V8 preparation/execution ledger, not all historical project
development. Bank preparation15.9941795 seconds is inside banker outer time;
do not add it again. The measurement helper's own execution is outside that ledger.

## Measured whole-run persistence

| Completed phase | Calls | Seconds | File-hash bytes read |
|---|---:|---:|---:|
| Create | 1 | 0.3221118 | 8,684,406 |
| Begin | 264 | 277.9762662 | 57,415,201,187 |
| Capture frontiers | 265 | 133.2704272 | 55,542,722,379 |
| Commit | 265 | 981.9194659 | 214,296,898,693 |

The 795 completed telemetry rows independently sum to 1393.4882711 seconds:
**63.15% of the corrected watchdog interval**. Commit alone is 44.50%.
This is the largest separately timed operation group. It includes validation,
hashing, private snapshots, serialization and publication; those inner components
have no separate clocks, so their individual time shares remain unknown.

The final safe boundary is264 (265 HEAD publications), SHA256
`e2acf10914cab2da891352e97eed4dc1e665e02e54e62508c58f858b3c94fdc4`.
There are 178 snapshots/full route-payload writes and 352 unchanged-route reuses.
The store writes 20,614,184,044 bytes and hashes 327,263,506,665 file bytes;
bounded metadata reads add 287,328,508 bytes and event parsing reads 420,178,380
bytes through separately named counters. These are software-accounted read
passes, not measured physical-device traffic: OS cache effects are unknown.
The completion inventory records 21,038,721,581 artifact bytes before its own
publication; retained artifact size is not cumulative bytes written.

Checkpoint tensor custody hashes process 164,747,745,672 in-memory bytes;
the separate parent custody ledger processes 89,710,984,352 more. Their sum,
254,458,730,024 bytes, is not disk I/O or the whole learner's hash traffic.
Snapshot tensor cloning accounts for 20,592,938,192 bytes and178 additional
legacy validation passes without internal operation instrumentation. Preserve
that limitation instead of describing the counters as exhaustive CPU accounting.

The Windows lease path authenticates 178 versions, records1056 reused checks
and122,192,007,168 avoided full-hash bytes. The latter is a counterfactual byte
amount under the ownership rule, not measured time or physical I/O saved.
All178 leases are released: final live handles0, peak178. Do not sum gauges.

## Timing and exposure boundaries

All four journals together report13.8698972 seconds for encoding, writing,
flush/fsync and row hashing. Checkpoint telemetry's own3.1132860-second append
ledger is inside the checkpoint wrapper's3.1158938-second journal interval;
count either scope once. Both are outside the four checkpoint phase intervals.
Final lease closure is0.0027760 seconds. Disk-budget inventories take50.2166340
seconds across866 checks; final retained-evidence hashing adds15.4181672 seconds.
Initialization is4.4282973 seconds, manifest validation2.4228217 and runtime
source capture0.1294071. These are nested components of the watchdog total,
not costs to add to it.

The report captures2190.1068938 seconds before final inventory. Completion
captures2205.7731723; the external watchdog's2206.5499429 includes remaining
process lifetime. Neither an earlier boundary's host timestamp nor the report
timestamp replaces the whole-run elapsed interval. Audit, banking and failed
launches require their own outer times; do not sum process-tree peaks.

For checkpoint work, sum each non-overlapping completed phase interval once.
Use the final cumulative work counters, or their consecutive deltas, never the
sum of cumulative snapshots. Report telemetry-journal append time, final lease
closure, disk accounting and final evidence inventory separately. Commit's
returned inner timing is nested inside the complete outer commit interval;
adding both would count it twice. Unfinished phases require explicit incomplete
accounting rather than an invented completed interval.

For neuronal work, aggregate canonical top-level source `work`, attempt `work`,
teacher `work` and attempt `ack_work` separately. Nested source-feedback and
payload/teacher receipts repeat portions of those ledgers and must not be added
again. Controller scan/free/teacher tick and draw counts are exposure counts;
old-prior calls and local action work are already inside their enclosing
prediction or correction work. The saved traces do not provide isolated scan,
prior and free-output wall times. Keep parent custody work and checkpoint
validation work in their own ledgers.

Source and teacher timers cover their declared operation intervals. Attempt
elapsed includes its separately recorded checkpoint-begin interval and actual
execution, so it cannot be added unadjusted to checkpoint-begin or child times.
The nested prediction-and-trace and acknowledgement timers explain an attempt;
they are not additional elapsed costs. Preserve these overlaps explicitly.
Likewise, bytes hashed while opening a payload lease are already charged to
file-hash read counters; do not add buffer-allocation sizes as disk traffic.

The audit's scalar aggregates supply the actual category denominators below.
This review has not opened the large attempt/source/teacher journals to create
another trace pass. Both neural routes have64 attempts,24 source observations,
two corrections and44 children; simple has the same transaction counts but no
neuronal prediction/teaching. Source observation is9884 symbols per neural route,
with only368 admitted factual slots:1614→1982. Simple retains all9884 new source
symbols as documents, so its available document history is11498 symbols.

| Actual controller category | Online ticks / draws | Disabled ticks / draws |
|---|---:|---:|
| Source scan | 9,884 / 32,079 | 9,884 / 32,077 |
| Public observation scan | 25,184 / 81,782 | 25,548 / 82,978 |
| Free output | 280 / 909 | 254 / 819 |
| Answer correction | 35 / 117 | 35 / 117 |

Public scans include832 status offers per route (flags plus their EOS),294 other
field EOS, and24,058 online /24,422 disabled content bytes. Their totals exceed
the nominal18,983 because actual workspaces and feedback determine observation
length. Free output has no target offer. Total categorical draws are114,887
online and115,991 disabled; RNG advances in both. The canonical work ledger
records280/254 free old-prior calls and35 teacher-prefix prior calls per route;
these are inside their response/correction costs, not separate complete predictions.
Online publishes controller slow state on24 sources,2 corrections and62
uncorrected acknowledgements; disabled publishes RNG on those same88 resolutions
and zero controller weight updates. Publication counts do not prove useful learning.

| Recorded route interval | Online seconds | Disabled seconds | Simple seconds |
|---|---:|---:|---:|
| Source operations | 40.1005061 | 41.2916442 | 0.0187335 |
| Attempts, including begin | 380.4334995 | 381.3874787 | 79.7248154 |
| Included checkpoint begin | 74.9136022 | 75.3247332 | 77.6060820 |
| Answer correction | 12.8255246 | 13.2159667 | 0.0006246 |

Subtracting the included begin gives305.5198973 /306.0627455 /2.1187334 seconds
for the remaining attempt scopes, still including actual child execution and
resolution. Do not call that pure neural inference latency. These category
timers and checkpoint phases explain the outer interval; adding unadjusted
attempt totals to all checkpoint phases would double-count begin.

## Selected next source design: owned journal prefixes

The canonical graph was queried first after verifying its checkout root; it
returned no matching digest/checkpoint symbols. Known-source fallback and the
immutable scalar telemetry identify repeated journal verification as the main
file-hash byte volume. This selects a source design, not implementation or a
measured speedup.

For each successful boundary n, let J_n be the combined byte length of
`attempts.jsonl`, `ingests.jsonl` and `teachers.jsonl`, with J_0=0. The actual
[`_events` reader](../../experiments/r4_integrated/persistent_session_checkpoint.py:604)
charges exactly the new complete row bytes to `checkpoint_event_bytes_read`.
Summing each commit's delta reconstructs J_n without opening those journals.
The final420,178,380 bytes match their report sizes
(311,424,915 +104,474,286 +4,279,179). Checkpoint telemetry itself is outside
these three authenticated outcome journals.

The call path is fixed:

- [`begin`](../../experiments/r4_integrated/persistent_session_checkpoint.py:752)
  calls `inspect → _frontiers`, hashing the previous prefix once.
- [`capture_frontiers`](../../experiments/r4_integrated/persistent_session_checkpoint.py:483)
  hashes the current prefix once.
- [`commit`](../../experiments/r4_integrated/persistent_session_checkpoint.py:783)
  calls [`_frontiers`](../../experiments/r4_integrated/persistent_session_checkpoint.py:519)
  for current evidence at entry and before HEAD publication, and for previous
  evidence once with `allow_tail=True`.

Across265 commits, ΣJ_n=55,539,382,506 bytes and ΣJ_(n−1)=55,119,204,126.
Thus journal-prefix hashing is **3ΣJ_n +2ΣJ_(n−1) =276,856,555,770 bytes**:
**84.5974%** of327,263,506,665 checkpoint file-hash bytes.
It accounts for55,119,204,126 bytes at begin (96.0011% of that phase's hash
bytes),55,539,382,506 at capture (99.9940%), and166,197,969,138 at commit
(77.5550%). These are source-derived read-byte attributions reconciled to saved
counters; they are not isolated time shares or physical-device I/O measurements.

Current ownership is insufficient for digest reuse.
[`SessionJournal`](../../experiments/r4_integrated/persistent_language_action_v7.py:45)
inherits an ordinary, exposed `path.open("xb")` handle from
[`Journal`](../../experiments/r4_integrated/persistent_language_action_v4.py:42).
Exclusive creation does not prove lifetime exclusion of other writes, truncation
or mappings. No cached prefix may replace current verification on that basis.

The proposed bounded contract covers only those three journals:

1. A new local NTFS journal owns a non-inherited append-only writer from exclusive
   creation to close. Prove that its access/share modes deny external writes,
   truncation, writable mappings, deletion and replacement; retain path/file
   identity checks. Existing payload read leases cannot simply be placed on a
   live writer. No writable handle or mapping may predate the ownership claim.
2. Preserve encoding and exact write counts, flush/fsync, then read and authenticate
   the actual new tail. Advance incremental SHA256 from those verified bytes,
   checking exact EOF, row count and LF. Hashing intended bytes alone is not the
   proposed disk-authentication proof.
3. Keep only current and last-committed frontier receipts. Same-live-store reuse
   requires the actual owner, identity, size and exact requested frontier to
   match. Freeze appends throughout checkpoint capture/publication; prevent
   reentrancy or another thread from advancing the writer during that scope.
4. Advance the committed receipt only after successful atomic HEAD publication.
   Short writes, mismatched tails, unexpected growth, ownership loss or failed
   publication preserve the dirty intent and forbid automatic continuation.
5. Fresh open/restore and unsupported ownership backends retain full prefix
   reads. Keep source bindings, execution-file inventories, private snapshots,
   payload authentication and bounded `_events` parsing unchanged. Count actual
   tail reads, identity checks and avoided prefix bytes separately.

These ownership and publication guarantees still require source review and
ordinary refusal/parity QA; they are not established by this cost calculation.
This is the single next persistence optimization design, without changed payload
serialization, weaker live-tensor checks or a V8 replay.

The earlier tensor buffer-view option is **not selected**: dominant commit
digests use shallow live `state_dict()`/tape aliases, and idle checks or tensor
versions do not exclude NumPy mutation. Proven private restore tensors and new
contiguous temporaries miss that dominant path. Keep its copying fallback;
do not add redundant snapshot digests to manufacture a use case. Serialized
write-time hashing also remains unselected. A future measured check must
establish benefit. V7 and V8 differ in learners, observations, journals and
transaction counts, so their percentages establish no matched speedup.
