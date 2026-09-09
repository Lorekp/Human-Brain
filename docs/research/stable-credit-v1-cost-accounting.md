# Stable credit V1 cost accounting

2026-09-09. Timing scopes reviewed against frozen comparison source
`5622f600921e6811d66254e25c8e956a31cd19ab`. The matched journal-path result
below is complete. Both workers and all scheduled probe callbacks completed and
the [final saved-only audit](../../.pytest-tmp/stable-credit-v1-comparison/review-final.json)
passes. The original outer run failed in its first saved audit; its failed status
is preserved. Post-run reporting corrections and their costs remain separate.
Supplemental diagnostic extraction also completed. This document adds no
execution, model access or experiment. The
[measurement bank](../../benchmarks/stable_language_action_credit_v1.measurement.json)
provides the consolidated measurement and receipt bindings.

The [stable comparison protocol](stable-language-action-credit-comparison-v1.md)
owns the workload and resource limits. The source references for the accounting
map are the [worker](../../experiments/r4_integrated/stable_language_action_credit_v1.py),
[probe callback](../../experiments/r4_integrated/persistent_language_action_credit_probes.py),
[initializer](../../experiments/r4_integrated/persistent_language_action_credit_initialize.py)
and [coordinator](../../experiments/r4_integrated/stable_language_action_credit_comparison_v1.py)
at the revision above. Timers use `time.perf_counter`.

## Matched journal-path measurement

The [completed result](owned-journal-prefix-cost-v1-result.md) and its linked
saved receipts preserve the separate measurement at source revision
`195bc0e4dcf8ab2d1c65d28170f93b415265030f`. Both lanes wrote the exact same
270 saved V8 rows, totaling 420,178,380 bytes, at all 265 saved boundaries.
All row coordinates, lengths and SHA256 values, boundary frontiers and fresh
final frontiers matched. The schedule checks the previous frontier before
append, the current frontier at capture, commit entry and prepublication, and
the previous frontier with an allowed tail during commit: three current checks
and two previous checks where a previous boundary exists.

| Scope or software counter | Ordinary | Owned |
|---|---:|---:|
| Complete scheduled journal path, seconds | 369.0633679 | 24.3657044 |
| Included append interval, seconds | 9.1783096 | 8.7093226 |
| Separate fresh final verification, seconds | 0.4833017 | 0.5635055 |
| Scheduled full-prefix hash bytes read | 276,856,555,770 | 0 |
| Scheduled full-prefix passes / authenticated prefix reuses | 3,969 | 3,969 |
| Actual new-tail bytes authenticated | — | 420,178,380 |
| Avoided scheduled prefix-hash bytes | 0 | 276,856,555,770 |
| Fresh final bytes read / full passes | 420,178,380 / 3 | 420,178,380 / 3 |
| Actual bytes written | 420,178,380 | 420,178,380 |

Append time is contained in scheduled-path time. Fresh final verification is
outside that timer and occurs after writer closure. Each lane also reads
420,178,380 saved row bytes to supply its writes. The ordinary lane records
270 writes and 270 fsync calls. The owned lane records 270 native write-through
completions, 578 actual tail reads, 8,481 identity checks and 1,587 freeze scopes;
all three writers close. Write-through is not counted as fsync. These byte
counters measure software reads and writes, not physical-device traffic.

The observed scheduled-path ratio is 15.1468 times, with 344.6976635 fewer
seconds in the owned lane. It is one serial ordinary-then-owned pair without
counterbalanced repeats or controlled OS cache state. The later lane may benefit
from prior reads. This establishes a reduction for this selected path under
those conditions; it establishes **no end-to-end checkpoint, worker or learning
speedup**, nor a general latency ratio. Tensor hashing, actor snapshots, payload
serialization and learning/action work are outside the measured schedule.

Separate authentication takes 1.7873505 seconds and reads 456,401,339 bytes in
1,073 passes. Internal elapsed time is 396.2719530 seconds through final checks,
excluding receipt publication. The outer receipt records exit zero,
396.7159483 seconds and 97.71484375 MiB, within 900 seconds and 4096 MiB.
The outer memory value is the largest sampled simultaneous sum of live owned
process peak working sets at 500 ms intervals; brief processes can be missed.
Both lane copies retain 840,356,760 bytes before receipt publication, within
the 1-GiB cap. These are separate preparation costs for the stable comparison.
No models, tensor decodes or task children run in this measurement.

## Worker timing overlap map

| Recorded interval | Included work | Boundaries and overlaps |
|---|---|---|
| Source row elapsed | Source observation, custody validation and hashes, feedback conversion | Starts after the append transaction's `begin` wrapper. Ends before ingest journal append; excludes subsequent capture and commit. |
| Attempt row elapsed | Observation preparation, custody hashes, entire `begin` wrapper, prediction/trace, actual execution/grading, acknowledgement and after-state controls | Initial per-attempt admission is before this clock. Ends before attempt journal append, teaching, host cursor bookkeeping, capture and commit. Prediction/trace, execution and acknowledgement diagnostics are nested. |
| `included_checkpoint_begin_seconds` | Observation encoding/hash plus the full `begin` wrapper | Nested in attempt elapsed. Includes both wrapper admission checks, store begin, checkpoint telemetry append and wrapper overhead; broader than the checkpoint begin phase alone. |
| Teacher row elapsed | Correction, feedback/receipt conversion, state/control validation and hashes | Starts after the attempt append and complete-file check. Ends before teacher journal append. Does not overlap the recorded attempt interval. |
| Main outcome journal `append_seconds` | Row encoding, write-through or fallback fsync, actual tail authentication and ownership checks | Outside the main source, attempt and teacher row clocks, which stop before their appends. |
| Checkpoint create/begin/capture/commit phases | The corresponding store operation and small phase bookkeeping | Begin starts after its first admission check. Phase time stops on entry to `checkpoint_event`, before telemetry row assembly and append. Store-returned commit timings are nested diagnostics. |
| Checkpoint telemetry `journal` | Telemetry append | Separate from the checkpoint phase clocks. Excludes row assembly before its append timer. Begin telemetry is nevertheless inside an attempt's included begin wrapper. |
| Outer checkpoint ownership scope | Entry ownership/freeze checks, capture, commit, scope exit checks | Entry occurs before the capture tick; final exit checks occur after commit timing. Entry/exit cost is absent from the phase sum. |
| Global disk-accounting seconds | Recursive artifact stat and free-space query inside `admit_work` | Some calls lie outside row/probe clocks; others lie inside attempt begin wrappers or probe callbacks. The global total is not an additive exclusive category. |

For an additive attempt category, subtract the recorded
`included_checkpoint_begin_seconds` from attempt elapsed before allocating the
begin wrapper elsewhere. Subtracting only the checkpoint begin phase leaves
overlapping admissions and telemetry. The receipts do not allocate every disk
check or wrapper gap to its enclosing interval, so they do not support an exact
fully exclusive partition. Unassigned time must not be labeled pure neuronal
compute.

## Probes, initialization and publication

| Recorded interval | Included work | Excluded or nested work |
|---|---|---|
| Probe callback elapsed | Admissions, fresh store open/restore, two disposable actor constructions, qualification, all 84 attempts and 51 actual executions, probe appends, CP0 comparison, cleanup/GC and destruction receipt, final full probe journal hash, final admissions and partial summary assembly | Starts after directory creation. Restore and probe-attempt timers are nested. Ends before `summary.json` publication. Caller live-state checks, caller admissions and checkpoint marker publication are outside. Probes restore without a main checkpoint capture/commit. |
| Probe attempt elapsed | The attempt and its preexecution admission when applicable | Starts after the first per-attempt admission. Ends before semantic signature, later admission and journal append. Per-route initial numeric hashes are outside attempt clocks but inside callback elapsed. |
| Initializer elapsed | Historical authentication, restricted payload decode, two actor constructions and copy restoration, baseline restoration, final historical verification | Excludes earlier module/runtime selection and store creation, later origin checks, initialization receipt publication and qualification. Some return-copy work follows the elapsed field. |
| Worker report elapsed | Work since runner entry, including main stream, probes, cleanup and final metric/hash assembly up to the elapsed field | Excludes report serialization/fsync, final source rehash, final file inventory, final disk checks and completion publication. |
| `final_inventory_seconds` | Full output-file hash inventory, including report, probes and payload files | Excludes preceding source rehash and subsequent final disk checks/completion. |
| Worker completion elapsed | Runner entry through final inventory and disk checks | Sampled before completion receipt write/fsync and final idempotent wrapper cleanup. Contains report elapsed and final inventory elapsed. |
| Coordinator worker phase elapsed | Log opening, process launch, interpreter startup/imports, worker execution, shutdown, wait and log closure | Excludes the coordinator's phase receipt publication, old CP0 reference hashing and interworker gaps. Contains worker report/completion intervals. |
| Coordinator saved-audit phase | Saved-only review and `review.json` publication | Review's own elapsed value is nested. |
| Coordinator comparison elapsed | Coordinator work up to its final receipt fields | Excludes final `comparison.json` serialization and publication. Its outer-accounted value also stops before that publication. |
| External monitor elapsed | Entire owned process tree, including coordinator startup, both launches, gaps, audit, publications and shutdown | This is the complete end-to-end timing boundary. It must come from the final monitor receipt. |

Main store work, disposable restore work and final full inventory reads have
different scopes. A counter absent from one receipt is unavailable there, not
zero. Present each counter with its owner and read scope; do not add cumulative
snapshots or nested timers.

## Audited completed worker costs

These figures come from the small saved [old report](../../.pytest-tmp/stable-credit-v1-comparison/old/report.json),
[causal report](../../.pytest-tmp/stable-credit-v1-comparison/causal/report.json),
their [old completion](../../.pytest-tmp/stable-credit-v1-comparison/old/completion.json)
and [causal completion](../../.pytest-tmp/stable-credit-v1-comparison/causal/completion.json)
receipts, and the [old launch](../../.pytest-tmp/stable-credit-v1-comparison/old-launch.json)
and [causal launch](../../.pytest-tmp/stable-credit-v1-comparison/causal-launch.json)
receipts. Both launches report exit zero. Seconds in the tables are rounded to
seven decimal places; the linked receipts preserve the original float values.
No payload or journal scan was performed to compile this section.

| Recorded timing, seconds | Old | Causal |
|---|---:|---:|
| Coordinator worker launch phase | 2910.7668660 | 3540.3105756 |
| Worker completion elapsed | 2910.2689351 | 3539.7718551 |
| Worker report elapsed | 2893.9214096 | 3523.4847231 |
| Initialization, inside worker | 7.1222096 | 6.2488978 |
| Final full file inventory, outside report elapsed | 16.0080966 | 15.7802560 |
| Checkpoint create | 0.4091986 | 0.3291608 |
| Checkpoint begin | 198.9998804 | 209.5486606 |
| Checkpoint capture | 61.7244867 | 65.8267867 |
| Checkpoint commit | 752.7904020 | 779.7542138 |
| Checkpoint telemetry journal | 2.0661448 | 2.0151227 |
| Sum of those five checkpoint phase counters | 1015.9901125 | 1057.4739446 |
| Store close, outside phase sum | 0.0035540 | 0.0039178 |
| Global disk accounting, overlapping other intervals | 167.9750342 | 527.1731721 |

Each worker records 1,774 disk admission checks, 192 main attempts, 132 actual
main executions, 72 source records, six teacher records and 265 safe boundary
publications. These counts describe the completed worker receipts; they do not
turn the failed original outer invocation into a successful audited comparison.
The passed final audit authenticates these worker receipts and their underlying
saved evidence. The larger causal disk-accounting total is an observed interval, not isolated
learner compute or a demonstrated causal effect of the local rule. Both workers
use owned journals, so their duration difference is not an ordinary/owned
journal comparison.

The auditor's `saved_costs` sums actual main row clocks by route. The last column
subtracts only the recorded included begin wrapper; it still includes custody,
prediction/trace, execution, grading and resolution work, rather than isolating
neuronal compute. It must not be added to the inclusive attempt column.

| Rule / route | Source seconds | Teacher seconds | Inclusive attempt seconds | Included begin seconds | Attempt minus included begin, seconds |
|---|---:|---:|---:|---:|---:|
| Old / online | 38.3045259 | 13.1489629 | 351.8478040 | 60.0405560 | 291.8072480 |
| Old / disabled | 39.2738598 | 12.7679434 | 360.8368464 | 60.2098657 | 300.6269807 |
| Old / simple | 0.0217929 | 0.0011216 | 64.5735350 | 62.6279739 | 1.9455611 |
| Causal / online | 47.0854968 | 13.3028334 | 425.6850268 | 87.9329072 | 337.7521196 |
| Causal / disabled | 49.0337424 | 13.4323570 | 436.5745137 | 90.4372909 | 346.1372228 |
| Causal / simple | 0.0225935 | 0.0010681 | 91.4748758 | 89.5225823 | 1.9522935 |

### Main checkpoint and journal work

These are final main-store counters after closure, not the earlier cumulative
snapshot embedded in `last_safe`. Neither table includes disposable probe stores
or the final independent output inventory.

| Main-store software counter | Old | Causal |
|---|---:|---:|
| Outcome rows authenticated / write-through completions | 270 / 270 | 270 / 270 |
| Actual outcome bytes written and tail bytes read | 519,642,399 | 547,276,430 |
| Actual tail read calls | 675 | 697 |
| Avoided repeated journal prefix-hash bytes | 345,641,496,717 | 361,816,861,795 |
| Authenticated journal prefix reuses | 3,969 | 3,969 |
| Journal identity checks | 23,565 | 23,565 |
| Journal freeze scopes / writer closes | 1,587 / 3 | 1,587 / 3 |
| Store file-hash bytes read, all included file categories | 50,876,009,189 | 50,801,488,830 |
| Store file-hash passes | 551,939 | 549,821 |
| Separate event bytes read | 519,642,399 | 547,276,430 |
| Store bytes written | 20,619,713,243 | 20,619,731,736 |
| Store fsync calls | 975 | 975 |
| Payload writes / authenticated versions | 178 / 178 | 178 / 178 |
| Payload reuses | 352 | 352 |
| Payload lease releases / remaining live handles | 178 / 0 | 178 / 0 |

Main-store file-hash bytes include source, payload and other store authentication;
they are not a journal-only fresh-read counter. Main reports do not expose a
separate fresh/full journal-check counter. Absence does not establish zero fresh
verification: the disposable restore receipts below record it explicitly.
The ordinary checkpoint telemetry journal separately writes 795 rows per worker,
5,573,740 old bytes and 5,573,812 causal bytes. Its timing is already represented
in the phase table, rather than added again from its journal receipt.

The audit also reconciles `checkpoint_between_phase_work`. Phase-start snapshots
were not saved: each `work_delta` begins after the preceding telemetry snapshot,
so their difference includes real work outside those phase counters. The
reconciliation uses authenticated row byte lengths and the frozen native writer
contract, rejects unexpected counters or gaps, and does not invent an elapsed
time for the gap.

| Audited work between phase counters | Old | Causal |
|---|---:|---:|
| Outcome rows / write-through completions | 270 / 270 | 270 / 270 |
| Outcome bytes written / actual tail bytes read | 519,642,399 | 547,276,430 |
| Tail read calls | 675 | 697 |
| Freeze scopes | 795 | 795 |
| Identity checks, opens and closes, each | 2,919 | 2,919 |
| Metadata queries | 23,352 | 23,352 |
| Read buffer bytes allocated | 180,894,924 | 181,986,784 |
| Write buffer bytes allocated | 519,642,669 | 547,276,700 |

These counters are subsets of main-store totals above, not additional work.
Read/write buffer allocation bytes are not additional file I/O. Final writer
and payload-handle cleanup remains in final-store totals, outside this
between-phase reconciliation. The [audit correction record](stable-credit-v1-audit-correction.md)
documents the original mistaken phase-accounting assumption and its correction.

### Scheduled disposable probe receipts

The four summaries per worker are [old CP0](../../.pytest-tmp/stable-credit-v1-comparison/old/probes/cp-0/summary.json),
[CP16](../../.pytest-tmp/stable-credit-v1-comparison/old/probes/cp-16/summary.json),
[CP32](../../.pytest-tmp/stable-credit-v1-comparison/old/probes/cp-32/summary.json),
[CP48](../../.pytest-tmp/stable-credit-v1-comparison/old/probes/cp-48/summary.json)
and [causal CP0](../../.pytest-tmp/stable-credit-v1-comparison/causal/probes/cp-0/summary.json),
[CP16](../../.pytest-tmp/stable-credit-v1-comparison/causal/probes/cp-16/summary.json),
[CP32](../../.pytest-tmp/stable-credit-v1-comparison/causal/probes/cp-32/summary.json),
[CP48](../../.pytest-tmp/stable-credit-v1-comparison/causal/probes/cp-48/summary.json).

| Probe callback elapsed, seconds | Old | Causal |
|---|---:|---:|
| CP0 | 267.8849824 | 313.2024911 |
| CP16 | 253.4864510 | 336.6603850 |
| CP32 | 270.2285554 | 356.7514714 |
| CP48 | 280.2460207 | 365.2149830 |
| Sum, already inside worker report elapsed | 1071.8460095 | 1371.8293305 |
| Included restore elapsed sum | 29.1242363 | 29.6630706 |

Each callback records 84 attempts, 51 actual executions, two new disposable
actors, one restore call and two tensor-load calls, with all actors/reference
released and no cleanup errors. Across both workers this is 672 probe attempts,
408 probe executions, 16 disposable actors and 16 probe tensor decodes. The
[old initialization](../../.pytest-tmp/stable-credit-v1-comparison/old/initialization.json)
and [causal initialization](../../.pytest-tmp/stable-credit-v1-comparison/causal/initialization.json)
receipts each record two main actor constructions and one retained-payload
decode. Thus these workers report **20 actor constructions and 18 tensor
decodes**, counting both initialization and all scheduled probes. They report
1,056 total main-plus-probe attempts and 672 actual executions. Construction and
decode counts are distinct; each initializer copies one decoded retained state
into two independent actors.

| Sum over four callbacks, per worker | Old | Causal |
|---|---:|---:|
| Probe journal bytes written / actual tail bytes read | 515,604,208 | 543,386,448 |
| Probe tail read calls | 720 | 743 |
| Probe authenticated rows / write-through completions | 336 / 336 | 336 / 336 |
| Probe writer closes | 4 | 4 |
| Restore fresh journal checks / full journal checks | 24 / 24 | 24 / 24 |
| Restore fresh payload checks / full payload checks | 16 / 16 | 16 / 16 |
| Restore file-hash bytes read, all included categories | 4,061,125,056 | 4,165,636,832 |
| Restore file-hash passes | 5,428 | 5,404 |

Fresh and full counts here label the same fresh full-verification operations;
they must not be summed as distinct reads. Each callback also performs a final
full hash of its completed probe journal. From the recorded final journal sizes
and that frozen source path, these four final reads total 515,604,208 old bytes
and 543,386,448 causal bytes, separately from the tail reads. This is a
source-derived accounting of the recorded sizes, not an additional measured
counter or a new read performed for this document. The final independent audit
inspector counters below are separate from these live probe restores.

### Original outer failure and saved disk snapshots

The [original outer monitor](../../.pytest-tmp/stable-credit-v1-comparison.launch.json)
records **failed**, exit one, 6472.8048409 seconds and 1453.3203125 MiB.
Both worker launches completed first; the initial saved auditor then failed
with `KeyError: 'grade'` because main reading rows do not contain the edit-only
grade field. This elapsed interval includes that failed audit and coordinator
shutdown. It excludes later diagnosis, preparation, focused QA and the corrected
saved-only audit. The memory measure is the maximum sampled simultaneous sum
of live owned process peak working sets at 500 ms intervals; brief unobserved
children may be missed. It is not a sum of independently attained process peaks.

| Saved comparison-root disk snapshot | Old worker point | Causal worker point |
|---|---:|---:|
| Last report admission: artifact bytes | 21,661,618,522 | 43,379,287,707 |
| Last report admission: filesystem free bytes | 38,035,922,944 | 16,897,273,856 |
| Completion: artifact bytes before completion publication | 21,661,693,602 | 43,379,362,555 |

The last causal completion snapshot is approximately 40.4002 GiB of artifacts;
its earlier last admission records approximately 15.7368 GiB free. These are
different sampling points. They are not a final post-audit disk inventory, and
the old completion figure is not a standalone old-directory size. Both cover
the comparison root at their respective times. No current recursive inventory
was run for this document.

## Passed final audit and separate reporting costs

The final [saved review](../../.pytest-tmp/stable-credit-v1-comparison/review-final.json)
passes with SHA256
`1bd3073fac22fdd54cc27e16ad7c8e75d106b15e9263e44bb1161b1a85c54919`.
Its corrected auditor SHA256 is
`116e77e187749b10812fda30bfbc6f770492e55cc4ea46861cf283a8bd650423`;
the execution revision remains the frozen `5622f60` revision above. The
[final audit monitor](../../.pytest-tmp/stable-credit-v1-final-audit.launch.json)
reports exit zero, **256.7525935 seconds and 115.03125 MiB**, within its separate
650-second/2048-MiB limits. Internal review elapsed is 255.2342021 seconds,
within the 600-second review ceiling. That internal interval is contained in
the outer interval. The audit constructs zero models, decodes zero tensors,
calls zero predictors and runs zero task children or teacher replays.

Within the final review, the existing byte-only store inspector performs these
fresh final checkpoint checks. Its timer and work are subsets of final audit
cost, not additions to that elapsed interval.

| Final audit inspector scope | Old | Causal |
|---|---:|---:|
| Inspector elapsed, seconds | 3.7269827 | 3.2418377 |
| File-hash bytes read | 769,347,258 | 796,841,406 |
| File-hash passes | 987 | 983 |
| Separate metadata bytes read | 693,157 | 692,116 |
| Fresh journal checks / full journal checks | 3 / 3 | 3 / 3 |
| Fresh payload checks / full payload checks | 2 / 2 | 2 / 2 |

As with probe restores, fresh and full counters describe the same operations.
The inspector is only one audit read scope. Complete saved inventories,
independent boundary/payload authentication and journal parsing also perform
real reads; inspector counters do not represent total audit I/O.

| Saved reporting attempt | Internal review seconds | Outer seconds | Outer observed peak MiB | Outcome |
|---|---:|---:|---:|---|
| Initial audit inside original comparison launch | 18.7546711 | Included in 6472.8048409 | Included in 1453.3203125 | Failed reading-grade lookup |
| First corrected audit | 86.3544813 | 88.0430764 | 101.90625 | Failed phase-work assumption |
| Phase-corrected audit | 95.5507661 | 97.3054548 | 101.5390625 | Failed missing explicit probe-reader argument |
| Final corrected audit | 255.2342021 | 256.7525935 | 115.03125 | Passed |

The [audit correction record](stable-credit-v1-audit-correction.md) preserves
each failed result and the additional source-preparation, diagnosis and focused
QA receipts. Those costs are not erased by the final pass. Their monitored
intervals remain separately attributed; no sum here combines nested intervals
or treats diagnosis as learner work. The final review revalidates both workers
and all probes, without reusing the direct probe diagnostic as a shortcut.
The separately monitored supplemental diagnostic extraction is outside all
intervals above; its completed cost follows.

## Completed supplemental diagnostic extraction

The [supplemental report](../../.pytest-tmp/stable-credit-v1-comparison/diagnostics.json)
completed on its first extraction attempt, with 2,780,209 bytes and SHA256
`23f574a1a30e21c6bd18ef12219b52721fcfcaa3f23226a53f5d780987cca0a2`.
Its [publication receipt](../../.pytest-tmp/stable-credit-v1-diagnostics.stdout)
records 34.3938003 internal seconds, successful output-size and postwrite disk
checks, and the [outer monitor](../../.pytest-tmp/stable-credit-v1-diagnostics.launch.json)
records exit zero, **34.9028742 seconds and 78.765625 MiB**, within 600 seconds
and 2048 MiB. Internal time is contained in the outer interval. The report fits
its 128-MiB output limit. This extraction reads saved diagnostics without model
construction, tensor decoding, task execution or additional teacher exposure.

| Confirmed extraction read scope | Bytes |
|---|---:|
| Journal index pass | 2,125,909,485 |
| Journal row parse pass | 2,125,909,485 |
| Metadata | 4,667,045 |
| Parser source closure | 443,026 |
| Total confirmed reads | 4,256,929,041 |

The extraction reports exact completed read accounting, 1,212 parsed saved rows
and 808 neural cases. These two real journal passes are supplemental analysis
cost; they are not new learner observations or additional checkpoint work.
The byte total measures software reads, not physical-device traffic.

Immediately before publication, its comparison-root snapshot records
43,384,670,805 artifact bytes and 14,685,614,080 free bytes. After publication,
it records **43,387,451,014 artifact bytes and 14,682,828,800 free bytes**, within
the 46-GiB artifact cap and 8-GiB free reserve. This is the diagnostics
postwrite snapshot, not a later measurement-bank/publication inventory. These
figures were transcribed from the small completed receipts; no additional
journal or payload read was performed to update this document.

## Final reporting boundary

The passed audit validates saved worker evidence; it does not change the
original outer launch's failed status or convert a post-run reporting correction
into part of the frozen launch. External elapsed and memory, worker launch/report/
completion timings, probes, checkpoint phases and final inventory retain the
separate scopes above. Artifact counts must name
their sampling point: `artifact_bytes_before_completion` covers the comparison
root at that moment and excludes the imminent completion receipt and later
coordinator outputs.

The fixed resource contract allows 10,800 seconds overall, with the two workers
sharing 10,200 seconds and 600 seconds reserved for saved audit. The 46-GiB new
artifact cap includes a 2-GiB pending-write reserve; it is not an extra allowance.
Admissions require used bytes plus that reserve within 46 GiB and at least
8 GiB free plus that reserve before new work. Postwrite checks and coordinator
receipt-size checks remain in force. These bound known parent/retained writes;
they are not a filesystem quota against unrelated writers.

Historical V8 and the stable comparison have different workloads and evidence
costs. Their elapsed differences cannot isolate the journal change or local
credit rule. The matched journal measurement above supplies the narrow storage
comparison; the audited stable run supplies behavior and its complete observed
costs.
