# V8 cost review — provisional accounting

Status: metadata snapshot only, 2026-09-07. The corrected session is still running.
There is no final runtime, speed comparison or selected optimization here.
The [initial failed launch](../../experiments/r4_integrated/persistent_language_action_v8_initialization_failure_result.md)
remains a separate development cost and exposure record.

## One completed checkpoint prefix

The single captured HEAD selected boundary 23, SHA256
`15ced845980321fa413b789cd440bb175b8b1172394e44a36a8c430289288d5e`.
The [immutable boundary](../../.pytest-tmp/persistent-language-action-v8-corrected-run/checkpoints/boundary-000000023.json)
records 14 resolved attempts and 9 source transactions across the three routes.
The corresponding completed prefix of
[checkpoint telemetry](../../.pytest-tmp/persistent-language-action-v8-corrected-run/checkpoints.jsonl)
contains 72 rows and ends at session elapsed 164.5094462 seconds. Later rows were
excluded. No task outputs, controller choices or tensor payloads were inspected.

| Completed phase | Calls | Recorded seconds |
|---|---:|---:|
| Create | 1 | 0.3221118 |
| Begin | 23 | 9.1434915 |
| Capture frontiers | 24 | 1.0236743 |
| Commit | 24 | 59.2994987 |

These four recorded phase intervals total 69.7887763 seconds. They exclude the
checkpoint telemetry journal's own append intervals and are not the complete
session cost. The boundary's host elapsed 162.1397177 seconds was captured before
its commit completed; it must not replace the later telemetry timestamp.

At this prefix, the store records 18 payload writes, 18 snapshots, 30 unchanged
route-payload reuses and 24 HEAD publications. Payloads are 115,712,128 bytes each;
the two current routes retain identical factual-memory hashes. The work ledger
records 2,083,734,155 bytes written, 7,281,104,920 file-hash bytes read,
23,024,884 bounded metadata bytes read and 41,424,470 event bytes read.
Tensor custody hashing separately processes 15,145,387,920 bytes in memory.
These quantities describe different operations; tensor bytes are not disk I/O.

The Windows lease path records 92 reused checks and 10,645,515,776
`checkpoint_payload_avoided_hash_bytes`. That last quantity is a counterfactual
full-read amount avoided by protected ownership, not measured I/O or time saved.
Live/peak handles both equal 18 here; handle gauges must not be summed across
telemetry rows. The initial attempt to read the live journal with ordinary
file sharing was refused. A read-only stream permitting the existing writer
then captured the completed prefix through the already captured HEAD; this was
an access-mode correction, not repeated progress polling.

## Final aggregation contract

Use the final watchdog for whole-run elapsed time and process-tree peak memory.
Add separately measured audit, banking and failed-launch costs when reporting
the complete development cost. Do not sum process-tree peaks or infer final
latency from this short prefix.

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

All source/frame and state-accounting changes are fixed for this run. Any next
optimization recommendation must follow the final saved timings, actual
category denominators and retained-file costs. This provisional note selects
no new mechanism, benchmark, replay or performance change.

## Source-only feasibility, awaiting final priority

The canonical graph was queried first after verifying its checkout root; it
returned no matching digest/checkpoint symbols. Direct source reads identify
two bounded possibilities. Neither is implemented or selected from the partial
timings.

1. **Avoid the extra tensor-to-Python-bytes copy when hashing owned stable
   storage.** [The tensor digest](../../src/humanbrain/neuronal_splice_learner.py)
   currently calls `detach().contiguous().numpy().tobytes()` for every tensor.
   A byte-cast buffer view can supply the same contiguous bytes to SHA256 while
   preserving the exact dtype/shape/byte-length header and mapping order.
   Keep any necessary contiguity copy, the tensor/array owner alive through the
   synchronous hash, and every before/after identity, mutation and digest check.
   Borrowed mutable storage must not gain a weaker concurrency contract; retain
   the copying path where stable ownership cannot be guaranteed. This removes
   an allocation/copy, not the required full hash scan, and creates no digest
   cache across public boundaries. The
   [lease helper](../../experiments/r4_integrated/persistent_checkpoint_payload_lease.py)
   already uses `memoryview(buffer).cast("B")` for its owned read buffer.
   Compatibility checks would cover exact old digest equality for supported
   scalar, empty, contiguous and noncontiguous tensors, plus mutation rejection.

2. **Potentially compute the expected serialized digest while writing, retaining
   the independent protected read.**
   [Checkpoint commit](../../experiments/r4_integrated/persistent_session_checkpoint.py)
   currently clones a private snapshot, writes/flushed/fsyncs it, rereads the
   temporary file with `_file`, renames it, then hashes the final file through
   the newly acquired lease. A narrow sequential writer could hash the exact
   successfully written archive bytes, replacing only the temporary reread on
   the eligible leased path. The installed PyTorch serialization source accepts
   a buffer writer with `write` and `flush`; that alone does not prove byte or
   write-order parity. Preserve the existing archive bytes, full-write checks,
   size limit, fsync, exclusive creation, rename, final leased physical hash,
   path/identity checks and fail-before-HEAD behavior. Non-leased fallback must
   retain actual file verification. A short write, unexpected seek, mismatched
   final digest or lease failure cannot publish HEAD. This is a possible
   future revision of the temporary-hash implementation, not permission to
   remove the independent handoff authentication or change the selected run.

The private snapshot itself still protects serialization from actor mutation:
`snapshot()` deep-copies model state and clones tape state. Neither possibility
removes that protection, strict schema/finite checks, source guards, frontier
verification or fresh restore reads. Assess their actual benefit only after
the final audit separates hash-copy traffic, serialization and disk verification
costs; neither offers a measured speedup here.
