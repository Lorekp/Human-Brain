# Local memory/byte v4: precise duration measurement

Status: prepared 2026-09-06; cost and acquisition unrun.

The v3 cost workload completed, but its frozen control's 33-decision capped
prediction was recorded as zero seconds. Python 3.11 on this Windows host uses
`GetTickCount64` for `monotonic`, with 15.625-ms resolution. The original strict
positive-sample projection therefore rejected that receipt as invalid timing.
V3's source, recipe and saved receipts remain preserved; it supplies no cost
admission, full acquisition or capability result.

V4 changes only duration measurement and its identity. The shared harness now
uses `perf_counter` consistently for start timestamps and elapsed differences,
including lessons, capped prediction, memory ingestion, serialization, worker
queries, action packages, progress and finalization. The memory export helper
is moved into the harness because its frozen v1 implementation captured the
old clock. Unix deadlines continue to use `time.time`; the outer launcher's
Stopwatch still measures the full process package. `outer_elapsed_seconds` is
an explicitly separate Unix-wall diagnostic; `duration_clock` describes the
phase and module durations. Clock domains are never subtracted from one another.

The clock's name, implementation, resolution, monotonic and adjustable flags
enter reports and admissions as `duration_clock`. Acquisition requires matching
cost, admission, current process and worker clock metadata. No zero floor,
substituted timing, margin reduction or extra evaluation is permitted. A
deterministic regression makes the actual capped prediction run against a
constant coarse clock and a fine counter; it must retain the fine elapsed time.

All [v3 workload and teaching rules](local_memory_byte_v3.md), five controls,
dimensions, seed, rate, source-removal geometry, action scaffolding, exposure
counts and scientific limitations remain unchanged. Require 8/8 fit plus 4/4
new-value, delayed and interference exact executed tagged outputs, strict
new-value advantage over writer/maps-frozen, frozen and writes-disabled, and
separate aggregate superiority to attribute a benefit to tags. Byte output
comes from the local decoder; host motor matching remains disclosed.

Allow one fresh disposable cost phase of at most 900 seconds total and only
one conditional 900-second acquisition. All attempts within a phase count.
The limit remains 2,048 MiB. Reuse the unchanged prospective formula:

```text
2 * sum_arm(1024 * (maximum complete sampled lesson + capped prediction))
  + 2 * (outer cost elapsed - sum all sampled lessons) + 30 seconds
```

Every sample remains finite and strictly positive, with no cap floor. Cost must
project <=900 seconds and satisfy memory limits before acquisition. Discard
engineering/cost weights. Freeze new source hashes, revision, package/native
identities, teaching schedule, clock metadata and exact command. Preserve all
old artifacts. Cost failure, nonfinite learning, failed fit or missing useful
advantage closes v4 without an automatic changed seed, rate or lesson budget.

Prepare the same admission fields as v3 plus
`duration_clock = harness.duration_clock()`, using the v4 `SPEC`, fresh output
paths and the same native/launcher environment. The exact command is the v3
recipe's launcher command with these four replacements:

```text
admission: .pytest-tmp/local-memory-byte-v4.cost-admission.json
output/log base: .pytest-tmp/local-memory-byte-v4-cost
module: experiments.r4_integrated.local_memory_byte_v4
schema: local-memory-byte-v4
```

Use `cost`, 900 seconds and 2,048 MiB. After saved-only audit and a passing
recomputed projection, bind the new completion and launcher hashes in an
`admitted` record and launch `run` with fresh `local-memory-byte-v4-run` paths.
Acquisition prints completed lesson counts and elapsed seconds every 128 episode
indices; it does no additional evaluation and all output overhead is counted.
