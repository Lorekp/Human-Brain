# Local memory/byte development: cost decision

2026-09-06. The full fit was **not admitted**. The fixed configuration projects
to **2,013.103 seconds**, above its 900-second limit. This is a runtime rejection,
not evidence that the local learner cannot acquire the task.

The frozen source and [recipe](local_memory_byte_v1.md) are commit
`35e2808452cf93e2e8485ea9ce24a32d66de078f`.
The [cost decision](../../benchmarks/local_memory_byte_v1.cost_decision.json)
records the prospective formula, measurements, artifact hashes and audit.

The disposable check completed four sequential lessons per each of five arms:
20 first byte attempts, 200 teacher byte/EOS offers, 160 enabled decoder-update
opportunities, then a fresh source-free process and 100 real Python actions plus
two motor demonstrations. The checkpoint contains 124,109,512 tensor bytes
across all five models and eight memory pools per model. Its weights are not
reused. No acquired capability, execution-feedback learning or recovery is claimed.

| Arm | Slowest complete sampled lesson | Additional capped prediction charge |
| --- | ---: | ---: |
| Tagged | 0.234 s | 0.015 s |
| Aggregate | 0.094 s | 0.016 s |
| Writer/maps frozen | 0.204 s | 0.016 s |
| Slow weights frozen | 0.203 s | 0.015 s |
| Writes disabled | 0.125 s | 0.016 s |

The successful disposable workload took 34.228 seconds. Conservative summed
parent/worker peaks were 805.398 MiB; the launcher's sampled process-tree peak
sum was 822.344 MiB, below 2,048 MiB. Runtime, rather than memory, rejected the
full fit. The declared two-times timing margin and additional capped-prediction
charge remain unchanged after measurement.

One earlier attempt failed native identity admission before any lesson or
action. Only `PATH` differed between the hashed build environments; compiler,
SDK, command and source location matched. Preparing the identity in the same
environment as the launcher resolved it. Both attempts and diagnosis remained
within the original cost-phase deadline: 411.674 seconds from first start to
successful completion. The earlier failed receipt remains preserved.

The saved-only independent audit passed: source/artifact hashes, initial/frozen
weights, checkpoint chains, shared two-key and delay memories, source-free state
schema, all 102 execution receipts and the 100 action workspaces were checked.
It did not rerun the learner. The Windows virtual-environment redirector PID
differs from actual Python; both appear in the recorded owned process tree.
The frozen runner's unexecuted full-fit admission compares those PIDs directly,
so any successor must correct that predicate before reuse.

Keep this configuration closed. Profile the measured local ingestion cost before
choosing a successor, without changing the learning rule merely to obtain a fit.
No additional seed, lessons or relaxed timing margin is admitted by this result.

A separate 2.43-second ingestion-only profile used one warmup and three measured
ingestions per tagged/aggregate arm, totaling 512 byte/terminator steps and no
teacher bytes. Instrumented means were 152 ms tagged and 43 ms aggregate.
Tag arithmetic and reductions dominate; finite checks account for about 20%,
while explicit cloning is below 0.5%. This narrows the next implementation work
to arithmetic buffers and redundant validation, without proving a speedup.
The decision JSON links the profile receipt; these timings do not replace the
original complete-lesson cost estimate.
