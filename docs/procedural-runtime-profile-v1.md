# Native procedural runtime engineering profile v1

Status: **preregistered and unexecuted**. No material profile artifact exists.

Raw protocol SHA-256:
`a9639593cb38dfdced12dc0cb0618e712cdec20f7f373d1aec2a7ead497a9533`.
Framed implementation source-bundle SHA-256:
`71a8a0172485ff351e10e0d8577b242b2ef04b2b97868637e3c75bdb51159439`.

This profile asks one bounded engineering question: with the same resident,
event, spike, fanout, and interval-distinct capacities (`R/Q/S/F/D`) and the
same externally injected work, do native work, native payload, and host timing
remain insensitive to the configured uint64 address-space size `N`? It compares
`N=1,000,000` with `N=100,000,000,000`. It does not simulate 100 billion
resident or updated cells.

The complete frozen definition is
`benchmarks/procedural_runtime_profile_protocol_v1.json`. Running
`python benchmarks/procedural_runtime_profile.py` prints a nonwriting plan.
The runner disables Python bytecode-cache writes before plan-time imports.
Execution requires `--execute`, the exact raw protocol SHA-256, and the exact
source-bundle SHA-256 printed by that plan. Execution remains held; this commit
does not run the profile.

## Measurement seam

`NativeProceduralRuntime.advance` returns `native_call_ns`, an integer host
wall-time observation from `perf_counter_ns` immediately around the single C
ABI `humanbrain_procedural_runtime_advance` call. It includes the ABI crossing
and Rust advance implementation. It is not a GPU kernel/device timer.

The runner measures `e2e_ns` immediately around the public `advance` call.
Native library import/build and identity, runtime construction, external-event
construction, process-memory observation, and result validation are outside
both timed intervals. Every row must satisfy
`0 <= native_call_ns <= e2e_ns`; raw nanosecond observations are retained.

The profile is CPU-only. Timed workers do not construct the dense reference,
import a GPU adapter, or report device work. `device_bytes` must be zero.

## Frozen workload and accounting

Both configured sizes use `R=96`, `Q=96`, `S=32`, `F=4`, `D=96`, seed 71,
one-step recurrent delay, and the `uniform_no_self` counter64-v2 topology.

- The quiet control advances 65,536 biological steps with no events. It must
  report zero resident, updated, spiking, delivered-event, generated-edge, and
  cell-step work.
- The driven case injects 16 identical 150,000 pA step-zero events and advances
  256 steps. Its frozen literals are 80 resident/updated distinct cells, 16
  spiking distinct cells and spike events, 20,416 cell-step evaluations, 64
  generated edges, 80 delivered events, and queue peak 64.

Those driven literals are not inferred after a run. The protocol commits the
canonical external-event hash plus the ordered counter64 row and target-union
hash for each configured `N`. Plan validation regenerates both counter64
derivations outside any timed interval and proves that each has 64 distinct
targets and an 80-GID source-plus-target union.

Every worker returns the complete `ScaleAccounting` record, native payload
bytes, retained nonzero count, capacity status, raw timings, and process
working-set/peak/private observations before construction, after construction,
and after advance. The process-memory implementation is the established
`benchmarks.native_scaling._process_memory` helper. Process memory is an OS
observation; native payload is the runtime's deterministic capacity accounting.

`configured_ids_scanned` is deliberately not an empirical profile counter. A
source-bound static regression instead checks that the production constructor
reserves resident/event/spike capacities and `RuntimeScratch`, and contains no
configured-`N` vector reservation or range scan. Configured/addressable cell
counts and configured edge totals are expected to differ with `N`; all
workload-dependent counters must be exactly equal.

## Repetitions and falsifiable gates

The raw protocol fixes a balanced order of 32 rows: two configured sizes, two
cases, and eight repetitions. Each row runs in a fresh subprocess. No CLI flag
can change the workload, order, repetition count, evidence path, or gates.

For each case, the profile passes only if:

- every raw row matches its worked accounting literals with no overflow and
  zero device bytes;
- workload-dependent counters and native payload bytes are exactly equal
  across configured sizes;
- the max/min ratio of cross-`N` median native-call time is at most 3.0;
- the max/min ratio of cross-`N` median end-to-end time is at most 3.0; and
- cross-`N` median private-memory deltas differ by at most 16 MiB.

All raw values are published even if a ratio gate fails. These are conservative
engineering gates, not throughput or capability targets.

## One-shot evidence and source identity

Before creating evidence, the runner requires an exact clean HEAD, the exact
raw protocol hash, the complete framed source bundle, and the portable identity
of the actually loaded native binary: byte size, binary SHA-256, and its source
cache digest. Absolute cache paths are not evidence. HEAD, every source byte,
the loaded binary identity, and Git status are checked before the first worker,
after every worker before publishing its record, before the summary, and after
every final success or failure publication.

After the exclusive fixed evidence root is created, Git status may contain only
the exact immutable files already published by this runner. Dirty paths outside
that root or unexpected files inside it fail closed. The runner retains each
publisher-returned file SHA-256 and rehashes every prior record at every seal, so
in-place changes to an already-untracked file also fail closed. Records use the
repository's atomic nonoverwriting canonical-JSON publisher.

The run-start record commits 32 independent random capability hashes, one per
frozen row. Each raw capability is passed only in its child's environment; a
worker must prove the matching run-start hash, live parent PID, and exact prior
evidence prefix before loading the native runtime. Raw capabilities are never
published. Interruption writes and attests a terminal failure record when
integrity can still be established and permanently consumes the root. Resume,
overwrite, alternate-path rerun, and retuning are forbidden.

## Claim boundary

Even a passing result would support only the bounded hypothesis that this
source-driven working-set implementation is insensitive to configured address
metadata for the registered workload. It would not establish 100-billion-cell
residency, 100-billion-cell updates, bounded biological steady state, token
throughput, software-engineering capability, or parity with Gemma. GPU,
DEV/SWE/canonical/private evaluation, capability evaluation, and Fly Brain are
outside this profile.
