# R3 persistence timing accounting — decision input

**Date:** 2026-08-27
**Status:** decision input for the still-not-runnable R3 calibration-v3 protocol.
**Claim boundary:** this note selects no scientific threshold, source policy,
winner policy, outcome, or C1 launch authority. It changes no protocol, runner,
checkpoint bytes, or artifact namespace. It recommends how a later Change Note
should define the existing 15-hour operational boundary.

## Recommendation

Freeze one metric named **cumulative active wall time**, not an unqualified
“wall clock”:

```text
timing_contract_id = r3-cumulative-active-wall-v1
cap_ns             = 54_000_000_000_000  # exactly 15 h
cumulative_ns      = prior_cumulative_ns + (session_stop_ns - session_start_ns)
```

Measure each session with CPython 3.11 `time.perf_counter_ns()`. A session is a
closed interval. It never pauses internally: scheduler delay, sleep, locks,
garbage collection, synchronous I/O, and a wait for an asynchronous operation
all remain charged. Only differences between counter readings are meaningful;
raw counter values from different sessions are provenance, not quantities to
add.

The contract should use these exact boundaries:

1. **Fresh start:** take `session_start_ns` before the first of production
   circuit construction, production TRAIN-source access, or creation of a
   frozen output parent. Environment installation, native compilation, source
   attestation, and the already-required fixture/cost gates must have completed
   before this point.
2. **Resume start:** take `session_start_ns` immediately before opening the
   exact frozen resume manifest. Charge manifest canonical/hash/pair validation,
   checkpoint opening and reading, full native import validation, all derived
   index rebuilding, identity comparison, and every later operation.
3. **Clean stop:** take `session_stop_ns` only after all scientific work for the
   boundary is complete and, at a checkpoint boundary, after the material
   checkpoint has been exported, durably published, freshly reopened, fully
   imported, had its derived indexes rebuilt, and passed identity/parity checks.
   Then construct and publish the small immutable manifest or terminal result
   envelope outside the metric.

This makes the expensive object that enables recovery part of the 15-hour
budget while treating the small object that reports the budget like post-run
benchmark evidence. A manifest is the commit point, but its publication is not
model-state persistence time.

### What is charged

The single interval total is authoritative. The following operations are all
inside it; optional phase timings are diagnostic subdivisions and may not be
subtracted from the total:

- production runtime allocation/construction and exact TRAIN reads and hashes;
- physical stepping, activity transfer, recognition, race, plasticity,
  evaluation, scoring, stream-hash updates, and final state identity work;
- checkpoint traversal/encoding and any staging copy;
- temporary-file creation, checkpoint bytes written, flush/synchronization,
  collision handling, atomic publication, parent-directory durability, and
  owned-temporary cleanup;
- a fresh open/read plus full checkpoint import, semantic validation, derived
  posting/index reconstruction, and checkpoint/state equality checks;
- waits for an asynchronous save to become durably published; and
- all incidental elapsed time between the session timestamps, including OS
  descheduling, contention, sleep/suspend observed by the selected clock, and
  error handling.

Checkpoint publication must be complete, not merely scheduled or copied into
staging memory. The same full fresh-import/index-rebuild exercise should run at
each frozen milestone on the uninterrupted confirmation branch, even if that
branch continues from its original handle. That makes persistence cost visible
and proves recoverability without granting the resumed branch a special,
cheaper checkpoint path.

### What is not charged

Exclude only work outside those closed intervals:

- dependency installation, native compilation/cache warming, protocol/source
  validation, and separate prerequisite fixtures or cost calibration;
- wall time while no experiment process is active between two **clean,
  manifest-committed** sessions (human approval, scheduler queueing, powered-off
  host, or deliberate harness handoff);
- construction, canonical serialization, hashing, durable publication, and
  post-publication reopen/confirmation of the small resume manifest or terminal
  result envelope;
- console formatting/flushing after the stop timestamp, report rendering,
  submission packaging, Git operations, upload, review, and archival.

No simulation, checkpoint bytes, import/rebuild work, evaluation, or result
calculation may be reclassified as evidence writing. Keep each excluded commit
record bounded (recommendation: at most 1 MiB) and byte-neutral; large traces or
arrays therefore cannot escape the cap by being attached to a manifest.

## Resume and branch rules

- Store `prior_cumulative_ns`, `session_active_ns`, and
  `cumulative_active_ns` in every clean manifest, with exact integer equality
  `prior + session == cumulative`. The first session has prior zero.
- A resume may use only the exact frozen manifest/checkpoint pair. Its local
  interval starts before reading that manifest, and the imported prior total is
  added once. The cap never resets at a book, pass, checkpoint, process, or host
  boundary.
- Report uninterrupted and checkpoint-resumed confirmation totals separately.
  State/spike/hash equality is the determinism criterion; equal time is not.
  When the protocol requires both branches, its conservative timing input is
  `max(uninterrupted_cumulative_ns, resumed_cumulative_ns)`.
- Check the cap within books and immediately before and after every atomic
  persistence operation. If it is crossed during an operation that cannot be
  safely cancelled, finish that operation and cleanup, record the exact
  overshoot, and emit an operational STOP; it cannot become an under-cap PASS.
- An unclean interruption after a session starts but before its next manifest
  makes the cap result **TIMING_INDETERMINATE**: lost monotonic time cannot be
  reconstructed honestly. The last checkpoint can remain diagnostic, but that
  branch cannot claim an under-15-hour result. This avoids silently deleting
  failed work from cumulative time.

The exclusion of inactive time between clean sessions is deliberate. It makes
a planned resume experiment comparable when a human or scheduler starts the
second process later, while still charging the checkpoint and recovery work the
resume mechanism controls. This is a project-specific engineering budget, not
an MLPerf-compliant time-to-train result.

## Why the manifest does not time its own publication

An immutable byte string must be finalized before its duration on stable
storage is known. Putting a receipt after it only moves the same problem to the
receipt; repeating that construction creates an infinite receipt chain. A
fixed surcharge would be an estimate rather than measured wall time, and an
additional mutable timing sidecar would weaken the immutable commit rule.

The simplest closed contract is therefore to exclude **only** the bounded
commit-record publication and confirmation. The manifest records the exact
charged time through successful checkpoint recovery validation, which is the
material cost. No receipt path, `latest` pointer, retry name, or timing sidecar
is needed. Publication duration may be printed as explicitly non-authoritative
operational telemetry, but it must not affect the cap or artifact validity.

## Primary-source conventions

The following are sourced facts; their application to R3 above is an explicit
inference, not a claim that R3 implements MLPerf, PyTorch DCP, or NeMo.

1. **MLPerf Training uses one contiguous wall-clock interval.** Its rules define
   a run result as wall-clock time, include required initialization excess,
   on-the-clock preprocessing, training, and quality evaluation, forbid pausing
   the clock, and permit stopping once target quality is determined. Required
   result logs are a separate submission/review obligation. This supports a
   conservative uninterrupted active interval and a distinct post-run evidence
   layer, although MLPerf itself does not define R3 manifests. ([MLPerf Training
   Rules, §§1.1 and 11](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#11-run-results))
2. **Checkpoint readiness includes storage and recovery, not serialization
   alone.** MLPerf Storage reports checkpoint write and read/recover time; its
   duration metric is the slowest process. A checkpoint must reach stable
   storage, and any remapping or delay before another host can read it is added
   to recovery. This is the strongest direct precedent for charging R3's durable
   publish and fresh import/index rebuild. ([MLPerf Storage checkpoint rules,
   “Metrics and Results Reporting” and “Requirements for Simultaneously Readable
   and Writable”](https://github.com/mlcommons/storage/blob/main/checkpointing/README.md#metrics-and-results-reporting))
3. **Asynchronous initiation is not completion.** PyTorch Distributed
   Checkpoint first stages state and saves it in another thread; the API returns
   a future, and its example waits on `future.result()`. PyTorch separately
   exposes staging and upload-completion futures. ([PyTorch 2.13 Distributed
   Checkpoint `async_save`](https://docs.pytorch.org/docs/2.13/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.async_save))
4. **Production checkpoint systems publish only complete checkpoints.** NeMo
   AutoModel marks a save in progress until all components are saved and the
   checkpoint is published; resume skips interrupted saves. Its async mode waits
   at the next save or clean shutdown to publish, and a terminated job may leave
   even finished background writes unpublished. This supports requiring R3's
   durable publication outcome before the charged checkpoint phase ends.
   ([NVIDIA NeMo AutoModel, “Interrupted Saves” and “Enable Asynchronous
   Checkpointing”](https://docs.nvidia.com/nemo/automodel/nightly/development/checkpointing#interrupted-saves))
5. **Use a monotonic difference clock for elapsed intervals.** Python 3.11
   documents `perf_counter()` as system-wide, highest-resolution, and inclusive
   of sleep, with `perf_counter_ns()` avoiding floating-point precision loss.
   Microsoft documents Windows QPC as independent of system time, suitable for
   interval measurement, and non-decreasing. ([Python 3.11 `time`
   documentation](https://docs.python.org/3.11/library/time.html#time.perf_counter_ns);
   [Microsoft QPC guidance](https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps))

## Decision consequence

If adopted, the later protocol refreeze should replace ambiguous “cumulative
wall clock” prose with the literal contract above, bind the timing-contract ID
and integer fields in resume/result payloads, and keep R3 `runnable=false` until
clean fresh/resumed tests prove the arithmetic and STOP behavior. This note is
research input only and does not authorize those edits or any fixture, outcome,
held-out, R3-F0, or C1 execution.
