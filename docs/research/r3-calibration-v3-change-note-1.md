# R3 calibration v3 preregistration — CHANGE NOTE 1 (durability admission refreeze)

**Date:** 2026-08-27
**Status:** **PRE-OUTCOME AMENDMENT — C1 REMAINS NOT RUNNABLE / NO-GO**

This note amends
[`r3-calibration-v3-preregistration.md`](r3-calibration-v3-preregistration.md)
before any calibration-v3 or C1 outcome-bearing execution. It closes two
filesystem-durability ambiguities found during the implementation audit. No
scientific result, held-out datum, threshold decision, or C1 measurement
influenced this amendment.

## Justification

Schema 4 placed output-parent creation inside a fresh active-wall interval and
did not freeze how those parents become durable. It also treated a resume
manifest as the transaction commit point without an admission rule for the
case where immutable final bytes exist but the publishing process could not
confirm their durability. File presence alone is not proof that the file and
its directory entry survived a crash or power loss.

Schema 5 replaces that ambiguous contract. It does not add a receipt sidecar,
retry an immutable publication, or weaken the manifest-last transaction order.

## What changes

1. **Durable directory preparation precedes timing.** For both fresh and resume
   entry, derive every parent directory from the exact 138 frozen artifact
   paths, stopping at and including the artifact root. Prepare the resulting 54
   directories root-first with
   `humanbrain.native_temporal_runtime.prepare_durable_directory(REPO_ROOT,
   relative_directory)` and require
   `TemporalDurabilityOutcome.DURABLE` for every entry before the first
   `time.perf_counter_ns` sample. Missing entries are created durably; existing
   entries are reconciled durably. Any failure is a pre-session STOP. Directory
   creation during active time and publisher-side directory creation are both
   forbidden.

2. **The directory authority is literal and checkable.** Its order is ascending
   path-component count, then UTF-8 bytewise path. Canonical compact JSON of the
   ordered list is exactly 5,305 bytes with SHA-256
   `20be5eb10a661a1ec91c693d336a9e024367cb4a2165285eab8fc6f812fd9055`.
   The machine-readable protocol carries the complete ordered list, count,
   byte length, and digest.

3. **Every resume reconciles the exact manifest bytes before checkpoint
   import.** A resumed active interval still starts immediately before opening
   the manifest. After a bounded stable read and successful decode, pass those
   exact accepted bytes—never a reserialization—to
   `humanbrain.native_temporal_runtime.reconcile_existing_immutable_bytes`.
   Require `TemporalDurabilityOutcome.DURABLE` before checkpoint open or
   import. This file-and-parent reconciliation is charged active time. Manifest
   presence alone is insufficient admission evidence. A same-process
   `PUBLISHED_BUT_NOT_CONFIRMED_DURABLE` publication remains terminal STOP
   without retry; a later
   process may admit the immutable bytes only by completing this exact native
   reconciliation. No persistent admission sidecar is introduced.

4. **Timing contract v2 moves only directory preparation.** A fresh interval
   now begins after successful directory preparation, immediately before the
   first production circuit construction or TRAIN-source access. A resume
   interval begins after the same preparation, immediately before manifest
   open. Manifest read, validation, reconciliation, checkpoint import, index
   rebuild, and parity checks remain charged. This supersedes only the old
   output-parent-creation clause in
   [`r3-persistence-timing-accounting.md`](r3-persistence-timing-accounting.md);
   all cumulative arithmetic, cap, overshoot, and indeterminate-session rules
   remain unchanged.

## Filesystem threat boundary

The C1 persistence protocol covers process crash, power loss, benign concurrent
directory preparation, and symlink or Windows reparse-point traversal and
substitution detection. A malicious concurrent namespace writer that already
has repository write or delete access is outside the C1 threat model. Bringing
that attacker into scope would require a descriptor-relative namespace
capability spanning preparation, publication, reconciliation, and import; a
path-based protocol must not claim that stronger guarantee.

## What does not change

Scientific screens, fixtures, arm order, seeds, thresholds, pass/fail gates,
the 15-hour cap, checkpoint-before-manifest ordering, exclusive immutable
publication, no-overwrite/no-retry policy, claim boundaries, and held-out
discipline do not change. Paths for unselected outputs remain absent even
though their frozen parent directories are prepared.

The machine protocol remains `execution_status=NOT_RUNNABLE_PRE_OUTCOME` and
`source_binding.runnable=false`, with the existing `_execute_frozen_plan`
blocker unchanged. This Change Note is not C1 launch authority.
