# Typed-program public DEV gate v1

**Status:** executed once; promotion failed at 0/2 pass@1; live runner retired

The raw protocols, committed TRAIN artifact, DEV evidence, and result report
remain immutable historical records. Layer-1 v4 deliberately removed the v3
checkpoint loader and both v1 execution CLIs. The bound historical commits are
the reproducibility anchors; DEV v1 will never be rerun or used for tuning.
The current unexecuted increment is specified in
[`typed-composition-a012-v1.md`](typed-composition-a012-v1.md).

This is the one-shot public development gate for the frozen typed-program
TRAIN result. It opens exactly the two teacher-free development definitions
already committed in `typed_program_public_train_protocol_v1.json`. It does
not grow or retrain a circuit and has no task, checkpoint, configuration,
seed, retry, repair, search, fallback, or evidence-path selector.

The exact raw SHA-256 of
`benchmarks/typed_program_public_dev_protocol_v1.json` is
`c7be743617c5fbc090891275f77a636bf1a7a6a89d27a1d1e7cf0e889d19ae56`.

## Frozen inputs

| Input | Identity |
| --- | --- |
| TRAIN source commit | `1ae5418b92e81dc48c6a2e36fb8822ceec7ce202` |
| Committed TRAIN artifact commit | `8291fe1a0039db7141817c438854a1817a0a8300` |
| TRAIN protocol raw SHA-256 | `1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191` |
| TRAIN protocol Git blob | `afdacef6f9d53a1e16b76155d8accc3a1ecf9e46` |
| TRAIN checkpoint raw SHA-256 | `0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d` |
| TRAIN checkpoint cell/synapse SHA-256 | `86b40631c49db55124dff3212645dc571555c5939fc4004a5d94c54b49e99779` |
| TRAIN checkpoint Git blob | `430faa1221c7721d97f43a9c311671e30bb65317` |
| TRAIN summary raw SHA-256 | `71133044d55e48a3bbe5cb147a33ce2f543e0a0dad855ae11e2e1b140b6b02d7` |
| TRAIN summary Git blob | `5ec6d687ed62aeb674231941c7ac52591d6f980f` |

Before either planning or execution returns, the runner checks the working
bytes and commit-tree Git blobs of the TRAIN protocol, the four relevant
source files, and all seven committed TRAIN artifact files. It also validates
the TRAIN summary as a successful 3/3 TRAIN result with no prior DEV opening,
and checks the checkpoint's embedded cell/synapse identity without restoring
it. A dirty, missing, relocated, uncommitted, or mismatched input fails closed.
Every direct read is capped at the committed byte count plus one before the
bytes are allocated or parsed, and the exact committed count is then required.
Source blobs are queried at the frozen source commit; the TRAIN protocol and
artifact blobs are queried at the committed TRAIN artifact commit. The
986,677-byte checkpoint uses the same 986,678-byte bounded read during both
preflight and each later raw-immutability audit.

## Public teacher-free tasks

| Task | Definition | Public semantic cases |
| --- | --- | --- |
| `dev_int_two_stage_offset` | integer signed add `+7`, then `-4` | `-8 -> -5`; `0 -> 3`; `11 -> 14` |
| `dev_list_two_stage_map_offset` | map signed add `-6`, then `+8` | `[] -> []`; `[-5, 0, 7] -> [-3, 2, 9]`; `[10, -11] -> [12, -9]` |

The DEV protocol contains no teacher-token field. Its loader requires these
records to equal the two validated definitions from the TRAIN protocol.

## Historical runner contract

The retired public runner was a plan by default. Planning made zero checkpoint
restores and zero completion calls, wrote nothing, and rejected an existing
fixed DEV evidence root. Its one authorized execution required `--execute`
plus the exact raw protocol SHA. There is deliberately no current command that
loads or executes this Layer-1 v3 record.

That command was executed exactly once on 2026-08-22. It used the fixed
evidence directory
`artifacts/typed_program_public_dev_v1/evidence`. If its parent exists, the run
stops. Otherwise it writes only these canonical JSON records:

1. `000-run-start.json`;
2. `001-dev_int_two_stage_offset.json`;
3. `002-dev_list_two_stage_map_offset.json`; and
4. `003-summary.json`.

Every record is flushed and then atomically hard-linked into a new ordinal
path. Existing files are never replaced. Partial evidence is terminal rather
than resumable.

Each task gets one fresh `FrozenCheckpoint.from_bytes` attempt from the same
checked byte snapshot and, after a valid restore, one `complete` attempt. An
exception is recorded without retry and does not skip the later task. The
task gates are:

1. fresh restore and attempt accounting;
2. teacher-free label boundary;
3. typed structural lowering;
4. Python syntax;
5. Python compilation and registered entry point;
6. all three registered semantics;
7. prompt identity;
8. typed-signature identity;
9. raw checkpoint immutability;
10. cell/synapse-state immutability;
11. fitted-cell race audit; and
12. timing, memory, and accounting validity.

The run additionally gates complete task recording, pass@1 accounting,
incremental evidence accounting, and run-level timing/memory accounting.
Caller-owned `tracemalloc` sessions are preserved.

## Primary result and promotion rule

The primary result is exact semantic pass@1 as an integer from 0 through 2.
Every failed or not-run gate, completion exception, and failed semantic case is
reported. There are no retries, replacement attempts, or alternate scores.

The preregistered promotion milestone was at least 1/2 tasks with semantics,
syntax, and compilation all valid and no recorded boundary violation. Meeting
that small milestone would justify the next capability experiment only. It
must not be reported as broad generalization.

Without another model call, each returned token stream is compared with the
three public TRAIN teacher streams and classified as either an exact public
TRAIN-label replay or not an exact public TRAIN-label replay. The second class
does not prove non-memorization or generalization.

## Expected cost and exclusions

Planning was expected to finish in under one second with zero model calls. The
single execution measured 298,252,300 ns wall time, 234,375,000 ns process
time, and 7,382,637 peak traced bytes.

This gate performs no teacher-program access, retraining, tuning, retries,
alternate checkpoint/configuration/seed, search, repair, fallback, Simple-SWE,
canonical or private evaluation, external language-model call, or additional
memorization-control completion. Its tests use injected fake restore and
completion adapters; no real DEV completion was performed while implementing
or verifying the gate. The later material result is reported separately in
`docs/typed-program-public-dev-v1-results.md`; it was not rerun during the
independent evidence audit.
