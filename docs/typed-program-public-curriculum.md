# Typed program public TRAIN curriculum v1

**Status:** executed once; live v1 runner retired by Layer-1 v4

The frozen protocol, checkpoint, evidence, and result report remain immutable
historical records. The v1 loader and execution CLI were removed when the
incompatible Layer-1 v4 checkpoint contract replaced v3; the commits named in
the result report are the reproducibility anchors. The current unexecuted
increment is specified in
[`typed-composition-a012-v1.md`](typed-composition-a012-v1.md).

This is the smallest public train-only curriculum around
`humanbrain.typed_program`. It exercises only the fitted recurrent-cell path.
It has one fixed `Layer1Protocol`, one growth call, one common checkpoint, and
one completion attempt per TRAIN task. It has no `DendriticCortex` adapter,
family macro, fallback model, search, retry, or benchmark integration.

The raw bytes of
[`typed_program_public_train_protocol_v1.json`](../benchmarks/typed_program_public_train_protocol_v1.json)
are preregistered as SHA-256
`1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191`.
The protocol is based on typed-stack commit
`600ef82bd2edddb00b5df237754bec62be93c0b1`.

## Public TRAIN set

The complete TRAIN set contains three programs, nine semantic cases, and 16
teacher DSL tokens. The cases below are protocol data, not generated samples.

| Task | Type | Public teacher tokens | Public semantic cases |
| --- | --- | --- | --- |
| `train_int_identity` | `Int -> Int` | `variable-0` | `-7 -> -7`; `0 -> 0`; `12 -> 12` |
| `train_int_add_negative_two` | `Int -> Int` | `add variable-0 integer - 2 end` | `-5 -> -7`; `0 -> -2`; `9 -> 7` |
| `train_list_map_add_positive_three` | `List[Int] -> List[Int]` | `map bind add variable-0 integer + 3 end variable-0` | `[] -> []`; `[-4, 0, 8] -> [-1, 3, 11]`; `[7, -9] -> [10, -6]` |

The loader structurally compiles each teacher program and checks these cases
without fitting or completing a circuit. If a later authorized run exactly
replays these labels, that result is **memorization, not generalization**.

## Public development definition

The protocol also preregisters two development task definitions and six public
semantic cases. Their family IDs, prompt-template IDs, and literal values are
all disjoint from TRAIN. They contain no teacher token field.

| Task | Family/template | Literals | Public cases |
| --- | --- | --- | --- |
| `dev_int_two_stage_offset` | `int_two_stage_signed_add` / `compose_two_scalar_offsets` | `+7`, `-4` | `-8 -> -5`; `0 -> 3`; `11 -> 14` |
| `dev_list_two_stage_map_offset` | `list_two_stage_map_signed_add` / `compose_two_offsets_inside_map` | `-6`, `+8` | `[] -> []`; `[-5, 0, 7] -> [-3, 2, 9]`; `[10, -11] -> [12, -9]` |

This task does not pass either development definition to completion or execute
its semantic cases against model output. The protocol therefore records zero
development runs and an unopened completion state. Nothing in this slice is a
development or held-out score.

## Historical runner contract

The retired v1 runner validated the frozen raw protocol and printed the TRAIN
counts without writing an artifact in its plan mode. It also failed if the
fixed artifact root, checkpoint, or incremental evidence directory already
existed. There is deliberately no Layer-1 v3 compatibility loader or live v1
CLI at current HEAD. Reproducing the historical executable requires checking
out the source commit bound by the result report; the preserved current-tree
artifacts are records, not executable inputs to v4.

The production module pins the repository root and protocol path from its own
installed source location. The CLI has no protocol, repository-root,
checkpoint, or evidence-path option, so copying the protocol cannot redirect
evidence. Tests relocate the runner only by replacing one private dependency
set that also supplies fake growth, restoration, and completion functions; that
seam is not part of the production interface.

The one material run was executed under the exact raw protocol SHA. No second
run, resume, or reconstruction under v4 is permitted.

On execution, the runner must:

1. call `grow(Layer1Protocol(), all_train_examples)` exactly once;
2. atomically publish the one nonoverwriting checkpoint;
3. construct a fresh `FrozenCheckpoint` from the same raw bytes for every task;
4. call `complete` exactly once for every TRAIN task, with no retry or
   replacement even if another task fails;
5. report label replay, typed lowering, Python syntax, Python compilation,
   public semantics, prompt/signature identity, checkpoint raw/state
   immutability, and fitted-cell race integrity as separate gates;
6. validate timing, traced memory, and accounting as an explicit gate on every
   task and again on the run summary, with both levels included in the final
   conjunction; and
7. record wall and process time, traced current/peak Python memory, checkpoint
   bytes, incremental evidence bytes, training sequences/tokens, completion
   attempts/tokens, and cell-race counts. A returned completion retains its
   program, prompt, checkpoint, token, and race identities even if a later
   audit gate fails.

The fixed evidence root is
`artifacts/typed_program_public_train_v1`. Each JSON event is first flushed and
then atomically linked into a new ordinal path; no existing file is replaced.
A partial run is deliberately terminal evidence, not a resumable or retryable
experiment.

Protocol loading reads at most 131,073 bytes—the 128 KiB limit plus one
sentinel byte—before hashing or JSON allocation. Execution preserves a caller's
pre-existing `tracemalloc` session and stops tracing only when the runner
started that session itself.

## Expected cost and exclusions

The unmeasured planning estimate is 1–3 minutes with a warm native cache and up
to roughly 10 minutes if the recurrent native runtime must build from cold,
with less than 512 MiB process memory. The checkpoint remains subject to the
fixed 2 MiB `Layer1Protocol` limit. These are planning estimates, not results.

This protocol performs no tuning, alternate configuration or seed, retry,
search, test feedback, Simple-SWE development/evaluation, canonical
freeze/reveal/private operation, external language-model call, Fly Brain
operation, or benchmark execution.

The separately hash-locked one-shot gate for the two teacher-free development
definitions is recorded in
[`typed-program-public-dev-gate-v1.md`](typed-program-public-dev-gate-v1.md).
It was executed once and failed 0/2; it is retired and will never be rerun or
used for tuning.
