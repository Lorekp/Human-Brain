# Typed program public TRAIN curriculum v1

**Status:** preregistered and implemented; material TRAIN run not executed

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

## Runner contract

The checked-in runner is plan-only by default:

```powershell
python benchmarks/typed_program_public_train.py
```

Planning validates the frozen raw protocol and prints the TRAIN counts without
writing an artifact. It also fails if the fixed artifact root, checkpoint, or
incremental evidence directory already exists.

A material run is intentionally not performed here. A later explicitly
authorized invocation must supply both `--execute` and the exact lowercase raw
protocol SHA through `--protocol-sha256`. No other seed, configuration,
checkpoint path, evidence path, or partition switch exists.

On execution, the runner must:

1. call `grow(Layer1Protocol(), all_train_examples)` exactly once;
2. atomically publish the one nonoverwriting checkpoint;
3. construct a fresh `FrozenCheckpoint` from the same raw bytes for every task;
4. call `complete` exactly once for every TRAIN task, with no retry or
   replacement even if another task fails;
5. report label replay, typed lowering, Python syntax, Python compilation,
   public semantics, prompt/signature identity, checkpoint raw/state
   immutability, and fitted-cell race integrity as separate gates; and
6. record wall and process time, traced current/peak Python memory, checkpoint
   bytes, incremental evidence bytes, training sequences/tokens, completion
   attempts/tokens, and cell-race counts.

The fixed evidence root is
`artifacts/typed_program_public_train_v1`. Each JSON event is first flushed and
then atomically linked into a new ordinal path; no existing file is replaced.
A partial run is deliberately terminal evidence, not a resumable or retryable
experiment.

## Expected cost and exclusions

The unmeasured planning estimate is 1–3 minutes with a warm native cache and up
to roughly 10 minutes if the recurrent native runtime must build from cold,
with less than 512 MiB process memory. The checkpoint remains subject to the
fixed 2 MiB `Layer1Protocol` limit. These are planning estimates, not results.

This protocol performs no tuning, alternate configuration or seed, retry,
search, test feedback, Simple-SWE development/evaluation, canonical
freeze/reveal/private operation, external language-model call, Fly Brain
operation, or benchmark execution.
