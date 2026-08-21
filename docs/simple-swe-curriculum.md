# Simple software-task curriculum

## Status

The v1 protocol and harness are implemented, but capability evaluation is
deliberately unopened. No canonical evaluation seed, private evaluation bundle,
cell completion, hidden-test result, or score has been generated. The next
authorized step is train/development work that produces one generic cell
checkpoint; only after its full inference configuration is frozen may the
harness reveal the 30-task evaluation cohort.

This is the immediate gate before any larger repository repair or Gemma
comparison. It asks a cell simulation to emit one complete Python function per
task. It is not a claim of LeetCode, HumanEval, MBPP, LiveCodeBench, SWE-bench,
or general software-engineering performance.

The canonical protocol is
`benchmarks/simple_swe_protocol_v1.json`, SHA-256
`f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2`.

## Established evaluation conventions

The design adopts a small set of established code-generation conventions:

- [HumanEval](https://github.com/openai/human-eval): one function completion,
  evaluator-only executable tests, and pass@1;
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp):
  entry-level Python task scale and explicit train/validation/test partitions;
  canonical MBPP exposes three assertions in each prompt, so this harness's
  private-test design is a variant and must not be called canonical MBPP; and
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench): private tests
  and a strict post-freeze reveal boundary. This internal generated cohort does
  not provide LiveCodeBench's temporal, externally sourced contamination
  boundary.

Prompts contain no input/output examples or executable reference logic. Common
format policy comes first; the task-specific signature and description come
last so a short cell context still reaches the semantics. Entry-point names are
opaque and do not reveal the family.

## Frozen task split

Families, not just instances, are disjoint across train, development, and
evaluation. Visible generation refuses the evaluation partition.

| Partition | Seed/status | Tasks | Families |
| --- | --- | ---: | --- |
| Train | fixed seed `1300781` | 90 | scalar clamp/offset/nonnegative; list add/absolute map; character-class filtering; clamp-then-sum, map-then-count, filter-then-reverse |
| Development | fixed seed `7100319` | 27 | digit sum/product; stable parity/range selection; longest character/target runs; digit-reduce-mod, select-then-span, run-length signature |
| Evaluation | 256-bit OS entropy only after freeze | 30 | bounded scalar/divisibility classification; adjacent equality/differences; token boundaries/word-length filter; clamped window sums, filtered adjacent changes, token-run signature |

The layers are intentionally small:

- Tier 0 proves the exact one-function output contract, parsing, signature, and
  simple scalar or boundary semantics.
- Tier 1 covers one-step numeric, list, and string transformations.
- Tier 2 covers held-out compositions, short windows/runs, and edge cases.
- Mini-repository bug fixes are a later layer and are not implemented by this
  gate.

Every task has four fixed regression cases and 32 generated property cases.
Each property set contains an explicit edge distribution as well as seeded
small random cases. Evaluation reports every preregistered task in frozen order;
there is no best-seed, best-task, or best-checkpoint selection.

## Cell-only completion boundary

The cell model's immutable synthesis request contains only a public prompt and
signature. The evaluator-side adapter restores the same checkpoint before
every task, verifies checkpoint/source commitments through its own filesystem
control plane, and performs a deterministic race among simulated human cells
for every emitted token. The cell model and synthesis request have no
filesystem, shell, network, tool, private-test, or feedback interface.

The adapter returns:

- final Python source;
- the exact model-emitted text and token IDs;
- adapter and deterministic compiler identities/hashes;
- prompt, emission, and source hashes; and
- the immutable on-disk checkpoint hash plus learned checkpoint-state hashes
  before and after synthesis.

Raw Python is the current compiler mode. A later deterministic grammar or
prefix DSL can use the same seam, but its compiler may only translate emitted
tokens and the public signature. It cannot choose an algorithm, inspect the
prompt semantics, run tests, search, repair, or retry. The protocol allows one
192-token sample and zero repairs per task.

Recurrent inference naturally changes membrane, threshold, and eligibility
traces. Those transient sequence variables are not checkpointed learned state.
The audit therefore compares an explicitly named learned checkpoint-state hash,
while independently requiring that the original checkpoint file remains
byte-identical and that learning is frozen. Dense-ridge, local recurrent v3,
and sparse-three-factor v3 checkpoint arrays use the same check.

## Freeze, reveal, and execution order

1. Freeze one checkpoint, tokenizer, model configuration, completion adapter,
   compiler factory, protocol, exact Python executable and `-I -S -B` flags,
   sandbox worker, native simulation artifact (when used), Git HEAD, and Git
   tree. All installed Human Brain execution modules must resolve under that
   source root and be Git-tracked.
2. Bind the manifest directory as the sole evidence root. Its canonical run
   path is derived from the canonical manifest bytes; alternate bundle, ledger,
   and run-root arguments are not exposed by the production APIs.
3. Revalidate a clean source tree and every hash, then append the public
   `run-start` commitment before requesting entropy.
4. Generate and atomically reserve one private 256-bit seed. Append its public
   hash commitment before generating any evaluation task, then stop.
5. Publish the returned run-start/seed commitment through an external channel.
   Only after an operator records a public HTTPS URI and receipt hash may the
   harness continue. The harness validates that receipt record structurally but
   does not itself contact or attest the external service.
6. Generate all tasks once from the reserved seed. Reload the canonical private
   bundle and regenerate every public/private field in frozen order before
   publishing task/test commitments.
7. Restore the checkpoint and synthesize exactly one completion for every task.
   Each completion is an atomic, hash-linked artifact. An interruption resumes
   in fixed order without resampling already committed tasks.
8. Seal the complete source/token/audit set. Revalidate the freeze and load one
   hash-checked immutable worker snapshot. Only after this seal may the executor
   receive private cases, so no test outcome can alter a completion.
9. Run every completion under the pre-frozen Python 3.11 runtime and append all
   results before computing pass@1 and tier/taxonomy summaries.

The public artifact ledger is canonical JSON, contiguous, append-only, and
SHA-256 linked. The private reservation/bundle contain the seed and cases; the
public ledger contains only commitments. Atomic creation uses a flushed
temporary file plus a no-overwrite hard link, so partial files are not treated
as results. The production runner cannot accept an injected adapter, compiler,
executor, runtime, worker, or in-memory bundle. A separate private test engine
permanently labels fixture ledgers as dependency-injected and noncanonical.

## Sandbox and failure reporting

Before parsing, the parent bounds source to 16,384 UTF-8 bytes; it also bounds
the AST to 4,096 nodes. Static policy permits exactly one plain function with
the requested signature and denies imports, classes, nested functions,
annotations, private/dunder attributes, and dangerous reflection/I/O built-ins.
The child receives restricted built-ins and no task-visible paths.

Each task is limited to two seconds wall time, one second CPU, 256 MiB process
memory, one process, and 65,536 output bytes. On Windows, the child blocks on
stdin until it is attached to a fail-closed Job Object enforcing CPU, memory,
process-count, and kill-on-close limits. If a required limit cannot be attached,
the outcome is `harness_error`; execution does not silently continue without
limits.

The fixed taxonomy is `format_error`, `syntax_error`, `policy_violation`,
`signature_error`, `timeout`, `memory_limit`, `runtime_error`,
`regression_failure`, `property_failure`, `harness_error`, and `passed`.
The summary reports pass@1, tier counts, and every taxonomy bucket.

This is language-policy isolation plus resource control, not an OS security
sandbox for hostile arbitrary Python. Windows Job Objects do not provide a
filesystem or network namespace. The accepted language currently cannot call
`open`, import modules, or reach private attributes, but hidden execution should
still run on a disposable host if adversarial-code containment becomes a goal.

## Anti-overfitting and claim boundary

All training and development choices must finish before the evaluation seed is
requested. Freeze fails if the checkpoint, tokenizer, configuration, adapter,
compiler, tracked source, or installed execution-module location drifts. There
is no validation/evaluation-time learning, task replay, checkpoint selection,
test feedback, repair, or cherry-picking.

This v1 cohort is nevertheless an internal development gate. Exact family
names and oracle implementations, including evaluation-family logic, are
committed in `src/humanbrain/simple_swe/tasks.py`. Family separation and a
post-freeze instance seed prevent ordinary instance tuning, but they do not
create a contamination-clean external benchmark. No headline capability claim
may rely on this score alone.

Before any external or Gemma headline comparison, preregister a later cohort
whose tasks and tests have never been inspected, source it after the checkpoint
cutoff where possible, use one unchanged checkpoint/configuration, and report
the whole cohort. Conventional-model baselines are aggregate reporting controls
only: the harness accepts them after the immutable cell summary, requires the
same task count, and provides no path by which baseline output can become cell
inference.

## Runtime estimate and remaining environment commitments

Visible train/development generation is sub-second. The 75 focused protocol,
freeze, adapter, reveal, sandbox, artifact, and runner tests complete in about
18 seconds on the current workstation. These are harness tests, not capability
evaluation.

Evaluation emits 30 x 192 = 5,760 cell-selected tokens. At the project's target
100 tokens/s that is about 58 seconds of cell inference. Sandboxed tests add
normally seconds and at most 60 seconds if all 30 tasks reach their two-second
wall limit. A realistic first gate is therefore roughly one to two minutes
after checkpoint loading and native compilation; reloading a very large
checkpoint 30 times may dominate. No honest runtime is yet available for a
100-billion-cell checkpoint.

The freeze probes the selected executable itself with isolated Python, enforces
the protocol's Python 3.11 major/minor, and records the executable SHA-256,
implementation, patch version, and exact launch flags before the seed exists.
For a recurrent checkpoint it also records and rehashes the loaded native DLL
or CUDA CUBIN plus the build source digest and architecture. The loaded cell
model must report the same backend/device before synthesis. This does not attest
the operating system, dependent DLLs, CUDA driver/runtime, firmware, hardware,
or other system libraries.

Git cleanliness includes tracked changes and ordinary untracked files, while
Git-ignored files are outside the tree commitment. Known execution modules are
protected by the stronger containment-and-tracked-file check, but ignored data,
compiler caches, environment variables, and system libraries remain reported
environment limitations. Production evaluation should record the host and
native build metadata alongside the ledger.

The one-run guarantee is procedural, not remote attestation. Canonical local
paths, no-overwrite writes, and an externally anchored seed commitment make
accidental reruns and ordinary cherry-picking visible, but a trusted operator
can delete local evidence or create another freeze manifest. The receipt API
records an operator-supplied publication claim; it does not prove publication.
Any headline use therefore requires the run-start/seed commitment to be
published through an independently retained channel before task generation.

## Work allowed before reveal

Use only train and development families to improve the cell simulation,
topology, grouping, growth/order, local learning, decoder, and speed. Compare
configurations with fixed development seeds and matched data/compute, preserve
failed scale results, and freeze one generic checkpoint only after those choices
are complete. Do not create the canonical evaluation seed or private bundle as
a smoke test; the unit suite exercises the state machine with synthetic fixture
bundles instead.
