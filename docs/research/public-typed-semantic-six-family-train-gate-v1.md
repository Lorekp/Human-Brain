# Public typed-semantic six-family TRAIN gate v1

Status: frozen, not spent.

Date: 2026-08-30.

## Purpose and claim boundary

This is the one-shot breadth gate licensed by the compact typed-output PASS.
It tests one task-independent bottom-up typed-semantic synthesizer on the six
remaining canonical public-TRAIN Simple-SWE families. Per task, the model may
see only the typed callable signature and four visible regression input/output
rows. It may not see prompt text, task/family/tier/ordinal/partition identity,
the 32 property rows, test commitments, sandbox results, prior completions, or
source/oracle code.

A PASS is public-TRAIN programming-by-examples evidence on these six families.
It is not prompt-only, family-disjoint, unseen-instance, contamination-clean,
DEV/eval, M0, held-out, protected, Path-2, C1, HumanEval, MBPP, LiveCodeBench,
SWE-bench, Gemma, or Opus evidence.

## Established conventions adopted

The protocol keeps the established patterns already chosen for Simple-SWE:

- [HumanEval](https://github.com/openai/human-eval): one complete function,
  deterministic pass@1, and executable evaluator-only tests. The entire
  declared cohort is attempted unless the frozen futility stop fires.
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp):
  explicit train/development/test partitions and entry-level Python scale.
  Unlike canonical MBPP prompts, these model inputs do not expose assertions.
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench): pin the
  complete cohort before execution and do not cherry-pick tasks. This
  generated public TRAIN cohort has no temporal-freshness claim.
- [EvalPlus](https://github.com/evalplus/evalplus): bind the expanded test
  oracle and require all additional cases, because a small visible suite alone
  is not a robust correctness result.

The 8/10 family threshold and third-failure futility stop are repo-specific
decisions, not conventions imported from those benchmarks.

## Frozen cohort

Protocol bytes: benchmarks/simple_swe_protocol_v1.json, SHA-256
f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2.
Partition: train, seed 1300781.

The six families are in their canonical protocol order:

1. t0_scalar_clamp;
2. t0_scalar_nonnegative;
3. t1_string_filter_class;
4. t2_clamp_then_sum;
5. t2_map_then_count;
6. t2_filter_then_reverse.

Every family contributes ordinals 000..009. Execution is ordinal-major
round-robin: all six ordinal-000 tasks, then all six ordinal-001 tasks, through
ordinal 009. The complete 60-ID tuple is frozen in the runner and has SHA-256
e1d3276b46a32e7811053851ba4776d3c5cddeed2292d6a10761b938a7334dd2.
There is no omission, replacement, resampling, unsupported-task filtering, or
order change.

## Exposure disclosure

The provenance-only registry is
experiments/r3_pilot/public_typed_semantic_six_family_train_exposure_registry_v1.json.
Its payload SHA-256 is
91a4b2db928cbc894dabc65355d77a0fc778ab28221994980406356598e68efc.
It was built from identifiers and artifact provenance only; no desired output,
regression row, property row, prompt text, completion, or task source was
opened.

All 60 prompts appeared in the historical public prompt-only baseline. Twenty
four task IDs have a task-specific R3 artifact mention, spanning 38 task/path
pairs; 36 have no such mention. The latter count is 2 clamp, 4 nonnegative,
10 String-filter, 3 clamp-sum, 7 map-count, and 10 filter-reverse. Identifier
occurrence does not prove desired-output exposure, so results are reported
descriptively rather than stratified into separate pass thresholds. Every
task remains in the gate to avoid exposure-based cherry-picking.

## Frozen mechanism

No new task or family operator is added. The existing generic basis already
expresses the intended breadth without reading a task row:

- maximum and nested minimum/maximum, costs 3 and 5;
- ASCII-class contains followed by filter, cost 6;
- reverse(filter(...)), cost 7;
- predicate binding plus count, at most cost 8;
- clamp binder plus map and sum, cost 9.

The literal bank is exactly -9..9, and maximum AST cost is 9. Frontier limits
remain 256 representatives per typed context, 6,000 declared roots, 30,000
operator predictions, and 80,000 physical races. A work-bound STOP is a
definitive task failure and is serialized with its exact completed-prefix
audit; it never licenses top-k pruning or a bound change.

The task-independent frozen identities are:

- semantic checkpoint
  af3eb36d62a5419bdd7855576e76e4fde832459b882bd35c39aa39dd58c25427;
- operator checkpoint
  92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e.

One model is built for the cohort. Both the stored and recomputed live
checkpoint must be exact before marker creation and remain exact before and
after every task. The host may enumerate grammar-valid typed programs and
compile the physically selected winner; it may not use family-specific
skeletons, inspect property rows during selection, rank candidates with host
execution, or feed sandbox results back into synthesis.

## Prerequisites

Preflight authenticates, without generating a task:

- the banked three-family visible-I/O PASS artifact, raw SHA-256
  09fbaef134d2ba18a68b513a89d7df5ab8708c8be84940e71a29b710f7ff9767;
- the compact output PASS artifact, raw SHA-256
  f503820711ebc597aeac6b96e6d89dce17c48be7b756b182619a7238b531e179
  and payload SHA-256
  beab356461d443fc5ac5f5cb5198e755e3e6927b48a8aa860bc8fdbd68b3604e;
- the protocol, source-closure, exposure-registry, semantic/operator
  checkpoint, host synthesis runtime, sandbox Python/worker, and tracked Git
  tree identities.

The exposure-registry payload is compared both with its self-hash and the
literal frozen digest above. The host process that performs NumPy-backed cell
synthesis is separately pinned to CPython 3.11.15 and NumPy 2.4.6, including
the venv launcher, base interpreter, python311.dll, NumPy core binary, build
configuration, and installed-package RECORD hashes. This is distinct from the
selected Python runtime used by the isolated canonical sandbox. Host identity
is checked before the marker, before the cohort, and after the cohort. The
tag/HEAD/tree identity is likewise recomputed immediately before marker
creation, closing the preflight-to-marker window.

The task-generator source itself is pinned at SHA-256
b0edfc2eedc25335bcb2318a7b13026f6ab7affc29643af76e16f4c6b57240c3;
a later committed generator with the same protocol metadata cannot silently
change the desired rows admitted by this gate.

The repository contains unrelated untracked research state, so cleanliness is
defined as: no staged or unstaged tracked change, every explicitly listed
mechanism/protocol/oracle file tracked, and every listed file hash captured.
Untracked paths are excluded rather than silently incorporated.

The commit itself is externally frozen by the lightweight Git tag
public-typed-semantic-six-family-train-gate-v1-freeze. Preflight requires HEAD
to resolve to that exact tag in addition to requiring a clean tracked tree;
the marker then binds the tag, commit, tree, and complete source hash bundle.

## One-shot spend and publication

Plain invocation is preflight only and must report PREFLIGHT_READY_NOT_RUN with
sealed_rows_materialized equal to 0. Execution requires both --execute and
--attest-exclusive-host.
After either the marker or terminal result exists, plain invocation reports
PREFLIGHT_BLOCKED_ALREADY_SPENT rather than another ready state.

Before generate_visible_partition is called, the runner exclusively creates
and file-fsyncs:

experiments/r3_pilot/public_typed_semantic_six_family_train_gate_v1.spent.json

The immutable marker binds the complete ID/order digest, Git tag/commit/tree,
source bundle, prerequisite and exposure hashes, model checkpoints, literal
bank, cost limit, runtime, worker, and canonical result path. An existing
marker or result forbids another spend. Only after the marker exists may the
60 rows be generated and their public/private test commitments bound.

The marker uses the repository's established atomic-create helper: temporary
contents are flushed and file-fsynced, then installed with an exclusive hard
link. The protocol does not claim parent-directory metadata fsync or
power-loss-proof directory durability.

Each task receives exactly one synthesis call and at most one canonical sandbox
execution. No retry, repair, learning, task-specific replay, replacement, or
test feedback is allowed. Program/source hashes are recorded before sandbox
execution. A task passes only when all four regressions and all 32 property
cases pass.

The terminal result is atomically published at:

experiments/r3_pilot/public_typed_semantic_six_family_train_gate_v1.json

The artifact records no literal regression/property rows. It records their
commitments and visible-problem hash, physical tier/race/work counters,
selected program/source hashes, sandbox taxonomy, checkpoints, timing, and
provenance. If a post-marker integrity or harness exception occurs, the same
canonical path receives an INVALID_INTEGRITY_OR_HARNESS terminal diagnostic;
the marker remains and the gate is not restarted. Exception detail is never
written verbatim because it could contain a row value; only a fixed error
taxonomy, exception type, and SHA-256 of the redacted detail are published.
Any safely completed task records are retained as a prefix, including the
program/source hashes and sandbox taxonomy for a harness-error task; failures
before cohort execution record zero attempted tasks.

## Verdict

Each family must pass at least 8 of its 10 tasks. Any synthesis STOP,
work-bound STOP, syntax/policy/signature/timeout/memory/runtime/regression/
property failure is one definitive task failure. The cohort stops immediately
when a family reaches its third failure because 8/10 is then impossible. There
is no early-success stop.

- PASS: all 60 tasks attempted, every family at least 8/10, exact source and
  checkpoints preserved.
- STOP_CAPABILITY: a family reaches three failures, a completed family is
  below 8/10, or a visible physical value is unsupported.
- INVALID_INTEGRITY_OR_HARNESS: source/checkpoint/protocol/runtime/oracle
  integrity fails or the canonical sandbox reports a harness error.

An early STOP is a decisive gate failure, not pass@1 over the attempted subset.
A PASS licenses building and freezing the TRAIN-only prompt-span-to-factor
bridge. DEV remains sealed until that bridge is frozen.
