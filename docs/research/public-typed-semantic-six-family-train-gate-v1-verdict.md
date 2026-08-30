# Public typed-semantic six-family TRAIN gate v1 verdict

Status: terminal `STOP_PUBLIC_TYPED_SEMANTIC_SIX_FAMILY_TRAIN_GATE_V1`.

Date: 2026-08-30.

## Frozen execution

The gate was frozen at commit
`4dfbc5a1cbae704f70d85511558bc086e1d904a6` and external tag
`public-typed-semantic-six-family-train-gate-v1-freeze`. Independent
stable-snapshot verification passed 9/9 gate-integrity tests and 58/58
mechanism tests before execution. Plain preflight returned
`PREFLIGHT_READY_NOT_RUN`, authenticated source bundle
`7e4f4de8152155d012ef3ac419cd427d4b0fb6584a1690d0e3d48e3207b1aa1d`,
and reported zero materialized rows with no marker or result.

The exclusive marker was then created before the canonical task generator was
called. Its raw SHA-256 is
`b218fc65f9d3c32ac67758892d05d3695d3e6710f188078b26d888055d627718`
and payload SHA-256 is
`a7f8c62be4a1a17502599ebf0c46136815e60b26a4cafe0e9c8013001666262e`.
The terminal artifact is
`experiments/r3_pilot/public_typed_semantic_six_family_train_gate_v1.json`,
raw SHA-256
`c663ef1d167d09ab8d8811c6766280b4490f2d22a194d0dde3fe872318d59538`
and payload SHA-256
`df016c57ba16020da999d14877d32d597a9d6435d01791c43033b459d1e170f4`.
Both payloads independently recompute, the result binds the exact marker, and
plain invocation now returns `PREFLIGHT_BLOCKED_ALREADY_SPENT`.

All 60 public-TRAIN task objects were materialized after the marker so their
fixed physical output support could be checked. The model saw only each
attempted task's typed signature and four ordered regression input/output
rows. Prompt/task/family identity stayed excluded from synthesis, and the 32
property rows per task reached only the canonical sandbox after a `DONE`
selection. No literal regression or property row is stored in the artifact.

## Result

The third scalar-clamp failure made its 8/10 gate impossible, so the frozen
ordinal-major round-robin stopped after 13 of 60 tasks. Six tasks passed. Wall
time inside the adjudicative cohort was 47.2223 seconds, and the shared cell
checkpoint remained exact.

| Family | Attempted | Passed | Failure mode |
|---|---:|---:|---|
| `t0_scalar_clamp` | 3 | 0 | three declared-root work-bound STOPs |
| `t0_scalar_nonnegative` | 2 | 2 | none observed |
| `t1_string_filter_class` | 2 | 2 | none observed |
| `t2_clamp_then_sum` | 2 | 0 | two property failures after visible exact match |
| `t2_map_then_count` | 2 | 0 | two retained-representative work-bound STOPs |
| `t2_filter_then_reverse` | 2 | 2 | none observed |

The six passes are real but narrow: both observed nonnegative tasks closed at
cost 3, both String-filter tasks at cost 6, and both filter-then-reverse tasks
at cost 7. They do not override the terminal family gate.

## Generic localization

### 1. Scalar clamp: complete cost-tier admission is too coarse

All three clamp tasks reached AST cost 5, the already-established generic cost
of nested minimum/maximum clamp. Before evaluating that tier, the runner's
complete-tier admission projected total declared-root counts of 7,897, 10,363,
and 9,130 from completed prefixes of 1,237, 1,261, and 1,249. The frozen limit
was 6,000, so no cost-5 candidate entered a desired-vector race.

This falsifies the sufficiency of raw complete-tier declaration as the safety
unit on this breadth, not the availability of generic clamp operators. A
successor must reduce or factor the desired-independent grammar tier before
physical evaluation; raising the bound and retrying these tasks is not
licensed.

### 2. Map then count: bound-context semantic diversity exceeds the cap

Both observed map-count tasks stopped at AST cost 5 while attempting the 257th
retained representative in the same bound `Int` context. Each completed 16,739
operator predictions, 40,520 physical races, 2,717 declared roots, 1,330
supported roots, and 358 retained representatives overall. The frozen
per-context limit was 256. The generic count solution tier was therefore never
reached.

This falsifies the calibrated 256-representative headroom for diverse
bound-value contexts. It does not falsify predicate binding, count, or the
compact cell output substrate. The successor needs a generic factorized or
normalized bound representation rather than a task-specific bank or silent
top-k pruning.

### 3. Clamp then sum: four visible rows do not identify the intended program

Both observed clamp-sum tasks selected the same unique exact visible-I/O
candidate at cost 2, then failed a sandbox-only property case. Recomputing the
generic candidate-node identity from the frozen source identifies
`d400f43a...` as exactly `sum(variable-0)`. The cell race therefore did exactly
what the current objective asks—select the lowest-cost observationally exact
program—but four examples admitted that cheaper spurious program instead of
the intended cost-9 composition.

This is an information/selection failure, not a race or syntax failure. More
unseen property feedback cannot be folded into the same run. The successor
must carry an explicit cross-cost ambiguity/version-space signal; a generic
learned prompt-semantic factor may resolve I/O-equivalent hypotheses but may
never override a worse visible-I/O tier.

## Decision and claim boundary

The six-family cohort will not be retried, resampled, narrowed, or used to tune
the literal bank, bounds, enumeration order, or family operators. Follow-up
mechanisms must be developed on target-independent or self-posed fixtures and
must preserve the immutable marker and result.

The named successor is a production-factorized semantic chart: keep typed root
production, operand/binder role, and literal slots in separate physical banks,
compose them only after factor races, and partition retained semantics by those
roles rather than one heterogeneous bound-Int bucket. Its self-posed frozen
matrix must reach nested clamp and count within the existing event budget and
must either resolve or honestly stop as `AMBIGUOUS_VISIBLE_SEMANTICS` on a
sum-versus-clamp-sum alias. No cap increase or silent pruning is licensed by
this result.

The result is public-TRAIN programming-by-examples evidence with three
localized boundaries and three small positive family slices. It is not a PASS,
prompt-only solving, family-disjoint transfer, DEV/eval, M0, held-out,
protected, Path-2, C1, SWE-bench, Gemma, or Opus evidence. DEV and every later
protected surface remain sealed.
