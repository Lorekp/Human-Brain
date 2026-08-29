# Typed semantic frontier work-bound protocol

Status: implemented and verified after the generic String semantic layer was
banked at commit `3daea4c`.

Date: 2026-08-29.

## Purpose

The bottom-up typed semantic synthesizer must have a deterministic upper bound
on teacher-free physical search work before the broad 60-task public-TRAIN
breadth gate.  The bound protects runtime without converting the retained
frontier into a host-ranked beam.

This protocol is an admission boundary, not silent top-k pruning.  A synthesis
that stays inside every bound preserves the existing exact-cost enumeration and
selection behavior.  A synthesis that would exceed a bound returns an explicit
scientific STOP before a partial tier can reach the desired-output race.  It
does not discard a representative and continue.

The implementation follows the banked String layer and enumerates every
aggregate and String physical prediction schema, including worst-case String
copy/filter/reverse work, in the frozen race-admission table authenticated by
the synthesizer checkpoint.

## Ownership and interface seam

The budget is a private in-process module owned by
`FrozenTypedSemanticSynthesizer`.  Its internal interface should remain small:

1. admit a complete exact-cost tier;
2. reserve one physical prediction and its conservative race upper bound;
3. retain one new semantic representative;
4. produce the completed-prefix STOP audit.

The caller-facing `synthesize(signature, case_arguments, desired_outputs)`
interface must remain unchanged.  The budget is not a caller-configurable
search policy, environment option, or benchmark parameter.  Tests exercise it
through ordinary synthesis decisions rather than through a new public budget
API.

## Frozen constants

| Internal constant | Limit |
| --- | ---: |
| `MAX_RETAINED_REPRESENTATIVES_PER_BUCKET` | 256 |
| `MAX_DECLARED_ROOT_CANDIDATES` | 6,000 |
| `MAX_OPERATOR_PREDICTIONS` | 30,000 |
| `MAX_PHYSICAL_RACES` | 80,000 |

These constants are fixed before any of the six unspent TRAIN-family desired
vectors are opened.  They may not be adapted by task, family, prompt, desired
output, observed failure, or wall-clock speed.

**Preimplementation supersession (2026-08-29, before any gated spend).** The
original 128-entry proposal was falsified by unchanged banked mechanism
regressions: the generic cost-6 predicate-count fixture reached 171 bound-Int
representatives and the scalar exhaustive no-solution fixture reached 168 root
Int representatives.  The cap is therefore transparently revised once to 256,
the next power of two.  This restores approximately the originally intended
headroom (171→256 is 50%, versus the preregistered 87→128 at 47%).  Sentinel B
was rerun diagnostically at both 192 and 256 and remained prediction-first at
exactly 30,000/30,001 with the identical completed prefix.  The other three
limits, check order, fixtures, and desired-independent protocol are unchanged.

The values leave deterministic headroom above the fixed 25-literal exhaustive
pressure sentinel:

- worst retained bucket: 87 versus 256;
- declared roots: 4,797 versus 6,000;
- operator predictions: 23,642 versus 30,000;
- physical races: 63,456 versus 80,000.

The 256-entry bucket bound also limits the next exact-cost scalar operand
cross-product independently of how many other typed context buckets exist.

## Desired-independent retention keys

### Context shape

Context shape is explicit audit state rather than an inferred property of a
flattened output tuple.

- Root context shape records the ordered visible case count:
  `("root", case_count)`.
- A bound container context records every flattened container in evaluation
  order:
  `("bound", ((case_index, parameter_index, container_type, length), ...))`.

For the current one-parameter `List[Int]` fixtures, the bound shape therefore
records one `(case, 0, "List[Int]", length)` entry per case.  The banked String
layer must use the same rule for Char contexts derived from String values.

### Bucket key

The retained-representative cap applies to:

```text
(
  environment_type_labels,
  result_type_label,
  context_shape,
)
```

It is cumulative across exact costs.  It is deliberately not partitioned by
production, syntax, task, family, or cost.

### Semantic key

A supported physical observation is deduplicated by:

```text
(
  environment_type_labels,
  result_type_label,
  context_shape,
  ordered_physical_output_sha256s,
)
```

The first deterministic lowest-cost node for a semantic key remains the only
retained representative.  Unsupported output tuples consume physical work but
do not consume representative capacity.

A compiler binder derives its retention identity from its already-retained
predicate and the predicate's bound context shape.  Binder nodes consume their
own `Function[...]` bucket capacity and may not bypass the 256-entry check.

Neither key contains desired outputs, prompt tokens, entry-point or parameter
names, task identifiers, family identifiers, property rows, sandbox scores,
DEV data, M0/evaluation data, held-out data, Path-2 data, or C1 data.

## Exact pre-admission order

The checks below have fixed precedence.  The first denial ends synthesis.

### 1. Admit the exact-cost tier

After deterministic DAG construction identifies the current root tier, but
before any node in that tier is physically processed, compute:

```text
prospective_declared = completed_declared + len(tier.candidates)
```

If `prospective_declared > 6_000`, return the work-bound STOP.  The current tier
is incomplete, is absent from completed `tier_attempts`, and never reaches a
desired-output race.

The representative cap from completed lower tiers bounds the operand frontier
used to construct this next DAG.  No candidate cap is checked after partially
processing a tier.

### 2. Reserve one physical prediction

Every `_CaseEvaluator._predict` call must reserve budget before invoking the
frozen operator.

First check prediction count.  If one more call would make the completed count
30,001, deny the call with `operator-predictions` as the bound name.

Second check physical races.  Each action schema has a frozen conservative
upper bound derived only from its physical banks and capacities.  Deny the
prediction if:

```text
completed_races + schema_max_races > 80_000
```

The implemented String-inclusive upper bounds are:

- ASCII-class terminal: 1 race;
- `contains(class, character)`: 2 races;
- unary absolute: 1 race;
- add, subtract, or less-than: 2 races;
- minimum or maximum: 4 races;
- `List[Int]` copy at capacity four: at most 6 races;
- String-position copy at capacity 26: at most 29 races for each filter source
  copy, filter output copy, and reverse output copy.

Every callable physical prediction schema is covered exhaustively; an unknown
schema is an implementation error, not permission to run without admission.
The completed race counter records actual races, not the conservative
reservation.  Completion also asserts that the physical call returned no more
races than its reserved schema maximum and cannot take the completed total past
80,000, so a stale understated schema cannot silently cross the hard bound.

Prediction check precedes race check so identical state always reports the same
first bound.

### 3. Retain a supported representative

After the full ordered supported output tuple and semantic key are available:

1. reuse the existing representative if the semantic key is already present;
2. otherwise derive the bucket key;
3. if the bucket already contains 256 representatives, deny attempted
   representative 257 before mutating `seen_semantics` or `retained`;
4. otherwise retain it normally.

The same ordering applies to a derived binder.

### 4. Race desired outputs only for a complete tier

`select_exact_semantic_vector` remains after all nodes in the current tier have
completed without a denial.  Desired outputs are not passed into the private
budget and cannot affect admission, retention, or STOP telemetry.

If any check fires mid-node or mid-tier, no desired-output race occurs for that
tier and no higher cost is enumerated.

## STOP decision and audit schema

The distinct status is:

```text
STOP_SEMANTIC_FRONTIER_WORK_BOUND
```

The normal exhaustive `STOP_NO_EXACT_TYPED_SEMANTIC_PROGRAM` remains separate.

The synthesis decision gains an optional immutable work-bound record containing
at least:

```text
bound_name
limit
completed
attempted
ast_cost
environment_type_labels
result_type_label
context_shape
completed_declared_candidates
completed_supported_root_candidates
completed_operator_predictions
completed_physical_races
completed_retained_representatives
completed_operator_races_sha256
```

Environment, result type, and context shape are null only for a whole-tier
candidate denial.  For prediction and representative denials they identify the
interrupted node context.

`tier_attempts` contains complete tiers only.  The interrupted cost is reported
by the work-bound record.  Candidate, prediction, race, and representative
counters report completed work only.  `attempted` reports the denied prospective
count: for example, 30,001 for a prediction denial or 257 for a representative
denial.

The ordered race digest covers exactly the completed physical-race prefix.
Checkpoint-before and checkpoint-after hashes must remain identical on every
STOP path.

## Checkpoint hashing

The synthesizer checkpoint must authenticate:

- all four constants;
- the context-shape and semantic-key version;
- the fixed check precedence;
- the complete aggregate-plus-String schema-to-maximum-races table;
- a new bounded-frontier domain separator.

Changing a limit, key, schema table, or check order invalidates the synthesizer
checkpoint.  The budget is readonly inference state and must not change the
underlying frozen operator checkpoint.

Implementation followed the String bank at `3daea4c`, so one checkpoint now
authenticates the complete generic aggregate-plus-String semantic basis rather
than a temporary aggregate-only race table.

## Sentinel A: exhaustive 25-literal pressure PASS

Signature:

```text
solve(values: List[Int]) -> Int
```

Fixture:

```text
literals = (-12, -11, ..., 11, 12)
arguments = (
  ([0, 0, 0, 0],),
  ([0, 0, 0, 0],),
)
desired_outputs = (0, 1)
```

The duplicate input with conflicting desired outputs deliberately forces an
exhaustive no-solution traversal without using an invalid output type.

Measured aggregate-layer telemetry:

- wall times: 19.18 s, 20.00 s, and 20.67 s;
- status: ordinary `STOP_NO_EXACT_TYPED_SEMANTIC_PROGRAM`;
- declared root candidates: 4,797;
- supported root candidates: 4,757;
- observed root semantic classes: 87;
- operator predictions: 23,642;
- physical races: 63,456;
- retained representatives before cost 9: 237;
- worst bucket: 87 representatives.

Exact tier `(cost, declared candidates, retained root representatives)` rows:

```text
(1, 25, 25)
(2, 26, 0)
(3, 1600, 24)
(4, 24, 0)
(5, 3024, 30)
(6, 2, 0)
(7, 24, 8)
(8, 0, 0)
(9, 72, 0)
```

Retained bucket totals before cost 9 were:

```text
root Int:                   87
bound Int:                  73
root List[Int]:             73
bound Bool:                  2
root Function[Int,Bool]:     2
```

Acceptance:

- the fixture completes with the ordinary exhaustive STOP, not the work-bound
  STOP;
- deterministic counters and tier rows above remain exact unless the banked
  String layer makes an explicitly reviewed shared-path change;
- the checkpoint remains immutable;
- host wall time is below 60 seconds, with below 30 seconds retained as the
  engineering target rather than a cross-machine scientific verdict.

## Sentinel B: diverse permutation work-bound STOP

Signature and literal bank are identical to Sentinel A.

Fixture:

```text
arguments = (
  ([-12, -4,  5, 12],),
  ([ 12,  5, -4, -12],),
  ([ -4, 12, -12,  5],),
  ([  5, -12, 12, -4],),
)
desired_outputs = (0, 1, 2, 3)
```

All rows are permutations of the same four values.  Every current
`List[Int] -> Int` aggregate is permutation-invariant, so the conflicting
desired vector is provably unsatisfiable while the bound body sees four diverse
values in every case.

An exact physical diagnostic made under the superseded 128-entry proposal and
without the prediction/race caps measured where representative 129 would have
been attempted.  It is retained as historical localization, not the current
acceptance boundary:

- cost: 3;
- production: `maximum`;
- overflowing bucket: bound `Int` in environment
  `("Int", "List[Int]")`;
- completed bucket representatives: 128;
- completed retained total: 178;
- declared roots through the tier: 1,651;
- processed nodes: 3,439;
- operator predictions: 34,280;
- physical races: 95,084;
- wall time: 4.05 seconds;
- completed diagnostic race-ledger SHA-256:
  `4cc08ba9f976ed279c3f98728d0ff9505960b4e31c7fe30c231f10a2790bfc5d`.

In the historical post-call diagnostic with the now-implemented dynamic caps,
the prediction boundary was reached first at cost 3:

- observed post-call predictions: 30,001;
- observed post-call races: 77,968;
- bound bucket representatives: 124;
- retained total: 174;
- processed nodes: 3,015;
- wall time: 3.76 seconds;
- diagnostic prefix SHA-256:
  `4f6ba3d331964ee0ec986708ec20ec08b300a0a46c0db021c1ad3d855d2100fc`.

The implemented pre-admission check denies that call at exactly 30,000
completed predictions.  Current verified pre-call telemetry is:

- bound: `operator-predictions`, `completed=30_000`, `attempted=30_001`;
- interrupted cost: 3, with no cost-3 tier attempt or desired-output race;
- completed physical races: 77,964;
- completed declared roots: 1,651;
- completed supported root candidates: 1,441;
- completed retained representatives: 174;
- completed-prefix race-ledger SHA-256:
  `506b110fac0558063293d08527c8c0ae6ad89849e5d97b24859d6bd82123970c`.

This is the completed pre-call prefix and therefore deliberately differs from
the historical post-call diagnostic digest above.  Changing the inconsistent
desired vector preserves the same audit and prefix digest.

Acceptance:

- return `STOP_SEMANTIC_FRONTIER_WORK_BOUND` in under 60 seconds on this host;
- report `operator-predictions` as the first bound under the frozen check order;
- complete no more than 30,000 predictions and 80,000 races;
- do not run a desired-output race for the interrupted cost-3 tier;
- preserve the frozen checkpoint;
- record the exact completed-prefix counters and race digest;
- rerun with another inconsistent desired vector and obtain the identical bound
  name, interrupted cost, counters, representative state, and race digest.

## Mechanism regression matrix

The work bound must leave the existing self-posed aggregate mechanisms intact:

- direct generic `sum` at exact AST cost 2;
- algebraically reduced generic predicate-count at exact AST cost 6;
- explicit map/add predicate structure enumerable at exact AST cost 8;
- clamp-map-sum selected at exact AST cost 9;
- desired-output-independent retained-frontier prefixes;
- unsupported aggregate outputs pruned only after their physical work is
  counted.

Boundary off-by-one tests must also establish:

- 256 representatives in one bucket are admitted and 257 is denied;
- 6,000 declared roots are admitted and 6,001 is denied before tier work;
- 30,000 completed predictions are permitted and prediction 30,001 is denied
  before physical execution;
- a prediction whose conservative race reservation would exceed 80,000 is
  denied before physical execution;
- no partial tier reaches the desired-output selector.

## Completeness and claim boundary

Inside all four bounds, this protocol removes no node or representative.  The
same complete exact-cost tiers, physical observational representatives, and
desired-vector races remain.  The admitted sum, count, and clamp compositions
therefore keep their existing exact-cost completeness.

When any bound fires, synthesis becomes intentionally undefined beyond the
completed prefix.  The STOP is not evidence that no higher-cost solution
exists, and no completeness claim survives for the interrupted tier or later
tiers.  Returning STOP rather than racing a truncated frontier is the mechanism
that preserves scientific honesty.

This is a target-independent runtime-safety protocol and self-posed mechanism
result only.  It is not a six-family TRAIN gate, prompt-only completion,
family-disjoint transfer, DEV/evaluation, M0, held-out, Path-2, C1, SWE-bench,
Gemma, or Opus capability result.  No canonical desired vectors or property
rows may be opened to implement or tune it.
