# Typed-composition A0/A1/A2 v1 decision and run contract

**Status:** implemented and frozen for review; material TRAIN, calibration, and
future-cohort execution is on hold

**Failure-analysis base:** `9ee5fb7fa1277001c1a8e386d6ddfcdb3250923d`

**Implementation base:** `a33d2ffd52e3365fb9e4f93c052453cd55ec8189`

**Retired evidence:** public DEV v1; it will never be rerun or used for tuning

**Raw protocol SHA-256:**
`13a3b443b253d61cb0f221c3522b5bb95c012502965b22d97ea0e63fe1686136`

## Decision

The smallest next end-to-end increment is one frozen three-lane causal
comparison over the same 128-cell fitted GLIF5 reservoir and ridge readout:

| Lane | Framing | Strengthen positions | Status |
| --- | --- | --- | --- |
| A0 | prompt, context delimiter, canonical opaque signature, input delimiter | every next-token transition | diagnostic only |
| A1 | canonical opaque signature, context delimiter, prompt, input delimiter | every next-token transition | diagnostic only |
| A2 | exactly the A1 observation stream | program-token targets only | sole eligible lane |

A0 preserves the failed v1 framing/readout-learning pattern inside the new
balanced corpus. A1 tests the representation hypothesis that putting prompt
bytes last prevents their direct traces from decaying behind a long signature.
A2 tests the learner hypothesis that excluding framing and prompt-prefix targets
prevents the ridge readout from being dominated by irrelevant byte prediction.
A1 and A2 observe the same token IDs and exact eligibility-state bytes in the
same order; only calls to `strengthen` differ. Tests freeze that call/state
parity. A0 and A1 use identical prompt bytes, signature JSON bytes, vocabulary,
cell count, seed, readout, teachers, order, and number of corpus passes.

No context latch, phrase tokenizer, sparse three-factor rule, reservoir growth,
recurrent-weight learning, search, beam, repair, constant folding, semantic
macro, or grammar-side semantic preference is part of this increment. This
deliberately diverges from doing every later research recommendation at once:
the v1 failure already exposes two much cheaper causal defects, so framing and
target filtering are isolated before adding state or learner capacity.

The primary-source basis and the distinction between published results and our
inferences are recorded in
[`typed-composition-primary-sources.md`](research/typed-composition-primary-sources.md).
In brief, fixed-reservoir/readout separation follows the established echo-state
and liquid-state pattern; the held operator/combinator cross-product follows
SCAN; and the grammar borrows only score-preserving syntactic/type rejection
from constrained decoding. Sparse three-factor/e-prop-like learning remains a
later matched lane because the present ridge comparison is the smallest causal
experiment.

## Why DEV v1 collapsed to `variable-0`

The immutable v1 TRAIN sequences had lengths `[180, 190, 281]`, or 651 tokens.
One corpus presentation therefore contained 648 next-token targets. Only 16
were program tokens; 632 were framing, prompt, signature, or already-observed
input transitions. The readout saw a 39.5:1 non-program/program target imbalance:
program targets were only 2.469% of updates. Across three presentations this was
1,944 strengthen calls: 1,896 non-program and 48 program. `variable-0` appeared
four times among the sixteen program targets, more than any complete short-label
alternative, and it was also the only one-token teacher.

The prompt was observed before the canonical signature. From the final prompt
byte to the first program target, the three committed suffix distances were 150,
154, and 233 token intervals. With the fixed 5 ms token interval and 80 ms
eligibility time constant, an unreverberated trace retains only
`exp(-distance*5/80)`: respectively `8.482e-5`, `6.606e-5`, and `4.738e-7`.
Both unseen typed prompts then produced the strongest short TRAIN label,
`variable-0`. This does not prove which defect is causal; it motivates A1 and
A2 as separately falsifiable changes.

## Public TRAIN and calibration corpus

The identifier-independent literal set is
`[-9, -7, -5, -3, -1, 0, 2, 4, 6, 8]`. Four ten-task factors are generated:

1. scalar `add(variable-0, literal)`;
2. scalar `subtract(variable-0, literal)`;
3. scalar `subtract(literal, variable-0)`; and
4. list `map(bind add(variable-0, literal), variable-0)`.

Scalar and list identity controls add two tasks, for 42 TRAIN tasks total. Each
family has two fixed surface templates; source-literal ordinal parity assigns
one template. The 42 trusted task specifications are sorted by a separate
domain-separated SHA-256 key before opaque ordinals are assigned, so family
blocks do not become a presentation-order cue. Each task has three public cases:

- scalar arguments `-11`, `0`, and `13`;
- list arguments `[]`, `[-11, 0, 13]`, and `[9, -4]`.

The exact TRAIN arithmetic is mechanically recomputed by the loader and tests:

| Quantity | Per pass | Three passes |
| --- | ---: | ---: |
| Tasks | 42 | 126 presentations |
| Semantic cases | 126 | scoring is once, not repeated |
| Program targets | 272 | 816 A2 strengthen calls |
| Framing tokens | 8,484 | 25,452 observations |
| Observed sequence states (`frame + teacher - final`) | 8,714 | 26,142 per lane |
| A0/A1 all-transition strengthen calls | 8,714 | 26,142 |

The 8,484 framing tokens decompose into 7,156 canonical signature bytes, 1,244
UTF-8 prompt bytes, and 84 delimiter tokens. Program targets are
`30*6 + 10*9 + 2*1 = 272`. Observations are
`8,484 + 272 - 42 = 8,714` because the final teacher token is not observed when
there is no following target.

Calibration is public but never grown. It uses literals `[-7, -1, 2, 8]` in
all four factors with the opposite prompt template from TRAIN, plus two
reworded identities: 18 tasks and 54 cases. Its teacher tokens, cases, family,
template, literal, task ID, and ordinal stay in trusted scoring records. They
are not arguments to `complete`, are not embedded in signatures, and cannot be
consulted until all 180 completions (three lanes times 60 tasks) have been
byte-sealed.

## Opaque typed-signature boundary

Every entry point is an eleven-character valid identifier of the form
`q_[bcdfghjklmnpqrstvwxyz]{9}`. Its body is derived only from
`SHA256(domain || NUL || ordinal_u16be)` by mapping the first nine digest bytes
onto the frozen consonant alphabet. The one parameter is always the neutral
five-byte name `datum`. The only meaningful signature information is the
required callable type (`Int -> Int` or `List[Int] -> List[Int]`).

Negative tests reject family, operation, literal, template, task, or teacher
text in signature JSON. Task IDs are retained solely to join sealed output to
trusted evidence after completion. The model boundary is exactly:

```text
complete(checkpoint, CompletionRequest.prompt, CompletionRequest.signature)
```

`CompletionRequest` has exactly those two fields. The host grammar may inspect
only the type and emitted prefix to reject invalid tokens. It never ranks an
allowed token, reads prompt semantics, chooses an algorithm, searches, retries,
repairs, or folds constants. `subtract : Int x Int -> Int` is one general
primitive and lowers literally to Python `ast.Sub` in either operand order.

## Frozen diagnostics and A2 capability gate

Each race seals the exact 1,024 little-endian bytes of its 128-float64
eligibility vector and the current of every allowed output cell before the race.
These are evidence-only measurements, never chooser inputs.

State separation uses all unordered TRAIN task pairs with the same result type
and different first teacher tokens. For first-race vectors `e_i,e_j`, distance
is `1 - cosine(e_i,e_j)`, with both-zero defined as zero and exactly-one-zero as
one. Count, minimum, p10, median, and mean are reported separately for `Int` and
`List[Int]`. The preregistered diagnostic hypothesis is that A1 exceeds A0 at
p10 and median in both strata. It explains outcomes but cannot veto an A2 pass.

Direct prompt-trace retention uses every TRAIN task's final committed prompt
byte and first program decision. New opaque signatures are exactly 151 bytes
for scalar tasks and 225 for list tasks. Therefore distances are A0 scalar 153,
A0 list 227, and A1/A2 one token for either type: 765 ms, 1,135 ms, and 5 ms.
Retention is exactly `exp(-(distance*5ms)/80ms)` and gets the same five summary
statistics by type. The A1>A0 hypothesis is also nonveto.

For every competitive TRAIN position in an exact-replay A2 completion, target
margin is `I(expected) - max(I(other allowed))` in pA. Count, minimum, p10,
median, and mean are reported, and every margin must be finite and strictly
positive. After seals exist, the scorer also groups decisions by exact
`(allowed_cell_ids, eligibility_bytes)` and fails if a group contains different
expected targets. This is an exact-byte state-collision check, not a hash-only
proxy.

A2 is eligible only if all of the following pass together:

- 42/42 TRAIN label replay and 126/126 TRAIN cases;
- 18/18 calibration label replay and 54/54 calibration cases;
- no same-allowed-set/different-target exact eligibility-byte collision;
- at least one competitive TRAIN decision and a strictly positive minimum
  trained target margin;
- protocol/source integrity, one valid cell race per emitted token, raw
  checkpoint and cell/synapse immutability, completion-attempt accounting,
  exact observation/strengthen accounting, and exclusive evidence creation.

A0 and A1 are diagnostic even if they score perfectly. There is no trained
variant selection and no calibration-driven hyperparameter choice.

## Cheap falsification gates before any larger brain

The frozen measurements distinguish four hypotheses without silently choosing
a new architecture:

| Hypothesis | Cheap falsifier | Consequence of failure |
| --- | --- | --- |
| Corpus | A2 replays TRAIN but fails opposite-template/literal calibration while states and margins remain distinct/positive | retire this calibration cohort; design a new public TRAIN acquisition protocol, never tune and rerun these 18 |
| Representation | A1 fails to improve first-decision separation or exact different-target state collisions remain | inspect byte framing/timescale representation; do not blame readout capacity first |
| Learner/readout | A1/A2 state bytes match exactly, states separate, but A1 fails and A2 gains positive margins/replay | program-only target filtering is causal; keep reservoir fixed |
| Reservoir capacity | A2 still has exact state collisions or inadequate separation after prompt-last framing, while target filtering and corpus integrity pass | preregister a separate reservoir-size lane; do not resize inside this run |

If fading distance is the defect, the next frozen stage tests heterogeneous
timescales before more cells. If states separate but margins fail, the next
matched stage tests the already-researched sparse three-factor readout against
ridge without calling it GLIF5 e-prop. A physical context latch (two dedicated
GLIF populations), a TRAIN-byte-learned/frozen phrase tokenizer, and reservoir
sizes above 128 are later, mutually isolated stages. Any tokenizer may learn
only raw public TRAIN prompt bytes without family, template, task, teacher, or
oracle labels. No trace may encode a host-chosen algorithm.

## Fresh future cohort and anti-overfit boundary

The future cohort is specified but does not exist. There is no seed key, anchor,
task file, or generated item. Only after A2 passes every frozen gate and this
source bundle is committed and reviewed may a new public seed be created once
and anchored before generation. It will contain exactly 16 teacher-free tasks:

- six `map(input - literal)` held compositions;
- six `map(literal - input)` held compositions;
- two scalar subtraction controls; and
- two map-add controls.

Every atomic token and both subtraction operand orders are public TRAIN facts,
but `map x subtract` is wholly held out, following the SCAN-style factor
cross-product. Future literals/templates are generated from the later anchored
seed and disjoint from calibration task bytes. No teacher enters growth or
completion. There is one checkpoint, one completion per task, semantic pass@1,
no retry, no alternate seed, no search/repair, and no resume.

DEV v1, hidden data, canonical tasks, Fly Brain, and any future benchmark output
are forbidden generator inputs. A failed TRAIN/calibration run is terminal and
preserved. Its calibration tasks become spent evidence, not a tuning set. A
later hypothesis requires a new protocol and a wholly fresh cohort.

## Execution and resource contract

The runner is plan-by-default. Planning only validates and hashes the protocol
and source bundle. Material execution additionally requires `--execute`, a
unique `--run-id`, the exact lowercase raw protocol SHA-256, and the exact
reviewed source-bundle SHA-256 printed by the pinned plan.
It exclusively creates the run directory and every record; an existing path is
a terminal refusal. It writes checkpoints as they finish, seals all completions
before scoring, preserves a terminal failure record on any exception, verifies
source and checkpoint bytes again at the end, and never resumes or overwrites.
The 128-cell comparison rejects a forced CUDA backend so all three lanes use
the same native CPU path. The native loader frames and hashes `Cargo.toml`,
`Cargo.lock`, an optional `build.rs`, and every recursive `src/**/*.rs` input in
normalized relative-path order into a full SHA-256 cache address. It attests
that content-addressed cache with a
canonical integrity manifest; execution rehashes the actually loaded
binary before the first model construction, after every lane, and immediately
before final publication. Evidence persists the portable pre/post source/cache
digest, binary size, and SHA-256, never an absolute machine path. A missing,
stale, or byte-mismatched cache manifest fails closed.

The old v1 measurement (651 sequence tokens, three presentations) took 0.770 s
wall and 3.469 s process for growth. Linear token-event scaling predicts about
31 s wall and 140 s aggregate process for three new lanes; 180 completions add
roughly 18-30 s based on prior fitted-cell races. The preregistered planning
estimate is therefore **45-90 s wall, 2-4 min aggregate process**, under
**128 MiB peak memory**, about **3 MiB of checkpoint bytes**, and under **10 MiB
of evidence**. These are estimates, not results or gates.

No material command has been run for this stage. The run hold remains in force
until the pinned implementation receives two-axis review and the exact protocol,
source-bundle, and commit hashes are reported.

## Implementation and test seams

- `src/humanbrain/typed_program.py`: v4 A0/A1/A2 growth protocol, program-only
  A2 strengthening, opaque prompt-last completion framing, exact race state and
  current audits, and the general subtraction lowering.
- `src/humanbrain/typed_composition.py`: frozen factor generator, opaque IDs,
  metadata-stripped requests, future commitments, and mechanical accounting.
- `src/humanbrain/typed_composition_runner.py`: plan/source binding, exclusive
  evidence, seal-before-score boundary, diagnostics, and exact eligibility gate.
- `src/humanbrain/native.py`: reusable portable loaded-library identity,
  source-addressed cache attestation, and exact pre/post binary rehash.
- `benchmarks/typed_composition_protocol_v1.json`: raw public commitments.
- `benchmarks/typed_composition.py`: plan-by-default CLI.
- focused tests cover subtraction, checkpoint v4 integrity, lane call/state
  parity, identifier leakage, exact arithmetic, source binding, sealing order,
  A2 gates, and synthetic failure preservation without capability training.

Checkpoint v3 is deliberately unsupported; there is no compatibility loader or
fallback. The retired v1 artifacts remain historical evidence only.
