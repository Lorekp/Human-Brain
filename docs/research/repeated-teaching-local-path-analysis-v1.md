# Interpreting the fixed repeated-teaching path

Source-only analysis, 2026-09-10; these are algebraic bounds and diagnostic
questions, not results of the proposed run. The
[fixed acquisition protocol](repeated-teaching-acquisition-v1.md) owns the
schedule. This note selects no component, learning-rate change or additional
teaching budget.

## What one offered target can change

The controller's [observe](../../src/humanbrain/recurrent_content_controller.py#L730)
adds `(one_hot(target) - p_literal) outer x` only when the actual mode selected
the literal branch. Its [finish](../../src/humanbrain/recurrent_content_controller.py#L919)
multiplies the teacher sum by `c = 0.01 / (K T)`, where `T` includes EOS and
`K = 2` for taught reading or `3` for taught editing. Division uses all external
teacher offers, not the number that selected literal. A successful computation
with learning disabled proposes this update but publishes none of it.

The [emission activity](../../src/humanbrain/recurrent_content_frame.py#L377)
contains nonnegative unit-energy values. For a fixed diagnostic activity `x*`,
let `rho = x dot x*`, so `0 <= rho <= 1`. One selected literal offer contributes
`c (1 - p_target) rho` to its target cell's current, and
`c (1 - p_target + p_j) rho` to its margin over another cell `j`. The respective
upper bounds are `c` and `2c`. At exactly uniform literal probabilities and
identical activity, the margin increment is `c`. These are contributions to a
linear current at a fixed activity, not bounds on changing recurrent choices.

| Scope: one particular target position only | Reading, T=12 | Editing, T=23 |
| --- | ---: | ---: |
| Coefficient per actual literal inclusion | 0.000416667 | 0.000144928 |
| Largest own-position target-current contribution across 32 inclusions | 0.0133333 | 0.00463768 |
| Largest own-position pairwise margin contribution across 32 inclusions | 0.0266667 | 0.00927536 |
| Illustrative uniform-mode exposure: eight inclusions in expectation | 0.00333333 | 0.00115942 |

The last row is an upper current scale under the additional hypothetical
assumption that each offer selects literal with probability 1/4. It is not an
observed inclusion count, independence assumption, or promise of eight actual
inclusions. Actual mode probabilities and counts replace that illustration.
At uniform literal probabilities, multiply those current scales by `256/257`;
for small isolated changes the increase in a rare token probability remains
small. Whole-output reliability cannot be inferred from those current scales.

## Shared positions and competing updates matter

The table is not a bound on the total movement of a cell. All positions share
the literal matrix. Repeated occurrences of a target byte can add positive
terms, with strength determined by their activity overlap. Other target bytes
add negative terms to that row. For the declared strings, `a` and `e` each occur
twice in reading and once in editing. Even if every such offer selects literal
and all activities overlap perfectly, their positive teacher contribution to a
fixed current is at most `32 * (2/2400 + 1/6900) = 0.03130435`. This is the
largest same-byte positive bound across these two strings; it excludes negative
teacher contributions and all public-input/action updates. EOS appears once
per string and has corresponding positive bound `0.01797101`.

Activity contains context h, previous byte, two register tokens and their
validity, phase, prior token and bias. It has no separate absolute output-position
feature. Different target positions may nevertheless differ through those
inputs and the recurrent/prior path. Matching hashes establish exact activity
reuse; different hashes do not establish disjoint features. Seven to nine active
features give nonzero overlap even when several content features differ.

Every live occasion also scans its actual public observation. The literal scan
term is independently normalized by `0.01 / (K S)`, with `S` all scan offers;
there is no source ingestion in this protocol. Scan and target terms touch the
same matrix and share some features despite their different phase. Their
contributions must not be collapsed into answer teaching.

For editing, an actual negative execution consequence adds
`-0.01 / [K (S + T + 129)]` times the sum of actual scan/free literal
eligibilities. This can lower a selected wrong cell and raise other cells; it
is not a direct correction toward the intended answer. It may affect features
also used by the target path. Other public outcomes give zero for this term.
Mode, context and gate use their own causal eligibility, held baselines and
additional length normalization. Literal teacher coverage alone therefore does
not measure whether those upstream choices became more useful.

## What the already planned records can distinguish

The experiment-only [observer](../../experiments/r4_integrated/repeated_teaching_diagnostics_v1.py)
uses the private literal-weight snapshot from before publication and the
actual teacher receipts. It reconstructs each activity and checks its hash
against the real mode draw. For actual literal branches it also checks the
recorded currents against the held snapshot. Missing literal distributions are
evaluated only in the observer, with no draws, prior calls or publication.

Use the saved measurements in this order:

1. Count offered, selected-literal and published-direct coverage by skill,
   position and EOS. A small count establishes sparse direct opportunity;
   counting 32 external corrections alone does not establish effective coverage.
2. Compare literal target log probabilities, actual inclusion counts and the
   exact per-resolution teacher coefficient. Coverage with only small changes
   is consistent with the bounded update scale or competing shared updates;
   it does not distinguish them by itself. Activity hashes and existing sampled
   context/gate/register records describe reuse. A delta norm alone cannot
   locate productive current movement.
3. Inspect each mode contribution to the target emission probability:
   `p0 I(prior=target) + p1 I(reg0=target) + p2 I(reg1=target) + p3 p_literal(target)`.
   Missing registers emit EOS. Improved literal probabilities with little full
   mixture change indicate that literal improvement has little influence at
   that sampled context/gate path. Changes in prior/register matches or mode
   probabilities identify a different contributing path without proving why
   its upstream choices changed.
4. Compare those target-prefix diagnostics with actual free bytes, prefix
   alignment, EOS/limit stopping, complete answer/execute grades and the two
   fixed historical sentinels. A good target-prefix score alongside divergent
   free output leaves conditioning/reuse and stochastic reliability unresolved.
   Degradation after the other skill's block is evidence of measured
   interference in that panel, not proof of a particular conflicting synapse.

Teacher replay begins from the retained pre-output context and cursor state,
but consumes RNG after the actual free response and advances on intended
prefix bytes ([teach](../../src/humanbrain/recurrent_content_learner.py#L405)).
The observer's summed mixture log score is conditioned on this sampled latent
path. It does not marginalize possible context/gate trajectories and is not an
exact complete-output probability or empirical free success frequency. Live
teacher scores precede that correction's publication; scheduled disabled
diagnostic teachers measure the checkpoint weights without publishing.

The existing free trace stops supporting intended-prefix alignment after its
first divergence. Correct raw bytes without EOS and changed edit proposals
without a successful executed grade remain failures of complete acquisition.
The finite panel has limited stochastic replication; no automatic conclusion
that more epochs, a new component, or a different rule is needed follows from
one failed or successful panel.
