# Causal observed-symbol credit in the persistent actor

2026-09-07. **Source design for the selected broader learning direction;
implementation and execution are not released by this document.** Continue the
same recurrent context/content actor. Preserve bounded upstream reward learning,
make its credit causal per offered symbol, and directly teach evaluated literal
output synapses. This is one refinement, with two explicitly different local
objectives; it is not a second learner or a claim of one global loss gradient.
The [roadmap](roadmap-reading-to-coding.md) owns selection.

## Evidence and intended change

The [complete V8 learning review](persistent-language-action-v8-learning-review.md)
finds zero neuronal task success, source raw matches129/9860 online versus132/9860
disabled, and no demonstrated transfer or repair advantage. All outputs stop at
EOS. These aggregates do not prove a particular internal cause or temporal
learning trend. They justify examining informative output/sequence credit before
assuming that more cells or repeated corrections solve the problem.

Keep V8's forward context/gate/mode/literal populations, causal raw frame, two
cursors, actual old prior, literal sampling and same execution path. No parser,
teacher address, role label, alternative action rollout or hidden pretrained
representation is added. Only already scheduled source/public-input symbols and
answer byte/EOS offers supervise this rule. Actual free output gets no invented
answer. BOS, absent markers and padding remain untaught; flags/EOS remain counted.

## Two direct local objectives and exact branch inclusion

For each actually sampled population, its own eligibility is still
`e_ij=(onehot(sample)_i-p_i)*x_j`. Preactivities, relays, previous states and
all downstream probabilities are stopped inputs, not derivative paths.

For **context, gate and mode** cells, use expected byte-match reward on the
actual sampled branch. Once the genuine target b arrives after the tick:

* If LITERAL was sampled, use `rho=2*p_literal[b]-1`, from the full literal
  probabilities already evaluated before target arrival.
* Otherwise use `rho=+1` if the actually selected prior/register output equals
  b, and `-1` otherwise, including invalid or absent prediction behavior.

The scalar rho modulates those cells' own eligibility. Do not differentiate it
through literal currents, relays or other cells. Do not evaluate or invent the
missing literal distribution on a non-LITERAL tick. Context/gate/mode noise
remains; this integrates out only the last literal sample's immediate reward.
In source and teacher scans that sampled literal is not the next input: the
actual arriving symbol advances the prefix. Free generation does feed its
sample back, and therefore keeps actual consequence credit unchanged.

For **literal output** cells, on an actually evaluated LITERAL offer use
`d_ij=(onehot(b)_i-p_literal_i)*z_j`. This is direct observed-byte log-likelihood
teaching at those output synapses, whether the sampled byte matched or not.
Non-LITERAL offers contribute zero d but remain in the external denominator.
Do not also apply sampled symbol-reward eligibility to these output synapses;
that would silently mix a second emission objective into the specified rule.
Their actual free/public-consequence term is retained below.

Thus upstream cells optimize expected match reward, while literal cells optimize
conditional log likelihood on their actual inclusion branch. This is a coupled
local design, not an unbiased gradient of a shared network objective. A literal
cross-entropy error may not be multiplied by an earlier weight transpose or
used to manufacture context/gate targets. The own-score reinforcement principle
is the established stochastic-unit mechanism; sample efficiency here is a
design hypothesis, not a literature result. [Williams, primary paper](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf).

## Fixed normalization and held baselines

Let s be the externally counted complete scan offers and t the offered answer
bytes plus EOS, or zero without a correction. Let `C=1+teacher_present+edit`.
For source-only interactions `C=1,L=s`; for observations `L=s+t+129`, preserving
V8's fixed free-decision allowance regardless of actual output length. The
learning-rate prefactor remains `.01`; no gain sweep is proposed. Teacher count
is declared after the actual response and before teacher draws, from the
external offer, never used as an input feature or inferred from correctness.

Keep three historical baseline slots: source scan, public scan, teacher. Their
values are held from before the interaction. For category q, write
`a_q,t=rho_q,t-b_q`, `A_q=sum a_q,t`, and
`G_q=sum_t a_q,t*E_q,<=t`, using that category's own cumulative eligibility.
Absent teacher terms are exactly zero. Proposed upstream updates are:

`Delta_up = .01/(C*L) * [G_scan/s + (A_teacher/t)*E_scan
                       + G_teacher/t + r*(E_scan+E_free)]`.

For source-only, omit teacher/public/free terms. Here r is the original validated
actual edit consequence: -1 for failure, otherwise0; STOP retains0 and the edit
category. Context/gate/mode common scan eligibility receives the whole later
teacher outcome; it is not reset or replaced by a second scan. Public reward
continues to use all original sampled eligibility, with no expected execution,
private grade or motor blame oracle.

For literal cells, let D_scan/D_teacher be sums of their direct d increments:

`Delta_literal = .01/C * [D_scan/s + D_teacher/t]
                 + .01/(C*L)*r*(E_literal_scan+E_literal_free)`.

These denominators include **all offered symbols**, including non-LITERAL
ticks. Per-category means keep a scarce offered correction from being diluted
by a long unrelated input scan. They do not normalize by number of matches,
sampled literal draws or free response length. Public consequence has exactly
its V8 scale and eligibility. Reading without correction still has no desired
answer reward on the actual free branch.

Compared with V8, upstream category means and the outer L division are retained;
causalization drops only past-reward terms from later choices. Under ideal
conditional sampling, those terms have zero expected own-score contribution.
This avoids an accidental O(s) upstream gain increase. Literal direct teaching
instead uses one category mean, without another division by L: this explicitly
changes its objective and length scaling. It is not described as variance-only
optimization. Update baselines from category mean rho only after the held
transaction, with the existing historical transaction counts. Do not condition
a cumulative-trace baseline on an already sampled context or target.

## Bounded publication and forward sparse implementation

Unit input implies an own categorical eligibility norm at most sqrt(2).
Across context/gate/mode the per-tick bound is sqrt(6). Since `|a|<=2` and
`|r|<=1`, the upstream proposal is conservatively bounded by `.02*sqrt(6)`.
Each direct literal increment is at most sqrt(2); the category means and
unchanged public term give a literal bound `.01*sqrt(2)`. The combined
four-matrix delta norm is therefore at most `sqrt(.0026)`, approximately0.051,
without clipping, target-dependent rescaling or another gain parameter.
Check this mathematical bound with a declared numerical tolerance before
publication. The bound is not a claim that each update is useful.

Avoid dense per-symbol accumulation. In each upstream category hold cumulative
E and auxiliary P. During a draw, before current target arrival, update only
active columns with `E+=e` and `P+=A_before*e`; after the offer, advance scalar
A. At finish use `G=A_final*E-P`. This equals the causal sum in real arithmetic
and requires no reverse traversal or stored future targets. All populations
within one tick use the same A_before. Direct literal D also accumulates only
its actual active columns when the target arrives. Compute dense proposals once
at finish; floating-point regrouping requires a finite arithmetic tolerance,
not a false bitwise-parity claim with V8's different estimator.

Retain the existing three eligibility banks for the original consequence and
ownership contract. Add two upstream P banks (context/gate/mode together:
2,802,448 values per bank) and two literal D banks (275,504 values per bank).
Their total is6,155,904 float64 values, **49,247,232 bytes per stage**. A root
plus correction fork adds98,494,464 bytes before scalar/tensor scratch, receipts
and existing state. This is retained additional state, not measured process RSS.
Sparse P adds one multiply/add per actual upstream active synapse; D adds its
local target-error products/accumulations only on evaluated literal offers.
Charge gathered columns, casts, validation, scalar reward work, final dense
products/subtractions, fork copies and complete seals. No per-symbol full
three-million-weight materialization or extra prior/feature evaluation occurs.

The smallest source seam is ControllerStage plus draw/observe/finish, with fork,
seal, discard and proposal validation extended for A/P/D and the retained
pre-target literal probabilities/activities. The latter are already evaluated
by the actual tick; seal them until observe and release after consumption.
Source admission, teacher ownership, actual action-before-correction and the
same actor's final joint validation stay at their existing boundaries.

## Continuation, custody and limits

Initialize both future neural routes as independent copies of completed V8
ONLINE, never by resetting the controller or replaying sources. The selected
historical payload is `route-online-000000262.pt`, SHA256
`f49a6496f91bca6dc431ae221034dfda9e897d0f2476ae18d2bb4b65f8617592`,
model hash `376c5a6223b727020a56bf09d62b7aed5ead069431fd989de5faf7f94adce853`,
with1982 factual slots. Authenticate that exact saved lineage in a separately
selected initializer. Preserve all numerical weights, old reader/prior state,
factual rows, baseline means/counts and RNG; adapt only declared algorithm
metadata. Existing sampled-reward baselines remain legal historical constants
when subsequent observations use conditional mean rewards.

No new learned idle-state tensor is necessary. A/P/D are transaction-private;
all must enter stage/fork ownership, digest and late-failure checks. Teacher
fork restores preoutput h/cursors but starts from the RNG frontier after actual
free output. Weights/baselines remain held through the common scan and both
branches. Disabled computes the complete proposed work but publishes no slow
state, advances acknowledged RNG, and retains the same selected factual writes.
Prepublication validation/receipt failure changes no live state; a later
journal failure stops at the preserved safe-boundary policy rather than
pretending live publication was rolled back.

Observed-label counts do not increase. Reports must nevertheless distinguish
all raw/status/EOS offers, actual literal-teaching inclusions, expected versus
sampled match signals, each local objective, free/public events and extra
computation. Do not claim unchanged biological mechanism, estimator or cost.
Public failure still cannot isolate payload blame from the motor operation/span,
and direct literal teaching can favor memorized byte regularity over useful
copying. No state-role supervision or new composition operation is supplied.
Ordinary arithmetic/custody review and any later broader exposure require root
selection; this design authorizes neither test execution nor a new benchmark.
