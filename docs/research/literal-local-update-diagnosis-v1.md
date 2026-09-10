# Literal update normalization and mode credit

Source-only diagnosis, 2026-09-10, following the banked repeated-teaching result
at `35be9bc`. This document proposes one bounded correction candidate for review;
it does not authorize implementation, training, a gain sweep or another mechanism.
The [roadmap](roadmap-reading-to-coding.md),
[active plan](persistent-language-action-development-v1.md) and
[measured learning review](repeated-teaching-acquisition-v1-learning-review.md)
own the current state and evidence.

## Exact current literal update

Let `eta=.01`, `S` be all public/source scan offers, `T` the answer bytes plus
EOS (zero without teaching), `K=1+teacher_present+edit`, and `L=S+T+129` for
reading/editing. Source-only uses `L=S`. Let `I_t` indicate an actual literal
selection, `x_t` its pre-target unit-energy emission activity, `p_t` its held
257-cell probabilities and `b_t` the subsequently observed symbol. The controller
[draw](../../src/humanbrain/recurrent_content_controller.py#L572) retains the
literal probabilities/activity before the target arrives; its
[observe](../../src/humanbrain/recurrent_content_controller.py#L730) accumulates

`d_t = I_t (one_hot(b_t) - p_t) outer x_t`.

Writing `D_c=sum d_t` and `E_c` for the actual sampled own-score eligibility in
category `c`, [finish](../../src/humanbrain/recurrent_content_controller.py#L919)
publishes, when enabled,

`Delta W_literal = eta/(K*S) D_scan + eta/(K*T) D_teacher`

`                  + eta/(K*L) r (E_scan + E_free)`.

The teacher term is absent when `T=0`. The public consequence `r` is -1 for a
validated failed non-STOP edit and zero otherwise; it is not a desired-byte
label. `T` includes nonliteral offers. The teacher is processed against held
weights and published once at the end, not after each symbol.

For any fixed diagnostic activity `x*`, an individual teacher contact contributes

`Delta z_b(x*) = eta/(K*T) (1-p_t[b]) (x_t dot x*)`

to its target row. Nonnegative unit-energy activity gives `0 <= x_t dot x* <= 1`.
Its target-versus-other-cell margin contribution is
`eta/(K*T) (1-p_t[b]+p_t[j]) (x_t dot x*)`, at most `2 eta/(K*T)`.
Other contacts may support or oppose that row through shared features.

## What the denominators mean

`D_teacher/T` is a mean over external answer-symbol offers; division by `K`
averages the transaction's scan, teacher and edit categories. This was an explicit
design in the [credit specification](causal-observation-credit-v1.md), with a
uniform transaction norm bound. It is not an arithmetic implementation bug.
It avoids diluting teacher credit by the unrelated scan length `S`, but it still
dilutes a particular contact by the answer's own length and the category count.

The denominators are not required for synaptic locality, target custody or
no-backpropagation. Neither local likelihood teaching nor the available data
uniquely selects a mean loss over whole answers rather than a sum of observed-byte
losses. The issue is the unit of experience: under the current rule an answer
transaction has bounded aggregate weight, while a constant per-contact rule
would give each evaluated observed symbol the same update coefficient.

| Actual taught setting | S | T | K | L | Coefficient per literal teacher contact | Relative to eta per contact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A00-R | 142 | 12 | 2 | 283 | 0.000416667 | 1/24 |
| A00-E | 387 | 23 | 3 | 539 | 0.000144928 | 1/69 |

For the same hypothetical contact and activity, increasing answer length reduces
its weight. Marking an otherwise equivalent taught transaction as editing reduces
the teacher coefficient from `eta/(2T)` to `eta/(3T)`, even when its public
reward is zero. These are real objective-weighting choices, not a necessary
property of the byte or its direct error. Calling them arbitrary should mean
that locality does not mandate these units, not that there was no stated reason
for the original normalization.

The measured 4–12 direct contacts per position imply own-position positive-current
bounds of 0.001667–0.005 for reading and 0.000580–0.001739 for editing. The
learning review finds fixed-feature target probabilities only 0.3278% and
0.2109% higher in relative terms, still near `1/257`. Coverage is no longer
wholly missing. This is evidence to examine attenuation; it does not prove that
rescaling alone will produce a correct sequence.

## Mode credit is a distinct issue

For mode, context and gate, the controller uses own eligibility
`e_t=(one_hot(sample_t)-pi_t) outer x_t`, not the literal CE vector. Once a
target arrives, its scalar signal is `rho_t=2q_t-1`, where `q_t` is the literal
target probability if literal was selected, or the selected prior/register
match indicator otherwise. The held category baseline is `b_c`; define
`a_t=rho_t-b_c`, `A=sum a_t`, and

`G_c = sum_t a_t sum_(u<=t) e_u = sum_u e_u sum_(t>=u) a_t`.

The implemented forward banks `E` and `P` produce `G=A_final E-P`. Mode receives

`Delta W_mode = eta/(K*L) [G_scan/S + (A_teacher/T) E_scan`

`                         + G_teacher/T + r(E_scan+E_free)]`.

The same causal form applies to context/gate with their own activities and
eligibilities. The common scan term is counted once, not replayed teacher input.

At one fixed sampled context/gate state, let `q_i` be each mode's conditional
target-match probability, including literal probability for mode 3. For ideal
categorical sampling, the expected **immediate** mode-score factor for cell i is

`E[(rho-b)(1[mode=i]-pi_i)] = 2 pi_i (q_i - sum_j pi_j q_j)`.

The held baseline cancels in this expectation; it still affects sample variance.
If the prior and both registers cannot match, `q=(0,0,0,p_literal[b])`. With
approximately uniform modes and `p_literal[b] approximately .0039`, the literal
cell's expected immediate factor is approximately 0.0014625. This is a small
absolute expected-match advantage, not the roughly unit-sized literal CE error
`1-p_literal[b]`. Its teacher immediate coefficient is additionally
`eta/(K*L*T)`: about `1.47232e-6` for reading and `2.68882e-7` for editing.
Their products are about `2.15e-9` and `3.93e-10` per offer in the literal mode
cell's expected current at the same activity. These illustrative values concern
only the immediate term at that stated condition, not the full realized mode
update or a prediction for the trained system.

If a prior/register already matches, it can instead have a large advantage over
near-uniform literal output. Thus small literal improvement need not raise every
full emission probability. The measured editing mixture mean fell slightly
despite improved literal probabilities at all its fixed-feature positions.

During teacher forcing, a selected mode's emitted symbol does not advance the
prefix: the genuine target does. Mode also does not change the sampled context
or gate cursor state. Under ideal independent uniforms, later teacher rewards
therefore have no causal dependence on that earlier mode choice once its
context/gate state is fixed. The delayed mode eligibility terms then contribute
zero expected credit for that choice while adding sampling variation. Literal
selection does consume an extra RNG draw, so fixed-seed sample paths can differ;
this statement is about the sampling expectation, not paired finite traces.
It requires a fixed declared target sequence, held parameters throughout the
teacher, no choice-dependent early termination, and fresh independent future
uniforms after the optional literal draw. The actual fixed-seed PRNG trace is
deterministic: consuming that extra draw changes which later numbers are used,
so changing one recorded mode can change every later sampled state. Zero expected
delayed mode credit does not imply zero realized credit, trace invariance, or
permission to delete those terms from the banked rule.
Context/gate choices do affect later state, and free mode choices affect later
generated prefixes; this reasoning does not discard their future consequences.

These observations diagnose the present objectives. They do not select a new
mode target, propagate literal errors upstream, or establish that mode credit
must change to acquire the current skills.

## What saved evidence can isolate without training

The current summary already establishes identical emission activity at all 35
checkpoint target positions, exact disabled constancy, slight enabled literal
improvement and some useful motor learning. It cannot separate every additive
weight contribution: final scalar probabilities are not a decomposition of the
updates that produced them.

A bounded, authenticated read of existing full receipts could provide the
following further diagnosis without learner, prior, executor or tensor-checkpoint
calls. Such a read is a possible next evidence action, not performed here.

1. Reconstruct each actual teacher emission activity from its prefix, h,
   registers, phase and prior, verifying the controller's exact activity hash.
   For a selected literal contact, its full pre-target probability vector is
   already saved. Reconstruct each diagnostic reference activity from the full
   checkpoint teacher receipt, not from its hash alone.
2. Project the teacher update onto that fixed reference activity, by summing
   `(1[target=j]-p_t[j]) (x_t dot x*)` with the actual coefficients. Separate
   the same position, other positions with the same byte, and competing targets.
   This computes an additive current contribution at fixed input, including
   overlap; it does not require a new prediction or target gradient.
3. Compute the same frozen-history projection with constant `eta` instead of
   `eta/(K*T)`. For each transaction's teacher component the ratio is exactly
   `K*T`, apart from declared floating-point regrouping tolerance. This falsifies
   or confirms an accounting claim about attenuation, not a claim about what
   a newly trained trajectory would have done.
4. Using the saved mode probabilities, actual selected modes and scalar credit,
   split realized mode contributions into same-tick teacher, later teacher,
   scan-to-teacher and actual public terms. Mode emission activities are similarly
   small. Complete scan/public projections additionally require reconstructing
   and authenticating their preceding public-input symbol chronology; do not
   infer them from teacher receipts or omit them while claiming a complete delta.

A large opposing contribution at identical input would support interference as
an additional explanation; tiny unopposed teacher contribution would support
attenuation. Either finding is more specific than a nonzero update norm. A
frozen-history counterfactual cannot identify post-change choices, subsequent
probabilities, altered RNG consumption, retention or execution success. Saved
probabilities do not supply an unobserved alternative latent trajectory.

## Smallest correction candidate supported by this diagnosis

For review, change only the **direct answer-teacher literal coefficient** to a
constant per evaluated contact:

`Delta W_literal_candidate = eta/(K*S) D_scan + eta D_teacher`

`                            + eta/(K*L) r(E_scan+E_free)`.

Keep actual literal inclusion, pre-target probabilities, held activities,
staged accumulation, one validated publication, mode/context/gate objectives,
public consequence, motor heads and forward generation unchanged. Do not divide
by the number of literal selections, teach missing branches, alter source-scan
weighting or publish intermediate teacher steps as part of this candidate.
Those would confound the normalization question with another learning change.

Removing `K*T` here preserves own-synapse locality and the forward architecture:
the target still meets only the already evaluated literal population after its
draw, and no downstream derivative, extra draw, prior evaluation or new state is
introduced. Constant per-contact scaling does **not** make branch inclusion
sample independent; which contacts exist and their activities remain sampled.
It also is not sequential online learning within the answer: all contacts still
use the held transaction weights. Unchanged parameters/RNG therefore preserve
pre-learning forward behavior exactly, while later behavior is intentionally
allowed to change through learning.

This is an explicit 24-fold or 69-fold increase in the direct teacher coefficient
for the present tasks, despite retaining the numerical prefactor `.01`. It must
be named as a change of effective learning scale/objective units, not a harmless
refactor or variance-only optimization. Its justification is a constant unit of
direct experience plus measured attenuation, not a fitted gain selected for these
two answers. It remains a candidate whose behavioral benefit is unmeasured.

In particular, removing the denominators is not evidence that the existing
32-correction budget would acquire either skill. Consider an explicitly
hypothetical frozen history with uniform literal probabilities, identical
activity on each contact to one target, and no other updates. Constant `.01`
teaching on 4–12 contacts adds only 0.04–0.12 to that target's logit margin over
each other cell. Even the 12-contact case changes its probability from `1/257`
to `exp(.12)/(256+exp(.12))`, approximately 0.00439. A probability of one half
in that same idealized population requires margin `log(256)`, about 5.545.
Shared same-byte contacts can add, competing targets and public updates can
oppose, and actual histories need not retain these probabilities or activities;
this example is a scale check, not an upper bound on the whole actor's learning.

The mode mixture is another unresolved limit: at a recorded context where only
literal can match, even perfect literal accuracy supplies at most the current
literal-mode probability (approximately one quarter here). A frozen-history
coefficient substitution holds that mixture and history fixed; a real continued
run could change both through later credit. Therefore the candidate can isolate
normalization's additive contribution without being a plausible demonstrated
solution to acquisition. Historical attribution should precede selecting it,
and its coefficient ratio must never be reported as a predicted probability or
task-success multiplier.

The existing `sqrt(.0026)` whole-update bound relies on category means and is
not valid after this change. An unnormalized teacher sum has norm at most
`eta*sqrt(2)*T` (or the tighter actual-inclusion count), so a new explicit finite
bound and matching metadata/receipts would be necessary. For example, retaining
the original upstream bound and conservatively bounding literal scan/public
terms gives `sqrt(.0024 + 2*eta^2*(T+1)^2)` for taught interactions. This is a
mathematical admission bound, not a proposed clipping or adaptive scaling rule.
Silently retaining the old bound, skipping it, or adding clipping would not be
the candidate described here.

Before any eventual comparison, preserve the continued factual state, learned
motor weights and paired forward behavior; declare the new rule and finite
exposure budget. The unresolved question is whether stronger direct local byte
teaching yields useful complete output under the unchanged mode and context
path, or whether shared representation and routing remain limiting. This
document selects no architecture addition, seed/gain search or automatic
training extension.
