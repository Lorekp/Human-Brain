# Integrated local learning revision

Selected for implementation, 2026-09-10, following retained bank `7271b9c`.
Implementation and [three focused QA checks](integrated-local-learning-v1-qa.md)
are complete; no experiment execution is released by this document. This coupled revision keeps the four existing stochastic populations,
the same reader/action actor and its forward computation. The
[roadmap](roadmap-reading-to-coding.md) and
[active plan](persistent-language-action-development-v1.md) retain authority.

The [historical attribution](repeated-teaching-literal-attribution-v1-result.md)
identifies small teacher contributions plus scan-driven competition from space.
Teacher-only coefficient rescaling does not resolve the weak first-byte mode
mixture or prove acquisition. The revision therefore changes the observed-symbol
scoring objective and causal update units together with where unconditional scan
emission learning is stored. It adds no request feature, population, host-selected
mode target, backpropagation, backward weight transport or reverse Jacobian.

## One rule, with explicit scope

Use these five connected changes:

1. Score the actual selected emitter with bounded, proper Brier utility.
2. Give context/gate causal credit in observed-event units using the mean of
   their own prefix eligibility, without the outer transaction `K*L` divisor.
3. Give mode immediate observed-event credit and literal direct per-contact CE
   teaching, without the transaction `K*T` divisor.
4. Center **scan-only** mode/literal associations on their own presynaptic phase
   mean; store unconditional scan error in the existing active phase afferent.
5. Keep the predecessor's actual public-consequence term and motor updates
   exactly unchanged.

Teacher emission associations remain uncentered. A feature that is constant
within one answer, such as its sampled task-carrying context, must remain able
to associate with that answer. Centering each teacher answer would erase precisely
that potential source of request separation. No gain or seed sweep is proposed;
`eta=.01` remains the only update prefactor, with changed effective units stated
openly below.

## Signal provenance and proper scoring

On a scan or teacher tick, the existing populations choose context, gate, mode
and, only when mode is literal, a literal symbol. All choices, probabilities,
activities and cursors are settled before the real target arrives.

If literal was selected, retain its pre-target probability vector `p`, its own
activity and the scalar `H=sum_i p_i^2`. Upon observing byte/EOS `b`, use

`rho = 2*p_b - H`.

For the actual deterministic prior/register mode, use `rho=+1` when its emitted
symbol equals `b`, and `rho=-1` otherwise. This is exactly the same Brier utility
for that emitter's one-hot distribution on the emitted-symbol alphabet, including
invalid sentinels; the literal distribution extends by zero on those sentinels.
Invalid/absent emissions remain actual
wrong deterministic predictions when they cannot match the offered target.
Do not evaluate an unselected literal distribution or draw another sample.

For true outcome probabilities `P`, expected utility is
`E[rho]=||P||^2-||p-P||^2`, uniquely maximized by truthful `p=P`. Also
`-1 <= rho <= 1`. A uniform 257-symbol literal distribution earns `1/257`;
an incorrect deterministic emitter earns -1. Their score difference is
`1+1/257`, versus `2/257` under the old expected-match utility: 129 times larger.
This is a scoring-objective change, not a fitted reward multiplier. A correct
deterministic emitter still earns the maximum +1.

Properness applies to the actual selected emitter's reported distribution. It
does not make the whole stochastic network an unbiased optimizer of marginal
sequence likelihood. Literal cells continue to receive direct CE error, a
distinct proper local objective. The Brier scalar modulates upstream cells'
own eligibility; it is never differentiated through literal probabilities or
used to construct desired context/gate/mode labels.

Hold the three existing category baselines from before the transaction and
write `a_t=rho_t-b_category`. Their values remain in [-1,1], hence `|a_t|<=2`.
After publication, use the existing count/mean update with the category's new
mean Brier utility. Retained historical means/counts are explicitly initial
control-variate priors under the revised score, not recomputed past Brier data.
Do not reset them or silently relabel their historical measurements.

## Context and gate: causal mean-prefix credit

For either population, its actual own eligibility on event t is
`e_t=(one_hot(sample_t)-p_t) outer x_t`, with unit-energy `x_t`.
Let `E_c,<=t=sum_(u<=t) e_c,u`, with separate scan and teacher banks. Let S be
all externally offered scan symbols and T the declared teacher symbols, including
EOS. The revised observed-symbol update is

`Delta W_up = eta [sum_(t=1..S) (a_scan,t/t) E_scan,<=t`

`                   + (sum_(t=1..T) a_teacher,t) E_scan/S`

`                   + sum_(t=1..T) (a_teacher,t/t) E_teacher,<=t]`.

Absent teacher terms are zero. The teacher's first event uses t=1 regardless
of answer length. Each received outcome credits the mean of its causal own
prefix, rather than a whole-transaction mean followed by another division by
`K*L`. The separate common-scan mean retains teacher credit to request/source
processing exactly once. The teacher fork inherits the actual scan but not the
actual free branch's context/cursor trajectory. No future teacher symbols enter
an activity, baseline or prediction before observation.

The existing forward A/P construction implements this without a reverse pass.
Within a category, accumulate `A'=sum_(observed t) a_t/t`; each new own draw adds
`e` to E and `A'_before*e` to P. At finish, `A'_final E-P` equals the desired
causal sum. Teacher scan transfer is the separate scalar `sum a_teacher`
times `E_scan/S`. No dense per-event matrix traversal or new recurrent derivative
is needed. These are different update units, not an unbiased rescaling of the
old estimator.

## Mode and literal: actual observed-event associations

For mode use the immediate postsynaptic vector
`g_mode,t = a_t (one_hot(actual_mode_t)-p_mode,t)`.
For literal use
`g_literal,t = one_hot(b_t)-p_literal,t` only on an actually selected literal
contact; other ticks contribute no literal update.

Teacher updates are simply
`eta sum_t g_mode,t outer x_mode,t` and
`eta sum_(selected literal t) g_literal,t outer x_literal,t`.
They retain all task/context afferents, actual inclusion and held pre-target
probabilities. They do not normalize by the answer length, category count,
number of selected literals, matches or free stopping time. This increases the
present teacher literal coefficient by 24 or 69 times; it does not multiply
probabilities or guarantee a correct output.

Mode receives no delayed forced-prefix outcome terms under this revision.
During input observation or teacher forcing, its emitted symbol does not set
the next input, h or cursor state. Under independent future uniforms and fixed
targets/held weights/no mode-dependent early termination, later observed rewards
have zero expected causal dependence on that mode choice. Removing those terms
removes sampling variation, not useful state-transition credit under those
conditions. The actual fixed-seed PRNG still has branch-dependent draw consumption;
no finite-path equality or zero realized delayed credit is claimed. Context/gate
future credit and actual free/public mode eligibility remain intact.

## Scan-only covariance and the existing phase afferent

The scan contains real predictive learning, not merely a nuisance to discard.
Keep its context/gate learning and its contextual mode/literal associations.
Relocate only the unconditional component of its emission update into the
already existing source/public phase afferent, which is inactive during output.

For each scan emission population separately, form a contact set: all actual
scan mode contacts for mode, and only actual selected scan literal contacts for
literal. A source scan has phase 0, a public-input scan phase 1. Never pool them
with each other or with teacher phase 2. Let N be this set's size. At each contact
the actual emission activity is `x_t=alpha_t v_t`, where `v_t` is the existing
binary active-contact vector and `alpha_t=1/sqrt(nnz_t)` is also the active phase
coordinate's value. No new forward normalization is introduced.

For N>0, retain the following transient own pre/post statistics:

`Z = sum_t alpha_t`, `m = (sum_t alpha_t v_t)/Z`,

`r_vec = sum_t alpha_t g_t`, `D = sum_t alpha_t g_t outer v_t`.

The complete scan emission proposal is

`Delta W_scan = eta [D - r_vec outer m + r_vec outer e_phase]`.

Here `e_phase` selects the already existing active phase column. Since that
coordinate is 1 in every `v_t`, its covariance column is zero; the last term
stores unconditional error there. Constant shared contacts such as tonic bias
likewise have zero covariance contribution. Context-varying contacts retain
their association with the postsynaptic signal. For N=0, all contributions are
exactly zero and no division is performed. For N=1, covariance is zero and the
single contact updates only its phase column.

The covariance is conditional on the actual contact set, particularly literal
selection. It uses completed, actually observed scan activity; no counterfactual
contact or host-chosen target is added. Each synapse combines its own presynaptic
mean/trace and its own postsynaptic error. The rank-one subtraction can be formed
once at finish, alongside the existing dense proposal. Mean/r/D statistics are
private transient learning state and must be sealed/copied with the stage/fork;
none becomes a forward feature or a persistent memory route.

This intentionally changes both shared scan learning and its effective gain.
It does not preserve the old scan delta at every input or keep its norm equal.
In particular, phase-only unconditional error affects an active output current
through its actual phase-coordinate value, rather than through all previously
shared active coordinates. Nor does centering guarantee that every scan-induced
target-versus-space margin becomes favorable: genuine scan covariance can still
interfere with output. Measure that remaining effect instead of claiming perfect
phase isolation. This is local covariance plasticity, not the exact gradient
of the unchanged forward logits' likelihood.

## Preserve actual action consequence and motor learning

For each of the four populations, add the exact predecessor term

`Delta W_public = eta*r_public/(K*L) (E_actual_scan + E_actual_free)`.

Here `K=1+teacher_present+edit`, `L=S+T+129`, and `r_public` retains its validated
actual execution meaning. These eligibility banks use the original actual
activities and actual samples, including literal samples, not the centered
scan association. This term was small in the historical attribution and is not
changed merely to make the formulas look uniform. Motor heads, their local
updates and their learned coordinates remain unchanged.

## Finite event stability and bounded publication

An own categorical error has norm at most sqrt(2). Brier advantage is at most
2 in magnitude. Consequently each context/gate scan-event term has norm at most
`2 eta sqrt(2)`. A teacher event has two mean eligibility terms (common scan and
teacher prefix), giving at most `4 eta sqrt(2)` per upstream population. These
bounds are independent of how many earlier symbols accumulated eligibility.

For a centered scan emission contact, the effective vector is
`alpha_t (v_t-m+e_phase)`. Existing geometry has 6–9 active contacts, at most
eight shared non-phase contacts, and `alpha_t<=1/sqrt(6)`. The shared part of
both v and its convex mean has norm at most sqrt(8). Thus the effective vector's
norm is at most `sqrt((4*8+1)/6)=sqrt(11/2)`. Mode's postsynaptic vector is bounded
by `2 sqrt(2)` and literal's by sqrt(2). For uncentered teacher contacts the
original activity norm is one.

Combining all four population bounds in quadrature gives at most
`eta sqrt(71)` per scan event and `eta sqrt(74)` per teacher event; these are
conservative bounds that allow a literal contact on every offer. If F is the
actual free tick count, the unchanged public term has whole-population bound
`eta |r_public| sqrt(8) (S+F)/(K*L)`, with `F<=129`.

A valid whole-transaction admission bound is their triangle sum:

`eta [sqrt(71)*S + sqrt(74)*T + |r_public|sqrt(8)(S+F)/(K*L)]`.

Source-only omits teacher/free/public terms. The predecessor `sqrt(.0026)` bound
is invalid for these new units and must not be retained or bypassed. The new
proof gives finite per-event steps and linear, not quadratic, dependence on
external offers; it is not a convergence theorem or a claim that every bounded
update is useful. Keep finite-value/probability participation checks and explicit
bound receipts. Do not add hidden clipping, adaptive gains, learning-rate searches
or target-dependent acceptance rules if a run fails a declared bound.

## Smallest implementation seam and remaining limits

The implementation seam is the existing controller's pre-target literal bank,
observed-symbol receipt, phase-local staged accumulators and finish calculation.
Add the held squared-probability norm, revised scalar Brier accounting, mode
immediate association, scan pre/post moments and their custody/work counters.
Update the rule metadata and verified bound. The learner continues to expose
the same observations, teacher fork, real execution and single atomic resolution.
Retained tensors, RNG and counters are numerical initialization, never reset.

Before learning, paired forward logits, choices, output bytes, motor proposals
and RNG must remain unchanged. Computing the pre-target norm of an already
evaluated literal distribution adds no random draw. Adding private learning
statistics does not alter acting, but their time/bytes must be accounted for.

The request population still enters context/gate, while emission sees their
existing h/register/prior relays rather than a new direct request input. Stronger
proper credit can encourage useful routing; it does not prove that the 32-state
relay or existing feature combinations preserve enough task information. The
teacher also still starts from its genuine pre-output fork, advances on intended
bytes and consumes RNG after the free response. Free output advances on its own
symbols. This rule does not erase that distribution mismatch or justify treating
a teacher-prefix score as free-response probability.

Required interpretation remains complete free reading and executed editing,
per-position coverage and full conditional mixture, preserved motor behavior,
and teacher-versus-free activity/trajectory alignment. Reuse the existing fixed
task difficulty and a separately declared finite teaching budget if this revision
is selected. The proposal addresses measured attenuation, inappropriate certainty
credit and unconditional scan competition jointly; its acquisition benefit and
remaining contextual interference are still unmeasured.

There is no automatic reuse of the previous 32-correction ceiling. For an
idealized single fixed unit input, one fixed target and initially uniform 257
cells under direct CE, the symmetric target margin obeys
`m(n+1)=m(n)+.01*(257/256)*(1-p(n))`,
`p(n)=exp(m(n))/(exp(m(n))+256)`, hence `m(n)<=n*.01*257/256`.
This source-derived timescale illustrates why 32 contacts alone do not make
the literal distribution confident. It is not an executed toy, cohort result
or bound on the full network: other paths, shared activities and scan updates
can change the trajectory. Before a new integrated run, choose a finite coverage
and resource budget that acknowledges this timescale; do not change eta or
execute an undeclared recurrence or training extension.
