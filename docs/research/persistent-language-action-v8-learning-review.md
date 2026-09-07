# V8 learning review: provisional signal analysis

2026-09-07. **PROVISIONAL — corrected-run outcomes and final saved audit pending.**
This note reads the selected source and protocol only. It contains no live
journal inspection, model/feature evaluation, checkpoint decoding or new
experiment. Planned counts below are not completed exposures. The earlier
serializer failure and its discarded activity belong in the eventual complete
cost/exposure ledger; they are not corrected-run capability results.

The [protocol](persistent-language-action-v8-protocol.md) offers a plausible
learning problem, not evidence that its signals identify useful language or
action policies. Negative task scores would constrain this implementation and
experience budget; they would prove neither that the broader goal is impossible
nor that a larger population alone fixes the problem.

## What actually teaches the cells

The [controller](../../src/humanbrain/recurrent_content_controller.py) samples
32 context, eight gate and four emission-mode alternatives, plus 257 literal
alternatives only when LITERAL is selected. For each sampled population its
own synapse accumulates `e_ij=(onehot_i-p_i)*x_j`, with unit-energy sparse input.
All alternatives receive this local eligibility; no downstream Jacobian,
correct address, semantic role or desired context-cell identity is supplied.

Let S/F/T be accumulated scan/free/teacher eligibility, s and t their offered
symbol counts, R_s and R_t mean exact-symbol rewards in {-1,+1}, and r the
public execution reward in {-1,0}. Baselines b_s/b_t are held pretransaction
means, with separate source-scan, public-scan and teacher slots. Absent teacher
error is zero. For an observation interaction the implemented update is

`ΔW = .01/(L*C) * [(R_s-b_s + R_t-b_t + r)S + rF + (R_t-b_t)T]`,

where `L=s+t+129` and `C=1+teacher_present+edit`. For a standalone source,
`ΔW=.01/s*(R_s-b_s)S`. Categories are externally scheduled; STOP retains the
edit denominator with r=0. Weights stay held throughout scan/free/teacher.
The teacher restores preoutput context/cursors, uses fresh unused RNG draws
and runs the old prior once per causal offered prefix. Disabled computes the
same proposals, retains weights/baselines and publishes acknowledged RNG.
[Actual same-actor integration](../../src/humanbrain/recurrent_content_learner.py).

Own-probability eligibility can associate earlier stochastic choices with
later reward without reverse error transport. The expectation argument needs
conditional sampling and action-independent baselines; it is not a guarantee
of informative finite-sample credit or an exact finite-PRNG gradient. Here it
is conditional on each sealed initial transaction state and fixed offered
input, not full-history differentiation through discarded eligibility or
earlier task-dependent workspace changes. [Williams's primary paper](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
supplies the stochastic reinforcement-learning basis; the implementation and
its application to this workload remain our design, not a reproduced result.

The signal sources have different meanings:

| Experience | Signal and actual credit | Limitation |
| --- | --- | --- |
| Whole source and public input scan | Predict the next arriving byte/EOS before seeing it; mean reward modulates S. | Predictable syntax, repeated bytes and status fields can reward policies unrelated to instruction relevance. |
| Two answer corrections | Mean offered-byte/EOS reward modulates common S and fresh T. | No desired intermediate binding is supplied; many wrong policies can earn the same common-byte matches. Actual free F gets no answer-derived credit. |
| Actual edit consequence | Failure gives r=-1 to S and F; otherwise zero. | Exit zero is not task success. Failure can come from operation/span choice, yet this consequence updates the controller, not those motor heads. |
| Uncorrected reading | Scan update only; F coefficient is zero. | Its private answer grade never teaches the emitted answer. |

The one edit correction can separately teach three existing motor heads.
Teacher-free rejection or runtime failure does not do so. A useful payload
chosen with a bad span can therefore receive the same negative controller
credit as a bad payload. This is a source-derived ambiguity, not an observed
cause of any V8 error.

## Reward density and normalization: derived bounds only

The planned stream contains 9,884 source offers, 18,983 nominal public-scan
offers, and 35 answer byte/EOS offers per neural route. Actual retries and
changed files/diagnostics can change the public count. Of each public scan,
13 offers are flags/EOS: 832 if all 64 scheduled attempts produce a scan.
Fewer answer corrections therefore does not mean less total supervision.

At zero weights all categorical distributions are uniform. At the first raw
symbol of a reset source, both registers are invalid and PRIOR is absent.
Exact prediction then has probability `1/(4*257)=1/1028`: only a matching
LITERAL draw can succeed. This is a first-tick analytical statement, not a
measured success rate for the full stream. For an EOS target with the same
invalid registers, the probability is `1/2+1/1028`; invalid-register modes
produce EOS. Stopping can consequently improve without useful content.
Later register validity/content and learned currents change these rates.

Negative rewards are frequent signals, not absent signals. However, when an
outcome is effectively constant across choices its expected score-function
credit is zero; a particular sampled update can still be noisy. A whole-source
mean does not tell a cell which of its many choices caused the few positive
matches. The historical mean baseline reduces a constant offset but is not a
context-specific predictor of which binding is useful.

Each tick's combined eligibility norm is at most `sqrt(8)` (at most four
unit-input categorical populations). The implementation bounds a complete
proposed update by `.02*sqrt(8)`, about 0.05657. This is a magnitude bound,
not a lower bound on useful learning. On a source of s symbols, changing one
reward from -1 to +1 while holding eligibility fixed changes the update by
`.02*S/s^2`, whose norm is at most `.02*sqrt(8)/s`. This isolates reward
sensitivity; changing an actual sampled trajectory also changes S. Longer
streams provide more experience while their mean-reward and length
normalization distribute each individual outcome's influence.

## Reusable capacity versus identifiable useful behavior

The [causal frame](../../src/humanbrain/recurrent_content_frame.py) lets the same
bind/advance/read policy copy different contents from two addresses. Thus
content substitution need not memorize each output literal. Learned recurrence
can distinguish ordered histories; it is not the previous frozen reservoir.
But the 32 cell identities have no supplied meanings and are identifiable only
through outcome differences, not as unique linguistic roles. Two distinct
addresses showing the same byte can present identical current inputs unless
the recurrent state preserves their provenance. Two cursors and 32 states
provide neither arbitrary working memory nor a program interpreter.

Source scans have no request Q or old prior. Consequently they never update
the request-specific afferent columns through source eligibility. They can
teach shared byte/context/gate circuitry, while request-conditioned control
must arise through public scans and the scarce downstream answer/consequence
signals. Emission inputs also distinguish scan and output phases. Good
next-source-symbol prediction alone is therefore not identified with good
answer generation. No published language-binding result should be used to
assume that raw requests already carry learned roles; the
[selected design's primary-source review](public-observation-sequence-context-v1.md)
explicitly rejects that inference.

There are 3,077,952 new weights (24,623,616 float64 bytes) but only 32 active
context alternatives. More weights, more simultaneously useful states, more
informative experience and better credit are different quantities. Sparse
tick products do not remove dense eligibility/staging or old-prior costs.
The [source resource ledger](recurrent-content-resource-plan-v1.md) supplies
those bounds; this note predicts neither elapsed time nor adequate sample size.

## Interpretation after the actual evidence is checked

Keep first acquisition, pure paraphrase, exact delayed repeat, new construction,
structural combination, historical regression and first/later/terminal recovery
separate. First encounters receive no answer correction before acting; earlier
fresh observations still become online training exposure. Both current neural
routes begin from V7 ONLINE with a new stochastic policy, so historical changes
are not an isolated measure of forgetting caused by current learning.

Read actual output bytes, EOS/limits, public assertions and private correctness
together. A source-reward gain with unchanged fresh accuracy supports learning
of the prediction objective, not instruction understanding. Fewer crashes
without correct terminal edits supports failure avoidance, not successful repair.
Any online/control difference combines source, public-scan, answer and
consequence updates. Saved scalar receipts can check their arithmetic and
causality; omitted eligibility and learned-weight effects cannot establish
which internal state or gate caused a result.

The one shared direction remains learning reusable context/content control in
the acting system. After evidence, identify whether it learns source regularity
but fails to bind the request, or whether useful credit is absent even there,
before selecting a change. A justified successor would address that specific
causal credit/representation gap using declared public experience in the same
loop. This provisional note selects no new mechanism, larger population,
gain/seed sweep, replay or isolated capability gate.

## Possible credit refinement, not selected or implemented

The following is source/math inference pending audited V8 evidence. It concerns
the same cells and already offered source/public/answer symbols, with no new
labels, parser, target address, alternative rollout or second learner.

**Preserve the byte-match objective.** After a causal tick has run and target b
arrives, let p be its literal distribution, when that population was actually
evaluated. Conditional on sampled context/gate/mode, replace the sampled
literal reward by `r_bar=2*p[b]-1` only when mode=LITERAL. For other sampled
modes use the actual deterministic selected content's +/-1 match. Context/gate
and mode synapses may multiply their own score eligibility by this scalar;
stop every derivative through r_bar, p, inputs and earlier states. This removes
one source of literal sampling noise without claiming to evaluate unchosen
gates or contexts. Literal synapses instead have the direct expected-match
increment `2*p[b]*(onehot(b)-p)*z`, conditional on the actual literal branch.
It is their own output-current rule, not an error transported to earlier cells.
Simply giving literal score eligibility r_bar would be wrong: that scalar is
independent of the sampled literal, so its conditional expected score product
is zero and omits the direct probability term.

At uniform literal probabilities the direct expected-match factor is `2/257`;
removing variance does not remove this small objective gradient. Nor can the
implementation marginalize over all four modes for free: non-LITERAL ticks
have no evaluated literal probabilities. Treating a missing probability as
zero, or only averaging all modes on sampled-LITERAL ticks, changes the
estimator. Any additional forward population would require explicit selection
and accounting; it is not assumed here.

**A hybrid with direct observed-byte teaching.** On each actually evaluated
literal population, use `ΔW_o ∝ (onehot(b)-p)*z` instead. This is direct local
cross-entropy teaching of output synapses from the already arriving byte,
whether or not the sampled byte matched. It avoids the `p[b]` attenuation;
at uniform probabilities the magnitude is 257/2 times the expected-match
gradient for the same prefactor. That is an objective change, not merely a
variance optimization or evidence that the present gain is wrong. Context and
gate credit can still use the stopped scalar match signal above. The hybrid
then has different output and control objectives, not a single claimed global
gradient. Better literal prediction could also reduce incentive to discover
content-copy gates; useful transfer must decide the tradeoff.

For either path, “per-symbol credit” should preserve causal history. A streaming
form `sum_t (r_t-b)*E_<=t / n`, where `E_<=t` contains own eligibility up to
that offered symbol, removes unrelated earlier rewards from later choices
while retaining earlier bindings' credit for later outcomes. For the fixed
causal source/teacher stream, dropping those past-reward terms preserves their
zero-expectation contribution under the conditional sampling assumptions.
This is an application of [score-function reinforcement credit](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf),
not a proof of lower total variance in this implementation. Using only the
current tick's eligibility would discard delayed binding credit and would not
estimate the same sequence objective. Keep common scan eligibility available
to every offered teacher symbol. A baseline multiplying cumulative eligibility
must be fixed before the credited choices; a baseline derived from the current
sampled context is not automatically valid for earlier choices.

Source/teacher next inputs are actual arrivals, so the sampled literal is not
fed back into those prefixes. Conditional emission averaging is consequently
applicable there. Actual free output changes future prefixes and execution;
its public outcome cannot be replaced by expected byte reward without a supplied
answer or unperformed alternatives. Keep that consequence tied to the sealed
actual path, including its unresolved motor-versus-payload credit ambiguity.

Both paths can retain old weights/state and stage changes until the existing
atomic boundary. Neither requires another answer exposure. Per-symbol
cumulative-credit accumulation can nevertheless add dense accumulator work,
and changed objectives require declared normalization/baselines and fresh
receipts. No cost-free implementation is claimed. Evidence should determine
whether this one credit-refinement direction addresses a measured bottleneck;
neither variant is released, and no additional experiment or gate is selected.
