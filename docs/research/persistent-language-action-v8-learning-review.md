# V8 learning review: complete outcomes and local credit

2026-09-07. **Corrected run complete; independent saved audit passed.** Both
neuronal routes score zero in all four outcome categories. More public
experience and a locally plastic recurrent controller did not establish useful
acquisition, new-form transfer, retention or repair in this session.

Evidence is the immutable [saved audit](../../.pytest-tmp/persistent-language-action-v8-corrected-review.json),
SHA256 `037f65e62c9170205ad063d40dd0e8f9a0069d0fb9e03f7499ff1df9233194bf`,
and [result bank](../../benchmarks/persistent_language_action_v8.result.json).
The audit passes on its first launch in104.7858249 seconds at93.3828125MiB;
the corrected runner takes2206.5499429 seconds at866.890625MiB. All192
attempts,132 executions,72 source rows and6 correction rows are present,
with265 safe boundaries. This review reads those saved summaries and the
source, plus one authorized streaming output-summary pass over the audited
311,424,915-byte attempts journal. That pass verifies journal SHA256
`3efbc004382b496602325d9c77b2f5a6e3fcb8148f322c31161fce7ef7a81a5c`
and takes4.0072871 internal seconds (4.2561156 shell seconds); its
[extraction](../../.pytest-tmp/persistent-language-action-v8-output-extraction.json)
is preserved at SHA256
`1bdd10439df4678e9739adb5d8650819c895f985b6da777a5d9ffb3e285cc950`.
This is separate post-bank report analysis, outside the already frozen
2362.395290-second outer experiment total; no combined whole-project total is
claimed. No model, feature, payload decode, replay or extra audit call occurs.
The first serializer-failed run's inferred89 source offers were discarded,
then N36 was deliberately reoffered from fresh V7 initialization. Those failed
offers have no saved match trace and are not mixed into the counts below.

## Measured behavior

| Outcome | Online | Disabled | Simple reference |
| --- | ---: | ---: | ---: |
| Acquisition first, before correction | 0/2 | 0/2 | 0/2 |
| Unchanged-requirement paraphrase | 0/2 | 0/2 | 1/2 |
| Exact delayed repeat | 0/2 | 0/2 | 1/2 |
| Fresh whole constructions | 0/10 | 0/10 | 0/10 |
| Fresh structural combinations | 0/10 | 0/10 | 0/10 |
| Historical regression | 0/14 | 0/14 | 9/14 |
| Recovery, any later and terminal | 0/8 | 0/8 | 0/8 |

All64 attempts per route are supported. Fresh strata each split five reading
and five editing cases; neither actual two-stage project chain succeeds.
Simple's acquisition successes are the edit paraphrase and exact edit revisit,
not fresh construction. Its historical9/14 consists of three readings and six
edits: eight previous successes retained, one old failure improved, none lost.
Both neural routes lose the same V7 ONLINE success A04-R on the matched R12-R
observation. Both began with a new stochastic controller, so this shared loss
does not isolate forgetting caused by online weight publication. The other13
historical failures are not13 newly forgotten skills. All historical raw
observations match their declared prior records. Exact repeats have61
intervening route attempts; factual occupancy grows from1614 to1982 without
eviction. Simple retains all source text and reaches11498 stored byte/EOS
positions, an explicitly different memory mechanism.

Both neural routes produce44 accepted but incorrect splices, with no STOP or
direct protected-driver rejection. Public checks pass only2 online and4
disabled executions, all privately incorrect; therefore clean public execution
does not imply completed requirements. Simple has eight correct changed
splices, eight incorrect accepted splices and28 incorrect STOPs. Its nine
public-check passes likewise include a privately incorrect case. The driver
policy prevents direct target splices, not arbitrary imported-code side effects;
these summary counts do not attribute failures to a particular integrity or
syntax cause.

The online public passes are R01-E/R13-E; disabled adds R05-E/R09-E. All are
historical cases, not fresh or recovery successes. Simple's one incorrect
public pass is R07-E with STOP. Its other eight public passes are the eight
correct edits.

Every recovery uses three attempts and16 actual feedback-consuming retries per
route. Consecutive proposals differ in all16 neural pairs, but none recovers;
simple repeats all16 pairs and also fails. Changed stochastic proposals are not
learned repair. Free output uses280 decisions online and254 disabled. The
actual response records confirm216 and190 emitted raw bytes, respectively,
averaging3.375 and2.96875 per response. All64 outputs in each route stop on
EOS, with zero LIMIT stops. Online has20 empty outputs (five readings and15
edits); disabled has23 (eight readings and15 edits). Maximum lengths are22
and18 bytes. Stopping is reliable here, but useful content is not: neither
route answers or edits correctly. Prior NLL remains a
frozen-prior diagnostic, not the stochastic policy's output likelihood.

## Measured symbol matches and sampled choices

| Already offered target category | Online matches/offers | Disabled matches/offers |
| --- | ---: | ---: |
| Source raw bytes | 129/9860 | 132/9860 |
| Source EOS | 7/24 | 6/24 |
| Public-scan raw content | 351/24058 | 350/24422 |
| Public-scan non-status EOS | 84/294 | 74/294 |
| Public flags/status, including their EOS | 106/832 | 117/832 |
| Answer-correction raw bytes | 5/33 | 5/33 |
| Answer-correction EOS | 1/2 | 1/2 |

Source raw accuracy is about1.31% online and1.34% disabled; public raw accuracy
about1.46% and1.43%. Each route has541 public-scan matches but different raw
denominators because actual files/diagnostics differ. Both receive9884 identical
source symbols and35 answer symbols; public-scan totals are25184 and25548.
These are aggregate matches, **not temporal improvement, a significance test,
or evidence of matched post-action observations**. EOS/status are easier here
than raw content, and combining them would obscure the latter's weakness.

For compact choice accounting, modes are `[PRIOR, REGISTER1, REGISTER2,
LITERAL]`; gates are `[HOLD, BIND1_CURRENT, BIND2_CURRENT, BIND1_RETRIEVED,
BIND2_RETRIEVED, ADVANCE1, ADVANCE2, SWAP]`.

| Category/route | Mode counts | Gate counts |
| --- | --- | --- |
| Source online | 2473,2459,2525,2427 | 1137,1244,1266,1253,1284,1198,1269,1233 |
| Source disabled | 2485,2463,2511,2425 | 1170,1256,1259,1231,1228,1185,1248,1307 |
| Public online | 6252,6377,6325,6230 | 3103,3149,3131,3009,3189,3279,3162,3162 |
| Public disabled | 6328,6456,6430,6334 | 3115,3188,3185,3083,3295,3329,3224,3129 |
| Teacher, each route | 8,8,7,12 | 1,3,2,3,9,5,9,3 |
| Free online | 68,71,72,69 | 38,40,31,37,36,32,35,31 |
| Free disabled | 68,61,68,57 | 31,35,31,31,32,32,31,31 |

All populations are sampled, with no large aggregate mode concentration.
Balanced counts do not prove equal conditional probabilities or an absence of
learned context-specific choices. Free choices have no invented target or
match count. Total draws are114887 online and115991 disabled; fewer draws can
reflect shorter outputs or changed observations, not greater intelligence.
The audit verifies saved probability/draw, public relay and reward arithmetic.
Omitted eligibility tensors, factual reads and learned-weight effects remain
source/custody-bound; no dominant internal cause is identified from these
aggregate records.

The [protocol](persistent-language-action-v8-protocol.md) offers a plausible
learning problem, but its signals did not establish useful language or action
policies here. Negative task scores constrain this implementation and
experience budget; they prove neither that the broader goal is impossible
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

The frozen stream specified 9,884 source offers, 18,983 nominal public-scan
offers, and 35 answer byte/EOS offers per neural route. Actual source/answer
counts match; actual public counts are reported above. Of each public scan,
13 offers are flags/EOS: all832 scheduled offers are present per neural route.
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

## Interpretation and one bounded next direction

Keep first acquisition, pure paraphrase, exact delayed repeat, new construction,
structural combination, historical regression and first/later/terminal recovery
separate. First encounters receive no answer correction before acting; earlier
fresh observations still become online training exposure. Both current neural
routes begin from V7 ONLINE with a new stochastic policy, so historical changes
are not an isolated measure of forgetting caused by current learning.

Read actual output bytes, EOS/limits, public assertions and private correctness
together. There is no demonstrated source-prediction advantage or correct
neural task here. Future source-reward gains without fresh accuracy would
support that prediction objective, not instruction understanding. Fewer crashes
without correct terminal edits would support avoidance, not successful repair.
Any online/control difference combines source, public-scan, answer and
consequence updates. Saved scalar receipts can check their arithmetic and
causality; omitted eligibility and learned-weight effects cannot establish
which internal state or gate caused a result.

Recommend one bounded design refinement of the same acting controller:
direct observed-byte teaching at actually evaluated literal output cells,
paired with causal per-symbol own-eligibility credit for context/gates using
available conditional match probabilities. The source-derived mechanism and
forward-only accumulation seam below make this concrete. Low raw matches,
zero task gain and no established source-prediction advantage justify examining
credit before assuming a learned language representation merely needs more
cells. They do not prove variance or attenuation caused the failure.

This recommendation changes the output objective explicitly and still must
resolve conditional branch accounting, held baselines/normalization and public
motor-versus-payload blame. It does not supply semantic roles, guarantee
composition, or authorize implementation, another run, gain/seed tuning, replay
or an isolated gate. Preserve the same broader learning/action pressures and
all V8 negative evidence; direct output learning alone is not a complete repair
or instruction-understanding mechanism.

## Possible credit refinement, not selected or implemented

The following is source/math inference, distinguished from the audited V8 results. It concerns
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

### Unselected source feasibility: sparse causal accumulation

The causal sum need not add a full three-million-weight eligibility matrix on
every symbol. For one category let `a_t=r_t-b`, `A_t=sum_{u<=t}a_u`, and e_t
be that tick's own sparse eligibility increment. Alongside existing cumulative
E, accumulate `P += A_(t-1)*e_t` only at the active afferent columns during
`draw`; advance scalar A only in `observe`, after the whole tick and target
arrival. At finish, one dense pass computes
`G=A_n*E_n-P=sum_t a_t*E_<=t`. This is forward bookkeeping of local
eligibility and a declared scalar signal, not backward transport, reverse
trajectory traversal or future target input. Every population's draws in one
tick use the same pre-arrival A. Floating-point regrouping is not bitwise
identity with a dense per-symbol reference and needs an explicit tolerance.

The narrow seam is private ControllerStage state plus draw/observe/finish and
their existing fork/seal validation. Keep plain scan E: subsequent teacher
reward contributes `(A_teacher/t)*E_scan`, and actual public reward contributes
`r*(E_scan+E_free)`. The scan's own causal contribution is `G_scan/s`; teacher's
is `G_teacher/t`. Retaining the current outer `.01/(L*C)` makes this a causal
estimator for the same category-mean objective under the stated assumptions;
dropping either s/t division would instead change length weighting. Baselines
must remain held before all credited choices, and zero-teacher categories must
not divide by zero. This bookkeeping works with whichever per-symbol scalar
signal is later selected; it does not select sampled reward, conditional
expected match or a hybrid objective.

A straightforward two-category P allocation adds 49,247,232 float64 bytes per
stage. If both root and correction fork own those banks, their additional
retained footprint is 98,494,464 bytes before scratch; lazy allocation could
reduce this but is not assumed. Each actual scan/teacher population draw adds
one scalar multiplication and accumulation per active afferent/cell product,
plus counted gathering and copying. Only finish needs whole-bank arithmetic.
No new learned checkpoint tensors are required at idle boundaries; pending
seals, private-fork copies and validation must include P/A. The new finish
rule needs versioned metadata even with unchanged learned tensor shapes.
Weights still publish only after the existing joint validation; disabled
interactions retain their real computed work and RNG progression. This is a
feasibility finding, not an implementation or runtime authorization.
