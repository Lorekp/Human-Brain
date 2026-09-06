# A local neuronal learning and action return

Status: bounded source/math review, 2026-09-06; prospective design, not execution
admission. No models, checkpoints, endpoint forwards or acquisition were run.
Root owns selection, protocol and implementation authorization. This note is the
only tracked file edited by this review.

Root selection: implement this batched forward-feature circuit and the
[four-arm executed-learning recipe](../../experiments/r4_integrated/local_joint_address_online_v1.md).
The mathematical and resource peer review passes. Implementation, authored QA
and fresh measured admission must precede acquisition. No frozen single-start
comparison or old-rule training is selected.

## Recommendation

Select one genuine forward-topology change: compute fixed query/key interaction
features before the learned address synapses, then apply a local output-neuron
delta rule. Keep the existing marginal free recall initially. Return from fresh
initial weights to an integrated sequence of actual action and feedback in the
same learner. Do not use the 82db teacher for further training and do not perform
the proposed frozen single-start shortcut.

The user's prohibition includes reverse-Jacobian error transport through fixed
maps. The old `K.T @ (p-b_T)` operation therefore fails the selected boundary.
The replacement must really construct its forward feature activity during both
acting and teaching; simply expanding the old backward product into a different
matrix expression would not satisfy this recommendation.

This review compares two candidates only: the forward-feature local rule below,
and stochastic neuronal eligibility with scalar action reward. Root replaced the
initial recalled-key-target candidate with the forward-feature design during this
review. The recalled-key target is not another selected option: its unit-norm
target would introduce an unmotivated contraction of address activity and would
need another learning objective and gain justification.

## Existing mechanism and the actual change

The examined source is
`82db27e847f00fb065f03d5464fc519116e6a6db:src/humanbrain/local_memory_learner.py`.
Its `_query` computes `Q=B.T@Z`, then scores `K vec(Q)`; `teach` lines 410-420
returns the output error through K.T and computes the local B products.
There is no BPTT, but that particular backward transport is explicit. See the
source-only [backpropagation boundary review](local-learning-backprop-boundary-review.md) for the
complete distinction from local plasticity and from autograd use.

Let Z be the existing fixed normalized 32-lag by 32-code-channel query activity;
let K_s be one stored normalized key reshaped to [32,32]. Keep the exact raw-byte
codes, key construction, query normalization and shared 32 by 32 B initialization.
Those initial B values retain the inherited temporal-sharing prior. They are not
new fan-in-Xavier values for a 1,024-input scalar neuron; make that choice explicit
rather than relabeling initialization to match the new topology.
For each occupied slot s, implement the actual forward circuit:

```text
X_s[i,j] = sum_d Z[i,d] * K_s[j,d]        # fixed dendritic interaction activity
score_s  = sum_i,j B[i,j] * X_s[i,j]      # only learned address output layer
p        = softmax(score)
```

Every slot uses the same rule, without a role parser, a target-dependent mask or
special identifier positions. X_s is produced before B and without any teacher
value. A valid implementation is one fixed 1,024-component dendritic population
with 32 channel products per component, followed by one 1,024-input score neuron
with shared B synapses, time-multiplexed over all occupied slots. Record this
nonlinear multiplication and channel-summing prior explicitly; do not describe
it as an ordinary linear neuron or hide its work in preprocessing.

The key activities/associations and query activities are the two forward inputs
to those dendritic compartments. The learned B synapse sees only its own X_s
preactivity and the current score neuron's teaching signal. No hidden projected
Q population is needed. Remove the Q and K.T-error kernels from the main path,
including unused receipt and accounting fields; preserve them only in immutable
historical source if needed to explain spent evidence.

In real arithmetic, these scores equal the old scores. That equality is useful
but is not the reason the rule passes the architectural test. The reason is the
new actual graph: fixed nonlinear inputs precede one learned output layer. An
optimizer that factors inference back into B.T@Z and KQ would restore the old
hidden-layer topology and is outside this contract.

## Direct local teaching in the same circuit

The same learner first produces its marginal free answer and executes that answer
through the existing action bridge. The external environment supplies the actual
action outcome and, for this supervised TRAIN return, the corrected byte sequence.
The existing motor's lossless execution check is not task correctness: the parent
must separately grade the actual executed result against the declared task.
Corrected-target teaching after execution is supervised local learning, not proof
that the scalar outcome causes plasticity. Keep the same actual model instance
and tape state throughout prediction, execution, external grading and feedback;
there is no separate offline learner. The current `teach(query, teacher, state,
learning_rate, learning_enabled=...)` interface may remain, with this custody
enforced by the integrated lesson. Do not present isolated teacher-only fitting
as that action/feedback cycle.
No answer or target selects a memory slot in host code. No external trainer
computes per-synapse updates.

The learner's existing codec and forward branch adjacency process that corrected
prefix. With all weights held, the address population starts at p and updates its
own posterior activity causally:

```text
b_s,0 = p_s
b_s,t proportional to b_s,t-1 * pi_s,t(teacher_t)
e_s = p_s - b_s,T
G[i,j] = sum_s X_s[i,j] * e_s
B_new = B - .01 * cast_float32(G_float64)
```

Here e_s is the local score/output neuron's actual versus teacher-driven activity
difference. It is not a signal returned through K, through B, or through past
feature operations. The shared physical B synapse accumulates its own local
products as slots are presented; weights remain held until the complete group
has finite proposals. The teacher-driven posterior comes from the learner's
actual codec probabilities and ordinary forward normalization, not host substring
matching or supplied correct-address labels.

Keep the codec's current-prefix-gated local delta rule and its own sensory
presynaptic input. It sends no weight-transposed error into another layer. As
before, the codec rule is a filtering approximation, not the exact whole-sequence
codec gradient. The address rule is an exact conditional output-layer gradient;
being a gradient does not make a single output-synapse delta operation a reverse
pass through hidden layers. No derivatives or errors are propagated through the
fixed dendritic feature generator. This is the concrete boundary, not a change
of terminology for the old K.T operation.

Keep every parameter outside autograd (`requires_grad=False`), and use the
declared mode and `learning_enabled` to gate local publication explicitly.
Do not repurpose an autograd flag as the synapse's plasticity switch. Parameter
and checkpoint checks must reject retained gradients/history. This guard is
additional to the actual no-backward-transport topology, not its justification.

Learning and executing use the same B, dendritic feature circuit, address
competition, codec, tape and forward cursors. Corrected-prefix teaching and free
generation are different input/activity phases of that learner: the former uses
offered tokens and the latter its actual emitted tokens. Both initially retain
the same all-start marginal recall topology. This proposal adds no single-winner
execution mode, no separately fitted executor and no distillation. The supervised
correction remains explicit assistance; this small TRAIN return is not autonomous
knowledge acquisition or human-equivalent cognition.

## Objective, conditioning and numerical boundary

For a held codec/tape, write
`a_s = sum_t log pi_s,t(teacher_t)`. The conditional address objective is

```text
L(B) = logsumexp_s(<B,X_s>) - logsumexp_s(<B,X_s> + a_s).
```

Its gradient at the sole learned output layer is G above. No global optimizer is
used to calculate or apply it. In real arithmetic, normalized keys satisfy
`||K_s||F <= 1` and the unchanged query has `||Z||F² <= 32`. Thus
`||X_s||F² <= 32`. The Hessian is the difference of the p and b_T covariance
matrices of vec(X_s); each covariance is positive semidefinite and bounded above
by 32 I. Consequently the Hessian operator norm is at most 32. Holding X and a
fixed, the ordinary descent lemma gives

```text
L(B - eta G) <= L(B) - eta*(1 - 16*eta)*||G||F².
eta=.01 gives coefficient .0084.
```

This conditional bound motivates retaining eta .01 without a rate, seed or gain
sweep. It does not guarantee improvement after the simultaneous approximate
codec update, improvement on other queries, exact-action improvement, convergence
or absence of interference. Degenerate features can carry no useful distinction.
It also does not guarantee actual float32 monotonicity on every lesson.

Use actual float32 forward X and scores, with a declared deterministic slot order;
use those float32 preactivities cast to float64 for the local products, one held
float64 accumulator, then one float32 cast for B publication. Reject nonfinite
features, signals, accumulators and all parameter proposals before either group
publishes. The multiplication/reduction order differs from the old factorization,
so neither bitwise trajectory parity nor the old endpoint scores are predicted.
No feature normalization, temperature or gain compensation is selected.

## Feasible activity and work ownership

Root selects a batched actual forward activity bank `X[N,32,32]`, in float32,
materialized before applying B in every free, query-only and teacher traversal.
This is the real presynaptic feature population, not a debug reconstruction of
the old projected-Q computation. Vectorized simulation applies one physical
shared B circuit across its stored slot activities. It does not introduce N
independently learned copies of B.

Hold the actual X bank through the current teacher prefix. When the score-neuron
errors arrive, form the local products from that bank and accumulate once with B
held. Discard the bank after the call; do not attach it to persistent model/tape
state or serialize it into the scientific journal. Free receipts need scores and
the existing policy evidence, not the full feature bank. There is no second
feature replay, per-slot Python kernel loop or backward reconstruction. This
bounded current presynaptic activity retention is not a history of derivatives.

For N slots, one feature traversal uses N*32,768 fixed products and N*31,744
channel additions; scoring additionally uses N*1,024 learned-synapse products
and N*1,023 sums. Feedback adds N*1,024 local credit products and
(N-1)*1,024 slot-reduction additions, using the already computed X. Query-only
delay reads incur scoring features too. Frozen controls still compute ordinary
forward/teacher signals when offered; publication gates alone change updates.

At N=128 a single feature traversal uses 4,194,304 fixed products and 4,063,232
channel additions. This is substantially more work than the old factorized query
projection and needs a new measured cost; old timings cannot admit it.
The full float32 X bank is at most 524,288 bytes. Retain the real feature values,
then cast the bank once to float64 for local learning (at most 1,048,576 bytes).
The selected explicit vectorized `X64 * e[:,None,None]` product bank adds at most
another 1,048,576 bytes. Reduction produces an 8,192-byte float64 B accumulator;
the finite float32 update cast and parameter proposal are 4,096 bytes each.
The normalized Z query activity is 4,096 bytes and the N-element float64 score
error is at most 1,024 bytes. These coexist with the actual query/codec state and
other validation/proposal arrays. Keep each named payload explicit; do not claim
the 0.5-MiB forward bank is the complete learning scratch allocation.

The bank's extended lifetime is one current query/teacher call, including the
complete corrected prefix before local publication. Parent action and feedback
receipts must not inadvertently retain an earlier full X bank alongside the next
teacher bank. Root selected this small bounded memory cost to avoid a second
feature traversal and many small Python/kernel calls. Changing to a fused local
reduction later would need an honest revised allocation contract; it must never
factor the forward graph back into Q or an adjoint.

The B and codec tensor sizes, tape payload and initialization draws need not grow.
Retire old projected-Q and K-cast/error-return buffers and counters honestly.
This source review does not supply an allocator bound: the source-frozen runner,
streaming receipts and whole process-tree watchdog must count actual peaks and
the extra forward work.

## Compared alternative: stochastic eligibility and scalar reward

A genuinely backprop-free alternative makes the actual address-output neurons
stochastic during action. For example, each channel application can emit
`q_j,d = sum_i B_i,j Z_i,d + epsilon_j,d`, with independent Gaussian noise of
declared variance sigma². Each physical B synapse accumulates its own eligibility
`E_i,j = sum_d Z_i,d epsilon_j,d / sigma²`. After the actual action returns scalar
reward r, the same synapse changes by `eta*(r-baseline)*E_i,j`. The baseline must
depend only on earlier trials or current pre-action information, not the current
noise. There is no K.T error, no backward weight or per-synapse host target.

For fixed downstream parameters and bounded reward, this likelihood-ratio rule
estimates the gradient of expected reward under the stochastic actor, even when
the action path includes discrete choices. It does not give the old deterministic
NLL descent guarantee. The method requires an actual sampled action and its
retained local eligibility, an auditable RNG state, reward timing and new work.
A counterfactual unperturbed run or multiple perturbation trials cannot be hidden
inside one experience. Learning with noisy actions but silently switching to a
different deterministic execution policy would need its own declared comparison.

Fiete and Seung's primary paper describes local eligibility from a neuron's
perturbation and its synapse's presynaptic activity, multiplied by scalar
reinforcement; it separately identifies the Gaussian likelihood-ratio connection
and the role of a noise-independent baseline. This supports the mechanism class,
not a claim that our workload will learn quickly or that their spiking model is
implemented here. [Fiete and Seung, 2006, equations 4-8 and discussion of
REINFORCE](https://fietelab.mit.edu/wp-content/uploads/2018/12/spike_learning_theory-1.pdf).

Its scale and variance are material. With 32 output-channel applications and
fixed Z, `E||E||F² = (32/sigma²)*||Z||F² <= 1024/sigma²`. Smaller noise increases
eligibility variance; larger noise changes actual action quality. Sparse exact
success reward may supply no early learning, while a graded reward changes the
task objective and must be declared. A local online baseline can reduce variance
but cannot establish that 1,024 experiences suffice. The new noise/reward/state
choices are avoidable here because the forward-feature output-layer rule has a
clear direct teacher signal and already justified step scale. Do not select the
stochastic candidate for this return or add it as another acquisition arm.

## One bounded integration return

After root selection, ordinary authored QA must establish the actual new graph:
forward features precede B in both acting and feedback; no old Q/K.T-error kernel
is called; a scalar local-product oracle agrees; the actual forward X bank is
held unchanged through feedback and discarded after the call; control publication
and finite atomicity hold. A small authored
real-arithmetic equality check against the old factorization is a shape/sign
oracle, not authorization to train or restore the prohibited old learner.

Run one newly frozen integrated protocol from fresh seed-identical initial
parameters, with learned, address-frozen, slow-frozen and writes-disabled matched
controls. Preserve the existing document/query schedule, offered teacher bytes,
rate, marginal policy and exposed fit/novel/delay/interference workload. Add the
missing explicit acquisition action custody: each first answer is actually
executed before its outcome/corrected prefix reaches this same learner. Every
action, teacher offer, forward feature traversal and read-only delay is counted.
No trained 82db checkpoint or additional old-rule training arm is part of this
return; historical scores remain descriptive spent evidence.

Keep 1,024 fixed experiences per each of four arms and the twenty final checks
per arm for the first return,
subject to a freshly measured cost and independent source/state/action audit.
This is 4,096 actual acquisition actions before their local feedback, plus 80
final generated-byte actions and two declared motor demonstrations. Root's protocol
must distinguish a scoped architecture-constraint repair that preserves useful
8/8 fit, 4/4 novel, 4/4 delay and at least 2/4 interference behavior with matched
control separation, from the still-required absolute 20/20 completion gate.
The prior 18/20 remains useful historical evidence and an incomplete absolute
result. An 18/20 compliant return must not be called completion or a retroactive
lowering of that old gate. No accuracy increase is predicted from changing this
topology. Failure is reported without an automatic epoch/gain/noise
extension. Preserve the 900-second aggregate run-plus-audit and 2,048-MiB ceiling
unless root explicitly selects a new budget before cost. More expensive actual
actions and the changed forward feature computation make fresh admission mandatory.

Successful small-workload behavior would establish a local output-layer learning
return, not continual semantic understanding. The next substantive return remains
broader content, wording, answer lengths and actions in the same learning system.
The single-start interference hypothesis stays an unexecuted historical proposal;
it is not combined with this required learning-mechanism correction.

## Evidence and scope

This recommendation is derived from the inspected source and the explicit
equations above. Hubble independently checked the changed graph, local ownership,
feature costs and conditional Hessian bound; Carson owns the same-instance
action/feedback integration contract. The research skill was used for bounded
primary-source reading. A task-specific Mem0 search returned older unrelated
development notes; none supplies scientific evidence or enters the model.
No behavioral result is inferred from literature or from this review.
