# Peer review: neuronal learning without reverse error transport

Status: bounded independent design review, 2026-09-06. No implementation,
model import, checkpoint read, simulation, fitting or endpoint evaluation.
Root owns selection. This review follows the user's no-backpropagation and
same-learning-and-executing-system requirements and the
[roadmap](roadmap-reading-to-coding.md). Historical source is inspected at
`82db27e847f00fb065f03d5464fc519116e6a6db`; its completed evidence stays frozen.

## Decision and boundary

Root selected the **forward joint-feature circuit with one plastic score
layer** and a batched float32 presynaptic bank, subject to the physical schedule
below, over starting a stochastic reward-learning experiment. This is one
integrated return, not authorization for two trials. Root has withdrawn both the old teacher's
reverse key-error transport and the frozen single-start inference shortcut.

The old implementation formed a learned query `Q = B.T @ Z`, then fixed-key
scores, and returned `K.T @ (p - b_T)` before updating B. That return is an actual
manual adjoint, regardless of `no_grad`. Renaming it associative recall or
reassociating its multiplication only during teaching does not remove it.

The replacement must instead construct, during every actual query,

```text
X_s[i,j] = sum_d Z[i,d] K_s[j,d]     fixed forward dendritic activity
a_s      = sum_i,j B[i,j] X_s[i,j]  one plastic score layer
p        = softmax(a)
```

There is no learned Q population in this graph. Fixed feature compartments are
upstream of B and receive no teaching error. Teaching may supply each score
application with its scalar leaf signal `e_s = p_s - b_T,s`. Each B synapse then
accumulates `e_s * X_s[i,j]`, using activity that actually reached that synapse.
No key-transpose error multiplication, hidden-layer signal, weight transpose
return, or reverse sequence traversal is required.

This is a genuine change of implemented topology if prediction and teaching
both take that route. It remains algebraically equivalent in real arithmetic to
the old score function and conditional B update. Equivalence is neither a
disqualification of a single-layer local delta nor sufficient evidence that a
new simulator implements it. A compiler/fused kernel that restores the old
learned-Q forward path would invalidate the declared topology even if outputs
match. Floating operation order changes mean bitwise trajectory parity must not
be promised from algebra alone.

## Physical ownership and schedule

Use one shared 1,024-input score neuron with 1,024 physical plastic synapses,
time-multiplexed over occupied slots and vectorized as a batch. Materialize the
complete float32 `X[N,32,32]` presynaptic bank before applying B. Do not instantiate per-slot plastic B
copies and silently synchronize them. The fixed feature population has 1,024
signed compartments; each combines 32 pairs of generic query and key activities.
Signed multiplicative compartments are an explicit custom-neuron prior, not a
claim about ordinary spiking cells or learned dendritic multiplication. The
generic connectivity uses every channel and lag pair; it contains no answer
positions, source roles or teacher-selected feature indices.

1. The same persistent learner ingests raw documents and receives the query.
   Fixed normalized Z and literal K drive the complete batched X bank before
   scoring. Each free, teacher or distractor query constructs its own bank once.
2. Present each X to the same B synapses with B held. Store its score in the
   slot's competition population and retain its forward X activity for the
   current call. The soft competition gives p. Free prediction discards its bank
   when that call finishes; it is not a cross-call feature cache.
3. The ordinary codec and cursor system produces the learner's actual action.
   Execute that action and record its outcome before applying feedback.
4. When correction is supplied, use that same system's declared teacher-clamped
   prefix path. Its forward likelihood evidence updates b sequentially. Hold B
   and codec weights throughout those offers. The teacher call constructs its
   own bank before B and retains it through the current prefix. A retained forward posterior is
   not a backward traversal of earlier neural states.
5. Pair the retained bank's X activities at the same score synapses with each slot's
   scalar difference of initial and teacher-conditioned competition activity.
   Each physical synapse owns its accumulated local product. Publish one finite
   proposal after the sequence. Count these local applications separately from
   forward scoring. Batched local accumulation uses this actual presynaptic bank;
   it does not reconstruct features or replay raw documents. Discard the bank
   after the call. There is no second teacher-only feature-construction pass.

The codec retains its own current-prefix responsibility-gated output delta.
The scalar b activity depends on held codec likelihoods, but no codec derivative
or codec-weight error is transported into B. This is a declared teaching circuit
using inferred responsibilities, not an observed label for the true source
address. The host may supply the teacher bytes, not a desired B matrix or chosen
source slot. A shared tensor sum is acceptable vectorization only when it
represents these individually owned synaptic products.

The integration must expose one continuing instance through action, feedback,
update and later action. Teacher clamping may differ from free emission, as the
roadmap allows, but both must use this X-before-B circuit and the same learned
synapses. External motor matching remains a supplied action schema; executing a
learner's byte choice is not evidence of a fully learned motor policy. A scalar
execution outcome that merely gets logged is not itself reward-driven learning.

## Conditional mathematics and numerical limits

The following is an independent derivation, not a theorem claimed by the papers
below. Let `x_s = vec(X_s)`, `w = vec(B)` and `l_s > 0` be the held codec's
likelihood of the whole corrected sequence on branch s. Its loss is

```text
L(w) = logsumexp_s(w · x_s) - logsumexp_s(w · x_s + log l_s)
g    = sum_s (p_s - b_T,s) x_s
H_L  = Cov_p(x) - Cov_bT(x).
```

The declared normalizations imply `||Z||_F <= sqrt(32)` and
`||K_s||_F <= 1`, so `||X_s||_F <= sqrt(32)`. For every unit vector v,
`v.T Cov_p(x) v <= E_p[(v · x)^2] <= 32`. Consequently
`H_L <= 32 I` everywhere for held l and x. The ordinary descent inequality gives

```text
L(w - eta g) <= L(w) - eta (1 - 16 eta) ||g||^2.
eta = .01 gives a decrease allowance of .0084 ||g||^2.
```

This does not assert convexity: the second covariance is subtracted. It gives
no positive lower bound on useful gradients, no global convergence, no successful
argmax/action guarantee and no joint decrease after the codec also changes.
Feature dependencies can give null directions or competing records. The joint
features keep exactly the old real-valued representation, so they do not by
themselves repair the two marginal-output interference failures.

Float32 normalization, feature construction and proposal casting perturb the
real-arithmetic argument. Finite checks and independent scalar arithmetic QA are
still required; a rounding-robust descent certificate has not been established.
Fix construction/accumulation precision prospectively, sum all slot contributions
once, and do not quietly introduce averaging by N or a different learning rate.

## Complete resource implications

At N occupied slots, fixed feature construction takes `N * 32,768` products
and `N * 31,744` channel reductions; score evaluation takes `N * 1,024`
plastic-synapse products and `N * 1,023` additions. Teaching adds `N * 1,024`
local products and `(N - 1) * 1,024` cross-slot accumulation additions. Feature
work applies to free, teacher and distractor queries, not just teaching.

At capacity 128 this is 4,194,304 fixed products and 4,063,232 fixed additions
per query, 131,072 score products, and 131,072 local teaching products.
The selected float32 bank costs at most 524,288 bytes. A whole-bank float64
learning cast adds 1,048,576 bytes while the float32 bank remains alive. The
selected explicit `X64 * e[:,None,None]` local-product bank adds another
1,048,576 bytes; sum it over slots, without assuming a fused reduction. The
float64 slot-error population adds at most 1,024 bytes, the local float64 B
accumulator 8,192 bytes, and its float32 cast and proposal 4,096 bytes each.
These named arrays sum to 2,638,848 bytes at capacity if simultaneously live;
they are not a complete peak estimate. Normalized Z alone adds 4,096 bytes,
and query lags, codec state, validation and other temporaries remain counted.
The batched fixed construction should not materialize an unnecessary
`N*32*32*32` product array. Temporary library workspace remains measured.

The number of learned B synapses remains 1,024, so unchanged B/codec/code shapes
alone retain the historical 567,568-byte model payload. This is not a complete
new model-byte promise: explicit fixed connectivity metadata, added state and
snapshots must be accounted. There are 32,768 logical pair-product sites per
slot application if parallelized, or the corresponding declared serialized
compartment applications. Compact index rules do not erase that work.

Old Q, reverse-key casts and projection-error transients are retired rather
than retained as a compatibility route. The tape and all codec branch activity
remain. Actual action execution on every learning experience, receipt I/O,
validation, cloning, finite checks, final evaluation and independent audit are
part of a fresh complete cost. Historical 101.385-second shared run timing cannot
admit the larger forward circuit or new executed-learning schedule. No timing
or allocator guarantee follows from these scalar counts.

## Bounded stochastic alternative and primary sources

Williams defines local characteristic eligibility from a stochastic neuron's
own response distribution; its reinforcement update combines that eligibility
with a reward difference. A logistic Bernoulli unit has eligibility
`(sample - probability) * presynaptic activity`. His paper also explicitly
develops hybrids using a backward pass through deterministic units; those
sections do not meet this project's rule. The local stochastic construction,
not the blanket label REINFORCE, is the relevant alternative. It offers expected
reward direction under its assumptions, not deterministic episode improvement.
[Williams 1992, sections 3–7 and 8](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf).

Fiete and Seung specify perturbed conductances, synapse-local eligibility from
coincident presynaptic activity and perturbation, and a later scalar reward
difference. Their physical proposal includes separate perturbing synapses; its
small-perturbation approximation, mean subtraction and baseline dependence are
substantive conditions. The paper does not require a backward network traversal
for that rule. It also explains why a reward baseline can greatly affect
variance. This supports feasibility of a declared perturbation-and-reward
circuit, not an unmeasured sample-efficiency claim for this memory task.
[Fiete and Seung 2006, equations 4–8 and biological interpretation](https://fietelab.mit.edu/wp-content/uploads/2018/12/spike_learning_theory-1.pdf).

For such a replacement, the eligibility must come from the stochastic neurons
that actually caused the executed action. Sampling a different unused branch
after a greedy action would not supply its policy eligibility. A scalar reward
must score that same action; eligibility, RNG state, reward baseline and their
lifetimes must be owned by the continuing learner. Every relevant trained stage
needs its own local stochastic mechanism; a policy-gradient library that silently
backpropagates into deterministic hidden units is not a shortcut. Exploration,
reward sparsity and estimator variance add uncalibrated acquisition risk here.
No old eta or exposure budget has a demonstrated stability guarantee for this
alternative. These are design inferences, not new experimental results.

Both papers were verified from primary full texts online on 2026-09-06. The
author's Williams index returned 403 on direct open; the publisher PDF was
available. Mem0 was queried for development context and returned unrelated old
facts; none supports these scientific claims. The exact historical code was read
directly from Git, without discovery or execution of a current model.

## Recommendation to the shared design owner

Choose the explicit joint-feature forward circuit as the smallest credible
return under the strict no-reverse-transport decision, provided its actual
simulation, ownership and same-instance action/teaching route are implemented
as described. Do not call it an algebra-only fix, biological proof or capability
improvement. Preserve the stochastic alternative as reviewed but unselected.
Close this review before one implementation and fresh complete admission; no
parallel parameter search, frozen WTA endpoint shortcut or old-teacher training
is authorized by this note.
