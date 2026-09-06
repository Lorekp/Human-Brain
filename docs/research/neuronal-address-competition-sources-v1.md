# Neuronal address competition: primary-source review

2026-09-06. Source and algebra only; no models, checkpoints, generated outputs,
actions or fits. The [active learning review](neuronal-splice-online-v1-learning-review.md)
motivates concentration of learned address activity; it does not establish that
committing to the highest address would produce correct complete payloads.

**Recommend one mechanism family for root's consideration: fixed threshold
comparison and inhibitory exclusion, followed by one retained address cursor.**
It has an exact finite circuit contract without a tuned gain or settling horizon.
Keep the existing direct local teacher as an explicitly different, all-address
supervised inference phase in the same learner. This is a proposed combination,
not an implementation selection or a published biological-learning theorem.

## What the primary sources establish

Lazzaro, Ryckebusch, Mahowald and Mead fabricated analog continuous-time WTA
circuits with linear interconnect complexity, including a local-inhibition
variant with multiple winners. That is physical support for competitive
selection, not a claim that a digital simulation obtains exact argmax in one
constant-cost step. [Original NeurIPS paper](https://papers.nips.cc/paper/1988/file/a8f15eda80c50adb0e71943adc8015cf-Paper.pdf).

Maass gives a two-layer threshold construction using `N(N−1)/2 + N` gates,
and a matching quadratic lower bound within that feedforward threshold-gate
model. His soft-WTA definition is rank based; it is not simply softmax with a
temperature parameter. An O(N) analog WTA and an O(N) comparison tournament
with analog value-routing primitives are different circuit models.
[Original paper, section 4](https://proceedings.neurips.cc/paper/1999/file/1c54985e4f95b7819ca0357c0cb9a09f-Paper.pdf).

Rutishauser, Douglas and Slotine analyze recurrent excitation and shared
inhibition using contraction theory. With rectification `[z]+`, a representative
special case is `τ dx_i/dt=−x_i+[I_i+αx_i−β1 h]+`,
`τ dh/dt=−h+[β2 Σ_i x_i]+`. In a single-winner active region the Jacobian is
`[[α−1,−β1],[β2,−1]]/τ`. Negative trace and positive determinant require
`α<2` and `β1β2>α−1`; these regional conditions alone are not their full
nonlinear stability theorem. Their stronger parameter conditions distinguish
stable winning states from unstable competing states. This does not supply a
uniform finite settling time for arbitrarily close inputs.
[Author-hosted full paper, sections 2.4–2.5](https://web.mit.edu/nsl/www/preprints/competition_2010.pdf).

Nessler, Pfeiffer, Buesing and Maass show local STDP/intrinsic plasticity in a
specific stochastic WTA model implementing spike-based online EM. Their rates
`r_s=exp(u_s−I)` give spike identity probability
`r_s/Σr=exp(u_s)/Σexp(u)`: common inhibition controls total rate, but cancels
from relative probabilities. Increasing that inhibition alone therefore does
not sharpen the address distribution. A posterior sample is also not MAP
selection. Their learning guarantees depend on the generative model, plasticity
and sampling assumptions; they do not transfer to our shared discriminative B,
hard cursor or teacher sequence merely because each is called WTA.
[Primary PLOS article, equations 2–5 and discussion](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003037).

## Concrete finite competition contract

The following tie convention and application are our adaptation of threshold
competition, not a quoted paper algorithm. Preserve the actual finite float32
address scores `a_s=Σ_ij B_ij X_sij`, where X is already materialized forward
from raw query/key activity before the sole learned address synapses.

For each pair `i<j`, fixed comparison neurons compute
`c_ij=1[a_i−a_j >= 0]`. For each address, fixed inhibitory inputs count losses:

```
L_i = sum_{j<i} c_ji + sum_{j>i} (1-c_ij)
w_i = 1[-L_i >= 0]
    = 1[i + sum_{j>i} c_ij - sum_{j<i} c_ji - (N-1) >= 0]
```

The lexicographic ordering (higher score, then lower address index) is total,
so exactly one address has zero losses. No teacher, source-byte equality, span
parser or post-loop host argmax selects it. Compare widened original scores to
avoid float32 difference overflow; do not perturb scores with an epsilon tie
bias that could reverse a genuine small margin. N=1 yields its sole winner.

For held inputs this acyclic circuit stabilizes after two logical evaluation
stages. It has no integration-step stability parameter and no convergence loop.
Exact ties choose the lowest physical address by declared wiring. For a unique
top gap Δ, perturbing every score by less than Δ/2 preserves its identity;
there is no noise-robustness guarantee as Δ approaches zero. This is an ideal
threshold-neuron abstraction, not a measured analog/spiking latency claim.

At N=128 the literal circuit has 8,128 comparison and 128 winner neurons;
comparison input connections and comparison-to-winner connections each number
16,256, excluding biases, score production and cursor relays. Two stages do not
mean O(1) arithmetic: every comparison and loss contribution must be counted.
Vectorizing this declared connectivity is legitimate. Replacing it with a host
sort or a tournament requires a separately declared circuit and accounting.

The winner activity initializes one cursor, which advances only through existing
forward adjacency, with document EOS absorbing. The same codec generates bytes
and its actual EOS; literal document termination does not replace learned
punctuation-to-output-EOS behavior. Fixed read gating may use a sparse gather,
provided its index comes from the neuronal winner. No later readdressing is
implied by this proposal.

## Local teacher ownership and objective boundary

A hard-masked corrective posterior cannot recover excluded addresses: starting
with one-hot w makes its posterior remain w, giving zero address credit. An
unmentioned switch back to soft branches is therefore not acceptable.

A coherent minimal contract explicitly retains the same B, codec, tape and raw
query in a teacher-driven mode after actual execution. The score population
forms `p=softmax(a)`; all candidate cursors receive the incoming raw teacher
bytes through the shared codec. Each address accumulates only its own forward
evidence `ell_s(t)=ell_s(t−1)+log pi_s,t(y_t)`. Current normalization produces
`b_s(t)=softmax_s(a_s+ell_s(t))`. This is forward latent inference, without a
host-supplied correct address. Retain the actual X bank for the current lesson.

Each physical shared B synapse then accumulates its own
`X_sij * (p_s−b_s(T))` while visiting score-neuron states, with held weights and
one finite publication. No comparator derivative, reverse key projection,
reversed history, externally solved per-synapse target or autograd is involved.
Current-prefix codec errors remain local responsibility-gated byte deltas;
future posterior values must not be sent backward to earlier codec events.

Algebraically the B delta is the exact conditional marginal-NLL gradient for
held codec likelihoods. It is **not** a gradient of the committed free policy
or task success. With `||X_s||²<=32`, its Hessian is the difference of two feature
covariances, operator norm at most32. In real arithmetic eta=.01 gives the
held-codec descent bound `ΔL<=−.0084 ||gradient||²`. This derivation does not
guarantee improvement after simultaneous codec learning, retained-task stability,
or successful hard recall. Calling the update contrastive does not eliminate
these objective differences. The same-instance teaching interpretation is
coherent only if those different activities and full teacher costs are explicit.

## Tradeoffs and decision boundary

Commitment removes off-target emission pooling but also discards useful duplicate
evidence and prefix-based disambiguation. Duplicated identical content may make
the tie harmless; duplicated prefixes with different continuations need not.
The circuit is MAP initial location plus greedy codec, not joint-MAP answer
generation. It cannot repair a wrong top address, guarantee escaping/composition,
or solve the remaining operation error. The score objective still optimizes a
marginal model, so its pressure for concentration may remain weak.

These limits favor the finite threshold family over adding a recurrent settling
process whose step count, tie behavior and plasticity would require additional
design decisions. They do not select WTA for deployment. Root must approve the
actual free/teacher contract and costs before implementation. Any successor
needs fresh withheld cases, matched controls, old-reading retention, ordinary
mechanics QA and measured admission; v1 endpoints stay exposed development data.
