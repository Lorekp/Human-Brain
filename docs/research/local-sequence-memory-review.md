# After the event regression: learn retrieval over an explicit neuronal sequence

2026-09-06. Selected structural design, not an admitted experiment.
No new model construction, checkpoint inference, teaching, parameter search or
protected observation. Roadmap and active plan reread; research skill applied.
Root selected this design after independent source/algebra review. This is one integrated replacement, not parallel
gate, encoder and output experiments.

## What the result establishes

The audited event system emits `cd` with EOS for all twenty cases in every
updating candidate, including writes-disabled. Tagged/aggregate/writer-frozen
and writes-disabled score 1/8 fit and zero new-value, delay and interference.
The immediate byte comparator reproduces its previous 1,024 lesson trajectory
and twenty outputs exactly, scoring 5/8, 2/4, 2/4, 0/4. Tagged first-attempt
success totals 104/1,024 versus byte 159; first/last 128-lesson half-SSE is
0.277614/0.254183 versus 0.274995/0.230540. Writer-frozen closely matches tagged.
These are exposed development results, not evidence of convergence or a
measurement of encoder saturation. Encoding amplitude, input clock, useful
representation and subsequent credit all changed together.

Evidence: `.pytest-tmp/local-memory-event-v1-description.json`, SHA256
`4f0274a5879298a0b85ca2188fe32085a322e7537300d2cd6bd7e61b02064601`;
strict audit SHA256
`ff392f2a3d9253175e529c53dce4b8451d26e2381d2f684d667c3641b01e1933`.
Decision `084bf3b0815338f672326035442482d42eca062c92ef2eb320e5bf1c3e3c8644`
closes the comparison; total monitored run/audit was 314.1363762 seconds.

The source-level downstream constraint is stronger than any diagnosis from the
loss curve: PhaseContextLMS supplies three separate affine classifiers of one
static context. Previous-byte identity is unused and memory is not reread during
generation. Its earlier supplied-factor certificate does not establish that an
arbitrary recurrent token code is linearly compositional. A replacement must
provide readable ordered content, not merely another unequal memory vector.
See the supporting `.pytest-tmp/downstream-content-interface-review.md` and
`.pytest-tmp/local-association-feasibility.md`.

## Three alternatives, one choice

**Choose explicit episodic sequence association with learned retrieval.** Separate
where a sensory event is stored from how the query locates it; retain its temporal
successor explicitly. This replaces both the recency-dependent writer and its
expensive delayed writer sensitivities. It gives current output error a direct
path to the actual retrieval neurons. It does not solve useful query addressing
in advance.

There is a relevant primary precedent: Melchior et al. associate incoming
patterns with an intrinsic sequence using Hebbian descent, separating sequence
dynamics from fast content associations. Their complete system includes offline
pretraining, and correlated inputs require pattern separation. Thus the paper
supports the structural separation, not a claim that its entire pipeline meets
our online-only acquisition contract or solves text queries. The proposed fixed
cursor below is an explicitly authored neural prior rather than importing their
pretrained network. [Paper and original code](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0304076),
[published correction](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0313130).

**Defer selective admission, including novelty gating.** PBWM provides an
established learned maintain/update distinction, but its actor-critic and phased
Leabra learning are a substantially different learning contract. Its discrete
working-memory tasks do not provide an answer-free gate for our byte stream.
Adding that gate now would leave both gate credit and readable content unresolved.
[O'Reilly and Frank, 2006](https://ccnlab.org/papers/OReillyFrank06.pdf).
Root's next-input-prediction idea has an honest local signal: each arriving byte
can teach a predictor. Nevertheless prediction error is surprise, not relevance;
unseen distractor bytes may be more surprising than the requested fact. A novel
gate would need its own acquired-policy evidence, counted sensory teachers and
retention tests. It should not be quietly substituted for learned role selection.

**Defer a larger learned contextual encoder.** It may eventually supply reusable
abstractions and compression, but the present evidence gives no identifiable
local target that guarantees task-useful, order-readable latent content. Adding
another reservoir or local feature objective while retaining the static decoder
repeats the unresolved interface. Eligibility and useful neuronal learning
signals are distinct requirements; their product does not make an arbitrary
representation objective task-correct. [Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y).

## Smallest proposed end-to-end contract

Use raw bytes again. Store all bytes and one separate EOS per document in a
128-slot neuronal episode, with an explicit capacity error rather than silent
wrap/truncation. A one-hot neuronal write cursor advances through generic fixed
adjacency. At its active slot, local presynaptic-cursor/postsynaptic-sensory
association stores a 257-neuron one-hot content vector. It also stores a generic
preceding-context key: the last 32 shared 32-dimensional byte codes in a fixed
lag population, zero padded at document start and normalized by the same
declared finite normalization circuit for every slot. Store the key *before*
consuming the current byte. All 32 lags and coordinates remain accessible to
the learned address map; there is no selected identifier lag or value boundary.
The lag population resets at each document EOS. These are general sequence
priors and a finite cue window, not learned segmentation or unbounded context.

The association rule at a newly activated slot is a local unit-rate overwrite
`w <- w + cursor * (sensory - w)` (and similarly for key features). It copies
observed input activity, never an answer label. Its unit rate follows from the
new-slot storage contract; slow learned parameters keep eta=0.01. Document
ingestion never receives a question, answer, target position or selected span.
During unrelated writes append to unused slots; all branches and old content
remain within the same fixed capacity and declared state.

Keep the existing raw-byte 32-neuron query cell and its declared local self-path
eligibility initially. Let `u` be its activity, `k_j` the stored key and `A` one
shared learned 32-by-1024 map. A single population first computes q=A^T*u; each occupied address neuron computes
`s_j = q^T k_j`; softmax competition supplies `p_j`. There are no separately
learned absolute-slot classifier weights. Initialize A by the existing private
Xavier convention at seed+4; do not fit initial features to the roster.

Every possible start-address candidate has its own forward continuation activity.
At output step t, branch j exposes sensory vector `x_j,t` from t forward neuronal
adjacency advances. At a document EOS, adjacency is absorbing; a branch cannot
cross into the next document. The weighted read is `r_t = sum_j p_j*x_j,t`.
There is no winning-span extraction in host code. A sparse gather is permissible
only as an exact implementation of this declared fixed connectivity; count all
allocated branches and all branches actually evaluated.

A shared 257-to-257 conditional codec is evaluated at every branch's current
sensory byte: `pi_j,t = softmax(W*x_j,t + bias)`. Output probabilities are the
mixture `P_t = sum_j b_j,t-1*pi_j,t`; generation emits their deterministic argmax. Here b is the causal prefix belief defined below, initially p.
This is **not** softmax of an averaged latent vector. The codec is one physical
neuronal circuit time-multiplexed over branches, with weights held fixed during
one output decision. It does not have independently tied trainable copies.

Initialize `W=log(256)*I` and `bias=0`. For any of the 257 sensory identities,
including document EOS, the initial same-identity output probability is exactly
1/2 in real arithmetic; each other output has probability 1/512. This finite,
smooth identity prior provides both a meaningful copy preference and positive
support for learning a different output, including termination after punctuation.
The magnitude is derived from this declared 50% copy/50% uniformly distributed
alternative prior, not fitted to the roster. Byte identity is an authored general
codec, not a learned achievement. No special period/EOS connection is initialized.

Retained literal sensory bytes in synapses are an explicit episodic memory.
Removing source files is not semantic compression, nor is learned retrieval
automatically language comprehension. The same cursor and codec can continue
beyond two bytes, but this first workload does not qualify that capability.

## Select one-start sequence likelihood with causal prefix credit

The prospective objective is the negative log probability of the complete
teacher sequence under one latent start address, including its EOS. Keep eta=0.01;
no extra temperature or learning-rate sweep. Jacobs, Jordan, Nowlan and Hinton
motivate a mixture likelihood whose expert updates reflect how well each expert
explains an observation, rather than only its prior gate weight. Their original
example uses Gaussian experts and gradient-trained networks, not this proposed
local neuronal implementation. Our categorical sequence specialization and
filtering-credit approximation are explicit below. [Original paper, equations 3-5](https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts.pdf),
[author-hosted copy](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf).

One address means one complete forward continuation, not a different selected
start for each output position. Initialize belief `b_j,0=p_j`. For current codec
probabilities pi and supplied byte y, compute

`P_t[y] = sum_j b_j,t-1*pi_j,t[y]`

`b_j,t = b_j,t-1*pi_j,t[y] / P_t[y]`.

Use log probabilities and log-sum-exp throughout; retain finite log belief even
when a represented probability underflows. There is no probability floor or
imputation. The reported joint loss is `L = -sum_t log(P_t[y_t])`. Its unit is
categorical negative log likelihood, not the predecessor's half-SSE.

During free generation the network emits argmax(P_t), filters belief using that
emitted byte, then advances every branch one neural adjacency step. EOS ends
that attempt; the declared cap bounds it. Teaching starts a new query traversal
and fresh b0, and filters using the corrected teacher prefix, offering every
teacher byte and EOS even after an earlier failed/terminated first attempt.
This is declared teacher forcing, not hidden recovery credit. A branch stays
within one document because its document EOS is absorbing. Prefix belief is
ephemeral neuronal activity, not retained source or answer memory.

### Exact publication order and physical locality

Hold A, query-cell parameters, W and codec bias fixed throughout all three teacher
positions. At each position run the single physical codec time-multiplexed over
all occupied branches with those weights, save its current probabilities,
compute the updated prefix belief, and accumulate the local codec contribution

`d_j,t = b_j,t * (pi_j,t - onehot(y_t))`

`E_W += d_j,t*x_j,t^T`, `E_bias += d_j,t`.

No independently tied codec copies are required: the shared synapse sees its
current presynaptic sensory activity and its postsynaptic error, modulated by
the current branch's neuronal responsibility. Process branches with frozen
weights; accumulators retain only their local sums. Stored N-by-257 probabilities
avoid a second codec pass solely for producing the error. Any recomputation is
additional counted work, not free backward processing.

At the final teacher EOS, each address neuron's error is

`e_j = p_j - b_j,T`.

Stored key connections return `h=sum_j e_j*k_j` to the single q population. Its
one shared A projection then has local update `dA=-eta*u*h^T`, and
`A_pre*h` is the query-cell neuron error for existing local self-path eligibility.
There are no separate learned absolute-slot matrices or external per-synapse
address updates. Reciprocal current address transport is a disclosed
weight-transport assumption; query recurrence still omits cross-neuron history.

After all teacher positions validate finite, publish `W-=eta*E_W`,
`bias-=eta*E_bias`, and the A/query proposals from the same held pre-update state.
**Sum**, do not average, the three codec contributions; do not divide by slots,
257 outputs or teacher count. The sequence-NLL address derivative p-bT is already
for the whole offered sequence and is also not divided by three. This retains
eta=.01 per contributed sensory/output teaching term while applying one held
proposal group. Count three teacher offers and the actual group publications.
Use float64 signal accumulation and one checked cast where the existing query
parameter dtype requires it. No whole-lesson atomicity is claimed unless the
implementation validates all proposals before publishing any of them.

For fixed features and held codec, p-bT is the exact initial-address-logit
derivative of the joint sequence NLL. **The codec rule is an explicit local
online filtering approximation**, not that full joint derivative: it weights
each current error by current prefix belief b_t. A full joint-likelihood codec
gradient would use final b_T on earlier errors, requiring a different delayed
credit contract. We do not retain earlier branch errors and retroactively
reweight them. Each local contribution is committed to its accumulator when
its teacher arrives. This leaves an honest distinction between an exact local
address signal and approximate temporal codec credit. No network history is
traversed backward, no target start location is computed, and no future teacher
is consulted when recording a current codec contribution.

Teacher byte activity is broadcast to codec output neurons; their learned
conditional probabilities provide the responsibility signal. The host does not
compare stored bytes/substrings against the target to create an address label.
All candidate continuations and their neuronal evidence are evaluated, including
wrong starts, rather than selecting the best target-matching span outside the
network. No output error modifies the stored literal sensory payload.

### Acquisition scale and punctuation, algebra before any fit

With identity-valued linear decoding and uniform addresses, half-SSE produces
individual address errors of order 1/N. Retaining it solely because the old
output used it would preserve a known dilution risk. Under the selected smooth
codec, at the first teacher byte with N uniform addresses and m branches whose
current sensory identity equals y, each such branch has posterior
`256/(256*m+N-m)`; each other branch has `1/(256*m+N-m)`.
For illustrative N=50,m=1, `P_y=305/25600`, posterior is 0.8393442623 and
prior-minus-posterior is -0.8193442623 for that branch. This is scalar algebra,
not a model trial or a convergence claim. Later address credit uses the final
prefix posterior rather than freezing that first-byte result.

Prefix accumulation also addresses the concrete period/EOS problem. Suppose one
branch predicts both correct value identities then a period, while an EOS-only
branch predicts neither value and stays at document EOS. Their three-byte
likelihood ratio at initialization is

`(.5*.5*(1/512)) / ((1/512)*(1/512)*.5) = 256`.

Thus the value branch can retain responsibility on the third correction and its
period input receives a meaningful local EOS error. Independent per-byte
responsibilities would instead let the EOS-only branch dominate that correction.
This example assumes the first two wrong identities really differ; duplicated
partial matches can still compete. There is no fixed period-to-EOS mapping or
host punctuation skip. A shared memoryless codec might ultimately overgeneralize
termination for punctuation-rich coding; no contextual extension is queued here.

The 50% identity prior is generic and smooth, includes EOS symmetrically and
leaves nonzero probability for all corrections. If every branch gives the same
teacher likelihood, current evidence cannot discriminate addresses; posterior
ratios stay unchanged. Repeated symbols, inadequate query features, diffuse
initial keys and the filtering approximation remain real acquisition risks.
Normalize scores only over occupied addresses with stable softmax. Stored keys
have unit L2 norm except an all-zero prefix; retain the query cell's existing
bounded activity scale. No extra rescaling, probability floor, temperature or
eta compensation is selected. Failure closes this fixed hypothesis rather than
triggering an automatic loss, rate or epoch adjustment.
## Cost and one integrated comparison

Retire the old 14,593 slow writer parameters, 933,952 tagged sensitivity values
per stream, writer feedback projections and tagged/aggregate modes. Do not retain
dummy zero arrays or claim learned admission. The literal key/content store is
128*(1024+257) float32 values = 655,872 bytes per episode, before occupancy,
cursors, adjacency, query state and snapshots. Count the full 128-slot allocation,
including unused slots, and all 128 continuation branches if evaluated densely.
Count any explicit 128-by-128 cursor/adjacency arrays; sparse implementation is
not permission to omit their declared connections or active state. A has 32,768
slow synapses; W has 66,049 plus 257 bias values. Compute total state from the
implemented arrays, including query eligibility and serialization. This is not
an unmeasured speedup claim.

The objective adds one 257-way conditional codec normalization per evaluated
branch per output step, then one mixture reduction and responsibility computation
per teacher offer. At 128 branches the saved probabilities contain 32,896 values
(263,168 bytes in float64), before logits and proposal accumulators. A codec pass
with one-hot input can use the active W column plus b: this is sparse evaluation
of declared sensory synapses, not host answer selection. Count N codec passes
and N*257 probability outputs even when identical sensory symbols share a
mathematically equivalent batched implementation. There are no hidden backward
cursor traversals or teacher-selected codec passes. The fresh cost admission must
include this full work; old static-decoder timing cannot be reused.

Use exactly five arms: proposed learner; address/query-learning frozen with the
same plastic readout and sensory storage; all slow learning frozen with sensory
storage intact; sensory writes disabled with slow learning enabled; strongest
immediate *byte* baseline from a1ab (explicit archive). The event regression is
closed and need not be rerun. Sensory storage is local learning in the frozen-slow
arms; none is misleadingly called entirely learning-disabled. The address-frozen
arm isolates learned retrieval; old writer-frozen/aggregate labels no longer apply.

One fresh seed-733 run, same ordered 1,024 raw-document lessons per arm, same
8 fit/4 swapped-new/4 delayed/4 interference cases, and the same real Python
motor. First attempt precedes three byte/EOS teachers. No supplied positions,
additional output fit, paraphrase tuning, replay increase or endpoint diagnostics.
The existing cohorts remain exposed development. Ordinary authored math/interface
QA must cover all byte identities, repeated bytes, EOS absorption, longer
continuations, finite proposals, local scalar credit and absence of host selection;
those are unit tests, not a preliminary learned component benchmark.

Preregister useful completion as 8/8 fit and 4/4 in each other stratum, both
same-question swapped assignments correct, and strictly more novel successes
than address-frozen, all-slow-frozen, memory-off and the immediate byte baseline.
Report useful system gains even if the stronger learned-retrieval claim fails.
Keep delayed duplicates separate from independent new-value successes. Preserve
source-free checkpoints, all scientific counts, exact baseline parity and actual
action results; the old host literal-edit motor remains declared scaffolding.

Admission is fresh: four lesson indices across five arms (20 lessons/60 teacher
offers), all 100 final action cases plus two demos, complete-cap measurements and
the original 2x projection plus 30 seconds under 900 seconds/2,048 MiB. Count every
raw input, sensory association, cursor/branch advance, normalization, potential
evaluation, teacher offer, snapshot and audit. Stop if cost or custody fails;
otherwise execute the single frozen integrated fit. On failure close this
specified retrieval/content hypothesis before selecting another mechanism. On
success the next shared milestone is longer content and varied document/query
form in the same action loop, not a semantic-comprehension or SWE claim.

## Implementation bindings before experiment freeze

Implement the smallest complete learner, with production tape capacity 128,
32 lags of 32-dimensional fixed codes, the existing 32-neuron query cell and
one learned 32-by-1024 address projection. Use seed733, query seed+1, code-table
seed+2 and private Xavier address seed+4. Query, address and sensory/key state
are CPU float32; codec weights/bias, log beliefs, probability reductions and
local signal accumulation are float64. Validate finite proposals before their
group publication. No autograd, optimizer or backward traversal is introduced.

Sparse active indices may represent one-hot write/recall cursor populations.
Report both their logical neuronal population/connectivity and stored tensor
bytes. Sparse successor gathers must exactly implement forward adjacency and
document-EOS absorption; they cannot inspect teacher values. Capacity overflow
must reject before changing caller-owned memory. Reads and teaching preserve
stored content, keys and adjacency. Append-only unrelated ingestion preserves
existing slots; generic document boundaries reset the temporary lag population.

The sensory-writes-disabled arm advances the same allocation and boundary clocks,
but stores zero sensory/key activity, including at EOS. Its layout metadata is
still retained and counted; it is not a control with literally no episode state.
The zero sensory input reaches the same codec bias and its softmax. Slow-frozen
and address-frozen arms retain normal sensory association writes. Name these
scopes explicitly rather than calling every frozen control learning-disabled.

Replace the obsolete main writer/phase-output composition directly. Preserve
closed experiment implementations only as explicit experiment references or at
their frozen commits; do not add a compatibility mode to the new learner. Reuse
the established rate-cell, byte-generation, snapshot and executor utilities where
their contracts still apply. Retire tag/aggregate/writer-feedback state from this
main path. Keep the byte baseline's exact original 1,024-lesson trajectory and
twenty outputs as a reference contract, not a checkpoint migration.

The implementation review must fix source-free transport, the five named arms,
actual teacher/group-update counts and complete input/cursor/codec work before
cost admission. Report every extra conditional-codec evaluation or replay. Ordinary
authored QA validates longer continuation and scalar local math, but does not
train a separate output model or inspect novel endpoints. One fresh cost run and
one conditional fixed integrated fit remain the next executions.
