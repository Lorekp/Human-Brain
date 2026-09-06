# Address competition with explicit corrective teaching

Source/math proposal, 2026-09-06. This implements no policy and authorizes no
model call, alternate decoding, fit or acquisition. The
[roadmap](roadmap-reading-to-coding.md) and completed
[learning review](neuronal-splice-online-v1-learning-review.md) own the scope.
The v1 endpoints are exposed evidence; they cannot qualify a successor.

Recommend one **initial address commitment circuit**, followed by the existing
byte codec at one actual cursor. Keep the existing raw request, forward joint
features, trained address synapses, action heads and persistent learner. The
necessary teaching contract is explicit: after the actual executed proposal,
supervised correction reopens the address population to compute forward evidence
for a direct output-neuron contrast. This is not the derivative of the committed
generation loss. Root must accept that contract before implementation.

## Current source path

The reviewed [reader](../../src/humanbrain/local_memory_learner.py) has four
relevant stages. `_query` (line 277) encodes raw bytes plus EOS into the normalized
32-by-32 lag population `Z`, materializes `X_s = Z K_s^T` for every occupied slot,
then scores `a_s = <B, X_s>`. `B` has 1,024 trained physical synapses. Float32
scores become float64 `log_softmax` probabilities `p`; there is no query/key
error transport. The raw query representation retains its last 31 bytes plus
EOS. No request extraction belongs inside the proposed circuit.

`predict_with_receipt` (line 371) initializes all slot cursors, evaluates the held
codec bank once, forms the probability mixture, selects a byte, updates the
branch posterior from that actual byte, and advances every cursor. Thus many
low-probability branches can jointly win an output token even when the intended
start has the largest individual address score.

`teach` (line 406) holds the same parameters throughout all supplied bytes and
EOS. Its forward prefix observer computes responsibilities, the codec receives
current prefix output errors, and the address leaf receives `p - b_final` on its
own `X_s`. Every parameter proposal is validated before either parameter group
publishes. `snapshot` and strict loading preserve physical weights and tape.

The [splice wrapper](../../src/humanbrain/neuronal_splice_learner.py) already owns
the pending proposal identity, actual execution envelope and publication order.
It stages action proposals before the reader's atomic correction. Explicit STOP
teaching calls only the operation head. Preserve those properties.

Graph discovery was attempted against the verified checkout project; its symbol
search returned no reader nodes. These traces therefore use the actual source.
External development-memory results did not supply scientific facts or design
requirements for this proposal.

## One proposed forward interface

After `_query`, a fixed comparator/inhibitory population receives the **actual
float32 address scores** for all `N <= 128` occupied slots. No target mask,
payload-validity filter, path parser, EOS exclusion or teacher enters it.
For `i < j`, a first-layer comparator neuron evaluates

`c_ij = H_ge0(float64(a_i) - float64(a_j))`.

The second-layer winner neuron evaluates

`w_i = H_ge0(i + sum_{j>i} c_ij - sum_{j<i} c_ji - (N-1))`.

The expression before the final subtraction counts pairwise wins. It reaches
`N-1` only at the first maximum. Float64 comparison currents receive the exact
finite float32 scores; no new trainable gain or numerical tolerance is used.
Exactly one address neuron remains active, including for `N=1`.
The index tie rule is a disclosed fixed wiring prior; it is not learned semantic
priority. With finite scores, this circuit is exactly the first maximum. The
explicit comparator topology is an abstract simulated neuronal circuit, not a
claim that a library argmax models biological settling dynamics.

Expose a small immutable `AddressCompetition` receipt containing score count,
winner activity, winner index, winning score, maximum tie count and work. A pure
internal `compete(scores)` evaluates these two fixed layers in each free
traversal. The index is the fixed identity readout `sum_i i*w_i`,
after validating the one-hot population. Vectorizing the actual upper-triangular
comparison bank is allowed; replacing this specified circuit with an uncounted
host argmax is not its implementation.

Initialize **one** actual cursor at the selected slot. Evaluate the existing
codec at that cursor, emit its largest-probability byte, and advance this cursor
using the unchanged tape adjacency. Continue through the same bounded byte/EOS
rule: at most 16 payload bytes plus one stopping decision for splice calls.
The codec owns EOS; the host never inserts a stopping byte or truncates an
incorrect payload into a correct one. A committed wrong start is not repaired
by later mixture filtering. All ordinary read-only reader calls would use this
same free policy, including old-reading retention, rather than hiding the change
behind an action-only special case.

The held codec bank may still be evaluated once per call and gathered at the
single cursor. That is the smallest reuse of the current implementation; it
does not earn a claim of reduced initial codec work. Evaluating only visited
rows would be a separate implementation choice requiring exact caching/work
review, not a prerequisite for this mechanism.

## Corrective phase and local credit ownership

The free result must execute and bind to the same pending learner/tape before
any corrective byte is provided. Its pending receipt already records the actual
commitment. Inside `teach`, recompute the same `_query` on held weights, then
activate an explicit supervised phase in which **all** address neurons are
available to the existing forward teacher-prefix evidence computation. There is
no redundant second competition pass, additional free output, or separate
parameter instance in this phase:

`b_0 = p`,
`b_{t,s} = b_{t-1,s} C_{s,t}(teacher_t) / sum_r b_{t-1,r} C_{r,t}(teacher_t)`.

Every teacher byte and EOS traverses the same codec and tape circuit. No
externally selected correct address is supplied, and no literal substring
search constructs a neuronal teaching target. In particular, an empty payload
offers EOS; it does not identify an authored punctuation boundary as its target.

Retain the current direct output contrast at each physical address synapse:

`delta B_ij = -eta sum_s X_s,ij (p_s - b_final,s)`, with `eta = .01`.

The materialized `X_s,ij` is that synapse's own forward presynaptic activity.
`p_s` and teacher-driven `b_final,s` are activities of its direct address output
population in two disclosed phases. The latter uses forward likelihood evidence,
not a reverse derivative through competition, query lags, keys, earlier bytes,
or the codec. No `B^T`, `K^T` error transport, competition Jacobian, unrolled
recurrent gradient or hidden eligibility approximation is introduced.

Codec teaching may retain the current all-branch, current-prefix responsibility
rule, using each actual sensory input and direct byte-output error. This is
supervised alternative-path exposure **after execution**, not a second executor
or independently trained reader. Count all of that activity, including branches
suppressed during free recall. Do not describe the resulting update as an exact
gradient of the one-cursor policy, or the actual execution grade as its numeric
cause: the grade remains external and raw supervised correction drives learning.

This reopening is essential, not optional compatibility boilerplate. If the
teacher retains exactly one-hot support and the codec likelihoods are positive,
every posterior equals that same one-hot vector. An address error formed as
`w - posterior` is identically zero and cannot correct a wrong commitment.
Conversely, substituting `w - b_final` for the existing rule without a separate
learning argument is a different update, not an innocuous receipt change.
The proposed contrast deliberately retains `p - b_final` and labels its scope.

All address/codec proposals still use held old weights and validate before
publication. The existing action staging and same-instance execution binding
remain intact. Frozen mode gates and `learning_enabled=False` suppress their
declared publications without suppressing actual evaluated work. No competing
checkpoint, teacher-specific model instance or rollback trainer is needed.

## Why this proposal rather than recurrent soft competition

Hard commitment directly addresses pooled emissions while keeping the forward
state small and the stopping rule exact. It loses prefix disambiguation and
combining evidence from duplicate valid starts; an early wrong winner can remain
wrong for the whole payload. T01 is an observed misranking that the saved rank
evidence does not resolve. Correct top rank alone also does not prove that every
subsequent codec output will be correct. The missed STOP remains a separate
observed failure of the existing system.

A recurrent soft population could preserve alternatives, for example through
fixed self-excitation and shared inhibition. But it needs a declared update,
iteration budget, initialization, gain and stopping rule. Finite recurrence does
not guarantee removal of pooled EOS or concentration on a unique target, and
near ties can require more iterations. Differentiating its final activity with
respect to `B` would require the forbidden reverse transport through recurrence;
silently retaining `p - b_final` as though it were that derivative is incorrect.
It would need an explicitly justified direct corrective contrast too. No existing
saved result selects a recurrence or gain. This makes it the larger unresolved
intervention for the current slice.

## Minimal implementation and evidence obligations

The existing `NeuronalSpliceLearner` constructor already accepts an injected
`LocalMemoryLearner` instance and permits subclasses. Prefer a new
`CompetitiveLocalMemoryLearner` subclass passed through that exact interface;
the frozen action source need not change. Inherit `_query`, `teach` and their
numerical feature/publication bodies unchanged. Override free prediction and
explicit version/phase metadata. This is one persistent combined model, not a
second learner fitted for execution. A committed prediction must report its selected start,
actual cursor/codec decisions and competition record. It must not manufacture
all-branch posteriors or feed infinities into the existing finite-log-belief
serializer. The uncommitted `p` may be retained once as a clearly named diagnostic.
Teacher receipts retain their real full-population evidence with an explicit
surrogate-phase label; they do not invent a second competitive traversal.
The splice wrapper needs the actual generation and work, while its compact
serializer, worker copy inventory, independent auditor and checkpoint metadata
must explicitly recognize the new receipt/schema. Root owns that integration
review; no generic compatibility framework is warranted.

There are no new slow parameters. The circuit has `N(N-1)/2` comparator neurons
and `N` winner neurons: 8,128 plus 128 at the maximum geometry, or 8,256 cells.
Each comparator has two fixed signed score inputs; each winner has `N-1`
fixed signed comparator inputs and its declared index-dependent bias. There are
two feedforward threshold layers, no recurrent settling or tournament relay.
Count the actual currents, thresholds, intermediate arrays and identity readout.
Free cursor/filter work
decreases to one path; the all-branch teacher and its feature/local-product
temporaries remain. Existing X and held-codec storage dominate these small
additional populations. No speed claim or reuse of old admission follows.

Meaningful ordinary QA should establish exact scalar/vector competition parity
including ties, genuine single-cursor traversal/EOS/cap behavior, reopened teacher
evidence after a wrong winner, and exact direct output-local update arithmetic.
Verify that keeping one-hot support gives the identified zero-credit failure,
that the chosen corrective phase avoids it, and that late validation failures
remain atomic. One same-instance real-action-before-correction fixture should
exercise the new receipt and worker boundary. Those are prospective mechanics
checks, not authorization to run the exposed v1 endpoint roster again.

Before any fit, root must decide whether the explicitly reopened corrective
phase satisfies the desired learning/execution contract. If teaching must remain
restricted to the committed path throughout, this proposal is blocked by its
address-credit failure and should not be implemented under an unchanged teacher
label. If accepted, freeze new withheld tasks, controls, complete accounting and
fresh admission before a single successor comparison.
