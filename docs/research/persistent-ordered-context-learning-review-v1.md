# Ordered context through local predictive learning

Status: bounded primary-source and repository review, 2026-09-07. No model,
feature, checkpoint, corpus, workload builder or experiment was run. Two
mechanisms were investigated. **No implementation-ready successor is selected
by this review.** Sparse dendritic temporal memory is the more relevant starting
point, but the raw-instruction-to-output binding rule remains unspecified.
Calling that gap an “ordered-context population” would conceal it.

## Decision required by the measured failures

The [roadmap](roadmap-reading-to-coding.md) and
[active plan](persistent-language-action-development-v1.md) require the same
persistent system to learn and act, with local signals and no backpropagation or
manually transported upstream derivatives. Fewer answer corrections must not
hide additional source labels. A new mechanism belongs in the broader language,
memory and action loop, not another isolated success gate.

The [closed reservoir history](roadmap-experiment-history.md#1-reservoir--readout-era--closed)
already distinguishes storage/separation from useful cross-context binding.
The [race-pivot result](race-pivot-gate-verdict.md) produced diverse wrong answers
despite complete training replay. More recently, exposing raw lag evidence
improved query binding relative to a small recurrent compression, although that
comparison did not prove information destruction in the encoder. See the
[lag-query review](local-lag-query-review.md). A fresh reservoir, a slower trace,
or a sparse code with a new local readout therefore needs a new causal binding
argument, not merely a larger capacity estimate.

V6 adds the same warning from a different architecture: private recognition
updates reached their intended input-defined owners, yet owner398 started60/62
online responses and `hold/express` appeared35 times. Those are
[partial recorded findings](persistent-language-action-v6-learning-review.md),
not a completed V6 score. The missing distinction is between recognizing an
observed occurrence and choosing useful content under a different instruction.

## Two primary-source mechanisms

### Sparse dendritic temporal memory

Hawkins and Ahmad distinguish a sensory column from context-specific cells
within it. A segment predicts a cell when enough connected presynaptic cells
are active. Predicted cells preferentially respond to the next matching sensory
input; unpredicted input activates more cells. Segment-local permanence changes
learn transitions. Their apical illustration supplies a stable feedback pattern
through a sequence; it does not learn that pattern from natural instructions.
These are sequence-memory results, not demonstrations of program composition.
[Hawkins and Ahmad, 2016, Figures2–5 and Methods](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2016.00023/full).

In compact notation, a connected contact is `c_ijk = 1[p_ijk >= tau]`, segment
activity is `d_ij(t) = 1[sum_k c_ijk a_k(t) >= theta]`, and prediction is
`pi_i(t) = OR_j d_ij(t)`. Reinforced contacts increase for active previous inputs
and decrease for inactive ones, with permanences bounded in[0,1]. These operations
use forward activity and local contact state; no output-error derivative through
the preceding state is required. This is the attractive difference from a fixed
recurrent encoder. [Same primary paper, Methods](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2016.00023/full).

The authors' implementation makes relevant policy choices concrete:
`compute(activeColumns, learn)` activates cells before computing the next
prediction. An unpredicted column chooses a learning cell/segment from previous
context and grows contacts to previous winners. Incorrect-prediction punishment
is configurable and defaults to zero. Segment limits can trigger replacement.
These choices must be explicit in a continuation; the library defaults are not
a project specification. In particular, initial permanence below the connection
threshold does not promise useful one-occurrence learning.
[Numenta's TemporalMemory source](https://github.com/numenta/nupic-legacy/blob/master/src/nupic/algorithms/temporal_memory.py).

### Self-organizing recurrent network, SORN

SORN combines threshold recurrence with three local mechanisms: temporally
asymmetric excitatory STDP, incoming-weight normalization and intrinsic threshold
adaptation. Schematically,
`delta W_ij = eta [a_i(t)a_j(t-1)-a_i(t-1)a_j(t)]`, followed by normalization;
`delta T_i = eta_IP [a_i(t)-r]`. These depend on pre/post activity and neuronal
state, rather than transported output derivatives.

However, the reported prediction study exposes the network for50,000 steps,
freezes its recurrent parameters, then fits a separate readout by a
Moore–Penrose pseudoinverse using5,000 activity steps. That performance is not
evidence for our direct-local, continuously acting, sparsely corrected system.
Replacing its fitted readout would create an unvalidated variant. The paper
supports learned temporal representations, but does not supply the missing
instruction/content relation. [Lazar, Pipa and Triesch, 2009, Methods and Results](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/neuro.10.023.2009/full).

SORN is therefore not recommended for the next integration. Its local recurrent
plasticity is materially different from a fixed reservoir, but the proposed
project benefit would still depend on precisely the representation-to-answer
mapping that the repository has repeatedly failed to obtain.

## What a dendritic candidate would and would not specify

The following is a **project design decomposition**, not a claim that the paper
implements the repository task. It identifies which pieces are already concrete
and which remaining decision prevents selection.

1. **Observed identity.** Use the same lossless raw-byte or explicitly admitted
   token identities for instruction, source, feedback and generated payload.
   Public field boundaries may be context inputs; do not create hidden
   identifier/value/answer roles. A sensory symbol can activate several fixed
   columns. Cells within those columns provide learned temporal context.
2. **Causal transition state.** Retain active/winning cells and segment activity
   across successive symbols within an explicitly delimited stream. Before
   observing symbol `x_t`, form predictions from state through`t-1`. When `x_t`
   actually arrives, only its sensory columns receive that input. Learning-cell
   choice within those columns uses preceding activity and current local state.
   The observed symbol is a counted postsynaptic teaching/input event, not an
   oracle memory address. Do not describe this as target-free output selection:
   the symbol has arrived before this particular learning update.
3. **Direct local teaching.** A reinforced segment can apply
   `p_k <- clip(p_k + eta_plus*a_k - eta_minus*(1-a_k), 0, 1)` using previous
   presynaptic activity. Novel segment allocation must be caused by the arriving
   input and preceding population, with a declared capacity/replacement policy.
   False predictions may receive an explicitly bounded local decrement. This
   bounds individual contact changes; it supplies no theorem that global
   sequence loss decreases or older behavior is preserved.
4. **Shared acting state.** During generation, predictions use the same cells
   and synapses. A chosen byte becomes the next causal input. Actual stdout,
   stderr and other admitted feedback enter later through the same sensory path.
   No teacher prefix enters a free response. Self-generated bytes are not
   silently promoted to verified source-training targets.

This can learn context-dependent transitions and distinguish some occurrences
of the same symbol. It is neither an event-index latch nor a full-context scalar
recognition gain. But it still leaves two coupled forward decisions unresolved:

- **Persistent instruction/content binding:** what actual neuronal activity
  remains available from the complete raw request while source and output
  symbols arrive, and what local experience makes changed wording activate a
  useful related population? Taking the last recurrent state risks losing early
  distinctions; pooling every active cell risks common-context attraction.
  Supplying a stable “task cue” from an episode, answer group or host semantic
  parser would insert the missing solution.
- **Autonomous byte selection:** temporal prediction can nominate several next
  symbols. The actor needs a fully specified current and competition rule,
  including ambiguity, no prediction and EOS. Summing predictions, taking a
  maximum, or adding an old-router prior make different choices. None follows
  automatically from a local permanence update. A new readout trained from the
  same broad, unbound state would repeat the old problem.

These are not requests for extra benchmark exposure. The missing design object
is the **causal forward population and local association that binds an
instruction-derived state to a content/payload state**, together with its actual
byte-selection rule. It must be defined before deciding that transition learning
is the next complete mechanism. In particular, a rule that strengthens only
adjacent source symbols does not explain why a question about one field should
read another field, why negation reverses a requirement, or why two learned
operations should compose. Locality and sequence capacity cannot supply those
relations by assertion.

## Actionable integration boundary

If that binding decision is resolved, the smallest honest implementation seam
would be one new persistent population owned by the existing reader/actor,
with these conceptual operations:

| Operation | Available signals and publication boundary |
|---|---|
| `begin_observation(raw_frame)` | Consume all admitted public fields with explicit order and boundaries; construct the actual instruction/content state. No grades or future targets. |
| `predict_next(state)` | Produce byte/EOS currents and a sealed decision from existing state. The unresolved binding and arbitration rules belong here, not in the runner. |
| `observe_external_symbol(symbol, provenance, learn)` | Compare with the already formed prediction, then update local transition state/contact proposals. Provenance controls exposure accounting, not semantic routing. |
| `accept_generated_symbol(decision)` | Advance the same causal state with the actual emitted byte; no implicit correctness label. |
| `prepare_correction(pending_response, target_stream)` | The actual response remains sealed. Earlier offered target bytes may construct correction state only after their counted offers. Sequential proposals are validated before joint publication. |
| `snapshot/restore` | Include new contacts, allocation state, transient stream state when resumable, old model/tape state and exposure cursors; preserve idle/pending guards. |

This is an interface boundary, not permission to implement unspecified bodies.
The old action heads and executor stay in the same system. A readout cell may
legally use `delta w_ci = eta*(T_c-v_c)*a_i/(epsilon+||a||²)` because `v_c` and
`a_i` are its own current and presynaptic activity. Using `W^T*(T-v)`, a manually
expanded chain derivative, or an inferred “correct hidden state” to teach the
earlier population would violate the present scope. Merely keeping autograd off
would not repair that violation.

## Exposure, continuation and scaling

**New experience must be visible.** For a fresh source stream of n bytes, a
declared BOS prediction followed by each arriving raw byte and one EOS creates
`n+1` symbol-teaching opportunities; omitting first-byte prediction instead gives
n. Choose one convention before exposure. Count field delimiters and prediction
opportunities on instruction/feedback bytes separately, even when their input
is already public. Re-presenting a whole source after ingestion is replay and
must be counted again. Reading a future symbol to form the state that predicts
it is invalid. Source syntax learning is self-supervised in the sense that its
targets arrive naturally; it is still additional neuronal teaching.

V6's884 source bytes plus12 EOS currently perform association writes, with no
source next-symbol teaching. A hypothetical896 next-symbol offers under the
first convention would be new work, not a reinterpretation of those recorded
writes. Nor would those limited bytes prove syntax acquisition. More raw
experience can reduce dependence on answer corrections, but does not reduce
total experience. Corpus origin, exact bytes, one-pass order, external feedback
and all correction symbols must remain in the shared evidence ledger.

**Continue recoverable state only.** The latest complete saved state is V5:
398 event owners and864 factual slots. V6's533/1760 state was observed in the
partial run but was not checkpointed. Preserve the actual V5 numerical model,
owners and tape; new transition contacts start empty or with a declared fixed
prior. Do not reconstruct V6 gains or knowledge through uncounted replay. Any
new input population changes forward semantics, so retained tensors do not
establish functional parity after switching to it.

**Cost is explicit even without a run.** For C columns, M cells/column,
D segments/cell and S potential contacts/segment, storage is `C*M*D*S` contacts.
With int32 presynaptic indices and float32 permanence, their payload alone is
`8*C*M*D*S` bytes. For illustration, `C=2048,M=8,D=4,S=24` gives1,572,864
contacts and12,582,912 bytes, excluding cell states, occupancy, allocation,
reverse indices, snapshots and validation. This is arithmetic, not a chosen
encoding or sufficient capacity. A full scan performs up to that many contact
tests per symbol. Sparse delivery can avoid inactive contacts only with exact,
maintained adjacency whose storage and mutation are counted; a selected answer
shortlist is not an equivalent optimization.

An unconstrained V-way dense output map adds `V*C*M` weights and corresponding
products per decision. For257 outputs and the illustrative16384 cells, float64
weights alone cost33,685,504 bytes. This is one reason not to silently append a
new dense readout to an attractive sparse transition model. Maintaining context
over a whole observation also costs every consumed raw symbol, rather than one
fixed-vector invocation. No speed or resource admission is inferred from these
nominal counts.

## Recommendation

Do not choose SORN or declare the dendritic substrate implementation-ready from
this evidence. Keep **one design completion question**: define an input-derived,
locally learned instruction/content binding and autonomous byte arbitration
around sparse dendritic temporal state, without stable episode cues or upstream
derivatives. That is the more concrete candidate family, because observed
sequence learning changes the causal state itself rather than only a final
prototype score. Its possible advantage is reusable ordered transitions; its
unresolved limit is relevance and composition across contexts.

Until those signal origins and equations are supplied, a new run would mainly
test another sequence store's output collapse. No extra fit, gain sweep,
pretrained encoder, corpus download or toy gate is proposed. Mem0 was searched
only as external development context; the conclusions here rely on the linked
repository evidence and the two primary mechanisms, not retrieved memory claims.
