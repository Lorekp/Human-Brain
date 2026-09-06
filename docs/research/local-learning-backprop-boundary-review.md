# Local learning and the backpropagation boundary

Root decision: apply the user's prohibition to manual reverse error transport
through fixed maps as well as learned maps. The current `K.T @ (p - b_T)`
teacher is withdrawn from further training. Preserve completed measurements;
they do not establish a backpropagation-free learner. The proposed frozen
single-address comparison is unselected because it would retain that teacher.
The [replacement review](local-neuronal-learning-return-review.md) owns the
next same-system, local-learning return. No old-rule training rerun, including
as a reference, is authorized by this disposition.

Source-only critical review, 2026-09-06. Examined exact
`82db27e847f00fb065f03d5464fc519116e6a6db:src/humanbrain/local_memory_learner.py`
through `git show`; root reports the current a500 source retains that learner.
No model, checkpoint, endpoint, fit or source modification was executed.

The user's requirement is: “do not use backpropagation. the learning and the
executing system should be the same (just like in humans).”

**Finding:** the current learner has no backward traversal through time and no
chain of transposed learned layers. However, it explicitly computes the exact
reverse error transport through the forward address-key map. Calling the complete
rule “not backpropagation” simply because the final synapse products are local,
or because autograd is disabled, would conceal that operation. The unchanged
teacher clause in the proposed single-address contract cannot be certified
unconditionally compliant with the user's new prohibition.

## What the code actually computes

Use Z for the fixed normalized query-lag activity, B for the only learned address
projection, Q=B.T Z, and K for occupied tape keys arranged as rows. Source lines
273-277 compute Q, scores s=K vec(Q), and p=softmax(s). K contains literal
source-derived preceding-context associations; it is held during a lesson, not
trained by the answer loss. Its fixed status does not remove its place in the
forward computation.

The teacher advances all branch cursors forward. At each offered token y_t it
computes the shared codec's probabilities pi_j,t and a forward Bayesian update
of branch activity:

```text
b_j,t proportional to b_j,t-1 * pi_j,t(y_t), with b_j,0 = p_j.
```

This is the actual log-space update at line 307, used by teach at lines 384-389.
There is no reverse loop over these activities. Saved per-step receipts are not
read backward to calculate learning signals.

After the last offered token, lines 410-420 perform:

```text
e = p - b_T
h = K.T e
H = reshape(h, 32, 32)
G = Z H.T
B_new = B - .01 * cast_float32(G_float64)
```

For a held codec and tape, e is the derivative of the sequence marginal negative
log likelihood with respect to initial address logits. h is the vector-Jacobian
product through s=K vec(Q). G is then the derivative at the B synapses, summed
over their 32 channel applications. Thus this is an explicitly coded conditional
gradient path through the fixed address readout into one learned projection.
The absence of a backward API does not alter that mathematical or operational
description. In particular, `F.linear(e, K.T)` is the backward transport in
question; the `.T` in the ordinary forward expression `B.T @ Z` is merely that
matrix's storage orientation and is not itself evidence of backpropagation.

There is no B.T error return into a trainable query encoder: query features are
fixed. There is no codec-weight transpose carrying errors into B. The codec
supplies likelihood activities for the posterior, not a cascade of weight
Jacobians. There are no gradients through the document writer, cursor history,
lag shift history or Bayesian filtering history. These are substantial limits
on the reverse credit path, but do not make the existing K.T error product vanish.

## Codec learning and local ownership

Lines 389-399 form the current signal
`b_j,t * (pi_j,t - onehot(y_t))`. Each shared codec synapse accumulates that
output-neuron error times its current sensory presynaptic activity. The
`index_add_` is sparse accumulation into one physical shared codec, reused across
branches; zero sensory activity reaches bias neurons but no active weight column.
Parameters stay held for the entire teacher sequence, then finite proposals are
published together.

This is a supervised single-layer delta update with a current posterior gate.
Its output errors are not propagated through codec weights to another learned
layer. Weight holding and summing local products do not create a history reverse
pass. Moreover, the codec uses each current prefix posterior, not a final
posterior retrospectively substituted for all earlier codec errors: it is not
the exact full-sequence codec gradient. This approximate local rule should not be
described as exact end-to-end gradient learning.

The B synapses likewise receive their own Z presynaptic activity and an H
postsynaptic error. The *plasticity product* is local. The separate question is
how H reaches those neurons, and current code answers it by an explicit transpose
of the forward key array. Local multiplication alone is insufficient evidence
about the locality or admissibility of that signal generator.

## Which feedback circuit is actually simulated?

The program stores K once and reuses its transpose for an arithmetic feedback
multiply. It does not instantiate separately owned reciprocal feedback synapses,
an additional neuronal reconstruction phase, or dynamics that establish those
feedback weights locally. An implementation can declare exact reciprocal
associative connectivity as a prior, but this current source has not simulated
its construction or independent physical state. Claiming such a circuit has
already been demonstrated would go beyond the code.

Algebraically, h equals the difference of two key expectations:
`sum_j p_j K_j - sum_j b_j,T K_j`. That suggests an associative activity-difference
interpretation. But merely computing those two sums instead of `K.T e`, renaming
H a teaching activity, or wrapping the multiplication in a neuron class would
leave exactly the same reverse credit operation. None is a substantive correction
to a prohibition that includes manual reverse-Jacobian transport. It is also
inaccurate to call this a hidden multilayer learned-weight backpropagation
network: the actual problematic boundary is the narrower fixed-key adjoint.

## Same learner versus identical execution schedule

Learning and action use the same B, codec, codes, tape and cursor connectivity;
there is no external trainer, distilled executor or separately fitted answer
solver. The current marginal free policy filters its own generated tokens;
teaching instead supplies the corrected prefix. The proposed single-start free
policy would activate one selected cursor while teaching still evaluates all
starts. Its maximum all-branch teaching circuit therefore remains allocated.

Shared physical parameters do not make these activity schedules identical. The
single-start proposal must expose that difference and cannot claim that its
frozen checkpoint comparison establishes online acquisition under the new policy.
Any later accepted policy must return to fresh action and local feedback in the
same learner. The existing no-autograd flag proves neither biological fidelity
nor compliance with a demand for identical learning/execution computations.

## Mandatory disposition before implementation

Do not dispatch the single-start change under an unqualified assertion that the
retained teacher is “no backpropagation.” The minimal correction now is to remove
that assertion and explicitly surface the fixed-key adjoint and the differing
teacher/free activity schedules in the selected mechanism contract. The source
already supports the narrower statement: no BPTT, no learned-layer reverse chain,
local single-layer plasticity, and one explicit exact fixed-key error return.

If root applies the user's prohibition to *any* manual reverse-Jacobian credit
transport, preserving the teacher is a blocker: the address teaching mechanism
must genuinely change before claiming compliance. There is no metadata-only or
algebra-only patch that fixes that interpretation. A replacement must specify an
actual forward/local neuronal teaching-activity generator and its own signal and
synapse ownership, rather than invoke the old adjoint under another name. Such
a learning-rule change invalidates the rationale for testing only a frozen output
policy; it requires newly declared ordinary QA and a fresh integrated acquisition
comparison. This review selects no substitute mechanism or parameter sweep.

If root instead accepts explicit associative feedback to a single learned layer
within the allowed local-error rule, that is a stated architectural boundary, not
a finding that this source contains no gradient transport. Keep its actual K.T
operation, exact signal claim, physical feedback assumption and computation costs
visible. Root must resolve that boundary against the user's new instruction;
the prior broader local-online permission alone cannot silently settle it.
