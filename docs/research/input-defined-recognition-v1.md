# Input-defined local recognition

Status: root selects candidate B below for the next shared implementation,
2026-09-07. The owned-free optimization already passes ordinary QA. This document
contains the reviewed source/math comparison and the implementation boundaries;
no model, feature, checkpoint, task or executor was evaluated for this decision.
Candidate A and sequence-latch proposals are not separate active queues.

## What the evidence requires us to distinguish

V5 online's 163 routed owners all differ from the independently prepared, newly
recruited event whose outgoing byte is taught. Thus even a positive current-byte
outcome updates the already routed evaluation, not the new event's recognition.
A negative **outcome** is not necessarily a negative current error: with desired
current d=-1, epsilon=d-s is positive if s<-1. Sign claims below concern epsilon.

The measured failure is not old output-vector erasure: all 163 additions are
novel and the previous 235 output vectors remain intact. Under four identical
online/control initial contexts, all previous 235 owners have lower online
currents and a new reading answer wins instead. Recruitment and shared-theta
plasticity are bundled, so this does not identify one feature block as the sole
cause. A coherent `idle` sequence dominates many unrelated code requests.

## A. Split signed flag differences, leaving shared credit unchanged

For each old flag difference d=f_current-f_owner, replace d by two fixed
nonnegative activities d_plus=max(d,0), d_minus=max(-d,0). Keep the signed
position difference u and abs(u). Normalize the new metadata block separately
to the same half-energy as before. Its squared raw norm is unchanged because
d_plus²+d_minus²=d². This is fixed directional coincidence, not a semantic flag.

There is a direct continuation mapping:

`theta_plus=theta_old_flag; theta_minus=-theta_old_flag`,

with the old lexical and two position coefficients retained. It preserves
currents in real arithmetic. A different float64 reduction order need not give
bitwise parity or identical decisions at arbitrarily small gaps. It requires a
new explicit schema, not in-place reinterpretation of an old checkpoint.

The old 12 signed flags become24 cells. Total logical width becomes1,033,316;
the shared f64 parameter adds96 bytes. Each evaluated sparse row adds12 actual
metadata values and corresponding products. All 398 inherited event identities,
receptive patterns, outputs and factual memory remain. Existing candidate raw
flags can produce the new activity without lesson metadata.

This removes a precise sign interaction. Under a complete literal context swap,
old lexical products are identical: the old full activity dot product is already
nonnegative (zero at equal positions with nonzero lexical blocks). That swap
cannot be claimed as a demonstrated negative-update boost. Different lexical
contexts with opposite flag directions can instead have a negative old dot
product. The split removes that flag contribution. The remaining position pair
has dot u*v+abs(u)*abs(v)>=0, and lexical products are nonnegative. Therefore
all complete new activity cross-products are nonnegative.

Consequently a shared update `delta_theta=epsilon*z_j/E_j` with epsilon<0
cannot increase another evaluation's current through a negative cross-product.
It can still **decrease many unrelated owners together**. Positive errors can
still increase many together. The measured broad suppression and the missing
positive association to a new taught owner are therefore not fixed by this
guarantee. A is a small removal of one algebraic risk, not evidence of improved
language relevance, retention or sequence credit. It adds no external labels;
its actual causal contribution would remain an empirical question.

## B. Input-defined, owner-specific recognition plasticity

Separate contextual recognition from current-byte utility. Preserve V5 theta as
a frozen prior and add private recognition synapses rho_j to each event owner.
Use the existing canonical sparse activity z_j(x) for that owner:

`s_j(x)=beta_j(x)+theta_prior·z_j(x)+rho_j·z_j(x)`.

Every occupied event still participates in competition. The byte readout and
same persistent actor remain unchanged. Set every inherited rho_j to zero, so
the initial current preserves the V5 function in the existing arithmetic path.
Do not reset the previous 235+163 cells or regroup them from journal lesson IDs.

During correction, the existing input-only exact/recruitment circuit activates
the prepared event k **before the current byte is supplied**. Its one-hot actual
postsynaptic activation a_k=1 and its own held presynaptic activity h=z_k(x)
can drive a normalized local recognition rule:

`delta_rho_k = (1-s_k(x))*h/(h·h)`, with zero-energy skip.

Other rho_j remain unchanged. This is a local association between the actual
input and the cell that the input-only circuit activated. The clamp current one
is the cell's recognition activity, not an externally supplied correct owner.
No byte likelihood, downstream derivative, teacher episode, requested answer
span or correct program identity determines k. The current offered byte then
trains that same event's outgoing vector using the existing local output rule.
All fallible stage, candidate-view and pending checks precede joint publication.

For this candidate, the new recognition rule **replaces** the shared byte-outcome
update; it is not secretly added as a second objective on theta. Stop updating
theta, but retain its numerical contribution. Recognition means "this actual
context activated this event", not "this event's old output was useful". Thus
an exact repeated context with a changed target can preserve recognition while
the outgoing byte is corrected. Mixing unconditional recognition reinforcement
and opposing byte-utility clamps on the same synapse would need another explicit
objective; it is not part of this minimal candidate.

This answers the positive-credit question narrowly: an input-only recruited
cell can receive a positive recognition association immediately, without an
external address label. For a fixed context and fixed candidate set, changing
rho_k changes no existing j!=k current. It therefore avoids the particular
cross-owner shared-weight suppression seen in V5. It can still attract future
unrelated inputs to k, displace otherwise useful competitors, and inherit the
already suppressed theta prior until new associations compensate it.

### Cost and an important minimality limit

A literal private sparse f64 row costs `8*sum_j(K_j+14)` bytes, where K_j is
the owner's stored lexical column count. At the maximum K_j=10,992 this is
35,043,104 bytes for398 owners and45,080,576 bytes for512. Fixed view columns
are reused. Free scoring adds one owner-specific product/reduction per visited
row; teaching evaluates and updates the prepared row as well. This is not free
because only one teacher owner changes. Old theta's8,266,432 bytes remain.

However, under strictly exact input-only preparation, each owner is taught at
its own same context. Its h is fixed and every recognition update is parallel
to h. A full private vector would then be redundant: rho_j=g_j*h_j. A fixed
recognition cell can compute kappa_j(x)=h_j·z_j(x), feeding one plastic gain g_j
to its owner's score. At its own context kappa_j=h_j·h_j, so

`delta_g_j=(1-s_j)/kappa_j`

is exactly the same current clamp in real arithmetic. A unit-normalized local
one-input rule `(1-s_j)*kappa_j/kappa_j²` gives this expression. Only512 f64
gains (4,096 bytes) are required; h_j is fixed wiring derived from the actual
stored canonical input, not a learned lesson label. Caching its f64 values can
still cost up to roughly45 MB; recomputing them incurs explicit products and
normalization. No hidden prior forward is permitted to manufacture these rows.

This rank-one form exposes the limitation honestly: it calibrates recognition
of an input-defined prototype; it does **not** learn arbitrary field importance
or semantic roles. At a self-context the present difference-metadata block is
zero, so this recognition trace also supplies no positive learning of flag
meaning. It may remain a similarity-based episodic recognizer, and broad lexical
overlap can still cause the wrong group to win. More variable contexts per owner
would require a separately defined, target-free ownership mechanism, not an
unstated generalization of this update. No sequence latch is implied.

## Selection and unresolved relevance

Between these two, B addresses the observed **ownership of plastic credit** more
directly: an activated new event learns recognition and unrelated old currents
do not change from its update. A removes a potential sign effect but retains the
shared suppression mechanism and lack of new-owner recognition. B's minimal
rank-one realization is correspondingly modest; it is not a claim that a new
prototype gain establishes instruction understanding or unseen composition.

Preserve all398 existing owners and their outputs, keep old theta frozen, and
count every new recognition association in the selected B implementation. The current
163 answer symbols would supply163 additional internal recognition opportunities
on the same causal inputs, with no extra answer bytes or correction occasions.
Disabled correction advances neither gains nor assemblies; ordinary factual
association writes remain matched. No retrospective gain fitting on the old
398 owners, old lessons, or this exposed V5 stream is authorized.

Fewer answer corrections and more naturally observed source/language experience
must remain separate quantities. Input recognition alone does not acquire syntax
that was never locally taught, and unchanged raw-document writes do not count as
source next-symbol training. Capacity, transfer, prior retention and actual tool
recovery remain joint broader-workload outcomes. Neither candidate authorizes an
extra epoch, hidden corpus, target-owner oracle, gain sweep or isolated gate.

## Root implementation boundaries

Implement the rank-one recognition form, using the same input-only prepared
event and direct local one-input current rule. Retain all 398 event identities,
outgoing values, existing numerical priors and the whole 864-slot factual tape.
Recognition gains start at zero; do not fit them from old journals or replay
earlier corrections. Derived self-activity is fixed wiring from authoritative
receptive patterns and must have explicit construction, storage and work counts.
All owners remain eligible. The free system and learning system share these
same gains, populations, assembly bank and action wrapper.

Recognition and output preparation must bind the same causal context before
the current target. A target may correct outgoing bytes only after the
input-defined owner is prepared. Keep sequential private staging, disabled
proposals without advancement, same-instance actual action before correction,
and joint prevalidation before any live publication. Freeze the prior shared
theta contribution rather than adding another competing byte-outcome update.
Do not interpret this as an external correct-owner supervision signal.

Use ordinary numerical, lifecycle and continuation QA. Then integrate one new
broader persistent workload with changed instruction forms, structural
combinations, longer experience and fewer answer corrections. Keep the current
successes and failures as exposed regressions; do not train until they pass or
reuse V5 as a new transfer set. Natural source learning is a separate exposure
policy and is not silently added by this recognition change.

The current event capacity has only 114 free owners and factual capacity only
160 free slots. The broader integration must explicitly bound offered events and
new source bytes, or select lossless capacity growth with its full resource
accounting. No extra model run, checkpoint load or cost probe is released by this
document. The roadmap and a concrete continuation protocol will select that
single broader execution, without a separate skill/perfection gate.
