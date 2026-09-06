# Neuronal operation intent: direct whole-request input

Status: root-selected source implementation, 2026-09-06. Implement the single
reviewed branch below in new versioned source, keeping frozen modules unchanged.
Freeze and review the exact ordinary QA scope before executing it. No standalone
fit, checkpoint load, genuine endpoint replay or acquisition is authorized here.
No implementation, model/checkpoint load, feature reconstruction, alternative
decoding, action, test or experiment was performed. Only finalized JSON and
source were read; saved two-neuron logits were subtracted. Mem0 was searched as
external development context and supplied no evidence or mechanism premise.
The graph returned no operation_features symbol, so its current source was read.

## What actually failed

The completed independent audit passes custody/scientific validity, while the
integrated adoption criterion fails. On the same fresh cohort the coincidence
learner solves 4/32 and the graded competitive reference 5/32. Both retain 14/16
actions and twenty old reader answers. The candidate has correct provisional
payload/EOS on all 28 edit-required cases, correct canonical location on 27,
and both correct on 27. It nevertheless chooses STOP on 23 edit requests.
All four explicit KEEP requests incorrectly choose SPLICE. The reference has
24/28 correct payloads, 27/28 correct locations, 23/28 jointly correct, 22 false
STOPs and four missed STOPs. These are overlapping error counts, not partitions.

Sartre's independent address extraction identifies all 27 nonempty candidate
literal starts as unique rank-one selections. All eight D and eight V cases
have the correct complete provisional edit but choose STOP. The reference's
four continuation payload failures retain the rank-two wrong-record problem.
The remaining candidate location error is O2-0; its retention misses are the
shared T01 payload and T08 location. These residual issues do not explain the
dominant fresh failure. No counterfactual edit was executed or counted correct.

Saved operation margin = SPLICE logit minus STOP logit:

| Group | Coincidence | Graded reference |
| --- | --- | --- |
| False STOP count | 23 | 22 |
| False STOP median margin | -0.303432 | -0.339754 |
| False STOP margin range | -0.588932 to -0.017142 | -0.542500 to -0.071442 |
| Four required STOP margins | +0.033097 to +1.315896 | +0.044711 to +1.340931 |

These saved margins describe errors in both operation classes. This is also
not merely an exact tie or a universal numerical saturation. The saved evidence
does not identify which feature block contributes each logit: that would require
weights/features beyond this saved-only scope. No such attribution is claimed.

## Explanation ranking

1. **Shared operation representation/generalization is the leading seam.**
   The source has 4244 operation inputs: query-by-local-context, query-by-path,
   span-by-replacement and three scalar quantities. It has no standalone request
   input. Query rows are EOS followed by reversed raw bytes, so prefix instruction
   positions move when the suffix length changes. Saved TRAIN request lengths
   are 15/16 bytes; fresh requests are 16/17. This concrete shift supports the
   representation hypothesis but does not prove it independently of path,
   workspace and corrected-versus-free context changes.
2. **A shared location failure is not the largest current gap.** The same action
   frontend achieves canonical location on 27/28 edits in both arms. Altering
   start/end feature geometry would spend exposure on a smaller observed error.
3. **Reader initial choice / hard commitment is not the candidate's current
   fresh bottleneck.** Its provisional payloads are 28/28. Commitment can still
   propagate a wrong rank into a wrong payload (reference and retention examples),
   but changing it does not target the 23 correct-payload false STOPs.
4. **Inherited numerical address priors have changed feature meaning.** That
   remains an explicit interpretation limit: final correct payloads do not prove
   representation-neutral initialization or universal transfer. The matched
   conditional payload gain is useful evidence, not successful integrated
   adoption. Different reader trajectories also change actual operation teaching
   contexts; the arms are not a frozen operation-head causal experiment.

The prior v2 20/32 versus present 4/32 difference is not a matched causal
subtraction: the fresh cohort changed. No new teacher rule or reader sweep is
justified by this outcome.

## The selected source candidate

Add a **parallel beginning-aligned whole-request input only to the operation
output population**. Encode the unchanged raw request, then its actual EOS, in
32 positions of the existing generic nine-bit dual rails (18 Boolean rails per
position); leave unused positions zero. Every raw byte is included. No host
tokenization, verb flag, key extraction, role parser or task-dependent mask is
introduced. This is a fixed positional representation prior, not learned
language understanding.

Let R be the flattened 576 rail activities. With L raw bytes, its squared norm
is 9(L+1), including EOS. Define q = sqrt(32) R / sqrt(9(L+1)). Preserve the old
operation feature vector phi and its separate energy-32 normalization exactly.
New logits are W phi + U q + b, with U[2,576] initialized exactly zero as an
explicit fresh prior. This preserves old logits at initialization in real
arithmetic while giving each operation neuron direct access to all request
positions. Preserve the actual old float32 product/reduction/bias expression,
then add the separately reduced request current. Extending one concatenated
reduction with zero products can change rounding and would not establish exact
initial parity. Count the additional 1150 reductions and two current additions.
Do not concatenate and renormalize the whole bank, which would
silently rescale the old branch. Start/end features, reader, teacher schedule,
pending action custody and actual execution remain the same.

Teacher operation error remains e = softmax(logits) - onehot(target). Its local
synapses receive delta W = -.01 e phi^T, delta U = -.01 e q^T and delta b = -.01 e.
No error passes through the rail encoder, normalization, payload, start/end
selection or any earlier population. SPLICE still uses its corrected context
after actual execution; STOP still teaches only the operation on its actual
proposal. The request-only branch is identical in free and corrective phases
even when the provisional span or payload was wrong. No extra reader teacher,
oracle operation route, or separately trained executor is added.

The actual source rule is an affine-softmax cross-entropy local update, **not a
squared-error LMS update**. Old augmented input energy is 33; new energy is 65.
For a held input, conservative real-arithmetic NLL smoothness L_s <= 65 implies
delta NLL <= -.01(1-.01*65/2)||G||^2 = -.00675||G||^2 for the full direct parameter
gradient G. The binary softmax covariance bound can tighten this; it is not
needed here. Float32 casts/rounding and subsequent changing inputs are outside
this real-arithmetic one-update bound. No empirical stability claim follows.

## Price, decisive QA and limits

The addition is 576 forward input currents, 1152 learned weights / 4608 float32
bytes per model, plus 1152 score products and 1152 local update products per
operation phase. Encoding, rail validation, norm/scaling, reductions, transient
update tensors, hashing and checkpoint schema must be counted in a selected
implementation; current v3 resource receipts cannot admit a successor. Existing
zero-initialized U does not entail inherited tensor restoration. Action controls
must gate U with the rest of the operation synapses, including all frozen modes.

Meaningful ordinary qualification would establish (a) raw/EOS/empty-padding truth
and unchanged prefix positions under different suffix lengths; (b) zero-U forward
parity and scalar direct output update with both branches, including controls
and publication atomicity; and (c) one same-instance real action-before-teacher
receipt with the identical raw request on free and corrected paths. These are
proposed checks, not executed fixtures or a new acquisition authorization.

The reusable scientific question is whether direct positional request inputs
learn operation intent across changed suffixes and workspaces, while retaining
context-dependent decisions and reader/location behavior. It needs a newly
frozen integrated comparison, actual first attempts and matched controls; the
closed v3 endpoints cannot choose weights, a gain, rate, seed or added rounds.
Beginning alignment favors prefix instructions; a linear rail branch alone
does not solve arbitrary word conjunctions, paraphrases, negation, requests
with instruction words elsewhere, or deciding that an unrequested edit is
already unnecessary. Those limitations must remain explicit, not hidden by a
verb-specific feature or bypass. Keep the unchanged contextual operation head available as the matched reference. This is one candidate, not a claim that the
new branch would have repaired the observed outputs.

## Evidence custody

The companion saved-only JSON
`.pytest-tmp/neuronal-splice-online-v3-operation-saved-review.json` has SHA256
`8d6e2feb9f3a80c32e7b5d6f20e35e0c9a5308451fc14e243e4f03a7fd24688b`.
It binds full SHA256 values for the finalized independent audit, frozen roster,
actual endpoint JSONL, current action learner source and selected comparison
plan, and retains all 64 learned/reference fresh saved operation-logit pairs.
Sartre owns the separate initial-address extraction; its findings above were
communicated from that saved-only analysis rather than recomputed here.
