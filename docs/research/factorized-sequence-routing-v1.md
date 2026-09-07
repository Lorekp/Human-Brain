# Factorized local sequence routing

Status: **selected implementation contract**, 2026-09-07. Root reviewed the
representation, direct outcome rule, sparse bounds and joint publication and
selected implementation. No model, checkpoint,
task builder, feature evaluation or probe has been run for this document.
The [roadmap](roadmap-reading-to-coding.md) and
[completed v4 learning review](persistent-language-action-v4-learning-review.md)
own the objective and evidence. This is one shared broader successor, not a
routing benchmark or a prerequisite that earlier tasks become perfect.

## Decision and candidate ownership

Retain the existing assembly bank and add a **shared, locally plastic routing
output cell** over token/field coincidence activity. Replace fixed whole-frame
Hamming as the free ranking score. Keep the current input-only recruitment and
local byte/EOS output teaching, with new routing teaching on the same offered
events. Do not adopt a byte-likelihood posterior as an upstream error generator.

An occupied candidate is one existing **context/next-byte event assembly**.
Its index owns its receptive pattern and 257 outgoing currents. It is not an
episode, task label, semantic role, complete program or automatically identified
reusable sequence. Preserve all 235 v4 assemblies with those exact identities.
Sequence reuse means repeated selection of event cells under the learner's
causal prefix; grouping events by teacher episode is not part of inference.
Do not build a correct-episode table from saved journals.

This event ownership is the smallest continuation compatible with the actual
state. It does **not** solve sequence-level credit or guarantee coherent group
selection. A later selected event can disagree with an earlier event, and
arbitrary switching does not establish composition. Those failures stay in the
broader workload rather than being hidden by a supplied sequence/group label.

## Fixed token and field activity

Use a stateless, lossless byte tokenization: raw byte IDs 0–255, EOS 256 and BOS
257, plus overlapping ordered pairs of these IDs. This is deliberately a
byte/byte-pair vocabulary, not a pretrained word model or a synonym dictionary.
There is no evolving tokenizer table, fitted BPE corpus or unknown-token policy.
The original raw observation remains the authoritative input and action payload.

Decode the selected context assembly's fixed receptive layout into the same
fifteen structural fields used for the current public context:

1. request;
2. four path fields in canonical file order;
3. four corresponding complete file fields;
4. stdout and stderr;
5. signed return-code bytes;
6. causal output prefix;
7. preceding and following memory-context fields.

The two memory fields preserve their existing orientation/absorbing-EOS
convention. No fact or workspace excerpt is selected by the tokenizer. Public
absence flags distinguish absent fields from present empty fields. Prefix has
BOS followed only by earlier raw output bytes; its current target/EOS never
enters the input. Other fields retain their existing EOS/padding rules.

For each field, form three population channels:

- token counts over 258 IDs;
- adjacent-pair counts over `258²` IDs, with pair index `258*a+b`;
- token counts in eight relative-position bins, over `8*258` IDs.

A token at position `i` in a nonempty field of length `L` occupies bin
`min(7, floor(8*i/L))`. Pair tokens never cross field boundaries. Divide each
nonzero channel by its Euclidean norm; an absent/zero channel stays zero.
These fixed populations provide shift-tolerant identity and some local order.
They do not preserve arbitrary long-range order injectively. Negation, reordered
clauses or code can still collide in these summaries; raw access is retained,
but the router cannot exploit distinctions absent from its own activities.

Let `u_f,c(k)` be current activity and `v_j,f,c(k)` candidate activity. Form actual
fixed rate-coincidence products

`g_j,f,c,k = u_f,c(k) * v_j,f,c(k)`.

These are presynaptic activities, not inferred correctness or host token matches
used to shortlist candidates. Add a separate metadata block with twelve **signed
flag differences** `flag_current-flag_j`, signed position difference
`(t-t_j)/128` and its absolute value. Equal flags, including shared absence, and
equal positions produce zero, not a constant positive activity. Their
subtraction/absolute-value operations are explicit. The logical feature width is

`D = 15*(258 + 258² + 8*258) + 14 = 1,033,304`.

Normalize the lexical product block and metadata-difference block independently:
`L_j=g_lex/||g_lex||₂`, `M_j=g_meta/||g_meta||₂`, with a zero block left zero.
Use `z_j=[L_j,M_j]/sqrt(2)`, without renormalizing the concatenation. Its squared
energy is zero, one half or one in real arithmetic. A zero complete vector skips
the routing update; there is no hidden target-dependent replacement feature.

This adjustment is required by a concrete interference bound. The earlier
positive agreement proposal could contribute twelve units of energy from common
flags alone, even when lexical product energy approaches zero. Its normalized
activities could therefore become almost identical across unrelated owners.
Here shared flags contribute exactly zero, and an active metadata block cannot
exceed one half of the total configured energy. There is no plastic bias term.
Lexical channels still overlap: their product energy is at most45 before
normalization, and two owners can have identical lexical summaries. If their
metadata differences also agree, their `z` vectors remain identical and a shared
update cannot improve their relative ranking. The change removes the identified
constant-flag dominance; it is not a theorem of language discrimination or a
claim that v3-style interference has disappeared.

### Sparse activity is a simulation choice, not a host retrieval policy

Store each candidate's active population columns as int32 and normalized channel
values as float32, with unique, sorted columns. Counts are exact nonnegative
integers before channel normalization; no field has more than513 symbol slots.
Validate each field/channel's counts and normalization before installing a view.
Build the current float32 population once per actual prefix. Cast gathered
activities to float64 for coincidence products, block norms, learned products
and routing potentials. The real-arithmetic bounds above allow finite-rounding
tolerance; no comparison silently rounds the final score back to float32.
For **every occupied candidate**, gather the current activity at its own stored
columns, multiply by its stored values, form the additional cells, normalize and
evaluate the learned output products. There is no inverted-index shortlist,
host nearest-neighbor function or early pruning. All candidates reach neuronal
competition, with lower-index ties. Count actual gathers, products, reductions,
normalizations and threshold activity, separately from logical inactive cells.

The shared float64 routing vector has `D` coefficients: 8,266,432 bytes. A dense
float32 current population would require 4,133,216 bytes. Candidate sparse
indices/values, temporary rows, staged weights and competition add costs; these
figures are not a run/RSS projection. At most3664 nonpadding token positions
per owner contribute at most10992 sparse entries before duplicate collapse.
Int32 indices plus float32 values therefore need at most45,023,232 bytes for512
owners, or20,664,960 for235, excluding offsets, metadata, stages and validation
scratch. A dense `candidate × D` bank must not be
materialized. Sparse storage must use source-defined numeric arrays and fixed
column routing, not a Python semantic matching engine. These bounds select
bounded implementation, not measured runtime admission.

## Forward selection and initialization

For a candidate, define a fixed baseline score as the arithmetic mean of its
45 within-field channel inner products, including zero absent channels:

`b_j = (1/45) * sum_f,c <u_f,c, v_j,f,c>`, with `0 <= b_j <= 1`.

The new shared scalar output circuit computes

`s_j = b_j + theta · z_j`; `theta` starts at zero.

Actual simulated **float64** pairwise threshold competition selects
`a = argmax_j s_j`, with the existing lower-index tie orientation. This requires
an explicit new float64 competition seam: the existing float32-only
`compete512`/memory competition functions cannot accept these scores unchanged.
Comparator currents and winner accumulations stay float64; every intermediate
must be finite. Count the actual all-owner pairs/thresholds and reject an invalid
current rather than silently quantizing or replacing the winner.
Read exactly candidate `a`'s 257 currents; choose its byte/EOS
by the current output rule. An empty bank retains the inherited fallback.
Do not mix every candidate's output currents into the free response.

This retains all numerical prior values but intentionally changes free routing.
It is not v4 functional parity. The fixed baseline is an explicit token/field
prior, not a measured improvement. Equal channel weighting is selected from the
normalization convention, not tuned on exposed v4 answers. The shared `theta`
learns which token coincidences, fields and local order are useful across event
owners; there is no new independent dense classifier per episode.

## Direct local teaching and the reverse-credit audit

First preserve the actual pending response and, for editing, its real execution.
The subsequent teacher call consumes its already offered raw bytes plus EOS
once. At teacher position `t`, construct the context from `teacher[:t]`, run the
same candidate circuit with the current private stage and retain its selected
candidate `a`, score `s_a`, input `z_a` and emitted byte `y_hat`.
Before exposing `y`, also prepare the existing target-independent exact-context
selection or novel recruitment from that same input and unchanged bank stage.
This is a **separate** exact-existence circuit. Retain the old `prepare_context`
full Hamming evaluation during teaching and count its gates/competition in
addition to token routing. A learned router's nonexact winner cannot prove that
no exact stored pattern exists. The old nearest winner has no authority over
free output or the routing outcome signal; it serves only the unchanged exact
reuse/recruit/exhaustion preparation. There is no old Hamming pass during free
token routing. A later exact-only optimization is not selected here.

The current offered byte `y` supplies a **direct outcome teaching signal**:

`d = +1 if y_hat == y, otherwise -1`.

If a candidate exists and `||z_a||²>0`, prepare the selected routing output
cell's own normalized potential update:

`theta_new = theta + (d - s_a) * z_a / ||z_a||²`.

The fixed baseline is held. In real arithmetic, the same selected cell's score
becomes exactly `d`. For its fixed single-event squared potential error, the
unit step reaches the minimum. This says nothing about the next winner,
sequence likelihood, other candidates or later tasks. Shared features cause
cross-candidate interference; finite arithmetic, weight growth and repeated
conflicting outcomes require explicit checks. There is no unproposed clipping,
temperature, learning-rate sweep or exploration/retry policy.
With the two-block normalization, a nonzero update has
`||delta_theta|| <= sqrt(2)*abs(d-s_a)` in real arithmetic; this is a one-event
bound, not a uniform bound over an arbitrarily long inconsistent stream.

The teaching signal is derived only from the candidate's **already emitted
byte** and the current offered byte. It is not a desired memory address, chosen
prototype, correct episode or gradient through the winner/output circuit.
It describes this teacher-prefix forward, not the earlier free task's success
or an invented correction of that free prefix.
No error is propagated into token construction, fixed coincidence cells,
receptive patterns, B/C/U, codec or the executor. A scalar output-neuron error
times that output cell's own presynaptic activity is the complete plastic rule.
The same physical routing weights score all candidates; only the activity of
the selected evaluation supplies this update.

The prepared assembly's own outgoing current clamp receives `y`, as in v4.
Compute the routing signal from the pre-clamp
emission, never from a newly recruited cell after it was taught the answer.
Stage routing and assembly proposals sequentially across teacher prefixes;
validate all state before one atomic publication. Disabled teaching advances
neither private state, though its offers and computed proposals are counted.
The prior B/C/U/codec remain held in this draft.

The transaction validates router weights, the assembly stage, all derived sparse
views, fixed metadata, pending identity and preallocated feedback **before the
first live copy**. Commit is one reviewed copy-only section over those validated
objects. Calling a fallible router `publish` and then a fallible assembly
`publish` is not atomic. No new normalization, casting, allocation or validation
may occur between the first and last live publication. Disabled proposals leave
private routing weights, assemblies and derived views unchanged alike.

### Why byte-posterior responsibility is not the selected rule

For `p=softmax(s)`, candidate byte likelihoods `q_j(y)` and mixture likelihood
`m(y)=sum_j p_j*q_j(y)`, the familiar responsibility is
`r_j=p_j*q_j(y)/m(y)`. Algebraically,

`d[-log m(y)]/d s_j = p_j-r_j`.

Manually sending that mixture-derived error upstream into routing weights is
reverse credit through a downstream computation. Calling it a responsibility,
omitting autograd or multiplying by local input afterward does not remove that
issue. **This draft does not implement it.** There is also no derivative of the
hard free winner or its byte selection in the selected outcome-clamp rule.

This stricter audit does not rewrite the contracts or scores of earlier frozen
experiments. Their negative evidence is relevant: v2's more concentrated latent
responsibilities and staged updates did not yield correct free answers. A
plausible local-looking algebra was not sufficient evidence of useful credit.

## Unresolved credit and generalization limits

The selected outcome rule is legal direct supervision, but may be insufficient.
A space, quote or EOS emitted by the wrong sequence earns the same positive
signal as the same byte from a relevant sequence. A wrong selected cell is
suppressed without being told which unselected cell would be useful. There is
no counterfactual evaluation, exploration guarantee or sequence-level reward
in this draft. It may simply learn local byte preferences or repeatedly switch
among irrelevant events. These are unresolved assumptions, not solved by the
unit-step potential bound.

Token/field factorization makes exact byte-position agreement unnecessary and
permits learned relevance of recurring byte patterns. It does not learn synonym
meaning automatically. Event ownership and byte-only feedback still fall short
of an identified reusable program/sequence group. Whole-sequence binding,
composition of multiple facts, prefix drift and action-span learning remain
capability questions. Root should reject or amend this draft if a stronger
ownership signal is required; it must not be filled in with a hidden oracle.

## API and state-continuation proposal

Selected module boundaries:

```text
FactorizedContext(populations, flags, position, raw_context_sha256)
FactorizedSequenceRouter(input_width=D)
  build_candidate_view(raw_receptive_pattern) -> immutable sparse numeric view
  evaluate(current_context, candidate_views, stage=None, work=None)
    -> RoutingDecision(scores, selected_index, selected_score, selected_activity)
  prepare_outcome(decision, emitted_byte, offered_byte,
                  stage, learning_enabled, work=None) -> RoutingUpdate
  begin_stage() / validate_stage()
  commit_validated(stage)  # copy-only, within the learner's joint commit

FactorizedSequenceMemoryLearner
  predict_with_receipt(raw_observation, tape, max_bytes)
  teach(raw_observation, target_bytes, tape, prediction=exact_pending,
        learning_enabled=True)
  acknowledge(exact_pending, tape)
```

Keep the existing same-instance action wrapper, pending observation/model/tape/
trace binding and execute-before-teach lifecycle. STOP and no-teacher resolution
produce no routing or payload labels. The routing decision must be sealed
against mutation; a current target is not accepted by `evaluate` or candidate
view construction. A teacher step stores selected pre-clamp currents, emitted
and offered bytes, scalar `d`, routing potential/error/norm, update accounting
and assembly recruitment/output receipt. Do not journal every million-wide row.

Continue the authenticated v4 online model, its 235 assembly patterns/outgoing
vectors, action weights, experience and whole 505-slot tape into two independent
routes. Build token views only by deterministic decoding of each actual saved
receptive pattern, never by searching its lesson ID or rereading the correct
task solution. New `theta=0` and derived fixed views are explicit added state;
strict loading checks their schema, indices, normalization and source identity
before publication. Fixed view construction is counted; it is not a prediction
or an uncharged replay of lessons. New recruited views derive from their actual
pre-target input and commit atomically with their cells.

## Broader continuation and exposure boundaries

Carson's [ignored workload proposal](../../.pytest-tmp/persistent-language-action-v5-workload-opportunities.md)
contains 64 episodes: 24 interleaved opportunities, sixteen unseen combinations,
eight three-attempt recoveries and sixteen exposed regressions. Eight fixed
corrections have a proposed ceiling of 200 byte/EOS events. These give at most
200 routing outcome offers and 200 assembly-output offers per neuronal route,
using the **same** 200 externally offered symbols; report both internal uses.
No second fit pass, extra answer or target-dependent retry is authorized.

The proposal's six complete public fact/program documents require explicit
1024-slot integration to grow beyond the current 505/512 occupancy. Its
[memory contract](../../.pytest-tmp/persistent-language-action-v5-memory-growth-contract.md)
preserves old rows and adjacency, with all occupied slots eligible and no
task shortlist. Capacity promotion and ordinary raw-document ingestion are
separate from routing/output corrections. Both matched routes may receive the
same association writes while only online receives correction plasticity;
label that control accurately. Causal source-language learning, if later added,
would be additional byte/EOS exposure, not free observation.

Keep exact delayed examples as labeled retention diagnostics alongside changed
wording, changed inputs and genuine combinations. None is a perfection gate.
The sparse representation, float64 competition and joint publication semantics
above close the implementation contract; the outcome rule's capability remains
an empirical question, not an additional perfection prerequisite. The generic
1024-capacity dependencies have ordinary mechanics QA, while integration into
this new reader still needs source review. Ordinary source/math, atomicity and raw-input mechanics
checks precede a broader monitored continuation; no model run is selected here.
