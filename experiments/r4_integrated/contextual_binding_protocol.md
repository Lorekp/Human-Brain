# Candidate-relative binding in the executed learning loop

Public development, declared before model outputs. Baseline `9c460ea` preserves
the [conditioning result](contextual_conditioned_result.md). This trial follows
the [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md);
it neither opens protected evaluations nor changes completed contracts.

## Failure, hypothesis and fixed representation

Gain 16 retains 11/16 taught edits under raw/mapped readouts but solves 0/8 fresh
tasks. Fresh answers favor old values or a neighboring entity. The hypothesis
is that reusable matching geometry can bind an unseen query name and requested
property to a candidate occurrence without learning the name or answer itself.

Compare **GAIN16** with **BINDING**, both from zero evidence synapses. BINDING
appends exactly 84 features to the existing 1,537 gain-16 coordinates. Keep the
pinned encoder, all generic 1–3-word candidates in 48-word windows, native
pairwise rule at learning rate 1, replay policy, fixed learned motor and source
custody. No entity recognition, relation parser, answer-head model or semantic
candidate filter is introduced. The new block has fixed gain 16, chosen before
outputs; it is not tuned on this cohort.

Let `i,k` index query words and `j,l` source words in the candidate's window.
Use existing byte word offsets and unit-normalized encoder span vectors:

```text
E[i,j] = 1 if casefold(query_word[i]) == casefold(source_word[j]), else 0
S[i,j] = max(0, dot(query_word_vector[i], source_word_vector[j]))
```

The six kernels are left/right exponential distance decay at scales 2, 8 and
32 words, ordered left 2/8/32 then right 2/8/32. Left is `exp(-abs(r)/s)` for
`r <= 0`, otherwise zero; right uses
`r >= 0`. Zero belongs to both. Distance from a candidate interval is negative
to its left, zero inside, positive to its right. All source interactions remain
within the same generic window; no selected document is supplied by an evaluator.

For query length `Q`, define:

```text
A[i,a,l] = max_j E[i,j] * K[a](j-l)
R[i,Y,l] = sum_{k != i} Y[k,l] / max(1,Q-1), Y in {E,S}
F[a,b,Y,c] = mean_i max_l A[i,a,l] * R[i,Y,l] * K[b](l-c)
H[b,Y,c] = mean_i max_j Y[i,j] * K[b](j-c)
```

Append 72 `F` coordinates in `(a,b,Y)` order (E before S), then 12 `H` coordinates
in `(b,Y)` order, multiplied by 16. Empty-query coordinates are zero. Reuse vectors from
one-word source candidates, encode query words once, and count retained index
maps and lexical payloads. Bind the representation into storage/checkpoint
identity; replay must reconstruct the exact original occurrence features.
Soft proximity is not grammatical role understanding or coreference; paired
entities and reversed order directly test this limitation.

## Fixed teaching and evaluation

The [cases](contextual_binding_cases.json) declare four paired-entity teaching
documents: eight entities, two properties each and sixteen unique answer spans.
Property/entity ordering is counterbalanced. Both arms ingest the same documents
and use the same 48-encounter schedule. Follow the
[executed correction contract](contextual_conditioned_protocol.md): at most 32
actual FAIL/teacher-PASS pairs, at most four distinct older certified pairs after
each correction and 128 replay offers overall. Successful attempts create no
invented negative. After discrete neuronal STOP, execute the raw learned-score
proposal, or external cosine if raw also STOPs. Teacher/fallback success is
separate from first-attempt neuronal success. Report differing on-policy ledgers.

After teaching, ingest six fresh documents once, remove live source inputs and
execute eight supported requests across four unseen entities, two unsupported
probes, and sixteen old requests without updates or retries. Fresh strata include
simple prose, neighboring entities, repeated values, and paraphrase/order change.
These are self-authored development fixtures, not independent exam evidence.

Each updater has a same-seed frozen evidence learner with identical ingestion and
feedback. Evaluate discrete and interpolated neuronal readouts for both updating
and frozen learners, plus raw/mapped learned-score and external-cosine references.
For BINDING, execute two additional read-only raw ablations: zero the new block;
or replace every positional kernel by one, collapsing its evidence to window
level. They are inference ablations of the trained weights, not independently
trained baselines. Reject ablation decisions for feedback or replay.

All policies share the fixed learned occurrence motor. These controls isolate
evidence learning/selection; they do not remove all neurons from the entire
system. Retained raw documents/vectors, encoder weights and replay ledger are
external resources. Source absence from requests is not semantic compression.
Keep original KS2 Q6a/b untuned, with official-rubric human scoring pending for
noncanonical answers. Do not broaden exam exposure during this comparison.

## Separate readout refinement

The previous run loses ten correct retained selections to same-step collisions.
Test one explicitly named linear endpoint-interpolation policy, preserving the
existing discrete policy and using it unchanged during all teaching. For cells
crossing in the earliest sampled step `k`, let `Fprev <= 0 < Fnow` be signed
voltage-minus-threshold distances. Estimate `alpha=-Fprev/(Fnow-Fprev)` and
`time=(k+alpha)*dt`; at `k=0`, the previous distance is the resting value.
Select only a unique earliest finite time. Times equal to or immediately adjacent
to the minimum in float64 are indistinguishable and remain STOP; invalid brackets
fail. The tonic STOP cell participates by the
same rule. Record timing identity, estimated time and coarse/refined tie counts.

This is a new numerical approximation to crossings between sampled endpoints,
not exact parity with the old discrete spike flags or the continuous ODE.
Validate fixed unequal same-bin/permuted currents, true ties, STOP and boundary
cases; compare a small fixed current grid with direct integration at `dt/16`
and `dt/64` over the same horizon. Report error/convergence without timing tuning.
In integration compare readouts at identical trained checkpoints. Recovering
correct existing rankings is readout progress, not evidence of improved ranking.

## Decision and budget

Representation progress requires at least 6/8 fresh and 12/16 retained executed
raw-reference successes, both properties for three fresh entities, and at least
three additional fresh successes over GAIN16. Report ablation effects and each
stratum. The neuronal milestone still requires 6/8 fresh and 12/16 retained
neuronal executions; reference-only gains cannot pass it. Report unsupported
behavior, teacher repairs, learning gain and retention separately.

Budget: one feature design/scale and one readout design, two hours implementation
and diagnosis each, one fixed integrated comparison. No extra epochs, parameter
sweep or feature changes after transfer outputs. A failed comparison triggers
analysis of the observed failure and alternatives, not an automatic extension.
Record source/protocol/case and NPZ hashes, actual checkout-native identity,
execution receipts, native work, timings, retained bytes and parent peak memory.
Use compact LF JSON and refuse existing output paths. Preserve completed results.
