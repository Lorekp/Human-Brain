# Value-neuron credit after the query-credit return

2026-09-06. Bounded source/mathematical review; no new fit, endpoint query,
teacher offer, forward evaluation or gradient oracle. Recommendation: one
integrated comparison replacing only the value-map error broadcast with the
actual read-neuron error. This is a candidate test, not automatic adoption or
evidence that the remaining writing problem is solved.

## What the completed evidence establishes

The [query-credit protocol](../../experiments/r4_integrated/local_memory_query_credit_v1.md)
at `f195a36323a4450b40e454e6a0d0df11ba126626` completes with tagged,
aggregate and writer-frozen each solving 5/8 fit, 1/4 new-value, 1/4 delayed and
0/4 interference actions. The matched old rule solves 2/8, 1/4, 1/4, 0/4.
Tagged gains the `aa`, `ab`, `bb` fit cases and loses no previously correct case.
Its two assignment swaps still produce `ca/ca` and `dd/dd`, respectively.
Writer-frozen differs on some wrong answers, including `cd/bd` for the second
swap pair, but has identical correctness on all twenty cases. Aggregate matches
tagged's twenty free outputs exactly.

The authenticated saved-only description is
`.pytest-tmp/local-memory-query-credit-v1-description.json`, SHA256
`77d181f05806a291e7a37c966e20ec05e9d572c2abf9174e49eed657f50a58d7`.
Its old-rule comparison checks 216 initial/final tensor entries and metadata,
all 1,024 lesson receipts and twenty final output receipts: all exactly match
the earlier `a94b466` tagged run after excluding arm names and durations.
The full audit is SHA256
`20d1dd19280f1a51fe44e8c40105dcb8717b6eacf81cb2b4aa210a0dc454af7b`.
The [result record](../../experiments/r4_integrated/local_memory_query_credit_v1_result.md)
owns the final resource and custody ledger.

This supports an effect of the repaired query-credit route on fitted behavior.
It does not establish useful learned writing. In the last 128 acquisition
episodes, first-exact counts are 32 tagged, 24 old-rule tagged, 35 writer-frozen
and 13 writes-disabled; mean teacher half-squared losses are respectively
0.234517, 0.251199, 0.232200 and 0.257694. Late improvement prevents a claim of
converged failure, but does not justify an automatic exposure extension. These
are exposed development examples, not fresh adjudication.

## One exact conditional link already present in the traces

The current [memory source](../../src/humanbrain/plastic_memory.py) implements
the delta overwrite, using column keys and row-oriented value memory:

```text
h_t in R^H, k_t in R^K, v_t = tanh(V h_t + b_v) in R^D
beta_t in [0,1], M_t in R^(D x K)
P_t = I - beta_t k_t k_t^T
M_t = M_(t-1) P_t + beta_t v_t k_t^T
r = M_T q, q in R^K
```

Fix the initial memory and the controller/key/strength parameters during one
ingestion interval. In this architecture the writer cell does not receive value
map outputs or fast reads, so its `h_t`, the keys and strengths do not depend on
`V` or `b_v`. Holding those quantities fixed is therefore the actual partial
derivative with respect to the value map, not an omitted indirect value-map path.

For one value-map synapse `V_ij`, memory rows remain independent:

```text
d M_(l,k) / d V_ij = 1[l=i] T_(i,j,k)
e_(t,i,j) = (1 - v_(t,i)^2) h_(t,j)
T_(t,i,j,:) = T_(t-1,i,j,:) P_t + beta_t e_(t,i,j) k_t^T
```

The bias has `e_(t,i) = 1 - v_(t,i)^2`. Differentiating the memory recurrence
directly proves this by induction, starting from zero sensitivity of a fixed
initial memory. No key normalization derivative is needed for this partial
derivative; the value map does not determine the key. The formula also holds
for clamped, nonunit keys. Inactive writes have zero beta and preserve the trace.

`write_local` already implements exactly this *real-arithmetic recurrence* for
the `value_weight` and `value_bias` tagged arrays: it subtracts the old
`T dot k`, adds the local tanh derivative times presynaptic activity, and writes
that correction along the actual key. Its shape is `[batch, D, H, K]` for weights
and `[batch, D, K]` for biases. The memory rows' independent dependence avoids
storing a redundant full Jacobian over every memory row for every parameter.

For a fixed final query key and a neuron error `g_r = d loss / d r`, the matching
parameter derivative is consequently

```text
d loss / d V_ij = g_(r,i) sum_k T_(i,j,k) q_k
```

The current `apply_local_feedback` already performs that query contraction and
the final learning-signal times eligibility product. The missing link is in
[the joined learner](../../src/humanbrain/local_memory_learner.py): it supplies
`value_feedback @ mean_context_signal`, a fixed random 64-by-96 projection,
instead of `mean_context_signal[:, query_size:]`, the existing 64 read-neuron
errors. That projection mixes query-activity error and unrelated read-neuron
errors before the row-specific trace. It is not the above conditional gradient.
It might nevertheless learn heuristically; its failure is empirical, not an
impossibility theorem about random feedback.

The proposed replacement is the direct read-error slice. **Do not multiply by
another tanh derivative, beta, `M.T`, learning rate, or output-count divisor.**
Tanh gain, beta and overwrite history are already in the trace; `M.T` belongs to
the separate key/query error route. Preserve the current mean over three output
teacher offers and the final `-eta * signal * eligibility` update sign.

The delta rule's fast-memory use has primary precedent in
[Schlag, Irie and Schmidhuber, ICML 2021](https://proceedings.mlr.press/v139/schlag21a.html).
Their learned controllers use gradient-based training; those experiments do not
qualify this locally trained writer. The distinction between local eligibility
and its neuronal teaching signal is also central to
[Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y).
The value-only derivation above is specific to the present source and follows by
differentiation of its recurrence; neither paper proves convergence here.

## Locality, numerical and history limits

Each value synapse uses its own existing key-tagged state, local presynaptic
activity/tanh gain during writing, declared key/beta/query activities, and its
corresponding read neuron's error at teaching. Those are explicit neuronal
signals; no host per-synapse error tensor, reverse-time history, external solver,
autograd or optimizer is needed. The stored key-vector eligibility still costs
O(number of writer synapses times key dimension). Mathematical conditional
exactness does not establish biological implementation or cheap large-scale
learning.

Only tagged value traces have this exactness claim. Aggregate traces discard the
key-dependent overwrite operator and remain an approximate control. Controller,
key and strength traces/signals remain approximate; simultaneous updates of those
parameters are not certified descent by the value partial derivative. The query
repair and local recurrent self-path approximations remain unchanged.

Keep slow value weights fixed throughout the complete ingestion interval,
including both documents and their terminators. Purging writer activity does not
retire the tagged trace; resetting it between documents would invalidate the
claim. Retiring traces after teaching deliberately truncates sensitivity to
earlier learning history. If later writes start from that retained memory, the
new interval treats it as fixed initial state. Do not call this an exact
derivative through arbitrary continual weight changes or previous lessons.

The recurrence proof is in real arithmetic. Production uses float32 represented
activities and tags, rounded tanh values and multiply/reduction/add operations;
`1 - v^2` is the declared local derivative estimate at those represented values.
It is not a derivative of the discontinuous floating implementation or a bitwise
autodiff equivalence claim. Keep the existing checked float64 output-signal sum,
mean and single upstream cast, all finite proposal checks and publication order.
The loss signal describes the local teacher computation, not derivatives through
the learner's previous output-weight updates. The lesson remains nontransactional
across its already separate output/memory/query publication groups.

Removing the obsolete `value_feedback` buffer has one matching trap: it is a
middle draw of the private generator. Preserve the subsequent `strength_feedback`
initial values exactly, for example by consuming the same private draw without
retaining it. Do not accidentally change a second learning signal. The old-rule
reference alone may retain the unused-by-candidate buffer under its frozen schema;
production should have one rule and no compatibility branch.

## Competing explanation and decisive comparison

The strongest alternative remains inadequate source-to-key binding and
interference. Different values cannot be recovered independently if their
effective write/read directions do not distinguish assignments sufficiently.
Earlier localization showed surviving memory/read differences, distinct query
keys, common-direction swap responses and strong second-document influence.
The query repair improves fitted answers without a novel-value or writer-trained
advantage. A correct value derivative cannot by itself make the controller write
the right fact at a separable address, prevent overwrite, or prove two-byte
composition from arbitrary continuous memory contexts. Those are remaining
explanations, not a competing mechanism queue to implement now.

Nevertheless, one direct-value comparison is justified: it isolates a known
algebraic mismatch in an existing pathway, retains the representation and exposure,
and can fail decisively on the actual workload. Do not add another gifted-context
diagnostic, tune epochs/rates/seeds, or change key/controller mechanisms alongside it.

Use five candidate controls plus one matched query-repaired/current-random-value
tagged reference from `f195a36`. Replace the previous historical comparison arm
with this immediate predecessor; do not accumulate a seventh arm. Preserve fresh
initial output weights, all common initial tensors, seed 733, 1,024 episodes,
rate 0.01, the twelve-memory roster, two-byte/EOS cap, source-free worker,
same-question swaps, read-only delays, unrelated writes and real Python actions.
Writer-frozen/frozen/writes-disabled must retain their prior forward/teaching
trajectories; their value parameter path cannot consume the changed signal.
The aggregate arm receives the same direct error with its declared approximate
trace. Report its outcome separately from tagged exact-conditional eligibility.

Before a fit, require small authored scalar recurrence/gradient checks for the
value-only path, including bias, overwrite, masking, saturation, frozen controls
and old/new forward parity. These are ordinary QA, not endpoint gradient probes
or teaching signals. Then freeze a fresh complete cost sample: 24 first attempts,
72 teacher offers, 60 enabled output opportunities, 120 actions, two motor
demonstrations and six capped predictions. Recompute the original twice-learning
plus twice-other-time plus 30-second projection, with 900-second/2,048-MiB bounds.
Do not reuse the preceding 853.567-second admission; stop if the new cost fails.

If admitted, run the single fixed integrated acquisition and its saved-only audit.
The full useful criterion remains 8/8 fit and 4/4 new-value, delayed and
interference actions, with strict new-value superiority over writer-frozen,
frozen, writes-disabled and the immediate-predecessor tagged rule. Record tagged
versus aggregate separately. A weaker fit/loss gain is descriptive development
progress, not adoption or an integrated pass. If meaningful writing and swap
behavior still fail, close this one comparison and reassess the binding/
interference explanation; do not automatically add more experience or another
broadcast tweak. A useful result returns directly to broader raw-document/action
learning and the still-explicit host-motor boundary.
