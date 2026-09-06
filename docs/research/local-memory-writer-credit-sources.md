# Value-map writer credit: primary sources and conditional derivation

2026-09-06. Bounded research requested after the closed query-credit comparison.
Three primary papers were checked online; repository source was inspected at
`f195a36323a4450b40e454e6a0d0df11ba126626`. No fit, model call, new probe or
performance measurement was performed. This is a research input to the
orchestrator's next decision, not authorization for another experiment.

**The existing tagged value-map eligibility supports exact conditional credit
from the actual read-neuron error.** Its overwrite transport already includes
the dependence of later memory writes on earlier values. Replacing only the
value neuron's fixed random broadcast with `g_read` would use that derivative;
it would not make controller, key or strength learning exact, nor establish
useful writing. The result below is our derivation for this code, not a theorem
claimed by the cited authors.

## What the primary sources establish

Schlag, Irie and Schmidhuber identify linear attention with fast weight
programming and introduce an error-correcting delta update. Their sections
4.1-4.2 discuss associative interference and editing existing associations.
The experiments do **not** demonstrate our required local slow-weight learning:
section 6.1 uses Adam for synthetic retrieval, while section 6.3 carries memory
between language-model segments with a bounded backpropagation span. Their
normalization and inputs also differ from our signed L2 keys and raw bytes.
Use this as the forward-memory precedent, not a transferable learning result.
[Linear Transformers Are Secretly Fast Weight Programmers, ICML 2021](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf).

Irie, Schlag, Csordas and Schmidhuber give a particularly clear specification
of the delta memory in section 2.2, equations 8-10: read the old value with the
write key, then add the gated outer product of the value residual and key.
Section 2.1 explicitly describes end-to-end gradient training; appendix A.1
uses Adam and a 256-token backpropagation span. Their recurrent extensions are
not needed for the derivative below and are not recommended as a second queue.
[Going Beyond Linear Transformers with Recurrent Fast Weight Programmers](https://arxiv.org/html/2106.06295v2).

Bellec and colleagues separate a forward-computable local eligibility from
a neuron-level learning signal in equations 1-4. The exact decomposition
requires the appropriate learning signal; practical e-prop approximates that
signal, including symmetric and fixed-random variants. Thus an available
eligibility is not sufficient reason to assume an arbitrary broadcast gives
the right gradient. This supports our distinction between trace correctness
and signal correctness, without certifying this custom fast-memory learner or
its biological plausibility.
[A solution to the learning dilemma for recurrent networks of spiking neurons, 2020](https://www.nature.com/articles/s41467-020-17236-y).

## Derivation for the existing value synapses

Use a single stream, value row `i`, writer neuron `j`, and memory key coordinate
`k`. All slow weights stay fixed during ingestion. Let

```text
v_t = tanh(W_value h_t + b_value)
P_t = I - beta_t k_t k_t^T
M_t = M_(t-1) P_t + beta_t v_t k_t^T
r = M_T q
```

Hold the initial memory, writer activities `h_t`, keys `k_t`, strengths
`beta_t`, and final query `q` fixed with respect to `W_value`. In the current
forward architecture, activities/keys/strengths do not receive value-map or
memory outputs, and the query path does not receive the memory read. Thus this
conditioning is consistent with a value-only partial derivative while the
other parameter blocks are fixed; it is not an invented input replacement.

Only row `i` depends on value parameter `W_value[i,j]`. Write its row derivative
as `T_ij`, initially zero for an initial memory treated as constant:

```text
d M_t[l,k] / d W_value[i,j] = indicator(l == i) * T_ij,t[k]
e_ij,t = (1 - v_t[i]^2) * h_t[j]
T_ij,t = T_ij,t-1 P_t + beta_t e_ij,t k_t^T
        = T_ij,t-1 + beta_t (e_ij,t - T_ij,t-1 dot k_t) k_t^T
```

This follows directly by differentiating the affine overwrite recurrence.
It is exactly the existing `write_local` tagged value-weight recurrence,
including its tanh gain. Value-bias eligibility follows by replacing `h_t[j]`
with one. The loss derivative for a supplied read-neuron signal is therefore

```text
g_read[i] = partial L / partial r[i]
partial L / partial W_value[i,j] = g_read[i] * (T_ij,T dot q)
partial L / partial b_value[i] = g_read[i] * (T_bias_i,T dot q)
```

`apply_local_feedback` already retrieves `T dot q` and multiplies it by the
per-neuron value signal before its negative learning-rate update. The one
missing factor is consequently the **direct read slice** of the mean LMS
context signal, not its random projection. Do not add another tanh gain,
write strength, query factor or `M^T`: those either already occur in the trace
or belong to the distinct query-credit path. Preserve the existing active-stream
mean and the once-per-lesson rate; no extra division by 257 is justified.
These statements come from inspection of
[PlasticMemory](../../src/humanbrain/plastic_memory.py) and
[LocalMemoryLearner](../../src/humanbrain/local_memory_learner.py).

Each value synapse maintains its own address-indexed trace. The final query is
an explicit address broadcast used to retrieve that trace, and the read error
is shared by the corresponding value neuron. The update does not require a
host-computed per-synapse correction or reverse traversal of a retained history.
It does require declaring this routing as part of the neuronal architecture;
vectorized execution alone does not prove biological locality.

The derivative statement is in real arithmetic. Floating trace transport and
reductions approximate it numerically; an implementation comparison needs
ordinary finite-difference or independent scalar reference QA, not a claim of
bitwise symbolic differentiation. A nonzero initial memory is permitted only
as a fixed boundary, or with its matching derivative initialized. Retiring
traces after a slow update truncates any older memory's parameter history.
Changing value weights during a trace interval without redefining that
derivative invalidates the stated contract. The aggregate-credit control lacks
addressed overwrite transport and does not inherit this exactness claim.

## What this would and would not resolve

The [query-credit result](../../experiments/r4_integrated/local_memory_query_credit_v1_result.md)
improves tagged fit from 2/8 to 5/8, but leaves new-value success at 1/4 and
interference at zero. Repaired writer-frozen also gets 5/8 and matches tagged
correctness on every final case. This shows that the repaired address route
helps fitted behavior without demonstrating a learned-writing contribution.
It does not prove that incorrect value credit is the remaining cause.

The competing explanation remains deficient source/key representation and
write interference. The preceding
[frozen localization](local-memory-localization-followup.md) found surviving
assignment differences but largely common-direction responses for queries
requiring opposite changes. A correct value derivative can change what values
are stored along those addresses; it cannot itself guarantee distinct write
keys or remove the writer controller's approximate credit. The paper's
associative-crosstalk discussion is consistent with this risk, but does not
diagnose our trained byte representation. Preserve both explanations when
interpreting any future single value-signal comparison; no rate, epoch,
representation or architecture sweep follows from these sources.

## Small integration and cost boundary for review

The minimal candidate is `signals['value'] = mean_signal[:, query_size:]`,
using the existing float64 phase accumulation/mean and single checked upstream
cast. Keep all other writer signals, query repair, LMS updates, trace clocks,
controls and finite proposal validation unchanged. Signals still get computed
and checked for frozen controls; existing parameter flags suppress their updates.
No evidence is supplied here to enlarge the comparison or alter its budget.

If removing the obsolete `value_feedback` buffer, preserve the private random
generator's existing draw before constructing `strength_feedback`: value is a
middle draw, so simply deleting it changes an unrelated control tensor. Make
the changed feedback metadata explicit and preserve all shared initial tensors
in a comparison. There is no reason to keep a production legacy route.

Existing value tags need no new allocation or token-time recursion. The proposed
signal removes one 64-by-96 random projection per lesson; the address-tag
contraction and large tagged writer transport remain. That is a small operation
count observation, **not a measured speedup**. The saved
`.pytest-tmp/query-credit-performance-summary.json` records 338.866 seconds in
complete lessons out of 366.330 seconds, with tagged/current-tagged about
102/101 seconds. It does not isolate trace arithmetic from checks and snapshots.
Tagged traces already dominate persistent state payload and are the previously
profiled expensive path. Preserve them and require fresh complete-workload
admission if a new learning comparison is selected; no optimization is selected
by this report.

Development context: the task-specific Mem0 search returned unrelated older
results and contributed no scientific premise. The knowledge-graph transport
was closed; exact known source files were read as the documented fallback.
