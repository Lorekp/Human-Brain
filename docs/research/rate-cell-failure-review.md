# Rate-cell acquisition failure: one next design question

Reviewed decision, 2026-09-05: prepare one comparison of direct query access under
the limits below. This note implements or scores nothing. The failed rate-cell
candidate remains closed. The
[roadmap](roadmap-reading-to-coding.md) owns adoption and broader priorities;
retain the [prior plan](contextual-rate-cell-preflight-plan.md) as historical.

## What the saved run establishes

The [closed result](../../experiments/r4_integrated/contextual_rate_cell_result.md)
owns scores, loss trajectories, costs and audits. Treatment solves 0/48 supported
requests in every phase and stops on 255/256 probes. Crucially, **zero correct
conditional span winners** exist in each phase; threshold tuning alone cannot
rescue these answers. Supported span loss improves while supported gate loss
worsens and unsupported gate loss improves. These are scheduled pre-update
observations, not a fixed-checkpoint evaluation curve. Numerical replay passes.

The evidence establishes failed acquisition under the fixed budget. It does not
identify shared-producer interference as the unique cause, prove query information
vanishes, or establish that rate cells/fixed features cannot solve the workload.
The earlier joint-ranking feasibility check remains inconclusive.

## Three materially different directions

**A. Persistent query alignment inside the cell producer.** The current
[`RateCellReader._forward`](../../src/humanbrain/rate_cell_reader.py) gives the
query to the recurrent prefix, then document positions receive document inputs
and the preceding state. Give each document position direct access to retained
query states as well. BiDAF computes token-to-query attention and passes query-aware
context into a recurrent modeling layer; its C2Q ablation supports this design
in its own much larger supervised setting, not ours. R-NET likewise uses gated
question–passage matching before recurrent processing.
[BiDAF §2, Table 1b](https://arxiv.org/pdf/1611.01603),
[R-NET §3](https://aclanthology.org/P17-1018.pdf).
Our hypothesis is a more usable query-dependent path to competing whole spans,
not a proven diagnosis of prefix forgetting. This changes learned computation,
not a hand-written property/entity selector.

**B. Separate answerability producer with multiple-instance evidence pooling.**
Use a distinct query-conditioned cell bank for answer existence, pooling across
windows rather than averaging every word from the span producer. Stop gate
backpropagation at the span-bank boundary. This removes direct parameter sharing
between the two objectives by construction. It does not establish adequate
answerability learning, and cannot alone repair the zero conditional winners.
Attention-based MIL learns a permutation-invariant weighted representation from
bag labels; its experiments concern other domains. Adapting it to retained
passages is a hypothesis. Softmax pooling can still shift when irrelevant windows
are added; noisy-OR can accumulate false evidence with corpus size. Neither is a
free solution to growth. [Ilse et al. §2](https://proceedings.mlr.press/v80/ilse18a.html).
Clark and Gardner establish the importance of comparable paragraph scores and
training on answer-absent paragraphs; they do not validate our separated gate.
[Clark and Gardner §3](https://aclanthology.org/P18-1078.pdf).

**C. Fixed features with separated objectives.** A fixed token/alignment producer
and trainable span/STOP heads offer the cheapest reference and isolate what heads
can acquire. They are useful within a matched comparison, not an independent
queue of fixed-feature fits. Reusing the old 1621-feature/three-word inventory
would confound the comparison with candidate coverage; use the same whole-span
inventory. Success would favor the simpler learner, but would not establish a
learned-cell contribution. The failed joint objective did not test every separated
fixed-feature design; the inconclusive LP supplies no contrary proof.

## Recommendation: one matched alignment question

**Does direct query access at every document position let learned rate cells
acquire executable whole-span choices beyond matched fixed computations?**
Prioritize A because it addresses the observed span failure as well as making
query information available to the unchanged gate. Keep B as an alternative
considered here, not another automatically scheduled experiment. Do not combine
alignment, independent gate producers, new pooling and optimizer changes into
one result and then attribute it to any one of them.

A concrete small candidate retains the 64 cells, frozen word inputs and query
prefix. Let u_j be each query-word rate state and r_t the document word's local
input projection. Add a query context at each document step:

`s_tj = r_t^T A u_j / sqrt(64); a_t = softmax_j(s_tj); c_t = sum_j a_tj u_j`

`h_t = .5 h_(t-1) + .5 tanh(W x_t + U h_(t-1) + V c_t + b)`.

A and V are learned 64×64 matrices; attention has no access to expected answers,
source labels or candidate indices. Use the existing whole-span heads and
word-weighted STOP pooling, separate losses, hierarchy and Adam mask policy.
This is disclosed differentiable attention plus BPTT-trained simulated rate cells,
not local biological plasticity or a conventional solver hidden behind a cell race.
The added 8192 float64 weights cost 65,536 bytes; two Adam moment arrays add
131,072 bytes per allocated arm, before gradients, query states and attention.
Attention adds O(Q·64² + T·Q·64) work when A·u is reused; V adds O(T·64²).
These are arithmetic/storage estimates, not measured runtime. No dense span
features or second encoder are needed. Freeze initialization of A/V, exact
projection/gradient arithmetic and bypass correspondence before target outputs.

## One integrated experiment, not a new gate ladder

Allow two hours preparation and one 45-minute attempt: at most 15 minutes of
synthetic numerical/full-inventory cost checks, then 30 minutes acquisition.
Keep one seed, update policy, exposed 64-task roster, 256 encounters and 511 offers;
no rate/epoch/window sweeps, extra fitting passes, solver weights or retries.
Reuse the existing full candidate inventory, authentic Python attempts/teachers,
source growth, actual/alternate probes and motor. This is acquisition development,
not new evidence from fresh tasks. Do not open official or protected evaluations.

Run the aligned learner, a matched prefix-only learner, aligned frozen cells
with trainable heads, aligned fixed bypass with trainable heads, fully frozen
evidence and external-only reference. Freeze parameter/memory differences.
All arms share original inputs, spans, labels and execution budgets. The aligned
frozen-cell comparison is essential: improved host heads or fixed attention alone
must not be credited as learned cell computation. Freeze **W/U/b and A/V** in
both aligned frozen-producer and fixed-bypass controls; only their heads learn.
The old frozen result stays
immutable; the prefix arm is a declared reference in a new comparison.

Retain raw sources, token offsets, original own-arm attempts, teacher authority,
source-pool hashes and current producer state. Replay recomputes alignment,
trajectories and gradients; historical attention/eligibility is not current credit.
No inference reads labels or lesson ledgers. Test attention/recurrent gradients,
frozen masks, duplicate spans and source-growth state invariance synthetically.
Reject before teaching if the six-arm runtime estimate exceeds 30 minutes,
parent RSS exceeds 2 GiB or counted learner payload exceeds 1 GiB.
First replace the original module-name-dependent payload traversal and verify
the exact `python -m` entry path on synthetic readers, shared input buffers and
optimizer state. The prior counter skipped `__main__` readers; its retrospective
estimate cannot substitute for a valid live count in the next admission check.

Keep the prior acquisition thresholds: 44/48 actual supported executions and
15/16 STOPs before/after growth, 15/16 STOPs in alternate phases, at least 80% in
each declared group, and at most 3/64 actual correctness lost after growth.
Require at least eight additional supported successes over both trained-head
controls without fewer STOPs. Report conditional gold winners/ranks separately
from gate rejection and execution; no threshold-selected rescue metric.
A failure, timeout or missing learned-cell advantage closes this candidate;
partial loss/rank gains do not authorize another alignment variant automatically.
A pass earns review for a genuinely broader persistent reading/action workload,
not continued optimization of these 64 questions or a general coding claim.
