# Learn over complete choices before adding a trainable representation

Development decision, 2026-09-05, following a bounded three-agent review.
The [projected comparison](../../experiments/r4_integrated/contextual_projected_binding_result.md)
failed. Neither arm fully acquires taught forms; available but previously
unexecuted competitors win despite mostly positive admitted pair margins.
This motivates better acquisition credit without establishing that the fixed
representation is incapable of transfer.

## Immediate work: one full-choice acquisition loop

Implement a small supervised learning interface in the same persistent
document -> evidence -> Python edit -> execution loop. Keep the existing fixed
BINDING features and one-to-three-word candidate family initially. Replace
immutable-negative pair replay with explicitly declared teacher imitation over
**all available TRAIN candidates and STOP**. This is a new learning contract,
not an extra epoch or repair of a spent experiment. Preserve the old AROW path
and frozen evidence until this alternative demonstrates useful behavior.

For scores z, candidate set C and teacher-authorized outputs G:

```text
L = logsumexp(z over C) - logsumexp(z over G)
dL/dz(c) = softmax_C(c) - 1[c in G] * softmax_G(c)
```

For fixed features, the gradient is the corresponding expected-feature
difference. Use a separately declared SGD update and learned STOP score;
do not pass it into AROW as an executed contrast. Unexecuted competitors receive
surrogate gradients, never fabricated FAIL receipts. Low probability still
permits negligible gradients; normalization is not a guarantee of acquisition.

An authentic FAIL/PASS pair proves only those two observed outcomes. The new
teacher contract must additionally authorize exact answer bytes or STOP.
For literal-copy tasks, G includes all occurrences emitting those bytes; this
does not teach their provenance or establish that other repairs would fail.
STOP requires explicit absent-evidence teaching, including present entities
with missing requested relations. A learner STOP is not its own target.
Training on already-correct lessons is disclosed supervised imitation too.
Inference cannot read labels, teacher ledgers or evaluator filters.

Use one fixed, complete TRAIN source pool for the first acquisition loop,
with consistent facts throughout. Read that pool once before teaching. Each
offer/replay recomputes all scores under the current learner and that same pool.
Bind original query/source bytes, output/teacher identity and genuine execution
receipts to a new current-state/loss/gradient receipt. Historical eligibility is
not current gradient. Later source growth is an updates-off interference test;
neither its sources nor fresh labels enter replay. A future expanding TRAIN
pool would require an explicit answer-preserving or recertification rule.

## Development scope and success criterion

Prepare a new reusable package of **64 requests: 48 supported and 16 unsupported**,
across at least sixteen source/task bundles. Include one-, two- and three-word
answers, competing entities, changed clause order, question paraphrases and
incidental lexical overlap. Counterfactual assignments must remain consistent
within the shared store; do not supply entity/property predicates to the learner.
Keep actual executed single-literal edits and report their narrow action scope.
These authored cases are development tools, not official reading evaluations.

Bound the first development effort to two hours of preparation and two hours
of acquisition execution. Permit one scheduled four-pass stream, at most 256
encounters/teacher opportunities and one replay per encounter. Fix the teacher
schedule, initialization, SGD step and gradient bound after synthetic numerical
preflight and before real TRAIN outputs. No subsequent rate/epoch sweep belongs
to this attempt. This larger budget is newly declared; the old 48/150 contract
does not apply. If cost preflight cannot fit, revise the design before teaching.

Require at least **44/48 supported actual-TRAIN executions and 15/16 correct
unsupported STOPs**, with at least 80% in each declared role/length and alternate
question-form group. Report exact counts and teacher repairs separately. After
unrelated source ingestion, rerun these requests without feedback and allow no
more than five percentage points aggregate loss while retaining the acquisition
thresholds. Alternate-form and source-growth probes remain public development.
Failure returns to a specific acquisition diagnosis; it does not authorize a
fresh-cohort run or show representational impossibility.

Use the same initialized no-learning mate and an external-only selector with
matched sources, output inventory and motor. Compare real first attempts,
whole-answer selection, executed edits, STOPs, final worst competitors and
retention. Count supervised offers, actual executions and all retained state.
An acquisition pass earns integration and a separately frozen fresh comparison;
it does not alone establish that the objective beats historical AROW scores.
Protected gates remain closed.

## Conditional producer design, not an automatic successor

If fixed-feature acquisition works but fresh behavior remains inadequate,
consider a small query-conditioned **leaky rate-cell token producer**:

```text
h_t = 0.5 h_(t-1) + 0.5 tanh(W x_t + U h_(t-1) + b)
span_score(i,j) = A h_i + B h_j + mean_(i..j)(C h_t)
```

The proposed envelope is 64 cells with frozen 384-dimensional token inputs plus
two segment indicators. Process query then document tokens, resetting per window.
Learn W/U/b and three readouts by backpropagation through time. This is conventional
gradient training of simulated rate neurons, not local biological plasticity.
The existing `FrozenContextEncoder.encode` already returns immutable float32
`ContextEncoding.token_features`, aligned UTF-8 `byte_spans`, word IDs and input/encoder
identities. Retain those arrays plus document/window origins in the successor
store; today the evidence store keeps pooled candidate vectors. No new encoder
wrapper is needed. Its `token_count` includes specials and can exceed stored rows.

Score structural contiguous token intervals plus STOP, without a semantic length
filter; use prefix sums for scalar interval scoring. Validate outputs longer than
three words. Within-window enumeration costs O(N squared); cross-window synthesis
and arbitrary program generation remain unsupported. Query-conditioned states
are transient; retain authenticated raw sources/frozen tokens and recompute after
producer updates. Existing exact-eligibility replay cannot certify this process.

Compare trainable versus same-initialization frozen producers with **identical
tokens, candidate family, trainable readouts, objective, teacher stream and budget**,
plus no-learning and core-disabled controls. Changing from old short spans to a
broader pointer family is not a producer-only effect. Isolate learned sequence
computation through matched decisions, not parameter mutation alone. Learned
pooled bilinear factors are smaller but still leave boundaries/order implicit;
they are not another rank-search queue.

The 64-cell sketch contains 29,057 float64 parameters (232,456 bytes), plus equal
gradient storage and no dense covariance. A frozen float32 token cache costs
1,536 bytes/token before offsets; 4,096 tokens occupy 6,291,456 bytes. Encoder,
raw sources, ledgers, trajectories, candidate scores and controls are additional.
These are estimates, not measured efficiency or a scaling claim. Confirm the
full interface and resource envelope before choosing this implementation.

## Research basis and authentic reading

[Collins, 2002](https://aclanthology.org/W02-1001/) supplies a precedent for updating
against the current decoded competitor; that alternative still focuses each
update on one winner. [Lee et al.](https://arxiv.org/abs/1611.01436) study recurrent
whole-span representations. [Clark and Gardner](https://aclanthology.org/P18-1078/)
discuss shared passage normalization, answer-occurrence marginalization and
no-answer modeling. These motivate the design; none establishes that this small
cell learner or teaching package is adequate.

The [official reading candidate](official-reading-next-candidate.md) remains
complementary preparation. Preserve original questions and marking keys; do not
rewrite option/table/ordering tasks to suit span extraction. Source custody and
structured responses must be ready before that diagnostic. Keep one integration
milestone and documentation proportional to the decision, rather than creating
independent mechanism queues.
