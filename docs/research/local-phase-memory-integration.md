# Two-byte local memory/LMS integration contract

2026-09-06. Recorded before implementation, after the four-arm LMS component
qualified and passed its independent audit. This is the selected return to actual
document-derived neuronal context, not another authored-factor fit. No trained
factor-input weights are transferred. Old experiments and their evidence remain
tied to their original commits; current learner replacement adds no compatibility
branch or checkpoint migration.

## Ownership, architecture and public input contract

Replace the retired recurrent output in `LocalMemoryLearner` directly with fresh
`PhaseContextLMS()`. Retain its existing memory/controller, query cell, write/read
credit, ingestion, source removal, retirement and five control modes.
Rename the constructor's `embedding_size` argument to `input_code_size`, default
32. Query size equals that input size. Read size defaults to 64 and their sum
must be the LMS module's fixed 96; compatible partitions are allowed for focused
QA. Writer hidden size and memory key size retain their existing options. The
production return uses input/query32, writer64, key64 and read64.

The learner owns a fixed `input_codes` buffer of shape `[258, input_code_size]`.
Initialize it with the first standard-normal draw from a private CPU generator
at seed+2, matching the old decoder embedding initialization exactly. Keep all
258 rows, including the unused input BOS row, counted and serialized. Documents
and questions use `F.embedding(raw_bytes + [256], input_codes)` with no text
normalization. Input codes are not a fictional output embedding and never learn.
Memory/query seeds remain seed/seed+1. Existing fixed upstream feedback matrices
retain their seed+3 draw order, shapes and scale.

The teacher must be exactly two raw bytes and learning rate exactly 0.01;
validate both before query traversal or any update. Prediction retains the
LMS module's at-most-two-byte cap. New learner/module metadata records owned
input codes, explicit input/query/read/writer dimensions, mixed precision,
half-squared-potential loss, phase clock and feedback semantics. Old snapshots
must fail configuration matching rather than acquiring a silent reinterpretation.

## Actual pre-update neuronal context signal

Let `r = onehot(target) - z` be the saved LMS residual and `p` the current
three-neuron phase activity. With the current output matrix before its update:

```text
V[j,k] = sum_phase p[phase] * W_pre[j, 3 + 96*phase + k]
context_signal[k] = -sum_j r[j] * V[j,k]
W <- W + 0.01 * r.T * features
```

Compute and finite-check this float64 signal before publishing output weights.
It is the instantaneous gradient of the half-sum squared potential loss with
respect to each context neuron's activity. The negative residual sign is required:
the output adds its teacher-minus-potential correction, whereas existing upstream
rules subtract `learning_rate * signal * local_eligibility`. Do not include another
learning rate or division by 257 in the context signal.

The signal requires explicit return paths using the forward phase/context output
weights. It is neuron-level error, not an externally computed per-synapse update.
No separate backward matrix is learned or stored. Only the immediate output-to-
context derivative is exact; existing memory/query traces and fixed module
projections remain approximate local learning rules, with no convergence claim.

Sum all three pre-update context signals in float64, checking each signal and
running sum; also reject overflow in accumulated teacher loss. Divide once by
the actual teacher count, finite-check the mean,
then cast once to the actual upstream context dtype/device and check finiteness
again. Feed that mean through the unchanged controller/key/value/strength/query
fixed projections and directly into the first `query_size` query-cell coordinates.
Upstream parameters update once per lesson and traces retire exactly as before.
Decoder parameters still update at each teacher byte; this is not an atomic
whole-lesson transaction. The first fresh lesson has zero upstream signal because
each phase's readout block starts at zero; later experience can provide a
nonzero signal. No artificial warm-start signal is added.

## Controls, accounting and verification scope

All five modes start with identical tensors. `frozen` must explicitly disable
LMS learning because `requires_grad_(False)` alone does not freeze its buffers.
Writer/maps-frozen and writes-disabled retain their existing parameter, fast
synapse and allocated-credit rules. Every mode follows the same actual signal
calculation; do not substitute zeros to bypass upstream learning. Frozen means
updates disabled, not labels unoffered.

Count the owned code table (33,024 bytes at default float32), LMS weights and
clock/gating buffers, input/query/memory parameters, retained fast synapses and
credit, plus returned signals and accumulation scratch. The 74,016 potential
context-return connections reuse forward readout weights and require no extra
stored weight array. A returned float64 context signal occupies 768 bytes.

Verification must preserve initial memory/query/fixed-broadcast tensors and
exact old input code values; check pre-update sign/gating numerically, genuine
later-lesson upstream changes, fixed/frozen/write-disabled behavior, independent
snapshots, early malformed-teacher/rate rejection and finite cast failures.
Integration owns the existing test file and QA; this agent runs no fit or timing.
The fresh two-byte return recipe must vary retained answers for the same query,
retain source-free execution and controls, and obtain its own cost admission.
Neither binary-factor qualification nor its LMS proof transfers to learned,
continuous, moving memory contexts or general action selection.

Implementation is complete. Independent source review confirms the signal sign,
pre-update timing, generic phase gating, mode handling and float64 accumulation
followed by one checked upstream cast. The identified accumulated-loss finite
check was added before upstream publication. Ruff and whitespace checks pass.
The lesson receipt now exposes actual changed decoder steps and all/enabled
nonzero residual offers, with its half-squared-potential loss label. The module
documents that whole-model `.float()` would violate the LMS float64 contract;
focused mixed-precision tests convert only upstream components. Numerical QA,
exact old-code-value verification and source-free transport pass all 94 affected
checks: 93 in `.pytest-tmp/two-byte-qa-final.xml` and one separately prepared
native transport check in `.pytest-tmp/two-byte-transport-qa.xml`. Doctor receipt:
`.pytest-tmp/dev/doctor-j6q_14qs/receipt.json`. The preserved initial QA failure
concerned a fixture rejecting the new finite learning-rate metadata; it was
corrected before the passing checks. No complete cost or acquisition run has
started at this implementation freeze.
