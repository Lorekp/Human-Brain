# Locate record identity before another binding fit

2026-09-06. Selected bounded diagnostic; no new model execution in this review.
Close the direct-value fit and observe the current frozen writer/read pathway.
Do not extend training or select another error broadcast automatically.

## Completed result and the unresolved question

The [value-credit return](../../experiments/r4_integrated/local_memory_value_credit_v1_result.md)
at `86cc7232320cf4cef9337cf4f3b79a017bb55171` leaves tagged at 5/8 fit,
1/4 new-value, 1/4 delayed and 0/4 interference actions. It produces exactly
the same twenty free outputs as its immediate predecessor: both assignment
swaps remain invariant (`ca/ca`, `dd/dd`). Aggregate falls to 4/8 fit; writer-frozen
retains the same correctness as tagged on all twenty cases. The conditional
value-gradient derivation has not produced useful binding or learned-writing
advantage in this fixed workload.

The strict audit is SHA256
`f29992900df13adeab81bdae91a3f5c9e6d6a35cdab4900b16b8ab531020f268`.
Saved-only description `.pytest-tmp/local-memory-value-credit-v1-description.json`,
SHA256 `ac8b875f73ed65c6dcd20e0d8cec07d33eac89651348ac750556dbb88de99480`,
passes in 2.200 seconds internally. Its immediate reference matches the prior
initial/final arrays and metadata, all 1,024 lesson receipts and twenty output
receipts exactly after excluding arm names and time. The full audit also verifies
unaffected-control parity with the explicitly declared metadata mapping.

Last-128 first-attempt counts are tagged 32, predecessor 32, writer-frozen 35 and
aggregate 38; corresponding teacher losses are 0.233903, 0.234517, 0.232200 and
0.227654. Better loss or more successes in that acquisition window does not
establish improved final binding. These curves do not establish convergence.

The decisive design question is narrower: **does record identity remain in
writer activity when value information arrives, and does that distinction enter
the write address and the actual question-conditioned read?** The latest endpoint
does not save its intermediate writer activities or read vectors. Weight changes
alone cannot answer that question.

## What is already known about timing

[The source](../../src/humanbrain/plastic_memory.py) uses
`h' = h + 0.5 * (tanh(Wx + Uh + b) - h)` with initial `U = 0.9 I`. Its initial
self-path carry is between 0.5 and 0.95 depending on activation, so the explicit
0.5 leak does not itself prove short memory. There are no explicitly separate
slow/fast populations, but trained recurrent dynamics could carry information
longer. The existing sigmoid strength map is already a neuronal write gate.

[Ingestion](../../src/humanbrain/local_memory_learner.py) advances writer activity
and tagged credit across both document terminators. It purges activity only after
the full document list; it retains fast memory and eligibility until teaching.
Each query starts a separate query cell from zero. A token-specific host parser,
per-document reset or supplied key/value factors would change this contract.

The completed historical saved analysis
`.pytest-tmp/historical-matched-prefix.json`, SHA256
`d6f4a67e1d8e5cb84512e0eb3568ca1c2a00bb848cc3dbd917afa295ec0d57d0`,
uses only the earlier `d724e06` localization traces. All 672 same-key/different-value
prefix checks of activity/key/value/beta are identical through byte 20, as the
identical inputs require. For tagged, different-key/same-value activity distance
from key end to pre-value falls from 3.695101 to 0.401011 in the first pair and
2.802161 to 0.284115 in the second: about 10.9% and 10.1% remain. End-of-document
ratios are about 3.7–4.5%. Write-key cosine rises to 0.993972/0.997551 before the
value arrives. At value end, changing the earlier key changes the write key by
0.048–0.092, whereas changing the value with the same key changes it by
0.682–0.755. Writer-frozen has the same qualitative pattern.

Those are measured relative sensitivities, not proof that identity is absent or
unrecoverable. Small nonzero Euclidean differences can still matter downstream.
The historical analysis resolves the old matched-prefix question; do not repeat
it or relabel it as measurement of the new value-trained checkpoint. The current
confirmation is justified because writer and query weights have changed.

## One fixed current-checkpoint localization

Use only the completed current `tagged` and `writer_maps_frozen` weights, held
fixed for every observation. Reuse the following already exposed raw-document
assignments in original and reversed document order:

| Stream | First assignment | Second assignment |
| --- | --- | --- |
| A | k00 = ab | k01 = bc |
| B | k00 = bc | k01 = ab |
| C | k02 = cd | k03 = da |
| D | k02 = da | k03 = cd |

Actual inputs remain `Record k00 has value ab.` and the analogous byte strings.
This gives eight two-document streams per checkpoint, sixteen total. Each stream
starts from zero memory/activity/eligibility. Carry state through both documents
and their terminators, then purge once. The first documents form complete 2-by-2
key/value comparisons at identical initial state: A-original, B-original,
A-reversed and B-reversed cover k00/k01 crossed with ab/bc; C/D cover the second
pair. This avoids confounding first-document identity with prior-document carry.

Reuse the unchanged `traced_ingest` and `observed_query` in
[the observation helper](../../experiments/r4_integrated/local_memory_localization_core.py),
with a new runner/worker identity. Old localization orchestration captures its
old configuration, factory and entrypoint and must not be reused by patching
globals. One ordinary one-shot ingestion mate per stream verifies the complete
traced final state, including tagged eligibility, exactly. Discard that mate;
it contributes no extra query or teaching. Count these sixteen parity ingestions.

Save every actual writer/key/value/beta/fast-memory prefix from the sixteen traced
ingestions. Analyze fixed first-document byte milestones 9 (key end), 20 (common
filler end), 21 and 22 (value bytes), 24 (document terminator), and global step 49
(second document terminator). Offsets are offline observation labels for this
fixed diagnostic only; they must never control writer updates, reset boundaries,
query inputs or the neuron model. A model-visible key/value parser is excluded.

After each full stream, ask both existing key questions once in a fresh source-free
worker. Each call obtains one actual query context, normalized query key, read,
ordinary generated bytes and the three actual phase potentials via `observed_query`.
Its post-EOS phase observations are supplemental diagnostic computations, never
emitted output. No prefix query, prefix decoder call, teacher correction, parameter
perturbation, delay loop, unrelated third document, action execution or motor
demonstration is included. Second-record overwrite/order is observed; do not call
it a new measurement of the full fit's unrelated-write interference challenge.

## Prespecified measurements and discrimination

For every recorded activity/key/value vector and beta, form the two matched
different-key/same-value differences and the two same-key/different-value
differences at the fixed milestones. Report their absolute norms, cosine where
defined, and key-contrast retention relative to that same field at key end.
Also report the 2-by-2 interaction `f11 - f10 - f01 + f00`. Preserve zero
denominators as null; do not add a floor or choose a best-looking projection.
Same-key prefixes must be identical through byte 20; audit this exact invariant.
The step-49 contrasts are whole-stream contrasts: both records' contents and
their order covary there. They do not isolate the first record's causal effect
as the matched zero-initial first-document comparisons do.

Require identical recorded query keys for the same question and fixed checkpoint
across memories. Use each saved final query key for explicit **derived prefix-read arithmetic**:
multiply each of the six saved prefix matrices by each of the two recorded query
keys. This is 192 saved matrix-vector projections, not 192 additional free
attempts or teacher evaluations. No output potentials are recomputed from these
derived reads. Compare the final derived read to the actually recorded read and
report its floating residual; do not claim a float64 reconstruction is bitwise
equal to the float32 forward computation.

For each source-swap pair and order, report `d0 = r_A(q0) - r_B(q0)` and
`d1 = r_A(q1) - r_B(q1)` at every milestone, their norms/cosine, common component
`(d0+d1)/2` and assignment-by-question interaction `(d0-d1)/2`. Repeat the final
contrast using the saved actual reads and saved phase potentials. Compare their
directions with the opposite target changes, but do not infer useful binding
merely from nonzero interaction. Useful behavior remains two correct different
answers for the swapped assignments. Score only the thirty-two actual final
generations, using labels held outside the worker.

Interpret the ladder without arbitrary distance thresholds:

- Activity identity contrast weakening before values motivates investigation of
  temporal representation, such as generic multiscale neuronal state or generic
  tokenization that shortens the event sequence. It does not prove information
  loss or select either implementation automatically.
- Activity contrast surviving while the address becomes primarily value-sensitive
  focuses the design question on routing and learned write timing. Since beta is
  already a gate, a new gate is not justified solely by observing failed answers.
- Source-dependent prefix reads surviving but remaining largely common across
  the two questions focuses the question on binding geometry/credit, rather than
  total erasure. More exact key/controller credit is substantially more coupled
  than the row-separable value partial; another broadcast replacement is not an
  exactness argument.

These are discriminating observations and hypotheses, not unique causal diagnoses.
If several stages show attenuation or contrasts survive throughout, report that
ambiguity. No trained host probe, fitted separator, all-coordinate gradient scan,
new factor representation or automatic next mechanism is permitted in this diagnostic.

## Exposure, resources and completion

The fixed ledger is 16 traced plus 16 one-shot parity ingestions: 32 total,
64 document presentations and 1,600 document byte/terminator steps. Only the
traced half retains 800 token observations and their 800 writer-map observation
sets. There are 32 actual query traversals, 832 query byte/terminator steps,
32 free attempts, 96 phase advances/potential evaluations including all suffix
observations, and 32 observation-only query-key map recomputations. Record actual
ordinary decision count and supplemental count separately; their sum is 96.
There are zero teacher offers, slow updates, weight perturbations, motor demos
and actions. Offline scoring may form 96 target-residual vectors from those same
saved phase potentials (24,672 scalar residuals), with no new phase evaluation or
signal delivered to the learner. The 192 prefix-read projections and all factor contrasts are
separately labeled saved arithmetic. Do not hide parity repetitions in QA totals.

Use a single 60-second aggregate run-plus-independent-audit ceiling and
2,048-MiB monitored process-tree limit. The auditor runs under the remaining
whole-second budget; stop if the run fails or the remainder is inadequate.
Do not extend the budget, add conditions, rerun failed exposures or reuse an
acquisition admission. Preserve attempted/completed progress when interrupted.

Completion requires authenticated current checkpoint/report/audit hashes, fresh
source and recipe closure, unchanged weights, exact one-shot ingestion parity,
query-only worker custody, complete prefix/phase records and the fixed ledger.
The saved-only audit computes and records all 192 prefix-read projections and
their contrasts within that same remaining budget, avoiding an additional
unbudgeted primary analysis. It verifies identity, array shapes/finiteness, same-prefix
equality and the declared contrast/projection arithmetic without re-executing
the learner. Report every planned cell even if uninformative. A complete
diagnostic may conclude ambiguity; audit passage is not a capability pass.

Only after that result should root select one reusable representation or learning
change and return it to actual source-absent document actions. Generic tokenization
is allowed; a host key/value parser or supplied answer factors would change the
scientific claim. Preserve failed fits and the unresolved neuronal action-selection
goal throughout this next step.
