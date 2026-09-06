# Local document memory: locate the failed binding before another fit

2026-09-06. Bounded research requested by the orchestrator after the two-byte
integrated return; the orchestrator subsequently selected this diagnostic.
No model, fit, test, or new evaluation
was executed for this review. The roadmap and active plan were reread. Graph
search returned `Transport closed`; this review uses the known source paths.
Mem0 returned unrelated development notes, which supplied no scientific evidence.

**Select one frozen-checkpoint localization experiment, capped at 60 seconds
including its saved audit and 2,048 MiB.** Keep actual raw-document ingestion,
source removal, neural output and Python execution. Measure where assignment
contrast disappears and test the existing local update direction on disposable
copies. Do not select a replacement learning rule, another output diagnostic,
additional epochs, or a larger brain from the current evidence.

## What the completed run establishes

At source `a94b466f1b2c34e83b53d19e43f9430e52191648`, tagged, aggregate and
writer/maps-frozen each execute 2/8 fit, 1/4 new-value, 1/4 delayed and zero
interference cases. Writes-disabled gets 1/8 fit and zero elsewhere. Tagged
matches all twenty writer-frozen outputs. Both assignments in the `ac/ca`
swap emit `ca`; both `bd/db` assignments emit `cd`. Unrelated document writes
change the first pair to `da` in both memories. Thus retained state influences
the output, but useful assignment binding and learned-writing advantage fail.

The authenticated saved diagnostic is
`.pytest-tmp/local-memory-two-byte-v1-diagnostics.json`, SHA256
`7e5d5cd23e9013a78616bc4ec34c7f1b97048921ccec6e9a6892cb8729edaf13`.
Tagged swapped matrices differ by relative Frobenius 0.050074 and 0.070630;
after interference these become 0.018368 and 0.025933. This rules out exact
equality of the retained matrices. It does **not** establish answer encoding,
adequate separation under the actual query, or usable output margins. Existing
receipts lack query/read vectors, phase potentials, gains and signal magnitudes.
Parameter movement and decreasing teacher loss do not locate the failure.

The successful [LMS component result](../../experiments/r4_integrated/local_context_lms_v1_result.md)
qualified learning with fixed supplied factors. Its separability/convergence
argument does not transfer to changing document-derived contexts. No further
answer-revealing positive control is justified.

## Two explanations, with distinct measurable links

**Writing/addressing may destroy or hide the distinction.**
[PlasticMemory.write_local](../../src/humanbrain/plastic_memory.py) writes every
byte and terminator, with keys and values from the recurrent writer. Its rule is
`M_t = M_(t-1) P_t + beta_t v_t k_t^T`, where
`P_t = I - beta_t k_t k_t^T`. Later nonorthogonal writes can suppress previous
contributions. The controller continues across document boundaries; final
activity is then purged. Shared suffixes and unrelated later writes are plausible
sources of interference, not demonstrated causes yet. The fast-weight delta
rule literature motivates this overwrite mechanism, but its gradient-trained
controller results do not validate our local controller credit.
[Schlag, Irie and Schmidhuber, 2021](https://proceedings.mlr.press/v139/schlag21a.html).

**Surviving distinctions may receive ineffective credit.**
[LocalMemoryLearner.teach](../../src/humanbrain/local_memory_learner.py) now
receives the LMS readout's genuine immediate context error. Its five upstream
groups nevertheless use independent fixed projections of that error. In
particular, the query-map signal is not the actual memory-dependent
`M^T g_read`; the query cell receives only the direct context slice, omitting
the indirect route through query normalization and memory read. Writer tags
transport self-path/map-diagonal traces, omitting key/strength effects on the
overwrite operator. These are declared approximations, not newly found coding
bugs. E-prop separates local eligibility from learning signals; useful traces
alone do not guarantee a useful signal for this added memory pathway.
[Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y).

Neither explanation warrants automatic symmetric feedback, activity gating or
write suppression. Their causal signatures should be measured first.

## One fixed diagnostic protocol

Use only the authenticated endpoint **tagged** and **writer_maps_frozen**
checkpoints. Preserve both, including LMS weights and input codes. Author these
four diagnostic memories before launch, using the unchanged raw grammar
`Record kNN has value vv.` and query `What is the value of kNN?`:

| Memory | First document | Second document |
| --- | --- | --- |
| A | k00 = ab | k01 = bc |
| B | k00 = bc | k01 = ab |
| C | k02 = cd | k03 = da |
| D | k02 = da | k03 = cd |

All values belong to integrated TRAIN. Query both keys in each memory. These
are exposed development combinations, not fresh evaluation. Repeat each memory
once with its **whole document order reversed**, keeping each sentence intact
and its key/value assignment fixed. This intervention is query-blind and makes
no answer-dependent choice of which bytes or writes survive. Across two
checkpoints this gives sixteen ingestions and thirty-two ordinary free queries.
Run their emitted bytes through the existing motor/executor, preserving genuine
EOS, exact stdout and unchanged decoy checks. Count its two motor demonstrations.

### A. Follow the actual write/read/output path

On these ordinary forwards, record all byte-step writer activity, key, value,
strength, normalization norm/local gain, and each prefix matrix. Log all positions
uniformly; do not pick a useful value span or feed any diagnostic back to cells.
Record the final query activity `u`, key `q`, read `r=Mq`, and actual phase
potentials. Runtime source-free requests still contain only queries and memory
IDs; tracing belongs outside the worker's learner input interface.

Preserve the original ingestion boundary: controller activity and eligibility
continue across the whole document list, including each document's terminator;
purging occurs only after the complete list. Repeated `ingest([document], state)`
calls would introduce extra purges and invalidate this comparison. Before the
diagnostic, affected engineering QA must show whole-state equality of instrumented
and ordinary ingestion, including the frozen control's existing forward-only
writer path. Compare the observed context with ordinary prediction and retain
its real generated output; diagnostics must not quietly implement a new reader.

For every same-query assignment pair report the entire prefix curve
`||Delta M_t||`, final `Delta r = Delta M q`, and each phase's
`Delta z_p = W_(p,read) Delta r`. The direct query and phase-only components
cancel within a same-query, same-checkpoint pair. Check these identities against
saved forward values with a stated floating tolerance, not bitwise equality of
different reduction orders. Report the complete vectors, norms and correct-minus-
competing-output margins; equal winners alone are insufficient.

As external receipt algebra, decompose each final read into source-step
contributions:

```text
M_T q = sum_t beta_t v_t [k_t^T P_(t+1) ... P_T q].
```

Check the reconstructed sum against the actual read; summarize contributions
by whole document and all byte positions. This identity is derived directly
from the implemented recurrence. It is not a learning rule, a reverse-mode
gradient, or a substitute answer. It quantifies overwrite of actual contributions
without assuming an oracle address. The two streams can have different keys,
so do not treat their matrix difference as evolving under one shared `P_t`.
Do not inject singular vectors, nearest-neighbor answers, learned probes or
teacher-selected read directions into generation.

#### Explicit floating-point accounting

Use the saved represented float32 prefix matrices, actual keys/values/strengths
and query keys, cast exactly to float64 for **external auditing only**. Do not
renormalize keys or replace their actual squared norm with one. Define

```text
F64(M,k,v,beta) = M + beta * (v - M k) k^T
E_t = saved_M_t - F64(saved_M_(t-1), k_t, v_t, beta_t)
A_0 = R_0 = 0
A_t = F64(A_(t-1), k_t, v_t, beta_t)
R_t = R_(t-1) - beta_t (R_(t-1) k_t) k_t^T + E_t.
```

Report every `E_t`, its norm, and both final closures: `A_T - saved_M_T`
(without residuals) and `A_T + R_T - saved_M_T` (with residuals). The latter
must satisfy the entrywise audit tolerance
`1e-10 * (1 + maxabs(saved_M_T) + maxabs(A_T) + maxabs(R_T))`.
`E_t` includes the observed float32 recurrence's rounding relative to this
specified float64 recurrence; it is not an exactly rational rounding certificate.
Record float64 implementation, reduction order and maximum closure residual.

Likewise preserve `rho = actual_read32 - saved_M_T64 @ actual_q64` as a separate
read-rounding residual. Transport each `E_t` through the subsequent observed
`P` factors alongside the source contributions. Report source-only read sum,
source-plus-recurrence-residual sum, and that sum plus `rho`. Check the last
against the actual read using `1e-10 * (1 + maxabs(actual_read) + sum(maxabs(c)))`,
where `c` includes all source, recurrence-residual and read-residual contributions.
This is a conservative fixed numerical audit tolerance, not a formal error
bound for arbitrary matrices. Save the actual discrepancies rather than
silently assigning closure error to an authored source position.

Implement the contribution calculation with a reverse vector recurrence:
initialize `q_tail=q`, accumulate `beta*v*(k dot q_tail)` and `E_t @ q_tail`
for the current step, then set `q_tail -= beta*k*(k dot q_tail)` before visiting
the preceding step. Never materialize all dense `P` products. Store bounded
arrays in tensor artifacts; JSON contains identities, dimensions, counts and
summaries rather than millions of decimal prefix entries.

Use the same scaled `1e-10` tolerance for float64 phase-potential decomposition,
with the sum of absolute component maxima in the scale. The actual same-query
keys must match exactly within a checkpoint. Their observed read contrast equals
`Delta M64 @ q64 + rho_A - rho_B` to the stated tolerance. Report the uncorrected
identity discrepancy as well. A distinction or directional loss difference
within ten times its recorded applicable audit tolerance is numerically
unresolved; it cannot select a mechanism. These checks do not require a model
forward, global gradient or fitted decoder in the auditor.

### B. Check only the existing local direction

On the eight original-order tagged query conditions, independently restore the
same endpoint, reingest the full raw documents, and call the existing teacher
once on a disposable copy: exactly two TRAIN bytes plus EOS, at rate 0.01.
This is eight isolated corrections, twenty-four teacher offers, never a fitted
sequence. Preserve all three pre-update signals and the actual proposed slow
parameter displacements. Discard the temporary output changes and state after
each correction; no candidate checkpoint becomes a new baseline.

Separate that displacement into exactly two predeclared blocks: writer/controller/
key/value/strength parameters, and query-cell/query-map parameters. For each
block evaluate `theta + Delta_block` and `theta - Delta_block`, with the original
LMS output weights fixed and all other parameters at the endpoint. Use the
actual rate-0.01 displacement; no coordinate scan, normalization, line search,
extra magnitude, sign selection or retained correction. Reingest raw documents
from empty memory for each evaluation, including address-only controls, so the
comparison has uniform custody and never evaluates stale memory after a writer
change. This adds thirty-two ingestions/free queries, also sent to real actions.

For the positive block, install the production teacher's **recorded after
parameters directly**. In float32, `before + (after - before)` need not recover
that array exactly. The negative block is explicitly
`before - (after - before)`, with both operations in the original parameter
dtype. Record the displacement and verify these separate positive/negative
custody rules exactly; they are symmetric real-arithmetic directions, not a
promise of bitwise symmetric floating perturbations.

Measure the same fixed-readout half-squared potential loss at all three clock
phases for baseline and perturbed contexts, using TRAIN labels solely in the
external scorer. If ordinary generation ends early, any additional potential
observation is separately counted diagnostic phase evaluation, not an emitted
byte or successful attempt. Log the loss and target margin differences for
every condition and the aggregate, with no statistical-population claim. This
is a finite **update-effect** comparison; an actual step can be too large for
a derivative interpretation. The negative direction is an external falsifier,
never a proposed neuronal teaching signal or a learned optimizer.

The complete budget is at most fifty-six two-document ingestions (including
the eight proposal copies), sixty-four free output/action attempts, two motor
demonstrations and twenty-four updating teacher offers. Count diagnostic phase
evaluations and all retained traces/snapshots separately. No new-value teacher
labels, optimizer, BPTT, automatic differentiation or per-synapse gradient oracle
are used. A 60-second aggregate watchdog includes execution and saved audit;
2,048 MiB applies to both process trees. Freeze source, checkpoint and roster
hashes first. On timeout preserve partial evidence and close without extending
the budget or launching a fit.

#### Evaluation accounting before freeze

There are exactly sixty-four free output attempts and sixty-four corresponding
action receipts, plus two separately identified motor demonstrations. Each
free attempt's generated phase observations are retained; finish only any
missing clock suffix after early EOS to obtain its three diagnostic potentials.
Those extra phases cannot append emitted bytes. Thus `free_generation_decisions
+ supplemental_phase_steps = 192` across the sixty-four conditions. Never call
generation and then silently recompute all three phases for tracing.

The eight actual proposal corrections add twenty-four teacher phase advances,
twenty-four teacher offers and 6,168 scalar residual offers. The current
`teach_local` also reevaluates its scores in `_check_step` twenty-four times;
record these validation score evaluations separately. With no duplicate tracing,
the totals are 216 phase advances and 240 full output-score evaluations,
plus twenty-four production pre-update context-signal projections. The selected
helper derives another twenty-four context-signal projections for receipt
verification, for forty-eight in total. It also repeats the query-key map once
per observed query (sixty-four times), and observes key/value/strength maps on
each of the eight hundred traced baseline byte steps. Count these observations
separately from production computation. If proposal diagnostics
derive per-phase signals from original weights and saved scores, label them
derived receipts and verify their mean against the production teacher's actual
mean. Any additional score or signal recomputation used for this verification
must have its own counter fixed in the recipe before launch; it is not a new
teacher offer. The three phase blocks are independent during a correction, so
earlier output updates do not change a later phase's active forward weights.
No uncounted baseline probe, duplicated `predict`, or teacher-copy generation is
permitted. All this work, serialization and the separate monitored audit share
the unchanged sixty-second aggregate ceiling.

## Interpretation and return

The completion criterion is an audited localization report, not a score pass:

- Large prefix contrast that contracts during subsequent actual writes, together
  with whole-document order sensitivity and suppressed earlier contributions,
  supports overwrite/recency as an immediate bottleneck. It does not prove that
  earlier activity represented the intended answer.
- Matrix contrast surviving while the actual `Delta M q` is tiny localizes
  attenuation at the current address. A nonzero read contrast that disappears
  through `W_(p,read)` instead localizes it at the currently learned output
  projection. Neither result proves a different learner can decode the matrices.
- Positive local updates that consistently worsen the fixed-readout loss while
  their negatives improve it falsify useful descent at this checkpoint and step
  for the corresponding block. Improvement supports useful local direction
  there; it does not establish adequate speed, global alignment or convergence.
- Mixed signs or multiple attenuation sites remain an unresolved result. Do not
  force a single explanation using an invented ratio threshold or select a
  winning perturbation as a trained model.

After this one measurement, select at most one repair addressing its dominant
observed link and return it directly to the full raw-document/action workload,
with fresh initialization/cost admission and the same strict five-control,
8/4/4/4 executed criteria. The diagnostic itself authorizes no repair or full
fit. Preserve the closed failures and previously exposed cohort labels. Source
conditioning, useful fact binding and the supplied host motor remain distinct
claims; no outcome here advances the general integrated gate.
