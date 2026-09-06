# Local writer arithmetic and validation development

2026-09-06. Bounded engineering follow-up to the cost-rejected local memory/byte
configuration. The original recipe, cost ceiling, timing margin and evidence
remain unchanged. This document does not authorize a learning experiment.

Observed failure: the fixed complete workload projected to 2,013.103 seconds
against 900 seconds. Its ingestion profile attributes the tagged overhead mainly
to arithmetic/reductions and finite validation, with cloning below 0.5%.

Hypothesis: private reusable tag product/correction arrays and removing repeated
entry checks inside an already checked sequence reduce cost without changing
the local rule. Keep the original elementwise operations and reduction order.
Clone caller-owned tags before mutation, handle noncontiguous restored state,
and retain per-step cell output checks plus full public input/output validation.
No global scratch, trace approximation, changed exposure, or altered parameters.

Comparison: committed allocating implementation at
`6558d7eec02fa53ce127f61893c655bb5d9c2df2` versus the reviewed working source.
Use matched model seeds and input tensors; check exact outputs and complete
local feedback continuation in float32/float64, masked batches and restored
noncontiguous state. Time disposable D32/H64/K64/V64, batch-one 64-byte tagged
and aggregate writes plus complete learner lessons; use one Torch CPU thread,
serial alternating baseline/candidate samples, two warmups and seven samples.
All timing fixtures are discarded and have no capability interpretation.

Completion: focused state/arithmetic/error tests pass and median complete-lesson
time improves without a meaningful aggregate regression. Report the isolated
write and integrated lesson costs separately. Return the implementation to the
shared persistent action workload; a speedup alone does not reopen v1.

Budget: one buffer candidate already present in the working tree, one redundant
validation refinement, at most 90 seconds of timing fixtures and 30 minutes of
implementation/QA. No parameter or architecture sweep. If these fail, preserve
the observation and return to the orchestrator for a fresh bounded decision.

## Result

The bounded comparison completed in 9.820 seconds. Median timings from seven
paired samples after two warmups, one Torch CPU thread:

| Work | Allocating baseline | Buffered/checked-loop candidate | Speedup |
| --- | ---: | ---: | ---: |
| Tagged 64-step write | 138.542 ms | 123.142 ms | 1.125x |
| Tagged complete lesson | 208.715 ms | 190.994 ms | 1.093x |
| Aggregate 64-step write | 45.570 ms | 35.964 ms | 1.267x |
| Aggregate complete lesson | 87.258 ms | 77.384 ms | 1.128x |

The complete lesson includes two-document ingestion, a first attempt capped at
24 bytes, and explicit byte/EOS teaching with upstream feedback and retirement.
It excludes process launch, persistence, motor execution and independent audit;
these figures are not a replacement complete-workload admission. Both arms used
saved public development episode 29 with fresh disposable models for every pair.
All resulting memory/credit, slow/fixed weights, predictions, mean losses and
feedback signals matched exactly. The 36 disposable lessons offered 360 teacher
bytes/EOS; they are engineering exposure, not acquired-capability evidence.

The implementation keeps tag arrays and scratch method-private. At the declared
dimensions, reusable product scratch totals 3,735,808 bytes and correction
scratch 58,372 bytes per active batch stream, beyond cloned credit and other
temporaries. These arrays disappear after the write; persistent state size does
not change. It trades repeated allocation for scratch held across that call.
The private cell step skips repeated entry validation, while public boundaries
and each intermediate cell output retain finite-state checks. Overflow cannot
mutate the caller's state or slow weights.

Focused QA: 46 memory/decoder/composition tests passed in 5.69 seconds, plus Ruff.
The initial noncontiguous parity fixture compared different input strides; the
corrected fixture restores the same snapshot on both sides, then checks exact
float32/float64 continuation through local feedback. Different tensor layouts
may themselves select different floating-point reductions; cross-layout bitwise
equivalence is not claimed. An initial default pytest temporary directory also
hit a Windows permission error; the retained successful rerun uses a unique
checkout-local temporary directory.

Receipts:

- Doctor: `.pytest-tmp/dev/doctor-dkckvnlq/receipt.json`.
- Focused tests: `.pytest-tmp/local-writer-speed-focused-v3.xml`.
- Paired timing source: `.pytest-tmp/local-writer-speed-compare.py`.
- Paired result: `.pytest-tmp/local-writer-speed-compare.result.json`, SHA-256
  `950ec564305c293e435df5ac9611420c66793ce43942c8db774e11faebee107b`.
- Measured `plastic_memory.py` SHA-256:
  `76cdaac279a385f3cec257ff8a37a6680f21df3ed40a51025a40db965339681d`.

The measurable gain satisfies this optimization check but is far below the
roughly 2.2x complete-workload reduction needed by the prior configuration.
Keep v1 closed. Return this implementation to integration; a successor retains
its own prospective cost admission and must not relax the original timing
margin merely because this isolated improvement exists.

## Follow-up: permanently frozen writer controls

Before execution: the measured 1.093x tagged lesson gain cannot plausibly close
the prior cost gap alone. Two declared controls still transport O(PK) writer
credit even though their writer/controller/maps never receive updates.

Hypothesis: use the existing forward writer for those controls and retain
allocated zero credit arrays for schema/accounting. Keep document bytes, cell
dynamics, fast synapses, query/decoder plasticity and exposure unchanged. This
intentionally changes unused writer credit; it is not full-state equivalence.
The all-learning-disabled control has no possible consumer of that credit.
For writer/maps-frozen, resolve the immutable-freezing contract explicitly before
implementation; silently supporting later delayed unfreezing would be unsound.

One implementation only, followed by exact forward-state/prediction and
learned query/decoder parameter checks against current controls. Exercise
continued ingestion, checkpoint restore and allocated tensor accounting.
One serial five-arm timing comparison, at most 60 seconds, with two warmups and
seven samples per arm. No new training roster or extra scientific evaluation.
Completion requires exact observable/learning parity and useful reduced lesson
cost. Return immediately to integration; updating tagged learning is unchanged.

The implemented shortcut applies only to `writer_maps_frozen` and `frozen`.
The first mode now explicitly rejects external writer/controller/map unfreezing
at the shared mode-validation boundary, including ingestion and teaching.
The second mode always disables teaching regardless of parameter flags. The
existing forward writer supplies the exact cell and fast-synapse arithmetic;
returned controller state is purged and credit is allocated as same-shaped
zeros. Incoming state and returned state retain full finite validation. There
is no query/decoder shortcut and no persistent tensor-byte reduction.

The one five-arm comparison finished in 13.002 seconds. It compares the buffered
implementation above with/without dead-credit omission, using the same saved
episode and seven paired samples after two warmups. Medians:

| Arm | Full credit | Omitted dead credit | Speedup |
| --- | ---: | ---: | ---: |
| Tagged | 195.623 ms | 198.858 ms | 0.984x |
| Aggregate | 82.066 ms | 83.184 ms | 0.987x |
| Writer/maps frozen | 198.350 ms | 73.329 ms | 2.705x |
| All slow weights frozen | 186.279 ms | 69.258 ms | 2.690x |
| Writes disabled | 91.238 ms | 93.373 ms | 0.977x |

All 90 disposable lessons had exact fast synapses, purged cell state, first
predictions, teacher losses/signals/counts and resulting slow/fixed parameters.
After explicit credit retirement the full local states also match. Credit
immediately after frozen-control ingestion intentionally differs, so its hashes
are not equivalent evidence. The run offered 900 teacher byte/EOS events and
5,760 document byte/terminator steps; no weights are reused. The three unchanged
arms differ by roughly 1–2% in measured medians, with no changed update arithmetic.

QA passed 53 focused memory, byte-decoder, composition and successor admission
tests, plus Ruff and `git diff --check`. The existing frozen-control test now
requires zero credit explicitly; five new cases compare cumulative ingestion,
snapshot continuation and exact float32/float64 learned query/decoder trajectories
against full-credit references, and reject later unfreezing before mutation.

Final follow-up receipts:

- Tests: `.pytest-tmp/local-dead-credit-focused-v3.xml` (53 passed, 5.47 seconds).
- Timing source: `.pytest-tmp/local-dead-credit-compare.py`.
- Timing result: `.pytest-tmp/local-dead-credit-compare.result.json`, SHA-256
  `ef0cdec9a671508451ed12b4c72d7ce1be8463798fc9371dc7bc60f1850c578d`.
- Measured `local_memory_learner.py` SHA-256:
  `3d4f351fcc046299c77544b251157d99683ec55afa5caa5123e99a5f26b62281`.

This follow-up is useful but still not a cost admission. The candidate sum of
five lesson medians is about 518 ms; even twice 1,024 times this sum exceeds
1,060 seconds before extra capped-prediction and process charges. A successor
must preserve its original ceiling/margin and prospective admission logic.
