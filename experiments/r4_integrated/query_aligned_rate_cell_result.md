# Query-aligned rate cells fail useful acquisition

Completed 2026-09-05 under the [frozen protocol](query_aligned_rate_cell_protocol.md),
source `8e403ba0bb5a8f9f725039e54f301ff491da4c74`. One synthetic cost preflight
passed, followed by one acquisition. No tuning, restart or extra teaching
followed. This is exposed, answer-supervised TRAIN development.

## Outcome and decision

**Close this candidate without adoption.** Direct query access produces no
supported executable successes at the final checkpoint. It does not establish
an advantage from learned cell computation over either trained-head control.
No integrated, fresh, official-exam or protected gate passes.

Each entry below is supported execution out of 48 / correct unsupported STOPs
out of 16. The prefix is a matched arm of this run.

| Arm | Actual before growth | Alternate before growth | Actual after growth | Alternate after growth |
| --- | ---: | ---: | ---: | ---: |
| Trainable aligned cells and heads | 0 / 16 | 0 / 8 | 0 / 15 | 0 / 6 |
| Trainable prefix cells and heads | 0 / 16 | 0 / 16 | 0 / 16 | 0 / 16 |
| Frozen aligned cells, trained heads | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Fixed aligned bypass, trained heads | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |
| Fully frozen aligned evidence learner | 0 / 7 | 0 / 8 | 0 / 8 | 0 / 9 |
| External token similarity | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The frozen-producer arm emits `The` everywhere; the bypass emits `boracite`
everywhere. Its one correct answer does not show request-specific selection.
Growth changes five actual and four alternate treatment decisions from STOP to
incorrect spans, losing one and two correct unsupported STOPs respectively.

The aligned treatment has **one correct conditional span winner out of 48**
in every phase: `lesser water plantain` for `pasir-audit`. Its gate rejects that
answer with STOP probabilities .612, .671, .578 and .641, in table order.
Changing the gate alone could recover at most this one answer with these span
scores. Median gold ranks are 88, 132.5, 89 and 143. Only treatment has saved
conditional-rank diagnostics; these are not matched rank comparisons to controls.

The treatment can now emit multiword spans, but none of its final emissions
executes successfully. One supported first attempt succeeds at teaching step
243; that request's final gold rank is 13 and the gate stops. This transient
success does not establish retained acquisition.

| Treatment loss before each new lesson | Pass 1 | Pass 2 | Pass 3 | Pass 4 |
| --- | ---: | ---: | ---: | ---: |
| Supported conditional-span loss | 8.986 | 8.500 | 7.847 | 7.000 |
| Supported answerability loss | 0.632 | 0.454 | 0.621 | 0.688 |
| Unsupported answerability loss | 1.103 | 1.086 | 0.821 | 0.589 |

Own-offer loss falls on 446/511 aligned updates. These are scheduled pre-update
observations, not a fixed-checkpoint learning curve. Lower loss and occasional
correct choices are insufficient here. They do not isolate forgetting, gradient
interference, query conditioning or feature capacity as the unique cause.

Return to the persistent executed-loop objective through a bounded acquisition
strategy review. Do not extend this configuration's attention, epochs, rates,
thresholds, labels or solver budget. The preceding failed candidates and
inconclusive feasibility diagnostic remain closed.

## Verification and resources

Preparation checks cover independent attention/recurrent gradients, finite
differences, span pooling, masked optimizer state, a genuine tiny Python adapter
round trip, inert runner success/failure paths, and payload traversal through the
actual `python -m` entry. Four focused tests, Python 3.11 compilation and focused
Ruff checks pass. No native code changed; broad native QA was not repeated.

The saved contract audit authenticates 3,066 offers and checks all six arms,
execution receipts, frozen masks, source identities and unchanged evaluation/
growth state. Four learning arms each change parameters on all 511 offers;
the two frozen arms change none. An independent gate calculation confirms failure.
The separate read-only workspace audit checks all **3,330 workspaces and 9,990
Python files**, including two motor demonstrations, against recorded actions,
stdout, untouched decoys and outcomes. Neither audit executes the work again.

**The independent accumulated numerical replay fails its frozen tolerance.**
At aligned offer 269, its STOP logit differs by 2.252e-10, versus an allowance
of approximately 1.099e-10. The failed receipt is preserved, with no tolerance
change or claim that the complete independent trajectory passes.

A separate saved-state diagnosis reproduces all 2,044 active updates using
the exact recorded arithmetic: 15,330 gradient-array hashes and 4,088 before/
after state hashes match exactly. Independent forward/BPTT and local Adam at
each exact state pass `1e-10 + 1e-10*abs(reference)`, with maximum coordinate
errors 2.243e-14 and 1.412e-15 respectively. All trained/final arrays and masked
states match; 3,066 adapter links and 148 input records authenticate. This
supports accumulated arithmetic-order drift rather than a same-state numerical
defect; it does not relabel the original failed replay. The matched prefix's
19 trained arrays also equal the prior prefix result byte-for-byte with identical
inputs, a descriptive reproducibility check rather than an added admission gate.

The cost preflight takes **2.840 seconds**, projecting **320.249 seconds**.
Acquisition takes **336.597 seconds** through checkpoint export, excluding final
JSON serialization/write. Frozen encoder loading and 148 distinct encodings
take 1.001 seconds. CPU, BLAS and encoder execution use one thread.
There are 1,536 teaching attempts, 256 teachers and 1,536 evaluation attempts,
plus two motor demonstrations. Teachers remain separate from learner success.

Peak parent working set is **297.313 MiB**, excluding execution children.
The corrected live Python/NumPy learner-and-motor count grows from **12,451,194
to 13,538,888 bytes**. Adding 90,882,978 encoder-file bytes gives a maximum
counted payload of **104,421,866 bytes**. Independent numerical/raw lower bounds
and actual-entry traversal checks support this accounting; the recorded counts
satisfy the 1 GiB payload and 2 GiB parent limits. Audit objects are separately
reported at 57,028,989 bytes; exported array payload is 21,216,768 bytes.
This fixes successor accounting without changing the preceding invalid receipt.

The system retains raw sources and frozen external word vectors, receives
explicit answer/STOP supervision, and shares the previously taught GLIF literal-
edit motor. The 64 rate cells and attention learn through conventional BPTT and
Adam. Controls isolate evidence computation; they do not remove the shared
neuronal motor. This is not answer-free internalization, autonomous recovery,
source-free memory or independent language generalization.

## Artifact custody

- [Cost preflight](../../benchmarks/query_aligned_rate_cell_preflight_v1.json),
  91,842 bytes, SHA256 `e0ea70ac1075dd898a1b2f21c9f5b9b669bcb6ea542ada0f7a4438a4c0ff4b71`.
- [Acquisition record](../../benchmarks/query_aligned_rate_cell_acquisition_v1.json),
  25,665,676 bytes, SHA256 `338e17e2f61d8965f787822a2e7942263aa5c09782fd29ddb4f9ac3aab088148`.
- [Inputs and initial/trained/final states](../../benchmarks/query_aligned_rate_cell_acquisition_v1.npz),
  11,357,023 bytes, SHA256 `7b07e0796cd6bc3b4218aeeff6dc6088280ad460402edec7f812063911a868bb`.
- [Scoped audits and numerical diagnosis](../../benchmarks/query_aligned_rate_cell_acquisition_v1.audit.json),
  1,635,509 bytes, SHA256 `76cd6ed8a19bfea5c984f54c4021296a31671376321748254cf9108b06c36386`.
  Includes contract, workspace and behavioral checks, the failed accumulated
  replay, separate passing same-state diagnosis, and their source/receipt identities.

The records bind source/protocol, fixture, encoder and native motor identities.
Source bytes stay unchanged during acquisition. Earlier frozen source, results
and their explicit limitations remain intact. Saved labels and ledgers are
audit material and do not enter inference.
