# Projected bilinear features regress executed transfer

Completed 2026-09-05 under the [frozen protocol](contextual_projected_binding_protocol.md),
source `dd0e59f11f8d01de07619b07fa3036903c9af5d7`. The one comparison completed in
537.1451703 seconds with exit 0, without interruption, retry or prefreeze model
preview. Preparation began at 06:45 UTC; reviewed source and fixtures were
committed before execution at approximately 07:05 UTC. Both budgets were met.

## Decision and behavior

**Reject PROJECTED.** Retain the existing AROW/BINDING architecture with RESERVED
allocation and affine actions. This experiment's identical mixed-length teaching
in both arms does not replace the earlier curriculum decision. Its new data and
questions also prevent a causal comparison with historical scores.

| Primary affine outcome | BASELINE | PROJECTED |
| --- | ---: | ---: |
| Fresh supported execution | 4/8 | 1/8 |
| Familiar / broader fresh execution | 2/4 / 2/4 | 1/4 / 0/4 |
| Common-query retained execution | 8/16 | 4/16 |
| Correct unsupported STOPs | 0/2 | 0/2 |
| Complete fresh / retained entities | 0/4 / 2/8 | 0/4 / 1/8 |
| Fresh whole two-word targets | 1/4 | 0/4 |
| First-attempt teaching execution | 10/48 | 16/48 |
| Actual taught-form postteaching selections | 9/16 | 7/16 |

PROJECTED gains no fresh task and loses three: jrev transport (`axinite` becomes
`the transport setting`), klym transport (`sweet briar` becomes `sweet`), and lorn
transport (`danburite` becomes the old value `galena`). Retention has two gains
and six losses. Neither relative adoption nor any absolute/integrated gate passes.
These are one small prospective authored development comparison, not independent
reading or general software-engineering evidence.

## What failed

At unchanged trained state, switching actual teaching questions to common forms
loses two correct answers and gains one for BASELINE (9/16 to 8/16), and loses
three for PROJECTED (7/16 to 4/16). Subsequent fresh ingestion changes neither
correctness set, for actual or common forms. Some already-wrong leaders change;
there is no correct-answer displacement in this run. These raw probes are
diagnostics, not additional executed successes.

| Learning evidence | BASELINE | PROJECTED |
| --- | ---: | ---: |
| New authentic comparisons / replay offers | 38 / 112 | 32 / 118 |
| Native updates / gate no-ops | 145 / 5 | 149 / 1 |
| New / replay native updates | 38 / 107 | 32 / 117 |
| Final negative / positive subunit / unit pair margins | 1 / 33 / 4 | 0 / 30 / 2 |
| Minimum final original-pair margin | -0.0141563 | 0.190545 |

BASELINE-35 falls from +0.0691065 after its last offer to -0.0141563, establishing
one erased contrast. Its rejected occurrence is not the final winning competitor.
All seven BASELINE and nine PROJECTED wrong actual-form leaders were previously
unexecuted and unadmitted occurrences for those exact questions. PROJECTED's
positive admitted margins therefore coexist with substantial acquisition and
competitor-coverage failures. Common-query joins are zero because all sixteen
question forms differ; the facts were taught.

Disabling PROJECTED's new block yields 2/8 fresh and 2/16 retained; keeping only
its projected-coordinate diagonal yields 3/8 and 3/16. These read-only diagnostics
show dependence and tradeoffs, not permission to adopt another variant. Neither
arm acquires all taught forms, so this result does **not** isolate a general
binding/representation limit. Close the candidate without a rank, projection,
wording, length, epoch or automatic AROMA successor. The next design review must
separate better learning over competing answers from a trainable token/role
representation, with explicit credit, replay and abstention supervision.

## Controls and verification

Both arms receive 48 encounters, 48 separately executed passing teacher repairs
and 150 offers. Teachers do not count as learner success. All 624 executions
across eleven BASELINE and thirteen PROJECTED policies preserve selected edits,
decoys and source absence from live action input. All 52 primary raw/affine
occurrence choices and outcomes agree. Frozen affine and external cosine solve
no supported task; frozen affine correctly stops on both unsupported requests.
The two primary learners act wrongly on both.

Each frozen mate receives 300 feedback events with zero synaptic/covariance
mutation. Fixed projections match the registered QR recipe exactly and remain
unchanged. Controls share the previously learned motor, so they isolate evidence
learning rather than the entire neuronal core. Frozen MiniLM, internally retained
raw documents and external correction ledgers remain part of the declared hybrid.
Live-source removal is not semantic compression.

The unchanged official KS2 Q6a/b helper runs once per arm without feedback.
Each primary arm has zero confirmed marks and two pending official-rubric human
judgments, not finalized 0/2 scores. Repeated items are regression observations.

Before freezing, all 60 core tests, exact default-policy parity, fixture semantics
and inert routing/export/mutation/failure checks passed. Unchanged native sources
justify reuse of ten FFI checks and both 278-test Rust suites. Ruff passes.
Independent saved-data audits authenticate all 38 source hashes, all 600
updater/frozen offers and all sixteen exported arrays. Exact state reconstruction
matches byte-for-byte; all 64 projected original eligibilities reconstruct with
zero error. Recorded gap error is zero; largest own-formula error is 6.22e-15,
and independent-reference errors are at most 6.81e-16 for weights and 1.34e-15
for covariance. No ambiguous unit gates occur. Symmetry is exact; final Cholesky
checks pass without repair (minimum updater diagonals 0.0417835 / 0.141587).
Audits use saved inputs and do not regenerate encoder preimages or rerun models.

## Cost and artifacts

| Resource | BASELINE | PROJECTED |
| --- | ---: | ---: |
| Teaching / transfer plus retention / exam | 156.00 / 52.14 / 14.40 s | 203.07 / 74.22 / 15.97 s |
| Covariance bytes per updater or frozen mate | 21,021,128 | 55,968,200 |
| Covariance scratch bytes | 12,968 | 21,160 |
| Fixed projection bytes per learner | 0 | 196,608 |
| Final retained array/byte payload per learner | 29,670,977 | 64,847,425 |
| Compact external ledger bytes | 2,669,145 | 3,644,952 |

Each final learner retains 11 documents, 2,742 candidates and 5,819 raw source
bytes, including the exam. Learned covariance is the same allocated matrix,
not additional payload. Ledger eligibility bytes (985,568 / 1,354,240) overlap
its serialization. The shared encoder occupies 90,882,978 bytes. Native
covariance-validation entries are 394,146,150 / 1,049,403,750; matrix-vector entries
381,007,945 / 1,042,407,725. These are scoped loop sizes, not total operations.
Covariance is unused at inference.

Total wall time includes child waits. Parent peak working set is 668.65 MiB,
ending private memory 860.99 MiB, and traced Python peak 461,022,399 bytes.
Memory excludes children and includes evaluator/export copies; do not sum
overlapping scopes. Sequential phase times are not a causal speed comparison.

- [JSON](../../benchmarks/contextual_projected_binding_v1.json): 13,910,056 bytes;
  SHA-256 `ea5ae15b8c19b09c5631d7d7b883c23998b3c5f45d0af0f46e7706973400d6a8`.
- [NPZ](../../benchmarks/contextual_projected_binding_v1.npz): 51,613,598 compressed
  bytes, 154,508,448 array bytes; SHA-256
  `42d1ab54b603d5642447a79315f3a05af32028c9ed8d8c5fd075f5ba15de2fdf`.
- Native source digest `7125922510b018089744714d6b99748a20ad6e768d5ab6480fc6b883f5285cf5`;
  DLL SHA-256 `21fe924ade8b88662b21398547a1c2c79771d25f6565d522c322afd7ffb64917`.

Preserve this result and protocol unchanged. The roadmap owns the next priority;
integrated and protected gates remain closed.
