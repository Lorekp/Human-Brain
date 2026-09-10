# Integrated local learning implementation verification

The [selected local revision](integrated-local-learning-revision-v1.md) is
implemented. Three focused tests passed on their first launch on 2026-09-10.
This verifies authored mechanics and copying, not acquisition or migration of
the retained working learner. No historical learner payload was decoded and
no additional integrated teaching was performed.

## Verified scope

- **Local arithmetic:** one controller, nine ticks, 34 actual draws, five scan
  offers and two teacher offers. An independent per-event/pairwise oracle
  verifies Brier utility, causal prefix means, scan covariance with varying
  activity magnitudes, zero/one-contact cases, uncentered teacher associations,
  unchanged public credit, full-support delta norms and the new admission bound.
- **Same actor lifecycle:** two authored actors, eight successful source offers,
  two offers rejected before publication, 105 public scan offers, three free
  responses, four teacher symbols and one real child execution. Checks cover
  disabled publication, exactly-once resolution, pending/final NumPy mutation
  rejection, joint atomicity and strict metadata validation.
- **Initializer and probe copying:** three authored actors and two source
  offers with learning disabled. The actual new copy and probe-clone functions
  preserve all numerical tensors, RNG, facts and authored counters. Original,
  payload, copy and clone have disjoint storage, including derived views. Only
  the three declared algorithm metadata dictionaries change. There are no
  tasks, teacher calls, child executions or payload decodes in this fixture.

These tests use the same acting classes and local publication path. They contain
no backpropagation or reverse weight transport. A correct authored STOP/execution
fixture does not demonstrate a learned edit. Preservation of the actual retained
20:24 motor behavior still requires a paired, bounded forward measurement before
the next integrated learning comparison.

## Receipts and costs

The [compact QA bank](../../benchmarks/integrated_local_learning_v1.qa.json)
has SHA256 `e4a4457fe0d369955793d5b7ce1e00f1a233d0e64c8b685c7cd1894bc8a7d6db`.
It records exact source hashes, scalar case summaries, full receipt identities
and the process monitor. The run used doctor
`.pytest-tmp/dev/doctor-qz4jmpnv/receipt.json`, with a 90-second/2,048-MiB cap.
Pytest reports three passes in 25.61 seconds; the outer process took
26.8182533 seconds at 1,099.453125 MiB observed tree peak. These nested times
must not be added. The pytest cache write was denied; the tests and their
receipts completed successfully.

The exact predecessor controller and learner source copies total 82,126 bytes;
source preparation took 0.0962958 seconds and constructed no model. They remain
under `.pytest-tmp/integrated-local-learning-v1-predecessor`. The new initializer
authenticates the completed parent's sources, journals and payloads through the
historical inspector, with exactly those two pinned source aliases. That actual
parent initialization path is implemented but has not yet been executed.

Full QA artifacts, totaling 22,778,595 bytes, are retained under
`.pytest-tmp/integrated-local-learning-v1-qa-artifacts`. Copying and compact-bank
preparation through serialization start took 0.157927 seconds; final bank writing
and hashing are outside that timer. The first copy preflight rejected a 16-MiB
receipt cap before writing anything; inspection established the actual size and
the subsequent copy used a declared 32-MiB cap. No tests were repeated.

Affected Ruff checks pass. Their first full invocation found one unsorted test
import block; sorting that block was the only post-test code edit. The bank
distinguishes the tested test-file hash from its final hash. Controller, learner
and initializer execution bytes were unchanged after the passing run.

## Next decision

Keep the architecture, difficulty and retained state fixed. The revised learning
path remains behaviorally unmeasured. Its per-contact step is still small, so
do not automatically reuse the previous 32-correction ceiling or launch another
micro-gate. Declare an adequate finite teaching/resource budget and matched
predecessor, disabled and simple-reference comparison before execution. Measure
effective contacts, intended-output probabilities, complete free responses and
interference. Expanded retention and changed-context work remains paused;
teacher-free recovery is still unestablished.
