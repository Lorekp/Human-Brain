# Local memory/byte v5: inference without unused query credit

Status: prepared 2026-09-06; cost and acquisition unrun.

V4 closed at a 995.972-second cost projection against its unchanged 900-second
ceiling. Its receipts and earlier frozen experiments remain preserved. A
read-only prediction still constructed query-cell and query-map eligibilities
that generation never consumed. The reusable hypothesis is that omitting this
work admits the same learning/action workload within the original budget.
The [bounded implementation comparison](../../docs/research/local-prediction-context-development.md)
measures its engineering effect; it is not acquisition or cost admission.

## Exact forward and teaching contracts

The new prediction context executes the same byte encoding, query-cell forward
arithmetic and fast-memory read, without constructing transient query credit.
Supplied persistent state, including retained credit, remains fully validated.
Actual encoded inputs, query activity, memory read/context and generated logits
must remain finite. Predictions retain no query state, cache or learning graph
and cannot update weights or persistent state.

This deliberately changes one unused-computation failure: prediction no longer
rejects a finite forward trajectory solely because a hypothetical query
eligibility would have overflowed. For example, zero query activity can remain
finite while an unused bias eligibility doubles each step. This is an explicitly
different inference error contract, not universal exception parity. Teaching
still constructs the original local query eligibilities and rejects their
nonfinite values before updating. No teaching arithmetic, signal, trace,
retirement, exposure or feedback timing changes.

Require exact normal-workload contexts, bytes, state, teacher losses/signals and
resulting parameters against the preceding query path. Include all five controls,
float32/64, read-only state, source-free restore, actual nonfinite-input/state/read/
logit rejection and same-query/different-memory causality. A parameter-authored
causal routing fixture is transport QA, not evidence that the learner acquired
an answer. No new teaching or evaluation cohort is introduced.

## Unchanged workload, budget and claims

The complete [v3 workload contract](local_memory_byte_v3.md) and
[v4 timing contract](local_memory_byte_v4.md) apply unchanged. The thin v5 wrapper
uses the existing shared harness, with its own identity, source manifest and
fresh admission. Seed 733, dimensions, rate 0.01, 1,024 sequential lessons per
each of five arms, full raw-document ingestion before questions/targets,
33-decision prediction cap, twenty final checks per arm, 100 real actions and
two motor demonstrations remain fixed. Teaching schedule SHA-256 remains
`5aa2b390194d28f8f58e8f9cc8ce0c0f1e0b6721ba848177101448b5564a14ac`.

Allow one fresh disposable cost phase <=900 seconds total and, only after
positive cost admission plus saved-only audit, one acquisition <=900 seconds.
All phase attempts count; memory limit remains 2,048 MiB. Discard cost weights.
Keep the exact original estimate:

```text
2 * sum_arm(1024 * (maximum complete sampled lesson + capped prediction))
  + 2 * (outer cost elapsed - sum all sampled lessons) + 30 seconds
```

Use strictly positive measured samples, no timing floor, `perf_counter` phase
durations and matching `duration_clock` metadata. Preserve Unix deadlines and
the outer process-tree memory/time limits. Do not reuse or reinterpret v4 costs.
Cost rejection, nonfinite learning, failed fit or missing useful advantage
closes v5 without automatic extra seeds, rates, lessons or relaxed margins.

Require tagged 8/8 fit, 4/4 new-value, 4/4 delayed and 4/4 interference exact
executed outputs and strict new-value advantage over writer/maps-frozen, frozen
and writes-disabled. Report aggregate superiority separately for tag-specific
attribution. The local decoder supplies bytes; the existing host-authored
occurrence motor remains a disclosed action scaffold. No broad neuron-only
action selection, prose comprehension, execution-feedback learning, recovery,
integrated-gate or protected coding claim follows.

## Launch

Prepare admission from v5 `SPEC.config`, `harness.sources(SPEC)`, current revision,
Torch/NumPy/native identity, the unchanged schedule hash and
`harness.duration_clock()`. Use the v3 recipe's exact launcher command with:

```text
admission: .pytest-tmp/local-memory-byte-v5.cost-admission.json
output/log base: .pytest-tmp/local-memory-byte-v5-cost
module: experiments.r4_integrated.local_memory_byte_v5
schema: local-memory-byte-v5
```

Launch `cost` with 900 seconds and 2,048 MiB in the prepared native environment.
After a passing audit and recomputed projection, write the hash-bound `admitted`
record and launch `run` with fresh `local-memory-byte-v5-run` paths. The unchanged
harness reports counts and elapsed time every 128 episode indices, with no
extra evaluation and all output overhead counted.
