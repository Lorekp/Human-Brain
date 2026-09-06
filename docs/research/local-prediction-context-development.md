# Forward-only prediction context: bounded development

2026-09-06. V4 closed after strict saved-evidence audit at 995.972 seconds
projected against 900 seconds, with a 774.773 MiB peak. Prior recipes, results,
the five-arm workload, timing margin and ceilings remain unchanged.

Observed dead work: read-only prediction calls the teaching query path, creating
input/recurrent/bias eligibility for each query byte and query-map eligibility,
then discards those arrays. Only final query activity and the memory read value
reach the byte decoder. The prior full-lesson profile identified both query
traversals as a remaining shared cost; no new profile is authorized here.

Before implementation, the explicit design decision is to create a private
prediction-context path using the same forward cells and memory read, omitting
only never-consumed query eligibility. Preserve full validation of supplied
retained memory/cell/credit state, raw bytes and encoded inputs, and finite
validation after every forward cell step and the final read/context. Preserve
read-only parameters, synapses and caller state; retain no query, cached answer,
history or autograd graph. Teaching continues to use the existing local query
path with unchanged traces, learning rules, clocks and byte exposure.

**Declared error-behavior change:** prediction may now succeed when only the
previously allocated but never-used query eligibility would have overflowed,
while every actual forward activity/read/output is finite. Those eligibility
arrays are no longer materialized. Do not claim universal exception parity.
Nonfinite supplied state, encoded input or forward activity/read remains invalid.

One candidate implementation and focused checks only. Require exact contexts,
generation races/bytes/EOS and later teacher updates on finite-forward fixtures,
all five controls, caller-state/RNG immutability, nonfinite-boundary rejection,
and an explicit finite-forward/overflowing-unused-trace diagnostic. Integration
owns the existing decoder/composition test file; this lane owns learner code
and this document.

After focused QA and source review, run one serial paired five-arm complete-
lesson comparison against the previous source, two warmups and seven samples
per arm, with at most a 60-second watchdog. Preserve hash/snapshot guards and
all lesson records except elapsed time; compare exact learned parameters.
No fit, new scientific roster, native fusion, configuration/budget sweep or cost
admission. Return the measured candidate to the orchestrator before considering
any successor full-workload cost run.

## Result

The new `_prediction_context` calls the same cell `_step` arithmetic directly,
matching the prior local step's hook behavior. It validates encoded inputs and
supplied retained state, checks each forward activity, uses the existing memory
read, and checks the resulting context. `predict` consumes this context;
teaching `_query` and every local feedback update remain unchanged. The output
is an independent temporary tensor with no retained graph or query state.

Combined QA passed 89 tests in 6.07 seconds. Twenty new integration cases include
exact float32/float64 contexts, predictions and two subsequent teacher-update
trajectories across all five modes; read-only state/weights; same-query versus
different-memory causal output; invalid supplied state, encoded input, transient
activity, memory read and logits; and the declared unused-trace overflow change.
The last diagnostic holds query activity at zero with recurrence 3, then uses
130/1,030-byte queries to overflow unused float32/float64 eligibility. Forward
prediction stays finite; the unchanged teaching path still rejects the overflow.
This confirms the scoped difference, not universal exception equivalence.
Ruff and `git diff --check` also pass.

One serial paired comparison completed in 8.168 seconds. It called the actual
v4 shared harness lesson, including precise duration clocks, hash/snapshot
guards, 32-byte generation cap and 0.01 teaching rate. The baseline was source
commit `e25bb2a9901a96e832a4018beac967ddc4882d6a`; each pair started matched
fresh disposable models on saved episode 29. Two warmups and seven measured
samples per arm produced these medians:

| Arm | Prior prediction | Forward-only prediction | Speedup |
| --- | ---: | ---: | ---: |
| Tagged | 128.459 ms | 126.283 ms | 1.017x |
| Aggregate | 63.419 ms | 57.696 ms | 1.099x |
| Writer/maps frozen | 53.545 ms | 44.874 ms | 1.193x |
| Fully frozen | 48.853 ms | 41.655 ms | 1.173x |
| Writes disabled | 58.261 ms | 49.823 ms | 1.169x |

The five-arm median sum falls from 352.537 ms to 320.331 ms (1.101x). All 90
lesson records matched exactly except elapsed time, including prediction bytes,
EOS/limit decisions, losses, mutation flags, before/after and memory hashes,
and complete resulting parameter tensors. Disposable exposure was 5,760 document
byte/terminator steps and 900 teacher byte/EOS offers. No model is reused as
capability evidence. Query bytes are still processed twice per taught lesson;
this change removes unused arithmetic within the read-only traversal.

Receipts:

- Doctor: `.pytest-tmp/dev/doctor-uk___txh/receipt.json`.
- Combined QA: `.pytest-tmp/local-prediction-combined-v1.xml` (89 passed).
- Timing source: `.pytest-tmp/local-prediction-context-compare.py`.
- Timing result: `.pytest-tmp/local-prediction-context-compare.result.json`, SHA-256
  `c038aa56bb9cda4f3b99c9eb0b44bf2c169b7dd44fc7980aae96a1b68cfcea0b`.
- Measured learner source SHA-256:
  `24e2f57072e90309d4ae51822b0b20f084cc9013e01355bc0a393be332462e35`.

The candidate passes its bounded implementation/performance check and returns
to integration. This measurement excludes a full source-free worker/motor run
and uses paired medians, so it does not establish cost admission under the
successor's maximum-sample, capped-prediction and process-overhead formula.
