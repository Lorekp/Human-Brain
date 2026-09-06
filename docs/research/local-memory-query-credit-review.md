# Instantaneous query-credit repair: independent preflight

Status: source review, 2026-09-06. This selects one six-arm integrated comparison;
it does not report a new fit or qualify document understanding.

The frozen localization evidence supports testing the missing source-dependent
query error, rather than reversing an apparently wrong update sign. Both source
assignment contrasts survive into memory and reads, but the two questions respond
largely in the same direction despite needing opposite answer changes. All saved
positive writer/address proposals lower loss and all negative proposals raise it.
See the [localization follow-up](local-memory-localization-followup.md) for the
authenticated evidence and the limits of this inference.

## Reviewed mathematical contract

For fixed retained memory `M`, query activity `u`, old query map `Q`, drive
`a = Q u + b`, norm `n`, normalized key `q`, and read `r = M q`, the selected
neuronal return is `e_q = M.T g_r`. For `n > 1e-8`, transport through normalization
is `e_a = (e_q - q (q.T e_q)) / n`. At and below the clamp it is `e_q / 1e-8`.
Equality deliberately uses the clamped branch; this is a declared boundary
choice, not a claim of universal agreement with an autodifferentiation convention.
The query-map update uses this signal with raw presynaptic `u` and unit bias
eligibility. It must not multiply a second diagonal normalization gain. The
query-cell teaching signal is `g_u + Q_old.T e_a` with its existing self-path
recurrent eligibility. No recurrent history is differentiated.

The implementation computes both transports and their finite checks before
publishing any memory/query-map update. Consequently the indirect query-cell
signal uses the old `Q`. The LMS output still computes each phase's loss signal
with that phase's pre-update output weights, sums signals in float64, divides by
the three teacher offers in float64, then performs one checked conversion to the
upstream dtype. The sign remains loss-gradient descent. No learning-rate factor
or output-count divisor is inserted into the transported error.

This is explicit symmetric instantaneous weight transport through two forward
connections, followed by local eligibility products. It is not BPTT, an external
per-synapse oracle, biological locality proof, or an exact gradient through the
learned document-writing history. Writer error broadcasts and writer/query-cell
self-path approximations remain unchanged. The original e-prop distinction
between neuron learning signals and local eligibility is useful terminology, but
does not establish correctness or convergence of this custom memory rule.
([Bellec et al., original paper](https://www.nature.com/articles/s41467-020-17236-y))

## Implementation boundaries

`LocalMemoryRead` now stores the norm and raw query-map traces. Its snapshot and
restore schema changes explicitly; old experiment checkpoints remain reproducible
at their original commits. Inactive queries use masked zero inputs and traces.
The production call holds memory and query weights fixed between the read and
transport. The event validator checks shapes, dtype, finiteness and nonnegative
norm; it is not a stale-event provenance certificate for arbitrary external calls.

With writes disabled, `M` is zero, so the indirect signal and query-map update are
zero; direct query-cell and LMS learning may continue. Frozen mode still computes
the declared signals but publishes no updates. Writer-frozen mode preserves the
writer maps while allowing the repaired query path. The added norm check can
reject a nonfinite norm even if an older forward happened to return a finite zero
key. Preserve this stronger validation without claiming universal exception parity.

Finite proposal checks are per publication group: the memory group is atomic,
then the query-cell group is validated/published; earlier LMS byte updates may
already exist. This does not make the whole lesson transactional. Repeated state
and event validation is retained for correctness; assess its cost in the complete
disposable workload before considering a separate exact optimization.

## Matched comparison and evidence custody

The six arms are repaired tagged, aggregate, writer-frozen, fully frozen,
writes-disabled, and experiment-only `current_tagged`. The latter preserves the
old learner and old read-event/diagonal-eligibility methods in a source-bound
reference module. Inheriting the new raw-trace read method would invalidate the
comparison. Its inherited unchanged methods and actual trajectories require
parity QA against the frozen source; runtime experiment execution must not load
Git source or monkeypatch production classes. Only its old fixed `query_feedback`
buffer differs in initial tensor membership; every common initial tensor is exact.
Production has one rule and no compatibility mode.

Keep the 1,024 lessons, fixed seed/rate, raw-document schedule, twelve retained
memories per arm, same-question `ac/ca` and `bd/db` swaps, read-only delays,
unrelated-write interference and source-free real Python actions. Qualification
requires tagged 8/8 fit and 4/4 in all three subsequent strata, plus strict new-value
superiority over writer-frozen, frozen, writes-disabled and `current_tagged`.
Aggregate superiority is separately reported. The exposed development cohort is
not an independent fresh benchmark. The host motor still supplies the action
schema; correct answer bytes do not establish learned action selection.

Before any full fit, a fresh six-arm cost run offers 24 first attempts, 72 teacher
steps, 60 enabled output-update opportunities, 120 actions and two demonstrations.
The original positive-duration projection is unchanged: twice the summed
1,024-lesson per-arm maximum complete-lesson plus additional capped-prediction
cost, twice the remaining measured cost-run wall time, and 30 seconds. The ceiling
is 900 seconds and 2,048 MiB. No prior admission transfers. Saved-only cost and
full auditors bind their own hashes in admission, all imported helper sources,
native identity, clocks, exact launcher/worker arguments and owned process
identities; reconstruct schedules, counts and checkpoint chains; verify source-free
state and action files; and confirm the full fit starts from fresh seed-identical
initial weights, not discarded cost-trained weights. They authenticate receipts and arrays without re-executing the learner
or independently recomputing a local gradient trajectory.
