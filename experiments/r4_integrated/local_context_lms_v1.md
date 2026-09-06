# Context LMS v1: local learning of all output potentials

Status: prepared 2026-09-06; full comparison unrun.

The closed integer reference learns all TRAIN winners but fails one or two
withheld compositions. Once TRAIN winners are correct its mistake rule cannot
change the remaining phase/opposite-factor weights. The
[composition review](../../docs/research/local-composition-return-review.md)
selects this one new potential-error rule. Preserve the old reference and
decoder experiments; neither is rerun here.

## Fixed model and prospective endpoint

Use fresh `PhaseContextLMS()` models: CPU float64, all 257×291 output synapses
initialized to zero, three supplied clock neurons, and continuous multiplication
neurons for every phase and every one of 96 context rates. Features are
`[phase, phase * context]`, treating all coordinates uniformly. Binary authored
contexts have the same two active factor units and 88 zeros as the earlier
diagnostics. The model also accepts finite signed continuous rates; they do not
inherit the binary-input proof. No fake embedding or upstream context teaching
interface is added in this component comparison.

At every teacher byte/EOS offer form all 257 preupdate residuals
`r = onehot(target) - saved_potentials`, then update each output synapse by
`0.01 * r_post * feature_pre`. Correct winners still receive this update.
The teacher supplies one desired activity for each output neuron; it does not
select features or provide externally computed per-synapse changes. There is no
softmax, output mask, optimizer, BPTT, clipping, decay or hidden fitting step.
Loss is `half_sum_squared_potential_error`: half the **sum** of squared residuals
over all 257 outputs, distinct from cross-entropy and the old 0/1 mistake loss.

The review's real-arithmetic derivation uses the TRAIN graph and its spectrum,
not a learner fit or missing-combination probe. With rate 0.01 it gives the
per-cycle squared-error contraction bound `q = 0.9684483676583155`; after 160
complete cycles the vector-error bound is 0.3997602429, implying an ideal
correct-versus-wrong margin above 0.2004795141. No proof separator, covariance
inverse or fitted target map enters runtime. Float64 execution is rounded:
this prospective proof is **not** a machine-arithmetic certificate. Measure
actual outputs and margins, and label any saved-weight residual verification
separately as a posteriori external analysis.

Fix **160 cycles, 1,920 lessons per arm**, rate 0.01, and a two-byte cap plus
one EOS decision. There are four fresh arms in this exact per-index order:

| Arm | Context | TRAIN order | Teaching |
| --- | --- | --- | --- |
| lexical | Correct authored rates | Lexical | Enabled |
| permuted | Correct authored rates | Private `Random(734)` permutation each cycle | Enabled |
| zero | Zero rates, clock retained | Lexical | Enabled |
| frozen | Correct authored rates | Lexical | Disabled; all residuals offered |

Experiment seed 733 is an identity convention; models initialize to zero without
RNG draws. Only stream shuffling uses RNG. Every cycle offers the same twelve
TRAIN pairs exactly once. `ab,bc,cd,da` remain excluded from TRAIN and permutations
but are **previously exposed development probes**, not a fresh assessment.
Serialize both complete streams and all context vectors before constructing
models or offering labels. First generation precedes all three teacher positions;
teacher clock traversal continues through byte1, byte2 and EOS even if the first
free attempt stopped early. No adaptive ordering or success-based stopping.

## One final free call per probe, with the actual potentials

After all 1,920 lessons, save final state and generate once for each of twelve
TRAIN and four withheld pairs per arm. There are no intermediate evaluations.
The runner's free generation traverses the model's `local_step` clock once and
records the exact potentials used for competition at each visited phase. No
target is supplied to that function. Learned EOS or the byte cap stops it;
neither bytes nor EOS are repaired.

Compute each final expected-byte/EOS margin from those **same recorded free-call
potentials**: target potential minus the largest competing output potential.
An absent phase has a null margin and fails; never add a teacher-prefix traversal
to fill a missing phase. Correct arms qualify individually only with 12/12 TRAIN
and 4/4 withheld exact bytes plus EOS, strict withheld superiority over both zero
and frozen controls, and **every one of their 16×3 final margins strictly >0.10**.
Report all outputs, stops, margins and controls even on failure. These are
measured numerical criteria, not proof of floating convergence in other settings.

## Counts, custody and budget

Totals are **7,680 first attempts, 23,040 byte/EOS label offers, 5,921,280 scalar
output-neuron residual offers including frozen, 17,280 enabled teacher steps,
4,440,960 enabled scalar residual offers, and 64 final free generations**.
Count actual changed steps, nonzero residual entries (all and enabled) and
generation decisions separately. Sparse logging is storage compression only:
every output participates in residual computation and updates, including zeros.

Each free-call phase records its winner and every nonzero potential and feature
as index/value pairs. Every unlisted vector entry means exact zero, with no
thresholding. Teacher receipts additionally record target, phase, all nonzero
preupdate residuals, half-sum loss, offered counts and changed flag. These records
permit checking all 257 desired activities and the winner without another model
call. The teacher target never affects the preceding free call.

Retain `schedule.json`, `admission.json`, all four complete `initial.pt` and
`final.pt` states, all lesson rows in `lessons.jsonl`, `report.json` and a
hash-bound completion. First/after-first/after-teacher hashes link lessons;
final free calls must be read-only and frozen state identical to initialization.
Record source/revision start and end, full model/fixed/step accounting, artifact
sizes, clock and dependency identities. The source manifest includes the frozen
v1 helper closure, this runner/recipe, the new LMS model and the mathematical
composition review, not mutable roadmap/status documents.

Allow **one complete comparison and saved-only audit within 120 seconds and
2,048 MiB**, including imports, sparse logging, hashing, snapshots and audit
startup. The root launches the learner under the existing watchdog, then launches
the independent saved-only auditor under the same watchdog with a time limit of
`floor(120 - measured_outer_run_seconds)` and the same 2,048 MiB ceiling. A
nonpositive remainder closes the attempt without an audit launch. This measures
audit startup, elapsed time and its owned process-tree peak. Require the sum of
both measured outer process durations to stay within 120 seconds; orchestration
gaps are not computation time. Report each process's time/peak separately and
do not sum nonsimultaneous memory peaks. Durations use the performance counter
with its metadata; Unix
deadline checks remain separate. Budget exhaustion or failed qualification
closes the attempt with no automatic extra epochs, rate, seed, margin or variant.

An independent saved-only audit checks all sparse-vector semantics, residual
identities/losses, source/artifact hashes, initial equality, state chains,
frozen/read-only behavior, margins, counts and launch ownership. It may evaluate
exact rational sums of saved TRAIN coefficients with a declared conservative
floating summation envelope as external evidence; that analysis must never feed
training or add a model evaluation. No numerical theorem for arbitrary memory
inputs follows from this binary-context check.

```powershell
$lmsOutput = '.pytest-tmp/local-context-lms-v1-run'
$lmsRevision = (git rev-parse HEAD).Trim()
$lmsSources = (& ./.venv/Scripts/python.exe -c 'from experiments.r4_integrated.local_context_lms_v1 import digest,sources; print(digest(sources()))').Trim()
./experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1 -Seconds 120 -MaxMiB 2048 -LogBase $lmsOutput -ChildArguments @('-X','utf8','-m','experiments.r4_integrated.local_context_lms_v1','--output',$lmsOutput,'--revision',$lmsRevision,'--sources-sha',$lmsSources)
```

Review and commit source before freezing this command. Hash-only preparation
constructs no model. Focused QA must use disposable small contexts/lessons, not
fit the full twelve-pair stream. A useful result earns one separately admitted
return to raw-document memory and actual executed actions with two-byte values,
removing the answer-revealing context. Upstream feedback and input codes need an
explicit interface there; this diagnostic is not reading, source-absent learning,
general output timing, learned action selection, coding ability or a gate pass.

## Preparation QA

Doctor passes at baseline `7c120e6`, receipt
`.pytest-tmp/dev/doctor-dwa2061l/receipt.json`. The existing decoder suite passes
**85 tests** in 6.30 seconds, receipt `.pytest-tmp/local-context-lms-v1-qa.xml`.
Review then strengthened the new update expectation to independent Python scalar
arithmetic on each affected synapse, with equality of all untouched entries.
The six new LMS cases pass again in 1.58 seconds, receipt
`.pytest-tmp/local-context-lms-v1-scalar-qa.xml`; 79 other cases are deselected.
Four unchanged new cases repeat in that selection and their exposure is counted.
No model arithmetic or experimental source was changed to obtain these passes.
Ruff, syntax compilation and diff checks pass; independent source review passes.

The new cases cover float32/64 signed inputs and uniform phase gating, exact
preupdate residuals including learning after a correct winner, frozen teaching,
stale/overflow/nonfinite rejection, saved sparse residual accounting, one-call
free margins and missing-phase failure, initial/checkpoint transport and the
fixed 160-cycle schedule without training it. Across both invocations these new
fixtures make 20 accepted disposable label offers (5,140 scalar residual offers,
2,570 enabled), six rejected teacher calls, ten successful read-only generation
calls and six rejected nonfinite/fixed-dynamics generation calls. Only two small
`aa` lessons per invocation are exercised; the twelve-pair stream is not fitted.
Existing regression fixtures also run in the first receipt. No full comparison
or final cohort has been executed during preparation.
