# Context learnability v1: one fixed exposure/order bracket

Status: prepared 2026-09-06; full comparison unrun.

The closed context v2 comparison has no useful composition: random feedback
fits 4/12 TRAIN and symmetric feedback 3/12, both 0/4 withheld probes. Saved
losses still decrease late; the random decoder begins improving exact first
attempts near the old endpoint. These data do not establish convergence or a
structural impossibility. The [learnability review](../../docs/research/local-decoder-learnability-review.md)
selects this single bracket to distinguish sufficient exposure/order from a
practical computation/credit limitation against a provably learnable reference.

## Fixed learners, inputs and streams

The existing decoder keeps default **random** feedback, exact initial seed 735
(experiment seed 733), context96/embedding32/hidden64, local eligibility, rate 0.01,
CPU float32, and all byte/EOS generation and teacher-forcing rules. Symmetric
feedback is not adopted. All reference models instead initialize int32 readout
ticks to zero; one tick represents rate 0.01. Cross-architecture initial parameter
equality is not claimed. Initial decoder states match each other; all four
reference states match each other.

The reference supplies an explicit three-neuron phase clock and generic
phase×all 96-context conjunction neurons. Its 291 features are three phase
activities plus all 288 conjunction activities, including those currently zero.
Every one of 257 byte/EOS output neurons receives all features. No phase-specific
factor selection, alphabet shortcut or proof separator initializes the model.
The two authored four-neuron context populations and remaining 88 zero rates
are identical to context v1. This is an informative supplied representation,
not a learned reader or learned timing policy.

At an incorrect teacher-step winner, the teacher neuron receives +1 signal,
the winning neuron −1, and each corresponding synapse changes its integer ticks
by presynaptic activity times that signal. Current exact int64 score competition
precedes any update. Correct winners cause no update. Input teacher bytes do not
select features: the explicit clock advances through all three teacher phases.
No optimizer, reverse gradient, decay, clipping or adaptive rate enters the rule.
Retain and count the fixed clock/conjunction wiring and all integer synapses.

The TRAIN roster is the same twelve lexical pairs over `a,b,c,d`, excluding
`ab,bc,cd,da`. Those four remain **previously exposed development probes**,
not a fresh assessment. Two complete TRAIN streams are serialized before model
construction or predictions: lexical order and a private `Random(734)` fresh
permutation per cycle. All twelve targets appear once each cycle in both.
There are 64 cycles and 768 lessons per arm, in this fixed per-index arm order:

| Arm | Learner/context | Stream | Teaching |
| --- | --- | --- | --- |
| decoder_lexical | Existing decoder, correct context | Lexical | Enabled |
| decoder_permuted | Same decoder, correct context | Permuted | Enabled |
| reference_lexical | Integer local reference, correct context | Lexical | Enabled |
| reference_permuted | Same reference, correct context | Same permuted stream | Enabled |
| reference_zero | Same reference, zero context and retained clock | Lexical | Enabled |
| reference_frozen | Same reference, correct context | Lexical | Disabled; labels offered |

First attempts precede all corrections. Every lesson then offers two target
bytes plus EOS, including frozen lessons. Generation emits at most two bytes
and gets one extra EOS decision; EOS/bytes are never repaired. Hidden/phase
state resets each lesson while learned synapses persist.

## Certificate, endpoints and accounting

The review derives a proof-only separator with squared norm 9 and margin 1 for
the two correct-context references. Their binary feature squared norm is 3.
From zero readout ticks, the multiclass perceptron makes at most
`2 × 3 × 9 = 54` teacher-position mistakes. Within 55 complete TRAIN cycles at
least one must be mistake-free; it updates nothing and covers every fixed input,
so all later cycles remain correct. The 64-cycle endpoint is fixed above this
bound prospectively, not selected through successive epoch trials. The proof
does not guarantee the four withheld combinations or apply to zero/frozen controls.

Observe exactly two checkpoints, after 384 and 768 lessons per arm. Each saves
all six complete model states and makes one read-only generation for each of
twelve TRAIN and four withheld pairs per arm. Checkpoint 384's lexical decoder
must reproduce authenticated context v2 random_correct first outputs, teacher
losses, endpoint outputs and state exactly. No extra generation or teaching is
added for this check; its audit uses saved records. Nothing adapts to checkpoint 384.

Totals: **4,608 first attempts, 13,824 teacher byte/EOS offers, 11,520 enabled
teacher-step opportunities, and 192 checkpoint probes**. Count actual changed
steps and generation decisions separately. Decoder `teacher_losses` are
cross-entropy; reference `teacher_losses` are exact 0/1 winner mistakes. Every
lesson and summary carries `loss_kind`; their numeric losses must not be compared
across architectures. Report per-position losses, first/last128 summaries,
first attempts, actual outputs, update counts and reference mistakes per cycle.

Reference certificate passage requires no more than 54 teacher mistakes,
a complete zero-mistake cycle by cycle 55, and 12/12 TRAIN at 768. A certificate
failure requires arithmetic/assumption investigation rather than a tuning sweep.
A useful final decoder/reference requires 12/12 TRAIN and 4/4 withheld exact bytes
with EOS and strict withheld superiority over both reference_zero and
reference_frozen. Report all six arms and both checkpoints regardless of outcome.

If only the longer lexical decoder succeeds, this establishes inadequate old
exposure for this case. If permuted training succeeds where lexical does not,
order is an observed intervention for this stream. If the reference fits TRAIN
but neither decoder does, prefer the explicit temporal representation for the
next scoped return; do not infer RNN impossibility or distinguish expressivity
from credit solely from that result. TRAIN-only reference success is not useful
composition. No automated new endpoint, phase variant, seed or threshold follows.

## Resource and artifact contract

Allow one full run within 120 seconds and 2,048 MiB under the existing monitored
launcher. All imports, six arms, both checkpoints, snapshots, hashing and logging
count. The performance counter measures durations; Unix deadline checks remain
separate. External process-tree monitoring supplies the memory limit. No native
motor or additional monitoring dependency enters the decoder-only bracket.

Keep frozen context v1/v2 source and receipts unchanged. Reuse v1's exact
context, rate/max-byte lesson, generation, hash and JSON helpers; none captures
the 384-lesson count for execution. This runner owns the 64-cycle stream, six-arm
factory, checkpoints and certificate. Sources include the full v1 closure, new
runner/recipe/reference module and the mathematical review.

Save `schedule.json`, `admission.json`, `initial.pt`, all 4,608 hash-chained rows
in `lessons.jsonl`, and `checkpoint-384.pt/.json` and `checkpoint-768.pt/.json`.
`report.json` contains both checkpoints, separately labeled learning metrics,
counts, tensor/artifact bytes, certificate and useful-endpoint decisions, and
start/end source/revision identities. Completion binds its hash. First generation
and both checkpoint probes must preserve model hashes; the frozen reference must
preserve its initial state. An independent saved-only audit reconstructs streams,
counts, state chains, teacher mistakes, exact outputs and launch ownership before
interpretation. No duplicate or resumed output directories are allowed.

Each reference lesson also records all three teacher phases, active presynaptic
feature indices, winner/target IDs and preupdate integer scores, 0/1 mistakes and
update flags. This distinguishes teacher-position mistakes from free-generation
failures. Reference accounting includes 644 neurons, 74,787 learned output
synapses, 585 allocated fixed synapses (578 nonzero), 288 thresholds and three
clock initial values: 302,652 stored tensor bytes, plus a 12-byte clock state,
3,232-byte step and supplied context. Python objects, copies and scratch remain
additional and are covered by external process-tree memory monitoring.

```powershell
$bracketOutput = '.pytest-tmp/local-context-learnability-v1-run'
$bracketRevision = (git rev-parse HEAD).Trim()
$bracketSources = (& ./.venv/Scripts/python.exe -c 'from experiments.r4_integrated.local_context_learnability_v1 import digest,sources; print(digest(sources()))').Trim()
./experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1 -Seconds 120 -MaxMiB 2048 -LogBase $bracketOutput -ChildArguments @('-X','utf8','-m','experiments.r4_integrated.local_context_learnability_v1','--output',$bracketOutput,'--revision',$bracketRevision,'--sources-sha',$bracketSources)
```

Review and commit source before binding this command. A useful compositional
result earns one return to document-derived neuronal context and actual actions
under fresh integrated admission, removing the answer-revealing authored inputs.
Neither the bound nor this two-byte task establishes reading, variable-length
output, learned action selection, general software capability or any gate pass.

## Preparation QA

Doctor passes at revision `671ec6e`, receipt
`.pytest-tmp/dev/doctor-s9q3bnep/receipt.json`. The existing decoder suite passes
**79 tests** in 6.58 seconds, receipt
`.pytest-tmp/local-context-learnability-v1-qa.xml`. Ruff, syntax compilation and
diff checks pass. The reference's final explicit-CPU source predates that QA
start. No full bracket or endpoint cohort was run during preparation.

Six added test instances cover balanced lexical/permuted schedules, certificate
boundary decisions, matching initial states and separate loss types, float32/64
generic clock/conjunction activity, exact integer teaching, stale-step and
overflow rejection, fixed/binary input checks, frozen behavior and checkpoint
round trips. They contain 15 accepted disposable teacher offers and four rejected
teacher calls (stale state, overflow or wrong rate), five successful read-only
generation calls and one rejected fixed-dynamics generation call. Existing
regression fixtures also run and are recorded in the complete receipt. These
interface checks do not constitute a fitted reference certificate or capability
result.
