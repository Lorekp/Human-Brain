# Fixed rehearsal allocation from authenticated after-A checkpoints

Status: preparation contract for one continuation comparison, not an adopted
learner. Preparation: **2026-09-05 17:01–17:46 UTC**. Freeze the exact source,
inputs and controls before execution. The [previous result](../../experiments/r4_integrated/real_input_qualification_result.md)
and its failed retention qualification remain unchanged.

## Hypothesis and evidence boundary

Cells acquired all eight A cases, then acquired B while forgetting six A cases.
The conventional control retained both blocks. Test whether allocating the same
number of new teaching offers across A and B restores useful retention without
losing B acquisition. This changes the distribution of supervised exposure,
not the core, optimizer, initialization, thresholds or total new-offer budget.
Rehearsal uses an external store of labeled lessons; it is not autonomous internal
consolidation. All 16 examples are exposed public TRAIN. Complete paragraphs and
pretrained MiniLM word vectors remain available; no source-absent memory,
fresh generalization, architecture adoption or roadmap gate pass is established.

## Inputs and exact restoration

Keep the [same fixture](../../experiments/r4_integrated/real_input_qualification_cases_v1.json):
A is its first eight items, B its next eight; each has six supported answers and
two STOP labels. Preserve original questions, complete paragraphs, accepted
annotations, UTF-8 offsets and paragraph-local 48-word span inventories.
Do not add, rewrite, filter or select examples by outcomes. Predictor inputs are
only the original query/source representations; annotation targets enter offers
and evaluator scoring. There is no new encoder inference or feature extraction.

- Original result source: `dd2bd7a43f25450931d36409cd1badd1d557dd76`.
- Original report SHA256: `073382d23ebc66a217f81873abca7d4e93f7b7cf6b7e34c89f4b185cadd60c11`.
- Original checkpoint archive SHA256: `921c1f253ade90afcc1c1c65534fcca0be0cd367dee0f334c5828dd3406ebb35`.
- Fixture SHA256: `aa988986d8f04d7e796e485ccea33f24c96cadad09f03779de844905ee5ad41b`.
- Saved-input NPZ SHA256: `5d60b2dad367a0a756c54fdce812b68865d51597b9b8ebfb92543e1968e12f9b`.

Authenticate the original report/completion/archive chain and every restored
after-A array, checkpoint identity, Adam moment, per-parameter step counter,
offer counter and update counter. Restore all state, not just weights. Retain
the original modes and parameter/configuration metadata. New offers start from
the inherited count of **2,048** for each parent. Saved inputs contain 32 original
word-vector arrays totaling 6,021,120 bytes, with no imported later learned state.
Keep the original SQuAD 2.0 TRAIN/CC BY-SA 4.0 attribution and exposure notices.

## Seven policies and four fixed streams

Create six independent physical core copies; external cosine is the seventh
policy and aliases `cell_no_learning`. It uses the existing sensory cosine
readout, not the cell producer's learned computation.

| Policy | Restored parent | New offers |
| --- | --- | --- |
| `cell_b_only` | after-A trainable cells | B0…B7, repeated 256 times |
| `cell_rehearsal` | same after-A trainable cells | B0,A0,…,B7,A7, repeated 128 times |
| `conventional_rehearsal` | after-A conventional | same alternating stream |
| `frozen_producer_rehearsal` | after-A frozen producer/trained heads | same alternating stream |
| `cell_no_learning` | same after-A trainable cells, original mode | zero |
| `conventional_no_learning` | same after-A conventional, original mode | zero |
| `external` | alias of `cell_no_learning` | zero |

Learning-disabled controls retain the **trained after-A parents**, not pristine
initializations. The runner withholds offers; it does not change their modes or
checkpoint identities. Check their full state remains unchanged throughout.
Frozen-producer rehearsal preserves its existing producer masks. Keep the original
Adam rate .001, betas .9/.999, epsilon 1e-8, active-coordinate clipping at 1,
hierarchical STOP and supported/STOP objective and optimizer masks unchanged.

Every updating continuation receives **2,048 new offers**, or **8,192 total**.
Each rehearsal stream supplies 1,024 A and 1,024 B offers; B-only supplies 2,048 B.
Count inherited A experience separately: these are branches of three previously
trained parents, not fresh training or additional executions of the original A
schedule. No extra replay, error-selected lesson, early stopping or budget growth.

Use 16 rounds. In each round, give each updating arm its next 128 fixed-stream
offers, in the table's updating-arm order, then probe all policies. Probe all
16 items for all seven policies initially and after each round: **1,904 predictions**
(112 + 16×112). No per-offer predictions. Preserve exact ordered rosters and
checkpoint immutability during probes. Retain all 112 final prediction objects
and reuse them directly for action, without another reader inference.

## Resume validity and resources

The cell B-only continuation must reproduce **every original B-stage update
receipt and the final checkpoint exactly**, including original source/query/target
identities, counters and learned arrays. Compare deterministic receipts, not new
wall-clock timings. The saved B trajectory supplies the concrete restoration
reference; no tolerance-based substitution for a mismatching checkpoint is allowed.
An unexplained mismatch invalidates the continuation comparison and stops it.

No new numerical-gradient qualification is required: unchanged cores already
passed actual-input admission. Byte-level restoration, source checks and the
B-only replication are the new validation obligations. Preparation QA must cover
restore/export custody, stream/probe counts, no-learning masks, immutable outputs,
partial failures and the new execution transport without actual scored fitting.

One **20-minute total run** includes loading/restoration, teaching, probes, motor
demonstrations, execution, authentication and artifact writing. Parent peak working
set must not exceed **2 GiB**. Preserve the outer process deadline and explicit
timing scopes. Count all six cores/optimizers, shared inputs once, labeled store,
inventories and retained experiment data. No new model, optimizer or seed variants.
Bind source/input/parent hashes in immutable report, checkpoint, completion and
launch artifacts. A timeout, resource, custody or serialization failure cannot pass;
preserve completed rows and the original error without an automatic rerun.

## Final qualification and action

Both full rehearsal learners (`cell_rehearsal` and `conventional_rehearsal`) must:

- Finish with A6/6 supported + 2/2 STOP **and** B6/6 supported + 2/2 STOP.
- Gain at least **one correct supported B answer** over their own matched after-A
  no-learning control. If that control is already at the supported ceiling, do
  not waive or redefine this criterion after observing the initial probe.
- Execute all 12 supported answers exactly and all four correct unchanged STOPs,
  with every decoy unchanged, from the same final checkpoints.

Additionally, rehearsal cells must restore all six A decisions lost by the
replicated B-only arm (A positions 0,1,2,3,4,7) while preserving all eight B cases.
Improved A at the expense of B fails. Report every intermediate loss and recovery;
there is **no additional requirement that every intermediate probe remain perfect**.
Frozen-producer rehearsal diagnoses producer contribution and is not required to
fail. A positive trained-head control limits attribution to producer learning.

Execute **112 actual policy/task attempts plus two shared motor demonstrations**.
Keep ordinal action metadata `qa000`/`qa_config000`/`Configure qa000` through 015,
stale `pending`, the unchanged learned motor, and generic lossless single-quote
escaping. Action addressing is independent of labels and model outputs; expected
answers enter scoring only. No teacher repair or learning occurs during execution.
Use the new `utf8_action_bridge.py` path with explicit child flags `-I -X utf8`:
isolated Python ignores environment-only codec settings. Reuse the original
`Case.files`, `ActionTask`, escaping and scoring contracts without changing frozen
files or global monkeypatches. Preserve raw stdout/stderr hex and hashes, decode
UTF-8 strictly, and compare successful captured values with selected/scoring
values. Before source freeze, qualify the new path with actual neutral Unicode
child roundtrips. Preserve the prior failed en-dash audit unchanged; its missing
raw bytes are not repaired retrospectively.

Passing supports this bounded supervised allocation strategy, not general learner
adoption. Failure closes the fixed comparison and requires a new evidence-based
choice, not more passes or an automatic consolidation/representation successor.
Official structured reading responses and the broader persistent action loop
remain pending under the [reading plan](reading-comprehension-evaluation.md).
