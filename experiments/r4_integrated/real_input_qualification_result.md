# Actual-language acquisition works; retention qualification fails

2026-09-05. Frozen source: `dd2bd7a43f25450931d36409cd1badd1d557dd76`.
The [protocol](../../docs/research/real-input-qualification-plan.md) is unchanged.
One run completed within budget. Cells learned both blocks and used learned
answers in Python, but failed the required retention and final-action criteria.
The conventional positive control passed those criteria. No architecture is adopted.

## Findings

The first 16 previously exposed SQuAD 2.0 TRAIN questions retain their complete
paragraphs, original annotations and legal span competition. A and B each contain
six supported questions and two STOP cases. Each updating policy received 256
ordered passes over A, then 256 over B: 4,096 offers each, 12,288 total. There was
no A rehearsal during B and no outcome-dependent extension. All policies share
the authenticated saved MiniLM vectors; no encoder inference was repeated.

Counts below separate exact supported answers from correct STOPs. Final execution
uses the same final predictions, with no new reading inference or learning.

| Policy | A after A (6 + 2) | A after B (6 + 2) | B after B (6 + 2) | Executed final (12 + 4) |
| --- | ---: | ---: | ---: | ---: |
| Trainable cells | 6 + 2 | 2 + 0 | 6 + 2 | 8 + 2 |
| Frozen producer, trained heads | 2 + 2 | 1 + 1 | 0 + 2 | 1 + 3 |
| Frozen cells | 0 + 0 | 0 + 0 | 0 + 0 | 0 + 0 |
| Conventional | 6 + 2 | 6 + 2 | 6 + 2 | 12 + 4 |
| Frozen conventional | 0 + 2 | 0 + 2 | 0 + 2 | 0 + 4 |
| External cosine | 0 + 0 | 0 + 0 | 0 + 0 | 0 + 0 |

All initial supported scores were zero. The conventional reader first appears
fully fitted on A at the 16-pass probe (128 offers); cells at 80 passes (640
offers). Both remain perfect on A through its final probe. These are sampled
first-fit times, not exact learning onset: probes occur every 16 passes.

During B, cells first fit all eight B cases at pass 48 and remain fitted through
pass 256. Meanwhile A falls to three supported/one STOP at B pass 16, two/one at
pass 32, then two/zero from pass 48 onward. **Six previously correct A decisions
are forgotten.** All four lost supported answers emit wrong spans; their final
annotated-answer ranks are 92, 2, 65 and 176. Both lost STOP cases emit spans.
The failure therefore includes lost answer selection as well as abstention.

The conventional control retains A and fits B by its first B probe, staying
perfect thereafter. The frozen-producer control never jointly fits a block.
Training the cell producer contributes useful acquisition under this teaching
schedule, while its retention remains weaker than the conventional comparator.
Different initializations and architectures prevent attribution to recurrence alone.

## Execution and transport limitation

All 96 policy/task attempts and two shared motor demonstrations ran in real
Python workspaces. The motor received an explicit ordinal `Configure qaNNN`
instruction and the already selected span, escaped as Python literal content.
It received neither the public reading question nor the expected answer.
All 294 retained files match their recorded contents; no decoys changed.
The action outputs reproduce the final reading scores in the table.

The original strict action audit **failed** on one stdout roundtrip:
`external-005` selected a wrong span containing an en dash. The stored Python
literal preserves the selected UTF-8 value exactly, but captured stdout contains
a replacement character. A Windows output-encoding/UTF-8-decoding mismatch is
compatible with this evidence; raw child stdout bytes and the child codec were
not retained, so CP1252 is an inference. The action was already wrong and failed.

All 12 official supported answers are ASCII. Every one of the 34 claimed action
successes, including correct unchanged STOPs, and both demonstrations has lossless
recorded stdout. Counts are unchanged. The strict failed audit, completed scoped
audit and transport supplement remain separate in the audit bundle. Repair the
execution transport before making a broader Unicode claim; do not rewrite this run.

## Costs and evidence

Actual-input numerical admission passed on disposable copies before teaching:
64 perturbed forward evaluations, four baseline gradients, four predictions and
four separately counted validation offers. All four validation offers exercise
clipping; warmed STOP preserves span weights, moments and counters. Maximum
directional errors are `7.65e-11` for cells and `1.18e-10` for the conventional
reader, within the fixed tolerances. Recorded independent Adam array comparisons
have zero error. This checks derivatives against the implemented objective and
local updates; it does not independently replay the whole training trajectory.

The runner takes **97.496 seconds**, including primary JSON/checkpoint writing
and authentication. The outer process takes **98.651 seconds**, including Python
bootstrap; its own final receipt write is excluded. Peak parent working set is
**315.234 MiB**, within the 20-minute/2-GiB limits. There was no startup retry.

- Shared retained word vectors: **6,021,120 bytes**; saved input archive: 5,778,426 bytes.
- Counted learner/optimizer, input/inventory and motor objects: **20,726,002 bytes**.
- Source/label object graph, separately: **657,635 bytes**.
- Counted complete experiment data, including checkpoint copies, cached final
  predictions and report objects: **88,830,720 bytes**. RSS also includes allocator,
  native and temporary work; these scopes are not interchangeable.
- Checkpoint archive: **5,898,865 bytes**, retained locally under the report's
  `.runtime` path, SHA256 `921c1f253ade90afcc1c1c65534fcca0be0cd367dee0f334c5828dd3406ebb35`.

[Preparation](../../benchmarks/real_input_qualification_preparation_v1.json)
binds the source/input custody, inert failure-path checks and native setup.
The [primary report](../../benchmarks/real_input_qualification_v1.json), SHA256
`073382d23ebc66a217f81873abca7d4e93f7b7cf6b7e34c89f4b185cadd60c11`,
has matching [completion](../../benchmarks/real_input_qualification_v1.completion.json)
and [launch](../../benchmarks/real_input_qualification_v1.launch.json) receipts.
The [saved audits](../../benchmarks/real_input_qualification_v1.audit.json) embed
their checkers and preserve the transport audit failure. Numerical receipts,
12,288 offer links/masks/counters, 2,400 probes and all 375 checkpoint arrays
authenticate. No logits, training trajectories or child executions were rerun
during these audits. Execution/custody status is separate from qualification.

This is concentrated supervised acquisition on exposed public TRAIN with retained
paragraphs and an external pretrained representation. It establishes neither
fresh generalization, source-absent semantic memory, autonomous recovery nor a
roadmap gate pass. The earlier broad comparison and its weak TRAIN fit remain
unchanged; this different exposure schedule does not repair their scores.

## Next decision

Run one bounded **fixed rehearsal-allocation comparison**, after freezing a new
protocol. Restore authenticated after-A weights, Adam state and counters. Compare
cells continuing B-only for 256 passes with cells alternating `B0,A0,...,B7,A7`
for 128 passes: **2,048 new offers each**, with no additional budget for rehearsal.
Use conventional and frozen-producer rehearsal controls at the same new-offer
budget: **8,192 new offers across four updating continuations**, plus matched
after-A learning-disabled and external-only readouts.
Count inherited A exposure separately. The B-only continuation must reproduce
the saved trajectory/checkpoint, providing a concrete restoration check.

Require both A8/8 and B8/8 from rehearsal cells and the conventional control,
followed by exact execution of all sixteen decisions at the same checkpoints.
Improved A at the expense of B fails. The existing B-only trace already fits B
and loses six A cases at pass 128, so simply halving B teaching did not retain A.
No error-selected replay, new epochs, optimizer or core variant is implied.

Choose this before consolidation or representation changes: it tests whether
allocating a fixed teaching budget across old and new experience can retain the
demonstrated acquisition. Consolidation adds protection state and a new learning
computation; changing representations would obscure the current acquisition
result. A failed rehearsal comparison must reopen that choice rather than grow
the budget. Rehearsal openly uses an external labeled lesson store; it is not
autonomous internal consolidation or fresh transfer.

Allow at most 45 minutes to prepare/validate the continuation and repair the
declared stdout transport in its new execution path, then one 20-minute run.
The official structured-response interface and general persistent-learner
obligations remain pending. Protected evaluations stay sealed.
