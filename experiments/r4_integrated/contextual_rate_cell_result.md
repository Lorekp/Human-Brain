# Rate-cell acquisition fails: abstention and incorrect span ranking

Completed 2026-09-05 under the [frozen protocol](contextual_rate_cell_protocol.md),
source `414f3920b0f222812944a77bf72d406ae3354960`. One synthetic cost preflight
passed, followed by one acquisition run. No tuning, restart or additional
training followed. This is exposed, answer-supervised TRAIN development.

## Outcome and decision

**Close this candidate without adoption.** Learned rate cells do not acquire
useful executed answers or outperform the trained-head controls. The treatment
solves no supported request, and its correct unsupported STOPs match the fully
frozen evidence learner. No integrated, fresh, official-exam or protected gate passes.

| Arm | Actual supported, before / after growth | Alternate supported, before / after growth | Correct unsupported STOPs, every phase |
| --- | ---: | ---: | ---: |
| Trainable cells and heads | 0 / 0 of 48 | 0 / 0 of 48 | 16/16 |
| Frozen cells, trained heads | 0 / 0 of 48 | 0 / 0 of 48 | 0/16 |
| Fixed query-conditioned bypass, trained heads | 1 / 1 of 48 | 1 / 1 of 48 | 0/16 |
| Frozen evidence learner | 0 / 0 of 48 | 0 / 0 of 48 | 16/16 |
| External token similarity | 0 / 0 of 48 | 0 / 0 of 48 | 0/16 |

The treatment stops on **255/256** final probes. Its sole emission, `blue`, is
incorrect and becomes STOP after growth. The bypass emits `boracite` for every
request; its one success is not evidence of request-specific selection. The
frozen-producer arm emits `For` before growth and `Garden` afterward.

The STOP gate is not the only failure: **none of the 48 supported targets wins
the conditional span ranking in any phase**. Before growth, the median gold
rank is 318.5 for actual wording and 376 for alternate wording. Every correct
answer scores below the best incorrect span. Changing the abstention threshold alone cannot
produce a correct supported answer with these final span scores.

The shared four-pass schedule supplies 256 encounters and 511 offers to each
arm. All three updating arms make 511 parameter changes; frozen and external
arms make none. Supported treatment execution is 0/48 on every teaching pass;
all 39 treatment emissions during teaching are incorrect single words.

| Treatment loss before each new lesson | Pass 1 | Pass 2 | Pass 3 | Pass 4 |
| --- | ---: | ---: | ---: | ---: |
| Supported conditional-span loss | 8.743 | 8.130 | 7.984 | 7.816 |
| Supported answerability loss | 0.691 | 0.888 | 1.059 | 1.173 |
| Unsupported answerability loss | 0.703 | 0.537 | 0.418 | 0.360 |

Own-offer total loss decreases on 437/511 treatment updates and increases on
74; Adam does not guarantee an immediate loss reduction. Improving span loss
without correct selections is inadequate acquisition. The opposite movement
of supported and unsupported gate losses motivates further diagnosis, but
does not identify shared-producer interference as the unique cause.

Return to the persistent executed-loop objective through a
[bounded failure review](../../docs/research/rate-cell-failure-review.md).
Do not tune this spent configuration's rate, thresholds, epochs or labels.
Neither this failure nor the earlier inconclusive LP proves fixed-feature
impossibility, or establishes that recurrence is necessary or sufficient.

## Verification and resource limits

Six focused synthetic/adapter tests and independent numerical checks passed
before source freeze. Python 3.11 compilation and focused Ruff checks passed.
Ruff now has a repository-owned Python 3.11 target: inherited parent settings
had generated unsupported exception syntax during preparation. The existing
71-test prefix and native implementation remain unchanged; broad native QA
was not repeated for this pure-NumPy evidence learner.

A read-only audit checks all **2,818** retained execution workspaces, including
two motor demonstrations: their 8,454 Python files match selected payloads or
unchanged baselines, recorded stdout and pass/fail outcomes. It executes no
code again. Teacher repairs and unchanged-workspace demonstrations are separate
from learner success.

Independent numerical replay reconstructs 1,533 updating offers and authenticates
2,555 adapter links. All 148 saved inputs match the earlier independent encoder
custody check. Initial/trained archive hashes and frozen parameter blocks match
exactly; reconstructed final weight/moment coordinates differ by at most
2.84e-13. Loss and gradient-norm differences pass the frozen
`1e-10 + 1e-10*abs(reference)` tolerance. Independently ordered arithmetic is not
claimed to regenerate every intermediate byte hash exactly: those intermediate
identities are checked as receipt chains. Evaluation and growth leave learned
and optimizer arrays unchanged. No numerical implementation defect was found.

The synthetic cost check takes **3.478 seconds** and projects **277.820 seconds**
for acquisition. Acquisition takes **299.067 seconds** through checkpoint export;
final JSON serialization/write is excluded from that timer. The 1,280 teaching
attempts, 256 shared teacher executions and 1,280 evaluation attempts account
for 2,816 executions, plus the two motor demonstrations. Input preparation,
including frozen MiniLM loading and 148 distinct source/query encodings, takes
1.124 seconds. The run uses CPU with single-thread BLAS and encoder execution.

Peak parent working set is **277.289 MiB**, excluding execution children. This
is below the 2 GiB parent limit. The separately recorded final audit-object
size is 48,274,239 bytes; exported array payload is 16,170,552 bytes.

**The frozen runner's learner-payload measurement is invalid.** Its object
walker recognizes imported project classes but skips `ExecutedReader` when
the module runs as `__main__`. Consequently, it reports only 11,804 Python/NumPy
learner/motor bytes at the end, omitting core and input state. Five arms' exported
checkpoint arrays, including step counters, occupy 3,494,760 bytes. The reported 90,894,782
byte total, dominated by 90,882,978 encoder-file bytes, must not certify the
1 GiB retained-payload gate. The original receipt remains unchanged; the
separate audit establishes a 9,366,069-byte recoverable numerical/raw lower bound
before candidate metadata and the encoder. An overinclusive saved-object
reconstruction estimates 109,261,853 bytes including encoder files. Original
object sharing and heap identities cannot be recovered exactly: this estimate
does not retrospectively certify the original payload gate. Fix this accounting
before any successor run.

This system retains raw sources and frozen external word vectors, uses explicit
answer/STOP supervision, and shares a previously taught GLIF literal-edit motor.
The new 64-cell recurrence learns through conventional BPTT and Adam; it directly
supplies evidence scores. These controls isolate evidence-learner changes, not
removal of the entire neuronal system. This is not answer-free internalization,
autonomous recovery, source-free memory or independent language generalization.

## Artifact custody

- [Cost preflight](../../benchmarks/contextual_rate_cell_preflight_v1.json),
  36,052 bytes, SHA256 `c60c171b7f9de4448669273d25ea34c9ad84213b7ef42243cda2e379859da8c9`.
- [Acquisition record](../../benchmarks/contextual_rate_cell_acquisition_v1.json),
  21,871,998 bytes, SHA256 `60cef41c7ef7c03d3615bbadaa4bba79b90576a0fb55fd1307f32f9a361ba62c`.
- [Inputs and initial/trained/final states](../../benchmarks/contextual_rate_cell_acquisition_v1.npz),
  9,406,607 bytes, SHA256 `caf03383858bf0b412296093e218b9d80792922f633f315e623a461e2d54f179`.
- [Independent audit and resource correction](../../benchmarks/contextual_rate_cell_acquisition_v1.audit.json),
  1,308,351 bytes, SHA256 `8ae9343b59c7a382a8b5116834dde9719634c2cb5e6993b8af9c8e192f71b566`.
  This includes numerical, contract, workspace and behavioral receipts, checker
  source, and the explicit qualification of the original payload-cap claim.

The records bind source/protocol, fixture, encoder and native motor identities.
Working bytes match committed source exactly except the explicitly declared
old full-choice helper's working-CRLF/Git-LF equivalence. Source hashes remain
unchanged throughout acquisition. Saved inputs and teacher ledgers are audit
material; labels do not enter inference. All earlier spent results remain intact.
