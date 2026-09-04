# Pairwise contextual evidence: result and teaching diagnosis

Completed 2026-09-05 under the [declared protocol](contextual_pairwise_protocol.md).
This is repeatable public development of a hybrid candidate, not a gate pass.

## Executed outcome

Sixteen teaching requests covered eight entities with two properties each.
All sixteen first attempts failed in both updating arms. Each correction pair
joins an actually executed wrong edit to a separately executed teacher PASS.
The absolute rule made 32 native updates; pairwise made 16. Each frozen control
received its updater's exact feedback with unchanged synapses.

| Evidence policy | Fresh executed | Old requests executed | Fresh entities with both properties | Old entities with both properties | Unsupported STOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Absolute updating neurons | 0/8 | 1/16 | 0/4 | 0/8 | 0/2 |
| Pairwise updating neurons | 0/8 | 1/16 | 0/4 | 0/8 | 0/2 |
| Conventional learned scores, either rule | 0/8 | 1/16 | 0/4 | 0/8 | 0/2 |
| Frozen neurons, either rule | 0/8 | 0/16 | 0/4 | 0/8 | 2/2 |
| External encoder cosine, either arm | 0/8 | 0/16 | 0/4 | 0/8 | 0/2 |

Both learned policies emit `canvas` on all ten transfer and sixteen retention
requests. Their one retained success is a constant-answer match. Conventional
and neuronal winners agree by candidate occurrence throughout these probes, so
spike resolution adds no failure here. Frozen controls always STOP; their two
unsupported STOPs do not demonstrate selective abstention. All decoys survive.

The original public KS2 Q6a/b remains untuned. Learned policies emit `canvas`
twice, external references emit `to` twice, and frozen controls STOP twice.
Under the existing official-rubric scoring contract, noncanonical outputs remain
pending human scoring: zero confirmed marks, two pending items. Frozen scores
0/2. Do not turn pending marks into a final zero or claim exam progress.

## What the recorded contrasts establish

The [receipt diagnostic](../../benchmarks/contextual_pairwise_diagnostic_v1.json)
reconstructs the original cumulative source store at every correction. All 32
eligibility hashes and 16 difference hashes match. One native replay pass from
zero exactly reproduces the saved weights and recorded immediate score gaps.

The feedback forms one acyclic ordering of values, ending at `canvas`. A linear
fit using only query-independent span features satisfies all sixteen margins
(rank 16; maximum residual 1.14e-14). Counterbalancing property questions did
not create feedback that requires conditional ranking. This fit concerns only
the sixteen executed contrasts, not all candidates or fresh tasks.

There is also interference: ten of eleven initially margin-satisfied contrasts
later violate the margin; five initially positive preferences reverse. These
are pair-margin measurements, not ten previously solved autonomous tasks.
Exactly 25 total native passes over the fixed ledger leave five margin violations
and one wrong pair. Further replay of this chain cannot supply the missing
conditional distinctions, even if optimization eventually fits it.

## Resources and reproducibility

The [initial run](../../benchmarks/contextual_pairwise_python_development_v1.json)
takes 45.32 seconds and peaks at 246.09 MiB working set. The
[checkout-native replay](../../benchmarks/contextual_pairwise_python_development_v1_local_native.json)
takes 38.64 seconds and peaks at 246.74 MiB; final private memory is 556.79 MiB.
These are whole sequential parent-process measurements including evaluation,
excluding child executor memory, not isolated kernel speed comparisons.

The initial run loaded the temporary-cache DLL before its `run()` environment
assignments. Importing `neuronal_relational_reader` builds its module-level
score currents and loads native code. Setting the environment before Python
starts selects the intended checkout cache; see [development.md](../../docs/development.md).
Both libraries have source digest `318e9734…`; actual binary hashes are recorded
in the artifacts (`66e048d6…` temporary, `62eb9eab…` checkout). The replay preserves
all 274 training/evaluation decision records, feedback, checkpoints, exam records
and saved synaptic arrays exactly. Both runs and archives remain unchanged.

After teaching, each memory retains 637 raw bytes and 300 candidates from eight
documents. After transfer and the exam, it retains 4,839 raw bytes and 2,319
candidates in 27 windows from fifteen documents. Counted arrays and byte payloads
total 7,283,897 bytes per arm, excluding Python object overhead and the shared
90,405,214-byte frozen encoder. Raw sources and vectors remain in external
storage; source removal refers to live input, not semantic compression.

The separate diagnostic takes 2.36 seconds, including 0.0075 seconds for 400
offered native updates. It uses no new negatives, global-candidate evaluation,
transfer evaluation, or protected material. Artifact/source/checkpoint hashes
were verified. Nine focused encoder/learning tests pass, including pairwise
margin change, frozen replay, and invalid/stale comparison rejection.

## Decision

Do not tune GLIF or scale this head. The [bounded research review](../../docs/research/contextual-learning-review.md)
recommends actual old-query revisitation, comparing it with the same schedule
plus bounded replay of executed correction receipts. The next ledger must
demonstrably require query dependence; request order alone cannot prove that.
Keep representation fixed for this comparison, then distinguish failure to fit,
forgetting, and retained learning without fresh transfer. The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns that next experiment; this completed cohort and its results stay intact.
