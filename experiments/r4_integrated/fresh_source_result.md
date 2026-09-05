# Fresh-source actions fail despite fitted paragraph-conditioned parents

2026-09-05. Frozen source: `6c2287afff09a1eb5b2265375ee584ea7c3e07a7`.
One session completes under its [revised contract](../../docs/research/fresh-source-action-plan.md)
and fails qualification. All policies solve **0/8 fresh supported questions**,
including with the correct paragraph supplied. Cells also fail source competition
on old fitted questions and lose eight old paragraph-conditioned answers after
the pooled updates. No architecture is adopted or roadmap gate passed.

## What ran

Start from the authenticated final [rehearsal parents](fixed_rehearsal_result.md),
including weights, Adam state and 4,096 inherited offers. Six policies begin with
all sixteen old paragraphs in one inventory; eight episode paragraphs arrive
sequentially, then four wholly untaught later paragraphs arrive after learning.
Actual predictions receive every current candidate, without correct-paragraph
routing. Separate read-only oracle predictions receive the original paragraph.

The twelve new questions and complete passages are unchanged original SQuAD 2.0
TRAIN items from the earlier project-exposed probe partition, absent from these
parents' sixteen-question experience. They are public development, not an
independent or contamination-free evaluation. Original attribution is retained.

Each updating policy receives **23 new offers**: seven fresh answer/STOP lessons
after first attempts and sixteen fixed old-lesson replays. There are 69 total
offers, no extension and no new optimizer, encoder or core variant. New input
encoding runs exactly 24 times; old saved representations are reused.

The [original preparation abort](../../benchmarks/fresh_source_aborted_preparation_v1.json)
is preserved. E3's "America" question is ambiguous when its passage discusses
Canadian representatives. L3's rocket/satellite wording makes its official STOP
key uncertain. The separate revised contract keeps both questions, their sources
and all their actions, with null semantic verdicts and no feedback or corrections.
Only E7 and L1 supply scorable fresh STOPs. This does not pass or reinterpret the
abandoned four-STOP criterion. No replacement questions or new labels were added.
The old O11 scientist/historian distinction remains an official-label limitation.

## Findings

Counts separate exact supported answers and correct STOPs. O denotes sixteen old
questions (12 supported +4 STOP); fresh denotes twelve first attempts (8 supported,
2 scorable STOP, 2 unscored). Supported correctness requires an original-source
annotated occurrence and exact execution, stricter than text-only QA matching.

| Policy | Initial pooled O | Final pooled O | Final oracle O | Fresh executed (8 +2) |
| --- | ---: | ---: | ---: | ---: |
| Updating cells | 0 +4 | 0 +4 | 4 +4 | 0 +1 |
| Frozen cells | 0 +4 | 0 +4 | 12 +4 | 0 +1 |
| Updating conventional | 5 +4 | 9 +4 | 12 +4 | 0 +1 |
| Frozen conventional | 5 +4 | 5 +4 | 12 +4 | 0 +1 |
| Frozen producer, updating heads | 1 +3 | 0 +3 | 2 +3 | 0 +0 |
| External cosine | 0 +0 | 0 +0 | 0 +0 | 0 +0 |

Three failures are distinguishable:

- **Immediate source competition:** before new learning, both full cell policies
  answer all 12 supported O questions under the oracle, but none in the pooled
  inventory. All twelve pooled selections come from another paragraph; gold
  ranks range from 89 to 3,182. These are wrong selections, not gate rejections.
  Conventional parents retain five correct pooled selections at the same stage.
- **Subsequent interference:** updating cells lose oracle answers O0, O1, O2, O5,
  O9, O10, O13 and O14, while frozen cells retain all twelve. The sixteen old
  replay offers do not prevent this loss. Conventional oracle retention stays
  perfect and its pooled supported score rises from five to nine, four above
  its final matched frozen control. This is useful old-task adaptation, not
  fresh transfer or autonomous recovery.
- **Unresolved acquisition and transfer:** all policies score zero supported
  fresh first attempts, even in the paragraph oracle. All six newly supervised
  supported episode questions also remain wrong in final oracle probes. The
  wholly untaught later tasks give 0/2 supported and 0/1 scorable STOP for every
  policy, including the oracle. There is no later supported learning gain.

The external cosine selections do identify the annotated source for supported
questions, but select wrong answer spans. Source addressing and answer extraction
therefore have different observed strengths. A routing change alone cannot
establish fresh comprehension. The frozen-producer control retains its weaker
historically trained parent and is not a matched freeze of the full producer.

## Execution, costs and evidence

There are **312 primary actions, 312 oracle predictions, 19 external teacher
repairs and two shared motor demonstrations**: 333 actual child executions in
314 workspaces. All 942 final files authenticate. Teachers repair the same failed
workspace; the original failed receipts are retained. E3/L3 contribute eighteen
primary and eighteen oracle rows across first/final phases, all unscored and
untaught. Teachers never count as learner successes.

Saved audits verify selected occurrences, literal escaping, strict UTF-8/LF raw
outputs, unchanged decoys, checkpoint custody and the corrected workspace state.
The all-six-first-attempt barrier is verified from frozen control flow and inert
QA; individual child records do not provide independent absolute start times.
No child or learner is reexecuted during these audits.

Runner time is **63.792 seconds**, including primary artifact writing; outer time
is **64.029 seconds**, excluding the launch receipt's own final write. Peak parent
working set is **351.832 MiB**, within 20 minutes and 2 GiB, with no retry.
The inventory grows from 41,300 candidates/46 windows to 70,898/78. Compact
inventory receipts authenticate the full candidate ordering without retaining
ten duplicate candidate manifests.

- Retained word vectors: **10,334,208 bytes** across 56 original/new inputs.
- Counted learner/input/motor graph: **49,585,867 bytes**; independently recoverable
  payload lower bound: 30,774,211 bytes. The separately counted source/label graph
  is 1,543,208 bytes; complete retained experiment graph is 113,046,933 bytes.
  These overlapping scopes are not additive and do not equal RSS.
- Local checkpoint/input archive: **11,447,286 bytes**, SHA256
  `49c70c5a625fded588d2478d565a74f3e6339f9ab34c8052b4001c6af9970b81`,
  retained at the report's `.runtime` path. All 274 arrays authenticate.

The [preparation bundle](../../benchmarks/fresh_source_preparation_v1.json) preserves
data review, meaningful neutral/inert QA, the earlier checker-only failure and
source derivations. The [report](../../benchmarks/fresh_source_v2.json), SHA256
`194480c4f0b09349db0fc3b7370d1eb4afda56130064b84c2344d5da7b53b60e`,
has matching [completion](../../benchmarks/fresh_source_v2.completion.json) and
[launch](../../benchmarks/fresh_source_v2.launch.json) receipts. The
[audit bundle](../../benchmarks/fresh_source_v2.audit.json) embeds three passed
saved-evidence audits and their checkers. All 42 frozen sources, 69 ordered offer
receipts, source/gold identities and ten inventories authenticate. Audits check
endpoint state and recorded update chains, not an independent gradient trajectory.

The result still retains complete raw documents, word vectors and candidate
bytes. It establishes neither source-absent knowledge nor answer-free learning.
No official exam, protected coding or larger SWE evaluation runs.

## Next decision

The [decision review](../../docs/research/pooled-reading-failure-review.md) compares
source addressing, varied acquisition and a principled representation or hybrid
change. It selects one 45-minute primary-source acquisition review to specify
an implementable comparison with a credible learning reference and exposure
budget before another fit. Source routing alone cannot solve fresh oracle
failure; more epochs on the sixteen old questions cannot establish transfer.
This session remains closed.
