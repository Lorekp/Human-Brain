# Frozen external reader fails qualification

Completed 2026-09-05 under the [fixed protocol](frozen_reader_qualification_protocol.md),
source `25689a9d2d97506591fe83a47f91a05ea37e2298`. One neutral runtime preflight
passed, followed by one complete qualification. No learning, Python actions,
question rewriting, retry, threshold change or official/protected evaluation.

## Result and decision

**Close this candidate without router fitting.** The pinned
`optimum/roberta-base-squad2` reader does not meet the required 44/48 exact
supported answers and 15/16 correct STOPs in either oracle-source wording.
The [conditional experience-addressing proposal](../../docs/research/acquisition-strategy-review.md)
therefore does not proceed with this reader. No neuronal or integration gate passes.

| Access | Wording | Exact supported /48 | Correct unsupported STOP /16 | Inference seconds |
| --- | --- | ---: | ---: | ---: |
| Known relevant source | Actual | 24 | 13 | 5.939 |
| Known relevant source | Alternate | 28 | 12 | 5.975 |
| All 16 sources | Actual | 20 | 2 | 96.302 |
| All 16 sources | Alternate | 13 | 4 | 97.095 |

The known-source condition is an evaluator-only ceiling diagnostic. Unsupported
requests receive their actual nonempty source too. Full-corpus inference receives
all original documents and unmodified requests, without gold-source selection.
These 64 requests are already exposed, authored TRAIN development; they are not
independent generalization or authentic exam evidence.

## What the failure establishes

With the correct source supplied, actual wording has 16 supported abstentions,
five wrong-entity answers and three wrong-property answers. Alternate wording has
16 abstentions, two wrong-entity answers and two overextended spans. For example,
both forms of derum's audit request select calvo's `anorthite` from the same
document instead of `common spotted orchid`. Kedru's actual audit request selects
its transport value `benitoite` instead of `hairy bitter cress`.

Only 5/16 actual and 10/16 alternate supported abstentions have the correct
recorded best span. These descriptive rankings are not credited successes or a
threshold study. The two overextensions faithfully copy multiple clauses; no
inside-word cuts, whitespace-only discrepancies or partial-target emissions
explain the result.

All-source access gains no correct cases over corresponding known-source access.
It loses four supported answers and eleven correct STOPs in actual wording,
fifteen answers and eight STOPs in alternate wording. All 128 repeated
correct-source window records are identical across access conditions: added
source competition under the fixed maximum-margin rule changes the decision.
This does not isolate model binding from aggregation calibration.

Reliable entity/property binding, whole-value extraction and evidence absence
are missing at this interface. A source router alone cannot repair the observed
within-source errors. This does not establish that every pretrained reader would
fail, that the model fails general QA, or that neuronal learning is impossible.
Do not automatically replace the reader or fit routing on its successful subset.
Return to a bounded acquisition-design review before another mechanism experiment.

## Verification and resources

Twelve focused synthetic adapter tests and inert runner checks passed before
freeze, including pinned-tokenizer overflow coverage, UTF-8 offsets, context
masking, span/null/tie arithmetic, malformed outputs and failure preservation.
Saved-only audits verify source/model/admission identities, all 256 rows, every
recorded best-window byte slice and decision aggregation, fixed denominators,
resource totals and unchanged correct-source windows. They do not independently
recompute unretained model logits or tokenizer outputs. No native code changed.

The neutral preflight takes **27.365 seconds**, with 689.102 MiB peak parent
working set. Its 51 longer synthetic windows project 1,094.598 seconds, below
the 1,800-second qualification limit. Actual qualification takes **206.841
seconds**, including 1.047 seconds loading and final source/model authentication;
205.311 seconds are inside reader calls. Final JSON serialization/write is
excluded. Elapsed time from preflight start to qualification end is 261.128
seconds, within the combined 2,700-second budget.

Qualification processes **2,176 query/document pairs and windows**, totaling
**132,124 paired tokens**. Peak parent working set is **689.441 MiB**, below
2 GiB. Retained raw sources contain 3,195 bytes. The six cached model/tokenizer/
configuration/card files total **498,422,266 bytes**; this disk payload is distinct
from process memory and excludes the local manifest. ONNX Runtime 1.29.0,
tokenizers 0.23.1 and NumPy 2.4.6 run on CPU with one execution thread.

The reader is a declared pretrained QA component, not the project's neuronal
learner. Its pinned export revision is `db07770d7fa024df076e60947b445b0b3290f8eb`;
the included card declares CC-BY-4.0 and conversion of deepset version 2. Tensor
equality to the current upstream checkpoint was not verified. Pretraining exposure
is not independently audited. No hybrid architecture is adopted by this check.

## Artifacts

- [Neutral preflight](../../benchmarks/frozen_reader_preflight_v1.json), 51,435 bytes,
  SHA256 `52a3b2366a5ee29bd44778c18beeef7347e6183bca803de57c8045c319568ccc`.
- [Complete qualification](../../benchmarks/frozen_reader_qualification_v1.json),
  1,759,989 bytes, SHA256
  `ad35e4bcc10d82ff8450750c953a19980dff24f33eacdf71a59592b91b232454`.
- [Scoped saved-data audits and diagnosis](../../benchmarks/frozen_reader_qualification_v1.audit.json),
  237,991 bytes, SHA256
  `e35dd7d60a6a02d4fffaab0a682c3ef76d206c3d2be897c69ec6a6a8d7e726ca`;
  includes checkers, source hashes and the detailed failure classification.

The preflight embeds the source-bound preparation receipt and download custody;
the qualification embeds that exact preflight. All original source/model hashes
remain bound to the frozen commit. Earlier failed experiments remain unchanged.
