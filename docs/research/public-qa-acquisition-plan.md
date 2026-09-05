# Broader public-QA acquisition, then return to execution

Status: one frozen comparison at the source revision recorded by its admission
receipt; not an adopted learner. Preparation window: 2026-09-05,
**12:20–14:20 UTC**. The committed data, source, numerical qualification and cost
contract below govern its one preflight and, if admitted, one acquisition. The
[roadmap](roadmap-reading-to-coding.md) owns priorities; the existing cell and
[failed frozen-reader](../../experiments/r4_integrated/frozen_reader_qualification_result.md)
experiments remain closed.

## Failure and hypothesis

The aligned cells acquired 0/48 final supported actions from repeated exposure
to 64 narrow authored requests. The subsequent external reader also failed
within-source entity/property binding, so the conditional episodic-router design
did not proceed. These results do not distinguish inadequate experience from
an inadequate trainable computation. Another small fit or replacement reader
would leave that question unresolved.

Test whether broader human-written questions, passages and answerability lessons
produce useful reading in the **unchanged query-aligned rate-cell computation**,
using the existing pinned MiniLM sensory features. Compare with an independently
trained conventional reader using those same features. This changes the teaching
package and paragraph-local competition, not dataset size alone. The test is
explicit supervised hybrid acquisition; it does not establish language learning
from raw text alone or answer-free fact internalization.

## Data and exposure

Use only the authors' **SQuAD 2.0 TRAIN** release. It supplies crowdwritten
extractive and unanswerable questions over Wikipedia passages, distributed under
CC BY-SA 4.0. Preserve attribution, source links, supplied notices and the license
for any redistributed subset; identify selection/format changes.
[Dataset authors](https://rajpurkar.github.io/SQuAD-explorer/),
[license](https://creativecommons.org/licenses/by-sa/4.0/).
This is acquisition material, not a substitute for authentic official-exam evaluation.
No SQuAD DEV/test, new exam or protected coding content enters this run.

Allocate whole articles by SHA256 of their original title: integer hash modulo
five equal to zero supplies probes, the rest teaching. Exclude exact passages
shared between different articles; deduplicate identical paragraph/question
examples without discarding conflicting annotations silently. Freeze full counts,
exclusions and original download/selection hashes before model outputs.

Keep original questions and paragraphs. Eligibility requires each original input
to fit the pinned encoder's 256-token limit, complete subtoken-to-regex-word
pooling, and every accepted annotation to fit exact word endpoints within the
unchanged 48-word nonoverlapping candidate windows. Preserve all accepted answer
texts/occurrences; do not shorten passages, rewrite questions or filter by model
scores. An insufficient eligible roster stops preparation.

Select by SHA256 question ID within each split/answerability group, then
interleave three supported items and one unsupported item:

- 1,024 teaching encounters: 768 supported, 256 unsupported.
- 256 article-disjoint final probes: 192 supported, 64 unsupported.
- 32 early TRAIN probes: 24/8 from the first 256 encounters, selected by ID hash.
- 32 late TRAIN probes: 24/8 from the last 256, selected the same way.

The latter two are subsets of teaching, not new exposure. Probe early TRAIN after
encounter 256 and both sets at the final checkpoint, with updates disabled.
These measure fit/stability with the passage visible, not delayed absent-source
fact memory. Article disjointness concerns this run's supervision; the frozen
encoder has external sentence/QA/code-pair training and unknown contamination.

The retained [selection](../../experiments/r4_integrated/public_qa_selection_v1.json)
has SHA256 `72fa7ac873643be36701c486a66c24ae67631fcd95c6fe18c2021539dd3cfe76`.
Its original TRAIN download has SHA256
`68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002`.
Selection gives 315 teaching articles/994 paragraphs and 80 probe articles/247
paragraphs, with zero article, paragraph, question-ID or exact-context overlap.
All 960 supported items retain their original accepted annotations. The 18,583
eligibility/deduplication exclusions and source/license custody are retained in
the [preparation evidence](../../benchmarks/public_qa_preparation_v1.json).

## Fixed learning and comparisons

One pass; after each new lesson except the first, replay that arm's immediately
preceding lesson. This gives 2,047 offers per updating arm. Retain the existing
cell initialization, objective, Adam rate .001 and clipping at norm 1. Do not
add epochs, select replay by outcomes, tune thresholds or change encoders.
An unanswerable label concerns its provided paragraph, not an enlarged corpus.
Dataset labels are supervised lessons, never fabricated Python PASS/FAIL events.
Record all five policies' first attempts before any arm updates on each encounter:
5,120 predictions, followed by 6,141 total update offers. The fixed early, final
early-then-late and article-disjoint probes add 1,760 read-only predictions.

Compare aligned cells and trained heads; the identical cells with all learning
disabled; frozen cell producer with trained heads; an independent trainable
attention/feedforward reader; and the existing sensory-only cosine reference.
All receive identical raw/token inputs and legal candidate spans. The conventional
producer has no recurrence: shared 64-dimensional token projection, learned
query attention, and a learned fusion of document/query-context vectors supply
the same span/STOP heads. Its 37,377 parameters are close to the aligned reader's
37,313. Precisely, q=tanh(W*xq+b), d=tanh(W*xd+b), attention=softmax(d*A*q.T/8),
context=attention*q, and h=tanh(F*[d;context]+bf). Both retain the existing modality
tags, three start/end/mean span heads and token-weighted mean STOP head.

The conventional control uses seed 0, drawing W[64,386] as normal/sqrt(2),
F[64,128] as normal/sqrt(128), then span[3,64] and STOP[64] heads as
normal*.01/sqrt(64), in that order. A=I64 and all biases are zero. Adam uses
rate .001, betas .9/.999, epsilon 1e-8 and active-coordinate norm clipping at 1;
unsupported updates leave span weights, moments and steps unchanged.

Qualification preserved one failed initial variant: normal/sqrt(386) W solved
only two of three fixed synthetic tasks after 255 offers. One separately recorded
same-budget diagnostic reproduced the failure. A single initialization correction
accounts for unit-normalized token features plus one modality flag (squared norm
2); it passed the unchanged three-task, 255-offer check. Six focused numerical,
optimizer and behavior checks pass on this corrected `normalized_input_v2` source.
The failed variant remains failed. No public-QA or motor outputs selected this
correction. Initialization differs from the frozen cell control, so this comparison
does not isolate recurrence alone. Exact source hashes and both variants' evidence
belong to the preparation receipt; no further variant is authorized in this run.

After learning, return **inside the same run** to the original exposed 64-request
Python workload, once in each wording, with all 16 retained sources. Keep the
same trained cores, updates disabled, actual literal-edit motor and real execution.
There are 640 policy attempts plus two shared motor demonstrations, no teachers,
gold-source input, question rewriting or retries. These measure transfer of the
training back to the project, not fresh evaluation or autonomous recovery.

## Decision rules

Separate useful acquisition from readiness of the returned action loop:

- Public QA: at least 96/192 exact supported answers and 52/64 correct STOPs;
  at least 20 more supported successes than each disabled-learning and
  frozen-producer control. Report all STOP differences; an always-STOP control
  does not impose an implicit 100% STOP requirement.
- Final TRAIN fit: at least 36/48 supported answers and 13/16 STOPs, with at
  least 18/24 and 6/8 respectively in each early/late subset. Early scores may
  lose at most two supported answers and one STOP since encounter 256.
- Conventional comparison: require no fewer supported public-QA successes and
  at most three fewer STOPs, while retaining the absolute floors. Report paired
  outcomes and costs; these chosen margins are not significance estimates.
- Motor readiness: 44/48 exact supported executions and 15/16 correct STOPs
  in **both** wordings, with decoys unchanged on every trained-core attempt.
  This is a new translated readiness criterion. It does
  not preserve earlier protocols' growth/stratum checks or pass their gates.

Passing acquisition but failing motor readiness is useful bounded reading
evidence with an unresolved integration failure. It does not earn general
integrated adoption. If only the conventional learner succeeds, investigate the
cell computation/update using that comparison. If both fit TRAIN but fail transfer,
report the domain/generalization limit. If neither fits, representation versus
optimization remains unresolved; stop this comparison without automatic expansion.

## Budget and return

Preparation is at most two hours. One subsequent two-hour run includes encoding,
all arms/offers, fixed probes, motor execution and authentication. A bounded
synthetic numerical/runtime preflight is part of that run budget and must admit
the full roster before scored work. That one preflight has a 900-second sublimit;
it uses synthetic maximum public-input shapes, original motor shapes, one neutral
256-token encoding, and one neutral Python child. It prices each policy's reads,
both supported/STOP update paths including their post-update forward, inventory
construction, all unique encodings and 642 Python executions. It reserves 120
seconds each for motor/custody overhead and artifact output. Synthetic update
states are discarded; acquisition starts fresh. Keep peak parent working set below 2 GiB and
retained token-vector cache below 768 MiB; measure raw/derived state separately.
Build span inventories on demand rather than retaining every candidate object.
Price early/final TRAIN probes, all-source action work and audit output as well
as training. Resource failure retains partial evidence and never passes.
The combined clock starts at preflight entry. Exact committed sources, encoder
files and loaded native library are authenticated. The JSON/NPZ report is followed
by a small authoritative `.completion.json` receipt: its status and report hash
must agree before admission or a pass. It measures elapsed time and peak parent
memory after primary artifact serialization, writing and hashing. Only Python
bootstrap and the completion receipt's own small write are outside that measure.
Original errors survive checkpoint/report export failures; missing or failed
completion receipts never pass. No intermediate score changes the fixed schedule.

Preserve concise, source-bound lessons, checkpoint/optimizer lineage, selected
original byte spans, paired outcomes and measured costs. Keep useful numerical
checks proportional to the new conventional producer; old frozen modules remain
unchanged. Return to the persistent executed-loop objective at the budget boundary.
Primary QA scoring accepts an exact original accepted answer text at any candidate
occurrence; saved `best_gold_rank` and `conditional_best_is_gold` diagnostics
specifically concern annotated occurrences and do not replace primary correctness.
Official reading preparation remains complementary under the
[exam plan](reading-comprehension-evaluation.md); no new exam run is authorized by
this acquisition design.
