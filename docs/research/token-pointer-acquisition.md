# Token-pointer acquisition: decision and preparation contract

Selected 2026-09-05 after the [pooled-source failure](../../experiments/r4_integrated/fresh_source_result.md).
Research window: **18:41–19:26 UTC**, at most 45 minutes. This review selects one
implementation and comparison; no experimental model was downloaded or run.
Research and decision review completed at **19:09 UTC**, before publication.
Preparation is next. Freeze its completed code, data and receipts before the one
comparison below. Earlier experiments and their failures remain unchanged.

## Decision and evidence

Keep the pinned generic MiniLM features. Replace the current reading recipe with
bidirectional rate-cell processing, learned tokenwise question/context fusion,
and start/end/NULL prediction. Use a conventional BiLSTM reader as a positive
learning reference and strong controls for learned cell contribution. This is a
new recipe comparison, not an isolation of one equation or a BiDAF reproduction.

The present cells already learn recurrent coupling and query attention. Their
failure does not prove those mechanisms cannot work. However, source inspection
finds three reasons not to repeat the old recipe:

- Unit word vectors plus a segment bit have squared norm 2. Cell input weights
  use variance `1/386`, while the conventional comparator uses `1/2`: initial
  input-drive variance differs by **193 times**. The impact on learning remains
  unmeasured. [Cell initialization](../../src/humanbrain/rate_cell_reader.py),
  [conventional initialization](../../src/humanbrain/conventional_attention_reader.py).
- Paragraph-local conditional span loss is unchanged by a common offset to all
  that paragraph's scores. The global mean-state STOP head also changes when
  irrelevant documents are appended. Neither establishes calibrated pool-wide
  answerability. [Reader/objective](../../src/humanbrain/query_aligned_rate_cell_reader.py),
  [training inventory](../../experiments/r4_integrated/public_qa_acquisition.py).
- The broad run gave each of 1,023 questions two presentations and the last one
  one; fitting the sixteen-case set used 256 per question. Those outcomes neither
  establish adequate varied training nor justify more epochs on sixteen answers.

BiDAF supplies a relevant learned computation: bidirectional contextual encoding,
tokenwise context-to-query and query-to-context attention, multiplicative fusion,
then contextual modeling and endpoint prediction. Its reported SQuAD1.1 recipe
used roughly 90,000 pairs, 12 epochs, batch 60 and about 2.6M parameters, taking
about 20 hours on a Titan X. Its 68.0 EM/77.3 F1 score has no unanswerable-question
claim. Borrow the computation principles, not its accuracy or obsolete runtime.
[Original paper](https://arxiv.org/pdf/1611.01603),
[authors' implementation](https://github.com/allenai/bi-att-flow).

BiDAF-No-Answer adds a shared learned null bias to both endpoint distributions.
Its separate slot-filling transfer experiments used much broader supervised
experience; they are not evidence of few-shot reading. SQuAD2 adds adversarial
unanswerable questions and separately evaluates this family. Keep original
annotations, including multiple accepted occurrences and unsupported cases.
[BNA paper](https://aclanthology.org/K17-1034.pdf),
[SQuAD2 paper](https://aclanthology.org/P18-2124.pdf).

Clark and Gardner show why paragraph confidence needs training across the scope
where answers compete. Their shared-normalization method samples paragraphs
from context documents. It does not make SQuAD2's paragraph-local STOP labels
valid over arbitrary unrelated documents. Broad acquisition below keeps local
label scope; only the reviewed integration pool supplies cross-source teaching.
[Multi-paragraph reading paper](https://aclanthology.org/P18-1078.pdf).

The researched alternative is frozen joint question/passage hidden states from
the cached `optimum/roberta-base-squad2` export. Its upstream card describes two
epochs of SQuAD2 QA tuning. Final states already support a linear answer head,
so learning another head on SQuAD can be surrogate-head learning. The local ONNX
export exposes logits only; adding hidden outputs needs a separately qualified
graph derivative. Select the generic-feature recipe first. RoBERTa is not a
second queued fit. Its earlier custom-language failure remains valid and does
not establish failure on natural SQuAD questions.
[Model card](https://huggingface.co/deepset/roberta-base-squad2),
[pinned export](https://huggingface.co/optimum/roberta-base-squad2/blob/db07770d7fa024df076e60947b445b0b3290f8eb/README.md),
[ONNX extraction API](https://onnx.ai/onnx/api/utils.html#extract-model),
[local result](../../experiments/r4_integrated/frozen_reader_qualification_result.md).

MiniLM is itself external knowledge: its sentence-contrastive training includes
QA/code-related pairs. Separate question/paragraph encoding supplies contextual
word features, not context-free words or a clean pretraining boundary. It supplies
no selected answer, source shortlist or action in this candidate.
[MiniLM card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## One concrete reader

Use the existing coupled rate equation `h' = .5*h + .5*tanh(W*x + U*h + b)`.
Two directions, 64 cells each, encode question and each complete paragraph;
encoder weights are shared between question and paragraphs. Reset state at
sequence boundaries and exclude padding in both directions. Retain every token
state. A learned trilinear similarity and the two attention directions produce
`[h; q_att; h*q_att; h*h_att]`. A second bidirectional 64-cell modeling layer
processes this fusion. Two linear heads on its 128 states predict endpoints.
This changes processing and supervision, without claiming novel cell dynamics.
Trainable rate-network dynamics have an established research basis; that is not
evidence this reading adaptation works.
[Rate-network framework](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004792).

Use fixed, non-affine layer normalization before each input projection, Xavier
input/head weights, `U=.9*I`, zero biases, and seed 20260905. The conventional
reference replaces both rate-cell stages with 64-unit-per-direction BiLSTMs;
use documented Xavier input, orthogonal recurrent and unit forget-bias
initialization. Share features, fusion, endpoint objective, data and exposure;
report its larger parameter count and measured cost rather than claiming parity.
Apply the same 0.2 input dropout during training to every sequence stage, including
frozen-producer stages. Hold each sequence's mask across token steps and use
matched mask streams where shapes agree. Disable dropout for every probe/action.
No initialization, width, leak or rate sweep is part of this comparison.

Append one shared learned NULL bias to each endpoint score vector. Minimize
`-log(sum_(s,e in original accepted pairs) p_start(s)*p_end(e))`; an unsupported
lesson targets `(NULL,NULL)`. Normalize each endpoint across the complete supplied
inventory plus NULL. The product is not a distribution normalized over only legal
spans. Decode the best legal logit sum `start[s]+end[e]` within one original
paragraph and compare it with `2*null_bias`. STOP if NULL is at least as large,
or if multiple distinct legal occurrences share the exact best score.
Mixed NULL/word pairs and
cross-paragraph spans cannot be emitted. Return the original document identity
and UTF-8 byte interval; no answer-type or gold-length filter. All within-paragraph
word-boundary intervals are eligible, including those crossing the old 48-word
boundaries. A running best start avoids materializing every span feature.

Use batched float32 PyTorch/autograd, including its BiLSTM and Adam, as a research
dependency in the checkout environment. Do not write another manual gradient or
optimizer engine. Qualify the custom cell recurrence and annotation objective
with small float64 finite-difference checks, plus masks, restore and real-action
behavior. This is a new arithmetic contract, not replay parity with old NumPy.
[Autograd](https://docs.pytorch.org/docs/stable/autograd.html),
[LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
[gradient checking](https://docs.pytorch.org/docs/stable/notes/gradcheck.html).

## Data, learning and controls

Use only the already pinned SQuAD2 TRAIN download. Keep the existing whole-article
hash split and original bytes/annotations. Select **8,192 teaching questions**
(6,144 supported, 2,048 STOP) by the existing question-ID hash order, explicitly
including the sixteen old regression questions within those totals. Select **256
untaught questions** (192/64) from the article-disjoint side, excluding all prior
project-selected probe IDs. These are fresh to these learners, not independent
adjudication or certified free of external pretraining exposure.

Keep complete inputs within the unchanged encoder's 256-subtoken contract and
exact original word endpoints. Remove the old 48-word span-window restriction;
do not otherwise select by answer length or predicted difficulty. Count and
describe exclusions before outputs. The longer-input/structured-response gaps
remain visible. Freeze the actual IDs, article/paragraph overlaps and label
counts after preparation; insufficient coverage stops preparation, not a redraw.

Eight deterministic shuffled epochs with batch 32 give **65,536 presentations
and 2,048 optimizer steps per updating arm**. Use mean per-example loss, Adam
rate .001, betas .9/.999, epsilon 1e-8, no weight decay and global gradient clip 1.
Every batch contains 24 supported and 8 STOP examples; gradients share the same
pre-update state. This is substantially broader exposure than the old run but
still much less than BiDAF's training. Eight epochs are a bounded test, not an
assertion of sufficiency. No best-checkpoint selection or early accuracy exit.

| Policy | Purpose |
| --- | --- |
| Full rate-cell learner | Learn the entire reader and output heads. |
| Exact-initial-state frozen producer | Freeze both cell stages and attention/fusion; train only endpoint/NULL heads. |
| Core-disabled trained reader | Replace each recurrent stage by a dimension-matched tokenwise affine+tanh map; train these maps, attention and heads. No simulated recurrent state. |
| Conventional BiLSTM reader | Stronger established learning reference under the same representation/task. |

Keep a fully frozen initial cell copy for acquisition measurements. Before
integration, clone each trained policy exactly, including optimizer/counters;
its frozen clone supplies the matched continuation control. Also retain sensory
cosine as a diagnostic external-only policy, explicitly too weak to be the sole
reference. Full cells equaling the trained bypass do not show useful recurrence.

Probe a fixed 128-question TRAIN subset (96/32) after epochs 1, 4 and 8. Probe
the 256 untaught questions at initialization and after epoch 8 with no teaching.
Keep both exact supported answers and correct STOPs separate; record word-length
and question/source strata and the exact selected occurrences. No extra probe
feeds a curriculum, threshold or checkpoint choice.

## Return to the persistent action loop in the same comparison

Reuse the 28-question [fresh-source development fixture](../../experiments/r4_integrated/fresh_source_cases_v1.json)
as an explicitly exposed integration regression, not another independent fresh
cohort. Its twelve episode/later questions are excluded from broad teaching.
Keep its original two ambiguous items diagnostic-only and untaught; the original
aborted admission and frozen session remain untouched. Its existing full-pool
semantic review applies only while source/question bytes remain identical.

From each newly acquired policy, run the same cumulative-source schedule:
O16 initial, E8 sequential first attempts, L4 wholly untaught after learning,
then O16+E8 final; use its matched frozen clone alongside it. Each updating arm
gets the prescribed seven E lessons and sixteen old replays: **23 extra offers**.
These offers use shared endpoint normalization over the entire current pool,
providing an explicit cross-source learning signal absent from broad local QA.
Do not turn other TRAIN paragraphs into arbitrary globally negative examples.
This small continuation may be insufficient; its results decide that.

The four updating policies, their four frozen clones and sensory cosine give
nine integration policies: **468 primary actions and 468 oracle probes**, plus
two shared demos and at most 28 teachers (**498 child executions maximum**).
The four updaters receive **92 continuation offers** in total. The initial frozen
cell control is used for acquisition, not as a tenth integration policy.
Every primary policy sees all current paragraphs. Correct-paragraph probes stay
diagnostic. Feed the selected original bytes into the existing learned literal
motor and strict UTF-8 Python execution path. Teacher corrections use the same
failed workspace and remain external supervision. Freeze exact policy/probe/child
counts with implementation before running; do not rebuild large span inventories
merely to satisfy an obsolete index-based adapter.

Report broad acquisition and the exposed integration regression separately.
Useful recipe evidence requires a **20/192 or greater supported untaught gain**
and **at least five percentage points of balanced-accuracy gain** over each of:
frozen initial cells, the frozen-producer learner and the trained core-disabled
reader. Balanced accuracy is `(supported_correct/192 + STOP_correct/64)/2`;
the full cells must also preserve at least **48/64 correct STOPs**. Report all
lane counts and any abstention tradeoff; do not claim Pareto improvement when
STOPs regress. An always-STOP control has perfect negative accuracy but solves
nothing, so the criterion measures a declared tradeoff instead of requiring its
negative-only score to be preserved almost exactly.

The conventional reader must meet the same supported/balanced gains against its
own initial state, reach at least 48/64 correct STOPs, and fit at least half the
96 supported TRAIN probes; otherwise the positive acquisition control is
unqualified. These are prospective engineering margins chosen to reject a
handful of chance gains, not targets borrowed from another benchmark or
significance claims. One seed does not establish a robust architecture result.

Integration additionally requires at least two more supported E+L first-attempt
executions than the matched frozen cell parent, no loss of any initially correct
old decision in final pooled or oracle probes, and no worse scorable fresh STOPs.
Report absent headroom as inconclusive rather than replacing tasks. Tiny exposed
counts do not establish independent transfer. TRAIN fit alone, regression gains
alone, or correctness supplied by external teachers does not meet this recipe's
joint criterion. No outcome here alone adopts an architecture or passes the
roadmap's integrated development gate; fresh streams and its other requirements
remain necessary. Stop after this comparison, with no automatic extension.

## Preparation, resource budget and next action

The old broad run measured 54.338 seconds for three arms' 6,141 offers plus
5,120 first predictions and inventory/report work, and 49.585 seconds for 2,521
encoded inputs. Scaling that mixed package to four arms' 262,144 presentations
gives about 2,320 seconds, plus roughly 315 seconds encoding at the old word-row
density. This is only a cost basis: new recurrence, batching, precision, controls,
probes, pooled actions and serialization can change it substantially. It is not
a throughput promise or evidence that the old short run budget is adequate.

The retained 1,024-lesson input cache is reusable without encoding: 2,018 arrays,
370,787,328 float64 bytes. At its observed density, 8,192 lessons need about
1.483 GB of float32 features before other scopes. Use one bounded indexed cache,
reuse existing inputs by verified identity, and avoid compressed/uncompressed
duplicates. The old `TokenInput` copies to float64; do not route a supposedly
float32 full cache through that constructor. Hash the new representation and
stream batches and compact update journals instead of retaining every trajectory.

At 18:43 UTC this host had 33.38 GB physical RAM, 9.27 GB available, 4.29 GB free
disk and a 12 GB RTX 4070 SUPER. Torch is absent. Prefer its CPU-only installation
for this bounded first recipe; pin a supported wheel and dependencies after
checking download/install sizes. CUDA setup is not a prerequisite or a parallel
optimization queue. [Official installation guidance](https://pytorch.org/get-started/locally/).

Start one **four-hour preparation window** when implementation begins. Build the
batched reader/control, exact word-pointer input/action seam and bounded cache in
parallel. Use the original sixteen questions only for a declared disposable
small-fit preflight: exactly 256 batch-16 steps per full cell/conventional branch,
at most 15 minutes total. Each batch contains all twelve supported and four STOP
questions with their original paragraphs. Probe all sixteen before and after
training, with dropout disabled; require both branches to finish at 12/12 exact
annotated supported answers and 4/4 correct STOPs. The old O11 label limitation
remains disclosed. These 8,192 presentations and 64 read-only probes verify
small-set acquisition and estimate cost; their states never seed the comparison.
Freeze this fixture/criterion before its scored preflight. Preserve failures;
do not sweep designs.

Admit one run only if measured batch/encoding/probe/action costs conservatively
project its full prescribed work within **four hours**, including artifacts,
with **4 GiB peak parent working set**. Before loading/caching, verify the actual
byte plan leaves at least 1 GiB free disk and includes dependency extraction,
cache, checkpoints and journals. Retain all existing experiment archives.
If cost or correctness admission fails, report the precise blocker and evidence
without silently shrinking data, controls or exposure. No research-review loop
is required before implementing this selected recipe.

Source-absent knowledge, answer-free acquisition, genuine structured Amelia
responses and autonomous multi-step software recovery remain open. This reader
is a component aimed at the shared persistent workload, not its completion.
