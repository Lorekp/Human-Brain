# One query-aligned rate-cell acquisition comparison

Exposed-TRAIN development, 2026-09-05. Prepare from **10:28:57 to 12:28:57 UTC**
at base `6b9607a`. Freeze this protocol, source, configuration and input identities
before target-scored learning. Then allow one numerical/cost phase of at most
15 minutes and one acquisition of at most 30 minutes, 45 minutes combined.
No retry, seed/rate/window/attention sweep, extra pass or label revision follows.

## Question and shared inputs

The [reviewed failure plan](../../docs/research/rate-cell-failure-review.md)
selects direct query access during document processing. The preceding reader's
failed conditional choices do not prove lost query information, an inadequate
representation, or shared-gradient interference as the unique cause. Test whether
learned alignment improves executed acquisition beyond trained-head controls.

Reuse `contextual_full_choice_train.json`, SHA256
`9c1055c8c1fbfc3f5e92d997c5e335913619962c6aa47f0bdf9b984a005f4059`:
64 requests, 48 exact-answer and 16 authorized STOP labels, 64 alternate requests,
16 TRAIN documents and 4 growth-only documents. Raw sources total 3195 TRAIN
bytes and 934 growth bytes. This is openly reused acquisition development, not
a new transfer cohort. No official/protected content is opened in this attempt.

Retain complete raw UTF-8 sources, word offsets and pinned MiniLM identity.
Mean-pool all complete frozen subtokens of each Unicode regex word, then normalize
its 384-vector with denominator `max(norm,1e-12)`. Add query/document indicators
`[1,0]`/`[0,1]`. All arms share 48-word nonoverlapping source windows and every
contiguous word-boundary span within them: 8162 TRAIN occurrences, 10864 after
growth. The persistent/audit inputs are pooled word vectors and word offsets;
subtoken counts are recorded, but raw subtoken vectors/offsets are temporary
encoder outputs. No current document is split. Cross-window spans remain unsupported;
there are no entity/property/source labels or answer-length filters in inference.

## Fixed learned computation

Use 64 float64 rate cells with leak .5. Prefix query states `u_j` use the existing
recurrence from zero in each window. For each document position define
`r_t=W*x_doc+b`, `s_tj=r_t^T*A*u_j/8`, `a_t=softmax_j(s_tj)`,
`c_t=sum_j(a_tj*u_j)`, then
`h_t=.5*h_prev+.5*tanh(r_t+U*h_prev+V*c_t)`.
Retain the last query state as the initial document state. With no query words,
context is zero and attention has shape `[windows, document_words, 0]`.
Shapes are `u:[windows,Q,64]`, `r/c:[windows,L,64]`, `a:[windows,L,Q]`.
Only valid query positions enter attention; query/padding states do not enter
document pooling. Attention reads no labels, candidate indices or lesson ledger.

Initialize exactly once with seed 0. Draw `W[64,386]=normal/sqrt(386)`, then
span heads `[3,64]` and STOP head `[64]` from `normal*.01/sqrt(64)`, in that order.
Set `U=.9I`, cell bias zero, STOP bias zero, `A=I64`, `V=I64`; these consume no
additional RNG draws. Parameter order is the old six arrays followed by
`attention_weights` and `query_context_weights`. Preserve the old span score:
start-head dot first state + end-head dot last state + interior-head dot interval
mean. STOP logit uses the word-weighted mean of all document states plus bias.

Supported loss is `softplus(z_STOP)+L_span`, where
`L_span=logsumexp(all spans)-logsumexp(all exact-answer occurrences)`.
STOP loss is `softplus(z_STOP)-z_STOP`; it supplies no span-head gradient.
Every exact-byte duplicate is a target without an added source/provenance label.
Use Adam eta .001, betas .9/.999, epsilon 1e-8, no decay, global L2 clip 1 over
active learnable coordinates. Each parameter has its own counter. STOP skips
span-head parameters, moments and counter; frozen producer controls skip W/U/b/A/V
and their optimizer state. All other active parameters use ordinary bias-corrected
Adam. Replay recomputes current alignment, trajectories and gradients.

Inference remains hierarchical: `p_STOP>=.5` means STOP. Otherwise emit the unique
maximum span score, even if negative; an exact span tie or empty inventory stops.
Do not compare STOP probability against an individual span's probability mass.
No threshold rescue, answer-conditioned fallback or extra evidence GLIF race is
added. The existing neuronal literal-edit motor retains its own GLIF path.
Learning here is disclosed attention and BPTT/Adam over simulated rate cells,
not local biological plasticity or a newly discovered neuronal learning rule.

## Six arms and authentic teaching

- `trainable`: aligned core, all active producer/head parameters learn.
- `prefix`: unchanged `RateCellReader(mode='trainable')` in this shared run.
- `frozen_producer`: aligned computation, fixed W/U/b/A/V, trainable heads.
- `bypass`: fixed query states `u_j=tanh(W0*x_query+b0)`; use the same attention
  formula and `h_doc=tanh(r_t+V0*c_t)`, with no recurrence U. W/U/b/A/V stay fixed;
  only heads learn. This is a different fixed computation, not the producer ablation.
- `frozen`: aligned initial core with all parameter/moment/counter mutations off.
- `external`: fixed normalized span-word-mean/query-word-mean cosine; unique
  maximum wins regardless of sign, empty/exact ties STOP. Allocated aligned core
  remains frozen; its supervised receipt computation never changes this selector.

All five aligned cores share initial arrays; the prefix shares the old six.
The complete TRAIN pool is fixed before teaching. Use the unchanged 256-encounter,
four-pass schedule. Every encounter first makes six label-blind real attempts in
source-absent Python workspaces through the same motor. Then execute one shared
teacher: exact literal repair for support, or authorized unchanged-stale-workspace
check for STOP. The STOP check proves no-action preservation, not a successful
unsupported repair. Each arm binds its own original attempt to that teacher.
Correct first attempts still receive teaching. Unexecuted competitors receive
surrogate supervised gradients, never fabricated FAIL receipts.

Each arm receives 256 new lessons and 255 replays: after every encounter except
the first, replay that arm's immediately preceding immutable lesson. Thus 511
offers per arm, no extra teacher executions or outcome-driven replay selection.
Frozen offers advance disclosed offer/receipt counters but not learned arrays.

After teaching execute 64 actual and 64 alternate queries; append all four growth
documents with updates disabled, then repeat both sets. Count 1536 first attempts,
1536 evaluation executions and 256 teachers: **3328 executions plus 2 motor demos**.
Record 256 additional read-only aligned diagnostics across the four probe phases:
conditional gold winners/ranks/gaps separately from STOP rejection and execution.

## Decision, numerical and resource contract

Require aligned actual support >=44/48 before and after growth, STOP >=15/16 in
all four phases, and >=80% in every declared phase group: role >=26/32, positive
length >=13/16, each form >=7/8, unsupported group >=13/16. The stricter aggregate
STOP threshold still applies. Actual total correctness may lose at most 3/64
after growth while all thresholds hold. In both actual phases require at least
8 more supported successes than EACH trained-head control, with no fewer STOPs.
Report the prefix comparison separately; historical scores are not matched arms.
Failure, timeout or missing learned-cell advantage closes this candidate; partial
loss/rank progress does not authorize another variant. A pass earns integration
review, not general coding, fresh-transfer, source-free memory or protected gates.

Before scored acquisition, independently verify attention/recurrent/span/STOP
gradients, duplicate marginal targets, empty-query behavior, query/padding masks,
clipping and inactive Adam states using synthetic inputs. Central finite differences
use epsilon 1e-6 and tolerance `2e-7+2e-5*abs(reference)`. Saved numerical replay
uses `1e-10+1e-10*abs(reference)` and exact stored hash authentication; numerical
roundoff is not permission to change a gate decision. No actual TRAIN preview
selects configuration. Bind held-source numerical receipts to the cost check.

Each aligned core allocates 37,313 float64 parameters, 298,504 parameter bytes;
two moments plus 8 packed counters give 895,576 checkpoint bytes. Five such cores
plus the prefix allocate 5,176,832 checkpoint bytes before inputs/trajectories.
Use corrected `rate_cell_session_support.retained_size(*roots)`: shared data and
array owners count once; all instance dictionaries and inherited slots are
traversed, including `__main__`. Validate the real `python -m` entry with synthetic
optimizer/shared-input objects. Reject unsupported opaque leaves instead of
silently counting headers. Exclude code objects and separately disclose native/
encoder allocation scope; parent RSS and encoder-file payload remain mandatory.
Require counted payload <=1 GiB and parent RSS <=2 GiB before and during acquisition.

Set CPU, checkout native cache and OPENBLAS/OMP/MKL thread variables to 1 before
Python. One synthetic full-inventory cost check must project all six control paths,
3066 offers, 3328 executions, 3072 first/evaluation predictions and 256 aligned
diagnostics within 1800 seconds. Its own wall time is <=900 seconds; combined
numerical/acquisition time is <=2700 seconds. Record projection omissions honestly.
Use seeded unit-vector synthetic inputs (seed 1), a 12-word query, 16 documents
of 34 words (9520 spans) and 20 of 37 words (14060 spans), covering the real TRAIN
and grown inventory sizes. No synthetic parameter is selected from target scores.
Require a passed source-bound actual-entry payload smoke receipt alongside the
independent numerical receipt before this cost phase. Acquisition timing includes
checkpoint export; final JSON serialization/write is reported as excluded.
Preserve all sources, candidate intervals, raw/query/input hashes, own decisions,
teacher/replay custody, initial/trained/final parameters and optimizer states.
Inference and growth must preserve learned bytes/counters exactly. Immutable
UTF-8/LF JSON/NPZ retain partial rows on failure; synthetic execution workspaces
remain inspectable. Audit artifacts never enter inference. Authentic reading still
requires the separately sourced/frozen [reading plan](../../docs/research/reading-comprehension-evaluation.md).
