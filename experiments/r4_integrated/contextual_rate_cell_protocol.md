# One rate-cell reading acquisition preflight

Public exposed-TRAIN development, 2026-09-05. Preparation runs from
**09:19:51 to 11:19:51 UTC**. Freeze source, this protocol, numerical policy and
artifacts before any target-scored fit. Then allow one 15-minute numerical/cost
preflight and one 30-minute acquisition, at most 45 minutes total. No retry,
initialization/rate/window/label search or fresh/protected comparison follows.
**Numerical policy fixed below; held-source/API review precedes the single scored run.**

## Question and fixed inputs

The [rate-cell plan](../../docs/research/contextual-rate-cell-preflight-plan.md)
selects learned sequence/span computation and separated answerability credit.
Earlier full-choice acquisition and joint-ranking feasibility remain failed and
inconclusive respectively; neither proved fixed-feature impossibility. Compare
learned cells against matched trained-head controls, not historical AROW scores.

Reuse `contextual_full_choice_train.json`, SHA256
`9c1055c8c1fbfc3f5e92d997c5e335913619962c6aa47f0bdf9b984a005f4059`:
64 actual requests (48 supported, 16 STOP), 64 alternate requests, 16 TRAIN documents,
4 growth-only documents and the unchanged 256-encounter/four-pass schedule.
TRAIN contains 3195 raw bytes; growth adds 934. These are openly exposed teaching
and diagnostic inputs, not a fresh cohort or independent official reading test.
Pin the existing local MiniLM encoder/model identity; retain raw UTF-8 sources,
frozen token features/offsets and encoder identity. No pretrained answer head,
entity/property/gold-source filter or task-specific span boundary rule is allowed.

## Computation and exact supervision

Use 64 float64 leaky rate cells on Unicode regex words: mean-pool all complete
frozen MiniLM subtokens of each word, then L2-normalize (floor 1e-12). This revises
the earlier subtoken sketch explicitly. Input is 384 word coordinates plus query/
document indicators [1,0]/[0,1]. Process query then document words; reset to zero
per 48-word nonoverlapping source window. Learn producer parameters and
start/end/interior span heads by disclosed BPTT with the frozen update policy.
This is conventional gradient training of simulated rate neurons, not local
biological plasticity. Encoder weights remain frozen. The 64-cell rate simulation
directly supplies span/STOP scores; there is no extra evidence GLIF race. The
previously learned motor retains its GLIF action path. Disclose this readout
change from earlier baselines; matched controls test learned cell contribution.

Candidates are all contiguous word-boundary spans inside each frozen window,
including lengths above 3 words: 8162 TRAIN occurrences, 2702 growth additions,
10864 after growth. No current document is split; cross-window answers remain
unsupported generally. Preserve UTF-8 raw slices and complete-word/subtoken
alignment. All five arms use exactly this broadened inventory.
No candidate receives a stored per-span copy of the old 1621-feature matrix.

Binary answerability uses `softplus(z_STOP)-y_STOP*z_STOP` on every lesson.
The supported conditional-span objective is
`L_span=logsumexp(all span scores)-logsumexp(all exact-answer occurrence scores)`.
Supported loss is gate+span with equal coefficients; STOP loss is gate only.
On STOP, skip span-head parameters, Adam moments AND their counters. Exact-byte
duplicates are all targets without provenance labels. Shared producer updates
can still interfere; clip only active learnable coordinates after arm/STOP masks.

Inference is strictly hierarchical: **p_STOP>=0.5 means STOP**; otherwise emit
the unique maximum conditional-span score, with a span tie also producing STOP.
Never compare STOP against `(1-p_STOP)*p_single_span`. No threshold tuning,
external fallback or teacher-conditioned decision occurs. A unique span maximum
wins even when all span logits are negative. Independent numerical score/gradient
checks verify rate-cell computation; no redundant raw/GLIF parity is claimed.

Each encounter first makes five label-blind real attempts through the same
previously learned neuronal literal-edit motor in source-absent workspaces.
Then execute one independent teacher action shared across arms: supported exact
literal repair or explicitly authorized unchanged-workspace STOP check. The latter
uses the declared stale baseline and proves no-action preservation only. Every
updating arm binds its OWN first decision/execution to this shared teacher;
never borrow another arm's attempt. Already-correct encounters still receive
supervised teaching. Unexecuted competitors get surrogate gradients, never fake
FAIL receipts; teacher repairs never count as learner successes.

## Five arms, replay and evaluation

1. TRAINABLE: producer and all heads learn.
2. FROZEN_PRODUCER: identical initial producer stays fixed; identical heads learn.
3. BYPASS: fixed query-conditioned projection replaces recurrence; matched heads learn.
4. FROZEN_EVIDENCE: treatment initialization, producer and heads both fixed.
5. EXTERNAL: fixed pretrained-token selector; no producer/head learning.

BYPASS uses `tanh(W0*[docvec,0,1]+W0*[mean(querywordvec),1,0]+b0)` at each
document word, with treatment's fixed initial W0/b0 and no recurrence. It is a
different fixed computation, not the producer-learning ablation. EXTERNAL picks
the unique maximum cosine between normalized span-word mean and normalized
query-word mean, without a learned gate; empty/exact ties STOP. The first two
arms share identical initialization; all share sources, candidate family, motor.
Frozen evidence receives the declared authority/order but mutates no learned state.

Seal the complete TRAIN source pool before any teaching. Each updating arm gets
256 new lessons and 255 replays: after each encounter except the first, replay
that arm's immediately preceding immutable lesson (a distinct task). Thus 511
offers per updating arm, no extra teacher executions or loss-driven scheduling.
Replay recomputes current trajectories/scores/gradients from retained inputs;
historical eligibility is not current credit. Growth closes further learning.

Execute 64 actual and 64 alternate probes before growth, append the four growth
documents without updates, then repeat both sets. Five arms give 1280 teaching
first attempts and 1280 evaluation executions; 256 shared teacher actions yield
**2816 executions**, excluding the two motor demonstrations. Count replay/offers,
actual teacher repairs, STOP checks and parameter updates separately.

Treatment must reach 44/48 supported actual executions before AND after growth,
15/16 correct STOPs in ALL four phases, and 80% in every declared group in each
phase: role>=26/32 (supported+STOP), positive length>=13/16, form>=7/8 for each of
8 forms. Actual/alternate denominators stay separate. Actual total correctness
may lose at most 3/64 after growth (five percentage points), while gates still hold.
In BOTH actual phases, treatment must exceed EACH trained-head control by at
least 8 supported executions without fewer correct STOPs. A tie with frozen
producer attributes improvement to heads/design, not learned cell computation.
Failure closes this candidate; a pass earns a reviewed integration decision only.

## Numerical, state and cost freeze

Initialize seed 0: W[64,386] normal/sqrt386, U=.9I, cellbias=0; spanheads[3,64]
and stophead[64] normal(.01/sqrt64), stopbias[1]=0. Use default_rng(0), drawing
W, spanheads, stophead sequentially; deterministic U/bias initialization consumes
no RNG. Span score is start-head dot h_start + end-head dot h_end +
interior-head dot mean(h_interval). STOP logit is stophead dot the word-weighted
mean of ALL document-word states across windows plus stopbias (no query/padding).
Recurrence is `h_t=.5*h_prev+.5*tanh(W*x_t+U*h_prev+cellbias)`. Use Adam
eta=.001, betas=.9/.999, eps=1e-8, no decay; global L2 clip 1 over active gradients.
Each active parameter has its own step counter: m=.9m+.1g, v=.999v+.001g**2;
subtract .001*(m/(1-.9^t))/(sqrt(v/(1-.999^t))+1e-8). Inactive parameters,
moments and counters stay exact. Frozen producer updates only heads; fully
frozen updates nothing. The hierarchical decision directly emits the selected bytes.
One full parameter set has 29121 float64 coordinates (232968 bytes), plus gradients
and retained Adam states; count allocated frozen/bypass arrays too. Use existing
NumPy runtime; no new autodiff dependency is assumed. Before Python starts, set
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`, explicit CPU and
the checkout native cache; reject and record any different runtime environment.

Before acquisition, synthetic finite differences verify recurrent/endpoint/
interior/answerability gradients (central epsilon 1e-6, abs 2e-7 + rel 2e-5),
duplicate marginal targets, zero span-head gradient on unsupported lessons, clipping and changed-producer replay. Verify read-only
inference and the exact frozen-producer/frozen-evidence/bypass mutation masks.
Use four-to-eight-word synthetic answers and representative full windows and
inventory sizes. Real token custody tests cover UTF-8 boundaries, retained raw
slices, resets and cross-window limits. Synthetic lessons are not capability
scores and may not select another configuration after failure.

The 15-minute preflight binds held-source independent numerical receipts, then
measures seeded synthetic full inventories (16 documents × 34 words = 9520 spans;
20 × 37 = 14060 spans; 12-word queries; independent unit-vector seed 1),
all five prediction paths, supported/STOP offers, and a six-word learned-motor/
real-Python roundtrip. It records source hashes, numeric `started_unix`, elapsed
seconds, parent RSS and projected acquisition seconds. Pass requires elapsed
<=900 seconds and projected acquisition <=1800 seconds; the acquisition also
ends within 2700 seconds of this preflight start. The projection estimates
full-inventory query/offer cost and the 2816-execution, 511-offer-per-updater
runtime, including frozen offers and all five control paths. Stop before teaching
if it cannot fit 30 minutes. Parent resident memory must remain <=2 GiB and
retained learner payload <=1 GiB; count all five arms, shared buffers once, encoder, motor, raw/token
storage, parameters, gradients, optimizer state, replay ledgers and trajectory
caches. Report parent/child memory scope and actual phase wall times. No silent
batch/window truncation or candidate filtering to meet the limits.

Bind working and Git source hashes. Require exact committed bytes except the
already archived full-choice helper, where documented working CRLF-to-Git LF
equivalence is allowed; never rewrite that old source. Preserve synthetic execution
workspaces in ignored scratch and publish immutable UTF-8/LF JSON.

Hash source/provenance/raw bytes, token arrays/offsets, windows, candidate intervals,
configuration and producer/head state. Each decision binds current input/source
universe and parent; each offer binds own original attempt, teacher authority,
current loss components/gradient hashes/norms/counters and resulting state.
Export initial/final parameters AND Adam moments/per-parameter counters, fixed
projection state and immutable encoded inputs/lesson ledgers. Independently
recompute gradients/updates from saved inputs (numerical reference tolerance
1e-10+1e-10*abs(reference), exact stored-state hashes); do not store 29k-gradient JSON
per offer or giant trajectory histories. Record all retained/cache/export bytes.
Count all audit artifacts/copies separately; they never enter inference.

Check input/source hashes before and after the run. Inference, alternate probes
and growth leave learned bytes and learning counters exact; storage identity may
grow. Preserve immutable JSON/NPZ and completed execution rows on failure; no
unreported resume. No target labels drive inference, diagnostic selection or
replay. Finish the bounded attempt with executed scores, control differences,
resource receipts and limitations. Authentic reading remains a separately frozen
[official companion](../../docs/research/reading-comprehension-evaluation.md);
this preflight opens no exam content and does not pass G0 or protected gates.
