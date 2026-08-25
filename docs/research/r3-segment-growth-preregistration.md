# R3 segment-growth pilot preregistration (path-2 mainline) — FROZEN BEFORE IMPLEMENTATION

Status: frozen 2026-08-25, before any winner-election or segment-growth
implementation exists on the GLIF3 dendritic lane. Branch `codex/path2-pilot`.
Sources adopted verbatim: `docs/research/path-decision-synthesis.md` @ e16d22c
(§Path 2 concrete-first-experiment + §2.3 adopt/adapt rows), HTM constants from
`docs/research/deep-dive-htm-tm.md` §1.2/§1.4/§Q3 (htm.core defaults verified
there: `activationThreshold=13` connected of `maxNewSynapseCount=20` grown,
`connectedPermanence=0.50`, learning-tier `minThreshold=10`, ±0.10 permanence),
book split from `cell-lm-corpus-bible.md` (PRIMARY gating split), cost anchor
from `roadmap-reading-to-coding.md` (1M-cell CUDA full-KJV ≈ 15 h, this box),
cell-only contract from `architecture.md`.

Nothing below may be edited after the first replay token is observed. Failure
is a publishable result. All thresholds literal; all arms seeded; no re-sweep,
no threshold edits, no seed changes, no alternate metrics after seeing results.

## 1. Scope (pinned)

This pilot converts the R3 episodic-store mainline from design to measured
mechanics: **winner-cell election + distal segment growth during passage
replay**, nothing else.

- Substrate: the existing GLIF3 dendritic lane (`architecture.md`: replayed
  sequences grow local synapses onto dendritic branches; NMDA-like plateau
  currents drive fitted GLIF3 cells; basket-cell race emits the first spike).
  What is added here is the missing piece named in [DD-HTM] Q3 mapping: a
  persistent winner cell per recurring context so repeated contexts converge
  instead of re-bursting, plus the growth/punishment rules below.
- Out of scope for THIS pilot: e-prop/three-factor credit assignment, race-bias
  consumption of predictive state (that is probe P-B), apical estimation
  (probe P-A), multiplicative `(1+β)` potentiation, any offline decoder head
  (§2.3 avoid row — recognition is scored by race/dynamics readouts only).
- Contract tripwires (mechanical, extend the §2.1 set): assert-no-forbidden-
  trainable-components (no gradient objects, no ANN modules anywhere in the
  diff); weight-snapshot clone-and-compare hash at every checkpoint;
  determinism: same (seed, corpus, constants) => bit-identical segment arrays
  and logs, always.

## 2. Frozen mechanism spec

Cell structure: each token identity owns a disjoint pool of `k = 32` candidate
cells (htm.core/NuPIC `cellsPerColumn`; the "column" of BAMI pseudocode maps to
the token pool). Feasibility assertion at build time:
`vocabulary_size × 32 ≤ configured_cells` (at vocab 299 per the R2 128k
manifest this is 9,568 cells — trivially inside 128k/1M). If a future corpus
violates it => STOP before the run; shrink the TRAIN book list (whole books
only, never splitting a book across sides), never the constants.

Distal segments: at most `maxSegmentsPerCell = 255` per cell, at most
`maxSynapsesPerSegment = 20` contacts per segment (**contact budget cap ≈ 20**,
adopted). Contacts carry float permanence `p ∈ [0, 1]`; connected iff
`p ≥ 0.50` (`connectedPermanence`). New contacts enter at `initialPermanence =
0.21` (htm.core default).

Recognition thresholds (two tiers, adopted):

| constant | value | meaning |
|---|---|---|
| `θ_a` (activation) | 13 connected-active synapses | segment depolarizes its cell (recall-competent; contributes plateau current) |
| `θ_m` (matching) | 10 potential-active synapses | segment "matching": eligible to learn / to win election |

Permanence dynamics (all ON, all local):

- Predicted-path confirm: active connected synapses on the confirmed segment
  `+= 0.10`; inactive synapses on that segment `-= 0.10`. No growth beyond cap.
- Growth toward recent winners: on an unpredicted step (burst), the elected
  winner's chosen segment grows up to `s = 20` new contacts sampled uniformly
  without replacement from the previous step's winner set `W_{t-1}` (seeded
  rng keyed `(construction_seed, step_index, cell_id)`); with cap = s = 20 a
  fresh segment fills on its first growth event ("θ ≈ 13 connected of ~20
  grown" [DD-HTM]; later events only adjust permanences unless removal freed
  budget). Contacts reaching `p ≤ 0` are removed and free budget.
- Negative-evidence decay (punish wrong predictor): **ON by default**
  (§2.3 adopt row; htm.core's default-off is explicitly not copied). Any
  segment that contributed depolarization toward a population which did not
  contain the actual next input gets `δ_neg = 0.04 × increment = 0.004`
  subtracted from its then-active-connected permanences (Frontiers eq. (7)
  analog). Same schedule in every growing arm; there is NO decay-off ablation
  arm in this pilot.

Winner-cell election (on burst of a token pool, one winner per burst):

1. Candidates: cells owning ≥ 1 matching segment (potential-active ≥ `θ_m`)
   against `W_{t-1}`.
2. If any candidates: elect the cell with the highest best-segment
   potential-active count; **tie-break**: fewest prior election wins, then
   lowest cell index (best-match / least-used tie-break, deterministic).
3. If none: elect the least-used cell (fewest owned segments, then fewest
   prior wins, then lowest index) and allocate exactly ONE brand-new segment
   grown toward `W_{t-1}`.
4. One-pattern-per-new-context allocation: at most one new segment per burst;
   a repeated (context, token) conjunction reinforces the existing
   best-matching segment and never allocates a duplicate while a matching
   segment exists (BAMI burst semantics, [DD-HTM] §1.4 lines 25–48).

`W_{t-1}` definition: elected winners of step t−1 for bursting pools; fired
(depolarized) cells themselves for pools that took the predicted path.

No eligibility traces, no third factor, no reward signal in this pilot — pure
TM-style local rules; the three-factor rows of §2.3 are follow-on work gated on
this pilot's outcome.

## 3. Corpus, split, seeds (pinned)

- Text: KJV via the PG10 pipeline and trim rule of `cell-lm-corpus-bible.md`;
  byte SHA-256 recomputed at load and asserted against the committed evidence
  manifest (mismatch => STOP, mirrors the ustate digest discipline).
- Split (PRIMARY gating split, whole blocks): TRAIN = 49 books (everything
  except the two held-out blocks); HELD-OUT = Minor Prophets #27–38
  (Hosea→Malachi) ∪ Gospels+Acts #39–43 (Matthew→Acts) — 17 books, author-
  disjoint by construction, no genre family split across sides.
- Token stream: the repo tokenizer with the vocabulary-alignment pass recorded
  in the R2 Bible-QA scale manifest (`experiments/bible_qa_scale/
 _qa_cuda_128k_run.json`, `corpus_sha256 8857eb77…`,
  `curriculum_sha256 ef24b1d3…` for the fixture lineage; the pilot logger
  recomputes and records both for the whole-book curriculum; any mismatch with
  its own committed manifest => STOP).
- Seeds: single construction seed `17` for ALL arms (continuity with the R2
  lane manifest). Seed-stability across seeds is a listed limitation; a
  multi-seed confirmation cohort is explicitly out of scope until after the
  verdict.

## 4. Arms (frozen)

- **A1 growth-on** (full mechanism): election + growth + negative-evidence ON.
- **A2 growth-off control** (burst baseline): identical replay and logging;
  election machinery runs for diagnostics but no segment is ever created and
  no permanence changes. Its predicted-path fraction is structurally 0; this
  identity is itself a tripwire (any nonzero value => leakage/bug => STOP).
- **A3 shuffled-winner control** (attribution): growth proceeds identically
  but presynaptic growth targets are sampled uniformly from the active cells
  of step t−1 instead of the elected winner set (destroys persistent-context
  binding; isolates what winner election buys beyond generic co-occurrence
  growth). Negative-evidence ON, same schedule.

Arms share corpus, split, seed, schedule, and instrumentation. No other arms;
no post-hoc additions.

## 5. Metric (frozen)

`f_pred(S)` = predicted-path fraction over sweep S: the fraction of replay
steps at which the arriving token's pool contained ≥ 1 segment-depolarized
(plateau-contributing) cell. Complement `burst_fraction` is logged per book
per pass (free anomaly metric, §2.1). Recognition is read from dynamics only;
no decoder touches these numbers.

Schedule: epoch-0 structural sweep (machinery armed, zero segments; asserts
`f_pred ≡ 0` in every arm — nonzero => STOP) → E = 3 ordered exposure passes
over TRAIN books (canon book order, verses in canon order) → ONE final
evaluation sweep over TRAIN ∪ HELD-OUT. Gates score only the final sweep; a
mid-training sweep after pass 1 is diagnostic logging, never gating.

## 6. Gates (frozen, mechanical)

Statistics: proportions scored with the Wilson score interval at
z = 1.959964; differences of proportions with the Newcombe MOVER-D interval
built from the two Wilson intervals. Pass requires the point estimate to clear
the margin AND the interval bound stated below.

- **G-S0 fixture + health (pre-cohort)**: red tests green — (i) a repeated
  bigram becomes predicted after one exposure; (ii) the same token in a novel
  context elects a distinct winner; (iii) sustained punishment drops a wrong
  predictor below `θ_a`; (iv) bit-identical reruns. Spike-rate health
  pre-flight (population rate ≥ 0.1% analytic steady-state check) before any
  long cohort; contract tripwires of §1 pass.
- **G-S1 engagement**: TRAIN `f_pred(final)` − 0 (epoch-0 floor) ≥ **Δ = 0.05**
  absolute with Wilson 95% lower bound > 0. Growth must memorize *something*
  before the transfer question is even asked.
- **G-S2 transfer**: HELD-OUT `f_pred(final)` − `baseline_burst` ≥ **Δ = 0.05**
  absolute with Wilson 95% lower bound > 0, where `baseline_burst` = A2's
  HELD-OUT `f_pred` (structurally 0, asserted). Margin pre-set: five points is
  the smallest movement that would have been unambiguous against the U-STATE
  P3 signature (38.3% TRAIN → 0% VAL), given ~239k held-out proxy tokens give
  tens of thousands of scored transition steps.
- **G-S3 attribution**: A1 HELD-OUT − A3 HELD-OUT ≥ **Δ = 0.05** with Newcombe
  interval excluding 0. Winner election must beat equally-fed generic growth,
  not merely do something.

Cohort ladder (stop-on-failure, cohort spent once):
1. **C1 @ 128k cells** — all gates. PASS = G-S1 ∧ G-S2 ∧ G-S3.
2. **C2 @ 1M cells** — runs ONLY on C1 pass; requires directional replication:
   HELD-OUT Wilson lower bound > 0 (margin may shrink, sign may not flip) and
   G-S3 sign holds.

## 7. Verdicts (mechanical, no judgment calls)

- `PROCEED_R3_MEASURED`: C1 pass ∧ C2 replication. R3 episodic-store proceeds
  with measured mechanics; three-factor and race-consumption work unblocks.
- `MIXED_SCALE_BLOCK`: C1 pass ∧ C2 fail. Report verbatim tables; mainline
  status = "works small, unproven at scale"; escalates to the user decision
  point (execution-order step 5) with measured gates attached. No retry.
- `KILL_PILOT_MEMORIZATION_ONLY`: G-S1 ∧ ¬G-S2 — TRAIN improves while
  held-out sits at burst baseline. This is the synthesis falsifier, verbatim:
  "TRAIN-book transition recognition improves while held-out books sit at
  burst baseline … means growth memorizes without transferable structure;
  kill this pilot and fall back to the probe-defined path 3 boundary."
  (path-decision-synthesis.md §Path 2.)
- `NEGATIVE_NO_ENGAGEMENT`: ¬G-S1 — growth did not engage even on TRAIN.
  Localization mandatory (segment allocation counts, burst-fraction
  trajectories, punishment statistics, `W_{t-1}` coverage), reported in
  FINDINGS; cohort spent once; no silent fixes, no reruns.

Secondary instrumentation (reported either way, never gating): burst fraction
per book per pass; segments/cell and contacts/segment distributions; winner
persistence (re-election entropy per token pool); permanence histograms;
wall-clock per pass vs the ≤ 15 h/cohort budget.

## 8. Cost and schedule (frozen)

Local box only; rental $0; the $70 reservation untouched. C1 (128k) is an
hours-scale cohort; C2 (1M) is bounded by the measured anchor (full-KJV 1M
CUDA ≈ 15 h, `roadmap-reading-to-coding.md`). Hard wall-clock cap: any cohort
exceeding 15 h aborts and reports partial logs — no silent extensions, no
downscaled reruns inside the same cohort slot.

Commit discipline on `codex/path2-pilot` (push each milestone):
D1 this document; D2 implementation + red tests (fixture-first);
D3 replay manifests (corpus/curriculum SHA-256s asserted equal to the pinned
values; binary arrays stay local with array-level SHA-256s committed);
D4 `FINDINGS` verdict document with the §6 gate table filled from raw logs.
