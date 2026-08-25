# Deep dive: HTM Temporal Memory — why TM sequences work where U-STATE decode failed

**Date:** 2026-08-25
**Status:** primary-source mechanism comparison; integration verdict
**Sources read in full this session (downloaded to `%TEMP%\htm_deepdive\`):**

- `tm_paper.txt` — Hawkins & Ahmad 2016, *Why Neurons Have Thousands of Synapses,
  a Theory of Sequence Memory in Neocortex*, Front. Neural Circuits 10:23
  (PMC4811948), incl. Materials & Methods equations.
- `cui_sdr.txt` — Ahmad & Hawkins 2015, *Properties of Sparse Distributed
  Representations and their Application to Hierarchical Temporal Memory*,
  arXiv:1503.07469 (note: authors are **Ahmad & Hawkins**, not Cui et al.;
  Cui et al. is arXiv:1512.05463 and is only cited inside these texts).
  Includes Appendix A "SDR Actuarial Tables" for unions.
- `ah2017.txt` — Ahmad & Hawkins, *How do neurons operate on sparse distributed
  representations? A mathematical theory of sparsity, neurons and active
  dendrites*, arXiv:1601.00720 (provisional/Frontiers typeset), incl. Fig. 4–7
  error-rate regimes.
- `bami_tm.txt` — BAMI *Temporal Memory Algorithm* chapter rev 0.56 (May 2021),
  full pseudocode.
- `TemporalMemory.hpp` / `.cpp` / `Connections.hpp` — htm.core implementation;
  defaults verified verbatim at HEAD of the download.

**Our side:** `ustate-preregistration.md`, `experiments/ustate_probes/FINDINGS.md`
(NEGATIVE_WITH_LOCALIZATION_V2), `src/humanbrain/sdr_union.py`,
`ustate-capacity-study.md`, `custom-architecture-directions.md`,
`apical-and-verdict.md`, `t-route-program-conclusion.md`, `../architecture.md`
cell-only contract.

---

## Bottom line

**TM sequences work where U-STATE decode failed because TM never puts the
stream into one vector.** It spends cells to make (input × context) a
first-class representational object — same columns, different cell per context
— stores each learned transition as a private subsampled coincidence detector
(distal segment) whose false-positive rate is super-exponentially small, keeps
unions only as a *momentary, bounded* predicted-state, and resolves ambiguity
by competition (depolarized cell spikes first) instead of by a learned decoder
over a static state. Our U-STATE did the opposite at every one of those
forks: one cumulative union vector over raw token identities, conjunctions
never created, decode outsourced to logistic/MLP heads. G-U2's failure
(best 2.19% vs majority-prior 4.07%) was structural information absence,
exactly as FINDINGS localized — and TM theory says precisely which structure
was missing.

Direction A **stays dead as specified** (cumulative union + external decode);
the falsifier fired correctly and nothing in the HTM sources rehabilitates a
global history union. But the deep dive finds Direction A was an
anti-TM design wearing TM clothes: every mechanism that makes TM work is
already present in, or maps cleanly onto, our GLIF3 dendritic lane + GLIF5
local-plasticity machinery under the cell-only contract. The adopt/adapt/avoid
table below routes each surviving piece.

---

## Q1. How TM's SDR math differs from our union-state

### 1.1 The two-representation trick: cells-per-column factoring

TM converts the feedforward column code into a **context-partitioned cell
code**: all cells in a column share the proximal receptive field, but which
*cell* fires encodes temporal context (Frontiers 2016, Fig. 2; BAMI §"Encoding
an input differently in different contexts"). With w active columns and M
cells/column there are up to M^w distinct representations of the *same*
columnar input — BAMI's example: 100 columns × 4 cells ⇒ 4^100 contexts, with
any two such patterns sharing only ~25/100 cells.

The Frontiers paper makes this the load-bearing requirement, with the exact
argument U-STATE violates:

> consider two abstract sequences "ABCD" and "XBCY" … the internal
> representation during the subsequence "BC" must be different in the two
> cases; otherwise the correct prediction cannot be made after "C".

`sdr_union.py` cannot satisfy this even in principle: token B contributes the
identical `s_B` bits whether preceded by A or X. `U_t = max(λU, s_t)` is a set
of recent tokens with decay weights; "B-after-A" and "B-after-X" collapse to
the same support bits. Every QA answer that depends on relations between
tokens rather than the tokens' identities is absent from the state — which is
precisely what P1/P2/P3 measured (decode 2.19%, self-store 38%/transfer 0%).
The λ sweep being flat (FINDINGS localization #4) is the expected signature:
no recency window can conjure conjunctions that were never written.

### 1.2 Storage is thousands of private detectors, not one pooled vector

TM memory = per-cell distal segments; each segment holds s ≈ 15–40 synapses
subsampling a ~400-cell presynaptic pattern, fires when ≥ θ connected synapses
are active (htm.core defaults verified: `activationThreshold=13` connected of
`maxNewSynapseCount=20` grown targets, `connectedPermanence=0.50`;
BAMI learning tier: `minThreshold=10` *potential* synapses marks a segment
"matching", eligible to learn). Ahmad–Hawkins overlap-set math gives the error
envelope: n=2048, w=40-class parameters put single-detector false positives
below 1e-11, subsampled detection below ~1e-11 at 50% noise tolerance, and
segment-mixed unions below ~1e-10 for M=10 patterns at n=20000 (arXiv:1601.00720
Figs. 4–7; optimal dendritic θ range 9–20).

So each stored transition lives in its own high-dimensional needle: cross-talk
between memories is bounded per-detector by combinatorics, not by how much
else is stored. A union vector has no such isolation — bit b carries a sum of
everything that ever touched it. Our capacity precondition measured the
consequence directly: mean union support 786 bits with universal-bit L2 mass
0.60–0.82 across arms (`capacity_precondition.json`) — i.e., common bytes own
the vector and item-specific bits drown.

### 1.3 Where unions legitimately appear in TM — and their bounds

The only union in TM is the **predicted state**: phase 2 matches every segment
against the current active-cell SDR, and all passing cells depolarize
simultaneously (arXiv:1503.07469 §III-C: "this step then determines a union of
possible temporal states for the next time step"). Three properties keep it
safe where our U_t was not:

1. **Momentary**: it represents predictions for exactly the next step, then is
   consumed by the incoming input. Nothing accumulates across the stream.
2. **Bounded membership**: typical operation predicts ~10 next-elements ≈ 2%
   of columns holding one predicted cell each (BAMI §2); the actuarial tables
   show n=65536 (=2048 cols × 32 cells) tolerates M≈600–1000 union members at
   θ=w with fp ≤ 1e-14, degrading gracefully to ~1e-3 at θ=32/w=40, M=1000.
   Members are also *cells*, i.e. already context-tagged, so union members are
   mutually far more distinct than raw token SDRs.
3. **Membership test, not content store**: the union is only ever queried by
   `match(X, y)` from the *next input* ("was this column predicted?"), never
   asked to regurgitate members through a generic decoder.

Our U_t violated all three: cumulative over ~200+ tokens, queried by external
heads for content, at n=2048 — the regime arXiv:1601.00720 Fig. 7B flags as
high mix-and-match error even for well-formed segment unions.

### 1.4 Bursting, winner election, and punishment

- **Predicted path**: if any cell in the winning column is depolarized, only
  those fire (sparse, context-disambiguated output); their segment gets
  permanence increments (+0.10) on active synapses, decrements (−0.10) on
  inactive ones, plus growth of new synapses toward the previous step's winner
  cells up to `SYNAPSE_SAMPLE_SIZE`/`maxNewSynapseCount` (BAMI pseudocode
  lines 10–24).
- **Unpredicted → burst**: all 32 cells fire; the column elects ONE winner
  cell (best-matching segment's cell if any segment had ≥10 potential-active
  synapses, else least-used cell + brand-new segment) and grows its segment
  toward previous winners (BAMI lines 25–48). This is how a novel transition
  acquires a dedicated detector without disturbing existing ones.
- **Punish wrong predictions**: matching segments in inactive-but-predicted
  columns get `predictedSegmentDecrement` (htm.core default 0.0 = off;
  recommended ≈ column-sparsity × increment, e.g. 0.0004 at 2% sparsity).
  Frontiers eq. (7): small decay on segments that caused a depolarization that
  wasn't confirmed.

None of these has any counterpart in `sdr_union.py`: no allocation of new
representational resources, no winner election, no negative evidence, no
distinction between expected and surprising input. The union writes every
token identically.

### 1.5 Capacity bookkeeping

TM capacity counts *transitions*: (cells-per-column ÷ column sparsity) ×
patterns recognized per neuron ≈ (32/0.02) × 200 = 320k transitions in the
paper's reference config — scaling linearly in cells/column and segments/cell
(`maxSegmentsPerCell=255` in htm.core). U-STATE capacity counted *bits of
support before saturation*. The first is a count of learned conjunctions; the
second is a count of co-stored atoms. Only the first corresponds to what the
QA gate needs.

---

## Q2. What HTM got right that U-STATE missed

Ranked by causal contribution to the G-U2 failure:

1. **Context-conditioned re-representation (the fatal one).** Same-input-
   different-cell coding creates the conjunction (context, token) *inside the
   representation*, so downstream selection never has to extract it. U-STATE
   required decoders to extract a conjunction from additive superposition of
   independent random projections — recovering w^2/n-overlap mixtures, not
   bindings. No head can learn a function of information that is not in its
   input; FINDINGS localization #1/#2 is this sentence in numbers.
2. **Online conjunctive feature creation via growth.** Segments grow toward
   the specific cells that were just active when something surprising happens
   (burst) or to top-up a barely-wrong predictor. Feature space expands on
   demand. A frozen seeded projection is feature-space-fixed forever; concept
   tokens (Change Note 1) changed the atoms but still could not create
   *combinations*.
3. **Prediction carried by a second state + competition, not by regression.**
   Depolarized cells fire earlier and inhibit column-mates (fast local
   inhibition); the network's output *is* the disambiguation. U-STATE's
   architecture asked logistic/MLP heads to play both roles offline against a
   static vector — the role HTM assigns to dynamics.
4. **Bounded, momentary unions.** See 1.3. Union-as-state vs
   union-as-transient-prediction-buffer is the difference between saturation
   and the actuarial tables' safe regime.
5. **Negative evidence and resource accounting.** Punishment of wrong
   predictors + one-winner-per-burst + least-used-cell allocation keep the
   detector population denoised and balanced. U-STATE had no way to be wrong.
6. **High effective dimensionality for anything union-like.** Cell-space n =
   64k+ at toy sim scale; our union ran at n=2048. Even a perfect TM-style
   predicted-union transplanted onto our n would sit in the flagged
   error-prone regime.

Equally important — **what TM does not give us even if adopted faithfully**:
TM is a *transition* memory. It recognizes a prefix-like active pattern and
depolarizes what followed it historically; its capacity metric is transitions,
its prediction horizon is the next element (with high-order context folded
into the cell identity), and it has no mechanism that maps a long passage to a
distant answer span. Bible-QA needs binding across tens of tokens plus answer
selection; TM supplies the representation discipline and the local learning
rules for that binding, but the associative store and the readout race remain
ours to build. This is consistent with `t-route-program-conclusion.md`: the
constraint that died was fixed connectivity + token-identity bits, and TM's
answer to that trap is structural (spend connectivity: grow segments).

---

## Q3. Re-expression under the cell-only contract; is Direction A fixable?

The cell-only contract (architecture.md): learned state = cell identity +
dendritic branches + local synaptic strength; next token = winner of a
simulated membrane/spike race; no ANN at inference. Mapping TM mechanisms:

| TM mechanism | Re-expression on our substrate | Status |
|---|---|---|
| Distal segment = subsampled coincidence detector | Dendritic branches grown by replayed sequences (GLIF3 lane); NMDA-like plateau currents | **Already implemented** — architecture.md describes exactly this; this deep dive confirms it is the closest ancestor, not an import |
| Predictive state = sub-threshold depolarization → fires earlier | Small sustained somatic current/threshold offset on cells whose branch matched, consumed by the existing first-spike basket-cell race | **Adaptable, untested in this form.** APICAL-AND killed *multiplicative plateau gating* as an amplifier (CI ≈ 1.03 vs ≥ 1.5; recruitment-limited regime) — not the race-head-start use. A matched-segment-count-proportional bias is TM-consistent, cheap, and rides machinery that is bit-exact and live |
| Permanence ±, grow-to-prev-winners ≤ 20 | Three-factor rule (eligibility × competitor × signed target) + capped structural growth toward recent winners; developmental recruitment = leastUsedCell analog (dormant prefix recruitment exists) | **Already implemented in kind**; constants adoptable (θ 9–20 active synapses; s 15–25 grown; grow cap ≈ 20; punish ≈ 4%×increment) |
| predictedSegmentDecrement (punish wrong predictor) | Anti-Hebbian/local decay term on branches that converged but lost the race | **Trivially addable**, local, currently absent |
| Cells-per-column context tagging | Output/token cells are already vocabulary-sized and identity-addressed; reservoir cells have private recurrent state. What is missing is *electing a persistent winner cell per new context* so repeated contexts converge | Design gap worth one preregistered probe, not a rebuild |
| Burst / anomaly signal | Fraction-of-columns-that-burst = free anomaly diagnostic on logged replays | Adoptable as instrumentation |

### Does Direction A have a fixable variant?

**No — as a state representation it stays dead, and the sources explain why the
death is principled rather than parametric.** The falsifier clause fired on a
flat λ sweep, twice-confirmed separation-vs-decode dissociation, and zero
concept-arm benefit; TM theory adds that a cumulative union of raw token
identities cannot express (context, token) bindings, so no readout could have
passed G-U2 honestly. Any "fixed variant" that adds conjunction creation stops
being a union state and becomes TM — i.e., becomes the dendritic mainline we
already run. Concretely, the salvage list from Direction A is short and
non-state-shaped:

- seeded SDR projections stay useful as **input encoders** feeding branch
  growth (deterministic, order-independent — reuse `token_sdr_indices`);
- the union actuarial math (`ustate-capacity-study.md` + arXiv:1503.07469
  Table 4) becomes a **bounding tool**: how many parallel branches/predictions
  may converge before mix-and-match errors matter, evaluated at true cell-space
  n, not n=2048;
- the observation-sink logging/replay harness and probes comparators carry
  forward unchanged.

---

## Adopt / adapt / avoid

| Verdict | Item | Why / source anchor |
|---|---|---|
| **ADOPT** | Context-conditioned re-representation as a design law: every future representation must make (input, context) explicit at encode time | Frontiers 2016 ABCD/XBCY argument; explains G-U2 failure mechanistically |
| **ADOPT** | Sub-sampled binary coincidence constants for dendritic branches: θ ≈ 9–20 active connected synapses, s ≈ 15–25 grown contacts, contact budget cap ≈ 20, one-pattern-per-new-context allocation | htm.core defaults (13/10/20/0.50); Ahmad–Hawkins error tables (fp ≤ 1e-11 regime) |
| **ADOPT** | Negative-evidence term: decay branches that converged but were not confirmed (analog of `predictedSegmentDecrement`, ≈ 4% × increment, ON by default) | Frontiers eq. (7); htm.core default-off is a known weakness we should not copy |
| **ADOPT** | Burst fraction as a free anomaly/surprise metric in replay logs | BAMI burst semantics; zero-cost instrumentation |
| **ADOPT** | Momentary bounded predicted-unions only, with membership tested by incoming input — never a cumulative state | arXiv:1503.07469 §III-C + Table 4 safe regime |
| **ADAPT** | Predictive state as a small proportional depolarizing bias (or threshold offset) entering the existing first-spike race — NOT multiplicative plateau gating | APICAL-AND verdict falsified the gate-as-amplifier form in the recruitment-limited regime; the race-head-start form is untested and TM-native |
| **ADAPT** | Winner-cell election: persist a dedicated cell per recurring context (least-used/best-match tie-break) so repeated contexts converge instead of re-bursting | BAMI `burstColumn`; maps onto dormant-cell recruitment + three-factor machinery |
| **ADAPT** | Two-tier branch scoring: "recall-competent" (connected-count ≥ θ_a) vs "learnable" (potential-count ≥ θ_m) used for different decisions | htm.core activationThreshold/minThreshold split; cleaner than a single threshold in branch lookup |
| **AVOID** | Cumulative union-of-token-identities as network state, in any encoding (byte or concept) | NEGATIVE_WITH_LOCALIZATION_V2 + TM theory: conjunctions structurally unwritable; flat λ sweep proves non-parametric |
| **AVOID** | Offline learned heads (logistic/MLP) as the arbiter of spike-state content in future probes | HTM resolves by competition; every program failure with offline heads (r0b/r0c/U-STATE/APICAL-AND/T-ROUTE) is consistent with the state lacking the needed function |
| **AVOID** | Porting the algorithmic HTM simulator (binary permanences, discrete pseudo-states) literally; adopting principles instead of the codebase | HTM cells are algorithmic objects, not fitted membranes; Numenta labels original HTM legacy/unmaintained; our contract requires fitted GLIF dynamics |
| **AVOID** | Union constructions at small n (n≈2048, w≈32) | arXiv:1601.00720 Fig. 7B flags n≈1000-class populations as mix-and-match-error-prone even for well-formed unions |

## Relation to open program decisions

`t-route-program-conclusion.md` lists three escape paths needing user
direction. This deep dive bears on path 2 (trained-by-construction
connectivity): TM is existence proof that *growing* connectivity toward
recent winners, gated by surprise, is both biologically attested and
compatible with local rules — it strengthens path 2 relative to paths 1/3 on
the representation axis. The predictive-depolarization race bias (ADAPT row)
is the cheapest untested TM-flavored probe and composes with either path; if
it is ever run, it should inherit the APICAL-AND early-warning diagnostic
(TRAIN-self-store vs VAL-generalization split) and score race-resolved
readouts, not offline heads.
