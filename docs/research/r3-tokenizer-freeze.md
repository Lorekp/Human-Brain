# R3 tokenizer freeze for C1 — concept-token encoding with pool-mapping convention

Status: **FROZEN 2026-08-26, before any D2 implementation exists**, on branch
`codex/path2-pilot`. This document fixes the token-identity encoding that the
R3 segment-growth pilot (C1 cohort @ 128k cells) will consume. It amends
`docs/research/r3-segment-growth-preregistration.md` §3 ("Token stream") only
in the choice of encoding; **every frozen constant, arm, metric, gate margin,
and seed in that preregistration is untouched.** As with the prereg: nothing
below may be edited after the first replay token is observed; failure is a
publishable result; no re-sweep, no threshold edits, no seed changes.

Sources adopted verbatim:

- `experiments/concept_sdr_composition/FINDINGS.md` + `results_v1.json`
  (`concept-sdr-composition-v1`, corpus sha256 verified against
  `FREEZE-CANDIDATE-v3-BIBLE.json` before encoding).
- `docs/research/r3-segment-growth-preregistration.md` (frozen 2026-08-25):
  winner-cell SETS, k=32 candidate cells per token pool, 20-contact budget per
  segment, θ_a=13 connected / θ_m=10 potential, construction seed 17.

## 1. Decision

**C1 uses the concept-token encoding** from the composition study: base byte
ids stay in `[0,256)`; mined concepts own ids in the band `[1024,1536)`
(512 slots, freq≥50, MI≥5 bits, 2–12 tokens, mined exactly as pinned in
`results_v1.json` protocol). The R2 byte tokenizer is **not** swapped back in.
Token ids feed the existing segment machinery unchanged — a token id selects
its disjoint 32-cell pool deterministically (§4), and segments grow toward
winner sets composed of those ids.

## 2. Evidence adopted (composition study, verbatim numbers)

Under identical projector (n=2048, w=32, seed=7) and union rule
`U_t = max(λ·U_{t-1}, s_t)`, concept tokens beat byte tokens on all three
metric families, at all three frozen λ, monotonically more at higher λ:

- Separability (median 1−cosine over 19,900 verse pairs): concept +26% /
  +64% / **+119%** at λ ∈ {0.90, 0.97, 0.99}
  (0.3864/0.2877/0.2525 vs byte 0.3060/0.1749/0.1153).
- Collapse tail (pairs > 0.90 cosine) at λ=0.99: 34.92% (byte) → **0.12%**
  (concept) — ~290× smaller; max cross-item cosine held at 0.9535.
- Mechanism confirmed by the `byte_matched` control (equal write count,
  strictly worse than even `byte_full` on the probe): the win comes from
  *which* SDRs get written, not fewer writes.
- Id-spread — the quantity this freeze turns on: concept streams used
  **439 distinct ids (367 in the concept band)** vs **74** for bytes;
  concept streams are 0.61× as long (28,410 vs 46,722 writes per 200 verses);
  universal-bit L2 mass drops 0.700 → **0.447** at λ=0.99.

The preregistered verdict stands: "swapping the week-1 byte tokenizer for the
concept stack … composes cleanly with the existing projector — no changes
needed, ids just extend into [1024,1536)." This freeze accepts that verdict
for C1 and specifies the one thing the composition study did not: how an id
becomes a cell pool.

## 3. The open question this freeze had to answer

> Does the concepts' wider id-spread (439 vs 74 distinct ids) help or hurt
> segment contact-budget dynamics, given segments hold ≤ 20 contacts and
> recognize at θ_a = 13 connected / θ_m = 10 potential?

Brief analysis (pre-registered reasoning, no pilot data consulted):

**Where spread helps.** Segment growth samples ≤ 20 contacts uniformly
without replacement from the previous step's winner set `W_{t-1}`. With a
74-id universe the marginal winner distribution is concentrated: two
independent 20-contact draws from a 74-id universe expect ≈ 20²/74 ≈ 5.4
shared ids; from a 439-id universe, ≈ 0.9. Byte-regime segments therefore
grow toward near-duplicate presynaptic sets — the segment-level analogue of
the union collapse tail the composition study measured (skewed bytes ' ', 'e',
't' re-hammering the same bits). Consequences: redundant segments inside a
cell pool spend their 20-contact budgets on overlapping ids, negative-evidence
decay punishes correlated segments together, and spurious cross-context
matches clear θ_a=13 too easily (over-generalization). Wider spread buys
distinctive contacts per budget unit — higher match precision.

**Where spread hurts.** Recognition is overlap-seeking: a segment fires when
≥13 of its ≤20 contacts are simultaneously active winners. Sparser winner-set
overlap means a segment matches almost only when its *own* context recurs.
First-pass engagement (`f_pred`) should therefore rise more slowly under
concepts than a byte-calibrated intuition predicts — the same-token recurring
bigram still engages (G-S0 fixture (i) is safe: identical ids → identical
pools), but generic co-occurrence glue weakens, which is precisely what G-S3
(A1 vs A3 shuffled-winner) is designed to detect. Partial mitigation already
measured: the top mined concepts are frequent function-word chunks
(' th', 'the', ' the '), giving winner sets a small shared backbone — the
honest analogue of universal bits, but reduced from 462 bits carrying 0.70 of
L2 mass to 350 carrying 0.45.

**Net read.** Spread helps the mechanism we are testing (context-specific
contact budgets spent on distinctive ids) and hurts naive engagement speed.
Neither effect is a reason to revert to bytes: the composition study shows
spread wins on separability at every λ, and slower generic engagement is a
predicted property of the treatment, not a defect. The risk it creates is
*interpretive*, not mechanical — a low `f_pred` could mean broken growth or
correctly sparse matching. That risk lands on gate interpretation, §5.

## 4. Frozen pool-mapping convention

Token id → cell pool, defined once, used identically in every arm:

- `initial_pool(id) = id mod k`, with `k = 32` (preregistration §2,
  `cellsPerColumn`). High-frequency concepts distribute uniformly across
  pools; no pool monopolizes elections.
- `slot(id) = id // k` (within-pool position). `(pool, slot)` ↔ `id` is a
  bijection on `[0, (max_id+1)·k)`: lossless, reversible, zero stored state.
- Determinism: the convention is pure arithmetic — no RNG at mapping time.
  Any residual stochastic step in the pipeline remains keyed to the single
  construction seed `17` exactly as preregistered; same (seed, corpus,
  constants) ⇒ bit-identical segment arrays, per the contract tripwires.
- Feasibility assertion at build time, replacing the vocab-299 form with the
  band-aware form: `(max_token_id + 1) × 32 ≤ configured_cells`. At the
  concept band's ceiling, `1536 × 32 = 49,152 ≤ 131,072` (C1 @ 128k) with
  2.7× headroom, and trivially inside 1M for C2. Violation ⇒ STOP before the
  run (shrink TRAIN books whole, never constants) — unchanged policy.
- No forbidden trainable components: the mapping introduces none (contract
  tripwire §1 of the preregistration applies as written).

## 5. Gate implications

Mechanical gates stay exactly as frozen (Wilson z=1.959964, Newcombe MOVER-D,
all Δ = 0.05 absolute). What the concept encoding changes is what failures
mean, so the following is fixed **now**, before data:

- **G-S1 engagement — concept-vocab-aware expected-engagement floor.** The
  Δ = 0.05 margin over the asserted-zero epoch-0 floor is retained verbatim.
  Additionally, before the first replay token, a calibration artifact must be
  committed: a hypergeometric simulation of P(≥θ_a of ≤20 grown contacts
  active) and P(≥θ_m active) under the concept-band winner-set overlap
  distribution derived from §3 (id universe 439, backbone statistics from
  `top20_concepts`), alongside the same computation for a 74-id byte regime.
  This artifact is computed from constants and composition-study statistics
  only — it never sees pilot results — and serves as the expected-engagement
  floor reference when reading G-S1. It is diagnostic, never gating.
- **Interpretation rule for ¬G-S1** (feeds the mandatory localization of
  verdict `NEGATIVE_NO_ENGAGEMENT`): localization MUST stratify `f_pred` by
  arriving-token class — concept-band ids (367 observed in the study) vs
  base-band bytes — and report segment-allocation and contact-overlap
  distributions for each stratum. A uniform failure across strata implicates
  the growth mechanism; a concept-band-specific deficit implicates match
  sparsity under wide id-spread. Either way: cohort spent once, no silent
  fixes, no reruns.
- **G-S0 fixtures unaffected**: identical ids map to identical pools, so
  fixture (i) (repeated bigram becomes predicted) and (ii) (same token, novel
  context ⇒ distinct winner) behave as specified; epoch-0 `f_pred ≡ 0`
  assertion unchanged.
- **G-S2/G-S3 unchanged and, if anything, sharpened**: the A3 control's
  generic co-occurrence growth loses its byte-regime advantage (fewer
  accidental threshold crossings under sparse overlap), so a positive
  G-S3 attribution gap is stronger evidence for winner election than it would
  have been under bytes.
- **KILL_PILOT_MEMORIZATION_ONLY semantics preserved**: if TRAIN engages but
  HELD-OUT sits at burst baseline under concept encoding, the synthesis
  falsifier fires verbatim — author-disjoint held-out books share fewer
  recurring ids, making a memorization-only failure *more* legible, not less.

## 6. What would revise this freeze (and what would not)

Not revisable post hoc: the encoding choice, the mod-k convention, the
band-aware feasibility assertion, the Δ margins, seeds, arms. Revisable only
by a new pre-registered document on a fresh branch: the concept inventory
itself (richer inventories should widen the measured gap — the study's stated
expectation — but swapping vocabularies mid-cohort is forbidden), and a
multi-seed confirmation cohort, explicitly out of scope until after the
verdict per preregistration §3.

Commit discipline: this is milestone D1-adjacent documentation landing before
D2 implementation; pushed to `codex/path2-pilot`.
